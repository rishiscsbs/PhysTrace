"""Deterministic world simulator: plant + PLC + field sensors + fieldbus + actuators.

Each step returns two strictly separated records:

* ``obs``   - ObservableFrame: what the defender's historian / SCADA can see
              (PLC-received sensor values, drive-side command log, PLC-reported
              tags, network-monitor statistics, watermark bookkeeping).
* ``truth`` - TruthFrame: simulation ground truth (true level, flows, overflow,
              active scenario ids). Used only for physical-impact metrics and
              post-run evaluation. It is never handed to the inference engine.
"""
from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..config import SimConfig
from .plant import Plant, pump_flow, valve_flow
from .rng import Streams, WatermarkSequence
from .scenarios import SENSOR_IDS, Injector, ScenarioSpec

RESPONSE_POLICIES = (
    "NONE",
    "BOUNDED_COMMAND_LIMITING",
    "FALLBACK_TO_CAMERA_OR_TWIN_ESTIMATE",
    "ISOLATE_REMOTE_COMMUNICATION_PATH",
    "HOLD_LAST_VERIFIED_SAFE_STATE",
    "CONTROLLED_SAFE_SHUTDOWN",
)

LIMIT_U_LO, LIMIT_U_HI, LIMIT_RATE = 0.45, 0.68, 0.01


@dataclass
class ResponseDirective:
    """Defender-issued simulated intervention. Applied by the independent safety layer."""

    policy: str = "NONE"
    fallback_level_m: float | None = None  # camera or twin estimate supplied by the defender
    hold_u: float | None = None
    hold_valve: float | None = None
    activated_at_s: float | None = None


@dataclass
class Packet:
    deliver_t: float
    seq: int
    sent_t: float
    value: float


@dataclass
class Channel:
    queue: deque = field(default_factory=deque)
    sent_log: deque = field(default_factory=lambda: deque(maxlen=1500))
    last_value: float = float("nan")
    last_seq: int = -1
    seq: int = 0
    arrivals: int = 0
    drops_seen: int = 0
    seq_anomalies: int = 0
    last_latency_s: float = float("nan")


def _quant(v: float, q: float) -> float:
    if q <= 0 or math.isnan(v):
        return v
    return round(v / q) * q


class World:
    def __init__(self, config: SimConfig, scenarios: list[ScenarioSpec] | None = None):
        self.cfg = config
        self.dt = config.dt_s
        self.k = 0
        self.t = 0.0
        self.rng = Streams(config.seed)
        self.wm = WatermarkSequence(config.watermark_seed, config.watermark.hold_steps)
        self.wm_scale = 1.0  # temporarily raised by ACTIVATE_WATERMARK_BURST
        self.plant = Plant(config.plant)
        self.injector = Injector(scenarios)
        c = config.controller
        self.golden = {"setpoint_m": c.setpoint_m, "kp": c.kp, "ki": c.ki, "u_ff": c.u_ff}
        self.plc_integral = 0.0
        self.plc_last_meas = config.plant.initial_level_m
        self.channels = {sid: Channel() for sid in SENSOR_IDS}
        self.field: dict[str, float] = {}
        self.redundant_level = config.plant.initial_level_m
        self.process_flow = 0.0
        self.valve_operator_offset = 0.0
        self.response = ResponseDirective()
        self.u_drive = 0.0
        self.last_u_limited = c.u_ff
        self.delivered_m3 = 0.0
        self.demand_m3 = 0.0
        self.unmet_m3 = 0.0
        self.pumped_m3 = 0.0
        self.energy_j = 0.0
        self.exogenous_log: list[tuple[float, float, float, float]] = []  # t, valve schedule, demand factor, supply head
        self.plc_registers = dict(self.golden)
        self.valve_test_offset = 0.0
        # Start the plant at the nominal steady state for the initial valve schedule.
        self.plant.state.pos = c.valve_base
        self.plant.state.n = c.u_ff
        self.u_drive = c.u_ff

    # ------------------------------------------------------------------ helpers
    def clone(self) -> "World":
        return copy.deepcopy(self)

    def add_scenario(self, spec: ScenarioSpec) -> ScenarioSpec:
        return self.injector.add(spec)

    def set_response(self, directive: ResponseDirective) -> None:
        if directive.policy not in RESPONSE_POLICIES:
            raise ValueError(f"unknown response policy {directive.policy}")
        if directive.activated_at_s is None:
            directive.activated_at_s = self.t
        self.response = directive

    def valve_schedule(self, t: float) -> float:
        c = self.cfg.controller
        return c.valve_base + c.valve_amplitude * math.sin(2 * math.pi * t / c.valve_period_s)

    # ------------------------------------------------- simulated out-of-band reads
    def read_field_direct(self) -> dict[str, float]:
        """Maintenance-port read of field devices, bypassing the fieldbus (diagnostic only)."""
        return dict(self.field)

    def poll_redundant_level(self) -> float:
        return self.redundant_level

    def probe_link(self, sid: str) -> dict[str, float]:
        """Active probe frames along one fieldbus link (measures what the link does to frames)."""
        mods = self.injector.network(self.t, sid)
        net = self.cfg.network
        lat = net.base_latency_s + 0.5 * net.jitter_s + mods["extra_delay_s"] + 0.5 * mods["jitter_s"]
        lat = max(self.dt, math.ceil(lat / self.dt - 1e-9) * self.dt)
        return {"latency_s": lat, "loss": max(net.drop_rate, mods["drop_rate"]), "stale": mods["stale_age_s"] > 0}

    def set_watermark_scale(self, scale: float) -> None:
        self.wm_scale = float(min(max(scale, 0.0), 3.0))  # bounded: never more than 3x epsilon

    def set_valve_test_offset(self, offset: float) -> None:
        self.valve_test_offset = float(min(max(offset, -0.05), 0.05))

    def read_plc_registers(self) -> dict[str, Any]:
        regs = dict(self.plc_registers)
        regs["logic_checksum"] = "GOLDEN-7F3A" if not self.injector.active(self.t, "CONTROLLER") or not any(
            s.type == "CTRL_LOGIC_BRANCH" for s in self.injector.active(self.t, "CONTROLLER")) else "MODIFIED-C21E"
        return regs

    # ----------------------------------------------------------------------- step
    def step(self) -> tuple[dict[str, Any], dict[str, Any]]:
        cfg, dt, t, inj, rng = self.cfg, self.dt, self.t, self.injector, self.rng
        s = self.plant.state
        sp = cfg.sensors

        # 1. exogenous inputs (drawn every step -> identical across paired worlds)
        pp = cfg.plant
        w = rng.n("process")
        sigma = pp.process_noise_lps / 1000.0
        self.process_flow = pp.process_noise_corr * self.process_flow + math.sqrt(1 - pp.process_noise_corr ** 2) * sigma * w
        dist = inj.disturbance(t)
        model = inj.model(t)
        valve_sched = self.valve_schedule(t)
        _ = rng.n("disturbance")
        self.exogenous_log.append((t, valve_sched, dist["demand_factor"], dist["supply_head_offset"]))
        if len(self.exogenous_log) > 200000:
            del self.exogenous_log[:100000]

        # 2. field sensors observe the true state
        noise = {sid: rng.n(sid) for sid in SENSOR_IDS}
        lt102_noise = rng.n("LT-102")
        true_vals = {
            "LT-101": s.h + sp.level_noise_m * noise["LT-101"],
            "FT-101": s.q_in * 1000.0 + sp.flow_noise_lps * noise["FT-101"],
            "FT-102": s.q_out * 1000.0 + sp.flow_noise_lps * noise["FT-102"],
            "ZT-101": s.pos * 100.0 + sp.valve_pos_noise_pct * noise["ZT-101"],
            "SC-101": s.n + sp.pump_speed_noise * noise["SC-101"],
        }
        quant = {"LT-101": sp.level_quant_m, "FT-101": sp.flow_quant_lps, "FT-102": sp.flow_quant_lps,
                 "ZT-101": 0.1, "SC-101": 0.001}
        for sid in SENSOR_IDS:
            v = inj.sensor(t, sid, _quant(true_vals[sid], quant[sid]), dt)
            self.field[sid] = v
        self.redundant_level = _quant(s.h + sp.redundant_level_noise_m * lt102_noise, sp.level_quant_m)

        # 3. fieldbus: field -> PLC (delay, loss, jitter, replay, MITM)
        net = cfg.network
        isolate = self.response.policy == "ISOLATE_REMOTE_COMMUNICATION_PATH"
        attack_u = inj.network_draw(t)
        net_stats: dict[str, dict[str, float]] = {}
        for sid in SENSOR_IDS:
            ch = self.channels[sid]
            u_drop = rng.u("network")
            u_jit = rng.u("network")
            mods = inj.network(t, sid)
            value = self.field[sid] + mods["bias"]
            sent_t, seq = t, ch.seq
            ch.sent_log.append((t, ch.seq, self.field[sid]))
            ch.seq += 1
            if mods["stale_age_s"] > 0:
                age_steps = int(mods["stale_age_s"] / dt)
                if len(ch.sent_log) > age_steps:
                    old_t, old_seq, old_v = ch.sent_log[-1 - age_steps]
                    sent_t, seq, value = old_t, old_seq, old_v
            latency = net.base_latency_s + net.jitter_s * u_jit + mods["extra_delay_s"] + mods["jitter_s"] * attack_u
            drop = u_drop < max(net.drop_rate, mods["drop_rate"])
            if isolate:
                # Hardwired local loop: PLC reads the field device directly; bypasses the remote path.
                ch.queue.clear()
                ch.queue.append(Packet(t, ch.seq - 1, t, self.field[sid]))  # delivered this step
            elif not drop:
                ch.queue.append(Packet(t + latency, seq, sent_t, value))
            arrivals, latency_seen = 0, float("nan")
            while ch.queue and ch.queue[0].deliver_t <= t + 1e-9:
                pkt = ch.queue.popleft()
                arrivals += 1
                if pkt.seq <= ch.last_seq:
                    ch.seq_anomalies += 1
                elif pkt.seq > ch.last_seq + 1 and ch.last_seq >= 0:
                    ch.drops_seen += pkt.seq - ch.last_seq - 1
                ch.last_seq = max(ch.last_seq, pkt.seq) if pkt.seq > ch.last_seq else ch.last_seq
                ch.last_value = pkt.value
                latency_seen = t - pkt.sent_t
                ch.last_latency_s = latency_seen
            # Jitter can reorder packets in the queue; keep it sorted by delivery time.
            if len(ch.queue) > 1 and mods["jitter_s"] > 0:
                ch.queue = deque(sorted(ch.queue, key=lambda p: p.deliver_t))
            ch.arrivals += arrivals
            net_stats[sid] = {"latency_s": latency_seen, "arrivals": arrivals, "drops_total": ch.drops_seen,
                              "seq_anomalies_total": ch.seq_anomalies, "queue": len(ch.queue)}
        recv = {sid: self.channels[sid].last_value for sid in SENSOR_IDS}

        # 4. PLC level controller (possibly manipulated)
        c = cfg.controller
        cmods = inj.controller(t)
        regs = dict(self.golden)
        if cmods["setpoint_m"] is not None and not isolate:
            regs["setpoint_m"] = cmods["setpoint_m"]
        regs["kp"] = self.golden["kp"] * cmods["kp_mult"]
        regs["ki"] = self.golden["ki"] * cmods["ki_mult"]
        self.plc_registers = regs
        meas = recv["LT-101"]
        if self.response.policy == "FALLBACK_TO_CAMERA_OR_TWIN_ESTIMATE" and self.response.fallback_level_m is not None:
            meas = self.response.fallback_level_m
        if math.isnan(meas):
            meas = self.plc_last_meas  # PLC holds last good value on loss of signal
        self.plc_last_meas = meas
        err = regs["setpoint_m"] - meas
        u_unsat = regs["u_ff"] + regs["kp"] * err + self.plc_integral + regs["ki"] * err * dt
        if 0.0 <= u_unsat <= 1.0:
            self.plc_integral += regs["ki"] * err * dt  # conditional-integration anti-windup
        u_plc = min(max(regs["u_ff"] + regs["kp"] * err + self.plc_integral, 0.0), 1.0)
        if cmods["forced_command"] is not None and meas > cmods["trigger_level_m"]:
            u_plc = cmods["forced_command"]
        amods = inj.actuator(t)
        if amods["valve_command_offset"] != 0.0 and self.valve_operator_offset == 0.0:
            self.valve_operator_offset = amods["valve_command_offset"]
        valve_cmd = min(max(valve_sched + self.valve_operator_offset + self.valve_test_offset, 0.0), 1.0)

        # 5. independent safety layer (simulated responses) + watermark at the drive
        pol = self.response.policy
        u_safe = u_plc
        if pol == "BOUNDED_COMMAND_LIMITING":
            # Clamp to the verified safe envelope, then rate-limit toward it.
            target = min(max(u_plc, LIMIT_U_LO), LIMIT_U_HI)
            r = LIMIT_RATE * dt
            u_safe = min(max(target, self.last_u_limited - r), self.last_u_limited + r)
        elif pol == "HOLD_LAST_VERIFIED_SAFE_STATE":
            if self.response.hold_u is not None:
                u_safe = self.response.hold_u
            if self.response.hold_valve is not None:
                valve_cmd = self.response.hold_valve
        elif pol == "CONTROLLED_SAFE_SHUTDOWN":
            u_safe = max(0.0, self.u_drive - 0.05 * dt)
            valve_cmd = 0.0
        self.last_u_limited = u_safe
        wmc = cfg.watermark
        sym = self.wm.symbol(self.k)
        delta_u = wmc.epsilon * self.wm_scale * sym if (wmc.enabled and pol != "CONTROLLED_SAFE_SHUTDOWN") else 0.0
        u_raw = u_safe + delta_u
        u_drive = min(max(u_raw, 0.0), 1.0)
        saturated = u_raw != u_drive
        self.u_drive = u_drive

        # 6. actuators + physics
        h_prev, ovf_prev = s.h, s.overflow_m3
        st = self.plant.step(
            dt, u_drive, valve_cmd,
            supply_head_offset=dist["supply_head_offset"], demand_factor=dist["demand_factor"],
            process_flow=self.process_flow, pump_flow_factor=amods["pump_flow_factor"], cv_factor=model["cv_factor"],
            valve_slew=amods["valve_slew"], valve_stuck=amods["valve_stuck"], pump_tripped=amods["pump_tripped"],
        )
        # Demand: nominal service flow at setpoint for the scheduled valve opening and real demand.
        demand = valve_flow(pp, valve_sched, c.setpoint_m, dist["demand_factor"], model["cv_factor"])
        prev_dry = getattr(self, "_prev_dry", 0.0)
        delivered = st.q_out * dt - (st.dry_clipped_m3 - prev_dry)
        self._prev_dry = st.dry_clipped_m3
        step_unmet = max(0.0, demand * dt - delivered)
        self.delivered_m3 += delivered
        self.pumped_m3 += st.q_in * dt
        self.demand_m3 += demand * dt
        self.unmet_m3 += step_unmet
        self.energy_j += st.power_w * dt

        obs = {
            "t": round(t, 6),
            "k": self.k,
            "recv": recv,
            "u_cmd": u_drive,
            "u_plc": u_plc,
            "valve_cmd": valve_cmd,
            "setpoint_reported": self.golden["setpoint_m"],
            "delta_u": delta_u,
            "wm_symbol": sym,
            "wm_eps": wmc.epsilon * self.wm_scale if wmc.enabled else 0.0,
            "wm_saturated": saturated,
            "net": net_stats,
            "response_policy": pol,
        }
        truth = {
            "t": round(t, 6),
            "h": st.h,
            "n": st.n,
            "pos": st.pos,
            "q_in_lps": st.q_in * 1000.0,
            "q_out_lps": st.q_out * 1000.0,
            "overflow_m3": st.overflow_m3,
            "power_w": st.power_w,
            "demand_lps": demand * 1000.0,
            "delivered_m3": self.delivered_m3,
            "unmet_m3": self.unmet_m3,
            "pumped_m3": self.pumped_m3,
            "energy_j": self.energy_j,
            "h_prev": h_prev,
            "step": {"pumped_m3": st.q_in * dt, "delivered_m3": delivered, "overflow_m3": st.overflow_m3 - ovf_prev,
                     "unmet_m3": step_unmet, "energy_j": st.power_w * dt},
            "active_scenarios": [x.scenario_id for x in inj.active(t)],
            "exogenous": {"valve_schedule": valve_sched, "demand_factor": dist["demand_factor"],
                          "supply_head_offset": dist["supply_head_offset"], "process_flow_lps": self.process_flow * 1000},
        }
        self.k += 1
        self.t = self.k * dt
        return obs, truth

    # ------------------------------------------------------------- camera scene
    def camera_scene_truth(self) -> dict[str, float]:
        """Physical scene the simulated camera 'sees' (true level, pump lamp)."""
        return {"h": self.plant.state.h, "tank_height_m": self.cfg.plant.tank_height_m,
                "pump_running": self.plant.state.n > 0.1}


def expected_inflow_lps(cfg: SimConfig, n: float, h: float) -> float:
    return pump_flow(cfg.plant, n, h) * 1000.0
