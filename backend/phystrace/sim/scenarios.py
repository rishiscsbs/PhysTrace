"""Attack & fault scenario lab (simulation-only).

Scenario specs are GROUND TRUTH. They are consumed by the simulator (World) and
by the evaluation ledger. The inference package never imports this module; a
static test enforces that (AT-011).

Everything here acts on the in-process simulator only: there is no network I/O
and no code path that talks to real PLCs or equipment.
"""
from __future__ import annotations

import uuid
from collections import deque
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

ScenarioType = Literal[
    "SENSOR_BIAS",
    "SENSOR_FREEZE",
    "SENSOR_REPLAY",
    "SENSOR_DRIFT",
    "SENSOR_DROPOUT",
    "COMM_DELAY",
    "COMM_PACKET_DROP",
    "COMM_JITTER",
    "COMM_STALE_REPLAY",
    "COMM_MITM_BIAS",
    "CTRL_SETPOINT_OVERRIDE",
    "CTRL_GAIN_TAMPER",
    "CTRL_LOGIC_BRANCH",
    "VALVE_STUCK",
    "VALVE_SLEW_DEGRADED",
    "PUMP_TRIP",
    "PUMP_CAVITATION",
    "DEMAND_SURGE",
    "INFLOW_FLUCTUATION",
    "TWIN_MODEL_MISMATCH",
]

SENSOR_IDS = ("LT-101", "FT-101", "FT-102", "ZT-101", "SC-101")

# name, family, expected hypothesis class, default target, allowed targets, params {name: (default, min, max, unit, help)}
CATALOG: dict[str, dict[str, Any]] = {
    "SENSOR_BIAS": dict(family="SENSOR", hypothesis="SENSOR_SPOOFING", target="LT-101", targets=list(SENSOR_IDS),
                        params={"bias": (-0.35, -1.0, 1.0, "sensor units", "Additive offset (m for LT, L/s for FT, % for ZT, 0..1 for SC)")}),
    "SENSOR_REPLAY": dict(family="SENSOR", hypothesis="SENSOR_SPOOFING", target="LT-101", targets=list(SENSOR_IDS),
                          params={"window_s": (60.0, 10.0, 120.0, "s", "Length of the recorded window replayed in a loop"),
                                  "offset": (-0.25, -1.0, 1.0, "sensor units", "Offset added to the replayed recording (masks a real excursion)")}),
    "SENSOR_FREEZE": dict(family="SENSOR", hypothesis="SENSOR_FAILURE", target="LT-101", targets=list(SENSOR_IDS), params={}),
    "SENSOR_DRIFT": dict(family="SENSOR", hypothesis="SENSOR_FAILURE", target="LT-101", targets=list(SENSOR_IDS),
                         params={"rate_per_s": (-0.003, -0.02, 0.02, "units/s", "Linear drift rate")}),
    "SENSOR_DROPOUT": dict(family="SENSOR", hypothesis="SENSOR_FAILURE", target="LT-101", targets=list(SENSOR_IDS), params={}),
    "COMM_DELAY": dict(family="COMM", hypothesis="COMMUNICATION_MANIPULATION_OR_DELAY", target="LT-101", targets=list(SENSOR_IDS),
                       params={"delay_s": (6.0, 0.5, 20.0, "s", "Added one-way latency on the fieldbus link")}),
    "COMM_PACKET_DROP": dict(family="COMM", hypothesis="COMMUNICATION_MANIPULATION_OR_DELAY", target="LT-101", targets=list(SENSOR_IDS),
                             params={"drop_rate": (0.85, 0.05, 0.99, "fraction", "Probability each packet is dropped")}),
    "COMM_JITTER": dict(family="COMM", hypothesis="COMMUNICATION_MANIPULATION_OR_DELAY", target="LT-101", targets=list(SENSOR_IDS),
                        params={"jitter_s": (3.0, 0.2, 10.0, "s", "Uniform random added latency")}),
    "COMM_STALE_REPLAY": dict(family="COMM", hypothesis="COMMUNICATION_MANIPULATION_OR_DELAY", target="LT-101", targets=list(SENSOR_IDS),
                              params={"age_s": (45.0, 5.0, 120.0, "s", "Age of the re-sent packets")}),
    "COMM_MITM_BIAS": dict(family="COMM", hypothesis="COMMUNICATION_MANIPULATION_OR_DELAY", target="LT-101", targets=list(SENSOR_IDS),
                           params={"bias": (-0.35, -1.0, 1.0, "sensor units", "Value rewritten in transit")}),
    "CTRL_SETPOINT_OVERRIDE": dict(family="CONTROLLER", hypothesis="CONTROLLER_MANIPULATION", target="PLC-101", targets=["PLC-101"],
                                   params={"setpoint_m": (2.85, 0.1, 3.0, "m", "Hidden setpoint written to the PLC register")}),
    "CTRL_GAIN_TAMPER": dict(family="CONTROLLER", hypothesis="CONTROLLER_MANIPULATION", target="PLC-101", targets=["PLC-101"],
                             params={"kp_mult": (25.0, 0.0, 50.0, "x", "Proportional gain multiplier"),
                                     "ki_mult": (8.0, 0.0, 50.0, "x", "Integral gain multiplier")}),
    "CTRL_LOGIC_BRANCH": dict(family="CONTROLLER", hypothesis="CONTROLLER_MANIPULATION", target="PLC-101", targets=["PLC-101"],
                              params={"forced_command": (0.85, 0.0, 1.0, "0..1", "Pump command forced by the malicious branch"),
                                      "trigger_level_m": (1.0, 0.0, 3.0, "m", "Branch activates when measured level exceeds this")}),
    "VALVE_STUCK": dict(family="ACTUATOR", hypothesis="ACTUATOR_FAULT", target="FV-101", targets=["FV-101"],
                        params={"position": (0.2, 0.0, 1.0, "0..1", "Position the valve is stuck at")}),
    "VALVE_SLEW_DEGRADED": dict(family="ACTUATOR", hypothesis="ACTUATOR_FAULT", target="FV-101", targets=["FV-101"],
                                params={"slew_rate_per_s": (0.0004, 0.0, 0.05, "1/s", "Degraded stem slew rate"),
                                        "command_step": (0.25, -0.4, 0.4, "0..1", "Operator valve step issued at onset")}),
    "PUMP_TRIP": dict(family="ACTUATOR", hypothesis="ACTUATOR_FAULT", target="P-101", targets=["P-101"], params={}),
    "PUMP_CAVITATION": dict(family="ACTUATOR", hypothesis="ACTUATOR_FAULT", target="P-101", targets=["P-101"],
                            params={"flow_loss": (0.45, 0.05, 0.9, "fraction", "Fraction of hydraulic flow lost")}),
    "DEMAND_SURGE": dict(family="DISTURBANCE", hypothesis="LEGITIMATE_DISTURBANCE", target="DEMAND", targets=["DEMAND"],
                         params={"factor": (1.6, 0.2, 3.0, "x", "Downstream demand multiplier")}),
    "INFLOW_FLUCTUATION": dict(family="DISTURBANCE", hypothesis="LEGITIMATE_DISTURBANCE", target="SUPPLY", targets=["SUPPLY"],
                               params={"head_amplitude_m": (2.0, 0.1, 4.0, "m", "Supply-head oscillation amplitude"),
                                       "period_s": (60.0, 10.0, 600.0, "s", "Oscillation period")}),
    "TWIN_MODEL_MISMATCH": dict(family="MODEL", hypothesis="TWIN_MODEL_MISMATCH", target="TWIN-01", targets=["TWIN-01"],
                                params={"cv_change": (0.25, -0.5, 0.5, "fraction", "Gradual change of the true outlet Cv (trim erosion / fouling)"),
                                        "ramp_s": (120.0, 10.0, 600.0, "s", "Time to reach full change")}),
}


class ScenarioSpec(BaseModel):
    scenario_id: str = Field(default_factory=lambda: "SCN-" + uuid.uuid4().hex[:8])
    type: ScenarioType
    target: str | None = None
    start_s: float = Field(60.0, ge=0)
    duration_s: float | None = Field(None, gt=0, description="None = persists until end of run")
    params: dict[str, float] = Field(default_factory=dict)
    seed: int = 0

    def resolved(self) -> "ScenarioSpec":
        cat = CATALOG[self.type]
        params = {k: v[0] for k, v in cat["params"].items()}
        params.update(self.params)
        return self.model_copy(update={"target": self.target or cat["target"], "params": params})

    def active(self, t: float) -> bool:
        if t < self.start_s:
            return False
        return self.duration_s is None or t < self.start_s + self.duration_s

    @property
    def family(self) -> str:
        return CATALOG[self.type]["family"]

    @property
    def expected_hypothesis(self) -> str:
        return CATALOG[self.type]["hypothesis"]


def catalog_for_api() -> list[dict[str, Any]]:
    out = []
    for name, c in CATALOG.items():
        out.append({
            "type": name,
            "family": c["family"],
            "default_target": c["target"],
            "targets": c["targets"],
            "params": [
                {"name": k, "default": v[0], "min": v[1], "max": v[2], "unit": v[3], "help": v[4]}
                for k, v in c["params"].items()
            ],
        })
    return out


class Injector:
    """Applies active scenarios at each simulator layer."""

    def __init__(self, scenarios: list[ScenarioSpec] | None = None):
        self.scenarios: list[ScenarioSpec] = [s.resolved() for s in (scenarios or [])]
        self._freeze: dict[str, float] = {}
        self._history: dict[str, deque] = {sid: deque(maxlen=2400) for sid in SENSOR_IDS}
        self._replay_buf: dict[str, list[float]] = {}
        self._gens: dict[str, np.random.Generator] = {}

    def add(self, spec: ScenarioSpec) -> ScenarioSpec:
        spec = spec.resolved()
        self.scenarios.append(spec)
        return spec

    def clear(self) -> None:
        self.scenarios.clear()

    def _gen(self, s: ScenarioSpec) -> np.random.Generator:
        g = self._gens.get(s.scenario_id)
        if g is None:
            g = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(s.seed), 31337])))
            self._gens[s.scenario_id] = g
        return g

    def active(self, t: float, family: str | None = None) -> list[ScenarioSpec]:
        return [s for s in self.scenarios if s.active(t) and (family is None or s.family == family)]

    # ---------------- sensor layer (field device output) ----------------
    def sensor(self, t: float, sensor_id: str, value: float, dt: float) -> float:
        hist = self._history[sensor_id]
        out = value
        for s in self.active(t, "SENSOR"):
            if s.target != sensor_id:
                continue
            key = s.scenario_id
            if s.type == "SENSOR_BIAS":
                out = out + s.params["bias"]
            elif s.type == "SENSOR_FREEZE":
                if key not in self._freeze:
                    self._freeze[key] = hist[-1] if hist else value
                out = self._freeze[key]
            elif s.type == "SENSOR_DRIFT":
                out = out + s.params["rate_per_s"] * (t - s.start_s)
            elif s.type == "SENSOR_DROPOUT":
                out = float("nan")
            elif s.type == "SENSOR_REPLAY":
                if key not in self._replay_buf:
                    n = max(1, int(s.params["window_s"] / dt))
                    self._replay_buf[key] = list(hist)[-n:] or [value]
                buf = self._replay_buf[key]
                idx = int(round((t - s.start_s) / dt)) % len(buf)
                out = buf[idx] + s.params.get("offset", 0.0)
        if sensor_id in self._history and not any(
            s.type in ("SENSOR_FREEZE", "SENSOR_REPLAY") and s.target == sensor_id for s in self.active(t, "SENSOR")
        ):
            hist.append(value)
        return out

    # ---------------- network layer ----------------
    def network(self, t: float, signal_id: str) -> dict[str, float]:
        mods = {"extra_delay_s": 0.0, "drop_rate": 0.0, "jitter_s": 0.0, "stale_age_s": 0.0, "bias": 0.0, "jitter_draw": 0.0}
        for s in self.active(t, "COMM"):
            if s.target != signal_id:
                continue
            if s.type == "COMM_DELAY":
                mods["extra_delay_s"] += s.params["delay_s"]
            elif s.type == "COMM_PACKET_DROP":
                mods["drop_rate"] = max(mods["drop_rate"], s.params["drop_rate"])
            elif s.type == "COMM_JITTER":
                mods["jitter_s"] += s.params["jitter_s"]
            elif s.type == "COMM_STALE_REPLAY":
                mods["stale_age_s"] = s.params["age_s"]
            elif s.type == "COMM_MITM_BIAS":
                mods["bias"] += s.params["bias"]
        return mods

    def network_draw(self, t: float) -> float:
        """One uniform draw per step for every active COMM scenario (state independent)."""
        v = 0.0
        for s in self.active(t, "COMM"):
            v = float(self._gen(s).random())
        return v

    # ---------------- controller (PLC internals) ----------------
    def controller(self, t: float) -> dict[str, Any]:
        mods: dict[str, Any] = {"setpoint_m": None, "kp_mult": 1.0, "ki_mult": 1.0, "forced_command": None, "trigger_level_m": None}
        for s in self.active(t, "CONTROLLER"):
            if s.type == "CTRL_SETPOINT_OVERRIDE":
                mods["setpoint_m"] = s.params["setpoint_m"]
            elif s.type == "CTRL_GAIN_TAMPER":
                mods["kp_mult"] *= s.params["kp_mult"]
                mods["ki_mult"] *= s.params["ki_mult"]
            elif s.type == "CTRL_LOGIC_BRANCH":
                mods["forced_command"] = s.params["forced_command"]
                mods["trigger_level_m"] = s.params["trigger_level_m"]
        return mods

    # ---------------- actuators ----------------
    def actuator(self, t: float) -> dict[str, Any]:
        mods: dict[str, Any] = {"valve_stuck": None, "valve_slew": None, "pump_tripped": False, "pump_flow_factor": 1.0,
                                "valve_command_offset": 0.0}
        for s in self.active(t, "ACTUATOR"):
            if s.type == "VALVE_STUCK":
                mods["valve_stuck"] = s.params["position"]
            elif s.type == "VALVE_SLEW_DEGRADED":
                mods["valve_slew"] = s.params["slew_rate_per_s"]
                mods["valve_command_offset"] = s.params.get("command_step", 0.0)
            elif s.type == "PUMP_TRIP":
                mods["pump_tripped"] = True
            elif s.type == "PUMP_CAVITATION":
                noise = float(self._gen(s).normal(0.0, 0.08))
                mods["pump_flow_factor"] = max(0.0, (1.0 - s.params["flow_loss"]) * (1.0 + noise))
        return mods

    # ---------------- exogenous disturbances & model mismatch ----------------
    def disturbance(self, t: float) -> dict[str, float]:
        mods = {"demand_factor": 1.0, "supply_head_offset": 0.0}
        for s in self.active(t, "DISTURBANCE"):
            if s.type == "DEMAND_SURGE":
                mods["demand_factor"] *= s.params["factor"]
            elif s.type == "INFLOW_FLUCTUATION":
                mods["supply_head_offset"] += s.params["head_amplitude_m"] * float(
                    np.sin(2 * np.pi * (t - s.start_s) / s.params["period_s"]))
        return mods

    def model(self, t: float) -> dict[str, float]:
        mods = {"cv_factor": 1.0}
        for s in self.active(t, "MODEL"):
            frac = min(1.0, (t - s.start_s) / max(s.params["ramp_s"], 1e-6))
            mods["cv_factor"] *= 1.0 + s.params["cv_change"] * frac
        return mods
