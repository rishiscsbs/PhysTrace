"""Digital-twin observer (defender side).

Predicts expected state x_hat and measurements y_hat from the known physics and
the *issued* commands (drive-side command log). Residual convention:

    r_t = y_observed,t - y_hat_predicted,t      (same units as the sensor)

Measurement correction is gated (FR-007): the twin is nudged toward LT-101 only
while the residual is small, no anomaly alarm is active, and a rolling
correction budget has not been exhausted. It never adapts on suspicious data.
"""
from __future__ import annotations

import math
from collections import deque

from ..config import SimConfig
from ..sim.plant import pump_flow, valve_flow  # pure physics functions, no ground truth

SENSOR_UNITS = {"LT-101": "m", "FT-101": "L/s", "FT-102": "L/s", "ZT-101": "%", "SC-101": "pu"}
RESIDUAL_SIGMA = {"LT-101": 0.006, "FT-101": 0.3, "FT-102": 0.3, "ZT-101": 0.6, "SC-101": 0.008}


def residual_sigma(cfg: SimConfig) -> dict[str, float]:
    """Residual tolerance per sensor, scaled by the configured (datasheet) sensor noise."""
    from ..config import SensorParams

    d, s = SensorParams(), cfg.sensors
    return {
        "LT-101": RESIDUAL_SIGMA["LT-101"] * max(1.0, s.level_noise_m / d.level_noise_m),
        "FT-101": RESIDUAL_SIGMA["FT-101"] * max(1.0, s.flow_noise_lps / d.flow_noise_lps),
        "FT-102": RESIDUAL_SIGMA["FT-102"] * max(1.0, s.flow_noise_lps / d.flow_noise_lps),
        "ZT-101": RESIDUAL_SIGMA["ZT-101"] * max(1.0, s.valve_pos_noise_pct / d.valve_pos_noise_pct),
        "SC-101": RESIDUAL_SIGMA["SC-101"] * max(1.0, s.pump_speed_noise / d.pump_speed_noise),
    }


class DigitalTwin:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        p = cfg.plant
        self.h = p.initial_level_m
        self.n = cfg.controller.u_ff
        self.pos = cfg.controller.valve_base
        self.dt = cfg.dt_s
        self.delay_steps = max(1, math.ceil((cfg.network.base_latency_s + 1e-9) / cfg.dt_s))
        self.pred_hist: deque = deque(maxlen=self.delay_steps + 2)
        self.gain = self.dt / 60.0
        self.gate_m = 0.02
        self.budget_window = int(120.0 / self.dt)
        self.budget_m = 0.03
        self.corrections: deque = deque(maxlen=self.budget_window)
        self._corr_sum = 0.0
        self.budget_exceeded = False
        self.correction_enabled = True
        self.frozen_reason: str | None = None
        self.invalid_regime = False

    def predict_measurements(self) -> dict[str, float]:
        p = self.cfg.plant
        return {
            "LT-101": self.h,
            "FT-101": pump_flow(p, self.n, self.h) * 1000.0,
            "FT-102": valve_flow(p, self.pos, self.h) * 1000.0,
            "ZT-101": self.pos * 100.0,
            "SC-101": self.n,
        }

    def step(self, obs: dict, suspicious: bool) -> tuple[dict[str, float], dict[str, float]]:
        """Returns (y_hat aligned with this step's received values, residuals)."""
        y_now = self.predict_measurements()
        self.pred_hist.append(y_now)
        # Received values lag the field by the nominal transport delay; compare like with like.
        y_hat = self.pred_hist[-1 - self.delay_steps] if len(self.pred_hist) > self.delay_steps else y_now
        recv = obs["recv"]
        res = {}
        for sid, pred in y_hat.items():
            v = recv.get(sid, float("nan"))
            res[sid] = (v - pred) if v is not None and not math.isnan(v) else float("nan")

        # Gated correction on level only.
        r = res["LT-101"]
        corr = 0.0
        self.budget_exceeded = self._corr_sum > self.budget_m
        if not suspicious and not self.budget_exceeded and not math.isnan(r) and abs(r) < self.gate_m:
            corr = self.gain * r
            self.h += corr
            self.frozen_reason = None
        else:
            self.frozen_reason = ("anomaly_alarm" if suspicious else "budget_exceeded" if self.budget_exceeded
                                  else "residual_gate" if not math.isnan(r) else "no_measurement")
        if len(self.corrections) == self.corrections.maxlen:
            self._corr_sum -= abs(self.corrections[0])
        self.corrections.append(corr)
        self._corr_sum = max(0.0, self._corr_sum + abs(corr))

        # Propagate with issued commands.
        p = self.cfg.plant
        alpha = 1.0 - math.exp(-self.dt / p.pump_time_constant_s)
        self.n += alpha * (obs["u_cmd"] - self.n)
        slew = p.valve_slew_rate_per_s * self.dt
        self.pos += max(-slew, min(slew, obs["valve_cmd"] - self.pos))
        qi = pump_flow(p, self.n, self.h)
        qo = valve_flow(p, self.pos, self.h)
        self.h = min(max(self.h + self.dt * (qi - qo) / p.area_m2, 0.0), p.tank_height_m)
        self.invalid_regime = self.h <= 0.05 or self.h >= p.tank_height_m - 0.01
        return y_hat, res

    def snapshot(self) -> dict:
        return {"h": self.h, "n": self.n, "pos": self.pos}

    def restore(self, snap: dict) -> None:
        self.h, self.n, self.pos = snap["h"], snap["n"], snap["pos"]

    def inflow_gain_lps(self) -> float:
        """dQ_in/dn at the current operating point (L/s per unit speed), for watermark templates."""
        p = self.cfg.plant
        q = pump_flow(p, self.n, self.h)
        if q <= 1e-6:
            return 0.0
        return 1000.0 * p.pump_shutoff_head_m * self.n / (p.system_resistance_s2_m5 * q)
