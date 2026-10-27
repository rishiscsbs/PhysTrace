"""Physical-impact metrics computed directly from ground-truth trajectory arrays (FR-025, FR-026).

Definitions
-----------
max_tank_level_m           max h(t)
overflow_volume_L          volume spilled over the rim (accumulated when h would exceed h_max)
water_loss_L               water that entered but was neither delivered nor retained:
                           ∫Q_in dt − ∫Q_delivered dt − (V_end − V_start). Computed independently from the
                           mass balance. The rim is the only loss path, so it should equal overflow_volume_L;
                           it is reported separately and NEVER added to overflow (no double counting).
                           `mass_balance_closure_L` = water_loss_L − overflow_volume_L (≈ 0 if consistent).
unmet_demand_L             ∫ max(0, demand − delivered) dt  (service loss, e.g. from a shutdown)
energy_consumption_kWh     ∫ P_pump dt
unsafe_duration_s          time with h > HH or h < LL
safety_violation_count     number of entries into the unsafe region
out_of_band_duration_s     time outside the normal operating band
recovery_time_s            time from attack onset until h re-enters the band and stays there to the end
                           (0 if it never left; null if it never re-enters)
intervention_time_s        time the response was applied (null if none)
attack_to_mitigation_delay_s  intervention_time_s − attack onset (null if either missing)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..config import SimConfig

LOWER_IS_BETTER = ("max_tank_level_m", "overflow_volume_L", "water_loss_L", "unmet_demand_L", "energy_consumption_kWh",
                   "unsafe_duration_s", "safety_violation_count", "out_of_band_duration_s", "recovery_time_s")
TOLERANCE = {"max_tank_level_m": 0.01, "overflow_volume_L": 0.5, "water_loss_L": 0.5, "unmet_demand_L": 5.0,
             "energy_consumption_kWh": 0.002, "unsafe_duration_s": 0.5, "safety_violation_count": 0.5,
             "out_of_band_duration_s": 0.5, "recovery_time_s": 1.0}
SAFETY_CRITICAL = ("overflow_volume_L", "unsafe_duration_s", "safety_violation_count", "unmet_demand_L", "max_tank_level_m")


def compute_metrics(truth: list[dict], cfg: SimConfig, attack_onset_s: float | None,
                    intervention_time_s: float | None) -> dict[str, Any]:
    dt = cfg.dt_s
    A = cfg.plant.area_m2
    lim = cfg.safety
    h = np.array([x["h"] for x in truth])
    t = np.array([x["t"] for x in truth])
    def total(key: str) -> float:
        return float(sum(x["step"][key] for x in truth))

    pumped, delivered = total("pumped_m3"), total("delivered_m3")
    overflow, unmet, energy = total("overflow_m3"), total("unmet_m3"), total("energy_j")
    v_start, v_end = truth[0]["h_prev"] * A, truth[-1]["h"] * A
    water_loss = pumped - delivered - (v_end - v_start)
    unsafe_mask = (h > lim.high_high_m) | (h < lim.low_low_m)
    oob_mask = (h > lim.band_high_m) | (h < lim.band_low_m)
    entries = int(np.sum(np.diff(unsafe_mask.astype(int)) == 1) + (1 if unsafe_mask[0] else 0))
    recovery = None
    if attack_onset_s is not None:
        after = t >= attack_onset_s
        if not oob_mask[after].any():
            recovery = 0.0
        else:
            idx = np.where(after)[0]
            sub = oob_mask[idx]
            last_bad = idx[np.where(sub)[0][-1]]
            if last_bad < len(h) - 1:
                recovery = float(t[last_bad + 1] - attack_onset_s)
    delay = None
    if intervention_time_s is not None and attack_onset_s is not None:
        delay = float(intervention_time_s - attack_onset_s)
    return {
        "max_tank_level_m": round(float(h.max()), 4),
        "min_tank_level_m": round(float(h.min()), 4),
        "overflow_volume_L": round(float(overflow * 1000), 3),
        "water_loss_L": round(float(water_loss * 1000), 3),
        "mass_balance_closure_L": round(float((water_loss - overflow) * 1000), 4),
        "unmet_demand_L": round(float(unmet * 1000), 2),
        "energy_consumption_kWh": round(float(energy / 3.6e6), 5),
        "unsafe_duration_s": round(float(unsafe_mask.sum() * dt), 2),
        "safety_violation_count": entries,
        "out_of_band_duration_s": round(float(oob_mask.sum() * dt), 2),
        "recovery_time_s": None if recovery is None else round(recovery, 2),
        "intervention_time_s": intervention_time_s,
        "attack_to_mitigation_delay_s": None if delay is None else round(delay, 2),
    }


def guarded_relative(a: float | None, b: float | None, eps: float = 1e-9) -> dict:
    """Δ = B − A and Δ/|A| with explicit handling of zero / missing denominators (never NaN or ±inf)."""
    if a is None or b is None:
        return {"delta": None, "relative": None, "note": "undefined: metric missing in one world"}
    d = b - a
    if abs(a) <= eps:
        note = "relative change undefined (World A value is 0)" if abs(d) > eps else "both zero"
        return {"delta": round(d, 6), "relative": None if abs(d) > eps else 0.0, "note": note}
    rel = d / abs(a)
    if not math.isfinite(rel):
        return {"delta": round(d, 6), "relative": None, "note": "non-finite relative change suppressed"}
    return {"delta": round(d, 6), "relative": round(rel, 6), "note": None}


def compare(ma: dict, mb: dict) -> dict:
    out: dict[str, Any] = {}
    adverse = []
    for k in LOWER_IS_BETTER:
        a, b = ma.get(k), mb.get(k)
        g = guarded_relative(a, b)
        tol = TOLERANCE.get(k, 1e-6)
        if k == "recovery_time_s" and (a is None) != (b is None):
            verdict = "IMPROVED" if a is None else "WORSENED"
            g["note"] = "World A never recovered" if a is None else "World B never recovered"
        elif g["delta"] is None:
            verdict = "UNDEFINED"
        elif g["delta"] > tol:
            verdict = "WORSENED"
        elif g["delta"] < -tol:
            verdict = "IMPROVED"
        else:
            verdict = "UNCHANGED"
        g["outcome"] = verdict
        out[k] = g
        if verdict == "WORSENED" and k in SAFETY_CRITICAL:
            adverse.append(k)
    return {"deltas": out, "adverse_metrics": adverse, "adverse_outcome": bool(adverse),
            "convention": "Δ = World B (protected) − World A (unprotected); relative = Δ / |A| (null when A = 0)"}
