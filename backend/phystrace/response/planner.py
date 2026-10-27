"""Safety-constrained response planner (FR-022, FR-023).

Projection uses the simulator *as a model*: a fresh World initialised from the
defender's estimated state (twin), with an ASSUMED scenario reconstructed from
the leading hypothesis and evidence-derived parameters. It never reads the live
world's ground truth. Results are DIGITAL_TWIN_PREDICTION, labelled with their
assumptions. All interventions are simulated only.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import SimConfig
from ..sim.scenarios import ScenarioSpec
from ..sim.world import RESPONSE_POLICIES, ResponseDirective, World

POLICY_INFO = {
    "BOUNDED_COMMAND_LIMITING": "Independent safety layer clamps the pump command to the verified envelope [0.45, 0.68] with rate limit.",
    "FALLBACK_TO_CAMERA_OR_TWIN_ESTIMATE": "Controller uses the camera waterline (if quality OK) or the twin estimate instead of LT-101.",
    "ISOLATE_REMOTE_COMMUNICATION_PATH": "PLC switches to hardwired local inputs; remote writes are blocked.",
    "HOLD_LAST_VERIFIED_SAFE_STATE": "Actuator commands frozen at the last state verified before the anomaly.",
    "CONTROLLED_SAFE_SHUTDOWN": "Pump ramps to zero and the outlet valve closes (inventory retained, supply interrupted).",
}


def assumed_scenario(defender, verdict: dict, hypotheses: list[dict]) -> tuple[list[ScenarioSpec], list[str]]:
    """Reconstruct a plausible scenario from defender evidence (no ground truth)."""
    h = defender.hist
    dt = defender.dt
    n10 = int(10 / dt)

    def mean(ch):
        w = h.window(ch, n10)
        w = w[~np.isnan(w)]
        return float(w.mean()) if w.size else float("nan")

    leading = verdict.get("leading")
    comp = verdict.get("component")
    if not leading:
        return [], ["No leading hypothesis: projection assumes nominal plant behaviour (INCONCLUSIVE / NOMINAL)."]
    notes = [f"Projection assumes {leading} at {comp} persists for the whole horizon."]
    if leading in ("SENSOR_SPOOFING", "SENSOR_FAILURE") and comp:
        bias = mean(f"res:{comp}")
        if math.isnan(bias):
            notes.append(f"{comp} unavailable: modelled as dropout.")
            return [ScenarioSpec(type="SENSOR_DROPOUT", target=comp, start_s=0)], notes
        notes.append(f"{comp} modelled as a constant bias {bias:+.3f} (mean residual, last 10 s).")
        return [ScenarioSpec(type="SENSOR_BIAS", target=comp, start_s=0, params={"bias": bias})], notes
    if leading == "COMMUNICATION_MANIPULATION_OR_DELAY" and comp:
        sid = comp.split("/", 1)[1]
        lat = mean(f"lat:{sid}")
        bias = mean(f"res:{sid}")
        if not math.isnan(bias) and abs(bias) > 5 * {"LT-101": 0.006}.get(sid, 0.3):
            notes.append(f"Link {sid} modelled as in-transit value rewrite {bias:+.3f}.")
            return [ScenarioSpec(type="COMM_MITM_BIAS", target=sid, start_s=0, params={"bias": bias})], notes
        d = max(0.5, (lat if not math.isnan(lat) else 1.0) - 0.1)
        notes.append(f"Link {sid} modelled as added latency {d:.1f} s.")
        return [ScenarioSpec(type="COMM_DELAY", target=sid, start_s=0, params={"delay_s": d})], notes
    if leading == "CONTROLLER_MANIPULATION":
        reg = (defender.diag_results.get("VERIFY_PLC_SETPOINT_REGISTER") or {}).get("measurements", {}).get("registers")
        if reg and abs(reg["setpoint_m"] - defender.golden["setpoint_m"]) > 1e-6:
            notes.append(f"Hidden setpoint {reg['setpoint_m']:.2f} m from register read.")
            return [ScenarioSpec(type="CTRL_SETPOINT_OVERRIDE", start_s=0, params={"setpoint_m": reg["setpoint_m"]})], notes
        u = mean("u_cmd")
        notes.append(f"Compromised logic modelled as forced pump command {u:.2f}.")
        return [ScenarioSpec(type="CTRL_LOGIC_BRANCH", start_s=0, params={"forced_command": u, "trigger_level_m": 0.0})], notes
    if leading == "ACTUATOR_FAULT" and comp == "FV-101":
        pos = mean("ZT-101") / 100.0
        notes.append(f"Valve modelled as stuck at feedback position {pos:.2f}.")
        return [ScenarioSpec(type="VALVE_STUCK", start_s=0, params={"position": pos})], notes
    if leading == "ACTUATOR_FAULT" and comp == "P-101":
        sc = mean("SC-101")
        if sc < 0.1:
            notes.append("Pump modelled as tripped (speed feedback ≈ 0).")
            return [ScenarioSpec(type="PUMP_TRIP", start_s=0)], notes
        ratio = mean("FT-101") / max(mean("yhat:FT-101"), 1e-3)
        loss = float(min(0.9, max(0.05, 1 - ratio)))
        notes.append(f"Pump modelled as hydraulic loss {loss:.0%}.")
        return [ScenarioSpec(type="PUMP_CAVITATION", start_s=0, params={"flow_loss": loss})], notes
    if leading == "LEGITIMATE_DISTURBANCE" and comp == "DEMAND":
        ratio = mean("FT-102") / max(mean("yhat:FT-102"), 1e-3)
        notes.append(f"Demand modelled as ×{ratio:.2f} of nominal.")
        return [ScenarioSpec(type="DEMAND_SURGE", start_s=0, params={"factor": float(max(0.2, ratio))})], notes
    if leading == "TWIN_MODEL_MISMATCH":
        ratio = mean("FT-102") / max(mean("yhat:FT-102"), 1e-3)
        notes.append(f"Outlet Cv modelled as ×{ratio:.2f} of the twin value.")
        return [ScenarioSpec(type="TWIN_MODEL_MISMATCH", start_s=0, params={"cv_change": float(ratio - 1), "ramp_s": 10.0})], notes
    notes.append("No parametric model for this hypothesis: nominal plant assumed.")
    return [], notes


def check_preconditions(policy: str, defender) -> list[str]:
    fails = []
    if policy == "FALLBACK_TO_CAMERA_OR_TWIN_ESTIMATE":
        if defender.fallback_source() == "TWIN_PREDICTION" and defender.twin.invalid_regime:
            fails.append("Twin outside valid operating regime and no good camera estimate")
    if policy == "HOLD_LAST_VERIFIED_SAFE_STATE" and defender.last_verified_commands() is None:
        fails.append("No verified pre-anomaly command snapshot available")
    return fails


def project(cfg: SimConfig, defender, policy: str, scenarios: list[ScenarioSpec], horizon_s: float = 180.0) -> dict:
    snap = defender.twin.snapshot()
    pcfg = cfg.model_copy(deep=True)
    pcfg.plant.initial_level_m = float(min(max(snap["h"], 0.0), cfg.plant.tank_height_m))
    pcfg.seed = cfg.seed + 7919  # independent projection noise
    w = World(pcfg, [s.model_copy() for s in scenarios])
    w.plant.state.n, w.plant.state.pos = snap["n"], snap["pos"]
    w.u_drive = snap["n"]
    w.plc_integral = defender.shadow_integral
    # the model's clock is aligned with the live clock so the valve schedule matches
    w.k = int(round(defender.last_t / cfg.dt_s))
    w.t = w.k * cfg.dt_s
    for s in w.injector.scenarios:
        s.start_s = w.t
    d = ResponseDirective(policy=policy, activated_at_s=w.t)
    if policy == "HOLD_LAST_VERIFIED_SAFE_STATE":
        cmds = defender.last_verified_commands()
        if cmds:
            d.hold_u, d.hold_valve = cmds
    w.set_response(d)
    lim = cfg.safety
    hs, unmet0, ovf0, e0 = [], w.unmet_m3, w.plant.state.overflow_m3, w.energy_j
    steps = int(horizon_s / cfg.dt_s)
    for _ in range(steps):
        if policy == "FALLBACK_TO_CAMERA_OR_TWIN_ESTIMATE":
            w.response.fallback_level_m = w.plant.state.h  # assumption: fallback estimate is accurate
        _, tr = w.step()
        hs.append(tr["h"])
    a = np.array(hs)
    unsafe = float(np.sum((a > lim.high_high_m) | (a < lim.low_low_m)) * cfg.dt_s)
    oob = float(np.sum((a > lim.band_high_m) | (a < lim.band_low_m)) * cfg.dt_s)
    return {"predicted_max_level_m": round(float(a.max()), 4), "predicted_min_level_m": round(float(a.min()), 4),
            "predicted_overflow_L": round((w.plant.state.overflow_m3 - ovf0) * 1000, 2),
            "predicted_unsafe_duration_s": round(unsafe, 1), "predicted_out_of_band_s": round(oob, 1),
            "predicted_unmet_demand_L": round((w.unmet_m3 - unmet0) * 1000, 1),
            "predicted_energy_kWh": round((w.energy_j - e0) / 3.6e6, 4),
            "trajectory": [round(float(x), 4) for x in a[:: int(2 / cfg.dt_s)]]}


def _hypothesis_set(defender, k: int = 3) -> list[dict]:
    """Leading verdict plus the strongest competing hypotheses (positive raw score)."""
    verdict = defender.result.get("verdict", {})
    hyps = [h for h in defender.result.get("hypotheses", []) if h["hypothesis_type"] != "INCONCLUSIVE" and h["raw_score"] > 0.5]
    hyps.sort(key=lambda h: -h["raw_score"])
    out = []
    if verdict.get("state") == "LEADING_HYPOTHESIS":
        out.append({"leading": verdict["leading"], "component": verdict.get("component"), "weight": "leading"})
    for h in hyps:
        if len(out) >= k:
            break
        if any(o["leading"] == h["hypothesis_type"] for o in out):
            continue
        out.append({"leading": h["hypothesis_type"], "component": h["component_id"], "weight": f"raw {h['raw_score']:.2f}"})
    return out or [{"leading": None, "component": None, "weight": "nominal"}]


WORST = ("predicted_max_level_m", "predicted_overflow_L", "predicted_unsafe_duration_s", "predicted_out_of_band_s",
         "predicted_unmet_demand_L", "predicted_energy_kWh")


def plan(cfg: SimConfig, defender, horizon_s: float = 300.0) -> dict:
    """Project every policy under each plausible hypothesis and rank by the worst case (minimax)."""
    verdict = defender.result.get("verdict", {})
    hset = _hypothesis_set(defender)
    worlds = []
    for hv in hset:
        scen, notes = assumed_scenario(defender, {"leading": hv["leading"], "component": hv["component"]}, [])
        worlds.append({"hypothesis": hv, "scenarios": scen, "notes": notes})
        sid = (hv["component"] or "").split("/")[-1]
        if hv["leading"] in ("SENSOR_SPOOFING", "SENSOR_FAILURE", "COMMUNICATION_MANIPULATION_OR_DELAY") and sid == "LT-101":
            # Worst case for an untrusted level signal: it no longer follows the process at all (replay / freeze).
            worlds.append({"hypothesis": {**hv, "weight": hv["weight"] + " (decoupled variant)"},
                           "scenarios": [ScenarioSpec(type="SENSOR_FREEZE", target=sid, start_s=0)],
                           "notes": [f"Decoupled variant: {sid} reading stops following the process (replay/freeze worst case)."]})
    out = []
    for pol in ("NONE",) + tuple(p for p in RESPONSE_POLICIES if p != "NONE"):
        fails = check_preconditions(pol, defender) if pol != "NONE" else []
        per = []
        for w in worlds:
            pred = project(cfg, defender, pol, w["scenarios"], horizon_s)
            per.append({"hypothesis": w["hypothesis"]["leading"], "component": w["hypothesis"]["component"], **pred})
        worst = {k: max(p[k] for p in per) for k in WORST}
        worst["predicted_min_level_m"] = min(p["predicted_min_level_m"] for p in per)
        out.append({"policy": pol, "description": POLICY_INFO.get(pol, "No intervention (baseline)."),
                    "preconditions_ok": not fails, "precondition_failures": fails, **worst,
                    "trajectory": per[0]["trajectory"], "per_hypothesis": per})

    benign = (verdict.get("state") == "NOMINAL") or verdict.get("leading") in ("LEGITIMATE_DISTURBANCE", "TWIN_MODEL_MISMATCH")

    def key(c):
        # Worst-case safety first (overflow, unsafe time), then service/band, then a fixed cost for intervening
        # at all (larger when the evidence points to a legitimate cause).
        intervene = 0.0 if c["policy"] == "NONE" else (40.0 if benign else 2.0)
        return (not c["preconditions_ok"], c["predicted_overflow_L"] > 0.5, round(c["predicted_unsafe_duration_s"], 0),
                c["predicted_out_of_band_s"] * 0.2 + c["predicted_unmet_demand_L"] * 0.05 + intervene)

    ranked = sorted(out, key=key)
    rec = ranked[0]["policy"]
    base = next(c for c in out if c["policy"] == "NONE")
    for c in out:
        c["recommended"] = c["policy"] == rec
        c["tradeoffs"] = _tradeoffs(c, base)
    return {"hypotheses_considered": [{**w["hypothesis"], "assumed_scenario": [s.model_dump() for s in w["scenarios"]],
                                       "assumptions": w["notes"]} for w in worlds],
            "assumed_scenario": [s.model_dump() for s in worlds[0]["scenarios"]],
            "assumptions": [n for w in worlds for n in w["notes"]] + ["Ranking uses the worst case across the hypotheses considered."],
            "horizon_s": horizon_s, "provenance": "DIGITAL_TWIN_PREDICTION", "recommended_policy": rec,
            "candidates": out, "ranking": "minimax over competing hypotheses: overflow, unsafe time, then band/service/intervention cost",
            "safety_boundary": "Simulated interventions only. No commands are issued to real equipment.", "verdict": verdict}


def _tradeoffs(c: dict, base: dict) -> list[str]:
    t = []
    d_ovf = c["predicted_overflow_L"] - base["predicted_overflow_L"]
    d_unmet = c["predicted_unmet_demand_L"] - base["predicted_unmet_demand_L"]
    d_unsafe = c["predicted_unsafe_duration_s"] - base["predicted_unsafe_duration_s"]
    if abs(d_ovf) > 0.5:
        t.append(f"overflow {'+' if d_ovf > 0 else ''}{d_ovf:.0f} L vs no action")
    if abs(d_unsafe) > 0.5:
        t.append(f"unsafe time {'+' if d_unsafe > 0 else ''}{d_unsafe:.0f} s")
    if abs(d_unmet) > 5:
        t.append(f"unmet demand {'+' if d_unmet > 0 else ''}{d_unmet:.0f} L" + (" (ADVERSE: service loss)" if d_unmet > 50 else ""))
    return t or ["no material difference predicted"]


def recommended_policy_for(verdict: dict) -> str:
    """Static fallback mapping (used when projection is skipped)."""
    lead = verdict.get("leading")
    comp = verdict.get("component") or ""
    if lead in ("SENSOR_SPOOFING", "SENSOR_FAILURE"):
        return "FALLBACK_TO_CAMERA_OR_TWIN_ESTIMATE"
    if lead == "COMMUNICATION_MANIPULATION_OR_DELAY":
        return "ISOLATE_REMOTE_COMMUNICATION_PATH"
    if lead == "CONTROLLER_MANIPULATION":
        return "BOUNDED_COMMAND_LIMITING"
    if lead == "ACTUATOR_FAULT":
        return "BOUNDED_COMMAND_LIMITING" if comp == "FV-101" else "CONTROLLED_SAFE_SHUTDOWN"
    if lead in ("LEGITIMATE_DISTURBANCE", "TWIN_MODEL_MISMATCH"):
        return "NONE"
    return "BOUNDED_COMMAND_LIMITING"
