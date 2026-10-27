"""Robustness & ablation lab (FR-031).

Ablation modes
  1 TWIN_RESIDUALS_ONLY     evidence restricted to twin residual features
  2 WATERMARK_ONLY          evidence restricted to watermark correlation features
  3 MULTIMODAL_FUSION       all passive evidence (residuals, watermark, camera, network, controller, mass balance)
  4 FUSION_ACTIVE           3 + up to two EIG-ranked active diagnostics after the alarm
  5 FULL_SYSTEM_RESPONSE    4 + planner-selected simulated response (physical impact measured vs. no response)

Ground-truth labels come from the scenario ledger and are used ONLY here, after
the defenders have produced their outputs.

Detection target ("positive"): an attack or equipment fault (sensor, communication,
controller, actuator families). Legitimate disturbances, model mismatch and
scenario-free runs are negatives. A run is predicted positive when an alarm is
raised in [onset, onset + decision window] and the verdict is not
LEGITIMATE_DISTURBANCE / TWIN_MODEL_MISMATCH (INCONCLUSIVE escalates -> positive).
"""
from __future__ import annotations

import copy
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable

from pydantic import BaseModel, Field

from ..config import SimConfig
from ..sim.scenarios import CATALOG, ScenarioSpec
from .harness import directive_for, make_episode, refresh_directive
from .metrics import compute_metrics

MODES = ("TWIN_RESIDUALS_ONLY", "WATERMARK_ONLY", "MULTIMODAL_FUSION", "FUSION_ACTIVE", "FULL_SYSTEM_RESPONSE")
POSITIVE_FAMILIES = ("SENSOR", "COMM", "CONTROLLER", "ACTUATOR")
NEGATIVE_VERDICTS = ("LEGITIMATE_DISTURBANCE", "TWIN_MODEL_MISMATCH")

PRESETS = {
    "quick": ["SENSOR_BIAS", "SENSOR_REPLAY", "SENSOR_FREEZE", "COMM_DELAY", "COMM_MITM_BIAS", "CTRL_SETPOINT_OVERRIDE",
              "VALVE_STUCK", "PUMP_TRIP", "DEMAND_SURGE", "TWIN_MODEL_MISMATCH", "NONE", "NONE"],
    "standard": list(CATALOG) + ["NONE", "NONE", "NONE"],
}


class AblationRequest(BaseModel):
    preset: str | None = "quick"
    scenario_types: list[str] | None = None
    noise_levels: list[float] = Field(default_factory=lambda: [1.0])
    latencies_s: list[float] = Field(default_factory=lambda: [0.05])
    seeds: list[int] = Field(default_factory=lambda: [11])
    onset_s: float = 90.0
    decision_window_s: float = 90.0
    duration_s: float = 420.0
    max_workers: int | None = None


def expected_component(spec: ScenarioSpec) -> str:
    fam = spec.family
    if fam == "COMM":
        return f"NET-1/{spec.target}"
    return spec.target or CATALOG[spec.type]["target"]


def _verdict_tuple(d) -> tuple[str, str | None, str | None]:
    v = d.result.get("verdict", {})
    return v.get("state", "NOMINAL"), v.get("leading"), v.get("component")


def _mode_outcome(d, onset: float, decision_t: float) -> dict:
    alarm_in = [a for a in d.alarm_history if onset - 1.0 <= a["raised_s"] <= decision_t]
    pre = [a for a in d.alarm_history if a["raised_s"] < onset - 1.0]
    state, lead, comp = _verdict_tuple(d)
    detected = bool(alarm_in) or (d.alarm_onset_s is not None and d.alarm_onset_s <= decision_t and d.alarm_onset_s >= onset - 1)
    predicted_positive = detected and state != "NOMINAL" and lead not in NEGATIVE_VERDICTS
    lat = (alarm_in[0]["raised_s"] - onset) if alarm_in else None
    return {"detected": detected, "pre_onset_alarm": bool(pre), "predicted_positive": predicted_positive,
            "verdict_state": state, "predicted_type": lead if state == "LEADING_HYPOTHESIS" else
            ("INCONCLUSIVE" if state == "INCONCLUSIVE" else None),
            "predicted_component": comp if state == "LEADING_HYPOTHESIS" else None,
            "detection_latency_s": None if lat is None else round(max(lat, 0.0), 2)}


def evaluate_case(case: dict) -> dict:
    t0 = time.perf_counter()
    cfg = SimConfig(seed=case["seed"])
    cfg.sensors.level_noise_m *= case["noise"]
    cfg.sensors.flow_noise_lps *= case["noise"]
    cfg.sensors.valve_pos_noise_pct *= case["noise"]
    cfg.sensors.pump_speed_noise *= case["noise"]
    cfg.network.base_latency_s = case["latency"]
    onset, decision_t, end_t = case["onset"], case["onset"] + case["window"], case["duration"]
    spec = None if case["type"] == "NONE" else ScenarioSpec(type=case["type"], start_s=onset, seed=case["seed"]).resolved()
    ep = make_episode(cfg, [spec] if spec else [], modes=("RESIDUAL_ONLY", "WATERMARK_ONLY", "FULL"), camera=True)
    full = ep.defenders["FULL"]
    branch = None
    while ep.world.t < decision_t - 1e-9:
        ep.step()
        if branch is None and full.first_alarm_s is not None and full.first_alarm_s >= onset - 1.0 \
                and ep.world.t >= full.first_alarm_s + 5.0:
            branch = copy.deepcopy(ep)
    out_modes = {
        "TWIN_RESIDUALS_ONLY": _mode_outcome(ep.defenders["RESIDUAL_ONLY"], onset, decision_t),
        "WATERMARK_ONLY": _mode_outcome(ep.defenders["WATERMARK_ONLY"], onset, decision_t),
        "MULTIMODAL_FUSION": _mode_outcome(full, onset, decision_t),
    }
    ep.run_until(end_t)
    metrics_a = compute_metrics(ep.truth, cfg, onset if spec else None, None)

    diag_run: list[dict] = []
    response_info: dict[str, Any] = {"policy": None}
    if branch is None:
        out_modes["FUSION_ACTIVE"] = dict(out_modes["MULTIMODAL_FUSION"])
        out_modes["FULL_SYSTEM_RESPONSE"] = dict(out_modes["MULTIMODAL_FUSION"])
        response_info["metrics_b"] = metrics_a
    else:
        bd = branch.defenders["FULL"]
        for _ in range(2):
            ranked = [r for r in bd.recommend() if r["executable"] and r["expected_information_gain_bits"] > 0.05]
            if not ranked:
                break
            job = bd.run_diagnostic(ranked[0]["test_id"], branch.world)
            if job.get("status") == "RUNNING":
                branch.run_until(job["due_s"] + branch.world.dt)
            diag_run.append({"test_id": ranked[0]["test_id"], "outcome": bd.diag_log[-1].get("outcome")})
        resp = copy.deepcopy(branch)
        branch.run_until(max(decision_t, branch.world.t))
        out_modes["FUSION_ACTIVE"] = _mode_outcome(bd, onset, max(decision_t, branch.world.t))
        rd = resp.defenders["FULL"]
        from ..response.planner import plan
        p = plan(cfg, rd, horizon_s=300.0)
        policy = p["recommended_policy"]
        t_int = resp.world.t
        if policy != "NONE":
            resp.world.set_response(directive_for(policy, rd, t_int))
        resp_decision = max(decision_t, resp.world.t)
        while resp.world.t < end_t - 1e-9:
            resp.step()
            refresh_directive(resp.world, rd)
            if abs(resp.world.t - resp_decision) < resp.world.dt / 2:
                out_modes["FULL_SYSTEM_RESPONSE"] = _mode_outcome(rd, onset, resp_decision)
        out_modes.setdefault("FULL_SYSTEM_RESPONSE", _mode_outcome(rd, onset, resp_decision))
        response_info = {"policy": policy, "intervention_time_s": t_int if policy != "NONE" else None,
                         "metrics_b": compute_metrics(resp.truth, cfg, onset if spec else None, t_int if policy != "NONE" else None)}
    positive = bool(spec) and spec.family in POSITIVE_FAMILIES
    truth_type = spec.expected_hypothesis if spec else None
    truth_comp = expected_component(spec) if spec else None
    for m, o in out_modes.items():
        o["correct_type"] = bool(positive and o["predicted_type"] == truth_type)
        o["correct_component"] = bool(positive and o["predicted_component"] == truth_comp)
    mb = response_info.get("metrics_b") or metrics_a
    return {
        "case": case, "ground_truth": {"type": case["type"], "positive": positive, "expected_hypothesis": truth_type,
                                       "expected_component": truth_comp, "provenance": "SIMULATION_GROUND_TRUTH"},
        "modes": out_modes, "diagnostics_run": diag_run,
        "impact": {"policy": response_info.get("policy"), "intervention_time_s": response_info.get("intervention_time_s"),
                   "overflow_A_L": metrics_a["overflow_volume_L"], "overflow_B_L": mb["overflow_volume_L"],
                   "unsafe_A_s": metrics_a["unsafe_duration_s"], "unsafe_B_s": mb["unsafe_duration_s"],
                   "unmet_A_L": metrics_a["unmet_demand_L"], "unmet_B_L": mb["unmet_demand_L"],
                   "max_level_A_m": metrics_a["max_tank_level_m"], "max_level_B_m": mb["max_tank_level_m"]},
        "runtime_s": round(time.perf_counter() - t0, 2),
    }


def _ratio(a: float, b: float) -> float | None:
    return None if b == 0 else round(a / b, 4)


def aggregate(results: list[dict]) -> dict:
    table = {}
    for m in MODES:
        tp = fp = fn = tn = 0
        lat, loc, cls, pre = [], [], [], 0
        for r in results:
            o = r["modes"][m]
            pos = r["ground_truth"]["positive"]
            pre += int(o["pre_onset_alarm"])
            if pos and o["predicted_positive"]:
                tp += 1
                if o["detection_latency_s"] is not None:
                    lat.append(o["detection_latency_s"])
                loc.append(o["correct_component"])
                cls.append(o["correct_type"])
            elif pos:
                fn += 1
            elif o["predicted_positive"]:
                fp += 1
            else:
                tn += 1
        prec, rec = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
        f1 = None if not prec or not rec else round(2 * prec * rec / (prec + rec), 4)
        row = {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": prec, "recall": rec, "f1": f1,
               "false_alarm_rate": _ratio(fp, fp + tn), "pre_onset_false_alarms": pre,
               "mean_detection_latency_s": round(sum(lat) / len(lat), 2) if lat else None,
               "localization_accuracy": _ratio(sum(loc), len(loc)), "classification_accuracy": _ratio(sum(cls), len(cls))}
        if m == "FULL_SYSTEM_RESPONSE":
            imp = [r["impact"] for r in results]
            row["mean_overflow_avoided_L"] = round(sum(i["overflow_A_L"] - i["overflow_B_L"] for i in imp) / max(len(imp), 1), 2)
            row["mean_unsafe_time_avoided_s"] = round(sum(i["unsafe_A_s"] - i["unsafe_B_s"] for i in imp) / max(len(imp), 1), 2)
            row["adverse_response_cases"] = sum(1 for i in imp if i["unmet_B_L"] - i["unmet_A_L"] > 50 or i["overflow_B_L"] > i["overflow_A_L"] + 0.5
                                                or i["unsafe_B_s"] > i["unsafe_A_s"] + 0.5)
        table[m] = row
    return table


def build_cases(req: AblationRequest) -> list[dict]:
    types = req.scenario_types or PRESETS.get(req.preset or "quick", PRESETS["quick"])
    cases = []
    for (i, typ), noise, lat, seed in itertools.product(enumerate(types), req.noise_levels, req.latencies_s, req.seeds):
        cases.append({"type": typ, "noise": noise, "latency": lat, "seed": seed + i * 101, "onset": req.onset_s,
                      "window": req.decision_window_s, "duration": req.duration_s})
    return cases


def run_ablation(req: AblationRequest, progress: Callable[[int, int], None] | None = None) -> dict:
    t0 = time.perf_counter()
    cases = build_cases(req)
    results: list[dict] = []
    workers = req.max_workers or max(1, min(len(cases), (os.cpu_count() or 2) - 1))
    if workers == 1:
        for i, c in enumerate(cases):
            results.append(evaluate_case(c))
            if progress:
                progress(i + 1, len(cases))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(evaluate_case, c) for c in cases]
            for i, f in enumerate(as_completed(futs)):
                results.append(f.result())
                if progress:
                    progress(i + 1, len(cases))
    results.sort(key=lambda r: (r["case"]["type"], r["case"]["noise"], r["case"]["latency"], r["case"]["seed"]))
    return {"request": req.model_dump(), "n_cases": len(cases), "matrix": aggregate(results), "cases": results,
            "modes": list(MODES), "runtime_s": round(time.perf_counter() - t0, 2), "workers": workers,
            "definitions": __doc__, "provenance": "SIMULATION_GROUND_TRUTH labels used for scoring only"}
