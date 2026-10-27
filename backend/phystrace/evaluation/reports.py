"""Structured investigation reports and exports (FR-030, FR-035)."""
from __future__ import annotations

import csv
import io
import time
from typing import Any

from ..config import INFERENCE_VERSION, MODEL_VERSION, TWIN_VERSION
from ..persistence.db import Database

PROVENANCE_LEGEND = {
    "SENSOR_OBSERVATION": "Value reported by a field instrument as received by the PLC/historian.",
    "CAMERA_VISUAL_ESTIMATE": "Waterline estimated by OpenCV from the selected camera source (live, replay or SIMULATED_VISUAL).",
    "DIGITAL_TWIN_PREDICTION": "Physics-model prediction driven by logged commands.",
    "INFERENCE": "Derived evidence, hypothesis evidence scores and verdicts (uncalibrated).",
    "SIMULATION_GROUND_TRUTH": "Simulator truth; used only for evaluation and physical-impact metrics.",
}

LIMITATIONS = [
    "Single-tank water-treatment plant; idealised pump curve and valve discharge model.",
    "Evidence scores are auditable heuristics, not calibrated probabilities.",
    "Response actions act only on the local simulator through an idealised independent safety layer.",
    "SIMULATED_VISUAL camera frames are synthetic renders and are labelled as such.",
    "Live camera estimates refer to whatever vessel is in view; they are not linked to the simulated plant's physics.",
]


def build_report(db: Database, run_id: str, include_ground_truth: bool = False) -> dict[str, Any]:
    run = db.get_run(run_id)
    if run is None:
        raise KeyError(run_id)
    events = db.events(run_id)
    snaps = db.hypothesis_snapshots(run_id)
    final = snaps[-1] if snaps else None
    verdicts = [{"t": s["timestamp"], **(s["verdict"] or {})} for s in snaps]
    report: dict[str, Any] = {
        "report_type": "PHYSTRACE_INVESTIGATION_REPORT",
        "generated_at": time.time(),
        "run": {k: run[k] for k in ("run_id", "created_at", "seed", "status", "label", "sim_time_s")},
        "configuration": run["config"],
        "model_versions": run.get("model_versions") or {"plant": MODEL_VERSION, "twin": TWIN_VERSION, "inference": INFERENCE_VERSION},
        "final_assessment": final,
        "verdict_history": verdicts,
        "diagnostics": db.diagnostics(run_id),
        "responses": db.responses(run_id),
        "events": events,
        "score_semantics": "EVIDENCE_SCORE: normalised, uncalibrated ranking aid. Not a probability.",
        "assumptions": sorted({a for s in ([final] if final else []) for h in s["hypotheses"] for a in h["assumptions"]}),
        "limitations": LIMITATIONS,
        "provenance_legend": PROVENANCE_LEGEND,
        "safety_boundary": "All attacks and responses were confined to the local simulator.",
    }
    if include_ground_truth:
        gt = db.ground_truth(run_id)
        report["evaluation"] = {"provenance": "SIMULATION_GROUND_TRUTH", "scenarios": gt,
                                "note": "Revealed for post-run evaluation only; never used by the inference engine."}
        if final and gt:
            from ..sim.scenarios import CATALOG, ScenarioSpec
            from .ablation import expected_component
            specs = [ScenarioSpec(**g["scenario"]) for g in gt if g["scenario"].get("type") in CATALOG]
            v = final["verdict"] or {}
            report["evaluation"]["comparison"] = [{
                "scenario_id": s.scenario_id, "type": s.type, "expected_hypothesis": s.expected_hypothesis,
                "expected_component": expected_component(s), "final_verdict": v.get("leading") or v.get("state"),
                "final_component": v.get("component"), "type_match": v.get("leading") == s.expected_hypothesis,
                "component_match": v.get("component") == expected_component(s)} for s in specs]
    return report


def telemetry_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})
    return buf.getvalue()


def experiment_csv(exp: dict) -> str:
    res = exp.get("result") or {}
    if exp.get("kind") == "COUNTERFACTUAL":
        s = res.get("series") or {}
        keys = list(s.keys())
        n = len(s.get("t", []))
        rows = [{k: s[k][i] for k in keys} for i in range(n)]
        return telemetry_csv(rows)
    if exp.get("kind") == "ABLATION":
        rows = []
        for c in res.get("cases", []):
            for m, o in c["modes"].items():
                rows.append({"scenario": c["case"]["type"], "noise": c["case"]["noise"], "latency_s": c["case"]["latency"],
                             "seed": c["case"]["seed"], "mode": m, "positive": c["ground_truth"]["positive"],
                             "expected_hypothesis": c["ground_truth"]["expected_hypothesis"],
                             "expected_component": c["ground_truth"]["expected_component"], **o})
        return telemetry_csv(rows)
    return ""
