"""Paired counterfactual impact twin (FR-024 – FR-026, AT-004 – AT-006).

World A (unprotected) and World B (protected) are two *independent* World +
Defender instances constructed from the same initial state x0, seed S, scenario
and exogenous disturbance generator. They are stepped in lockstep; only World B
receives the response policy at t_int. Before t_int both trajectories are
bit-identical; the isolation check proves it.
"""
from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from ..config import MODEL_VERSION, TWIN_VERSION, INFERENCE_VERSION, SimConfig
from ..sim.scenarios import ScenarioSpec
from ..sim.world import RESPONSE_POLICIES
from .harness import directive_for, make_episode, refresh_directive
from .metrics import compare, compute_metrics


class CounterfactualRequest(BaseModel):
    config: SimConfig = Field(default_factory=SimConfig)
    scenarios: list[ScenarioSpec] = Field(default_factory=list)
    policy: str = "BOUNDED_COMMAND_LIMITING"
    trigger: Literal["FIXED_TIME", "ON_DETECTION"] = "ON_DETECTION"
    intervention_time_s: float | None = None  # FIXED_TIME
    detection_delay_s: float = 5.0  # ON_DETECTION: extra delay after the alarm
    duration_s: float = Field(400.0, gt=10, le=3600)
    camera: bool = True
    sample_hz: float = 2.0


def run_counterfactual(req: CounterfactualRequest) -> dict[str, Any]:
    if req.policy not in RESPONSE_POLICIES:
        raise ValueError(f"policy must be one of {RESPONSE_POLICIES}")
    if req.trigger == "FIXED_TIME" and req.intervention_time_s is None:
        raise ValueError("intervention_time_s required for FIXED_TIME")
    t0 = time.perf_counter()
    cfg = req.config
    scen_a = [s.model_copy(deep=True) for s in req.scenarios]
    scen_b = [s.model_copy(deep=True) for s in req.scenarios]
    ep_a = make_episode(cfg.model_copy(deep=True), scen_a, camera=req.camera)
    ep_b = make_episode(cfg.model_copy(deep=True), scen_b, camera=req.camera)
    da, db = ep_a.defenders["FULL"], ep_b.defenders["FULL"]
    onset = min((s.start_s for s in req.scenarios), default=None)
    t_int: float | None = None
    trigger_note = None
    steps = int(round(req.duration_s / cfg.dt_s))
    decim = max(1, int(round(1.0 / (req.sample_hz * cfg.dt_s))))
    series: dict[str, list] = {k: [] for k in ("t", "A_h", "B_h", "A_lt", "B_lt", "A_twin", "B_twin", "A_qin", "B_qin",
                                               "A_qout", "B_qout", "A_u", "B_u", "A_overflow_L", "B_overflow_L",
                                               "A_unmet_L", "B_unmet_L")}
    pre_dev = 0.0
    for i in range(steps):
        oa, ta = ep_a.step()
        ob, tb = ep_b.step()
        if t_int is None:
            pre_dev = max(pre_dev, abs(ta["h"] - tb["h"]), abs(ta["q_in_lps"] - tb["q_in_lps"]))
            fire = False
            if req.trigger == "FIXED_TIME":
                fire = ep_b.world.t >= req.intervention_time_s - 1e-9
            elif db.first_alarm_s is not None and ep_b.world.t >= db.first_alarm_s + req.detection_delay_s - 1e-9:
                fire = True
            if fire:
                t_int = ep_b.world.t
                ep_b.world.set_response(directive_for(req.policy, db, t_int))
        else:
            refresh_directive(ep_b.world, db)
        if i % decim == 0:
            series["t"].append(round(ta["t"], 3))
            for w, tr, o, d in (("A", ta, oa, da), ("B", tb, ob, db)):
                series[f"{w}_h"].append(round(tr["h"], 4))
                lt = o["recv"]["LT-101"]
                series[f"{w}_lt"].append(None if lt != lt else round(lt, 4))
                series[f"{w}_twin"].append(round(d.hist.last("yhat:LT-101"), 4))
                series[f"{w}_qin"].append(round(tr["q_in_lps"], 3))
                series[f"{w}_qout"].append(round(tr["q_out_lps"], 3))
                series[f"{w}_u"].append(round(o["u_cmd"], 4))
                series[f"{w}_overflow_L"].append(round(tr["overflow_m3"] * 1000, 2))
                series[f"{w}_unmet_L"].append(round(tr["unmet_m3"] * 1000, 2))
    if t_int is None:
        trigger_note = ("No anomaly alarm in World B: response never triggered; worlds remain identical."
                        if req.trigger == "ON_DETECTION" else "Intervention time beyond horizon.")
    ma = compute_metrics(ep_a.truth, cfg, onset, None)
    mb = compute_metrics(ep_b.truth, cfg, onset, t_int)
    cmp = compare(ma, mb)
    ex_a = np.array(ep_a.world.exogenous_log)
    ex_b = np.array(ep_b.world.exogenous_log)
    isolation = {
        "same_seed": True,
        "initial_state_identical": ep_a.truth[0]["h_prev"] == ep_b.truth[0]["h_prev"],
        "exogenous_disturbance_max_abs_diff": float(np.max(np.abs(ex_a - ex_b))) if ex_a.size else 0.0,
        "pre_intervention_max_abs_state_diff": pre_dev,
        "independent_instances": ep_a.world is not ep_b.world and ep_a.world.plant is not ep_b.world.plant,
    }
    isolation["valid"] = (isolation["initial_state_identical"] and isolation["exogenous_disturbance_max_abs_diff"] == 0.0
                          and pre_dev == 0.0 and isolation["independent_instances"])
    return {
        "world_a": {"label": "World A — unprotected baseline", "metrics": ma,
                    "final_verdict": da.result.get("verdict"), "first_alarm_s": da.first_alarm_s},
        "world_b": {"label": f"World B — protected ({req.policy})", "metrics": mb, "policy": req.policy,
                    "final_verdict": db.result.get("verdict"), "first_alarm_s": db.first_alarm_s},
        "comparison": cmp,
        "intervention_time_s": t_int,
        "attack_onset_s": onset,
        "trigger": req.trigger,
        "trigger_note": trigger_note,
        "isolation_check": isolation,
        "series": series,
        "provenance": "SIMULATION_GROUND_TRUTH",
        "provenance_note": "Physical impact metrics are computed from the simulator's ground-truth trajectories of both worlds.",
        "model_versions": {"plant": MODEL_VERSION, "twin": TWIN_VERSION, "inference": INFERENCE_VERSION},
        "runtime_s": round(time.perf_counter() - t0, 3),
        "limitations": [
            "Single-tank plant model; pump/valve curves are idealised.",
            "SIMULATED_VISUAL camera used inside both worlds (explicitly synthetic).",
            "Response policies act through an idealised independent safety layer.",
        ],
    }
