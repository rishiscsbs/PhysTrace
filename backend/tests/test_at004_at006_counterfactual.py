"""AT-004 counterfactual isolation; AT-005 metric computation & zero denominators; AT-006 adverse outcomes."""
import json
import math

from phystrace.evaluation.counterfactual import CounterfactualRequest, run_counterfactual
from phystrace.evaluation.metrics import compare, guarded_relative
from phystrace.sim.scenarios import ScenarioSpec


def _finite(o):
    if isinstance(o, dict):
        return all(_finite(v) for v in o.values())
    if isinstance(o, list):
        return all(_finite(v) for v in o)
    if isinstance(o, float):
        return math.isfinite(o)
    return True


def test_at004_paired_worlds_are_isolated_and_identical_before_intervention():
    req = CounterfactualRequest(scenarios=[ScenarioSpec(type="CTRL_SETPOINT_OVERRIDE", start_s=40)],
                                policy="BOUNDED_COMMAND_LIMITING", trigger="FIXED_TIME", intervention_time_s=70.0,
                                duration_s=200.0, camera=False)
    r = run_counterfactual(req)
    iso = r["isolation_check"]
    assert iso["valid"] and iso["independent_instances"]
    assert iso["exogenous_disturbance_max_abs_diff"] == 0.0
    assert iso["pre_intervention_max_abs_state_diff"] == 0.0
    assert r["intervention_time_s"] == 70.0
    s = r["series"]
    idx_pre = [i for i, t in enumerate(s["t"]) if t < 70.0]
    idx_post = [i for i, t in enumerate(s["t"]) if t > 120.0]
    assert all(s["A_h"][i] == s["B_h"][i] for i in idx_pre)
    assert any(abs(s["A_h"][i] - s["B_h"][i]) > 0.05 for i in idx_post)


def test_at005_metrics_computed_from_trajectories_and_finite():
    req = CounterfactualRequest(scenarios=[ScenarioSpec(type="CTRL_SETPOINT_OVERRIDE", start_s=30, params={"setpoint_m": 3.0})],
                                policy="BOUNDED_COMMAND_LIMITING", trigger="FIXED_TIME", intervention_time_s=60.0,
                                duration_s=420.0, camera=False)
    r = run_counterfactual(req)
    a, b = r["world_a"]["metrics"], r["world_b"]["metrics"]
    for k in ("max_tank_level_m", "overflow_volume_L", "water_loss_L", "energy_consumption_kWh", "unsafe_duration_s",
              "safety_violation_count", "recovery_time_s", "intervention_time_s", "attack_to_mitigation_delay_s"):
        assert k in a and k in b
    assert a["max_tank_level_m"] > b["max_tank_level_m"]
    assert a["unsafe_duration_s"] > 0 and b["unsafe_duration_s"] == 0
    assert b["attack_to_mitigation_delay_s"] == 30.0 and a["intervention_time_s"] is None
    assert r["comparison"]["deltas"]["unsafe_duration_s"]["outcome"] == "IMPROVED"
    assert _finite(r) and "NaN" not in json.dumps(r) and "Infinity" not in json.dumps(r)


def test_at005_guarded_relative_handles_zero_and_missing():
    assert guarded_relative(0.0, 0.0) == {"delta": 0.0, "relative": 0.0, "note": "both zero"}
    z = guarded_relative(0.0, 12.5)
    assert z["delta"] == 12.5 and z["relative"] is None and "undefined" in z["note"]
    assert guarded_relative(None, 3.0)["delta"] is None
    assert guarded_relative(10.0, 5.0)["relative"] == -0.5
    c = compare({"recovery_time_s": None, "overflow_volume_L": 0.0}, {"recovery_time_s": 12.0, "overflow_volume_L": 0.0})
    assert c["deltas"]["recovery_time_s"]["outcome"] == "IMPROVED"
    assert c["deltas"]["overflow_volume_L"]["relative"] == 0.0


def test_at006_inappropriate_response_is_reported_as_adverse():
    # Premature shutdown during a legitimate demand surge: no safety gain, large supply loss.
    req = CounterfactualRequest(scenarios=[ScenarioSpec(type="DEMAND_SURGE", start_s=40)], policy="CONTROLLED_SAFE_SHUTDOWN",
                                trigger="FIXED_TIME", intervention_time_s=50.0, duration_s=300.0, camera=False)
    r = run_counterfactual(req)
    cmp = r["comparison"]
    assert cmp["adverse_outcome"] is True
    assert "unmet_demand_L" in cmp["adverse_metrics"]
    assert cmp["deltas"]["unmet_demand_L"]["outcome"] == "WORSENED"
    assert r["world_b"]["metrics"]["unmet_demand_L"] > 10 * max(r["world_a"]["metrics"]["unmet_demand_L"], 1.0)
