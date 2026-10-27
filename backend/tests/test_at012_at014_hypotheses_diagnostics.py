"""AT-012 inconclusive handling; AT-013 evidence-score semantics & anti double counting;
AT-014 diagnostic execution appends an Observation and recomputes scores."""
from conftest import run_episode

from phystrace.inference.hypotheses import score
from phystrace.schemas import EvidenceItem


def _ev(feature, comp, strength, group, source="SENSOR"):
    return EvidenceItem(evidence_id=f"EV:{feature}@{comp}", feature=feature, component_id=comp, strength=strength,
                        description="t", independence_group=group, source_type=source, timestamp=0.0)


def test_at012_no_anomaly_is_nominal_not_ranked():
    r = score([], anomaly_active=False)
    assert r["verdict"]["state"] == "NOMINAL"
    assert all(h["status"] == "NOT_EVALUATED" for h in r["hypotheses"])


def test_at012_weak_evidence_is_inconclusive():
    r = score([_ev("residual_step:LT-101", "LT-101", 0.2, "LT-101", "TWIN_PREDICTION")], anomaly_active=True)
    assert r["verdict"]["state"] == "INCONCLUSIVE"
    inc = next(h for h in r["hypotheses"] if h["hypothesis_type"] == "INCONCLUSIVE")
    assert inc["status"] == "INCONCLUSIVE" and inc["missing_evidence"]


def test_at012_symmetric_tie_is_inconclusive():
    # A clean LT-101 residual step alone cannot separate in-transit MITM from sensor spoofing well.
    ev = [_ev("redundant_disagrees", "LT-101", 1.0, "LT-102")]
    r = score(ev, anomaly_active=True)
    assert r["verdict"]["state"] == "INCONCLUSIVE"
    assert "tied" in r["verdict"]["reason"]


def test_at012_live_episode_without_attack_never_alarms():
    ep = run_episode(None, seed=21, until=300)
    assert ep.defenders["FULL"].first_alarm_s is None
    assert ep.defenders["FULL"].result["verdict"]["state"] == "NOMINAL"


def test_at013_scores_are_evidence_scores_not_probabilities():
    ep = run_episode("VALVE_STUCK", until=160)
    res = ep.defenders["FULL"].result
    assert res["score_type"] == "EVIDENCE_SCORE" and "Not calibrated" in res["score_note"]
    hyps = [h for h in res["hypotheses"] if h["hypothesis_type"] != "INCONCLUSIVE"]
    assert all(h["score_type"] == "EVIDENCE_SCORE" for h in res["hypotheses"])
    assert abs(sum(h["score"] for h in hyps) - 1.0) < 1e-3
    lead = hyps[0]
    assert lead["supporting_evidence_refs"] and "assumptions" in lead and "missing_evidence" in lead
    assert {h["hypothesis_type"] for h in res["hypotheses"]} == {
        "SENSOR_SPOOFING", "SENSOR_FAILURE", "COMMUNICATION_MANIPULATION_OR_DELAY", "CONTROLLER_MANIPULATION",
        "ACTUATOR_FAULT", "LEGITIMATE_DISTURBANCE", "TWIN_MODEL_MISMATCH", "INCONCLUSIVE"}


def test_at013_correlated_evidence_is_not_double_counted():
    one = [_ev("residual_step:LT-101", "LT-101", 1.0, "LT-101", "TWIN_PREDICTION")]
    dup = one + [_ev("residual_ramp:LT-101", "LT-101", 1.0, "LT-101", "TWIN_PREDICTION"),
                 _ev("wm_absent:LT-101", "LT-101", 1.0, "LT-101", "WATERMARK")]
    indep = one + [_ev("camera_disagrees:LT-101", "LT-101", 1.0, "CAM-01", "CAMERA_ESTIMATE")]

    def raw(ev):
        return next(c["raw_score"] for c in score(ev, True)["candidates"]
                    if c["hypothesis_type"] == "SENSOR_SPOOFING" and c["component_id"] == "LT-101")

    same_stream_gain = raw(dup) - raw(one)
    independent_gain = raw(indep) - raw(one)
    assert same_stream_gain < 0.25 * (0.6 + 2.0) + 1e-9  # diminishing returns inside one data stream
    assert independent_gain >= 1.5 - 1e-9  # an independent modality counts fully


def test_at014_diagnostic_appends_observation_and_updates_scores():
    ep = run_episode("COMM_MITM_BIAS", until=120)
    d = ep.defenders["FULL"]
    assert d.result["verdict"]["leading"] in ("SENSOR_SPOOFING", None)
    d.emit = True
    n_obs = len(d.observations)
    before = {h["hypothesis_type"]: h["raw_score"] for h in d.result["hypotheses"]}
    res = d.run_diagnostic("READ_FIELD_DEVICE_DIRECT", ep.world)
    assert res["status"] == "COMPLETE" and res["outcome"] == "MISMATCH"
    assert len(d.observations) > n_obs
    assert any(o.metadata.get("diagnostic_test") == "READ_FIELD_DEVICE_DIRECT" for o in d.observations)
    after = {h["hypothesis_type"]: h["raw_score"] for h in d.result["hypotheses"]}
    assert after["COMMUNICATION_MANIPULATION_OR_DELAY"] > before["COMMUNICATION_MANIPULATION_OR_DELAY"] + 1.0
    assert d.result["verdict"]["leading"] == "COMMUNICATION_MANIPULATION_OR_DELAY"
    assert res["scores_before"] != res["scores_after"]


def test_at014_ranking_uses_eig_and_preconditions():
    ep = run_episode("SENSOR_BIAS", until=120)
    d = ep.defenders["FULL"]
    ranking = d.recommend()
    assert {r["test_id"] for r in ranking} >= {"POLL_REDUNDANT_SENSOR", "ACTIVATE_WATERMARK_BURST", "CAPTURE_VISUAL_WITNESS_FRAME",
                                               "VERIFY_PLC_SETPOINT_REGISTER", "RUN_SHORT_HORIZON_TWIN_CHECK"}
    assert all("expected_information_gain_bits" in r and "utility" in r for r in ranking)
    top = ranking[0]
    assert top["executable"]
    d.run_diagnostic(top["test_id"], ep.world)
    again = next(r for r in d.recommend() if r["test_id"] == top["test_id"])
    assert again["expected_information_gain_bits"] == 0.0  # already-acquired information is not re-recommended


def test_at014_extended_test_completes_after_duration():
    ep = run_episode("SENSOR_REPLAY", until=120)
    d = ep.defenders["FULL"]
    ctx = d.diagnostic_context()
    if ctx["level_estimate_m"] is None or not (1.0 <= ctx["level_estimate_m"] <= 2.0):
        return
    job = d.run_diagnostic("ACTIVATE_WATERMARK_BURST", ep.world)
    assert job["status"] == "RUNNING" and ep.world.wm_scale == 3.0
    ep.run_until(job["due_s"] + 0.2)
    assert ep.world.wm_scale == 1.0
    assert d.diag_results["ACTIVATE_WATERMARK_BURST"]["status"] == "COMPLETE"
