"""Competing-hypothesis evidence scoring (transparent, auditable).

Scoring rule (documented, uncalibrated):

1. Candidates are (hypothesis_type, component) pairs.
2. Every EvidenceItem e contributes  w(feature(e), type) * strength(e)  to each
   candidate whose component the feature is relevant to (RELEVANCE below).
3. Anti double-counting: contributions are grouped by e.independence_group
   (one physical data stream). Within a group the strongest support counts
   fully and every further support counts 25%; the same for contradictions.
4. raw_score(candidate) = sum over groups. The best candidate per type is reported.
5. score = softmax(raw_score / T) across the 7 types. This is a normalised
   EVIDENCE SCORE - a ranking aid, **not** a calibrated probability.
6. INCONCLUSIVE leads when an anomaly is active and the top raw score is below
   MIN_SUPPORT or the margin over the runner-up is below MIN_MARGIN.
"""
from __future__ import annotations

import math
from collections import defaultdict

from ..schemas import Hypothesis, EvidenceItem

TYPES = ("SENSOR_SPOOFING", "SENSOR_FAILURE", "COMMUNICATION_MANIPULATION_OR_DELAY", "CONTROLLER_MANIPULATION",
         "ACTUATOR_FAULT", "LEGITIMATE_DISTURBANCE", "TWIN_MODEL_MISMATCH")
SS, SF, CM, CT, AF, LD, TM = TYPES
SENSORS = ("LT-101", "FT-101", "FT-102", "ZT-101", "SC-101")
MIN_SUPPORT = 1.2
MIN_MARGIN = 0.45
TEMPERATURE = 1.0
SAME_GROUP_FACTOR = 0.25

CANDIDATES: list[tuple[str, str]] = (
    [(SS, s) for s in SENSORS] + [(SF, s) for s in SENSORS] + [(CM, f"NET-1/{s}") for s in SENSORS]
    + [(CT, "PLC-101"), (AF, "FV-101"), (AF, "P-101"), (LD, "DEMAND"), (LD, "SUPPLY"), (TM, "TWIN-01")]
)

# Physical paths: which sensors can be affected by a fault at a given component.
PATH = {
    "FV-101": {"FT-102", "ZT-101", "LT-101"},
    "P-101": {"FT-101", "SC-101", "LT-101"},
    "DEMAND": {"FT-102", "LT-101"},
    "SUPPLY": {"FT-101", "LT-101"},
    "TWIN-01": {"FT-102", "FT-101", "LT-101"},
}

# Weight table w(feature_prefix, type). Positive = supports, negative = contradicts.
W: dict[str, dict[str, float]] = {
    "residual_step:LT-101": {SS: 2.0, SF: 0.6, CM: 1.0, AF: 0.6, LD: 0.4, TM: -0.3, CT: -0.6},
    "residual_ramp:LT-101": {SS: 0.6, SF: 1.8, CM: 0.3, AF: 0.7, LD: 0.4, TM: 1.0, CT: -0.6},
    "residual_step:FT-101": {SS: 1.2, SF: 0.5, CM: 0.6, AF: 1.0, LD: 1.0, TM: -0.3, CT: -0.6},
    "residual_ramp:FT-101": {SS: 0.5, SF: 1.2, CM: 0.2, AF: 0.6, LD: 0.6, TM: 0.6, CT: -0.6},
    "residual_step:FT-102": {SS: 1.2, SF: 0.5, CM: 0.6, AF: 0.8, LD: 1.6, TM: 0.0, CT: -0.6},
    "residual_ramp:FT-102": {SS: 0.4, SF: 1.0, CM: 0.2, AF: 0.5, LD: 0.5, TM: 1.8, CT: -0.6},
    "residual_step:ZT-101": {SS: 1.0, SF: 1.0, CM: 0.5, AF: 1.5, CT: -0.6},
    "residual_ramp:ZT-101": {SS: 0.6, SF: 1.2, CM: 0.3, AF: 1.5, CT: -0.6},
    "residual_step:SC-101": {SS: 1.0, SF: 1.0, CM: 0.5, AF: 1.5, CT: -0.6},
    "residual_ramp:SC-101": {SS: 0.6, SF: 1.2, CM: 0.3, AF: 1.5, CT: -0.6},
    "frozen": {SF: 2.5, CM: 0.8, SS: 0.5, AF: -0.5, LD: -0.5, TM: -0.5},
    "dropout": {SF: 2.8, CM: 1.0, AF: -0.5, LD: -0.5, TM: -0.5},
    "wm_absent:LT-101": {SS: 2.0, SF: 0.6, CM: 1.0, AF: 0.2, TM: -0.5, LD: -0.5, CT: -0.3},
    "wm_absent:FT-101": {SS: 1.2, SF: 0.8, CM: 0.8, AF: 1.4, TM: -0.5, LD: -0.5, CT: -0.3},
    "wm_present": {SF: -0.3, SS: -0.2, CM: -0.2},
    "wm_lag_shift": {CM: 2.2, SF: 0.2},
    "camera_disagrees": {SS: 1.5, SF: 1.5, CM: 1.5, AF: -1.2, LD: -1.2, TM: -0.8, CT: -1.2},
    "camera_agrees": {SS: -1.5, SF: -1.5, CM: -0.4, AF: 0.4, LD: 0.4, TM: 0.4, CT: 0.6},
    "mass_balance_inconsistent": {SS: 1.2, SF: 1.2, CM: 0.8, AF: -1.0, LD: -1.2, TM: -1.0, CT: -1.0},
    "mass_balance_consistent": {SS: -1.0, SF: -1.0, CM: -0.5, AF: 0.6, LD: 0.8, TM: 0.8, CT: 0.6},
    "latency_elevated": {CM: 2.5},
    "jitter_elevated": {CM: 2.0},
    "packet_loss": {CM: 2.5, SF: 0.3},
    "seq_anomaly": {CM: 2.8},
    "control_law_inconsistent": {CT: 2.5},
    "control_law_consistent": {CT: -1.5},
    "setpoint_departure_unexplained": {CT: 1.5, LD: 0.3, TM: 0.3},
    "oscillation": {CT: 1.2, CM: 1.0, TM: 0.3},
    "valve_actuator_divergence": {AF: 2.5, SS: -1.0, SF: -1.0},
    "valve_feedback_inconsistent_with_flow": {SS: 1.5, SF: 1.5, AF: -1.5},
    "pump_speed_divergence": {AF: 2.5, SS: -1.0, SF: -1.0},
    "pump_hydraulic_deficit": {AF: 1.5, LD: 0.8},
    "valve_feedback_tracks_command": {AF: -1.5},
    "pump_speed_tracks_command": {AF: -0.4},
    "inflow_oscillatory": {LD: 2.0, AF: 0.3},
    "inflow_hf_noise": {AF: 1.5},
    "twin_budget_exceeded": {TM: 0.8, SF: 0.8},
    # active diagnostics
    "redundant_disagrees": {SS: 2.0, SF: 2.0, CM: 2.0, AF: -1.0, LD: -1.0, TM: -1.0, CT: -1.0},
    "redundant_agrees": {SS: -2.0, SF: -2.0, CM: -0.6, AF: 0.5, LD: 0.5, TM: 0.5, CT: 0.5},
    "field_received_mismatch": {CM: 3.0, SS: -1.5, SF: -1.5},
    "field_received_match": {CM: -1.5},
    "plc_register_mismatch": {CT: 3.5},
    "plc_register_verified": {CT: -2.5},
    "stroke_test_failed": {AF: 3.0},
    "stroke_test_passed": {AF: -2.0},
    "probe_link_degraded": {CM: 2.5},
    "probe_link_nominal": {CM: -1.5},
    "burst_wm_absent": {SS: 1.8, SF: 1.2, CM: 1.0, AF: 0.8, LD: -0.8, TM: -0.8, CT: -0.5},
    "burst_wm_present": {SS: -0.8, SF: -1.2, LD: 0.3, TM: 0.3, CT: 0.3},
    "twin_check_single_sensor": {SS: 1.0, SF: 1.0, CM: 1.0, AF: -0.8, LD: -0.8, TM: -0.8, CT: -0.5},
    "twin_check_outflow_step": {LD: 1.5, TM: 0.2, SS: -0.5, SF: -0.5},
    "twin_check_outflow_gradual": {TM: 1.5, LD: 0.3, SS: -0.5, SF: -0.5},
    "twin_check_valve_path": {AF: 1.5},
    "twin_check_inflow_path": {AF: 0.8, LD: 0.8},
    "twin_check_all_consistent": {CT: 1.0, SS: -0.3, AF: -0.8, LD: -0.8, TM: -0.8},
}

ASSUMPTIONS = {
    SS: ["A single data source is compromised; the attacker does not also control the camera feed or the redundant sensor.",
         "Spoofed values are plausible in range (bias, replay) rather than physically impossible."],
    SF: ["Failure modes manifest as stuck, missing, or drifting values.", "Sensor noise specifications are known from datasheets."],
    CM: ["Network-monitor timestamps are trustworthy (synchronised clocks).", "Field devices themselves report correctly."],
    CT: ["The golden PLC configuration (setpoint, gains, logic checksum) is authentic.", "Drive-side command logging is trusted."],
    AF: ["Valve position and pump speed feedback sensors are healthy.", "Commands reach the actuator unmodified."],
    LD: ["Downstream demand and supply head are not instrumented; they are inferred from the flow balance.",
         "All sensors and actuators are healthy."],
    TM: ["All sensors healthy; twin parameters may be stale (e.g. valve trim erosion).",
         "Residual drift is slow relative to the attack-detection window."],
}

MISSING_CHECKS = {
    SS: [("camera", "Independent CAMERA_ESTIMATE of the waterline"), ("POLL_REDUNDANT_SENSOR", "Redundant level transmitter LT-102 reading"),
         ("READ_FIELD_DEVICE_DIRECT", "Field-side device value (bypassing fieldbus)"), ("watermark", "Detectable watermark correlation")],
    SF: [("camera", "Independent CAMERA_ESTIMATE of the waterline"), ("POLL_REDUNDANT_SENSOR", "Redundant level transmitter LT-102 reading"),
         ("watermark", "Detectable watermark correlation")],
    CM: [("READ_FIELD_DEVICE_DIRECT", "Field-side device value (bypassing fieldbus)"), ("NETWORK_LATENCY_PROBE", "Active link latency/loss probe")],
    CT: [("VERIFY_PLC_SETPOINT_REGISTER", "Out-of-band PLC register / logic checksum read")],
    AF: [("PARTIAL_STROKE_VALVE_TEST", "Partial-stroke valve test"), ("RUN_SHORT_HORIZON_TWIN_CHECK", "Short-horizon twin replay")],
    LD: [("RUN_SHORT_HORIZON_TWIN_CHECK", "Short-horizon twin replay"), ("camera", "Independent CAMERA_ESTIMATE of the waterline")],
    TM: [("RUN_SHORT_HORIZON_TWIN_CHECK", "Short-horizon twin replay")],
}


def _feature_key(feature: str) -> str | None:
    if feature in W:
        return feature
    base = feature.split(":")[0]
    return base if base in W else None


def _relevant(ev: EvidenceItem, htype: str, comp: str) -> bool:
    """Does this evidence item bear on candidate (htype, comp)?"""
    ec = ev.component_id
    if htype == CT:
        return True  # controller hypotheses are judged against all evidence (most is contradicting)
    if htype in (SS, SF):
        if ev.feature.startswith(("mass_balance", "oscillation", "control_law", "setpoint_departure", "twin_check_all")):
            return True
        if ec == comp:
            return True
        if ev.feature.startswith(("valve_actuator_divergence",)) and comp in ("ZT-101", "FT-102"):
            return True
        if ev.feature.startswith("pump_speed_divergence") and comp in ("SC-101", "FT-101"):
            return True
        if ev.feature.startswith("valve_feedback_inconsistent") and comp == "ZT-101":
            return True
        return False
    if htype == CM:
        link = comp.split("/", 1)[1]
        if ec in (comp, link):
            return True
        return ev.feature.startswith(("mass_balance", "oscillation", "probe_link", "twin_check_all"))
    # physical-path hypotheses (actuator, disturbance, model mismatch)
    sensors = PATH.get(comp, set())
    if ec == comp or ec in sensors or ec == "TK-101":
        return True
    if ev.feature.startswith(("oscillation", "control_law", "setpoint_departure", "twin_check_all", "camera_", "redundant_")):
        return True
    return False


def score(evidence: list[EvidenceItem], anomaly_active: bool, available: dict[str, bool] | None = None,
          feature_filter=None) -> dict:
    available = available or {}
    if feature_filter is not None:
        evidence = [e for e in evidence if feature_filter(e.feature)]
    cand_scores: dict[tuple[str, str], dict] = {}
    for htype, comp in CANDIDATES:
        groups_pos: dict[str, list[tuple[float, str]]] = defaultdict(list)
        groups_neg: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for ev in evidence:
            key = _feature_key(ev.feature)
            if key is None or not _relevant(ev, htype, comp):
                continue
            w = W[key].get(htype, 0.0)
            if w == 0.0:
                continue
            c = w * ev.strength
            (groups_pos if c > 0 else groups_neg)[ev.independence_group].append((c, ev.evidence_id))
        total, sup, con = 0.0, [], []
        for g, items in groups_pos.items():
            items.sort(reverse=True)
            total += items[0][0] + SAME_GROUP_FACTOR * sum(c for c, _ in items[1:])
            sup += [i for _, i in items]
        for g, items in groups_neg.items():
            items.sort()
            total += items[0][0] + SAME_GROUP_FACTOR * sum(c for c, _ in items[1:])
            con += [i for _, i in items]
        cand_scores[(htype, comp)] = {"raw": total, "support": sup, "contra": con}

    best: dict[str, tuple[str, dict]] = {}
    for (htype, comp), v in cand_scores.items():
        if htype not in best or v["raw"] > best[htype][1]["raw"]:
            best[htype] = (comp, v)
    raws = {t: best[t][1]["raw"] for t in TYPES}
    mx = max(raws.values())
    exps = {t: math.exp((raws[t] - mx) / TEMPERATURE) for t in TYPES}
    z = sum(exps.values())
    shares = {t: exps[t] / z for t in TYPES}
    order = sorted(TYPES, key=lambda t: raws[t], reverse=True)
    top, second = order[0], order[1]
    margin = raws[top] - raws[second]
    entropy = -sum(p * math.log(p) for p in shares.values() if p > 0) / math.log(len(TYPES))

    if not anomaly_active:
        verdict = {"state": "NOMINAL", "leading": None, "reason": "No active anomaly alarm; hypotheses not ranked."}
    elif raws[top] < MIN_SUPPORT:
        verdict = {"state": "INCONCLUSIVE", "leading": None,
                   "reason": f"Anomaly active but strongest hypothesis has weak support (raw {raws[top]:.2f} < {MIN_SUPPORT})."}
    elif margin < MIN_MARGIN:
        verdict = {"state": "INCONCLUSIVE", "leading": None,
                   "reason": f"{top} and {second} are nearly tied (margin {margin:.2f} < {MIN_MARGIN}); collect discriminating evidence."}
    else:
        verdict = {"state": "LEADING_HYPOTHESIS", "leading": top, "component": best[top][0],
                   "reason": f"{top} leads by raw margin {margin:.2f}."}
    verdict.update(margin=round(margin, 3), normalized_entropy=round(entropy, 3))

    hyps: list[Hypothesis] = []
    for rank, t in enumerate(order, start=1):
        comp, v = best[t]
        missing = []
        for key, label in MISSING_CHECKS.get(t, []):
            if not available.get(key, False):
                missing.append(label)
        if not anomaly_active:
            status = "NOT_EVALUATED"
        elif verdict["state"] == "LEADING_HYPOTHESIS" and t == top:
            status = "LEADING"
        elif v["raw"] < 0:
            status = "DISFAVOURED"
        else:
            status = "COMPETING"
        alts = sorted(((c, cs["raw"]) for (ht, c), cs in cand_scores.items() if ht == t and c != comp),
                      key=lambda x: -x[1])[:2]
        hyps.append(Hypothesis(
            hypothesis_id=f"HYP:{t}", hypothesis_type=t, component_id=comp, score=round(shares[t], 4), raw_score=round(v["raw"], 4),
            rank=rank, status=status, supporting_evidence_refs=v["support"], contradicting_evidence_refs=v["contra"],
            missing_evidence=missing, assumptions=ASSUMPTIONS[t],
            explanation=f"Best component {comp}; alternatives " + ", ".join(f"{c} ({r:.2f})" for c, r in alts),
        ))
    inconclusive_score = round(entropy, 4) if anomaly_active else 0.0
    hyps.append(Hypothesis(
        hypothesis_id="HYP:INCONCLUSIVE", hypothesis_type="INCONCLUSIVE", component_id=None, score=inconclusive_score,
        raw_score=round(max(0.0, MIN_SUPPORT - raws[top]) + max(0.0, MIN_MARGIN - margin), 4), rank=0,
        status="INCONCLUSIVE" if verdict["state"] == "INCONCLUSIVE" else ("NOT_EVALUATED" if not anomaly_active else "DISFAVOURED"),
        missing_evidence=sorted({m for h in hyps[:3] for m in h.missing_evidence}),
        assumptions=["Score is the normalised entropy of the evidence-score distribution (1 = completely undecided)."],
        explanation=verdict["reason"],
    ))
    candidates = [{"hypothesis_type": h, "component_id": c, "raw_score": round(v["raw"], 4)} for (h, c), v in cand_scores.items()]
    return {"verdict": verdict, "hypotheses": [h.model_dump() for h in hyps], "candidates": candidates,
            "score_type": "EVIDENCE_SCORE",
            "score_note": "Normalised evidence scores (softmax of auditable evidence totals). Not calibrated probabilities."}
