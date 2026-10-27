"""Active investigation: diagnostic registry, preconditions, EIG ranking, execution.

Ranking heuristic (documented): the current candidate scores are turned into
uncertainty weights b(c) = softmax(raw_score(c)). These weights are NOT
calibrated probabilities. For each test with an outcome model P(o | c):

    EIG = H(b) - sum_o P(o) H(b | o),  P(o) = sum_c b(c) P(o | c)   [bits]
    utility = EIG - 0.3 * cost - 0.3 * process_disturbance

Tests whose safety preconditions fail are listed but not executable.

Execution talks to the simulator only through a narrow out-of-band port
(maintenance reads, redundant poll, register read, probe, small bounded
excitations). It never reads simulation ground truth.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Protocol

import numpy as np

from ..schemas import Observation
from .hypotheses import CANDIDATES, SS, SF, CM, CT, AF, LD, TM
from .twin import DigitalTwin, residual_sigma


class DiagnosticPort(Protocol):
    def read_field_direct(self) -> dict[str, float]: ...
    def poll_redundant_level(self) -> float: ...
    def read_plc_registers(self) -> dict[str, Any]: ...
    def probe_link(self, sid: str) -> dict[str, float]: ...
    def set_watermark_scale(self, scale: float) -> None: ...
    def set_valve_test_offset(self, offset: float) -> None: ...


def _lt_types(c):
    h, comp = c
    return h in (SS, SF) and comp == "LT-101" or (h == CM and comp == "NET-1/LT-101")


REGISTRY: dict[str, dict[str, Any]] = {
    "POLL_REDUNDANT_SENSOR": dict(
        label="Poll redundant level transmitter LT-102", cost=0.05, disturbance=0.0, duration_s=0.0,
        description="Reads the independent radar transmitter LT-102 over its own path and compares it with LT-101.",
        outcomes={"DISAGREE": lambda c: 0.9 if _lt_types(c) else 0.05}),
    "READ_FIELD_DEVICE_DIRECT": dict(
        label="Read field devices via maintenance port", cost=0.1, disturbance=0.0, duration_s=0.0,
        description="Reads each transmitter directly (bypassing the fieldbus) and compares with PLC-received values.",
        outcomes={"MISMATCH": lambda c: 0.85 if c[0] == CM else 0.05}),
    "VERIFY_PLC_SETPOINT_REGISTER": dict(
        label="Verify PLC setpoint / gain registers", cost=0.1, disturbance=0.0, duration_s=0.0,
        description="Out-of-band read of setpoint, PI gains and logic checksum versus the golden configuration.",
        outcomes={"MISMATCH": lambda c: 0.95 if c[0] == CT else 0.02}),
    "NETWORK_LATENCY_PROBE": dict(
        label="Fieldbus latency / loss probe", cost=0.05, disturbance=0.0, duration_s=0.0,
        description="Sends timestamped probe frames along each sensor link and measures latency and loss.",
        outcomes={"DEGRADED": lambda c: 0.75 if c[0] == CM else 0.05}),
    "CAPTURE_VISUAL_WITNESS_FRAME": dict(
        label="Capture visual witness frame", cost=0.02, disturbance=0.0, duration_s=0.0,
        description="Captures a camera frame now and estimates the waterline in the calibrated ROI.",
        outcomes={"DISAGREE": lambda c: 0.85 if _lt_types(c) else 0.05}),
    "RUN_SHORT_HORIZON_TWIN_CHECK": dict(
        label="Short-horizon twin replay check", cost=0.05, disturbance=0.0, duration_s=0.0,
        description="Re-runs a fresh twin from the last verified snapshot with the logged commands and localises which "
                    "measurement path disagrees.",
        outcomes={
            "SINGLE_SENSOR": lambda c: 0.8 if c[0] in (SS, SF, CM) else 0.04,
            "OUTFLOW": lambda c: 0.8 if c in ((LD, "DEMAND"), (TM, "TWIN-01")) else 0.04,
            "VALVE": lambda c: 0.8 if c == (AF, "FV-101") else 0.04,
            "INFLOW": lambda c: 0.8 if c in ((AF, "P-101"), (LD, "SUPPLY")) else 0.04,
            "ALL_CONSISTENT": lambda c: 0.8 if c[0] == CT else 0.04,
        }),
    "ACTIVATE_WATERMARK_BURST": dict(
        label="Activate watermark burst (3× ε for 30 s)", cost=0.2, disturbance=0.3, duration_s=30.0,
        description="Temporarily triples the bounded watermark amplitude and checks whether the response appears in "
                    "LT-101 and FT-101.",
        outcomes={"ABSENT": lambda c: (0.7 if c[0] in (SS, SF, CM) and c[1].endswith(("LT-101", "FT-101")) else
                                       0.7 if c == (AF, "P-101") else 0.08)}),
    "PARTIAL_STROKE_VALVE_TEST": dict(
        label="Partial-stroke valve test (+4 % for 20 s)", cost=0.3, disturbance=0.4, duration_s=20.0,
        description="Commands a small bounded valve excursion and checks that position feedback and outlet flow follow.",
        outcomes={"FAIL": lambda c: 0.9 if c == (AF, "FV-101") else 0.05}),
}


def _entropy(p: np.ndarray) -> float:
    p = p[p > 1e-12]
    return float(-(p * np.log2(p)).sum())


def candidate_weights(candidates: list[dict]) -> tuple[list[tuple[str, str]], np.ndarray]:
    lookup = {(c["hypothesis_type"], c["component_id"]): c["raw_score"] for c in candidates}
    keys = list(CANDIDATES)
    raw = np.array([lookup.get(k, 0.0) for k in keys])
    w = np.exp(raw - raw.max())
    return keys, w / w.sum()


def expected_information_gain(test_id: str, keys, b: np.ndarray) -> float:
    outcomes = REGISTRY[test_id]["outcomes"]
    likelihoods = []
    for fn in outcomes.values():
        likelihoods.append(np.array([fn(k) for k in keys]))
    if len(likelihoods) == 1:
        likelihoods.append(1.0 - likelihoods[0])
    else:
        rest = np.clip(1.0 - np.sum(likelihoods, axis=0), 0.02, 1.0)
        likelihoods.append(rest)
    L = np.vstack(likelihoods)
    L = L / L.sum(axis=0, keepdims=True)
    h0 = _entropy(b)
    eh = 0.0
    for row in L:
        po = float((row * b).sum())
        if po <= 1e-12:
            continue
        post = row * b / po
        eh += po * _entropy(post)
    return max(0.0, h0 - eh)


def check_preconditions(test_id: str, ctx: dict) -> tuple[bool, list[str]]:
    reasons = []
    lvl = ctx.get("level_estimate_m")
    lim = ctx["safety"]
    if test_id in ("ACTIVATE_WATERMARK_BURST", "PARTIAL_STROKE_VALVE_TEST"):
        if lvl is None or not (lim.band_low_m + 0.2 <= lvl <= lim.band_high_m - 0.2):
            reasons.append("Level estimate must be inside the safe operating band with 0.2 m margin")
        if ctx.get("response_policy") in ("CONTROLLED_SAFE_SHUTDOWN", "HOLD_LAST_VERIFIED_SAFE_STATE"):
            reasons.append(f"Not permitted while {ctx.get('response_policy')} is active")
        if test_id in ctx.get("running", []):
            reasons.append("Already running")
    if test_id == "ACTIVATE_WATERMARK_BURST":
        if not ctx.get("watermark_enabled", True):
            reasons.append("Watermarking disabled")
        if ctx.get("saturation_fraction", 0.0) > 0.1:
            reasons.append("Actuator saturated: burst would be clipped")
    if test_id == "CAPTURE_VISUAL_WITNESS_FRAME":
        if not ctx.get("camera_ready", False):
            reasons.append("No camera source active and calibrated (explicit user selection required)")
    return (not reasons), reasons


def rank_tests(candidates: list[dict], ctx: dict) -> list[dict]:
    keys, b = candidate_weights(candidates)
    out = []
    for tid, spec in REGISTRY.items():
        ok, reasons = check_preconditions(tid, ctx)
        recent = ctx.get("recent", {}).get(tid)
        # A result still inside its validity window is already in the evidence set: repeating it adds no information.
        eig = 0.0 if recent is not None else expected_information_gain(tid, keys, b)
        util = eig - 0.3 * spec["cost"] - 0.3 * spec["disturbance"]
        out.append({
            "test_id": tid, "label": spec["label"], "description": spec["description"],
            "expected_information_gain_bits": round(eig, 4), "cost": spec["cost"], "process_disturbance": spec["disturbance"],
            "utility": round(util, 4), "executable": ok, "precondition_failures": reasons, "duration_s": spec["duration_s"],
            "recently_executed_at_s": recent, "ranking_method": "EIG over softmax evidence weights (heuristic, uncalibrated)",
        })
    out.sort(key=lambda r: (not r["executable"], -r["utility"]))
    return out


def _ev(feature, component, strength, description, group, source_type, value=None):
    return {"feature": feature, "component": component, "strength": round(float(min(1.0, max(0.0, strength))), 4),
            "description": description, "group": group, "source_type": source_type, "value": value}


# --------------------------------------------------------------------------- execution
def start(test_id: str, t: float, port: DiagnosticPort) -> dict:
    """Begin a (possibly extended) test. Returns a job dict; finalise() completes it."""
    spec = REGISTRY[test_id]
    job = {"test_id": test_id, "label": spec["label"], "started_s": t, "due_s": t + spec["duration_s"], "status": "RUNNING"}
    if test_id == "ACTIVATE_WATERMARK_BURST":
        port.set_watermark_scale(3.0)
    elif test_id == "PARTIAL_STROKE_VALVE_TEST":
        port.set_valve_test_offset(0.04)
    return job


def finalise(job: dict, t: float, port: DiagnosticPort, defender) -> dict:
    tid = job["test_id"]
    h, cfg, dt = defender.hist, defender.cfg, defender.cfg.dt_s
    res: dict[str, Any] = dict(job)
    res["completed_s"] = t
    ev: list[dict] = []
    obs: Observation | None = None
    lt_recv = float(np.nanmean(h.window("LT-101", int(1 / dt)))) if len(h) else float("nan")

    if tid == "POLL_REDUNDANT_SENSOR":
        v = port.poll_redundant_level()
        tol = 4 * math.hypot(cfg.sensors.level_noise_m, cfg.sensors.redundant_level_noise_m) + 0.01
        d = v - lt_recv
        obs = Observation(timestamp=t, source_type="SENSOR", source_id="LT-102", component_id="TK-101", measurement_type="level",
                          value=v, unit="m", uncertainty=cfg.sensors.redundant_level_noise_m,
                          provenance={"class": "SENSOR_OBSERVATION", "path": "independent hardwired loop", "diagnostic": tid},
                          metadata={"independence_group": "LT-102"})
        if math.isnan(lt_recv) or abs(d) > tol:
            ev.append(_ev("redundant_disagrees", "LT-101", 1.0 if math.isnan(lt_recv) else min(1.0, abs(d) / (2.5 * tol)) * 0.6 + 0.4,
                          f"LT-102 reads {v:.3f} m vs LT-101 {lt_recv:.3f} m (Δ {d:+.3f} m, tol {tol:.3f})", "LT-102", "SENSOR", d))
            res["outcome"] = "DISAGREE"
        else:
            ev.append(_ev("redundant_agrees", "LT-101", 1.0 - abs(d) / tol,
                          f"LT-102 {v:.3f} m corroborates LT-101 {lt_recv:.3f} m", "LT-102", "SENSOR", d))
            res["outcome"] = "AGREE"
        res["measurements"] = {"LT-102_m": v, "LT-101_received_m": lt_recv, "difference_m": d, "tolerance_m": tol}

    elif tid == "READ_FIELD_DEVICE_DIRECT":
        field = port.read_field_direct()
        out = {}
        mismatch_any = False
        for sid, fv in field.items():
            rv = h.last(sid)
            sig = residual_sigma(cfg)[sid]
            d = (rv - fv) if not (math.isnan(rv) or math.isnan(fv)) else float("nan")
            # allow for the nominal 1-sample transport lag
            tol = 5 * sig
            out[sid] = {"field": fv, "received": rv, "difference": d}
            if math.isnan(d):
                if math.isnan(rv) != math.isnan(fv):
                    ev.append(_ev("field_received_mismatch", sid, 1.0, f"{sid}: field device and PLC-received availability differ",
                                  "FIELD-DIRECT", "NETWORK_MONITOR"))
                    mismatch_any = True
                continue
            if abs(d) > tol:
                mismatch_any = True
                ev.append(_ev("field_received_mismatch", sid, min(1.0, abs(d) / (3 * tol)) * 0.5 + 0.5,
                              f"{sid}: field device {fv:.3f} vs PLC-received {rv:.3f} (Δ {d:+.3f}) - value altered or delayed in transit",
                              "FIELD-DIRECT", "NETWORK_MONITOR", d))
            elif sid in defender.anomalous_components():
                ev.append(_ev("field_received_match", sid, 1.0 - abs(d) / tol,
                              f"{sid}: field device matches PLC-received value - transit path not altering data",
                              "FIELD-DIRECT", "NETWORK_MONITOR", d))
        res["outcome"] = "MISMATCH" if mismatch_any else "MATCH"
        res["measurements"] = out
        obs = Observation(timestamp=t, source_type="NETWORK_MONITOR", source_id="MAINT-PORT", component_id="NET-1",
                          measurement_type="field_vs_received", value=res["outcome"], unit="-",
                          provenance={"class": "SENSOR_OBSERVATION", "diagnostic": tid}, metadata={"independence_group": "FIELD-DIRECT"})

    elif tid == "VERIFY_PLC_SETPOINT_REGISTER":
        regs = port.read_plc_registers()
        golden = defender.golden
        diffs = {}
        for k in ("setpoint_m", "kp", "ki"):
            if abs(regs[k] - golden[k]) > 1e-6 * max(1.0, abs(golden[k])):
                diffs[k] = {"register": regs[k], "golden": golden[k]}
        if regs.get("logic_checksum") != "GOLDEN-7F3A":
            diffs["logic_checksum"] = {"register": regs.get("logic_checksum"), "golden": "GOLDEN-7F3A"}
        if diffs:
            ev.append(_ev("plc_register_mismatch", "PLC-101", 1.0, "PLC registers differ from golden configuration: " +
                          ", ".join(f"{k}={v['register']} (golden {v['golden']})" for k, v in diffs.items()), "PLC-REG", "CONTROLLER_LOG"))
            res["outcome"] = "MISMATCH"
        else:
            ev.append(_ev("plc_register_verified", "PLC-101", 1.0, "PLC setpoint, gains and logic checksum match golden configuration",
                          "PLC-REG", "CONTROLLER_LOG"))
            res["outcome"] = "VERIFIED"
        res["measurements"] = {"registers": regs, "golden": golden, "differences": diffs}
        obs = Observation(timestamp=t, source_type="CONTROLLER_LOG", source_id="PLC-101", component_id="PLC-101",
                          measurement_type="register_integrity", value=res["outcome"], unit="-",
                          provenance={"class": "SENSOR_OBSERVATION", "diagnostic": tid}, metadata={"independence_group": "PLC-REG"})

    elif tid == "NETWORK_LATENCY_PROBE":
        out, degraded = {}, False
        nominal = math.ceil((cfg.network.base_latency_s + 1e-9) / dt) * dt
        for sid in ("LT-101", "FT-101", "FT-102", "ZT-101", "SC-101"):
            pr = port.probe_link(sid)
            out[sid] = pr
            bad = pr["loss"] > 0.1 or pr["latency_s"] > nominal + 0.4 or pr.get("stale", False)
            if bad:
                degraded = True
                ev.append(_ev("probe_link_degraded", f"NET-1/{sid}", 1.0,
                              f"Probe on {sid} link: latency {pr['latency_s']:.2f} s, loss {pr['loss']:.0%}"
                              + (", stale-sequence replies" if pr.get("stale") else ""), "NET-PROBE", "NETWORK_MONITOR", pr["latency_s"]))
        if not degraded:
            ev.append(_ev("probe_link_nominal", "NET-1/LT-101", 1.0, "All fieldbus links probe nominal", "NET-PROBE", "NETWORK_MONITOR"))
            for sid in ("FT-101", "FT-102", "ZT-101", "SC-101"):
                ev.append(_ev("probe_link_nominal", f"NET-1/{sid}", 1.0, "All fieldbus links probe nominal", "NET-PROBE", "NETWORK_MONITOR"))
        res["outcome"] = "DEGRADED" if degraded else "NOMINAL"
        res["measurements"] = out
        obs = Observation(timestamp=t, source_type="NETWORK_MONITOR", source_id="PROBE-1", component_id="NET-1",
                          measurement_type="link_probe", value=res["outcome"], unit="-",
                          provenance={"class": "SENSOR_OBSERVATION", "diagnostic": tid}, metadata={"independence_group": "NET-PROBE"})

    elif tid == "CAPTURE_VISUAL_WITNESS_FRAME":
        cam = defender.capture_camera(t)
        if cam is None or cam.get("status") != "OK" or cam.get("level_m") is None:
            res["status"] = "UNAVAILABLE"
            res["outcome"] = "UNAVAILABLE"
            res["summary"] = "Camera estimate unavailable: " + (cam or {}).get("reason", "no active camera")
            res["measurements"] = {"camera": cam}
        else:
            cl, cu = cam["level_m"], cam.get("uncertainty_m", 0.03)
            tol = 3 * math.hypot(cu, cfg.sensors.level_noise_m)
            d = cl - lt_recv
            res["measurements"] = {"camera_level_m": cl, "LT-101_received_m": lt_recv, "difference_m": d, "tolerance_m": tol,
                                   "camera_source": cam.get("source_mode")}
            if abs(d) > tol:
                ev.append(_ev("camera_disagrees", "LT-101", min(1.0, abs(d) / (2.5 * tol)) * 0.6 + 0.4,
                              f"Visual witness {cl:.3f} m vs LT-101 {lt_recv:.3f} m", "CAM-01", "CAMERA_ESTIMATE", d))
                res["outcome"] = "DISAGREE"
            else:
                ev.append(_ev("camera_agrees", "LT-101", 1.0 - abs(d) / tol, f"Visual witness {cl:.3f} m corroborates LT-101",
                              "CAM-01", "CAMERA_ESTIMATE", d))
                res["outcome"] = "AGREE"
            obs = Observation(timestamp=t, source_type="CAMERA_ESTIMATE", source_id="CAM-01", component_id="TK-101",
                              measurement_type="waterline_level", value=cl, unit="m", uncertainty=cu,
                              quality="GOOD", provenance={"class": "CAMERA_VISUAL_ESTIMATE", "mode": cam.get("source_mode"), "diagnostic": tid},
                              metadata={"independence_group": "CAM-01"})

    elif tid == "RUN_SHORT_HORIZON_TWIN_CHECK":
        res.update(_twin_check(defender, t))
        ev.extend(res.pop("evidence"))
        obs = Observation(timestamp=t, source_type="TWIN_PREDICTION", source_id="TWIN-CHECK", component_id="TWIN-01",
                          measurement_type="replay_consistency", value=res["outcome"], unit="-",
                          provenance={"class": "DIGITAL_TWIN_PREDICTION", "diagnostic": tid, "anchor_s": res.get("anchor_s")},
                          metadata={"independence_group": "TWIN-CHECK"})

    elif tid == "ACTIVATE_WATERMARK_BURST":
        port.set_watermark_scale(1.0)
        n = int(30.0 / dt)
        an = defender.watermark.analyze(_TailView(h, n + int(20 / dt)), defender.twin.inflow_gain_lps())
        res["measurements"] = an
        absent = []
        for sid, e in (an.get("signals") or {}).items():
            st = e.get("status")
            if st in ("ABSENT", "WEAK") and e.get("detectability") != "UNDETECTABLE":
                absent.append(sid)
                ev.append(_ev("burst_wm_absent", sid, 1.0 if st == "ABSENT" else 0.5,
                              f"Burst watermark not reflected in {sid} (ρ={e.get('correlation')}, expected {e.get('expected_correlation')})",
                              f"WM-BURST:{sid}", "WATERMARK", e.get("correlation")))
            elif st == "CORRELATED":
                ev.append(_ev("burst_wm_present", sid, 1.0, f"Burst watermark clearly present in {sid} (ρ={e.get('correlation')})",
                              f"WM-BURST:{sid}", "WATERMARK", e.get("correlation")))
        res["outcome"] = "ABSENT" if absent else "PRESENT"
        obs = Observation(timestamp=t, source_type="WATERMARK", source_id="WM-BURST", component_id="P-101",
                          measurement_type="burst_correlation", value=res["outcome"], unit="-",
                          provenance={"class": "INFERENCE", "diagnostic": tid}, metadata={"independence_group": "WM-BURST"})

    elif tid == "PARTIAL_STROKE_VALVE_TEST":
        port.set_valve_test_offset(0.0)
        n = int(job["due_s"] - job["started_s"]) * int(1 / dt)
        zt = h.window("ZT-101", n)
        pre = float(np.nanmean(h.window("ZT-101", n + int(5 / dt))[: int(5 / dt)]))
        moved = float(np.nanmax(zt) - pre) if zt.size else 0.0
        expected = 4.0
        fo = h.window("FT-102", n)
        fo_pre = float(np.nanmean(h.window("FT-102", n + int(5 / dt))[: int(5 / dt)]))
        dq = float(np.nanmean(fo[-int(5 / dt):]) - fo_pre) if fo.size else 0.0
        res["measurements"] = {"feedback_movement_pct": moved, "expected_pct": expected, "outlet_flow_change_lps": dq}
        if moved < 0.4 * expected:
            ev.append(_ev("stroke_test_failed", "FV-101", 1.0 - moved / (0.4 * expected) * 0.5,
                          f"Valve moved {moved:.1f} % of commanded {expected:.0f} % stroke (stuck / degraded actuator)", "STROKE",
                          "SENSOR", moved))
            res["outcome"] = "FAIL"
        else:
            ev.append(_ev("stroke_test_passed", "FV-101", 1.0, f"Valve followed stroke ({moved:.1f} %), outlet flow Δ {dq:+.2f} L/s",
                          "STROKE", "SENSOR", moved))
            res["outcome"] = "PASS"
        obs = Observation(timestamp=t, source_type="SENSOR", source_id="ZT-101", component_id="FV-101",
                          measurement_type="partial_stroke", value=res["outcome"], unit="-",
                          provenance={"class": "SENSOR_OBSERVATION", "diagnostic": tid}, metadata={"independence_group": "STROKE"})

    if res.get("status") in (None, "RUNNING"):
        res["status"] = "COMPLETE"
    res["evidence"] = ev
    if obs is not None:
        obs.metadata["diagnostic_test"] = tid
        res["observation"] = obs.model_dump()
        res["observation_id"] = obs.observation_id
    res.setdefault("summary", "; ".join(e["description"] for e in ev[:3]))
    return res


class _TailView:
    """Restricts a History to its last n samples (for burst-window analysis)."""

    def __init__(self, hist, n):
        self._h, self._n = hist, n

    def __len__(self):
        return min(len(self._h), self._n)

    def window(self, ch, n):
        return self._h.window(ch, min(n, len(self)))


def _twin_check(defender, t: float) -> dict:
    h, cfg, dt = defender.hist, defender.cfg, defender.cfg.dt_s
    snap = defender.verified_snapshot()
    ev: list[dict] = []
    if snap is None:
        return {"status": "UNAVAILABLE", "outcome": "UNAVAILABLE", "summary": "No verified twin snapshot available", "evidence": []}
    anchor_t, state = snap
    n = int(round((t - anchor_t) / dt))
    n = min(n, len(h) - 1)
    if n < int(10 / dt):
        return {"status": "UNAVAILABLE", "outcome": "UNAVAILABLE", "summary": "Replay horizon too short", "evidence": []}
    tw = DigitalTwin(cfg)
    tw.restore(state)
    u = h.window("u_cmd", n)
    v = h.window("valve_cmd", n)
    preds = {s: np.empty(n) for s in ("LT-101", "FT-101", "FT-102", "ZT-101", "SC-101")}
    for i in range(n):
        y = tw.predict_measurements()
        for s in preds:
            preds[s][i] = y[s]
        tw.step({"recv": {}, "u_cmd": float(u[i]), "valve_cmd": float(v[i])}, suspicious=True)
    rms = {}
    lag = tw.delay_steps
    for s, p in preds.items():
        meas = h.window(s, n)
        d = meas[lag:] - p[:-lag] if lag else meas - p
        d = d[~np.isnan(d)]
        rms[s] = float(np.sqrt(np.mean(d[-int(20 / dt):] ** 2)) / residual_sigma(cfg)[s]) if d.size else float("nan")
    bad = {s for s, z in rms.items() if not math.isnan(z) and z > 4.0}
    # Outflow-coefficient ratio from *measured* level and valve feedback: FT-102 / (Cv * ZT * sqrt(LT)).
    # A step in this ratio indicates a demand change; a gradual drift indicates a stale twin parameter.
    from ..sim.plant import valve_flow
    ratio_blocks = []
    fo, lt_m, zt_m = h.window("FT-102", n), h.window("LT-101", n), h.window("ZT-101", n)
    bl = int(5 / dt)
    for i in range(0, n - bl + 1, bl):
        lvl = float(np.nanmean(lt_m[i:i + bl]))
        pos = float(np.nanmean(zt_m[i:i + bl])) / 100.0
        m = valve_flow(cfg.plant, pos, max(lvl, 0.0)) * 1000
        if m > 0.5:
            ratio_blocks.append(round(float(np.nanmean(fo[i:i + bl]) / m), 4))
    level_flow = bad & {"LT-101", "FT-101", "FT-102"}
    if "ZT-101" in bad and bad & {"FT-102", "LT-101"}:
        outcome = "VALVE"
        ev.append(_ev("twin_check_valve_path", "FV-101", 1.0, "Replay: valve-position path disagrees with logged commands", "TWIN-CHECK",
                      "TWIN_PREDICTION"))
    elif "SC-101" in bad and bad & {"FT-101", "LT-101"}:
        outcome = "INFLOW"
        ev.append(_ev("twin_check_inflow_path", "P-101", 1.0, "Replay: pump speed and inflow path disagree with logged commands",
                      "TWIN-CHECK", "TWIN_PREDICTION"))
    elif len(bad) == 1:
        s_only = next(iter(bad))
        outcome = "SINGLE_SENSOR"
        ev.append(_ev("twin_check_single_sensor", s_only, 1.0,
                      f"Replay: only {s_only} disagrees; every other measurement path matches the twin", "TWIN-CHECK",
                      "TWIN_PREDICTION"))
    elif "FT-102" in level_flow:
        outcome = "OUTFLOW"
        rb = np.array(ratio_blocks)
        jumps = np.abs(np.diff(rb)) if rb.size > 2 else np.array([0.0])
        total = abs(rb[-1] - rb[0]) if rb.size > 1 else 0.0
        abrupt = total > 0 and float(jumps.max()) > 0.6 * total
        if abrupt:
            ev.append(_ev("twin_check_outflow_step", "DEMAND", 1.0,
                          f"Replay: outflow ratio stepped abruptly ({rb[0]:.2f}→{rb[-1]:.2f}) - consistent with a demand change",
                          "TWIN-CHECK", "TWIN_PREDICTION"))
        else:
            ev.append(_ev("twin_check_outflow_gradual", "TWIN-01", 1.0,
                          f"Replay: outflow ratio drifted gradually ({rb[0] if rb.size else 1:.2f}→{rb[-1] if rb.size else 1:.2f}) - "
                          "consistent with twin parameter mismatch", "TWIN-CHECK", "TWIN_PREDICTION"))
    elif "FT-101" in level_flow:
        outcome = "INFLOW"
        for comp in ("P-101", "SUPPLY"):
            ev.append(_ev("twin_check_inflow_path", comp, 1.0, "Replay: inflow path (FT-101) disagrees with pump-curve prediction",
                          "TWIN-CHECK", "TWIN_PREDICTION"))
    elif not bad:
        outcome = "ALL_CONSISTENT"
        ev.append(_ev("twin_check_all_consistent", "PLC-101", 1.0,
                      "Replay: every measurement matches the twin driven by logged commands - anomaly originates in the commands",
                      "TWIN-CHECK", "TWIN_PREDICTION"))
    else:
        outcome = "MIXED"
    return {"outcome": outcome, "anchor_s": anchor_t, "measurements": {"normalized_rms": rms, "outflow_ratio_blocks": ratio_blocks},
            "evidence": ev}


def outcome_from_result(res: dict) -> str:
    return res.get("outcome", "UNKNOWN")


Runner = Callable[[str], dict]
