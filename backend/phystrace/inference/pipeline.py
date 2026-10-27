"""Defender pipeline: consumes ObservableFrames only (never TruthFrames).

Per step:  digital twin, shadow controller, ring-buffer history, diagnostic jobs.
Per 1 s:   watermark correlation, evidence features, alarm detector, hypothesis
           scoring, Observation normalisation (SRS 5.1) and events.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Callable

import numpy as np

from ..config import SimConfig
from ..schemas import Observation
from . import diagnostics as dg
from .features import SENSORS, FeatureContext, extract_features
from .hypotheses import score
from .history import History
from .twin import SENSOR_UNITS, DigitalTwin, residual_sigma
from .watermark import WatermarkCorrelator

POSITIVE_ONLY = ("present", "consistent", "agrees", "verified", "passed", "nominal", "match")

MODE_FILTERS: dict[str, Callable[[str], bool] | None] = {
    "RESIDUAL_ONLY": lambda f: f.startswith(("residual_", "twin_budget")),
    "WATERMARK_ONLY": lambda f: f.startswith(("wm_", "burst_wm")),
    "FUSION": lambda f: not f.startswith(("redundant_", "field_received", "plc_register", "stroke_test", "probe_link",
                                          "burst_wm", "twin_check")),
    "FULL": None,
}


def _is_detection_feature(f: str) -> bool:
    base = f.split(":")[0]
    return not any(base.endswith(s) or f"_{s}" in base for s in POSITIVE_ONLY)


class Defender:
    def __init__(self, cfg: SimConfig, *, mode: str = "FULL", camera_provider: Callable[[float], dict | None] | None = None,
                 emit: bool = True, sink: Callable[[str, Any], None] | None = None, history_s: float = 300.0):
        self.cfg = cfg
        self.dt = cfg.dt_s
        self.mode = mode
        self.filter = MODE_FILTERS[mode]
        self.twin = DigitalTwin(cfg)
        self.watermark = WatermarkCorrelator(cfg)
        c = cfg.controller
        self.golden = {"setpoint_m": c.setpoint_m, "kp": c.kp, "ki": c.ki, "u_ff": c.u_ff}
        chans = ["t", "u_cmd", "u_plc", "valve_cmd", "delta_u", "wm_saturated", "setpoint_reported", "shadow_err",
                 "response_active", "cam_h"]
        for s in SENSORS:
            chans += [s, f"yhat:{s}", f"res:{s}", f"lat:{s}", f"arr:{s}", f"seqa:{s}"]
        self.hist = History(int(history_s / self.dt), chans)
        self.camera_provider = camera_provider
        self.emit = emit
        self.sink = sink
        self.shadow_integral = 0.0
        self.shadow_last = cfg.plant.initial_level_m
        self.alarm_onset_s: float | None = None
        self._raise_count = 0
        self._clear_count = 0
        self.alarm_history: list[dict] = []
        self.first_alarm_s: float | None = None
        self.evidence: list = []
        self.result: dict = {"verdict": {"state": "NOMINAL"}, "hypotheses": [], "candidates": []}
        self.wm_result: dict | None = None
        self.feature_diag: dict = {}
        self.camera_latest: dict | None = None
        self.diag_results: dict[str, dict] = {}
        self.diag_log: list[dict] = []
        self.jobs: list[dict] = []
        self.observations: deque = deque(maxlen=4000)
        self.obs_refs: dict[str, str] = {}
        self.snapshots: deque = deque(maxlen=400)  # (t, twin snapshot, u_cmd, valve_cmd) while nominal
        self.first_seen: dict[str, float] = {}
        self.response_policy = "NONE"
        self.last_obs: dict | None = None
        self.last_t = 0.0
        self.anomaly_level = 0.0
        self.verdict_log: list[dict] = []
        self.diagnostic_validity_s = 120.0
        self.shape_memory: dict[str, dict] = {}

    # ------------------------------------------------------------------ per step
    def step(self, obs: dict, port: dg.DiagnosticPort | None = None) -> None:
        t = obs["t"]
        self.last_t = t
        self.last_obs = obs
        self.response_policy = obs.get("response_policy", "NONE")
        y_hat, res = self.twin.step(obs, suspicious=self.alarm_onset_s is not None)

        # shadow controller: golden PI on the same received measurement
        meas = obs["recv"]["LT-101"]
        if self.response_policy == "FALLBACK_TO_CAMERA_OR_TWIN_ESTIMATE":
            meas = self.fallback_level()
        if meas is None or math.isnan(meas):
            meas = self.shadow_last
        self.shadow_last = meas
        g = self.golden
        err = g["setpoint_m"] - meas
        u_unsat = g["u_ff"] + g["kp"] * err + self.shadow_integral + g["ki"] * err * self.dt
        if 0.0 <= u_unsat <= 1.0:
            self.shadow_integral += g["ki"] * err * self.dt
        u_shadow = min(max(g["u_ff"] + g["kp"] * err + self.shadow_integral, 0.0), 1.0)
        shadow_err = (obs["u_cmd"] - obs["delta_u"]) - u_shadow if not obs.get("wm_saturated") else 0.0

        row = {"t": t, "u_cmd": obs["u_cmd"], "u_plc": obs["u_plc"], "valve_cmd": obs["valve_cmd"], "delta_u": obs["delta_u"],
               "wm_saturated": 1.0 if obs["wm_saturated"] else 0.0, "setpoint_reported": obs["setpoint_reported"],
               "shadow_err": shadow_err, "response_active": 0.0 if self.response_policy == "NONE" else 1.0,
               "cam_h": np.nan}
        for s in SENSORS:
            row[s] = obs["recv"][s]
            row[f"yhat:{s}"] = y_hat[s]
            row[f"res:{s}"] = res[s]
            ns = obs["net"][s]
            row[f"lat:{s}"] = ns["latency_s"]
            row[f"arr:{s}"] = ns["arrivals"]
            row[f"seqa:{s}"] = ns["seq_anomalies_total"]
        k = obs["k"]
        cam_period = max(1, int(round(1.0 / (self.cfg.camera_rate_hz * self.dt))))
        if self.camera_provider is not None and k % cam_period == 0:
            cam = self.camera_provider(t)
            if cam is not None:
                self._ingest_camera(cam, t)
        if self.camera_latest and self.camera_latest.get("status") == "OK" and abs(self.camera_latest["timestamp"] - t) < 1e-6:
            row["cam_h"] = self.camera_latest["level_m"]
        self.hist.push(row)

        # extended diagnostic jobs
        for job in list(self.jobs):
            if t >= job["due_s"] and port is not None:
                self.jobs.remove(job)
                self._complete(dg.finalise(job, t, port, self))

        if k % int(round(1.0 / self.dt)) == 0:
            self.analyze(t)

    def _ingest_camera(self, cam: dict, t: float) -> None:
        cam = dict(cam)
        cam.setdefault("timestamp", t)
        quality = {"OK": "GOOD", "LOW_QUALITY": "LOW_QUALITY"}.get(cam.get("status"), "UNAVAILABLE")
        o = self._observe(t, "CAMERA_ESTIMATE", "CAM-01", "TK-101", "waterline_level", cam.get("level_m"), "m",
                          quality=quality, uncertainty=cam.get("uncertainty_m"),
                          provenance={"class": "CAMERA_VISUAL_ESTIMATE", "mode": cam.get("source_mode"),
                                      "simulated": cam.get("source_mode") == "SIMULATED_VISUAL"},
                          metadata={"independence_group": "CAM-01", "reasons": cam.get("reasons", []),
                                    "quality_metrics": cam.get("quality", {})}, key="CAMERA")
        if o is not None:
            cam["observation_id"] = o
        self.camera_latest = cam

    # ----------------------------------------------------------------- analysis
    def analyze(self, t: float) -> None:
        h = self.hist
        if len(h) < 10:
            return
        k = int(round(t / self.dt))
        if self.wm_result is None or k % int(round(2.0 / self.dt)) == 0:
            lat = h.window("lat:LT-101", int(10 / self.dt))
            lat = lat[~np.isnan(lat)]
            self.wm_result = self.watermark.analyze(h, self.twin.inflow_gain_lps(), float(lat.mean()) if lat.size else None)
        if self.emit:
            self._emit_observations(t)
        valid_diag = {tid: r for tid, r in self.diag_results.items()
                      if t - r.get("completed_s", -1e9) <= self.diagnostic_validity_s and r.get("status") == "COMPLETE"}
        ctx = FeatureContext(cfg=self.cfg, hist=h, t=t, alarm_onset_s=self.alarm_onset_s, watermark=self.wm_result,
                             camera=self.camera_latest, diagnostics=valid_diag, obs_refs=self.obs_refs,
                             response_active=self.response_policy != "NONE", shape_memory=self.shape_memory)
        items, diag = extract_features(ctx)
        if self.twin.budget_exceeded:
            from ..schemas import EvidenceItem
            items.append(EvidenceItem(evidence_id="EV:twin_budget_exceeded@TWIN-01", feature="twin_budget_exceeded",
                                      component_id="TWIN-01", strength=0.6,
                                      description="Twin correction budget exhausted: persistent model-vs-measurement disagreement",
                                      independence_group="TWIN", source_type="TWIN_PREDICTION", timestamp=t))
        if self.filter is not None:
            items = [e for e in items if self.filter(e.feature)]
        self.evidence = items
        self.feature_diag = diag

        # alarm detector with hysteresis
        level = max((e.strength for e in items if _is_detection_feature(e.feature)), default=0.0)
        self.anomaly_level = level
        if self.alarm_onset_s is None:
            if level >= 0.35:
                self._raise_count += 1
                if self._raise_count >= 2:
                    self.alarm_onset_s = t - 1.0
                    self.first_alarm_s = self.first_alarm_s if self.first_alarm_s is not None else self.alarm_onset_s
                    self.first_seen = {}
                    self.alarm_history.append({"raised_s": self.alarm_onset_s, "cleared_s": None})
                    self._event(t, "WARNING", "ANOMALY_ALARM", None,
                                f"Anomaly alarm raised (max evidence strength {level:.2f})", {"level": level})
            else:
                self._raise_count = 0
                self.snapshots.append((t, self.twin.snapshot(), self.hist.last("u_cmd"), self.hist.last("valve_cmd")))
        else:
            if level < 0.15:
                self._clear_count += 1
                if self._clear_count >= 30:
                    self.alarm_history[-1]["cleared_s"] = t
                    self._event(t, "NOTICE", "ANOMALY_CLEARED", None, "Anomaly alarm cleared after 30 s of nominal evidence")
                    self.alarm_onset_s = None
                    self._clear_count = 0
                    self._raise_count = 0
            else:
                self._clear_count = 0
        if self.alarm_onset_s is not None:
            for e in items:
                if e.strength >= 0.3 and _is_detection_feature(e.feature):
                    comp = e.component_id.split("/")[0] if e.component_id.startswith("NET-1") else e.component_id
                    self.first_seen.setdefault(comp, t)
                    if e.component_id.startswith("NET-1"):
                        self.first_seen.setdefault(e.component_id, t)

        available = {
            "camera": bool(self.camera_latest and self.camera_latest.get("status") == "OK"),
            "watermark": bool(self.wm_result and any(s.get("detectability") == "DETECTABLE"
                                                     for s in (self.wm_result.get("signals") or {}).values())),
        }
        for tid in valid_diag:
            available[tid] = True
        prev = self.result.get("verdict", {})
        self.result = score(items, self.alarm_onset_s is not None, available)
        v = self.result["verdict"]
        if (v.get("state"), v.get("leading"), v.get("component")) != (prev.get("state"), prev.get("leading"), prev.get("component")):
            self.verdict_log.append({"t": t, **v})
            if v["state"] != "NOMINAL":
                self._event(t, "WARNING" if v["state"] == "INCONCLUSIVE" else "CRITICAL", "VERDICT_CHANGE", v.get("component"),
                            f"Verdict: {v['state']}" + (f" — {v['leading']} @ {v.get('component')}" if v.get("leading") else "")
                            + f". {v['reason']}", {"verdict": v})
            if self.sink:
                self.sink("hypotheses", {"t": t, "result": self.result})

    # ------------------------------------------------------------- observations
    def _observe(self, t, source_type, source_id, component_id, mtype, value, unit, *, quality="GOOD", uncertainty=None,
                 freshness=0.0, provenance=None, metadata=None, key=None) -> str | None:
        if not self.emit:
            return None
        if isinstance(value, float) and math.isnan(value):
            value, quality = None, "BAD"
        o = Observation(timestamp=t, source_type=source_type, source_id=source_id, component_id=component_id,
                        measurement_type=mtype, value=value, unit=unit, quality=quality, uncertainty=uncertainty,
                        freshness=freshness, provenance=provenance or {}, metadata=metadata or {})
        self.observations.append(o)
        if key:
            self.obs_refs[key] = o.observation_id
        if self.sink:
            self.sink("observation", o)
        return o.observation_id

    def _emit_observations(self, t: float) -> None:
        h = self.hist
        comp = {"LT-101": "TK-101", "FT-101": "P-101", "FT-102": "FV-101", "ZT-101": "FV-101", "SC-101": "P-101"}
        spec = {"LT-101": self.cfg.sensors.level_noise_m, "FT-101": self.cfg.sensors.flow_noise_lps,
                "FT-102": self.cfg.sensors.flow_noise_lps, "ZT-101": self.cfg.sensors.valve_pos_noise_pct,
                "SC-101": self.cfg.sensors.pump_speed_noise}
        for s in SENSORS:
            v = h.last(s)
            lat = h.last(f"lat:{s}")
            sid = self._observe(t, "SENSOR", s, comp[s], "value", v, SENSOR_UNITS[s], uncertainty=spec[s],
                                freshness=0.0 if math.isnan(lat) else lat,
                                provenance={"class": "SENSOR_OBSERVATION", "path": f"{s} → NET-1 → PLC-101"},
                                metadata={"independence_group": s}, key=f"SENSOR:{s}")
            r = h.last(f"res:{s}")
            self._observe(t, "TWIN_PREDICTION", "TWIN-01", comp[s], f"predicted:{s}", h.last(f"yhat:{s}"), SENSOR_UNITS[s],
                          uncertainty=residual_sigma(self.cfg)[s],
                          provenance={"class": "DIGITAL_TWIN_PREDICTION", "derived_from": [sid], "inputs": "issued commands"},
                          metadata={"independence_group": s, "residual": None if math.isnan(r) else r,
                                    "residual_convention": "r = y_observed - y_hat_predicted",
                                    "correction_frozen": self.twin.frozen_reason}, key=f"TWIN:{s}")
            self._observe(t, "NETWORK_MONITOR", "NETMON-1", f"NET-1/{s}", "latency", None if math.isnan(lat) else lat, "s",
                          provenance={"class": "SENSOR_OBSERVATION"}, metadata={"independence_group": f"NET:{s}"}, key=f"NET:{s}")
        self._observe(t, "CONTROLLER_LOG", "DRIVE-P101", "P-101", "pump_command", h.last("u_cmd"), "0..1",
                      provenance={"class": "SENSOR_OBSERVATION", "path": "drive-side command log"},
                      metadata={"independence_group": "PLC", "valve_command": h.last("valve_cmd"),
                                "setpoint_reported": h.last("setpoint_reported")}, key="CTRL:u_cmd")
        self._observe(t, "CONTROLLER_LOG", "SHADOW-PI", "PLC-101", "control_law_deviation", h.last("shadow_err"), "0..1",
                      provenance={"class": "INFERENCE", "method": "golden PI replica on received measurement"},
                      metadata={"independence_group": "PLC"}, key="CTRL:shadow")
        if self.wm_result and self.wm_result.get("signals"):
            lt = self.wm_result["signals"].get("LT-101", {})
            self._observe(t, "WATERMARK", "WM-1", "P-101", "correlation:LT-101", lt.get("correlation"), "rho",
                          quality="GOOD" if lt.get("detectability") == "DETECTABLE" else "UNCERTAIN",
                          provenance={"class": "INFERENCE", "method": "sliding-window template cross-correlation"},
                          metadata={"independence_group": "LT-101", "signals": self.wm_result["signals"],
                                    "warnings": self.wm_result.get("warnings", [])}, key="WATERMARK")

    def _event(self, t, severity, category, component, message, data=None):
        if self.sink:
            self.sink("event", {"timestamp": t, "severity": severity, "category": category, "component_id": component,
                                "message": message, "provenance": "INFERENCE", "data": data or {}})

    # ----------------------------------------------------------------- helpers
    def fallback_level(self) -> float:
        cam = self.camera_latest
        if cam and cam.get("status") == "OK" and cam.get("level_m") is not None and self.last_t - cam["timestamp"] < 3.0:
            return float(cam["level_m"])
        return self.twin.h

    def fallback_source(self) -> str:
        cam = self.camera_latest
        if cam and cam.get("status") == "OK" and cam.get("level_m") is not None and self.last_t - cam["timestamp"] < 3.0:
            return "CAMERA_ESTIMATE"
        return "TWIN_PREDICTION"

    def verified_snapshot(self):
        if not self.snapshots:
            return None
        if self.alarm_onset_s is None:
            t, snap, *_ = self.snapshots[-1]
            return (t, snap) if self.last_t - t >= 10 else (self.snapshots[0][0], self.snapshots[0][1])
        target = self.alarm_onset_s - 10.0
        best = None
        for t, snap, *_ in self.snapshots:
            if t <= target:
                best = (t, snap)
        return best or (self.snapshots[0][0], self.snapshots[0][1])

    def last_verified_commands(self) -> tuple[float, float] | None:
        for t, _snap, u, v in reversed(self.snapshots):
            if self.alarm_onset_s is None or t <= self.alarm_onset_s - 5.0:
                return (u, v)
        return None

    def anomalous_components(self) -> set[str]:
        return {e.component_id for e in self.evidence if e.strength >= 0.3 and _is_detection_feature(e.feature)}

    def camera_ready(self) -> bool:
        return self.camera_provider is not None and bool(self.camera_latest) and self.camera_latest.get("status") in ("OK", "LOW_QUALITY")

    def capture_camera(self, t: float) -> dict | None:
        if self.camera_provider is None:
            return None
        cam = self.camera_provider(t)
        if cam is not None:
            self._ingest_camera(cam, t)
        return self.camera_latest

    def diagnostic_context(self) -> dict:
        lvl = self.hist.last("LT-101")
        if math.isnan(lvl) or self.alarm_onset_s is not None:
            lvl = self.twin.h  # conservative: do not trust a possibly compromised sensor for preconditions
        return {"level_estimate_m": lvl, "safety": self.cfg.safety, "response_policy": self.response_policy,
                "running": [j["test_id"] for j in self.jobs], "watermark_enabled": self.cfg.watermark.enabled,
                "saturation_fraction": (self.wm_result or {}).get("saturation_fraction", 0.0),
                "camera_ready": self.camera_ready(),
                "recent": {tid: r["completed_s"] for tid, r in self.diag_results.items()
                           if self.last_t - r.get("completed_s", -1e9) <= self.diagnostic_validity_s}}

    def recommend(self) -> list[dict]:
        return dg.rank_tests(self.result.get("candidates", []), self.diagnostic_context())

    def run_diagnostic(self, test_id: str, port: dg.DiagnosticPort, force: bool = False) -> dict:
        if test_id not in dg.REGISTRY:
            raise KeyError(test_id)
        ok, reasons = dg.check_preconditions(test_id, self.diagnostic_context())
        if not ok and not (force and test_id == "CAPTURE_VISUAL_WITNESS_FRAME"):
            res = {"test_id": test_id, "status": "REJECTED", "precondition_failures": reasons, "started_s": self.last_t,
                   "completed_s": self.last_t, "evidence": []}
            self.diag_log.append(res)
            return res
        before = self._score_snapshot()
        job = dg.start(test_id, self.last_t, port)
        job["scores_before"] = before
        if job["due_s"] <= self.last_t:
            return self._complete(dg.finalise(job, self.last_t, port, self))
        self.jobs.append(job)
        self._event(self.last_t, "INFO", "DIAGNOSTIC_STARTED", None, f"Diagnostic {test_id} started ({job['due_s'] - self.last_t:.0f} s)")
        return job

    def _score_snapshot(self) -> dict:
        return {h["hypothesis_type"]: {"score": h["score"], "raw_score": h["raw_score"], "component_id": h["component_id"]}
                for h in self.result.get("hypotheses", [])}

    def _complete(self, res: dict) -> dict:
        if res.get("observation") and self.emit:
            o = Observation(**res["observation"])
            self.observations.append(o)
            if self.sink:
                self.sink("observation", o)
        self.diag_results[res["test_id"]] = res
        self.analyze(self.last_t)  # live recalculation (AT-014)
        res["scores_after"] = self._score_snapshot()
        res["verdict_after"] = self.result.get("verdict")
        self.diag_log.append(res)
        self._event(self.last_t, "NOTICE", "DIAGNOSTIC_RESULT", None,
                    f"{res['test_id']}: {res.get('outcome', res.get('status'))} — {res.get('summary', '')}"[:400],
                    {"test_id": res["test_id"], "outcome": res.get("outcome")})
        if self.sink:
            self.sink("diagnostic", res)
        return res
