"""Live simulation engine (FR-001 – FR-004).

Single-loop integrity: one asyncio task at most, guarded by an asyncio.Lock;
start/stop/pause/reset are idempotent. All buffers are bounded.
"""
from __future__ import annotations

import asyncio
import copy
import math
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .camera.manager import CameraManager
from .config import INFERENCE_VERSION, MODEL_VERSION, TWIN_VERSION, SimConfig
from .evaluation.harness import directive_for, refresh_directive
from .inference.causal import build_graph
from .inference.features import SENSORS
from .inference.pipeline import Defender
from .persistence.db import Database
from .schemas import new_id
from .sim.scenarios import ScenarioSpec
from .sim.world import RESPONSE_POLICIES, World

VERSIONS = {"plant": MODEL_VERSION, "twin": TWIN_VERSION, "inference": INFERENCE_VERSION}
TICK_S = 0.05
MAX_STEPS_PER_TICK = 400


class LiveEngine:
    def __init__(self, db: Database, data_dir: Path):
        self.db = db
        self.data_dir = Path(data_dir)
        self.lock = asyncio.Lock()
        self.task: asyncio.Task | None = None
        self.status = "STOPPED"
        self.speed = 1.0
        self.cfg = SimConfig()
        self.camera = CameraManager()
        for cal in db.active_calibrations():
            self.camera.set_calibration(cal["mode"], cal)
        self.subscribers: set[asyncio.Queue] = set()
        self.loop_count = 0  # number of loop tasks ever started (single-loop test)
        self.steps_total = 0
        self.points: deque = deque(maxlen=1200)
        self._last_frame_saved = -1e9
        self._new_run(self.cfg, label="initial")

    # ----------------------------------------------------------------- lifecycle
    def _new_run(self, cfg: SimConfig, label: str = "") -> None:
        self.cfg = cfg
        self.run_id = "RUN-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        self.world = World(cfg)
        self.world.exogenous_log = deque(maxlen=20000)
        self.camera.set_scene_provider(self.world.camera_scene_truth)
        if self.camera.mode == "SIMULATED_VISUAL":
            self.camera.select("SIMULATED_VISUAL")
        self.defender = Defender(cfg, mode="FULL", camera_provider=self._camera_provider, emit=True, sink=self._sink)
        self.points.clear()
        self.response_log: list[dict] = []
        self.db.create_run(self.run_id, cfg.seed, cfg.model_dump(), label, VERSIONS)
        self.frames_dir = self.data_dir / "runs" / self.run_id / "frames"
        self._last_frame_saved = -1e9
        self._emit_event(0.0, "INFO", "RUN_CREATED", None, f"Run {self.run_id} created (seed {cfg.seed})", "SENSOR_OBSERVATION")

    async def start(self) -> dict:
        async with self.lock:
            if self.task is not None and not self.task.done():
                if self.status == "PAUSED":
                    self.status = "RUNNING"
                return self.state_summary()
            self.status = "RUNNING"
            self.loop_count += 1
            self.task = asyncio.create_task(self._loop(), name="phystrace-sim-loop")
            self.db.update_run(self.run_id, status="RUNNING")
            self._emit_event(self.world.t, "INFO", "SIM_STARTED", None, "Simulation loop started", "SENSOR_OBSERVATION")
            return self.state_summary()

    async def pause(self) -> dict:
        async with self.lock:
            if self.status == "RUNNING":
                self.status = "PAUSED"
                self.db.update_run(self.run_id, status="PAUSED")
            return self.state_summary()

    async def stop(self) -> dict:
        async with self.lock:
            await self._stop_locked()
            return self.state_summary()

    async def _stop_locked(self) -> None:
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass
            self.task = None
        if self.status != "STOPPED":
            self.status = "STOPPED"
            self.db.update_run(self.run_id, status="STOPPED", sim_time_s=self.world.t)
            self._emit_event(self.world.t, "INFO", "SIM_STOPPED", None, "Simulation loop stopped", "SENSOR_OBSERVATION")

    async def reset(self, cfg: SimConfig | None = None, label: str = "") -> dict:
        async with self.lock:
            await self._stop_locked()
            self._new_run(cfg or self.cfg.model_copy(deep=True), label)
            return self.state_summary()

    def loop_running(self) -> bool:
        return self.task is not None and not self.task.done()

    async def _loop(self) -> None:
        last = time.monotonic()
        acc = 0.0
        last_push = 0.0
        try:
            while True:
                await asyncio.sleep(TICK_S)
                now = time.monotonic()
                elapsed, last = now - last, now
                if self.status != "RUNNING":
                    continue
                acc += elapsed * self.speed
                n = min(int(acc / self.cfg.dt_s), MAX_STEPS_PER_TICK)
                acc -= n * self.cfg.dt_s
                acc = min(acc, self.cfg.dt_s * MAX_STEPS_PER_TICK)  # never accumulate an unbounded backlog
                for _ in range(n):
                    self._step()
                if now - last_push >= 0.25:
                    last_push = now
                    self._broadcast(self.live_payload())
        except asyncio.CancelledError:
            raise

    def _step(self) -> None:
        obs, truth = self.world.step()
        self.defender.step(obs, port=self.world)
        refresh_directive(self.world, self.defender)
        self.steps_total += 1
        k = obs["k"]
        if k % 5 == 0:  # 2 Hz flight-recorder telemetry (observables only)
            row = self._telemetry_row(obs)
            self.points.append(row)
            self.db.add_telemetry(self.run_id, row["t"], row)
        if k % 10 == 0:  # 1 Hz ground truth -> evaluation ledger only
            self.db.add_truth(self.run_id, truth["t"], {k2: truth[k2] for k2 in
                                                       ("t", "h", "n", "pos", "q_in_lps", "q_out_lps", "overflow_m3",
                                                        "unmet_m3", "energy_j", "active_scenarios")})
            self.db.update_run(self.run_id, sim_time_s=truth["t"])

    def _telemetry_row(self, obs: dict) -> dict:
        h = self.defender.hist
        row = {"t": round(obs["t"], 2)}
        for s in SENSORS:
            v = obs["recv"][s]
            row[s] = None if v != v else round(v, 5)
            yh = h.last(f"yhat:{s}")
            row[f"yhat:{s}"] = None if yh != yh else round(yh, 5)
            r = h.last(f"res:{s}")
            row[f"res:{s}"] = None if r != r else round(r, 5)
            row[f"lat:{s}"] = None if obs["net"][s]["latency_s"] != obs["net"][s]["latency_s"] else round(obs["net"][s]["latency_s"], 3)
        row.update(u_cmd=round(obs["u_cmd"], 4), u_plc=round(obs["u_plc"], 4), valve_cmd=round(obs["valve_cmd"], 4),
                   delta_u=round(obs["delta_u"], 4), setpoint_reported=obs["setpoint_reported"],
                   response=obs["response_policy"], alarm=self.defender.alarm_onset_s is not None,
                   anomaly_level=round(self.defender.anomaly_level, 3))
        cam = self.defender.camera_latest
        row["cam_h"] = cam.get("level_m") if cam and cam.get("status") == "OK" and abs(cam["timestamp"] - obs["t"]) < 1.5 else None
        row["cam_status"] = cam.get("status") if cam else None
        return row

    # --------------------------------------------------------------- camera
    def _camera_provider(self, t: float) -> dict | None:
        if self.camera.mode == "NONE":
            return None
        est = self.camera.capture(t)
        if t - self._last_frame_saved >= 1.0 and self.camera.last_frame is not None:
            self._last_frame_saved = t
            try:
                self.frames_dir.mkdir(parents=True, exist_ok=True)
                p = self.frames_dir / f"{t:010.2f}.jpg"
                cv2.imwrite(str(p), self.camera.last_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                self.db.add_frame(self.run_id, t, str(p), est.get("status"), est.get("level_m"), self.camera.mode)
            except Exception:  # noqa: BLE001
                pass
        return est

    # ------------------------------------------------------------------ sinks
    def _sink(self, kind: str, payload: Any) -> None:
        if kind == "observation":
            self.db.add_observation(self.run_id, payload)
        elif kind == "event":
            self._emit_event(payload["timestamp"], payload["severity"], payload["category"], payload.get("component_id"),
                             payload["message"], payload.get("provenance", "INFERENCE"), payload.get("data"))
        elif kind == "hypotheses":
            self.db.add_hypothesis_snapshot(self.run_id, payload["t"], payload["result"], new_id("SNAP"))
        elif kind == "diagnostic":
            self.db.add_diagnostic(self.run_id, new_id("DIAG"), _jsonable(payload))

    def _emit_event(self, t, severity, category, component, message, provenance="INFERENCE", data=None) -> dict:
        e = {"event_id": new_id("EVT"), "run_id": self.run_id, "timestamp": round(float(t), 3), "severity": severity,
             "category": category, "component_id": component, "message": message, "provenance": provenance,
             "data": _jsonable(data or {})}
        self.db.add_event(e)
        self._broadcast({"type": "event", "event": e})
        return e

    # ----------------------------------------------------------- websockets
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=8)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)

    def _broadcast(self, msg: dict) -> None:
        for q in list(self.subscribers):
            if q.full():
                try:
                    q.get_nowait()  # drop oldest: slow clients never cause unbounded growth
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    # -------------------------------------------------------------- actions
    def inject(self, spec: ScenarioSpec, relative: bool = True) -> dict:
        spec = spec.resolved()
        if relative:
            spec = spec.model_copy(update={"start_s": self.world.t + spec.start_s})
        self.world.add_scenario(spec)
        # Ground truth goes to the evaluation ledger only - no inference-visible event.
        self.db.record_ground_truth(self.run_id, self.world.t, spec.model_dump())
        return {"scheduled": True, "scenario_id": spec.scenario_id, "start_s": spec.start_s,
                "note": "Recorded in the evaluation ledger. Hidden from the inference engine."}

    def clear_scenarios(self) -> None:
        self.world.injector.clear()
        self.db.record_ground_truth(self.run_id, self.world.t, {"type": "CLEAR_ALL", "start_s": self.world.t})

    def run_diagnostic(self, test_id: str) -> dict:
        res = self.defender.run_diagnostic(test_id, self.world)
        return _jsonable(res)

    def apply_response(self, policy: str) -> dict:
        if policy not in RESPONSE_POLICIES:
            raise ValueError(policy)
        d = directive_for(policy, self.defender, self.world.t)
        self.world.set_response(d)
        info = {"policy": policy, "activated_at_s": self.world.t, "fallback_source": self.defender.fallback_source()
                if policy == "FALLBACK_TO_CAMERA_OR_TWIN_ESTIMATE" else None,
                "hold_u": d.hold_u, "hold_valve": d.hold_valve}
        self.response_log.append(info)
        self.db.add_response(self.run_id, self.world.t, policy, info)
        self._emit_event(self.world.t, "WARNING" if policy != "NONE" else "NOTICE", "OPERATOR_RESPONSE", None,
                         f"Simulated response applied: {policy}", "INFERENCE", info)
        return info

    def defender_snapshot(self) -> Defender:
        """Copy for off-loop analysis (planner)."""
        d = copy.copy(self.defender)
        d.hist = copy.deepcopy(self.defender.hist)
        d.twin = copy.deepcopy(self.defender.twin)
        d.result = copy.deepcopy(self.defender.result)
        d.snapshots = copy.deepcopy(self.defender.snapshots)
        d.diag_results = copy.deepcopy(self.defender.diag_results)
        d.sink = None
        return d

    # ---------------------------------------------------------------- views
    def state_summary(self) -> dict:
        return {"run_id": self.run_id, "status": self.status, "t": round(self.world.t, 2), "speed": self.speed,
                "seed": self.cfg.seed, "loop_running": self.loop_running(), "loop_tasks_started": self.loop_count,
                "steps_total": self.steps_total, "db_dropped_writes": self.db.dropped, "versions": VERSIONS}

    def live_payload(self) -> dict:
        d, w = self.defender, self.world
        h = d.hist
        obs = d.last_obs or {}
        observed = {s: _f(h.last(s)) for s in SENSORS}
        twin = {s: _f(h.last(f"yhat:{s}")) for s in SENSORS}
        res = {s: _f(h.last(f"res:{s}")) for s in SENSORS}
        net = {s: {"latency_s": _f(h.last(f"lat:{s}")), **{k: v for k, v in (obs.get("net", {}).get(s, {}) or {}).items()
                                                          if k in ("drops_total", "seq_anomalies_total")}} for s in SENSORS}
        wm = d.wm_result or {}
        cam = self.camera.status()
        hyps = [{k: x[k] for k in ("hypothesis_type", "component_id", "score", "raw_score", "status", "score_type")}
                for x in d.result.get("hypotheses", [])[:4]]
        new_points = list(self.points)[-6:]
        return _jsonable({
            "type": "state", **self.state_summary(),
            "observed": observed, "twin": twin, "residuals": res,
            "twin_state": {"h": d.twin.h, "n": d.twin.n, "pos": d.twin.pos, "correction_frozen": d.twin.frozen_reason,
                           "budget_exceeded": d.twin.budget_exceeded, "invalid_regime": d.twin.invalid_regime},
            "commands": {"u_cmd": obs.get("u_cmd"), "u_plc": obs.get("u_plc"), "valve_cmd": obs.get("valve_cmd"),
                         "delta_u": obs.get("delta_u"), "wm_eps": obs.get("wm_eps"), "wm_symbol": obs.get("wm_symbol"),
                         "setpoint_reported": obs.get("setpoint_reported")},
            "network": net,
            "watermark": {"status": wm.get("status"), "signals": wm.get("signals"), "warnings": wm.get("warnings", []),
                          "epsilon": wm.get("epsilon"), "scale": w.wm_scale},
            "camera": {"mode": cam["mode"], "state": cam["source"]["state"], "calibrated": cam["calibrated"],
                       "simulated": cam["simulated"], "estimate": cam["estimate"], "error": cam["source"].get("error")},
            "alarm": {"active": d.alarm_onset_s is not None, "onset_s": d.alarm_onset_s, "level": d.anomaly_level,
                      "first_alarm_s": d.first_alarm_s},
            "verdict": d.result.get("verdict"), "hypotheses": hyps,
            "response": {"policy": w.response.policy, "activated_at_s": w.response.activated_at_s,
                         "fallback_source": d.fallback_source() if w.response.policy == "FALLBACK_TO_CAMERA_OR_TWIN_ESTIMATE" else None},
            "jobs": [{"test_id": j["test_id"], "due_s": j["due_s"], "started_s": j["started_s"]} for j in d.jobs],
            "limits": self.cfg.safety.model_dump(),
            "plant": {"area_m2": self.cfg.plant.area_m2, "tank_height_m": self.cfg.plant.tank_height_m,
                      "setpoint_m": self.cfg.controller.setpoint_m},
            "points": new_points,
        })

    def telemetry(self, seconds: float = 300.0) -> list[dict]:
        n = int(seconds * 2)
        return list(self.points)[-n:]

    def causal_graph(self) -> dict:
        return _jsonable(build_graph(self.defender))


def _f(v):
    if v is None:
        return None
    try:
        return None if math.isnan(v) else round(float(v), 5)
    except TypeError:
        return v


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, deque)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if hasattr(o, "model_dump"):
        return _jsonable(o.model_dump())
    return o
