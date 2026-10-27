"""FastAPI application: REST + WebSocket API for the PhysTrace workstation."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from ..camera.manager import MODES as CAMERA_MODES
from ..config import SimConfig
from ..engine import VERSIONS, LiveEngine, _jsonable
from ..evaluation.ablation import AblationRequest, build_cases, run_ablation
from ..evaluation.counterfactual import CounterfactualRequest, run_counterfactual
from ..evaluation.reports import build_report, experiment_csv, telemetry_csv
from ..inference import diagnostics as dg
from ..persistence.db import Database
from ..response.planner import plan
from ..schemas import CameraCalibration, new_id
from ..sim.scenarios import ScenarioSpec, catalog_for_api
from ..sim.world import RESPONSE_POLICIES

DATA_DIR = Path(os.environ.get("PHYSTRACE_DATA", Path(__file__).resolve().parents[3] / "data"))
_threads = ThreadPoolExecutor(max_workers=4, thread_name_prefix="phystrace-job")
_procs: ProcessPoolExecutor | None = None


def _proc_pool() -> ProcessPoolExecutor:
    global _procs
    if _procs is None:
        _procs = ProcessPoolExecutor(max_workers=2)
    return _procs


@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = Database(os.environ.get("PHYSTRACE_DB", DATA_DIR / "phystrace.sqlite3"))
    # Experiments interrupted by a restart are marked failed rather than left "RUNNING" forever.
    for e in db.list_experiments():
        if e["status"] in ("RUNNING", "PENDING"):
            full = db.get_experiment(e["experiment_id"])
            full.update(status="FAILED", error="interrupted by server restart")
            db.save_experiment(full)
    app.state.db = db
    app.state.engine = LiveEngine(db, DATA_DIR)
    yield
    await app.state.engine.stop()
    app.state.engine.camera.stop()
    db.close()
    if _procs is not None:
        _procs.shutdown(cancel_futures=True)


app = FastAPI(title="PhysTrace", version="1.0.0", lifespan=lifespan)
# Any loopback port: Vite moves to 5174+ when 5173 is taken. Non-local origins stay blocked.
app.add_middleware(CORSMiddleware, allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
                   allow_methods=["*"], allow_headers=["*"])


def eng(request: Request) -> LiveEngine:
    return request.app.state.engine


def dbh(request: Request) -> Database:
    return request.app.state.db


# ------------------------------------------------------------------ health / state
@app.get("/api/health")
def health(request: Request):
    e = eng(request)
    return {"status": "ok", "service": "phystrace-backend", "time": time.time(), "versions": VERSIONS,
            "simulation": e.state_summary(), "camera_mode": e.camera.mode,
            "safety_boundary": "Simulation only: no external network scanning, no commands to real PLCs or equipment."}


@app.get("/api/state")
def state(request: Request):
    return eng(request).live_payload()


class ResetBody(BaseModel):
    seed: int | None = None
    label: str = ""
    config: SimConfig | None = None


@app.post("/api/sim/start")
async def sim_start(request: Request):
    return await eng(request).start()


@app.post("/api/sim/pause")
async def sim_pause(request: Request):
    return await eng(request).pause()


@app.post("/api/sim/stop")
async def sim_stop(request: Request):
    return await eng(request).stop()


@app.post("/api/sim/reset")
async def sim_reset(request: Request, body: ResetBody | None = None):
    e = eng(request)
    body = body or ResetBody()
    cfg = (body.config or e.cfg).model_copy(deep=True)
    if body.seed is not None:
        cfg.seed = body.seed
    return await e.reset(cfg, body.label)


class SpeedBody(BaseModel):
    speed: float = Field(1.0, ge=0.1, le=20.0)


@app.put("/api/sim/speed")
def sim_speed(request: Request, body: SpeedBody):
    eng(request).speed = body.speed
    return eng(request).state_summary()


@app.get("/api/sim/config")
def sim_config(request: Request):
    return eng(request).cfg.model_dump()


@app.get("/api/telemetry")
def telemetry(request: Request, seconds: float = 300.0):
    return {"provenance": "observables (sensor / twin / controller log)", "points": eng(request).telemetry(seconds)}


# -------------------------------------------------------------------- scenarios
@app.get("/api/scenarios/catalog")
def scenario_catalog():
    return {"scenarios": catalog_for_api(), "note": "Simulation-only attack & fault lab."}


class InjectBody(BaseModel):
    type: str
    target: str | None = None
    start_in_s: float = Field(5.0, ge=0)
    duration_s: float | None = None
    params: dict[str, float] = Field(default_factory=dict)
    seed: int = 0


@app.post("/api/scenarios")
def inject(request: Request, body: InjectBody):
    try:
        spec = ScenarioSpec(type=body.type, target=body.target, start_s=body.start_in_s, duration_s=body.duration_s,
                            params=body.params, seed=body.seed)
    except Exception as ex:  # noqa: BLE001
        raise HTTPException(422, str(ex))
    return eng(request).inject(spec, relative=True)


@app.post("/api/scenarios/clear")
def clear_scenarios(request: Request):
    eng(request).clear_scenarios()
    return {"cleared": True}


# ------------------------------------------------------------- inference views
@app.get("/api/hypotheses")
def hypotheses(request: Request):
    d = eng(request).defender
    return _jsonable({**d.result, "t": d.last_t, "alarm_onset_s": d.alarm_onset_s})


@app.get("/api/evidence")
def evidence(request: Request):
    d = eng(request).defender
    return _jsonable({"t": d.last_t, "evidence": [e.model_dump() for e in d.evidence], "diagnostics": d.feature_diag,
                      "watermark": d.wm_result})


@app.get("/api/observations")
def observations(request: Request, limit: int = 200, source_type: str | None = None):
    d = eng(request).defender
    obs = [o.model_dump() for o in list(d.observations)[-2000:] if source_type is None or o.source_type == source_type]
    return _jsonable({"observations": obs[-limit:]})


@app.get("/api/causal-graph")
def causal_graph(request: Request):
    return eng(request).causal_graph()


# ------------------------------------------------------------------ diagnostics
@app.get("/api/diagnostics")
def diagnostics(request: Request):
    d = eng(request).defender
    return _jsonable({"ranking": d.recommend(), "running": d.jobs, "log": d.diag_log[-50:],
                      "method": dg.__doc__})


@app.post("/api/diagnostics/{test_id}/run")
def run_diag(request: Request, test_id: str):
    if test_id not in dg.REGISTRY:
        raise HTTPException(404, f"unknown test {test_id}")
    return eng(request).run_diagnostic(test_id)


# -------------------------------------------------------------------- responses
@app.post("/api/responses/evaluate")
async def responses_evaluate(request: Request, horizon_s: float = 300.0):
    e = eng(request)
    snap = e.defender_snapshot()
    cfg = e.cfg
    res = await asyncio.get_running_loop().run_in_executor(_threads, plan, cfg, snap, horizon_s)
    return _jsonable(res)


class ResponseBody(BaseModel):
    policy: str


@app.post("/api/responses/apply")
def responses_apply(request: Request, body: ResponseBody):
    if body.policy not in RESPONSE_POLICIES:
        raise HTTPException(422, f"policy must be one of {RESPONSE_POLICIES}")
    return eng(request).apply_response(body.policy)


# --------------------------------------------------------------- counterfactual
def _new_experiment(db: Database, kind: str, config: dict, label: str = "") -> dict:
    exp = {"experiment_id": new_id("EXP"), "kind": kind, "created_at": time.time(), "status": "RUNNING",
           "config": config, "result": None, "model_versions": VERSIONS, "error": None, "progress": 0.0, "label": label}
    db.save_experiment(exp)
    return exp


async def _finish(db: Database, exp: dict, fut) -> None:
    try:
        res = await asyncio.wrap_future(fut)
        exp.update(status="COMPLETE", result=_jsonable(res), progress=1.0)
    except Exception as ex:  # noqa: BLE001
        exp.update(status="FAILED", error=f"{type(ex).__name__}: {ex}")
    db.save_experiment(exp)


@app.post("/api/counterfactual")
async def counterfactual(request: Request, body: CounterfactualRequest):
    if body.policy not in RESPONSE_POLICIES:
        raise HTTPException(422, f"policy must be one of {RESPONSE_POLICIES}")
    db = dbh(request)
    exp = _new_experiment(db, "COUNTERFACTUAL", body.model_dump(), f"{body.policy} vs no response")
    fut = _proc_pool().submit(run_counterfactual, body)
    asyncio.create_task(_finish(db, exp, fut))
    return {"experiment_id": exp["experiment_id"], "status": "RUNNING"}


class CounterfactualFromLive(BaseModel):
    policy: str = "BOUNDED_COMMAND_LIMITING"
    trigger: str = "ON_DETECTION"
    intervention_time_s: float | None = None
    duration_s: float = 400.0


@app.post("/api/counterfactual/from-live")
async def counterfactual_from_live(request: Request, body: CounterfactualFromLive):
    """Replays the live run's configuration and ground-truth scenario ledger (evaluation side) in a paired experiment."""
    e = eng(request)
    gt = dbh(request).ground_truth(e.run_id)
    specs = [ScenarioSpec(**g["scenario"]) for g in gt if g["scenario"].get("type") != "CLEAR_ALL"]
    req = CounterfactualRequest(config=e.cfg.model_copy(deep=True), scenarios=specs, policy=body.policy,
                                trigger=body.trigger, intervention_time_s=body.intervention_time_s,
                                duration_s=max(body.duration_s, max((s.start_s for s in specs), default=0) + 120))
    return await counterfactual(request, req)


@app.get("/api/experiments")
def experiments(request: Request, kind: str | None = None):
    return {"experiments": dbh(request).list_experiments(kind)}


@app.get("/api/experiments/{eid}")
def experiment(request: Request, eid: str):
    exp = dbh(request).get_experiment(eid)
    if exp is None:
        raise HTTPException(404, "experiment not found")
    return exp


# ---------------------------------------------------------------------- ablation
@app.post("/api/ablation")
async def ablation(request: Request, body: AblationRequest):
    db = dbh(request)
    exp = _new_experiment(db, "ABLATION", body.model_dump(), f"ablation {body.preset or 'custom'}")

    def progress(i, n):
        db.update_experiment_progress(exp["experiment_id"], i / n)

    fut = _threads.submit(run_ablation, body, progress)
    asyncio.create_task(_finish(db, exp, fut))
    return {"experiment_id": exp["experiment_id"], "status": "RUNNING", "n_cases": len(build_cases(body))}


# ------------------------------------------------------------------------ camera
class CameraModeBody(BaseModel):
    mode: str
    device_index: int = 0
    video: str | None = None
    loop: bool = True


@app.get("/api/camera/status")
def camera_status(request: Request):
    return _jsonable(eng(request).camera.status())


@app.post("/api/camera/mode")
def camera_mode(request: Request, body: CameraModeBody):
    e = eng(request)
    if body.mode not in CAMERA_MODES:
        raise HTTPException(422, f"mode must be one of {CAMERA_MODES}")
    path = None
    if body.mode == "VIDEO_REPLAY":
        if not body.video:
            raise HTTPException(422, "video filename required")
        path = _video_path(body.video)
        if not path.is_file():
            raise HTTPException(404, "video not found in local video library")
    st = e.camera.select(body.mode, device_index=body.device_index, video_path=str(path) if path else None, loop=body.loop)
    e._emit_event(e.world.t, "NOTICE", "CAMERA_SOURCE_SELECTED", "CAM-01",
                  f"Camera source explicitly selected: {body.mode} → {st['source']['state']}"
                  + (f" ({st['source']['error']})" if st["source"].get("error") else ""), "CAMERA_VISUAL_ESTIMATE")
    return _jsonable(st)


@app.post("/api/camera/stop")
def camera_stop(request: Request):
    eng(request).camera.stop()
    return {"mode": "NONE"}


@app.post("/api/camera/capture")
def camera_capture(request: Request):
    e = eng(request)
    return _jsonable(e.camera.capture(e.world.t))


@app.get("/api/camera/frame.jpg")
def camera_frame(request: Request, annotated: bool = True):
    e = eng(request)
    if e.camera.mode != "NONE" and e.status != "RUNNING":
        e.camera.capture(e.world.t)  # keep preview fresh while paused (no substitution on failure)
    jpg = e.camera.jpeg(annotated)
    if jpg is None:
        return JSONResponse({"status": "UNAVAILABLE", "mode": e.camera.mode}, status_code=404)
    return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


class CalibrationBody(BaseModel):
    mode: str
    roi: dict[str, int]
    points: list[dict[str, float]]
    indicator_roi: dict[str, int] | None = None
    component_id: str = "TK-101"
    notes: str = ""


@app.post("/api/camera/calibration")
def camera_calibrate(request: Request, body: CalibrationBody):
    if len(body.points) != 2 or abs(body.points[0]["pixel_y"] - body.points[1]["pixel_y"]) < 2:
        raise HTTPException(422, "two calibration points with distinct pixel rows are required")
    cal = CameraCalibration(mode=body.mode, roi=body.roi, points=body.points, indicator_roi=body.indicator_roi,
                            component_id=body.component_id, created_at=time.time(), notes=body.notes).model_dump()
    dbh(request).save_calibration(cal)
    eng(request).camera.set_calibration(body.mode, cal)
    return cal


@app.post("/api/camera/calibration/simulated-default")
def camera_calibrate_sim(request: Request):
    d = eng(request).camera.default_simulated_calibration()
    body = CalibrationBody(mode="SIMULATED_VISUAL", roi=d["roi"], points=d["points"], indicator_roi=d["indicator_roi"],
                           notes="Known synthetic geometry of the SIMULATED_VISUAL renderer")
    return camera_calibrate(request, body)


@app.get("/api/camera/calibration")
def camera_calibrations(request: Request):
    return {"calibrations": dbh(request).active_calibrations()}


class DegradeBody(BaseModel):
    blur: int = 0
    brightness: float = 1.0
    occlude: bool = False
    freeze: bool = False


@app.post("/api/camera/simulated/degrade")
def camera_degrade(request: Request, body: DegradeBody):
    src = eng(request).camera.source
    if src is None or src.mode != "SIMULATED_VISUAL":
        raise HTTPException(409, "degradation controls apply only to SIMULATED_VISUAL")
    src.degrade = body.model_dump()
    return {"degrade": src.degrade}


def _video_dir() -> Path:
    p = DATA_DIR / "videos"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _video_path(name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)
    return _video_dir() / safe


@app.get("/api/camera/videos")
def camera_videos():
    return {"videos": [{"name": p.name, "bytes": p.stat().st_size} for p in sorted(_video_dir().glob("*")) if p.is_file()],
            "directory": "data/videos (local only)"}


@app.post("/api/camera/videos")
async def camera_upload(request: Request, filename: str = Query(...)):
    if not re.search(r"\.(mp4|avi|mov|mkv|webm)$", filename, re.I):
        raise HTTPException(422, "unsupported video extension")
    data = await request.body()
    if len(data) > 500 * 1024 * 1024:
        raise HTTPException(413, "file too large")
    p = _video_path(filename)
    p.write_bytes(data)
    return {"name": p.name, "bytes": len(data), "stored": "local data/videos directory"}


# --------------------------------------------------------- flight recorder / runs
@app.get("/api/runs")
def runs(request: Request):
    return {"runs": dbh(request).list_runs(), "current_run_id": eng(request).run_id}


@app.get("/api/runs/{run_id}")
def run_detail(request: Request, run_id: str):
    r = dbh(request).get_run(run_id)
    if r is None:
        raise HTTPException(404, "run not found")
    return r


@app.get("/api/runs/{run_id}/timeline")
def run_timeline(request: Request, run_id: str, t0: float = 0, t1: float = 1e18, every: int = 1,
                 severity: str | None = None):
    db = dbh(request)
    if db.get_run(run_id) is None:
        raise HTTPException(404, "run not found")
    eng(request).db.flush(1.0)
    sev = severity.split(",") if severity else None
    return _jsonable({"run_id": run_id, "telemetry": db.telemetry(run_id, t0, t1, every), "events": db.events(run_id, t0, t1, sev),
                      "hypothesis_snapshots": db.hypothesis_snapshots(run_id, t0, t1),
                      "frames": [{"t": f["t"], "status": f["status"], "level_m": f["level_m"], "mode": f["mode"]} for f in db.frames(run_id)],
                      "diagnostics": db.diagnostics(run_id), "responses": db.responses(run_id),
                      "note": "Recorded history only; gaps are shown as gaps, never interpolated."})


@app.get("/api/runs/{run_id}/frame")
def run_frame(request: Request, run_id: str, t: float):
    frames = dbh(request).frames(run_id)
    prior = [f for f in frames if f["t"] <= t + 1e-6]
    if not prior or t - prior[-1]["t"] > 2.5:
        return JSONResponse({"status": "NO_RECORDED_FRAME", "t": t}, status_code=404)
    p = Path(prior[-1]["path"])
    if not p.is_file():
        return JSONResponse({"status": "FRAME_FILE_MISSING"}, status_code=404)
    return FileResponse(p, media_type="image/jpeg", headers={"X-Frame-Time": str(prior[-1]["t"])})


@app.get("/api/runs/{run_id}/observations")
def run_observations(request: Request, run_id: str, t0: float = 0, t1: float = 1e18, source_type: str | None = None, limit: int = 500):
    return {"observations": dbh(request).observations(run_id, t0, t1, source_type, limit)}


# --------------------------------------------------------------- evaluation (GT)
@app.get("/api/evaluation/ground-truth/{run_id}")
def ground_truth(request: Request, run_id: str):
    db = dbh(request)
    return {"provenance": "SIMULATION_GROUND_TRUTH", "run_id": run_id, "scenarios": db.ground_truth(run_id),
            "truth_telemetry": db.truth(run_id)[::2],
            "warning": "Evaluation only. This data is never available to the inference engine."}


# ---------------------------------------------------------------- export/reports
@app.get("/api/reports/{run_id}")
def report(request: Request, run_id: str, include_ground_truth: bool = False):
    db = dbh(request)
    db.flush(1.0)
    try:
        rep = build_report(db, run_id, include_ground_truth)
    except KeyError:
        raise HTTPException(404, "run not found")
    db.save_report(new_id("REP"), run_id, rep)
    return _jsonable(rep)


@app.get("/api/export/run/{run_id}.json")
def export_run_json(request: Request, run_id: str, include_ground_truth: bool = False):
    db = dbh(request)
    db.flush(1.0)
    try:
        rep = build_report(db, run_id, include_ground_truth)
    except KeyError:
        raise HTTPException(404, "run not found")
    rep["telemetry"] = db.telemetry(run_id)
    rep["observations"] = db.observations(run_id, limit=100000)
    body = json.dumps(_jsonable(rep), indent=1)
    return Response(body, media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{run_id}.json"'})


@app.get("/api/export/run/{run_id}.csv")
def export_run_csv(request: Request, run_id: str):
    db = dbh(request)
    db.flush(1.0)
    return PlainTextResponse(telemetry_csv(db.telemetry(run_id)), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{run_id}-telemetry.csv"'})


@app.get("/api/export/experiment/{eid}.json")
def export_exp_json(request: Request, eid: str):
    exp = dbh(request).get_experiment(eid)
    if exp is None:
        raise HTTPException(404, "experiment not found")
    return Response(json.dumps(exp, indent=1), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{eid}.json"'})


@app.get("/api/export/experiment/{eid}.csv")
def export_exp_csv(request: Request, eid: str):
    exp = dbh(request).get_experiment(eid)
    if exp is None:
        raise HTTPException(404, "experiment not found")
    return PlainTextResponse(experiment_csv(exp), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{eid}.csv"'})


@app.get("/api/meta")
def meta():
    return {"response_policies": list(RESPONSE_POLICIES), "camera_modes": list(CAMERA_MODES),
            "diagnostics": {k: {"label": v["label"], "description": v["description"], "cost": v["cost"],
                                "disturbance": v["disturbance"], "duration_s": v["duration_s"]} for k, v in dg.REGISTRY.items()},
            "versions": VERSIONS}


# --------------------------------------------------------------------- websocket
@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    e: LiveEngine = ws.app.state.engine
    q = e.subscribe()
    try:
        await ws.send_json(e.live_payload())
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=2.0)
            except asyncio.TimeoutError:
                msg = e.live_payload()  # heartbeat with current state while paused/stopped
            await ws.send_json(msg)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        e.unsubscribe(q)
