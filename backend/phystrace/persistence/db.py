"""Local-first SQLite persistence (FR-027 – FR-030).

Ground truth lives in its own tables (`evaluation_ledger`, `truth_telemetry`), read only
through the evaluation API. Inference code never queries them.
"""
from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, created_at REAL, seed INTEGER, config_json TEXT, status TEXT, label TEXT,
  sim_time_s REAL DEFAULT 0, model_versions_json TEXT);
CREATE TABLE IF NOT EXISTS observations (
  observation_id TEXT PRIMARY KEY, run_id TEXT, timestamp REAL, source_type TEXT, source_id TEXT, component_id TEXT,
  measurement_type TEXT, value_json TEXT, unit TEXT, quality TEXT, uncertainty REAL, freshness REAL,
  provenance_json TEXT, metadata_json TEXT);
CREATE INDEX IF NOT EXISTS ix_obs_run_t ON observations(run_id, timestamp);
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY, run_id TEXT, timestamp REAL, severity TEXT, category TEXT, component_id TEXT,
  message TEXT, provenance TEXT, data_json TEXT);
CREATE INDEX IF NOT EXISTS ix_evt_run_t ON events(run_id, timestamp);
CREATE TABLE IF NOT EXISTS hypotheses (
  snapshot_id TEXT, run_id TEXT, timestamp REAL, hypothesis_id TEXT, hypothesis_type TEXT, component_id TEXT,
  score REAL, raw_score REAL, score_type TEXT, rank INTEGER, status TEXT, supporting_json TEXT,
  contradicting_json TEXT, missing_json TEXT, assumptions_json TEXT, verdict_json TEXT);
CREATE INDEX IF NOT EXISTS ix_hyp_run_t ON hypotheses(run_id, timestamp);
CREATE TABLE IF NOT EXISTS telemetry (run_id TEXT, t REAL, data_json TEXT);
CREATE INDEX IF NOT EXISTS ix_tel_run_t ON telemetry(run_id, t);
CREATE TABLE IF NOT EXISTS diagnostics (
  diagnostic_id TEXT PRIMARY KEY, run_id TEXT, test_id TEXT, started_s REAL, completed_s REAL, status TEXT, outcome TEXT,
  result_json TEXT);
CREATE TABLE IF NOT EXISTS experiments (
  experiment_id TEXT PRIMARY KEY, kind TEXT, created_at REAL, status TEXT, config_json TEXT, result_json TEXT,
  model_versions_json TEXT, error TEXT, progress REAL DEFAULT 0, label TEXT);
CREATE TABLE IF NOT EXISTS camera_calibrations (
  calibration_id TEXT PRIMARY KEY, camera_id TEXT, component_id TEXT, mode TEXT, roi_json TEXT, points_json TEXT,
  indicator_roi_json TEXT, created_at REAL, active INTEGER, notes TEXT);
CREATE TABLE IF NOT EXISTS camera_frames (run_id TEXT, t REAL, path TEXT, status TEXT, level_m REAL, mode TEXT);
CREATE INDEX IF NOT EXISTS ix_frames_run_t ON camera_frames(run_id, t);
CREATE TABLE IF NOT EXISTS responses (run_id TEXT, t REAL, policy TEXT, data_json TEXT);
CREATE TABLE IF NOT EXISTS reports (report_id TEXT PRIMARY KEY, run_id TEXT, created_at REAL, report_json TEXT);
-- Evaluation-only ground truth (never read by inference)
CREATE TABLE IF NOT EXISTS evaluation_ledger (run_id TEXT, recorded_at REAL, sim_time_s REAL, scenario_json TEXT);
CREATE TABLE IF NOT EXISTS truth_telemetry (run_id TEXT, t REAL, data_json TEXT);
CREATE INDEX IF NOT EXISTS ix_truth_run_t ON truth_telemetry(run_id, t);
"""


def _j(v: Any) -> str:
    return json.dumps(v, default=str, separators=(",", ":"))


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.executescript(SCHEMA)
            self.conn.commit()
        self._q: queue.Queue = queue.Queue(maxsize=50000)
        self.dropped = 0
        self._stop = threading.Event()
        self._writer = threading.Thread(target=self._write_loop, daemon=True, name="db-writer")
        self._writer.start()

    # ------------------------------------------------------------------ writer
    def enqueue(self, sql: str, params: tuple) -> None:
        try:
            self._q.put_nowait((sql, params))
        except queue.Full:
            self.dropped += 1  # bounded: never grows without limit

    def _write_loop(self) -> None:
        while not self._stop.is_set() or not self._q.empty():
            batch = []
            try:
                batch.append(self._q.get(timeout=0.25))
            except queue.Empty:
                continue
            while len(batch) < 2000:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            with self._lock:
                try:
                    for sql, params in batch:
                        self.conn.execute(sql, params)
                    self.conn.commit()
                except sqlite3.Error:
                    self.conn.rollback()

    def flush(self, timeout: float = 5.0) -> None:
        t0 = time.time()
        while not self._q.empty() and time.time() - t0 < timeout:
            time.sleep(0.02)
        time.sleep(0.05)
        with self._lock:
            self.conn.commit()

    def close(self) -> None:
        self.flush()
        self._stop.set()
        self._writer.join(timeout=3)
        with self._lock:
            self.conn.close()

    def query(self, sql: str, params: Iterable = ()) -> list[dict]:
        with self._lock:
            cur = self.conn.execute(sql, tuple(params))
            return [dict(r) for r in cur.fetchall()]

    def execute(self, sql: str, params: Iterable = ()) -> None:
        with self._lock:
            self.conn.execute(sql, tuple(params))
            self.conn.commit()

    # -------------------------------------------------------------------- runs
    def create_run(self, run_id: str, seed: int, config: dict, label: str, versions: dict) -> None:
        self.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                     (run_id, time.time(), seed, _j(config), "CREATED", label, 0.0, _j(versions)))

    def update_run(self, run_id: str, status: str | None = None, sim_time_s: float | None = None) -> None:
        if status is not None:
            self.enqueue("UPDATE runs SET status=? WHERE run_id=?", (status, run_id))
        if sim_time_s is not None:
            self.enqueue("UPDATE runs SET sim_time_s=? WHERE run_id=?", (sim_time_s, run_id))

    def list_runs(self, limit: int = 100) -> list[dict]:
        rows = self.query("SELECT run_id, created_at, seed, status, label, sim_time_s FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        for r in rows:
            r["event_count"] = self.query("SELECT COUNT(*) AS n FROM events WHERE run_id=?", (r["run_id"],))[0]["n"]
        return rows

    def get_run(self, run_id: str) -> dict | None:
        rows = self.query("SELECT * FROM runs WHERE run_id=?", (run_id,))
        if not rows:
            return None
        r = rows[0]
        r["config"] = json.loads(r.pop("config_json") or "{}")
        r["model_versions"] = json.loads(r.pop("model_versions_json") or "{}")
        return r

    # ------------------------------------------------------------- records
    def add_observation(self, run_id: str, o) -> None:
        self.enqueue("INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (o.observation_id, run_id, o.timestamp, o.source_type, o.source_id, o.component_id, o.measurement_type,
                      _j(o.value), o.unit, o.quality, o.uncertainty, o.freshness, _j(o.provenance), _j(o.metadata)))

    def add_event(self, e: dict) -> None:
        self.enqueue("INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?,?)",
                     (e["event_id"], e["run_id"], e["timestamp"], e["severity"], e["category"], e.get("component_id"),
                      e["message"], e.get("provenance", "INFERENCE"), _j(e.get("data", {}))))

    def add_hypothesis_snapshot(self, run_id: str, t: float, result: dict, snapshot_id: str) -> None:
        v = _j(result.get("verdict"))
        for h in result.get("hypotheses", []):
            self.enqueue("INSERT INTO hypotheses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (snapshot_id, run_id, t, h["hypothesis_id"], h["hypothesis_type"], h.get("component_id"), h["score"],
                          h["raw_score"], h["score_type"], h["rank"], h["status"], _j(h["supporting_evidence_refs"]),
                          _j(h["contradicting_evidence_refs"]), _j(h["missing_evidence"]), _j(h["assumptions"]), v))

    def add_telemetry(self, run_id: str, t: float, row: dict) -> None:
        self.enqueue("INSERT INTO telemetry VALUES (?,?,?)", (run_id, t, _j(row)))

    def add_truth(self, run_id: str, t: float, row: dict) -> None:
        self.enqueue("INSERT INTO truth_telemetry VALUES (?,?,?)", (run_id, t, _j(row)))

    def add_diagnostic(self, run_id: str, did: str, res: dict) -> None:
        self.enqueue("INSERT OR REPLACE INTO diagnostics VALUES (?,?,?,?,?,?,?,?)",
                     (did, run_id, res["test_id"], res.get("started_s"), res.get("completed_s"), res.get("status"),
                      res.get("outcome"), _j(res)))

    def add_frame(self, run_id: str, t: float, path: str, status: str, level: float | None, mode: str) -> None:
        self.enqueue("INSERT INTO camera_frames VALUES (?,?,?,?,?,?)", (run_id, t, path, status, level, mode))

    def add_response(self, run_id: str, t: float, policy: str, data: dict) -> None:
        self.enqueue("INSERT INTO responses VALUES (?,?,?,?)", (run_id, t, policy, _j(data)))

    def record_ground_truth(self, run_id: str, sim_time_s: float, scenario: dict) -> None:
        self.execute("INSERT INTO evaluation_ledger VALUES (?,?,?,?)", (run_id, time.time(), sim_time_s, _j(scenario)))

    def ground_truth(self, run_id: str) -> list[dict]:
        rows = self.query("SELECT * FROM evaluation_ledger WHERE run_id=? ORDER BY sim_time_s", (run_id,))
        return [{"recorded_at": r["recorded_at"], "sim_time_s": r["sim_time_s"], "scenario": json.loads(r["scenario_json"])}
                for r in rows]

    # ----------------------------------------------------------- experiments
    def save_experiment(self, exp: dict) -> None:
        self.execute("INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (exp["experiment_id"], exp["kind"], exp["created_at"], exp["status"], _j(exp.get("config", {})),
                      _j(exp.get("result")) if exp.get("result") is not None else None, _j(exp.get("model_versions", {})),
                      exp.get("error"), exp.get("progress", 0.0), exp.get("label")))

    def update_experiment_progress(self, eid: str, progress: float) -> None:
        self.enqueue("UPDATE experiments SET progress=? WHERE experiment_id=?", (progress, eid))

    def get_experiment(self, eid: str) -> dict | None:
        rows = self.query("SELECT * FROM experiments WHERE experiment_id=?", (eid,))
        if not rows:
            return None
        r = rows[0]
        r["config"] = json.loads(r.pop("config_json") or "{}")
        res = r.pop("result_json")
        r["result"] = json.loads(res) if res else None
        r["model_versions"] = json.loads(r.pop("model_versions_json") or "{}")
        return r

    def list_experiments(self, kind: str | None = None, limit: int = 100) -> list[dict]:
        if kind:
            return self.query("SELECT experiment_id, kind, created_at, status, progress, label, error FROM experiments "
                              "WHERE kind=? ORDER BY created_at DESC LIMIT ?", (kind, limit))
        return self.query("SELECT experiment_id, kind, created_at, status, progress, label, error FROM experiments "
                          "ORDER BY created_at DESC LIMIT ?", (limit,))

    # ----------------------------------------------------------- calibration
    def save_calibration(self, cal: dict) -> None:
        with self._lock:
            self.conn.execute("UPDATE camera_calibrations SET active=0 WHERE mode=? AND camera_id=?", (cal["mode"], cal["camera_id"]))
            self.conn.execute("INSERT INTO camera_calibrations VALUES (?,?,?,?,?,?,?,?,?,?)",
                              (cal["calibration_id"], cal["camera_id"], cal["component_id"], cal["mode"], _j(cal["roi"]),
                               _j(cal["points"]), _j(cal.get("indicator_roi")), cal["created_at"], 1, cal.get("notes", "")))
            self.conn.commit()

    def active_calibrations(self) -> list[dict]:
        rows = self.query("SELECT * FROM camera_calibrations WHERE active=1")
        out = []
        for r in rows:
            out.append({"calibration_id": r["calibration_id"], "camera_id": r["camera_id"], "component_id": r["component_id"],
                        "mode": r["mode"], "roi": json.loads(r["roi_json"]), "points": json.loads(r["points_json"]),
                        "indicator_roi": json.loads(r["indicator_roi_json"]) if r["indicator_roi_json"] else None,
                        "created_at": r["created_at"], "notes": r["notes"]})
        return out

    # ------------------------------------------------------------ readers
    def events(self, run_id: str, t0: float = 0, t1: float = 1e18, severity: list[str] | None = None, limit: int = 5000) -> list[dict]:
        sql = "SELECT * FROM events WHERE run_id=? AND timestamp BETWEEN ? AND ?"
        params: list = [run_id, t0, t1]
        if severity:
            sql += " AND severity IN (%s)" % ",".join("?" * len(severity))
            params += severity
        sql += " ORDER BY timestamp LIMIT ?"
        params.append(limit)
        rows = self.query(sql, params)
        for r in rows:
            r["data"] = json.loads(r.pop("data_json") or "{}")
        return rows

    def telemetry(self, run_id: str, t0: float = 0, t1: float = 1e18, every: int = 1) -> list[dict]:
        rows = self.query("SELECT t, data_json FROM telemetry WHERE run_id=? AND t BETWEEN ? AND ? ORDER BY t", (run_id, t0, t1))
        return [json.loads(r["data_json"]) for i, r in enumerate(rows) if i % max(1, every) == 0]

    def truth(self, run_id: str, t0: float = 0, t1: float = 1e18) -> list[dict]:
        rows = self.query("SELECT data_json FROM truth_telemetry WHERE run_id=? AND t BETWEEN ? AND ? ORDER BY t", (run_id, t0, t1))
        return [json.loads(r["data_json"]) for r in rows]

    def hypothesis_snapshots(self, run_id: str, t0: float = 0, t1: float = 1e18) -> list[dict]:
        rows = self.query("SELECT * FROM hypotheses WHERE run_id=? AND timestamp BETWEEN ? AND ? ORDER BY timestamp, rank",
                          (run_id, t0, t1))
        snaps: dict[str, dict] = {}
        for r in rows:
            s = snaps.setdefault(r["snapshot_id"], {"snapshot_id": r["snapshot_id"], "timestamp": r["timestamp"],
                                                    "verdict": json.loads(r["verdict_json"] or "{}"), "hypotheses": []})
            s["hypotheses"].append({"hypothesis_id": r["hypothesis_id"], "hypothesis_type": r["hypothesis_type"],
                                    "component_id": r["component_id"], "score": r["score"], "raw_score": r["raw_score"],
                                    "score_type": r["score_type"], "rank": r["rank"], "status": r["status"],
                                    "supporting_evidence_refs": json.loads(r["supporting_json"]),
                                    "contradicting_evidence_refs": json.loads(r["contradicting_json"]),
                                    "missing_evidence": json.loads(r["missing_json"]), "assumptions": json.loads(r["assumptions_json"])})
        return sorted(snaps.values(), key=lambda s: s["timestamp"])

    def observations(self, run_id: str, t0: float = 0, t1: float = 1e18, source_type: str | None = None, limit: int = 2000) -> list[dict]:
        sql = "SELECT * FROM observations WHERE run_id=? AND timestamp BETWEEN ? AND ?"
        params: list = [run_id, t0, t1]
        if source_type:
            sql += " AND source_type=?"
            params.append(source_type)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self.query(sql, params)
        for r in rows:
            r["value"] = json.loads(r.pop("value_json"))
            r["provenance"] = json.loads(r.pop("provenance_json"))
            r["metadata"] = json.loads(r.pop("metadata_json"))
        return rows

    def diagnostics(self, run_id: str) -> list[dict]:
        rows = self.query("SELECT result_json FROM diagnostics WHERE run_id=? ORDER BY started_s", (run_id,))
        return [json.loads(r["result_json"]) for r in rows]

    def frames(self, run_id: str) -> list[dict]:
        return self.query("SELECT t, path, status, level_m, mode FROM camera_frames WHERE run_id=? ORDER BY t", (run_id,))

    def responses(self, run_id: str) -> list[dict]:
        rows = self.query("SELECT t, policy, data_json FROM responses WHERE run_id=? ORDER BY t", (run_id,))
        return [{"t": r["t"], "policy": r["policy"], "data": json.loads(r["data_json"])} for r in rows]

    def save_report(self, report_id: str, run_id: str, report: dict) -> None:
        self.execute("INSERT OR REPLACE INTO reports VALUES (?,?,?,?)", (report_id, run_id, time.time(), _j(report)))
