import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phystrace.config import SimConfig  # noqa: E402
from phystrace.evaluation.harness import make_episode  # noqa: E402
from phystrace.sim.scenarios import ScenarioSpec  # noqa: E402


@pytest.fixture
def api(tmp_path, monkeypatch):
    """FastAPI TestClient bound to an isolated data directory."""
    from fastapi.testclient import TestClient

    import phystrace.api.app as appmod

    monkeypatch.setattr(appmod, "DATA_DIR", tmp_path)
    monkeypatch.setenv("PHYSTRACE_DB", str(tmp_path / "test.sqlite3"))
    with TestClient(appmod.app) as client:
        client.data_dir = tmp_path
        yield client


def wait_until(fn, timeout=30.0, interval=0.1):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(interval)
    return fn()


def run_episode(typ=None, seed=7, until=200.0, onset=90.0, camera=True, params=None, target=None):
    scen = [] if typ is None else [ScenarioSpec(type=typ, start_s=onset, seed=2, params=params or {}, target=target)]
    ep = make_episode(SimConfig(seed=seed), scen, camera=camera)
    ep.run_until(until)
    return ep
