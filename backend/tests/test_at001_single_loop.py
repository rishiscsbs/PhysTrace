"""AT-001 single simulation loop under repeated start/stop/pause/reset; bounded buffers."""
import asyncio

from conftest import wait_until


def test_at001_repeated_start_spawns_one_loop(api):
    for _ in range(5):
        r = api.post("/api/sim/start").json()
    assert r["loop_tasks_started"] == 1 and r["loop_running"]
    api.put("/api/sim/speed", json={"speed": 20})
    assert wait_until(lambda: api.get("/api/state").json()["t"] > 5, timeout=20)
    eng = api.app.state.engine
    tasks = [t for t in asyncio.all_tasks(eng.task.get_loop()) if t.get_name() == "phystrace-sim-loop"]
    assert len(tasks) == 1


def test_at001_stop_pause_reset_are_idempotent(api):
    api.post("/api/sim/start")
    api.post("/api/sim/pause")
    t_paused = api.get("/api/state").json()["t"]
    import time
    time.sleep(0.4)
    assert api.get("/api/state").json()["t"] == t_paused  # paused: no stepping
    r = api.post("/api/sim/start").json()  # resume, still one loop
    assert r["status"] == "RUNNING" and r["loop_tasks_started"] == 1
    api.post("/api/sim/stop")
    api.post("/api/sim/stop")
    s = api.get("/api/state").json()
    assert s["status"] == "STOPPED" and not s["loop_running"]
    r1 = api.post("/api/sim/reset", json={"seed": 99}).json()
    assert r1["t"] == 0.0 and r1["seed"] == 99 and not r1["loop_running"]
    api.post("/api/sim/start")
    r2 = api.post("/api/sim/start").json()
    assert r2["loop_tasks_started"] == 2 and r2["loop_running"]  # exactly one new loop after the reset
    api.post("/api/sim/stop")


def test_at001_buffers_are_bounded(api):
    eng = api.app.state.engine
    assert eng.points.maxlen is not None
    assert eng.defender.observations.maxlen is not None
    assert eng.defender.hist.capacity == int(300 / eng.cfg.dt_s)
    assert eng.world.exogenous_log.maxlen is not None
    for ch in eng.world.channels.values():
        assert ch.sent_log.maxlen is not None
    q = eng.subscribe()
    for i in range(100):
        eng._broadcast({"i": i})
    assert q.qsize() <= q.maxsize  # slow subscriber cannot grow memory
    eng.unsubscribe(q)
