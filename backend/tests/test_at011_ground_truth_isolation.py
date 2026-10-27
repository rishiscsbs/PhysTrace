"""AT-011 hidden scenario labels never reach the inference engine."""
import ast
import copy
from pathlib import Path

from phystrace.config import SimConfig
from phystrace.inference.pipeline import Defender
from phystrace.schemas import Observation
from phystrace.sim.scenarios import ScenarioSpec
from phystrace.sim.world import World

INFERENCE_DIR = Path(__file__).resolve().parents[1] / "phystrace" / "inference"
FORBIDDEN = ("scenarios", "world", "evaluation", "persistence", "engine", "api")


def test_at011_inference_package_never_imports_ground_truth_modules():
    for py in INFERENCE_DIR.glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split(".")
                assert not any(p in FORBIDDEN for p in parts), f"{py.name} imports {node.module}"
                assert node.module != "sim" or all(a.name in ("plant",) for a in node.names)
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert not any(p in FORBIDDEN for p in a.name.split(".")), f"{py.name} imports {a.name}"


def test_at011_observable_frames_carry_no_truth_fields():
    w = World(SimConfig(), [ScenarioSpec(type="SENSOR_BIAS", start_s=0.0)])
    obs, truth = w.step()
    flat = repr(obs)
    for key in ("active_scenarios", "scenario", "SENSOR_BIAS", "\"h\"", "overflow"):
        assert key not in flat
    assert "active_scenarios" in truth
    assert set(Observation.model_fields) == {"observation_id", "timestamp", "source_type", "source_id", "component_id",
                                             "measurement_type", "value", "unit", "quality", "uncertainty", "freshness",
                                             "provenance", "metadata"}


def test_at011_inference_output_invariant_to_ground_truth_labels():
    """Same observations, different hidden labels -> identical hypotheses."""
    w = World(SimConfig(seed=3), [ScenarioSpec(type="SENSOR_BIAS", start_s=60.0, params={"bias": -0.4})])
    frames = [w.step()[0] for _ in range(1500)]
    d1 = Defender(SimConfig(seed=3), emit=False)
    d2 = Defender(SimConfig(seed=3), emit=False)
    relabelled = copy.deepcopy(frames)
    for f in relabelled:
        f["ground_truth_label"] = "COMM_DELAY"  # an adversarially injected label must be ignored
    for a, b in zip(frames, relabelled):
        d1.step(a)
        d2.step(b)
    assert d1.result["verdict"] == d2.result["verdict"]
    assert [h["raw_score"] for h in d1.result["hypotheses"]] == [h["raw_score"] for h in d2.result["hypotheses"]]


def test_at011_api_inject_does_not_emit_inference_visible_event(api):
    r = api.post("/api/scenarios", json={"type": "PUMP_TRIP", "start_in_s": 30}).json()
    assert r["scheduled"]
    run_id = api.get("/api/runs").json()["current_run_id"]
    api.app.state.db.flush()
    events = api.get(f"/api/runs/{run_id}/timeline").json()["events"]
    assert not any("PUMP_TRIP" in e["message"] or e["category"].startswith("SCENARIO") for e in events)
    assert "PUMP_TRIP" not in api.get("/api/state").text
    gt = api.get(f"/api/evaluation/ground-truth/{run_id}").json()
    assert gt["provenance"] == "SIMULATION_GROUND_TRUTH" and gt["scenarios"][0]["scenario"]["type"] == "PUMP_TRIP"
