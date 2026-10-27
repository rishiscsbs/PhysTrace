"""AT-017 bounded reproducible watermark; correlation evidence and degradation warnings."""
import numpy as np

from conftest import run_episode
from phystrace.config import SimConfig
from phystrace.sim.rng import WatermarkSequence
from phystrace.sim.world import World


def test_at017_watermark_bounded_and_reproducible():
    cfg = SimConfig()
    w1, w2 = World(cfg), World(cfg)
    d1 = np.array([w1.step()[0]["delta_u"] for _ in range(2000)])
    d2 = np.array([w2.step()[0]["delta_u"] for _ in range(2000)])
    assert np.array_equal(d1, d2)
    assert np.max(np.abs(d1)) <= cfg.watermark.epsilon + 1e-12
    assert set(np.unique(np.sign(d1))) <= {-1.0, 1.0}
    a, b = WatermarkSequence(1, 30), WatermarkSequence(2, 30)
    assert [a.symbol(k) for k in range(0, 3000, 30)] != [b.symbol(k) for k in range(0, 3000, 30)]


def test_at017_watermark_correlated_nominally_and_lost_under_replay():
    nominal = run_episode(None, seed=5, until=200, camera=False).defenders["FULL"].wm_result["signals"]
    assert nominal["FT-101"]["status"] == "CORRELATED" and nominal["FT-101"]["detectability"] == "DETECTABLE"
    assert nominal["LT-101"]["status"] == "CORRELATED"
    replay = run_episode("SENSOR_REPLAY", seed=5, until=190, camera=False).defenders["FULL"].wm_result["signals"]
    assert replay["LT-101"]["status"] in ("ABSENT", "WEAK")
    assert replay["FT-101"]["status"] == "CORRELATED"  # the unaffected signal still carries it


def test_at017_degradation_warning_under_high_noise():
    cfg = SimConfig(seed=5)
    cfg.sensors.level_noise_m *= 12
    cfg.sensors.flow_noise_lps *= 12
    from phystrace.evaluation.harness import make_episode
    ep = make_episode(cfg, [], camera=False)
    ep.run_until(200)
    wm = ep.defenders["FULL"].wm_result
    assert any(w.startswith("HIGH_NOISE_LOW_SNR") for w in wm["warnings"])
    assert wm["signals"]["LT-101"]["detectability"] in ("DEGRADED", "UNDETECTABLE")


def test_at017_missing_correlation_is_evidence_not_proof():
    ep = run_episode("PUMP_TRIP", seed=5, until=170, camera=False)
    d = ep.defenders["FULL"]
    wm_items = [e for e in d.evidence if e.feature.startswith("wm_absent")]
    assert wm_items and all(e.strength <= 1.0 for e in wm_items)
    # A pump trip also removes the watermark: the verdict must rest on more than watermark loss.
    assert d.result["verdict"]["leading"] == "ACTUATOR_FAULT"
