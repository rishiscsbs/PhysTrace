"""AT-002 mass balance & physical constraints; AT-003 seeded reproducibility."""
import math

import numpy as np

from phystrace.config import SimConfig
from phystrace.evaluation.metrics import compute_metrics
from phystrace.sim.plant import Plant, pump_flow, valve_flow
from phystrace.sim.scenarios import ScenarioSpec
from phystrace.sim.world import World


def test_at002_rk4_step_satisfies_mass_balance():
    cfg = SimConfig()
    p = Plant(cfg.plant)
    p.state.h, p.state.n, p.state.pos = 1.2, 0.7, 0.45
    h0 = p.state.h
    st = p.step(0.1, 0.7, 0.45, supply_head_offset=0.0, demand_factor=1.0, process_flow=0.0)
    # dh/dt = (Q_in - Q_out) / A with the stage-weighted RK4 flows
    assert math.isclose(st.h - h0, 0.1 * (st.q_in - st.q_out) / cfg.plant.area_m2, rel_tol=0, abs_tol=1e-12)


def test_at002_pump_and_valve_curves_are_physical():
    p = SimConfig().plant
    assert pump_flow(p, 0.0, 1.0) == 0.0
    assert pump_flow(p, 1.0, 1.0) > pump_flow(p, 0.8, 1.0) > 0
    assert pump_flow(p, 0.8, 2.5) < pump_flow(p, 0.8, 0.5)  # higher head -> less flow
    assert valve_flow(p, 0.5, 0.0) == 0.0
    assert math.isclose(valve_flow(p, 0.5, 1.44), p.valve_cv * 0.5 * 1.2)


def test_at002_volume_accounting_closes_with_overflow_and_dry_tank():
    # Force overflow: hidden setpoint far above the rim.
    w = World(SimConfig(seed=3), [ScenarioSpec(type="CTRL_SETPOINT_OVERRIDE", start_s=5, params={"setpoint_m": 3.0}),
                                   ScenarioSpec(type="VALVE_STUCK", start_s=5, params={"position": 0.05})])
    truth = [w.step()[1] for _ in range(4000)]
    m = compute_metrics(truth, w.cfg, 5.0, None)
    assert m["overflow_volume_L"] > 100
    assert max(x["h"] for x in truth) <= w.cfg.plant.tank_height_m + 1e-12
    assert abs(m["mass_balance_closure_L"]) < 1e-6  # water_loss independently equals overflow (no double count)
    # Force a dry tank: pump trip with outlet open.
    w2 = World(SimConfig(seed=3), [ScenarioSpec(type="PUMP_TRIP", start_s=1)])
    truth2 = [w2.step()[1] for _ in range(6000)]
    assert min(x["h"] for x in truth2) >= 0.0
    m2 = compute_metrics(truth2, w2.cfg, 1.0, None)
    assert abs(m2["mass_balance_closure_L"]) < 1e-6
    assert m2["unmet_demand_L"] > 0


def test_at002_nominal_steady_state_near_setpoint():
    w = World(SimConfig(seed=1))
    hs = np.array([w.step()[1]["h"] for _ in range(3000)])
    assert abs(hs.mean() - w.cfg.controller.setpoint_m) < 0.03
    assert hs.min() > w.cfg.safety.band_low_m and hs.max() < w.cfg.safety.band_high_m


def _trajectory(seed, scen=None, n=1500):
    w = World(SimConfig(seed=seed), scen or [])
    out = []
    for _ in range(n):
        o, t = w.step()
        out.append((t["h"], o["recv"]["LT-101"], o["recv"]["FT-101"], o["u_cmd"], o["delta_u"]))
    return np.array(out, dtype=float)


def test_at003_identical_seed_reproduces_trajectory():
    s = [ScenarioSpec(type="SENSOR_REPLAY", start_s=60, seed=4)]
    a, b = _trajectory(42, s), _trajectory(42, [x.model_copy() for x in s])
    assert np.array_equal(np.nan_to_num(a), np.nan_to_num(b))


def test_at003_different_seed_changes_noise_but_not_physics_scale():
    a, b = _trajectory(1), _trajectory(2)
    assert not np.array_equal(a[:, 1], b[:, 1])
    assert abs(a[:, 0].mean() - b[:, 0].mean()) < 0.02
