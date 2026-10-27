"""Physical tank / pump / valve model (simulation ground truth).

Mass balance:      dh/dt = (Q_in - Q_out) / A
Pump curve:        H_pump(Q, n) = H0 n^2 - K Q^2  (affinity laws, quadratic loss)
Operating point:   H0 n^2 - K Q^2 = H_static + h + d_supply
                   => Q_in = sqrt(max(0, H0 n^2 - H_static - h - d_supply) / K)
Valve discharge:   Q_out = Cv * pos * sqrt(h) * demand_factor
Pump power:        P = rho g Q_in H_pump / eta + P_idle   (when running)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import PlantParams

RHO = 1000.0
G = 9.81


def pump_flow(p: PlantParams, n: float, h: float, supply_head_offset: float = 0.0, flow_factor: float = 1.0) -> float:
    """Inlet flow Q_in [m^3/s] for pump speed n (0..1) at tank level h."""
    n = min(max(n, 0.0), 1.2)
    head_avail = p.pump_shutoff_head_m * n * n - p.static_lift_m - max(h, 0.0) - supply_head_offset
    if head_avail <= 0.0:
        return 0.0
    return flow_factor * math.sqrt(head_avail / p.system_resistance_s2_m5)


def valve_flow(p: PlantParams, pos: float, h: float, demand_factor: float = 1.0, cv_factor: float = 1.0) -> float:
    """Outlet flow Q_out [m^3/s] = Cv * pos * sqrt(h)."""
    pos = min(max(pos, 0.0), 1.0)
    if h <= 0.0:
        return 0.0
    return p.valve_cv * cv_factor * pos * math.sqrt(h) * demand_factor


def pump_power_w(p: PlantParams, n: float, q_in: float, h: float) -> float:
    if n < 0.02:
        return 0.0
    head = p.static_lift_m + max(h, 0.0) + p.system_resistance_s2_m5 * 0.5 * q_in * q_in
    hydraulic = RHO * G * q_in * head
    return hydraulic / p.pump_efficiency + p.pump_idle_power_kw * 1000.0


@dataclass
class PlantState:
    h: float  # level [m]
    n: float  # actual pump speed (0..1)
    pos: float  # actual valve position (0..1)
    overflow_m3: float = 0.0
    q_in: float = 0.0
    q_out: float = 0.0
    power_w: float = 0.0
    dry_clipped_m3: float = 0.0


class Plant:
    """Ground-truth plant. Integrated with classical RK4 on h; actuator states use exact
    first-order / slew-limited updates."""

    def __init__(self, params: PlantParams):
        self.p = params
        self.state = PlantState(h=params.initial_level_m, n=0.0, pos=0.5)

    def step(
        self,
        dt: float,
        n_target: float,
        pos_target: float,
        *,
        supply_head_offset: float,
        demand_factor: float,
        process_flow: float,
        pump_flow_factor: float = 1.0,
        cv_factor: float = 1.0,
        valve_slew: float | None = None,
        valve_stuck: float | None = None,
        pump_tripped: bool = False,
    ) -> PlantState:
        p, s = self.p, self.state
        # Actuators
        if pump_tripped:
            n_target = 0.0
        alpha = 1.0 - math.exp(-dt / p.pump_time_constant_s)
        s.n = s.n + alpha * (n_target - s.n)
        if valve_stuck is not None:
            s.pos = valve_stuck
        else:
            slew = (valve_slew if valve_slew is not None else p.valve_slew_rate_per_s) * dt
            s.pos = s.pos + max(-slew, min(slew, pos_target - s.pos))
        s.pos = min(max(s.pos, 0.0), 1.0)

        n, pos = s.n, s.pos

        def flows(h: float) -> tuple[float, float]:
            qi = max(pump_flow(p, n, h, supply_head_offset, pump_flow_factor) + process_flow, 0.0)
            qo = valve_flow(p, pos, h, demand_factor, cv_factor)
            return qi, qo

        A = p.area_m2
        h0 = s.h
        i1, o1 = flows(h0)
        i2, o2 = flows(h0 + 0.5 * dt * (i1 - o1) / A)
        i3, o3 = flows(h0 + 0.5 * dt * (i2 - o2) / A)
        i4, o4 = flows(h0 + dt * (i3 - o3) / A)
        # RK4 stage-weighted flows: h1 = h0 + dt (q_in - q_out) / A holds exactly,
        # so volume accounting closes to machine precision.
        q_in = (i1 + 2 * i2 + 2 * i3 + i4) / 6.0
        q_out = (o1 + 2 * o2 + 2 * o3 + o4) / 6.0
        h1 = h0 + dt * (q_in - q_out) / A
        if h1 > p.tank_height_m:
            s.overflow_m3 += (h1 - p.tank_height_m) * p.area_m2
            h1 = p.tank_height_m
        if h1 < 0.0:
            # Non-negative volume: the outlet cannot deliver water that is not there.
            s.dry_clipped_m3 += -h1 * p.area_m2
            h1 = 0.0
        s.h = h1
        s.q_in = q_in
        s.q_out = q_out
        s.power_w = pump_power_w(p, n, q_in, h0)
        return s
