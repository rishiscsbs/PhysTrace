"""Typed configuration for the simulated water-treatment plant.

All physical quantities use SI units internally (m, m^2, m^3/s, s, W).
Flow values presented to operators are converted to L/s at the sensor layer.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

MODEL_VERSION = "phystrace-plant-1.0.0"
TWIN_VERSION = "phystrace-twin-1.0.0"
INFERENCE_VERSION = "phystrace-evidence-1.0.0"


class PlantParams(BaseModel):
    area_m2: float = Field(1.5, gt=0, description="Tank cross-sectional area A")
    tank_height_m: float = Field(3.0, gt=0, description="Overflow rim height h_max")
    pump_shutoff_head_m: float = Field(20.0, gt=0, description="Pump head H0 at rated speed n=1")
    static_lift_m: float = Field(2.0, ge=0, description="Static lift from reservoir to tank floor")
    system_resistance_s2_m5: float = Field(20000.0, gt=0, description="Combined pump + pipe quadratic loss K")
    pump_efficiency: float = Field(0.65, gt=0, le=1)
    pump_idle_power_kw: float = Field(0.15, ge=0)
    pump_time_constant_s: float = Field(2.0, gt=0)
    valve_cv: float = Field(0.023, gt=0, description="Outlet discharge coefficient Cv [m^2.5/s]")
    valve_slew_rate_per_s: float = Field(0.05, gt=0)
    initial_level_m: float = Field(1.5, ge=0)
    process_noise_lps: float = Field(0.15, ge=0, description="AR(1) unmodelled inflow turbulence sigma, L/s")
    process_noise_corr: float = Field(0.98, ge=0, lt=1)


class SafetyLimits(BaseModel):
    high_high_m: float = 2.7
    band_high_m: float = 2.2
    band_low_m: float = 0.8
    low_low_m: float = 0.3


class ControllerParams(BaseModel):
    setpoint_m: float = 1.5
    kp: float = 0.8
    ki: float = 0.02
    u_ff: float = 0.611
    valve_base: float = 0.5
    valve_amplitude: float = 0.08
    valve_period_s: float = 900.0


class SensorParams(BaseModel):
    level_noise_m: float = 0.003
    level_quant_m: float = 0.001
    redundant_level_noise_m: float = 0.004
    flow_noise_lps: float = 0.15
    flow_quant_lps: float = 0.05
    valve_pos_noise_pct: float = 0.3
    pump_speed_noise: float = 0.004


class NetworkParams(BaseModel):
    base_latency_s: float = 0.05
    jitter_s: float = 0.02
    drop_rate: float = 0.0


class WatermarkParams(BaseModel):
    enabled: bool = True
    epsilon: float = Field(0.02, ge=0, le=0.1, description="Bound on |delta u_k|")
    hold_steps: int = Field(30, ge=1, description="Samples each pseudorandom symbol is held")
    window_s: float = 90.0
    max_lag_s: float = 4.0


class SimConfig(BaseModel):
    dt_s: float = 0.1
    seed: int = 1234
    watermark_seed: int = 9001
    plant: PlantParams = PlantParams()
    safety: SafetyLimits = SafetyLimits()
    controller: ControllerParams = ControllerParams()
    sensors: SensorParams = SensorParams()
    network: NetworkParams = NetworkParams()
    watermark: WatermarkParams = WatermarkParams()
    camera_rate_hz: float = 1.0
    simulated_camera: bool = Field(
        False, description="Render the explicitly labelled SIMULATED_VISUAL source inside the world"
    )
