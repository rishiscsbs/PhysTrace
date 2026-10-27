"""Evidence feature extraction (defender side, observable data only).

Each EvidenceItem has a strength in [0, 1] computed with documented saturating
ramps, an `independence_group` naming the underlying data stream (used by the
hypothesis engine to avoid double counting), and references to the
Observation records it was derived from.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import SimConfig
from ..schemas import EvidenceItem
from ..sim.plant import pump_flow, valve_flow
from .twin import residual_sigma

SENSORS = ("LT-101", "FT-101", "FT-102", "ZT-101", "SC-101")


def ramp(x: float, lo: float, hi: float) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


@dataclass
class FeatureContext:
    cfg: SimConfig
    hist: Any
    t: float
    alarm_onset_s: float | None
    watermark: dict | None
    camera: dict | None  # latest camera observation dict (status, level_m, uncertainty_m, timestamp, observation_id)
    diagnostics: dict[str, dict] = field(default_factory=dict)  # test_id -> latest result
    obs_refs: dict[str, str] = field(default_factory=dict)  # source key -> latest observation_id
    response_active: bool = False
    # Onset-shape memory per sensor: {sid: {"frac_jump": .., "since": t, "last_anomalous": t}}. The residual shape
    # (abrupt step vs gradual ramp) is characterised around onset and remembered while the anomaly persists, so it
    # does not change merely because the onset scrolls out of the bounded history.
    shape_memory: dict[str, dict] = field(default_factory=dict)


class _Builder:
    def __init__(self, t: float):
        self.t = t
        self.items: list[EvidenceItem] = []

    def add(self, feature: str, component: str, strength: float, desc: str, group: str, source: str,
            refs: list[str], value: Any = None, provenance: str = "INFERENCE") -> None:
        if strength <= 0.0:
            return
        if isinstance(value, float) and math.isnan(value):
            value = None
        self.items.append(EvidenceItem(
            evidence_id=f"EV:{feature}@{component}", feature=feature, component_id=component,
            strength=round(float(strength), 4), value=None if value is None else (round(value, 5) if isinstance(value, float) else value),
            description=desc, independence_group=group, observation_refs=[r for r in refs if r],
            source_type=source, provenance=provenance, timestamp=self.t,
        ))


def _blocks(x: np.ndarray, n: int) -> np.ndarray:
    m = x.size // n
    if m == 0:
        return np.empty(0)
    return np.nanmean(x[x.size - m * n:].reshape(m, n), axis=1)


def extract_features(ctx: FeatureContext) -> tuple[list[EvidenceItem], dict[str, Any]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _extract(ctx)


def _extract(ctx: FeatureContext) -> tuple[list[EvidenceItem], dict[str, Any]]:
    cfg, h, dt = ctx.cfg, ctx.hist, ctx.cfg.dt_s
    b = _Builder(ctx.t)
    diag: dict[str, Any] = {"residual_z": {}, "shape": {}}
    n10, n5, n30, n60 = int(10 / dt), int(5 / dt), int(30 / dt), int(60 / dt)
    if len(h) < n30:
        return b.items, diag
    ref = ctx.obs_refs
    alarm = ctx.alarm_onset_s is not None
    SIG = residual_sigma(cfg)

    # ---------------------------------------------------------- per-sensor residuals
    res_strength: dict[str, float] = {}
    # Shape window spans the anomaly onset so an old step is not mistaken for a ramp.
    n_shape = n60
    if alarm:
        n_shape = int(min(len(h), max(n60, (ctx.t - ctx.alarm_onset_s + 15.0) / dt)))
    for sid in SENSORS:
        r = h.window(f"res:{sid}", n_shape)
        y = h.window(sid, n5)
        sig = SIG[sid]
        rs = [ref.get(f"SENSOR:{sid}"), ref.get(f"TWIN:{sid}")]
        nan_frac = float(np.isnan(y).mean()) if y.size else 1.0
        if nan_frac > 0.5:
            b.add(f"dropout:{sid}", sid, nan_frac, f"{sid} reports missing values ({nan_frac:.0%} of last 5 s)",
                  sid, "SENSOR", rs, nan_frac)
            continue
        yy = y[~np.isnan(y)]
        noise_spec = {"LT-101": cfg.sensors.level_noise_m, "FT-101": cfg.sensors.flow_noise_lps,
                      "FT-102": cfg.sensors.flow_noise_lps, "ZT-101": cfg.sensors.valve_pos_noise_pct,
                      "SC-101": cfg.sensors.pump_speed_noise}[sid]
        if yy.size >= n5 * 0.8:
            sd = float(np.std(yy))
            if sd < 0.2 * noise_spec:
                b.add(f"frozen:{sid}", sid, 1.0 - sd / (0.2 * noise_spec),
                      f"{sid} value is static (std {sd:.2g} vs spec noise {noise_spec:.2g})", sid, "SENSOR", rs, sd)
        rr = r[~np.isnan(r)]
        if rr.size < n10:
            continue
        m10 = float(np.mean(rr[-n10:]))
        z = abs(m10) / sig
        diag["residual_z"][sid] = round(z, 2)
        s_z = ramp(z, 3.0, 8.0)
        res_strength[sid] = s_z
        if s_z <= 0:
            continue
        blocks = _blocks(rr, int(1 / dt))
        jump = float(np.max(np.abs(np.diff(blocks)))) if blocks.size > 2 else 0.0
        frac_jump = min(1.0, jump / max(abs(m10), 1e-9))
        mem = ctx.shape_memory.get(sid)
        if mem is None or ctx.t - mem["last_anomalous"] > 30.0:
            mem = {"frac_jump": frac_jump, "since": ctx.t, "last_anomalous": ctx.t}
        elif ctx.t - mem["since"] <= 30.0:
            mem["frac_jump"] = max(mem["frac_jump"], frac_jump)  # still characterising the onset
        else:
            frac_jump = mem["frac_jump"]
        mem["last_anomalous"] = ctx.t
        ctx.shape_memory[sid] = mem
        step = s_z * ramp(frac_jump, 0.2, 0.6)
        rmp = s_z * ramp(0.6 - frac_jump, 0.0, 0.4)
        diag["shape"][sid] = round(frac_jump, 3)
        unit = {"LT-101": "m", "FT-101": "L/s", "FT-102": "L/s", "ZT-101": "%", "SC-101": "pu"}[sid]
        b.add(f"residual_step:{sid}", sid, step,
              f"{sid} residual r=y-ŷ is {m10:+.3g} {unit} ({z:.1f}σ) with abrupt onset", sid, "TWIN_PREDICTION", rs, m10)
        b.add(f"residual_ramp:{sid}", sid, rmp,
              f"{sid} residual r=y-ŷ is {m10:+.3g} {unit} ({z:.1f}σ), growing gradually", sid, "TWIN_PREDICTION", rs, m10)

    any_residual = max(res_strength.values(), default=0.0)

    # ----------------------------------------------------------------- watermark
    wm = ctx.watermark or {}
    wref = [ref.get("WATERMARK")]
    for sid, e in (wm.get("signals") or {}).items():
        st = e.get("status")
        exp = e.get("expected_correlation") or 0.0
        if e.get("detectability") == "UNDETECTABLE" or st in (None, "NO_DATA", "UNDETECTABLE"):
            continue
        corr = e.get("correlation")
        if st == "ABSENT":
            s = ramp(exp, 0.15, 0.5)
            b.add(f"wm_absent:{sid}", sid, s, f"Watermark response missing in {sid} (ρ={corr} vs expected {exp:.2f})",
                  sid, "WATERMARK", wref + [ref.get(f"SENSOR:{sid}")], corr)
        elif st == "WEAK":
            b.add(f"wm_absent:{sid}", sid, 0.4 * ramp(exp, 0.15, 0.5),
                  f"Watermark response weak in {sid} (ρ={corr} vs expected {exp:.2f})", sid, "WATERMARK", wref, corr)
        elif st == "CORRELATED":
            lag_shift = (e.get("lag_s") or 0.0) - (e.get("nominal_lag_s") or 0.0)
            if lag_shift > 1.0 and sid == "FT-101":
                b.add(f"wm_lag_shift:{sid}", sid, ramp(lag_shift, 1.0, 3.0),
                      f"Watermark response in {sid} arrives {lag_shift:.1f} s late", f"NET:{sid}", "WATERMARK", wref, lag_shift)
            if alarm:
                b.add(f"wm_present:{sid}", sid, ramp(exp, 0.15, 0.5) * min(1.0, corr / max(exp, 1e-6)),
                      f"Watermark response present in {sid} (ρ={corr:.2f}): data is fresh and causally linked to the drive",
                      sid, "WATERMARK", wref, corr)
        if e.get("lag_at_search_edge") and st in ("CORRELATED", "WEAK") and sid == "FT-101":
            b.add(f"wm_lag_shift:{sid}", sid, 0.6, f"Watermark lag in {sid} at search-window edge (possible transport delay)",
                  f"NET:{sid}", "WATERMARK", wref, e.get("lag_s"))

    # -------------------------------------------------------------- camera witness
    cam = ctx.camera
    lt = h.window("LT-101", int(2 / dt))
    lt_now = float(np.nanmean(lt)) if lt.size and not np.isnan(lt).all() else float("nan")
    twin_now = h.last("yhat:LT-101")
    if cam and cam.get("status") == "OK" and cam.get("level_m") is not None and (ctx.t - cam.get("timestamp", -1e9)) < 5.0:
        cl, cu = float(cam["level_m"]), float(cam.get("uncertainty_m") or 0.03)
        tol = 3.0 * math.hypot(cu, cfg.sensors.level_noise_m)
        dis = abs(cl - lt_now) if not math.isnan(lt_now) else float("nan")
        crefs = [cam.get("observation_id"), ref.get("SENSOR:LT-101")]
        if not math.isnan(dis) and dis > tol:
            agrees_twin = abs(cl - twin_now) <= tol
            b.add("camera_disagrees:LT-101", "LT-101", ramp(dis / tol, 1.0, 2.5),
                  f"Camera waterline {cl:.3f} m disagrees with LT-101 {lt_now:.3f} m (tol {tol:.3f} m)"
                  + ("; camera agrees with twin" if agrees_twin else ""), "CAM-01", "CAMERA_ESTIMATE", crefs, dis)
        elif not math.isnan(dis) and alarm:
            b.add("camera_agrees:LT-101", "LT-101", 1.0 - dis / tol,
                  f"Camera waterline {cl:.3f} m corroborates LT-101 {lt_now:.3f} m", "CAM-01", "CAMERA_ESTIMATE", crefs, dis)

    # ----------------------------------------------------------- mass balance
    A = cfg.plant.area_m2
    n_hist = len(h)
    if alarm:
        n_mb = int(min(n_hist - int(3 / dt), max(int(20 / dt), (ctx.t - ctx.alarm_onset_s + 10.0) / dt)))
    else:
        n_mb = min(n_hist - int(3 / dt), n60)
    n_mb = max(n_mb, n30)
    if n_mb > n30 - 1 and n_hist > n_mb + int(2 / dt):
        lt_all = h.window("LT-101", n_mb + int(2 / dt))
        fi = h.window("FT-101", n_mb)
        fo = h.window("FT-102", n_mb)
        if not (np.isnan(lt_all).mean() > 0.3 or np.isnan(fi).mean() > 0.3 or np.isnan(fo).mean() > 0.3):
            w2 = int(2 / dt)
            dl = float(np.nanmean(lt_all[-w2:]) - np.nanmean(lt_all[:w2]))
            integ = float(np.nansum(fi - fo) / 1000.0 * dt / A)
            sd = math.sqrt(2 * cfg.sensors.level_noise_m ** 2 / w2 +
                           (math.sqrt(2) * cfg.sensors.flow_noise_lps / 1000 * dt * math.sqrt(n_mb) / A) ** 2)
            tol = 3 * sd + 0.012
            d = dl - integ
            mrefs = [ref.get("SENSOR:LT-101"), ref.get("SENSOR:FT-101"), ref.get("SENSOR:FT-102")]
            diag["mass_balance"] = {"delta_level_m": dl, "integrated_net_flow_m": integ, "discrepancy_m": d, "tolerance_m": tol,
                                    "window_s": n_mb * dt}
            if abs(d) > tol:
                b.add("mass_balance_inconsistent", "TK-101", ramp(abs(d) / tol, 1.0, 2.5),
                      f"Measured level change {dl:+.3f} m ≠ ∫(FT-101−FT-102)/A dt {integ:+.3f} m over {n_mb*dt:.0f} s",
                      "MASS-BALANCE", "SENSOR", mrefs, d)
            elif alarm and any_residual > 0.2:
                b.add("mass_balance_consistent", "TK-101", (1.0 - abs(d) / tol) * min(1.0, any_residual * 1.5),
                      f"Sensors mutually consistent: Δlevel {dl:+.3f} m ≈ ∫net flow {integ:+.3f} m (sensor set agrees with itself)",
                      "MASS-BALANCE", "SENSOR", mrefs, d)

    # ----------------------------------------------------------- network monitor
    for sid in SENSORS:
        lat = h.window(f"lat:{sid}", n10)
        lat = lat[~np.isnan(lat)]
        nref = [ref.get(f"NET:{sid}")]
        nominal = math.ceil((cfg.network.base_latency_s + 1e-9) / dt) * dt
        if lat.size:
            mlat = float(np.mean(lat))
            b.add(f"latency_elevated:{sid}", f"NET-1/{sid}", ramp(mlat - nominal, 0.3, 2.0),
                  f"Fieldbus latency for {sid} {mlat:.2f} s (nominal {nominal:.2f} s)", f"NET:{sid}", "NETWORK_MONITOR", nref, mlat)
            jit = float(np.std(lat))
            b.add(f"jitter_elevated:{sid}", f"NET-1/{sid}", ramp(jit, 0.15, 1.0),
                  f"Latency jitter for {sid} {jit:.2f} s", f"NET:{sid}", "NETWORK_MONITOR", nref, jit)
        arr = h.window(f"arr:{sid}", n10)
        if arr.size >= n10:
            arrived = float(np.nansum(arr))
            loss = max(0.0, 1.0 - arrived / n10)
            b.add(f"packet_loss:{sid}", f"NET-1/{sid}", ramp(loss, 0.08, 0.4),
                  f"{sid}: {loss:.0%} of expected packets missing in last 10 s", f"NET:{sid}", "NETWORK_MONITOR", nref, loss)
        sq = h.window(f"seqa:{sid}", n10)
        if sq.size >= 2:
            cnt = float(sq[-1] - sq[0])
            b.add(f"seq_anomaly:{sid}", f"NET-1/{sid}", ramp(cnt, 0.5, 20.0),
                  f"{sid}: {cnt:.0f} out-of-order / replayed sequence numbers in 10 s", f"NET:{sid}", "NETWORK_MONITOR", nref, cnt)

    # ------------------------------------------------------- controller consistency
    cref = [ref.get("CTRL:u_cmd"), ref.get("CTRL:shadow")]
    se = h.window("shadow_err", n30)
    ra = h.window("response_active", n30)
    if se.size and not (np.nanmax(ra) > 0 if ra.size else False):
        ms = float(np.nanmean(np.abs(se)))
        if ms > 0.02:
            b.add("control_law_inconsistent", "PLC-101", ramp(ms, 0.02, 0.08),
                  f"Issued pump command deviates from the golden control law by {ms:.3f} (30 s mean)", "PLC", "CONTROLLER_LOG", cref, ms)
        elif alarm:
            b.add("control_law_consistent", "PLC-101", ramp(0.02 - ms, 0.0, 0.015),
                  f"Issued commands match golden control law (mean |Δu| {ms:.4f})", "PLC", "CONTROLLER_LOG", cref, ms)
    lt60 = h.window("LT-101", n60)
    spr = h.last("setpoint_reported")
    if lt60.size >= n30 and not math.isnan(spr):
        dev = float(np.nanmean(lt60[-n30:]) - spr)
        if abs(dev) > 0.25 and any_residual < 0.2:
            b.add("setpoint_departure_unexplained", "PLC-101", ramp(abs(dev), 0.25, 0.8),
                  f"Level held {dev:+.2f} m away from reported setpoint {spr:.2f} m with nominal residuals", "PLC", "CONTROLLER_LOG",
                  cref + [ref.get("SENSOR:LT-101")], dev)
    u60 = h.window("u_cmd", n60)
    if u60.size >= n60:
        du = u60 - h.window("delta_u", n60)
        bl = _blocks(du, int(2 / dt))
        if bl.size > 4:
            sign_changes = int(np.sum(np.diff(np.sign(np.diff(bl))) != 0))
            amp = float(np.std(bl))
            b.add("oscillation", "PLC-101", ramp(amp, 0.04, 0.15) * ramp(sign_changes, 4, 10),
                  f"Control loop oscillating (command std {amp:.3f}, {sign_changes} reversals/60 s)", "LOOP", "CONTROLLER_LOG", cref, amp)

    # ------------------------------------------------------- actuator-specific checks
    p = cfg.plant
    zt = h.window("ZT-101", n10)
    fo10 = h.window("FT-102", n10)
    lt10 = h.window("LT-101", n10)
    zt_hat = h.window("yhat:ZT-101", n10)
    if zt.size >= n10 and res_strength.get("ZT-101", 0) > 0 and not np.isnan(zt).all():
        lvl = max(float(np.nanmean(lt10)), 0.0)
        q_fb = valve_flow(p, float(np.nanmean(zt)) / 100.0, lvl) * 1000
        q_cmd = valve_flow(p, float(np.nanmean(zt_hat)) / 100.0, lvl) * 1000
        q_meas = float(np.nanmean(fo10))
        if abs(q_meas - q_fb) < abs(q_meas - q_cmd):
            b.add("valve_actuator_divergence", "FV-101", res_strength["ZT-101"] * ramp(abs(q_meas - q_cmd) - abs(q_meas - q_fb), 0.2, 1.5),
                  f"Valve feedback {np.nanmean(zt):.1f}% ≠ command-model {np.nanmean(zt_hat):.1f}%, and FT-102 {q_meas:.2f} L/s matches the feedback",
                  "FV-101", "SENSOR", [ref.get("SENSOR:ZT-101"), ref.get("SENSOR:FT-102")], q_meas - q_cmd)
        else:
            b.add("valve_feedback_inconsistent_with_flow", "ZT-101", res_strength["ZT-101"],
                  f"ZT-101 deviates from command but FT-102 {q_meas:.2f} L/s matches the commanded position",
                  "FV-101", "SENSOR", [ref.get("SENSOR:ZT-101"), ref.get("SENSOR:FT-102")], q_meas - q_fb)
    if res_strength.get("FT-102", 0) > 0.3 and res_strength.get("ZT-101", 0) == 0 and zt.size >= n10:
        b.add("valve_feedback_tracks_command", "FV-101", res_strength["FT-102"],
              f"Outlet flow anomalous while valve feedback ZT-101 tracks its command ({np.nanmean(zt):.1f}%)",
              "FV-101", "SENSOR", [ref.get("SENSOR:ZT-101")], float(np.nanmean(zt)))
    sc = h.window("SC-101", n10)
    fi10 = h.window("FT-101", n10)
    if sc.size >= n10 and not np.isnan(sc).all():
        lvl = max(float(np.nanmean(lt10)), 0.0)
        q_meas = float(np.nanmean(fi10))
        q_fb = pump_flow(p, float(np.nanmean(sc)), lvl) * 1000
        q_cmd = float(np.nanmean(h.window("yhat:FT-101", n10)))
        if res_strength.get("SC-101", 0) > 0:
            if abs(q_meas - q_fb) < abs(q_meas - q_cmd):
                b.add("pump_speed_divergence", "P-101", res_strength["SC-101"],
                      f"Pump speed feedback {np.nanmean(sc):.3f} ≠ commanded model; FT-101 {q_meas:.2f} L/s matches feedback",
                      "P-101", "SENSOR", [ref.get("SENSOR:SC-101"), ref.get("SENSOR:FT-101")], float(np.nanmean(sc)))
        elif res_strength.get("FT-101", 0) > 0:
            b.add("pump_speed_tracks_command", "P-101", res_strength["FT-101"],
                  f"Inlet flow anomalous while pump speed SC-101 tracks its command ({np.nanmean(sc):.3f})",
                  "P-101-SPEED", "SENSOR", [ref.get("SENSOR:SC-101")], float(np.nanmean(sc)))
            r_in = h.window("res:FT-101", max(n60, n_shape))
            bl = _blocks(r_in, int(5 / dt))
            hf = float(np.nanstd(np.diff(h.window("FT-101", n10))))
            spec_hf = math.sqrt(2) * cfg.sensors.flow_noise_lps
            if bl.size >= 6 and np.sum(np.diff(np.sign(bl)) != 0) >= 2 and abs(np.nanmean(bl)) < 0.6 * np.nanmax(np.abs(bl)):
                b.add("inflow_oscillatory", "SUPPLY", res_strength["FT-101"],
                      "Inlet-flow residual oscillates with speed feedback nominal (supply head fluctuation signature)",
                      "P-101", "SENSOR", [ref.get("SENSOR:FT-101"), ref.get("SENSOR:SC-101")], float(np.nanstd(bl)))
            elif q_meas < q_fb:
                b.add("pump_hydraulic_deficit", "P-101", res_strength["FT-101"],
                      f"Pump speed nominal but inlet flow {q_meas:.2f} L/s below pump-curve {q_fb:.2f} L/s",
                      "P-101", "SENSOR", [ref.get("SENSOR:FT-101"), ref.get("SENSOR:SC-101")], q_meas - q_fb)
            if hf > 2.5 * spec_hf:
                b.add("inflow_hf_noise", "P-101", ramp(hf / spec_hf, 2.5, 6.0),
                      f"Inlet flow high-frequency noise {hf:.2f} L/s ≫ spec {spec_hf:.2f} (cavitation signature)",
                      "P-101-HF", "SENSOR", [ref.get("SENSOR:FT-101")], hf)

    # ------------------------------------------------------- active diagnostics
    for tid, res in ctx.diagnostics.items():
        for ev in res.get("evidence", []):
            b.add(ev["feature"], ev["component"], ev["strength"], ev["description"], ev["group"], ev["source_type"],
                  [res.get("observation_id")] + ev.get("refs", []), ev.get("value"))
    return b.items, diag
