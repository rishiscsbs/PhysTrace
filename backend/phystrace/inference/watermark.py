"""Bounded physical watermark correlator (defender side).

The defender knows the exact pseudorandom sequence delta_u_k it injected at the
pump drive. Through the known physics, delta_u produces:

  * inlet flow (FT-101):  dQ_k  ~ g * lag_tau(delta_u)           (memoryless after pump lag)
  * tank level (LT-101):  dh_k  ~ sum(dQ) * dt / A               (integrating response)

We correlate the expected response template with the received measurement over
a sliding window (both band-passed identically), searching over lags. A drop in
correlation is *evidence* (e.g. of replay or a frozen signal), never proof.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from ..config import SimConfig


def _moving_avg(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    if n == 1 or x.size == 0:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x)
    for_full = (c[n:] - c[:-n]) / n
    out[n - 1:] = for_full
    # ramp-in for the first samples
    out[: n - 1] = c[1:n] / np.arange(1, n)
    return out


def _bandpass(x: np.ndarray, lp: int, hp: int) -> np.ndarray:
    y = _moving_avg(x, lp)
    return y - _moving_avg(y, hp)


def _ffill(x: np.ndarray) -> tuple[np.ndarray, float]:
    mask = np.isnan(x)
    frac = float(mask.mean()) if x.size else 1.0
    if not mask.any():
        return x, 0.0
    idx = np.where(~mask, np.arange(x.size), 0)
    np.maximum.accumulate(idx, out=idx)
    y = x[idx]
    if np.isnan(y[0]):
        first = np.nanmean(x) if frac < 1.0 else 0.0
        y = np.where(np.isnan(y), first, y)
    return y, frac


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _best_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[float, int]:
    """Pearson correlation of a[:N-l] with b[l:] for l = 0..max_lag, vectorised via prefix sums."""
    n = a.size
    best, best_lag = -2.0, 0
    if n < 20:
        return best, best_lag
    ca, cb = np.concatenate(([0.0], np.cumsum(a))), np.concatenate(([0.0], np.cumsum(b)))
    ca2, cb2 = np.concatenate(([0.0], np.cumsum(a * a))), np.concatenate(([0.0], np.cumsum(b * b)))
    for lag in range(0, min(max_lag, n - 11) + 1):
        m = n - lag
        sa, sb = ca[m], cb[n] - cb[lag]
        va = ca2[m] - sa * sa / m
        vb = cb2[n] - cb2[lag] - sb * sb / m
        if va <= 1e-18 or vb <= 1e-18:
            continue
        cov = float(np.dot(a[:m], b[lag:])) - sa * sb / m
        c = cov / math.sqrt(va * vb)
        if c > best:
            best, best_lag = c, lag
    return best, best_lag


class WatermarkCorrelator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.dt = cfg.dt_s
        self.n_win = int(cfg.watermark.window_s / self.dt)
        self.max_lag = int(cfg.watermark.max_lag_s / self.dt)
        self.nominal_lag = max(1, math.ceil((cfg.network.base_latency_s + 1e-9) / self.dt))

    def analyze(self, hist, flow_gain_lps: float, measured_latency_s: float | None = None) -> dict:
        cfg = self.cfg
        n = min(len(hist), self.n_win)
        result = {"window_s": n * self.dt, "epsilon": float(cfg.watermark.epsilon), "signals": {}, "warnings": []}
        if not cfg.watermark.enabled:
            result["warnings"].append("WATERMARK_DISABLED")
            result["status"] = "DISABLED"
            return result
        if n < int(60.0 / self.dt):
            result["status"] = "WARMING_UP"
            return result
        du = hist.window("delta_u", n)
        du = np.nan_to_num(du)
        sat = hist.window("wm_saturated", n)
        # Clipped samples never reached the actuator: exclude them from the expected response.
        du = np.where(np.nan_to_num(sat) > 0.5, 0.0, du)
        sat_frac = float(np.nanmean(sat)) if sat.size else 0.0
        p = cfg.plant
        alpha = 1.0 - math.exp(-self.dt / p.pump_time_constant_s)
        dn = lfilter([alpha], [1.0, -(1.0 - alpha)], du)  # first-order pump lag
        dq = flow_gain_lps * dn  # L/s
        dh = np.cumsum(dq / 1000.0) * self.dt / p.area_m2  # m
        s = cfg.sensors
        specs = {
            # signal: template, low-pass samples, high-pass samples, white noise sigma, extra wander sigma
            "FT-101": (dq, 1, int(8 / self.dt), s.flow_noise_lps, p.process_noise_lps * 0.6),
            "LT-101": (dh, int(1.0 / self.dt), int(12 / self.dt), s.level_noise_m, 0.0012),
        }
        for sid, (tmpl, lp, hp, sigma, wander) in specs.items():
            y_raw = hist.window(sid, n)
            y, nan_frac = _ffill(y_raw)
            entry: dict = {"nan_fraction": nan_frac}
            if nan_frac > 0.5:
                entry.update(status="NO_DATA", correlation=None, expected_correlation=None, lag_s=None)
                result["signals"][sid] = entry
                continue
            tb = _bandpass(tmpl, lp, hp)
            yb = _bandpass(y, lp, hp)
            skip = hp  # discard filter ramp-in
            tb, yb = tb[skip:], yb[skip:]
            # Robust scale (MAD) and winsorising: isolated step transients must not dominate the statistic.
            med = float(np.median(yb))
            rsd = 1.4826 * float(np.median(np.abs(yb - med)))
            if rsd > 0:
                yb = np.clip(yb, med - 3 * rsd, med + 3 * rsd)
            best, best_lag = _best_lag(tb, yb, self.max_lag)
            sig_var = float(np.var(tb))
            noise_var = sigma ** 2 / max(lp, 1) + wander ** 2
            obs_var = rsd ** 2
            # Expected correlation uses the larger of the modelled (signal + noise) variance and the observed
            # band-passed variance: unexplained in-band disturbance honestly lowers detectability instead of
            # producing a spurious "watermark absent".
            denom = max(obs_var, sig_var + noise_var)
            expected = math.sqrt(sig_var / denom) if sig_var > 0 and denom > 0 else 0.0
            masked = obs_var > 3.0 * (sig_var + noise_var)
            if masked:
                result["warnings"].append(f"DISTURBANCE_MASKING:{sid}")
            if best == -2.0:
                corr = float("nan")
            else:
                corr = best
            if expected < 0.15:
                status = "UNDETECTABLE"
            elif math.isnan(corr):
                status = "ABSENT"  # zero-variance measurement (frozen) cannot carry the watermark
            elif corr >= 0.5 * expected:
                status = "CORRELATED"
            elif corr >= 0.25 * expected:
                status = "WEAK"
            else:
                status = "ABSENT"
            detectability = "DETECTABLE" if expected >= 0.35 else "DEGRADED" if expected >= 0.15 else "UNDETECTABLE"
            entry.update(
                status=status,
                correlation=None if math.isnan(corr) else round(corr, 4),
                expected_correlation=round(expected, 4),
                lag_s=round(best_lag * self.dt, 2),
                nominal_lag_s=round(self.nominal_lag * self.dt, 2),
                detectability=detectability,
                lag_at_search_edge=best_lag >= self.max_lag - 1,
                observed_band_variance=obs_var,
                masked_by_disturbance=masked,
            )
            result["signals"][sid] = entry
            if detectability != "DETECTABLE" and not masked:
                result["warnings"].append(f"HIGH_NOISE_LOW_SNR:{sid}")
        if sat_frac > 0.1:
            result["warnings"].append(f"ACTUATOR_SATURATION:{sat_frac:.0%} of window clipped")
        if measured_latency_s is not None and measured_latency_s > cfg.watermark.max_lag_s:
            result["warnings"].append(f"TRANSPORT_DELAY_EXCEEDS_LAG_SEARCH:{measured_latency_s:.1f}s")
        result["saturation_fraction"] = round(sat_frac, 4)
        result["status"] = "OK"
        return result
