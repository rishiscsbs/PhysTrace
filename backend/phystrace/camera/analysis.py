"""OpenCV waterline estimation and image-quality gating (FR-012, AT-008, AT-009).

Never invents a level: when quality checks fail the result status is
LOW_QUALITY or UNAVAILABLE and `level_m` is None.
"""
from __future__ import annotations

import cv2
import numpy as np

BLUR_MIN_LAPLACIAN_VAR = 20.0
DARK_MAX_MEAN = 30.0
BRIGHT_MIN_MEAN = 235.0
FROZEN_DIFF = 0.25
FROZEN_FRAMES = 3
N_STRIPS = 5


class FrameAnalyzer:
    def __init__(self) -> None:
        self._prev_gray: np.ndarray | None = None
        self._frozen_count = 0

    def reset(self) -> None:
        self._prev_gray = None
        self._frozen_count = 0

    def analyze(self, frame: np.ndarray | None, calibration: dict | None, t: float, source_mode: str) -> dict:
        out: dict = {"timestamp": t, "source_mode": source_mode, "status": "UNAVAILABLE", "level_m": None,
                     "uncertainty_m": None, "reasons": [], "quality": {}, "indicator": None, "waterline_px": None}
        if frame is None:
            out["reasons"].append("NO_FRAME")
            out["reason"] = "no frame from selected source"
            return out
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frozen-frame check on the full image
        if self._prev_gray is not None and self._prev_gray.shape == gray_full.shape:
            diff = float(np.mean(cv2.absdiff(gray_full, self._prev_gray)))
            self._frozen_count = self._frozen_count + 1 if diff < FROZEN_DIFF else 0
            out["quality"]["frame_diff"] = round(diff, 3)
        self._prev_gray = gray_full
        if calibration is None:
            out["status"] = "UNCALIBRATED"
            out["reasons"].append("NO_ROI_CALIBRATION")
            out["reason"] = "ROI / 2-point calibration required before estimating a level"
            out["quality"]["brightness"] = round(float(gray_full.mean()), 2)
            return out
        r = calibration["roi"]
        H, W = gray_full.shape
        x0, y0 = max(0, int(r["x"])), max(0, int(r["y"]))
        x1, y1 = min(W, x0 + int(r["w"])), min(H, y0 + int(r["h"]))
        if x1 - x0 < 10 or y1 - y0 < 20:
            out["reasons"].append("ROI_OUT_OF_FRAME")
            out["reason"] = "calibrated ROI does not fit this frame"
            return out
        roi = gray_full[y0:y1, x0:x1].astype(np.float32)
        brightness = float(roi.mean())
        lap_var = float(cv2.Laplacian(roi, cv2.CV_32F).var())
        out["quality"].update(brightness=round(brightness, 2), laplacian_var=round(lap_var, 2))
        if calibration.get("indicator_roi"):
            out["indicator"] = indicator_state(frame, calibration["indicator_roi"])

        fatal, degraded = [], []
        if self._frozen_count >= FROZEN_FRAMES:
            fatal.append("FROZEN_FRAME")
        if brightness < DARK_MAX_MEAN:
            fatal.append("UNDEREXPOSED")
        if brightness > BRIGHT_MIN_MEAN:
            fatal.append("OVEREXPOSED")
        if lap_var < BLUR_MIN_LAPLACIAN_VAR:
            degraded.append("BLURRED")

        # waterline per vertical strip: strongest horizontal edge (Sobel-y) in the ROI
        sm = cv2.GaussianBlur(roi, (5, 5), 0)
        gy = np.abs(cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3))
        strips = np.array_split(np.arange(roi.shape[1]), N_STRIPS)
        rows, confs = [], []
        for cols in strips:
            prof = gy[:, cols].mean(axis=1)
            prof[:3] = 0
            prof[-3:] = 0
            k = int(np.argmax(prof))
            peak = float(prof[k])
            base = float(np.median(prof)) + 1e-3
            conf = peak / base
            if 0 < k < prof.size - 1:
                a, b_, c = prof[k - 1], prof[k], prof[k + 1]
                den = a - 2 * b_ + c
                sub = 0.5 * (a - c) / den if abs(den) > 1e-6 else 0.0
            else:
                sub = 0.0
            rows.append(k + float(np.clip(sub, -0.5, 0.5)))
            confs.append(conf)
        rows_a, confs_a = np.array(rows), np.array(confs)
        good = confs_a > 4.0
        out["quality"]["strip_confidence"] = [round(float(c), 2) for c in confs_a]
        if good.sum() < 3:
            degraded.append("NO_CLEAR_WATERLINE")
        else:
            spread = float(np.max(rows_a[good]) - np.min(rows_a[good]))
            out["quality"]["strip_spread_px"] = round(spread, 2)
            if spread > 6.0 or good.sum() < N_STRIPS - 1:
                degraded.append("OCCLUDED_OR_INCONSISTENT")
            else:
                # A second comparably strong horizontal edge means an object spans the ROI.
                prof_all = gy.mean(axis=1)
                prof_all[:3] = 0
                prof_all[-3:] = 0
                k = int(np.argmax(prof_all))
                masked = prof_all.copy()
                masked[max(0, k - 6):k + 7] = 0
                if masked.max() > 0.5 * prof_all[k]:
                    degraded.append("OCCLUDED_OR_INCONSISTENT")
        dark_frac = float((roi < 50).mean())
        out["quality"]["dark_fraction"] = round(dark_frac, 3)
        if dark_frac > 0.15 and "OCCLUDED_OR_INCONSISTENT" not in degraded:
            degraded.append("OCCLUDED_OR_INCONSISTENT")
        if fatal:
            out["status"] = "UNAVAILABLE"
            out["reasons"] = fatal + degraded
            out["reason"] = ", ".join(out["reasons"])
            return out
        if degraded:
            out["status"] = "LOW_QUALITY"
            out["reasons"] = degraded
            out["reason"] = ", ".join(degraded)
            return out

        row = float(np.median(rows_a[good]))
        # contrast check: liquid below vs air above must differ
        ri = int(round(row))
        above = roi[max(0, ri - 25):max(1, ri - 3)].mean() if ri > 4 else np.nan
        below = roi[ri + 3:ri + 25].mean() if ri + 4 < roi.shape[0] else np.nan
        contrast = float(abs(above - below)) if not (np.isnan(above) or np.isnan(below)) else 0.0
        out["quality"]["contrast"] = round(contrast, 2)
        if contrast < 12.0:
            out["status"] = "LOW_QUALITY"
            out["reasons"] = ["LOW_CONTRAST"]
            out["reason"] = "LOW_CONTRAST"
            return out
        py = y0 + row
        p1, p2 = calibration["points"][0], calibration["points"][1]
        if abs(p2["pixel_y"] - p1["pixel_y"]) < 1e-6:
            out["status"] = "UNAVAILABLE"
            out["reasons"] = ["DEGENERATE_CALIBRATION"]
            return out
        m_per_px = (p2["height_m"] - p1["height_m"]) / (p2["pixel_y"] - p1["pixel_y"])
        level = p1["height_m"] + (py - p1["pixel_y"]) * m_per_px
        px_unc = float(np.std(rows_a[good])) + 1.0
        out.update(status="OK", level_m=round(float(level), 4), uncertainty_m=round(abs(m_per_px) * px_unc + 0.005, 4),
                   waterline_px=round(py, 2), reason="")
        return out


def indicator_state(frame: np.ndarray, r: dict) -> dict:
    x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
    patch = frame[max(0, y):y + h, max(0, x):x + w]
    if patch.size == 0:
        return {"state": "UNKNOWN", "lit_fraction": None}
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 100, 120), (90, 255, 255))
    frac = float(mask.mean() / 255.0)
    return {"state": "ON" if frac > 0.2 else "OFF", "lit_fraction": round(frac, 3), "hue_band": "green 40-90"}


def annotate(frame: np.ndarray, calibration: dict | None, est: dict | None) -> np.ndarray:
    img = frame.copy()
    if calibration:
        r = calibration["roi"]
        cv2.rectangle(img, (int(r["x"]), int(r["y"])), (int(r["x"] + r["w"]), int(r["y"] + r["h"])), (255, 82, 0), 1)
        for p in calibration["points"]:
            y = int(p["pixel_y"])
            cv2.line(img, (int(r["x"]) - 8, y), (int(r["x"]), y), (255, 82, 0), 1)
        if calibration.get("indicator_roi"):
            q = calibration["indicator_roi"]
            cv2.rectangle(img, (int(q["x"]), int(q["y"])), (int(q["x"] + q["w"]), int(q["y"] + q["h"])), (6, 119, 217), 1)
    if est and est.get("waterline_px") is not None and calibration:
        r = calibration["roi"]
        y = int(round(est["waterline_px"]))
        cv2.line(img, (int(r["x"]), y), (int(r["x"] + r["w"]), y), (0, 46, 255), 1)
    return img
