"""SIMULATED_VISUAL renderer: an explicitly labelled synthetic camera view of TK-101.

The scene is rendered from the simulator's physical tank state. Every frame
carries a burnt-in "SIMULATED_VISUAL" badge so it can never be mistaken for a
real camera feed.
"""
from __future__ import annotations

import cv2
import numpy as np

FRAME_W, FRAME_H = 240, 360
TANK_X0, TANK_X1 = 60, 180
TANK_TOP_Y, TANK_BOTTOM_Y = 30, 330  # pixel rows of the rim (h=3.0 m) and floor (h=0)

DEFAULT_SIM_CALIBRATION = {
    "roi": {"x": 70, "y": 34, "w": 100, "h": 292},
    "points": [{"pixel_y": TANK_BOTTOM_Y, "height_m": 0.0}, {"pixel_y": TANK_TOP_Y, "height_m": 3.0}],
    "indicator_roi": {"x": 196, "y": 44, "w": 28, "h": 28},
}


def level_to_pixel(h: float, tank_height_m: float = 3.0) -> float:
    return TANK_BOTTOM_Y - (TANK_BOTTOM_Y - TANK_TOP_Y) * (h / tank_height_m)


def render_tank(scene: dict, noise: np.ndarray | None = None, *, blur: int = 0, brightness: float = 1.0,
                occlude: bool = False) -> np.ndarray:
    img = np.empty((FRAME_H, FRAME_W, 3), np.float32)
    ramp = np.linspace(185, 160, FRAME_H, dtype=np.float32)[:, None]
    img[:] = ramp[..., None]
    # tank interior (empty part) and walls
    img[TANK_TOP_Y:TANK_BOTTOM_Y, TANK_X0:TANK_X1] = (226, 228, 230)
    img[TANK_TOP_Y - 4:TANK_BOTTOM_Y + 4, TANK_X0 - 6:TANK_X0] = (64, 64, 70)
    img[TANK_TOP_Y - 4:TANK_BOTTOM_Y + 4, TANK_X1:TANK_X1 + 6] = (64, 64, 70)
    img[TANK_BOTTOM_Y:TANK_BOTTOM_Y + 6, TANK_X0 - 6:TANK_X1 + 6] = (64, 64, 70)
    # graduation marks every 0.5 m on the left wall (outside the ROI)
    for k in range(7):
        y = int(round(level_to_pixel(k * 0.5, scene["tank_height_m"])))
        img[y:y + 1, TANK_X0 - 14:TANK_X0 - 6] = (40, 40, 40)
    # liquid
    y_s = level_to_pixel(min(max(scene["h"], 0.0), scene["tank_height_m"]), scene["tank_height_m"])
    yi = int(np.floor(y_s))
    frac = y_s - yi
    if yi < TANK_BOTTOM_Y:
        top = max(yi + 1, TANK_TOP_Y)
        depth = np.linspace(0, 1, max(TANK_BOTTOM_Y - top, 1), dtype=np.float32)[:, None, None]
        water = np.array([176, 118, 38], np.float32) * (1 - 0.25 * depth)
        img[top:TANK_BOTTOM_Y, TANK_X0:TANK_X1] = water
        if TANK_TOP_Y <= yi < TANK_BOTTOM_Y:
            blend = (1 - frac) * np.array([226, 228, 230], np.float32) + frac * np.array([176, 118, 38], np.float32)
            img[yi, TANK_X0:TANK_X1] = blend
        # meniscus highlight just below the surface
        my = min(max(yi + 1, TANK_TOP_Y), TANK_BOTTOM_Y - 1)
        img[my:my + 1, TANK_X0:TANK_X1] = (205, 170, 110)
    # pump-running indicator lamp
    lamp = (40, 210, 40) if scene.get("pump_running") else (30, 55, 30)
    cv2.circle(img, (210, 58), 10, lamp, -1)
    cv2.circle(img, (210, 58), 12, (50, 50, 50), 2)
    if occlude:
        cv2.rectangle(img, (TANK_X0 - 10, 90), (TANK_X1 + 10, 300), (35, 35, 35), -1)
    img *= brightness
    if noise is not None:
        img += noise
    out = np.clip(img, 0, 255).astype(np.uint8)
    if blur:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    cv2.rectangle(out, (0, FRAME_H - 18), (FRAME_W, FRAME_H), (0, 0, 0), -1)
    cv2.putText(out, "SIMULATED_VISUAL", (6, FRAME_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)
    return out
