"""Camera mode state machine.

Rules (FR-010, AT-007):
* No source is active until the user explicitly selects one.
* LIVE_CAMERA failure / disconnect -> UNAVAILABLE. Never a silent switch to SIMULATED_VISUAL.
* Recordings stay local (data/ directory) by default.
"""
from __future__ import annotations

import threading
from typing import Callable

import cv2
import numpy as np

from .analysis import FrameAnalyzer, annotate
from .render import DEFAULT_SIM_CALIBRATION
from .sources import FrameSource, LiveCameraSource, SimulatedVisualSource, VideoReplaySource

MODES = ("LIVE_CAMERA", "VIDEO_REPLAY", "SIMULATED_VISUAL")


class CameraManager:
    def __init__(self, scene_provider: Callable[[], dict] | None = None):
        self._scene_provider = scene_provider
        self.source: FrameSource | None = None
        self.mode = "NONE"
        self.calibrations: dict[str, dict] = {}  # mode -> calibration dict
        self.analyzer = FrameAnalyzer()
        self.last_frame: np.ndarray | None = None
        self.last_estimate: dict | None = None
        self._lock = threading.RLock()
        self.history: list[dict] = []

    def set_scene_provider(self, fn: Callable[[], dict]) -> None:
        self._scene_provider = fn

    def select(self, mode: str, *, device_index: int = 0, video_path: str | None = None, loop: bool = True) -> dict:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        with self._lock:
            self.stop()
            self.analyzer.reset()
            if mode == "LIVE_CAMERA":
                self.source = LiveCameraSource(device_index).start()
            elif mode == "VIDEO_REPLAY":
                if not video_path:
                    raise ValueError("video_path required for VIDEO_REPLAY")
                self.source = VideoReplaySource(video_path, loop).start()
            else:
                if self._scene_provider is None:
                    raise ValueError("no simulator scene available")
                self.source = SimulatedVisualSource(self._scene_provider)
            self.mode = mode
            return self.status()

    def stop(self) -> None:
        with self._lock:
            if self.source is not None:
                self.source.close()
            self.source = None
            self.mode = "NONE"
            self.last_frame = None
            self.last_estimate = None

    def calibration(self) -> dict | None:
        return self.calibrations.get(self.mode)

    def set_calibration(self, mode: str, cal: dict) -> None:
        self.calibrations[mode] = cal

    def default_simulated_calibration(self) -> dict:
        return dict(DEFAULT_SIM_CALIBRATION)

    def ready(self) -> bool:
        return self.source is not None and self.source.state == "ACTIVE" and self.calibration() is not None

    def capture(self, t: float) -> dict:
        """Grab and analyse one frame from the selected source (no substitution on failure)."""
        with self._lock:
            if self.source is None:
                est = {"timestamp": t, "status": "UNAVAILABLE", "level_m": None, "reason": "no camera source selected",
                       "reasons": ["NO_SOURCE_SELECTED"], "source_mode": "NONE", "quality": {}}
                self.last_estimate = est
                return est
            frame = self.source.read()
            if frame is None:
                st = self.source.state
                est = {"timestamp": t, "status": "UNAVAILABLE" if st != "ERROR" else "ERROR", "level_m": None,
                       "reason": self.source.error or f"source state {st}", "reasons": [f"SOURCE_{st}"],
                       "source_mode": self.mode, "quality": {}}
                self.last_frame = None
                self.last_estimate = est
                return est
            self.last_frame = frame
            est = self.analyzer.analyze(frame, self.calibration(), t, self.mode)
            self.last_estimate = est
            return est

    def jpeg(self, annotated: bool = True) -> bytes | None:
        with self._lock:
            if self.last_frame is None:
                return None
            img = annotate(self.last_frame, self.calibration(), self.last_estimate) if annotated else self.last_frame
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buf.tobytes() if ok else None

    def status(self) -> dict:
        src = self.source.info() if self.source else {"mode": "NONE", "state": "IDLE", "error": None, "frames": 0}
        return {
            "mode": self.mode,
            "source": src,
            "calibrated": self.calibration() is not None,
            "calibration": self.calibration(),
            "estimate": self.last_estimate,
            "simulated": self.mode == "SIMULATED_VISUAL",
            "label": "SIMULATED_VISUAL — synthetic render, not a real camera" if self.mode == "SIMULATED_VISUAL" else None,
            "frame_size": None if self.last_frame is None else {"w": int(self.last_frame.shape[1]), "h": int(self.last_frame.shape[0])},
        }
