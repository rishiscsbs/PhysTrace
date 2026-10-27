"""Camera frame sources. Each source reports its own failure; none substitutes another."""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Callable

import cv2
import numpy as np

from .render import render_tank


class SourceError(RuntimeError):
    pass


class FrameSource:
    mode = "NONE"

    def __init__(self) -> None:
        self.state = "IDLE"  # IDLE | ACTIVE | UNAVAILABLE | ERROR | ENDED
        self.error: str | None = None
        self.frames = 0

    def read(self) -> np.ndarray | None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def info(self) -> dict:
        return {"mode": self.mode, "state": self.state, "error": self.error, "frames": self.frames}


class SimulatedVisualSource(FrameSource):
    """Explicitly labelled synthetic view rendered from the simulator scene."""

    mode = "SIMULATED_VISUAL"

    def __init__(self, scene_provider: Callable[[], dict], seed: int = 0) -> None:
        super().__init__()
        self._scene = scene_provider
        self._rng = np.random.default_rng(seed)
        self.state = "ACTIVE"
        self.degrade: dict = {}  # lab controls: blur, brightness, occlude, freeze

    def read(self) -> np.ndarray | None:
        if self.degrade.get("freeze") and getattr(self, "_last", None) is not None:
            self.frames += 1
            return self._last
        noise = self._rng.normal(0.0, 3.0, (360, 240, 3))
        f = render_tank(self._scene(), noise, blur=int(self.degrade.get("blur", 0)),
                        brightness=float(self.degrade.get("brightness", 1.0)), occlude=bool(self.degrade.get("occlude", False)))
        self._last = f
        self.frames += 1
        return f


class _ThreadedCapture(FrameSource):
    """Background reader for cv2.VideoCapture with explicit failure states."""

    MAX_CONSECUTIVE_FAILS = 10

    def __init__(self) -> None:
        super().__init__()
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fps = 15.0

    def _open(self) -> cv2.VideoCapture:
        raise NotImplementedError

    def _on_eof(self) -> bool:
        """Return True to continue (e.g. loop), False to stop."""
        return False

    def start(self) -> "FrameSource":
        try:
            cap = self._open()
        except Exception as e:  # noqa: BLE001
            self.state, self.error = "UNAVAILABLE", f"open failed: {e}"
            return self
        if cap is None or not cap.isOpened():
            self.state, self.error = "UNAVAILABLE", "device/file could not be opened"
            if cap is not None:
                cap.release()
            return self
        self._cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and 1 <= fps <= 120:
            self._fps = float(fps)
        self.state = "ACTIVE"
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"camera-{self.mode}")
        self._thread.start()
        return self

    def _run(self) -> None:
        fails = 0
        period = 1.0 / self._fps
        while not self._stop.is_set():
            t0 = time.monotonic()
            ok, frame = self._cap.read() if self._cap is not None else (False, None)
            if ok and frame is not None:
                fails = 0
                with self._lock:
                    self._frame = frame
                self.frames += 1
            else:
                if self._is_file() and self._on_eof():
                    continue
                if self._is_file():
                    self.state, self.error = "ENDED", "end of video file"
                    break
                fails += 1
                if fails >= self.MAX_CONSECUTIVE_FAILS:
                    # Disconnect: report it. Never switch to another source.
                    self.state, self.error = "UNAVAILABLE", "camera disconnected (read failures)"
                    with self._lock:
                        self._frame = None
                    break
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _is_file(self) -> bool:
        return False

    def read(self) -> np.ndarray | None:
        if self.state != "ACTIVE":
            return None
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        if self.state == "ACTIVE":
            self.state = "IDLE"

    def simulate_disconnect(self) -> None:
        """Test hook: behave exactly like a USB unplug."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class LiveCameraSource(_ThreadedCapture):
    mode = "LIVE_CAMERA"

    def __init__(self, device_index: int = 0) -> None:
        super().__init__()
        self.device_index = int(device_index)

    def _open(self) -> cv2.VideoCapture:
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        return cv2.VideoCapture(self.device_index, backend)

    def info(self) -> dict:
        d = super().info()
        d["device_index"] = self.device_index
        return d


class VideoReplaySource(_ThreadedCapture):
    mode = "VIDEO_REPLAY"

    def __init__(self, path: str, loop: bool = True) -> None:
        super().__init__()
        self.path = path
        self.loop = loop

    def _open(self) -> cv2.VideoCapture:
        if not os.path.isfile(self.path):
            raise SourceError(f"video file not found: {self.path}")
        return cv2.VideoCapture(self.path)

    def _is_file(self) -> bool:
        return True

    def _on_eof(self) -> bool:
        if self.loop and self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return True
        return False

    def info(self) -> dict:
        d = super().info()
        d["path"] = os.path.basename(self.path)
        d["loop"] = self.loop
        return d
