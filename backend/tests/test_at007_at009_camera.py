"""AT-007 no silent camera fallback; AT-008 calibrated waterline accuracy; AT-009 quality gating."""
import time

import cv2
import numpy as np
import pytest

from phystrace.camera.analysis import FrameAnalyzer
from phystrace.camera.manager import CameraManager
from phystrace.camera.render import DEFAULT_SIM_CALIBRATION, render_tank
from phystrace.camera.sources import LiveCameraSource

CAL = DEFAULT_SIM_CALIBRATION
RNG = np.random.default_rng(0)


def _frame(h, **kw):
    return render_tank({"h": h, "tank_height_m": 3.0, "pump_running": True}, RNG.normal(0, 3, (360, 240, 3)), **kw)


class _FakeCap:
    def __init__(self):
        self.ok = True

    def isOpened(self):
        return True

    def get(self, prop):
        return 30.0

    def read(self):
        return (True, _frame(1.4)) if self.ok else (False, None)

    def release(self):
        self.ok = False


def test_at007_no_source_until_explicit_selection():
    m = CameraManager(scene_provider=lambda: {"h": 1.5, "tank_height_m": 3.0, "pump_running": True})
    est = m.capture(0.0)
    assert m.mode == "NONE" and est["status"] == "UNAVAILABLE" and est["level_m"] is None


def test_at007_live_camera_open_failure_is_unavailable_not_simulated(monkeypatch):
    class Broken:
        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(LiveCameraSource, "_open", lambda self: Broken())
    m = CameraManager(scene_provider=lambda: {"h": 1.5, "tank_height_m": 3.0, "pump_running": True})
    st = m.select("LIVE_CAMERA", device_index=7)
    assert st["mode"] == "LIVE_CAMERA" and st["source"]["state"] == "UNAVAILABLE"
    est = m.capture(1.0)
    assert est["status"] == "UNAVAILABLE" and est["level_m"] is None
    assert m.mode == "LIVE_CAMERA" and not m.status()["simulated"]
    assert m.last_frame is None


def test_at007_live_camera_disconnect_transitions_to_unavailable(monkeypatch):
    monkeypatch.setattr(LiveCameraSource, "_open", lambda self: _FakeCap())
    m = CameraManager(scene_provider=lambda: {"h": 2.9, "tank_height_m": 3.0, "pump_running": True})
    m.set_calibration("LIVE_CAMERA", CAL)
    m.select("LIVE_CAMERA", device_index=0)
    time.sleep(0.3)
    est = m.capture(1.0)
    assert est["status"] == "OK" and abs(est["level_m"] - 1.4) < 0.03
    m.source.simulate_disconnect()
    deadline = time.time() + 3
    while m.source.state == "ACTIVE" and time.time() < deadline:
        time.sleep(0.05)
    est2 = m.capture(2.0)
    assert m.source.state == "UNAVAILABLE"
    assert est2["status"] == "UNAVAILABLE" and est2["level_m"] is None
    assert m.mode == "LIVE_CAMERA"  # never switched to SIMULATED_VISUAL
    m.stop()


@pytest.mark.parametrize("h", [0.3, 0.9, 1.5, 2.1, 2.7])
def test_at008_calibrated_waterline_estimate_accuracy(h):
    est = FrameAnalyzer().analyze(_frame(h), CAL, 0.0, "SIMULATED_VISUAL")
    assert est["status"] == "OK"
    assert abs(est["level_m"] - h) <= 0.02
    assert est["uncertainty_m"] > 0
    assert est["indicator"]["state"] == "ON"


def test_at008_video_replay_fixture(tmp_path):
    path = str(tmp_path / "tank.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (240, 360))
    for i in range(30):
        vw.write(_frame(1.0 + 0.01 * i))
    vw.release()
    m = CameraManager()
    m.set_calibration("VIDEO_REPLAY", CAL)
    st = m.select("VIDEO_REPLAY", video_path=path, loop=False)
    assert st["source"]["state"] == "ACTIVE"
    time.sleep(0.4)
    est = m.capture(0.0)
    assert est["status"] == "OK" and 0.98 <= est["level_m"] <= 1.33
    deadline = time.time() + 6
    while m.source.state == "ACTIVE" and time.time() < deadline:
        time.sleep(0.1)
    assert m.source.state == "ENDED"
    assert m.capture(9.0)["status"] == "UNAVAILABLE"  # end of file is reported, not papered over
    m.stop()


@pytest.mark.parametrize("kw,reason", [({"blur": 6}, "BLURRED"), ({"brightness": 0.1}, "UNDEREXPOSED"),
                                       ({"occlude": True}, "OCCLUDED_OR_INCONSISTENT")])
def test_at009_quality_failures_return_no_numeric_level(kw, reason):
    est = FrameAnalyzer().analyze(_frame(1.5, **kw), CAL, 0.0, "SIMULATED_VISUAL")
    assert est["status"] in ("LOW_QUALITY", "UNAVAILABLE")
    assert est["level_m"] is None
    assert reason in est["reasons"]


def test_at009_frozen_frame_detected():
    a = FrameAnalyzer()
    f = _frame(1.5)
    for i in range(4):
        est = a.analyze(f, CAL, float(i), "SIMULATED_VISUAL")
    assert est["status"] == "UNAVAILABLE" and "FROZEN_FRAME" in est["reasons"] and est["level_m"] is None


def test_at009_uncalibrated_returns_no_level():
    est = FrameAnalyzer().analyze(_frame(1.5), None, 0.0, "LIVE_CAMERA")
    assert est["status"] == "UNCALIBRATED" and est["level_m"] is None
