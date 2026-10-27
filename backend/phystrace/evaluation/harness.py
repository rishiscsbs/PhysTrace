"""Batch episode harness: World (truth) + Defender (observables only).

Used by the counterfactual engine, the ablation lab and the test-suite. The
harness is the only place where both halves meet, and it passes the defender
nothing but ObservableFrames, camera estimates, and the narrow diagnostic port.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..camera.analysis import FrameAnalyzer
from ..camera.render import DEFAULT_SIM_CALIBRATION, render_tank
from ..config import SimConfig
from ..inference.pipeline import Defender
from ..sim.scenarios import ScenarioSpec
from ..sim.world import ResponseDirective, World


class SimulatedCamera:
    """SIMULATED_VISUAL witness for batch runs (explicitly labelled synthetic input)."""

    def __init__(self, world: World, seed: int):
        self.world = world
        self.analyzer = FrameAnalyzer()
        self.rng = np.random.default_rng([seed, 99])
        self.calibration = dict(DEFAULT_SIM_CALIBRATION)
        self._bank = self.rng.standard_normal((4, 360, 240, 3), dtype=np.float32) * 3.0

    def __call__(self, t: float) -> dict:
        # Noise frame from a small bank with a random 2-D roll: unique per frame, cheap to draw.
        i, dy, dx = int(self.rng.integers(4)), int(self.rng.integers(360)), int(self.rng.integers(240))
        noise = np.roll(self._bank[i], (dy, dx), axis=(0, 1))
        frame = render_tank(self.world.camera_scene_truth(), noise)
        return self.analyzer.analyze(frame, self.calibration, t, "SIMULATED_VISUAL")


@dataclass
class Episode:
    cfg: SimConfig
    world: World
    defenders: dict[str, Defender]
    truth: list[dict] = field(default_factory=list)
    obs: list[dict] = field(default_factory=list)
    keep_obs: bool = False

    def step(self) -> tuple[dict, dict]:
        o, tr = self.world.step()
        for d in self.defenders.values():
            d.step(o, port=self.world)
        self.truth.append(tr)
        if self.keep_obs:
            self.obs.append(o)
        return o, tr

    def run_until(self, t_end: float) -> None:
        while self.world.t < t_end - 1e-9:
            self.step()


def make_episode(cfg: SimConfig, scenarios: list[ScenarioSpec], *, modes=("FULL",), camera: bool = True,
                 keep_obs: bool = False) -> Episode:
    world = World(cfg, scenarios)
    defenders = {}
    for m in modes:
        cam = SimulatedCamera(world, cfg.seed) if camera and m not in ("RESIDUAL_ONLY", "WATERMARK_ONLY") else None
        defenders[m] = Defender(cfg, mode=m, camera_provider=cam, emit=False)
    return Episode(cfg=cfg, world=world, defenders=defenders, keep_obs=keep_obs)


def clone_episode(ep: Episode) -> Episode:
    """Deep copy (world + defenders + camera) so a branch can diverge from an identical state."""
    import copy

    return copy.deepcopy(ep)


def directive_for(policy: str, defender: Defender, t: float) -> ResponseDirective:
    d = ResponseDirective(policy=policy, activated_at_s=t)
    if policy == "FALLBACK_TO_CAMERA_OR_TWIN_ESTIMATE":
        d.fallback_level_m = defender.fallback_level()
    if policy == "HOLD_LAST_VERIFIED_SAFE_STATE":
        cmds = defender.last_verified_commands()
        if cmds:
            d.hold_u, d.hold_valve = cmds
    return d


def refresh_directive(world: World, defender: Defender) -> None:
    """Fallback estimate must be refreshed every step from the defender's latest estimate."""
    if world.response.policy == "FALLBACK_TO_CAMERA_OR_TWIN_ESTIMATE":
        world.response.fallback_level_m = defender.fallback_level()


def summarize_truth(truth: list[dict]) -> dict[str, Any]:
    if not truth:
        return {}
    h = np.array([x["h"] for x in truth])
    return {"max_level_m": float(h.max()), "min_level_m": float(h.min()), "overflow_m3": truth[-1]["overflow_m3"]}
