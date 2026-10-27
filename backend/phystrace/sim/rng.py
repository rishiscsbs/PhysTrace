"""Seeded, independent pseudorandom streams.

Every subsystem draws from its own stream spawned from a single SeedSequence,
and every stream is consumed at a fixed rate per simulation step regardless of
plant state. This is what makes paired counterfactual runs valid: an
intervention in World B can never shift the noise sequence seen by World A.
"""
from __future__ import annotations

import numpy as np

STREAM_NAMES = (
    "process",
    "disturbance",
    "LT-101",
    "LT-102",
    "FT-101",
    "FT-102",
    "ZT-101",
    "SC-101",
    "network",
    "attack",
    "camera",
)


class NormalStream:
    """Block-buffered standard normal draws (fast scalar access)."""

    def __init__(self, seed_seq: np.random.SeedSequence, block: int = 4096):
        self._gen = np.random.Generator(np.random.PCG64(seed_seq))
        self._block = block
        self._buf = self._gen.standard_normal(block)
        self._i = 0

    def next(self) -> float:
        if self._i >= self._block:
            self._buf = self._gen.standard_normal(self._block)
            self._i = 0
        v = self._buf[self._i]
        self._i += 1
        return float(v)


class UniformStream:
    def __init__(self, seed_seq: np.random.SeedSequence, block: int = 4096):
        self._gen = np.random.Generator(np.random.PCG64(seed_seq))
        self._block = block
        self._buf = self._gen.random(block)
        self._i = 0

    def next(self) -> float:
        if self._i >= self._block:
            self._buf = self._gen.random(self._block)
            self._i = 0
        v = self._buf[self._i]
        self._i += 1
        return float(v)


class Streams:
    """Named streams: '<name>' -> normal, '<name>:u' -> uniform."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        root = np.random.SeedSequence(self.seed)
        children = root.spawn(len(STREAM_NAMES) * 2)
        self.normal: dict[str, NormalStream] = {}
        self.uniform: dict[str, UniformStream] = {}
        for i, name in enumerate(STREAM_NAMES):
            self.normal[name] = NormalStream(children[2 * i])
            self.uniform[name] = UniformStream(children[2 * i + 1])

    def n(self, name: str) -> float:
        return self.normal[name].next()

    def u(self, name: str) -> float:
        return self.uniform[name].next()

    def image_noise(self, shape: tuple[int, ...], sigma: float) -> np.ndarray:
        # Camera noise is drawn from its own generator per frame; frames are rendered on a
        # fixed schedule, so consumption is still state-independent.
        gen = self._camera_gen()
        return gen.normal(0.0, sigma, size=shape)

    def _camera_gen(self) -> np.random.Generator:
        if not hasattr(self, "_cam"):
            self._cam = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed, 77])))
        return self._cam


class WatermarkSequence:
    """Reproducible bounded pseudorandom watermark delta_u_k in [-eps, +eps].

    A random binary sequence (+1/-1) held for `hold_steps` samples, scaled by
    epsilon. Seeded with a defender-held secret seed, so the defender can
    regenerate the exact sequence and the attacker cannot predict it.
    """

    def __init__(self, seed: int, hold_steps: int):
        self._gen = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), 4242])))
        self.hold = max(1, int(hold_steps))
        self._symbols: list[float] = []

    def symbol(self, k: int) -> float:
        idx = k // self.hold
        while len(self._symbols) <= idx:
            block = self._gen.integers(0, 2, size=256) * 2 - 1
            self._symbols.extend(float(b) for b in block)
        return self._symbols[idx]
