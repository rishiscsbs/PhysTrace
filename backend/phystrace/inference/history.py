"""Bounded ring-buffer history of defender-observable signals."""
from __future__ import annotations

import numpy as np


class History:
    def __init__(self, capacity: int, channels: list[str]):
        self.capacity = int(capacity)
        self.data = {c: np.full(self.capacity, np.nan) for c in channels}
        self.i = 0  # total samples written
        self.channels = channels

    def push(self, row: dict[str, float]) -> None:
        j = self.i % self.capacity
        for c, arr in self.data.items():
            v = row.get(c, np.nan)
            arr[j] = np.nan if v is None else float(v)
        self.i += 1

    def __len__(self) -> int:
        return min(self.i, self.capacity)

    def window(self, channel: str, n: int) -> np.ndarray:
        n = min(int(n), len(self))
        if n <= 0:
            return np.empty(0)
        arr = self.data[channel]
        end = self.i % self.capacity
        if self.i <= self.capacity:
            return arr[max(0, self.i - n): self.i].copy()
        idx = (np.arange(end - n, end)) % self.capacity
        return arr[idx]

    def last(self, channel: str) -> float:
        if self.i == 0:
            return float("nan")
        return float(self.data[channel][(self.i - 1) % self.capacity])
