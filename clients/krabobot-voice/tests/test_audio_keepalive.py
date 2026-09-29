"""Tests for stream keepalive + interruptible read helpers (no real devices)."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.audio_io import with_stream_keepalive  # noqa: E402


class _CountingMic:
    def __init__(self) -> None:
        self.drains = 0
        self.lock = threading.Lock()
        self.sample_rate = 16000
        self.block = 480

    def drain_available(self) -> None:
        with self.lock:
            self.drains += 1
        time.sleep(0.01)

    def read_block(self) -> np.ndarray:
        self.drain_available()
        return np.zeros(self.block, dtype=np.int16)


def test_with_stream_keepalive_drains_during_slow_work() -> None:
    mic = _CountingMic()

    def _slow() -> str:
        time.sleep(0.12)
        return "ok"

    out = with_stream_keepalive(mic, _slow)
    assert out == "ok"
    assert mic.drains >= 3


def test_with_stream_keepalive_stops_after_fn() -> None:
    mic = _CountingMic()
    with_stream_keepalive(mic, lambda: time.sleep(0.05))
    after = mic.drains
    time.sleep(0.08)
    assert mic.drains == after
