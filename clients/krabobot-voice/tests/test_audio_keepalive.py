"""Tests for stream keepalive + interruptible read helpers (no real devices)."""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.audio_io import play_beeps, with_stream_keepalive  # noqa: E402


class _CountingMic:
    def __init__(self) -> None:
        self.drains = 0
        self.reads = 0
        self.lock = threading.Lock()
        self.sample_rate = 16000
        self.block = 480

    def drain_available(self) -> None:
        with self.lock:
            self.drains += 1
        time.sleep(0.01)

    def read_block(self) -> np.ndarray:
        with self.lock:
            self.reads += 1
        time.sleep(0.01)
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


def test_with_stream_keepalive_sink_buffers_blocks() -> None:
    mic = _CountingMic()
    sink: deque[np.ndarray] = deque(maxlen=50)

    def _slow() -> str:
        time.sleep(0.12)
        return "ok"

    out = with_stream_keepalive(mic, _slow, sink=sink)
    assert out == "ok"
    assert len(sink) >= 2
    assert all(isinstance(b, np.ndarray) for b in sink)
    # With sink, drain_available is not used — read_block path.
    assert mic.reads >= 2


def test_play_beeps_zero_is_noop(monkeypatch: object) -> None:
    calls: list[int] = []

    def _fake_beep(**kwargs: object) -> None:
        calls.append(1)

    import krabobot_voice.audio_io as audio_io

    monkeypatch.setattr(audio_io, "play_beep", _fake_beep)  # type: ignore[attr-defined]
    play_beeps(0)
    assert calls == []
    play_beeps(2, gap_ms=0)
    assert len(calls) == 2
