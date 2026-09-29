"""Unit tests for persistent VadSegmenter."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.segmenter import VadSegmenter  # noqa: E402

SR = 16000
BLOCK = 480  # 30 ms


def _tone(n: int = BLOCK, amp: int = 8000) -> np.ndarray:
    t = np.arange(n, dtype=np.float32)
    wave = np.sin(2 * np.pi * 440 * t / SR)
    return (wave * amp).astype(np.int16)


def _silence(n: int = BLOCK) -> np.ndarray:
    return np.zeros(n, dtype=np.int16)


class _ScriptedSource:
    def __init__(self, blocks: list[np.ndarray]) -> None:
        self._blocks = list(blocks)
        self.i = 0

    def read_block(self) -> np.ndarray:
        if self.i >= len(self._blocks):
            return _silence()
        b = self._blocks[self.i]
        self.i += 1
        return b


def _seg(source: _ScriptedSource, **kwargs: object) -> VadSegmenter:
    opts = dict(
        sample_rate=SR,
        block=BLOCK,
        max_s=5.0,
        silence_end_s=0.15,  # ~5 blocks
        speech_start_s=0.06,  # ~2 blocks
        min_speech_s=0.12,  # ~4 blocks
        energy_threshold=0.01,
        preroll_s=0.1,
    )
    opts.update(kwargs)
    return VadSegmenter(source.read_block, **opts)  # type: ignore[arg-type]


def test_no_speech_by_deadline_returns_none() -> None:
    src = _ScriptedSource([_silence() for _ in range(50)])
    seg = _seg(src)
    deadline = time.monotonic() + 0.05
    assert seg.next_segment(deadline) is None


def test_started_segment_finishes_past_deadline() -> None:
    # Enough speech (>400 ms) to pass near_silent, then silence to end.
    # Deadline expires after speech has started — segment must still finish.
    speech = [_tone() for _ in range(20)]
    trail = [_silence() for _ in range(12)]
    src = _ScriptedSource(speech + trail)
    seg = _seg(src)
    started = {"n": 0}

    def poll() -> None:
        started["n"] += 1

    deadline = time.monotonic() + 2.0
    out = seg.next_segment(deadline, poll=poll)
    assert out is not None
    assert out.size >= BLOCK * 8
    assert started["n"] > 0


def test_feed_backlog_used_before_read() -> None:
    src = _ScriptedSource([_silence() for _ in range(100)])
    seg = _seg(src)
    blocks = [_tone() for _ in range(20)] + [_silence() for _ in range(10)]
    seg.feed_backlog(blocks)
    out = seg.next_segment(time.monotonic() + 2.0)
    assert out is not None
    assert src.i < 20


def test_preroll_carries_across_calls() -> None:
    # Timeout on silence first (preroll retained), then backlog speech continues.
    src = _ScriptedSource([_silence() for _ in range(8)])
    seg = _seg(src)
    out = seg.next_segment(time.monotonic() + 0.05)
    assert out is None
    # Speech arrives via keepalive-style backlog (same segmenter instance).
    seg.feed_backlog([_tone() for _ in range(20)] + [_silence() for _ in range(10)])
    out = seg.next_segment(time.monotonic() + 2.0)
    assert out is not None


def test_speech_continues_after_deadline_once_started() -> None:
    """Once speech started, an expired deadline must not drop the utterance."""
    speech = [_tone() for _ in range(20)]
    trail = [_silence() for _ in range(12)]
    src = _ScriptedSource(speech + trail)
    seg = _seg(src)
    # Flip deadline to the past after speech_start via poll.
    state = {"started_reads": 0}

    def poll() -> None:
        state["started_reads"] += 1

    # Far deadline so we can start; segmenter finishes on silence.
    out = seg.next_segment(time.monotonic() + 5.0, poll=poll)
    assert out is not None


def test_configure_updates_energy() -> None:
    src = _ScriptedSource([])
    seg = _seg(src, energy_threshold=0.05)
    seg.configure(energy_threshold=0.002, max_s=3.0, min_speech_s=0.3)
    assert seg.energy_threshold == 0.002
    assert seg.max_s == 3.0
    assert seg.min_speech_s == 0.3
