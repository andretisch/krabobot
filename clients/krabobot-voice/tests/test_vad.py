"""Unit tests for utterance VAD / capture (no mic / no server)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.vad import (  # noqa: E402
    detect_speech_segment,
    pcm_stats,
    record_utterance,
)

SR = 16000
BLOCK = 480  # 30 ms


def _silence(n: int = BLOCK) -> np.ndarray:
    return np.zeros(n, dtype=np.int16)


def _tone(n: int = BLOCK, amp: float = 0.2, freq: float = 440.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / float(SR)
    x = amp * np.sin(2 * np.pi * freq * t)
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)


class _BlockQueue:
    def __init__(self, blocks: list[np.ndarray]) -> None:
        self._blocks = list(blocks)
        self.i = 0

    def __call__(self) -> np.ndarray:
        if self.i >= len(self._blocks):
            return _silence()
        out = self._blocks[self.i]
        self.i += 1
        return out


def test_record_rejects_brief_noise_then_silence() -> None:
    """False start after beep must not produce a near-empty upload."""
    # settle dumps first N; then brief noise (would have triggered old VAD), then silence
    blocks = [_silence() for _ in range(40)]  # settle + wait
    # ~90 ms of "speech" (3 frames) — enough for old speech_start_s=0.15 at threshold
    blocks.extend([_tone(amp=0.15) for _ in range(5)])
    blocks.extend([_silence() for _ in range(80)])

    pcm = record_utterance(
        _BlockQueue(blocks),
        sample_rate=SR,
        block=BLOCK,
        settle_s=0.3,
        speech_start_s=0.15,
        min_speech_s=1.2,
        silence_end_s=0.6,
        energy_threshold=0.01,
        no_speech_timeout_s=4.0,
        max_s=6.0,
    )
    assert pcm is None


def test_record_keeps_min_speech_before_vad_end() -> None:
    """Once speech starts, silence must not end capture before min_speech_s."""
    blocks: list[np.ndarray] = [_silence() for _ in range(20)]  # settle
    # Continuous speech for ~1.5s then silence
    speech_blocks = int(1.5 * SR / BLOCK)
    blocks.extend([_tone(amp=0.2) for _ in range(speech_blocks)])
    blocks.extend([_silence() for _ in range(60)])

    pcm = record_utterance(
        _BlockQueue(blocks),
        sample_rate=SR,
        block=BLOCK,
        settle_s=0.3,
        speech_start_s=0.2,
        min_speech_s=1.0,
        silence_end_s=0.5,
        energy_threshold=0.01,
        preroll_s=0.2,
        no_speech_timeout_s=5.0,
        max_s=8.0,
    )
    assert pcm is not None
    stats = pcm_stats(pcm, sample_rate=SR)
    assert stats.duration_ms >= 900
    assert stats.rms > 0.02


def test_pcm_stats_near_silent() -> None:
    stats = pcm_stats(_silence(SR // 10), sample_rate=SR)
    assert stats.near_silent


def test_detect_speech_segment_min_duration() -> None:
    # 200 ms tone + 1.5 s silence — too short for min_speech_s=1.0
    frame = BLOCK
    tone_n = int(0.2 * SR)
    silence_n = int(2.0 * SR)
    pcm = np.concatenate([_tone(tone_n, amp=0.25), _silence(silence_n)])
    # Pad to whole frames
    pad = (-pcm.size) % frame
    if pad:
        pcm = np.concatenate([pcm, _silence(pad)])
    out = detect_speech_segment(
        pcm,
        sample_rate=SR,
        frame_ms=30,
        energy_threshold=0.01,
        speech_start_s=0.1,
        silence_end_s=0.5,
        min_speech_s=1.0,
    )
    assert out is None


def test_record_no_speech_timeout_with_intermittent_noise() -> None:
    """Brief energy spikes must not hang past no_speech_timeout_s."""
    # Pattern: settle, then repeating short bursts that never reach min_speech.
    blocks: list[np.ndarray] = [_silence() for _ in range(15)]
    for _ in range(40):
        blocks.extend([_tone(amp=0.15) for _ in range(3)])  # ~90 ms
        blocks.extend([_silence() for _ in range(20)])  # ~600 ms silence

    pcm = record_utterance(
        _BlockQueue(blocks),
        sample_rate=SR,
        block=BLOCK,
        settle_s=0.2,
        speech_start_s=0.08,
        min_speech_s=1.2,
        silence_end_s=0.4,
        energy_threshold=0.01,
        preroll_s=0.15,
        no_speech_timeout_s=1.5,
        max_s=20.0,  # would hang here without hard no-speech deadline
    )
    assert pcm is None
    # Queue should not have been drained for the full max_s window.
    q = _BlockQueue(blocks)
    record_utterance(
        q,
        sample_rate=SR,
        block=BLOCK,
        settle_s=0.2,
        speech_start_s=0.08,
        min_speech_s=1.2,
        silence_end_s=0.4,
        energy_threshold=0.01,
        preroll_s=0.15,
        no_speech_timeout_s=1.5,
        max_s=20.0,
    )
    # ~1.5s + settle ≈ far fewer than all intermittent blocks.
    assert q.i < len(blocks)


def test_record_wall_clock_timeout_when_reads_are_slow() -> None:
    """Even if block counting is slow, wall-clock no_speech must return."""

    class _SlowSilence:
        def __init__(self) -> None:
            self.i = 0

        def __call__(self) -> np.ndarray:
            self.i += 1
            time.sleep(0.05)
            return _silence()

    t0 = time.monotonic()
    pcm = record_utterance(
        _SlowSilence(),
        sample_rate=SR,
        block=BLOCK,
        settle_s=0.0,
        speech_start_s=0.2,
        min_speech_s=1.0,
        silence_end_s=0.5,
        energy_threshold=0.01,
        preroll_s=0.1,
        no_speech_timeout_s=0.35,
        max_s=30.0,
    )
    elapsed = time.monotonic() - t0
    assert pcm is None
    assert elapsed < 2.0
