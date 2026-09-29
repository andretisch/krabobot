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
    """With a far deadline, a started utterance still finishes on silence."""
    speech = [_tone() for _ in range(20)]
    trail = [_silence() for _ in range(12)]
    src = _ScriptedSource(speech + trail)
    seg = _seg(src)
    state = {"started_reads": 0}

    def poll() -> None:
        state["started_reads"] += 1

    out = seg.next_segment(time.monotonic() + 5.0, poll=poll)
    assert out is not None
    assert state["started_reads"] > 0


def test_hard_deadline_aborts_after_speech_start() -> None:
    """Sticky continuous speech past listen deadline → None (no max_s hang)."""
    src = _ScriptedSource([_tone() for _ in range(800)])
    seg = _seg(src, max_s=60.0, silence_end_s=5.0, min_speech_s=0.12)

    def poll() -> None:
        # Pace the loop so wall-clock deadline can fire mid-utterance.
        time.sleep(0.025)

    deadline = time.monotonic() + 0.12
    t0 = time.monotonic()
    out = seg.next_segment(deadline, poll=poll)
    assert out is None
    assert time.monotonic() - t0 < 1.5
    assert not seg._started

def test_configure_updates_energy() -> None:
    src = _ScriptedSource([])
    seg = _seg(src, energy_threshold=0.05)
    seg.configure(energy_threshold=0.002, max_s=3.0, min_speech_s=0.3)
    assert seg.energy_threshold == 0.002
    assert seg.max_s == 3.0
    assert seg.min_speech_s == 0.3


def test_early_check_closes_without_silence() -> None:
    """Continuous energy (music-like): early_check match ends before max_s / silence."""
    # Long continuous tone — no silence trail; max_s is far away.
    src = _ScriptedSource([_tone() for _ in range(400)])
    seg = _seg(src, max_s=12.0, silence_end_s=2.0, min_speech_s=0.12)
    probes: list[int] = []

    def early(pcm: np.ndarray) -> bool:
        probes.append(int(pcm.size))
        # Accept once we have ~0.8s of audio (enough for a short command).
        return pcm.size >= int(0.8 * SR)

    t0 = time.monotonic()
    out = seg.next_segment(
        time.monotonic() + 20.0,
        early_check=early,
        early_check_interval_s=0.15,
        early_check_min_s=0.4,
    )
    elapsed = time.monotonic() - t0
    assert out is not None
    assert probes, "early_check should run at least once"
    # Closed well before the 12s max (command-oriented exit).
    assert elapsed < 3.0
    assert out.size < int(4.0 * SR)


def test_early_check_false_still_ends_on_silence() -> None:
    speech = [_tone() for _ in range(20)]
    trail = [_silence() for _ in range(12)]
    src = _ScriptedSource(speech + trail)
    seg = _seg(src)
    probes = {"n": 0}

    def early(_pcm: np.ndarray) -> bool:
        probes["n"] += 1
        return False

    out = seg.next_segment(
        time.monotonic() + 5.0,
        early_check=early,
        early_check_interval_s=0.1,
        early_check_min_s=0.2,
    )
    assert out is not None
    assert probes["n"] >= 1


def test_early_check_false_hits_max_without_silence() -> None:
    """Unmatched continuous energy still ends on max_s (music fallback)."""
    src = _ScriptedSource([_tone() for _ in range(200)])
    seg = _seg(src, max_s=0.6, silence_end_s=5.0, min_speech_s=0.12)

    def early(_pcm: np.ndarray) -> bool:
        return False

    out = seg.next_segment(
        time.monotonic() + 5.0,
        early_check=early,
        early_check_interval_s=0.2,
        early_check_min_s=0.2,
    )
    assert out is not None
    assert out.size <= int(1.2 * SR)


def test_max_s_counts_from_speech_start_not_idle_wait() -> None:
    """Long silence before speech must not eat the wake/command max_s budget."""
    # ~1.5 s silence, then continuous tone (no trailing silence) → close on max_s=0.6.
    silence = [_silence() for _ in range(50)]
    speech = [_tone() for _ in range(80)]
    src = _ScriptedSource(silence + speech)
    seg = _seg(src, max_s=0.6, silence_end_s=5.0, min_speech_s=0.12, preroll_s=0.1)
    out = seg.next_segment(None)
    assert out is not None
    # Cap is ~0.6 s of utterance (+ preroll); must not be near-empty from early max.
    assert out.size >= int(0.35 * SR)
    assert out.size <= int(1.2 * SR)


def test_wake_command_max_s_default_two_seconds() -> None:
    """Default wake/meeting command window is 2.0 s."""
    from krabobot_voice.config import VoiceClientConfig

    assert VoiceClientConfig().wake_max_s == 2.0


def test_early_check_first_probe_within_0_8s_before_silence() -> None:
    """First early_check at ≤0.8s voiced audio; match closes before silence_end."""
    # Continuous speech well past 0.8s; silence_end / max_s would be much later.
    src = _ScriptedSource([_tone() for _ in range(400)])
    seg = _seg(src, max_s=2.0, silence_end_s=5.0, min_speech_s=0.12, speech_start_s=0.06)
    probes: list[float] = []

    def early(pcm: np.ndarray) -> bool:
        voiced_s = pcm.size / SR
        probes.append(voiced_s)
        # Match on first probe once we have ~0.8s of buffer.
        return voiced_s >= 0.75

    out = seg.next_segment(
        None,
        early_check=early,
        early_check_interval_s=0.6,
        early_check_min_s=0.8,
    )
    assert out is not None
    assert probes, "early_check should run at least once"
    assert probes[0] <= 1.05, f"first probe too late: {probes[0]:.2f}s"
    # Closed well before 2s max / silence (immediate on match).
    assert out.size <= int(1.3 * SR)


def test_early_check_works_with_silero_speech_gate() -> None:
    """Silero gate still allows early_check cadence (not energy-only)."""
    src = _ScriptedSource([_tone() for _ in range(400)])

    def gate(_pcm: np.ndarray) -> bool:
        return True

    seg = _seg(
        src,
        max_s=2.0,
        silence_end_s=5.0,
        min_speech_s=0.12,
        speech_start_s=0.06,
        speech_gate=gate,
        energy_pregate=0.0,
    )
    probes: list[int] = []

    def early(pcm: np.ndarray) -> bool:
        probes.append(int(pcm.size))
        return pcm.size >= int(0.8 * SR)

    out = seg.next_segment(
        None,
        early_check=early,
        early_check_interval_s=0.6,
        early_check_min_s=0.8,
    )
    assert out is not None
    assert probes, "early_check must run with Silero speech_gate"
    assert out.size < int(1.5 * SR)


def test_early_check_cadence_not_every_block() -> None:
    """Misses must not spam ASR every block — only first@min then ~interval."""
    src = _ScriptedSource([_tone() for _ in range(200)])
    seg = _seg(src, max_s=2.0, silence_end_s=5.0, min_speech_s=0.12, speech_start_s=0.06)
    probes = {"n": 0}

    def early(_pcm: np.ndarray) -> bool:
        probes["n"] += 1
        return False

    out = seg.next_segment(
        None,
        early_check=early,
        early_check_interval_s=0.6,
        early_check_min_s=0.8,
    )
    assert out is not None  # closed on max_s
    # ~2s max, first at 0.8s, then every 0.6s → about 3 probes, not dozens.
    assert 1 <= probes["n"] <= 5, f"unexpected probe count: {probes['n']}"


def test_commit_includes_preroll_pcm() -> None:
    """Leading lookbehind stays in committed PCM (wake head not clipped)."""
    # Unique marker in preroll, then speech, then silence to close.
    marker = (np.arange(BLOCK, dtype=np.int16) % 200 + 50).astype(np.int16)
    preroll = [marker] + [_silence() for _ in range(3)]
    speech = [_tone() for _ in range(20)]
    trail = [_silence() for _ in range(12)]
    src = _ScriptedSource(preroll + speech + trail)
    seg = _seg(src, preroll_s=0.15, silence_end_s=0.15, speech_start_s=0.06)
    out = seg.next_segment(time.monotonic() + 5.0)
    assert out is not None
    # Marker block must appear at the front of the committed utterance.
    assert np.array_equal(out[:BLOCK], marker)


def test_commit_silence_trim_leaves_padding() -> None:
    """Silence-end commit keeps ~commit_silence_pad_s, not a hard strip of all trail."""
    speech = [_tone() for _ in range(20)]
    # silence_end_s=0.30 → ~10 blocks; pad 0.18 → keep ~6 of them.
    trail = [_silence() for _ in range(20)]
    src = _ScriptedSource(speech + trail)
    seg = _seg(
        src,
        silence_end_s=0.30,
        speech_start_s=0.06,
        preroll_s=0.0,
        commit_silence_pad_s=0.18,
        min_speech_s=0.12,
    )
    out = seg.next_segment(time.monotonic() + 5.0)
    assert out is not None
    speech_samples = BLOCK * 20
    pad_min = int(0.12 * SR)  # allow block quantization slack
    pad_max = int(0.30 * SR)
    trailing = int(out.size) - speech_samples
    assert pad_min <= trailing <= pad_max, f"trailing silence samples={trailing}"


def test_speech_start_opens_faster_with_mock_gate() -> None:
    """Default-ish speech_start_s=0.10 opens after ~3–4 frames with Silero gate."""
    # Gate: first 2 blocks silent, then speech forever until we flip off.
    calls = {"n": 0}

    def gate(_pcm: np.ndarray) -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    # Continuous loud tones (energy_pregate off); gate controls start.
    src = _ScriptedSource([_tone() for _ in range(80)] + [_silence() for _ in range(20)])
    seg = _seg(
        src,
        speech_start_s=0.10,  # ~3–4 blocks at 30 ms
        silence_end_s=0.15,
        min_speech_s=0.12,
        preroll_s=0.1,
        speech_gate=gate,
        energy_pregate=0.0,
    )
    out = seg.next_segment(time.monotonic() + 5.0)
    assert out is not None
    # Opened after speech_start (~0.10s), not the old 0.25s (~8+ frames).
    # Gate call count at start ≈ 2 silent + start_need (~4) ≈ ≤ 8 before speech.
    assert calls["n"] >= 5
    # Committed well before old 0.25s delay would have allowed a long idle wait.
    assert out.size >= int(0.3 * SR)
