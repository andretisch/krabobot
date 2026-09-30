"""Listen countdown counts silence only; VAD speech pauses and resets it."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.app import _Driver  # noqa: E402
from krabobot_voice.config import VoiceClientConfig  # noqa: E402
from krabobot_voice.dialog import SetDeadline, VoiceSession, VoiceSessionConfig  # noqa: E402
from krabobot_voice.segmenter import (  # noqa: E402
    ListenCountdown,
    VadSegmenter,
    listen_countdown_status,
)

SR = 16000
BLOCK = 480  # 30 ms
STEP = BLOCK / SR


def _tone(n: int = BLOCK, amp: int = 8000) -> np.ndarray:
    t = np.arange(n, dtype=np.float32)
    wave = np.sin(2 * np.pi * 440 * t / SR)
    return (wave * amp).astype(np.int16)


def _silence(n: int = BLOCK) -> np.ndarray:
    return np.zeros(n, dtype=np.int16)


class _Clock:
    def __init__(self) -> None:
        self.t = 1_000.0

    def __call__(self) -> float:
        return self.t


class _Scripted:
    def __init__(self, blocks: list[np.ndarray]) -> None:
        self.blocks = list(blocks)
        self.i = 0

    def read_block(self) -> np.ndarray:
        if self.i >= len(self.blocks):
            return _silence()
        block = self.blocks[self.i]
        self.i += 1
        return block


def _seg(read, **kwargs: object) -> VadSegmenter:
    opts: dict[str, object] = dict(
        sample_rate=SR,
        block=BLOCK,
        max_s=30.0,
        silence_end_s=0.15,
        speech_start_s=0.06,
        min_speech_s=0.12,
        energy_threshold=0.01,
        preroll_s=0.1,
    )
    opts.update(kwargs)
    return VadSegmenter(read, **opts)  # type: ignore[arg-type]


def _advancing(clock: _Clock, src: _Scripted):
    def read() -> np.ndarray:
        clock.t += STEP
        return src.read_block()

    return read


def test_countdown_ticks_only_during_silence() -> None:
    clock = _Clock()
    cd = ListenCountdown(10.0, clock=clock)
    assert listen_countdown_status(cd) == "listening… (10s left)"

    clock.t += 4.0
    cd.note(False)
    assert cd.speech_active is False
    assert cd.remaining() == pytest.approx(6.0)
    assert listen_countdown_status(cd) == "listening… (6s left)"
    assert cd.expired() is False

    # Speech resets the silence budget and freezes the visible countdown.
    cd.note(True)
    clock.t += 3.0
    cd.note(True)
    assert cd.speech_active is True
    assert cd.expired() is False
    assert cd.remaining() == pytest.approx(10.0)
    assert listen_countdown_status(cd) == "listening… (speaking)"
    assert "left" not in listen_countdown_status(cd)

    cd.note(False)
    clock.t += 2.0
    assert cd.speech_active is False
    assert listen_countdown_status(cd) == "listening… (8s left)"
    assert listen_countdown_status(cd, follow_up=True) == "follow-up listening… (8s left)"

    clock.t += 8.0
    assert cd.expired() is True


def test_speech_hold_ignores_original_deadline() -> None:
    clock = _Clock()
    cd = ListenCountdown(1.0, clock=clock)
    clock.t += 30.0
    cd.note(True)
    assert cd.expired() is False
    assert listen_countdown_status(cd, follow_up=True) == "follow-up listening… (speaking)"


def test_segmenter_does_not_end_listen_while_speaking() -> None:
    """Speech longer than the budget must not abort; later silence still can."""
    clock = _Clock()
    speech_n = 20  # 0.6 s > 0.2 s budget
    src = _Scripted([_tone() for _ in range(speech_n)] + [_silence() for _ in range(40)])
    cd = ListenCountdown(0.2, clock=clock)
    seg = _seg(
        _advancing(clock, src),
        silence_end_s=5.0,
        max_s=30.0,
    )
    out = seg.next_segment(None, countdown=cd)
    assert out is None
    assert src.i > speech_n
    assert src.i < speech_n + 20


def test_segmenter_silence_end_still_commits() -> None:
    clock = _Clock()
    src = _Scripted([_tone() for _ in range(20)] + [_silence() for _ in range(12)])
    cd = ListenCountdown(2.0, clock=clock)
    seg = _seg(_advancing(clock, src), silence_end_s=0.15, min_speech_s=0.12)
    out = seg.next_segment(None, countdown=cd)
    assert out is not None
    assert out.size >= BLOCK * 8


def test_segmenter_silence_before_speech_still_times_out() -> None:
    clock = _Clock()
    src = _Scripted([_silence() for _ in range(80)])
    cd = ListenCountdown(0.2, clock=clock)
    seg = _seg(_advancing(clock, src))
    assert seg.next_segment(None, countdown=cd) is None
    assert src.i < 20


def test_non_speech_gate_does_not_hold_countdown() -> None:
    """Loud energy that Silero rejects is silence for the countdown."""
    clock = _Clock()
    src = _Scripted([_tone() for _ in range(80)])
    cd = ListenCountdown(0.2, clock=clock)
    seg = _seg(
        _advancing(clock, src),
        speech_gate=lambda _pcm: False,
        energy_pregate=None,
        silence_end_s=5.0,
    )
    assert seg.next_segment(None, countdown=cd) is None
    assert src.i < 20
    assert cd.speech_active is False


def test_early_check_still_closes_during_speech_hold() -> None:
    clock = _Clock()
    src = _Scripted([_tone() for _ in range(80)])
    cd = ListenCountdown(0.1, clock=clock)
    seg = _seg(_advancing(clock, src), max_s=30.0, silence_end_s=5.0)
    probes: list[int] = []

    def early(pcm: np.ndarray) -> bool:
        probes.append(int(pcm.size))
        return pcm.size >= SR // 4

    out = seg.next_segment(
        None,
        countdown=cd,
        early_check=early,
        early_check_interval_s=0.6,
        early_check_min_s=0.4,
    )
    assert out is not None
    assert probes
    assert cd.speech_active is True


def test_set_deadline_installs_silence_countdown() -> None:
    driver = _Driver(
        VoiceClientConfig(),
        object(),  # type: ignore[arg-type]
        VoiceSession(VoiceSessionConfig()),
        asr=None,
        kws=None,
        ptt=None,
        meeting_hk=None,
        wake_energy=0.02,
    )
    driver._apply([SetDeadline(8.0)])
    assert driver._countdown is not None
    assert driver._countdown.budget_s == pytest.approx(8.0)
    driver._countdown.note(True)
    assert driver._countdown.speech_active is True
    driver._apply([SetDeadline(None)])
    assert driver._countdown is None
    assert driver._deadline is None
