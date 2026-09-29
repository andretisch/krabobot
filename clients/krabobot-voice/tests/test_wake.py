"""Unit tests for wake phrase matching and decode-window advance (no mic)."""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.app import Trigger, _wait_for_wake_asr  # noqa: E402
from krabobot_voice.wake import (  # noqa: E402
    decide_after_wake_decode,
    drop_ring_samples,
    matches_wake_phrase,
    normalize_text,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Эй, Арнольд", True),
        ("эй арнольд", True),
        ("Ну эй арнольд как дела", True),
        ("Hey Arnold", True),
        ("hey, arnold!", True),
        ("эй арнолд", True),  # ASR glitch
        ("ей арнольд", True),
        ("эйарнольд", True),
        # Log spam from user — must wake
        ("Привет, Арнольд.", True),
        ("Привет, Арнольд", True),
        ("привет арнольд", True),
        ("Привет, Арнольд!", True),
        ("Привет", False),
        ("Арнольд.", False),  # name alone, no greeting
        ("эй александр", False),
        ("эй бот", False),
        ("", False),
    ],
)
def test_matches_wake_phrase(text: str, expected: bool) -> None:
    assert matches_wake_phrase(text) is expected


def test_custom_phrases() -> None:
    assert matches_wake_phrase("ок арнольд", phrases=["ок арнольд"])
    assert not matches_wake_phrase("привет арнольд", phrases=["ок арнольд"], greetings=[])


def test_normalize_yo() -> None:
    assert "е" in normalize_text("Ёжик")
    assert normalize_text("Эй, Арнольд!") == "эй арнольд"


def test_decide_match_consumes_and_prints_once() -> None:
    d1 = decide_after_wake_decode(
        "Привет, Арнольд.",
        last_printed="",
        window_n=32000,
        sample_rate=16000,
    )
    assert d1.matched is True
    assert d1.print_text == "Привет, Арнольд."
    assert d1.consume_samples >= 32000
    assert d1.cooldown_samples > 0

    d2 = decide_after_wake_decode(
        "Привет, Арнольд.",
        last_printed="Привет, Арнольд.",
        window_n=32000,
        sample_rate=16000,
    )
    assert d2.matched is True
    assert d2.print_text is None  # debounced


def test_decide_miss_still_consumes() -> None:
    d = decide_after_wake_decode(
        "шум в комнате",
        last_printed="",
        window_n=32000,
        sample_rate=16000,
        cooldown_s=0.8,
    )
    assert d.matched is False
    assert d.print_text == "шум в комнате"
    assert d.consume_samples >= 32000
    assert d.cooldown_samples == int(0.8 * 16000)


def test_drop_ring_samples_advances_past_window() -> None:
    sr = 16000
    block = 800
    ring: deque[np.ndarray] = deque()
    total = 0
    for _ in range(50):
        chunk = np.ones(block, dtype=np.int16)
        ring.append(chunk)
        total += chunk.size
    before = total
    total = drop_ring_samples(ring, total, window_n := 2 * sr)
    assert total == before - window_n
    assert sum(c.size for c in ring) == total


class _FakeMic:
    """Loud PCM then silence — enough for one wake window, then idle."""

    def __init__(self, *, sample_rate: int = 16000, block: int = 800, loud_blocks: int = 50):
        self.sample_rate = sample_rate
        self.block = block
        self._i = 0
        self._loud_blocks = loud_blocks

    def read_block(self) -> np.ndarray:
        self._i += 1
        if self._i <= self._loud_blocks:
            return (np.ones(self.block, dtype=np.int16) * 12000)
        return np.zeros(self.block, dtype=np.int16)


class _FakeAsr:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0
        self.texts_seen: list[str] = []

    def transcribe_pcm16(self, pcm: np.ndarray, sample_rate: int = 16000) -> str:
        self.calls += 1
        self.texts_seen.append(self.text)
        return self.text


def test_wait_for_wake_asr_transitions_once_on_privet_arnold() -> None:
    """Dry-run: mock ASR returns log line; must wake once, not N times."""
    mic = _FakeMic(loud_blocks=80)
    asr = _FakeAsr("Привет, Арнольд.")
    trigger, detail = _wait_for_wake_asr(
        mic,  # type: ignore[arg-type]
        asr,
        window_s=2.0,
        hop_s=0.5,
        energy_threshold=0.012,
        ptt=None,
        meeting=None,
        cooldown_s=0.8,
    )
    assert trigger is Trigger.WAKE
    assert "Арнольд" in detail
    # Match on first successful decode — must not rescore the same utterance forever.
    assert asr.calls == 1


def _run_wake_miss_loop(*, consume: bool, loud_blocks: int = 40) -> tuple[int, int]:
    """Drive fake mic; return (asr_calls, print_count)."""
    from krabobot_voice.vad import frame_rms

    mic = _FakeMic(loud_blocks=loud_blocks)
    asr = _FakeAsr("шум в комнате")
    sr = mic.sample_rate
    window_n = int(2.0 * sr)
    hop_n = int(0.5 * sr)
    ring: deque[np.ndarray] = deque()
    total = 0
    since_hop = 0
    cooldown = 0
    last_print = ""
    prints = 0

    for _ in range(200):
        chunk = mic.read_block()
        ring.append(chunk)
        total += chunk.size
        since_hop += chunk.size
        if cooldown > 0:
            cooldown = max(0, cooldown - chunk.size)
        while total > window_n + mic.block * 2:
            dropped = ring.popleft()
            total -= dropped.size
        if cooldown > 0 or since_hop < hop_n or total < int(0.6 * sr):
            continue
        since_hop = 0
        pcm = np.concatenate(list(ring))[-window_n:]
        if frame_rms(pcm) < 0.012:
            continue
        text = asr.transcribe_pcm16(pcm, sample_rate=sr)
        decision = decide_after_wake_decode(
            text,
            last_printed=last_print,
            window_n=window_n,
            sample_rate=sr,
            cooldown_s=0.8,
        )
        if decision.print_text is not None:
            prints += 1
            last_print = decision.print_text
        if consume:
            total = drop_ring_samples(ring, total, decision.consume_samples)
            cooldown = decision.cooldown_samples
            since_hop = 0
    return asr.calls, prints


def test_wait_for_wake_asr_miss_does_not_rescore_forever() -> None:
    """Same loud utterance: without consume → many ASC rescored; with fix → few + 1 print."""
    buggy_calls, buggy_prints = _run_wake_miss_loop(consume=False)
    fixed_calls, fixed_prints = _run_wake_miss_loop(consume=True)

    assert buggy_calls >= 5, f"baseline spam expected, got {buggy_calls}"
    assert buggy_prints >= 1
    assert fixed_prints == 1
    assert fixed_calls < buggy_calls
    assert fixed_calls <= 3
