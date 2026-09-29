"""Dry-run: mock ASR with log strings → wake once (no mic).

Run from clients/krabobot-voice:
  python scripts/dry_run_wake.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.app import Trigger, _wait_for_wake_asr  # noqa: E402
from krabobot_voice.wake import matches_wake_phrase  # noqa: E402


class _FakeMic:
    def __init__(self, *, sample_rate: int = 16000, block: int = 800, loud_blocks: int = 80):
        self.sample_rate = sample_rate
        self.block = block
        self._i = 0
        self._loud_blocks = loud_blocks

    def read_block(self) -> np.ndarray:
        self._i += 1
        if self._i <= self._loud_blocks:
            return np.ones(self.block, dtype=np.int16) * 12000
        return np.zeros(self.block, dtype=np.int16)


class _FakeAsr:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def transcribe_pcm16(self, pcm: np.ndarray, sample_rate: int = 16000) -> str:
        self.calls += 1
        return self.text


def main() -> int:
    log_strings = [
        "Привет, Арнольд.",
        "Привет, Арнольд",
        "Привет",
        "Привет, Арнольд!",
        "Арнольд.",
    ]
    print("Matcher vs log strings:")
    for s in log_strings:
        print(f"  {s!r} -> {matches_wake_phrase(s)}")

    assert matches_wake_phrase("Привет, Арнольд.") is True
    assert matches_wake_phrase("Привет, Арнольд!") is True
    assert matches_wake_phrase("Привет") is False
    assert matches_wake_phrase("Арнольд.") is False

    mic = _FakeMic()
    asr = _FakeAsr("Привет, Арнольд.")
    trigger, detail = _wait_for_wake_asr(
        mic,  # type: ignore[arg-type]
        asr,
        window_s=2.0,
        hop_s=0.5,
        energy_threshold=0.012,
        ptt=None,
        meeting=None,
    )
    print(f"\nFake PCM + mock ASR: trigger={trigger.value!r} detail={detail!r} asr_calls={asr.calls}")
    if trigger is not Trigger.WAKE or asr.calls != 1:
        print("FAIL: expected single WAKE transition")
        return 1
    print("OK: transitioned to wake/listening path once")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
