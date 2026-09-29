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

from krabobot_voice.dialog import Beep, Mode, Segment, VoiceSession, VoiceSessionConfig  # noqa: E402
from krabobot_voice.segmenter import VadSegmenter  # noqa: E402
from krabobot_voice.wake import matches_wake_phrase  # noqa: E402


class _FakeMic:
    def __init__(self, *, sample_rate: int = 16000, block: int = 800, loud_blocks: int = 40):
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

    def transcribe_pcm16(self, pcm: np.ndarray, sample_rate: int = 16000, **_: object) -> str:
        self.calls += 1
        return self.text


def main() -> int:
    log_strings = [
        "Привет, Арнольд.",
        "Привет, Арнольд",
        "Привет",
        "Привет, Арнольд!",
        "Арнольд.",
        "Давай скажем эй арнольд",
    ]
    print("Matcher vs log strings:")
    for s in log_strings:
        print(f"  {s!r} -> {matches_wake_phrase(s)}")

    assert matches_wake_phrase("Привет, Арнольд.") is True
    assert matches_wake_phrase("Привет, Арнольд!") is True
    assert matches_wake_phrase("Давай скажем эй арнольд") is True
    assert matches_wake_phrase("Привет") is False
    assert matches_wake_phrase("Арнольд.") is False

    mic = _FakeMic()
    asr = _FakeAsr("Привет, Арнольд.")
    seg = VadSegmenter(
        mic.read_block,
        sample_rate=mic.sample_rate,
        block=mic.block,
        energy_threshold=0.012,
        silence_end_s=0.4,
        max_s=4.0,
        min_speech_s=0.35,
        speech_start_s=0.15,
        preroll_s=0.2,
    )
    session = VoiceSession(
        VoiceSessionConfig(
            wake_phrases=["Эй, Арнольд", "привет арнольд"],
            wake_greetings=["эй", "привет", "hey"],
        )
    )
    pcm = seg.next_segment(None)
    assert pcm is not None
    text = asr.transcribe_pcm16(pcm, sample_rate=mic.sample_rate)
    effects = session.on_event(Segment(text))
    print(
        f"\nFake PCM + mock ASR: mode={session.mode.value!r} "
        f"detail={text!r} asr_calls={asr.calls} effects={effects!r}"
    )
    if session.mode is not Mode.LISTEN or asr.calls != 1:
        print("FAIL: expected LISTEN after single wake segment")
        return 1
    if not any(isinstance(e, Beep) and e.count == 2 for e in effects):
        print("FAIL: expected Beep(2)")
        return 1
    print("OK: VAD segment -> ASR once -> wake (2 beeps, LISTEN)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
