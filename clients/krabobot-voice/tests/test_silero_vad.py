"""Unit tests for Silero VAD wrapper + segmenter speech_gate (mocked, no onnx)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.segmenter import VadSegmenter  # noqa: E402
from krabobot_voice.silero_vad import SileroOnnxVad, ensure_silero_model  # noqa: E402

SR = 16000
BLOCK = 480


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
    opts: dict = dict(
        sample_rate=SR,
        block=BLOCK,
        max_s=5.0,
        silence_end_s=0.15,
        speech_start_s=0.06,
        min_speech_s=0.12,
        energy_threshold=0.01,
        preroll_s=0.1,
        energy_pregate=0.0005,
    )
    opts.update(kwargs)
    return VadSegmenter(source.read_block, **opts)  # type: ignore[arg-type]


def test_energy_backend_still_closes_on_silence() -> None:
    speech = [_tone() for _ in range(20)]
    trail = [_silence() for _ in range(12)]
    src = _ScriptedSource(speech + trail)
    seg = _seg(src, speech_gate=None)
    out = seg.next_segment(time.monotonic() + 2.0)
    assert out is not None


def test_silero_gate_rejects_music_like_tone() -> None:
    """High-energy tone that energy-VAD would treat as speech; gate says no."""
    # Continuous "music" + late deadline — must NOT start a segment.
    src = _ScriptedSource([_tone() for _ in range(80)])
    calls = {"n": 0}

    def gate(_pcm: np.ndarray) -> bool:
        calls["n"] += 1
        return False  # Silero: not speech

    seg = _seg(src, speech_gate=gate, energy_pregate=0.0)
    deadline = time.monotonic() + 0.08
    out = seg.next_segment(deadline)
    assert out is None
    assert calls["n"] > 0


def test_silero_gate_starts_and_ends_without_waiting_for_rms_silence() -> None:
    """Speech per gate, then non-speech while tone still has energy → close."""
    # All blocks are loud tones; gate flips speech→silence.
    blocks = [_tone() for _ in range(60)]
    src = _ScriptedSource(blocks)
    state = {"i": 0}

    def gate(_pcm: np.ndarray) -> bool:
        # First ~15 blocks = speech, then not speech (music continues).
        state["i"] += 1
        return state["i"] <= 15

    seg = _seg(
        src,
        speech_gate=gate,
        energy_pregate=0.0,
        speech_start_s=0.06,
        min_speech_s=0.12,
        silence_end_s=0.15,
        max_s=8.0,
    )
    out = seg.next_segment(time.monotonic() + 3.0)
    assert out is not None
    # Closed early — did not consume the whole music buffer.
    assert src.i < 55


def test_energy_pregate_skips_gate_on_silence() -> None:
    src = _ScriptedSource([_silence() for _ in range(30)])
    calls = {"n": 0}

    def gate(_pcm: np.ndarray) -> bool:
        calls["n"] += 1
        return True

    seg = _seg(src, speech_gate=gate, energy_pregate=0.001)
    out = seg.next_segment(time.monotonic() + 0.05)
    assert out is None
    assert calls["n"] == 0  # silence never reached Silero


def test_silero_push_buffers_to_512(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a real ONNX session: inject a fake _run_window."""
    vad = object.__new__(SileroOnnxVad)
    vad.sample_rate = SR
    vad.threshold = 0.5
    vad._window = 512
    vad._context_n = 64
    vad._pending = np.zeros(0, dtype=np.float32)
    vad._context = np.zeros((1, 64), dtype=np.float32)
    vad._state = np.zeros((2, 1, 128), dtype=np.float32)
    vad._last_prob = 0.0
    runs: list[int] = []

    def _fake_run(window: np.ndarray) -> float:
        runs.append(int(window.size))
        vad._last_prob = 0.9
        return 0.9

    vad._run_window = _fake_run  # type: ignore[method-assign]
    # One 480-block is not enough for a 512 window.
    assert vad.push_pcm16(_tone(480)) == 0.0
    assert runs == []
    # 480 + 480 = 960 → one full 512 window, 448 pending
    assert vad.push_pcm16(_tone(480)) == pytest.approx(0.9)
    assert runs == [512]
    assert vad.is_speech(_tone(512)) is True
    assert len(runs) >= 2


def test_ensure_silero_model_uses_existing(tmp_path: Path) -> None:
    model = tmp_path / "silero_vad.onnx"
    model.write_bytes(b"x" * 20_000)
    assert ensure_silero_model(model) == model


def test_config_vad_section(tmp_path: Path) -> None:
    from krabobot_voice.config import VoiceClientConfig

    path = tmp_path / "cfg.yaml"
    path.write_text(
        "vad:\n  backend: energy\n  threshold: 0.6\n  energy_pregate: 0.001\n",
        encoding="utf-8",
    )
    cfg = VoiceClientConfig.load(path)
    assert cfg.vad_backend == "energy"
    assert cfg.vad_threshold == 0.6
    assert cfg.vad_energy_pregate == 0.001
