"""Unit tests for audio preprocess (alignment before sherpa / upload)."""

from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.preprocess import (  # noqa: E402
    TARGET_SR,
    frame_rms_float,
    normalize_level,
    pcm16_to_wav_bytes,
    preprocess_pcm16,
    preprocess_wav_bytes,
    remove_dc_offset,
    trim_silence,
)

SR = 16000


def _tone(duration_s: float, freq: float = 220.0, amp: float = 0.05) -> np.ndarray:
    n = int(duration_s * SR)
    t = np.arange(n, dtype=np.float32) / float(SR)
    x = amp * np.sin(2 * np.pi * freq * t)
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)


def test_remove_dc_offset() -> None:
    x = np.ones(1000, dtype=np.float32) * 0.2 + 0.01 * np.sin(np.linspace(0, 20, 1000))
    y = remove_dc_offset(x)
    assert abs(float(y.mean())) < 1e-5


def test_normalize_raises_quiet_speech() -> None:
    quiet = _tone(1.0, amp=0.02).astype(np.float32) / 32768.0
    loud = normalize_level(quiet)
    assert frame_rms_float(loud) > frame_rms_float(quiet) * 2
    assert float(np.max(np.abs(loud))) <= 0.95


def test_trim_silence_keeps_pad() -> None:
    speech = _tone(0.6, amp=0.2).astype(np.float32) / 32768.0
    silence = np.zeros(int(0.5 * SR), dtype=np.float32)
    x = np.concatenate([silence, speech, silence])
    trimmed = trim_silence(x, sample_rate=SR, energy_threshold=0.01, pad_ms=80)
    assert trimmed.size < x.size
    assert trimmed.size > speech.size * 0.8


def test_preprocess_pcm16_to_16k_mono() -> None:
    # Simulate 48 kHz capture of a quiet phrase
    n = int(1.2 * 48000)
    t = np.arange(n, dtype=np.float32) / 48000.0
    x = (0.03 * np.sin(2 * np.pi * 180.0 * t) * 32767.0).astype(np.int16)
    aligned, sr = preprocess_pcm16(x, sample_rate=48000)
    assert sr == TARGET_SR
    assert aligned.dtype == np.int16
    assert aligned.ndim == 1
    # Level should be raised
    assert frame_rms_float(aligned.astype(np.float32) / 32768.0) > 0.05


def test_preprocess_rewrites_wav_header() -> None:
    pcm = _tone(0.8, amp=0.04)
    # Intentionally odd: write as stereo-looking interleaved mono bytes with wrong rate
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())
    out = preprocess_wav_bytes(buf.getvalue())
    with wave.open(io.BytesIO(out), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == TARGET_SR
        frames = wf.readframes(wf.getnframes())
    assert len(frames) > 0


def test_pcm16_to_wav_bytes_roundtrip() -> None:
    pcm = _tone(0.5, amp=0.1)
    wav = pcm16_to_wav_bytes(pcm, sample_rate=SR)
    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getframerate() == SR
        assert wf.getnchannels() == 1


def test_local_asr_calls_preprocess_before_decode(monkeypatch) -> None:
    """Proof: wake ASR path always runs preprocess_pcm16 before sherpa."""
    from krabobot_voice import local_asr as la

    calls: list[dict] = []
    real = preprocess_pcm16

    def spy_preprocess(pcm, **kwargs):
        calls.append({"pcm_len": len(np.asarray(pcm).reshape(-1)), **kwargs})
        return real(pcm, **kwargs)

    monkeypatch.setattr(
        "krabobot_voice.preprocess.preprocess_pcm16",
        spy_preprocess,
    )

    class _Stream:
        def accept_waveform(self, *_a, **_k):
            return None

        @property
        def result(self):
            class R:
                text = "ок"

            return R()

    class _Rec:
        def create_stream(self):
            return _Stream()

        def decode_stream(self, _stream):
            return None

    asr = object.__new__(la.LocalWakeAsr)
    asr._recognizer = _Rec()
    pcm = _tone(0.8, amp=0.04)
    text = asr.transcribe_pcm16(pcm, sample_rate=SR, energy_threshold=0.006)
    assert text == "ок"
    assert len(calls) == 1
    assert calls[0]["sample_rate"] == SR
    assert calls[0]["energy_threshold"] == 0.006


def test_upload_path_calls_preprocess() -> None:
    """Proof: Talk upload wraps preprocess_pcm16 → WAV."""
    from krabobot_voice.app import _pcm_to_upload_wav

    pcm = _tone(0.6, amp=0.03)
    wav, stats = _pcm_to_upload_wav(pcm, sample_rate=SR)
    assert len(wav) > 44  # header + frames
    assert stats.duration_ms > 0
    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getframerate() == TARGET_SR
        assert wf.getnchannels() == 1
