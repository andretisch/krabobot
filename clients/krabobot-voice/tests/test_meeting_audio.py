"""Unit tests for meeting mix / resample helpers (no devices)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.audio_io import (  # noqa: E402
    downmix_to_mono_float,
    float_to_pcm16,
    mix_pcm16_average,
    pcm16_to_wav_bytes,
    resample_mono,
)
from krabobot_voice.config import VoiceClientConfig  # noqa: E402
from krabobot_voice.meeting import MeetingCaptureConfig, MeetingRecorder  # noqa: E402


def test_resample_mono_doubles_length() -> None:
    x = np.linspace(-0.5, 0.5, 100, dtype=np.float32)
    y = resample_mono(x, 8000, 16000)
    assert y.dtype == np.float32
    assert abs(y.size - 200) <= 1


def test_resample_mono_same_rate_noop_size() -> None:
    x = np.arange(50, dtype=np.float32)
    y = resample_mono(x, 16000, 16000)
    assert y.size == 50


def test_downmix_stereo_int16() -> None:
    left = np.full(8, 1000, dtype=np.int16)
    right = np.full(8, 3000, dtype=np.int16)
    stereo = np.column_stack([left, right])
    mono = downmix_to_mono_float(stereo)
    assert mono.shape == (8,)
    # ~2000/32768
    assert abs(mono.mean() - (2000 / 32768.0)) < 1e-4


def test_mix_pcm16_average_clip_protect() -> None:
    a = np.full(4, 20000, dtype=np.int16)
    b = np.full(4, 20000, dtype=np.int16)
    mixed = mix_pcm16_average(a, b)
    assert mixed.dtype == np.int16
    assert mixed.tolist() == [20000, 20000, 20000, 20000]


def test_mix_pcm16_truncates_to_shorter() -> None:
    a = np.arange(10, dtype=np.int16)
    b = np.arange(4, dtype=np.int16)
    mixed = mix_pcm16_average(a, b)
    assert mixed.size == 4
    assert mixed.tolist() == [0, 1, 2, 3]


def test_float_to_pcm16_clips() -> None:
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    pcm = float_to_pcm16(x)
    assert pcm[0] == -32768
    assert pcm[-1] == 32767


def test_pcm16_to_wav_bytes_header() -> None:
    pcm = np.zeros(1600, dtype=np.int16)
    wav = pcm16_to_wav_bytes(pcm, sample_rate=16000)
    assert wav[:4] == b"RIFF"
    assert b"WAVE" in wav[:16]


def test_meeting_capture_default_is_mix() -> None:
    cfg = VoiceClientConfig()
    assert cfg.meeting_capture == "mix"
    assert cfg.meeting_enabled is True


def test_config_loads_meeting_section(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "\n".join(
            [
                "base_url: http://127.0.0.1:8900",
                "device_id: t1",
                "meeting:",
                "  capture: loopback",
                "  hotkey: ctrl+shift+m",
                "  loopback_device: Headset",
                "audio:",
                "  output_device: Speakers",
            ]
        ),
        encoding="utf-8",
    )
    cfg = VoiceClientConfig.load(path)
    assert cfg.meeting_capture == "loopback"
    assert cfg.meeting_hotkey == "ctrl+shift+m"
    assert cfg.meeting_loopback_device == "Headset"
    assert cfg.audio_output_device == "Speakers"


def test_config_rejects_bad_capture_to_mix(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("meeting:\n  capture: stereo\n", encoding="utf-8")
    cfg = VoiceClientConfig.load(path)
    assert cfg.meeting_capture == "mix"


def test_meeting_recorder_mic_with_mocked_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = [np.full(480, i, dtype=np.int16) for i in range(5)]

    class FakeMic:
        def __init__(self, *a, **k):
            self._i = 0
            self.sample_rate = 16000
            self.block = 480

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def read_block(self):
            if self._i >= len(blocks):
                return np.zeros(480, dtype=np.int16)
            out = blocks[self._i]
            self._i += 1
            return out

    monkeypatch.setattr("krabobot_voice.meeting.MicStream", FakeMic)
    monkeypatch.setattr("krabobot_voice.meeting.require_sounddevice", lambda: None)

    rec = MeetingRecorder(MeetingCaptureConfig(capture="mic", sample_rate=16000, max_s=10))
    rec.start()
    # Let a few blocks land
    import time

    time.sleep(0.15)
    result = rec.stop()
    assert result.capture == "mic"
    assert result.frames >= 480
    assert result.wav_bytes[:4] == b"RIFF"
