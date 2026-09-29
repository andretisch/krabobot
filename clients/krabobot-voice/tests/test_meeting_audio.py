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
    open_capture_stream,
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


def test_listen_source_default_is_mic() -> None:
    cfg = VoiceClientConfig()
    assert cfg.audio_listen_source == "mic"


def test_config_default_run_test_includes_infinitive() -> None:
    cfg = VoiceClientConfig()
    assert "выполнить тест" in cfg.cmd_run_test
    assert "выполни тест" in cfg.cmd_run_test


def test_config_loads_audio_listen_source(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "\n".join(
            [
                "audio:",
                "  listen_source: loopback",
                "  output_device: Headphones",
                "meeting:",
                "  loopback_device: Headset Loopback",
            ]
        ),
        encoding="utf-8",
    )
    cfg = VoiceClientConfig.load(path)
    assert cfg.audio_listen_source == "loopback"
    assert cfg.audio_output_device == "Headphones"
    assert cfg.meeting_loopback_device == "Headset Loopback"


def test_config_loads_wake_input_alias(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("wake:\n  input: loopback\n", encoding="utf-8")
    cfg = VoiceClientConfig.load(path)
    assert cfg.audio_listen_source == "loopback"


def test_config_audio_listen_source_wins_over_wake_input(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "audio:\n  listen_source: mic\nwake:\n  input: loopback\n",
        encoding="utf-8",
    )
    cfg = VoiceClientConfig.load(path)
    assert cfg.audio_listen_source == "mic"


def test_config_rejects_bad_listen_source_to_mic(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("audio:\n  listen_source: speakers\n", encoding="utf-8")
    cfg = VoiceClientConfig.load(path)
    assert cfg.audio_listen_source == "mic"


def test_env_listen_source_overrides_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("audio:\n  listen_source: mic\n", encoding="utf-8")
    monkeypatch.setenv("KRABOBOT_VOICE_LISTEN_SOURCE", "loopback")
    cfg = VoiceClientConfig.load(path)
    assert cfg.audio_listen_source == "loopback"


def test_env_wake_phrase_overrides_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text('wake:\n  phrase: "Эй, Арнольд"\n', encoding="utf-8")
    monkeypatch.setenv("KRABOBOT_VOICE_WAKE_PHRASE", "Ок, Бот")
    cfg = VoiceClientConfig.load(path)
    assert cfg.wake_phrases[0] == "Ок, Бот"


def test_open_capture_stream_mic_vs_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeMic:
        def __init__(self, **kwargs):
            calls.append(("mic", kwargs))

    class FakeLb:
        def __init__(self, **kwargs):
            calls.append(("loopback", kwargs))

    monkeypatch.setattr("krabobot_voice.audio_io.MicStream", FakeMic)
    monkeypatch.setattr("krabobot_voice.audio_io.LoopbackStream", FakeLb)

    mic = open_capture_stream("mic", sample_rate=16000, input_device="USB Mic")
    assert isinstance(mic, FakeMic)
    assert calls[-1][0] == "mic"
    assert calls[-1][1]["device"] == "USB Mic"

    lb = open_capture_stream(
        "loopback",
        sample_rate=16000,
        loopback_device="Headset",
        output_device="Speakers",
    )
    assert isinstance(lb, FakeLb)
    assert calls[-1][0] == "loopback"
    assert calls[-1][1]["device"] == "Headset"
    assert calls[-1][1]["output_device"] == "Speakers"
    assert calls[-1][1].get("gain", 1.0) > 1.0

    # Unknown → mic
    open_capture_stream("weird")
    assert calls[-1][0] == "mic"


def test_resolve_loopback_prefers_default_wasapi_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Device selection must use PyAudioWPatch loopback analogues (not mic)."""
    from krabobot_voice import audio_io

    class FakePa:
        def get_loopback_device_info_generator(self):
            yield {
                "index": 16,
                "name": "Speakers (USB Headset) [Loopback]",
                "maxInputChannels": 2,
                "defaultSampleRate": 48000.0,
                "isLoopbackDevice": True,
            }

        def get_default_wasapi_loopback(self):
            return {
                "index": 16,
                "name": "Speakers (USB Headset) [Loopback]",
                "maxInputChannels": 2,
                "defaultSampleRate": 48000.0,
                "isLoopbackDevice": True,
            }

        def get_wasapi_loopback_analogue_by_index(self, idx: int):
            raise LookupError(idx)

        def get_wasapi_loopback_analogue_by_dict(self, info: dict):
            raise LookupError(info)

        def get_device_info_generator_by_host_api(self, host_api_type=None):
            return iter(())

        def terminate(self):
            return None

    monkeypatch.setattr(audio_io, "pyaudio", type("M", (), {"PyAudio": FakePa, "paWASAPI": 2})())
    monkeypatch.setattr(audio_io, "require_pyaudiowpatch", lambda: None)

    info = audio_io.resolve_loopback_device_info()
    assert info["index"] == 16
    assert info["isLoopbackDevice"] is True
    assert "Loopback" in info["name"]


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
