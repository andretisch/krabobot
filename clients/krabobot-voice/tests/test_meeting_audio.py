"""Unit tests for meeting mix / resample helpers (no devices)."""

from __future__ import annotations

import sys
import threading
import time
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


def test_meeting_recorder_mic_with_mocked_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    rec = MeetingRecorder(
        MeetingCaptureConfig(capture="mic", sample_rate=16000, max_s=10),
        save_dir=tmp_path,
        use_process=False,
    )
    wav_path = rec.start()
    assert wav_path.parent == tmp_path
    assert wav_path.suffix == ".wav"
    import time

    time.sleep(0.15)
    result = rec.stop()
    assert result.capture == "mic"
    assert result.frames >= 480
    assert result.wav_path.is_file()
    assert result.wav_path.read_bytes()[:4] == b"RIFF"


def test_new_meeting_wav_path_creates_dir(tmp_path: Path) -> None:
    from krabobot_voice.meeting import new_meeting_wav_path

    root = tmp_path / "meetings"
    path = new_meeting_wav_path(root)
    assert path.parent == root
    assert root.is_dir()
    assert path.name.endswith(".wav")


def test_config_meeting_save_and_upload_as(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "\n".join(
            [
                "meeting:",
                "  save_dir: D:/meetings",
                "  upload_as: file",
            ]
        ),
        encoding="utf-8",
    )
    cfg = VoiceClientConfig.load(path)
    assert cfg.meeting_save_dir == "D:/meetings"
    assert cfg.meeting_upload_as == "file"
    assert "закончить запись совещания" in cfg.cmd_meeting_stop


def test_meeting_upload_uses_files_not_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Smoke: meeting upload posts files=, not audio=."""
    from krabobot_voice.app import _Driver
    from krabobot_voice.dialog import VoiceSession, VoiceSessionConfig
    from krabobot_voice.http_client import VoiceTurnResult

    wav = tmp_path / "20260101-120000.wav"
    wav.write_bytes(pcm16_to_wav_bytes(np.zeros(1600, dtype=np.int16), sample_rate=16000))

    calls: list[dict] = []

    class FakeHttp:
        def turn(self, **kwargs):
            calls.append(kwargs)
            return VoiceTurnResult(
                status_code=200,
                audio_wav=None,
                transcript="",
                reply="ok",
                device_id="t",
            )

    cfg = VoiceClientConfig()
    cfg.meeting_upload_as = "file"
    cfg.meeting_instruct = "summarize meeting"
    session = VoiceSession(VoiceSessionConfig())
    driver = _Driver(
        cfg,
        FakeHttp(),  # type: ignore[arg-type]
        session,
        asr=None,
        kws=None,
        ptt=None,
        meeting_hk=None,
        wake_energy=0.01,
    )
    driver._handle_meeting_upload(wav, duration_s=1.0, capture="mix")
    t = driver._meeting_upload_thread
    assert t is not None
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert len(calls) == 1
    assert calls[0].get("files") == [wav]
    assert calls[0].get("audio_bytes") is None
    assert calls[0].get("audio_path") is None
    assert calls[0].get("instruct") == "summarize meeting"
    assert calls[0].get("async_meeting") is True


def test_meeting_worker_start_stop_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker process entry: mocked capture writes WAV then stops via flag file."""
    from krabobot_voice import meeting_worker

    out = tmp_path / "out.wav"
    stop = tmp_path / "stop.flag"
    status = tmp_path / "status.json"
    blocks = [np.full(480, 100, dtype=np.int16) for _ in range(3)]

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
                stop.write_text("1", encoding="utf-8")
                return np.zeros(480, dtype=np.int16)
            out_block = blocks[self._i]
            self._i += 1
            return out_block

    monkeypatch.setattr("krabobot_voice.meeting.MicStream", FakeMic)
    monkeypatch.setattr("krabobot_voice.meeting.require_sounddevice", lambda: None)

    rc = meeting_worker.main(
        [
            "--out",
            str(out),
            "--stop-file",
            str(stop),
            "--status-file",
            str(status),
            "--capture",
            "mic",
            "--max-s",
            "5",
        ]
    )
    assert rc == 0
    assert out.is_file()
    assert out.read_bytes()[:4] == b"RIFF"
    assert status.is_file()


def test_meeting_recorder_stop_survives_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl+C during stop() must not hang; returns best-effort result."""
    wav = tmp_path / "m.wav"
    # Non-trivial WAV so empty-frames fallback accepts the file.
    wav.write_bytes(pcm16_to_wav_bytes(np.zeros(1600, dtype=np.int16), sample_rate=16000))

    rec = MeetingRecorder(
        MeetingCaptureConfig(capture="mic", sample_rate=16000, max_s=10),
        save_dir=tmp_path,
        use_process=False,
    )
    rec._active = True
    rec._wav_path = wav
    rec._started_at = time.monotonic()
    rec._stop_event = threading.Event()
    rec._result = {
        "frames": 1600,
        "duration_s": 0.1,
        "capture": "mic",
        "error": "",
        "wav_path": str(wav),
    }
    thread = threading.Thread(target=lambda: None, daemon=True)
    thread.start()
    thread.join(timeout=1.0)
    rec._thread = thread

    def flaky_join(self, timeout=None):  # noqa: ANN001
        raise KeyboardInterrupt()

    monkeypatch.setattr(threading.Thread, "join", flaky_join)
    result = rec.stop()
    assert result.wav_path == wav
    assert result.frames == 1600


def test_mix_degrades_to_mic_when_loopback_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mix mode must keep recording on mic if WASAPI loopback open fails."""
    blocks = [np.full(480, 42, dtype=np.int16) for _ in range(4)]

    class FakeMic:
        def __init__(self, *a, **k):
            self._i = 0

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

    class BoomLb:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            raise RuntimeError("Не удалось открыть WASAPI loopback idx=16 [Errno -9996]")

        def __exit__(self, *exc):
            return None

    monkeypatch.setattr("krabobot_voice.meeting.MicStream", FakeMic)
    monkeypatch.setattr("krabobot_voice.meeting.LoopbackStream", BoomLb)
    monkeypatch.setattr("krabobot_voice.meeting.require_sounddevice", lambda: None)

    rec = MeetingRecorder(
        MeetingCaptureConfig(capture="mix", sample_rate=16000, max_s=10),
        save_dir=tmp_path,
        use_process=False,
    )
    wav_path = rec.start()
    assert rec.ready is True
    assert "loopback open failed" in rec.warning
    time.sleep(0.12)
    result = rec.stop()
    assert result.capture == "mic"
    assert result.frames >= 480
    assert wav_path.is_file()


def test_meeting_start_raises_when_open_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """start() must raise before 'recording…' if capture cannot open."""

    class BoomMic:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            raise RuntimeError("mic open failed")

        def __exit__(self, *exc):
            return None

    monkeypatch.setattr("krabobot_voice.meeting.MicStream", BoomMic)
    monkeypatch.setattr("krabobot_voice.meeting.require_sounddevice", lambda: None)

    rec = MeetingRecorder(
        MeetingCaptureConfig(capture="mic", sample_rate=16000, max_s=10),
        save_dir=tmp_path,
        use_process=False,
    )
    with pytest.raises(RuntimeError, match="mic open failed"):
        rec.start()
    assert rec.active is False


def test_stop_tags_open_phase_in_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Open failures surface as [open] so the app can say «не удалось начать»."""
    rec = MeetingRecorder(
        MeetingCaptureConfig(capture="mic", sample_rate=16000, max_s=10),
        save_dir=tmp_path,
        use_process=False,
    )
    rec._active = True
    rec._wav_path = tmp_path / "x.wav"
    rec._started_at = time.monotonic()
    rec._stop_event = threading.Event()
    rec._ready = False
    rec._result = {
        "frames": 0,
        "duration_s": 0.0,
        "capture": "mic",
        "error": "Не удалось открыть WASAPI loopback",
        "error_phase": "open",
        "wav_path": str(tmp_path / "x.wav"),
    }
    with pytest.raises(RuntimeError, match=r"^\[open\]"):
        rec.stop()


def test_loopback_open_tries_fallback_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid device on default rate must retry other rates before giving up."""
    from krabobot_voice import audio_io

    class FakePa:
        paInt16 = 8
        paFloat32 = 1

        def __init__(self):
            self.calls: list[dict] = []

        def open(self, **kwargs):
            self.calls.append(kwargs)
            rate = int(kwargs.get("rate") or 0)
            if rate == 48000:
                raise OSError(-9996, "Invalid device")
            return object()

    info = {
        "index": 16,
        "name": "Наушники гарнитуры (Lenovo USB Headset) [Loopback]",
        "maxInputChannels": 2,
        "defaultSampleRate": 48000.0,
        "isLoopbackDevice": True,
    }
    monkeypatch.setattr(
        audio_io,
        "pyaudio",
        type(
            "M",
            (),
            {
                "paInt16": 8,
                "paFloat32": 1,
            },
        )(),
    )
    pa = FakePa()
    stream, fmt, ch, rate = audio_io._try_open_pyaudio_loopback(pa, info)
    assert stream is not None
    assert rate == 44100
    assert any(c.get("rate") == 48000 for c in pa.calls)
    assert any(c.get("rate") == 44100 for c in pa.calls)
    assert any(c.get("frames_per_buffer") == 1024 for c in pa.calls)
    assert fmt in ("int16", "float32")
    assert ch in (1, 2)


def test_frozen_defaults_to_thread_meeting_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    from krabobot_voice import meeting as meeting_mod

    monkeypatch.setattr(
        "krabobot_voice.config.is_frozen", lambda: True
    )
    monkeypatch.delenv("KRABOBOT_VOICE_MEETING_PROCESS", raising=False)
    assert meeting_mod.default_meeting_use_process() is False


def test_env_forces_meeting_process(monkeypatch: pytest.MonkeyPatch) -> None:
    from krabobot_voice import meeting as meeting_mod

    monkeypatch.setenv("KRABOBOT_VOICE_MEETING_PROCESS", "1")
    monkeypatch.setattr("krabobot_voice.config.is_frozen", lambda: True)
    assert meeting_mod.default_meeting_use_process() is True


def test_mic_tap_ipc_during_mic_only_meeting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Degraded mix→mic-only must still push mic PCM to the parent tap queue."""

    class FakeMic:
        def __init__(self, *a, **k):
            self._i = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def read_block(self):
            # Pace like a real device so the tap queue is not flooded with
            # silence that drops the early loud frames (maxsize ring).
            time.sleep(0.01)
            self._i += 1
            return np.full(480, 8000, dtype=np.int16)

    class BoomLb:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            raise RuntimeError("Invalid sample rate")

        def __exit__(self, *exc):
            return None

    monkeypatch.setattr("krabobot_voice.meeting.MicStream", FakeMic)
    monkeypatch.setattr("krabobot_voice.meeting.LoopbackStream", BoomLb)
    monkeypatch.setattr("krabobot_voice.meeting.require_sounddevice", lambda: None)

    rec = MeetingRecorder(
        MeetingCaptureConfig(capture="mix", sample_rate=16000, block_ms=30, max_s=10),
        save_dir=tmp_path,
        use_process=False,
    )
    rec.start()
    assert "loopback open failed" in rec.warning
    tap = rec.mic_tap_reader()
    loud = 0
    for _ in range(15):
        chunk = tap.read_block()
        if float(np.max(np.abs(chunk))) > 1000:
            loud += 1
    assert loud >= 3, "parent must hear mic via IPC tap while meeting records"
    assert tap.blocks_read >= 3
    result = rec.stop()
    assert result.capture == "mic"
    assert result.frames >= 480


def test_meeting_stop_via_mic_tap_segment(monkeypatch, tmp_path: Path) -> None:
    """Voice stop during meeting: mic-tap ASR segment → StopMeeting → upload."""
    from krabobot_voice.app import _Driver
    from krabobot_voice.dialog import Mode, VoiceSession, VoiceSessionConfig
    from krabobot_voice.http_client import VoiceTurnResult
    from krabobot_voice.meeting import MeetingSessionResult

    monkeypatch.setattr("krabobot_voice.app.play_beeps", lambda *_a, **_k: None)

    session = VoiceSession(
        VoiceSessionConfig(
            wake_phrases=["эй арнольд"],
            wake_greetings=["эй"],
            cmd_meeting_stop=["закончить запись совещания", "закончить запись"],
            cmd_meeting_start=["начать запись"],
            cmd_run_test=["выполни тест"],
            cmd_exit=["хватит"],
        )
    )
    session.mode = Mode.MEETING

    wav_path = tmp_path / "meeting.wav"
    wav_path.write_bytes(pcm16_to_wav_bytes(np.zeros(1600, dtype=np.int16), sample_rate=16000))
    uploads: list[dict] = []

    class FakeHttp:
        def turn(self, **kwargs):
            uploads.append(kwargs)
            return VoiceTurnResult(
                status_code=202,
                audio_wav=None,
                transcript="",
                reply="",
                device_id="t",
                queued=True,
            )

    cfg = VoiceClientConfig()
    cfg.talk_beeps = False
    cfg.meeting_upload_as = "file"
    driver = _Driver(
        cfg,
        FakeHttp(),  # type: ignore[arg-type]
        session,
        asr=None,
        kws=None,
        ptt=None,
        meeting_hk=None,
        wake_energy=0.02,
    )

    stop_n = {"n": 0}

    class FakeRecorder:
        active = True
        use_process = False

        def mic_tap_reader(self):
            return type(
                "T",
                (),
                {
                    "sample_rate": 16000,
                    "block": 480,
                    "last_rms": 0.1,
                    "blocks_read": 10,
                    "empty_reads": 0,
                    "read_block": staticmethod(lambda: np.zeros(480, dtype=np.int16)),
                },
            )()

        def stop(self):
            stop_n["n"] += 1
            FakeRecorder.active = False
            return MeetingSessionResult(
                wav_path=wav_path,
                duration_s=2.0,
                capture="mic",
                frames=32000,
                wav_bytes=wav_path.read_bytes(),
            )

    driver._recorder = FakeRecorder()  # type: ignore[assignment]

    class FakeSegmenter:
        def next_segment(self, *a, **k):
            driver._early_asr_text = "Закончить запись совещания."
            return np.full(16000, 1000, dtype=np.int16)

        def feed_backlog(self, _sink):
            return None

    monkeypatch.setattr(
        "krabobot_voice.app._make_segmenter",
        lambda *a, **k: FakeSegmenter(),
    )

    driver.run_meeting_loop()
    assert session.mode is Mode.IDLE
    assert stop_n["n"] == 1
    t = driver._meeting_upload_thread
    if t is not None:
        t.join(timeout=2.0)
    assert len(uploads) == 1
    assert uploads[0].get("async_meeting") is True
