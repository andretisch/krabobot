"""Meeting stop must not trigger TTS playback or a prior dialog turn."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from krabobot_voice.app import _Driver
from krabobot_voice.config import VoiceClientConfig
from krabobot_voice.dialog import Mode, Segment, VoiceSession, VoiceSessionConfig
from krabobot_voice.http_client import VoiceTurnResult
from krabobot_voice.meeting import MeetingSessionResult


def _join_meeting_upload(driver: _Driver, *, timeout: float = 2.0) -> None:
    t = driver._meeting_upload_thread
    if t is not None:
        t.join(timeout=timeout)
        assert not t.is_alive(), "meeting upload thread did not finish"


def test_finish_turn_skips_playback_on_meeting_stop(monkeypatch) -> None:
    played: list[object] = []
    monkeypatch.setattr(
        "krabobot_voice.app._play_turn_result",
        lambda r: played.append(r) or True,
    )

    session = VoiceSession(VoiceSessionConfig())
    driver = _Driver(
        MagicMock(),
        MagicMock(),
        session,
        asr=None,
        kws=None,
        ptt=None,
        meeting_hk=None,
        wake_energy=0.01,
    )
    result = VoiceTurnResult(
        status_code=200,
        audio_wav=b"fake",
        transcript="",
        reply="Стоп",
        device_id="x",
        actions=["meeting_stop"],
    )
    assert driver._finish_turn(result, mic=None) is True
    assert played == []


def test_meeting_stop_phrase_skips_dialog_turn_before_stop(monkeypatch, tmp_path: Path) -> None:
    """Stop utterance must not hit sync http.turn; only async meeting upload after STOP."""
    monkeypatch.setattr("krabobot_voice.app.play_beeps", lambda *_a, **_k: None)

    # Stale stop list (no primary phrase) — same failure mode as user config.yaml.
    session = VoiceSession(
        VoiceSessionConfig(
            wake_phrases=["эй арнольд"],
            wake_greetings=["эй"],
            cmd_meeting_stop=[
                "закончить совещание",
                "завершить запись",
                "стоп запись",
            ],
            cmd_meeting_start=["начать запись"],
            cmd_run_test=["выполни тест"],
            cmd_exit=["хватит"],
        )
    )
    session.mode = Mode.MEETING
    session._meeting_listen = True

    wav_path = tmp_path / "meeting.wav"
    wav_path.write_bytes(b"RIFF")

    dialog_turns: list[dict] = []
    meeting_turns: list[dict] = []

    class FakeHttp:
        def turn(self, **kwargs):
            if kwargs.get("async_meeting") or kwargs.get("files"):
                meeting_turns.append(kwargs)
                return VoiceTurnResult(
                    status_code=202,
                    audio_wav=None,
                    transcript="",
                    reply="",
                    device_id="t",
                    queued=True,
                )
            dialog_turns.append(kwargs)
            return VoiceTurnResult(
                status_code=200,
                audio_wav=None,
                transcript="x",
                reply="should not run",
                device_id="t",
            )

    cfg = VoiceClientConfig()
    cfg.meeting_upload_as = "file"
    cfg.talk_beeps = False
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
    driver._recorder = MagicMock()
    driver._recorder.stop.return_value = MeetingSessionResult(
        wav_path=wav_path,
        duration_s=12.0,
        capture="mix",
        wav_bytes=b"RIFF",
    )
    # Meeting loop would stash the stop clip here before applying Segment effects.
    driver._pending_pcm = np.zeros(1600, dtype=np.int16)

    driver._apply(
        session.on_event(Segment("Закончить запись совещания."))
    )

    assert dialog_turns == [], "stop must not call sync /v1/voice/turn for dialog audio"
    assert session.mode is Mode.IDLE
    assert driver._pending_pcm is None
    _join_meeting_upload(driver)
    assert len(meeting_turns) == 1
    assert meeting_turns[0].get("async_meeting") is True
    assert meeting_turns[0].get("files") == [wav_path]


def test_meeting_early_asr_stop_bypasses_quiet_rms_gate(
    monkeypatch, tmp_path: Path
) -> None:
    """early-asr hit «Закончить запись» must StopMeeting even if PCM RMS is near zero.

    Regression: meeting loop discarded quiet early-hit clips via frame_rms gate,
    cleared ``_early_asr_text`` on the next iteration, and kept recording.
    """
    monkeypatch.setattr("krabobot_voice.app.play_beeps", lambda *_a, **_k: None)

    session = VoiceSession(
        VoiceSessionConfig(
            wake_phrases=["эй арнольд", "арнольд"],
            wake_greetings=["эй"],
            cmd_meeting_stop=["закончить запись"],
            cmd_meeting_start=["начать запись"],
            cmd_run_test=["выполни тест"],
            cmd_exit=["хватит"],
        )
    )
    session.mode = Mode.MEETING

    wav_path = tmp_path / "meeting.wav"
    wav_path.write_bytes(b"RIFF")
    meeting_turns: list[dict] = []

    class FakeHttp:
        def turn(self, **kwargs):
            meeting_turns.append(kwargs)
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
    cfg.meeting_capture = "mix"
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

    stop_calls = {"n": 0}

    class FakeRecorder:
        active = True

        def mic_tap_reader(self):
            return MagicMock(
                sample_rate=16000,
                block=480,
                read_block=lambda: np.zeros(480, dtype=np.int16),
            )

        def stop(self):
            stop_calls["n"] += 1
            FakeRecorder.active = False
            return MeetingSessionResult(
                wav_path=wav_path,
                duration_s=3.0,
                capture="mix",
                wav_bytes=b"RIFF",
            )

    driver._recorder = FakeRecorder()  # type: ignore[assignment]

    probes = {"n": 0}

    class FakeSegmenter:
        def next_segment(self, *a, **k):
            probes["n"] += 1
            if probes["n"] > 1:
                raise AssertionError(
                    "meeting must stop after early-asr hit; must not keep recording"
                )
            # Near-silent clip: RMS gate would discard without the early-hit bypass.
            driver._early_asr_text = "Закончить запись."
            return np.zeros(16000, dtype=np.int16)

        def feed_backlog(self, _sink):
            return None

    monkeypatch.setattr(
        "krabobot_voice.app._make_segmenter",
        lambda *a, **k: FakeSegmenter(),
    )

    driver.run_meeting_loop()

    assert session.mode is Mode.IDLE
    assert stop_calls["n"] == 1
    assert probes["n"] == 1
    _join_meeting_upload(driver)
    assert len(meeting_turns) == 1
    assert meeting_turns[0].get("async_meeting") is True


def test_meeting_stop_upload_does_not_block_main(monkeypatch, tmp_path: Path) -> None:
    """Stop path must return to IDLE before slow HTTP upload finishes."""
    monkeypatch.setattr("krabobot_voice.app.play_beeps", lambda *_a, **_k: None)

    session = VoiceSession(
        VoiceSessionConfig(
            wake_phrases=["эй арнольд"],
            wake_greetings=["эй"],
            cmd_meeting_stop=["закончить запись"],
            cmd_meeting_start=["начать запись"],
            cmd_run_test=["выполни тест"],
            cmd_exit=["хватит"],
        )
    )
    session.mode = Mode.MEETING
    session._meeting_listen = True

    wav_path = tmp_path / "meeting.wav"
    wav_path.write_bytes(b"RIFF")

    upload_started = threading.Event()
    upload_release = threading.Event()
    meeting_turns: list[dict] = []

    class SlowHttp:
        def turn(self, **kwargs):
            meeting_turns.append(kwargs)
            upload_started.set()
            # Block until test releases — simulates multi-MB POST latency.
            assert upload_release.wait(timeout=5.0)
            return VoiceTurnResult(
                status_code=202,
                audio_wav=None,
                transcript="",
                reply="",
                device_id="t",
                queued=True,
            )

    cfg = VoiceClientConfig()
    cfg.meeting_upload_as = "file"
    cfg.talk_beeps = False
    driver = _Driver(
        cfg,
        SlowHttp(),  # type: ignore[arg-type]
        session,
        asr=None,
        kws=None,
        ptt=None,
        meeting_hk=None,
        wake_energy=0.01,
    )
    driver._recorder = MagicMock()
    driver._recorder.stop.return_value = MeetingSessionResult(
        wav_path=wav_path,
        duration_s=12.0,
        capture="mix",
        wav_bytes=b"RIFF",
    )

    t0 = time.monotonic()
    driver._apply(session.on_event(Segment("Закончить запись.")))
    elapsed = time.monotonic() - t0

    assert session.mode is Mode.IDLE
    assert elapsed < 0.5, f"stop path blocked on upload ({elapsed:.2f}s)"
    assert upload_started.wait(timeout=2.0), "background upload never started"
    assert driver._meeting_upload_thread is not None
    assert driver._meeting_upload_thread.is_alive()
    assert meeting_turns  # turn entered before release

    upload_release.set()
    _join_meeting_upload(driver)
    assert meeting_turns[0].get("async_meeting") is True
    assert meeting_turns[0].get("files") == [wav_path]


def test_meeting_early_asr_wake_plus_stop_typo(monkeypatch, tmp_path: Path) -> None:
    """«Арнольд закончить запись совещать» early hit must stop (substring + typo)."""
    monkeypatch.setattr("krabobot_voice.app.play_beeps", lambda *_a, **_k: None)

    session = VoiceSession(
        VoiceSessionConfig(
            wake_phrases=["эй арнольд", "арнольд"],
            wake_greetings=["эй"],
            cmd_meeting_stop=["закончить запись"],
            cmd_meeting_start=["начать запись"],
            cmd_run_test=["выполни тест"],
            cmd_exit=["хватит"],
        )
    )
    session.mode = Mode.MEETING

    wav_path = tmp_path / "meeting.wav"
    wav_path.write_bytes(b"RIFF")

    class FakeHttp:
        def turn(self, **kwargs):
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

    class FakeRecorder:
        active = True

        def mic_tap_reader(self):
            return MagicMock(
                sample_rate=16000,
                block=480,
                read_block=lambda: np.zeros(480, dtype=np.int16),
            )

        def stop(self):
            FakeRecorder.active = False
            return MeetingSessionResult(
                wav_path=wav_path,
                duration_s=1.0,
                capture="mix",
                wav_bytes=b"RIFF",
            )

    driver._recorder = FakeRecorder()  # type: ignore[assignment]

    class FakeSegmenter:
        def next_segment(self, *a, **k):
            driver._early_asr_text = "Арнольд закончить запись совещать."
            return np.zeros(8000, dtype=np.int16)

        def feed_backlog(self, _sink):
            return None

    monkeypatch.setattr(
        "krabobot_voice.app._make_segmenter",
        lambda *a, **k: FakeSegmenter(),
    )

    driver.run_meeting_loop()
    assert session.mode is Mode.IDLE
    _join_meeting_upload(driver)
