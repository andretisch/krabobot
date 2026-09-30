"""Meeting-start cue: three beeps only after capture is actually open."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.app import _Driver, cue_meeting_recording_started  # noqa: E402
from krabobot_voice.config import VoiceClientConfig  # noqa: E402
from krabobot_voice.dialog import VoiceSession, VoiceSessionConfig  # noqa: E402


def _driver() -> _Driver:
    return _Driver(
        VoiceClientConfig(),
        object(),  # type: ignore[arg-type]
        VoiceSession(VoiceSessionConfig()),
        asr=None,
        kws=None,
        ptt=None,
        meeting_hk=None,
        wake_energy=0.02,
    )


def test_cue_plays_three_distinct_beeps() -> None:
    seen: list[int] = []
    thread = cue_meeting_recording_started(lambda n: seen.append(n), background=False)
    assert thread is None
    assert seen == [3]


def test_cue_background_does_not_block_caller() -> None:
    seen: list[int] = []
    thread = cue_meeting_recording_started(lambda n: seen.append(n), background=True)
    assert thread is not None
    assert thread.daemon
    assert thread.name == "meeting-start-beep"
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert seen == [3]


def test_start_meeting_beeps_only_after_ready(monkeypatch, tmp_path: Path) -> None:
    cues: list[str] = []
    monkeypatch.setattr(
        "krabobot_voice.app.cue_meeting_recording_started",
        lambda: cues.append("ready"),
    )

    class _Ready:
        warning = ""
        use_process = False

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def start(self) -> Path:
            return tmp_path / "meeting.wav"

    monkeypatch.setattr("krabobot_voice.app.MeetingRecorder", _Ready)
    driver = _driver()
    driver._start_meeting()
    assert cues == ["ready"]
    assert driver._recorder is not None


def test_start_meeting_no_beep_when_capture_fails(monkeypatch) -> None:
    cues: list[str] = []
    monkeypatch.setattr(
        "krabobot_voice.app.cue_meeting_recording_started",
        lambda: cues.append("ready"),
    )

    class _Boom:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def start(self) -> Path:
            raise RuntimeError("device busy")

    monkeypatch.setattr("krabobot_voice.app.MeetingRecorder", _Boom)
    driver = _driver()
    driver._start_meeting()
    assert cues == []
    assert driver._recorder is None
