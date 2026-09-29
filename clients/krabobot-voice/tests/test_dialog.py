"""Unit tests for VoiceSession state machine (no mic / no server)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.dialog import (  # noqa: E402
    Beep,
    Hotkey,
    Mode,
    RunTest,
    Segment,
    SendAudio,
    SendText,
    SetDeadline,
    StartMeeting,
    StopMeeting,
    Timeout,
    TurnDone,
    VoiceSession,
    VoiceSessionConfig,
)


def _cfg(**kwargs: object) -> VoiceSessionConfig:
    base = VoiceSessionConfig(
        listen_timeout_s=10.0,
        follow_up_s=10.0,
        wake_phrases=["эй арнольд", "hey arnold"],
        wake_greetings=["эй", "hey"],
        cmd_meeting_start=["начать запись", "начни запись"],
        cmd_meeting_stop=["стоп запись", "завершить запись"],
        cmd_run_test=[
            "выполни тест",
            "выполнить тест",
            "сделай тест",
            "запусти тест",
        ],
        cmd_exit=["хватит", "выход"],
        meeting_enabled=True,
    )
    for k, v in kwargs.items():
        setattr(base, k, v)
    return base


def _effects(session: VoiceSession, event: object) -> list[object]:
    return session.on_event(event)  # type: ignore[arg-type]


def test_idle_miss_stays_idle() -> None:
    s = VoiceSession(_cfg())
    effects = _effects(s, Segment("просто шум"))
    assert effects == []
    assert s.mode is Mode.IDLE


def test_wake_only_two_beeps_listen() -> None:
    s = VoiceSession(_cfg())
    effects = _effects(s, Segment("эй арнольд"))
    assert any(isinstance(e, Beep) and e.count == 2 for e in effects)
    assert any(isinstance(e, SetDeadline) and e.seconds == 10.0 for e in effects)
    assert s.mode is Mode.LISTEN


def test_wake_trailing_send_text() -> None:
    s = VoiceSession(_cfg())
    effects = _effects(s, Segment("эй арнольд который час"))
    assert any(isinstance(e, Beep) and e.count == 2 for e in effects)
    texts = [e for e in effects if isinstance(e, SendText)]
    assert len(texts) == 1
    assert "час" in texts[0].text.lower() or texts[0].text


def test_wake_trailing_local_meeting_start() -> None:
    s = VoiceSession(_cfg())
    effects = _effects(s, Segment("эй арнольд начни запись"))
    assert any(isinstance(e, Beep) and e.count == 2 for e in effects)
    assert any(isinstance(e, StartMeeting) for e in effects)
    assert s.mode is Mode.MEETING


def test_listen_timeout_one_beep_idle() -> None:
    s = VoiceSession(_cfg())
    _effects(s, Segment("эй арнольд"))
    assert s.mode is Mode.LISTEN
    effects = _effects(s, Timeout())
    assert any(isinstance(e, Beep) and e.count == 1 for e in effects)
    assert s.mode is Mode.IDLE


def test_listen_segment_send_audio() -> None:
    s = VoiceSession(_cfg())
    _effects(s, Segment("эй арнольд"))
    effects = _effects(s, Segment("какой сегодня день"))
    assert any(isinstance(e, SendAudio) for e in effects)
    assert not any(isinstance(e, Beep) for e in effects)


def test_listen_local_exit() -> None:
    s = VoiceSession(_cfg())
    _effects(s, Segment("эй арнольд"))
    effects = _effects(s, Segment("хватит"))
    assert any(isinstance(e, Beep) and e.count == 1 for e in effects)
    assert s.mode is Mode.IDLE


def test_listen_local_run_test() -> None:
    s = VoiceSession(_cfg())
    _effects(s, Segment("эй арнольд"))
    effects = _effects(s, Segment("выполни тест"))
    assert any(isinstance(e, RunTest) for e in effects)
    assert not any(isinstance(e, SendAudio) for e in effects)
    assert s.mode is Mode.DIALOG
    assert any(isinstance(e, SetDeadline) and e.seconds == 10.0 for e in effects)


def test_listen_local_run_test_infinitive() -> None:
    """ASR often returns infinitive «выполнить» instead of imperative «выполни»."""
    s = VoiceSession(_cfg())
    _effects(s, Segment("эй арнольд"))
    effects = _effects(s, Segment("Выполнить тест."))
    assert any(isinstance(e, RunTest) for e in effects)
    assert not any(isinstance(e, SendAudio) for e in effects)
    assert s.mode is Mode.DIALOG


def test_dialog_local_run_test_no_upload() -> None:
    s = VoiceSession(_cfg())
    s.mode = Mode.DIALOG
    effects = _effects(s, Segment("выполнить тест"))
    assert any(isinstance(e, RunTest) for e in effects)
    assert not any(isinstance(e, SendAudio) for e in effects)


def test_run_test_action() -> None:
    s = VoiceSession(_cfg())
    s.mode = Mode.DIALOG
    effects = _effects(s, TurnDone(ok=True, actions=("run_test",)))
    assert any(isinstance(e, RunTest) for e in effects)
    assert s.mode is Mode.DIALOG
    assert any(isinstance(e, SetDeadline) and e.seconds == 10.0 for e in effects)


def test_turn_ok_enters_dialog() -> None:
    s = VoiceSession(_cfg())
    _effects(s, Segment("эй арнольд"))
    _effects(s, Segment("привет"))
    effects = _effects(s, TurnDone(ok=True))
    assert s.mode is Mode.DIALOG
    assert any(isinstance(e, SetDeadline) and e.seconds == 10.0 for e in effects)


def test_dialog_timeout_one_beep() -> None:
    s = VoiceSession(_cfg())
    s.mode = Mode.DIALOG
    effects = _effects(s, Timeout())
    assert any(isinstance(e, Beep) and e.count == 1 for e in effects)
    assert s.mode is Mode.IDLE


def test_dialog_segment_send_audio() -> None:
    s = VoiceSession(_cfg())
    s.mode = Mode.DIALOG
    effects = _effects(s, Segment("ещё вопрос"))
    assert any(isinstance(e, SendAudio) for e in effects)


def test_empty_asr_discards_without_upload() -> None:
    """Music/noise with empty local ASR must not force a server turn."""
    s = VoiceSession(_cfg())
    _effects(s, Segment("эй арнольд"))
    assert s.mode is Mode.LISTEN
    effects = _effects(s, Segment(""))
    assert not any(isinstance(e, SendAudio) for e in effects)
    assert s.mode is Mode.LISTEN


def test_empty_asr_upload_if_empty_fallback() -> None:
    """When local ASR engine is absent, empty text still uploads PCM."""
    s = VoiceSession(_cfg())
    s.mode = Mode.DIALOG
    effects = _effects(s, Segment("", upload_if_empty=True))
    assert any(isinstance(e, SendAudio) for e in effects)


def test_end_dialog_action() -> None:
    s = VoiceSession(_cfg())
    s.mode = Mode.DIALOG
    effects = _effects(s, TurnDone(ok=True, actions=("end_dialog",)))
    assert any(isinstance(e, Beep) and e.count == 1 for e in effects)
    assert s.mode is Mode.IDLE


def test_follow_up_disabled() -> None:
    s = VoiceSession(_cfg(follow_up_s=0))
    s.mode = Mode.LISTEN
    effects = _effects(s, TurnDone(ok=True))
    assert s.mode is Mode.IDLE
    assert any(isinstance(e, Beep) and e.count == 1 for e in effects)


def test_failed_turn_idle() -> None:
    s = VoiceSession(_cfg())
    s.mode = Mode.LISTEN
    effects = _effects(s, TurnDone(ok=False))
    assert s.mode is Mode.IDLE
    assert any(isinstance(e, Beep) and e.count == 1 for e in effects)


def test_meeting_hotkey_start_stop() -> None:
    s = VoiceSession(_cfg())
    effects = _effects(s, Hotkey("meeting"))
    assert any(isinstance(e, StartMeeting) for e in effects)
    assert s.mode is Mode.MEETING
    effects = _effects(s, Hotkey("meeting"))
    assert any(isinstance(e, StopMeeting) for e in effects)
    assert any(isinstance(e, Beep) and e.count == 1 for e in effects)
    assert s.mode is Mode.IDLE


def test_meeting_local_stop_without_wake() -> None:
    s = VoiceSession(_cfg())
    s.mode = Mode.MEETING
    effects = _effects(s, Segment("стоп запись"))
    assert any(isinstance(e, StopMeeting) for e in effects)
    assert s.mode is Mode.IDLE


def test_meeting_stop_asr_truncation_without_wake() -> None:
    s = VoiceSession(_cfg())
    s.mode = Mode.MEETING
    effects = _effects(s, Segment("Кончить запись совещания."))
    assert any(isinstance(e, StopMeeting) for e in effects)
    assert s.mode is Mode.IDLE


def test_meeting_wake_plus_stop_one_utterance_custom_wake() -> None:
    """Configurable wake + stop in one ASR line must StopMeeting (no Arnold)."""
    s = VoiceSession(
        _cfg(
            wake_phrases=["ок бот"],
            wake_greetings=["ок"],
        )
    )
    s.mode = Mode.MEETING
    effects = _effects(s, Segment("Ок Бот закончить запись"))
    assert any(isinstance(e, StopMeeting) for e in effects)
    assert not any(isinstance(e, (SendAudio, SendText)) for e in effects)
    assert s.mode is Mode.IDLE


def test_meeting_wake_plus_stop_full_phrase() -> None:
    s = VoiceSession(_cfg())
    s.mode = Mode.MEETING
    effects = _effects(s, Segment("Эй, Арнольд, закончить запись совещания"))
    assert any(isinstance(e, StopMeeting) for e in effects)
    assert s.mode is Mode.IDLE


def test_meeting_stop_wins_over_armed_listen_no_send_audio() -> None:
    """After wake arms meeting-listen, stop phrase must StopMeeting — never SendAudio."""
    # Stale cfg without the advertised primary phrase (mirrors real user YAML).
    s = VoiceSession(
        _cfg(
            cmd_meeting_stop=[
                "закончить совещание",
                "завершить запись",
                "стоп запись",
            ]
        )
    )
    s.mode = Mode.MEETING
    # Wake-only arms listen for the next segment.
    wake_effects = _effects(s, Segment("эй арнольд"))
    assert s._meeting_listen is True
    assert any(isinstance(e, Beep) for e in wake_effects)
    assert not any(isinstance(e, SendAudio) for e in wake_effects)

    effects = _effects(s, Segment("Закончить запись совещания."))
    assert any(isinstance(e, StopMeeting) for e in effects)
    assert not any(isinstance(e, (SendAudio, SendText)) for e in effects)
    assert s.mode is Mode.IDLE
    assert s._meeting_listen is False


def test_meeting_server_stop_action() -> None:
    s = VoiceSession(_cfg())
    s.mode = Mode.MEETING
    effects = _effects(s, TurnDone(ok=True, actions=("meeting_stop",)))
    assert any(isinstance(e, StopMeeting) for e in effects)
    assert any(isinstance(e, Beep) and e.count == 1 for e in effects)
    assert s.mode is Mode.IDLE


def test_ptt_hotkey_arms_listen() -> None:
    s = VoiceSession(_cfg())
    effects = _effects(s, Hotkey("ptt"))
    assert s.mode is Mode.LISTEN
    assert any(isinstance(e, Beep) and e.count == 2 for e in effects)


def test_client_mode_meeting_state() -> None:
    s = VoiceSession(_cfg())
    assert s.client_mode() == "idle"
    assert s.meeting_state() == "idle"
    s.mode = Mode.MEETING
    assert s.client_mode() == "meeting"
    assert s.meeting_state() == "recording"
