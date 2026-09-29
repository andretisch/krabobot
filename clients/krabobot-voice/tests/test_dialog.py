"""Unit tests for Talk dialog follow-up planning (no mic / no server)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.dialog import (  # noqa: E402
    TalkPhase,
    plan_after_local_command,
    plan_after_no_speech,
    plan_after_turn,
    plan_initial_listen,
)


def test_initial_listen_beeps() -> None:
    plan = plan_initial_listen(no_speech_timeout_s=5.0, settle_s=0.35)
    assert plan.phase is TalkPhase.LISTEN
    assert plan.play_beep is True
    assert plan.no_speech_timeout_s == 5.0


def test_follow_up_after_ok_turn() -> None:
    plan = plan_after_turn(
        turn_ok=True,
        follow_up_s=8.0,
        follow_up_beep=False,
        settle_s=0.35,
        default_no_speech_s=5.0,
    )
    assert plan.phase is TalkPhase.FOLLOW_UP
    assert plan.play_beep is False
    assert plan.no_speech_timeout_s == 8.0
    assert plan.settle_s <= 0.25


def test_follow_up_disabled() -> None:
    plan = plan_after_turn(
        turn_ok=True,
        follow_up_s=0,
        follow_up_beep=False,
        settle_s=0.35,
        default_no_speech_s=5.0,
    )
    assert plan.phase is TalkPhase.WAIT_WAKE


def test_failed_turn_back_to_wake() -> None:
    plan = plan_after_turn(
        turn_ok=False,
        follow_up_s=8.0,
        follow_up_beep=True,
        settle_s=0.35,
        default_no_speech_s=5.0,
    )
    assert plan.phase is TalkPhase.WAIT_WAKE
    assert plan.play_beep is False


def test_no_speech_leaves_follow_up() -> None:
    assert plan_after_no_speech(was_follow_up=True) is TalkPhase.WAIT_WAKE
    assert plan_after_no_speech(was_follow_up=False) is TalkPhase.WAIT_WAKE


def test_local_command_exits_dialog() -> None:
    assert plan_after_local_command(command="exit") is TalkPhase.WAIT_WAKE
    assert plan_after_local_command(command="meeting_start") is TalkPhase.WAIT_WAKE
