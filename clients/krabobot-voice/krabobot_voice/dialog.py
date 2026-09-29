"""Pure Talk dialog / follow-up state transitions (unit-testable, no I/O)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TalkPhase(str, Enum):
    """High-level Talk loop phases (Meeting is separate)."""

    WAIT_WAKE = "wait_wake"
    LISTEN = "listen"
    FOLLOW_UP = "follow_up"


@dataclass(frozen=True)
class ListenPlan:
    """How the next mic capture should behave."""

    phase: TalkPhase
    play_beep: bool
    no_speech_timeout_s: float
    settle_s: float


def plan_initial_listen(
    *,
    no_speech_timeout_s: float,
    settle_s: float,
    listen_source: str = "mic",
    play_beep: bool | None = None,
) -> ListenPlan:
    """After wake/PTT: plan the first Talk capture.

    Loopback: skip beep (it fights WASAPI + loses continuous playback) and use
    settle_s=0 — stream keepalive already drained backlog during ASR.
    """
    is_loopback = (listen_source or "").strip().lower() == "loopback"
    if play_beep is None:
        beep = not is_loopback
    else:
        beep = bool(play_beep)
    settle = 0.0 if is_loopback else max(0.0, float(settle_s))
    return ListenPlan(
        phase=TalkPhase.LISTEN,
        play_beep=beep,
        no_speech_timeout_s=max(0.1, float(no_speech_timeout_s)),
        settle_s=settle,
    )


def plan_after_turn(
    *,
    turn_ok: bool,
    follow_up_s: float,
    follow_up_beep: bool,
    settle_s: float,
    default_no_speech_s: float,
) -> ListenPlan:
    """After TTS/playback: stay in conversation listen or return to wake wait.

    ``follow_up_s <= 0`` disables dialog mode.
    """
    if turn_ok and float(follow_up_s) > 0:
        return ListenPlan(
            phase=TalkPhase.FOLLOW_UP,
            play_beep=bool(follow_up_beep),
            no_speech_timeout_s=float(follow_up_s),
            # Slightly shorter settle when no wake beep (flush TTS echo / buffer).
            settle_s=max(0.0, min(float(settle_s), 0.25) if not follow_up_beep else float(settle_s)),
        )
    return ListenPlan(
        phase=TalkPhase.WAIT_WAKE,
        play_beep=False,
        no_speech_timeout_s=max(0.1, float(default_no_speech_s)),
        settle_s=max(0.0, float(settle_s)),
    )


def plan_after_no_speech(*, was_follow_up: bool) -> TalkPhase:
    """No utterance within the listen window → always back to wake wait."""
    _ = was_follow_up
    return TalkPhase.WAIT_WAKE


def plan_after_local_command(*, command: str) -> TalkPhase:
    """Local commands never stay in follow-up (exit / meeting leave dialog)."""
    _ = command
    return TalkPhase.WAIT_WAKE
