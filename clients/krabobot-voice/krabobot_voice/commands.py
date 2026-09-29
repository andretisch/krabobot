"""Local voice-command matching (no server) — meeting start/stop, run_test, exit dialog."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

from krabobot_voice.wake import command_after_wake, normalize_text

DEFAULT_MEETING_START = (
    "начать совещание",
    "начать запись",
    "запиши совещание",
    "начни совещание",
    "начни запись",
)

DEFAULT_MEETING_STOP = (
    "закончить запись совещания",
    "закончить запись",
    "закончить совещание",
    "завершить запись",
    "стоп запись",
    "стоп запись совещания",
    "закончи совещание",
    "останови запись",
    "стоп совещание",
    # Sherpa often drops the leading «За» of «Закончить» → «Кончить…»
    "кончить запись совещания",
    "кончить запись",
)

DEFAULT_RUN_TEST = (
    "выполни тест",
    "выполнить тест",
    "сделай тест",
    "сделать тест",
    "запусти тест",
    "запустить тест",
    "проведи тест",
    "провести тест",
)

DEFAULT_EXIT = (
    "хватит",
    "выход",
    "спокойной ночи",
    "отмена",
    "закончили",
    "пока",
)


class LocalCommand(str, Enum):
    MEETING_START = "meeting_start"
    MEETING_STOP = "meeting_stop"
    RUN_TEST = "run_test"
    EXIT = "exit"


def _phrase_hits(norm: str, phrase: str) -> bool:
    """True if normalized utterance contains phrase as whole words."""
    p = normalize_text(phrase)
    if not p or not norm:
        return False
    return f" {p} " in f" {norm} "


def _with_default_stop_phrases(
    configured: list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Configured stop phrases first; append any missing built-in defaults.

    Stale YAML often drops «закончить запись совещания» while the UI still
    advertises it — keep DEFAULT_MEETING_STOP as a safety floor so meeting
    stop never falls through to a blocking Talk turn.
    """
    if configured is None:
        return list(DEFAULT_MEETING_STOP)
    out = list(configured)
    seen = {normalize_text(p) for p in out if p}
    for phrase in DEFAULT_MEETING_STOP:
        key = normalize_text(phrase)
        if key and key not in seen:
            out.append(phrase)
            seen.add(key)
    return out


def match_local_command(
    text: str,
    *,
    meeting_start: list[str] | tuple[str, ...] | None = None,
    meeting_stop: list[str] | tuple[str, ...] | None = None,
    run_test: list[str] | tuple[str, ...] | None = None,
    exit_dialog: list[str] | tuple[str, ...] | None = None,
) -> LocalCommand | None:
    """Return the first matching local command, or None for ordinary Talk.

    Priority: meeting_stop → meeting_start → run_test → exit
    (stop/start/test before short exits).
    """
    norm = normalize_text(text)
    if not norm:
        return None

    # Stop phrases always keep built-in defaults (see _with_default_stop_phrases).
    stop = _with_default_stop_phrases(meeting_stop)
    start = meeting_start if meeting_start is not None else DEFAULT_MEETING_START
    test = run_test if run_test is not None else DEFAULT_RUN_TEST
    exits = exit_dialog if exit_dialog is not None else DEFAULT_EXIT

    # Longer / more specific phrases first within each group.
    for phrase in sorted(stop, key=lambda s: len(normalize_text(s)), reverse=True):
        if _phrase_hits(norm, phrase):
            return LocalCommand.MEETING_STOP
    for phrase in sorted(start, key=lambda s: len(normalize_text(s)), reverse=True):
        if _phrase_hits(norm, phrase):
            return LocalCommand.MEETING_START
    for phrase in sorted(test, key=lambda s: len(normalize_text(s)), reverse=True):
        if _phrase_hits(norm, phrase):
            return LocalCommand.RUN_TEST
    for phrase in sorted(exits, key=lambda s: len(normalize_text(s)), reverse=True):
        if _phrase_hits(norm, phrase):
            return LocalCommand.EXIT
    return None


def match_utterance_command(
    text: str,
    *,
    wake_phrases: Sequence[str] | None = None,
    wake_greetings: Sequence[str] | None = None,
    meeting_start: list[str] | tuple[str, ...] | None = None,
    meeting_stop: list[str] | tuple[str, ...] | None = None,
    run_test: list[str] | tuple[str, ...] | None = None,
    exit_dialog: list[str] | tuple[str, ...] | None = None,
) -> LocalCommand | None:
    """Match a local command in a full ASR utterance.

    1. Phrase match anywhere in the normalized text (meeting stop alone works).
    2. Else strip the *configured* wake via ``command_after_wake`` and match
       the trailing remainder (wake + stop in one utterance).

    Wake name/phrases are never hard-coded — callers pass ``wake_phrases`` /
    ``wake_greetings`` from config.
    """
    cmd_kwargs = {
        "meeting_start": meeting_start,
        "meeting_stop": meeting_stop,
        "run_test": run_test,
        "exit_dialog": exit_dialog,
    }
    hit = match_local_command(text, **cmd_kwargs)
    if hit is not None:
        return hit
    trailing = command_after_wake(
        text,
        phrases=wake_phrases,
        greetings=wake_greetings,
    )
    if not trailing or trailing == normalize_text(text):
        return None
    return match_local_command(trailing, **cmd_kwargs)
