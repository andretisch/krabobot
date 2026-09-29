"""Local voice-command matching (no server) — meeting start/stop, run_test, exit dialog."""

from __future__ import annotations

from enum import Enum

from krabobot_voice.wake import normalize_text

DEFAULT_MEETING_START = (
    "начать совещание",
    "начать запись",
    "запиши совещание",
    "начни совещание",
    "начни запись",
)

DEFAULT_MEETING_STOP = (
    "закончить совещание",
    "завершить запись",
    "стоп запись",
    "закончи совещание",
    "останови запись",
    "стоп совещание",
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

    stop = meeting_stop if meeting_stop is not None else DEFAULT_MEETING_STOP
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
