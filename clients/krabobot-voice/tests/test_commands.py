"""Unit tests for local voice-command matching (no mic / no server)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.commands import (  # noqa: E402
    LocalCommand,
    match_local_command,
    match_utterance_command,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("начать совещание", LocalCommand.MEETING_START),
        ("Ну давай начать запись пожалуйста", LocalCommand.MEETING_START),
        ("запиши совещание", LocalCommand.MEETING_START),
        ("закончить запись совещания", LocalCommand.MEETING_STOP),
        ("закончить совещание", LocalCommand.MEETING_STOP),
        ("пожалуйста завершить запись", LocalCommand.MEETING_STOP),
        ("стоп запись", LocalCommand.MEETING_STOP),
        # Sherpa truncation: «Закончить» → «Кончить» (leading «За» dropped)
        ("Кончить запись совещания.", LocalCommand.MEETING_STOP),
        ("кончить запись", LocalCommand.MEETING_STOP),
        ("стоп запись совещания", LocalCommand.MEETING_STOP),
        ("закончить запись", LocalCommand.MEETING_STOP),
        # Bare «кончить» without «запись» must not match
        ("кончить", None),
        ("хватит", LocalCommand.EXIT),
        ("спокойной ночи", LocalCommand.EXIT),
        ("отмена", LocalCommand.EXIT),
        ("выход", LocalCommand.EXIT),
        ("выполни тест", LocalCommand.RUN_TEST),
        ("выполнить тест", LocalCommand.RUN_TEST),
        ("пожалуйста сделай тест", LocalCommand.RUN_TEST),
        ("сделать тест сейчас", LocalCommand.RUN_TEST),
        ("запусти тест", LocalCommand.RUN_TEST),
        ("запустить тест", LocalCommand.RUN_TEST),
        ("проведи тест", LocalCommand.RUN_TEST),
        ("провести тест", LocalCommand.RUN_TEST),
        ("какая сегодня погода", None),
        ("привет арнольд", None),
        ("", None),
    ],
)
def test_match_local_command(text: str, expected: LocalCommand | None) -> None:
    assert match_local_command(text) is expected


@pytest.mark.parametrize(
    "text,wake_phrases,wake_greetings,expected",
    [
        ("Кончить запись совещания.", ["ок бот"], ["ок"], LocalCommand.MEETING_STOP),
        (
            "Эй, Арнольд, закончить запись совещания",
            ["эй арнольд"],
            ["эй"],
            LocalCommand.MEETING_STOP,
        ),
        (
            "эй арнольд закончить запись",
            ["эй арнольд"],
            ["эй"],
            LocalCommand.MEETING_STOP,
        ),
        (
            "Ок Бот закончить запись",
            ["ок бот"],
            ["ок"],
            LocalCommand.MEETING_STOP,
        ),
        (
            "ок бот закончить запись совещания",
            ["ок бот"],
            ["ок"],
            LocalCommand.MEETING_STOP,
        ),
        (
            "Ок Бот кончить запись совещания",
            ["ок бот"],
            ["ок"],
            LocalCommand.MEETING_STOP,
        ),
        # Custom wake alone — not a command
        ("ок бот", ["ок бот"], ["ок"], None),
    ],
)
def test_match_utterance_wake_plus_stop(
    text: str,
    wake_phrases: list[str],
    wake_greetings: list[str],
    expected: LocalCommand | None,
) -> None:
    assert (
        match_utterance_command(
            text,
            wake_phrases=wake_phrases,
            wake_greetings=wake_greetings,
        )
        is expected
    )


def test_custom_exit_phrases() -> None:
    assert (
        match_local_command("всё хватит говорить", exit_dialog=["хватит говорить"])
        is LocalCommand.EXIT
    )
    assert match_local_command("хватит", exit_dialog=["спокойной ночи"]) is None


def test_stop_beats_exit_when_both_present() -> None:
    # Unlikely ASR, but stop phrases are checked first.
    text = "стоп запись и выход"
    assert match_local_command(text) is LocalCommand.MEETING_STOP


def test_stale_stop_list_keeps_default_primary_phrase() -> None:
    """YAML without «закончить запись совещания» must still match it."""
    stale = [
        "закончить совещание",
        "завершить запись",
        "стоп запись",
        "закончи совещание",
        "останови запись",
        "стоп совещание",
    ]
    assert (
        match_local_command(
            "Закончить запись совещания.",
            meeting_stop=stale,
        )
        is LocalCommand.MEETING_STOP
    )


def test_stale_stop_list_keeps_asr_truncation_aliases() -> None:
    """Stale YAML without truncation aliases still matches via defaults floor."""
    stale = [
        "закончить запись совещания",
        "закончить совещание",
        "завершить запись",
        "стоп запись",
    ]
    assert (
        match_local_command(
            "Кончить запись совещания.",
            meeting_stop=stale,
        )
        is LocalCommand.MEETING_STOP
    )


def test_stop_substring_with_wake_and_asr_typo() -> None:
    """Wake name + «закончить запись» + typo «совещать» still matches stop."""
    assert (
        match_utterance_command(
            "Арнольд закончить запись совещать.",
            wake_phrases=["эй арнольд", "арнольд"],
            wake_greetings=["эй"],
        )
        is LocalCommand.MEETING_STOP
    )
    assert match_local_command("Закончить запись.") is LocalCommand.MEETING_STOP
