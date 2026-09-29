"""Unit tests for local voice-command matching (no mic / no server)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.commands import LocalCommand, match_local_command  # noqa: E402


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
