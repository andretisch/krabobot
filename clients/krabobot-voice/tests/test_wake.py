"""Unit tests for wake phrase normalization (no mic / no server)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.wake import matches_wake_phrase, normalize_text  # noqa: E402


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Эй, Арнольд", True),
        ("эй арнольд", True),
        ("Ну эй арнольд как дела", True),
        ("Hey Arnold", True),
        ("hey, arnold!", True),
        ("эй александр", False),
        ("привет арнольд", False),
        ("эй бот", False),
        ("", False),
    ],
)
def test_matches_wake_phrase(text: str, expected: bool) -> None:
    assert matches_wake_phrase(text) is expected


def test_normalize_yo() -> None:
    assert "е" in normalize_text("Ёжик")
    assert normalize_text("Эй, Арнольд!") == "эй арнольд"
