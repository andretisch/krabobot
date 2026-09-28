"""Unit tests for PTT hotkey parsing / matching (no keyboard listener)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.ptt import combo_matches, parse_hotkey  # noqa: E402


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("ctrl+alt+space", frozenset({"ctrl", "alt", "space"})),
        ("CTRL+ALT+SPACE", frozenset({"ctrl", "alt", "space"})),
        ("control+option+space", frozenset({"ctrl", "alt", "space"})),
        ("", frozenset({"ctrl", "alt", "space"})),
        ("f8", frozenset({"f8"})),
    ],
)
def test_parse_hotkey(spec: str, expected: frozenset[str]) -> None:
    assert parse_hotkey(spec) == expected


def test_combo_matches_requires_all_keys() -> None:
    combo = parse_hotkey("ctrl+alt+space")
    assert not combo_matches({"ctrl", "alt"}, combo)
    assert combo_matches({"ctrl", "alt", "space"}, combo)
    assert combo_matches({"ctrl", "alt", "space", "shift"}, combo)
