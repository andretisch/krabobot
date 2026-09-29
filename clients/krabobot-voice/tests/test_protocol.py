"""Unit tests for client_state / voice actions protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.protocol import ClientState, parse_actions  # noqa: E402


def test_client_state_to_json() -> None:
    state = ClientState(mode="dialog", meeting="recording")
    raw = state.to_json()
    data = json.loads(raw)
    assert data["mode"] == "dialog"
    assert data["meeting"] == "recording"
    assert "meeting_start" in data["capabilities"]
    assert "meeting_stop" in data["capabilities"]
    assert "end_dialog" in data["capabilities"]
    assert "run_test" in data["capabilities"]


def test_parse_actions_from_header() -> None:
    payload = quote(json.dumps([{"action": "meeting_stop"}, {"action": "end_dialog"}]))
    headers = {"X-Krabobot-Voice-Actions": payload}
    assert parse_actions(headers, None) == ["meeting_stop", "end_dialog"]


def test_parse_actions_from_json_body() -> None:
    body = {"actions": [{"action": "meeting_start"}]}
    assert parse_actions({}, body) == ["meeting_start"]


def test_parse_actions_run_test() -> None:
    body = {"actions": [{"action": "run_test"}]}
    assert parse_actions({}, body) == ["run_test"]


def test_parse_actions_unknown_ignored() -> None:
    logs: list[str] = []
    body = {"actions": [{"action": "explode"}, {"action": "end_dialog"}]}
    out = parse_actions({}, body, log=logs.append)
    assert out == ["end_dialog"]
    assert any("explode" in m for m in logs)


def test_parse_actions_dedupes() -> None:
    body = {"actions": ["meeting_stop", {"action": "meeting_stop"}]}
    assert parse_actions({}, body) == ["meeting_stop"]


def test_parse_actions_string_items() -> None:
    payload = quote(json.dumps(["end_dialog"]))
    assert parse_actions({"X-Krabobot-Voice-Actions": payload}, None) == ["end_dialog"]
