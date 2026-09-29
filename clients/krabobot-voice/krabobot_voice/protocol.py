"""Client ↔ server voice protocol: client_state form field and action list."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote

DEFAULT_CAPABILITIES: tuple[str, ...] = (
    "meeting_start",
    "meeting_stop",
    "end_dialog",
    "run_test",
)

KNOWN_ACTIONS = frozenset({"meeting_start", "meeting_stop", "end_dialog", "run_test"})


@dataclass
class ClientState:
    """JSON form field ``client_state`` for POST /v1/voice/turn."""

    mode: str = "idle"  # idle | listen | dialog | meeting
    meeting: str = "idle"  # idle | recording
    capabilities: list[str] = field(
        default_factory=lambda: list(DEFAULT_CAPABILITIES)
    )

    def to_json(self) -> str:
        meeting = self.meeting if self.meeting in {"idle", "recording"} else "idle"
        mode = self.mode if self.mode else "idle"
        caps = [str(c).strip() for c in self.capabilities if str(c).strip()]
        return json.dumps(
            {"mode": mode, "meeting": meeting, "capabilities": caps},
            ensure_ascii=False,
            separators=(",", ":"),
        )


def _normalize_action_item(item: Any) -> str | None:
    if isinstance(item, str):
        name = item.strip()
        return name or None
    if isinstance(item, dict):
        name = str(item.get("action") or item.get("name") or "").strip()
        return name or None
    return None


def parse_actions(
    headers: dict[str, str] | Any | None = None,
    json_body: dict[str, Any] | None = None,
    *,
    log: Any | None = None,
) -> list[str]:
    """Parse server actions from header and/or JSON body.

    Header ``X-Krabobot-Voice-Actions`` is URL-encoded JSON
    ``[{"action":"meeting_stop"}, ...]``. JSON body may carry ``actions``.
    Unknown actions are logged (if ``log`` given) and skipped.
    """
    raw_items: list[Any] = []

    header_val = ""
    if headers is not None:
        try:
            header_val = str(headers.get("X-Krabobot-Voice-Actions") or "")
        except Exception:
            header_val = ""
        if not header_val:
            try:
                # httpx Headers are case-insensitive
                header_val = str(headers.get("x-krabobot-voice-actions") or "")
            except Exception:
                header_val = ""

    if header_val.strip():
        try:
            decoded = unquote(header_val.strip())
            parsed = json.loads(decoded)
            if isinstance(parsed, list):
                raw_items.extend(parsed)
            elif isinstance(parsed, dict):
                raw_items.append(parsed)
        except Exception as e:
            if log is not None:
                log(f"WARNING: bad X-Krabobot-Voice-Actions header: {e}")

    if isinstance(json_body, dict) and "actions" in json_body:
        body_actions = json_body.get("actions")
        if isinstance(body_actions, list):
            raw_items.extend(body_actions)
        elif body_actions is not None:
            raw_items.append(body_actions)

    out: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        name = _normalize_action_item(item)
        if not name:
            continue
        if name not in KNOWN_ACTIONS:
            if log is not None:
                log(f"WARNING: unknown voice action ignored: {name!r}")
            continue
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out
