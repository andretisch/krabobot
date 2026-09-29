"""Voice-client commands tool — invoke client-owned voice commands."""

from __future__ import annotations

import threading
from typing import Any

from krabobot.agent.tools.base import Tool

VOICE_CLIENT_ACTIONS = ("meeting_start", "meeting_stop", "end_dialog", "run_test")

_pending_lock = threading.Lock()
# chat_id / device_id -> list of {"action": "..."}
_pending_actions: dict[str, list[dict[str, str]]] = {}


def queue_voice_action(chat_id: str, action: str) -> None:
    """Append a voice-client command for the given voice chat/device."""
    key = (chat_id or "").strip()
    if not key or action not in VOICE_CLIENT_ACTIONS:
        return
    with _pending_lock:
        _pending_actions.setdefault(key, []).append({"action": action})


def pop_voice_actions(chat_id: str) -> list[dict[str, str]]:
    """Remove and return all pending commands for chat_id (empty if none)."""
    key = (chat_id or "").strip()
    if not key:
        return []
    with _pending_lock:
        return _pending_actions.pop(key, [])


class VoiceClientActionTool(Tool):
    """Queue a voice-client command for the current turn response."""

    def __init__(self) -> None:
        self._channel = ""
        self._chat_id = ""

    def set_context(self, channel: str, chat_id: str, sender_id: str = "") -> None:
        """Set the current session context (channel / device chat_id)."""
        self._channel = channel or ""
        self._chat_id = chat_id or ""

    @property
    def name(self) -> str:
        return "voice"

    @property
    def description(self) -> str:
        return (
            "REQUIRED for voice-client commands listed in client_state.capabilities. "
            "When the user asks to run a test, start/stop a meeting, or end dialog, "
            "you MUST call this tool with the matching action — do not only acknowledge "
            "in text (saying 'тест запущен' without calling the tool does nothing). "
            "Only available on the voice channel. Commands: "
            "meeting_start (begin local meeting recording), "
            "meeting_stop (stop recording), "
            "end_dialog (exit dialog mode / return to idle), "
            "run_test (print a test confirmation on the client terminal — "
            "use when user says выполнить/выполни/запусти тест). "
            "The client executes the command after the reply audio is played."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(VOICE_CLIENT_ACTIONS),
                    "description": "Voice-client command to invoke",
                },
            },
            "required": ["action"],
        }

    async def execute(self, action: str, **kwargs: Any) -> str:
        if (self._channel or "").strip().lower() != "voice":
            return "Error: voice tool is only available on the voice channel"
        if not self._chat_id:
            return "Error: no session context (chat_id)"
        if action not in VOICE_CLIENT_ACTIONS:
            return f"Error: unknown command '{action}'"
        queue_voice_action(self._chat_id, action)
        return f"Queued voice client command: {action}"
