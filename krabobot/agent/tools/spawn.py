"""Spawn tool for creating background subagents."""

from typing import TYPE_CHECKING, Any

from krabobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from krabobot.agent.subagent import SubagentManager


class SpawnTool(Tool):
    """Tool to spawn a subagent for background task execution."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._session_key = "cli:direct"
        self._user_id: str | None = None

    def set_context(self, channel: str, chat_id: str, user_id: str | None = None) -> None:
        """Set the origin context for subagent announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        base = f"{channel}:{chat_id}"
        self._session_key = f"user:{user_id}:{base}" if user_id else base
        self._user_id = user_id

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a background subagent for heavy independent work you should not block on. "
            "USE spawn when: large read/search across many files; multi-step web research "
            "(web_search / web_fetch); long exploration where only a summary is needed back. "
            "Do NOT spawn when: a simple reply, one quick tool call, a single-file edit, "
            "or anything that needs immediate back-and-forth with the user — do those yourself. "
            "This tool returns only an ack that the subagent started; when it finishes, "
            "a compact structured summary (goal / findings / paths / open questions) is "
            "injected into this dialogue — not the full subagent transcript or tool noise. "
            "For deliverables or existing projects, inspect the workspace first "
            "and use a dedicated subdirectory when helpful."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "Self-contained task brief for the subagent. Include goal, "
                        "constraints, and what to return; it has no parent chat history."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": "Optional short label for the task (for display)",
                },
            },
            "required": ["task"],
        }

    async def execute(self, task: str, label: str | None = None, **kwargs: Any) -> str:
        """Spawn a subagent to execute the given task."""
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
            session_key=self._session_key,
            user_id=self._user_id,
        )
