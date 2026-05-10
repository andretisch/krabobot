"""Message tool for sending messages to users."""

from typing import Any, Awaitable, Callable

from krabobot.agent.tools.base import Tool
from krabobot.bus.events import OutboundMessage
from krabobot.users.resolver import UserResolver


class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        default_message_id: str | None = None,
        *,
        user_resolver: UserResolver | None = None,
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id
        self._default_message_id = default_message_id
        self._user_resolver = user_resolver
        self._context_user_id: str | None = None
        self._sent_in_turn: bool = False

    def set_context(
        self,
        channel: str,
        chat_id: str,
        message_id: str | None = None,
        *,
        user_id: str | None = None,
    ) -> None:
        """Set the current message context."""
        self._default_channel = channel
        self._default_chat_id = chat_id
        self._default_message_id = message_id
        self._context_user_id = user_id

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    def start_turn(self) -> None:
        """Reset per-turn send tracking."""
        self._sent_in_turn = False

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return (
            "Send a message to the user, optionally with file attachments. "
            "This is the ONLY way to deliver files (images, documents, audio, video) to the user. "
            "Use the 'media' parameter with file paths to attach files. "
            "Do NOT use read_file to send files — that only reads content for your own analysis."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (telegram, vk, email, etc.)"
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional: target chat/user ID"
                },
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "REQUIRED when sending files: list of absolute file paths to attach. "
                        "Use this for documents (.docx, .pdf), images, etc. Example: media=[\"/path/to/file.docx\"]"
                    )
                }
            },
            "required": ["content"]
        }

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        message_id: str | None = None,
        media: list[str] | None = None,
        **kwargs: Any
    ) -> str:
        channel = channel or self._default_channel
        chat_id = chat_id or self._default_chat_id
        message_id = message_id or self._default_message_id

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"

        if not self._send_callback:
            return "Error: Message sending not configured"

        metadata: dict[str, Any] = {}
        if message_id:
            metadata["message_id"] = message_id

        recipient_user_id = self._context_user_id
        if self._user_resolver:
            dc = (self._default_channel or "").strip().lower()
            dh = self._default_chat_id or ""
            tc = channel.strip().lower()
            cid = chat_id
            same_target = tc == dc and cid == dh
            if not same_target:
                recipient_user_id = await self._user_resolver.lookup(tc, cid)
            meta_tts = False
            if recipient_user_id:
                try:
                    meta_tts = await self._user_resolver.get_tts_enabled(
                        recipient_user_id,
                        default=False,
                    )
                except Exception:
                    meta_tts = False
            metadata["_tts_enabled_for_user"] = meta_tts

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            media=media or [],
            metadata=metadata,
        )

        try:
            await self._send_callback(msg)
            if channel == self._default_channel and chat_id == self._default_chat_id:
                self._sent_in_turn = True
            media_info = f" with {len(media)} attachments" if media else ""
            return (
                f"Message queued to {channel}:{chat_id}{media_info}. "
                "Delivery depends on channel state/configuration."
            )
        except Exception as e:
            return f"Error sending message: {str(e)}"
