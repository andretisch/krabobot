"""Cross-process outbound spool (serve → gateway).

``krabobot serve`` and ``krabobot gateway`` are separate processes with separate
in-memory ``MessageBus`` instances. When the web/api agent calls the ``message``
tool targeting a real channel (email, telegram, vk, …), outbound is published on
serve's bus. This module bridges those messages to the gateway via a shared
workspace directory that ChannelManager polls.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from krabobot.bus.events import OutboundMessage
from krabobot.utils.helpers import ensure_dir

# Virtual channels with no ChannelManager adapter (handled in-process / session).
LOCAL_OUTBOUND_CHANNELS = frozenset({"api", "cli", "voice", "system", ""})


def is_local_outbound_channel(channel: str | None) -> bool:
    """Return True when *channel* is delivered locally (not via gateway adapters)."""
    return (channel or "").strip().lower() in LOCAL_OUTBOUND_CHANNELS


def outbound_spool_dir(workspace: Path) -> Path:
    """Directory for pending cross-process outbound messages."""
    return ensure_dir(Path(workspace) / "runtime" / "outbound_spool")


def enqueue_outbound(workspace: Path, msg: OutboundMessage) -> Path:
    """Write *msg* into the workspace spool for the gateway to deliver.

    Returns the final ``.json`` path. Uses a temp file + replace for atomicity.
    """
    directory = outbound_spool_dir(workspace)
    payload: dict[str, Any] = {
        "channel": msg.channel,
        "chat_id": msg.chat_id,
        "content": msg.content,
        "reply_to": msg.reply_to,
        "media": list(msg.media or []),
        "metadata": dict(msg.metadata or {}),
    }
    name = f"{time.time_ns()}_{uuid.uuid4().hex[:8]}.json"
    path = directory / name
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def _msg_from_payload(data: dict[str, Any]) -> OutboundMessage:
    media = data.get("media") or []
    if not isinstance(media, list):
        media = []
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    reply_to = data.get("reply_to")
    return OutboundMessage(
        channel=str(data.get("channel") or ""),
        chat_id=str(data.get("chat_id") or ""),
        content=str(data.get("content") or ""),
        reply_to=str(reply_to) if reply_to is not None else None,
        media=[str(p) for p in media],
        metadata=dict(metadata),
    )


def claim_spooled_outbound(workspace: Path, *, limit: int = 32) -> list[OutboundMessage]:
    """Atomically claim and return pending spool messages (oldest first).

    Each file is renamed to ``.processing`` then deleted after a successful parse.
    Corrupt files are renamed to ``.failed.json`` so they do not block the queue.
    """
    directory = outbound_spool_dir(workspace)
    claimed: list[OutboundMessage] = []
    for path in sorted(directory.glob("*.json")):
        if len(claimed) >= limit:
            break
        if path.name.endswith(".failed.json"):
            continue
        processing = path.with_suffix(".processing")
        try:
            path.replace(processing)
        except OSError:
            continue
        try:
            data = json.loads(processing.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("spool payload is not an object")
            claimed.append(_msg_from_payload(data))
            processing.unlink(missing_ok=True)
        except Exception:
            logger.exception("Corrupt outbound spool file {}", processing.name)
            failed = path.with_name(path.stem + ".failed.json")
            try:
                processing.replace(failed)
            except OSError:
                try:
                    processing.unlink(missing_ok=True)
                except OSError:
                    pass
    return claimed


def route_serve_outbound(msg: OutboundMessage) -> OutboundMessage | None:
    """Decide how serve should handle an outbound message from its local bus.

    Returns ``None`` for local/virtual channels (drop — already in session history).
    Returns a copy marked ``force_send`` for real channels that must be spooled
    to the gateway process.
    """
    if is_local_outbound_channel(msg.channel):
        return None
    meta = dict(msg.metadata or {})
    meta.setdefault("force_send", True)
    return OutboundMessage(
        channel=msg.channel,
        chat_id=msg.chat_id,
        content=msg.content,
        reply_to=msg.reply_to,
        media=list(msg.media or []),
        metadata=meta,
    )
