"""Built-in slash command handlers."""

from __future__ import annotations

import asyncio
import os
import sys

from krabobot import __version__
from krabobot.bus.events import OutboundMessage
from krabobot.command.router import CommandContext, CommandRouter
from krabobot.utils.helpers import build_status_content


async def cmd_stop(ctx: CommandContext) -> OutboundMessage:
    """Cancel all active tasks and subagents for the session."""
    loop = ctx.loop
    msg = ctx.msg
    runtime = ctx.runtime or loop._default_runtime
    tasks = loop._active_tasks.pop(msg.dispatch_key, [])
    cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
    for t in tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    sub_cancelled = await runtime.subagents.cancel_by_session(msg.dispatch_key)
    total = cancelled + sub_cancelled
    content = f"Stopped {total} task(s)." if total else "No active task to stop."
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)


async def cmd_restart(ctx: CommandContext) -> OutboundMessage:
    """Restart the process in-place via os.execv."""
    msg = ctx.msg

    async def _do_restart():
        await asyncio.sleep(1)
        os.execv(sys.executable, [sys.executable, "-m", "krabobot"] + sys.argv[1:])

    asyncio.create_task(_do_restart())
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="Restarting...")


async def cmd_status(ctx: CommandContext) -> OutboundMessage:
    """Build an outbound status message for a session."""
    loop = ctx.loop
    runtime = ctx.runtime or loop._default_runtime
    session = ctx.session or runtime.sessions.get_or_create(ctx.key)
    ctx_est = 0
    try:
        ctx_est, _ = runtime.memory_consolidator.estimate_session_prompt_tokens(session)
    except Exception:
        pass
    if ctx_est <= 0:
        ctx_est = loop._last_usage.get("prompt_tokens", 0)
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_status_content(
            version=__version__, model=loop.model,
            start_time=loop._start_time, last_usage=loop._last_usage,
            context_window_tokens=loop.context_window_tokens,
            session_msg_count=len(session.get_history(max_messages=0)),
            context_tokens_estimate=ctx_est,
        ),
        metadata={"render_as": "text"},
    )


async def cmd_new(ctx: CommandContext) -> OutboundMessage:
    """Start a fresh session."""
    loop = ctx.loop
    runtime = ctx.runtime or loop._default_runtime
    session = ctx.session or runtime.sessions.get_or_create(ctx.key)
    snapshot = session.messages[session.last_consolidated:]
    session.clear()
    runtime.sessions.save(session)
    runtime.sessions.invalidate(session.key)
    if snapshot:
        loop._schedule_background(runtime.memory_consolidator.archive_messages(snapshot))
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content="New session started.",
    )


async def cmd_help(ctx: CommandContext) -> OutboundMessage:
    """Return available slash commands."""
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_help_text(),
        metadata={"render_as": "text"},
    )


def build_help_text() -> str:
    """Build canonical help text shared across channels."""
    lines = [
        "🦀 krabobot commands:",
        "/new — Start a new conversation",
        "/stop — Stop the current task",
        "/restart — Restart the bot",
        "/status — Show bot status",
        "/id — Show your IDs",
        "/link — Link account across channels",
        "/tts on|off|status — Per-user voice replies",
        "/help — Show available commands",
    ]
    return "\n".join(lines)


async def cmd_id(ctx: CommandContext) -> OutboundMessage:
    """Return caller identifiers useful for manual linking/debug."""
    msg = ctx.msg
    lines = [
        "Your identifiers:",
        f"- channel: {msg.channel}",
        f"- sender_id: {msg.sender_id}",
        f"- chat_id: {msg.chat_id}",
    ]
    if msg.user_id:
        lines.append(f"- user_id: {msg.user_id}")
    return OutboundMessage(
        channel=msg.channel,
        chat_id=msg.chat_id,
        content="\n".join(lines),
        metadata={"render_as": "text"},
    )


async def cmd_link(ctx: CommandContext) -> OutboundMessage:
    """Generate or consume one-time link codes for cross-channel account mapping."""
    loop = ctx.loop
    msg = ctx.msg
    code = (ctx.args or "").strip().upper()
    if not loop.multi_user_enabled:
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="Multi-user mode is disabled in config.",
        )
    if not msg.user_id:
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="Unable to resolve your account identity.",
        )
    if not code:
        generated = await loop.user_resolver.create_link_code(msg.user_id)
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=(
                "Link code created.\n"
                f"Use `/link {generated}` in your other channel account within the TTL window."
            ),
            metadata={"render_as": "text"},
        )

    result = await loop.user_resolver.consume_link_code(code, msg.channel, msg.sender_id)
    if not result.ok:
        await loop.user_resolver.register_failed_link_attempt(code)
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=f"Link failed: {result.error or 'invalid_code'}",
            metadata={"render_as": "text"},
        )
    msg.user_id = result.user_id
    return OutboundMessage(
        channel=msg.channel,
        chat_id=msg.chat_id,
        content="Account linked successfully. Your future messages will share the same user data.",
        metadata={"render_as": "text"},
    )


async def cmd_tts(ctx: CommandContext) -> OutboundMessage:
    """Manage per-user TTS preference: /tts on|off|status."""
    msg = ctx.msg
    loop = ctx.loop
    if not loop.multi_user_enabled:
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="Per-user TTS requires tools.multiUser.enabled=true.",
            metadata={"render_as": "text"},
        )
    if not msg.user_id:
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="Unable to resolve your account identity.",
            metadata={"render_as": "text"},
        )

    arg = (ctx.args or "").strip().lower()
    if arg in {"", "status"}:
        enabled = await loop.user_resolver.get_tts_enabled(msg.user_id, default=False)
        state = "ON" if enabled else "OFF"
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=f"TTS for your user is {state}. Use `/tts on` or `/tts off`.",
            metadata={"render_as": "text"},
        )
    if arg in {"on", "off"}:
        enabled = arg == "on"
        await loop.user_resolver.set_tts_enabled(msg.user_id, enabled)
        state = "enabled" if enabled else "disabled"
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=f"TTS is now {state} for your user.",
            metadata={"render_as": "text"},
        )
    return OutboundMessage(
        channel=msg.channel,
        chat_id=msg.chat_id,
        content="Usage: /tts on | /tts off | /tts status",
        metadata={"render_as": "text"},
    )


def register_builtin_commands(router: CommandRouter) -> None:
    """Register the default set of slash commands."""
    router.priority("/stop", cmd_stop)
    router.priority("/restart", cmd_restart)
    router.priority("/status", cmd_status)
    router.exact("/new", cmd_new)
    router.exact("/status", cmd_status)
    router.exact("/id", cmd_id)
    router.prefix("/link", cmd_link)
    router.prefix("/tts", cmd_tts)
    router.exact("/help", cmd_help)
