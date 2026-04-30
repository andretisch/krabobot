"""Built-in slash command handlers."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Literal

from krabobot import __version__
from krabobot.bus.events import OutboundMessage
from krabobot.command.router import CommandContext, CommandRouter
from krabobot.utils.helpers import build_status_content


@dataclass(frozen=True)
class BuiltinCommandSpec:
    """Single built-in slash command registration and menu metadata."""

    command: str
    route: Literal["priority", "exact", "prefix"]
    handler_name: str
    help_line: str


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
    content = f"Остановлено задач: {total}." if total else "Нет активных задач для остановки."
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)


async def cmd_restart(ctx: CommandContext) -> OutboundMessage:
    """Restart the process in-place via os.execv."""
    msg = ctx.msg

    async def _do_restart():
        await asyncio.sleep(1)
        os.execv(sys.executable, [sys.executable, "-m", "krabobot"] + sys.argv[1:])

    asyncio.create_task(_do_restart())
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="Перезапускаюсь...")


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
        content="Новая сессия начата.",
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
    lines = ["🦀 Команды krabobot:"]
    lines.extend(spec.help_line for spec in _builtin_command_specs())
    return "\n".join(lines)


async def cmd_id(ctx: CommandContext) -> OutboundMessage:
    """Return caller identifiers useful for manual linking/debug."""
    msg = ctx.msg
    lines = [
        "Ваши идентификаторы:",
        f"- канал: {msg.channel}",
        f"- sender_id: {msg.sender_id}",
        f"- chat_id: {msg.chat_id}",
    ]
    if msg.user_id:
        lines.append(f"- user_id: {msg.user_id}")
        linked_accounts = await ctx.loop.user_resolver.accounts_for_user(msg.user_id)
        if linked_accounts:
            lines.append("")
            lines.append("Привязанные каналы:")
            for account in linked_accounts:
                channel, sep, sender_id = account.partition(":")
                if sep:
                    lines.append(f"- {channel}: {sender_id}")
                else:
                    lines.append(f"- {account}")
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
            content="Режим multi-user отключен в конфиге.",
        )
    if not msg.user_id:
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="Не удалось определить вашу учетную запись.",
        )
    if not code:
        generated = await loop.user_resolver.create_link_code(msg.user_id)
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=(
                "Код привязки создан.\n"
                f"Используйте `/link {generated}` в вашем другом канале в течение времени жизни кода."
            ),
            metadata={"render_as": "text"},
        )

    result = await loop.user_resolver.consume_link_code(code, msg.channel, msg.sender_id)
    if not result.ok:
        await loop.user_resolver.register_failed_link_attempt(code)
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=f"Привязка не выполнена: {result.error or 'invalid_code'}",
            metadata={"render_as": "text"},
        )
    msg.user_id = result.user_id
    return OutboundMessage(
        channel=msg.channel,
        chat_id=msg.chat_id,
        content="Аккаунт успешно привязан. Дальнейшие сообщения будут использовать общий профиль пользователя.",
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
            content="Персональный TTS требует tools.multiUser.enabled=true.",
            metadata={"render_as": "text"},
        )
    if not msg.user_id:
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="Не удалось определить вашу учетную запись.",
            metadata={"render_as": "text"},
        )

    arg = (ctx.args or "").strip().lower()
    if arg in {"", "status"}:
        enabled = await loop.user_resolver.get_tts_enabled(msg.user_id, default=False)
        state = "включен" if enabled else "выключен"
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=f"TTS для вашего пользователя {state}. Используйте `/tts on` или `/tts off`.",
            metadata={"render_as": "text"},
        )
    if arg in {"on", "off"}:
        enabled = arg == "on"
        await loop.user_resolver.set_tts_enabled(msg.user_id, enabled)
        state = "включен" if enabled else "выключен"
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=f"TTS теперь {state} для вашего пользователя.",
            metadata={"render_as": "text"},
        )
    return OutboundMessage(
        channel=msg.channel,
        chat_id=msg.chat_id,
        content="Использование: /tts on | /tts off | /tts status",
        metadata={"render_as": "text"},
    )


def register_builtin_commands(router: CommandRouter) -> None:
    """Register the default set of slash commands."""
    handlers = {
        "cmd_stop": cmd_stop,
        "cmd_restart": cmd_restart,
        "cmd_status": cmd_status,
        "cmd_new": cmd_new,
        "cmd_id": cmd_id,
        "cmd_link": cmd_link,
        "cmd_tts": cmd_tts,
        "cmd_help": cmd_help,
    }
    for spec in _builtin_command_specs():
        handler = handlers[spec.handler_name]
        if spec.route == "priority":
            router.priority(spec.command, handler)
        elif spec.route == "exact":
            router.exact(spec.command, handler)
        else:
            router.prefix(spec.command, handler)


def builtin_menu_commands() -> list[str]:
    """Return canonical slash commands for channel menu generation."""
    commands: list[str] = []
    seen: set[str] = set()
    for spec in _builtin_command_specs():
        if not spec.command.startswith("/"):
            continue
        if spec.command in seen:
            continue
        seen.add(spec.command)
        commands.append(spec.command)
    return commands


def builtin_menu_entries() -> list[tuple[str, str]]:
    """Return unique menu entries as (command_without_slash, description)."""
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for spec in _builtin_command_specs():
        if not spec.command.startswith("/"):
            continue
        if spec.command in seen:
            continue
        seen.add(spec.command)
        cmd = spec.command.lstrip("/")
        desc = spec.help_line.split(" — ", 1)[1] if " — " in spec.help_line else spec.help_line
        entries.append((cmd, desc))
    return entries


def _builtin_command_specs() -> list[BuiltinCommandSpec]:
    """Canonical built-in command registry."""
    return [
        BuiltinCommandSpec(
            command="/new",
            route="exact",
            handler_name="cmd_new",
            help_line="/new — Новый разговор",
        ),
        BuiltinCommandSpec(
            command="/stop",
            route="priority",
            handler_name="cmd_stop",
            help_line="/stop — Остановить текущую задачу",
        ),
        BuiltinCommandSpec(
            command="/restart",
            route="priority",
            handler_name="cmd_restart",
            help_line="/restart — Перезапустить бота",
        ),
        BuiltinCommandSpec(
            command="/status",
            route="priority",
            handler_name="cmd_status",
            help_line="/status — Показать статус бота",
        ),
        BuiltinCommandSpec(
            command="/status",
            route="exact",
            handler_name="cmd_status",
            help_line="/status — Показать статус бота",
        ),
        BuiltinCommandSpec(
            command="/id",
            route="exact",
            handler_name="cmd_id",
            help_line="/id — Показать ваши ID",
        ),
        BuiltinCommandSpec(
            command="/link",
            route="prefix",
            handler_name="cmd_link",
            help_line="/link — Привязать аккаунт между каналами",
        ),
        BuiltinCommandSpec(
            command="/tts",
            route="prefix",
            handler_name="cmd_tts",
            help_line="/tts on|off|status — Голосовые ответы для вашего пользователя",
        ),
        BuiltinCommandSpec(
            command="/help",
            route="exact",
            handler_name="cmd_help",
            help_line="/help — Показать список команд",
        ),
    ]
