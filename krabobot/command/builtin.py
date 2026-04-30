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
    denied = await _owner_only_guard(ctx, action_label="перезапуск бота")
    if denied is not None:
        return denied

    async def _do_restart():
        await asyncio.sleep(1)
        os.execv(sys.executable, [sys.executable, "-m", "krabobot"] + sys.argv[1:])

    asyncio.create_task(_do_restart())
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="Перезапускаюсь...")


async def cmd_status(ctx: CommandContext) -> OutboundMessage:
    """Build an outbound status message for a session."""
    denied = await _owner_only_guard(ctx, action_label="просмотр статуса")
    if denied is not None:
        return denied
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


async def cmd_clear_memory(ctx: CommandContext) -> OutboundMessage:
    """Clear current user's long-term memory and archive previous content."""
    runtime = ctx.runtime or ctx.loop._default_runtime
    archived = runtime.memory_consolidator.store.clear_and_archive()
    content = (
        "Память очищена. Предыдущее содержимое архивировано в HISTORY.md."
        if archived
        else "Память очищена."
    )
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=content,
        metadata={"render_as": "text"},
    )


async def cmd_start(ctx: CommandContext) -> OutboundMessage:
    """Start/registration entrypoint."""
    msg = ctx.msg
    registered = bool(msg.user_id and await ctx.loop.user_resolver.is_registered(msg.channel, msg.sender_id))
    if registered:
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="👋 Привет! Доступ активен. Отправьте сообщение или /help.",
            metadata={"render_as": "text"},
        )
    return OutboundMessage(
        channel=msg.channel,
        chat_id=msg.chat_id,
        content=(
            "👋 Привет! Для доступа нужна регистрация.\n"
            "Отправьте /reg [кто вы] и дождитесь подтверждения владельца,\n"
            "или /reg <одноразовый_код>."
        ),
        metadata={"render_as": "text"},
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


async def _owner_only_guard(ctx: CommandContext, *, action_label: str) -> OutboundMessage | None:
    """Allow command execution only for owner user."""
    msg = ctx.msg
    if msg.channel in {"cli", "system"}:
        return None
    if not msg.user_id:
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=f"Команда недоступна: только владелец может выполнить {action_label}.",
            metadata={"render_as": "text"},
        )
    is_owner = await ctx.loop.user_resolver.is_owner(msg.user_id)
    if is_owner:
        return None
    return OutboundMessage(
        channel=msg.channel,
        chat_id=msg.chat_id,
        content=f"Команда недоступна: только владелец может выполнить {action_label}.",
        metadata={"render_as": "text"},
    )


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
    if not code:
        if not msg.user_id:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Не удалось определить вашу учетную запись.",
            )
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


async def cmd_reg(ctx: CommandContext) -> OutboundMessage:
    """Self-registration and owner moderation: /reg."""
    msg = ctx.msg
    loop = ctx.loop
    raw_args = (ctx.args or "").strip()
    args = raw_args.split()
    owner_id = await loop.user_resolver.get_owner_user_id()
    is_owner = bool(msg.user_id and await loop.user_resolver.is_owner(msg.user_id))

    if is_owner and args and args[0].lower() in {"list", "approve", "reject"}:
        action = args[0].lower()
        if action == "list":
            requests = await loop.user_resolver.list_registration_requests()
            if not requests:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Очередь регистрации пуста.",
                    metadata={"render_as": "text"},
                )
            lines = ["Очередь регистрации:"]
            for req in requests:
                note = f" | note: {req.note}" if req.note else ""
                lines.append(f"- {req.request_id}: {req.channel}:{req.sender_id}{note}")
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="\n".join(lines),
                metadata={"render_as": "text"},
            )
        if len(args) < 2:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Использование: /reg approve <request_id> | /reg reject <request_id> | /reg list",
                metadata={"render_as": "text"},
            )
        request_id = args[1]
        if action == "approve":
            result = await loop.user_resolver.approve_registration(request_id)
            if not result.ok:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"Не удалось подтвердить регистрацию: {result.error}.",
                    metadata={"render_as": "text"},
                )
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Регистрация {request_id.upper()} подтверждена.",
                metadata={"render_as": "text"},
            )
        result = await loop.user_resolver.reject_registration(request_id)
        if not result.ok:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Не удалось отклонить регистрацию: {result.error}.",
                metadata={"render_as": "text"},
            )
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=f"Регистрация {request_id.upper()} отклонена.",
            metadata={"render_as": "text"},
        )

    if not owner_id:
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="Владелец еще не определен. Повторите попытку позже.",
            metadata={"render_as": "text"},
        )
    if await loop.user_resolver.is_registered(msg.channel, msg.sender_id):
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="Ваш аккаунт уже зарегистрирован.",
            metadata={"render_as": "text"},
        )

    if raw_args and len(raw_args) == 8 and raw_args.isalnum():
        consumed = await loop.user_resolver.consume_registration_code(raw_args, msg.channel, msg.sender_id)
        if consumed.ok:
            msg.user_id = consumed.user_id
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Регистрация по коду подтверждена. Доступ открыт.",
                metadata={"render_as": "text"},
            )

    request = await loop.user_resolver.create_registration_request(
        msg.channel,
        msg.sender_id,
        note=raw_args,
    )
    await _notify_owner_about_registration(ctx, request_id=request.request_id)
    return OutboundMessage(
        channel=msg.channel,
        chat_id=msg.chat_id,
        content=(
            "Заявка на регистрацию отправлена владельцу.\n"
            f"ID заявки: {request.request_id}.\n"
            "Ожидайте подтверждения."
        ),
        metadata={"render_as": "text"},
    )


async def cmd_regcode(ctx: CommandContext) -> OutboundMessage:
    """Owner-only registration codes: /regcode create [ttl_seconds]."""
    denied = await _owner_only_guard(ctx, action_label="управление регистрационными кодами")
    if denied is not None:
        return denied
    msg = ctx.msg
    args = (ctx.args or "").strip().split()
    if not args or args[0].lower() != "create":
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="Использование: /regcode create [ttl_seconds]",
            metadata={"render_as": "text"},
        )
    ttl = 3600
    if len(args) > 1:
        try:
            ttl = max(60, int(args[1]))
        except ValueError:
            ttl = 3600
    code = await ctx.loop.user_resolver.create_registration_code(msg.user_id or "", ttl_seconds=ttl)
    return OutboundMessage(
        channel=msg.channel,
        chat_id=msg.chat_id,
        content=f"Одноразовый код регистрации: {code} (TTL: {ttl}s).",
        metadata={"render_as": "text"},
    )


def register_builtin_commands(router: CommandRouter) -> None:
    """Register the default set of slash commands."""
    handlers = {
        "cmd_stop": cmd_stop,
        "cmd_restart": cmd_restart,
        "cmd_status": cmd_status,
        "cmd_new": cmd_new,
        "cmd_clear_memory": cmd_clear_memory,
        "cmd_start": cmd_start,
        "cmd_id": cmd_id,
        "cmd_link": cmd_link,
        "cmd_tts": cmd_tts,
        "cmd_reg": cmd_reg,
        "cmd_regcode": cmd_regcode,
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
            command="/start",
            route="exact",
            handler_name="cmd_start",
            help_line="/start — Начало работы и проверка доступа",
        ),
        BuiltinCommandSpec(
            command="/new",
            route="exact",
            handler_name="cmd_new",
            help_line="/new — Новый разговор",
        ),
        BuiltinCommandSpec(
            command="/clear_memory",
            route="exact",
            handler_name="cmd_clear_memory",
            help_line="/clear_memory — Очистить память пользователя (с архивом)",
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
        BuiltinCommandSpec(
            command="/reg",
            route="prefix",
            handler_name="cmd_reg",
            help_line="/reg [о_себе|код] — Запросить регистрацию или ввести код",
        ),
        BuiltinCommandSpec(
            command="/regcode",
            route="prefix",
            handler_name="cmd_regcode",
            help_line="/regcode create [ttl] — Создать одноразовый код регистрации (owner)",
        ),
    ]


async def _notify_owner_about_registration(ctx: CommandContext, *, request_id: str) -> None:
    """Best-effort notification to owner account when new registration arrives."""
    owner_id = await ctx.loop.user_resolver.get_owner_user_id()
    if not owner_id:
        return
    accounts = await ctx.loop.user_resolver.accounts_for_user(owner_id)
    if not accounts:
        return
    target = accounts[0]
    channel, sep, sender = target.partition(":")
    if not sep:
        return
    chat_id = _owner_chat_id_from_sender(channel, sender)
    await ctx.loop.bus.publish_outbound(
        OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=(
                "Новая заявка на регистрацию.\n"
                f"ID: {request_id}\n"
                f"Канал: {ctx.msg.channel}\n"
                f"Отправитель: {ctx.msg.sender_id}\n\n"
                f"Подтвердить: /reg approve {request_id}\n"
                f"Отклонить: /reg reject {request_id}"
            ),
            metadata={"render_as": "text"},
        )
    )


def _owner_chat_id_from_sender(channel: str, sender_id: str) -> str:
    """Derive chat_id for owner notification from sender_id."""
    if channel == "telegram" and "|" in sender_id:
        return sender_id.split("|", 1)[0]
    return sender_id
