"""Async voice meeting processing and owner email delivery."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from krabobot.agent.loop import AgentLoop
from krabobot.agent.tools.message import linked_email_addresses
from krabobot.agent.tools.voice import pop_voice_actions
from krabobot.bus.events import InboundMessage, OutboundMessage
from krabobot.utils.helpers import ensure_dir, safe_filename

_VOICE_AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".m4a", ".flac", ".opus", ".aac"}
_VOICE_VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}


def voice_media_content_notes(paths: list[str]) -> list[str]:
    """Annotate attached files in agent content (same as voice HTTP turn)."""
    notes: list[str] = []
    for raw in paths:
        p = Path(raw)
        ext = p.suffix.lower()
        if ext in _VOICE_VIDEO_EXTS:
            notes.append(f"[video: {p}]")
        elif ext in _VOICE_AUDIO_EXTS:
            notes.append(f"[audio: {p}]")
        else:
            notes.append(f"[Файл сохранён в workspace: {p}]")
    return notes

_MEETING_EMAIL_HINT = (
    "Результат нужен для письма администратору (не для голосового TTS).\n"
    "В самом начале ответа ровно две строки:\n"
    "SUBJECT: <дата события и короткий заголовок; участники через запятую, если известны>\n"
    "EMAIL_SUMMARY: <краткий связный текст на 2–4 предложения>\n"
    "Далее — полный протокол совещания в Markdown (участники, темы, решения, action items).\n"
)

_SUBJECT_RE = re.compile(r"^SUBJECT:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_SUMMARY_RE = re.compile(r"^EMAIL_SUMMARY:\s*(.+)$", re.MULTILINE | re.IGNORECASE)


def parse_meeting_agent_reply(text: str) -> tuple[str, str, str]:
    """Return (email_subject, email_summary, markdown_body)."""
    raw = (text or "").strip()
    if not raw:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        return f"{today} — итоги совещания", "", ""

    subject_m = _SUBJECT_RE.search(raw)
    summary_m = _SUMMARY_RE.search(raw)
    subject = subject_m.group(1).strip() if subject_m else ""
    summary = summary_m.group(1).strip() if summary_m else ""

    body_lines: list[str] = []
    for line in raw.splitlines():
        if _SUBJECT_RE.match(line) or _SUMMARY_RE.match(line):
            continue
        body_lines.append(line)
    body = "\n".join(body_lines).strip()

    if not subject:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        first_line = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        title = first_line.lstrip("#").strip()[:80] if first_line else "итоги совещания"
        subject = f"{today} — {title}"

    if not summary:
        plain = re.sub(r"[#*_`>\[\]()]", " ", body)
        plain = " ".join(plain.split())
        summary = plain[:600] + ("…" if len(plain) > 600 else "")

    if not body:
        body = raw
    return subject, summary, body


async def resolve_owner_admin_email(agent_loop: AgentLoop) -> str | None:
    """Primary mailbox for the deployment owner (linked ``email:`` account)."""
    owner_id = await agent_loop.user_resolver.get_owner_user_id()
    if not owner_id:
        return None
    accounts = await agent_loop.user_resolver.accounts_for_user(owner_id)
    emails = linked_email_addresses(accounts)
    return emails[0] if emails else None


def _meeting_notes_path(workspace: Path, device_id: str, stem: str) -> Path:
    sub = safe_filename(device_id)[:80] or "device"
    root = ensure_dir(workspace.resolve() / "meetings" / "voice" / sub)
    return root / f"{stem}-notes.md"


async def run_voice_meeting_job(
    agent_loop: AgentLoop,
    *,
    device_id: str,
    instruct: str,
    media_paths: list[str],
    session_key: str,
    timeout_s: float,
) -> None:
    """Process meeting media in background and email Markdown notes to the owner."""
    pop_voice_actions(device_id)
    content_parts = [p for p in (instruct,) if p.strip()]
    if media_paths:
        content_parts.extend(voice_media_content_notes(media_paths))
    if not content_parts:
        content_parts.append("Обработай приложенные файлы записи совещания.")
    body = "\n\n".join(content_parts)
    content = f"{_MEETING_EMAIL_HINT}\n\n{body}"

    try:
        result = await asyncio.wait_for(
            agent_loop.process_direct(
                content,
                session_key=session_key,
                channel="voice",
                chat_id=device_id,
                sender_id=device_id,
                media=media_paths or None,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.error("voice meeting job timed out after {}s device_id={}", timeout_s, device_id)
        return
    except Exception:
        logger.exception("voice meeting background job failed device_id={}", device_id)
        return

    reply = ""
    if result is not None:
        reply = str(getattr(result, "content", None) or "").strip()
    if not reply:
        logger.warning("voice meeting job empty reply device_id={}", device_id)
        return

    subject, summary, md_body = parse_meeting_agent_reply(reply)
    msg = InboundMessage(
        channel="voice",
        sender_id=device_id,
        chat_id=device_id,
        content="",
    )
    await agent_loop._ensure_identity(msg)
    runtime = await agent_loop._runtime_for_message(msg)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    notes_path = _meeting_notes_path(runtime.workspace, device_id, stamp)
    try:
        notes_path.write_text(md_body, encoding="utf-8")
    except OSError as e:
        logger.error("Failed to write meeting notes {}: {}", notes_path, e)
        return

    admin_email = await resolve_owner_admin_email(agent_loop)
    if not admin_email:
        logger.warning(
            "Meeting notes saved to {} but owner has no linked email:… account "
            "(link owner email in web UI: Users → Links → channel email)",
            notes_path,
        )
        return

    outbound = OutboundMessage(
        channel="email",
        chat_id=admin_email,
        content=summary or "Итоги совещания — см. вложение.",
        media=[str(notes_path.resolve())],
        metadata={"subject": subject, "force_send": True},
    )
    ok = await agent_loop.deliver_outbound(outbound)
    if ok:
        logger.info(
            "Meeting notes emailed to {} (subject={!r}, attachment={})",
            admin_email,
            subject,
            notes_path.name,
        )
    else:
        logger.warning("Failed to deliver meeting email to {}", admin_email)
