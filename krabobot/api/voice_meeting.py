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
    "Сохрани расшифровку в `<каталог записи>/<имя_без_расширения>_stt/transcript.txt`.\n"
    "Если расшифровка пустая, ответь одной строкой TRANSCRIPT_EMPTY и не пиши протокол.\n"
    "В самом начале ответа ровно две строки:\n"
    "SUBJECT: <дата события и короткий заголовок; участники через запятую, если известны>\n"
    "EMAIL_SUMMARY: <краткий связный текст на 2–4 предложения>\n"
    "Далее — полный протокол совещания в Markdown (участники, темы, решения, action items).\n"
)

_TRANSCRIPT_EMPTY_RE = re.compile(r"(?im)^\s*TRANSCRIPT_EMPTY\s*$")
_EMPTY_REPORT_BODY = "отчёт пустой"

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


def _transcript_candidates(media: Path) -> list[Path]:
    """Paths where meeting STT is expected to leave ``transcript.txt``."""
    parent = media.parent
    stem = media.stem
    names = [
        parent / f"{stem}_stt" / "transcript.txt",
        parent / f"{stem}-stt" / "transcript.txt",
        parent / stem / "transcript.txt",
    ]
    if not parent.is_dir():
        return names
    prefix = stem.lower()
    try:
        children = list(parent.iterdir())
    except OSError:
        return names
    for child in children:
        if not child.is_dir():
            continue
        label = child.name.lower()
        if label.startswith(prefix) and "stt" in label:
            names.append(child / "transcript.txt")
    return names


def _existing_transcripts(media_paths: list[str]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for raw in media_paths:
        for path in _transcript_candidates(Path(raw)):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            found.append(path)
    return found


def meeting_decoding_is_empty(reply: str, media_paths: list[str]) -> bool:
    """True when this meeting's transcript file or agent marker says decoding is empty.

    A missing transcript file is not enough: only an existing blank ``transcript.txt``
    next to the recording, or an explicit ``TRANSCRIPT_EMPTY`` line, counts.
    Non-empty transcript text wins over the marker.
    """
    files = _existing_transcripts(media_paths)
    if files:
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                text = ""
            if text.strip():
                return False
        return True
    return bool(_TRANSCRIPT_EMPTY_RE.search(reply or ""))


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
    decoding_empty = meeting_decoding_is_empty(reply, media_paths)
    if not reply and not decoding_empty:
        logger.warning("voice meeting job empty reply device_id={}", device_id)
        return

    msg = InboundMessage(
        channel="voice",
        sender_id=device_id,
        chat_id=device_id,
        content="",
    )
    await agent_loop._ensure_identity(msg)
    runtime = await agent_loop._runtime_for_message(msg)

    notes_path: Path | None = None
    if decoding_empty:
        subject = f"{datetime.now(UTC).strftime('%Y-%m-%d')} — отчёт пустой"
        summary = _EMPTY_REPORT_BODY
        logger.info("voice meeting transcript empty device_id={} — email without attachment", device_id)
    else:
        subject, summary, md_body = parse_meeting_agent_reply(reply)
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        notes_path = _meeting_notes_path(runtime.workspace, device_id, stamp)
        try:
            notes_path.write_text(md_body, encoding="utf-8")
        except OSError as e:
            logger.error("Failed to write meeting notes {}: {}", notes_path, e)
            return

    admin_email = await resolve_owner_admin_email(agent_loop)
    if not admin_email:
        if notes_path is not None:
            logger.warning(
                "Meeting notes saved to {} but owner has no linked email:… account "
                "(link owner email in web UI: Users → Links → channel email)",
                notes_path,
            )
        else:
            logger.warning(
                "Meeting transcript empty and owner has no linked email:… account "
                "(link owner email in web UI: Users → Links → channel email)"
            )
        return

    outbound = OutboundMessage(
        channel="email",
        chat_id=admin_email,
        content=summary or "Итоги совещания — см. вложение.",
        media=[str(notes_path.resolve())] if notes_path is not None else [],
        metadata={"subject": subject, "force_send": True},
    )
    ok = await agent_loop.deliver_outbound(outbound)
    if ok:
        logger.info(
            "Meeting notes emailed to {} (subject={!r}, attachment={})",
            admin_email,
            subject,
            notes_path.name if notes_path is not None else "-",
        )
    else:
        logger.warning("Failed to deliver meeting email to {}", admin_email)
