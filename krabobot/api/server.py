"""OpenAI-compatible HTTP API server for a fixed krabobot session.

Provides /v1/chat/completions and /v1/models endpoints.
All requests route to a single persistent API session.
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import re
import shutil
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from aiohttp import web
from loguru import logger
from pydantic import ValidationError

from krabobot.agent.context import ContextBuilder
from krabobot.agent.loop import AgentLoop
from krabobot.agent.tools.voice import pop_voice_actions
from krabobot.api.voice_io import synthesize_speech_wav, transcribe_audio
from krabobot.api.web_auth import (
    auth_middleware,
    handle_auth_login,
    handle_auth_logout,
    handle_auth_setup,
    handle_auth_status,
)
from krabobot.api.web_config import (
    build_web_config_payload,
    list_config_backups,
    restore_backup,
    save_web_config_sections,
)
from krabobot.api.web_files import handle_web_file_download
from krabobot.api.web_users import (
    handle_registration_approve,
    handle_registration_reject,
    handle_registrations_list,
    handle_user_delete,
    handle_user_get,
    handle_user_link_add,
    handle_user_link_remove,
    handle_user_patch,
    handle_users_create,
    handle_users_list,
)
from krabobot.bus.events import InboundMessage, is_background_wakeup
from krabobot.utils.helpers import ensure_dir, safe_filename


def web_static_dir() -> Path:
    """Directory with bundled static chat UI (index.html, app.js, …)."""
    return Path(__file__).resolve().parent.parent / "web" / "static"


async def handle_chat_index(_request: web.Request) -> web.StreamResponse:
    """Serve single-page chat at GET /. Inject mtime cache-bust on app.js / CSS."""
    static = web_static_dir()
    idx = static / "index.html"
    if not idx.is_file():
        return web.Response(status=404, text="Web UI not found on server.")
    try:
        html = idx.read_text(encoding="utf-8")
    except OSError:
        return web.Response(status=500, text="Failed to read Web UI.")
    app_js = static / "app.js"
    if app_js.is_file():
        ver = str(int(app_js.stat().st_mtime))
        html = html.replace('src="/static/app.js"', f'src="/static/app.js?v={ver}"')
    css = static / "chat.css"
    if css.is_file():
        cver = str(int(css.stat().st_mtime))
        html = html.replace('href="/static/chat.css"', f'href="/static/chat.css?v={cver}"')
    return web.Response(
        text=html,
        content_type="text/html",
        charset="utf-8",
        headers={"Cache-Control": "no-store"},
    )

API_SESSION_KEY = "api:default"
API_CHAT_ID = "default"
# aiohttp defaults to 1 MiB — base64 images in JSON exceed that and the body is truncated → JSON parse fails.
MAX_REQUEST_BODY_BYTES = 64 * 1024 * 1024
# Soft threshold: ≤ this size uses the normal chat attach path; above → stream-to-disk «folder» path.
MAX_UPLOAD_SOFT_BYTES = 50 * 1024 * 1024
# Backward-compatible alias (soft mode / former hard cap for in-memory multipart).
MAX_UPLOAD_FILE_BYTES = MAX_UPLOAD_SOFT_BYTES
# Default hard max for large multipart uploads (overridable via api.maxUploadMb).
DEFAULT_MAX_UPLOAD_HARD_MB = 2048


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _error_json(status: int, message: str, err_type: str = "invalid_request_error") -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": err_type, "code": status}},
        status=status,
    )


def _chat_completion_response(content: str, model: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# Injected into every voice turn so TTS gets speech-friendly plain text.
_VOICE_REPLY_HINT = (
    "Отвечай для голосового интерфейса: без Markdown, без эмодзи, обычный текст."
)

_VOICE_VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
_VOICE_AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".m4a", ".flac", ".opus", ".aac"}


def _voice_media_content_notes(paths: list[str]) -> list[str]:
    """Annotate attached files in content (same pattern as Telegram/web video/docs).

    Long meeting WAVs must appear as path notes — not as the ``audio`` STT field.
    """
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


def _format_voice_client_state_line(raw: str) -> str | None:
    """Build a short context line from client_state JSON, or None if unusable."""
    text = (raw or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    mode = str(data.get("mode") or "").strip() or "unknown"
    meeting = str(data.get("meeting") or "").strip() or "idle"
    caps_raw = data.get("capabilities")
    caps: list[str] = []
    if isinstance(caps_raw, list):
        caps = [str(c).strip() for c in caps_raw if str(c).strip()]
    if caps:
        return (
            f"[voice client: mode={mode}, meeting={meeting}; "
            f"available commands: {', '.join(caps)}. "
            "If the user requests one of these, CALL the voice tool with that "
            "action — do not only acknowledge in text.]"
        )
    return f"[voice client: mode={mode}, meeting={meeting}]"


def _encode_voice_actions_header(actions: list[dict[str, str]]) -> str:
    """URL-encode JSON list for X-Krabobot-Voice-Actions (latin-1-safe)."""
    return quote(json.dumps(actions, ensure_ascii=False, separators=(",", ":")), safe="")


def _response_text(value: Any) -> str:
    """Normalize process_direct output to plain assistant text."""
    if value is None:
        return ""
    if hasattr(value, "content"):
        return str(getattr(value, "content") or "")
    return str(value)


def _coerce_api_user_content(raw: Any) -> str | list[dict[str, Any]]:
    """Normalize OpenAI-style message content from JSON to str or content blocks."""
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, list):
        return str(raw) if raw is not None else ""
    blocks: list[dict[str, Any]] = []
    for part in raw:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            blocks.append({"type": "text", "text": str(part.get("text", "") if part.get("text") is not None else "")})
        elif ptype == "image_url":
            iu = part.get("image_url")
            url = ""
            if isinstance(iu, dict):
                url = str(iu.get("url", "") or "")
            elif isinstance(iu, str):
                url = iu
            if url.startswith("data:"):
                blocks.append({"type": "image_url", "image_url": {"url": url}})
        elif ptype == "input_audio":
            ia = part.get("input_audio")
            if isinstance(ia, dict) and isinstance(ia.get("data"), str):
                blocks.append({
                    "type": "input_audio",
                    "input_audio": {
                        "data": ia["data"],
                        "format": str(ia.get("format") or "wav"),
                    },
                })
        elif ptype == "kb_file":
            kbf = part.get("kb_file")
            if isinstance(kbf, dict) and isinstance(kbf.get("data"), str):
                blocks.append({
                    "type": "kb_file",
                    "kb_file": {
                        "filename": str(kbf.get("filename") or "file.bin"),
                        "mime": str(kbf.get("mime") or "application/octet-stream"),
                        "data": kbf["data"],
                    },
                })
    if not blocks:
        return ""
    if len(blocks) == 1 and blocks[0].get("type") == "text":
        return str(blocks[0].get("text", ""))
    return blocks


def _content_preview_for_log(content: Any, limit: int = 120) -> str:
    coerced = _coerce_api_user_content(content) if not isinstance(content, str) else content
    if isinstance(coerced, str):
        return coerced[:limit]
    parts: list[str] = []
    for b in coerced:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(str(b.get("text", "")))
    text = " ".join(parts)
    if any(
        isinstance(b, dict)
        and b.get("type") in ("image_url", "input_audio", "kb_file")
        for b in coerced
    ):
        text = (text + " " if text else "") + "[attachments]"
    return text[:limit]


def _parse_data_url(url: str) -> tuple[bytes, str]:
    """Decode a data: URL into raw bytes and MIME type."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return b"", ""
    try:
        comma = url.index(",")
    except ValueError:
        return b"", ""
    header = url[5:comma]
    payload = url[comma + 1 :]
    mime = "application/octet-stream"
    for segment in header.split(";"):
        seg = segment.strip()
        if not seg or seg.lower() == "base64":
            continue
        if "/" in seg:
            mime = seg
            break
    try:
        raw = base64.b64decode(payload, validate=False)
    except Exception:
        return b"", ""
    return raw, mime


def _image_ext_for_mime(mime: str) -> str:
    m = (mime or "").lower()
    if "png" in m:
        return ".png"
    if "jpeg" in m or "jpg" in m:
        return ".jpg"
    if "gif" in m:
        return ".gif"
    if "webp" in m:
        return ".webp"
    return ".img"


def _format_size_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.0f}"


async def _stream_multipart_part_to_path(part: Any, dest: Path, limit: int) -> int:
    """Stream a multipart file part to disk without holding the whole body in RAM."""
    total = 0
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await part.read_chunk(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise ValueError(
                        f"Файл слишком большой (больше {_format_size_mb(limit)} МБ). "
                        f"Максимум — {_format_size_mb(limit)} МБ."
                    )
                out.write(chunk)
    except Exception:
        try:
            if dest.is_file():
                dest.unlink()
        except OSError:
            pass
        raise
    return total


def _resolve_max_upload_bytes(max_upload_mb: int | None) -> tuple[int, int]:
    """Return (hard_mb, hard_bytes) from config or defaults."""
    mb = DEFAULT_MAX_UPLOAD_HARD_MB if max_upload_mb is None else int(max_upload_mb)
    if mb < 1:
        mb = 1
    return mb, mb * 1024 * 1024


def _web_upload_dir(workspace: Path, session_id: str) -> Path:
    sub = safe_filename(session_id)[:80] or "default"
    return ensure_dir(workspace.resolve() / "uploads" / "web" / sub)


def _persist_web_uploads(
    workspace: Path,
    session_id: str,
    coerced: str | list[dict[str, Any]],
) -> tuple[str, list[str]]:
    """Save web UI uploads under workspace/uploads/web/<session>/.

    Returns plaintext/caption for the agent plus absolute paths for image ``media``
    (same mechanism as other channels). Audio/docs are persisted and referenced in text.
    """
    if isinstance(coerced, str):
        return coerced, []

    ws = workspace.resolve()
    upload_dir = _web_upload_dir(ws, session_id)
    tag = uuid.uuid4().hex[:10]

    text_parts: list[str] = []
    image_paths: list[str] = []
    attachment_paths: list[str] = []  # audio + PDF/Office/прочий бинарник (kb_file)

    for i, block in enumerate(coerced):
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            t = str(block.get("text", "") or "")
            m = re.match(r"^\s*---\s*(.+?)\s*---\s*\n", t, flags=re.DOTALL)
            if m:
                fname = safe_filename(m.group(1).strip())[:160]
                rest = t[m.end() :]
                if fname and rest.strip():
                    fp = upload_dir / fname
                    try:
                        fp.write_text(rest, encoding="utf-8")
                        rp = fp.resolve()
                        try:
                            rp.relative_to(ws)
                            logger.info("Web UI saved text attachment to {}", rp)
                        except ValueError:
                            pass
                    except OSError as e:
                        logger.warning("Failed to write text upload {}: {}", fp, e)
            text_parts.append(t)
        elif btype == "image_url":
            iu = block.get("image_url")
            url = ""
            if isinstance(iu, dict):
                url = str(iu.get("url", "") or "")
            elif isinstance(iu, str):
                url = iu
            if not url.startswith("data:"):
                continue
            raw, mime = _parse_data_url(url)
            if not raw:
                logger.warning("Skipping web image block {} (decode failed or empty)", i)
                continue
            ext = _image_ext_for_mime(mime)
            fn = f"{tag}_{i}_image{ext}"
            path = upload_dir / fn
            try:
                path.write_bytes(raw)
            except OSError as e:
                logger.warning("Failed to write web upload {}: {}", path, e)
                continue
            resolved = path.resolve()
            try:
                resolved.relative_to(ws)
            except ValueError:
                logger.warning("Upload path outside workspace, dropping {}", resolved)
                continue
            image_paths.append(str(resolved))
            logger.info("Web UI saved image to {}", resolved)
        elif btype == "input_audio":
            ia = block.get("input_audio")
            if not isinstance(ia, dict) or not isinstance(ia.get("data"), str):
                continue
            try:
                raw = base64.b64decode(ia["data"], validate=False)
            except Exception:
                continue
            fmt = str(ia.get("format") or "wav").lower().strip(".")
            ext = "." + fmt if fmt else ".bin"
            fn = f"{tag}_{i}_audio{ext}"
            path = upload_dir / fn
            try:
                path.write_bytes(raw)
            except OSError as e:
                logger.warning("Failed to write web audio {}: {}", path, e)
                continue
            resolved = path.resolve()
            try:
                resolved.relative_to(ws)
            except ValueError:
                continue
            attachment_paths.append(str(resolved))
            logger.info("Web UI saved audio to {}", resolved)
        elif btype == "kb_file":
            kbf = block.get("kb_file")
            if not isinstance(kbf, dict) or not isinstance(kbf.get("data"), str):
                continue
            try:
                raw = base64.b64decode(kbf["data"], validate=False)
            except Exception:
                continue
            if not raw:
                logger.warning("Skipping web kb_file block {} (empty decode)", i)
                continue
            mime = str(kbf.get("mime") or "application/octet-stream").split(";")[0].strip()
            orig_name = Path(str(kbf.get("filename") or "file.bin")).name
            base = safe_filename(orig_name)[:160] or "file.bin"
            if "." not in base:
                guess = mimetypes.guess_extension(mime) or ".bin"
                base = f"{base}{guess}"
            fn = f"{tag}_{i}_{base}"
            path = upload_dir / fn
            try:
                path.write_bytes(raw)
            except OSError as e:
                logger.warning("Failed to write web document {}: {}", path, e)
                continue
            resolved = path.resolve()
            try:
                resolved.relative_to(ws)
            except ValueError:
                logger.warning("Upload path outside workspace, dropping {}", resolved)
                continue
            attachment_paths.append(str(resolved))
            logger.info("Web UI saved document to {} ({})", resolved, mime)

    user_text = "\n\n".join(t for t in text_parts if str(t).strip()).strip()

    if attachment_paths:
        note = "\n\n".join(f"[Файл сохранён в workspace: {p}]" for p in attachment_paths)
        user_text = (user_text + "\n\n" + note).strip() if user_text else note.strip()

    if image_paths:
        if not user_text:
            user_text = "Please describe or analyze the image(s) above."
        return user_text, image_paths

    if not user_text:
        user_text = "…"
    return user_text, []


def _strip_stored_user_text(text: str) -> str:
    """Hide LLM-only prefixes from session history shown in the web UI."""
    if text.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
        parts = text.split("\n\n", 1)
        text = parts[1] if len(parts) > 1 else ""
    return AgentLoop._strip_linked_accounts(text)


def _ui_text_from_stored_content(content: Any) -> str:
    """Session history → plain text for the web UI."""
    if isinstance(content, str):
        return _strip_stored_user_text(content)
    if not isinstance(content, list):
        return str(content)
    lines: list[str] = []
    for b in content:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t in ("text", "input_text"):
            lines.append(_strip_stored_user_text(str(b.get("text", ""))))
        elif t == "image_url":
            lines.append("[изображение]")
        elif t == "input_audio":
            lines.append("[аудио]")
        else:
            lines.append("[вложение]")
    return "\n".join(lines) if lines else "[сложное сообщение]"


_SESSION_TITLE_MAX = 50
_SESSION_TITLE_CUSTOM_MAX = 120


def _truncate_session_title(text: str, max_len: int = _SESSION_TITLE_MAX) -> str:
    """Collapse whitespace and truncate for sidebar labels."""
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


def derive_session_title(
    messages: list[dict[str, Any]] | None,
    metadata: dict[str, Any] | None = None,
    *,
    max_len: int = _SESSION_TITLE_MAX,
) -> str:
    """
    Human-readable dialog title for the web sidebar.

    Prefers a custom ``metadata["title"]``; otherwise the first user message
    (UI-stripped). Empty string if neither is available (UI falls back to date).
    """
    meta = metadata or {}
    custom = str(meta.get("title") or "").strip()
    if custom:
        return _truncate_session_title(custom, max_len=_SESSION_TITLE_CUSTOM_MAX)

    for msg in messages or []:
        if msg.get("role") != "user":
            continue
        raw = _ui_text_from_stored_content(msg.get("content")).strip()
        title = _truncate_session_title(raw, max_len=max_len)
        if title:
            return title
        break
    return ""


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

async def handle_chat_completions(request: web.Request) -> web.Response:
    """POST /v1/chat/completions"""

    # --- Parse body ---
    try:
        body = await request.json()
    except Exception as exc:
        logger.warning("POST /v1/chat/completions: JSON parse failed: {}", exc)
        return _error_json(400, "Invalid JSON body")

    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        return _error_json(400, "Only a single user message is supported")

    # Stream not yet supported
    if body.get("stream", False):
        return _error_json(400, "stream=true is not supported yet. Set stream=false or omit it.")

    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        return _error_json(400, "Only a single user message is supported")
    raw_content = message.get("content", "")
    coerced_content = _coerce_api_user_content(raw_content)

    agent_loop = request.app["agent_loop"]
    timeout_s: float = request.app.get("request_timeout", 120.0)
    model_name: str = request.app.get("model_name", "krabobot")
    if (requested_model := body.get("model")) and requested_model != model_name:
        return _error_json(400, f"Only configured model '{model_name}' is available")

    sid = str(body.get("session_id") or "default")
    session_key = f"api:{body['session_id']}" if body.get("session_id") else API_SESSION_KEY
    session_locks: dict[str, asyncio.Lock] = request.app["session_locks"]
    session_lock = session_locks.setdefault(session_key, asyncio.Lock())

    logger.info("API request session_key={} content={}", session_key, _content_preview_for_log(raw_content))

    fallback_empty = "I've completed processing but have no response to give."

    try:
        async with session_lock:
            try:
                sm = await agent_loop.session_manager_for_api(sid)
                final_content, media_paths = _persist_web_uploads(sm.workspace, sid, coerced_content)
                response = await asyncio.wait_for(
                    agent_loop.process_direct(
                        content=final_content,
                        media=media_paths if media_paths else None,
                        session_key=session_key,
                        channel="api",
                        chat_id=sid,
                        sender_id=sid,
                    ),
                    timeout=timeout_s,
                )
                response_text = _response_text(response)

                if not response_text or not response_text.strip():
                    logger.warning(
                        "Empty response for session {}, retrying",
                        session_key,
                    )
                    retry_response = await asyncio.wait_for(
                        agent_loop.process_direct(
                            content=final_content,
                            media=media_paths if media_paths else None,
                            session_key=session_key,
                            channel="api",
                            chat_id=sid,
                            sender_id=sid,
                        ),
                        timeout=timeout_s,
                    )
                    response_text = _response_text(retry_response)
                    if not response_text or not response_text.strip():
                        logger.warning(
                            "Empty response after retry for session {}, using fallback",
                            session_key,
                        )
                        response_text = fallback_empty

            except asyncio.TimeoutError:
                return _error_json(504, f"Request timed out after {timeout_s}s")
            except Exception:
                logger.exception("Error processing request for session {}", session_key)
                return _error_json(500, "Internal server error", err_type="server_error")
    except Exception:
        logger.exception("Unexpected API lock error for session {}", session_key)
        return _error_json(500, "Internal server error", err_type="server_error")

    return web.json_response(_chat_completion_response(response_text, model_name))


async def handle_models(request: web.Request) -> web.Response:
    """GET /v1/models"""
    model_name = request.app.get("model_name", "krabobot")
    return web.json_response({
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": 0,
                "owned_by": "krabobot",
            }
        ],
    })


async def handle_health(request: web.Request) -> web.Response:
    """GET /health"""
    soft = int(request.app.get("max_upload_soft_bytes", MAX_UPLOAD_SOFT_BYTES))
    hard = int(request.app.get("max_upload_bytes", DEFAULT_MAX_UPLOAD_HARD_MB * 1024 * 1024))
    hard_mb = int(request.app.get("max_upload_mb", DEFAULT_MAX_UPLOAD_HARD_MB))
    return web.json_response({
        "status": "ok",
        "maxUploadSoftMb": soft // (1024 * 1024),
        "maxUploadHardMb": hard_mb,
        "maxUploadSoftBytes": soft,
        "maxUploadHardBytes": hard,
    })


async def handle_web_uploads(request: web.Request) -> web.Response:
    """POST /v1/web/uploads — multipart file upload for web chat (video/docs).

    Streams each part to ``workspace/uploads/web/<session>/`` so large files
    (up to api.maxUploadMb, default 2 GiB) do not need to fit in RAM or JSON base64.
    """
    ctype = (request.content_type or "").lower()
    if "multipart/" not in ctype:
        return _error_json(400, "Ожидается multipart/form-data")

    agent_loop = request.app["agent_loop"]
    hard_limit = int(request.app.get("max_upload_bytes", DEFAULT_MAX_UPLOAD_HARD_MB * 1024 * 1024))
    session_id = "default"
    saved: list[dict[str, Any]] = []
    upload_dir: Path | None = None
    ws: Path | None = None
    tag = uuid.uuid4().hex[:10]
    file_index = 0

    async def _bind_session(sid: str) -> web.Response | None:
        nonlocal upload_dir, ws
        try:
            sm = await agent_loop.session_manager_for_api(sid)
        except Exception:
            logger.exception("Failed to resolve session for upload sid={}", sid)
            return _error_json(500, "Internal server error", err_type="server_error")
        upload_dir = _web_upload_dir(sm.workspace, sid)
        ws = sm.workspace.resolve()
        return None

    try:
        reader = await request.multipart()
        while True:
            part = await reader.next()
            if part is None:
                break
            name = part.name or ""
            if name == "session_id":
                session_id = (await part.text()).strip() or "default"
                err = await _bind_session(session_id)
                if err is not None:
                    return err
                continue
            if name not in ("files", "file"):
                await part.read(decode=False)
                continue

            if upload_dir is None or ws is None:
                err = await _bind_session(session_id)
                if err is not None:
                    return err

            filename = Path(part.filename or "file.bin").name
            mime = "application/octet-stream"
            hdr_ct = None
            if hasattr(part, "headers"):
                hdr_ct = part.headers.get("Content-Type")
            if hdr_ct and "/" in str(hdr_ct):
                mime = str(hdr_ct).split(";")[0].strip()
            elif mimetypes.guess_type(filename)[0]:
                mime = mimetypes.guess_type(filename)[0] or mime

            assert upload_dir is not None and ws is not None
            base = safe_filename(filename)[:160] or "file.bin"
            if "." not in base:
                guess = mimetypes.guess_extension(mime) or ".bin"
                base = f"{base}{guess}"
            fn = f"{tag}_{file_index}_{base}"
            path = upload_dir / fn
            try:
                size = await _stream_multipart_part_to_path(part, path, hard_limit)
            except ValueError as e:
                return _error_json(413, str(e))
            except OSError as e:
                logger.warning("Failed to write multipart upload {}: {}", path, e)
                return _error_json(
                    500, f"Не удалось сохранить файл: {filename}", err_type="server_error"
                )
            if size == 0:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
                return _error_json(400, f"Пустой файл: {filename}")

            resolved = path.resolve()
            try:
                resolved.relative_to(ws)
            except ValueError:
                logger.warning("Upload path outside workspace, dropping {}", resolved)
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
                return _error_json(500, "Путь загрузки вне workspace", err_type="server_error")

            saved.append({
                "filename": filename,
                "saved_as": fn,
                "path": str(resolved),
                "mime": mime,
                "size": size,
            })
            logger.info("Web UI multipart saved {} ({}, {} bytes)", resolved, mime, size)
            file_index += 1
    except Exception:
        logger.exception("Failed to parse multipart web upload")
        return _error_json(400, "Не удалось прочитать загрузку")

    if not saved:
        return _error_json(400, "Нет файлов в запросе")

    return web.json_response({"object": "upload.result", "data": saved})


async def handle_web_sessions_list(request: web.Request) -> web.Response:
    """GET /v1/web/sessions — list saved api:* chat sessions for the web UI."""
    agent_loop = request.app["agent_loop"]
    try:
        sm = await agent_loop.session_manager_for_api("default")
    except Exception:
        logger.exception("Failed to resolve API session manager")
        return _error_json(500, "Internal server error", err_type="server_error")
    items: list[dict[str, Any]] = []
    for info in sm.list_sessions():
        key = str(info.get("key") or "")
        if not key.startswith("api:"):
            continue
        sid = key.split(":", 1)[1]
        session = sm.get_or_create(key)
        preview = ""
        for m in reversed(session.messages):
            if m.get("role") == "user":
                preview = _ui_text_from_stored_content(m.get("content"))[:160]
                break
        title = derive_session_title(session.messages, session.metadata)
        items.append({
            "id": sid,
            "key": key,
            "updated_at": info.get("updated_at"),
            "created_at": info.get("created_at"),
            "message_count": len(session.messages),
            "preview": preview,
            "title": title,
        })
    return web.json_response({"object": "list", "data": items})


async def handle_web_sessions_patch(request: web.Request) -> web.Response:
    """PATCH /v1/web/sessions/{session_id} — rename dialog (stored in session metadata)."""
    session_id = unquote(request.match_info.get("session_id", "")).strip()
    if not session_id:
        return _error_json(400, "Missing session id")
    try:
        body = await request.json()
    except Exception:
        return _error_json(400, "Invalid JSON body")
    if not isinstance(body, dict) or "title" not in body:
        return _error_json(400, "Field 'title' is required")
    raw_title = body.get("title")
    if raw_title is None:
        title = ""
    elif isinstance(raw_title, str):
        title = " ".join(raw_title.split()).strip()
    else:
        return _error_json(400, "Field 'title' must be a string")
    if len(title) > _SESSION_TITLE_CUSTOM_MAX:
        title = title[:_SESSION_TITLE_CUSTOM_MAX].rstrip()

    key = f"api:{session_id}"
    agent_loop = request.app["agent_loop"]
    try:
        sm = await agent_loop.session_manager_for_api(session_id)
    except Exception:
        logger.exception("Failed to resolve API session manager for patch")
        return _error_json(500, "Internal server error", err_type="server_error")
    sm.invalidate(key)
    path = sm._get_session_path(key)
    if not path.exists():
        return _error_json(404, "Session not found")
    session = sm.get_or_create(key)
    if title:
        session.metadata["title"] = title
    else:
        session.metadata.pop("title", None)
    sm.save(session)
    derived = derive_session_title(session.messages, session.metadata)
    return web.json_response({
        "object": "session",
        "id": session_id,
        "title": derived,
    })


async def handle_web_sessions_delete(request: web.Request) -> web.Response:
    """DELETE /v1/web/sessions/{session_id}"""
    session_id = unquote(request.match_info.get("session_id", "")).strip()
    if not session_id:
        return _error_json(400, "Missing session id")
    key = f"api:{session_id}"
    agent_loop = request.app["agent_loop"]
    try:
        sm = await agent_loop.session_manager_for_api(session_id)
    except Exception:
        logger.exception("Failed to resolve API session manager for delete")
        return _error_json(500, "Internal server error", err_type="server_error")
    ok = sm.delete_session(key)
    locks: dict[str, asyncio.Lock] = request.app["session_locks"]
    locks.pop(key, None)
    if not ok:
        return _error_json(404, "Session not found")
    try:
        if await agent_loop.user_resolver.unlink_account("api", session_id):
            logger.info("Removed user_links entry for {}", key)
    except Exception:
        logger.exception("Failed to unlink api account after web session delete")
    return web.json_response({"object": "session.deleted", "id": session_id, "ok": True})


async def handle_web_session_messages(request: web.Request) -> web.Response:
    """GET /v1/web/sessions/{session_id}/messages — history for the web UI."""
    session_id = unquote(request.match_info.get("session_id", "")).strip()
    if not session_id:
        return _error_json(400, "Missing session id")
    key = f"api:{session_id}"
    agent_loop = request.app["agent_loop"]
    try:
        sm = await agent_loop.session_manager_for_api(session_id)
    except Exception:
        logger.exception("Failed to resolve API session manager for messages")
        return _error_json(500, "Internal server error", err_type="server_error")
    # Gateway may have appended notify messages on disk; drop stale in-memory cache.
    sm.invalidate(key)
    session = sm.get_or_create(key)
    out: list[dict[str, Any]] = []
    for m in session.messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content")
        # Match live web chat: only final assistant text is shown, not intermediate
        # tool-call turns (role=tool is already skipped above).
        if role == "assistant" and not content and m.get("tool_calls"):
            continue
        text = _ui_text_from_stored_content(content)
        if role == "assistant" and not text.strip():
            continue
        # Exec/subagent notices wake the model. They are not something the user typed.
        if role == "user" and is_background_wakeup(text):
            continue
        out.append({"role": role, "content": text})
    return web.json_response(
        {"object": "list", "data": out},
        headers={"Cache-Control": "no-store"},
    )


async def handle_web_config(_request: web.Request) -> web.Response:
    """GET /v1/web/config — redacted config for the settings UI."""
    try:
        payload = build_web_config_payload()
    except FileNotFoundError as e:
        return web.json_response(
            {"error": {"message": str(e), "type": "not_found", "code": 404}},
            status=404,
        )
    except Exception:
        logger.exception("Failed to build web config payload")
        return _error_json(500, "Internal server error", err_type="server_error")
    return web.json_response(payload)


async def handle_web_config_put(request: web.Request) -> web.Response:
    """PUT /v1/web/config — merge UI sections into config.json (backup before write)."""
    try:
        raw = await request.json()
    except Exception:
        return _error_json(400, "Invalid JSON body")

    if not isinstance(raw, dict):
        return _error_json(400, "Expected a JSON object")

    sections = {
        "core": raw.get("core") if isinstance(raw.get("core"), dict) else {},
        "channels": raw.get("channels") if isinstance(raw.get("channels"), dict) else {},
        "other": raw.get("other") if isinstance(raw.get("other"), dict) else {},
    }

    try:
        path_resolved, bak = save_web_config_sections(sections)
    except FileNotFoundError as e:
        return web.json_response(
            {"error": {"message": str(e), "type": "not_found", "code": 404}},
            status=404,
        )
    except ValidationError as e:
        return web.json_response(
            {
                "error": {
                    "message": "Configuration failed validation.",
                    "type": "validation_error",
                    "code": 422,
                    "detail": e.errors(),
                },
            },
            status=422,
        )
    except Exception:
        logger.exception("Failed to save web config")
        return _error_json(500, "Internal server error", err_type="server_error")

    return web.json_response(
        {
            "ok": True,
            "path": str(path_resolved),
            "backupCreated": str(bak),
        }
    )


async def handle_web_config_backups(_request: web.Request) -> web.Response:
    """GET /v1/web/config/backups — list automatic config backups newest first."""
    try:
        data = list_config_backups()
    except Exception:
        logger.exception("Failed to list web config backups")
        return _error_json(500, "Internal server error", err_type="server_error")
    return web.json_response({"object": "list", "data": data})


async def handle_web_config_restore(request: web.Request) -> web.Response:
    """POST /v1/web/config/restore — restore config.json from a named backup."""
    try:
        raw = await request.json()
    except Exception:
        return _error_json(400, "Invalid JSON body")

    if not isinstance(raw, dict):
        return _error_json(400, "Expected a JSON object")

    name = raw.get("backup") or raw.get("file") or ""
    if not isinstance(name, str):
        return _error_json(400, "backup must be a string")

    try:
        cfg_path, pre_restore_backup = restore_backup(name.strip())
    except ValueError as e:
        return _error_json(400, str(e))
    except FileNotFoundError:
        return _error_json(404, "Backup not found")
    except Exception:
        logger.exception("Failed to restore web config from backup")
        return _error_json(500, "Internal server error", err_type="server_error")

    return web.json_response(
        {
            "ok": True,
            "path": str(cfg_path),
            "restoredFrom": name.strip(),
            "previousBackedUpAs": str(pre_restore_backup),
        }
    )


async def handle_web_backup_download(request: web.Request) -> web.StreamResponse:
    """GET /v1/web/backup/download — .tar.gz with config + workspace (not media/models/history)."""
    from krabobot.cli.backup import create_archive
    from krabobot.config.loader import get_config_path

    cfg_path = get_config_path()
    if not cfg_path.is_file():
        return _error_json(404, "Config not found")

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    filename = f"krabobot-backup-{stamp}.tar.gz"
    tmp_dir = Path(tempfile.mkdtemp(prefix="krabobot-web-backup-"))
    archive_path = tmp_dir / filename

    try:
        # Same defaults as ``krabobot backup`` without --full / --with-*.
        create_archive(
            archive_path,
            config_path=cfg_path,
            include_workspace=True,
            include_media=False,
            include_models=False,
            include_history=False,
        )
    except FileNotFoundError as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return _error_json(404, str(e))
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.exception("Failed to create web backup archive")
        return _error_json(500, "Internal server error", err_type="server_error")

    size = archive_path.stat().st_size
    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "application/gzip",
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(size),
            "Cache-Control": "no-store",
        },
    )
    await resp.prepare(request)
    try:
        with archive_path.open("rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                await resp.write(chunk)
        await resp.write_eof()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return resp


# ---------------------------------------------------------------------------
# Voice channel (virtual: STT → agent → TTS)
# ---------------------------------------------------------------------------

def _voice_upload_dir(workspace: Path, device_id: str) -> Path:
    sub = safe_filename(device_id)[:80] or "device"
    return ensure_dir(workspace.resolve() / "uploads" / "voice" / sub)


async def handle_voice_turn(request: web.Request) -> web.Response:
    """POST /v1/voice/turn — multipart audio (+ optional instruct/files) → WAV reply.

    Form fields:
      - device_id (required): stable sender_id for channel ``voice``
      - audio (optional): speech clip for server STT
      - instruct (optional): text override/addition (meeting upload+instruct path)
      - client_state (optional): JSON string with mode/meeting/capabilities
      - files / file (optional): saved under the linked user's workspace

    Accepts ``multipart/form-data`` (preferred) or ``application/x-www-form-urlencoded``
    for text-only instruct turns.

    Successful WAV responses may include ``X-Krabobot-Voice-Actions`` (URL-encoded JSON
    list of ``{"action": "..."}``). JSON bodies (e.g. TTS unavailable) include ``actions``.
    """
    ctype = (request.content_type or "").lower()
    agent_loop: AgentLoop = request.app["agent_loop"]
    hard_limit = int(request.app.get("max_upload_bytes", DEFAULT_MAX_UPLOAD_HARD_MB * 1024 * 1024))
    timeout_s = float(request.app.get("request_timeout", 120.0))

    device_id = ""
    instruct = ""
    client_state_raw = ""
    async_meeting = False
    audio_path: Path | None = None
    media_paths: list[str] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="voice_turn_"))
    tag = uuid.uuid4().hex[:10]
    file_index = 0

    try:
        if "multipart/" in ctype:
            reader = await request.multipart()
            while True:
                part = await reader.next()
                if part is None:
                    break
                name = (part.name or "").strip()
                if name == "device_id":
                    device_id = (await part.text()).strip()
                    continue
                if name == "instruct":
                    instruct = (await part.text()).strip()
                    continue
                if name == "client_state":
                    client_state_raw = (await part.text()).strip()
                    continue
                if name == "async":
                    val = (await part.text()).strip().lower()
                    async_meeting = val in ("1", "true", "yes", "on")
                    continue
                if name == "audio":
                    filename = Path(part.filename or "audio.wav").name
                    base = safe_filename(filename)[:160] or "audio.wav"
                    audio_path = tmp_dir / f"{tag}_{base}"
                    try:
                        size = await _stream_multipart_part_to_path(part, audio_path, hard_limit)
                    except ValueError as e:
                        return _error_json(413, str(e))
                    if size <= 0:
                        return _error_json(400, "Пустой audio")
                    continue
                if name in ("files", "file"):
                    filename = Path(part.filename or "file.bin").name
                    mime = "application/octet-stream"
                    hdr_ct = part.headers.get("Content-Type") if hasattr(part, "headers") else None
                    if hdr_ct and "/" in str(hdr_ct):
                        mime = str(hdr_ct).split(";")[0].strip()
                    elif mimetypes.guess_type(filename)[0]:
                        mime = mimetypes.guess_type(filename)[0] or mime
                    base = safe_filename(filename)[:160] or "file.bin"
                    if "." not in base:
                        base = f"{base}{mimetypes.guess_extension(mime) or '.bin'}"
                    staging = tmp_dir / f"media_{file_index}_{base}"
                    try:
                        size = await _stream_multipart_part_to_path(part, staging, hard_limit)
                    except ValueError as e:
                        return _error_json(413, str(e))
                    if size <= 0:
                        return _error_json(400, f"Пустой файл: {filename}")
                    media_paths.append(str(staging))
                    file_index += 1
                    continue
                await part.read(decode=False)
        elif "application/x-www-form-urlencoded" in ctype or ctype in ("", "text/plain"):
            post = await request.post()
            device_id = str(post.get("device_id") or "").strip()
            instruct = str(post.get("instruct") or "").strip()
            client_state_raw = str(post.get("client_state") or "").strip()
            async_meeting = str(post.get("async") or "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        else:
            return _error_json(400, "Ожидается multipart/form-data")

        if not device_id:
            return _error_json(400, "device_id is required")

        stub = InboundMessage(
            channel="voice",
            sender_id=device_id,
            chat_id=device_id,
            content="",
        )
        await agent_loop._ensure_identity(stub)
        if not stub.user_id:
            return _error_json(
                403,
                "device_id is not linked; admin must POST /v1/web/users/{id}/links "
                "with {channel:\"voice\", sender_id:<device_id>}",
                err_type="forbidden",
            )
        registered = await agent_loop.user_resolver.is_registered("voice", device_id)
        if not registered:
            return _error_json(
                403,
                "device_id is not linked to a registered user",
                err_type="forbidden",
            )

        runtime = await agent_loop._runtime_for_message(stub)
        # Move staged media into the voice user's workspace.
        final_media: list[str] = []
        if media_paths:
            upload_dir = _voice_upload_dir(runtime.workspace, device_id)
            for src in media_paths:
                src_p = Path(src)
                dest = upload_dir / src_p.name
                try:
                    shutil.move(str(src_p), str(dest))
                    final_media.append(str(dest.resolve()))
                except OSError as e:
                    logger.warning("Failed to place voice upload {}: {}", src_p, e)
                    final_media.append(str(src_p.resolve()))

        if request.path.rstrip("/").endswith("/v1/voice/meeting"):
            async_meeting = True
            if not final_media or audio_path is not None:
                return _error_json(
                    400,
                    "POST /v1/voice/meeting expects multipart files= without audio",
                )

        if async_meeting and final_media and audio_path is None:
            from krabobot.api.voice_meeting import run_voice_meeting_job

            session_key = f"voice:{device_id}"
            pop_voice_actions(device_id)
            asyncio.create_task(
                run_voice_meeting_job(
                    agent_loop,
                    device_id=device_id,
                    instruct=instruct,
                    media_paths=list(final_media),
                    session_key=session_key,
                    timeout_s=timeout_s,
                )
            )
            return web.json_response(
                {
                    "object": "voice.meeting",
                    "status": "queued",
                    "device_id": device_id,
                    "message": "Совещание в очереди на обработку; результат придёт на почту администратора.",
                },
                status=202,
                headers={"X-Krabobot-Device-Id": device_id},
            )

        transcript = ""
        if audio_path is not None:
            from krabobot.config.loader import load_config

            try:
                audio_size = audio_path.stat().st_size
            except OSError:
                audio_size = 0
            logger.info(
                "voice turn STT input: device_id={} bytes={} file={}",
                device_id,
                audio_size,
                audio_path.name,
            )
            try:
                stt_cfg = load_config().stt
            except Exception:
                stt_cfg = None
            transcript, stt_err = await transcribe_audio(audio_path, stt=stt_cfg)
            if not transcript:
                return _error_json(
                    400,
                    stt_err or "Не удалось распознать речь",
                    err_type="stt_error",
                )

        content_parts = [p for p in (transcript, instruct) if p]
        # Meeting / chat-style attachments: path in content (like video), not STT.
        if final_media:
            content_parts.extend(_voice_media_content_notes(final_media))
        state_line = _format_voice_client_state_line(client_state_raw)
        if state_line:
            content_parts.append(state_line)
        if not content_parts and not final_media:
            return _error_json(400, "Нужен audio, instruct или файл")
        body = "\n\n".join(content_parts) if content_parts else "Обработай приложенные файлы."
        content = f"{_VOICE_REPLY_HINT}\n\n{body}"

        # Drop any stale actions from a previous failed turn for this device.
        pop_voice_actions(device_id)

        session_key = f"voice:{device_id}"
        try:
            result = await asyncio.wait_for(
                agent_loop.process_direct(
                    content,
                    session_key=session_key,
                    channel="voice",
                    chat_id=device_id,
                    sender_id=device_id,
                    media=final_media or None,
                ),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            pop_voice_actions(device_id)
            return _error_json(504, f"Request timed out after {timeout_s}s")
        except Exception:
            pop_voice_actions(device_id)
            logger.exception("voice turn failed for device_id={}", device_id)
            return _error_json(500, "Internal server error", err_type="server_error")

        actions = pop_voice_actions(device_id)
        reply = _response_text(result).strip()
        if not reply:
            reply = "[empty message]"

        action_names = {
            str(a.get("action") or "").strip()
            for a in actions
            if isinstance(a, dict)
        }
        skip_tts = "meeting_stop" in action_names

        # Header values must be latin-1-safe for aiohttp; percent-encode UTF-8.
        headers = {
            "X-Krabobot-Device-Id": device_id,
            "X-Krabobot-Transcript": quote(transcript[:2000], safe=""),
            "X-Krabobot-Reply": quote(reply[:2000], safe=""),
            "X-Krabobot-Voice-Actions": _encode_voice_actions_header(actions),
        }
        if skip_tts:
            return web.json_response(
                {
                    "object": "voice.turn",
                    "device_id": device_id,
                    "transcript": transcript,
                    "reply": reply,
                    "audio": None,
                    "actions": actions,
                },
                headers=headers,
            )

        from krabobot.config.loader import load_config

        try:
            tts_cfg = load_config().tts
        except Exception:
            tts_cfg = None
        wav = await synthesize_speech_wav(reply, tts=tts_cfg)
        if wav is None:
            return web.json_response(
                {
                    "object": "voice.turn",
                    "device_id": device_id,
                    "transcript": transcript,
                    "reply": reply,
                    "audio": None,
                    "error": "TTS unavailable",
                    "actions": actions,
                },
                headers=headers,
            )
        return web.Response(
            body=wav,
            content_type="audio/wav",
            headers=headers,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    agent_loop,
    model_name: str = "krabobot",
    request_timeout: float = 120.0,
    *,
    max_upload_mb: int | None = None,
) -> web.Application:
    """Create the aiohttp application.

    Args:
        agent_loop: An initialized AgentLoop instance.
        model_name: Model name reported in responses.
        request_timeout: Per-request timeout in seconds.
        max_upload_mb: Hard max for multipart web uploads (default 2048). Soft UI
            threshold between in-chat and folder-upload modes stays 50 MiB.
    """
    hard_mb, hard_bytes = _resolve_max_upload_bytes(max_upload_mb)
    client_max = max(MAX_REQUEST_BODY_BYTES, hard_bytes)
    app = web.Application(client_max_size=client_max, middlewares=[auth_middleware])
    app["agent_loop"] = agent_loop
    app["model_name"] = model_name
    app["request_timeout"] = request_timeout
    app["session_locks"] = {}  # per-user locks, keyed by session_key
    app["web_sessions"] = {}  # auth session token -> expiry unix ts
    app["max_upload_mb"] = hard_mb
    app["max_upload_bytes"] = hard_bytes
    app["max_upload_soft_bytes"] = MAX_UPLOAD_SOFT_BYTES

    static = web_static_dir()
    if static.is_dir():
        app.router.add_get("/", handle_chat_index)
        app.router.add_static("/static/", static, name="web_static")
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    app.router.add_post("/v1/voice/turn", handle_voice_turn)
    app.router.add_post("/v1/voice/meeting", handle_voice_turn)
    app.router.add_get("/v1/web/auth/status", handle_auth_status)
    app.router.add_post("/v1/web/auth/setup", handle_auth_setup)
    app.router.add_post("/v1/web/auth/login", handle_auth_login)
    app.router.add_post("/v1/web/auth/logout", handle_auth_logout)
    app.router.add_post("/v1/web/uploads", handle_web_uploads)
    app.router.add_get("/v1/web/sessions", handle_web_sessions_list)
    app.router.add_patch("/v1/web/sessions/{session_id}", handle_web_sessions_patch)
    app.router.add_delete("/v1/web/sessions/{session_id}", handle_web_sessions_delete)
    app.router.add_get("/v1/web/sessions/{session_id}/messages", handle_web_session_messages)
    app.router.add_get("/v1/web/config", handle_web_config)
    app.router.add_put("/v1/web/config", handle_web_config_put)
    app.router.add_get("/v1/web/config/backups", handle_web_config_backups)
    app.router.add_post("/v1/web/config/restore", handle_web_config_restore)
    app.router.add_get("/v1/web/backup/download", handle_web_backup_download)
    app.router.add_get("/v1/web/files", handle_web_file_download)
    app.router.add_get("/v1/web/users", handle_users_list)
    app.router.add_post("/v1/web/users", handle_users_create)
    app.router.add_get("/v1/web/users/{user_id}", handle_user_get)
    app.router.add_patch("/v1/web/users/{user_id}", handle_user_patch)
    app.router.add_delete("/v1/web/users/{user_id}", handle_user_delete)
    app.router.add_post("/v1/web/users/{user_id}/links", handle_user_link_add)
    app.router.add_delete("/v1/web/users/{user_id}/links", handle_user_link_remove)
    app.router.add_get("/v1/web/registrations", handle_registrations_list)
    app.router.add_post(
        "/v1/web/registrations/{request_id}/approve",
        handle_registration_approve,
    )
    app.router.add_post(
        "/v1/web/registrations/{request_id}/reject",
        handle_registration_reject,
    )
    return app
