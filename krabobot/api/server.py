"""OpenAI-compatible HTTP API server for a fixed krabobot session.

Provides /v1/chat/completions and /v1/models endpoints.
All requests route to a single persistent API session.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import re
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from aiohttp import web
from loguru import logger

from krabobot.utils.helpers import ensure_dir, safe_filename


def web_static_dir() -> Path:
    """Directory with bundled static chat UI (index.html, app.js, …)."""
    return Path(__file__).resolve().parent.parent / "web" / "static"


async def handle_chat_index(_request: web.Request) -> web.StreamResponse:
    """Serve single-page chat at GET /."""
    idx = web_static_dir() / "index.html"
    if not idx.is_file():
        return web.Response(status=404, text="Web UI not found on server.")
    return web.FileResponse(idx)

API_SESSION_KEY = "api:default"
API_CHAT_ID = "default"
# aiohttp defaults to 1 MiB — base64 images in JSON exceed that and the body is truncated → JSON parse fails.
MAX_REQUEST_BODY_BYTES = 64 * 1024 * 1024


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
    sub = safe_filename(session_id)[:80] or "default"
    upload_dir = ensure_dir(ws / "uploads" / "web" / sub)
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


def _ui_text_from_stored_content(content: Any) -> str:
    """Session history → plain text for the web UI."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    lines: list[str] = []
    for b in content:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t in ("text", "input_text"):
            lines.append(str(b.get("text", "")))
        elif t == "image_url":
            lines.append("[изображение]")
        elif t == "input_audio":
            lines.append("[аудио]")
        else:
            lines.append("[вложение]")
    return "\n".join(lines) if lines else "[сложное сообщение]"


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
                        chat_id=API_CHAT_ID,
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
                            chat_id=API_CHAT_ID,
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
    return web.json_response({"status": "ok"})


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
        items.append({
            "id": sid,
            "key": key,
            "updated_at": info.get("updated_at"),
            "created_at": info.get("created_at"),
            "message_count": len(session.messages),
            "preview": preview,
        })
    return web.json_response({"object": "list", "data": items})


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
    session = sm.get_or_create(key)
    out: list[dict[str, Any]] = []
    for m in session.messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content")
        if role == "assistant" and not content and m.get("tool_calls"):
            text = "[инструменты…]"
        else:
            text = _ui_text_from_stored_content(content)
        out.append({"role": role, "content": text})
    return web.json_response({"object": "list", "data": out})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(agent_loop, model_name: str = "krabobot", request_timeout: float = 120.0) -> web.Application:
    """Create the aiohttp application.

    Args:
        agent_loop: An initialized AgentLoop instance.
        model_name: Model name reported in responses.
        request_timeout: Per-request timeout in seconds.
    """
    app = web.Application(client_max_size=MAX_REQUEST_BODY_BYTES)
    app["agent_loop"] = agent_loop
    app["model_name"] = model_name
    app["request_timeout"] = request_timeout
    app["session_locks"] = {}  # per-user locks, keyed by session_key

    static = web_static_dir()
    if static.is_dir():
        app.router.add_get("/", handle_chat_index)
        app.router.add_static("/static/", static, name="web_static")
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/web/sessions", handle_web_sessions_list)
    app.router.add_delete("/v1/web/sessions/{session_id}", handle_web_sessions_delete)
    app.router.add_get("/v1/web/sessions/{session_id}/messages", handle_web_session_messages)
    return app
