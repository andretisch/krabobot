"""Download a file that lives inside the signed-in user's workspace."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from aiohttp import web

_DENIED_ROOTS = frozenset({"identity", "sessions", "users", ".git"})


def resolve_workspace_download(workspace: Path, raw: str) -> Path | None:
    """Return a regular file inside *workspace*, or None.

    Absolute paths are accepted when they still resolve inside the workspace.
    ``identity``, ``sessions``, ``users`` and dot-directories stay closed.
    """
    text = (raw or "").strip().strip('"').strip("'")
    if not text or "\x00" in text or len(text) > 1024:
        return None
    if text.startswith("~/") or text.startswith("~\\"):
        text = str(Path.home() / text[2:])
    root = workspace.resolve()
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        target = candidate.resolve()
    except OSError:
        return None
    if not target.is_file():
        return None
    try:
        rel = target.relative_to(root)
    except ValueError:
        return None
    if not rel.parts:
        return None
    if rel.parts[0] in _DENIED_ROOTS or rel.parts[0].startswith("."):
        return None
    return target


def content_disposition(name: str) -> str:
    """Attachment header that keeps a Unicode filename."""
    cleaned = name.replace("\r", "").replace("\n", "").replace('"', "").strip()
    if not cleaned or cleaned in {".", ".."}:
        cleaned = "download"
    ascii_name = cleaned.encode("ascii", "replace").decode("ascii").replace("?", "_")
    if not ascii_name or ascii_name in {".", ".."}:
        ascii_name = "download"
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(cleaned)}"


def _error(status: int, message: str, err_type: str) -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": err_type, "code": status}},
        status=status,
    )


async def handle_web_file_download(request: web.Request) -> web.StreamResponse:
    """GET /v1/web/files?path=...&session_id=... — attachment from the web workspace."""
    raw = request.query.get("path") or ""
    sid = (request.query.get("session_id") or "default").strip() or "default"
    agent_loop = request.app["agent_loop"]
    try:
        sm = await agent_loop.session_manager_for_api(sid)
    except Exception:
        return _error(500, "Internal server error", "server_error")
    target = resolve_workspace_download(Path(sm.workspace), raw)
    if target is None:
        return _error(404, "File not found", "not_found")
    return web.FileResponse(
        target,
        headers={
            "Content-Disposition": content_disposition(target.name),
            "Cache-Control": "no-store",
        },
    )
