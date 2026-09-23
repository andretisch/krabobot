"""Admin user management handlers for /v1/web/users*."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from aiohttp import web
from loguru import logger

from krabobot.users.user_md import update_user_md_display_name


def _error(status: int, message: str, err_type: str = "invalid_request_error") -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": err_type, "code": status}},
        status=status,
    )


def _users_root(agent_loop: Any) -> Path:
    return Path(agent_loop._users_root)


def _workspace_for(agent_loop: Any, user_id: str) -> Path:
    return agent_loop.user_resolver.user_workspace(_users_root(agent_loop), user_id)


def _drop_user_runtime(agent_loop: Any, user_id: str) -> None:
    runtimes = getattr(agent_loop, "_user_runtimes", None)
    if isinstance(runtimes, dict):
        runtimes.pop(user_id, None)


def _parse_account_key(account: str) -> tuple[str, str] | None:
    if ":" not in account:
        return None
    channel, sender_id = account.split(":", 1)
    channel = channel.strip()
    sender_id = sender_id.strip()
    if not channel or not sender_id:
        return None
    return channel, sender_id


def _user_payload(agent_loop: Any, entry: dict[str, Any]) -> dict[str, Any]:
    uid = str(entry["user_id"])
    ws = _workspace_for(agent_loop, uid)
    return {
        "user_id": uid,
        "display_name": entry.get("display_name") or "",
        "workspace": str(ws),
        "is_owner": bool(entry.get("is_owner")),
        "tts_enabled": bool(entry.get("tts_enabled")),
        "accounts": list(entry.get("accounts") or []),
    }


async def handle_users_list(request: web.Request) -> web.Response:
    """GET /v1/web/users"""
    agent_loop = request.app["agent_loop"]
    users = await agent_loop.user_resolver.list_users()
    data = [_user_payload(agent_loop, u) for u in users]
    return web.json_response({"object": "list", "data": data})


async def handle_users_create(request: web.Request) -> web.Response:
    """POST /v1/web/users — {display_name?, channel?, sender_id?}"""
    agent_loop = request.app["agent_loop"]
    try:
        body = await request.json()
    except Exception:
        return _error(400, "Invalid JSON body")
    body = body or {}
    display_name = str(body.get("display_name") or "").strip()
    channel = str(body.get("channel") or "").strip()
    sender_id = str(body.get("sender_id") or "").strip()
    if (channel and not sender_id) or (sender_id and not channel):
        return _error(400, "channel and sender_id must be provided together")

    user_id = await agent_loop.user_resolver.create_user(
        display_name=display_name,
        channel=channel,
        sender_id=sender_id,
    )
    if display_name:
        update_user_md_display_name(_workspace_for(agent_loop, user_id), display_name)

    users = await agent_loop.user_resolver.list_users()
    entry = next((u for u in users if u["user_id"] == user_id), None)
    if entry is None:
        entry = {
            "user_id": user_id,
            "display_name": display_name,
            "accounts": ([f"{channel}:{sender_id}"] if channel and sender_id else []),
            "tts_enabled": False,
            "is_owner": False,
        }
    return web.json_response(
        {"object": "user", **_user_payload(agent_loop, entry)},
        status=201,
    )


async def handle_user_get(request: web.Request) -> web.Response:
    """GET /v1/web/users/{user_id}"""
    agent_loop = request.app["agent_loop"]
    user_id = request.match_info.get("user_id", "").strip()
    if not user_id:
        return _error(400, "Missing user id")
    if not await agent_loop.user_resolver.user_exists(user_id):
        return _error(404, "User not found", err_type="not_found")
    users = await agent_loop.user_resolver.list_users()
    entry = next((u for u in users if u["user_id"] == user_id), None)
    if entry is None:
        return _error(404, "User not found", err_type="not_found")
    return web.json_response({"object": "user", **_user_payload(agent_loop, entry)})


async def handle_user_patch(request: web.Request) -> web.Response:
    """PATCH /v1/web/users/{user_id} — {display_name?, tts_enabled?}"""
    agent_loop = request.app["agent_loop"]
    user_id = request.match_info.get("user_id", "").strip()
    if not user_id:
        return _error(400, "Missing user id")
    if not await agent_loop.user_resolver.user_exists(user_id):
        return _error(404, "User not found", err_type="not_found")
    try:
        body = await request.json()
    except Exception:
        return _error(400, "Invalid JSON body")
    body = body or {}

    if "display_name" in body:
        display_name = str(body.get("display_name") or "").strip()
        await agent_loop.user_resolver.set_display_name(user_id, display_name)
        update_user_md_display_name(_workspace_for(agent_loop, user_id), display_name)

    if "tts_enabled" in body:
        await agent_loop.user_resolver.set_tts_enabled(user_id, bool(body.get("tts_enabled")))

    users = await agent_loop.user_resolver.list_users()
    entry = next((u for u in users if u["user_id"] == user_id), None)
    if entry is None:
        return _error(404, "User not found", err_type="not_found")
    return web.json_response({"object": "user", **_user_payload(agent_loop, entry)})


async def handle_user_delete(request: web.Request) -> web.Response:
    """DELETE /v1/web/users/{user_id} — unlink accounts, wipe workspace; owner blocked."""
    agent_loop = request.app["agent_loop"]
    user_id = request.match_info.get("user_id", "").strip()
    if not user_id:
        return _error(400, "Missing user id")
    if not await agent_loop.user_resolver.user_exists(user_id):
        return _error(404, "User not found", err_type="not_found")

    ok, err = await agent_loop.user_resolver.delete_user(user_id)
    if not ok:
        if err == "cannot_delete_owner":
            return _error(403, "Cannot delete owner", err_type="forbidden")
        return _error(400, err or "Delete failed")

    ws = _workspace_for(agent_loop, user_id)
    try:
        if ws.is_dir():
            shutil.rmtree(ws)
            logger.info("Wiped user workspace {}", ws)
    except OSError:
        logger.exception("Failed to wipe workspace {}", ws)
        return _error(500, "User unlinked but workspace wipe failed", err_type="server_error")

    _drop_user_runtime(agent_loop, user_id)
    return web.json_response({"object": "user.deleted", "id": user_id, "ok": True})


async def handle_user_link_add(request: web.Request) -> web.Response:
    """POST /v1/web/users/{user_id}/links — {channel, sender_id}"""
    agent_loop = request.app["agent_loop"]
    user_id = request.match_info.get("user_id", "").strip()
    if not user_id:
        return _error(400, "Missing user id")
    if not await agent_loop.user_resolver.user_exists(user_id):
        return _error(404, "User not found", err_type="not_found")
    try:
        body = await request.json()
    except Exception:
        return _error(400, "Invalid JSON body")
    body = body or {}
    channel = str(body.get("channel") or "").strip()
    sender_id = str(body.get("sender_id") or "").strip()
    if not channel or not sender_id:
        return _error(400, "channel and sender_id are required")
    await agent_loop.user_resolver.link_account(user_id, channel, sender_id)
    users = await agent_loop.user_resolver.list_users()
    entry = next((u for u in users if u["user_id"] == user_id), None)
    if entry is None:
        return _error(404, "User not found", err_type="not_found")
    return web.json_response({"object": "user", **_user_payload(agent_loop, entry)})


async def handle_user_link_remove(request: web.Request) -> web.Response:
    """DELETE /v1/web/users/{user_id}/links — {channel, sender_id} or account=channel:id"""
    agent_loop = request.app["agent_loop"]
    user_id = request.match_info.get("user_id", "").strip()
    if not user_id:
        return _error(400, "Missing user id")
    if not await agent_loop.user_resolver.user_exists(user_id):
        return _error(404, "User not found", err_type="not_found")

    channel = ""
    sender_id = ""
    if request.can_read_body:
        try:
            body = await request.json()
            if isinstance(body, dict):
                channel = str(body.get("channel") or "").strip()
                sender_id = str(body.get("sender_id") or "").strip()
                account = str(body.get("account") or "").strip()
                if account and (not channel or not sender_id):
                    parsed = _parse_account_key(account)
                    if parsed:
                        channel, sender_id = parsed
        except Exception:
            pass
    if not channel or not sender_id:
        account_q = (request.rel_url.query.get("account") or "").strip()
        if account_q:
            parsed = _parse_account_key(account_q)
            if parsed:
                channel, sender_id = parsed
        else:
            channel = (request.rel_url.query.get("channel") or "").strip()
            sender_id = (request.rel_url.query.get("sender_id") or "").strip()
    if not channel or not sender_id:
        return _error(400, "channel and sender_id (or account) are required")

    linked = await agent_loop.user_resolver.lookup(channel, sender_id)
    if linked != user_id:
        return _error(404, "Link not found for this user", err_type="not_found")
    await agent_loop.user_resolver.unlink_account(channel, sender_id)

    users = await agent_loop.user_resolver.list_users()
    entry = next((u for u in users if u["user_id"] == user_id), None)
    if entry is None:
        # User may have had only this link and no prefs — still ok
        return web.json_response(
            {
                "object": "user",
                "user_id": user_id,
                "display_name": "",
                "workspace": str(_workspace_for(agent_loop, user_id)),
                "is_owner": await agent_loop.user_resolver.is_owner(user_id),
                "tts_enabled": False,
                "accounts": [],
            }
        )
    return web.json_response({"object": "user", **_user_payload(agent_loop, entry)})


async def handle_registrations_list(request: web.Request) -> web.Response:
    """GET /v1/web/registrations"""
    agent_loop = request.app["agent_loop"]
    items = await agent_loop.user_resolver.list_registration_requests()
    data = [
        {
            "request_id": r.request_id,
            "channel": r.channel,
            "sender_id": r.sender_id,
            "note": r.note,
            "created_at": r.created_at,
        }
        for r in items
    ]
    return web.json_response({"object": "list", "data": data})


async def handle_registration_approve(request: web.Request) -> web.Response:
    """POST /v1/web/registrations/{request_id}/approve"""
    agent_loop = request.app["agent_loop"]
    request_id = request.match_info.get("request_id", "").strip()
    decision = await agent_loop.user_resolver.approve_registration(request_id)
    if not decision.ok:
        code = 404 if decision.error == "request_not_found" else 400
        return _error(code, decision.error or "Approve failed")
    return web.json_response(
        {"object": "registration.approved", "ok": True, "user_id": decision.user_id}
    )


async def handle_registration_reject(request: web.Request) -> web.Response:
    """POST /v1/web/registrations/{request_id}/reject"""
    agent_loop = request.app["agent_loop"]
    request_id = request.match_info.get("request_id", "").strip()
    decision = await agent_loop.user_resolver.reject_registration(request_id)
    if not decision.ok:
        code = 404 if decision.error == "request_not_found" else 400
        return _error(code, decision.error or "Reject failed")
    return web.json_response({"object": "registration.rejected", "ok": True})
