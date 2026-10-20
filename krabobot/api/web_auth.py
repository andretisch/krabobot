"""Web/API authentication: password hash, sessions, aiohttp middleware."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiohttp import web
from loguru import logger

from krabobot.config.loader import get_config_path, load_config, save_config
from krabobot.config.schema import ApiAuthConfig, LdapConfig

COOKIE_NAME = "krabobot_session"
SESSION_TTL_SECONDS = 7 * 24 * 3600
_PBKDF2_ITERATIONS = 200_000
_MIN_PASSWORD_LEN = 6
# Chat stays open. These routes are owner/admin only (UI hide is not enough).
_ADMIN_ONLY_PREFIXES = (
    "/v1/web/config",
    "/v1/web/backup",
    "/v1/web/users",
    "/v1/web/registrations",
    "/v1/voice",
)


@dataclass(slots=True)
class WebPrincipal:
    """Authenticated web caller. Password and adminToken sessions are admins."""

    is_admin: bool
    user_id: str | None = None
    display_name: str = ""
    expires_at: float = 0.0


def hash_password(password: str) -> str:
    """Return a pbkdf2_sha256$… digest suitable for config.api.auth.passwordHash."""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        _PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time verify against a stored pbkdf2 hash."""
    if not password or not stored:
        return False
    parts = stored.split("$")
    if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
        return False
    try:
        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
        expected = bytes.fromhex(parts[3])
    except (ValueError, TypeError):
        return False
    got = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(got, expected)


def load_auth_config() -> ApiAuthConfig:
    """Load api.auth from the active config file."""
    cfg = load_config(get_config_path())
    return cfg.api.auth


def load_ldap_config() -> LdapConfig:
    """Load the LDAP block. Missing block is the disabled default."""
    return load_config(get_config_path()).ldap


def ldap_feature_enabled(cfg: LdapConfig | None = None) -> bool:
    """True when LDAP sign-in should be offered (host and base DN required)."""
    c = cfg if cfg is not None else load_ldap_config()
    return bool(c.enabled and str(c.host or "").strip() and str(c.base_dn or "").strip())


def auth_is_configured(auth: ApiAuthConfig | None = None) -> bool:
    """True when a password hash and/or admin token is present."""
    a = auth if auth is not None else load_auth_config()
    return bool(str(a.password_hash or "").strip() or str(a.admin_token or "").strip())


def save_password_hash(password_hash: str) -> None:
    """Persist passwordHash into config.json (keeps other fields)."""
    path = get_config_path()
    cfg = load_config(path)
    cfg.api.auth.password_hash = password_hash
    save_config(cfg, path)


def _sessions(app: web.Application) -> dict[str, WebPrincipal | float]:
    store = app.get("web_sessions")
    if store is None:
        store = {}
        app["web_sessions"] = store
    return store  # type: ignore[return-value]


def issue_session(
    app: web.Application,
    *,
    user_id: str | None = None,
    is_admin: bool = True,
    display_name: str = "",
) -> str:
    """Create an in-memory session token and return it.

    The default is an admin session (local password or adminToken). LDAP login
    passes the directory user's id and ``is_admin`` only when that user is owner.
    """
    token = secrets.token_urlsafe(32)
    _sessions(app)[token] = WebPrincipal(
        is_admin=is_admin,
        user_id=user_id,
        display_name=display_name,
        expires_at=time.time() + SESSION_TTL_SECONDS,
    )
    return token


def revoke_session(app: web.Application, token: str | None) -> None:
    if not token:
        return
    _sessions(app).pop(token, None)


def _expiry_of(raw: WebPrincipal | float) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    return float(raw.expires_at)


def _purge_expired(app: web.Application) -> None:
    now = time.time()
    store = _sessions(app)
    dead = [k for k, raw in store.items() if _expiry_of(raw) <= now]
    for k in dead:
        store.pop(k, None)


def _session_principal(app: web.Application, token: str | None) -> WebPrincipal | None:
    if not token:
        return None
    _purge_expired(app)
    raw = _sessions(app).get(token)
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        if raw <= time.time():
            return None
        return WebPrincipal(is_admin=True, expires_at=float(raw))
    if raw.expires_at <= time.time():
        return None
    return raw


def session_valid(app: web.Application, token: str | None) -> bool:
    return _session_principal(app, token) is not None


def extract_bearer(request: web.Request) -> str | None:
    header = request.headers.get("Authorization") or ""
    if header.lower().startswith("bearer "):
        return header[7:].strip() or None
    return None


def extract_session_token(request: web.Request) -> str | None:
    bearer = extract_bearer(request)
    if bearer:
        return bearer
    cookie = request.cookies.get(COOKIE_NAME)
    return cookie.strip() if cookie else None


def _auth_candidates(request: web.Request) -> list[str]:
    """Bearer and cookie tokens (Bearer first). Prefer any valid candidate."""
    out: list[str] = []
    bearer = extract_bearer(request)
    if bearer:
        out.append(bearer)
    cookie = request.cookies.get(COOKIE_NAME)
    if cookie:
        c = cookie.strip()
        if c and c not in out:
            out.append(c)
    return out


def resolve_principal(request: web.Request) -> WebPrincipal | None:
    """Cookie/Bearer session, or a matching adminToken (always an admin)."""
    candidates = _auth_candidates(request)
    if not candidates:
        return None
    for token in candidates:
        principal = _session_principal(request.app, token)
        if principal is not None:
            return principal
    auth = load_auth_config()
    admin = str(auth.admin_token or "").strip()
    if not admin:
        return None
    for token in candidates:
        # compare_digest requires equal length; skip mismatches safely
        if len(token) == len(admin) and hmac.compare_digest(token, admin):
            return WebPrincipal(is_admin=True, expires_at=time.time() + SESSION_TTL_SECONDS)
    return None


def is_authenticated(request: web.Request) -> bool:
    """True if cookie/Bearer session or configured adminToken matches."""
    return resolve_principal(request) is not None


def web_chat_user_id(request: web.Request) -> str | None:
    """LDAP (or other scoped) user whose web chats must stay in their workspace.

    Local password and adminToken sessions have no user id and keep the owner path.
    """
    principal = request.get("web_principal")
    if not isinstance(principal, WebPrincipal):
        principal = resolve_principal(request)
    if principal is None or not principal.user_id:
        return None
    return str(principal.user_id)


def _is_admin_only_path(path: str) -> bool:
    for prefix in _ADMIN_ONLY_PREFIXES:
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


def set_session_cookie(response: web.Response, token: str) -> None:
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="Lax",
        path="/",
    )


def clear_session_cookie(response: web.Response) -> None:
    response.del_cookie(COOKIE_NAME, path="/")


def _auth_public_path(path: str) -> bool:
    if path in {"/v1/web/auth/status", "/v1/web/auth/setup", "/v1/web/auth/login"}:
        return True
    return False


@web.middleware
async def auth_middleware(request: web.Request, handler):
    """Require auth for /v1/* except public auth endpoints."""
    path = request.path
    if not path.startswith("/v1/"):
        return await handler(request)
    if _auth_public_path(path):
        return await handler(request)
    principal = resolve_principal(request)
    if principal is not None:
        request["web_principal"] = principal
        if _is_admin_only_path(path) and not principal.is_admin:
            return web.json_response(
                {
                    "error": {
                        "message": "Недостаточно прав. Раздел доступен только администратору.",
                        "type": "forbidden",
                        "code": 403,
                    }
                },
                status=403,
            )
        return await handler(request)

    configured = auth_is_configured()
    body: dict[str, Any] = {
        "error": {
            "message": (
                "Требуется аутентификация. Задайте пароль веб-интерфейса или api.auth.adminToken."
                if not configured
                else "Требуется аутентификация."
            ),
            "type": "authentication_error",
            "code": 401,
            "setup_required": not configured,
        }
    }
    return web.json_response(body, status=401)


def _status_payload(request: web.Request) -> dict[str, Any]:
    configured = auth_is_configured()
    ldap_on = ldap_feature_enabled()
    principal = resolve_principal(request) if (configured or ldap_on) else None
    return {
        "object": "auth.status",
        "configured": configured,
        "authenticated": principal is not None,
        "ldap_enabled": ldap_on,
        "is_admin": bool(principal and principal.is_admin),
        "user_id": principal.user_id if principal else None,
        "display_name": principal.display_name if principal else "",
    }


async def handle_auth_status(request: web.Request) -> web.Response:
    """GET /v1/web/auth/status — whether auth is configured / current session ok."""
    return web.json_response(_status_payload(request))


async def handle_auth_setup(request: web.Request) -> web.Response:
    """POST /v1/web/auth/setup — first-run password (only when not yet configured)."""
    if auth_is_configured():
        return web.json_response(
            {
                "error": {
                    "message": "Пароль уже задан. Используйте вход.",
                    "type": "invalid_request_error",
                    "code": 409,
                }
            },
            status=409,
        )
    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {
                "error": {
                    "message": "Некорректное тело запроса (ожидается JSON).",
                    "type": "invalid_request_error",
                    "code": 400,
                }
            },
            status=400,
        )
    password = str((body or {}).get("password") or "")
    if len(password) < _MIN_PASSWORD_LEN:
        return web.json_response(
            {
                "error": {
                    "message": f"Пароль должен быть не короче {_MIN_PASSWORD_LEN} символов.",
                    "type": "invalid_request_error",
                    "code": 400,
                }
            },
            status=400,
        )
    save_password_hash(hash_password(password))
    token = issue_session(request.app)
    resp = web.json_response({"object": "auth.setup", "ok": True, "token": token})
    set_session_cookie(resp, token)
    return resp


def _json_error(status: int, message: str, err_type: str, **extra: Any) -> web.Response:
    body: dict[str, Any] = {"error": {"message": message, "type": err_type, "code": status}}
    body["error"].update(extra)
    return web.json_response(body, status=status)


def _directory_for(request: web.Request, cfg: LdapConfig):
    """Test double on the app, otherwise the ldap3 client."""
    custom = request.app.get("ldap_authenticator")
    if custom is not None:
        return custom
    from krabobot.users.ldap_directory import Ldap3Directory

    return Ldap3Directory(cfg)


async def _login_ldap(request: web.Request, username: str, password: str) -> web.Response:
    """Bind against the directory, then reuse a linked user or auto-provision."""
    cfg = load_ldap_config()
    if not ldap_feature_enabled(cfg):
        return _json_error(400, "Вход через LDAP отключён.", "invalid_request_error")
    if not password:
        return _json_error(401, "Неверное доменное имя или пароль.", "authentication_error")

    directory = _directory_for(request, cfg)
    try:
        directory_user = directory.authenticate(username, password)
    except Exception as exc:
        logger.warning("LDAP login error for {}: {}", username, type(exc).__name__)
        return _json_error(401, "Неверное доменное имя или пароль.", "authentication_error")
    if directory_user is None:
        logger.info("LDAP login failed for {}", username)
        return _json_error(401, "Неверное доменное имя или пароль.", "authentication_error")

    agent_loop = request.app["agent_loop"]
    resolver = agent_loop.user_resolver
    user_id, created = await resolver.resolve_ldap_identity(
        directory_user.username,
        display_name=directory_user.display_name,
        auto_create=cfg.auto_create,
    )
    if not user_id:
        return _json_error(
            403,
            "Доменная учётная запись не привязана, а автосоздание выключено.",
            "forbidden",
        )
    if created:
        try:
            from krabobot.users.user_md import update_user_md_display_name

            workspace = resolver.user_workspace(Path(agent_loop._users_root), user_id)
            update_user_md_display_name(workspace, directory_user.display_name)
        except Exception:
            logger.exception("Failed to write USER.md for LDAP user {}", user_id)
        logger.info("LDAP user provisioned username={} user_id={}", directory_user.username, user_id)
    else:
        logger.info("LDAP user signed in username={} user_id={}", directory_user.username, user_id)

    is_admin = await resolver.is_owner(user_id)
    display = (await resolver.get_display_name(user_id)) or directory_user.display_name
    token = issue_session(
        request.app,
        user_id=user_id,
        is_admin=is_admin,
        display_name=display,
    )
    resp = web.json_response(
        {
            "object": "auth.login",
            "ok": True,
            "token": token,
            "is_admin": is_admin,
            "user_id": user_id,
            "display_name": display,
            "created": created,
        }
    )
    set_session_cookie(resp, token)
    return resp


async def handle_auth_login(request: web.Request) -> web.Response:
    """POST /v1/web/auth/login — {password}, {token}, or {username, password} for LDAP."""
    try:
        body = await request.json()
    except Exception:
        return _json_error(400, "Некорректное тело запроса (ожидается JSON).", "invalid_request_error")
    body = body or {}
    username = str(body.get("username") or "").strip()
    password = str(body.get("password") or "")
    if username:
        return await _login_ldap(request, username, password)

    auth = load_auth_config()
    if not auth_is_configured(auth):
        return web.json_response(
            {
                "error": {
                    "message": "Пароль ещё не задан. Сначала выполните первичную настройку.",
                    "type": "invalid_request_error",
                    "code": 400,
                    "setup_required": True,
                }
            },
            status=400,
        )
    token_in = str(body.get("token") or "").strip()

    ok = False
    if password and auth.password_hash and verify_password(password, auth.password_hash):
        ok = True
    admin = str(auth.admin_token or "").strip()
    if not ok and token_in and admin and hmac.compare_digest(token_in, admin):
        ok = True
    if not ok:
        return web.json_response(
            {
                "error": {
                    "message": "Неверный пароль или API-ключ.",
                    "type": "authentication_error",
                    "code": 401,
                }
            },
            status=401,
        )

    token = issue_session(request.app)
    resp = web.json_response(
        {"object": "auth.login", "ok": True, "token": token, "is_admin": True}
    )
    set_session_cookie(resp, token)
    return resp


async def handle_auth_logout(request: web.Request) -> web.Response:
    """POST /v1/web/auth/logout — clear cookie and revoke session."""
    token = extract_session_token(request)
    revoke_session(request.app, token)
    resp = web.json_response({"object": "auth.logout", "ok": True})
    clear_session_cookie(resp)
    return resp
