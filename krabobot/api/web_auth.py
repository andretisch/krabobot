"""Web/API authentication: password hash, sessions, aiohttp middleware."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Any

from aiohttp import web

from krabobot.config.loader import get_config_path, load_config, save_config
from krabobot.config.schema import ApiAuthConfig

COOKIE_NAME = "krabobot_session"
SESSION_TTL_SECONDS = 7 * 24 * 3600
_PBKDF2_ITERATIONS = 200_000
_MIN_PASSWORD_LEN = 6


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


def _sessions(app: web.Application) -> dict[str, float]:
    store = app.get("web_sessions")
    if store is None:
        store = {}
        app["web_sessions"] = store
    return store  # type: ignore[return-value]


def issue_session(app: web.Application) -> str:
    """Create an in-memory session token and return it."""
    token = secrets.token_urlsafe(32)
    _sessions(app)[token] = time.time() + SESSION_TTL_SECONDS
    return token


def revoke_session(app: web.Application, token: str | None) -> None:
    if not token:
        return
    _sessions(app).pop(token, None)


def _purge_expired(app: web.Application) -> None:
    now = time.time()
    store = _sessions(app)
    dead = [k for k, exp in store.items() if exp <= now]
    for k in dead:
        store.pop(k, None)


def session_valid(app: web.Application, token: str | None) -> bool:
    if not token:
        return False
    _purge_expired(app)
    exp = _sessions(app).get(token)
    return bool(exp and exp > time.time())


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


def is_authenticated(request: web.Request) -> bool:
    """True if cookie/Bearer session or configured adminToken matches."""
    candidates = _auth_candidates(request)
    if not candidates:
        return False
    for token in candidates:
        if session_valid(request.app, token):
            return True
    auth = load_auth_config()
    admin = str(auth.admin_token or "").strip()
    if not admin:
        return False
    for token in candidates:
        # compare_digest requires equal length; skip mismatches safely
        if len(token) == len(admin) and hmac.compare_digest(token, admin):
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
    if is_authenticated(request):
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


async def handle_auth_status(request: web.Request) -> web.Response:
    """GET /v1/web/auth/status — whether auth is configured / current session ok."""
    configured = auth_is_configured()
    authenticated = is_authenticated(request) if configured else False
    return web.json_response(
        {
            "object": "auth.status",
            "configured": configured,
            "authenticated": authenticated,
        }
    )


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


async def handle_auth_login(request: web.Request) -> web.Response:
    """POST /v1/web/auth/login — {password} or {token} (adminToken)."""
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
    body = body or {}
    password = str(body.get("password") or "")
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
    resp = web.json_response({"object": "auth.login", "ok": True, "token": token})
    set_session_cookie(resp, token)
    return resp


async def handle_auth_logout(request: web.Request) -> web.Response:
    """POST /v1/web/auth/logout — clear cookie and revoke session."""
    token = extract_session_token(request)
    revoke_session(request.app, token)
    resp = web.json_response({"object": "auth.logout", "ok": True})
    clear_session_cookie(resp)
    return resp
