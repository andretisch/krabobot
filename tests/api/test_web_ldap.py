"""Web LDAP login, admin link, and non-admin API gate. Directory is faked."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from krabobot.api.server import create_app
from krabobot.api.web_auth import hash_password
from krabobot.users.ldap_auth import LdapDirectoryUser
from krabobot.users.resolver import UserResolver

try:
    from aiohttp.test_utils import TestClient, TestServer

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytest_plugins = ("pytest_asyncio",)


class FakeDirectory:
    """In-memory directory. authenticate returns None on a bad password."""

    def __init__(self, users: dict[str, tuple[str, str]]) -> None:
        self.users = users
        self.calls: list[str] = []

    def authenticate(self, username: str, password: str) -> LdapDirectoryUser | None:
        self.calls.append(username)
        rec = self.users.get(username.strip().lower())
        if rec is None or rec[0] != password:
            return None
        login, _pw, display = username.strip().lower(), rec[0], rec[1]
        return LdapDirectoryUser(username=login, display_name=display, email="", dn=f"cn={login}")


def _config(ws: Path, *, ldap: dict | None, password: str | None = "secret12") -> dict:
    api: dict = {"host": "127.0.0.1", "port": 8900, "auth": {}}
    if password:
        api["auth"] = {"passwordHash": hash_password(password)}
    data: dict = {
        "agents": {"defaults": {"model": "x", "provider": "custom", "workspace": str(ws)}},
        "providers": {"custom": {"apiKey": "k", "apiBase": ""}},
        "channels": {},
        "api": api,
    }
    if ldap is not None:
        data["ldap"] = ldap
    return data


def _ldap_block(*, auto_create: bool = True) -> dict:
    return {
        "enabled": True,
        "host": "ldap.example.test",
        "port": 389,
        "baseDn": "DC=example,DC=test",
        "bindDn": "CN=svc,DC=example,DC=test",
        "bindPassword": "not-a-real-bind",
        "userFilter": "(sAMAccountName=%{username})",
        "autoCreate": auto_create,
    }


def _mock_agent(tmp_path: Path) -> MagicMock:
    from krabobot.session.manager import SessionManager

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    users_root = workspace / "users"
    users_root.mkdir(exist_ok=True)
    identity = workspace / "identity"
    identity.mkdir(exist_ok=True)
    mock = MagicMock()
    mock.process_direct = AsyncMock(return_value="ok")
    mock.session_manager_for_api = AsyncMock(return_value=SessionManager(tmp_path))
    mock.user_resolver = UserResolver(identity)
    mock._users_root = users_root
    mock._user_runtimes = {}
    return mock


def _patch_config(monkeypatch: pytest.MonkeyPatch, cfg: Path) -> None:
    target = cfg.resolve()

    def _path() -> Path:
        return target

    monkeypatch.setattr("krabobot.config.loader.get_config_path", _path)
    monkeypatch.setattr("krabobot.api.web_auth.get_config_path", _path)
    monkeypatch.setattr("krabobot.api.web_config.get_config_path", _path)


async def _start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data: dict, directory: FakeDirectory):
    krabot = tmp_path / ".krabobot"
    krabot.mkdir()
    cfg = krabot / "config.json"
    cfg.write_text(json.dumps(data), encoding="utf-8")
    _patch_config(monkeypatch, cfg)
    agent = _mock_agent(tmp_path)
    app = create_app(agent, model_name="t", request_timeout=5)
    app["ldap_authenticator"] = directory
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, agent


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_ldap_login_uses_linked_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = FakeDirectory({"alice": ("pw-ok", "Alice Directory")})
    client, agent = await _start(
        tmp_path,
        monkeypatch,
        _config(tmp_path / "ws", ldap=_ldap_block()),
        directory,
    )
    try:
        admin = await client.post("/v1/web/auth/login", json={"password": "secret12"})
        assert admin.status == 200, await admin.text()
        admin_token = (await admin.json())["token"]
        headers = {"Authorization": f"Bearer {admin_token}"}

        created = await client.post(
            "/v1/web/users",
            json={"display_name": "Owner"},
            headers=headers,
        )
        assert created.status == 201, await created.text()
        guest = await client.post(
            "/v1/web/users",
            json={"display_name": "Alice Local", "channel": "telegram", "sender_id": "423648236"},
            headers=headers,
        )
        assert guest.status == 201, await guest.text()
        guest_id = (await guest.json())["user_id"]
        linked = await client.post(
            f"/v1/web/users/{guest_id}/ldap",
            json={"username": r"CORP\Alice"},
            headers=headers,
        )
        assert linked.status == 200, await linked.text()
        body = await linked.json()
        assert body["ldap_usernames"] == ["alice"]
        assert "telegram:423648236" in body["accounts"]
        assert all(not str(a).startswith("ldap:") for a in body["accounts"])

        before = await agent.user_resolver.list_users()
        login = await client.post(
            "/v1/web/auth/login",
            json={"username": "Alice", "password": "pw-ok"},
        )
        assert login.status == 200, await login.text()
        logged = await login.json()
        assert logged["user_id"] == guest_id
        assert logged["created"] is False
        assert logged["is_admin"] is False
        after = await agent.user_resolver.list_users()
        assert len(after) == len(before)

        token = logged["token"]
        user_headers = {"Authorization": f"Bearer {token}"}
        denied = await client.get("/v1/web/config", headers=user_headers)
        assert denied.status == 403
        denied_users = await client.get("/v1/web/users", headers=user_headers)
        assert denied_users.status == 403
        denied_regs = await client.get("/v1/web/registrations", headers=user_headers)
        assert denied_regs.status == 403
        denied_put = await client.put("/v1/web/config", json={"core": {}}, headers=user_headers)
        assert denied_put.status == 403

        still_admin = await client.get("/v1/web/config", headers=headers)
        assert still_admin.status == 200
        still_users = await client.get("/v1/web/users", headers=headers)
        assert still_users.status == 200

        chat = await client.post(
            "/v1/chat/completions",
            json={"model": "t", "messages": [{"role": "user", "content": "hi"}]},
            headers=user_headers,
        )
        assert chat.status == 200, await chat.text()
        assert agent.process_direct.await_args.kwargs["user_id"] == guest_id
        sessions = await client.get("/v1/web/sessions", headers=user_headers)
        assert sessions.status == 200
    finally:
        await client.close()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_ldap_autocreate_uses_directory_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = FakeDirectory({"bob": ("pw-ok", "Иванов Иван")})
    client, agent = await _start(
        tmp_path,
        monkeypatch,
        _config(tmp_path / "ws", ldap=_ldap_block()),
        directory,
    )
    try:
        bad = await client.post(
            "/v1/web/auth/login",
            json={"username": "bob", "password": "wrong"},
        )
        assert bad.status == 401
        assert await agent.user_resolver.list_users() == []

        ok = await client.post(
            "/v1/web/auth/login",
            json={"username": "bob", "password": "pw-ok"},
        )
        assert ok.status == 200, await ok.text()
        logged = await ok.json()
        assert logged["created"] is True
        assert logged["is_admin"] is False
        assert logged["display_name"] == "Иванов Иван"
        users = await agent.user_resolver.list_users()
        assert len(users) == 1
        assert users[0]["user_id"] == logged["user_id"]
        assert users[0]["display_name"] == "Иванов Иван"
        assert users[0]["is_owner"] is False
        assert await agent.user_resolver.get_owner_user_id() in (None, "")

        status = await client.get(
            "/v1/web/auth/status",
            headers={"Authorization": f"Bearer {logged['token']}"},
        )
        payload = await status.json()
        assert payload["authenticated"] is True
        assert payload["is_admin"] is False
        assert payload["ldap_enabled"] is True
    finally:
        await client.close()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_ldap_without_autocreate_does_not_provision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = FakeDirectory({"carol": ("pw-ok", "Carol")})
    client, agent = await _start(
        tmp_path,
        monkeypatch,
        _config(tmp_path / "ws", ldap=_ldap_block(auto_create=False)),
        directory,
    )
    try:
        denied = await client.post(
            "/v1/web/auth/login",
            json={"username": "carol", "password": "pw-ok"},
        )
        assert denied.status == 403
        assert await agent.user_resolver.list_users() == []
    finally:
        await client.close()
