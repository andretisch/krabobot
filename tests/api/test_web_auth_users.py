"""Tests for web auth gate and /v1/web/users admin API."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from krabobot.api.server import create_app
from krabobot.api.web_auth import hash_password
from krabobot.users import UserResolver
from krabobot.users.user_md import update_user_md_display_name

try:
    from aiohttp.test_utils import TestClient, TestServer

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytest_plugins = ("pytest_asyncio",)


def _minimal_config(ws: Path) -> dict:
    return {
        "agents": {
            "defaults": {
                "model": "x",
                "provider": "custom",
                "workspace": str(ws),
            }
        },
        "providers": {
            "custom": {"apiKey": "k", "apiBase": ""},
            "openrouter": {"apiKey": "", "apiBase": None},
            "proxyapi": {"apiKey": "", "apiBase": None},
            "gptunnel": {"apiKey": "", "apiBase": None},
            "ollama": {"apiKey": "", "apiBase": None},
        },
        "channels": {},
        "api": {"host": "127.0.0.1", "port": 8900, "auth": {}},
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


async def _setup_and_login(client: TestClient, password: str = "secret12") -> str:
    r = await client.post("/v1/web/auth/setup", json={"password": password})
    assert r.status == 200, await r.text()
    data = await r.json()
    assert data.get("token")
    return str(data["token"])


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_auth_gate_requires_setup(tmp_path: Path, monkeypatch) -> None:
    krabot = tmp_path / ".krabobot"
    krabot.mkdir()
    cfg = krabot / "config.json"
    cfg.write_text(json.dumps(_minimal_config(tmp_path / "ws")), encoding="utf-8")
    monkeypatch.setattr(
        "krabobot.config.loader.get_config_path",
        lambda: cfg.resolve(),
        raising=True,
    )
    monkeypatch.setattr(
        "krabobot.api.web_auth.get_config_path",
        lambda: cfg.resolve(),
        raising=True,
    )

    app = create_app(_mock_agent(tmp_path), model_name="t", request_timeout=5)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        st = await client.get("/v1/web/auth/status")
        assert st.status == 200
        body = await st.json()
        assert body["configured"] is False
        assert body["authenticated"] is False

        denied = await client.get("/v1/web/config")
        assert denied.status == 401
        err = await denied.json()
        assert err["error"]["setup_required"] is True

        token = await _setup_and_login(client)
        ok = await client.get(
            "/v1/web/config",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert ok.status == 200

        st2 = await client.get(
            "/v1/web/auth/status",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert (await st2.json())["authenticated"] is True

        # Cookie session remains valid even if a stale Bearer is also sent
        chat = await client.post(
            "/v1/chat/completions",
            json={
                "model": "t",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"Authorization": "Bearer stale-not-a-session"},
        )
        assert chat.status == 200, await chat.text()
    finally:
        await client.close()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_auth_login_with_admin_token(tmp_path: Path, monkeypatch) -> None:
    krabot = tmp_path / ".krabobot"
    krabot.mkdir()
    cfg = krabot / "config.json"
    data = _minimal_config(tmp_path / "ws")
    data["api"]["auth"] = {"adminToken": "tok-abc-xyz-123456", "passwordHash": ""}
    cfg.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(
        "krabobot.config.loader.get_config_path",
        lambda: cfg.resolve(),
        raising=True,
    )
    monkeypatch.setattr(
        "krabobot.api.web_auth.get_config_path",
        lambda: cfg.resolve(),
        raising=True,
    )

    app = create_app(_mock_agent(tmp_path), model_name="t", request_timeout=5)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        bad = await client.get(
            "/v1/models",
            headers={"Authorization": "Bearer wrong"},
        )
        assert bad.status == 401

        # Direct Bearer with adminToken works without login session
        good = await client.get(
            "/v1/models",
            headers={"Authorization": "Bearer tok-abc-xyz-123456"},
        )
        assert good.status == 200

        login = await client.post(
            "/v1/web/auth/login",
            json={"token": "tok-abc-xyz-123456"},
        )
        assert login.status == 200
    finally:
        await client.close()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_users_crud_and_owner_protected(tmp_path: Path, monkeypatch) -> None:
    krabot = tmp_path / ".krabobot"
    krabot.mkdir()
    cfg = krabot / "config.json"
    data = _minimal_config(tmp_path / "workspace")
    data["api"]["auth"] = {"passwordHash": hash_password("secret12")}
    cfg.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(
        "krabobot.config.loader.get_config_path",
        lambda: cfg.resolve(),
        raising=True,
    )
    monkeypatch.setattr(
        "krabobot.api.web_auth.get_config_path",
        lambda: cfg.resolve(),
        raising=True,
    )

    agent = _mock_agent(tmp_path)
    app = create_app(agent, model_name="t", request_timeout=5)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        login = await client.post("/v1/web/auth/login", json={"password": "secret12"})
        assert login.status == 200
        token = (await login.json())["token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Create owner-ish first user
        r1 = await client.post(
            "/v1/web/users",
            json={"display_name": "Owner", "channel": "telegram", "sender_id": "1"},
            headers=headers,
        )
        assert r1.status == 201
        owner = await r1.json()
        assert owner["display_name"] == "Owner"
        assert owner["is_owner"] is True
        assert "telegram:1" in owner["accounts"]

        user_md = Path(owner["workspace"]) / "USER.md"
        assert user_md.is_file()
        assert "Owner" in user_md.read_text(encoding="utf-8")

        r2 = await client.post(
            "/v1/web/users",
            json={"display_name": "Alice", "channel": "vk", "sender_id": "99"},
            headers=headers,
        )
        assert r2.status == 201
        alice = await r2.json()
        assert alice["is_owner"] is False

        # Patch name → USER.md
        patch = await client.patch(
            f"/v1/web/users/{alice['user_id']}",
            json={"display_name": "Alice Updated", "tts_enabled": True},
            headers=headers,
        )
        assert patch.status == 200
        patched = await patch.json()
        assert patched["display_name"] == "Alice Updated"
        assert patched["tts_enabled"] is True
        alice_md = Path(alice["workspace"]) / "USER.md"
        assert "Alice Updated" in alice_md.read_text(encoding="utf-8")

        # Add/remove link (api channel visible)
        add_link = await client.post(
            f"/v1/web/users/{alice['user_id']}/links",
            json={"channel": "api", "sender_id": "sess-1"},
            headers=headers,
        )
        assert add_link.status == 200
        assert "api:sess-1" in (await add_link.json())["accounts"]

        rm_link = await client.delete(
            f"/v1/web/users/{alice['user_id']}/links",
            json={"channel": "api", "sender_id": "sess-1"},
            headers=headers,
        )
        assert rm_link.status == 200
        assert "api:sess-1" not in (await rm_link.json())["accounts"]

        # Cannot delete owner
        deny = await client.delete(
            f"/v1/web/users/{owner['user_id']}",
            headers=headers,
        )
        assert deny.status == 403

        # Delete alice + wipe workspace
        ws_alice = Path(alice["workspace"])
        assert ws_alice.is_dir()
        gone = await client.delete(
            f"/v1/web/users/{alice['user_id']}",
            headers=headers,
        )
        assert gone.status == 200
        assert not ws_alice.exists()

        listed = await client.get("/v1/web/users", headers=headers)
        ids = [u["user_id"] for u in (await listed.json())["data"]]
        assert owner["user_id"] in ids
        assert alice["user_id"] not in ids
    finally:
        await client.close()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_registration_approve_reject(tmp_path: Path, monkeypatch) -> None:
    krabot = tmp_path / ".krabobot"
    krabot.mkdir()
    cfg = krabot / "config.json"
    data = _minimal_config(tmp_path / "workspace")
    data["api"]["auth"] = {"passwordHash": hash_password("secret12")}
    cfg.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(
        "krabobot.config.loader.get_config_path",
        lambda: cfg.resolve(),
        raising=True,
    )
    monkeypatch.setattr(
        "krabobot.api.web_auth.get_config_path",
        lambda: cfg.resolve(),
        raising=True,
    )

    agent = _mock_agent(tmp_path)
    await agent.user_resolver.create_registration_request(
        "telegram", "555", note="please"
    )

    app = create_app(agent, model_name="t", request_timeout=5)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        login = await client.post("/v1/web/auth/login", json={"password": "secret12"})
        token = (await login.json())["token"]
        headers = {"Authorization": f"Bearer {token}"}

        regs = await client.get("/v1/web/registrations", headers=headers)
        assert regs.status == 200
        items = (await regs.json())["data"]
        assert len(items) == 1
        rid = items[0]["request_id"]

        appr = await client.post(
            f"/v1/web/registrations/{rid}/approve",
            headers=headers,
        )
        assert appr.status == 200
        assert (await appr.json())["user_id"]

        empty = await client.get("/v1/web/registrations", headers=headers)
        assert (await empty.json())["data"] == []
    finally:
        await client.close()


def test_update_user_md_creates_and_updates(tmp_path: Path) -> None:
    ws = tmp_path / "u1"
    p = update_user_md_display_name(ws, "Bob")
    assert p.is_file()
    assert "**Name**: Bob" in p.read_text(encoding="utf-8")
    update_user_md_display_name(ws, "Robert")
    assert "**Name**: Robert" in p.read_text(encoding="utf-8")
