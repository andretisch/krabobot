"""Async meeting upload and email helper tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from krabobot.agent.loop import AgentLoop
from krabobot.api.server import create_app
from krabobot.api.voice_meeting import parse_meeting_agent_reply, resolve_owner_admin_email
from krabobot.bus.events import InboundMessage, OutboundMessage
from krabobot.bus.queue import MessageBus
from krabobot.config.schema import ToolsConfig
from krabobot.providers.base import GenerationSettings, LLMResponse

try:
    from aiohttp import FormData
    from aiohttp.test_utils import TestClient, TestServer

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


def _make_loop(tmp_path: Path) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=512)
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="ok", tool_calls=[]))
    return AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        restrict_to_workspace=True,
        multi_user_config=ToolsConfig.MultiUserConfig(enabled=True),
    )


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


def test_parse_meeting_agent_reply_extracts_subject_and_summary() -> None:
    text = (
        "SUBJECT: 2026-09-29 — планёрка (Иван, Мария)\n"
        "EMAIL_SUMMARY: Обсудили релиз и сроки.\n\n"
        "# Протокол\n\n- Решение: ship"
    )
    subject, summary, body = parse_meeting_agent_reply(text)
    assert "планёрка" in subject
    assert "Иван" in subject
    assert "релиз" in summary
    assert "# Протокол" in body


@pytest.mark.asyncio
async def test_resolve_owner_admin_email_from_linked_account(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    seed = InboundMessage(channel="cli", sender_id="owner", chat_id="d", content="")
    await loop._ensure_identity(seed)
    owner = seed.user_id
    assert owner
    await loop.user_resolver.link_account(owner, "email", "admin@example.com")
    assert await resolve_owner_admin_email(loop) == "admin@example.com"


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_voice_meeting_async_returns_202(tmp_path: Path, monkeypatch) -> None:
    krabot = tmp_path / ".krabobot"
    krabot.mkdir()
    cfg = krabot / "config.json"
    data = _minimal_config(tmp_path / "workspace")
    data["api"]["auth"] = {"adminToken": "tok-admin"}
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

    loop = _make_loop(tmp_path / "workspace")
    seed = InboundMessage(channel="cli", sender_id="owner", chat_id="d", content="")
    await loop._ensure_identity(seed)
    owner = seed.user_id
    assert owner
    await loop.user_resolver.link_account(owner, "voice", "dev-meet")

    loop.process_direct = AsyncMock()

    app = create_app(loop, model_name="t", request_timeout=5)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        with patch("krabobot.api.voice_meeting.run_voice_meeting_job", new_callable=AsyncMock):
            form = FormData()
            form.add_field("device_id", "dev-meet")
            form.add_field("instruct", "протокол")
            form.add_field("async", "1")
            form.add_field(
                "files",
                b"RIFF",
                filename="meet.wav",
                content_type="audio/wav",
            )
            r = await client.post(
                "/v1/voice/turn",
                data=form,
                headers={"Authorization": "Bearer tok-admin"},
            )
        assert r.status == 202, await r.text()
        body = await r.json()
        assert body.get("status") == "queued"
        loop.process_direct.assert_not_awaited()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_run_voice_meeting_job_emails_owner(tmp_path: Path) -> None:
    from krabobot.api.voice_meeting import run_voice_meeting_job

    loop = _make_loop(tmp_path)
    seed = InboundMessage(channel="cli", sender_id="owner", chat_id="d", content="")
    await loop._ensure_identity(seed)
    owner = seed.user_id
    assert owner
    await loop.user_resolver.link_account(owner, "email", "owner@corp.test")
    await loop.user_resolver.link_account(owner, "voice", "v1")

    loop.process_direct = AsyncMock(
        return_value=OutboundMessage(
            channel="voice",
            chat_id="v1",
            content=(
                "SUBJECT: 2026-09-29 — sync\n"
                "EMAIL_SUMMARY: Кратко.\n\n# Notes\n\nHello"
            ),
        )
    )
    loop.deliver_outbound = AsyncMock(return_value=True)

    media = [str(tmp_path / "meet.wav")]
    Path(media[0]).write_bytes(b"wav")

    await run_voice_meeting_job(
        loop,
        device_id="v1",
        instruct="summarize",
        media_paths=media,
        session_key="voice:v1",
        timeout_s=30.0,
    )

    loop.deliver_outbound.assert_awaited_once()
    out = loop.deliver_outbound.await_args.args[0]
    assert out.channel == "email"
    assert out.chat_id == "owner@corp.test"
    assert out.metadata.get("force_send") is True
    assert out.media and out.media[0].endswith("-notes.md")
