"""Tests for virtual voice channel identity and POST /v1/voice/turn."""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from krabobot.agent.loop import AgentLoop
from krabobot.api.server import create_app
from krabobot.api.web_auth import hash_password
from krabobot.bus.events import InboundMessage, OutboundMessage
from krabobot.bus.queue import MessageBus
from krabobot.config.schema import ToolsConfig
from krabobot.providers.base import GenerationSettings, LLMResponse
from krabobot.session.manager import SessionManager
from krabobot.users import UserResolver

try:
    from aiohttp import FormData
    from aiohttp.test_utils import TestClient, TestServer

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytest_plugins = ("pytest_asyncio",)


def _make_loop(tmp_path: Path) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=512)
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="ok", tool_calls=[]))
    provider.chat_stream_with_retry = AsyncMock(
        return_value=LLMResponse(content="ok", tool_calls=[])
    )
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


def _tiny_wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 160)  # 10 ms silence
    return buf.getvalue()


@pytest.mark.asyncio
async def test_voice_does_not_auto_link_to_owner(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    # Seed owner via first non-voice channel
    owner_msg = InboundMessage(channel="cli", sender_id="owner", chat_id="direct", content="")
    await loop._ensure_identity(owner_msg)
    owner_id = owner_msg.user_id
    assert owner_id

    voice_msg = InboundMessage(
        channel="voice", sender_id="device-abc", chat_id="device-abc", content=""
    )
    await loop._ensure_identity(voice_msg)
    assert voice_msg.user_id is None
    assert await loop.user_resolver.lookup("voice", "device-abc") is None


@pytest.mark.asyncio
async def test_api_still_auto_links_to_owner(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    owner_msg = InboundMessage(channel="cli", sender_id="owner", chat_id="direct", content="")
    await loop._ensure_identity(owner_msg)
    owner_id = owner_msg.user_id
    assert owner_id

    api_msg = InboundMessage(channel="api", sender_id="web-sess", chat_id="web-sess", content="")
    await loop._ensure_identity(api_msg)
    assert api_msg.user_id == owner_id
    assert await loop.user_resolver.lookup("api", "web-sess") == owner_id


@pytest.mark.asyncio
async def test_voice_prelink_resolves_identity(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    owner_msg = InboundMessage(channel="cli", sender_id="owner", chat_id="direct", content="")
    await loop._ensure_identity(owner_msg)
    owner_id = owner_msg.user_id
    assert owner_id

    await loop.user_resolver.link_account(owner_id, "voice", "rpi-kitchen")
    voice_msg = InboundMessage(
        channel="voice", sender_id="rpi-kitchen", chat_id="rpi-kitchen", content=""
    )
    await loop._ensure_identity(voice_msg)
    assert voice_msg.user_id == owner_id


@pytest.mark.asyncio
async def test_local_outbound_includes_voice() -> None:
    assert "voice" in AgentLoop._LOCAL_OUTBOUND_CHANNELS


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_admin_can_link_voice_device(tmp_path: Path, monkeypatch) -> None:
    krabot = tmp_path / ".krabobot"
    krabot.mkdir()
    cfg = krabot / "config.json"
    data = _minimal_config(tmp_path / "workspace")
    data["api"]["auth"] = {"passwordHash": hash_password("secret12"), "adminToken": "tok-admin"}
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

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    users_root = workspace / "users"
    users_root.mkdir()
    identity = workspace / "identity"
    identity.mkdir()

    agent = MagicMock()
    agent.process_direct = AsyncMock(return_value="ok")
    agent.session_manager_for_api = AsyncMock(return_value=SessionManager(tmp_path))
    agent.user_resolver = UserResolver(identity)
    agent._users_root = users_root
    agent._user_runtimes = {}
    owner = await agent.user_resolver.resolve_or_create("cli", "owner")
    await agent.user_resolver.ensure_owner(owner)

    app = create_app(agent, model_name="t", request_timeout=5)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        headers = {"Authorization": "Bearer tok-admin"}
        r = await client.post(
            f"/v1/web/users/{owner}/links",
            json={"channel": "voice", "sender_id": "dev-1"},
            headers=headers,
        )
        assert r.status == 200, await r.text()
        body = await r.json()
        assert "voice:dev-1" in body["accounts"]
        assert await agent.user_resolver.lookup("voice", "dev-1") == owner
    finally:
        await client.close()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_voice_turn_unlinked_device_forbidden(tmp_path: Path, monkeypatch) -> None:
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
    # Ensure owner exists so voice does not become first-user bootstrap
    seed = InboundMessage(channel="cli", sender_id="owner", chat_id="d", content="")
    await loop._ensure_identity(seed)

    app = create_app(loop, model_name="t", request_timeout=5)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        form = FormData()
        form.add_field("device_id", "unknown-device")
        form.add_field("instruct", "привет")
        r = await client.post(
            "/v1/voice/turn",
            data=form,
            headers={"Authorization": "Bearer tok-admin"},
        )
        assert r.status == 403
    finally:
        await client.close()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_voice_turn_with_mocks(tmp_path: Path, monkeypatch) -> None:
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
    monkeypatch.setattr(
        "krabobot.config.loader.load_config",
        lambda: MagicMock(stt=MagicMock(
            sherpa_models_dir=str(tmp_path),
            sherpa_model_id="m",
            sherpa_num_threads=1,
            sherpa_provider="cpu",
        ), tts=MagicMock(
            sherpa_models_dir=str(tmp_path),
            sherpa_model_id="m",
            sherpa_speed=1.0,
            provider="sherpa_onnx",
        )),
        raising=True,
    )

    loop = _make_loop(tmp_path / "workspace")
    seed = InboundMessage(channel="cli", sender_id="owner", chat_id="d", content="")
    await loop._ensure_identity(seed)
    owner = seed.user_id
    assert owner
    await loop.user_resolver.link_account(owner, "voice", "pi-01")

    loop.process_direct = AsyncMock(
        return_value=OutboundMessage(channel="voice", chat_id="pi-01", content="Ответ бота")
    )

    fake_wav = _tiny_wav_bytes()

    async def _fake_stt(path, *, stt=None):
        return "привет мир", None

    async def _fake_tts(text, *, tts=None):
        assert "Ответ" in text or "бота" in text or text
        return fake_wav

    app = create_app(loop, model_name="t", request_timeout=5)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        with (
            patch("krabobot.api.server.transcribe_audio", side_effect=_fake_stt),
            patch("krabobot.api.server.synthesize_speech_wav", side_effect=_fake_tts),
        ):
            form = FormData()
            form.add_field("device_id", "pi-01")
            form.add_field(
                "audio",
                fake_wav,
                filename="clip.wav",
                content_type="audio/wav",
            )
            # Meeting path: optional file + instruct alongside audio
            form.add_field(
                "file",
                b"meeting notes",
                filename="notes.txt",
                content_type="text/plain",
            )
            form.add_field("instruct", "учти файл")
            r = await client.post(
                "/v1/voice/turn",
                data=form,
                headers={"Authorization": "Bearer tok-admin"},
            )
        assert r.status == 200, await r.text()
        assert r.headers.get("Content-Type", "").startswith("audio/wav")
        body = await r.read()
        assert body == fake_wav
        assert r.headers.get("X-Krabobot-Device-Id") == "pi-01"

        loop.process_direct.assert_awaited_once()
        call_kw = loop.process_direct.await_args.kwargs
        assert call_kw["channel"] == "voice"
        assert call_kw["sender_id"] == "pi-01"
        assert call_kw["chat_id"] == "pi-01"
        assert call_kw["session_key"] == "voice:pi-01"
        content = loop.process_direct.await_args.args[0]
        assert "привет мир" in content
        assert "учти файл" in content
        assert "без Markdown" in content
        assert "без эмодзи" in content
        media = call_kw.get("media") or []
        assert media and any("notes.txt" in p for p in media)
    finally:
        await client.close()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_voice_turn_client_state_and_actions_header(tmp_path: Path, monkeypatch) -> None:
    from urllib.parse import unquote

    from krabobot.agent.tools.voice import queue_voice_action

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
    monkeypatch.setattr(
        "krabobot.config.loader.load_config",
        lambda: MagicMock(
            stt=MagicMock(),
            tts=MagicMock(),
        ),
        raising=True,
    )

    loop = _make_loop(tmp_path / "workspace")
    seed = InboundMessage(channel="cli", sender_id="owner", chat_id="d", content="")
    await loop._ensure_identity(seed)
    owner = seed.user_id
    assert owner
    await loop.user_resolver.link_account(owner, "voice", "pi-state")

    async def _process_with_action(content, **kwargs):
        queue_voice_action("pi-state", "meeting_stop")
        return OutboundMessage(channel="voice", chat_id="pi-state", content="Стоп")

    loop.process_direct = AsyncMock(side_effect=_process_with_action)
    fake_wav = _tiny_wav_bytes()

    async def _fake_tts(text, *, tts=None):
        return fake_wav

    app = create_app(loop, model_name="t", request_timeout=5)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        with patch("krabobot.api.server.synthesize_speech_wav", side_effect=_fake_tts):
            form = FormData()
            form.add_field("device_id", "pi-state")
            form.add_field("instruct", "останови запись")
            form.add_field(
                "client_state",
                json.dumps(
                    {
                        "mode": "dialog",
                        "meeting": "recording",
                        "capabilities": [
                            "meeting_start",
                            "meeting_stop",
                            "end_dialog",
                            "run_test",
                        ],
                    }
                ),
            )
            r = await client.post(
                "/v1/voice/turn",
                data=form,
                headers={"Authorization": "Bearer tok-admin"},
            )
        assert r.status == 200, await r.text()
        assert r.headers.get("Content-Type", "").startswith("audio/wav")
        raw_actions = r.headers.get("X-Krabobot-Voice-Actions")
        assert raw_actions is not None
        actions = json.loads(unquote(raw_actions))
        assert actions == [{"action": "meeting_stop"}]

        content = loop.process_direct.await_args.args[0]
        assert "останови запись" in content
        assert "[voice client: mode=dialog, meeting=recording;" in content
        assert "available commands:" in content
        assert "run_test" in content
        assert "CALL the voice tool" in content
    finally:
        await client.close()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_voice_turn_tts_unavailable_includes_actions_json(tmp_path: Path, monkeypatch) -> None:
    from urllib.parse import unquote

    from krabobot.agent.tools.voice import queue_voice_action

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
    monkeypatch.setattr(
        "krabobot.config.loader.load_config",
        lambda: MagicMock(stt=MagicMock(), tts=MagicMock()),
        raising=True,
    )

    loop = _make_loop(tmp_path / "workspace")
    seed = InboundMessage(channel="cli", sender_id="owner", chat_id="d", content="")
    await loop._ensure_identity(seed)
    owner = seed.user_id
    assert owner
    await loop.user_resolver.link_account(owner, "voice", "pi-tts")

    async def _process_with_action(content, **kwargs):
        queue_voice_action("pi-tts", "end_dialog")
        return OutboundMessage(channel="voice", chat_id="pi-tts", content="Пока")

    loop.process_direct = AsyncMock(side_effect=_process_with_action)

    async def _no_tts(text, *, tts=None):
        return None

    app = create_app(loop, model_name="t", request_timeout=5)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        with patch("krabobot.api.server.synthesize_speech_wav", side_effect=_no_tts):
            form = FormData()
            form.add_field("device_id", "pi-tts")
            form.add_field("instruct", "закончи диалог")
            r = await client.post(
                "/v1/voice/turn",
                data=form,
                headers={"Authorization": "Bearer tok-admin"},
            )
        assert r.status == 200, await r.text()
        body = await r.json()
        assert body.get("actions") == [{"action": "end_dialog"}]
        assert body.get("error") == "TTS unavailable"
        raw_actions = r.headers.get("X-Krabobot-Voice-Actions")
        assert raw_actions is not None
        assert json.loads(unquote(raw_actions)) == [{"action": "end_dialog"}]
    finally:
        await client.close()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_voice_turn_empty_stt_returns_clear_error(tmp_path: Path, monkeypatch) -> None:
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
    monkeypatch.setattr(
        "krabobot.config.loader.load_config",
        lambda: MagicMock(stt=MagicMock(), tts=MagicMock()),
        raising=True,
    )

    loop = _make_loop(tmp_path / "workspace")
    seed = InboundMessage(channel="cli", sender_id="owner", chat_id="d", content="")
    await loop._ensure_identity(seed)
    owner = seed.user_id
    assert owner
    await loop.user_resolver.link_account(owner, "voice", "pi-empty")

    async def _empty_stt(path, *, stt=None):
        return "", "sherpa_onnx returned empty transcription (audio_bytes=44; speak after the beep)"

    app = create_app(loop, model_name="t", request_timeout=5)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        with patch("krabobot.api.server.transcribe_audio", side_effect=_empty_stt):
            form = FormData()
            form.add_field("device_id", "pi-empty")
            form.add_field(
                "audio",
                _tiny_wav_bytes(),
                filename="clip.wav",
                content_type="audio/wav",
            )
            r = await client.post(
                "/v1/voice/turn",
                data=form,
                headers={"Authorization": "Bearer tok-admin"},
            )
        assert r.status == 400
        body = await r.json()
        err = body.get("error") or body
        msg = err.get("message") if isinstance(err, dict) else str(body)
        assert "empty transcription" in msg
        assert "audio_bytes" in msg or "beep" in msg
    finally:
        await client.close()
