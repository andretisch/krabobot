"""Focused tests for the fixed-session OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from krabobot.session.manager import SessionManager

from krabobot.api.server import (
    API_CHAT_ID,
    API_SESSION_KEY,
    _chat_completion_response,
    _error_json,
    create_app,
    handle_chat_completions,
    web_static_dir,
)

try:
    from aiohttp.test_utils import TestClient, TestServer

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytest_plugins = ("pytest_asyncio",)


def _make_mock_agent(response_text: str = "mock response", workspace: Path | None = None) -> MagicMock:
    agent = MagicMock()
    agent.process_direct = AsyncMock(return_value=response_text)
    agent._connect_mcp = AsyncMock()
    agent.close_mcp = AsyncMock()
    ws = workspace if workspace is not None else Path("/tmp/krabobot_api_test_ws")
    agent.session_manager_for_api = AsyncMock(return_value=SessionManager(ws))
    return agent


@pytest.fixture
def mock_agent(tmp_path):
    return _make_mock_agent(workspace=tmp_path)


@pytest.fixture
def app(mock_agent):
    return create_app(mock_agent, model_name="test-model", request_timeout=10.0)


def _attach_workspace(agent: MagicMock, workspace: Path) -> None:
    agent.session_manager_for_api = AsyncMock(return_value=SessionManager(workspace))


@pytest_asyncio.fixture
async def aiohttp_client():
    clients: list[TestClient] = []

    async def _make_client(app):
        client = TestClient(TestServer(app))
        await client.start_server()
        clients.append(client)
        return client

    try:
        yield _make_client
    finally:
        for client in clients:
            await client.close()


def test_error_json() -> None:
    resp = _error_json(400, "bad request")
    assert resp.status == 400
    body = json.loads(resp.body)
    assert body["error"]["message"] == "bad request"
    assert body["error"]["code"] == 400


def test_chat_completion_response() -> None:
    result = _chat_completion_response("hello world", "test-model")
    assert result["object"] == "chat.completion"
    assert result["model"] == "test-model"
    assert result["choices"][0]["message"]["content"] == "hello world"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["id"].startswith("chatcmpl-")


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_missing_messages_returns_400(aiohttp_client, app) -> None:
    client = await aiohttp_client(app)
    resp = await client.post("/v1/chat/completions", json={"model": "test"})
    assert resp.status == 400


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_no_user_message_returns_400(aiohttp_client, app) -> None:
    client = await aiohttp_client(app)
    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "system", "content": "you are a bot"}]},
    )
    assert resp.status == 400


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_stream_true_returns_400(aiohttp_client, app) -> None:
    client = await aiohttp_client(app)
    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
    )
    assert resp.status == 400
    body = await resp.json()
    assert "stream" in body["error"]["message"].lower()


@pytest.mark.asyncio
async def test_model_mismatch_returns_400() -> None:
    request = MagicMock()
    request.json = AsyncMock(
        return_value={
            "model": "other-model",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    request.app = {
        "agent_loop": _make_mock_agent(),
        "model_name": "test-model",
        "request_timeout": 10.0,
        "session_lock": asyncio.Lock(),
    }

    resp = await handle_chat_completions(request)
    assert resp.status == 400
    body = json.loads(resp.body)
    assert "test-model" in body["error"]["message"]


@pytest.mark.asyncio
async def test_single_user_message_required() -> None:
    request = MagicMock()
    request.json = AsyncMock(
        return_value={
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "previous reply"},
            ],
        }
    )
    request.app = {
        "agent_loop": _make_mock_agent(),
        "model_name": "test-model",
        "request_timeout": 10.0,
        "session_lock": asyncio.Lock(),
    }

    resp = await handle_chat_completions(request)
    assert resp.status == 400
    body = json.loads(resp.body)
    assert "single user message" in body["error"]["message"].lower()


@pytest.mark.asyncio
async def test_single_user_message_must_have_user_role() -> None:
    request = MagicMock()
    request.json = AsyncMock(
        return_value={
            "messages": [{"role": "system", "content": "you are a bot"}],
        }
    )
    request.app = {
        "agent_loop": _make_mock_agent(),
        "model_name": "test-model",
        "request_timeout": 10.0,
        "session_lock": asyncio.Lock(),
    }

    resp = await handle_chat_completions(request)
    assert resp.status == 400
    body = json.loads(resp.body)
    assert "single user message" in body["error"]["message"].lower()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_successful_request_uses_fixed_api_session(aiohttp_client, mock_agent) -> None:
    app = create_app(mock_agent, model_name="test-model")
    client = await aiohttp_client(app)
    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["choices"][0]["message"]["content"] == "mock response"
    assert body["model"] == "test-model"
    mock_agent.process_direct.assert_called_once_with(
        content="hello",
        media=None,
        session_key=API_SESSION_KEY,
        channel="api",
        chat_id=API_CHAT_ID,
        sender_id="default",
    )


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_followup_requests_share_same_session_key(aiohttp_client, tmp_path) -> None:
    call_log: list[str] = []

    async def fake_process(content, session_key="", channel="", chat_id="", **kwargs):
        call_log.append(session_key)
        return f"reply to {content}"

    agent = MagicMock()
    agent.process_direct = fake_process
    agent._connect_mcp = AsyncMock()
    agent.close_mcp = AsyncMock()
    _attach_workspace(agent, tmp_path)

    app = create_app(agent, model_name="m")
    client = await aiohttp_client(app)

    r1 = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "first"}]},
    )
    r2 = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "second"}]},
    )

    assert r1.status == 200
    assert r2.status == 200
    assert call_log == [API_SESSION_KEY, API_SESSION_KEY]


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_fixed_session_requests_are_serialized(aiohttp_client, tmp_path) -> None:
    order: list[str] = []

    async def slow_process(content, session_key="", channel="", chat_id="", **kwargs):
        order.append(f"start:{content}")
        await asyncio.sleep(0.1)
        order.append(f"end:{content}")
        return content

    agent = MagicMock()
    agent.process_direct = slow_process
    agent._connect_mcp = AsyncMock()
    agent.close_mcp = AsyncMock()
    _attach_workspace(agent, tmp_path)

    app = create_app(agent, model_name="m")
    client = await aiohttp_client(app)

    async def send(msg: str):
        return await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": msg}]},
        )

    r1, r2 = await asyncio.gather(send("first"), send("second"))
    assert r1.status == 200
    assert r2.status == 200
    # Verify serialization: one process must fully finish before the other starts
    if order[0] == "start:first":
        assert order.index("end:first") < order.index("start:second")
    else:
        assert order.index("end:second") < order.index("start:first")


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_models_endpoint(aiohttp_client, app) -> None:
    client = await aiohttp_client(app)
    resp = await client.get("/v1/models")
    assert resp.status == 200
    body = await resp.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "test-model"


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_health_endpoint(aiohttp_client, app) -> None:
    client = await aiohttp_client(app)
    resp = await client.get("/health")
    assert resp.status == 200
    body = await resp.json()
    assert body["status"] == "ok"


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_root_serves_bundled_chat_ui(aiohttp_client, app) -> None:
    if not web_static_dir().is_dir() or not (web_static_dir() / "index.html").is_file():
        pytest.skip("bundled web UI not present")
    client = await aiohttp_client(app)
    resp = await client.get("/")
    assert resp.status == 200
    text = await resp.text()
    assert "<html" in text.lower()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_multimodal_content_extracts_text(aiohttp_client, mock_agent) -> None:
    app = create_app(mock_agent, model_name="m")
    client = await aiohttp_client(app)
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "data:image/png;base64,"
                                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
                                    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                                )
                            },
                        },
                    ],
                }
            ]
        },
    )
    assert resp.status == 200
    kw = mock_agent.process_direct.call_args.kwargs
    assert kw["content"] == "describe this"
    assert kw.get("media") and len(kw["media"]) == 1
    assert Path(kw["media"][0]).is_file()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_empty_response_retry_then_success(aiohttp_client, tmp_path) -> None:
    call_count = 0

    async def sometimes_empty(content, session_key="", channel="", chat_id="", **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ""
        return "recovered response"

    agent = MagicMock()
    agent.process_direct = sometimes_empty
    agent._connect_mcp = AsyncMock()
    agent.close_mcp = AsyncMock()
    _attach_workspace(agent, tmp_path)

    app = create_app(agent, model_name="m")
    client = await aiohttp_client(app)
    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["choices"][0]["message"]["content"] == "recovered response"
    assert call_count == 2


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_empty_response_falls_back(aiohttp_client, tmp_path) -> None:
    call_count = 0

    async def always_empty(content, session_key="", channel="", chat_id="", **kwargs):
        nonlocal call_count
        call_count += 1
        return ""

    agent = MagicMock()
    agent.process_direct = always_empty
    agent._connect_mcp = AsyncMock()
    agent.close_mcp = AsyncMock()
    _attach_workspace(agent, tmp_path)

    app = create_app(agent, model_name="m")
    client = await aiohttp_client(app)
    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["choices"][0]["message"]["content"] == "I've completed processing but have no response to give."
    assert call_count == 2


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_web_sessions_list_uses_agent_session_manager(aiohttp_client) -> None:
    sm = MagicMock()
    sm.list_sessions.return_value = [
        {
            "key": "api:aa-bb-cc",
            "updated_at": "2026-01-02T12:00:00",
            "created_at": "2026-01-01T10:00:00",
        },
    ]
    sess = MagicMock()
    sess.messages = [{"role": "user", "content": "hello there"}]
    sm.get_or_create.return_value = sess

    agent = _make_mock_agent()
    agent.session_manager_for_api = AsyncMock(return_value=sm)

    app = create_app(agent, model_name="m")
    client = await aiohttp_client(app)
    resp = await client.get("/v1/web/sessions")
    assert resp.status == 200
    body = await resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["id"] == "aa-bb-cc"
    assert "hello" in body["data"][0]["preview"]


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_web_session_delete_and_messages(aiohttp_client) -> None:
    sm = MagicMock()
    sm.list_sessions.return_value = []
    sm.get_or_create.return_value = MagicMock(messages=[])
    sm.delete_session.return_value = True

    agent = _make_mock_agent()
    agent.session_manager_for_api = AsyncMock(return_value=sm)

    app = create_app(agent, model_name="m")
    client = await aiohttp_client(app)

    resp = await client.delete("/v1/web/sessions/xyz")
    assert resp.status == 200
    sm.delete_session.assert_called_once_with("api:xyz")

    resp2 = await client.get("/v1/web/sessions/xyz/messages")
    assert resp2.status == 200
    body = await resp2.json()
    assert body["data"] == []
