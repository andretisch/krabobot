"""Focused tests for the fixed-session OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from krabobot.api.server import (
    API_CHAT_ID,
    API_SESSION_KEY,
    DEFAULT_MAX_UPLOAD_HARD_MB,
    MAX_UPLOAD_FILE_BYTES,
    MAX_UPLOAD_SOFT_BYTES,
    _chat_completion_response,
    _error_json,
    _persist_web_uploads,
    _stream_multipart_part_to_path,
    create_app,
    handle_chat_completions,
    web_static_dir,
)
from krabobot.session.manager import SessionManager

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
    assert resp.headers.get("Cache-Control") == "no-store"
    text = await resp.text()
    assert "<html" in text.lower()
    # mtime cache-bust so browsers pick up app.js after deploy without hard refresh
    assert re.search(r'src="/static/app\.js\?v=\d+"', text)


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
    agent.user_resolver.unlink_account = AsyncMock(return_value=True)

    app = create_app(agent, model_name="m")
    client = await aiohttp_client(app)

    resp = await client.delete("/v1/web/sessions/xyz")
    assert resp.status == 200
    sm.delete_session.assert_called_once_with("api:xyz")
    agent.user_resolver.unlink_account.assert_awaited_once_with("api", "xyz")

    resp2 = await client.get("/v1/web/sessions/xyz/messages")
    assert resp2.status == 200
    body = await resp2.json()
    assert body["data"] == []


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_web_session_messages_strips_linked_accounts(aiohttp_client) -> None:
    """UI history must not leak [Linked Accounts] blocks into the chat transcript."""
    linked = (
        "[Linked Accounts]\n"
        "user_id: u1\n"
        "accounts: telegram:123, vk:456\n\n"
        "Hello from user"
    )
    sm = MagicMock()
    sm.invalidate = MagicMock()
    sm.get_or_create.return_value = MagicMock(
        messages=[
            {"role": "user", "content": linked},
            {"role": "assistant", "content": "Hi!"},
        ]
    )

    agent = _make_mock_agent()
    agent.session_manager_for_api = AsyncMock(return_value=sm)

    app = create_app(agent, model_name="m")
    client = await aiohttp_client(app)
    resp = await client.get("/v1/web/sessions/s1/messages")
    assert resp.status == 200
    body = await resp.json()
    assert body["data"][0]["content"] == "Hello from user"
    assert "Linked Accounts" not in body["data"][0]["content"]
    assert body["data"][1]["content"] == "Hi!"


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_web_uploads_saves_mp4(aiohttp_client, tmp_path) -> None:
    from aiohttp import FormData

    agent = _make_mock_agent(workspace=tmp_path)
    app = create_app(agent, model_name="m")
    client = await aiohttp_client(app)

    payload = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64
    form = FormData()
    form.add_field("session_id", "sess-video-1")
    form.add_field(
        "files",
        payload,
        filename="Screen Recording 2026-09-21.mp4",
        content_type="video/mp4",
    )

    resp = await client.post("/v1/web/uploads", data=form)
    assert resp.status == 200
    body = await resp.json()
    assert body["object"] == "upload.result"
    assert len(body["data"]) == 1
    row = body["data"][0]
    assert row["mime"] == "video/mp4"
    assert row["size"] == len(payload)
    assert row["filename"].endswith(".mp4")
    saved = Path(row["path"])
    assert saved.is_file()
    assert saved.read_bytes() == payload
    assert "uploads" in saved.parts
    assert "web" in saved.parts


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_web_uploads_rejects_oversized(aiohttp_client, tmp_path) -> None:
    from aiohttp import FormData

    agent = _make_mock_agent(workspace=tmp_path)
    app = create_app(agent, model_name="m", max_upload_mb=1)
    # Keep test fast: tiny hard limit (client_max_size still allows the small body).
    app["max_upload_bytes"] = 32
    client = await aiohttp_client(app)

    form = FormData()
    form.add_field("session_id", "s1")
    form.add_field(
        "files",
        b"x" * 64,
        filename="big.mp4",
        content_type="video/mp4",
    )
    resp = await client.post("/v1/web/uploads", data=form)
    assert resp.status == 413
    body = await resp.json()
    assert "слишком большой" in body["error"]["message"].lower()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_web_uploads_streams_to_disk(aiohttp_client, tmp_path) -> None:
    from aiohttp import FormData

    agent = _make_mock_agent(workspace=tmp_path)
    app = create_app(agent, model_name="m", max_upload_mb=8)
    client = await aiohttp_client(app)

    # ~256 KiB — larger than a single read_chunk in the soft path, still fast.
    payload = b"\x00\x00\x00\x18ftypmp42" + (b"V" * (256 * 1024))
    form = FormData()
    form.add_field("session_id", "sess-stream")
    form.add_field(
        "files",
        payload,
        filename="bigish.mp4",
        content_type="video/mp4",
    )
    resp = await client.post("/v1/web/uploads", data=form)
    assert resp.status == 200
    body = await resp.json()
    row = body["data"][0]
    saved = Path(row["path"])
    assert saved.is_file()
    assert saved.read_bytes() == payload
    assert row["size"] == len(payload)
    assert (tmp_path / "uploads" / "web").exists()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_health_reports_upload_limits(aiohttp_client, tmp_path) -> None:
    agent = _make_mock_agent(workspace=tmp_path)
    app = create_app(agent, model_name="m", max_upload_mb=512)
    client = await aiohttp_client(app)
    resp = await client.get("/health")
    assert resp.status == 200
    body = await resp.json()
    assert body["status"] == "ok"
    assert body["maxUploadSoftMb"] == 50
    assert body["maxUploadHardMb"] == 512
    assert body["maxUploadHardBytes"] == 512 * 1024 * 1024


@pytest.mark.asyncio
async def test_stream_multipart_part_to_path_enforces_limit(tmp_path: Path) -> None:
    class _FakePart:
        def __init__(self, data: bytes, chunk: int = 8) -> None:
            self._data = data
            self._chunk = chunk
            self._pos = 0

        async def read_chunk(self, size: int = 64 * 1024) -> bytes:
            n = min(size, self._chunk, len(self._data) - self._pos)
            if n <= 0:
                return b""
            out = self._data[self._pos : self._pos + n]
            self._pos += n
            return out

    dest = tmp_path / "out.bin"
    with pytest.raises(ValueError, match="слишком большой"):
        await _stream_multipart_part_to_path(_FakePart(b"abcdefghij"), dest, limit=5)
    assert not dest.exists()


def test_persist_web_uploads_kb_file_mp4(tmp_path: Path) -> None:
    import base64

    raw = b"fake-mp4-bytes"
    b64 = base64.b64encode(raw).decode("ascii")
    coerced = [
        {"type": "text", "text": "посмотри видео"},
        {
            "type": "kb_file",
            "kb_file": {
                "filename": "clip.mp4",
                "mime": "video/mp4",
                "data": b64,
            },
        },
    ]
    text, media = _persist_web_uploads(tmp_path, "sid-kb", coerced)
    assert media == []
    assert "посмотри видео" in text
    assert "Файл сохранён в workspace:" in text
    assert ".mp4" in text
    # File actually written
    upload_root = tmp_path / "uploads" / "web"
    files = list(upload_root.rglob("*.mp4"))
    assert len(files) == 1
    assert files[0].read_bytes() == raw


def test_max_upload_soft_under_body_limit() -> None:
    """Soft 50 MiB is the UI attach threshold; multipart endpoint enforces hard max."""
    from krabobot.api.server import MAX_REQUEST_BODY_BYTES

    assert MAX_UPLOAD_SOFT_BYTES < MAX_REQUEST_BODY_BYTES
    assert MAX_UPLOAD_SOFT_BYTES == 50 * 1024 * 1024
    assert MAX_UPLOAD_FILE_BYTES == MAX_UPLOAD_SOFT_BYTES
    assert DEFAULT_MAX_UPLOAD_HARD_MB == 2048


def test_create_app_soft_and_hard_upload_limits_distinct() -> None:
    """Regression: soft (UI) stays 50 MiB while hard follows max_upload_mb."""
    agent = MagicMock()
    app = create_app(agent, model_name="m", max_upload_mb=512)
    assert app["max_upload_soft_bytes"] == MAX_UPLOAD_SOFT_BYTES
    assert app["max_upload_bytes"] == 512 * 1024 * 1024
    assert app["max_upload_soft_bytes"] < app["max_upload_bytes"]


def test_create_app_client_max_covers_hard_upload() -> None:
    agent = MagicMock()
    app = create_app(agent, model_name="m", max_upload_mb=100)
    assert app["max_upload_mb"] == 100
    assert app["max_upload_bytes"] == 100 * 1024 * 1024
    assert app._client_max_size >= 100 * 1024 * 1024


def test_api_config_max_upload_mb_alias() -> None:
    from krabobot.config.schema import ApiConfig

    cfg = ApiConfig.model_validate({"maxUploadMb": 1024})
    assert cfg.max_upload_mb == 1024
    dumped = cfg.model_dump(by_alias=True)
    assert dumped["maxUploadMb"] == 1024
