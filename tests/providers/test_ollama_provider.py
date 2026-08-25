"""Tests for Ollama provider with auto local/cloud detection."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from krabobot.config.schema import Config
from krabobot.providers.factory import make_provider
from krabobot.providers.ollama_provider import (
    CLOUD_OLLAMA_HOST,
    LOCAL_OLLAMA_HOST,
    OllamaProvider,
    is_ollama_cloud_base,
    probe_local_ollama,
    resolve_ollama_api_key,
    resolve_ollama_connection,
)


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        ("https://ollama.com", True),
        ("https://ollama.com/", True),
        ("http://localhost:11434/v1", False),
        ("http://ollama-host:11434", False),
        (None, False),
    ],
)
def test_is_ollama_cloud_base(api_base: str | None, expected: bool) -> None:
    assert is_ollama_cloud_base(api_base) is expected


def test_resolve_ollama_api_key_prefers_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "env-key")
    assert resolve_ollama_api_key("config-key") == "config-key"


def test_resolve_ollama_api_key_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "env-key")
    assert resolve_ollama_api_key("") == "env-key"


def test_resolve_ollama_connection_explicit_local_host() -> None:
    conn = resolve_ollama_connection("http://ollama-host:11434", None, probe=lambda: False)

    assert conn.host == "http://ollama-host:11434"
    assert conn.is_cloud is False
    assert conn.api_key is None
    assert conn.auto_detected is False


def test_resolve_ollama_connection_explicit_cloud_host() -> None:
    conn = resolve_ollama_connection("https://ollama.com", "cloud-key", probe=lambda: True)

    assert conn.host == CLOUD_OLLAMA_HOST
    assert conn.is_cloud is True
    assert conn.api_key == "cloud-key"
    assert conn.auto_detected is False


def test_resolve_ollama_connection_auto_prefers_local_when_reachable() -> None:
    conn = resolve_ollama_connection(None, "unused-key", probe=lambda: True)

    assert conn.host == LOCAL_OLLAMA_HOST
    assert conn.is_cloud is False
    assert conn.api_key is None
    assert conn.auto_detected is True


def test_resolve_ollama_connection_auto_falls_back_to_cloud_with_key() -> None:
    conn = resolve_ollama_connection(None, "cloud-key", probe=lambda: False)

    assert conn.host == CLOUD_OLLAMA_HOST
    assert conn.is_cloud is True
    assert conn.api_key == "cloud-key"
    assert conn.auto_detected is True


def test_probe_local_ollama_returns_true_on_200() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_client = MagicMock()
    mock_client.get.return_value = mock_response
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with patch("krabobot.providers.ollama_provider.httpx.Client", return_value=mock_client):
        assert probe_local_ollama() is True

    mock_client.get.assert_called_once_with(f"{LOCAL_OLLAMA_HOST}/api/tags")


@pytest.mark.asyncio
async def test_ollama_provider_uses_async_client_for_cloud() -> None:
    provider = OllamaProvider(api_key="secret-key", api_base="https://ollama.com")
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "message": {"role": "assistant", "content": "hi"},
        "done": True,
    }

    mock_client = AsyncMock()
    mock_client.chat.return_value = mock_response
    provider._client = mock_client

    result = await provider.chat([{"role": "user", "content": "hello"}])

    assert result.content == "hi"
    kwargs = mock_client.chat.call_args.kwargs
    assert kwargs["stream"] is False


@pytest.mark.asyncio
async def test_ollama_provider_parses_tool_calls() -> None:
    provider = OllamaProvider(api_key="secret-key")
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
            }],
        },
        "done": True,
    }

    mock_client = AsyncMock()
    mock_client.chat.return_value = mock_response
    provider._client = mock_client

    result = await provider.chat([{"role": "user", "content": "weather?"}])

    assert result.finish_reason == "tool_calls"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments == {"city": "Paris"}


def test_config_get_api_base_keeps_ollama_cloud_without_v1_suffix() -> None:
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "ollama", "model": "gemma4:cloud"}},
            "providers": {
                "ollama": {"apiKey": "key", "apiBase": "https://ollama.com"},
            },
        }
    )

    assert config.get_api_base() == "https://ollama.com"


def test_make_provider_uses_ollama_provider_for_explicit_cloud() -> None:
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "ollama", "model": "gemma4:cloud"}},
            "providers": {
                "ollama": {"apiKey": "cloud-key", "apiBase": "https://ollama.com"},
            },
        }
    )

    provider = make_provider(config)

    assert isinstance(provider, OllamaProvider)
    assert provider.api_key == "cloud-key"
    assert provider.api_base == "https://ollama.com"


def test_make_provider_auto_uses_local_when_reachable() -> None:
    config = Config()

    with patch(
        "krabobot.providers.ollama_provider.resolve_ollama_connection",
        return_value=resolve_ollama_connection(None, None, probe=lambda: True),
    ):
        provider = make_provider(config)

    assert isinstance(provider, OllamaProvider)
    assert provider.api_base == LOCAL_OLLAMA_HOST
    assert provider.api_key is None


def test_make_provider_auto_uses_cloud_when_local_unreachable() -> None:
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "ollama", "model": "gemma4:cloud"}},
            "providers": {"ollama": {"apiKey": "cloud-key"}},
        }
    )

    with patch(
        "krabobot.providers.ollama_provider.resolve_ollama_connection",
        return_value=resolve_ollama_connection(None, "cloud-key", probe=lambda: False),
    ):
        provider = make_provider(config)

    assert isinstance(provider, OllamaProvider)
    assert provider.api_base == CLOUD_OLLAMA_HOST
    assert provider.api_key == "cloud-key"


def test_make_provider_auto_cloud_requires_api_key() -> None:
    config = Config()

    with patch(
        "krabobot.providers.ollama_provider.resolve_ollama_connection",
        return_value=resolve_ollama_connection(None, None, probe=lambda: False),
    ):
        with pytest.raises(ValueError, match="No local Ollama found"):
            make_provider(config)


def test_make_provider_cloud_accepts_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "env-cloud-key")
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "ollama", "model": "gemma4:cloud"}},
            "providers": {
                "ollama": {"apiBase": "https://ollama.com"},
            },
        }
    )

    provider = make_provider(config)

    assert isinstance(provider, OllamaProvider)
    assert provider.api_key == "env-cloud-key"
