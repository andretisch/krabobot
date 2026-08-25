"""Tests for Ollama Cloud native provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from krabobot.config.schema import Config
from krabobot.providers.factory import make_provider
from krabobot.providers.ollama_provider import (
    OllamaProvider,
    is_ollama_cloud_base,
    resolve_ollama_api_key,
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


@pytest.mark.asyncio
async def test_ollama_provider_applies_bearer_auth() -> None:
    provider = OllamaProvider(api_key="secret-key", api_base="https://ollama.com")
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "message": {"role": "assistant", "content": "hi"},
        "done": True,
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("krabobot.providers.ollama_provider.httpx.AsyncClient", return_value=mock_client):
        result = await provider.chat([{"role": "user", "content": "hello"}])

    assert result.content == "hi"
    kwargs = mock_client.post.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer secret-key"
    assert kwargs["json"]["stream"] is False
    assert mock_client.post.call_args.args[0] == "https://ollama.com/api/chat"


@pytest.mark.asyncio
async def test_ollama_provider_parses_tool_calls() -> None:
    provider = OllamaProvider(api_key="secret-key")
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
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
    mock_client.post.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("krabobot.providers.ollama_provider.httpx.AsyncClient", return_value=mock_client):
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


def test_make_provider_uses_ollama_provider_for_cloud() -> None:
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


def test_make_provider_keeps_openai_compat_for_local_ollama() -> None:
    config = Config()

    with patch("krabobot.providers.openai_compat_provider.AsyncOpenAI") as mock_async_openai:
        provider = make_provider(config)

    from krabobot.providers.openai_compat_provider import OpenAICompatProvider

    assert isinstance(provider, OpenAICompatProvider)
    kwargs = mock_async_openai.call_args.kwargs
    assert kwargs["base_url"] == "http://localhost:11434/v1"


def test_make_provider_cloud_requires_api_key() -> None:
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "ollama", "model": "gemma4:cloud"}},
            "providers": {
                "ollama": {"apiBase": "https://ollama.com"},
            },
        }
    )

    with pytest.raises(ValueError, match="Ollama Cloud requires"):
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
