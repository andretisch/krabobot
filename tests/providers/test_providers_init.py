"""Tests for lazy provider exports from krabobot.providers."""

from __future__ import annotations

import importlib
import sys


def test_importing_providers_package_is_lazy(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "krabobot.providers", raising=False)
    monkeypatch.delitem(sys.modules, "krabobot.providers.openai_compat_provider", raising=False)

    providers = importlib.import_module("krabobot.providers")

    assert "krabobot.providers.openai_compat_provider" not in sys.modules
    assert providers.__all__ == [
        "LLMProvider",
        "LLMResponse",
        "OpenAICompatProvider",
    ]


def test_explicit_provider_import_still_works(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "krabobot.providers", raising=False)
    monkeypatch.delitem(sys.modules, "krabobot.providers.openai_compat_provider", raising=False)

    namespace: dict[str, object] = {}
    exec("from krabobot.providers import OpenAICompatProvider", namespace)

    assert namespace["OpenAICompatProvider"].__name__ == "OpenAICompatProvider"
    assert "krabobot.providers.openai_compat_provider" in sys.modules
