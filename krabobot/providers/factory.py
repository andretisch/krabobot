"""Shared LLM provider factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from krabobot.config.schema import Config
    from krabobot.providers.base import LLMProvider


def make_provider(config: Config) -> LLMProvider:
    """Create the LLM provider from config."""
    from dataclasses import replace

    from krabobot.providers.base import GenerationSettings
    from krabobot.providers.ollama_provider import OllamaProvider, resolve_ollama_connection
    from krabobot.providers.openai_compat_provider import OpenAICompatProvider
    from krabobot.providers.registry import find_by_name

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)
    spec = find_by_name(provider_name) if provider_name else None
    if spec and p and getattr(p, "use_max_completion_tokens", False):
        spec = replace(spec, supports_max_completion_tokens=True)
    backend = spec.backend if spec else "openai_compat"

    if spec and spec.name == "ollama":
        conn = resolve_ollama_connection(
            p.api_base if p else None,
            p.api_key if p else None,
        )
        if conn.is_cloud and not conn.api_key:
            raise ValueError(
                "No local Ollama found. Install Ollama (https://ollama.com) "
                "or set OLLAMA_API_KEY / providers.ollama.apiKey for cloud access."
            )
        provider: LLMProvider = OllamaProvider(
            api_key=conn.api_key,
            api_base=conn.host,
            default_model=model,
        )
    else:
        api_base = config.get_api_base(model)
        if backend == "openai_compat" and not model.startswith("bedrock/"):
            needs_key = not (p and p.api_key)
            exempt = spec and (spec.is_oauth or spec.is_direct or spec.is_local)
            if needs_key and not exempt:
                raise ValueError(f"No API key configured for provider '{provider_name}'.")

        provider = OpenAICompatProvider(
            api_key=p.api_key if p else None,
            api_base=api_base,
            default_model=model,
            extra_headers=p.extra_headers if p else None,
            spec=spec,
        )

    defaults = config.agents.defaults
    provider.generation = GenerationSettings(
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        reasoning_effort=defaults.reasoning_effort,
    )
    from krabobot.providers.anonymizing import wrap_provider_if_anonymize

    return wrap_provider_if_anonymize(provider, defaults.anonymize)
