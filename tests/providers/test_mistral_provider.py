"""Tests for provider registration in the simplified baseline."""

from krabobot.config.schema import ProvidersConfig
from krabobot.providers.registry import PROVIDERS


def test_proxyapi_config_field_exists():
    """ProvidersConfig should expose a proxyapi field."""
    config = ProvidersConfig()
    assert hasattr(config, "proxyapi")


def test_proxyapi_provider_in_registry():
    """ProxyAPI should be registered in the provider registry."""
    specs = {s.name: s for s in PROVIDERS}
    assert "proxyapi" in specs

    proxyapi = specs["proxyapi"]
    assert proxyapi.env_key == "PROXYAPI_API_KEY"
    assert proxyapi.default_api_base == "https://api.proxyapi.ru/openai/v1"
