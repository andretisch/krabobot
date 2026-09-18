"""Tests for /v1/web/config merge, payloads, backups, and restore helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from krabobot.api.server import create_app

try:
    from aiohttp.test_utils import TestClient, TestServer

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytest_plugins = ("pytest_asyncio",)


def test_build_web_config_redacts_and_splits_channels(monkeypatch, tmp_path: Path) -> None:
    from krabobot.api import web_config as wc

    krabot_dir = tmp_path / ".krabobot"
    krabot_dir.mkdir()
    cfg_file = krabot_dir / "config.json"
    minimal = {
        "agents": {
            "defaults": {
                "model": "gpt-x",
                "provider": "custom",
                "workspace": "~/w",
            }
        },
        "providers": {
            "custom": {
                "apiKey": "sk-secret",
                "apiBase": "https://example/v1",
            }
        },
        "channels": {
            "sendProgress": False,
            "sendToolHints": True,
            "telegram": {"enabled": True, "token": "123:telegram"},
            "vk": {"enabled": False, "token": "vk-token"},
        },
        "api": {"host": "127.0.0.1", "port": 8900},
    }
    cfg_file.write_text(json.dumps(minimal, ensure_ascii=False, indent=2), encoding="utf-8")

    monkeypatch.setattr(
        "krabobot.api.web_config.get_config_path",
        lambda: cfg_file.resolve(),
        raising=True,
    )

    payload = wc.build_web_config_payload()
    assert payload["path"] == str(cfg_file.resolve())
    assert payload["core"]["providers"]["custom"]["apiKey"] == "••••••••"
    assert payload["channels"]["common"]["sendProgress"] is False
    assert payload["channels"]["named"]["telegram"]["token"] == "••••••••"
    assert "sendProgress" not in payload["channels"]["named"]
    assert payload["other"]["api"]["host"] == "127.0.0.1"
    vals = [c["value"] for c in payload["providerChoices"]]
    assert "auto" in vals and "custom" in vals and "openrouter" in vals


def test_redact_leaves_counters_and_masks_secrets() -> None:
    from krabobot.api.web_config import redact_secrets

    tree = {
        "agents": {"defaults": {"model": "m", "contextWindowTokens": 65536}},
        "apiKeyFlat": "keep-or-mask",
        "telegram": {"accessToken": "abc"},
    }
    redact_secrets(tree)
    assert tree["agents"]["defaults"]["contextWindowTokens"] == 65536
    assert tree["apiKeyFlat"] == "keep-or-mask"  # not "apiKey" suffix pattern
    assert tree["telegram"]["accessToken"] == "••••••••"


def test_sections_payload_merge_keeps_masked_api_key(monkeypatch, tmp_path: Path) -> None:
    from krabobot.api import web_config as wc
    from krabobot.config.loader import load_config

    krabot_dir = tmp_path / ".krabobot"
    krabot_dir.mkdir()
    cfg_file = krabot_dir / "config.json"
    minimal = {
        "agents": {
            "defaults": {
                "model": "gpt-x",
                "provider": "custom",
                "workspace": "~/w",
            }
        },
        "providers": {
            "custom": {
                "apiKey": "sk-secret",
                "apiBase": "https://example/v1",
            }
        },
        "channels": {"sendProgress": True, "telegram": {"enabled": False}},
        "api": {"host": "127.0.0.1", "port": 8900},
    }
    cfg_file.write_text(json.dumps(minimal, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(wc, "get_config_path", lambda: cfg_file.resolve(), raising=True)

    incoming = wc.build_web_config_payload()
    incoming["core"]["agents"]["defaults"]["model"] = "gpt-y"

    merged = wc.sections_payload_merge_into_config(load_config(cfg_file), incoming)
    dumped = merged.model_dump(by_alias=True, mode="json")
    assert dumped["agents"]["defaults"]["model"] == "gpt-y"
    assert dumped["providers"]["custom"]["apiKey"] == "sk-secret"


def test_save_web_config_writes_and_backup(monkeypatch, tmp_path: Path) -> None:
    from krabobot.api import web_config as wc
    from krabobot.config.loader import load_config

    krabot_dir = tmp_path / ".krabobot"
    krabot_dir.mkdir()
    cfg_file = krabot_dir / "config.json"
    minimal = {
        "agents": {
            "defaults": {
                "model": "m1",
                "provider": "openrouter",
                "workspace": "~/w",
            }
        },
        "providers": {
            "custom": {"apiKey": "", "apiBase": None},
            "openrouter": {"apiKey": "key1", "apiBase": ""},
            "proxyapi": {"apiKey": "", "apiBase": None},
            "gptunnel": {"apiKey": "", "apiBase": None},
            "ollama": {"apiKey": "", "apiBase": None},
        },
        "channels": {"sendProgress": True},
        "api": {"host": "127.0.0.1", "port": 8900},
    }
    cfg_file.write_text(json.dumps(minimal, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(wc, "get_config_path", lambda: cfg_file.resolve(), raising=True)

    payload = wc.build_web_config_payload()
    payload["core"]["agents"]["defaults"]["model"] = "m2"

    wc.save_web_config_sections(payload)

    backups = list(cfg_file.parent.glob("config.backup.*.json"))
    assert len(backups) == 1

    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert data["agents"]["defaults"]["model"] == "m2"

    reload = load_config(cfg_file).model_dump(by_alias=True, mode="json")
    assert reload["providers"]["openrouter"]["apiKey"] == "key1"


def test_restore_roundtrip(monkeypatch, tmp_path: Path) -> None:
    from krabobot.api import web_config as wc

    krabot_dir = tmp_path / ".krabobot"
    krabot_dir.mkdir()
    cfg_file = krabot_dir / "config.json"
    v1 = {
        "agents": {"defaults": {"model": "a", "provider": "custom", "workspace": "~"}},
        "providers": {"custom": {"apiKey": "k", "apiBase": ""}},
        "channels": {"sendProgress": True},
        "api": {"host": "127.0.0.1", "port": 8900},
    }
    cfg_file.write_text(json.dumps(v1, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(wc, "get_config_path", lambda: cfg_file.resolve(), raising=True)

    bn = wc.backup_config_now(cfg_file).name
    cfg_file.write_text(
        json.dumps({**v1, "agents": {**v1["agents"], "defaults": {**v1["agents"]["defaults"], "model": "b"}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    wc.restore_backup(bn)
    restored = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert restored["agents"]["defaults"]["model"] == "a"


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_put_web_config_persists(tmp_path: Path, monkeypatch) -> None:
    from krabobot.api import web_config as wc
    from krabobot.session.manager import SessionManager

    krabot_dir = tmp_path / ".krabobot"
    krabot_dir.mkdir()
    cfg_file = krabot_dir / "config.json"
    minimal = {
        "agents": {
            "defaults": {"model": "x", "provider": "custom", "workspace": "~/w"}
        },
        "providers": {
            "custom": {"apiKey": "sec", "apiBase": "https://x"},
            "openrouter": {"apiKey": "", "apiBase": None},
            "proxyapi": {"apiKey": "", "apiBase": None},
            "gptunnel": {"apiKey": "", "apiBase": None},
            "ollama": {"apiKey": "", "apiBase": None},
        },
        "channels": {},
        "api": {"host": "127.0.0.1", "port": 8900},
    }
    cfg_file.write_text(json.dumps(minimal, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(wc, "get_config_path", lambda: cfg_file.resolve(), raising=True)

    mock_agent = MagicMock()
    mock_agent.process_direct = AsyncMock(return_value="ok")
    mock_agent.session_manager_for_api = AsyncMock(return_value=SessionManager(tmp_path))

    app = create_app(mock_agent, model_name="t", request_timeout=5)

    payload = wc.build_web_config_payload()
    payload["core"]["agents"]["defaults"]["model"] = "y"

    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.put("/v1/web/config", json=payload)
        assert resp.status == 200
        data = json.loads(cfg_file.read_text(encoding="utf-8"))
        assert data["agents"]["defaults"]["model"] == "y"
        assert data["providers"]["custom"]["apiKey"] == "sec"
    finally:
        await client.close()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_web_backup_download_config_and_workspace(tmp_path: Path, monkeypatch) -> None:
    """GET /v1/web/backup/download packs config + workspace, not media/models/history."""
    import io
    import tarfile

    from krabobot.config.loader import save_config
    from krabobot.config.schema import Config
    from krabobot.session.manager import SessionManager

    krabot_dir = tmp_path / ".krabobot"
    krabot_dir.mkdir()
    cfg_file = krabot_dir / "config.json"
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "marker.txt").write_text("workspace-ok", encoding="utf-8")

    media = krabot_dir / "media"
    media.mkdir()
    (media / "skip.bin").write_bytes(b"\x00\x01")
    models = krabot_dir / "models"
    models.mkdir()
    (models / "big.onnx").write_bytes(b"model")
    history = krabot_dir / "history"
    history.mkdir()
    (history / "cli.log").write_text("hist", encoding="utf-8")

    cfg = Config()
    cfg.agents.defaults.workspace = str(ws)
    cfg.agents.defaults.model = "test-model"
    cfg.agents.defaults.provider = "custom"
    save_config(cfg, cfg_file)

    monkeypatch.setattr(
        "krabobot.config.loader.get_config_path",
        lambda: cfg_file.resolve(),
        raising=True,
    )

    mock_agent = MagicMock()
    mock_agent.process_direct = AsyncMock(return_value="ok")
    mock_agent.session_manager_for_api = AsyncMock(return_value=SessionManager(tmp_path))

    app = create_app(mock_agent, model_name="t", request_timeout=5)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.get("/v1/web/backup/download")
        assert resp.status == 200
        assert "application/gzip" in (resp.headers.get("Content-Type") or "")
        cd = resp.headers.get("Content-Disposition") or ""
        assert "attachment" in cd
        assert "krabobot-backup-" in cd
        assert cd.endswith('.tar.gz"') or ".tar.gz" in cd

        body = await resp.read()
        assert body[:2] == b"\x1f\x8b"  # gzip magic

        with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tf:
            names = set(tf.getnames())
            assert "manifest.json" in names
            assert "payload/config.json" in names
            assert any(n.startswith("payload/workspace/") for n in names)
            assert not any(n.startswith("payload/media") for n in names)
            assert not any(n.startswith("payload/models") for n in names)
            assert not any(n.startswith("payload/history") for n in names)

            m = json.loads(tf.extractfile("manifest.json").read().decode("utf-8"))
            assert m["entries"] == ["config", "workspace"]

            marker = tf.extractfile("payload/workspace/marker.txt")
            assert marker is not None
            assert marker.read().decode("utf-8") == "workspace-ok"
    finally:
        await client.close()
