"""Tests for config YAML merge and meetings path resolution helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.app import resolve_meetings_save_dir  # noqa: E402
from krabobot_voice.config import (  # noqa: E402
    VoiceClientConfig,
    merge_config_yaml,
)
from krabobot_voice.meeting import default_meetings_dir  # noqa: E402
from krabobot_voice.status_bus import StatusBus, set_status_bus  # noqa: E402


def test_merge_config_yaml_nested(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "base_url: http://127.0.0.1:8900\nwake:\n  phrase: old\n",
        encoding="utf-8",
    )
    merge_config_yaml(
        path,
        {
            "base_url": "http://127.0.0.1:9000",
            "wake": {"phrase": "Ок, Бот"},
            "meeting": {"save_dir": str(tmp_path / "meetings")},
        },
    )
    cfg = VoiceClientConfig.load(path)
    assert cfg.base_url == "http://127.0.0.1:9000"
    assert cfg.wake_phrases[0] == "Ок, Бот"
    assert cfg.meeting_save_dir == str(tmp_path / "meetings")


def test_ui_start_minimized_default_false(tmp_path: Path) -> None:
    """Missing ui.start_minimized → False (do not start hidden on first run)."""
    path = tmp_path / "config.yaml"
    path.write_text("base_url: http://127.0.0.1:8900\n", encoding="utf-8")
    cfg = VoiceClientConfig.load(path)
    assert cfg.ui_start_minimized is False


def test_ui_start_minimized_parse_and_merge(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "base_url: http://127.0.0.1:8900\nui:\n  start_minimized: true\n",
        encoding="utf-8",
    )
    cfg = VoiceClientConfig.load(path)
    assert cfg.ui_start_minimized is True

    merge_config_yaml(path, {"ui": {"start_minimized": False}})
    cfg2 = VoiceClientConfig.load(path)
    assert cfg2.ui_start_minimized is False


def test_resolve_meetings_save_dir_default_and_override(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KRABOBOT_VOICE_CONFIG_DIR", str(tmp_path))
    cfg = VoiceClientConfig(meeting_save_dir="")
    assert resolve_meetings_save_dir(cfg) == default_meetings_dir().resolve()

    override = tmp_path / "custom-meetings"
    cfg2 = VoiceClientConfig(meeting_save_dir=str(override))
    assert resolve_meetings_save_dir(cfg2) == override.resolve()


def test_log_hooks_status_bus() -> None:
    from krabobot_voice.app import _log

    bus = StatusBus()
    set_status_bus(bus)
    try:
        _log("hello-from-log")
        snap = bus.snapshot()
        assert "hello-from-log" in snap.log_lines
    finally:
        set_status_bus(None)


def test_turn_activity_thinking_clears() -> None:
    """Non-blocking turn publishes thinking activity via StatusBus."""
    bus = StatusBus()
    bus.set_mode("listen")
    bus.set_activity("thinking")
    assert bus.snapshot().activity == "thinking"
    bus.set_activity("")
    assert bus.snapshot().activity == ""
