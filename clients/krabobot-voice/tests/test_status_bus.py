"""Unit tests for StatusBus (queue + snapshot)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.status_bus import (  # noqa: E402
    StatusBus,
    display_mode,
    get_status_bus,
    set_status_bus,
)


def test_emit_log_and_snapshot() -> None:
    bus = StatusBus(max_log=3)
    bus.emit_log("one")
    bus.emit_log("two")
    bus.emit_log("three")
    bus.emit_log("four")
    snap = bus.snapshot()
    assert snap.log_lines == ["two", "three", "four"]
    events = bus.drain_events()
    assert [e.kind for e in events] == ["log", "log", "log", "log"]
    assert events[-1].payload == "four"


def test_mode_activity_uploading() -> None:
    bus = StatusBus()
    bus.set_mode("listen")
    bus.set_activity("thinking")
    bus.set_status("thinking…")
    snap = bus.snapshot()
    assert snap.mode == "listen"
    assert snap.activity == "thinking"
    assert display_mode(snap) == "thinking"

    bus.set_uploading(True)
    snap = bus.snapshot()
    assert snap.uploading is True
    assert snap.activity == "uploading"
    assert display_mode(snap) == "uploading"

    bus.set_uploading(False)
    bus.set_activity("")
    bus.set_mode("meeting")
    snap = bus.snapshot()
    assert snap.meeting == "recording"
    assert display_mode(snap) == "meeting"


def test_set_paths_and_global_bus() -> None:
    bus = StatusBus()
    bus.set_paths(config_path="C:/app/config.yaml", meetings_dir="C:/app/meetings")
    snap = bus.snapshot()
    assert snap.config_path.endswith("config.yaml")
    assert snap.meetings_dir.endswith("meetings")

    set_status_bus(bus)
    try:
        assert get_status_bus() is bus
    finally:
        set_status_bus(None)
    assert get_status_bus() is None


def test_duplicate_status_suppressed() -> None:
    bus = StatusBus()
    bus.set_status("waiting…")
    bus.set_status("waiting…")
    events = bus.drain_events()
    assert len([e for e in events if e.kind == "status"]) == 1
