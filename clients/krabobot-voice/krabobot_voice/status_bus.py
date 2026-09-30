"""Thread-safe status / log bus between the voice loop and the Windows UI."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class StatusEvent:
    """One published change (UI drains these for incremental updates)."""

    kind: str  # mode | status | log | meeting | uploading | paths | activity
    payload: Any = None
    ts: float = field(default_factory=time.monotonic)


@dataclass
class StatusSnapshot:
    """Latest UI-facing state (copy-on-read)."""

    mode: str = "idle"  # idle | listen | dialog | meeting
    activity: str = ""  # thinking | uploading | "" (overlay on mode)
    status: str = ""
    meeting: str = "idle"  # idle | recording
    uploading: bool = False
    config_path: str = ""
    meetings_dir: str = ""
    log_lines: list[str] = field(default_factory=list)


class StatusBus:
    """Queue + snapshot for cross-thread UI updates."""

    def __init__(self, *, max_log: int = 400, max_queue: int = 2000) -> None:
        self._lock = threading.Lock()
        self._snap = StatusSnapshot()
        self._max_log = max(1, int(max_log))
        self._events: queue.Queue[StatusEvent | None] = queue.Queue(maxsize=max(100, int(max_queue)))

    def snapshot(self) -> StatusSnapshot:
        with self._lock:
            return replace(self._snap, log_lines=list(self._snap.log_lines))

    def drain_events(self, *, max_n: int = 200) -> list[StatusEvent]:
        out: list[StatusEvent] = []
        for _ in range(max(1, int(max_n))):
            try:
                item = self._events.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            out.append(item)
        return out

    def set_mode(self, mode: str) -> None:
        cleaned = (mode or "idle").strip().lower() or "idle"
        with self._lock:
            if self._snap.mode == cleaned:
                return
            self._snap.mode = cleaned
            self._snap.meeting = "recording" if cleaned == "meeting" else "idle"
        self._push(StatusEvent("mode", cleaned))

    def set_activity(self, activity: str) -> None:
        cleaned = (activity or "").strip().lower()
        with self._lock:
            if self._snap.activity == cleaned:
                return
            self._snap.activity = cleaned
        self._push(StatusEvent("activity", cleaned))

    def set_status(self, message: str) -> None:
        msg = str(message or "")
        with self._lock:
            if self._snap.status == msg:
                return
            self._snap.status = msg
        self._push(StatusEvent("status", msg))

    def set_uploading(self, uploading: bool) -> None:
        flag = bool(uploading)
        with self._lock:
            if self._snap.uploading == flag:
                return
            self._snap.uploading = flag
            if flag:
                self._snap.activity = "uploading"
            elif self._snap.activity == "uploading":
                self._snap.activity = ""
        self._push(StatusEvent("uploading", flag))
        if flag:
            self._push(StatusEvent("activity", "uploading"))
        else:
            self._push(StatusEvent("activity", ""))

    def set_paths(self, *, config_path: str = "", meetings_dir: str = "") -> None:
        cfg = str(config_path or "")
        meetings = str(meetings_dir or "")
        with self._lock:
            changed = False
            if cfg and self._snap.config_path != cfg:
                self._snap.config_path = cfg
                changed = True
            if meetings and self._snap.meetings_dir != meetings:
                self._snap.meetings_dir = meetings
                changed = True
            if not changed:
                return
            payload = {
                "config_path": self._snap.config_path,
                "meetings_dir": self._snap.meetings_dir,
            }
        self._push(StatusEvent("paths", payload))

    def emit_log(self, line: str) -> None:
        text = str(line or "").rstrip()
        if not text:
            return
        with self._lock:
            lines = self._snap.log_lines
            lines.append(text)
            overflow = len(lines) - self._max_log
            if overflow > 0:
                del lines[:overflow]
        self._push(StatusEvent("log", text))

    def _push(self, event: StatusEvent) -> None:
        try:
            self._events.put_nowait(event)
        except queue.Full:
            try:
                self._events.get_nowait()
            except queue.Empty:
                pass
            try:
                self._events.put_nowait(event)
            except queue.Full:
                pass


_bus: StatusBus | None = None
_bus_lock = threading.Lock()


def get_status_bus() -> StatusBus | None:
    return _bus


def set_status_bus(bus: StatusBus | None) -> None:
    global _bus
    with _bus_lock:
        _bus = bus


def display_mode(snapshot: StatusSnapshot | None = None) -> str:
    """UI label: activity overlay wins over session mode when set."""
    snap = snapshot if snapshot is not None else (
        _bus.snapshot() if _bus is not None else StatusSnapshot()
    )
    activity = (snap.activity or "").strip().lower()
    if activity in {"thinking", "uploading"}:
        return activity
    mode = (snap.mode or "idle").strip().lower() or "idle"
    return mode
