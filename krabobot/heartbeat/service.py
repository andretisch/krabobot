"""HEARTBEAT.md parsing helpers (built-in scheduler task)."""

from __future__ import annotations

import re

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_COMPLETED_HEADING = re.compile(
    r"^#{1,6}\s+(completed|done|выполнено|выполненные|готово|сделано)\b",
    re.IGNORECASE,
)
_ACTIVE_HEADING = re.compile(
    r"^#{1,6}\s+(active|open|todo|tasks?|активн\w*|задач\w*|текущ\w*)\b",
    re.IGNORECASE,
)
_CHECKBOX = re.compile(r"^(?:[-*+]|\d+[.)])\s+\[([ xX])\]\s*(.*)$")
_BULLET = re.compile(r"^(?:[-*+]|\d+[.)])\s+(.*)$")

DEFAULT_TICK_INTERVAL_S = 60
DEFAULT_HEARTBEAT_INTERVAL_S = 30 * 60


def unfinished_heartbeat_tasks(content: str) -> list[str]:
    """Return open tasks in ``HEARTBEAT.md`` (Completed section ignored)."""
    text = _HTML_COMMENT.sub("", content)
    section = "preamble"
    tasks: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if _COMPLETED_HEADING.match(line):
                section = "completed"
            elif _ACTIVE_HEADING.match(line):
                section = "active"
            else:
                section = "preamble"
            continue
        if section == "completed":
            continue
        checkbox = _CHECKBOX.match(line)
        if checkbox:
            if checkbox.group(1) == " ":
                task = checkbox.group(2).strip()
                if task:
                    tasks.append(task)
            continue
        bullet = _BULLET.match(line)
        if bullet:
            task = bullet.group(1).strip()
            if task:
                tasks.append(task)
            continue
        if section == "active":
            tasks.append(line)
    return tasks


def heartbeat_file_is_boilerplate(content: str) -> bool:
    text = _HTML_COMMENT.sub("", content)
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        return False
    return True


def scheduler_tick_interval(interval_s: int) -> int:
    if interval_s <= 0:
        return DEFAULT_TICK_INTERVAL_S
    return interval_s


def heartbeat_check_interval(interval_s: int) -> int:
    if interval_s <= 0:
        return DEFAULT_HEARTBEAT_INTERVAL_S
    return interval_s
