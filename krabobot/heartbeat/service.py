"""In-process task scheduler (cron due jobs + HEARTBEAT.md)."""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from loguru import logger

from krabobot.utils.file_lock import InterProcessLock

if TYPE_CHECKING:
    from krabobot.providers.base import LLMProvider

_HEARTBEAT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "heartbeat",
            "description": "Report heartbeat decision after reviewing tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["skip", "run"],
                        "description": "skip = nothing to do, run = has active tasks",
                    },
                    "tasks": {
                        "type": "string",
                        "description": "Natural-language summary of active tasks (required for run)",
                    },
                },
                "required": ["action"],
            },
        },
    }
]


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
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

DEFAULT_TICK_INTERVAL_S = 60
DEFAULT_HEARTBEAT_INTERVAL_S = 30 * 60


def unfinished_heartbeat_tasks(content: str) -> list[str]:
    """Return tasks that are still open in ``HEARTBEAT.md``.

    Unchecked boxes and list items count. Plain lines count only under an
    Active/Tasks heading. A Completed/Done/Выполнено section is ignored, even
    when its boxes are still unchecked.
    """
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
    """True when the file has no tasks left after headers and comments."""
    text = _HTML_COMMENT.sub("", content)
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        return False
    return True


def scheduler_tick_interval(interval_s: int) -> int:
    """Cron poll period. Non-positive values fall back to one minute."""
    if interval_s <= 0:
        return DEFAULT_TICK_INTERVAL_S
    return interval_s


def heartbeat_check_interval(interval_s: int) -> int:
    """Built-in HEARTBEAT.md task period. Non-positive → 30 minutes."""
    if interval_s <= 0:
        return DEFAULT_HEARTBEAT_INTERVAL_S
    return interval_s


class HeartbeatService:
    """Task scheduler: cron jobs every tick; ``HEARTBEAT.md`` on its own interval."""

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        on_execute: Callable[[str], Coroutine[Any, Any, str]] | None = None,
        on_notify: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        tick_interval_s: int = DEFAULT_TICK_INTERVAL_S,
        interval_s: int = DEFAULT_HEARTBEAT_INTERVAL_S,
        enabled: bool = True,
        timezone: str | None = None,
        cron_service: Any | None = None,
    ):
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.on_execute = on_execute
        self.on_notify = on_notify
        self.tick_interval_s = scheduler_tick_interval(tick_interval_s)
        self.heartbeat_interval_s = heartbeat_check_interval(interval_s)
        self.enabled = enabled
        self.timezone = timezone
        self.cron_service = cron_service
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_heartbeat_check: float | None = None
        self._file_lock = InterProcessLock(workspace / "runtime" / "heartbeat.lock")

    @property
    def heartbeat_file(self) -> Path:
        return self.workspace / "HEARTBEAT.md"

    def _heartbeat_due(self) -> bool:
        if self._last_heartbeat_check is None:
            return True
        return (time.monotonic() - self._last_heartbeat_check) >= self.heartbeat_interval_s

    def _read_heartbeat_file(self) -> str | None:
        if self.heartbeat_file.exists():
            try:
                return self.heartbeat_file.read_text(encoding="utf-8")
            except Exception:
                return None
        return None

    async def _decide(self, content: str) -> tuple[str, str]:
        """Phase 1: ask LLM to decide skip/run via virtual tool call.

        Returns (action, tasks) where action is 'skip' or 'run'.
        """
        from krabobot.utils.helpers import current_time_str

        response = await self.provider.chat_with_retry(
            messages=[
                {"role": "system", "content": "You are a heartbeat agent. Call the heartbeat tool to report your decision."},
                {"role": "user", "content": (
                    f"Current Time: {current_time_str(self.timezone)}\n\n"
                    "Review the following HEARTBEAT.md and decide whether there are unfinished tasks.\n"
                    "Unfinished means unchecked '- [ ]' items, open list items, and lines under "
                    "Active Tasks. Items under Completed/Done/Выполнено are finished — do not run those.\n"
                    "If any unfinished task remains, action must be 'run' and tasks must list them.\n"
                    "Use 'skip' only when every task is done or the file has no tasks.\n\n"
                    f"{content}"
                )},
            ],
            tools=_HEARTBEAT_TOOL,
            model=self.model,
        )

        if not response.has_tool_calls:
            return "skip", ""

        args = response.tool_calls[0].arguments
        return args.get("action", "skip"), args.get("tasks", "")

    async def start(self) -> None:
        """Start the scheduler."""
        if not self.enabled:
            logger.info("Scheduler disabled (gateway.heartbeat.enabled=false)")
            return
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="krabobot-scheduler")
        logger.info(
            "Task scheduler started (cron every {}s, HEARTBEAT.md every {}s)",
            self.tick_interval_s,
            self.heartbeat_interval_s,
        )

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        """Poll cron every tick; HEARTBEAT.md only when its interval elapsed."""
        first = True
        while self._running:
            try:
                if not first:
                    await asyncio.sleep(self.tick_interval_s)
                first = False
                if self._running:
                    await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scheduler error: {}", e)

    async def _run_due_cron_jobs(self) -> None:
        """Run cron jobs whose deadline has arrived."""
        cron = self.cron_service
        if cron is None:
            return
        try:
            ran = await cron.run_due_jobs()
        except Exception:
            logger.exception("Scheduler: failed to run due cron jobs")
            return
        if ran:
            logger.info("Scheduler: ran {} due cron job(s)", ran)

    async def _tasks_to_run(self, content: str) -> str | None:
        """Resolve unfinished HEARTBEAT.md work. ``None`` means nothing to run."""
        open_tasks = unfinished_heartbeat_tasks(content)
        if not open_tasks:
            if heartbeat_file_is_boilerplate(content):
                logger.debug("Scheduler: HEARTBEAT.md has no tasks")
            else:
                logger.debug("Scheduler: no unfinished HEARTBEAT.md tasks")
            return None
        return "\n".join(open_tasks)

    async def _run_heartbeat_file_task(self) -> None:
        """Built-in task: check ``HEARTBEAT.md`` for open recurring work."""
        from krabobot.utils.evaluator import evaluate_response

        content = self._read_heartbeat_file()
        if not content:
            logger.debug("Scheduler: HEARTBEAT.md missing or empty")
            return

        logger.info("Scheduler: HEARTBEAT.md check...")
        try:
            tasks = await self._tasks_to_run(content)
            if not tasks:
                return

            logger.info("Scheduler: unfinished HEARTBEAT.md tasks, executing...")
            if self.on_execute:
                response = await self.on_execute(tasks)

                if response:
                    should_notify = await evaluate_response(
                        response, tasks, self.provider, self.model,
                    )
                    if should_notify and self.on_notify:
                        logger.info("Scheduler: HEARTBEAT.md work done, delivering")
                        await self.on_notify(response)
                    else:
                        logger.info("Scheduler: HEARTBEAT.md response silenced by evaluation")
        except Exception:
            logger.exception("Scheduler: HEARTBEAT.md task failed")

    async def _tick(self) -> None:
        """One scheduler tick: cron always; HEARTBEAT.md when due."""
        if not self._file_lock.try_acquire():
            logger.debug("Scheduler: skip tick, another process holds the lock")
            return
        try:
            await self._run_due_cron_jobs()

            if not self._heartbeat_due():
                return

            self._last_heartbeat_check = time.monotonic()
            await self._run_heartbeat_file_task()
        finally:
            self._file_lock.release()

    async def trigger_now(self) -> str | None:
        """Manual run: due cron jobs + immediate HEARTBEAT.md check."""
        await self._run_due_cron_jobs()
        content = self._read_heartbeat_file()
        if not content or not self.on_execute:
            return None
        tasks = await self._tasks_to_run(content)
        if not tasks:
            return None
        self._last_heartbeat_check = time.monotonic()
        return await self.on_execute(tasks)
