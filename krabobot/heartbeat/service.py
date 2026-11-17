"""Heartbeat service - periodic agent wake-up to check for tasks."""

from __future__ import annotations

import asyncio
import re
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

DEFAULT_SCHEDULER_INTERVAL_S = 30 * 60


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


def scheduler_interval(interval_s: int) -> int:
    """Return the poll period. Non-positive values fall back to 30 minutes."""
    if interval_s <= 0:
        return DEFAULT_SCHEDULER_INTERVAL_S
    return interval_s


class HeartbeatService:
    """In-process scheduler (config key ``gateway.heartbeat``).

    On each tick it runs due cron jobs, then checks ``HEARTBEAT.md`` for open
    recurring tasks. Unfinished items are unchecked boxes, list items, and lines
    under Active Tasks; the Completed section is ignored.
    """

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        on_execute: Callable[[str], Coroutine[Any, Any, str]] | None = None,
        on_notify: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        interval_s: int = DEFAULT_SCHEDULER_INTERVAL_S,
        enabled: bool = True,
        timezone: str | None = None,
        cron_service: Any | None = None,
    ):
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.on_execute = on_execute
        self.on_notify = on_notify
        self.interval_s = scheduler_interval(interval_s)
        self.enabled = enabled
        self.timezone = timezone
        self.cron_service = cron_service
        self._running = False
        self._task: asyncio.Task | None = None
        self._file_lock = InterProcessLock(workspace / "runtime" / "heartbeat.lock")

    @property
    def heartbeat_file(self) -> Path:
        return self.workspace / "HEARTBEAT.md"

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
        """Start the heartbeat service."""
        if not self.enabled:
            logger.info("Heartbeat disabled")
            return
        if self._running:
            logger.warning("Heartbeat already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="krabobot-heartbeat")
        logger.info(
            "In-process scheduler started (every {}s: cron jobs + HEARTBEAT.md)",
            self.interval_s,
        )

    def stop(self) -> None:
        """Stop the heartbeat service."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        """Main heartbeat loop. The first pass does not wait a full interval."""
        first = True
        while self._running:
            try:
                if not first:
                    await asyncio.sleep(self.interval_s)
                first = False
                if self._running:
                    await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat error: {}", e)

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
        """Resolve unfinished HEARTBEAT.md work. ``None`` means nothing to run.

        The phase-1 model call used to be allowed to answer ``skip`` for open
        checklist items. Unfinished tasks are taken from the file directly.
        """
        open_tasks = unfinished_heartbeat_tasks(content)
        if not open_tasks:
            if heartbeat_file_is_boilerplate(content):
                logger.debug("Heartbeat: HEARTBEAT.md has no tasks")
            else:
                logger.debug("Heartbeat: no unfinished tasks")
            return None
        return "\n".join(open_tasks)

    async def _tick(self) -> None:
        """Execute a single heartbeat tick."""
        from krabobot.utils.evaluator import evaluate_response

        if not self._file_lock.try_acquire():
            logger.debug("Heartbeat: skip tick, another process holds the lock")
            return
        try:
            await self._run_due_cron_jobs()

            content = self._read_heartbeat_file()
            if not content:
                logger.debug("Heartbeat: HEARTBEAT.md missing or empty")
                return

            logger.info("Heartbeat: checking for unfinished tasks...")
            try:
                tasks = await self._tasks_to_run(content)
                if not tasks:
                    return

                logger.info("Heartbeat: unfinished tasks found, executing...")
                if self.on_execute:
                    response = await self.on_execute(tasks)

                    if response:
                        should_notify = await evaluate_response(
                            response, tasks, self.provider, self.model,
                        )
                        if should_notify and self.on_notify:
                            logger.info("Heartbeat: completed, delivering response")
                            await self.on_notify(response)
                        else:
                            logger.info("Heartbeat: silenced by post-run evaluation")
            except Exception:
                logger.exception("Heartbeat execution failed")
        finally:
            self._file_lock.release()

    async def trigger_now(self) -> str | None:
        """Manually trigger a heartbeat."""
        await self._run_due_cron_jobs()
        content = self._read_heartbeat_file()
        if not content or not self.on_execute:
            return None
        tasks = await self._tasks_to_run(content)
        if not tasks:
            return None
        return await self.on_execute(tasks)
