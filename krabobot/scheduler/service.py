"""Task scheduler: one timer from process start, cron store + HEARTBEAT.md."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from loguru import logger

from krabobot.cron.service import CronService
from krabobot.heartbeat.service import (
    heartbeat_check_interval,
    heartbeat_file_is_boilerplate,
    scheduler_tick_interval,
    unfinished_heartbeat_tasks,
)
from krabobot.utils.file_lock import InterProcessLock

if TYPE_CHECKING:
    from krabobot.providers.base import LLMProvider

DEFAULT_TICK_INTERVAL_S = 60
DEFAULT_HEARTBEAT_INTERVAL_S = 30 * 60


class TaskScheduler:
    """The only in-process clock while gateway/serve run.

    From startup it ticks every ``tick_interval_s`` (default 1 minute), reloads
    ``jobs.json`` through :class:`~krabobot.cron.service.CronService`, and runs
  due jobs. ``HEARTBEAT.md`` is a built-in task on ``heartbeat_interval_s``
    (default 30 minutes). This is not the system crontab.
    """

    def __init__(
        self,
        cron: CronService,
        *,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        on_heartbeat_execute: Callable[[str], Coroutine[Any, Any, str]] | None = None,
        on_heartbeat_notify: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        tick_interval_s: int = DEFAULT_TICK_INTERVAL_S,
        heartbeat_interval_s: int = DEFAULT_HEARTBEAT_INTERVAL_S,
        enabled: bool = True,
    ):
        self._cron = cron
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.on_heartbeat_execute = on_heartbeat_execute
        self.on_heartbeat_notify = on_heartbeat_notify
        self.tick_interval_s = scheduler_tick_interval(tick_interval_s)
        self.heartbeat_interval_s = heartbeat_check_interval(heartbeat_interval_s)
        self.enabled = enabled
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_heartbeat_check: float | None = None
        self._file_lock = InterProcessLock(workspace / "runtime" / "scheduler.lock")

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

    async def start(self) -> None:
        """Bootstrap the job store and start the tick loop from now."""
        if not self.enabled:
            logger.info("Task scheduler disabled (gateway.heartbeat.enabled=false)")
            return
        if self._running:
            logger.warning("Task scheduler already running")
            return

        await self._cron.bootstrap()
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="krabobot-scheduler")
        logger.info(
            "Task scheduler started (cron every {}s, HEARTBEAT.md every {}s)",
            self.tick_interval_s,
            self.heartbeat_interval_s,
        )

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        """Fixed interval from process start — first tick runs immediately."""
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
            except Exception:
                logger.exception("Task scheduler tick failed")

    async def _tasks_to_run(self, content: str) -> str | None:
        open_tasks = unfinished_heartbeat_tasks(content)
        if not open_tasks:
            if heartbeat_file_is_boilerplate(content):
                logger.debug("Task scheduler: HEARTBEAT.md has no tasks")
            return None
        return "\n".join(open_tasks)

    async def _run_heartbeat_builtin(self) -> None:
        from krabobot.utils.evaluator import evaluate_response

        content = self._read_heartbeat_file()
        if not content:
            logger.debug("Task scheduler: HEARTBEAT.md missing or empty")
            return

        logger.info("Task scheduler: HEARTBEAT.md (built-in task)...")
        tasks = await self._tasks_to_run(content)
        if not tasks or not self.on_heartbeat_execute:
            return

        logger.info("Task scheduler: running HEARTBEAT.md tasks...")
        try:
            response = await self.on_heartbeat_execute(tasks)
            if not response:
                return
            should_notify = await evaluate_response(
                response, tasks, self.provider, self.model,
            )
            if should_notify and self.on_heartbeat_notify:
                await self.on_heartbeat_notify(response)
        except Exception:
            logger.exception("Task scheduler: HEARTBEAT.md task failed")

    async def _tick(self) -> None:
        if not self._file_lock.try_acquire():
            logger.debug("Task scheduler: skip tick, lock held by another process")
            return
        try:
            ran = await self._cron.run_due_jobs()
            if ran:
                logger.info("Task scheduler: ran {} cron job(s)", ran)

            if not self._heartbeat_due():
                return

            self._last_heartbeat_check = time.monotonic()
            await self._run_heartbeat_builtin()
        finally:
            self._file_lock.release()

    async def trigger_now(self) -> str | None:
        """Run due cron jobs and force a HEARTBEAT.md pass."""
        await self._cron.run_due_jobs()
        content = self._read_heartbeat_file()
        if not content or not self.on_heartbeat_execute:
            return None
        tasks = await self._tasks_to_run(content)
        if not tasks:
            return None
        self._last_heartbeat_check = time.monotonic()
        return await self.on_heartbeat_execute(tasks)
