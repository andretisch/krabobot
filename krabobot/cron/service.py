"""In-process scheduler for agent tasks.

This is not the system crontab. ``gateway`` and ``serve`` each run a scheduler
loop against ``jobs.json``. A file lock makes sure only one process executes
a due job. Missed deadlines are caught up once after downtime or a restart
instead of being pushed a full interval into the future.
"""

import asyncio
import contextlib
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine

from loguru import logger

from krabobot.cron.types import (
    CronJob,
    CronJobState,
    CronPayload,
    CronRunRecord,
    CronSchedule,
    CronStore,
)
from krabobot.utils.file_lock import InterProcessLock


def _now_ms() -> int:
    return int(time.time() * 1000)


def _compute_next_run(schedule: CronSchedule, now_ms: int) -> int | None:
    """Compute next run time in ms."""
    if schedule.kind == "at":
        return schedule.at_ms if schedule.at_ms and schedule.at_ms > now_ms else None

    if schedule.kind == "every":
        if not schedule.every_ms or schedule.every_ms <= 0:
            return None
        # Next interval from now
        return now_ms + schedule.every_ms

    if schedule.kind == "cron" and schedule.expr:
        try:
            from zoneinfo import ZoneInfo

            from croniter import croniter
            # Use caller-provided reference time for deterministic scheduling
            base_time = now_ms / 1000
            tz = ZoneInfo(schedule.tz) if schedule.tz else datetime.now().astimezone().tzinfo
            base_dt = datetime.fromtimestamp(base_time, tz=tz)
            cron = croniter(schedule.expr, base_dt)
            next_dt = cron.get_next(datetime)
            return int(next_dt.timestamp() * 1000)
        except Exception as exc:
            logger.warning(
                "Cron: cannot compute next run for expr={!r} tz={!r}: {}",
                schedule.expr,
                schedule.tz,
                exc,
            )
            return None

    return None


def _resume_next_run(
    schedule: CronSchedule,
    persisted_ms: int | None,
    now_ms: int,
    last_run_at_ms: int | None = None,
) -> int | None:
    """Pick the next deadline when the scheduler starts.

    Recomputing every job from ``now`` postpones ``every`` schedules by a full
    interval on each gateway restart, so a process that restarts more often
    than the interval never fires. Overdue deadlines are kept so the missed
    run happens once, then the schedule advances.
    """
    if schedule.kind == "every" and schedule.every_ms and schedule.every_ms > 0:
        horizon = now_ms + schedule.every_ms
        if persisted_ms is not None and persisted_ms <= horizon:
            return persisted_ms
        if last_run_at_ms:
            nxt = last_run_at_ms + schedule.every_ms
            return nxt if nxt > now_ms else now_ms
        return _compute_next_run(schedule, now_ms)

    if schedule.kind == "at":
        if not schedule.at_ms or last_run_at_ms:
            return None
        # A one-shot that was due while the process was down still runs once.
        return schedule.at_ms

    if schedule.kind == "cron":
        if persisted_ms is not None:
            return persisted_ms
        return _compute_next_run(schedule, now_ms)

    return _compute_next_run(schedule, now_ms)


def _validate_schedule_for_add(schedule: CronSchedule) -> None:
    """Validate schedule fields that would otherwise create non-runnable jobs."""
    if schedule.tz and schedule.kind != "cron":
        raise ValueError("tz can only be used with cron schedules")

    if schedule.kind == "cron" and schedule.tz:
        try:
            from zoneinfo import ZoneInfo

            ZoneInfo(schedule.tz)
        except Exception:
            raise ValueError(f"unknown timezone '{schedule.tz}'") from None


class CronService:
    """Service for managing and executing scheduled jobs."""

    _MAX_RUN_HISTORY = 20

    def __init__(
        self,
        store_path: Path,
        on_job: Callable[[CronJob], Coroutine[Any, Any, str | None]] | None = None,
        *,
        max_sleep_ms: int = 30_000,
    ):
        self.store_path = store_path
        self.on_job = on_job
        self._store: CronStore | None = None
        self._last_mtime: float = 0.0
        self._timer_task: asyncio.Task | None = None
        self._running = False
        # Wake the loop early when a job is added or removed. Never cancel the
        # running tick: that task is also the one executing the job.
        self._wake = asyncio.Event()
        self._run_lock = asyncio.Lock()
        self._executing = False
        self._max_sleep_ms = max(1_000, max_sleep_ms)
        self._file_lock = InterProcessLock(store_path.parent / f"{store_path.name}.lock")

    def _load_store(self) -> CronStore:
        """Load jobs from disk. Reloads automatically if file was modified externally.

        While a job callback is running, keep the in-memory store. Reloading
        mid-callback drops the deadline this tick already advanced and the same
        job fires again forever.
        """
        if self._executing and self._store is not None:
            return self._store
        if self._store and self.store_path.exists():
            mtime = self.store_path.stat().st_mtime
            if mtime != self._last_mtime:
                logger.info("Cron: jobs.json modified externally, reloading")
                self._store = None
        if self._store:
            return self._store

        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text(encoding="utf-8"))
                jobs = []
                for j in data.get("jobs", []):
                    jobs.append(CronJob(
                        id=j["id"],
                        name=j["name"],
                        enabled=j.get("enabled", True),
                        schedule=CronSchedule(
                            kind=j["schedule"]["kind"],
                            at_ms=j["schedule"].get("atMs"),
                            every_ms=j["schedule"].get("everyMs"),
                            expr=j["schedule"].get("expr"),
                            tz=j["schedule"].get("tz"),
                        ),
                        payload=CronPayload(
                            kind=j["payload"].get("kind", "agent_turn"),
                            message=j["payload"].get("message", ""),
                            deliver=j["payload"].get("deliver", False),
                            channel=j["payload"].get("channel"),
                            to=j["payload"].get("to"),
                            from_sender_id=j["payload"].get("fromSenderId"),
                        ),
                        state=CronJobState(
                            next_run_at_ms=j.get("state", {}).get("nextRunAtMs"),
                            last_run_at_ms=j.get("state", {}).get("lastRunAtMs"),
                            last_status=j.get("state", {}).get("lastStatus"),
                            last_error=j.get("state", {}).get("lastError"),
                            run_history=[
                                CronRunRecord(
                                    run_at_ms=r["runAtMs"],
                                    status=r["status"],
                                    duration_ms=r.get("durationMs", 0),
                                    error=r.get("error"),
                                )
                                for r in j.get("state", {}).get("runHistory", [])
                            ],
                        ),
                        created_at_ms=j.get("createdAtMs", 0),
                        updated_at_ms=j.get("updatedAtMs", 0),
                        delete_after_run=j.get("deleteAfterRun", False),
                    ))
                self._store = CronStore(jobs=jobs)
            except Exception as e:
                logger.warning("Failed to load cron store: {}", e)
                self._store = CronStore()
        else:
            self._store = CronStore()

        return self._store

    def _save_store(self) -> None:
        """Save jobs to disk."""
        if not self._store:
            return

        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": self._store.version,
            "jobs": [
                {
                    "id": j.id,
                    "name": j.name,
                    "enabled": j.enabled,
                    "schedule": {
                        "kind": j.schedule.kind,
                        "atMs": j.schedule.at_ms,
                        "everyMs": j.schedule.every_ms,
                        "expr": j.schedule.expr,
                        "tz": j.schedule.tz,
                    },
                    "payload": {
                        "kind": j.payload.kind,
                        "message": j.payload.message,
                        "deliver": j.payload.deliver,
                        "channel": j.payload.channel,
                        "to": j.payload.to,
                        "fromSenderId": j.payload.from_sender_id,
                    },
                    "state": {
                        "nextRunAtMs": j.state.next_run_at_ms,
                        "lastRunAtMs": j.state.last_run_at_ms,
                        "lastStatus": j.state.last_status,
                        "lastError": j.state.last_error,
                        "runHistory": [
                            {
                                "runAtMs": r.run_at_ms,
                                "status": r.status,
                                "durationMs": r.duration_ms,
                                "error": r.error,
                            }
                            for r in j.state.run_history
                        ],
                    },
                    "createdAtMs": j.created_at_ms,
                    "updatedAtMs": j.updated_at_ms,
                    "deleteAfterRun": j.delete_after_run,
                }
                for j in self._store.jobs
            ]
        }

        self.store_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self._last_mtime = self.store_path.stat().st_mtime

    async def start(self, *, start_loop: bool = True) -> None:
        """Load the store and, unless the scheduler owns the clock, arm the timer.

        The in-process scheduler calls :meth:`run_due_jobs` on its interval.
        ``start_loop=False`` keeps this object as the job store without a second
        loop that would run the same job twice.
        """
        if self._running:
            return
        if self._file_lock.try_acquire():
            try:
                self._load_store()
                self._recompute_next_runs()
                self._save_store()
            finally:
                self._file_lock.release()
        else:
            self._load_store()
        if not start_loop:
            logger.info(
                "Cron store ready ({} jobs); in-process scheduler runs due jobs",
                len(self._store.jobs if self._store else []),
            )
            return
        self._running = True
        self._timer_task = asyncio.create_task(self._scheduler_loop(), name="krabobot-cron")
        logger.info(
            "In-process cron scheduler started with {} jobs",
            len(self._store.jobs if self._store else []),
        )

    def stop(self) -> None:
        """Stop the cron service."""
        self._running = False
        self._wake.set()
        task = self._timer_task
        self._timer_task = None
        if task and not task.done():
            task.cancel()

    def _recompute_next_runs(self) -> None:
        """Fill deadlines on startup without postponing pending or overdue jobs."""
        if not self._store:
            return
        now = _now_ms()
        for job in self._store.jobs:
            if not job.enabled:
                continue
            job.state.next_run_at_ms = _resume_next_run(
                job.schedule,
                job.state.next_run_at_ms,
                now,
                job.state.last_run_at_ms,
            )

    def _get_next_wake_ms(self) -> int | None:
        """Get the earliest next run time across all jobs."""
        if not self._store:
            return None
        times = [j.state.next_run_at_ms for j in self._store.jobs
                 if j.enabled and j.state.next_run_at_ms]
        return min(times) if times else None

    def _sleep_seconds(self) -> float:
        """How long to idle before the next check.

        The cap re-reads ``jobs.json`` and the wall clock so an NTP step or an
        edit from another process is noticed even when the next deadline is far
        away. A single long ``asyncio.sleep`` until that deadline is not.
        """
        nxt = self._get_next_wake_ms()
        if nxt is None:
            return self._max_sleep_ms / 1000
        delay_ms = min(self._max_sleep_ms, max(0, nxt - _now_ms()))
        return delay_ms / 1000

    def _arm_timer(self) -> None:
        """Ask the scheduler loop to wake up. Does not cancel a running job."""
        self._wake.set()

    async def _wait(self, delay_s: float) -> None:
        if delay_s <= 0 or not self._running:
            return
        sleep_task = asyncio.create_task(asyncio.sleep(delay_s))
        wake_task = asyncio.create_task(self._wake.wait())
        try:
            done, pending = await asyncio.wait(
                {sleep_task, wake_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            sleep_task.cancel()
            wake_task.cancel()
            await asyncio.gather(sleep_task, wake_task, return_exceptions=True)
            raise
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if wake_task in done:
            self._wake.clear()

    async def _scheduler_loop(self) -> None:
        """Keep scheduling until stopped. One failed tick does not kill the loop."""
        while self._running:
            try:
                delay = self._sleep_seconds()
                if delay > 0:
                    await self._wait(delay)
                if not self._running:
                    break
                acquired = await self._on_timer()
                if not acquired and self._running:
                    await self._wait(1.0)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Cron: scheduler tick failed; retrying")
                if self._running:
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._wait(1.0)

    async def _on_timer(self) -> bool:
        """Run due jobs. Return False when another process holds the lock."""
        async with self._run_lock:
            if not self._file_lock.try_acquire():
                logger.debug("Cron: tick skipped, another process holds the scheduler lock")
                return False
            try:
                self._load_store()
                if not self._store:
                    return True

                now = _now_ms()
                due_ids = [
                    j.id for j in self._store.jobs
                    if j.enabled and j.state.next_run_at_ms and now >= j.state.next_run_at_ms
                ]
                if not due_ids:
                    return True

                self._executing = True
                try:
                    for job_id in due_ids:
                        job = next((j for j in self._store.jobs if j.id == job_id), None)
                        if job is None or not job.enabled:
                            continue
                        due_at = job.state.next_run_at_ms
                        if not due_at or due_at > _now_ms():
                            continue
                        if due_at < _now_ms() - 1000:
                            logger.info("Cron: running overdue job '{}' ({})", job.name, job.id)
                        await self._execute_job(job)
                    self._save_store()
                finally:
                    self._executing = False
                return True
            finally:
                self._file_lock.release()

    async def _execute_job(self, job: CronJob) -> None:
        """Execute a single job."""
        start_ms = _now_ms()
        logger.info("Cron: executing job '{}' ({})", job.name, job.id)

        try:
            if self.on_job:
                await self.on_job(job)

            job.state.last_status = "ok"
            job.state.last_error = None
            logger.info("Cron: job '{}' completed", job.name)

        except Exception as e:
            job.state.last_status = "error"
            job.state.last_error = str(e)
            logger.error("Cron: job '{}' failed: {}", job.name, e)

        end_ms = _now_ms()
        job.state.last_run_at_ms = start_ms
        job.updated_at_ms = end_ms

        job.state.run_history.append(CronRunRecord(
            run_at_ms=start_ms,
            status=job.state.last_status,
            duration_ms=end_ms - start_ms,
            error=job.state.last_error,
        ))
        job.state.run_history = job.state.run_history[-self._MAX_RUN_HISTORY:]

        # Handle one-shot jobs
        if job.schedule.kind == "at":
            if job.delete_after_run:
                self._store.jobs = [j for j in self._store.jobs if j.id != job.id]
            else:
                job.enabled = False
                job.state.next_run_at_ms = None
        else:
            # Compute next run
            job.state.next_run_at_ms = _compute_next_run(job.schedule, _now_ms())

    # ========== Public API ==========

    def list_jobs(self, include_disabled: bool = False) -> list[CronJob]:
        """List all jobs."""
        store = self._load_store()
        jobs = store.jobs if include_disabled else [j for j in store.jobs if j.enabled]
        return sorted(jobs, key=lambda j: j.state.next_run_at_ms or float('inf'))

    def add_job(
        self,
        name: str,
        schedule: CronSchedule,
        message: str,
        deliver: bool = False,
        channel: str | None = None,
        to: str | None = None,
        from_sender_id: str | None = None,
        delete_after_run: bool = False,
    ) -> CronJob:
        """Add a new job."""
        store = self._load_store()
        _validate_schedule_for_add(schedule)
        now = _now_ms()

        job = CronJob(
            id=str(uuid.uuid4())[:8],
            name=name,
            enabled=True,
            schedule=schedule,
            payload=CronPayload(
                kind="agent_turn",
                message=message,
                deliver=deliver,
                channel=channel,
                to=to,
                from_sender_id=from_sender_id,
            ),
            state=CronJobState(next_run_at_ms=_compute_next_run(schedule, now)),
            created_at_ms=now,
            updated_at_ms=now,
            delete_after_run=delete_after_run,
        )

        store.jobs.append(job)
        self._save_store()
        self._arm_timer()

        logger.info("Cron: added job '{}' ({})", name, job.id)
        return job

    def remove_job(self, job_id: str) -> bool:
        """Remove a job by ID."""
        store = self._load_store()
        before = len(store.jobs)
        store.jobs = [j for j in store.jobs if j.id != job_id]
        removed = len(store.jobs) < before

        if removed:
            self._save_store()
            self._arm_timer()
            logger.info("Cron: removed job {}", job_id)

        return removed

    def enable_job(self, job_id: str, enabled: bool = True) -> CronJob | None:
        """Enable or disable a job."""
        store = self._load_store()
        for job in store.jobs:
            if job.id == job_id:
                job.enabled = enabled
                job.updated_at_ms = _now_ms()
                if enabled:
                    job.state.next_run_at_ms = _compute_next_run(job.schedule, _now_ms())
                else:
                    job.state.next_run_at_ms = None
                self._save_store()
                self._arm_timer()
                return job
        return None

    async def run_due_jobs(self) -> int:
        """Run every enabled job whose deadline has arrived.

        Called by the in-process scheduler. A job is due when
        ``next_run_at_ms`` is now or earlier; there is no extra grace period.
        """
        async with self._run_lock:
            if not self._file_lock.try_acquire():
                logger.debug("Cron: run_due_jobs skipped, another process holds the lock")
                return 0
            try:
                self._load_store()
                if not self._store:
                    return 0
                now = _now_ms()
                due_ids = [
                    job.id
                    for job in self._store.jobs
                    if job.enabled and job.state.next_run_at_ms and now >= job.state.next_run_at_ms
                ]
                missed_at = [
                    job.id
                    for job in self._store.jobs
                    if job.enabled
                    and job.state.next_run_at_ms is None
                    and job.schedule.kind == "at"
                    and job.schedule.at_ms
                    and job.schedule.at_ms <= now
                    and job.state.last_run_at_ms is None
                ]
                due_ids.extend(job_id for job_id in missed_at if job_id not in due_ids)
                if not due_ids:
                    return 0

                ran = 0
                self._executing = True
                try:
                    for job_id in due_ids:
                        job = next((item for item in self._store.jobs if item.id == job_id), None)
                        if job is None or not job.enabled:
                            continue
                        due_at = job.state.next_run_at_ms
                        if due_at is not None and due_at > _now_ms():
                            continue
                        logger.info("Cron: running due job '{}' ({})", job.name, job.id)
                        await self._execute_job(job)
                        ran += 1
                    if ran:
                        self._save_store()
                finally:
                    self._executing = False
                return ran
            finally:
                self._file_lock.release()

    def overdue_jobs(self, *, grace_ms: int = 60_000) -> list[CronJob]:
        """Enabled jobs whose deadline passed and the scheduler has not run them.

        ``grace_ms`` leaves a window for the scheduler loop so heartbeat does
        not execute a job that is about to fire on its own.
        """
        now = _now_ms()
        cutoff = now - max(0, grace_ms)
        overdue: list[CronJob] = []
        for job in self.list_jobs(include_disabled=False):
            due = job.state.next_run_at_ms
            if due is None:
                if (
                    job.schedule.kind == "at"
                    and job.schedule.at_ms
                    and job.schedule.at_ms <= cutoff
                    and job.state.last_run_at_ms is None
                ):
                    overdue.append(job)
                continue
            if due <= cutoff:
                overdue.append(job)
        return overdue

    async def run_job(
        self,
        job_id: str,
        force: bool = False,
        *,
        only_if_due: bool = False,
    ) -> bool:
        """Run a job now.

        ``force`` runs a disabled job. ``only_if_due`` is the heartbeat
        backstop: skip the job when the scheduler has already moved its
        deadline forward.
        """
        async with self._run_lock:
            if not self._file_lock.try_acquire():
                logger.debug("Cron: run_job skipped, another process holds the scheduler lock")
                return False
            try:
                store = self._load_store()
                for job in store.jobs:
                    if job.id != job_id:
                        continue
                    if not force and not job.enabled:
                        return False
                    due = job.state.next_run_at_ms
                    # A future deadline means the scheduler already advanced it.
                    # ``None`` still runs: one-shots that were missed never got one.
                    if only_if_due and due is not None and due > _now_ms():
                        return False
                    self._executing = True
                    try:
                        await self._execute_job(job)
                        self._save_store()
                    finally:
                        self._executing = False
                    self._arm_timer()
                    return True
                return False
            finally:
                self._file_lock.release()

    def get_job(self, job_id: str) -> CronJob | None:
        """Get a job by ID."""
        store = self._load_store()
        return next((j for j in store.jobs if j.id == job_id), None)

    def status(self) -> dict:
        """Get service status."""
        store = self._load_store()
        return {
            "enabled": self._running,
            "jobs": len(store.jobs),
            "next_wake_at_ms": self._get_next_wake_ms(),
        }
