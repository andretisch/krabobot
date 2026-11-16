import asyncio
import contextlib
import json

import pytest

from krabobot.cron.service import CronService, _resume_next_run
from krabobot.cron.types import CronSchedule


def test_add_job_rejects_unknown_timezone(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")

    with pytest.raises(ValueError, match="unknown timezone 'America/Vancovuer'"):
        service.add_job(
            name="tz typo",
            schedule=CronSchedule(kind="cron", expr="0 9 * * *", tz="America/Vancovuer"),
            message="hello",
        )

    assert service.list_jobs(include_disabled=True) == []


def test_add_job_accepts_valid_timezone(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")

    job = service.add_job(
        name="tz ok",
        schedule=CronSchedule(kind="cron", expr="0 9 * * *", tz="America/Vancouver"),
        message="hello",
    )

    assert job.schedule.tz == "America/Vancouver"
    assert job.state.next_run_at_ms is not None


def test_add_job_persists_from_sender_id(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"
    service = CronService(store_path)
    job = service.add_job(
        name="vk remind",
        schedule=CronSchedule(kind="cron", expr="0 15 * * *"),
        message="ping",
        channel="vk",
        to="12345",
        from_sender_id="98765",
    )
    assert job.payload.from_sender_id == "98765"

    raw = json.loads(store_path.read_text())
    assert raw["jobs"][0]["payload"]["fromSenderId"] == "98765"

    fresh = CronService(store_path)
    reloaded = fresh.get_job(job.id)
    assert reloaded is not None
    assert reloaded.payload.from_sender_id == "98765"


@pytest.mark.asyncio
async def test_execute_job_records_run_history(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"
    service = CronService(store_path, on_job=lambda _: asyncio.sleep(0))
    job = service.add_job(
        name="hist",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="hello",
    )
    await service.run_job(job.id)

    loaded = service.get_job(job.id)
    assert loaded is not None
    assert len(loaded.state.run_history) == 1
    rec = loaded.state.run_history[0]
    assert rec.status == "ok"
    assert rec.duration_ms >= 0
    assert rec.error is None


@pytest.mark.asyncio
async def test_run_history_records_errors(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"

    async def fail(_):
        raise RuntimeError("boom")

    service = CronService(store_path, on_job=fail)
    job = service.add_job(
        name="fail",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="hello",
    )
    await service.run_job(job.id)

    loaded = service.get_job(job.id)
    assert len(loaded.state.run_history) == 1
    assert loaded.state.run_history[0].status == "error"
    assert loaded.state.run_history[0].error == "boom"


@pytest.mark.asyncio
async def test_run_history_trimmed_to_max(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"
    service = CronService(store_path, on_job=lambda _: asyncio.sleep(0))
    job = service.add_job(
        name="trim",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="hello",
    )
    for _ in range(25):
        await service.run_job(job.id)

    loaded = service.get_job(job.id)
    assert len(loaded.state.run_history) == CronService._MAX_RUN_HISTORY


@pytest.mark.asyncio
async def test_run_history_persisted_to_disk(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"
    service = CronService(store_path, on_job=lambda _: asyncio.sleep(0))
    job = service.add_job(
        name="persist",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="hello",
    )
    await service.run_job(job.id)

    raw = json.loads(store_path.read_text())
    history = raw["jobs"][0]["state"]["runHistory"]
    assert len(history) == 1
    assert history[0]["status"] == "ok"
    assert "runAtMs" in history[0]
    assert "durationMs" in history[0]

    fresh = CronService(store_path)
    loaded = fresh.get_job(job.id)
    assert len(loaded.state.run_history) == 1
    assert loaded.state.run_history[0].status == "ok"


@pytest.mark.asyncio
async def test_running_service_honors_external_disable(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"
    called: list[str] = []

    async def on_job(job) -> None:
        called.append(job.id)

    service = CronService(store_path, on_job=on_job)
    job = service.add_job(
        name="external-disable",
        schedule=CronSchedule(kind="every", every_ms=200),
        message="hello",
    )
    await service.start()
    try:
        # Wait slightly to ensure file mtime is definitively different
        await asyncio.sleep(0.05)
        external = CronService(store_path)
        updated = external.enable_job(job.id, enabled=False)
        assert updated is not None
        assert updated.enabled is False

        await asyncio.sleep(0.35)
        assert called == []
    finally:
        task = service._timer_task
        service.stop()
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task


def test_resume_keeps_pending_every_and_overdue_deadline() -> None:
    every_ms = 86_400_000
    schedule = CronSchedule(kind="every", every_ms=every_ms)
    now = 1_700_000_000_000
    pending = now + every_ms - 3_600_000
    assert _resume_next_run(schedule, pending, now + 3_600_000) == pending

    overdue = now - 5_000
    assert _resume_next_run(schedule, overdue, now) == overdue


def test_resume_keeps_missed_one_shot_until_it_runs() -> None:
    at_ms = 1_700_000_000_000
    schedule = CronSchedule(kind="at", at_ms=at_ms)
    assert _resume_next_run(schedule, None, at_ms + 60_000, last_run_at_ms=None) == at_ms
    assert _resume_next_run(schedule, at_ms, at_ms + 60_000, last_run_at_ms=at_ms + 1) is None


@pytest.mark.asyncio
async def test_restart_does_not_postpone_every_job(tmp_path, monkeypatch) -> None:
    base = 1_700_000_000_000
    monkeypatch.setattr("krabobot.cron.service._now_ms", lambda: base)
    store = tmp_path / "cron" / "jobs.json"
    service = CronService(store)
    job = service.add_job(
        name="daily",
        schedule=CronSchedule(kind="every", every_ms=86_400_000),
        message="ping",
    )
    assert job.state.next_run_at_ms == base + 86_400_000

    monkeypatch.setattr("krabobot.cron.service._now_ms", lambda: base + 3_600_000)
    restarted = CronService(store)
    await restarted.start()
    task = restarted._timer_task
    try:
        loaded = restarted.get_job(job.id)
        assert loaded is not None
        assert loaded.state.next_run_at_ms == base + 86_400_000
    finally:
        restarted.stop()
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_overdue_one_shot_runs_after_restart(tmp_path, monkeypatch) -> None:
    base = 1_700_000_000_000
    monkeypatch.setattr("krabobot.cron.service._now_ms", lambda: base)
    store = tmp_path / "cron" / "jobs.json"
    called: list[str] = []

    async def on_job(job) -> None:
        called.append(job.id)

    service = CronService(store, on_job=on_job)
    job = service.add_job(
        name="once",
        schedule=CronSchedule(kind="at", at_ms=base + 1_000),
        message="ping",
    )
    monkeypatch.setattr("krabobot.cron.service._now_ms", lambda: base + 60_000)
    await service.start()
    task = service._timer_task
    try:
        for _ in range(50):
            if called:
                break
            await asyncio.sleep(0.02)
        assert called == [job.id]
        loaded = service.get_job(job.id)
        assert loaded is not None
        assert loaded.enabled is False
    finally:
        service.stop()
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_list_during_job_does_not_replay(tmp_path, monkeypatch) -> None:
    base = 1_700_000_000_000
    monkeypatch.setattr("krabobot.cron.service._now_ms", lambda: base)
    store = tmp_path / "cron" / "jobs.json"
    called: list[str] = []
    service = CronService(store)

    async def on_job(_job) -> None:
        service.list_jobs()
        called.append("ran")

    service.on_job = on_job
    job = service.add_job(
        name="replay",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="ping",
    )
    job.state.next_run_at_ms = base - 1_000
    service._save_store()

    await service.start()
    task = service._timer_task
    try:
        await asyncio.sleep(0.2)
        assert called == ["ran"]
    finally:
        service.stop()
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_add_during_job_does_not_cancel_execution(tmp_path, monkeypatch) -> None:
    base = 1_700_000_000_000
    monkeypatch.setattr("krabobot.cron.service._now_ms", lambda: base)
    store = tmp_path / "cron" / "jobs.json"
    service = CronService(store)
    finished = asyncio.Event()

    async def on_job(_job) -> None:
        service.add_job(
            name="later",
            schedule=CronSchedule(kind="every", every_ms=3_600_000),
            message="later",
        )
        await asyncio.sleep(0.05)
        finished.set()

    service.on_job = on_job
    job = service.add_job(
        name="due",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="ping",
    )
    job.state.next_run_at_ms = base - 1
    service._save_store()

    await service.start()
    task = service._timer_task
    try:
        await asyncio.wait_for(finished.wait(), timeout=2)
    finally:
        service.stop()
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task
