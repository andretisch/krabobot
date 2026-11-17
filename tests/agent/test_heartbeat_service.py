import asyncio
import time
from pathlib import Path

import pytest

from krabobot.heartbeat.service import (
    heartbeat_check_interval,
    scheduler_tick_interval,
    unfinished_heartbeat_tasks,
)
from krabobot.providers.base import LLMProvider, LLMResponse
from krabobot.scheduler.service import TaskScheduler


class DummyProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__()

    async def chat(self, *args, **kwargs) -> LLMResponse:
        return LLMResponse(content="", tool_calls=[])

    def get_default_model(self) -> str:
        return "test-model"


class FakeCron:
    def __init__(self) -> None:
        self.calls = 0

    async def bootstrap(self) -> None:
        pass

    async def run_due_jobs(self) -> int:
        self.calls += 1
        return 0


def test_scheduler_intervals() -> None:
    assert scheduler_tick_interval(60) == 60
    assert scheduler_tick_interval(0) == 60
    assert heartbeat_check_interval(1800) == 1800
    assert heartbeat_check_interval(0) == 30 * 60


def test_template_and_russian_sections() -> None:
    template = Path("krabobot/templates/HEARTBEAT.md").read_text(encoding="utf-8")
    assert unfinished_heartbeat_tasks(template) == []

    content = (
        "## Активные задачи\n"
        "Проверить бэкапы\n"
        "- [x] уже сделано\n"
        "\n"
        "## Выполнено\n"
        "Старый отчёт\n"
        "- [ ] это уже перенесено\n"
    )
    assert unfinished_heartbeat_tasks(content) == ["Проверить бэкапы"]


@pytest.mark.asyncio
async def test_start_is_idempotent(tmp_path) -> None:
    cron = FakeCron()
    scheduler = TaskScheduler(
        cron,  # type: ignore[arg-type]
        workspace=tmp_path,
        provider=DummyProvider(),
        model="test",
        tick_interval_s=9999,
        enabled=True,
    )
    await scheduler.start()
    first_task = scheduler._task
    await scheduler.start()
    assert scheduler._task is first_task
    scheduler.stop()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_trigger_now_executes_open_tasks(tmp_path) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] do thing", encoding="utf-8")
    called_with: list[str] = []

    async def _on_execute(tasks: str) -> str:
        called_with.append(tasks)
        return "done"

    scheduler = TaskScheduler(
        FakeCron(),  # type: ignore[arg-type]
        workspace=tmp_path,
        provider=DummyProvider(),
        model="test",
        on_heartbeat_execute=_on_execute,
    )
    result = await scheduler.trigger_now()
    assert result == "done"
    assert called_with == ["do thing"]


@pytest.mark.asyncio
async def test_trigger_now_skips_completed_section(tmp_path) -> None:
    (tmp_path / "HEARTBEAT.md").write_text(
        "## Completed\n- [ ] already done\n",
        encoding="utf-8",
    )
    async def _never(tasks: str) -> str:
        return tasks

    scheduler = TaskScheduler(
        FakeCron(),  # type: ignore[arg-type]
        workspace=tmp_path,
        provider=DummyProvider(),
        model="test",
        on_heartbeat_execute=_never,
    )
    assert await scheduler.trigger_now() is None


@pytest.mark.asyncio
async def test_tick_notifies_when_evaluator_says_yes(tmp_path, monkeypatch) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] check deployments", encoding="utf-8")
    executed: list[str] = []
    notified: list[str] = []

    async def _on_execute(tasks: str) -> str:
        executed.append(tasks)
        return "deployment failed on staging"

    async def _on_notify(response: str) -> None:
        notified.append(response)

    scheduler = TaskScheduler(
        FakeCron(),  # type: ignore[arg-type]
        workspace=tmp_path,
        provider=DummyProvider(),
        model="test",
        on_heartbeat_execute=_on_execute,
        on_heartbeat_notify=_on_notify,
    )

    async def _eval_notify(*_a, **_k):
        return True

    monkeypatch.setattr("krabobot.utils.evaluator.evaluate_response", _eval_notify)
    await scheduler._tick()
    assert executed == ["check deployments"]
    assert notified == ["deployment failed on staging"]


@pytest.mark.asyncio
async def test_tick_suppresses_when_evaluator_says_no(tmp_path, monkeypatch) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] check status", encoding="utf-8")
    executed: list[str] = []
    notified: list[str] = []

    async def _on_execute(tasks: str) -> str:
        executed.append(tasks)
        return "everything is fine"

    async def _on_notify(response: str) -> None:
        notified.append(response)

    scheduler = TaskScheduler(
        FakeCron(),  # type: ignore[arg-type]
        workspace=tmp_path,
        provider=DummyProvider(),
        model="test",
        on_heartbeat_execute=_on_execute,
        on_heartbeat_notify=_on_notify,
    )

    monkeypatch.setattr(
        "krabobot.utils.evaluator.evaluate_response",
        lambda *_a, **_k: asyncio.sleep(0, result=False),  # type: ignore[misc]
    )
    await scheduler._tick()
    assert executed == ["check status"]
    assert notified == []


@pytest.mark.asyncio
async def test_tick_runs_unchecked_tasks_from_file(tmp_path, monkeypatch) -> None:
    (tmp_path / "HEARTBEAT.md").write_text(
        "## Active Tasks\n- [ ] write report\n",
        encoding="utf-8",
    )
    executed: list[str] = []

    async def _on_execute(tasks: str) -> str:
        executed.append(tasks)
        return "ok"

    scheduler = TaskScheduler(
        FakeCron(),  # type: ignore[arg-type]
        workspace=tmp_path,
        provider=DummyProvider(),
        model="test",
        on_heartbeat_execute=_on_execute,
    )
    monkeypatch.setattr(
        "krabobot.utils.evaluator.evaluate_response",
        lambda *_a, **_k: asyncio.sleep(0, result=False),  # type: ignore[misc]
    )
    await scheduler._tick()
    assert executed == ["write report"]


@pytest.mark.asyncio
async def test_tick_runs_due_cron_jobs(tmp_path) -> None:
    class CountingCron(FakeCron):
        async def run_due_jobs(self) -> int:
            self.calls += 1
            return 2

    cron = CountingCron()
    scheduler = TaskScheduler(
        cron,  # type: ignore[arg-type]
        workspace=tmp_path,
        provider=DummyProvider(),
        model="test",
        heartbeat_interval_s=42,
    )
    assert scheduler.heartbeat_interval_s == 42
    await scheduler._tick()
    assert cron.calls == 1


@pytest.mark.asyncio
async def test_heartbeat_file_not_checked_every_cron_tick(tmp_path, monkeypatch) -> None:
    checks = 0
    cron = FakeCron()

    async def _fake_heartbeat(self) -> None:
        nonlocal checks
        checks += 1

    monkeypatch.setattr(TaskScheduler, "_run_heartbeat_builtin", _fake_heartbeat)

    scheduler = TaskScheduler(
        cron,  # type: ignore[arg-type]
        workspace=tmp_path,
        provider=DummyProvider(),
        model="test",
        heartbeat_interval_s=30,
    )
    t = 1_000.0
    monkeypatch.setattr(time, "monotonic", lambda: t)

    await scheduler._tick()
    assert checks == 1

    t += 1
    await scheduler._tick()
    assert checks == 1

    t += 30
    await scheduler._tick()
    assert checks == 2
