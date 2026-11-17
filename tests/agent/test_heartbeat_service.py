import asyncio
import time
from pathlib import Path

import pytest

from krabobot.heartbeat.service import (
    HeartbeatService,
    heartbeat_check_interval,
    scheduler_tick_interval,
    unfinished_heartbeat_tasks,
)
from krabobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class DummyProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]):
        super().__init__()
        self._responses = list(responses)
        self.calls = 0

    async def chat(self, *args, **kwargs) -> LLMResponse:
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(content="", tool_calls=[])

    def get_default_model(self) -> str:
        return "test-model"


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
    provider = DummyProvider([])

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
        tick_interval_s=9999,
        interval_s=1800,
        enabled=True,
    )

    await service.start()
    first_task = service._task
    await service.start()

    assert service._task is first_task

    service.stop()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_decide_returns_skip_when_no_tool_call(tmp_path) -> None:
    provider = DummyProvider([LLMResponse(content="no tool call", tool_calls=[])])
    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
    )

    action, tasks = await service._decide("heartbeat content")
    assert action == "skip"
    assert tasks == ""


@pytest.mark.asyncio
async def test_trigger_now_executes_when_decision_is_run(tmp_path) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] do thing", encoding="utf-8")

    provider = DummyProvider([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="hb_1",
                    name="heartbeat",
                    arguments={"action": "run", "tasks": "check open tasks"},
                )
            ],
        )
    ])

    called_with: list[str] = []

    async def _on_execute(tasks: str) -> str:
        called_with.append(tasks)
        return "done"

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
    )

    result = await service.trigger_now()
    assert result == "done"
    assert called_with == ["do thing"]


@pytest.mark.asyncio
async def test_trigger_now_returns_none_when_decision_is_skip(tmp_path) -> None:
    (tmp_path / "HEARTBEAT.md").write_text(
        "## Completed\n- [ ] already done\n- finished item\n",
        encoding="utf-8",
    )

    provider = DummyProvider([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="hb_1",
                    name="heartbeat",
                    arguments={"action": "skip"},
                )
            ],
        )
    ])

    async def _on_execute(tasks: str) -> str:
        return tasks

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
    )

    assert await service.trigger_now() is None


@pytest.mark.asyncio
async def test_tick_notifies_when_evaluator_says_yes(tmp_path, monkeypatch) -> None:
    """Phase 1 run -> Phase 2 execute -> Phase 3 evaluate=notify -> on_notify called."""
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] check deployments", encoding="utf-8")

    provider = DummyProvider([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="hb_1",
                    name="heartbeat",
                    arguments={"action": "run", "tasks": "check deployments"},
                )
            ],
        ),
    ])

    executed: list[str] = []
    notified: list[str] = []

    async def _on_execute(tasks: str) -> str:
        executed.append(tasks)
        return "deployment failed on staging"

    async def _on_notify(response: str) -> None:
        notified.append(response)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
        on_notify=_on_notify,
    )

    async def _eval_notify(*a, **kw):
        return True

    monkeypatch.setattr("krabobot.utils.evaluator.evaluate_response", _eval_notify)

    await service._tick()
    assert executed == ["check deployments"]
    assert notified == ["deployment failed on staging"]


@pytest.mark.asyncio
async def test_tick_suppresses_when_evaluator_says_no(tmp_path, monkeypatch) -> None:
    """Phase 1 run -> Phase 2 execute -> Phase 3 evaluate=silent -> on_notify NOT called."""
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] check status", encoding="utf-8")

    provider = DummyProvider([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="hb_1",
                    name="heartbeat",
                    arguments={"action": "run", "tasks": "check status"},
                )
            ],
        ),
    ])

    executed: list[str] = []
    notified: list[str] = []

    async def _on_execute(tasks: str) -> str:
        executed.append(tasks)
        return "everything is fine, no issues"

    async def _on_notify(response: str) -> None:
        notified.append(response)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
        on_notify=_on_notify,
    )

    async def _eval_silent(*a, **kw):
        return False

    monkeypatch.setattr("krabobot.utils.evaluator.evaluate_response", _eval_silent)

    await service._tick()
    assert executed == ["check status"]
    assert notified == []


@pytest.mark.asyncio
async def test_decide_retries_transient_error_then_succeeds(tmp_path, monkeypatch) -> None:
    provider = DummyProvider([
        LLMResponse(content="429 rate limit", finish_reason="error"),
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="hb_1",
                    name="heartbeat",
                    arguments={"action": "run", "tasks": "check open tasks"},
                )
            ],
        ),
    ])

    delays: list[int] = []

    async def _fake_sleep(delay: int) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
    )

    action, tasks = await service._decide("heartbeat content")

    assert action == "run"
    assert tasks == "check open tasks"
    assert provider.calls == 2
    assert delays == [1]


@pytest.mark.asyncio
async def test_decide_prompt_includes_current_time(tmp_path) -> None:
    """Phase 1 user prompt must contain current time so the LLM can judge task urgency."""

    captured_messages: list[dict] = []

    class CapturingProvider(LLMProvider):
        async def chat(self, *, messages=None, **kwargs) -> LLMResponse:
            if messages:
                captured_messages.extend(messages)
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="hb_1", name="heartbeat",
                        arguments={"action": "skip"},
                    )
                ],
            )

        def get_default_model(self) -> str:
            return "test-model"

    service = HeartbeatService(
        workspace=tmp_path,
        provider=CapturingProvider(),
        model="test-model",
    )

    await service._decide("- [ ] check servers at 10:00 UTC")

    user_msg = captured_messages[1]
    assert user_msg["role"] == "user"
    assert "Current Time:" in user_msg["content"]
    assert "unfinished" in user_msg["content"].lower()


@pytest.mark.asyncio
async def test_tick_runs_unchecked_tasks_when_model_would_skip(tmp_path, monkeypatch) -> None:
    """Open checklist items run even if the phase-1 model is never consulted."""
    (tmp_path / "HEARTBEAT.md").write_text(
        "## Active Tasks\n- [ ] write report\n- [x] already sent\n\n"
        "## Completed\n- [ ] moved here\n",
        encoding="utf-8",
    )
    provider = DummyProvider([])
    executed: list[str] = []
    notified: list[str] = []

    async def _on_execute(tasks: str) -> str:
        executed.append(tasks)
        return "sent the report"

    async def _on_notify(response: str) -> None:
        notified.append(response)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="test-model",
        on_execute=_on_execute,
        on_notify=_on_notify,
    )

    async def _eval_notify(*_a, **_k):
        return True

    monkeypatch.setattr("krabobot.utils.evaluator.evaluate_response", _eval_notify)

    await service._tick()

    assert provider.calls == 0
    assert executed == ["write report"]
    assert notified == ["sent the report"]


@pytest.mark.asyncio
async def test_tick_runs_due_cron_jobs(tmp_path) -> None:
    class _Cron:
        def __init__(self) -> None:
            self.calls = 0

        async def run_due_jobs(self) -> int:
            self.calls += 1
            return 2

    cron = _Cron()
    provider = DummyProvider([])
    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="test-model",
        tick_interval_s=60,
        interval_s=42,
        cron_service=cron,
    )
    assert service.heartbeat_interval_s == 42
    await service._tick()
    assert cron.calls == 1


@pytest.mark.asyncio
async def test_heartbeat_file_not_checked_every_cron_tick(tmp_path, monkeypatch) -> None:
  checks = 0

  class _Cron:
      async def run_due_jobs(self) -> int:
          return 0

  async def _fake_heartbeat(self) -> None:
      nonlocal checks
      checks += 1

  monkeypatch.setattr(HeartbeatService, "_run_heartbeat_file_task", _fake_heartbeat)

  service = HeartbeatService(
      workspace=tmp_path,
      provider=DummyProvider([]),
      model="test-model",
      tick_interval_s=1,
      interval_s=30,
      cron_service=_Cron(),
  )
  t = 1_000.0
  monkeypatch.setattr(time, "monotonic", lambda: t)

  await service._tick()
  assert checks == 1

  t += 1
  await service._tick()
  assert checks == 1

  t += 30
  await service._tick()
  assert checks == 2

