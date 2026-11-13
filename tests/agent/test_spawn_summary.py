"""Spawn / subagent parent-summary contract."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from krabobot.agent.runner import TOOL_VISION_BRIDGE_PREFIX
from krabobot.agent.subagent import SubagentManager
from krabobot.agent.tools.spawn import SpawnTool
from krabobot.bus.queue import MessageBus


def test_spawn_tool_description_guides_when_to_spawn():
    tool = SpawnTool(manager=MagicMock())
    text = tool.description.lower()
    assert "use spawn when" in text
    assert "do not spawn" in text
    assert "compact structured summary" in text
    assert "not the full subagent transcript" in text
    task_desc = tool.parameters["properties"]["task"]["description"].lower()
    assert "no parent chat history" in task_desc


def test_subagent_prompt_requires_structured_summary(tmp_path):
    mgr = SubagentManager(
        provider=MagicMock(get_default_model=MagicMock(return_value="m")),
        workspace=tmp_path,
        bus=MessageBus(),
    )
    prompt = mgr._build_subagent_prompt(tmp_path)
    assert "Цель / Goal:" in prompt
    assert "Находки / Findings:" in prompt
    assert "Пути / Paths:" in prompt
    assert "Открытые вопросы / Open questions:" in prompt
    assert "ONLY thing the main agent sees" in prompt
    assert "no parent chat history" in prompt.lower()


def test_prepare_parent_summary_strips_vision_bridge_and_truncates():
    long_tail = "Z" * 5000
    raw = "\n".join(
        [
            "Цель / Goal: dig",
            f"{TOOL_VISION_BRIDGE_PREFIX} secret pixels",
            'data:image/png;base64,AAAA',
            "Находки / Findings: ok",
            long_tail,
        ]
    )
    out = SubagentManager._prepare_parent_summary(raw, max_chars=800)
    assert TOOL_VISION_BRIDGE_PREFIX not in out
    assert "data:image" not in out
    assert "[image omitted" in out
    assert "Цель / Goal: dig" in out
    assert "chars truncated" in out
    assert len(out) <= 800 + 80  # truncation marker overhead


@pytest.mark.asyncio
async def test_spawn_ack_mentions_compact_summary(tmp_path):
    mgr = SubagentManager(
        provider=MagicMock(get_default_model=MagicMock(return_value="m")),
        workspace=tmp_path,
        bus=MessageBus(),
    )
    mgr._run_subagent = AsyncMock()  # type: ignore[method-assign]
    ack = await mgr.spawn(task="research X", label="research")
    assert "compact summary" in ack
    assert "not the full subagent transcript" in ack
    assert "id:" in ack


@pytest.mark.asyncio
async def test_announce_embeds_prepared_summary_only(tmp_path):
    bus = MessageBus()
    mgr = SubagentManager(
        provider=MagicMock(get_default_model=MagicMock(return_value="m")),
        workspace=tmp_path,
        bus=bus,
    )
    huge = "FINDING " + ("x" * 9000)
    await mgr._announce_result(
        "abc",
        "label",
        "do research",
        f"{TOOL_VISION_BRIDGE_PREFIX} leak\n{huge}",
        {"channel": "cli", "chat_id": "direct"},
        "ok",
    )
    msg = await bus.consume_inbound()
    assert msg.channel == "system"
    assert msg.sender_id == "subagent"
    assert "Summary (compact" in msg.content
    assert TOOL_VISION_BRIDGE_PREFIX not in msg.content
    assert "chars truncated" in msg.content
    assert "do not expect tool traces" in msg.content


@pytest.mark.asyncio
async def test_run_subagent_passes_final_content_through_prepare(tmp_path):
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "m"
    mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=bus)
    published: list = []

    async def capture(msg):
        published.append(msg)

    mgr._publish_wakeup = capture  # type: ignore[method-assign]
    mgr.runner.run = AsyncMock(
        return_value=SimpleNamespace(
            stop_reason="completed",
            final_content=f"{TOOL_VISION_BRIDGE_PREFIX} no\nЦель / Goal: done\nНаходки / Findings: 1",
            error=None,
            tool_events=[],
        )
    )

    await mgr._run_subagent(
        "sub-1", "task brief", "lbl", {"channel": "test", "chat_id": "c1"}
    )

    assert len(published) == 1
    body = published[0].content
    assert TOOL_VISION_BRIDGE_PREFIX not in body
    assert "Цель / Goal: done" in body
    assert "Summary (compact" in body
