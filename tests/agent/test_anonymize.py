"""Tests for turn-local PII anonymization."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from krabobot.agent.anonymize import (
    TokenMap,
    anonymize_messages,
    decode_messages,
    decode_tool_arguments,
    load_default_rules,
)
from krabobot.agent.loop import AgentLoop
from krabobot.bus.events import InboundMessage
from krabobot.bus.queue import MessageBus
from krabobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest


def test_rules_load_and_include_fio_org() -> None:
    rules = load_default_rules()
    cats = {r.category for r in rules}
    assert "FIO" in cats
    assert "ORG" in cats
    assert "PHONE" in cats
    assert "EMAIL" in cats
    assert "KPP" not in cats  # disabled in JSON


def test_encode_decode_roundtrip_fio_org_phone_email() -> None:
    text = (
        "Клиент Иванов И.И. из ООО «Ромашка», "
        "тел. +7 999 123-45-67, mail user@example.com"
    )
    tm = TokenMap()
    encoded = tm.encode(text)
    assert "Иванов И.И." not in encoded
    assert "ООО «Ромашка»" not in encoded
    assert "+7 999 123-45-67" not in encoded
    assert "user@example.com" not in encoded
    assert "[FIO-" in encoded
    assert "[ORG-" in encoded
    assert "[PHONE-" in encoded
    assert "[EMAIL-" in encoded
    assert tm.decode(encoded) == text


def test_fio_two_and_three_part_with_surname_morphology() -> None:
    """Hybrid E: 2-part needs surname-like ending; 3-part needs ending + patronymic."""
    cases = [
        "Тишкин Андрей",
        "Тишкин Андрей Иванович",
        "Иванов И.И.",
        # Feminine nominative endings (−ова/−ина/−ская) + name / patronymic (−овна/−евна)
        "Тишкина Анна",
        "Иванова Мария",
        "Смирнова Елена",
        "Петровская Ольга",
        "Кузнецова Наталья",
        "Тишкина Анна Ивановна",
        "Иванова Мария Петровна",
        "Смирнова Елена Сергеевна",
    ]
    for text in cases:
        tm = TokenMap()
        encoded = tm.encode(text)
        assert text not in encoded, text
        assert "[FIO-" in encoded, text
        assert tm.decode(encoded) == text


def test_fio_does_not_mask_oblique_declensions() -> None:
    """Regex is nominative-ending based; declined endings (−у/−ой/−ым…) are out of scope."""
    for text in (
        "Тишкину Андрею",
        "Тишкиным Андреем",
        "Тишкиной Анне",
        "к Тишкиной Анне",
        "Андрея Тишкина",
    ):
        tm = TokenMap()
        encoded = tm.encode(text)
        assert encoded == text, text
        assert "[FIO-" not in encoded, text


def test_fio_masks_screenshot_user_test_string() -> None:
    text = "Тишкин Андрей - персональные данные спрятаны? Это тест"
    tm = TokenMap()
    encoded = tm.encode(text)
    assert "Тишкин Андрей" not in encoded
    assert "[FIO-" in encoded
    assert "персональные данные спрятаны?" in encoded
    assert tm.decode(encoded) == text


def test_fio_skips_capitalized_non_names() -> None:
    for text in ("Это Тест", "Новый Год", "Уважаемый Клиент", "Московский Университет"):
        tm = TokenMap()
        encoded = tm.encode(text)
        assert encoded == text
        assert "[FIO-" not in encoded


def test_same_value_reuses_token_within_turn() -> None:
    tm = TokenMap()
    a = tm.encode("Звоните +79991234567")
    b = tm.encode("Ещё раз +79991234567")
    tokens_a = [t for t in tm.token_to_original if t in a]
    tokens_b = [t for t in tm.token_to_original if t in b]
    assert tokens_a == tokens_b
    assert len(tm.token_to_original) == 1


def test_skip_already_tokenized_spans() -> None:
    tm = TokenMap()
    text = "Уже есть [ORG-00099] и ООО «Ромашка»"
    encoded = tm.encode(text)
    assert "[ORG-00099]" in encoded
    # Existing token must remain; new ORG gets a fresh id.
    assert "ООО «Ромашка»" not in encoded
    assert encoded.count("[ORG-") >= 2


def test_skip_linked_accounts_block() -> None:
    block = "[Linked Accounts]\nuser_id: u1\naccounts: telegram:1\n\n"
    body = "ООО «Ромашка» и Иванов И.И."
    tm = TokenMap()
    encoded = tm.encode(block + body)
    assert encoded.startswith("[Linked Accounts]\nuser_id: u1\naccounts: telegram:1\n\n")
    assert "ООО «Ромашка»" not in encoded
    assert "Иванов И.И." not in encoded
    assert tm.decode(encoded) == block + body


def test_overlap_prefers_non_overlapping_longer_first() -> None:
    # Phone compact and spaced forms should not double-replace.
    tm = TokenMap()
    encoded = tm.encode("Номер +7 999 123-45-67 конец")
    assert encoded.count("[PHONE-") == 1


def test_anonymize_messages_skips_system_and_handles_multimodal() -> None:
    tm = TokenMap()
    messages = [
        {"role": "system", "content": "Keep ООО «Система» as-is"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Пишите Иванов И.И."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
            ],
        },
    ]
    out = anonymize_messages(messages, tm)
    assert out[0]["content"] == "Keep ООО «Система» as-is"
    assert "Иванов И.И." not in out[1]["content"][0]["text"]
    assert out[1]["content"][1]["type"] == "image_url"
    restored = decode_messages(out, tm)
    assert restored[1]["content"][0]["text"] == "Пишите Иванов И.И."


def _make_provider() -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=512)
    return provider


@pytest.mark.asyncio
async def test_loop_anonymize_llm_sees_tokens_user_gets_decoded(tmp_path: Path) -> None:
    captured: list[list[dict]] = []

    async def chat_with_retry(*, messages, **kwargs):
        captured.append(messages)
        user = messages[-1]["content"]
        text = user if isinstance(user, str) else str(user)

        m = re.search(r"\[ORG-\d{5}\]", text)
        org_token = m.group(0) if m else None
        return LLMResponse(
            content=f"Принято: {org_token or 'ok'}",
            tool_calls=[],
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        )

    provider = _make_provider()
    provider.chat_with_retry = chat_with_retry
    provider.chat_stream_with_retry = AsyncMock()

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        anonymize=True,
    )

    out = await loop.process_direct(
        "Договор с ООО «Ромашка»",
        session_key="api:default",
        channel="api",
        chat_id="default",
        sender_id="tester",
    )

    assert captured, "LLM should have been called"
    llm_user = captured[0][-1]["content"]
    llm_text = llm_user if isinstance(llm_user, str) else str(llm_user)
    assert "ООО «Ромашка»" not in llm_text
    assert "[ORG-" in llm_text
    assert out is not None
    assert "ООО «Ромашка»" in out.content
    assert "[ORG-" not in out.content

    runtime = await loop._runtime_for_message(
        InboundMessage(channel="api", sender_id="tester", chat_id="default", content="")
    )
    session = runtime.sessions.get_or_create("api:default")
    user_msgs = [m for m in session.messages if m.get("role") == "user"]
    assert user_msgs
    assert "ООО «Ромашка»" in str(user_msgs[-1]["content"])


@pytest.mark.asyncio
async def test_loop_anonymize_disables_streaming(tmp_path: Path) -> None:
    stream_called = {"n": 0}

    async def chat_with_retry(**kwargs):
        return LLMResponse(content="ok", tool_calls=[], usage={})

    async def chat_stream_with_retry(**kwargs):
        stream_called["n"] += 1
        return LLMResponse(content="streamed", tool_calls=[], usage={})

    provider = _make_provider()
    provider.chat_with_retry = chat_with_retry
    provider.chat_stream_with_retry = chat_stream_with_retry

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        anonymize=True,
    )

    async def on_stream(delta: str) -> None:
        pass

    final, _, _ = await loop._run_agent_loop(
        loop._default_runtime,
        [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
        on_stream=on_stream,
    )
    assert final == "ok"
    assert stream_called["n"] == 0


def test_decode_tool_arguments_restores_nested_strings() -> None:
    tm = TokenMap()
    encoded = tm.encode("ООО «Ромашка»")
    args = {"path": f"/data/{encoded}.txt", "meta": {"note": encoded}, "tags": [encoded]}
    decoded = decode_tool_arguments(args, tm)
    assert decoded["path"] == "/data/ООО «Ромашка».txt"
    assert decoded["meta"]["note"] == "ООО «Ромашка»"
    assert decoded["tags"] == ["ООО «Ромашка»"]


@pytest.mark.asyncio
async def test_provider_wrapper_tool_roundtrip_same_token_map() -> None:
    """AnonymizingProvider shares TokenMap across chat calls in a runner turn."""
    from krabobot.agent.runner import AgentRunner, AgentRunSpec
    from krabobot.providers.anonymizing import AnonymizingProvider

    executed_args: list[dict] = []
    captured: list[list[dict]] = []
    call_count = {"n": 0}

    async def chat_with_retry(*, messages, **kwargs):
        captured.append(messages)
        call_count["n"] += 1
        if call_count["n"] == 1:
            user = messages[-1]["content"]
            text = user if isinstance(user, str) else str(user)
            m = re.search(r"\[ORG-\d{5}\]", text)
            org_token = m.group(0) if m else "[ORG-00001]"
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="read_file",
                        arguments={"path": f"/tmp/{org_token}.txt"},
                    )
                ],
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            )
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        tool_text = str(tool_msgs[-1]["content"]) if tool_msgs else ""
        m = re.search(r"\[PHONE-\d{5}\]", tool_text)
        phone_token = m.group(0) if m else "missing"
        return LLMResponse(
            content=f"Номер: {phone_token}",
            tool_calls=[],
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        )

    inner = _make_provider()
    inner.chat_with_retry = chat_with_retry
    provider = AnonymizingProvider(inner)

    tools = MagicMock()
    tools.get_definitions.return_value = []

    async def execute(name, params):
        executed_args.append(params)
        return "Контакт: +79991234567"

    tools.execute = execute

    runner = AgentRunner(provider)
    result = await runner.run(
        AgentRunSpec(
            initial_messages=[
                {"role": "system", "content": "s"},
                {"role": "user", "content": "Прочитай ООО «Ромашка»"},
            ],
            tools=tools,
            model="test-model",
            max_iterations=3,
        )
    )

    assert executed_args, "tool should have run"
    assert "ООО «Ромашка»" in executed_args[0]["path"]
    assert "[ORG-" not in executed_args[0]["path"]

    assert len(captured) == 2
    assert "ООО «Ромашка»" not in str(captured[0][-1]["content"])
    assert "+79991234567" not in str(captured[1])
    assert "[PHONE-" in str(next(m for m in captured[1] if m.get("role") == "tool")["content"])

    assert result.final_content is not None
    assert "+79991234567" in result.final_content
    assert "[PHONE-" not in result.final_content

    tool_saved = next(m for m in result.messages if m.get("role") == "tool")
    assert "+79991234567" in str(tool_saved["content"])


@pytest.mark.asyncio
async def test_consolidator_anonymizes_via_provider(tmp_path: Path) -> None:
    from krabobot.agent.memory import MemoryStore
    from krabobot.providers.anonymizing import AnonymizingProvider

    captured: list[list[dict]] = []

    async def chat_with_retry(*, messages, **kwargs):
        captured.append(messages)
        blob = str(messages)
        m = re.search(r"\[ORG-\d{5}\]", blob)
        org_token = m.group(0) if m else "[ORG-00001]"
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="m1",
                    name="save_memory",
                    arguments={
                        "history_entry": f"met {org_token}",
                        "memory_update": f"Client: {org_token}",
                    },
                )
            ],
        )

    inner = _make_provider()
    inner.chat_with_retry = chat_with_retry
    provider = AnonymizingProvider(inner)

    store = MemoryStore(tmp_path)
    ok = await store.consolidate(
        [{"role": "user", "content": "Договор с ООО «Ромашка»"}],
        provider,
        "test-model",
    )
    assert ok
    assert captured
    assert "ООО «Ромашка»" not in str(captured[0])
    assert "[ORG-" in str(captured[0])
    # Decoded tool args → MEMORY.md stores original
    assert "ООО «Ромашка»" in store.read_long_term()
    assert "[ORG-" not in store.read_long_term()


@pytest.mark.asyncio
async def test_anonymize_flag_false_skips_wrapper(tmp_path: Path) -> None:
    captured: list[list[dict]] = []

    async def chat_with_retry(*, messages, **kwargs):
        captured.append(messages)
        return LLMResponse(content="ok", tool_calls=[], usage={})

    provider = _make_provider()
    provider.chat_with_retry = chat_with_retry
    provider.chat_stream_with_retry = AsyncMock()

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        anonymize=False,
    )
    from krabobot.providers.anonymizing import AnonymizingProvider

    assert not isinstance(loop.provider, AnonymizingProvider)

    await loop.process_direct(
        "Договор с ООО «Ромашка»",
        session_key="api:default",
        channel="api",
        chat_id="default",
        sender_id="tester",
    )
    assert captured
    blob = str(captured[0])
    assert "ООО «Ромашка»" in blob
    assert "[ORG-" not in blob


@pytest.mark.asyncio
async def test_loop_tool_roundtrip_same_token_map(tmp_path: Path) -> None:
    """Tool results are masked toward LLM; decoded messages restore originals."""
    captured: list[list[dict]] = []
    executed_args: list[dict] = []
    call_count = {"n": 0}

    async def chat_with_retry(*, messages, **kwargs):
        captured.append(messages)
        call_count["n"] += 1
        if call_count["n"] == 1:
            user = messages[-1]["content"]
            text = user if isinstance(user, str) else str(user)

            m = re.search(r"\[ORG-\d{5}\]", text)
            org_token = m.group(0) if m else "[ORG-00001]"
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="c1",
                        name="read_file",
                        arguments={"path": f"docs/{org_token}.md"},
                    )
                ],
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            )
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        tool_text = str(tool_msgs[-1]["content"]) if tool_msgs else ""

        m = re.search(r"\[EMAIL-\d{5}\]", tool_text)
        email_token = m.group(0) if m else "ok"
        return LLMResponse(
            content=f"Нашёл: {email_token}",
            tool_calls=[],
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        )

    provider = _make_provider()
    provider.chat_with_retry = chat_with_retry
    provider.chat_stream_with_retry = AsyncMock()

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        anonymize=True,
    )

    mock_tools = MagicMock()
    mock_tools.get_definitions.return_value = []

    async def execute(name, params):
        executed_args.append(params)
        return "Пишите на secret@example.com"

    mock_tools.execute = execute
    runtime = loop._default_runtime
    runtime.tools = mock_tools

    final, _, saved = await loop._run_agent_loop(
        runtime,
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "Договор с ООО «Ромашка»"},
        ],
    )

    assert len(captured) == 2
    assert "ООО «Ромашка»" not in str(captured[0][-1]["content"])
    assert "secret@example.com" not in str(captured[1])
    assert "[EMAIL-" in str(next(m for m in captured[1] if m.get("role") == "tool")["content"])

    assert executed_args
    assert "ООО «Ромашка»" in executed_args[0]["path"]

    assert final is not None
    assert "secret@example.com" in final
    assert "[EMAIL-" not in final

    tool_saved = next(m for m in saved if m.get("role") == "tool")
    assert "secret@example.com" in str(tool_saved["content"])
