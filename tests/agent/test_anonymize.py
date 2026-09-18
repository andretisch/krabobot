"""Tests for turn-local PII anonymization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from krabobot.agent.anonymize import (
    TokenMap,
    anonymize_messages,
    decode_messages,
    load_default_rules,
)
from krabobot.agent.loop import AgentLoop
from krabobot.bus.events import InboundMessage
from krabobot.bus.queue import MessageBus
from krabobot.providers.base import GenerationSettings, LLMResponse


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
        import re

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
