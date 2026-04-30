from pathlib import Path

import pytest

from krabobot.users import UserResolver


@pytest.mark.asyncio
async def test_resolve_or_create_is_stable(tmp_path: Path):
    resolver = UserResolver(tmp_path)
    first = await resolver.resolve_or_create("telegram", "42")
    second = await resolver.resolve_or_create("telegram", "42")
    assert first == second
    assert await resolver.lookup("telegram", "42") == first


@pytest.mark.asyncio
async def test_link_code_links_second_account(tmp_path: Path):
    resolver = UserResolver(tmp_path, code_ttl_seconds=600, code_attempt_limit=3)
    user = await resolver.resolve_or_create("telegram", "42")
    code = await resolver.create_link_code(user)

    result = await resolver.consume_link_code(code, "email", "alice@example.com")
    assert result.ok is True
    assert result.user_id == user
    assert await resolver.lookup("email", "alice@example.com") == user


@pytest.mark.asyncio
async def test_invalid_code_records_attempt(tmp_path: Path):
    resolver = UserResolver(tmp_path, code_ttl_seconds=600, code_attempt_limit=1)
    user = await resolver.resolve_or_create("telegram", "42")
    _ = await resolver.create_link_code(user)
    bad = await resolver.consume_link_code("INVALID", "telegram", "100")
    assert bad.ok is False


@pytest.mark.asyncio
async def test_accounts_for_user_returns_all_linked_accounts(tmp_path: Path):
    resolver = UserResolver(tmp_path, code_ttl_seconds=600, code_attempt_limit=3)
    user = await resolver.resolve_or_create("telegram", "42")
    await resolver.link_account(user, "vk", "123")
    await resolver.link_account(user, "email", "alice@example.com")

    accounts = await resolver.accounts_for_user(user)

    assert "telegram:42" in accounts
    assert "vk:123" in accounts
    assert "email:alice@example.com" in accounts


@pytest.mark.asyncio
async def test_tts_preference_roundtrip(tmp_path: Path):
    resolver = UserResolver(tmp_path)
    user = await resolver.resolve_or_create("telegram", "42")

    assert await resolver.get_tts_enabled(user, default=False) is False

    await resolver.set_tts_enabled(user, True)
    assert await resolver.get_tts_enabled(user, default=False) is True


@pytest.mark.asyncio
async def test_first_user_becomes_owner(tmp_path: Path):
    resolver = UserResolver(tmp_path)
    first = await resolver.resolve_or_create("telegram", "42")
    owner = await resolver.ensure_owner(first)

    assert owner == first
    assert await resolver.get_owner_user_id() == first
    assert await resolver.is_owner(first) is True


@pytest.mark.asyncio
async def test_owner_is_not_overwritten_by_next_user(tmp_path: Path):
    resolver = UserResolver(tmp_path)
    first = await resolver.resolve_or_create("telegram", "42")
    second = await resolver.resolve_or_create("telegram", "43")
    await resolver.ensure_owner(first)
    owner = await resolver.ensure_owner(second)

    assert owner == first
    assert await resolver.is_owner(second) is False
