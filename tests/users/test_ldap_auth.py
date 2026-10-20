"""LDAP filter helpers and identity store. No directory network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from krabobot.users.ldap_auth import (
    build_user_filter,
    escape_ldap_filter,
    normalize_ldap_username,
    pick_display_name,
)
from krabobot.users.resolver import UserResolver


def test_filter_replaces_username_placeholder_and_escapes() -> None:
    assert escape_ldap_filter("a*b") == r"a\2ab"
    built = build_user_filter("(sAMAccountName=%{username})", "sAMAccountName", "a*b")
    assert built == r"(sAMAccountName=a\2ab)"


def test_filter_ands_username_attribute_when_placeholder_missing() -> None:
    built = build_user_filter(
        "(memberOf=cn=app,dc=example,dc=com)",
        "sAMAccountName",
        "alice",
    )
    assert built == "(&(sAMAccountName=alice)(memberOf=cn=app,dc=example,dc=com))"


def test_normalize_domain_and_upn() -> None:
    assert normalize_ldap_username(r"CORP\JSmith") == "jsmith"
    assert normalize_ldap_username("JSmith@corp.example") == "jsmith"
    assert normalize_ldap_username("bad*name") == ""


def test_display_name_falls_back_to_login() -> None:
    assert pick_display_name("", "  ", fallback="jsmith") == "jsmith"
    assert pick_display_name("Иванов Иван", fallback="jsmith") == "Иванов Иван"


@pytest.mark.asyncio
async def test_ldap_link_does_not_touch_channel_accounts(tmp_path: Path) -> None:
    resolver = UserResolver(tmp_path)
    owner = await resolver.create_user(display_name="Owner", channel="telegram", sender_id="42")
    guest = await resolver.create_user(display_name="Guest", channel="vk", sender_id="7")
    await resolver.link_ldap(guest, r"CORP\Alice")

    assert await resolver.lookup("telegram", "42") == owner
    assert await resolver.lookup("vk", "7") == guest
    assert await resolver.lookup_ldap("alice") == guest
    assert await resolver.lookup_ldap("ALICE") == guest
    accounts = await resolver.accounts_for_user(guest)
    assert accounts == ["vk:7"]
    raw = json.loads((tmp_path / "user_links.json").read_text(encoding="utf-8"))
    assert "ldap:alice" not in raw["accounts"]
    assert raw["ldap_identities"]["alice"] == guest

    with pytest.raises(ValueError, match="ldap_identity_taken"):
        await resolver.link_ldap(owner, "alice")


@pytest.mark.asyncio
async def test_resolve_ldap_autocreate_skips_owner_and_failed_path(tmp_path: Path) -> None:
    resolver = UserResolver(tmp_path)
    missing, created = await resolver.resolve_ldap_identity(
        "nobody",
        display_name="Nobody",
        auto_create=False,
    )
    assert missing is None and created is False
    assert await resolver.list_users() == []

    user_id, created = await resolver.resolve_ldap_identity(
        "bob",
        display_name="Bob Directory",
        auto_create=True,
    )
    assert created is True
    assert user_id
    again, created_again = await resolver.resolve_ldap_identity(
        "BOB",
        display_name="Other",
        auto_create=True,
    )
    assert again == user_id and created_again is False
    users = await resolver.list_users()
    assert len(users) == 1
    assert users[0]["display_name"] == "Bob Directory"
    assert users[0]["is_owner"] is False
    assert users[0]["ldap_usernames"] == ["bob"]
    assert await resolver.get_owner_user_id() in (None, "")
