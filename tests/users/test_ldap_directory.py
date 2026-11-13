"""Ldap3Directory open/bind sequencing. Uses fakes — no real LDAP network."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from krabobot.config.schema import LdapConfig
from krabobot.users.ldap_directory import Ldap3Directory


def _cfg(**overrides: object) -> LdapConfig:
    data = {
        "enabled": True,
        "host": "ldap.example.test",
        "port": 389,
        "base_dn": "DC=example,DC=test",
        "bind_dn": "CN=svc,DC=example,DC=test",
        "bind_password": "svc-secret",
        "user_filter": "(sAMAccountName=%{username})",
    }
    data.update(overrides)
    return LdapConfig(**data)


class _FakeEntry:
    def __init__(self, dn: str, attrs: dict[str, str]) -> None:
        self.entry_dn = dn
        self._attrs = attrs

    def __contains__(self, name: object) -> bool:
        return str(name) in self._attrs

    def __getitem__(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(value=self._attrs[name])


def test_authenticate_treats_open_none_as_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """ldap3 SyncStrategy.open() returns None on success; listening is the signal."""
    entry = _FakeEntry(
        "CN=Alice,DC=example,DC=test",
        {"sAMAccountName": "alice", "displayName": "Alice", "mail": "", "cn": "Alice"},
    )

    search_conn = MagicMock()
    search_conn.open.return_value = None
    search_conn.listening = True
    search_conn.start_tls.return_value = True
    search_conn.bind.return_value = True
    search_conn.search.return_value = True
    search_conn.entries = [entry]

    user_conn = MagicMock()
    user_conn.open.return_value = None
    user_conn.listening = True
    user_conn.start_tls.return_value = True
    user_conn.bind.return_value = True

    connections = iter([search_conn, user_conn])

    fake_ldap3 = SimpleNamespace(
        NONE="NONE",
        SUBTREE="SUBTREE",
        Connection=lambda *a, **k: next(connections),
        Server=MagicMock(),
        Tls=MagicMock(),
    )
    fake_exc = SimpleNamespace(LDAPException=Exception)

    import sys

    monkeypatch.setitem(sys.modules, "ldap3", fake_ldap3)
    monkeypatch.setitem(sys.modules, "ldap3.core.exceptions", fake_exc)

    result = Ldap3Directory(_cfg()).authenticate("alice", "user-pw")
    assert result is not None
    assert result.username == "alice"
    assert result.dn == "CN=Alice,DC=example,DC=test"
    search_conn.open.assert_called_once()
    user_conn.open.assert_called_once()


def test_authenticate_fails_when_bind_password_missing() -> None:
    result = Ldap3Directory(_cfg(bind_password="")).authenticate("alice", "user-pw")
    assert result is None


def test_authenticate_fails_when_not_listening(monkeypatch: pytest.MonkeyPatch) -> None:
    search_conn = MagicMock()
    search_conn.open.return_value = None
    search_conn.listening = False

    fake_ldap3 = SimpleNamespace(
        NONE="NONE",
        SUBTREE="SUBTREE",
        Connection=lambda *a, **k: search_conn,
        Server=MagicMock(),
        Tls=MagicMock(),
    )
    fake_exc = SimpleNamespace(LDAPException=Exception)

    import sys

    monkeypatch.setitem(sys.modules, "ldap3", fake_ldap3)
    monkeypatch.setitem(sys.modules, "ldap3.core.exceptions", fake_exc)

    assert Ldap3Directory(_cfg()).authenticate("alice", "user-pw") is None


def _install_search_capture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sam: str = "Andrey.Tishkin",
) -> MagicMock:
    """Fake ldap3; return the search connection so callers can inspect the filter."""
    entry = _FakeEntry(
        f"CN={sam},DC=example,DC=test",
        {
            "sAMAccountName": sam,
            "displayName": sam,
            "mail": "",
            "cn": sam,
        },
    )
    search_conn = MagicMock()
    search_conn.open.return_value = None
    search_conn.listening = True
    search_conn.start_tls.return_value = True
    search_conn.bind.return_value = True
    search_conn.search.return_value = True
    search_conn.entries = [entry]

    user_conn = MagicMock()
    user_conn.open.return_value = None
    user_conn.listening = True
    user_conn.start_tls.return_value = True
    user_conn.bind.return_value = True

    connections = iter([search_conn, user_conn])
    fake_ldap3 = SimpleNamespace(
        NONE="NONE",
        SUBTREE="SUBTREE",
        Connection=lambda *a, **k: next(connections),
        Server=MagicMock(),
        Tls=MagicMock(),
    )
    fake_exc = SimpleNamespace(LDAPException=Exception)
    import sys

    monkeypatch.setitem(sys.modules, "ldap3", fake_ldap3)
    monkeypatch.setitem(sys.modules, "ldap3.core.exceptions", fake_exc)
    return search_conn


@pytest.mark.parametrize(
    "typed_login",
    [
        "andrey.tishkin@wefry.ru",
        r"WEFRY\Andrey.Tishkin",
        r"wefry\andrey.tishkin",
        "Andrey.Tishkin",
    ],
)
def test_authenticate_normalizes_upn_and_domain_before_filter(
    monkeypatch: pytest.MonkeyPatch,
    typed_login: str,
) -> None:
    """UPN / DOMAIN\\user must become short name before %{username} substitution."""
    search_conn = _install_search_capture(monkeypatch)
    result = Ldap3Directory(_cfg()).authenticate(typed_login, "user-pw")
    assert result is not None
    assert result.username == "andrey.tishkin"
    kwargs = search_conn.search.call_args.kwargs
    assert kwargs["search_filter"] == "(sAMAccountName=andrey.tishkin)"


def test_authenticate_rejects_unusable_login_before_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid login must fail before ldap3 is imported / connected."""
    called = {"ldap": False}

    class _Boom:
        def __getattr__(self, name: str) -> object:
            called["ldap"] = True
            raise AssertionError(f"ldap3 should not load for bad login ({name})")

    import sys

    monkeypatch.setitem(sys.modules, "ldap3", _Boom())
    monkeypatch.setitem(sys.modules, "ldap3.core.exceptions", _Boom())
    assert Ldap3Directory(_cfg()).authenticate("bad*name@wefry.ru", "pw") is None
    assert called["ldap"] is False
