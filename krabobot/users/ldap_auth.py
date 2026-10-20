"""LDAP/AD identity helpers for web sign-in. No network and no ldap3 import."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class LdapDirectoryUser:
    """Directory account after a successful credential check."""

    username: str  # stable identity key (normalized sAMAccountName / uid)
    display_name: str
    email: str = ""
    dn: str = ""


class LdapDirectory(Protocol):
    """Search-then-bind directory. Implementations must not log passwords."""

    def authenticate(self, username: str, password: str) -> LdapDirectoryUser | None:
        """Return the directory user, or None if credentials are rejected."""


def normalize_ldap_username(raw: str) -> str:
    """Stable identity key: sAMAccountName, without DOMAIN\\ or @realm, lowercased."""
    text = (raw or "").strip()
    if not text or any(ord(ch) < 32 for ch in text):
        return ""
    if "\\" in text:
        text = text.split("\\")[-1].strip()
    if "@" in text:
        text = text.split("@", 1)[0].strip()
    if not text or len(text) > 128:
        return ""
    if any(ch in text for ch in '*()\\/\x00:'):
        return ""
    return text.lower()


def escape_ldap_filter(value: str) -> str:
    """RFC 4515 filter escaping so a login cannot change the search filter."""
    out: list[str] = []
    for ch in value:
        if ch in {"\\", "*", "(", ")", "\x00"}:
            out.append("\\" + format(ord(ch), "02x"))
        else:
            out.append(ch)
    return "".join(out)


def build_user_filter(user_filter: str, username_attribute: str, username: str) -> str:
    """GitLab-style filter.

    ``%{username}`` is replaced with the escaped login. If the placeholder is
    absent, ``(username_attribute=login)`` is AND-ed with ``user_filter``.
    """
    escaped = escape_ldap_filter(username)
    template = (user_filter or "").strip()
    if "%{username}" in template:
        return template.replace("%{username}", escaped)
    attr = (username_attribute or "sAMAccountName").strip() or "sAMAccountName"
    uid_clause = f"({attr}={escaped})"
    if not template:
        return uid_clause
    return f"(&{uid_clause}{template})"


def pick_display_name(*candidates: str, fallback: str) -> str:
    """First non-empty directory name, otherwise the login."""
    for raw in candidates:
        text = (raw or "").strip()
        if text:
            return text
    return fallback
