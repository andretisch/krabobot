"""Tests for WinRT mic permission helper (mocked; no MessageBox / Settings)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice import mic_permission as mp  # noqa: E402


def test_ensure_skipped_on_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mp.sys, "platform", "linux")
    r = mp.ensure_microphone_access(interactive=False)
    assert r.status == "skipped"


def test_ensure_allowed_no_dialog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mp.sys, "platform", "win32")

    async def _ok() -> tuple[str, str]:
        return "allowed", "AppCapability request=ALLOWED"

    monkeypatch.setattr(mp, "_request_access_async", _ok)
    shown: list[str] = []
    monkeypatch.setattr(mp, "show_mic_access_dialog", lambda *a, **k: shown.append("x"))
    monkeypatch.setattr(mp, "open_windows_mic_privacy_settings", lambda: True)
    r = mp.ensure_microphone_access(interactive=True)
    assert r.status == "allowed"
    assert shown == []


def test_ensure_denied_shows_dialog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mp.sys, "platform", "win32")

    async def _deny() -> tuple[str, str]:
        return "denied", "DENIED_BY_USER"

    monkeypatch.setattr(mp, "_request_access_async", _deny)
    shown: list[str] = []
    monkeypatch.setattr(
        mp, "show_mic_access_dialog", lambda text, **k: shown.append(text)
    )
    monkeypatch.setattr(mp, "open_windows_mic_privacy_settings", lambda: True)
    r = mp.ensure_microphone_access(interactive=True)
    assert r.status == "denied"
    assert r.opened_settings is True
    assert shown and "микрофон" in shown[0].lower()


def test_ensure_denied_silent_when_not_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mp.sys, "platform", "win32")

    async def _deny() -> tuple[str, str]:
        return "denied", "x"

    monkeypatch.setattr(mp, "_request_access_async", _deny)
    monkeypatch.setattr(
        mp, "show_mic_access_dialog", lambda *a, **k: (_ for _ in ()).throw(AssertionError())
    )
    r = mp.ensure_microphone_access(interactive=False)
    assert r.status == "denied"
    assert r.opened_settings is False


def test_status_name() -> None:
    assert mp._status_name(SimpleNamespace())  # does not raise
