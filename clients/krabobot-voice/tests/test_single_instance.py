"""Single-instance mutex: second launch is refused; one-shot flags are not."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice import single_instance as si  # noqa: E402
from krabobot_voice.app import _main_inner  # noqa: E402

_MSG = "krabobot-voice уже запущен"


@pytest.fixture(autouse=True)
def _reset_mutex_state() -> None:
    si.release()
    si._held = None
    si._close_owned = None


def test_already_exists_refuses_and_closes_duplicate() -> None:
    closed: list[int] = []

    def create(name: str) -> tuple[int, int]:
        assert name == si.MUTEX_NAME
        return 42, si.ERROR_ALREADY_EXISTS

    assert si.acquire(create=create, close=closed.append) is False
    assert closed == [42]
    assert si._held is None


def test_acquire_holds_until_release() -> None:
    closed: list[int] = []

    def create(name: str) -> tuple[int, int]:
        assert name == r"Local\krabobot-voice"
        return 7, 0

    assert si.acquire(create=create, close=closed.append) is True
    assert si._held == 7
    si.release()
    assert closed == [7]
    assert si._held is None


def test_one_shot_flags_are_not_resident() -> None:
    assert si.is_resident_launch(["--list-devices"]) is False
    assert si.is_resident_launch(["--test-loopback", "voice.yaml"]) is False
    assert si.is_resident_launch([]) is True
    assert si.is_resident_launch(["--console"]) is True
    assert si.is_resident_launch(["--minimized", "voice.yaml"]) is True


def test_notify_console_prints_without_dialog(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(si.sys, "platform", "win32")
    monkeypatch.setattr(
        si, "_message_box", lambda text: (_ for _ in ()).throw(AssertionError(text))
    )
    si.notify_already_running(console=True)
    assert capsys.readouterr().out.strip() == _MSG


def test_notify_window_uses_message_box(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shown: list[str] = []
    monkeypatch.setattr(si.sys, "platform", "win32")
    monkeypatch.setattr(si, "_message_box", shown.append)
    si.notify_already_running(console=False)
    assert shown == [_MSG]
    assert capsys.readouterr().out == ""


def test_list_devices_does_not_take_mutex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "krabobot_voice.app.ensure_single_instance",
        lambda **k: (_ for _ in ()).throw(AssertionError("mutex")),
    )
    monkeypatch.setattr("krabobot_voice.audio_io.require_sounddevice", lambda: None)
    monkeypatch.setattr("krabobot_voice.audio_io.list_input_devices", lambda: [])
    monkeypatch.setattr("krabobot_voice.audio_io.format_input_devices_lines", lambda devices: "")
    monkeypatch.setattr("krabobot_voice.audio_io.mic_privacy_hint", lambda: "")
    assert _main_inner(["--list-devices"]) == 0


def test_test_loopback_does_not_take_mutex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "krabobot_voice.app.ensure_single_instance",
        lambda **k: (_ for _ in ()).throw(AssertionError("mutex")),
    )
    monkeypatch.setattr("krabobot_voice.app.VoiceClientConfig.load", lambda path: object())
    monkeypatch.setattr("krabobot_voice.app.run_test_loopback", lambda cfg, duration_s=3.0: 0)
    assert _main_inner(["--test-loopback"]) == 0


def test_second_console_launch_exits_before_loop(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("krabobot_voice.single_instance.acquire", lambda: False)
    monkeypatch.setattr(
        "krabobot_voice.app.VoiceClientConfig.load",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("config")),
    )
    monkeypatch.setattr(
        "krabobot_voice.app.run_loop",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("loop")),
    )
    assert _main_inner(["--console"]) == 1
    assert capsys.readouterr().out.strip() == _MSG


def test_second_window_launch_message_box_before_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    shown: list[str] = []
    monkeypatch.setattr(si.sys, "platform", "win32")
    monkeypatch.setattr("krabobot_voice.single_instance.acquire", lambda: False)
    monkeypatch.setattr("krabobot_voice.single_instance._message_box", shown.append)
    monkeypatch.setattr(
        "krabobot_voice.app.VoiceClientConfig.load",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("config")),
    )
    assert _main_inner([]) == 1
    assert shown == [_MSG]


def test_real_mutex_second_create_already_exists() -> None:
    if sys.platform != "win32":
        pytest.skip("named mutex is Windows-only")
    name = r"Local\krabobot-voice-unit-test"
    assert si.acquire(name) is True
    try:
        handle, err = si._win_create_mutex(name)
        assert err == si.ERROR_ALREADY_EXISTS
        assert handle
        si._win_close_handle(handle)
    finally:
        si.release()
    handle, err = si._win_create_mutex(name)
    try:
        assert err != si.ERROR_ALREADY_EXISTS
        assert handle
    finally:
        if handle:
            si._win_close_handle(handle)
