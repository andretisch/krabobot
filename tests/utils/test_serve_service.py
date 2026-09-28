"""Unit tests for serve user-service install helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from krabobot.utils import serve_service as ss


def _proc(code: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    m = MagicMock()
    m.returncode = code
    m.stdout = stdout
    m.stderr = stderr
    return m


def test_build_serve_argv_includes_host_port_config(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    args = ss.build_serve_argv(config=str(cfg), host="127.0.0.1", port=8900)
    assert args[:3] == ["-m", "krabobot", "serve"]
    assert "--host" in args and "127.0.0.1" in args
    assert "--port" in args and "8900" in args
    assert "--config" in args


def test_serve_command_line_uses_sys_executable() -> None:
    cmd = ss.serve_command_line(host="0.0.0.0", port=9000)
    assert cmd[0] == sys.executable
    assert "-m" in cmd and "krabobot" in cmd and "serve" in cmd


def test_windows_task_xml_contains_restart_and_command(tmp_path: Path) -> None:
    cmd = [sys.executable, "-m", "krabobot", "serve", "--host", "127.0.0.1"]
    xml = ss._windows_task_xml(cmd, tmp_path)
    assert "LogonTrigger" in xml
    assert "RestartOnFailure" in xml
    assert "LeastPrivilege" in xml
    assert "-m" in xml and "krabobot" in xml and "serve" in xml


def test_linux_unit_text_execstart(tmp_path: Path) -> None:
    cmd = [sys.executable, "-m", "krabobot", "serve", "--port", "8900"]
    text = ss._linux_unit_text(cmd, tmp_path)
    assert "[Service]" in text
    assert "ExecStart=" in text
    assert "Restart=on-failure" in text
    assert "krabobot" in text
    assert str(tmp_path) in text or "WorkingDirectory=" in text


def test_install_windows_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ss.sys, "platform", "win32")
    monkeypatch.setattr(ss.shutil, "which", lambda n: "schtasks" if n == "schtasks" else None)
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        if cmd[:2] == ["schtasks", "/Create"]:
            return _proc(0, stdout="SUCCESS: The scheduled task")
        if cmd[:2] == ["schtasks", "/Run"]:
            return _proc(0, stdout="SUCCESS")
        return _proc(1, stderr=f"unexpected: {cmd}")

    monkeypatch.setattr(ss, "_run", fake_run)
    result = ss.install_serve_service(
        host="127.0.0.1",
        port=8900,
        start_now=True,
        working_directory=tmp_path,
    )
    assert result.ok
    assert any(c[:2] == ["schtasks", "/Create"] for c in calls)
    assert any(c[:2] == ["schtasks", "/Run"] for c in calls)
    assert any("/XML" in c for c in calls if c[:2] == ["schtasks", "/Create"])


def test_install_windows_access_denied_hints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ss.sys, "platform", "win32")
    monkeypatch.setattr(ss.shutil, "which", lambda n: "schtasks" if n == "schtasks" else None)

    def fake_run(cmd, **_kwargs):
        if cmd[:2] == ["schtasks", "/Create"]:
            return _proc(1, stderr="ERROR: Access is denied.")
        return _proc(0)

    monkeypatch.setattr(ss, "_run", fake_run)
    result = ss.install_serve_service(working_directory=tmp_path)
    assert not result.ok
    assert any("администратора" in h.lower() or "admin" in h.lower() for h in result.hints)


def test_uninstall_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ss.sys, "platform", "win32")
    monkeypatch.setattr(ss.shutil, "which", lambda n: "schtasks" if n == "schtasks" else None)

    def fake_run(cmd, **_kwargs):
        if cmd[:2] == ["schtasks", "/Delete"]:
            return _proc(0, stdout="SUCCESS")
        return _proc(0)

    monkeypatch.setattr(ss, "_run", fake_run)
    result = ss.uninstall_serve_service()
    assert result.ok


def test_install_linux_writes_unit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ss.sys, "platform", "linux")
    monkeypatch.setattr(ss.shutil, "which", lambda n: "systemctl" if n == "systemctl" else None)
    unit_dir = tmp_path / ".config" / "systemd" / "user"
    unit_path = unit_dir / ss.LINUX_UNIT_NAME
    monkeypatch.setattr(ss, "_linux_unit_path", lambda: unit_path)

    def fake_run(cmd, **_kwargs):
        if cmd[:2] == ["systemctl", "--user"]:
            return _proc(0)
        return _proc(1, stderr=str(cmd))

    monkeypatch.setattr(ss, "_run", fake_run)
    result = ss.install_serve_service(
        host="127.0.0.1",
        port=8900,
        start_now=True,
        working_directory=tmp_path / "repo",
    )
    assert result.ok
    assert unit_path.is_file()
    body = unit_path.read_text(encoding="utf-8")
    assert "ExecStart=" in body
    assert "krabobot" in body
    assert "serve" in body


def test_uninstall_linux_removes_unit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ss.sys, "platform", "linux")
    monkeypatch.setattr(ss.shutil, "which", lambda n: "systemctl" if n == "systemctl" else None)
    unit_path = tmp_path / ".config" / "systemd" / "user" / ss.LINUX_UNIT_NAME
    unit_path.parent.mkdir(parents=True)
    unit_path.write_text("[Unit]\nDescription=x\n", encoding="utf-8")
    monkeypatch.setattr(ss, "_linux_unit_path", lambda: unit_path)

    monkeypatch.setattr(ss, "_run", lambda *_a, **_k: _proc(0))
    result = ss.uninstall_serve_service()
    assert result.ok
    assert not unit_path.exists()


def test_unsupported_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ss.sys, "platform", "darwin")
    result = ss.install_serve_service()
    assert not result.ok
