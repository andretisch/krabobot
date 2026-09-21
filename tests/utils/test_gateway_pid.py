"""Tests for gateway/serve PID helpers and restart cleanup."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from krabobot.utils.gateway_pid import (
    gateway_pid_path,
    is_gateway_running,
    prepare_process_restart,
    serve_pid_path,
    stop_gateway_process,
)


def test_gateway_pid_path_uses_config_parent(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg" / "config.json"
    assert gateway_pid_path(cfg) == tmp_path / "cfg" / "gateway.pid"
    assert serve_pid_path(cfg) == tmp_path / "cfg" / "serve.pid"


def test_is_gateway_running_reads_live_pid(tmp_path: Path, monkeypatch) -> None:
    pid_file = tmp_path / "gateway.pid"
    pid_file.write_text("12345\n", encoding="utf-8")
    monkeypatch.setattr(
        "krabobot.utils.gateway_pid.gateway_pid_path",
        lambda _config_path=None: pid_file,
    )
    monkeypatch.setattr("krabobot.utils.gateway_pid.is_process_alive", lambda pid: pid == 12345)

    running, pid = is_gateway_running()
    assert running is True
    assert pid == 12345


def test_is_gateway_running_clears_stale_pid(tmp_path: Path, monkeypatch) -> None:
    pid_file = tmp_path / "gateway.pid"
    pid_file.write_text("12345\n", encoding="utf-8")
    monkeypatch.setattr(
        "krabobot.utils.gateway_pid.gateway_pid_path",
        lambda _config_path=None: pid_file,
    )
    monkeypatch.setattr("krabobot.utils.gateway_pid.is_process_alive", lambda _pid: False)

    running, pid = is_gateway_running()
    assert running is False
    assert pid is None
    assert not pid_file.exists()


def test_stop_gateway_process_terminates_other_pid(tmp_path: Path, monkeypatch) -> None:
    pid_file = tmp_path / "gateway.pid"
    pid_file.write_text("9999\n", encoding="utf-8")
    monkeypatch.setattr(
        "krabobot.utils.gateway_pid.gateway_pid_path",
        lambda _config_path=None: pid_file,
    )
    monkeypatch.setattr("krabobot.utils.gateway_pid.is_process_alive", lambda pid: pid == 9999)
    seen: list[int] = []

    def _terminate(pid: int, *, timeout_s: float = 15.0) -> bool:
        seen.append(pid)
        return True

    monkeypatch.setattr("krabobot.utils.gateway_pid.terminate_pid", _terminate)

    stopped = stop_gateway_process()
    assert stopped == 9999
    assert seen == [9999]
    assert not pid_file.exists()


def test_prepare_process_restart_from_serve_stops_gateway(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["krabobot", "serve", "--port", "8080"])
    cleared: list[object] = []
    stopped: list[object] = []
    monkeypatch.setattr(
        "krabobot.utils.gateway_pid.clear_serve_pid",
        lambda **kwargs: cleared.append(kwargs),
    )
    monkeypatch.setattr(
        "krabobot.utils.gateway_pid.stop_gateway_process",
        lambda: stopped.append(True) or 42,
    )

    prepare_process_restart()

    assert cleared
    assert stopped == [True]


def test_prepare_process_restart_from_gateway_clears_own_pid(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["krabobot", "gateway"])
    cleared: list[object] = []
    monkeypatch.setattr(
        "krabobot.utils.gateway_pid.clear_gateway_pid",
        lambda **kwargs: cleared.append(kwargs),
    )
    monkeypatch.setattr(
        "krabobot.utils.gateway_pid.stop_gateway_process",
        lambda: (_ for _ in ()).throw(AssertionError("must not stop other gateway")),
    )

    prepare_process_restart()

    assert cleared and "only_pid" in cleared[0]


@pytest.mark.asyncio
async def test_cmd_restart_calls_prepare_before_execv() -> None:
    from krabobot.bus.events import InboundMessage
    from krabobot.command.builtin import cmd_restart
    from krabobot.command.router import CommandContext
    from tests.cli.test_restart_command import _make_loop

    loop, _bus = _make_loop()
    msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/restart")
    ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/restart", loop=loop)

    prepared: list[bool] = []
    with patch(
        "krabobot.utils.gateway_pid.prepare_process_restart",
        lambda: prepared.append(True),
    ):
        with patch("krabobot.command.builtin.os.execv") as mock_execv:
            out = await cmd_restart(ctx)
            assert "Перезапускаюсь" in out.content
            await asyncio.sleep(1.5)
            assert prepared == [True]
            mock_execv.assert_called_once()
