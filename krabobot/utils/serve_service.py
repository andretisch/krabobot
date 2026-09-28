"""Install / uninstall ``krabobot serve`` as a user-level background service."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

SERVICE_NAME = "krabobot-serve"
WINDOWS_TASK_NAME = "krabobot-serve"
LINUX_UNIT_NAME = "krabobot-serve.service"


@dataclass(frozen=True)
class ServiceResult:
    """Outcome of install / uninstall / start helper."""

    ok: bool
    message: str
    hints: tuple[str, ...] = ()


def build_serve_argv(
    *,
    config: str | None = None,
    workspace: str | None = None,
    host: str | None = None,
    port: int | None = None,
    verbose: bool = False,
) -> list[str]:
    """Argv for ``python -m krabobot serve …`` (no executable prefix)."""
    args = ["-m", "krabobot", "serve"]
    if config:
        args.extend(["--config", str(Path(config).expanduser().resolve())])
    if workspace:
        args.extend(["--workspace", str(Path(workspace).expanduser().resolve())])
    if host is not None:
        args.extend(["--host", host])
    if port is not None:
        args.extend(["--port", str(port)])
    if verbose:
        args.append("--verbose")
    return args


def serve_command_line(
    *,
    config: str | None = None,
    workspace: str | None = None,
    host: str | None = None,
    port: int | None = None,
    verbose: bool = False,
) -> list[str]:
    """Full command: interpreter + serve argv."""
    return [sys.executable, *build_serve_argv(
        config=config,
        workspace=workspace,
        host=host,
        port=port,
        verbose=verbose,
    )]


def _quote_win(arg: str) -> str:
    if not arg:
        return '""'
    if any(c in arg for c in ' \t"&<>|^'):
        return '"' + arg.replace('"', '\\"') + '"'
    return arg


def _run(cmd: list[str], *, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _windows_task_xml(cmd: list[str], working_directory: Path) -> str:
    """Task Scheduler XML (UTF-16 written by caller) for ONLOGON + restart."""
    command = cmd[0]
    arguments = " ".join(_quote_win(a) for a in cmd[1:])
    # Escape XML special chars in paths/args.
    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    return textwrap.dedent(
        f"""\
        <?xml version="1.0" encoding="UTF-16"?>
        <Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
          <RegistrationInfo>
            <Description>krabobot OpenAI-compatible API server (user logon)</Description>
          </RegistrationInfo>
          <Triggers>
            <LogonTrigger>
              <Enabled>true</Enabled>
            </LogonTrigger>
          </Triggers>
          <Principals>
            <Principal id="Author">
              <LogonType>InteractiveToken</LogonType>
              <RunLevel>LeastPrivilege</RunLevel>
            </Principal>
          </Principals>
          <Settings>
            <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
            <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
            <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
            <AllowHardTerminate>true</AllowHardTerminate>
            <StartWhenAvailable>true</StartWhenAvailable>
            <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
            <RestartOnFailure>
              <Interval>PT1M</Interval>
              <Count>3</Count>
            </RestartOnFailure>
          </Settings>
          <Actions Context="Author">
            <Exec>
              <Command>{esc(command)}</Command>
              <Arguments>{esc(arguments)}</Arguments>
              <WorkingDirectory>{esc(str(working_directory))}</WorkingDirectory>
            </Exec>
          </Actions>
        </Task>
        """
    )


def _install_windows(
    cmd: list[str],
    *,
    working_directory: Path,
    start_now: bool,
) -> ServiceResult:
    if shutil.which("schtasks") is None:
        return ServiceResult(
            ok=False,
            message="schtasks not found — Task Scheduler is required on Windows.",
            hints=(
                "Install via an elevated PowerShell if Task Scheduler is disabled, "
                "or run: krabobot serve (foreground).",
            ),
        )

    xml = _windows_task_xml(cmd, working_directory)
    # schtasks expects UTF-16 LE XML when /XML is used.
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".xml",
        delete=False,
        encoding="utf-16",
    ) as tmp:
        tmp.write(xml)
        xml_path = tmp.name

    try:
        create = _run(
            ["schtasks", "/Create", "/TN", WINDOWS_TASK_NAME, "/XML", xml_path, "/F"]
        )
    finally:
        try:
            os.unlink(xml_path)
        except OSError:
            pass

    out = ((create.stdout or "") + (create.stderr or "")).strip()
    if create.returncode != 0:
        elevated = any(
            s in out.lower()
            for s in ("access is denied", "отказано в доступе", "privilege", "elevation")
        )
        hints: list[str] = []
        if elevated:
            hints.append(
                "Запустите терминал от имени администратора и повторите "
                "`krabobot serve --install-as-service`, либо создайте задачу вручную "
                "в Планировщике заданий (триггер: при входе пользователя)."
            )
        hints.extend(_windows_hints())
        return ServiceResult(
            ok=False,
            message=out or f"schtasks failed with code {create.returncode}",
            hints=tuple(hints),
        )

    hints = list(_windows_hints())
    if start_now:
        start = _run(["schtasks", "/Run", "/TN", WINDOWS_TASK_NAME])
        if start.returncode != 0:
            hints.insert(
                0,
                "Задача создана, но не удалось запустить сейчас: "
                + ((start.stderr or start.stdout or "").strip() or f"code {start.returncode}"),
            )
        else:
            hints.insert(0, "Задача запущена сейчас (`schtasks /Run`).")

    return ServiceResult(
        ok=True,
        message=f"Задача Планировщика «{WINDOWS_TASK_NAME}» создана (старт при входе пользователя).",
        hints=tuple(hints),
    )


def _uninstall_windows() -> ServiceResult:
    if shutil.which("schtasks") is None:
        return ServiceResult(ok=False, message="schtasks not found")
    # End first (ignore errors), then delete.
    _run(["schtasks", "/End", "/TN", WINDOWS_TASK_NAME])
    delete = _run(["schtasks", "/Delete", "/TN", WINDOWS_TASK_NAME, "/F"])
    out = ((delete.stdout or "") + (delete.stderr or "")).strip()
    if delete.returncode != 0:
        not_found = any(
            s in out.lower()
            for s in ("cannot find", "не удается найти", "does not exist", "не найден")
        )
        if not_found:
            return ServiceResult(
                ok=True,
                message=f"Задачи «{WINDOWS_TASK_NAME}» нет — уже удалена.",
            )
        return ServiceResult(ok=False, message=out or f"schtasks delete failed ({delete.returncode})")
    return ServiceResult(
        ok=True,
        message=f"Задача Планировщика «{WINDOWS_TASK_NAME}» удалена.",
        hints=_windows_hints(),
    )


def _windows_hints() -> tuple[str, ...]:
    return (
        f"Статус:  schtasks /Query /TN {WINDOWS_TASK_NAME} /V /FO LIST",
        f"Старт:   schtasks /Run /TN {WINDOWS_TASK_NAME}",
        f"Стоп:    schtasks /End /TN {WINDOWS_TASK_NAME}",
        "Удалить: krabobot serve --uninstall-as-service",
    )


def _linux_unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / LINUX_UNIT_NAME


def _linux_unit_text(cmd: list[str], working_directory: Path) -> str:
    exec_start = " ".join(_quote_posix(a) for a in cmd)
    return textwrap.dedent(
        f"""\
        [Unit]
        Description=krabobot OpenAI-compatible API server
        After=network-online.target
        Wants=network-online.target

        [Service]
        Type=simple
        WorkingDirectory={working_directory}
        ExecStart={exec_start}
        Restart=on-failure
        RestartSec=5

        [Install]
        WantedBy=default.target
        """
    )


def _quote_posix(arg: str) -> str:
    if not arg:
        return "''"
    if all(c.isalnum() or c in "/._-:+@" for c in arg):
        return arg
    return "'" + arg.replace("'", "'\"'\"'") + "'"


def _systemctl_user(*args: str) -> subprocess.CompletedProcess[str]:
    return _run(["systemctl", "--user", *args])


def _install_linux(
    cmd: list[str],
    *,
    working_directory: Path,
    start_now: bool,
) -> ServiceResult:
    if shutil.which("systemctl") is None:
        return ServiceResult(
            ok=False,
            message="systemctl not found — systemd user services require systemd.",
        )

    unit_path = _linux_unit_path()
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(_linux_unit_text(cmd, working_directory), encoding="utf-8")

    reload = _systemctl_user("daemon-reload")
    if reload.returncode != 0:
        return ServiceResult(
            ok=False,
            message=((reload.stderr or reload.stdout or "daemon-reload failed").strip()),
            hints=_linux_hints(),
        )

    enable_args = ["enable", "--now", LINUX_UNIT_NAME] if start_now else ["enable", LINUX_UNIT_NAME]
    enable = _systemctl_user(*enable_args)
    if enable.returncode != 0:
        return ServiceResult(
            ok=False,
            message=((enable.stderr or enable.stdout or "systemctl enable failed").strip()),
            hints=_linux_hints(),
        )

    return ServiceResult(
        ok=True,
        message=f"systemd --user unit «{LINUX_UNIT_NAME}» установлен"
        + (" и запущен." if start_now else " (enable)."),
        hints=_linux_hints(),
    )


def _uninstall_linux() -> ServiceResult:
    if shutil.which("systemctl") is None:
        return ServiceResult(ok=False, message="systemctl not found")

    _systemctl_user("disable", "--now", LINUX_UNIT_NAME)
    unit_path = _linux_unit_path()
    try:
        if unit_path.is_file():
            unit_path.unlink()
    except OSError as exc:
        return ServiceResult(ok=False, message=f"Cannot remove {unit_path}: {exc}")

    _systemctl_user("daemon-reload")
    return ServiceResult(
        ok=True,
        message=f"Unit «{LINUX_UNIT_NAME}» отключён и удалён.",
        hints=_linux_hints(),
    )


def _linux_hints() -> tuple[str, ...]:
    return (
        f"Статус:  systemctl --user status {LINUX_UNIT_NAME}",
        f"Старт:   systemctl --user start {LINUX_UNIT_NAME}",
        f"Стоп:    systemctl --user stop {LINUX_UNIT_NAME}",
        f"Логи:    journalctl --user -u {LINUX_UNIT_NAME} -f",
        "Удалить: krabobot serve --uninstall-as-service",
        "Если нужно без входа в сессию: loginctl enable-linger $USER",
    )


def install_serve_service(
    *,
    config: str | None = None,
    workspace: str | None = None,
    host: str | None = None,
    port: int | None = None,
    verbose: bool = False,
    start_now: bool = False,
    working_directory: Path | None = None,
) -> ServiceResult:
    """Register a user-level service that runs ``krabobot serve``."""
    cmd = serve_command_line(
        config=config,
        workspace=workspace,
        host=host,
        port=port,
        verbose=verbose,
    )
    cwd = (working_directory or Path.cwd()).resolve()
    if sys.platform == "win32":
        return _install_windows(cmd, working_directory=cwd, start_now=start_now)
    if sys.platform.startswith("linux"):
        return _install_linux(cmd, working_directory=cwd, start_now=start_now)
    return ServiceResult(
        ok=False,
        message=f"Установка сервиса не поддерживается на {sys.platform}.",
        hints=("Используйте Linux (systemd --user) или Windows (Планировщик заданий).",),
    )


def uninstall_serve_service() -> ServiceResult:
    """Remove the user-level serve service if present."""
    if sys.platform == "win32":
        return _uninstall_windows()
    if sys.platform.startswith("linux"):
        return _uninstall_linux()
    return ServiceResult(
        ok=False,
        message=f"Удаление сервиса не поддерживается на {sys.platform}.",
    )
