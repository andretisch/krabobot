"""PID file helpers so serve can detect/start a standalone gateway process."""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path


def gateway_pid_path(config_path: Path | None = None) -> Path:
    """PID file next to the active config (supports multi-instance via -c)."""
    from krabobot.config.loader import get_config_path

    base = (config_path or get_config_path()).parent
    return base / "gateway.pid"


def serve_pid_path(config_path: Path | None = None) -> Path:
    """PID file for the HTTP serve process."""
    from krabobot.config.loader import get_config_path

    base = (config_path or get_config_path()).parent
    return base / "serve.pid"


def is_process_alive(pid: int) -> bool:
    """Return True if *pid* refers to a running process."""
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            import ctypes

            # PROCESS_QUERY_LIMITED_INFORMATION
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid_file(path: Path) -> int | None:
    """Read an integer PID from *path*, or None if missing/invalid."""
    try:
        text = path.read_text(encoding="utf-8").strip()
        return int(text)
    except (OSError, ValueError):
        return None


def read_gateway_pid(path: Path | None = None) -> int | None:
    """Read PID from the gateway pidfile, or None if missing/invalid."""
    return read_pid_file(path or gateway_pid_path())


def is_gateway_running(config_path: Path | None = None) -> tuple[bool, int | None]:
    """Return (running, pid). Clears a stale pidfile when the process is gone."""
    pid_path = gateway_pid_path(config_path)
    pid = read_gateway_pid(pid_path)
    if pid is None:
        return False, None
    if is_process_alive(pid):
        return True, pid
    try:
        pid_path.unlink(missing_ok=True)
    except OSError:
        pass
    return False, None


def write_pid_file(path: Path, pid: int | None = None) -> Path:
    """Write *pid* (default: current) to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{pid if pid is not None else os.getpid()}\n", encoding="utf-8")
    return path


def write_gateway_pid(config_path: Path | None = None, pid: int | None = None) -> Path:
    """Write the current (or given) PID to the gateway pidfile."""
    return write_pid_file(gateway_pid_path(config_path), pid)


def write_serve_pid(config_path: Path | None = None, pid: int | None = None) -> Path:
    """Write the current (or given) PID to the serve pidfile."""
    return write_pid_file(serve_pid_path(config_path), pid)


def clear_pid_file(path: Path, *, only_pid: int | None = None) -> None:
    """Remove *path*. If *only_pid* is set, only remove when it matches."""
    if only_pid is not None:
        current = read_pid_file(path)
        if current != only_pid:
            return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def clear_gateway_pid(config_path: Path | None = None, *, only_pid: int | None = None) -> None:
    """Remove the gateway pidfile. If *only_pid* is set, only remove when it matches."""
    clear_pid_file(gateway_pid_path(config_path), only_pid=only_pid)


def clear_serve_pid(config_path: Path | None = None, *, only_pid: int | None = None) -> None:
    """Remove the serve pidfile. If *only_pid* is set, only remove when it matches."""
    clear_pid_file(serve_pid_path(config_path), only_pid=only_pid)


def terminate_pid(pid: int, *, timeout_s: float = 15.0) -> bool:
    """Ask *pid* to exit; escalate to kill if needed. Returns True if it exited."""
    if pid <= 0 or pid == os.getpid() or not is_process_alive(pid):
        return True
    try:
        if sys.platform == "win32":
            os.kill(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError:
        return not is_process_alive(pid)

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not is_process_alive(pid):
            return True
        time.sleep(0.1)

    try:
        if sys.platform == "win32":
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(1, False, pid)  # PROCESS_TERMINATE
            if handle:
                ctypes.windll.kernel32.TerminateProcess(handle, 1)
                ctypes.windll.kernel32.CloseHandle(handle)
        else:
            os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    return not is_process_alive(pid)


def stop_gateway_process(config_path: Path | None = None) -> int | None:
    """Stop the process recorded in gateway.pid (if any). Returns stopped pid or None."""
    running, pid = is_gateway_running(config_path)
    if not running or pid is None:
        clear_gateway_pid(config_path)
        return None
    if pid == os.getpid():
        clear_gateway_pid(config_path, only_pid=pid)
        return pid
    terminate_pid(pid)
    clear_gateway_pid(config_path, only_pid=pid)
    return pid


def prepare_process_restart() -> None:
    """Clean up sibling processes before ``os.execv`` so channels come back cleanly.

    - ``krabobot gateway``: clear our pidfile (new process rewrites it).
    - ``krabobot serve``: stop the gateway process so the new serve can spawn a fresh one.
    """
    argv = [a.lower() for a in sys.argv]
    if "gateway" in argv:
        clear_gateway_pid(only_pid=os.getpid())
        return
    if "serve" in argv:
        clear_serve_pid(only_pid=os.getpid())
        stop_gateway_process()
