"""One resident krabobot-voice process per Windows session.

Dev (`python -m krabobot_voice`) and the frozen exe share one named mutex.
One-shot CLI flags must not acquire it — see ``ONE_SHOT_FLAGS``.
"""

from __future__ import annotations

import atexit
import sys
from collections.abc import Callable
from typing import Any

MUTEX_NAME = r"Local\krabobot-voice"
ALREADY_RUNNING_MESSAGE = "krabobot-voice уже запущен"
ALREADY_RUNNING_EXIT_CODE = 1

# Do not start the audio loop / tray / window.
ONE_SHOT_FLAGS = ("--list-devices", "--test-loopback")

ERROR_ALREADY_EXISTS = 183

_MB_OK = 0x00000000
_MB_ICONINFORMATION = 0x00000040
_MB_SETFOREGROUND = 0x00010000
_MB_TOPMOST = 0x00040000

_held: int | None = None
_close_owned: Callable[[int], None] | None = None
_atexit_installed = False
_kernel32: Any = None


def is_resident_launch(argv: list[str]) -> bool:
    """True when this argv starts the long-running client."""
    return not any(flag in argv for flag in ONE_SHOT_FLAGS)


def ensure_single_instance(*, console: bool) -> bool:
    """Acquire the mutex. Return False if another instance is already running.

    On refusal, a windowed launch shows a message box; ``--console`` prints
    the same text. The mutex is held until process exit.
    """
    if acquire():
        return True
    notify_already_running(console=console)
    return False


def acquire(
    name: str = MUTEX_NAME,
    *,
    create: Callable[[str], tuple[int, int]] | None = None,
    close: Callable[[int], None] | None = None,
) -> bool:
    """Return True if this process now holds ``name``.

    ``create`` returns ``(handle, last_error)``. Injected for tests.
    Non-Windows with no injected ``create`` does not guard.
    """
    global _held, _close_owned
    if _held:
        return True
    if create is None and sys.platform != "win32":
        return True

    create_fn = create or _win_create_mutex
    close_duplicate = close or _win_close_handle
    close_owned = close or _win_release_owned
    handle, err = create_fn(name)
    if err == ERROR_ALREADY_EXISTS:
        if handle:
            close_duplicate(handle)
        return False
    if not handle:
        return True
    _held = handle
    _close_owned = close_owned
    _install_atexit()
    return True


def release() -> None:
    """Drop the named mutex so a later launch can start."""
    global _held, _close_owned
    handle, close_fn = _held, _close_owned
    _held = None
    _close_owned = None
    if handle and close_fn is not None:
        close_fn(handle)


def notify_already_running(*, console: bool) -> None:
    if console or sys.platform != "win32":
        print(ALREADY_RUNNING_MESSAGE, flush=True)
        return
    _message_box(ALREADY_RUNNING_MESSAGE)


def _install_atexit() -> None:
    global _atexit_installed
    if _atexit_installed:
        return
    atexit.register(release)
    _atexit_installed = True


def _kernel32_dll() -> Any:
    global _kernel32
    if _kernel32 is None:
        import ctypes
        from ctypes import wintypes

        dll = ctypes.WinDLL("kernel32", use_last_error=True)
        dll.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
        dll.CreateMutexW.restype = wintypes.HANDLE
        dll.ReleaseMutex.argtypes = (wintypes.HANDLE,)
        dll.ReleaseMutex.restype = wintypes.BOOL
        dll.CloseHandle.argtypes = (wintypes.HANDLE,)
        dll.CloseHandle.restype = wintypes.BOOL
        _kernel32 = dll
    return _kernel32


def _win_create_mutex(name: str) -> tuple[int, int]:
    import ctypes

    dll = _kernel32_dll()
    ctypes.set_last_error(0)
    raw = dll.CreateMutexW(None, True, name)
    err = int(ctypes.get_last_error())
    return int(raw or 0), err


def _win_close_handle(handle: int) -> None:
    _kernel32_dll().CloseHandle(handle)


def _win_release_owned(handle: int) -> None:
    """Release ownership and close our handle so the name disappears."""
    dll = _kernel32_dll()
    dll.ReleaseMutex(handle)
    dll.CloseHandle(handle)


def _message_box(text: str) -> None:
    if sys.platform != "win32":
        print(text, flush=True)
        return
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.MessageBoxW.argtypes = (
            wintypes.HWND,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.UINT,
        )
        user32.MessageBoxW.restype = ctypes.c_int
        user32.MessageBoxW(
            None,
            text,
            "krabobot-voice",
            _MB_OK | _MB_ICONINFORMATION | _MB_SETFOREGROUND | _MB_TOPMOST,
        )
    except Exception:
        print(text, flush=True)
