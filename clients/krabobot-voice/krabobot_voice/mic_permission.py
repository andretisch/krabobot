"""Windows microphone access request (WinRT) for portable / unpackaged exe.

Primary path: AppCapability.RequestAccessAsync + MediaCapture.InitializeAsync
(system consent UI when Windows can show it). Settings page is last resort via
a single-button MessageBox when access is Denied / NotDeclared.

WinRT is loaded via importlib (not static imports) so PyInstaller Analysis does
not pull winrt into the same isolated process as onnxruntime (AV crash).

**Runtime order:** callers must import ``onnxruntime`` *before* this module
loads any ``winrt.*`` extension (see ``silero_vad.preload_onnxruntime``).
WinRT-then-onnxruntime → ACCESS_VIOLATION / silent process exit on Windows.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from dataclasses import dataclass
from typing import Any, Literal


MicAccessStatus = Literal["allowed", "denied", "unavailable", "skipped"]


@dataclass(frozen=True)
class MicAccessResult:
    status: MicAccessStatus
    detail: str = ""
    opened_settings: bool = False


def open_windows_mic_privacy_settings() -> bool:
    """Open ms-settings:privacy-microphone (last-resort UX)."""
    if sys.platform != "win32":
        return False
    try:
        import os

        os.startfile("ms-settings:privacy-microphone")  # noqa: S606
        return True
    except Exception:
        try:
            import subprocess

            subprocess.Popen(  # noqa: S603
                ["cmd", "/c", "start", "", "ms-settings:privacy-microphone"],
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False


def _message_box(text: str, *, title: str = "krabobot-voice") -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(  # type: ignore[attr-defined]
            0,
            text,
            title,
            0x00000030 | 0x00000000,  # MB_ICONWARNING | MB_OK
        )
    except Exception:
        pass


def show_mic_access_dialog(text: str, *, title: str = "krabobot-voice") -> None:
    """Native MessageBox (Windows); no-op elsewhere."""
    _message_box(text, title=title)


def _status_name(value: object) -> str:
    try:
        return str(value).split(".")[-1]
    except Exception:
        return repr(value)


def _winrt_import(module: str) -> Any:
    return importlib.import_module(module)


def _ensure_com_sta() -> None:
    """WinRT capture/consent expects an STA apartment on Win32."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        # COINIT_APARTMENTTHREADED = 0x2; ignore RPC_E_CHANGED_MODE
        ctypes.windll.ole32.CoInitializeEx(None, 0x2)  # type: ignore[attr-defined]
    except Exception:
        pass


async def _request_access_async() -> tuple[str, str]:
    """Return (bucket, detail) where bucket is allowed|denied|prompted|error."""
    parts: list[str] = []
    _ensure_com_sta()

    # 1) AppCapability.RequestAccessAsync — shows consent when UserPromptRequired.
    try:
        aca = _winrt_import(
            "winrt.windows.security.authorization.appcapabilityaccess"
        )
        AppCapability = aca.AppCapability
        AppCapabilityAccessStatus = aca.AppCapabilityAccessStatus
        cap = AppCapability.create("Microphone")
        before = cap.check_access()
        status = await asyncio.wait_for(cap.request_access_async(), timeout=6.0)
        parts.append(
            f"AppCapability check={_status_name(before)} "
            f"request={_status_name(status)}"
        )
        if status == AppCapabilityAccessStatus.ALLOWED:
            return "allowed", "; ".join(parts)
    except TimeoutError:
        parts.append("AppCapability request timed out")
    except Exception as e:
        parts.append(f"AppCapability unavailable: {e}")

    # 2) MediaCapture.InitializeAsync — MS docs: launches consent prompt.
    try:
        mc_mod = _winrt_import("winrt.windows.media.capture")
        MediaCapture = mc_mod.MediaCapture
        mc = MediaCapture()
        await asyncio.wait_for(mc.initialize_async(), timeout=8.0)
        try:
            mc.close()
        except Exception:
            pass
        parts.append("MediaCapture.InitializeAsync OK")
        return "allowed", "; ".join(parts)
    except TimeoutError:
        parts.append("MediaCapture timed out")
        # Do not block portable startup — PortAudio open is the real gate.
        return "allowed", "; ".join(parts)
    except Exception as e:
        parts.append(f"MediaCapture: {e}")
        msg = str(e).lower()
        denied_hints = ("access", "denied", "privacy", "0x80070005", "e_accessdenied")
        if any(h in msg for h in denied_hints):
            return "denied", "; ".join(parts)
        # Unknown WinRT error: continue; MicStream failure still has a dialog.
        return "allowed", "; ".join(parts)


def ensure_microphone_access(
    *,
    interactive: bool = True,
    force: bool = False,  # noqa: ARG001 — reserved for callers / tests
) -> MicAccessResult:
    """Proactively request microphone access (WinRT) before PortAudio open.

    On non-Windows returns ``skipped``. Idempotent when already Allowed.
    WinRT timeouts do not block startup (PortAudio open remains the gate).
    """
    if sys.platform != "win32":
        return MicAccessResult("skipped", "not Windows")

    try:
        bucket, detail = asyncio.run(_request_access_async())
    except Exception as e:
        # Prefer continuing to device open over a hard deny on WinRT glue issues.
        return MicAccessResult("allowed", f"WinRT skipped: {e}")

    if bucket == "allowed":
        return MicAccessResult("allowed", detail)

    opened = False
    if interactive:
        show_mic_access_dialog(
            "Krabobot Voice нужен доступ к микрофону.\n\n"
            "Нажмите OK — откроются параметры Windows «Микрофон».\n"
            "Включите доступ для этого приложения и запустите снова.\n\n"
            f"({detail})"
        )
        opened = open_windows_mic_privacy_settings()
    return MicAccessResult("denied", detail, opened_settings=opened)
