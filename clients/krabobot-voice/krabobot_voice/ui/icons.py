"""App icon assets (window title / taskbar / system tray)."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

# Distinct from python.exe so Windows taskbar does not show the Python logo.
_APP_USER_MODEL_ID = "krabobot.voice"


def _assets_dir() -> Path:
    """``krabobot_voice/ui/assets`` — works for ``python -m`` and PyInstaller."""
    here = Path(__file__).resolve().parent / "assets"
    if here.is_dir():
        return here
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        alt = Path(meipass) / "krabobot_voice" / "ui" / "assets"
        if alt.is_dir():
            return alt
    return here


def asset_path(name: str) -> Path:
    """Path to a bundled UI asset (may not exist yet — callers check)."""
    return _assets_dir() / name


def favicon_png() -> Path:
    return asset_path("favicon.png")


def favicon_ico() -> Path:
    return asset_path("favicon.ico")


def set_app_user_model_id(app_id: str = _APP_USER_MODEL_ID) -> None:
    """Tell Windows this process is not generic python.exe (taskbar grouping).

    Call before creating the first top-level window. No-op on non-Windows.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(str(app_id))
    except Exception:
        pass


def _win32_set_taskbar_icon(root: Any, ico: Path) -> None:
    """Apply .ico via WM_SETICON so the taskbar uses crab, not python.exe."""
    import ctypes
    from ctypes import wintypes

    try:
        root.update_idletasks()
    except Exception:
        pass

    try:
        hwnd = int(root.winfo_id())
    except Exception:
        return
    if not hwnd:
        return

    user32 = ctypes.windll.user32
    # Tk client HWND → outer frame (what the taskbar associates with).
    GA_ROOT = 2
    try:
        ancestor = int(user32.GetAncestor(hwnd, GA_ROOT) or 0)
        if ancestor:
            hwnd = ancestor
    except Exception:
        try:
            parent = int(user32.GetParent(hwnd) or 0)
            if parent:
                hwnd = parent
        except Exception:
            pass

    IMAGE_ICON = 1
    LR_LOADFROMFILE = 0x0010
    WM_SETICON = 0x0080
    ICON_SMALL = 0
    ICON_BIG = 1

    LoadImage = user32.LoadImageW
    LoadImage.argtypes = [
        wintypes.HINSTANCE,
        wintypes.LPCWSTR,
        wintypes.UINT,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
    ]
    LoadImage.restype = wintypes.HANDLE

    ico_abs = str(ico.resolve())
    h_small = LoadImage(None, ico_abs, IMAGE_ICON, 16, 16, LR_LOADFROMFILE)
    h_big = LoadImage(None, ico_abs, IMAGE_ICON, 32, 32, LR_LOADFROMFILE)
    if not h_small and not h_big:
        # Fall back to default size from file.
        h_big = LoadImage(None, ico_abs, IMAGE_ICON, 0, 0, LR_LOADFROMFILE)
        h_small = h_big
    if not h_small and not h_big:
        return

    SendMessage = user32.SendMessageW
    if h_small:
        SendMessage(hwnd, WM_SETICON, ICON_SMALL, h_small)
    if h_big:
        SendMessage(hwnd, WM_SETICON, ICON_BIG, h_big)

    # Keep HICON alive for the window lifetime (do not DestroyIcon).
    root._krabobot_hicons = (h_small, h_big)  # noqa: SLF001


@lru_cache(maxsize=1)
def load_tray_image() -> Any:
    """PIL Image for pystray (RGBA). Raises if Pillow/PNG missing."""
    from PIL import Image

    path = favicon_png()
    if not path.is_file():
        raise FileNotFoundError(f"tray icon not found: {path}")
    return Image.open(path).convert("RGBA")


def apply_window_icon(root: Any) -> None:
    """Set Tk/CTk window icon for title bar and Windows taskbar.

    Prefer ``iconbitmap`` (.ico) on Windows for reliable taskbar; also set
    ``iconphoto`` from PNG. On Windows, additionally set AppUserModelID and
    ``WM_SETICON`` so the taskbar is not tied to ``python.exe``. Keep PhotoImage
    / HICON refs on ``root`` to avoid GC.
    """
    ico = favicon_ico()
    png = favicon_png()
    if sys.platform == "win32":
        set_app_user_model_id()
        if ico.is_file():
            ico_abs = str(ico.resolve())
            try:
                root.iconbitmap(default=ico_abs)
            except Exception:
                try:
                    root.iconbitmap(ico_abs)
                except Exception:
                    pass
            try:
                _win32_set_taskbar_icon(root, ico)
            except Exception:
                pass
    if not png.is_file():
        return
    try:
        from PIL import Image, ImageTk
    except ImportError:
        try:
            photo = __import__("tkinter").PhotoImage(file=str(png))
        except Exception:
            return
        else:
            root._krabobot_icon_photo = photo  # noqa: SLF001 — keep alive
            try:
                root.iconphoto(True, photo)
            except Exception:
                pass
            return
    try:
        im = Image.open(png).convert("RGBA")
        # Windows taskbar often looks better at 32/64; keep original if small.
        photo = ImageTk.PhotoImage(im, master=root)
        root._krabobot_icon_photo = photo  # noqa: SLF001 — keep alive
        root.iconphoto(True, photo)
    except Exception:
        pass
