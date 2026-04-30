"""Helpers for locating ffmpeg in a cross-platform way."""

from __future__ import annotations

import shutil


def resolve_ffmpeg_exe() -> str | None:
    """Return ffmpeg executable path from bundled package or system PATH."""
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe:
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")
