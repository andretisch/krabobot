"""System tray icon (pystray) for krabobot-voice."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from krabobot_voice.status_bus import StatusBus, display_mode
from krabobot_voice.ui.icons import load_tray_image


def _make_icon_image() -> Any:
    """Branded crab favicon; fall back to a simple circle if assets missing."""
    try:
        return load_tray_image()
    except Exception:
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse((4, 4, 60, 60), fill=(30, 120, 200, 255))
        return img


class TrayApp:
    """Background tray; callbacks run on the tray thread — marshal to UI as needed."""

    def __init__(
        self,
        bus: StatusBus,
        *,
        on_show: Callable[[], None],
        on_meeting: Callable[[], None],
        on_ptt: Callable[[], None],
        on_open_config: Callable[[], None],
        on_open_meetings: Callable[[], None],
        on_quit: Callable[[], None],
    ) -> None:
        self.bus = bus
        self._on_show = on_show
        self._on_meeting = on_meeting
        self._on_ptt = on_ptt
        self._on_open_config = on_open_config
        self._on_open_meetings = on_open_meetings
        self._on_quit = on_quit
        self._icon: Any = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import pystray
            from pystray import MenuItem as Item
        except ImportError as e:
            raise RuntimeError(
                'pystray required for tray UI — pip install -e ".\\clients\\krabobot-voice[ui]"'
            ) from e

        menu = pystray.Menu(
            Item("Show", lambda: self._on_show()),
            Item("Meeting start/stop", lambda: self._on_meeting()),
            Item("PTT (listening)", lambda: self._on_ptt()),
            Item("Open config", lambda: self._on_open_config()),
            Item("Meetings folder", lambda: self._on_open_meetings()),
            Item("Quit", lambda: self._quit()),
        )
        self._icon = pystray.Icon(
            "krabobot-voice",
            _make_icon_image(),
            "krabobot-voice",
            menu,
        )
        self._thread = threading.Thread(target=self._run, name="krabobot-tray", daemon=True)
        self._thread.start()
        self._schedule_refresh()

    def _run(self) -> None:
        if self._icon is not None:
            self._icon.run()

    def _quit(self) -> None:
        self.stop()
        self._on_quit()

    def stop(self) -> None:
        icon = self._icon
        self._icon = None
        if icon is not None:
            try:
                icon.stop()
            except Exception:
                pass

    def _schedule_refresh(self) -> None:
        def _loop() -> None:
            while self._icon is not None:
                try:
                    self._refresh_title()
                except Exception:
                    pass
                import time

                time.sleep(0.5)

        threading.Thread(target=_loop, name="krabobot-tray-status", daemon=True).start()

    def _refresh_title(self) -> None:
        icon = self._icon
        if icon is None:
            return
        snap = self.bus.snapshot()
        mode = display_mode(snap)
        title = f"krabobot-voice — {mode}"
        if snap.status:
            title = f"{title}: {snap.status[:60]}"
        try:
            icon.title = title
        except Exception:
            pass
