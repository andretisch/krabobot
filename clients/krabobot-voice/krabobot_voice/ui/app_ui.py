"""Entry: tray + window + voice loop on a worker thread (Architecture A)."""

from __future__ import annotations

import threading
from pathlib import Path

from krabobot_voice.config import VoiceClientConfig, ensure_app_config, resolve_config_file_path
from krabobot_voice.status_bus import StatusBus, set_status_bus
from krabobot_voice.ui.controller import VoiceController
from krabobot_voice.ui.tray import TrayApp
from krabobot_voice.ui.window import VoiceWindow


def run_ui(
    config: VoiceClientConfig | None = None,
    *,
    config_path: str | Path | None = None,
    start_minimized: bool = False,
) -> int:
    """Start UI on the main thread; voice ``run_loop`` on a daemon worker."""
    _ = config  # loaded again inside the worker so Apply/restart always re-reads YAML
    # Before any top-level window: distinct AppUserModelID so Windows taskbar
    # does not inherit the python.exe icon when running ``python -m``.
    try:
        from krabobot_voice.ui.icons import set_app_user_model_id

        set_app_user_model_id()
    except Exception:
        pass

    path: Path | None
    if config_path is not None:
        path = Path(config_path).expanduser()
    else:
        ensured = ensure_app_config()
        path = ensured or resolve_config_file_path(None)

    bus = StatusBus()
    set_status_bus(bus)
    controller = VoiceController(config_path=path, status_bus=bus)

    quit_flag = threading.Event()
    window_holder: dict[str, VoiceWindow | None] = {"w": None}
    tray_holder: dict[str, TrayApp | None] = {"t": None}

    def do_quit() -> None:
        if quit_flag.is_set():
            return
        quit_flag.set()
        t = tray_holder["t"]
        tray_holder["t"] = None
        if t is not None:
            t.stop()
        controller.stop()
        w = window_holder["w"]
        if w is not None and not w._closed:
            try:
                w.root.after(0, w.quit_app)
            except Exception:
                pass

    def show_window() -> None:
        w = window_holder["w"]
        if w is not None:
            try:
                w.root.after(0, w.show)
            except Exception:
                pass

    controller.start()

    window = VoiceWindow(
        controller,
        on_quit=do_quit,
        start_minimized=start_minimized,
    )
    window_holder["w"] = window

    tray_ok = False
    try:
        tray = TrayApp(
            bus,
            on_show=show_window,
            on_meeting=controller.meeting_toggle,
            on_ptt=controller.ptt_listen,
            on_open_config=lambda: window.root.after(0, window._open_config),
            on_open_meetings=lambda: window.root.after(0, window._open_meetings),
            on_quit=do_quit,
        )
        tray.start()
        tray_holder["t"] = tray
        tray_ok = True
    except Exception as e:
        bus.emit_log(f"WARNING: tray unavailable ({e}) — window only")

    # Hide to tray only when tray is available; otherwise keep the window visible.
    if start_minimized and tray_ok:
        window.hide()
    else:
        window.show()
    window.run()

    if not quit_flag.is_set():
        controller.stop()
        t = tray_holder["t"]
        if t is not None:
            t.stop()
    set_status_bus(None)
    return int(controller.last_exit_code or 0)
