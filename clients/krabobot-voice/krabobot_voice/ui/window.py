"""Main status / config window (customtkinter with tkinter fallback)."""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext
from typing import Any, Callable

from krabobot_voice.status_bus import StatusBus, display_mode
from krabobot_voice.ui.controller import VoiceController

try:
    import customtkinter as ctk

    _HAS_CTK = True
except ImportError:  # pragma: no cover - optional UI dep
    ctk = None  # type: ignore[assignment]
    _HAS_CTK = False


def _open_path(path: Path) -> None:
    path = Path(path)
    if path.is_dir():
        target = path
    else:
        target = path if path.is_file() else path.parent
        if not target.exists() and path.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
            target = path.parent
        elif not target.exists():
            path.mkdir(parents=True, exist_ok=True)
            target = path
    if sys.platform == "win32":
        os.startfile(str(target))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])


def _open_config_file(path: Path) -> None:
    path = Path(path)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            path.write_text("# krabobot-voice config\n", encoding="utf-8")
    if sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class VoiceWindow:
    """Tk / CustomTkinter shell: status, log, meeting/PTT, config apply."""

    def __init__(
        self,
        controller: VoiceController,
        *,
        on_quit: Callable[[], None] | None = None,
        start_minimized: bool = False,
    ) -> None:
        self.controller = controller
        self.bus: StatusBus = controller.status_bus
        self._on_quit = on_quit
        self._log_seen = 0
        self._closed = False
        self._log_seeded = False

        if _HAS_CTK:
            ctk.set_appearance_mode("System")
            ctk.set_default_color_theme("blue")
            self.root = ctk.CTk()
        else:
            self.root = tk.Tk()
        self.root.title("krabobot-voice")
        self.root.geometry("720x560")
        self.root.minsize(520, 420)
        self.root.protocol("WM_DELETE_WINDOW", self.hide)
        # Withdraw before map to avoid a brief flash when starting hidden to tray.
        if start_minimized:
            try:
                self.root.withdraw()
            except Exception:
                pass
        try:
            from krabobot_voice.ui.icons import apply_window_icon

            apply_window_icon(self.root)
            # Re-apply after map: first WM_SETICON can race with Tk frame creation.
            self.root.after(50, lambda: apply_window_icon(self.root))
        except Exception:
            pass

        self._build()
        self._load_settings_fields()
        self.root.after(200, self._tick)

    def _frame(self, parent: Any, **kwargs: Any) -> Any:
        if _HAS_CTK:
            return ctk.CTkFrame(parent, **kwargs)
        return tk.Frame(parent, **kwargs)

    def _label(self, parent: Any, text: str = "", **kwargs: Any) -> Any:
        if _HAS_CTK:
            return ctk.CTkLabel(parent, text=text, **kwargs)
        return tk.Label(parent, text=text, **kwargs)

    def _button(self, parent: Any, text: str, command: Callable[[], None], **kwargs: Any) -> Any:
        if _HAS_CTK:
            return ctk.CTkButton(parent, text=text, command=command, **kwargs)
        return tk.Button(parent, text=text, command=command, **kwargs)

    def _entry(self, parent: Any, **kwargs: Any) -> Any:
        if _HAS_CTK:
            return ctk.CTkEntry(parent, **kwargs)
        return tk.Entry(parent, **kwargs)

    def _build(self) -> None:
        top = self._frame(self.root)
        top.pack(fill="x", padx=12, pady=8)

        self.mode_label = self._label(top, text="mode: idle", anchor="w")
        self.mode_label.pack(fill="x")
        self.status_label = self._label(top, text="status: —", anchor="w")
        self.status_label.pack(fill="x")

        btns = self._frame(self.root)
        btns.pack(fill="x", padx=12, pady=4)
        self._button(btns, "Meeting start/stop", self.controller.meeting_toggle, width=140).pack(
            side="left", padx=4
        )
        self._button(btns, "PTT (listening)", self.controller.ptt_listen, width=120).pack(
            side="left", padx=4
        )
        self._button(btns, "Open config", self._open_config, width=100).pack(side="left", padx=4)
        self._button(btns, "Meetings folder", self._open_meetings, width=120).pack(
            side="left", padx=4
        )
        self._button(btns, "Quit", self.quit_app, width=80).pack(side="right", padx=4)

        settings = self._frame(self.root)
        settings.pack(fill="x", padx=12, pady=6)
        self._label(settings, text="Settings (Apply = save YAML + restart voice loop)").pack(
            anchor="w"
        )

        grid = self._frame(settings)
        grid.pack(fill="x", pady=4)
        self._fields: dict[str, Any] = {}
        rows = [
            ("base_url", "base_url"),
            ("device_id", "device_id"),
            ("wake.phrase", "wake.phrase"),
            ("token", "token"),
            ("meeting.save_dir", "meeting.save_dir (empty=<app>/meetings)"),
            ("ptt.hotkey", "ptt.hotkey"),
            ("meeting.hotkey", "meeting.hotkey"),
            ("audio.listen_source", "audio.listen_source"),
        ]
        for i, (key, label) in enumerate(rows):
            self._label(grid, text=label).grid(row=i, column=0, sticky="w", padx=2, pady=2)
            entry = self._entry(grid, width=48)
            entry.grid(row=i, column=1, sticky="ew", padx=2, pady=2)
            self._fields[key] = entry
        grid.columnconfigure(1, weight=1)

        self._start_minimized_var = tk.BooleanVar(value=False)
        if _HAS_CTK:
            ctk.CTkCheckBox(
                settings,
                text="Start minimized to tray",
                variable=self._start_minimized_var,
            ).pack(anchor="w", pady=2)
        else:
            tk.Checkbutton(
                settings,
                text="Start minimized to tray",
                variable=self._start_minimized_var,
            ).pack(anchor="w")

        self._button(settings, "Apply & restart", self._apply_settings, width=140).pack(
            anchor="w", pady=4
        )

        self._label(self.root, text="Log").pack(anchor="w", padx=12)
        self.log = scrolledtext.ScrolledText(self.root, height=14, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def _set_label(self, widget: Any, text: str) -> None:
        try:
            widget.configure(text=text)
        except Exception:
            pass

    def _entry_get(self, key: str) -> str:
        w = self._fields[key]
        try:
            return str(w.get()).strip()
        except Exception:
            return ""

    def _entry_set(self, key: str, value: str) -> None:
        w = self._fields[key]
        try:
            if _HAS_CTK:
                w.delete(0, "end")
                w.insert(0, value)
            else:
                w.delete(0, tk.END)
                w.insert(0, value)
        except Exception:
            pass

    def _load_settings_fields(self) -> None:
        from krabobot_voice.config import VoiceClientConfig

        path = self.controller._resolve_path()
        try:
            cfg = VoiceClientConfig.load(path)
        except Exception:
            return
        self._entry_set("base_url", cfg.base_url)
        self._entry_set("device_id", cfg.device_id)
        phrase = cfg.wake_phrases[0] if cfg.wake_phrases else ""
        self._entry_set("wake.phrase", phrase)
        self._entry_set("token", cfg.token)
        self._entry_set("meeting.save_dir", cfg.meeting_save_dir)
        self._entry_set("ptt.hotkey", cfg.ptt_hotkey)
        self._entry_set("meeting.hotkey", cfg.meeting_hotkey)
        self._entry_set("audio.listen_source", cfg.audio_listen_source)
        try:
            self._start_minimized_var.set(bool(cfg.ui_start_minimized))
        except Exception:
            pass

    def _apply_settings(self) -> None:
        from krabobot_voice.config import (
            ensure_app_config,
            merge_config_yaml,
            resolve_config_file_path,
        )

        path = self.controller._resolve_path()
        if path is None:
            ensured = ensure_app_config()
            path = ensured or resolve_config_file_path(None)
        if path is None:
            messagebox.showerror("krabobot-voice", "Не найден config.yaml")
            return
        updates: dict[str, Any] = {
            "base_url": self._entry_get("base_url"),
            "device_id": self._entry_get("device_id"),
            "token": self._entry_get("token"),
            "wake": {"phrase": self._entry_get("wake.phrase")},
            "ptt": {"hotkey": self._entry_get("ptt.hotkey")},
            "meeting": {
                "hotkey": self._entry_get("meeting.hotkey"),
                "save_dir": self._entry_get("meeting.save_dir"),
            },
            "audio": {"listen_source": self._entry_get("audio.listen_source") or "mic"},
            "ui": {"start_minimized": bool(self._start_minimized_var.get())},
        }
        try:
            merge_config_yaml(path, updates)
        except Exception as e:
            messagebox.showerror("krabobot-voice", f"Не удалось сохранить конфиг:\n{e}")
            return
        self.controller.config_path = Path(path)
        self.bus.emit_log(f"config saved → {path}")
        self.controller.restart_loop()
        messagebox.showinfo("krabobot-voice", "Конфиг сохранён, voice loop перезапускается.")

    def _open_config(self) -> None:
        snap = self.bus.snapshot()
        path = snap.config_path or str(self.controller._resolve_path() or "")
        if not path:
            messagebox.showwarning("krabobot-voice", "Путь к config.yaml неизвестен")
            return
        _open_config_file(Path(path))

    def _open_meetings(self) -> None:
        snap = self.bus.snapshot()
        meetings = snap.meetings_dir
        if not meetings:
            from krabobot_voice.meeting import default_meetings_dir

            meetings = str(default_meetings_dir())
        root = Path(meetings)
        root.mkdir(parents=True, exist_ok=True)
        _open_path(root)

    def _append_log(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _tick(self) -> None:
        if self._closed:
            return
        snap = self.bus.snapshot()
        mode = display_mode(snap)
        self._set_label(self.mode_label, f"mode: {mode}")
        self._set_label(self.status_label, f"status: {snap.status or '—'}")
        events = self.bus.drain_events(max_n=300)
        if not self._log_seeded:
            self._log_seeded = True
            if not any(e.kind == "log" for e in events):
                for line in snap.log_lines:
                    self._append_log(line)
        for ev in events:
            if ev.kind == "log" and isinstance(ev.payload, str):
                self._append_log(ev.payload)
        self.root.after(250, self._tick)

    def show(self) -> None:
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
        except Exception:
            pass

    def hide(self) -> None:
        try:
            self.root.withdraw()
        except Exception:
            pass

    def quit_app(self) -> None:
        if self._closed:
            return
        self._closed = True
        cb = self._on_quit
        self._on_quit = None
        if cb is not None:
            try:
                cb()
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self) -> None:
        self.root.mainloop()
