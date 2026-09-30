"""Voice-loop controller: worker thread + restart/stop for the UI."""

from __future__ import annotations

import queue
import threading
from pathlib import Path

from krabobot_voice.config import VoiceClientConfig, resolve_config_file_path
from krabobot_voice.status_bus import StatusBus, set_status_bus


class VoiceController:
    """Owns the voice ``run_loop`` on a daemon thread; UI posts commands here."""

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        status_bus: StatusBus | None = None,
    ) -> None:
        self.config_path = Path(config_path).expanduser() if config_path else None
        self.status_bus = status_bus or StatusBus()
        self.command_queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._restart = threading.Event()
        self._quit = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.last_exit_code = 0

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._quit.clear()
            self._stop.clear()
            self._restart.clear()
            set_status_bus(self.status_bus)
            self._thread = threading.Thread(
                target=self._voice_main,
                name="krabobot-voice-loop",
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, join_timeout_s: float = 8.0) -> None:
        self._quit.set()
        self._stop.set()
        self.post("quit")
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=join_timeout_s)
        set_status_bus(None)

    def restart_loop(self) -> None:
        """Save already done by UI — stop current loop and reload config."""
        self._restart.set()
        self._stop.set()
        self.post("quit")

    def post(self, command: str) -> None:
        cmd = (command or "").strip().lower()
        if not cmd:
            return
        try:
            self.command_queue.put_nowait(cmd)
        except queue.Full:
            pass

    def meeting_toggle(self) -> None:
        self.post("meeting")

    def ptt_listen(self) -> None:
        """Arm listen mode (same as PTT hotkey edge without hold-to-talk)."""
        self.post("ptt")

    def _resolve_path(self) -> Path | None:
        if self.config_path is not None:
            return self.config_path
        return resolve_config_file_path(None)

    def _voice_main(self) -> None:
        from krabobot_voice.app import run_loop

        while not self._quit.is_set():
            self._stop.clear()
            self._restart.clear()
            path = self._resolve_path()
            try:
                cfg = VoiceClientConfig.load(path)
            except Exception as e:  # noqa: BLE001
                self.status_bus.emit_log(f"ERROR: config load: {e}")
                self.last_exit_code = 2
                break
            code = run_loop(
                cfg,
                stop_event=self._stop,
                command_queue=self.command_queue,
                status_bus=self.status_bus,
                config_path=path,
            )
            self.last_exit_code = int(code or 0)
            if self._quit.is_set():
                break
            if self._restart.is_set():
                self.status_bus.emit_log("config applied — restarting voice loop…")
                # Drain stale commands from the previous loop.
                while True:
                    try:
                        self.command_queue.get_nowait()
                    except queue.Empty:
                        break
                continue
            break
