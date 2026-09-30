"""Meeting recording: subprocess capture, local WAV persist, mic-tap for stop ASR."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
import wave
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from krabobot_voice.audio_io import (
    LoopbackStream,
    MicStream,
    mix_pcm16_average,
    require_sounddevice,
)


def default_meetings_dir() -> Path:
    """``%LOCALAPPDATA%/krabobot-voice/meetings`` (or ``~/…`` fallback)."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "krabobot-voice" / "meetings"


def new_meeting_wav_path(save_dir: str | Path | None = None) -> Path:
    """Return ``<save_dir>/YYYYMMDD-HHMMSS.wav`` (dir created)."""
    root = Path(save_dir) if save_dir else default_meetings_dir()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = root / f"{stamp}.wav"
    # Avoid clobbering if started twice in the same second.
    if path.exists():
        path = root / f"{stamp}-{os.getpid()}.wav"
    return path


@dataclass
class MeetingCaptureConfig:
    """Capture settings for a meeting session."""

    capture: str = "mix"  # mic | loopback | mix
    sample_rate: int = 16000
    block_ms: int = 30
    input_device: str = ""
    output_device: str = ""
    loopback_device: str = ""
    max_s: float = 7200.0


@dataclass
class MeetingSessionResult:
    """Stopped meeting payload ready for upload / local archive."""

    wav_path: Path
    duration_s: float
    capture: str
    frames: int = 0
    wav_bytes: bytes = b""  # optional in-memory copy (tests / legacy)


class _StopFlag(Protocol):
    def is_set(self) -> bool: ...


def _decode_mic_tap_item(item: Any, *, block: int) -> np.ndarray:
    """Normalize queue items (bytes or ndarray) to int16 mono of ``block`` samples."""
    if isinstance(item, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(item, dtype=np.int16).reshape(-1).copy()
    else:
        arr = np.asarray(item, dtype=np.int16).reshape(-1)
    if arr.size == block:
        return arr
    if arr.size > block:
        return arr[:block].copy()
    out = np.zeros(block, dtype=np.int16)
    if arr.size:
        out[: arr.size] = arr
    return out


class _MicTapReader:
    """Block source backed by MeetingRecorder mic-tap queue (VadSegmenter-compatible)."""

    def __init__(
        self,
        q: Any,
        *,
        sample_rate: int,
        block: int,
        active: Callable[[], bool],
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.block = max(1, int(block))
        self._q = q
        self._active = active
        self.last_rms = 0.0
        self.blocks_read = 0
        self.empty_reads = 0

    def read_block(self) -> np.ndarray:
        # Short timeout so meeting hotkey poll stays responsive, but avoid
        # flooding VAD with silence when the worker is only briefly late.
        idle_zeros = 0
        while True:
            try:
                item = self._q.get(timeout=0.05)
            except queue.Empty:
                if not self._active():
                    return np.zeros(self.block, dtype=np.int16)
                idle_zeros += 1
                if idle_zeros >= 4:  # ~200 ms with no tap → silence frame
                    self.empty_reads += 1
                    self.last_rms = 0.0
                    return np.zeros(self.block, dtype=np.int16)
                continue
            arr = _decode_mic_tap_item(item, block=self.block)
            self.blocks_read += 1
            # Cheap peak proxy (full RMS every block is fine at 30 ms).
            peak = float(np.max(np.abs(arr))) if arr.size else 0.0
            self.last_rms = peak / 32768.0
            return arr


def run_meeting_capture(
    cfg: MeetingCaptureConfig,
    wav_path: str | Path,
    *,
    stop: _StopFlag,
    mic_tap_queue: Any | None = None,
    on_progress: Callable[[int, float], None] | None = None,
    on_ready: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Record until ``stop.is_set()`` or ``max_s``; write mono PCM16 WAV to ``wav_path``.

    Returns a dict with frames / duration_s / capture / error (empty if ok).
    Mic tap blocks (for parent stop-phrase ASR) are pushed to ``mic_tap_queue`` when set.
    ``on_ready`` fires once capture streams are open (before the first frame);
    it may be called as ``on_ready()`` or ``on_ready({"warning": ..., "capture": ...})``.
    """
    mode = (cfg.capture or "mix").strip().lower()
    if mode not in {"mic", "loopback", "mix"}:
        mode = "mix"
    sample_rate = int(cfg.sample_rate)
    out = Path(wav_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    frames = 0
    started = time.monotonic()
    error = ""
    error_phase = ""
    actual_capture = mode
    warning = ""

    def budget_ok() -> bool:
        if cfg.max_s <= 0:
            return True
        return (time.monotonic() - started) < float(cfg.max_s)

    def push_mic(pcm: np.ndarray) -> None:
        if mic_tap_queue is None:
            return
        # Raw PCM16 bytes pickle cheaper/more reliably across spawn than ndarray.
        payload = np.asarray(pcm, dtype=np.int16).reshape(-1).tobytes()
        try:
            mic_tap_queue.put_nowait(payload)
        except Exception:
            try:
                mic_tap_queue.get_nowait()
            except Exception:
                pass
            try:
                mic_tap_queue.put_nowait(payload)
            except Exception:
                pass

    def write_pcm(wf: wave.Wave_write, pcm: np.ndarray) -> None:
        nonlocal frames
        arr = np.asarray(pcm, dtype=np.int16).reshape(-1)
        if arr.size == 0:
            return
        wf.writeframes(arr.tobytes())
        frames += int(arr.size)
        if on_progress is not None and frames % (sample_rate * 5) < arr.size:
            on_progress(frames, time.monotonic() - started)

    def signal_ready(status: dict[str, Any] | None = None) -> None:
        if on_ready is not None:
            try:
                if status:
                    on_ready(status)
                else:
                    on_ready()
            except TypeError:
                try:
                    on_ready()
                except Exception:
                    pass
            except Exception:
                pass

    def record_mic_only(wf: wave.Wave_write) -> None:
        require_sounddevice()
        with MicStream(
            sample_rate=sample_rate,
            block_ms=cfg.block_ms,
            device=cfg.input_device or None,
        ) as mic:
            signal_ready({"capture": "mic"})
            while not stop.is_set() and budget_ok():
                chunk = mic.read_block()
                write_pcm(wf, chunk)
                push_mic(chunk)

    try:
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            error_phase = "open"
            if mode == "mic":
                record_mic_only(wf)
            elif mode == "loopback":
                with LoopbackStream(
                    sample_rate=sample_rate,
                    block_ms=cfg.block_ms,
                    device=cfg.loopback_device or None,
                    output_device=cfg.output_device or None,
                ) as lb:
                    signal_ready({"capture": "loopback"})
                    error_phase = "record"
                    while not stop.is_set() and budget_ok():
                        write_pcm(wf, lb.read_block())
            else:
                # mix: open loopback *before* mic so exclusive mic open cannot
                # block WASAPI loopback on the same USB headset (-9996).
                require_sounddevice()
                lb_cm = LoopbackStream(
                    sample_rate=sample_rate,
                    block_ms=cfg.block_ms,
                    device=cfg.loopback_device or None,
                    output_device=cfg.output_device or None,
                )
                lb = None
                lb_err: Exception | None = None
                try:
                    lb = lb_cm.__enter__()
                except Exception as e:
                    lb_err = e
                try:
                    with MicStream(
                        sample_rate=sample_rate,
                        block_ms=cfg.block_ms,
                        device=cfg.input_device or None,
                    ) as mic:
                        if lb is None:
                            actual_capture = "mic"
                            warning = (
                                "loopback open failed — meeting degraded to mic-only: "
                                f"{lb_err}"
                            )
                            signal_ready(
                                {
                                    "capture": "mic",
                                    "warning": warning,
                                }
                            )
                            error_phase = "record"
                            while not stop.is_set() and budget_ok():
                                chunk = mic.read_block()
                                write_pcm(wf, chunk)
                                push_mic(chunk)
                        else:
                            signal_ready({"capture": "mix"})
                            error_phase = "record"
                            while not stop.is_set() and budget_ok():
                                a = mic.read_block()
                                b = lb.read_block()
                                write_pcm(wf, mix_pcm16_average(a, b))
                                push_mic(a)
                finally:
                    if lb is not None:
                        try:
                            lb_cm.__exit__(None, None, None)
                        except Exception:
                            pass
    except Exception as e:  # pragma: no cover - device failures
        error = str(e)
        if not error_phase:
            error_phase = "open"

    duration = max(0.0, time.monotonic() - started)
    if not error and frames <= 0:
        error = "Meeting recording is empty"
        if not error_phase:
            error_phase = "record"
    return {
        "frames": frames,
        "duration_s": duration,
        "capture": actual_capture,
        "error": error,
        "error_phase": error_phase if error else "",
        "warning": warning,
        "wav_path": str(out),
    }


def _process_main(
    cfg_dict: dict[str, Any],
    wav_path: str,
    stop_event: Any,
    mic_q: Any,
    result_q: Any,
    ready_event: Any | None = None,
    status_q: Any | None = None,
) -> None:
    """multiprocessing entry: must stay top-level for Windows spawn."""
    cfg = MeetingCaptureConfig(**cfg_dict)

    def _ready(status: dict[str, Any] | None = None) -> None:
        if status_q is not None and status:
            try:
                status_q.put_nowait(status)
            except Exception:
                pass
        if ready_event is not None:
            try:
                ready_event.set()
            except Exception:
                pass

    try:
        result = run_meeting_capture(
            cfg,
            wav_path,
            stop=stop_event,
            mic_tap_queue=mic_q,
            on_ready=_ready,
        )
    except KeyboardInterrupt:
        # Parent may Ctrl+C while we still hold PortAudio; treat as stop.
        try:
            stop_event.set()
        except Exception:
            pass
        result = {
            "frames": 0,
            "duration_s": 0.0,
            "capture": cfg.capture,
            "error": "",
            "error_phase": "",
            "warning": "",
            "wav_path": str(wav_path),
        }
    try:
        result_q.put(result)
    except Exception:
        pass


def default_meeting_use_process() -> bool:
    """Whether meeting capture should run in a spawned process.

    Portable/frozen builds default to an in-process thread: PyInstaller spawn +
    ``multiprocessing.Queue`` often fails to deliver mic-tap PCM to the parent,
    so voice stop («закончить запись») never reaches ASR. Dev (non-frozen) may
    still use a process for PortAudio crash isolation; override with
    ``KRABOBOT_VOICE_MEETING_PROCESS=0|1``.
    """
    env = (os.environ.get("KRABOBOT_VOICE_MEETING_PROCESS") or "").strip().lower()
    if env in {"0", "false", "no", "thread"}:
        return False
    if env in {"1", "true", "yes", "process"}:
        return True
    try:
        from krabobot_voice.config import is_frozen

        return not is_frozen()
    except Exception:
        return False


class MeetingRecorder:
    """Meeting capture in a worker (thread or process); parent reads mic-tap for stop ASR.

    Note (Windows): the worker owns mix/mic+loopback devices. Wake on the main
    capture stream may be unavailable while recording; stop via hotkey or the
    mic-tap side channel from this recorder. Frozen/portable defaults to a
    thread so stop-phrase ASR keeps working when loopback degrades to mic-only.
    """

    def __init__(
        self,
        cfg: MeetingCaptureConfig,
        *,
        save_dir: str | Path | None = None,
        use_process: bool | None = None,
    ) -> None:
        mode = (cfg.capture or "mix").strip().lower()
        if mode not in {"mic", "loopback", "mix"}:
            mode = "mix"
        self.cfg = MeetingCaptureConfig(
            capture=mode,
            sample_rate=int(cfg.sample_rate),
            block_ms=int(cfg.block_ms),
            input_device=str(cfg.input_device or ""),
            output_device=str(cfg.output_device or ""),
            loopback_device=str(cfg.loopback_device or ""),
            max_s=float(cfg.max_s),
        )
        self.save_dir = Path(save_dir) if save_dir else default_meetings_dir()
        self.use_process = (
            default_meeting_use_process() if use_process is None else bool(use_process)
        )
        self._lock = threading.Lock()
        self._stop_event: Any | None = None
        self._mic_q: Any | None = None
        self._result_q: Any | None = None
        self._ready_event: Any | None = None
        self._status_q: Any | None = None
        self._proc: mp.Process | None = None
        self._thread: threading.Thread | None = None
        self._wav_path: Path | None = None
        self._error: str = ""
        self._error_phase: str = ""
        self._warning: str = ""
        self._started_at = 0.0
        self._active = False
        self._ready = False
        self._result: dict[str, Any] | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            if not self._active:
                return False
            if self._proc is not None:
                return self._proc.is_alive()
            if self._thread is not None:
                return self._thread.is_alive()
            return self._active

    @property
    def capture_mode(self) -> str:
        return self.cfg.capture

    @property
    def wav_path(self) -> Path | None:
        return self._wav_path

    @property
    def ready(self) -> bool:
        with self._lock:
            if self._ready:
                return True
            ev = self._ready_event
        if ev is not None and ev.is_set():
            with self._lock:
                self._ready = True
            return True
        return False

    @property
    def warning(self) -> str:
        with self._lock:
            return self._warning

    def start(self) -> Path:
        with self._lock:
            if self._active:
                raise RuntimeError("Meeting already recording")
            wav_path = new_meeting_wav_path(self.save_dir)
            self._wav_path = wav_path
            self._error = ""
            self._error_phase = ""
            self._warning = ""
            self._result = None
            self._ready = False
            self._started_at = time.monotonic()
            self._active = True

            if self.use_process:
                ctx = mp.get_context("spawn")
                self._stop_event = ctx.Event()
                self._ready_event = ctx.Event()
                self._mic_q = ctx.Queue(maxsize=400)
                self._result_q = ctx.Queue(maxsize=1)
                self._status_q = ctx.Queue(maxsize=1)
                self._proc = ctx.Process(
                    target=_process_main,
                    args=(
                        asdict(self.cfg),
                        str(wav_path),
                        self._stop_event,
                        self._mic_q,
                        self._result_q,
                        self._ready_event,
                        self._status_q,
                    ),
                    name="meeting-capture",
                    daemon=True,
                )
                self._proc.start()
            else:
                self._stop_event = threading.Event()
                self._ready_event = threading.Event()
                self._mic_q = queue.Queue(maxsize=400)
                self._status_q = queue.Queue(maxsize=1)
                self._thread = threading.Thread(
                    target=self._run_thread,
                    name="meeting-capture",
                    daemon=True,
                )
                self._thread.start()

        # Block until streams open or worker dies — avoids "recording…" while open failed.
        try:
            self._wait_until_ready(timeout_s=8.0)
            self._consume_status()
        except Exception:
            self._abort_failed_start()
            raise
        return wav_path

    def _consume_status(self) -> None:
        q = self._status_q
        if q is None:
            return
        try:
            status = q.get_nowait()
        except Exception:
            return
        if not isinstance(status, dict):
            return
        warn = str(status.get("warning") or "")
        if warn:
            with self._lock:
                self._warning = warn

    def _wait_until_ready(self, *, timeout_s: float) -> None:
        deadline = time.monotonic() + max(0.5, float(timeout_s))
        while time.monotonic() < deadline:
            if self.ready:
                return
            if not self.active:
                err, phase = self._peek_worker_error()
                phase = phase or "open"
                raise RuntimeError(err or "Meeting worker failed to start")
            time.sleep(0.05)
        # Still alive but slow to signal — allow recording to continue.
        if not self.active:
            err, phase = self._peek_worker_error()
            raise RuntimeError(err or "Meeting worker failed to start")

    def _peek_worker_error(self) -> tuple[str, str]:
        result = self._result
        result_q = self._result_q
        if result is None and result_q is not None:
            try:
                result = result_q.get(timeout=0.2)
                with self._lock:
                    self._result = result
            except Exception:
                result = None
        if not result:
            with self._lock:
                return self._error, self._error_phase
        err = str(result.get("error") or "")
        phase = str(result.get("error_phase") or "")
        warning = str(result.get("warning") or "")
        with self._lock:
            if err:
                self._error = err
                self._error_phase = phase
            if warning:
                self._warning = warning
        return err, phase

    def _abort_failed_start(self) -> None:
        with self._lock:
            stop_event = self._stop_event
            proc = self._proc
            thread = self._thread
            result_q = self._result_q
        if stop_event is not None:
            try:
                stop_event.set()
            except Exception:
                pass
        if proc is not None and proc.is_alive():
            try:
                proc.terminate()
                proc.join(timeout=2.0)
            except Exception:
                pass
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=2.0)
            except Exception:
                pass
        if result_q is not None:
            try:
                result_q.get(timeout=0.2)
            except Exception:
                pass
        with self._lock:
            self._proc = None
            self._thread = None
            self._active = False
            self._ready = False
            self._stop_event = None
            self._ready_event = None
            self._mic_q = None
            self._result_q = None
            self._status_q = None

    def stop(self) -> MeetingSessionResult:
        with self._lock:
            if not self._active and self._proc is None and self._thread is None:
                raise RuntimeError("Meeting is not recording")
            wav_path = self._wav_path
            stop_event = self._stop_event
            proc = self._proc
            thread = self._thread
            result_q = self._result_q
            was_ready = self._ready or (
                self._ready_event is not None and self._ready_event.is_set()
            )
        if stop_event is not None:
            stop_event.set()

        # Ctrl+C during join/get must not hang the parent on a dead worker.
        interrupted = False
        try:
            if proc is not None:
                proc.join(timeout=30.0)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5.0)
            if thread is not None:
                thread.join(timeout=30.0)
        except KeyboardInterrupt:
            interrupted = True
            if proc is not None and proc.is_alive():
                try:
                    proc.terminate()
                    proc.join(timeout=2.0)
                except Exception:
                    pass

        result = self._result
        if result is None and result_q is not None:
            try:
                result = result_q.get(timeout=0.5 if interrupted else 2.0)
            except (Exception, KeyboardInterrupt):
                result = None
        if result is None:
            result = {}

        with self._lock:
            err = str(result.get("error") or self._error or "")
            phase = str(result.get("error_phase") or self._error_phase or "")
            warning = str(result.get("warning") or self._warning or "")
            capture = str(result.get("capture") or self.cfg.capture)
            frames = int(result.get("frames") or 0)
            duration = float(result.get("duration_s") or 0.0)
            if duration <= 0 and self._started_at:
                duration = max(0.0, time.monotonic() - self._started_at)
            path = Path(str(result.get("wav_path") or wav_path or ""))
            self._proc = None
            self._thread = None
            self._active = False
            self._ready = False
            self._stop_event = None
            self._ready_event = None
            self._mic_q = None
            self._result_q = None
            self._status_q = None
            self._result = None
            self._warning = warning
            self._error_phase = phase

        if err:
            if not phase and not was_ready:
                phase = "open"
            prefix = "open" if phase == "open" else "record"
            raise RuntimeError(f"[{prefix}] {err}")
        if wav_path is None or not path.is_file():
            raise RuntimeError("Meeting recording is empty")
        if frames <= 0:
            # File exists but no frames reported — still allow if non-trivial size.
            try:
                if path.stat().st_size < 64:
                    raise RuntimeError("Meeting recording is empty")
            except OSError as e:
                raise RuntimeError("Meeting recording is empty") from e

        wav_bytes = b""
        try:
            # Small recordings only; long meetings stay on disk.
            if path.stat().st_size <= 2_000_000:
                wav_bytes = path.read_bytes()
        except OSError:
            wav_bytes = b""

        return MeetingSessionResult(
            wav_path=path,
            wav_bytes=wav_bytes,
            duration_s=duration,
            capture=capture,
            frames=frames,
        )

    def mic_tap_reader(self) -> _MicTapReader:
        """Adapter: blocking int16 mono blocks from the mic tap (for VadSegmenter)."""
        block = max(1, int(self.cfg.sample_rate * self.cfg.block_ms / 1000))
        q = self._mic_q
        if q is None:
            q = queue.Queue()
        return _MicTapReader(
            q,
            sample_rate=self.cfg.sample_rate,
            block=block,
            active=lambda: self.active,
        )

    def recent_mic_pcm(self, seconds: float = 2.5) -> np.ndarray | None:
        """Best-effort: drain a short window from the tap queue (may be sparse)."""
        q = self._mic_q
        if q is None:
            return None
        block = max(1, int(self.cfg.sample_rate * self.cfg.block_ms / 1000))
        chunks: list[np.ndarray] = []
        need = max(1, int(float(seconds) * self.cfg.sample_rate))
        got = 0
        while got < need:
            try:
                item = q.get_nowait()
            except Exception:
                break
            arr = _decode_mic_tap_item(item, block=block)
            chunks.append(arr)
            got += int(arr.size)
        if not chunks:
            return None
        pcm = np.concatenate(chunks)
        if pcm.size < self.cfg.sample_rate // 4:
            return None
        return pcm[-need:]

    def _run_thread(self) -> None:
        assert self._stop_event is not None and self._wav_path is not None
        ready_event = self._ready_event
        status_q = self._status_q

        def _ready(status: dict[str, Any] | None = None) -> None:
            if status_q is not None and status:
                try:
                    status_q.put_nowait(status)
                except Exception:
                    pass
            if ready_event is not None:
                try:
                    ready_event.set()
                except Exception:
                    pass

        result = run_meeting_capture(
            self.cfg,
            self._wav_path,
            stop=self._stop_event,
            mic_tap_queue=self._mic_q,
            on_ready=_ready,
        )
        with self._lock:
            self._result = result
            if result.get("error"):
                self._error = str(result["error"])
                self._error_phase = str(result.get("error_phase") or "")
            if result.get("warning"):
                self._warning = str(result["warning"])
            self._active = False
