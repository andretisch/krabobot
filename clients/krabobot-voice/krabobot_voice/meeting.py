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

    def read_block(self) -> np.ndarray:
        while True:
            try:
                return self._q.get(timeout=0.35)
            except queue.Empty:
                if not self._active():
                    return np.zeros(self.block, dtype=np.int16)
                return np.zeros(self.block, dtype=np.int16)


def run_meeting_capture(
    cfg: MeetingCaptureConfig,
    wav_path: str | Path,
    *,
    stop: _StopFlag,
    mic_tap_queue: Any | None = None,
    on_progress: Callable[[int, float], None] | None = None,
) -> dict[str, Any]:
    """Record until ``stop.is_set()`` or ``max_s``; write mono PCM16 WAV to ``wav_path``.

    Returns a dict with frames / duration_s / capture / error (empty if ok).
    Mic tap blocks (for parent stop-phrase ASR) are pushed to ``mic_tap_queue`` when set.
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

    def budget_ok() -> bool:
        if cfg.max_s <= 0:
            return True
        return (time.monotonic() - started) < float(cfg.max_s)

    def push_mic(pcm: np.ndarray) -> None:
        if mic_tap_queue is None:
            return
        arr = np.asarray(pcm, dtype=np.int16).reshape(-1).copy()
        try:
            mic_tap_queue.put_nowait(arr)
        except Exception:
            try:
                mic_tap_queue.get_nowait()
            except Exception:
                pass
            try:
                mic_tap_queue.put_nowait(arr)
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

    try:
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            if mode == "mic":
                require_sounddevice()
                with MicStream(
                    sample_rate=sample_rate,
                    block_ms=cfg.block_ms,
                    device=cfg.input_device or None,
                ) as mic:
                    while not stop.is_set() and budget_ok():
                        chunk = mic.read_block()
                        write_pcm(wf, chunk)
                        push_mic(chunk)
            elif mode == "loopback":
                with LoopbackStream(
                    sample_rate=sample_rate,
                    block_ms=cfg.block_ms,
                    device=cfg.loopback_device or None,
                    output_device=cfg.output_device or None,
                ) as lb:
                    while not stop.is_set() and budget_ok():
                        write_pcm(wf, lb.read_block())
            else:
                require_sounddevice()
                with (
                    MicStream(
                        sample_rate=sample_rate,
                        block_ms=cfg.block_ms,
                        device=cfg.input_device or None,
                    ) as mic,
                    LoopbackStream(
                        sample_rate=sample_rate,
                        block_ms=cfg.block_ms,
                        device=cfg.loopback_device or None,
                        output_device=cfg.output_device or None,
                    ) as lb,
                ):
                    while not stop.is_set() and budget_ok():
                        a = mic.read_block()
                        b = lb.read_block()
                        write_pcm(wf, mix_pcm16_average(a, b))
                        push_mic(a)
    except Exception as e:  # pragma: no cover - device failures
        error = str(e)

    duration = max(0.0, time.monotonic() - started)
    if not error and frames <= 0:
        error = "Meeting recording is empty"
    return {
        "frames": frames,
        "duration_s": duration,
        "capture": mode,
        "error": error,
        "wav_path": str(out),
    }


def _process_main(
    cfg_dict: dict[str, Any],
    wav_path: str,
    stop_event: Any,
    mic_q: Any,
    result_q: Any,
) -> None:
    """multiprocessing entry: must stay top-level for Windows spawn."""
    cfg = MeetingCaptureConfig(**cfg_dict)
    result = run_meeting_capture(
        cfg,
        wav_path,
        stop=stop_event,
        mic_tap_queue=mic_q,
    )
    try:
        result_q.put(result)
    except Exception:
        pass


class MeetingRecorder:
    """Meeting capture in a child process; parent keeps mic-tap for stop ASR.

    Note (Windows): the worker owns mix/mic+loopback devices. Wake on the main
    capture stream may be unavailable while recording; stop via hotkey or the
    mic-tap side channel from this recorder.
    """

    def __init__(
        self,
        cfg: MeetingCaptureConfig,
        *,
        save_dir: str | Path | None = None,
        use_process: bool = True,
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
        self.use_process = bool(use_process)
        self._lock = threading.Lock()
        self._stop_event: Any | None = None
        self._mic_q: Any | None = None
        self._result_q: Any | None = None
        self._proc: mp.Process | None = None
        self._thread: threading.Thread | None = None
        self._wav_path: Path | None = None
        self._error: str = ""
        self._started_at = 0.0
        self._active = False
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

    def start(self) -> Path:
        with self._lock:
            if self._active:
                raise RuntimeError("Meeting already recording")
            wav_path = new_meeting_wav_path(self.save_dir)
            self._wav_path = wav_path
            self._error = ""
            self._result = None
            self._started_at = time.monotonic()
            self._active = True

            if self.use_process:
                ctx = mp.get_context("spawn")
                self._stop_event = ctx.Event()
                self._mic_q = ctx.Queue(maxsize=200)
                self._result_q = ctx.Queue(maxsize=1)
                self._proc = ctx.Process(
                    target=_process_main,
                    args=(
                        asdict(self.cfg),
                        str(wav_path),
                        self._stop_event,
                        self._mic_q,
                        self._result_q,
                    ),
                    name="meeting-capture",
                    daemon=True,
                )
                self._proc.start()
            else:
                self._stop_event = threading.Event()
                self._mic_q = queue.Queue(maxsize=200)
                self._thread = threading.Thread(
                    target=self._run_thread,
                    name="meeting-capture",
                    daemon=True,
                )
                self._thread.start()
            return wav_path

    def stop(self) -> MeetingSessionResult:
        with self._lock:
            if not self._active and self._proc is None and self._thread is None:
                raise RuntimeError("Meeting is not recording")
            wav_path = self._wav_path
            stop_event = self._stop_event
            proc = self._proc
            thread = self._thread
            result_q = self._result_q
        if stop_event is not None:
            stop_event.set()
        if proc is not None:
            proc.join(timeout=30.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)
        if thread is not None:
            thread.join(timeout=30.0)

        result = self._result
        if result is None and result_q is not None:
            try:
                result = result_q.get(timeout=2.0)
            except Exception:
                result = None
        if result is None:
            result = {}

        with self._lock:
            err = str(result.get("error") or self._error or "")
            capture = str(result.get("capture") or self.cfg.capture)
            frames = int(result.get("frames") or 0)
            duration = float(result.get("duration_s") or 0.0)
            if duration <= 0 and self._started_at:
                duration = max(0.0, time.monotonic() - self._started_at)
            path = Path(str(result.get("wav_path") or wav_path or ""))
            self._proc = None
            self._thread = None
            self._active = False
            self._stop_event = None
            self._mic_q = None
            self._result_q = None
            self._result = None

        if err:
            raise RuntimeError(err)
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
        chunks: list[np.ndarray] = []
        need = max(1, int(float(seconds) * self.cfg.sample_rate))
        got = 0
        while got < need:
            try:
                arr = q.get_nowait()
            except Exception:
                break
            chunks.append(np.asarray(arr, dtype=np.int16).reshape(-1))
            got += int(chunks[-1].size)
        if not chunks:
            return None
        pcm = np.concatenate(chunks)
        if pcm.size < self.cfg.sample_rate // 4:
            return None
        return pcm[-need:]

    def _run_thread(self) -> None:
        assert self._stop_event is not None and self._wav_path is not None
        result = run_meeting_capture(
            self.cfg,
            self._wav_path,
            stop=self._stop_event,
            mic_tap_queue=self._mic_q,
        )
        with self._lock:
            self._result = result
            if result.get("error"):
                self._error = str(result["error"])
            self._active = False
