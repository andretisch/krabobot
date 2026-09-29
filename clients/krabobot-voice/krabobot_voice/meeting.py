"""Meeting recording: start/stop thread, capture mic/loopback/mix, upload helper."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from krabobot_voice.audio_io import (
    LoopbackStream,
    MicStream,
    mix_pcm16_average,
    pcm16_to_wav_bytes,
    require_sounddevice,
)


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
    """Stopped meeting payload ready for upload."""

    wav_bytes: bytes
    duration_s: float
    capture: str
    frames: int = 0


class _MicTapReader:
    """Block source backed by MeetingRecorder mic-tap queue (VadSegmenter-compatible)."""

    def __init__(
        self,
        q: queue.Queue[np.ndarray],
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


class MeetingRecorder:
    """Background meeting capture; toggle start/stop from the main loop."""

    def __init__(self, cfg: MeetingCaptureConfig) -> None:
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
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._chunks: list[np.ndarray] = []
        self._error: str = ""
        self._started_at = 0.0
        self._active = False
        # Mic-only ring for local stop-phrase ASR while recording (mic / mix).
        self._mic_ring: deque[np.ndarray] = deque()
        self._mic_samples = 0
        self._mic_ring_max = max(1, int(self.cfg.sample_rate * 3.0))
        self._mic_tap_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def capture_mode(self) -> str:
        return self.cfg.capture

    def start(self) -> None:
        with self._lock:
            if self._active:
                raise RuntimeError("Meeting already recording")
            self._stop.clear()
            self._chunks = []
            self._error = ""
            self._mic_ring.clear()
            self._mic_samples = 0
            while True:
                try:
                    self._mic_tap_q.get_nowait()
                except queue.Empty:
                    break
            self._started_at = time.monotonic()
            self._active = True
            self._thread = threading.Thread(
                target=self._run,
                name="meeting-capture",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> MeetingSessionResult:
        with self._lock:
            if not self._active and self._thread is None:
                raise RuntimeError("Meeting is not recording")
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=30.0)
        with self._lock:
            err = self._error
            chunks = list(self._chunks)
            started = self._started_at
            capture = self.cfg.capture
            self._thread = None
            self._active = False
            self._chunks = []
        if err:
            raise RuntimeError(err)
        if not chunks:
            raise RuntimeError("Meeting recording is empty")
        pcm = np.concatenate(chunks)
        duration = max(0.0, time.monotonic() - started) if started else (
            pcm.size / float(self.cfg.sample_rate)
        )
        wav = pcm16_to_wav_bytes(pcm, sample_rate=self.cfg.sample_rate)
        return MeetingSessionResult(
            wav_bytes=wav,
            duration_s=duration,
            capture=capture,
            frames=int(pcm.size),
        )

    def mic_tap_reader(self) -> _MicTapReader:
        """Adapter: blocking int16 mono blocks from the mic tap (for VadSegmenter)."""
        block = max(1, int(self.cfg.sample_rate * self.cfg.block_ms / 1000))
        return _MicTapReader(
            self._mic_tap_q,
            sample_rate=self.cfg.sample_rate,
            block=block,
            active=lambda: self.active,
        )

    def _run(self) -> None:  # noqa: C901 — capture modes are intentionally explicit
        try:
            if self.cfg.capture == "mic":
                self._run_mic_only()
            elif self.cfg.capture == "loopback":
                self._run_loopback_only()
            else:
                self._run_mix()
        except Exception as e:  # pragma: no cover - device failures
            with self._lock:
                self._error = str(e)
        finally:
            with self._lock:
                self._active = False

    def _append(self, pcm: np.ndarray) -> None:
        with self._lock:
            self._chunks.append(np.asarray(pcm, dtype=np.int16).reshape(-1))

    def _append_mic_tap(self, pcm: np.ndarray) -> None:
        """Keep a short mic-only window + live queue for VadSegmenter."""
        arr = np.asarray(pcm, dtype=np.int16).reshape(-1)
        with self._lock:
            self._mic_ring.append(arr)
            self._mic_samples += int(arr.size)
            while self._mic_samples > self._mic_ring_max and self._mic_ring:
                dropped = self._mic_ring.popleft()
                self._mic_samples -= int(dropped.size)
        try:
            self._mic_tap_q.put_nowait(arr.copy())
        except queue.Full:
            try:
                self._mic_tap_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._mic_tap_q.put_nowait(arr.copy())
            except queue.Full:
                pass

    def recent_mic_pcm(self, seconds: float = 2.5) -> np.ndarray | None:
        """Copy the latest mic tap window (None if empty / loopback-only)."""
        need = max(1, int(float(seconds) * self.cfg.sample_rate))
        with self._lock:
            if not self._mic_ring:
                return None
            pcm = np.concatenate(list(self._mic_ring))
        if pcm.size < self.cfg.sample_rate // 4:
            return None
        return pcm[-need:]

    def _budget_ok(self) -> bool:
        if self.cfg.max_s <= 0:
            return True
        return (time.monotonic() - self._started_at) < self.cfg.max_s

    def _run_mic_only(self) -> None:
        require_sounddevice()
        with MicStream(
            sample_rate=self.cfg.sample_rate,
            block_ms=self.cfg.block_ms,
            device=self.cfg.input_device or None,
        ) as mic:
            while not self._stop.is_set() and self._budget_ok():
                chunk = mic.read_block()
                self._append(chunk)
                self._append_mic_tap(chunk)

    def _run_loopback_only(self) -> None:
        with LoopbackStream(
            sample_rate=self.cfg.sample_rate,
            block_ms=self.cfg.block_ms,
            device=self.cfg.loopback_device or None,
            output_device=self.cfg.output_device or None,
        ) as lb:
            while not self._stop.is_set() and self._budget_ok():
                self._append(lb.read_block())

    def _run_mix(self) -> None:
        """Best-effort sync: read both streams each iteration, average, clip-protect."""
        require_sounddevice()
        with (
            MicStream(
                sample_rate=self.cfg.sample_rate,
                block_ms=self.cfg.block_ms,
                device=self.cfg.input_device or None,
            ) as mic,
            LoopbackStream(
                sample_rate=self.cfg.sample_rate,
                block_ms=self.cfg.block_ms,
                device=self.cfg.loopback_device or None,
                output_device=self.cfg.output_device or None,
            ) as lb,
        ):
            while not self._stop.is_set() and self._budget_ok():
                a = mic.read_block()
                b = lb.read_block()
                self._append(mix_pcm16_average(a, b))
                self._append_mic_tap(a)
