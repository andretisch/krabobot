"""Persistent energy-VAD segmenter for a live PCM block stream."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from krabobot_voice.vad import frame_rms, pcm_stats


@dataclass
class VadSegmenter:
    """Utterance segmenter whose preroll/state survive across phases.

    Lives for the lifetime of an open capture stream. ``next_segment`` returns
    ``None`` only when **no speech has started** by ``deadline``. Once speech
    is confirmed, the segment is always finished (subject to ``max_s``).
    """

    read_block: Callable[[], np.ndarray]
    sample_rate: int
    block: int
    max_s: float = 15.0
    silence_end_s: float = 2.0
    speech_start_s: float = 0.25
    min_speech_s: float = 1.2
    energy_threshold: float = 0.008
    preroll_s: float = 0.4

    def __post_init__(self) -> None:
        sr = max(1, int(self.sample_rate))
        block = max(1, int(self.block))
        self.sample_rate = sr
        self.block = block
        self._start_need = max(1, int(float(self.speech_start_s) * sr / block))
        self._end_need = max(1, int(float(self.silence_end_s) * sr / block))
        self._min_speech_blocks = max(1, int(float(self.min_speech_s) * sr / block))
        self._preroll_blocks = max(1, int(float(self.preroll_s) * sr / block))
        self._max_blocks = max(1, int(float(self.max_s) * sr / block))
        self._backlog: deque[np.ndarray] = deque()
        self._buf: list[np.ndarray] = []
        self._speech_run = 0
        self._silence_run = 0
        self._started = False
        self._speech_blocks = 0
        self._elapsed = 0

    def feed_backlog(self, blocks: Iterable[np.ndarray] | None) -> None:
        """Queue PCM blocks captured during ASR/HTTP (keepalive sink)."""
        if not blocks:
            return
        for raw in blocks:
            arr = np.asarray(raw, dtype=np.int16).reshape(-1)
            if arr.size == 0:
                continue
            self._backlog.append(arr)

    def reset_utterance(self) -> None:
        """Clear in-progress utterance state; keep backlog and energy settings."""
        self._buf = []
        self._speech_run = 0
        self._silence_run = 0
        self._started = False
        self._speech_blocks = 0
        self._elapsed = 0

    def configure(
        self,
        *,
        energy_threshold: float | None = None,
        max_s: float | None = None,
        silence_end_s: float | None = None,
        min_speech_s: float | None = None,
    ) -> None:
        """Update VAD thresholds without dropping backlog / preroll."""
        sr = self.sample_rate
        block = self.block
        if energy_threshold is not None:
            self.energy_threshold = float(energy_threshold)
        if max_s is not None:
            self.max_s = float(max_s)
            self._max_blocks = max(1, int(self.max_s * sr / block))
        if silence_end_s is not None:
            self.silence_end_s = float(silence_end_s)
            self._end_need = max(1, int(self.silence_end_s * sr / block))
        if min_speech_s is not None:
            self.min_speech_s = float(min_speech_s)
            self._min_speech_blocks = max(1, int(self.min_speech_s * sr / block))

    def _read(self) -> np.ndarray:
        if self._backlog:
            chunk = self._backlog.popleft()
            # Normalize to one block when possible (keepalive may yield full blocks).
            if chunk.size == self.block:
                return chunk
            if chunk.size > self.block:
                head = chunk[: self.block]
                rest = chunk[self.block :]
                if rest.size:
                    self._backlog.appendleft(rest)
                return head
            # Short chunk — pad so VAD frame math stays stable.
            out = np.zeros(self.block, dtype=np.int16)
            out[: chunk.size] = chunk
            return out
        return np.asarray(self.read_block(), dtype=np.int16).reshape(-1)

    def _commit(self) -> np.ndarray | None:
        keep = max(0, len(self._buf) - self._end_need)
        data = np.concatenate(self._buf[:keep]) if keep else np.concatenate(self._buf)
        min_samples = int(float(self.min_speech_s) * 0.6 * self.sample_rate)
        if data.size < max(self.sample_rate // 4, min_samples):
            return None
        stats = pcm_stats(data, sample_rate=self.sample_rate)
        if stats.near_silent:
            return None
        return data

    def _finish_max(self) -> np.ndarray | None:
        if not self._started or not self._buf:
            return None
        if self._speech_blocks < self._min_speech_blocks:
            return None
        data = np.concatenate(self._buf)
        if data.size < self.sample_rate // 4:
            return None
        stats = pcm_stats(data, sample_rate=self.sample_rate)
        if stats.near_silent:
            return None
        return data

    def next_segment(
        self,
        deadline: float | None,
        poll: Callable[[], None] | None = None,
    ) -> np.ndarray | None:
        """Wait for the next VAD-closed utterance.

        ``deadline`` is an absolute ``time.monotonic()`` timestamp, or ``None``
        for no listen timeout (idle forever). Returns ``None`` only if speech
        never started before the deadline.
        """
        thr = float(self.energy_threshold)
        # max_s is per attempt; preroll/started state may carry across calls.
        self._elapsed = 0
        while True:
            if poll is not None:
                poll()

            now = time.monotonic()
            if deadline is not None and now >= deadline and not self._started:
                return None

            chunk = self._read()
            if chunk.size != self.block:
                # Defensive: coerce odd device reads to block size.
                if chunk.size > self.block:
                    chunk = chunk[: self.block]
                else:
                    pad = np.zeros(self.block, dtype=np.int16)
                    pad[: chunk.size] = chunk
                    chunk = pad

            speaking = frame_rms(chunk) >= thr
            self._elapsed += 1

            if not self._started:
                self._buf.append(chunk)
                if len(self._buf) > self._preroll_blocks and not speaking:
                    self._buf = self._buf[-self._preroll_blocks :]
                if speaking:
                    self._speech_run += 1
                    if self._speech_run >= self._start_need:
                        self._started = True
                        self._silence_run = 0
                        self._speech_blocks = self._speech_run
                else:
                    self._speech_run = 0
                if deadline is not None and time.monotonic() >= deadline and not self._started:
                    return None
                continue

            self._buf.append(chunk)
            if speaking:
                self._silence_run = 0
                self._speech_blocks += 1
            else:
                self._silence_run += 1
                if self._silence_run < self._end_need:
                    if self._elapsed >= self._max_blocks:
                        data = self._finish_max()
                        self.reset_utterance()
                        return data
                    continue
                # Trailing silence.
                if self._speech_blocks >= self._min_speech_blocks:
                    data = self._commit()
                    self.reset_utterance()
                    # Keep a short preroll from the silence tail for the next call.
                    return data
                # False start — reset and keep listening (deadline still applies).
                self._started = False
                self._speech_run = 0
                self._silence_run = 0
                self._speech_blocks = 0
                if len(self._buf) > self._preroll_blocks:
                    self._buf = self._buf[-self._preroll_blocks :]
                if deadline is not None and time.monotonic() >= deadline:
                    return None
                continue

            if self._elapsed >= self._max_blocks:
                data = self._finish_max()
                self.reset_utterance()
                return data
