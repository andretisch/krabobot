"""Persistent VAD segmenter for a live PCM block stream.

Speech start/end: ``speech_gate`` (Silero) when set; else energy-RMS fallback.
``energy_pregate`` only skips Silero on near-silence. Optional ``early_check``
is an energy-backend helper (unused when Silero is active in the app).
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from krabobot_voice.vad import frame_rms, pcm_stats

_UNSET: Any = object()


@dataclass
class VadSegmenter:
    """Utterance segmenter whose preroll/state survive across phases.

    Lives for the lifetime of an open capture stream. ``next_segment`` returns
    ``None`` when the wall-clock ``deadline`` fires **before a commit** (even if
    speech already started — safety against sticky VAD / long trails). Otherwise
    the segment finishes on silence, ``max_s`` (counted from speech start), or an
    optional ``early_check`` hit (local command / wake without waiting for silence).
    """

    read_block: Callable[[], np.ndarray]
    sample_rate: int
    block: int
    max_s: float = 15.0
    silence_end_s: float = 2.0
    speech_start_s: float = 0.25
    min_speech_s: float = 1.2
    energy_threshold: float = 0.008
    preroll_s: float = 0.8
    # Optional classifier: pcm16 block → speaking (Silero or test mocks).
    speech_gate: Callable[[np.ndarray], bool] | None = None
    # When speech_gate is set: treat as non-speech if RMS is below this (CPU skip
    # for pure silence). Music passes the pre-gate; Silero still rejects it.
    energy_pregate: float | None = 0.0008

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

    def _speaking(self, chunk: np.ndarray) -> bool:
        """Speech decision: Silero gate when set, else energy RMS."""
        if self.speech_gate is None:
            return frame_rms(chunk) >= float(self.energy_threshold)
        pre = self.energy_pregate
        if pre is not None and float(pre) > 0.0 and frame_rms(chunk) < float(pre):
            return False
        return bool(self.speech_gate(chunk))

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
        speech_gate: Any = _UNSET,
        energy_pregate: Any = _UNSET,
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
        if speech_gate is not _UNSET:
            self.speech_gate = speech_gate
        if energy_pregate is not _UNSET:
            self.energy_pregate = energy_pregate

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

    def _finish_early(self) -> np.ndarray | None:
        """Close on early_check hit (command/wake) — no silence trim."""
        if not self._started or not self._buf:
            return None
        data = np.concatenate(self._buf)
        self.reset_utterance()
        if data.size < self.sample_rate // 4:
            return None
        return data

    def next_segment(
        self,
        deadline: float | None,
        poll: Callable[[], None] | None = None,
        *,
        early_check: Callable[[np.ndarray], bool] | None = None,
        early_check_interval_s: float = 1.0,
        early_check_min_s: float = 0.7,
    ) -> np.ndarray | None:
        """Wait for the next VAD-closed utterance.

        ``deadline`` is an absolute ``time.monotonic()`` timestamp, or ``None``
        for no listen timeout (idle forever). When the deadline fires before a
        commit, returns ``None`` even if speech already started (hard listen
        wall-clock — avoids hanging until ``max_s`` on sticky VAD).

        When ``early_check`` is set, after ``early_check_min_s`` of voiced audio
        the callback runs about every ``early_check_interval_s`` on the growing
        buffer. Returning True closes immediately (local command / wake match)
        without waiting for silence. Silence_end and max_s remain fallbacks.
        """
        sr = self.sample_rate
        block = self.block
        interval_blocks = max(1, int(float(early_check_interval_s) * sr / block))
        min_early_blocks = max(1, int(float(early_check_min_s) * sr / block))
        blocks_since_check = 0
        # max_s is per attempt; preroll/started state may carry across calls.
        self._elapsed = 0
        while True:
            if poll is not None:
                poll()

            now = time.monotonic()
            if deadline is not None and now >= deadline:
                # Hard wall-clock: abandon uncommitted utterance (incl. sticky speech).
                if self._started:
                    self.reset_utterance()
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

            speaking = self._speaking(chunk)
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
                        # max_s caps the utterance after speech appears, not idle wait.
                        self._elapsed = self._speech_blocks
                        blocks_since_check = 0
                else:
                    self._speech_run = 0
                if deadline is not None and time.monotonic() >= deadline:
                    return None
                continue

            self._buf.append(chunk)
            blocks_since_check += 1
            if speaking:
                self._silence_run = 0
                self._speech_blocks += 1
            else:
                self._silence_run += 1
                if self._silence_run < self._end_need:
                    if (
                        early_check is not None
                        and self._speech_blocks >= min_early_blocks
                        and blocks_since_check >= interval_blocks
                    ):
                        blocks_since_check = 0
                        if early_check(np.concatenate(self._buf)):
                            return self._finish_early()
                    if deadline is not None and time.monotonic() >= deadline:
                        self.reset_utterance()
                        return None
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
                blocks_since_check = 0
                if len(self._buf) > self._preroll_blocks:
                    self._buf = self._buf[-self._preroll_blocks :]
                if deadline is not None and time.monotonic() >= deadline:
                    return None
                continue

            # Continuous speech (or gate false-positives): probe command/wake.
            if (
                early_check is not None
                and self._speech_blocks >= min_early_blocks
                and blocks_since_check >= interval_blocks
            ):
                blocks_since_check = 0
                if early_check(np.concatenate(self._buf)):
                    return self._finish_early()

            if deadline is not None and time.monotonic() >= deadline:
                self.reset_utterance()
                return None

            if self._elapsed >= self._max_blocks:
                data = self._finish_max()
                self.reset_utterance()
                return data
