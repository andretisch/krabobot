"""Energy-based VAD helpers for utterance capture.

Live wake/Talk capture prefers Silero via ``VadSegmenter.speech_gate``
(``krabobot_voice.silero_vad``). These helpers remain the energy fallback
and offline buffer utilities used by tests / legacy ``record_utterance``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def frame_rms(pcm16: np.ndarray) -> float:
    """RMS of int16 mono PCM (normalized to ~[-1, 1])."""
    if pcm16.size == 0:
        return 0.0
    x = pcm16.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x)))


def is_speech(pcm16: np.ndarray, *, threshold: float) -> bool:
    return frame_rms(pcm16) >= threshold


@dataclass(frozen=True)
class PcmStats:
    """Diagnostics for a captured PCM clip."""

    samples: int
    duration_ms: float
    rms: float
    peak: float

    @property
    def near_silent(self) -> bool:
        return self.rms < 0.004 or self.duration_ms < 400


def pcm_stats(pcm16: np.ndarray, *, sample_rate: int) -> PcmStats:
    arr = np.asarray(pcm16, dtype=np.int16).reshape(-1)
    samples = int(arr.size)
    duration_ms = 1000.0 * samples / float(sample_rate) if sample_rate > 0 else 0.0
    if samples == 0:
        return PcmStats(samples=0, duration_ms=0.0, rms=0.0, peak=0.0)
    rms = frame_rms(arr)
    peak = float(np.max(np.abs(arr.astype(np.float32))) / 32768.0)
    return PcmStats(samples=samples, duration_ms=duration_ms, rms=rms, peak=peak)


def record_utterance(
    read_block: Callable[[], np.ndarray],
    *,
    sample_rate: int,
    block: int,
    max_s: float = 15.0,
    silence_end_s: float = 1.2,
    speech_start_s: float = 0.10,
    min_speech_s: float = 1.2,
    energy_threshold: float = 0.008,
    settle_s: float = 0.35,
    preroll_s: float = 1.2,
    no_speech_timeout_s: float = 5.0,
    commit_silence_pad_s: float = 0.18,
) -> np.ndarray | None:
    """Capture one utterance from a blocking ``read_block`` source.

    After wake/beep, callers should pass a live mic reader. This function:
      1. Discards ``settle_s`` of audio (flush PortAudio buffer / beep echo).
      2. Waits for sustained speech (``speech_start_s``).
      3. Ends on trailing silence only after ``min_speech_s`` of *voiced* frames
         (false starts from clicks/beep echo are reset instead of uploaded).
    """
    sr = int(sample_rate)
    block = max(1, int(block))
    max_blocks = max(1, int(max_s * sr / block))
    start_need = max(1, int(speech_start_s * sr / block))
    end_need = max(1, int(silence_end_s * sr / block))
    min_speech_blocks = max(1, int(min_speech_s * sr / block))
    preroll_blocks = max(1, int(preroll_s * sr / block))
    no_speech_blocks = max(1, int(no_speech_timeout_s * sr / block))
    settle_blocks = max(0, int(settle_s * sr / block))

    for _ in range(settle_blocks):
        read_block()

    buf: list[np.ndarray] = []
    speech_run = 0
    silence_run = 0
    started = False
    speech_blocks = 0  # voiced frames since (re)start
    # Wall-clock deadline — survives hung/slow reads that still eventually return.
    listen_deadline = time.monotonic() + max(0.1, float(no_speech_timeout_s))

    for elapsed in range(max_blocks):
        if time.monotonic() >= listen_deadline and not started:
            return None
        if (
            time.monotonic() >= listen_deadline
            and started
            and speech_blocks < min_speech_blocks
        ):
            return None
        chunk = np.asarray(read_block(), dtype=np.int16).reshape(-1)
        speaking = frame_rms(chunk) >= energy_threshold

        if not started:
            buf.append(chunk)
            if len(buf) > preroll_blocks and not speaking:
                buf = buf[-preroll_blocks:]
            if speaking:
                speech_run += 1
                if speech_run >= start_need:
                    started = True
                    silence_run = 0
                    speech_blocks = speech_run
            else:
                speech_run = 0
            # Absolute listen deadline (survives false-start resets).
            if elapsed + 1 >= no_speech_blocks:
                return None
            continue

        buf.append(chunk)
        # Hard listen deadline while utterance is still unconfirmed (false starts /
        # loopback noise) — do not hang past no_speech_timeout_s.
        if speech_blocks < min_speech_blocks and elapsed + 1 >= no_speech_blocks:
            return None
        if speaking:
            silence_run = 0
            speech_blocks += 1
        else:
            silence_run += 1
            if silence_run < end_need:
                continue
            # Trailing silence — only commit if we actually heard enough speech.
            # Soft trim: leave ~commit_silence_pad_s of silence; keep preroll.
            if speech_blocks >= min_speech_blocks:
                pad_blocks = max(1, int(float(commit_silence_pad_s) * sr / block))
                strip = max(0, end_need - pad_blocks)
                keep = max(0, len(buf) - strip)
                data = np.concatenate(buf[:keep]) if keep else np.concatenate(buf)
                min_samples = int(min_speech_s * 0.6 * sr)
                if data.size < max(sr // 4, min_samples):
                    return None
                stats = pcm_stats(data, sample_rate=sr)
                if stats.near_silent:
                    return None
                return data
            # False start (beep echo / click): reset and keep listening.
            started = False
            speech_run = 0
            silence_run = 0
            speech_blocks = 0
            if len(buf) > preroll_blocks:
                buf = buf[-preroll_blocks:]
            if elapsed + 1 >= no_speech_blocks:
                return None
            if time.monotonic() >= listen_deadline:
                return None

    if not started or not buf:
        return None
    data = np.concatenate(buf)
    if speech_blocks < min_speech_blocks:
        return None
    if data.size < sr // 4:
        return None
    stats = pcm_stats(data, sample_rate=sr)
    if stats.near_silent:
        return None
    return data


def detect_speech_segment(
    pcm16_mono: bytes | np.ndarray,
    *,
    sample_rate: int = 16000,
    frame_ms: int = 30,
    energy_threshold: float = 0.008,
    speech_start_s: float = 0.10,
    silence_end_s: float = 1.2,
    min_speech_s: float = 1.2,
) -> bytes | None:
    """Return first speech segment (PCM16 bytes) from a buffer, or None."""
    if isinstance(pcm16_mono, (bytes, bytearray)):
        arr = np.frombuffer(pcm16_mono, dtype=np.int16)
    else:
        arr = np.asarray(pcm16_mono, dtype=np.int16).reshape(-1)

    frame = max(1, int(sample_rate * frame_ms / 1000))
    start_need = max(1, int(speech_start_s * 1000 / frame_ms))
    end_need = max(1, int(silence_end_s * 1000 / frame_ms))
    min_speech_frames = max(1, int(min_speech_s * 1000 / frame_ms))

    speech_run = 0
    silence_run = 0
    started = False
    seg_start = 0
    seg_end = 0
    speech_frames = 0

    for i in range(0, len(arr) - frame + 1, frame):
        chunk = arr[i : i + frame]
        speaking = is_speech(chunk, threshold=energy_threshold)
        if not started:
            if speaking:
                speech_run += 1
                if speech_run >= start_need:
                    started = True
                    seg_start = max(0, i - frame * start_need)
                    seg_end = i + frame
                    silence_run = 0
                    speech_frames = speech_run
            else:
                speech_run = 0
        else:
            seg_end = i + frame
            if speaking:
                silence_run = 0
                speech_frames += 1
            else:
                silence_run += 1
                if silence_run < end_need:
                    continue
                if speech_frames >= min_speech_frames:
                    end = max(seg_start, seg_end - frame * end_need)
                    return arr[seg_start:end].tobytes()
                # False start — resume search after this click/noise.
                started = False
                speech_run = 0
                silence_run = 0
                speech_frames = 0
                seg_start = 0
                seg_end = 0

    if started and speech_frames >= min_speech_frames:
        return arr[seg_start:seg_end].tobytes()
    return None
