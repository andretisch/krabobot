"""Align / normalize audio before sherpa-onnx STT (defensive server path)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

TARGET_SR = 16000


def remove_dc_offset(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    return arr - float(arr.mean())


def resample_mono(x: np.ndarray, src_rate: int, dst_rate: int = TARGET_SR) -> np.ndarray:
    arr = np.asarray(x).reshape(-1)
    if src_rate == dst_rate or arr.size == 0:
        return arr.astype(np.float32, copy=False)
    n_dst = max(1, int(round(arr.size * float(dst_rate) / float(src_rate))))
    xp = np.linspace(0.0, 1.0, num=arr.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
    return np.interp(x_new, xp, arr.astype(np.float64)).astype(np.float32)


def frame_rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))


def trim_silence(
    x: np.ndarray,
    *,
    sample_rate: int = TARGET_SR,
    frame_ms: int = 30,
    energy_threshold: float = 0.008,
    pad_ms: int = 120,
    min_keep_s: float = 0.25,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    frame = max(1, int(sample_rate * frame_ms / 1000))
    pad = max(0, int(sample_rate * pad_ms / 1000))
    n_frames = max(1, arr.size // frame)
    energies = [frame_rms(arr[i * frame : (i + 1) * frame]) for i in range(n_frames)]
    voiced = [e >= energy_threshold for e in energies]
    if not any(voiced):
        return arr
    first = next(i for i, v in enumerate(voiced) if v)
    last = len(voiced) - 1 - next(i for i, v in enumerate(reversed(voiced)) if v)
    start = max(0, first * frame - pad)
    end = min(arr.size, (last + 1) * frame + pad)
    out = arr[start:end]
    min_keep = int(min_keep_s * sample_rate)
    if out.size < min_keep and arr.size >= min_keep:
        mid = (start + end) // 2
        half = min_keep // 2
        start2 = max(0, mid - half)
        end2 = min(arr.size, start2 + min_keep)
        return arr[start2:end2]
    return out


def normalize_level(
    x: np.ndarray,
    *,
    target_peak: float = 0.89,
    target_rms: float = 0.12,
    max_gain: float = 24.0,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    peak = float(np.max(np.abs(arr)))
    rms = frame_rms(arr)
    if peak < 1e-8:
        return arr
    gain_peak = target_peak / peak
    gain_rms = (target_rms / rms) if rms > 1e-8 else gain_peak
    gain = min(max(gain_rms, 1.0), gain_peak, max_gain)
    if gain < 1.01 and peak <= target_peak:
        if peak > 0.99:
            return arr * (target_peak / peak)
        return arr
    out = arr * float(gain)
    peak2 = float(np.max(np.abs(out)))
    if peak2 > 0.99:
        out = out * (target_peak / peak2)
    return out.astype(np.float32)


def align_waveform_float(
    waveform: np.ndarray,
    *,
    sample_rate: int = TARGET_SR,
    trim: bool = True,
    energy_threshold: float = 0.008,
) -> np.ndarray:
    """DC-remove, optional trim, level-normalize a float mono waveform."""
    x = remove_dc_offset(np.asarray(waveform, dtype=np.float32).reshape(-1))
    if int(sample_rate) != TARGET_SR:
        x = resample_mono(x, int(sample_rate), TARGET_SR)
    if trim:
        x = trim_silence(x, sample_rate=TARGET_SR, energy_threshold=energy_threshold)
    return normalize_level(x)


def float_to_pcm16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64) * 32767.0, -32768, 32767).astype(np.int16)


def rewrite_wav_16k_mono(path: str | Path, waveform: np.ndarray) -> Path:
    """Overwrite ``path`` as clean 16 kHz mono int16 WAV."""
    p = Path(path)
    pcm = float_to_pcm16(waveform)
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(pcm.tobytes())
    return p
