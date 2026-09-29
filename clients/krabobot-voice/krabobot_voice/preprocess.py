"""Align / normalize PCM for sherpa-onnx (wake + Talk upload).

Goals:
- 16 kHz mono int16 with a correct WAV header
- DC removal, careful silence trim
- peak/RMS gain so quiet mics still decode (without hard clipping)
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

TARGET_SR = 16000


def remove_dc_offset(x: np.ndarray) -> np.ndarray:
    """Subtract mean from float waveform."""
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    return arr - float(arr.mean())


def resample_mono(x: np.ndarray, src_rate: int, dst_rate: int = TARGET_SR) -> np.ndarray:
    """Linear-resample 1-D float/int array to ``dst_rate``."""
    arr = np.asarray(x).reshape(-1)
    if src_rate == dst_rate or arr.size == 0:
        return arr.astype(np.float32, copy=False)
    n_dst = max(1, int(round(arr.size * float(dst_rate) / float(src_rate))))
    xp = np.linspace(0.0, 1.0, num=arr.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
    return np.interp(x_new, xp, arr.astype(np.float64)).astype(np.float32)


def _to_float_mono(pcm: np.ndarray | bytes, *, channels: int = 1) -> np.ndarray:
    if isinstance(pcm, (bytes, bytearray)):
        arr = np.frombuffer(pcm, dtype=np.int16)
    else:
        arr = np.asarray(pcm)
    if np.issubdtype(arr.dtype, np.integer):
        x = arr.astype(np.float32) / 32768.0
    else:
        x = arr.astype(np.float32, copy=False)
        if float(np.max(np.abs(x))) > 1.5:
            x = x / 32768.0
    x = x.reshape(-1)
    ch = max(1, int(channels))
    if ch > 1 and x.size % ch == 0:
        x = x.reshape(-1, ch).mean(axis=1)
    return x.astype(np.float32, copy=False)


def float_to_pcm16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64) * 32767.0, -32768, 32767).astype(np.int16)


def frame_rms_float(x: np.ndarray) -> float:
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
    """Trim leading/trailing low-energy frames; keep a small pad."""
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    frame = max(1, int(sample_rate * frame_ms / 1000))
    pad = max(0, int(sample_rate * pad_ms / 1000))
    n_frames = max(1, arr.size // frame)
    energies = [
        frame_rms_float(arr[i * frame : (i + 1) * frame]) for i in range(n_frames)
    ]
    voiced = [e >= energy_threshold for e in energies]
    if not any(voiced):
        # Keep original if nothing passes the gate (avoid emptying short clips).
        return arr
    first = next(i for i, v in enumerate(voiced) if v)
    last = len(voiced) - 1 - next(i for i, v in enumerate(reversed(voiced)) if v)
    start = max(0, first * frame - pad)
    end = min(arr.size, (last + 1) * frame + pad)
    out = arr[start:end]
    min_keep = int(min_keep_s * sample_rate)
    if out.size < min_keep and arr.size >= min_keep:
        # Prefer a centered window if trim collapsed too hard.
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
    """Raise quiet speech toward target RMS/peak without hard clipping."""
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    peak = float(np.max(np.abs(arr)))
    rms = frame_rms_float(arr)
    if peak < 1e-8:
        return arr
    gain_peak = target_peak / peak
    gain_rms = (target_rms / rms) if rms > 1e-8 else gain_peak
    # Prefer RMS lift when quiet; never exceed peak budget or max_gain.
    gain = min(max(gain_rms, 1.0), gain_peak, max_gain)
    if gain < 1.01 and peak <= target_peak:
        # Already loud enough — only gently pull peaks down if clipping-risk.
        if peak > 0.99:
            return arr * (target_peak / peak)
        return arr
    out = arr * float(gain)
    peak2 = float(np.max(np.abs(out)))
    if peak2 > 0.99:
        out = out * (target_peak / peak2)
    return out.astype(np.float32)


def preprocess_pcm16(
    pcm: np.ndarray | bytes,
    *,
    sample_rate: int = TARGET_SR,
    channels: int = 1,
    trim: bool = True,
    energy_threshold: float = 0.008,
) -> tuple[np.ndarray, int]:
    """Return aligned int16 mono @ 16 kHz ready for sherpa / upload."""
    x = _to_float_mono(pcm, channels=channels)
    x = remove_dc_offset(x)
    if int(sample_rate) != TARGET_SR:
        x = resample_mono(x, int(sample_rate), TARGET_SR)
    if trim:
        x = trim_silence(x, sample_rate=TARGET_SR, energy_threshold=energy_threshold)
    x = normalize_level(x)
    return float_to_pcm16(x), TARGET_SR


def pcm16_to_wav_bytes(pcm16: np.ndarray | bytes, *, sample_rate: int = TARGET_SR) -> bytes:
    """Write a clean mono 16-bit PCM WAV (always rewrites header)."""
    if isinstance(pcm16, np.ndarray):
        data = np.asarray(pcm16, dtype=np.int16).reshape(-1).tobytes()
    else:
        data = bytes(pcm16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(data)
    return buf.getvalue()


def load_wav_pcm16(path: str | Path | bytes) -> tuple[np.ndarray, int, int]:
    """Load WAV → (int16 samples possibly interleaved, sample_rate, channels)."""
    if isinstance(path, (bytes, bytearray)):
        src: io.BytesIO | str = io.BytesIO(path)
    else:
        src = str(path)
    with wave.open(src, "rb") as wf:  # type: ignore[arg-type]
        rate = int(wf.getframerate())
        ch = int(wf.getnchannels())
        width = int(wf.getsampwidth())
        n = int(wf.getnframes())
        raw = wf.readframes(n)
    if width != 2:
        # Best-effort: reinterpret / reject exotic widths by zeroing.
        raise ValueError(f"expected 16-bit PCM WAV, got sampwidth={width}")
    arr = np.frombuffer(raw, dtype=np.int16)
    return arr, rate, ch


def preprocess_wav_bytes(
    wav_bytes: bytes,
    *,
    trim: bool = True,
    energy_threshold: float = 0.008,
) -> bytes:
    """Repair/normalize a WAV blob for sherpa (header + level + 16 kHz mono)."""
    try:
        pcm, rate, ch = load_wav_pcm16(wav_bytes)
    except Exception:
        # Broken header — treat payload as raw int16 @ assumed 16k mono.
        pcm = np.frombuffer(wav_bytes, dtype=np.int16)
        rate, ch = TARGET_SR, 1
    aligned, sr = preprocess_pcm16(
        pcm, sample_rate=rate, channels=ch, trim=trim, energy_threshold=energy_threshold
    )
    return pcm16_to_wav_bytes(aligned, sample_rate=sr)


def preprocess_wav_file(
    path: str | Path,
    *,
    inplace: bool = True,
    trim: bool = True,
    energy_threshold: float = 0.008,
) -> Path:
    """Normalize a WAV on disk; rewrite with a correct 16 kHz mono header."""
    p = Path(path)
    raw = p.read_bytes()
    out = preprocess_wav_bytes(raw, trim=trim, energy_threshold=energy_threshold)
    dest = p if inplace else p.with_name(p.stem + "_aligned.wav")
    dest.write_bytes(out)
    return dest
