"""Lightweight keyword spotting for «Эй, Арнольд» (no full ASR).

Default approach: energy-gated streaming MFCC embeddings + cosine match
against short reference WAVs (synthetic TTS and/or user enrollments).

This avoids continuous sherpa-onnx wake ASR. Optional openWakeWord ONNX
models can be plugged in later via ``oww_model`` without changing the
streaming loop contract.
"""

from __future__ import annotations

import math
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from krabobot_voice.vad import frame_rms


def default_refs_dir() -> Path:
    import os

    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "krabobot-voice" / "wake_refs"


def _pcm16_from_wav(path: Path) -> tuple[np.ndarray, int]:
    """Return int16 mono PCM and sample rate."""
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM WAV: {path}")
        rate = int(wf.getframerate())
        ch = int(wf.getnchannels())
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return arr.reshape(-1), rate


def _resample_mono(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or x.size == 0:
        return x.astype(np.float32)
    n_dst = max(1, int(round(x.size * float(dst_rate) / float(src_rate))))
    xp = np.linspace(0.0, 1.0, num=x.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
    return np.interp(x_new, xp, x.astype(np.float64)).astype(np.float32)


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_fft: int, sr: int, n_mels: int = 26) -> np.ndarray:
    f_min, f_max = 80.0, min(8000.0, sr / 2.0 - 1.0)
    mels = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    hz = np.array([_mel_to_hz(float(m)) for m in mels])
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center <= left or right <= center:
            continue
        for j in range(left, center):
            fb[i, j] = (j - left) / float(center - left)
        for j in range(center, right):
            fb[i, j] = (right - j) / float(right - center)
    return fb


def _dct_matrix(n_mfcc: int, n_mels: int) -> np.ndarray:
    n = np.arange(n_mels, dtype=np.float64)
    k = np.arange(n_mfcc, dtype=np.float64)[:, None]
    return np.cos(math.pi / n_mels * (n + 0.5) * k)


def mfcc_embedding(
    pcm: np.ndarray,
    *,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    frame_ms: int = 25,
    hop_ms: int = 10,
) -> np.ndarray:
    """Mean+std MFCC vector (lightweight speaker/phrase fingerprint)."""
    if pcm.dtype == np.int16:
        x = pcm.astype(np.float32) / 32768.0
    else:
        x = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if np.max(np.abs(x)) > 1.5:
            x = x / 32768.0

    if x.size < sample_rate // 8:
        return np.zeros(n_mfcc * 2, dtype=np.float32)

    frame = max(64, int(sample_rate * frame_ms / 1000))
    hop = max(32, int(sample_rate * hop_ms / 1000))
    n_fft = 1
    while n_fft < frame:
        n_fft *= 2
    window = np.hanning(frame).astype(np.float64)
    fb = _mel_filterbank(n_fft, sample_rate)
    dct = _dct_matrix(n_mfcc, fb.shape[0])

    feats: list[np.ndarray] = []
    for start in range(0, max(0, x.size - frame + 1), hop):
        chunk = x[start : start + frame].astype(np.float64) * window
        spec = np.abs(np.fft.rfft(chunk, n=n_fft)) ** 2
        mel = np.maximum(fb @ spec, 1e-10)
        log_mel = np.log(mel)
        mfcc = dct @ log_mel
        feats.append(mfcc.astype(np.float32))

    if not feats:
        return np.zeros(n_mfcc * 2, dtype=np.float32)
    mat = np.stack(feats, axis=0)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0) + 1e-6
    emb = np.concatenate([mean, std]).astype(np.float32)
    norm = float(np.linalg.norm(emb))
    if norm > 1e-8:
        emb /= norm
    return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def load_reference_embeddings(
    refs_dir: Path,
    *,
    sample_rate: int = 16000,
) -> list[np.ndarray]:
    """Load *.wav from refs_dir and return MFCC embeddings."""
    refs_dir = Path(refs_dir)
    if not refs_dir.is_dir():
        return []
    out: list[np.ndarray] = []
    for path in sorted(refs_dir.glob("*.wav")):
        try:
            pcm, rate = _pcm16_from_wav(path)
            if rate != sample_rate:
                pcm_f = _resample_mono(pcm.astype(np.float32) / 32768.0, rate, sample_rate)
                pcm = np.clip(pcm_f * 32767.0, -32768, 32767).astype(np.int16)
            emb = mfcc_embedding(pcm, sample_rate=sample_rate)
            if float(np.linalg.norm(emb)) > 0.1:
                out.append(emb)
        except Exception:
            continue
    return out


def synthesize_wake_refs_windows(
    refs_dir: Path,
    *,
    phrase: str = "Эй, Арнольд",
    sample_rate: int = 16000,
    count: int = 3,
) -> list[Path]:
    """Generate reference WAVs via Windows SAPI (best-effort)."""
    import subprocess
    import sys

    if sys.platform != "win32":
        return []

    refs_dir = Path(refs_dir)
    refs_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    try:
        for i in range(count):
            out_path = refs_dir / f"sapi_ref_{i}.wav"
            # Rate -2..+2; vary slightly for robustness
            rate = -1 + i  # -1, 0, +1
            safe_phrase = phrase.replace("'", "''")
            safe_out = str(out_path).replace("'", "''")
            ps = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Rate = {rate}; $s.Volume = 100; "
                "$tmp = [System.IO.Path]::GetTempFileName() + '.wav'; "
                "$s.SetOutputToWaveFile($tmp); "
                f"$s.Speak('{safe_phrase}'); "
                "$s.Dispose(); "
                f"Copy-Item -LiteralPath $tmp -Destination '{safe_out}' -Force; "
                "Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if out_path.is_file() and out_path.stat().st_size > 1000:
                try:
                    pcm, rate_in = _pcm16_from_wav(out_path)
                    if rate_in != sample_rate:
                        pcm_f = _resample_mono(
                            pcm.astype(np.float32) / 32768.0, rate_in, sample_rate
                        )
                        pcm16 = np.clip(pcm_f * 32767.0, -32768, 32767).astype(np.int16)
                        with wave.open(str(out_path), "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(sample_rate)
                            wf.writeframes(pcm16.tobytes())
                    written.append(out_path)
                except Exception:
                    out_path.unlink(missing_ok=True)
    except Exception:
        return written
    return written


def ensure_wake_refs(
    refs_dir: Path | None = None,
    *,
    sample_rate: int = 16000,
    auto_tts: bool = True,
) -> Path:
    """Ensure refs dir exists; optionally synthesize SAPI samples if empty."""
    d = Path(refs_dir) if refs_dir else default_refs_dir()
    d.mkdir(parents=True, exist_ok=True)
    wavs = list(d.glob("*.wav"))
    if not wavs and auto_tts:
        synthesize_wake_refs_windows(d, sample_rate=sample_rate)
    return d


@dataclass
class KwsHit:
    score: float
    threshold: float


class EmbeddingKws:
    """Streaming energy-gated MFCC cosine matcher."""

    def __init__(
        self,
        references: list[np.ndarray],
        *,
        sample_rate: int = 16000,
        threshold: float = 0.82,
        window_s: float = 1.6,
        hop_s: float = 0.3,
        energy_threshold: float = 0.012,
        block: int = 480,
    ) -> None:
        self.references = list(references)
        self.sample_rate = sample_rate
        self.threshold = float(threshold)
        self.window_n = max(block, int(window_s * sample_rate))
        self.hop_n = max(block, int(hop_s * sample_rate))
        self.energy_threshold = float(energy_threshold)
        self.block = block
        self._ring: deque[np.ndarray] = deque()
        self._total = 0
        self._since_hop = 0
        self._cooldown = 0

    @classmethod
    def from_refs_dir(
        cls,
        refs_dir: Path | str,
        *,
        sample_rate: int = 16000,
        threshold: float = 0.82,
        window_s: float = 1.6,
        hop_s: float = 0.3,
        energy_threshold: float = 0.012,
        block: int = 480,
        auto_tts: bool = True,
    ) -> EmbeddingKws:
        d = ensure_wake_refs(Path(refs_dir), sample_rate=sample_rate, auto_tts=auto_tts)
        refs = load_reference_embeddings(d, sample_rate=sample_rate)
        return cls(
            refs,
            sample_rate=sample_rate,
            threshold=threshold,
            window_s=window_s,
            hop_s=hop_s,
            energy_threshold=energy_threshold,
            block=block,
        )

    @property
    def ready(self) -> bool:
        return len(self.references) > 0

    def reset(self) -> None:
        self._ring.clear()
        self._total = 0
        self._since_hop = 0
        self._cooldown = int(0.8 * self.sample_rate / max(1, self.block)) * self.block

    def best_score(self, pcm16: np.ndarray) -> float:
        if not self.references:
            return 0.0
        emb = mfcc_embedding(pcm16, sample_rate=self.sample_rate)
        return max(cosine_similarity(emb, ref) for ref in self.references)

    def process(self, chunk: np.ndarray) -> KwsHit | None:
        """Feed one mic block; return hit when wake phrase likely detected."""
        if not self.references:
            return None
        arr = np.asarray(chunk, dtype=np.int16).reshape(-1)
        if self._cooldown > 0:
            self._cooldown = max(0, self._cooldown - arr.size)
            return None

        self._ring.append(arr)
        self._total += arr.size
        self._since_hop += arr.size
        while self._total > self.window_n + self.block * 2:
            dropped = self._ring.popleft()
            self._total -= dropped.size

        if self._since_hop < self.hop_n or self._total < int(0.5 * self.sample_rate):
            return None
        self._since_hop = 0

        pcm = np.concatenate(list(self._ring))[-self.window_n :]
        if frame_rms(pcm) < self.energy_threshold:
            return None
        score = self.best_score(pcm)
        if score >= self.threshold:
            self.reset()
            return KwsHit(score=score, threshold=self.threshold)
        return None


def score_phrase_against_refs(
    pcm16: np.ndarray,
    references: list[np.ndarray],
    *,
    sample_rate: int = 16000,
) -> float:
    """Unit-test helper: max cosine score of clip vs reference embeddings."""
    if not references:
        return 0.0
    emb = mfcc_embedding(pcm16, sample_rate=sample_rate)
    return max(cosine_similarity(emb, ref) for ref in references)
