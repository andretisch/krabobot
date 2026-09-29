"""Local sherpa-onnx ASR for wake-word windows (reuses ~/.krabobot STT models)."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np


def default_stt_model_dir() -> Path:
    return (
        Path.home()
        / ".krabobot"
        / "models"
        / "stt"
        / "sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16"
    )


def resolve_stt_model_dir(explicit: str | None = None) -> Path:
    if explicit and explicit.strip():
        p = Path(explicit).expanduser().resolve()
        if (p / "tokens.txt").is_file():
            return p
        raise FileNotFoundError(f"STT model dir missing tokens.txt: {p}")
    base = Path.home() / ".krabobot" / "models" / "stt"
    preferred = default_stt_model_dir()
    if (preferred / "tokens.txt").is_file():
        return preferred
    if base.is_dir():
        for child in sorted(base.iterdir()):
            if child.is_dir() and (child / "tokens.txt").is_file():
                return child
    raise FileNotFoundError(
        f"No sherpa STT model under {base}. Run `krabobot serve` once to download models."
    )


def _pick(base: Path, patterns: list[str], label: str) -> Path:
    for patt in patterns:
        found = sorted(base.glob(patt))
        if found:
            return found[0]
    raise FileNotFoundError(f"{label} onnx not found in {base}")


class LocalWakeAsr:
    """Reusable OfflineRecognizer for short wake windows."""

    def __init__(self, model_dir: str | Path, *, num_threads: int = 0, provider: str = "cpu") -> None:
        import sherpa_onnx  # type: ignore[import-not-found]

        from krabobot_voice.config import resolve_stt_num_threads

        base = Path(model_dir).expanduser().resolve()
        tokens = base / "tokens.txt"
        if not tokens.is_file():
            raise FileNotFoundError(f"tokens.txt not found in {base}")
        encoder = _pick(base, ["*encoder*.onnx", "encoder*.onnx"], "encoder")
        decoder = _pick(base, ["*decoder*.onnx", "decoder*.onnx"], "decoder")
        joiner = _pick(base, ["*joiner*.onnx", "joiner*.onnx"], "joiner")
        threads = resolve_stt_num_threads(num_threads)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            tokens=str(tokens),
            num_threads=threads,
            sample_rate=16000,
            feature_dim=80,
            provider=provider,
            decoding_method="greedy_search",
            model_type="nemo_transducer",
        )
        self.model_dir = base
        self.num_threads = threads

    def transcribe_pcm16(
        self,
        pcm16: np.ndarray | bytes,
        *,
        sample_rate: int = 16000,
        energy_threshold: float = 0.006,
    ) -> str:
        """Decode after alignment: DC → 16 kHz mono → trim → normalize → sherpa."""
        from krabobot_voice.preprocess import TARGET_SR, preprocess_pcm16

        if isinstance(pcm16, (bytes, bytearray)):
            arr = np.frombuffer(pcm16, dtype=np.int16)
        else:
            arr = np.asarray(pcm16, dtype=np.int16).reshape(-1)
        if arr.size < max(1, int(sample_rate) // 10):
            return ""
        # Alignment MUST run before sherpa (quiet mic / DC / wrong rate).
        aligned, sr = preprocess_pcm16(
            arr,
            sample_rate=int(sample_rate),
            energy_threshold=float(energy_threshold),
        )
        if aligned.size < TARGET_SR // 10:
            return ""
        waveform = (aligned.astype(np.float32) / 32768.0).tolist()
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sr, waveform)
        self._recognizer.decode_stream(stream)
        result = stream.result
        return str(getattr(result, "text", "") or "").strip()

    def transcribe_wav_bytes(
        self,
        wav_bytes: bytes,
        *,
        energy_threshold: float = 0.006,
    ) -> str:
        from krabobot_voice.preprocess import preprocess_wav_bytes

        # Align header/rate/level first, then decode.
        aligned = preprocess_wav_bytes(wav_bytes, energy_threshold=float(energy_threshold))
        with wave.open(io.BytesIO(aligned), "rb") as wf:
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            pcm = np.frombuffer(frames, dtype=np.int16)
        return self.transcribe_pcm16(pcm, sample_rate=rate, energy_threshold=energy_threshold)
