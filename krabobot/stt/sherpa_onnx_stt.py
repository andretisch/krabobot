"""Local STT backend using sherpa-onnx offline transducer."""

from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np


class SherpaOnnxTranscriber:
    """Wrapper around sherpa-onnx OfflineRecognizer for file transcription."""

    @classmethod
    def transcribe(
        cls,
        file_path: str | Path,
        *,
        model_dir: str | Path,
        num_threads: int = 2,
        provider: str = "cpu",
    ) -> str:
        import sherpa_onnx  # type: ignore[import-not-found]

        model = cls._resolve_transducer_model(model_dir)
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=model["encoder"],
            decoder=model["decoder"],
            joiner=model["joiner"],
            tokens=model["tokens"],
            num_threads=int(max(1, num_threads)),
            sample_rate=16000,
            feature_dim=80,
            provider=provider,
            decoding_method="greedy_search",
            model_type="nemo_transducer",
        )
        waveform = cls._load_audio_16k_mono(file_path)
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, waveform.tolist())
        recognizer.decode_stream(stream)
        result = stream.result
        return str(getattr(result, "text", "") or "").strip()

    @staticmethod
    def _resolve_transducer_model(model_dir: str | Path) -> dict[str, str]:
        base = Path(model_dir).expanduser().resolve()
        tokens = base / "tokens.txt"
        if not tokens.is_file():
            raise FileNotFoundError(f"tokens.txt not found in {base}")

        def _pick(patterns: list[str], label: str) -> Path:
            for patt in patterns:
                found = sorted(base.glob(patt))
                if found:
                    return found[0]
            raise FileNotFoundError(f"{label} onnx not found in {base}")

        encoder = _pick(["*encoder*.onnx", "encoder*.onnx"], "encoder")
        decoder = _pick(["*decoder*.onnx", "decoder*.onnx"], "decoder")
        joiner = _pick(["*joiner*.onnx", "joiner*.onnx"], "joiner")
        return {
            "encoder": str(encoder),
            "decoder": str(decoder),
            "joiner": str(joiner),
            "tokens": str(tokens),
        }

    @staticmethod
    def _load_audio_16k_mono(file_path: str | Path) -> np.ndarray:
        """Load arbitrary audio into mono float32 waveform in [-1, 1] at 16k."""
        src = Path(file_path).expanduser().resolve()
        if src.suffix.lower() == ".wav":
            try:
                with wave.open(str(src), "rb") as wf:
                    rate = wf.getframerate()
                    channels = wf.getnchannels()
                    width = wf.getsampwidth()
                    frames = wf.readframes(wf.getnframes())
                if rate == 16000 and channels == 1 and width == 2:
                    arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    return arr
            except Exception:
                pass

        ffmpeg = "ffmpeg"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            cmd = [
                ffmpeg,
                "-y",
                "-i",
                str(src),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                tmp.name,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with wave.open(tmp.name, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
            return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
