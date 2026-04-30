"""Local TTS backend using sherpa-onnx."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any


class SherpaOnnxTTS:
    """Thin wrapper over sherpa-onnx offline TTS API."""

    @staticmethod
    def _ctor_kwargs(cls: type, **kwargs: Any) -> dict[str, Any]:
        """Filter kwargs to only those accepted by a constructor."""
        try:
            sig = inspect.signature(cls)
        except (TypeError, ValueError):
            return kwargs
        accepted = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in accepted}

    @classmethod
    def synthesize_to_wav(
        cls,
        *,
        text: str,
        model_dir: str | Path,
        out_path: str | Path,
        speed: float = 1.0,
        sid: int = 0,
    ) -> None:
        """Generate speech into a WAV file."""
        import sherpa_onnx  # type: ignore[import-not-found]

        base = Path(model_dir).expanduser().resolve()
        model = cls._resolve_onnx_model(base)
        tokens = base / "tokens.txt"
        lexicon = base / "lexicon.txt"
        data_dir = base / "espeak-ng-data"
        dict_dir = base / "dict"

        if not model.is_file():
            raise FileNotFoundError(f"sherpa-onnx TTS model not found: {model}")
        if not tokens.is_file():
            raise FileNotFoundError(f"sherpa-onnx TTS tokens not found: {tokens}")

        vits_kwargs = {
            "model": str(model),
            "tokens": str(tokens),
            "lexicon": str(lexicon) if lexicon.is_file() else "",
            "data_dir": str(data_dir) if data_dir.is_dir() else "",
            "dict_dir": str(dict_dir) if dict_dir.is_dir() else "",
        }
        vits_cfg = sherpa_onnx.OfflineTtsVitsModelConfig(
            **cls._ctor_kwargs(sherpa_onnx.OfflineTtsVitsModelConfig, **vits_kwargs)
        )
        model_cfg = sherpa_onnx.OfflineTtsModelConfig(vits=vits_cfg, provider="cpu", debug=False)
        tts_cfg = sherpa_onnx.OfflineTtsConfig(model=model_cfg)
        tts = sherpa_onnx.OfflineTts(tts_cfg)

        generated = tts.generate(text, sid=sid, speed=float(speed))
        samples = generated.samples
        sample_rate = generated.sample_rate
        sherpa_onnx.write_wave(str(out_path), samples=samples, sample_rate=sample_rate)

    @staticmethod
    def _resolve_onnx_model(model_dir: Path) -> Path:
        """Resolve ONNX model filename in a sherpa/piper model directory."""
        direct = model_dir / "model.onnx"
        if direct.is_file():
            return direct
        onnx_files = sorted(model_dir.glob("*.onnx"))
        if not onnx_files:
            return direct
        preferred = [p for p in onnx_files if "vits" in p.name.lower() or "piper" in p.name.lower()]
        if preferred:
            return preferred[0]
        return onnx_files[0]
