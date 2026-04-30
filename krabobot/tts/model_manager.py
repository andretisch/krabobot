"""Model management helpers for local sherpa-onnx TTS."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from krabobot.config.schema import TTSConfig
from krabobot.utils.helpers import ensure_dir


def ensure_sherpa_tts_models(cfg: TTSConfig) -> None:
    """Ensure configured sherpa-onnx TTS models exist locally."""
    if (cfg.provider or "").strip().lower() not in {"sherpa", "sherpa_onnx", "sherpa-onnx"}:
        return
    if not cfg.auto_download_models:
        return
    model_id = (cfg.sherpa_model_id or "").strip()
    if not model_id:
        return
    base = Path(cfg.sherpa_models_dir).expanduser().resolve()
    model_name = model_id.split("/", 1)[-1] if "/" in model_id else model_id
    model_dir = str((base / model_name).resolve())
    _ensure_model(model_id, model_dir)


def _ensure_model(model_id: str, model_dir: str) -> None:
    model_id = (model_id or "").strip()
    if not model_id:
        return
    out_dir = ensure_dir(Path(model_dir).expanduser().resolve())
    model_file = out_dir / "model.onnx"
    tokens_file = out_dir / "tokens.txt"
    any_onnx = any(out_dir.glob("*.onnx"))
    if (model_file.is_file() or any_onnx) and tokens_file.is_file():
        return
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "huggingface_hub is not installed; cannot auto-download sherpa TTS model {}",
            model_id,
        )
        return
    logger.info("Downloading sherpa-onnx TTS model {} to {}", model_id, out_dir)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(out_dir),
        allow_patterns=[
            "*.onnx",
            "tokens.txt",
            "lexicon.txt",
            "espeak-ng-data/**",
            "dict/**",
        ],
    )
