"""Model management helpers for local sherpa-onnx STT."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from krabobot.config.schema import STTConfig
from krabobot.utils.helpers import ensure_dir


def ensure_sherpa_stt_model(cfg: STTConfig) -> None:
    """Ensure configured sherpa-onnx STT model exists locally."""
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
    out_dir = ensure_dir(Path(model_dir).expanduser().resolve())
    has_tokens = (out_dir / "tokens.txt").is_file()
    has_encoder = any(out_dir.glob("*encoder*.onnx"))
    has_decoder = any(out_dir.glob("*decoder*.onnx"))
    has_joiner = any(out_dir.glob("*joiner*.onnx"))
    if has_tokens and has_encoder and has_decoder and has_joiner:
        return

    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "huggingface_hub is not installed; cannot auto-download sherpa STT model {}",
            model_id,
        )
        return

    logger.info("Downloading sherpa-onnx STT model {} to {}", model_id, out_dir)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(out_dir),
        allow_patterns=[
            "*.onnx",
            "tokens.txt",
            "*.txt",
            "*.md",
        ],
    )
