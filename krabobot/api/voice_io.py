"""Shared STT/TTS helpers for the voice HTTP channel (no BaseChannel required)."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from loguru import logger

from krabobot.config.schema import STTConfig, TTSConfig


def resolve_sherpa_stt_model_dir(cfg: STTConfig | None = None) -> str:
    """Resolve sherpa-onnx STT model directory from config/env."""
    if cfg is not None:
        base = Path(cfg.sherpa_models_dir).expanduser().resolve()
        model_id = (cfg.sherpa_model_id or "").strip()
        model_name = model_id.split("/", 1)[-1] if "/" in model_id else model_id
        if model_name:
            return str((base / model_name).resolve())
        return str(base.resolve())
    default_dir = str(
        (
            Path.home()
            / ".krabobot"
            / "models"
            / "stt"
            / "sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16"
        ).resolve()
    )
    return (os.getenv("SHERPA_STT_MODEL_DIR", default_dir) or "").strip()


def resolve_sherpa_tts_model_dir(cfg: TTSConfig | None = None) -> str:
    """Resolve sherpa-onnx TTS model directory from config/env."""
    if cfg is not None:
        base = Path(cfg.sherpa_models_dir).expanduser().resolve()
        model_id = (cfg.sherpa_model_id or "").strip()
        model_name = model_id.split("/", 1)[-1] if "/" in model_id else model_id
        if model_name:
            return str((base / model_name).resolve())
        return str(base.resolve())
    default_dir = str(
        (Path.home() / ".krabobot" / "models" / "tts" / "vits-piper-ru_RU-irina-medium").resolve()
    )
    return (os.getenv("SHERPA_TTS_MODEL_DIR", default_dir) or "").strip()


async def transcribe_audio(
    file_path: str | Path,
    *,
    stt: STTConfig | None = None,
) -> tuple[str, str | None]:
    """Transcribe audio via sherpa-onnx. Returns ``(text, error)``."""
    path = Path(file_path)
    try:
        size = path.stat().st_size if path.is_file() else 0
    except OSError:
        size = 0
    logger.info("voice STT: path={} bytes={}", path.name, size)
    try:
        from krabobot.stt.audio_preprocess import align_waveform_float, rewrite_wav_16k_mono
        from krabobot.stt.sherpa_onnx_stt import SherpaOnnxTranscriber

        # Rewrite upload as aligned 16 kHz mono before decode (header/level/trim).
        try:
            raw_wave = await asyncio.to_thread(SherpaOnnxTranscriber._load_audio_16k_mono, path)
            aligned = align_waveform_float(raw_wave, sample_rate=16000)
            await asyncio.to_thread(rewrite_wav_16k_mono, path, aligned)
            try:
                size = path.stat().st_size if path.is_file() else size
            except OSError:
                pass
            logger.info(
                "voice STT aligned: path={} bytes={} samples={}",
                path.name,
                size,
                int(aligned.size),
            )
        except Exception as e:
            logger.warning("voice STT preprocess skipped: {}", e)

        model_dir = resolve_sherpa_stt_model_dir(stt)
        num_threads = int(max(1, stt.sherpa_num_threads)) if stt else 2
        provider = (stt.sherpa_provider if stt else "cpu") or "cpu"
        text = await asyncio.to_thread(
            SherpaOnnxTranscriber.transcribe,
            path,
            model_dir=model_dir,
            num_threads=num_threads,
            provider=str(provider).strip() or "cpu",
        )
        if text:
            return text, None
        hint = (
            f"sherpa_onnx returned empty transcription "
            f"(audio_bytes={size}; speak after the beep, ≥1–2s)"
        )
        logger.warning("voice STT empty: {}", hint)
        return "", hint
    except Exception as e:
        logger.warning("voice STT failed: {}", e)
        return "", f"sherpa_onnx failed: {e}"


async def synthesize_speech_wav(
    text: str,
    *,
    tts: TTSConfig | None = None,
) -> bytes | None:
    """Synthesize speech via sherpa-onnx and return WAV bytes."""
    clean = (text or "").strip()
    if not clean:
        return None
    from krabobot.tts.sherpa_onnx_tts import SherpaOnnxTTS

    model_dir = resolve_sherpa_tts_model_dir(tts)
    if not model_dir:
        logger.warning("voice TTS: sherpa model dir is not configured")
        return None
    speed = 1.0
    if tts is not None:
        speed = max(0.5, min(2.0, float(tts.sherpa_speed)))
    fd, out_path = tempfile.mkstemp(prefix="voice_tts_", suffix=".wav")
    os.close(fd)
    try:
        await asyncio.to_thread(
            SherpaOnnxTTS.synthesize_to_wav,
            text=clean,
            model_dir=model_dir,
            out_path=out_path,
            speed=speed,
            sid=0,
        )
        return Path(out_path).read_bytes()
    except Exception as e:
        logger.warning("voice sherpa-onnx TTS failed: {}", e)
        return None
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass
