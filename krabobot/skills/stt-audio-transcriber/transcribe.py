#!/usr/bin/env python3
"""Transcribe audio with krabobot sherpa-onnx STT; write transcript.txt (+ optional progress)."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

from krabobot.config.loader import load_config
from krabobot.stt.model_manager import ensure_sherpa_stt_model
from krabobot.stt.sherpa_onnx_stt import SherpaOnnxTranscriber


def _model_dir(cfg) -> Path:
    ensure_sherpa_stt_model(cfg.stt)
    model_id = (cfg.stt.sherpa_model_id or "").strip()
    model_name = model_id.split("/", 1)[-1] if "/" in model_id else model_id
    return Path(cfg.stt.sherpa_models_dir).expanduser() / model_name


def transcribe_oneshot(src: Path, transcript_path: Path, cfg) -> str:
    text = SherpaOnnxTranscriber.transcribe(
        src,
        model_dir=_model_dir(cfg),
        num_threads=cfg.stt.sherpa_num_threads,
        provider=cfg.stt.sherpa_provider or "cpu",
    )
    transcript_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
    return text


def transcribe_chunked(
    wav_path: Path,
    transcript_path: Path,
    progress_path: Path,
    cfg,
    *,
    chunk_sec: int = 45,
    sample_rate: int = 16000,
) -> Path:
    """Chunked STT: reuses one recognizer; persists transcript + progress after each chunk."""
    import sherpa_onnx  # type: ignore[import-not-found]

    model = SherpaOnnxTranscriber._resolve_transducer_model(_model_dir(cfg))
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=model["encoder"],
        decoder=model["decoder"],
        joiner=model["joiner"],
        tokens=model["tokens"],
        num_threads=int(max(1, cfg.stt.sherpa_num_threads or 2)),
        sample_rate=sample_rate,
        feature_dim=80,
        provider=cfg.stt.sherpa_provider or "cpu",
        decoding_method="greedy_search",
        model_type="nemo_transducer",
    )

    with wave.open(str(wav_path), "rb") as wf:
        if not (
            wf.getframerate() == sample_rate
            and wf.getnchannels() == 1
            and wf.getsampwidth() == 2
        ):
            raise SystemExit(
                f"expected {sample_rate} Hz mono 16-bit wav; got "
                f"rate={wf.getframerate()} ch={wf.getnchannels()} width={wf.getsampwidth()}"
            )
        nframes = wf.getnframes()
        duration = nframes / sample_rate
        chunk_frames = chunk_sec * sample_rate
        total = (nframes + chunk_frames - 1) // chunk_frames
        start = 0
        if progress_path.exists():
            start = int(json.loads(progress_path.read_text(encoding="utf-8")).get("next_chunk", 0))
        parts: list[str] = []
        if start and transcript_path.exists():
            parts = transcript_path.read_text(encoding="utf-8").split("\n\n")

        for i in range(start, total):
            wf.setpos(i * chunk_frames)
            arr = np.frombuffer(wf.readframes(chunk_frames), dtype=np.int16).astype(np.float32)
            arr = arr / 32768.0
            if arr.size == 0:
                break
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, arr.tolist())
            recognizer.decode_stream(stream)
            text = str(getattr(stream.result, "text", "") or "").strip()
            parts.append(text)
            body = "\n\n".join(p for p in parts if p)
            transcript_path.write_text(body + ("\n" if body else ""), encoding="utf-8")
            processed = min(duration, (i * chunk_frames + arr.size) / sample_rate)
            progress_path.write_text(
                json.dumps(
                    {
                        "next_chunk": i + 1,
                        "total_chunks": total,
                        "processed_s": processed,
                        "duration_s": duration,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[{i + 1}/{total}] {processed:.1f}s chars={len(text)}", flush=True)

    return transcript_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="Wav (chunked) or short media path (oneshot)")
    p.add_argument("out_dir", type=Path, help="Output folder for transcript.txt")
    p.add_argument(
        "--chunked",
        action="store_true",
        help="Chunked mode for long 16 kHz mono wav (default if duration unknown: use for ~>60s)",
    )
    p.add_argument("--chunk-sec", type=int, default=45, help="Chunk length in seconds (default 45)")
    args = p.parse_args(argv)

    src = args.input.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not src.is_file():
        print(f"input not found: {src}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = out_dir / "transcript.txt"
    cfg = load_config()

    if args.chunked:
        progress_path = out_dir / "stt_progress.json"
        path = transcribe_chunked(
            src,
            transcript_path,
            progress_path,
            cfg,
            chunk_sec=max(1, args.chunk_sec),
        )
        print("DONE", path)
        return 0

    text = transcribe_oneshot(src, transcript_path, cfg)
    print(transcript_path)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
