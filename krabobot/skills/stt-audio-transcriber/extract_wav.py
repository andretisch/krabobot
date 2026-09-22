#!/usr/bin/env python3
"""Extract 16 kHz mono PCM wav from audio/video (streamed via ffmpeg; no full-file RAM load)."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from krabobot.utils.ffmpeg import resolve_ffmpeg_exe


def probe_media(ffmpeg: str, src: Path) -> list[str]:
    """Return ffmpeg stderr lines about Duration / Audio / Video streams."""
    r = subprocess.run(
        [ffmpeg, "-i", str(src)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return [
        line
        for line in r.stderr.splitlines()
        if "Duration" in line or ("Stream" in line and ("Audio" in line or "Video" in line))
    ]


def extract_wav(src: Path, out_dir: Path, *, wav_name: str = "audio_16k_mono.wav") -> Path:
    ffmpeg = resolve_ffmpeg_exe()
    if not ffmpeg:
        raise SystemExit(
            "ffmpeg unavailable after resolve_ffmpeg_exe(); install: pip install imageio-ffmpeg"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / wav_name
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-vn",
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(wav),
        ],
        check=True,
    )
    return wav


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="Source audio/video path")
    p.add_argument(
        "out_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Output folder (required unless --probe-only)",
    )
    p.add_argument(
        "--probe-only",
        action="store_true",
        help="Print Duration/stream lines and exit (no extract)",
    )
    p.add_argument(
        "--wav-name",
        default="audio_16k_mono.wav",
        help="Output wav filename inside out_dir (default: audio_16k_mono.wav)",
    )
    args = p.parse_args(argv)

    src = args.input.expanduser().resolve()
    if not src.is_file():
        print(f"input not found: {src}", file=sys.stderr)
        return 1

    ffmpeg = resolve_ffmpeg_exe()
    if not ffmpeg:
        print(
            "ffmpeg unavailable after resolve_ffmpeg_exe(); install: pip install imageio-ffmpeg",
            file=sys.stderr,
        )
        return 1

    info = probe_media(ffmpeg, src)
    for line in info:
        print(line)

    if args.probe_only:
        joined = "\n".join(info)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", joined)
        if m:
            h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            print(f"duration_s={h * 3600 + mi * 60 + s:.3f}")
        return 0

    if args.out_dir is None:
        print("out_dir is required unless --probe-only", file=sys.stderr)
        return 2

    wav = extract_wav(src, args.out_dir.expanduser().resolve(), wav_name=args.wav_name)
    print(wav)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
