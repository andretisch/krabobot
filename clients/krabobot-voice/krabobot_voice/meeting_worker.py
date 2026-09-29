"""Standalone meeting capture process.

Run::

    python -m krabobot_voice.meeting_worker --out PATH --stop-file PATH \\
        [--capture mix] [--sample-rate 16000] ...

Writes a mono PCM16 WAV to ``--out`` until ``--stop-file`` appears (or max duration).
Used by ``MeetingRecorder`` via ``multiprocessing``; can also be launched as a
plain subprocess for crash isolation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stop_file = Path(args.stop_file)
    status_file = Path(args.status_file) if args.status_file else None

    # Local import keeps CLI startup light and matches MeetingRecorder config.
    from krabobot_voice.meeting import MeetingCaptureConfig, run_meeting_capture

    cfg = MeetingCaptureConfig(
        capture=args.capture,
        sample_rate=int(args.sample_rate),
        block_ms=int(args.block_ms),
        input_device=args.input_device or "",
        output_device=args.output_device or "",
        loopback_device=args.loopback_device or "",
        max_s=float(args.max_s),
    )

    class _FileStop:
        def is_set(self) -> bool:
            return stop_file.is_file()

    def _status(payload: dict) -> None:
        if status_file is None:
            return
        try:
            status_file.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    _status({"active": True, "out": str(out), "capture": cfg.capture})
    try:
        result = run_meeting_capture(
            cfg,
            out,
            stop=_FileStop(),
            mic_tap_queue=None,
            on_progress=lambda frames, dur: _status(
                {
                    "active": True,
                    "out": str(out),
                    "capture": cfg.capture,
                    "frames": frames,
                    "duration_s": round(dur, 2),
                }
            ),
        )
    except Exception as e:  # pragma: no cover - device failures
        _status({"active": False, "error": str(e), "out": str(out)})
        print(f"meeting_worker ERROR: {e}", file=sys.stderr)
        return 1

    err = str(result.get("error") or "")
    _status(
        {
            "active": False,
            "out": str(out),
            "capture": result.get("capture") or cfg.capture,
            "frames": int(result.get("frames") or 0),
            "duration_s": float(result.get("duration_s") or 0.0),
            "error": err,
        }
    )
    if err:
        print(f"meeting_worker ERROR: {err}", file=sys.stderr)
        return 1
    return 0


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="krabobot-voice meeting capture worker")
    p.add_argument("--out", required=True, help="Output WAV path")
    p.add_argument("--stop-file", required=True, help="Create this file to stop")
    p.add_argument("--status-file", default="", help="Optional JSON status path")
    p.add_argument("--capture", default="mix", choices=("mic", "loopback", "mix"))
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--block-ms", type=int, default=30)
    p.add_argument("--input-device", default="")
    p.add_argument("--output-device", default="")
    p.add_argument("--loopback-device", default="")
    p.add_argument("--max-s", type=float, default=7200.0)
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
