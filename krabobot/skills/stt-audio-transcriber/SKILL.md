---
name: stt-audio-transcriber
description: Transcribe short and long audio files with local STT (GigaAM ONNX) while preserving original files.
---

# STT Audio Transcriber

Use this skill when the user asks to transcribe voice/audio files, especially in Russian.

## Goals

- Produce accurate transcript text.
- Keep the original audio file untouched.
- Handle both short and long recordings.

## Workflow

1. Verify that the source audio file exists and is readable.
2. If the file is short (about up to 60 seconds), transcribe directly.
3. If the file is long, split into time chunks (for example 30-60s with overlap), transcribe each chunk, then merge in order.
4. Return:
   - full transcript,
   - optional chunked transcript with timestamps,
   - short summary on request.

## Important rules

- Never delete or overwrite the original audio file.
- If a temporary converted/chunked file is needed, store it in the current workspace.
- If transcription quality is poor, mention likely causes (noise, music, low bitrate, multiple speakers).
- If STT backend is unavailable, clearly report missing runtime/dependencies and suggest exact install steps.

## krabobot-specific notes

- Channel ingestion may include `[voice: /path/to/file]` or media paths in context.
- Prefer those paths directly for STT.
- For very long files, process in chunks and provide a stitched final text.

