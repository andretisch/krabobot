---
name: stt-audio-transcriber
description: Transcribe audio/video with built-in sherpa-onnx STT and always save wav + full transcript on disk. For summaries or meeting minutes, use the summarize skill.
---

# STT Audio Transcriber

## Goal (priority order)

1. **Primary:** extract audio + transcription + **save source artifacts** on disk:
   - keep the **original media** untouched (never delete/overwrite);
   - save extracted **16 kHz mono wav** (or note the existing wav path if already suitable);
   - save the **full transcript** to `transcript.txt` (persist during chunked STT).
2. **Not this skill:** short summary or formal meeting report — use skill **`summarize`** on `transcript.txt` if needed.

Default deliverables: `audio_16k_mono.wav` + `transcript.txt`.

## Reality check (anti-blocker)

- Helper scripts live **next to this SKILL.md** — `read_file` them if needed, then **`exec`** with Python. Runtime STT is krabobot **sherpa-onnx** + ffmpeg via `krabobot.utils.ffmpeg.resolve_ffmpeg_exe` / imageio-ffmpeg.
- **Do NOT** refuse because scripts are “missing” without checking this folder, or invent «нет ffmpeg» without calling `resolve_ffmpeg_exe()` (scripts do this).
- No agent tool named «transcribe». Use the scripts below.
- Channel auto-STT (Telegram/VK voice) may already put text in context. Web/uploads and video usually **do not** — run STT yourself.
- If the file is already under workspace `uploads/` (or any given path), **work that path**. Do not ask the user to re-upload first.
- Large / multi-GB media: never load the whole file into RAM — extract wav, then `--chunked` STT. One-shot `transcribe()` is OK only for short clips (~≤60s).

## Scripts (same directory as this file)

Let `{base}` = this skill directory (path of this `SKILL.md`).

| Script | Purpose |
|--------|---------|
| `{base}/extract_wav.py` | Probe duration/streams; extract `audio_16k_mono.wav` |
| `{base}/transcribe.py` | One-shot or `--chunked` STT → `transcript.txt` (+ `stt_progress.json`) |

```bash
python "{base}/extract_wav.py" --probe-only "PATH/TO/media"
python "{base}/extract_wav.py" "PATH/TO/media" "PATH/TO/out_folder"
python "{base}/transcribe.py" "PATH/TO/out_folder/audio_16k_mono.wav" "PATH/TO/out_folder"
python "{base}/transcribe.py" --chunked "PATH/TO/out_folder/audio_16k_mono.wav" "PATH/TO/out_folder"
```

(`--chunked` for long wav; resume via existing `stt_progress.json` + `transcript.txt`.)

## Workflow

1. Confirm source exists; probe with `extract_wav.py --probe-only`.
2. Create an out folder next to the source (e.g. `.../TOIR_stt/`). Never overwrite the original.
3. Extract 16 kHz mono wav into that folder (or note path if source is already suitable).
4. STT: short → oneshot `transcribe.py`; long (~>60s) → `--chunked` (persists after each chunk). Prefer `exec` with `background=true` so completion returns to this session; use **cron** only if needed for recovery (e.g. after gateway restart).
5. Done when `transcript.txt` exists. For a short summary or meeting minutes, use skill **`summarize`**.
6. If the session cannot finish: save **partial** transcript and state how much was done.

## Output layout (example)

```text
Videos/TOIR_stt/
  audio_16k_mono.wav
  transcript.txt
  stt_progress.json    # chunked resume
# original media stays untouched
```

## Rules

- Never delete or overwrite the original media.
- Always save wav (or note path) + `transcript.txt` under the out folder.
- Refuse only after a real failed attempt (`resolve_ffmpeg_exe()` is `None` / model missing / tool error). Then report the exact error (`pip install imageio-ffmpeg`; models under `~/.krabobot/models/stt`).
- Poor quality: mention noise/music/low bitrate/overlap — after a transcript attempt.
