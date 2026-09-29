"""Console ASR/PTT → beep → record → POST /v1/voice/turn → play loop.

Wake default: VAD utterance → local sherpa-onnx ASR (user-configured wake.phrase).
Optional KWS mode scores the same VAD segments. Dialog follow-up after Talk reply.
Meeting: toggle capture.
"""

from __future__ import annotations

import os
import sys
import time
from enum import Enum

import numpy as np

from krabobot_voice.audio_io import open_capture_stream, play_beep, play_wav_bytes, with_stream_keepalive
from krabobot_voice.commands import LocalCommand, match_local_command
from krabobot_voice.config import VoiceClientConfig
from krabobot_voice.dialog import (
    TalkPhase,
    plan_after_no_speech,
    plan_after_turn,
    plan_initial_listen,
)
from krabobot_voice.http_client import VoiceHttpClient
from krabobot_voice.kws import EmbeddingKws, default_refs_dir
from krabobot_voice.link import ensure_device_linked
from krabobot_voice.meeting import MeetingCaptureConfig, MeetingRecorder
from krabobot_voice.ptt import PttHotkey
from krabobot_voice.vad import frame_rms, pcm_stats, record_utterance
from krabobot_voice.wake import command_after_wake, matches_wake_phrase

# Loopback after downmix/resample is often quieter than a close mic.
_LOOPBACK_WAKE_ENERGY_SCALE = 0.4
_LOOPBACK_WAKE_ENERGY_FLOOR = 0.0015
_WAKE_DEBUG_HEARTBEAT_S = 30.0
_LOOPBACK_SILENCE_WARN_S = 25.0


def _log(msg: str) -> None:
    print(msg, flush=True)


def _debug_enabled() -> bool:
    return os.environ.get("KRABOBOT_VOICE_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _log_debug(msg: str) -> None:
    if _debug_enabled():
        _log(f"[debug] {msg}")


def effective_wake_energy(threshold: float, *, listen_source: str) -> float:
    """Lower energy gate for WASAPI loopback (system audio RMS is often low)."""
    thr = float(threshold)
    if (listen_source or "").strip().lower() != "loopback":
        return thr
    return min(thr, max(_LOOPBACK_WAKE_ENERGY_FLOOR, thr * _LOOPBACK_WAKE_ENERGY_SCALE))


class Trigger(str, Enum):
    WAKE = "wake"
    PTT = "ptt"
    MEETING = "meeting"


class _AbortWakeWaitError(Exception):
    """Raised from a wrapped read_block when PTT / meeting wins the wake wait."""

    def __init__(self, trigger: Trigger, detail: str) -> None:
        super().__init__(detail)
        self.trigger = trigger
        self.detail = detail


def _interruptible_read_block(
    mic: object,
    *,
    ptt: PttHotkey | None,
    meeting: PttHotkey | None,
):
    """Return a read_block that also polls PTT / meeting hotkeys."""
    read_block = getattr(mic, "read_block")

    def _read() -> np.ndarray:
        if meeting is not None and meeting.consume_edge_down():
            raise _AbortWakeWaitError(Trigger.MEETING, "meeting")
        if ptt is not None and ptt.consume_edge_down():
            raise _AbortWakeWaitError(Trigger.PTT, "ptt")
        return read_block()

    return _read


def _capture_wake_utterance(
    mic: object,
    *,
    energy_threshold: float,
    silence_end_s: float,
    max_s: float,
    min_speech_s: float,
    speech_start_s: float,
    preroll_s: float,
    ptt: PttHotkey | None,
    meeting: PttHotkey | None,
) -> np.ndarray | None:
    """One VAD-closed utterance for wake, or None on silence timeout / near-silent.

    Raises ``_AbortWakeWaitError`` if PTT or meeting fires while capturing.
    """
    sr = int(getattr(mic, "sample_rate"))
    block = int(getattr(mic, "block"))
    return record_utterance(
        _interruptible_read_block(mic, ptt=ptt, meeting=meeting),
        sample_rate=sr,
        block=block,
        max_s=max_s,
        silence_end_s=silence_end_s,
        speech_start_s=speech_start_s,
        min_speech_s=min_speech_s,
        energy_threshold=energy_threshold,
        settle_s=0.0,
        preroll_s=preroll_s,
        # Outer loop restarts on None so idle wake keeps listening forever.
        no_speech_timeout_s=max(8.0, max_s),
    )


def _wait_for_wake_asr(
    mic: object,
    asr: object,
    *,
    energy_threshold: float,
    silence_end_s: float,
    max_s: float,
    min_speech_s: float,
    speech_start_s: float,
    preroll_s: float,
    ptt: PttHotkey | None,
    meeting: PttHotkey | None,
    phrases: list[str] | tuple[str, ...] | None = None,
    greetings: list[str] | tuple[str, ...] | None = None,
    listen_source: str = "mic",
) -> tuple[Trigger, str]:
    """Wake path: VAD speech_start→end → one sherpa ASR → phrase match.

    Misses discard the segment and immediately listen for the next VAD utterance.
    Status line is printed by the caller once on enter — not every hop.
    """
    phrase_list = tuple(phrases) if phrases is not None else None
    greet_list = tuple(greetings) if greetings is not None else None
    is_loopback = (listen_source or "").strip().lower() == "loopback"
    entered_at = time.monotonic()
    last_heartbeat = entered_at
    silence_warned = False
    saw_energy = False

    while True:
        try:
            pcm = _capture_wake_utterance(
                mic,
                energy_threshold=energy_threshold,
                silence_end_s=silence_end_s,
                max_s=max_s,
                min_speech_s=min_speech_s,
                speech_start_s=speech_start_s,
                preroll_s=preroll_s,
                ptt=ptt,
                meeting=meeting,
            )
        except _AbortWakeWaitError as abort:
            return abort.trigger, abort.detail

        now = time.monotonic()
        if now - last_heartbeat >= _WAKE_DEBUG_HEARTBEAT_S:
            _log_debug("still waiting for wake…")
            last_heartbeat = now

        if pcm is None:
            if is_loopback and not silence_warned and not saw_energy:
                if now - entered_at >= _LOOPBACK_SILENCE_WARN_S:
                    silence_warned = True
                    _log(
                        "WARNING: loopback RMS ниже wake energy — "
                        "увеличьте громкость видео / выберите audio.output_device "
                        "того выхода, куда играет звук "
                        f"(thr={energy_threshold:.4f}). "
                        "Диагностика: python -m krabobot_voice --test-loopback"
                    )
            continue

        if frame_rms(pcm) >= energy_threshold:
            saw_energy = True

        # Keep draining loopback/mic while sherpa runs — otherwise PortAudio
        # overflows and the next stream.read can hang forever (Ctrl+C ignored).
        def _transcribe() -> str:
            return asr.transcribe_pcm16(  # type: ignore[attr-defined]
                pcm,
                sample_rate=int(getattr(mic, "sample_rate")),
                energy_threshold=min(energy_threshold, 0.006),
            )

        text = with_stream_keepalive(mic, _transcribe)
        stripped = (text or "").strip()
        if stripped:
            _log(f"  [wake-asr] {stripped}")
        if matches_wake_phrase(stripped, phrases=phrase_list, greetings=greet_list):
            return Trigger.WAKE, stripped
        # Miss: discard segment; immediately ready for the next VAD utterance.


def _wait_for_wake_kws(
    mic: object,
    kws: EmbeddingKws,
    *,
    energy_threshold: float,
    silence_end_s: float,
    max_s: float,
    min_speech_s: float,
    speech_start_s: float,
    preroll_s: float,
    ptt: PttHotkey | None,
    meeting: PttHotkey | None,
) -> tuple[Trigger, str]:
    """KWS wake: VAD segment → one embedding score (no sliding-window spam)."""
    while True:
        try:
            pcm = _capture_wake_utterance(
                mic,
                energy_threshold=energy_threshold,
                silence_end_s=silence_end_s,
                max_s=max_s,
                min_speech_s=min_speech_s,
                speech_start_s=speech_start_s,
                preroll_s=preroll_s,
                ptt=ptt,
                meeting=meeting,
            )
        except _AbortWakeWaitError as abort:
            return abort.trigger, abort.detail

        if pcm is None:
            continue

        score = kws.best_score(pcm)
        if score >= kws.threshold:
            _log(f"  [kws] score={score:.3f} (thr={kws.threshold:.3f})")
            kws.reset()
            return Trigger.WAKE, f"kws:{score:.3f}"


def _wait_for_ptt_or_meeting(
    mic: object,
    *,
    ptt: PttHotkey | None,
    meeting: PttHotkey | None,
) -> Trigger:
    """Block until PTT or meeting (wake_mode=off)."""
    last_heartbeat = time.monotonic()
    read_block = getattr(mic, "read_block")
    while True:
        if meeting is not None and meeting.consume_edge_down():
            return Trigger.MEETING
        if ptt is not None and ptt.consume_edge_down():
            return Trigger.PTT
        read_block()
        now = time.monotonic()
        if now - last_heartbeat >= _WAKE_DEBUG_HEARTBEAT_S:
            _log_debug("still waiting for PTT / meeting…")
            last_heartbeat = now


def _record_while_held(
    mic: object,
    ptt: PttHotkey,
    *,
    max_s: float,
    settle_s: float,
) -> np.ndarray | None:
    """Hold-to-talk: capture PCM while PTT chord is down."""
    sr = int(getattr(mic, "sample_rate"))
    block = int(getattr(mic, "block"))
    read_block = getattr(mic, "read_block")
    settle_blocks = max(0, int(settle_s * sr / block))
    for _ in range(settle_blocks):
        read_block()
        if not ptt.is_down():
            return None

    chunks: list[np.ndarray] = []
    max_blocks = max(1, int(max_s * sr / block))
    for _ in range(max_blocks):
        if not ptt.is_down():
            break
        chunks.append(read_block())
    # Drain edge-up
    ptt.consume_edge_up()
    if not chunks:
        return None
    data = np.concatenate(chunks)
    stats = pcm_stats(data, sample_rate=sr)
    if stats.near_silent or data.size < sr // 5:
        return None
    return data


def _pcm_to_upload_wav(pcm: np.ndarray, *, sample_rate: int) -> tuple[bytes, object]:
    """Align PCM (16 kHz mono, level, trim) and wrap as WAV for /v1/voice/turn."""
    from krabobot_voice.preprocess import pcm16_to_wav_bytes, preprocess_pcm16

    aligned, sr = preprocess_pcm16(pcm, sample_rate=sample_rate)
    stats = pcm_stats(aligned, sample_rate=sr)
    return pcm16_to_wav_bytes(aligned, sample_rate=sr), stats


def _play_turn_result(result: object) -> bool:
    """Log transcript/reply and play TTS. True if playback succeeded (or none)."""
    transcript = getattr(result, "transcript", "") or ""
    reply = getattr(result, "reply", "") or ""
    audio_wav = getattr(result, "audio_wav", None)
    if transcript:
        _log(f"  you: {transcript}")
    if reply:
        _log(f"  bot: {reply}")
    if audio_wav:
        _log("playing…")
        try:
            play_wav_bytes(audio_wav)
        except Exception as e:
            _log(f"ERROR: воспроизведение: {e}")
            return False
    else:
        _log("нет audio/wav в ответе (TTS недоступен?)")
    return True


def _handle_turn(
    cfg: VoiceClientConfig,
    http: VoiceHttpClient,
    pcm: np.ndarray,
) -> bool:
    """Upload utterance and play reply. True if the turn completed successfully."""
    wav, stats = _pcm_to_upload_wav(pcm, sample_rate=cfg.sample_rate)
    _log(
        f"upload: {len(wav)} bytes, {stats.duration_ms:.0f} ms, "
        f"rms={stats.rms:.4f}, peak={stats.peak:.4f}"
    )
    if stats.near_silent:
        _log("WARNING: clip looks near-silent — speak louder / closer after beep")
    _log("thinking…")
    try:
        result = http.turn(audio_bytes=wav, audio_filename="utterance.wav")
    except Exception as e:
        _log(f"ERROR: запрос /v1/voice/turn: {e}")
        return False

    if result.status_code >= 400:
        _log(
            f"ERROR HTTP {result.status_code}: "
            f"{result.error_message or result.reply or 'unknown'}"
        )
        return False
    return _play_turn_result(result)


def _handle_text_turn(
    cfg: VoiceClientConfig,
    http: VoiceHttpClient,
    text: str,
) -> bool:
    """Send wake leftover / text-only instruct to /v1/voice/turn (no second listen)."""
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    _log(f"  you (wake): {cleaned}")
    _log("thinking…")
    try:
        result = http.turn(instruct=cleaned)
    except Exception as e:
        _log(f"ERROR: запрос /v1/voice/turn: {e}")
        return False

    if result.status_code >= 400:
        _log(
            f"ERROR HTTP {result.status_code}: "
            f"{result.error_message or result.reply or 'unknown'}"
        )
        return False
    return _play_turn_result(result)


def _apply_local_command(
    cfg: VoiceClientConfig,
    cmd: LocalCommand,
    *,
    kws: EmbeddingKws | None,
) -> tuple[bool, bool]:
    """Handle a matched local command.

    Returns ``(start_meeting, follow_up)``. ``follow_up`` is always False after
    local commands (back to wake wait).
    """
    if cmd is LocalCommand.EXIT:
        _log("local: exit dialog → wake")
        if kws is not None:
            kws.reset()
        return False, False
    if cmd is LocalCommand.MEETING_START:
        if not cfg.meeting_enabled:
            _log("local: meeting start ignored (meeting disabled)")
            if kws is not None:
                kws.reset()
            return False, False
        _log("local: meeting start")
        if kws is not None:
            kws.reset()
        return True, False
    if cmd is LocalCommand.MEETING_STOP:
        _log("local: meeting stop (not recording) — ignored")
        if kws is not None:
            kws.reset()
        return False, False
    return False, False


def _handle_meeting_upload(
    cfg: VoiceClientConfig,
    http: VoiceHttpClient,
    wav: bytes,
    *,
    duration_s: float,
    capture: str,
) -> None:
    from krabobot_voice.preprocess import preprocess_wav_bytes

    aligned = preprocess_wav_bytes(wav)
    _log(
        f"meeting upload: {len(aligned)} bytes, {duration_s:.1f} s, capture={capture}"
    )
    _log("thinking (meeting)…")
    try:
        result = http.turn(
            audio_bytes=aligned,
            audio_filename="meeting.wav",
            instruct=cfg.meeting_instruct or None,
        )
    except Exception as e:
        _log(f"ERROR: запрос /v1/voice/turn (meeting): {e}")
        return

    if result.status_code >= 400:
        _log(
            f"ERROR HTTP {result.status_code}: "
            f"{result.error_message or result.reply or 'unknown'}"
        )
        return
    if result.transcript:
        _log(f"  stt: {result.transcript[:200]}{'…' if len(result.transcript) > 200 else ''}")
    if result.reply:
        _log(f"  bot: {result.reply}")
    if result.audio_wav:
        _log("playing…")
        try:
            play_wav_bytes(result.audio_wav)
        except Exception as e:
            _log(f"ERROR: воспроизведение: {e}")


def _run_meeting_session(
    cfg: VoiceClientConfig,
    http: VoiceHttpClient,
    meeting_hk: PttHotkey | None,
    *,
    asr: object | None = None,
) -> None:
    """Record until toggle / voice stop / max_s, then upload+instruct."""
    recorder = MeetingRecorder(
        MeetingCaptureConfig(
            capture=cfg.meeting_capture,
            sample_rate=cfg.sample_rate,
            input_device=cfg.audio_input_device,
            output_device=cfg.audio_output_device,
            loopback_device=cfg.meeting_loopback_device,
            max_s=cfg.meeting_max_s,
        )
    )
    echo_hint = ""
    if cfg.meeting_capture == "mix":
        echo_hint = (
            " (mix: лучше наушники — иначе локальный голос может задвоиться с колонок)"
        )
    _log(f"meeting START capture={cfg.meeting_capture}{echo_hint}")
    play_beep(freq=880, duration_ms=120)
    try:
        recorder.start()
    except Exception as e:
        _log(f"ERROR: не удалось начать запись встречи: {e}")
        return

    last_print = 0.0
    last_asr = 0.0
    stop_reason = "hotkey"
    try:
        while recorder.active:
            if meeting_hk is not None and meeting_hk.consume_edge_down():
                stop_reason = "hotkey"
                break
            now = time.monotonic()
            if now - last_print > 10.0:
                hk = meeting_hk.hotkey_label if meeting_hk is not None else "voice"
                _log(f"meeting recording… (stop: {hk} / «стоп запись»)")
                last_print = now
            # Voice stop via local ASR on mic tap (mic / mix only).
            if asr is not None and now - last_asr >= 1.2:
                last_asr = now
                tap = recorder.recent_mic_pcm(2.5)
                if tap is not None and frame_rms(tap) >= cfg.energy_threshold:
                    try:
                        text = asr.transcribe_pcm16(  # type: ignore[attr-defined]
                            tap, sample_rate=cfg.sample_rate
                        )
                    except Exception:
                        text = ""
                    if text:
                        cmd = match_local_command(
                            text,
                            meeting_start=cfg.cmd_meeting_start,
                            meeting_stop=cfg.cmd_meeting_stop,
                            exit_dialog=cfg.cmd_exit,
                        )
                        if cmd is LocalCommand.MEETING_STOP:
                            _log(f"local stop: {text}")
                            stop_reason = "voice"
                            break
            time.sleep(0.05)
        try:
            result = recorder.stop()
        except Exception as e:
            _log(f"ERROR: остановка записи встречи: {e}")
            return
    except KeyboardInterrupt:
        try:
            result = recorder.stop()
        except Exception:
            raise
        raise

    play_beep(freq=660, duration_ms=120)
    _log(f"meeting STOP ({result.duration_s:.1f} s, via {stop_reason})")
    _handle_meeting_upload(
        cfg,
        http,
        result.wav_bytes,
        duration_s=result.duration_s,
        capture=result.capture,
    )


def _resolve_local_command(
    cfg: VoiceClientConfig,
    asr: object | None,
    pcm: np.ndarray,
) -> LocalCommand | None:
    """Transcribe utterance with local sherpa and match a command, if any."""
    if asr is None:
        return None
    try:
        text = asr.transcribe_pcm16(pcm, sample_rate=cfg.sample_rate)  # type: ignore[attr-defined]
    except Exception as e:
        _log(f"WARNING: local ASR for commands failed: {e}")
        return None
    if text:
        _log(f"  [local-asr] {text}")
    return match_local_command(
        text,
        meeting_start=cfg.cmd_meeting_start,
        meeting_stop=cfg.cmd_meeting_stop,
        exit_dialog=cfg.cmd_exit,
    )


def run_test_loopback(config: VoiceClientConfig | None = None, *, duration_s: float = 3.0) -> int:
    """Print live loopback RMS for a few seconds (diagnostic; no wake/ASR)."""
    from krabobot_voice.audio_io import probe_capture_rms, resolve_loopback_device_info

    cfg = config or VoiceClientConfig.load()
    _log("=== loopback test ===")
    try:
        info = resolve_loopback_device_info(
            device=cfg.meeting_loopback_device or None,
            output_device=cfg.audio_output_device or None,
        )
        _log(
            f"resolved: index={info.get('index')} "
            f"ch={info.get('maxInputChannels')} "
            f"rate={info.get('defaultSampleRate')} "
            f"name={info.get('name')!r}"
        )
    except Exception as e:
        _log(f"ERROR: resolve loopback: {e}")
        return 1
    _log(f"Измерьте {duration_s:.0f}s — включите YouTube/TTS на том же выходе.")
    try:
        stats = probe_capture_rms(
            "loopback",
            duration_s=duration_s,
            sample_rate=cfg.sample_rate,
            loopback_device=cfg.meeting_loopback_device or None,
            output_device=cfg.audio_output_device or None,
        )
    except Exception as e:
        _log(f"ERROR: open/read loopback: {e}")
        return 1
    _log(
        f"device: index={stats.get('device_index')} name={stats.get('device_name')!r}"
    )
    _log(
        f"RMS mean={stats['rms_mean']:.4f} max={stats['rms_max']:.4f} "
        f"blocks={stats['blocks']}"
    )
    if stats["ok"]:
        _log("OK: RMS > 0 — loopback слышит системный звук.")
        return 0
    _log(
        "FAIL: RMS≈0 — звук играет не на этом устройстве, "
        "или Privacy->Microphone / громкость. "
        "Задайте audio.output_device / meeting.loopback_device."
    )
    return 2


def run_loop(config: VoiceClientConfig | None = None) -> int:
    cfg = config or VoiceClientConfig.load()
    wake_energy = effective_wake_energy(
        cfg.wake_energy_threshold, listen_source=cfg.audio_listen_source
    )
    _log(f"krabobot-voice → {cfg.base_url}")
    _log(f"device_id={cfg.device_id}")
    _log(f"wake_mode={cfg.wake_mode}, ptt={'on' if cfg.ptt_enabled else 'off'}")
    _log(f"audio.listen_source={cfg.audio_listen_source}")
    if cfg.audio_listen_source == "loopback":
        from krabobot_voice.audio_io import probe_capture_rms, resolve_loopback_device_info

        try:
            info = resolve_loopback_device_info(
                device=cfg.meeting_loopback_device or None,
                output_device=cfg.audio_output_device or None,
            )
            _log(
                f"loopback device: index={info.get('index')} "
                f"name={info.get('name')!r}"
            )
        except Exception as e:
            _log(f"WARNING: resolve loopback failed: {e}")
        _log(
            "wake/listen: WASAPI loopback (системный звук). "
            "Видео/TTS должно играть на том же выходе. "
            "Проверка: python -m krabobot_voice --test-loopback"
        )
        try:
            probe = probe_capture_rms(
                "loopback",
                duration_s=1.0,
                sample_rate=cfg.sample_rate,
                loopback_device=cfg.meeting_loopback_device or None,
                output_device=cfg.audio_output_device or None,
            )
            _log(
                f"loopback probe 1s: rms_mean={probe['rms_mean']:.4f} "
                f"rms_max={probe['rms_max']:.4f} "
                f"({'OK' if probe['ok'] else 'тихо — включите звук на этом устройстве'})"
            )
        except Exception as e:
            _log(f"WARNING: loopback probe failed: {e}")
    if cfg.talk_follow_up_s > 0:
        _log(
            f"dialog follow-up: {cfg.talk_follow_up_s:.0f}s "
            f"(beep={'on' if cfg.talk_follow_up_beep else 'off'})"
        )
    if cfg.meeting_enabled:
        _log(f"meeting: capture={cfg.meeting_capture}, hotkey={cfg.meeting_hotkey}")

    link = ensure_device_linked(cfg)
    if not link.ok:
        _log(f"ERROR: привязка device_id не удалась: {link.message}")
        return 2
    _log(f"link: {link.message}")

    if cfg.wake_mode == "off" and not cfg.ptt_enabled and not cfg.meeting_enabled:
        _log("ERROR: wake_mode=off, PTT и meeting выключены — нечем активировать запись")
        return 3

    kws: EmbeddingKws | None = None
    asr = None
    if cfg.wake_mode == "kws":
        refs = cfg.kws_refs_dir.strip() or str(default_refs_dir())
        _log(f"KWS refs: {refs}")
        kws = EmbeddingKws.from_refs_dir(
            refs,
            sample_rate=cfg.sample_rate,
            threshold=cfg.kws_threshold,
            energy_threshold=cfg.kws_energy_threshold,
            auto_tts=cfg.kws_auto_enroll_tts,
        )
        if not kws.ready:
            _log(
                "WARNING: нет reference WAV для KWS. "
                "Положите записи «Эй, Арнольд» в refs dir "
                "или включите auto_enroll_tts на Windows. "
                "PTT всё равно доступен."
                if cfg.ptt_enabled or cfg.meeting_enabled
                else "ERROR: нет KWS references и PTT/meeting выключены."
            )
            if not cfg.ptt_enabled and not cfg.meeting_enabled:
                return 3
            kws = None
        else:
            _log(f"KWS ready ({len(kws.references)} refs, thr={cfg.kws_threshold})")
    elif cfg.wake_mode == "asr":
        from krabobot_voice.local_asr import LocalWakeAsr, resolve_stt_model_dir

        try:
            model_dir = resolve_stt_model_dir(cfg.stt_model_dir)
        except FileNotFoundError as e:
            _log(f"ERROR: {e}")
            return 3
        _log(f"wake ASR model: {model_dir.name}")
        try:
            asr = LocalWakeAsr(model_dir, num_threads=cfg.stt_num_threads)
        except Exception as e:
            _log(f"ERROR: не удалось загрузить sherpa-onnx: {e}")
            return 3

    http = VoiceHttpClient(cfg)
    ptt: PttHotkey | None = None
    if cfg.ptt_enabled:
        try:
            ptt = PttHotkey(cfg.ptt_hotkey)
            ptt.start()
            _log(f"PTT: удерживайте {ptt.hotkey_label}")
        except Exception as e:
            _log(f"WARNING: PTT недоступен ({e})")
            ptt = None
            if cfg.wake_mode == "off" and not cfg.meeting_enabled:
                return 3

    meeting_hk: PttHotkey | None = None
    if cfg.meeting_enabled:
        try:
            meeting_hk = PttHotkey(cfg.meeting_hotkey)
            meeting_hk.start()
            _log(f"Meeting: {meeting_hk.hotkey_label} — старт/стоп записи")
        except Exception as e:
            _log(f"WARNING: meeting hotkey недоступен ({e})")
            meeting_hk = None

    hints: list[str] = []
    wake_label = (cfg.wake_phrases[0] if cfg.wake_phrases else "wake").strip() or "wake"
    if cfg.wake_mode == "kws":
        hints.append(f'скажите «{wake_label}»')
        _log(
            f"wake KWS: VAD max={cfg.wake_max_s:.1f}s "
            f"silence_end={cfg.wake_silence_end_s:.2f}s "
            f"energy≥{wake_energy:.4f}"
        )
    elif cfg.wake_mode == "asr":
        hints.append(f'скажите «{wake_label}» (ASR)')
        _log(
            f"wake ASR: VAD max={cfg.wake_max_s:.1f}s "
            f"silence_end={cfg.wake_silence_end_s:.2f}s "
            f"energy≥{wake_energy:.4f} phrases={cfg.wake_phrases!r}"
        )
    if ptt is not None:
        hints.append(f"или PTT {ptt.hotkey_label}")
    if meeting_hk is not None:
        hints.append(f"или meeting {meeting_hk.hotkey_label}")
    _log("Готов. " + " ".join(hints) + ". Ctrl+C — выход.")

    talk_enabled = cfg.wake_mode != "off" or ptt is not None

    try:
        while True:
            # Meeting can run without holding the wake MicStream (frees device for mix).
            if meeting_hk is not None and meeting_hk.consume_edge_down():
                _run_meeting_session(cfg, http, meeting_hk, asr=asr)
                if kws is not None:
                    kws.reset()
                continue

            if not talk_enabled:
                # Meeting-only: poll hotkey without opening the mic.
                time.sleep(0.05)
                continue

            start_meeting = False
            follow_up = False
            # Shared capture for wake + Talk/PTT/follow-up (mic default; loopback for wake tests).
            with open_capture_stream(
                cfg.audio_listen_source,
                sample_rate=cfg.sample_rate,
                input_device=cfg.audio_input_device or None,
                loopback_device=cfg.meeting_loopback_device or None,
                output_device=cfg.audio_output_device or None,
            ) as mic:
                while True:
                    if meeting_hk is not None and meeting_hk.consume_edge_down():
                        start_meeting = True
                        break

                    trigger: Trigger | None = None
                    wake_detail = ""
                    wake_command = ""
                    if follow_up:
                        plan = plan_after_turn(
                            turn_ok=True,
                            follow_up_s=cfg.talk_follow_up_s,
                            follow_up_beep=cfg.talk_follow_up_beep,
                            settle_s=cfg.settle_s,
                            default_no_speech_s=cfg.no_speech_timeout_s,
                        )
                        _log(
                            f"follow-up listening… ({plan.no_speech_timeout_s:.0f}s)"
                        )
                    else:
                        _log("waiting for wake / PTT / meeting…")
                        wake_kwargs = dict(
                            energy_threshold=wake_energy,
                            silence_end_s=max(0.25, float(cfg.wake_silence_end_s)),
                            max_s=max(2.0, float(cfg.wake_max_s)),
                            min_speech_s=max(0.2, float(cfg.wake_min_speech_s)),
                            speech_start_s=cfg.speech_start_s,
                            preroll_s=cfg.preroll_s,
                            ptt=ptt,
                            meeting=meeting_hk,
                        )
                        if cfg.wake_mode == "asr" and asr is not None:
                            trigger, detail = _wait_for_wake_asr(
                                mic,
                                asr,
                                phrases=cfg.wake_phrases,
                                greetings=cfg.wake_greetings,
                                listen_source=cfg.audio_listen_source,
                                **wake_kwargs,
                            )
                            if trigger is Trigger.MEETING:
                                start_meeting = True
                                break
                            if trigger is Trigger.WAKE:
                                wake_detail = detail
                                _log(f"wake: {detail}")
                                wake_command = command_after_wake(
                                    detail,
                                    phrases=cfg.wake_phrases,
                                    greetings=cfg.wake_greetings,
                                )
                        elif cfg.wake_mode == "kws" and kws is not None:
                            trigger, detail = _wait_for_wake_kws(
                                mic,
                                kws,
                                **wake_kwargs,
                            )
                            if trigger is Trigger.MEETING:
                                start_meeting = True
                                break
                            if trigger is Trigger.WAKE:
                                wake_detail = detail
                                _log(f"wake: {detail}")
                        else:
                            trigger = _wait_for_ptt_or_meeting(
                                mic, ptt=ptt, meeting=meeting_hk
                            )
                            if trigger is Trigger.MEETING:
                                start_meeting = True
                                break
                        plan = plan_initial_listen(
                            no_speech_timeout_s=cfg.no_speech_timeout_s,
                            settle_s=cfg.settle_s,
                            listen_source=cfg.audio_listen_source,
                        )

                    # Same-utterance command after wake phrase → skip second listen.
                    if (
                        not follow_up
                        and trigger is Trigger.WAKE
                        and wake_command.strip()
                    ):
                        _log(f"wake command: {wake_command}")
                        cmd = match_local_command(
                            wake_command,
                            meeting_start=cfg.cmd_meeting_start,
                            meeting_stop=cfg.cmd_meeting_stop,
                            exit_dialog=cfg.cmd_exit,
                        )
                        if cmd is not None:
                            start_meeting, follow_up = _apply_local_command(
                                cfg, cmd, kws=kws
                            )
                            if start_meeting:
                                break
                            continue
                        turn_ok = _handle_text_turn(cfg, http, wake_command)
                        next_plan = plan_after_turn(
                            turn_ok=turn_ok,
                            follow_up_s=cfg.talk_follow_up_s,
                            follow_up_beep=cfg.talk_follow_up_beep,
                            settle_s=cfg.settle_s,
                            default_no_speech_s=cfg.no_speech_timeout_s,
                        )
                        follow_up = next_plan.phase is TalkPhase.FOLLOW_UP
                        if kws is not None:
                            kws.reset()
                        continue

                    if plan.play_beep:
                        play_beep(freq=1000, duration_ms=150)

                    if (
                        not follow_up
                        and trigger is Trigger.PTT
                        and ptt is not None
                    ):
                        _log("listening (PTT hold)…")
                        pcm = _record_while_held(
                            mic,
                            ptt,
                            max_s=cfg.utterance_max_s,
                            settle_s=plan.settle_s,
                        )
                    else:
                        if not follow_up:
                            _log(
                                f"listening… ({plan.no_speech_timeout_s:.0f}s, "
                                "no speech → idle)"
                            )
                            if trigger is Trigger.WAKE and wake_detail:
                                _log(
                                    "  (wake phrase only — waiting for next utterance)"
                                )
                        # Slightly lower floor when local commands are available
                        # so short exits («хватит») still pass VAD.
                        min_speech = cfg.min_speech_s
                        if asr is not None:
                            min_speech = min(cfg.min_speech_s, 0.75)
                        # Loopback: reuse wake energy gate — talk.energy_threshold is
                        # tuned for close mics and misses quieter system audio.
                        if cfg.audio_listen_source == "loopback":
                            talk_energy = wake_energy
                        else:
                            talk_energy = effective_wake_energy(
                                cfg.energy_threshold,
                                listen_source=cfg.audio_listen_source,
                            )
                        pcm = record_utterance(
                            mic.read_block,
                            sample_rate=mic.sample_rate,
                            block=mic.block,
                            max_s=cfg.utterance_max_s,
                            silence_end_s=cfg.silence_end_s,
                            speech_start_s=cfg.speech_start_s,
                            min_speech_s=min_speech,
                            energy_threshold=talk_energy,
                            settle_s=plan.settle_s,
                            preroll_s=cfg.preroll_s,
                            no_speech_timeout_s=plan.no_speech_timeout_s,
                        )

                    if pcm is None:
                        phase = plan_after_no_speech(was_follow_up=follow_up)
                        if follow_up:
                            _log("follow-up timeout — back to wake")
                        else:
                            _log("no speech — back to idle")
                        follow_up = phase is TalkPhase.FOLLOW_UP
                        if kws is not None:
                            kws.reset()
                        continue

                    # Local commands (sherpa) before any server upload.
                    cmd = _resolve_local_command(cfg, asr, pcm)
                    if cmd is not None:
                        start_meeting, follow_up = _apply_local_command(
                            cfg, cmd, kws=kws
                        )
                        if start_meeting:
                            break
                        continue

                    turn_ok = _handle_turn(cfg, http, pcm)
                    next_plan = plan_after_turn(
                        turn_ok=turn_ok,
                        follow_up_s=cfg.talk_follow_up_s,
                        follow_up_beep=cfg.talk_follow_up_beep,
                        settle_s=cfg.settle_s,
                        default_no_speech_s=cfg.no_speech_timeout_s,
                    )
                    follow_up = next_plan.phase is TalkPhase.FOLLOW_UP
                    if kws is not None:
                        kws.reset()

            if start_meeting:
                _run_meeting_session(cfg, http, meeting_hk, asr=asr)
                if kws is not None:
                    kws.reset()
    except KeyboardInterrupt:
        _log("stopped.")
        return 0
    except RuntimeError as e:
        _log(f"ERROR: {e}")
        return 4
    finally:
        if ptt is not None:
            ptt.stop()
        if meeting_hk is not None:
            meeting_hk.stop()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--test-loopback" in args:
        args = [a for a in args if a != "--test-loopback"]
        path = args[0] if args and not args[0].startswith("-") else None
        cfg = VoiceClientConfig.load(path)
        return run_test_loopback(cfg, duration_s=3.0)
    path = None
    if args and not args[0].startswith("-"):
        path = args[0]
    cfg = VoiceClientConfig.load(path)
    return run_loop(cfg)
