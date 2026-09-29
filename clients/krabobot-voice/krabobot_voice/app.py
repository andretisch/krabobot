"""Console ASR/PTT → beep → record → POST /v1/voice/turn → play loop.

Wake default: local sherpa-onnx ASR («Эй, Арнольд») with audio preprocess.
Optional KWS mode. Dialog follow-up after Talk reply. Meeting: toggle capture.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from enum import Enum

import numpy as np

from krabobot_voice.audio_io import MicStream, play_beep, play_wav_bytes
from krabobot_voice.commands import LocalCommand, match_local_command
from krabobot_voice.config import VoiceClientConfig
from krabobot_voice.dialog import (
    TalkPhase,
    plan_after_local_command,
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
from krabobot_voice.wake import (
    DEFAULT_WAKE_GREETINGS,
    DEFAULT_WAKE_PHRASES,
    decide_after_wake_decode,
    drop_ring_samples,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


class Trigger(str, Enum):
    WAKE = "wake"
    PTT = "ptt"
    MEETING = "meeting"


def _wait_for_wake_asr(
    mic: MicStream,
    asr: object,
    *,
    window_s: float,
    hop_s: float,
    energy_threshold: float,
    ptt: PttHotkey | None,
    meeting: PttHotkey | None,
    phrases: list[str] | tuple[str, ...] | None = None,
    greetings: list[str] | tuple[str, ...] | None = None,
    cooldown_s: float = 0.8,
) -> tuple[Trigger, str]:
    """Sherpa wake path: energy-gate → preprocess (inside ASR) → fuzzy match.

    After each decode (hit or miss) the ring advances past the scored window and a
    short cooldown applies so the same utterance is not rescored / reprinted forever.
    """
    sr = mic.sample_rate
    window_n = max(mic.block, int(window_s * sr))
    hop_n = max(mic.block, int(hop_s * sr))
    ring: deque[np.ndarray] = deque()
    total = 0
    since_hop = 0
    cooldown = 0
    last_status_print = 0.0
    last_asr_print = ""
    phrase_list = tuple(phrases) if phrases is not None else DEFAULT_WAKE_PHRASES
    greet_list = tuple(greetings) if greetings is not None else DEFAULT_WAKE_GREETINGS

    while True:
        if meeting is not None and meeting.consume_edge_down():
            return Trigger.MEETING, "meeting"
        if ptt is not None and ptt.consume_edge_down():
            return Trigger.PTT, "ptt"
        chunk = mic.read_block()
        ring.append(chunk)
        total += chunk.size
        since_hop += chunk.size
        if cooldown > 0:
            cooldown = max(0, cooldown - chunk.size)
        while total > window_n + mic.block * 2:
            dropped = ring.popleft()
            total -= dropped.size

        now = time.monotonic()
        if now - last_status_print > 5.0:
            _log("waiting for wake…")
            last_status_print = now

        if cooldown > 0 or since_hop < hop_n or total < int(0.6 * sr):
            continue
        since_hop = 0
        pcm = np.concatenate(list(ring))[-window_n:]
        if frame_rms(pcm) < energy_threshold:
            continue
        text = asr.transcribe_pcm16(pcm, sample_rate=sr)  # type: ignore[attr-defined]
        decision = decide_after_wake_decode(
            text or "",
            last_printed=last_asr_print,
            window_n=window_n,
            sample_rate=sr,
            phrases=phrase_list,
            greetings=greet_list,
            cooldown_s=cooldown_s,
        )
        if decision.print_text is not None:
            _log(f"  [wake-asr] {decision.print_text}")
            last_asr_print = decision.print_text
        total = drop_ring_samples(ring, total, decision.consume_samples)
        since_hop = 0
        cooldown = decision.cooldown_samples
        if decision.matched:
            return Trigger.WAKE, (text or "").strip() or decision.print_text or ""


def _wait_for_trigger(
    mic: MicStream,
    *,
    kws: EmbeddingKws | None,
    ptt: PttHotkey | None,
    meeting: PttHotkey | None,
) -> Trigger:
    """Block until KWS hit, PTT key-down, or meeting toggle."""
    last_print = 0.0
    while True:
        if meeting is not None and meeting.consume_edge_down():
            return Trigger.MEETING
        if ptt is not None and ptt.consume_edge_down():
            return Trigger.PTT
        chunk = mic.read_block()
        now = time.monotonic()
        if now - last_print > 5.0:
            _log("waiting for wake / PTT / meeting…")
            last_print = now
        if kws is None:
            continue
        hit = kws.process(chunk)
        if hit is not None:
            _log(f"  [kws] score={hit.score:.3f} (thr={hit.threshold:.3f})")
            return Trigger.WAKE


def _record_while_held(
    mic: MicStream,
    ptt: PttHotkey,
    *,
    max_s: float,
    settle_s: float,
) -> np.ndarray | None:
    """Hold-to-talk: capture PCM while PTT chord is down."""
    sr = mic.sample_rate
    settle_blocks = max(0, int(settle_s * sr / mic.block))
    for _ in range(settle_blocks):
        mic.read_block()
        if not ptt.is_down():
            return None

    chunks: list[np.ndarray] = []
    max_blocks = max(1, int(max_s * sr / mic.block))
    for _ in range(max_blocks):
        if not ptt.is_down():
            break
        chunks.append(mic.read_block())
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
    if result.transcript:
        _log(f"  you: {result.transcript}")
    if result.reply:
        _log(f"  bot: {result.reply}")
    if result.audio_wav:
        _log("playing…")
        try:
            play_wav_bytes(result.audio_wav)
        except Exception as e:
            _log(f"ERROR: воспроизведение: {e}")
            return False
    else:
        _log("нет audio/wav в ответе (TTS недоступен?)")
    return True


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


def run_loop(config: VoiceClientConfig | None = None) -> int:
    cfg = config or VoiceClientConfig.load()
    _log(f"krabobot-voice → {cfg.base_url}")
    _log(f"device_id={cfg.device_id}")
    _log(f"wake_mode={cfg.wake_mode}, ptt={'on' if cfg.ptt_enabled else 'off'}")
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
            window_s=cfg.wake_window_s,
            hop_s=cfg.wake_hop_s,
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
    if cfg.wake_mode == "kws":
        hints.append('скажите «Эй, Арнольд»')
    elif cfg.wake_mode == "asr":
        hints.append('скажите «Эй, Арнольд» (ASR)')
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
            with MicStream(
                sample_rate=cfg.sample_rate,
                device=cfg.audio_input_device or None,
            ) as mic:
                while True:
                    if meeting_hk is not None and meeting_hk.consume_edge_down():
                        start_meeting = True
                        break

                    trigger: Trigger | None = None
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
                        if cfg.wake_mode == "asr" and asr is not None:
                            trigger, detail = _wait_for_wake_asr(
                                mic,
                                asr,
                                window_s=max(cfg.wake_window_s, 2.0),
                                hop_s=max(cfg.wake_hop_s, 0.5),
                                energy_threshold=cfg.wake_energy_threshold,
                                ptt=ptt,
                                meeting=meeting_hk,
                                phrases=cfg.wake_phrases,
                                greetings=cfg.wake_greetings,
                            )
                            if trigger is Trigger.MEETING:
                                start_meeting = True
                                break
                            if trigger is Trigger.WAKE:
                                _log(f"wake: {detail}")
                        else:
                            trigger = _wait_for_trigger(
                                mic, kws=kws, ptt=ptt, meeting=meeting_hk
                            )
                            if trigger is Trigger.MEETING:
                                start_meeting = True
                                break
                            if trigger is Trigger.WAKE:
                                _log("wake: KWS")
                        plan = plan_initial_listen(
                            no_speech_timeout_s=cfg.no_speech_timeout_s,
                            settle_s=cfg.settle_s,
                        )

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
                            _log("listening…")
                        # Slightly lower floor when local commands are available
                        # so short exits («хватит») still pass VAD.
                        min_speech = cfg.min_speech_s
                        if asr is not None:
                            min_speech = min(cfg.min_speech_s, 0.75)
                        pcm = record_utterance(
                            mic.read_block,
                            sample_rate=mic.sample_rate,
                            block=mic.block,
                            max_s=cfg.utterance_max_s,
                            silence_end_s=cfg.silence_end_s,
                            speech_start_s=cfg.speech_start_s,
                            min_speech_s=min_speech,
                            energy_threshold=cfg.energy_threshold,
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
                    if cmd is LocalCommand.EXIT:
                        _log("local: exit dialog → wake")
                        follow_up = (
                            plan_after_local_command(command=cmd.value)
                            is TalkPhase.FOLLOW_UP
                        )
                        if kws is not None:
                            kws.reset()
                        continue
                    if cmd is LocalCommand.MEETING_START:
                        if not cfg.meeting_enabled:
                            _log("local: meeting start ignored (meeting disabled)")
                            follow_up = False
                            if kws is not None:
                                kws.reset()
                            continue
                        _log("local: meeting start")
                        start_meeting = True
                        follow_up = False
                        if kws is not None:
                            kws.reset()
                        break
                    if cmd is LocalCommand.MEETING_STOP:
                        _log("local: meeting stop (not recording) — ignored")
                        # Stay in follow-up if we were there; else back to wake wait.
                        if kws is not None:
                            kws.reset()
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
    path = None
    if args and not args[0].startswith("-"):
        path = args[0]
    cfg = VoiceClientConfig.load(path)
    return run_loop(cfg)
