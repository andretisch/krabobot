"""Console KWS/PTT → beep → record → POST /v1/voice/turn → play loop."""

from __future__ import annotations

import sys
import time
from collections import deque
from enum import Enum

import numpy as np

from krabobot_voice.audio_io import MicStream, pcm16_to_wav_bytes, play_beep, play_wav_bytes
from krabobot_voice.config import VoiceClientConfig
from krabobot_voice.http_client import VoiceHttpClient
from krabobot_voice.kws import EmbeddingKws, default_refs_dir
from krabobot_voice.link import ensure_device_linked
from krabobot_voice.ptt import PttHotkey
from krabobot_voice.vad import frame_rms, pcm_stats, record_utterance
from krabobot_voice.wake import matches_wake_phrase


def _log(msg: str) -> None:
    print(msg, flush=True)


class Trigger(str, Enum):
    WAKE = "wake"
    PTT = "ptt"


def _wait_for_wake_asr(
    mic: MicStream,
    asr: object,
    *,
    window_s: float,
    hop_s: float,
    ptt: PttHotkey | None,
) -> tuple[Trigger, str]:
    """Legacy sherpa wake path (optional)."""
    sr = mic.sample_rate
    window_n = max(mic.block, int(window_s * sr))
    hop_n = max(mic.block, int(hop_s * sr))
    ring: deque[np.ndarray] = deque()
    total = 0
    since_hop = 0
    last_print = 0.0

    while True:
        if ptt is not None and ptt.consume_edge_down():
            return Trigger.PTT, "ptt"
        chunk = mic.read_block()
        ring.append(chunk)
        total += chunk.size
        since_hop += chunk.size
        while total > window_n + mic.block * 2:
            dropped = ring.popleft()
            total -= dropped.size

        now = time.monotonic()
        if now - last_print > 5.0:
            _log("waiting for wake…")
            last_print = now

        if since_hop < hop_n or total < int(0.6 * sr):
            continue
        since_hop = 0
        pcm = np.concatenate(list(ring))[-window_n:]
        if frame_rms(pcm) < 0.008:
            continue
        text = asr.transcribe_pcm16(pcm, sample_rate=sr)  # type: ignore[attr-defined]
        if text:
            _log(f"  [wake-asr] {text}")
        if matches_wake_phrase(text):
            return Trigger.WAKE, text


def _wait_for_trigger(
    mic: MicStream,
    *,
    kws: EmbeddingKws | None,
    ptt: PttHotkey | None,
) -> Trigger:
    """Block until KWS hit or PTT key-down."""
    last_print = 0.0
    while True:
        if ptt is not None and ptt.consume_edge_down():
            return Trigger.PTT
        chunk = mic.read_block()
        now = time.monotonic()
        if now - last_print > 5.0:
            _log("waiting for wake / PTT…")
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


def _handle_turn(
    cfg: VoiceClientConfig,
    http: VoiceHttpClient,
    pcm: np.ndarray,
) -> None:
    stats = pcm_stats(pcm, sample_rate=cfg.sample_rate)
    wav = pcm16_to_wav_bytes(pcm, sample_rate=cfg.sample_rate)
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
        return

    if result.status_code >= 400:
        _log(
            f"ERROR HTTP {result.status_code}: "
            f"{result.error_message or result.reply or 'unknown'}"
        )
        return
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
    else:
        _log("нет audio/wav в ответе (TTS недоступен?)")


def run_loop(config: VoiceClientConfig | None = None) -> int:
    cfg = config or VoiceClientConfig.load()
    _log(f"krabobot-voice → {cfg.base_url}")
    _log(f"device_id={cfg.device_id}")
    _log(f"wake_mode={cfg.wake_mode}, ptt={'on' if cfg.ptt_enabled else 'off'}")

    link = ensure_device_linked(cfg)
    if not link.ok:
        _log(f"ERROR: привязка device_id не удалась: {link.message}")
        return 2
    _log(f"link: {link.message}")

    if cfg.wake_mode == "off" and not cfg.ptt_enabled:
        _log("ERROR: wake_mode=off и PTT выключен — нечем активировать запись")
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
                if cfg.ptt_enabled
                else "ERROR: нет KWS references и PTT выключен."
            )
            if not cfg.ptt_enabled:
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
        _log(f"legacy wake ASR model: {model_dir.name}")
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
            if cfg.wake_mode == "off":
                return 3

    hints: list[str] = []
    if cfg.wake_mode == "kws":
        hints.append('скажите «Эй, Арнольд»')
    elif cfg.wake_mode == "asr":
        hints.append('скажите «Эй, Арнольд» (ASR)')
    if ptt is not None:
        hints.append(f"или PTT {ptt.hotkey_label}")
    _log("Готов. " + " ".join(hints) + ". Ctrl+C — выход.")

    try:
        with MicStream(sample_rate=cfg.sample_rate) as mic:
            while True:
                _log("waiting for wake / PTT…")
                if cfg.wake_mode == "asr" and asr is not None:
                    trigger, detail = _wait_for_wake_asr(
                        mic,
                        asr,
                        window_s=max(cfg.wake_window_s, 2.0),
                        hop_s=max(cfg.wake_hop_s, 0.5),
                        ptt=ptt,
                    )
                    if trigger is Trigger.WAKE:
                        _log(f"wake: {detail}")
                else:
                    trigger = _wait_for_trigger(mic, kws=kws, ptt=ptt)
                    if trigger is Trigger.WAKE:
                        _log("wake: KWS")

                play_beep(freq=1000, duration_ms=150)

                if trigger is Trigger.PTT and ptt is not None:
                    _log("listening (PTT hold)…")
                    pcm = _record_while_held(
                        mic,
                        ptt,
                        max_s=cfg.utterance_max_s,
                        settle_s=cfg.settle_s,
                    )
                else:
                    _log("listening…")
                    pcm = record_utterance(
                        mic.read_block,
                        sample_rate=mic.sample_rate,
                        block=mic.block,
                        max_s=cfg.utterance_max_s,
                        silence_end_s=cfg.silence_end_s,
                        speech_start_s=cfg.speech_start_s,
                        min_speech_s=cfg.min_speech_s,
                        energy_threshold=cfg.energy_threshold,
                        settle_s=cfg.settle_s,
                        preroll_s=cfg.preroll_s,
                        no_speech_timeout_s=cfg.no_speech_timeout_s,
                    )

                if pcm is None:
                    _log("no speech — back to idle")
                    if kws is not None:
                        kws.reset()
                    continue

                _handle_turn(cfg, http, pcm)
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


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    path = None
    if args and not args[0].startswith("-"):
        path = args[0]
    cfg = VoiceClientConfig.load(path)
    return run_loop(cfg)
