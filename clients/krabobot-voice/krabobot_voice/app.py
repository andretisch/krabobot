"""Console ASR/PTT → segment stream → VoiceSession → effects driver.

Wake default: persistent VAD segmenter → local sherpa ASR → pure state machine.
Dialog follow-up after Talk reply. Meeting: same segmenter on mic tap.
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np

from krabobot_voice.audio_io import (
    open_capture_stream,
    play_beeps,
    play_wav_bytes,
    with_stream_keepalive,
)
from krabobot_voice.commands import match_local_command
from krabobot_voice.config import VoiceClientConfig
from krabobot_voice.dialog import (
    Beep,
    Hotkey,
    Mode,
    RunTest,
    Segment,
    SendAudio,
    SendText,
    SetDeadline,
    StartMeeting,
    StopMeeting,
    Timeout,
    TurnDone,
    VoiceSession,
    VoiceSessionConfig,
)
from krabobot_voice.http_client import VoiceHttpClient
from krabobot_voice.kws import EmbeddingKws, default_refs_dir
from krabobot_voice.link import ensure_device_linked
from krabobot_voice.meeting import MeetingCaptureConfig, MeetingRecorder
from krabobot_voice.protocol import ClientState
from krabobot_voice.ptt import PttHotkey
from krabobot_voice.segmenter import VadSegmenter
from krabobot_voice.vad import frame_rms, pcm_stats
from krabobot_voice.wake import matches_wake_phrase

# Loopback after downmix/resample is often quieter than a close mic
# (energy pre-gate / preprocess only — speech decision is Silero).
_LOOPBACK_WAKE_ENERGY_SCALE = 0.4
_LOOPBACK_WAKE_ENERGY_FLOOR = 0.0015
_WAKE_DEBUG_HEARTBEAT_S = 30.0
_KEEPALIVE_SINK_MAX = 200


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
    """Lower energy *pre-gate* for WASAPI loopback (system audio RMS is often low).

    Not used for speech start/end when Silero ``speech_gate`` is active.
    """
    thr = float(threshold)
    if (listen_source or "").strip().lower() != "loopback":
        return thr
    return min(thr, max(_LOOPBACK_WAKE_ENERGY_FLOOR, thr * _LOOPBACK_WAKE_ENERGY_SCALE))


def _session_config(cfg: VoiceClientConfig) -> VoiceSessionConfig:
    return VoiceSessionConfig(
        listen_timeout_s=float(cfg.talk_listen_timeout_s),
        follow_up_s=float(cfg.talk_follow_up_s),
        wake_phrases=list(cfg.wake_phrases),
        wake_greetings=list(cfg.wake_greetings),
        cmd_meeting_start=list(cfg.cmd_meeting_start),
        cmd_meeting_stop=list(cfg.cmd_meeting_stop),
        cmd_run_test=list(cfg.cmd_run_test),
        cmd_exit=list(cfg.cmd_exit),
        meeting_enabled=bool(cfg.meeting_enabled),
    )


def _client_state(session: VoiceSession) -> ClientState:
    return ClientState(
        mode=session.client_mode(),
        meeting=session.meeting_state(),
    )


def _make_segmenter(
    read_block: Callable[[], np.ndarray],
    *,
    sample_rate: int,
    block: int,
    cfg: VoiceClientConfig,
    energy_threshold: float,
    max_s: float | None = None,
    silence_end_s: float | None = None,
    min_speech_s: float | None = None,
    speech_gate: Callable[[np.ndarray], bool] | None = None,
    energy_pregate: float | None = None,
) -> VadSegmenter:
    min_speech = float(min_speech_s if min_speech_s is not None else cfg.min_speech_s)
    pregate = (
        float(cfg.vad_energy_pregate)
        if energy_pregate is None
        else energy_pregate
    )
    return VadSegmenter(
        read_block,
        sample_rate=sample_rate,
        block=block,
        max_s=float(max_s if max_s is not None else cfg.utterance_max_s),
        silence_end_s=float(
            silence_end_s if silence_end_s is not None else cfg.silence_end_s
        ),
        speech_start_s=cfg.speech_start_s,
        min_speech_s=min_speech,
        energy_threshold=float(energy_threshold),
        preroll_s=cfg.preroll_s,
        speech_gate=speech_gate,
        energy_pregate=pregate if speech_gate is not None else None,
    )


def _load_speech_gate(
    cfg: VoiceClientConfig,
) -> tuple[Callable[[np.ndarray], bool] | None, object | None, str]:
    """Return (gate_fn, silero_instance_or_None, backend_label)."""
    backend = (cfg.vad_backend or "silero").strip().lower()
    if backend == "energy":
        return None, None, "energy"
    if backend != "silero":
        _log(f"WARNING: unknown vad.backend={backend!r}; using energy")
        return None, None, "energy"
    try:
        from krabobot_voice.silero_vad import load_silero_vad, silero_available
    except Exception as e:
        _log(f"WARNING: Silero VAD import failed ({e}); falling back to energy")
        return None, None, "energy"
    if not silero_available():
        _log(
            "WARNING: onnxruntime not installed — VAD falls back to energy. "
            'Install: pip install -e ".\\clients\\krabobot-voice[asr]"'
        )
        return None, None, "energy"
    try:
        silero = load_silero_vad(
            model_path=cfg.vad_model_path or None,
            threshold=float(cfg.vad_threshold),
            sample_rate=int(cfg.sample_rate),
        )
    except Exception as e:
        _log(f"WARNING: Silero VAD load failed ({e}); falling back to energy")
        return None, None, "energy"
    return silero.is_speech, silero, "silero"


def _record_while_held(
    mic: object,
    ptt: PttHotkey,
    *,
    max_s: float,
) -> np.ndarray | None:
    """Hold-to-talk: capture PCM while PTT chord is down."""
    sr = int(getattr(mic, "sample_rate"))
    read_block = getattr(mic, "read_block")
    chunks: list[np.ndarray] = []
    max_blocks = max(1, int(max_s * sr / max(1, int(getattr(mic, "block")))))
    for _ in range(max_blocks):
        if not ptt.is_down():
            break
        chunks.append(read_block())
    ptt.consume_edge_up()
    if not chunks:
        return None
    data = np.concatenate(chunks)
    stats = pcm_stats(data, sample_rate=sr)
    if stats.near_silent or data.size < sr // 5:
        return None
    return data


def _pcm_to_upload_wav(pcm: np.ndarray, *, sample_rate: int) -> tuple[bytes, object]:
    from krabobot_voice.preprocess import pcm16_to_wav_bytes, preprocess_pcm16

    aligned, sr = preprocess_pcm16(pcm, sample_rate=sample_rate)
    stats = pcm_stats(aligned, sample_rate=sr)
    return pcm16_to_wav_bytes(aligned, sample_rate=sr), stats


def _play_turn_result(result: object) -> bool:
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


def _drain_mic(mic: object, blocks: int = 8) -> None:
    """Drop a few blocks (beep / TTS echo) without feeding the segmenter."""
    read = getattr(mic, "read_block", None)
    drain = getattr(mic, "drain_available", None)
    for _ in range(max(0, int(blocks))):
        try:
            if callable(drain):
                drain()
            elif callable(read):
                read()
        except Exception:
            break


def _transcribe(
    asr: object | None,
    pcm: np.ndarray,
    *,
    sample_rate: int,
    energy_threshold: float | None = None,
) -> str:
    if asr is None:
        return ""
    kwargs: dict[str, Any] = {"sample_rate": sample_rate}
    if energy_threshold is not None:
        kwargs["energy_threshold"] = energy_threshold
    try:
        return str(asr.transcribe_pcm16(pcm, **kwargs) or "").strip()  # type: ignore[attr-defined]
    except TypeError:
        try:
            return str(asr.transcribe_pcm16(pcm, sample_rate=sample_rate) or "").strip()  # type: ignore[attr-defined]
        except Exception as e:
            _log(f"WARNING: local ASR failed: {e}")
            return ""
    except Exception as e:
        _log(f"WARNING: local ASR failed: {e}")
        return ""


def _kws_score(kws: EmbeddingKws, pcm: np.ndarray) -> float:
    try:
        return float(kws.best_score(pcm))
    except Exception:
        return 0.0


class _Driver:
    """Applies VoiceSession effects and owns the segment → ASR loop."""

    def __init__(
        self,
        cfg: VoiceClientConfig,
        http: VoiceHttpClient,
        session: VoiceSession,
        *,
        asr: object | None,
        kws: EmbeddingKws | None,
        ptt: PttHotkey | None,
        meeting_hk: PttHotkey | None,
        wake_energy: float,
        speech_gate: Callable[[np.ndarray], bool] | None = None,
        silero: object | None = None,
    ) -> None:
        self.cfg = cfg
        self.http = http
        self.session = session
        self.asr = asr
        self.kws = kws
        self.ptt = ptt
        self.meeting_hk = meeting_hk
        self.wake_energy = wake_energy
        self.speech_gate = speech_gate
        self._silero = silero
        self._deadline: float | None = None
        self._pending_pcm: np.ndarray | None = None
        self._recorder: MeetingRecorder | None = None
        self._last_status = ""
        self._early_asr_text: str | None = None

    def _reset_vad_state(self) -> None:
        reset = getattr(self._silero, "reset", None)
        if callable(reset):
            reset()

    def _status(self, msg: str) -> None:
        if msg != self._last_status:
            _log(msg)
            self._last_status = msg

    def _apply(self, effects: list[object], *, mic: object | None = None) -> None:
        for eff in effects:
            if isinstance(eff, Beep):
                if self.cfg.talk_beeps and eff.count > 0:
                    play_beeps(eff.count)
                    if mic is not None:
                        _drain_mic(mic)
            elif isinstance(eff, SetDeadline):
                if eff.seconds is None:
                    self._deadline = None
                else:
                    self._deadline = time.monotonic() + max(0.1, float(eff.seconds))
            elif isinstance(eff, SendText):
                self._do_text_turn(eff.text, mic=mic)
            elif isinstance(eff, SendAudio):
                self._do_audio_turn(mic=mic)
            elif isinstance(eff, StartMeeting):
                self._start_meeting()
            elif isinstance(eff, StopMeeting):
                self._stop_meeting_and_upload()
            elif isinstance(eff, RunTest):
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _log(f"[test] {now} — тест выполнен")

    def _do_text_turn(self, text: str, *, mic: object | None) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            self._apply(
                self.session.on_event(TurnDone(ok=False)),
                mic=mic,
            )
            return
        _log(f"  you (wake): {cleaned}")
        _log("thinking…")
        sink: deque[np.ndarray] | None = None
        try:
            if mic is not None:
                sink = deque(maxlen=_KEEPALIVE_SINK_MAX)

                def _call() -> object:
                    return self.http.turn(
                        instruct=cleaned,
                        client_state=_client_state(self.session),
                    )

                result = with_stream_keepalive(mic, _call, sink=sink)
            else:
                result = self.http.turn(
                    instruct=cleaned,
                    client_state=_client_state(self.session),
                )
        except Exception as e:
            _log(f"ERROR: запрос /v1/voice/turn: {e}")
            self._apply(self.session.on_event(TurnDone(ok=False)), mic=mic)
            return
        finally:
            # Backlog feed happens in the talk loop after apply returns if segmenter set.
            self._last_sink = sink

        ok = self._finish_turn(result, mic=mic)
        self._apply(
            self.session.on_event(
                TurnDone(ok=ok, actions=tuple(getattr(result, "actions", None) or ()))
            ),
            mic=mic,
        )

    def _do_audio_turn(self, *, mic: object | None) -> None:
        pcm = self._pending_pcm
        self._pending_pcm = None
        if pcm is None:
            self._apply(self.session.on_event(TurnDone(ok=False)), mic=mic)
            return
        wav, stats = _pcm_to_upload_wav(pcm, sample_rate=self.cfg.sample_rate)
        _log(
            f"upload: {len(wav)} bytes, {stats.duration_ms:.0f} ms, "
            f"rms={stats.rms:.4f}, peak={stats.peak:.4f}"
        )
        if stats.near_silent:
            _log("WARNING: clip looks near-silent — speak louder / closer after beep")
        _log("thinking…")
        sink: deque[np.ndarray] | None = None
        try:
            if mic is not None:
                sink = deque(maxlen=_KEEPALIVE_SINK_MAX)

                def _call() -> object:
                    return self.http.turn(
                        audio_bytes=wav,
                        audio_filename="utterance.wav",
                        client_state=_client_state(self.session),
                    )

                result = with_stream_keepalive(mic, _call, sink=sink)
            else:
                result = self.http.turn(
                    audio_bytes=wav,
                    audio_filename="utterance.wav",
                    client_state=_client_state(self.session),
                )
        except Exception as e:
            _log(f"ERROR: запрос /v1/voice/turn: {e}")
            self._apply(self.session.on_event(TurnDone(ok=False)), mic=mic)
            return
        finally:
            self._last_sink = sink

        ok = self._finish_turn(result, mic=mic)
        self._apply(
            self.session.on_event(
                TurnDone(ok=ok, actions=tuple(getattr(result, "actions", None) or ()))
            ),
            mic=mic,
        )

    def _finish_turn(self, result: object, *, mic: object | None) -> bool:
        status = int(getattr(result, "status_code", 500) or 500)
        if status >= 400:
            _log(
                f"ERROR HTTP {status}: "
                f"{getattr(result, 'error_message', None) or getattr(result, 'reply', None) or 'unknown'}"
            )
            return False
        ok = _play_turn_result(result)
        if mic is not None:
            _drain_mic(mic, blocks=12)
        return ok

    def _start_meeting(self) -> None:
        if self._recorder is not None and self._recorder.active:
            return
        recorder = MeetingRecorder(
            MeetingCaptureConfig(
                capture=self.cfg.meeting_capture,
                sample_rate=self.cfg.sample_rate,
                input_device=self.cfg.audio_input_device,
                output_device=self.cfg.audio_output_device,
                loopback_device=self.cfg.meeting_loopback_device,
                max_s=self.cfg.meeting_max_s,
            )
        )
        echo_hint = ""
        if self.cfg.meeting_capture == "mix":
            echo_hint = (
                " (mix: лучше наушники — иначе локальный голос может задвоиться с колонок)"
            )
        _log(f"meeting START capture={self.cfg.meeting_capture}{echo_hint}")
        try:
            recorder.start()
        except Exception as e:
            _log(f"ERROR: не удалось начать запись встречи: {e}")
            self.session.mode = Mode.IDLE
            return
        self._recorder = recorder

    def _stop_meeting_and_upload(self) -> None:
        recorder = self._recorder
        self._recorder = None
        if recorder is None:
            return
        try:
            result = recorder.stop()
        except Exception as e:
            _log(f"ERROR: остановка записи встречи: {e}")
            return
        _log(f"meeting STOP ({result.duration_s:.1f} s)")
        self._handle_meeting_upload(result.wav_bytes, duration_s=result.duration_s, capture=result.capture)

    def _handle_meeting_upload(
        self,
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
            result = self.http.turn(
                audio_bytes=aligned,
                audio_filename="meeting.wav",
                instruct=self.cfg.meeting_instruct or None,
                client_state=ClientState(mode="idle", meeting="idle"),
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
            _log(
                f"  stt: {result.transcript[:200]}"
                f"{'…' if len(result.transcript) > 200 else ''}"
            )
        if result.reply:
            _log(f"  bot: {result.reply}")
        if result.audio_wav:
            _log("playing…")
            try:
                play_wav_bytes(result.audio_wav)
            except Exception as e:
                _log(f"ERROR: воспроизведение: {e}")

    def _poll_hotkeys(self) -> list[object] | None:
        """Return effects if a hotkey was consumed; else None."""
        if self.meeting_hk is not None and self.meeting_hk.consume_edge_down():
            return self.session.on_event(Hotkey("meeting"))
        if self.ptt is not None and self.ptt.consume_edge_down():
            return self.session.on_event(Hotkey("ptt"))
        return None

    def _asr_pcm(
        self,
        pcm: np.ndarray,
        *,
        mic: object,
        segmenter: VadSegmenter,
        for_wake: bool,
    ) -> str:
        sink: deque[np.ndarray] = deque(maxlen=_KEEPALIVE_SINK_MAX)
        energy = min(self.wake_energy, 0.006) if for_wake else None

        if self.cfg.wake_mode == "kws" and self.kws is not None and for_wake:
            score = with_stream_keepalive(
                mic, lambda: _kws_score(self.kws, pcm), sink=sink  # type: ignore[arg-type]
            )
            segmenter.feed_backlog(sink)
            if score >= self.kws.threshold:
                _log(f"  [kws] score={score:.3f} (thr={self.kws.threshold:.3f})")
                self.kws.reset()
                # Synthetic wake text so VoiceSession matches without ASR phrase.
                phrase = (self.cfg.wake_phrases[0] if self.cfg.wake_phrases else "wake").strip()
                return phrase or "wake"
            return ""

        if self.asr is None:
            segmenter.feed_backlog(sink)
            return ""

        text = with_stream_keepalive(
            mic,
            lambda: _transcribe(
                self.asr,
                pcm,
                sample_rate=self.cfg.sample_rate,
                energy_threshold=energy,
            ),
            sink=sink,
        )
        segmenter.feed_backlog(sink)
        return text

    def _make_early_check(self, *, for_wake: bool) -> Callable[[np.ndarray], bool] | None:
        """Optional energy-backend helper; unused when Silero speech_gate is active."""
        # Silero closes music segments correctly — do not thrash partial ASR.
        if self.speech_gate is not None:
            return None
        if for_wake and self.cfg.wake_mode == "kws" and self.kws is not None:

            def _kws_check(pcm: np.ndarray) -> bool:
                score = _kws_score(self.kws, pcm)  # type: ignore[arg-type]
                if score < float(self.kws.threshold):  # type: ignore[union-attr]
                    return False
                phrase = (
                    self.cfg.wake_phrases[0] if self.cfg.wake_phrases else "wake"
                ).strip()
                self._early_asr_text = phrase or "wake"
                _log_debug(f"early-kws hit score={score:.3f}")
                return True

            return _kws_check

        if self.asr is None:
            return None

        energy = min(self.wake_energy, 0.006) if for_wake else None

        def _asr_check(pcm: np.ndarray) -> bool:
            text = _transcribe(
                self.asr,
                pcm,
                sample_rate=self.cfg.sample_rate,
                energy_threshold=energy,
            )
            if not text:
                return False
            if for_wake:
                hit = matches_wake_phrase(
                    text,
                    phrases=self.cfg.wake_phrases or None,
                    greetings=self.cfg.wake_greetings or None,
                )
            else:
                hit = (
                    match_local_command(
                        text,
                        meeting_start=self.cfg.cmd_meeting_start,
                        meeting_stop=self.cfg.cmd_meeting_stop,
                        run_test=self.cfg.cmd_run_test,
                        exit_dialog=self.cfg.cmd_exit,
                    )
                    is not None
                )
            if hit:
                self._early_asr_text = text
                _log_debug(f"early-asr hit: {text}")
            return hit

        return _asr_check

    def run_talk_loop(self, mic: object) -> None:
        """Drive one open capture stream until meeting starts or stream should reopen."""
        self._last_sink = None
        wake_max = max(2.0, float(self.cfg.wake_max_s))
        wake_silence = max(0.25, float(self.cfg.wake_silence_end_s))
        wake_min = max(0.2, float(self.cfg.wake_min_speech_s))
        talk_min = float(self.cfg.min_speech_s)
        if self.asr is not None:
            talk_min = min(talk_min, 0.75)
        talk_energy = (
            self.wake_energy
            if self.cfg.audio_listen_source == "loopback"
            else effective_wake_energy(
                self.cfg.energy_threshold, listen_source=self.cfg.audio_listen_source
            )
        )

        # Idle uses wake VAD params; listen/dialog use talk params — switch energy
        # via segmenter.energy_threshold per mode.
        self._reset_vad_state()
        segmenter = _make_segmenter(
            mic.read_block,
            sample_rate=int(mic.sample_rate),
            block=int(mic.block),
            cfg=self.cfg,
            energy_threshold=self.wake_energy,
            max_s=wake_max,
            silence_end_s=wake_silence,
            min_speech_s=wake_min,
            speech_gate=self.speech_gate,
        )
        last_heartbeat = time.monotonic()

        while True:
            # Meeting session owns its own mic tap; leave the talk stream.
            if self.session.mode is Mode.MEETING:
                return

            hot = self._poll_hotkeys()
            if hot is not None:
                # PTT: arm then capture while held.
                if (
                    self.ptt is not None
                    and any(isinstance(e, Beep) for e in hot)
                    and self.session.mode is Mode.LISTEN
                    and self.ptt.is_down()
                ):
                    self._apply(hot, mic=mic)
                    self._status("listening (PTT hold)…")
                    pcm = _record_while_held(
                        mic, self.ptt, max_s=self.cfg.utterance_max_s
                    )
                    if pcm is None:
                        self._apply(self.session.on_event(Timeout()), mic=mic)
                        continue
                    self._pending_pcm = pcm
                    text = self._asr_pcm(
                        pcm, mic=mic, segmenter=segmenter, for_wake=False
                    )
                    if text:
                        _log(f"  [local-asr] {text}")
                    self._apply(
                        self.session.on_event(
                            Segment(
                                text,
                                source="ptt",
                                upload_if_empty=self.asr is None,
                            )
                        ),
                        mic=mic,
                    )
                    self._feed_last_sink(segmenter)
                    continue
                self._apply(hot, mic=mic)
                if self.session.mode is Mode.MEETING:
                    return
                continue

            mode = self.session.mode
            if mode is Mode.IDLE:
                segmenter.configure(
                    energy_threshold=self.wake_energy,
                    max_s=wake_max,
                    silence_end_s=wake_silence,
                    min_speech_s=wake_min,
                )
                self._status("waiting for wake / PTT / meeting…")
            else:
                segmenter.configure(
                    energy_threshold=talk_energy,
                    max_s=float(self.cfg.utterance_max_s),
                    silence_end_s=float(self.cfg.silence_end_s),
                    min_speech_s=talk_min,
                )
                if mode is Mode.LISTEN:
                    self._status(
                        f"listening… ({self.cfg.talk_listen_timeout_s:.0f}s, "
                        "no speech → idle)"
                    )
                elif mode is Mode.DIALOG:
                    self._status(
                        f"follow-up listening… ({self.cfg.talk_follow_up_s:.0f}s)"
                    )

            def _poll() -> None:
                if self.meeting_hk is not None and self.meeting_hk.consume_edge_down():
                    raise _HotkeyAbortError("meeting")
                if self.ptt is not None and self.ptt.consume_edge_down():
                    raise _HotkeyAbortError("ptt")

            for_wake = mode is Mode.IDLE
            early = self._make_early_check(for_wake=for_wake)
            self._early_asr_text = None

            try:
                # With Silero, early is None — segment closes on Silero non-speech.
                pcm = segmenter.next_segment(
                    self._deadline,
                    poll=_poll,
                    early_check=early,
                    early_check_interval_s=1.0,
                    early_check_min_s=0.7,
                )
            except _HotkeyAbortError as abort:
                effects = self.session.on_event(Hotkey(abort.kind))
                if (
                    abort.kind == "ptt"
                    and self.ptt is not None
                    and self.ptt.is_down()
                ):
                    self._apply(effects, mic=mic)
                    self._status("listening (PTT hold)…")
                    held = _record_while_held(
                        mic, self.ptt, max_s=self.cfg.utterance_max_s
                    )
                    if held is None:
                        self._apply(self.session.on_event(Timeout()), mic=mic)
                        continue
                    self._pending_pcm = held
                    text = self._asr_pcm(
                        held, mic=mic, segmenter=segmenter, for_wake=False
                    )
                    if text:
                        _log(f"  [local-asr] {text}")
                    self._apply(
                        self.session.on_event(
                            Segment(
                                text,
                                source="ptt",
                                upload_if_empty=self.asr is None,
                            )
                        ),
                        mic=mic,
                    )
                    self._feed_last_sink(segmenter)
                    continue
                self._apply(effects, mic=mic)
                if self.session.mode is Mode.MEETING:
                    return
                continue

            now = time.monotonic()
            if now - last_heartbeat >= _WAKE_DEBUG_HEARTBEAT_S:
                _log_debug(f"still in {self.session.mode.value}…")
                last_heartbeat = now

            if pcm is None:
                self._apply(self.session.on_event(Timeout()), mic=mic)
                continue

            early_text = self._early_asr_text
            self._early_asr_text = None
            if early_text is not None:
                text = early_text
            else:
                text = self._asr_pcm(
                    pcm, mic=mic, segmenter=segmenter, for_wake=for_wake
                )
            if text:
                label = "wake-asr" if for_wake else "local-asr"
                tag = "early" if early_text is not None else label
                _log(f"  [{tag}] {text}")
            elif for_wake and self.cfg.wake_mode == "kws":
                continue

            self._pending_pcm = pcm
            self._apply(
                self.session.on_event(
                    Segment(text, upload_if_empty=self.asr is None)
                ),
                mic=mic,
            )
            self._feed_last_sink(segmenter)
            if self.session.mode is Mode.MEETING:
                return

    def _feed_last_sink(self, segmenter: VadSegmenter) -> None:
        sink = getattr(self, "_last_sink", None)
        self._last_sink = None
        if sink:
            segmenter.feed_backlog(sink)

    def run_meeting_loop(self) -> None:
        """While recording: same VadSegmenter on mic tap for stop / wake / dialog."""
        recorder = self._recorder
        if recorder is None or not recorder.active:
            self.session.mode = Mode.IDLE
            return

        tap = recorder.mic_tap_reader()
        # Loopback-only meetings have no mic tap — poll hotkey only.
        if self.cfg.meeting_capture == "loopback":
            self._meeting_hotkey_only(recorder)
            return

        energy = effective_wake_energy(
            self.cfg.wake_energy_threshold, listen_source="mic"
        )
        self._reset_vad_state()
        segmenter = _make_segmenter(
            tap.read_block,
            sample_rate=tap.sample_rate,
            block=tap.block,
            cfg=self.cfg,
            energy_threshold=energy,
            max_s=max(2.0, float(self.cfg.wake_max_s)),
            silence_end_s=max(0.25, float(self.cfg.wake_silence_end_s)),
            min_speech_s=max(0.2, float(self.cfg.wake_min_speech_s)),
            speech_gate=self.speech_gate,
        )
        last_print = 0.0
        while self.session.mode is Mode.MEETING and recorder.active:
            if self.meeting_hk is not None and self.meeting_hk.consume_edge_down():
                self._apply(self.session.on_event(Hotkey("meeting")))
                break
            now = time.monotonic()
            if now - last_print > 10.0:
                hk = self.meeting_hk.hotkey_label if self.meeting_hk else "voice"
                _log(f"meeting recording… (stop: {hk} / «стоп запись»)")
                last_print = now

            def _poll() -> None:
                if self.meeting_hk is not None and self.meeting_hk.consume_edge_down():
                    raise _HotkeyAbortError("meeting")
                if not recorder.active:
                    raise _HotkeyAbortError("meeting")

            # With Silero, speech end is the gate — skip partial-ASR thrash.
            early = None
            if self.speech_gate is None and self.asr is not None:

                def early(pcm: np.ndarray) -> bool:
                    text = _transcribe(
                        self.asr, pcm, sample_rate=self.cfg.sample_rate
                    )
                    if not text:
                        return False
                    if match_local_command(
                        text,
                        meeting_start=self.cfg.cmd_meeting_start,
                        meeting_stop=self.cfg.cmd_meeting_stop,
                        run_test=self.cfg.cmd_run_test,
                        exit_dialog=self.cfg.cmd_exit,
                    ) is not None:
                        self._early_asr_text = text
                        return True
                    if matches_wake_phrase(
                        text,
                        phrases=self.cfg.wake_phrases or None,
                        greetings=self.cfg.wake_greetings or None,
                    ):
                        self._early_asr_text = text
                        return True
                    return False

            self._early_asr_text = None
            try:
                pcm = segmenter.next_segment(
                    self._deadline,
                    poll=_poll,
                    early_check=early,
                    early_check_interval_s=1.0,
                    early_check_min_s=0.7,
                )
            except _HotkeyAbortError:
                self._apply(self.session.on_event(Hotkey("meeting")))
                break

            if pcm is None:
                self._apply(self.session.on_event(Timeout()))
                continue
            if frame_rms(pcm) < energy * 0.5:
                continue

            early_text = self._early_asr_text
            self._early_asr_text = None
            if early_text is not None:
                text = early_text
            else:
                text = ""
                if self.asr is not None:
                    try:
                        text = _transcribe(
                            self.asr, pcm, sample_rate=self.cfg.sample_rate
                        )
                    except Exception:
                        text = ""
            if text:
                tag = "early" if early_text is not None else "meeting-asr"
                _log(f"  [{tag}] {text}")
            self._pending_pcm = pcm
            self._apply(
                self.session.on_event(
                    Segment(text, upload_if_empty=self.asr is None)
                )
            )
            if self.session.mode is not Mode.MEETING:
                break

        if self._recorder is not None and self.session.mode is not Mode.MEETING:
            # StopMeeting effect already handled upload.
            pass

    def _meeting_hotkey_only(self, recorder: MeetingRecorder) -> None:
        last_print = 0.0
        while self.session.mode is Mode.MEETING and recorder.active:
            if self.meeting_hk is not None and self.meeting_hk.consume_edge_down():
                self._apply(self.session.on_event(Hotkey("meeting")))
                break
            now = time.monotonic()
            if now - last_print > 10.0:
                hk = self.meeting_hk.hotkey_label if self.meeting_hk else "hotkey"
                _log(f"meeting recording… (stop: {hk})")
                last_print = now
            time.sleep(0.05)


class _HotkeyAbortError(Exception):
    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


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
            f"(beeps={'on' if cfg.talk_beeps else 'off'})"
        )
    _log(
        f"listen_timeout: {cfg.talk_listen_timeout_s:.0f}s, "
        f"beeps={'on' if cfg.talk_beeps else 'off'}"
    )
    if cfg.meeting_enabled:
        _log(f"meeting: capture={cfg.meeting_capture}, hotkey={cfg.meeting_hotkey}")

    speech_gate, silero, vad_label = _load_speech_gate(cfg)
    if vad_label == "silero":
        _log(
            f"vad: silero (thr={cfg.vad_threshold:.2f}, "
            f"pregate_rms={cfg.vad_energy_pregate:.4f})"
        )
    else:
        _log(f"vad: energy (thr≥{wake_energy:.4f})")

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

    # Local commands during talk/meeting need sherpa even when wake_mode=kws.
    if asr is None and cfg.wake_mode == "kws":
        try:
            from krabobot_voice.local_asr import LocalWakeAsr, resolve_stt_model_dir

            model_dir = resolve_stt_model_dir(cfg.stt_model_dir)
            asr = LocalWakeAsr(model_dir, num_threads=cfg.stt_num_threads)
            _log(f"command ASR model: {model_dir.name}")
        except Exception:
            asr = None

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
            f"backend={vad_label}"
        )
    elif cfg.wake_mode == "asr":
        hints.append(f'скажите «{wake_label}» (ASR)')
        _log(
            f"wake ASR: VAD max={cfg.wake_max_s:.1f}s "
            f"silence_end={cfg.wake_silence_end_s:.2f}s "
            f"backend={vad_label} phrases={cfg.wake_phrases!r}"
        )
    if ptt is not None:
        hints.append(f"или PTT {ptt.hotkey_label}")
    if meeting_hk is not None:
        hints.append(f"или meeting {meeting_hk.hotkey_label}")
    _log("Готов. " + " ".join(hints) + ". Ctrl+C — выход.")

    session = VoiceSession(_session_config(cfg))
    driver = _Driver(
        cfg,
        http,
        session,
        asr=asr,
        kws=kws,
        ptt=ptt,
        meeting_hk=meeting_hk,
        wake_energy=wake_energy,
        speech_gate=speech_gate,
        silero=silero,
    )
    talk_enabled = cfg.wake_mode != "off" or ptt is not None

    try:
        while True:
            if session.mode is Mode.MEETING:
                if driver._recorder is None:
                    driver._start_meeting()
                driver.run_meeting_loop()
                continue

            if meeting_hk is not None and meeting_hk.consume_edge_down():
                driver._apply(session.on_event(Hotkey("meeting")))
                continue

            if not talk_enabled:
                time.sleep(0.05)
                continue

            with open_capture_stream(
                cfg.audio_listen_source,
                sample_rate=cfg.sample_rate,
                input_device=cfg.audio_input_device or None,
                loopback_device=cfg.meeting_loopback_device or None,
                output_device=cfg.audio_output_device or None,
            ) as mic:
                driver.run_talk_loop(mic)
    except KeyboardInterrupt:
        _log("stopped.")
        return 0
    except RuntimeError as e:
        _log(f"ERROR: {e}")
        return 4
    finally:
        if driver._recorder is not None:
            try:
                driver._recorder.stop()
            except Exception:
                pass
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
