"""Console ASR/PTT → segment stream → VoiceSession → effects driver.

Wake default: persistent VAD segmenter → local sherpa ASR → pure state machine.
Dialog follow-up after Talk reply. Meeting: same segmenter on mic tap.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from krabobot_voice.audio_io import (
    open_capture_stream,
    play_beeps,
    play_wav_bytes,
    with_stream_keepalive,
)
from krabobot_voice.commands import match_utterance_command
from krabobot_voice.config import VoiceClientConfig, resolve_stt_num_threads
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
from krabobot_voice.meeting import MeetingCaptureConfig, MeetingRecorder, default_meetings_dir
from krabobot_voice.protocol import ClientState
from krabobot_voice.ptt import PttHotkey
from krabobot_voice.segmenter import ListenCountdown, VadSegmenter, listen_countdown_status
from krabobot_voice.single_instance import (
    ALREADY_RUNNING_EXIT_CODE,
    ensure_single_instance,
)
from krabobot_voice.status_bus import StatusBus, get_status_bus
from krabobot_voice.vad import frame_rms, pcm_stats
from krabobot_voice.wake import command_after_wake, matches_wake_phrase

# Loopback after downmix/resample is often quieter than a close mic
# (energy pre-gate / preprocess only — speech decision is Silero).
_LOOPBACK_WAKE_ENERGY_SCALE = 0.4
_LOOPBACK_WAKE_ENERGY_FLOOR = 0.0015
_WAKE_DEBUG_HEARTBEAT_S = 30.0
_KEEPALIVE_SINK_MAX = 200
# Cadence between early ASR probes after the first (wake.early_asr_s) hit window.
_EARLY_ASR_INTERVAL_S = 0.6


def _log(msg: str) -> None:
    print(msg, flush=True)
    bus = get_status_bus()
    if bus is not None:
        bus.emit_log(str(msg))


def resolve_meetings_save_dir(cfg: VoiceClientConfig) -> Path:
    """Explicit ``meeting.save_dir`` or portable ``<app>/meetings``."""
    raw = (cfg.meeting_save_dir or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return default_meetings_dir().resolve()


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


def _is_wake_or_command(
    text: str,
    *,
    cfg: VoiceClientConfig,
    for_wake: bool,
) -> bool:
    """True if ASR text is a wake hit and/or a local voice command."""
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if for_wake and matches_wake_phrase(
        cleaned,
        phrases=cfg.wake_phrases or None,
        greetings=cfg.wake_greetings or None,
    ):
        return True
    if match_utterance_command(
        cleaned,
        wake_phrases=cfg.wake_phrases or None,
        wake_greetings=cfg.wake_greetings or None,
        meeting_start=cfg.cmd_meeting_start,
        meeting_stop=cfg.cmd_meeting_stop,
        run_test=cfg.cmd_run_test,
        exit_dialog=cfg.cmd_exit,
    ) is not None:
        return True
    # Meeting mic-tap also accepts wake (arm listen / trailing command).
    if not for_wake and matches_wake_phrase(
        cleaned,
        phrases=cfg.wake_phrases or None,
        greetings=cfg.wake_greetings or None,
    ):
        return True
    return False


def _match_cfg_command(text: str, *, cfg: VoiceClientConfig) -> object | None:
    """Local command in utterance (full text or after configured wake strip)."""
    return match_utterance_command(
        text,
        wake_phrases=cfg.wake_phrases or None,
        wake_greetings=cfg.wake_greetings or None,
        meeting_start=cfg.cmd_meeting_start,
        meeting_stop=cfg.cmd_meeting_stop,
        run_test=cfg.cmd_run_test,
        exit_dialog=cfg.cmd_exit,
    )


def _log_asr_text(text: str, *, tag: str, matched: bool) -> None:
    """Print wake/meeting/local ASR lines only on match (or KRABOBOT_VOICE_DEBUG)."""
    cleaned = (text or "").strip()
    if not cleaned:
        return
    if matched or _debug_enabled():
        _log(f"  [{tag}] {cleaned}")


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
        from krabobot_voice.config import is_frozen

        silero = load_silero_vad(
            model_path=cfg.vad_model_path or None,
            threshold=float(cfg.vad_threshold),
            sample_rate=int(cfg.sample_rate),
            download=not is_frozen(),
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


# Audible cue once meeting capture is actually open (not when the command is queued).
_MEETING_START_BEEPS = 3


def cue_meeting_recording_started(
    play: Callable[[int], None] | None = None,
    *,
    background: bool = True,
) -> threading.Thread | None:
    """Play three short beeps after meeting capture is ready.

    The worker already owns the mic, so this runs on a daemon thread and does
    not block the UI or the meeting stop / mic-tap loop. Gaps come from
    ``play_beeps`` (winsound on Windows, generated tone otherwise).
    """
    player = play if play is not None else play_beeps

    def _run() -> None:
        try:
            player(_MEETING_START_BEEPS)
        except Exception as e:
            _log(f"WARNING: meeting start beep failed: {e}")

    if not background:
        _run()
        return None
    thread = threading.Thread(target=_run, name="meeting-start-beep", daemon=True)
    thread.start()
    return thread


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
        stop_event: threading.Event | None = None,
        command_queue: queue.Queue[str] | None = None,
        status_bus: StatusBus | None = None,
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
        self._stop_event = stop_event
        self._command_queue = command_queue
        self._status_bus = status_bus if status_bus is not None else get_status_bus()
        self._deadline: float | None = None
        self._countdown: ListenCountdown | None = None
        self._pending_pcm: np.ndarray | None = None
        self._recorder: MeetingRecorder | None = None
        self._last_status = ""
        self._early_asr_text: str | None = None
        self._meeting_upload_lock = threading.Lock()
        self._meeting_upload_thread: threading.Thread | None = None
        self._meeting_upload_path: Path | None = None
        self._turn_inflight = False
        self._turn_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._turn_sink: deque[np.ndarray] | None = None
        self._publish_mode()

    def _stopped(self) -> bool:
        return self._stop_event is not None and self._stop_event.is_set()

    def _bus(self) -> StatusBus | None:
        return self._status_bus if self._status_bus is not None else get_status_bus()

    def _publish_mode(self) -> None:
        bus = self._bus()
        if bus is not None:
            bus.set_mode(self.session.mode.value)

    def _reset_vad_state(self) -> None:
        reset = getattr(self._silero, "reset", None)
        if callable(reset):
            reset()

    def _status(self, msg: str) -> None:
        if msg != self._last_status:
            _log(msg)
            self._last_status = msg
            bus = self._bus()
            if bus is not None:
                bus.set_status(msg)
                self._publish_mode()

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
                    self._countdown = None
                else:
                    self._countdown = ListenCountdown(float(eff.seconds))
                    self._deadline = self._countdown.deadline
            elif isinstance(eff, SendText):
                self._do_text_turn(eff.text, mic=mic)
            elif isinstance(eff, SendAudio):
                self._do_audio_turn(mic=mic)
            elif isinstance(eff, StartMeeting):
                # Defer device open: talk loop may still hold the mic exclusively.
                # Outer run() starts MeetingRecorder after open_capture_stream closes.
                pass
            elif isinstance(eff, StopMeeting):
                # Drop any mic-tap clip so a later SendAudio cannot upload the stop phrase.
                self._pending_pcm = None
                self._stop_meeting_and_upload()
            elif isinstance(eff, RunTest):
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _log(f"[test] {now} — тест выполнен")
        self._publish_mode()

    def _start_turn_worker(
        self,
        call: Callable[[], object],
        *,
        mic: object | None,
    ) -> None:
        """Run HTTP turn off the voice control path; loop polls for completion."""
        bus = self._bus()
        if bus is not None:
            bus.set_activity("thinking")
        self._status("thinking…")
        sink: deque[np.ndarray] | None = None
        if mic is not None:
            sink = deque(maxlen=_KEEPALIVE_SINK_MAX)
        self._turn_sink = sink
        self._turn_inflight = True

        def _worker() -> None:
            try:
                result = call()
                self._turn_queue.put(("ok", result))
            except Exception as e:  # noqa: BLE001 — delivered to voice loop
                self._turn_queue.put(("err", e))

        threading.Thread(
            target=_worker,
            name="voice-turn",
            daemon=True,
        ).start()

    def _do_text_turn(self, text: str, *, mic: object | None) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            self._apply(
                self.session.on_event(TurnDone(ok=False)),
                mic=mic,
            )
            return
        _log(f"  you (wake): {cleaned}")
        state = _client_state(self.session)

        def _call() -> object:
            return self.http.turn(instruct=cleaned, client_state=state)

        self._start_turn_worker(_call, mic=mic)

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
        state = _client_state(self.session)

        def _call() -> object:
            return self.http.turn(
                audio_bytes=wav,
                audio_filename="utterance.wav",
                client_state=state,
            )

        self._start_turn_worker(_call, mic=mic)

    def _drain_turn_keepalive(self, mic: object | None) -> None:
        sink = self._turn_sink
        if mic is None or sink is None:
            return
        try:
            drain = getattr(mic, "drain_available", None)
            if callable(drain):
                drain()
            block = getattr(mic, "read_block", None)
            if callable(block):
                sink.append(block())
        except Exception:
            pass

    def _poll_pending_turn(self, *, mic: object | None) -> bool:
        """Keep mic drained while HTTP runs. Return True if still in flight."""
        if not self._turn_inflight:
            return False
        self._drain_turn_keepalive(mic)
        try:
            kind, payload = self._turn_queue.get_nowait()
        except queue.Empty:
            return True

        self._turn_inflight = False
        self._last_sink = self._turn_sink
        self._turn_sink = None
        bus = self._bus()
        if bus is not None and bus.snapshot().activity == "thinking":
            bus.set_activity("")

        if kind == "err":
            _log(f"ERROR: запрос /v1/voice/turn: {payload}")
            self._apply(self.session.on_event(TurnDone(ok=False)), mic=mic)
            return False

        result = payload
        ok = self._finish_turn(result, mic=mic)
        self._apply(
            self.session.on_event(
                TurnDone(ok=ok, actions=tuple(getattr(result, "actions", None) or ()))
            ),
            mic=mic,
        )
        return False

    def _finish_turn(self, result: object, *, mic: object | None) -> bool:
        status = int(getattr(result, "status_code", 500) or 500)
        if status >= 400:
            _log(
                f"ERROR HTTP {status}: "
                f"{getattr(result, 'error_message', None) or getattr(result, 'reply', None) or 'unknown'}"
            )
            return False
        actions = tuple(getattr(result, "actions", None) or ())
        if "meeting_stop" in actions:
            if mic is not None:
                _drain_mic(mic, blocks=12)
            return True
        ok = _play_turn_result(result)
        if mic is not None:
            _drain_mic(mic, blocks=12)
        return ok

    def _start_meeting(self) -> None:
        if self._recorder is not None and self._recorder.active:
            return
        save_dir = str(resolve_meetings_save_dir(self.cfg))
        recorder = MeetingRecorder(
            MeetingCaptureConfig(
                capture=self.cfg.meeting_capture,
                sample_rate=self.cfg.sample_rate,
                input_device=self.cfg.audio_input_device,
                output_device=self.cfg.audio_output_device,
                loopback_device=self.cfg.meeting_loopback_device,
                max_s=self.cfg.meeting_max_s,
            ),
            save_dir=save_dir,
        )
        echo_hint = ""
        if self.cfg.meeting_capture == "mix":
            echo_hint = (
                " (mix: лучше наушники — иначе локальный голос может задвоиться с колонок;"
                " worker держит mic+loopback — wake на основном стриме на паузе)"
            )
        try:
            wav_path = recorder.start()
        except Exception as e:
            _log(f"ERROR: не удалось начать запись встречи: {e}")
            self._recorder = None
            self.session.mode = Mode.IDLE
            self._publish_mode()
            return
        warn = getattr(recorder, "warning", "") or ""
        if warn:
            _log(f"WARNING: {warn}")
        worker = "process" if getattr(recorder, "use_process", False) else "thread"
        _log(
            f"meeting START capture={self.cfg.meeting_capture} "
            f"worker={worker} → {wav_path}{echo_hint}"
        )
        self._recorder = recorder
        # Capture is open (start() waited until ready). Cue off the voice thread.
        cue_meeting_recording_started()

    def _stop_meeting_and_upload(self) -> None:
        recorder = self._recorder
        self._recorder = None
        if recorder is None:
            return
        try:
            result = recorder.stop()
        except Exception as e:
            msg = str(e)
            if msg.startswith("[open]") or "Не удалось открыть" in msg:
                _log(f"ERROR: не удалось начать запись встречи: {msg.removeprefix('[open] ').removeprefix('[record] ')}")
            else:
                detail = msg.removeprefix("[record] ").removeprefix("[open] ")
                _log(f"ERROR: остановка записи встречи: {detail}")
            self.session.mode = Mode.IDLE
            self._publish_mode()
            return
        warn = getattr(recorder, "warning", "") or ""
        if warn:
            _log(f"WARNING: {warn}")
        capture = result.capture
        if (
            self.cfg.meeting_capture == "mix"
            and capture == "mic"
        ):
            _log("WARNING: meeting saved as mic-only (loopback was unavailable)")
        _log(
            f"meeting STOP ({result.duration_s:.1f} s) saved={result.wav_path}"
        )
        self._handle_meeting_upload(
            result.wav_path,
            duration_s=result.duration_s,
            capture=result.capture,
            wav_bytes=result.wav_bytes,
        )

    def _finalize_meeting_session(self) -> None:
        """Ensure IDLE + recorder cleaned after meeting loop exits (success or fail)."""
        if self._recorder is not None:
            # Worker died or loop exited without StopMeeting effect.
            if self.session.mode is Mode.MEETING:
                self.session.mode = Mode.IDLE
                self._publish_mode()
            self._stop_meeting_and_upload()
        elif self.session.mode is Mode.MEETING:
            self.session.mode = Mode.IDLE
            self._publish_mode()

    def _handle_meeting_upload(
        self,
        wav_path: Path | str,
        *,
        duration_s: float,
        capture: str,
        wav_bytes: bytes = b"",
    ) -> None:
        """Log path and schedule HTTP upload off the voice loop thread."""
        path = Path(wav_path)
        upload_as = (self.cfg.meeting_upload_as or "file").strip().lower()
        size = 0
        try:
            size = path.stat().st_size
        except OSError:
            size = len(wav_bytes)
        _log(
            f"meeting upload as={upload_as}: {path.name} ({size} bytes, "
            f"{duration_s:.1f} s, capture={capture})"
        )
        with self._meeting_upload_lock:
            prev = self._meeting_upload_thread
            if (
                prev is not None
                and prev.is_alive()
                and self._meeting_upload_path == path
            ):
                _log("meeting upload already in progress — skip duplicate")
                return
            _log("upload meeting (async)…")
            bus = self._bus()
            if bus is not None:
                bus.set_uploading(True)
            thread = threading.Thread(
                target=self._meeting_upload_worker,
                args=(path, upload_as, wav_bytes),
                name="meeting-upload",
                daemon=True,
            )
            self._meeting_upload_thread = thread
            self._meeting_upload_path = path
            thread.start()

    def _meeting_upload_worker(
        self,
        path: Path,
        upload_as: str,
        wav_bytes: bytes,
    ) -> None:
        """Background HTTP POST; must not raise into the main process."""
        meeting_timeout = min(float(self.cfg.timeout_s), 60.0)
        bus = self._bus()
        try:
            if upload_as == "audio":
                # Legacy short-clip path (server STT on `audio`) — not for long meetings.
                from krabobot_voice.preprocess import preprocess_wav_bytes

                raw = wav_bytes or path.read_bytes()
                aligned = preprocess_wav_bytes(raw)
                result = self.http.turn(
                    audio_bytes=aligned,
                    audio_filename=path.name or "meeting.wav",
                    instruct=self.cfg.meeting_instruct or None,
                    client_state=ClientState(mode="idle", meeting="idle"),
                    timeout_s=meeting_timeout,
                )
            else:
                # Long meetings: multipart `files` — async queue on server (no TTS wait).
                result = self.http.turn(
                    files=[path],
                    instruct=self.cfg.meeting_instruct or None,
                    client_state=ClientState(mode="idle", meeting="idle"),
                    async_meeting=True,
                    timeout_s=meeting_timeout,
                )
        except Exception as e:
            _log(f"ERROR: запрос /v1/voice/turn (meeting): {e}")
            return
        finally:
            if bus is not None:
                bus.set_uploading(False)
        if result.status_code >= 400:
            _log(
                f"ERROR HTTP {result.status_code}: "
                f"{result.error_message or result.reply or 'unknown'}"
            )
            return
        if result.queued or result.status_code == 202:
            _log("meeting queued — результат придёт на почту")
            return
        _log("WARNING: сервер принял встречу без async=queued; проверьте версию krabobot serve")

    def _poll_ui_command(self) -> str | None:
        q = self._command_queue
        if q is None:
            return None
        try:
            raw = str(q.get_nowait() or "").strip().lower()
        except queue.Empty:
            return None
        if raw in {"meeting", "ptt", "quit"}:
            return raw
        return None

    def _poll_hotkeys(self) -> list[object] | None:
        """Return effects if a hotkey/UI command was consumed; else None."""
        ui = self._poll_ui_command()
        if ui == "quit":
            if self._stop_event is not None:
                self._stop_event.set()
            return None
        if ui == "meeting":
            return self.session.on_event(Hotkey("meeting"))
        if ui == "ptt":
            return self.session.on_event(Hotkey("ptt"))
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

    def _early_asr_cadence(self) -> tuple[float, float]:
        """Return (min_s, interval_s) for wake/meeting early ASR probes.

        First probe once buffered voiced audio reaches ``wake.early_asr_s``
        (default 0.8 s, capped by ``wake.max_s``); then every ~0.6 s until
        silence_end or max_s.
        """
        max_s = max(0.5, float(self.cfg.wake_max_s))
        min_s = float(self.cfg.wake_early_asr_s)
        min_s = max(0.3, min(min_s, max_s))
        return min_s, _EARLY_ASR_INTERVAL_S

    def _make_early_check(
        self,
        *,
        for_wake: bool,
        mic: object | None = None,
        segmenter: VadSegmenter | None = None,
        match_wake: bool | None = None,
        match_commands: bool | None = None,
    ) -> Callable[[np.ndarray], bool] | None:
        """Partial-utterance probe after speech_start (Silero or energy).

        Idle wake: wake phrase / KWS. LISTEN/DIALOG: local commands. Meeting
        mic-tap: commands only (``match_commands``; wake+stop via configured
        wake strip in ``match_utterance_command``). Wake-only during meeting
        waits for silence so continuous wake+command is not cut mid-phrase.
        Match → immediate commit; misses stay quiet unless KRABOBOT_VOICE_DEBUG.
        """
        want_wake = for_wake if match_wake is None else bool(match_wake)
        want_cmd = (not for_wake) if match_commands is None else bool(match_commands)

        if want_wake and self.cfg.wake_mode == "kws" and self.kws is not None:

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

        energy = min(self.wake_energy, 0.006) if want_wake and not want_cmd else None

        def _asr_check(pcm: np.ndarray) -> bool:
            def _run() -> str:
                return _transcribe(
                    self.asr,
                    pcm,
                    sample_rate=self.cfg.sample_rate,
                    energy_threshold=energy,
                )

            if mic is not None and segmenter is not None:
                sink: deque[np.ndarray] = deque(maxlen=_KEEPALIVE_SINK_MAX)
                text = with_stream_keepalive(mic, _run, sink=sink)
                segmenter.feed_backlog(sink)
            else:
                text = _run()
            if not text:
                return False
            hit = False
            if want_cmd and _match_cfg_command(text, cfg=self.cfg) is not None:
                hit = True
            if want_wake and matches_wake_phrase(
                text,
                phrases=self.cfg.wake_phrases or None,
                greetings=self.cfg.wake_greetings or None,
            ):
                # Idle wake early-hit. If this check also looks for commands
                # (unusual), only commit when trailing is empty so continuous
                # wake+command is not truncated.
                if want_cmd:
                    trailing = command_after_wake(
                        text,
                        phrases=self.cfg.wake_phrases or None,
                        greetings=self.cfg.wake_greetings or None,
                    )
                    if not trailing.strip():
                        hit = True
                else:
                    hit = True
            if hit:
                self._early_asr_text = text
                _log_debug(f"early-asr hit: {text}")
            return hit

        return _asr_check

    def run_talk_loop(self, mic: object) -> None:
        """Drive one open capture stream until meeting starts or stream should reopen."""
        self._last_sink = None
        # Wake + local-command clips: hard ≤ wake_max_s (default 2 s).
        wake_max = max(0.5, float(self.cfg.wake_max_s))
        wake_silence = max(0.25, float(self.cfg.wake_silence_end_s))
        wake_min = max(0.2, float(self.cfg.wake_min_speech_s))
        talk_min = float(self.cfg.min_speech_s)
        if self.asr is not None:
            talk_min = min(talk_min, 0.75)
        listen_silence = max(0.35, float(self.cfg.talk_listen_silence_end_s))
        talk_energy = (
            self.wake_energy
            if self.cfg.audio_listen_source == "loopback"
            else effective_wake_energy(
                self.cfg.energy_threshold, listen_source=self.cfg.audio_listen_source
            )
        )
        # LISTEN free-form dialog upload may use utterance_max_s; wake/meeting
        # command clips stay on wake_max (≤2 s). Local-command early_check still
        # commits short phrases without waiting for the longer max.
        dialog_max = max(wake_max, float(self.cfg.utterance_max_s))

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
            if self._stopped():
                return

            # HTTP turn runs on a worker; keep draining mic and accept meeting stop.
            if self._turn_inflight:
                if self._stopped():
                    return
                self._status("thinking…")
                hot = self._poll_hotkeys()
                if hot is not None:
                    # Allow meeting toggle even while thinking (stop/start).
                    if any(isinstance(e, StartMeeting) or isinstance(e, StopMeeting) for e in hot):
                        self._apply(hot, mic=mic)
                        if self.session.mode is Mode.MEETING:
                            return
                    # Ignore PTT arm while turn is in flight.
                if self._poll_pending_turn(mic=mic):
                    time.sleep(0.01)
                    continue
                self._feed_last_sink(segmenter)
                continue

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
                    _log_asr_text(
                        text,
                        tag="local-asr",
                        matched=_is_wake_or_command(
                            text, cfg=self.cfg, for_wake=False
                        ),
                    )
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
                # LISTEN/DIALOG: snappy silence_end; local-command early_check
                # commits ≤2 s phrases. Segment max stays dialog-sized for uploads.
                segmenter.configure(
                    energy_threshold=talk_energy,
                    max_s=dialog_max,
                    silence_end_s=listen_silence,
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
                if self._stopped():
                    raise _HotkeyAbortError("stop")
                ui = self._poll_ui_command()
                if ui == "quit":
                    if self._stop_event is not None:
                        self._stop_event.set()
                    raise _HotkeyAbortError("stop")
                if ui == "meeting":
                    raise _HotkeyAbortError("meeting")
                if ui == "ptt":
                    raise _HotkeyAbortError("ptt")
                if self.meeting_hk is not None and self.meeting_hk.consume_edge_down():
                    raise _HotkeyAbortError("meeting")
                if self.ptt is not None and self.ptt.consume_edge_down():
                    raise _HotkeyAbortError("ptt")
                cd = self._countdown
                if mode in (Mode.LISTEN, Mode.DIALOG) and cd is not None:
                    # Speech hold freezes the line; silence publishes the seconds left.
                    self._status(
                        listen_countdown_status(cd, follow_up=mode is Mode.DIALOG)
                    )

            for_wake = mode is Mode.IDLE
            early = self._make_early_check(
                for_wake=for_wake, mic=mic, segmenter=segmenter
            )
            early_min_s, early_interval_s = self._early_asr_cadence()
            self._early_asr_text = None

            try:
                # Idle wake + LISTEN local-commands: cadence ASR after speech_start
                # (Silero remains the speech gate; no sliding-window spam).
                # LISTEN/DIALOG: silence-only countdown (held while VAD speech).
                listen_cd = (
                    self._countdown if mode in (Mode.LISTEN, Mode.DIALOG) else None
                )
                pcm = segmenter.next_segment(
                    None if listen_cd is not None else self._deadline,
                    poll=_poll,
                    early_check=early,
                    early_check_interval_s=early_interval_s,
                    early_check_min_s=early_min_s,
                    countdown=listen_cd,
                )
            except _HotkeyAbortError as abort:
                if abort.kind == "stop":
                    return
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
                    _log_asr_text(
                        text,
                        tag="local-asr",
                        matched=_is_wake_or_command(
                            text, cfg=self.cfg, for_wake=False
                        ),
                    )
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
                # early_text is only set on a probe hit → always a match.
                matched = early_text is not None or _is_wake_or_command(
                    text, cfg=self.cfg, for_wake=for_wake
                )
                _log_asr_text(text, tag=tag, matched=matched)
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
            self._finalize_meeting_session()
            return

        try:
            tap = recorder.mic_tap_reader()
            # Loopback-only meetings have no mic tap — poll hotkey only.
            if self.cfg.meeting_capture == "loopback":
                self._meeting_hotkey_only(recorder)
                return

            energy = effective_wake_energy(
                self.cfg.wake_energy_threshold, listen_source="mic"
            )
            # Same ≤2 s command window as idle wake (stop phrase / wake+command).
            cmd_max = max(0.5, float(self.cfg.wake_max_s))
            self._reset_vad_state()
            segmenter = _make_segmenter(
                tap.read_block,
                sample_rate=tap.sample_rate,
                block=tap.block,
                cfg=self.cfg,
                energy_threshold=energy,
                max_s=cmd_max,
                silence_end_s=max(0.25, float(self.cfg.wake_silence_end_s)),
                min_speech_s=max(0.2, float(self.cfg.wake_min_speech_s)),
                speech_gate=self.speech_gate,
            )
            last_print = 0.0
            while self.session.mode is Mode.MEETING and recorder.active:
                if self._stopped():
                    break
                if self._turn_inflight:
                    if self._poll_pending_turn(mic=None):
                        time.sleep(0.01)
                        continue
                ui = self._poll_ui_command()
                if ui == "quit":
                    if self._stop_event is not None:
                        self._stop_event.set()
                    break
                if ui == "meeting" or (
                    self.meeting_hk is not None and self.meeting_hk.consume_edge_down()
                ):
                    self._apply(self.session.on_event(Hotkey("meeting")))
                    break
                now = time.monotonic()
                if now - last_print > 10.0:
                    hk = self.meeting_hk.hotkey_label if self.meeting_hk else "voice"
                    peak = float(getattr(tap, "last_rms", 0.0) or 0.0)
                    got = int(getattr(tap, "blocks_read", 0) or 0)
                    empty = int(getattr(tap, "empty_reads", 0) or 0)
                    tap_note = f", tap_peak={peak:.4f} blocks={got}"
                    if got == 0 or (empty > got and peak < 1e-4):
                        tap_note += " (тихо/нет IPC — стоп-фраза может не слышаться)"
                    _log(
                        f"meeting recording… (stop: {hk} / «закончить запись совещания»"
                        f"{tap_note})"
                    )
                    last_print = now

                def _poll() -> None:
                    if self._stopped():
                        raise _HotkeyAbortError("stop")
                    ui_cmd = self._poll_ui_command()
                    if ui_cmd == "quit":
                        if self._stop_event is not None:
                            self._stop_event.set()
                        raise _HotkeyAbortError("stop")
                    if ui_cmd == "meeting":
                        raise _HotkeyAbortError("meeting")
                    if self.meeting_hk is not None and self.meeting_hk.consume_edge_down():
                        raise _HotkeyAbortError("meeting")
                    if not recorder.active:
                        raise _HotkeyAbortError("meeting")

                # Early ASR: local commands only (stop alone, or wake+stop after
                # configured wake strip). Wake-only waits for silence so continuous
                # «wake + stop» is not cut off mid-phrase.
                early = self._make_early_check(
                    for_wake=False,
                    match_wake=False,
                    match_commands=True,
                )
                early_min_s, early_interval_s = self._early_asr_cadence()

                self._early_asr_text = None
                try:
                    # Same silence-only listen budget as talk mode when a
                    # follow-up window is armed; idle meeting recording has none.
                    pcm = segmenter.next_segment(
                        None if self._countdown is not None else self._deadline,
                        poll=_poll,
                        early_check=early,
                        early_check_interval_s=early_interval_s,
                        early_check_min_s=early_min_s,
                        countdown=self._countdown,
                    )
                except _HotkeyAbortError as abort:
                    if abort.kind == "stop":
                        break
                    self._apply(self.session.on_event(Hotkey("meeting")))
                    break

                # Capture early hit *before* any discard — next iteration clears
                # ``_early_asr_text`` at the top of the loop.
                early_text = self._early_asr_text
                self._early_asr_text = None

                if pcm is None:
                    self._apply(self.session.on_event(Timeout()))
                    continue
                # Quiet clips are noise — but an early ASR/KWS hit already matched
                # a stop/wake command; never drop that (RMS gate was discarding
                # «закончить запись» and leaving the meeting recording).
                if early_text is None and frame_rms(pcm) < energy * 0.5:
                    continue

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
                    matched = early_text is not None or _is_wake_or_command(
                        text, cfg=self.cfg, for_wake=False
                    )
                    _log_asr_text(text, tag=tag, matched=matched)
                self._pending_pcm = pcm
                self._apply(
                    self.session.on_event(
                        Segment(text, upload_if_empty=self.asr is None)
                    )
                )
                if self.session.mode is not Mode.MEETING:
                    break
        finally:
            # Worker death / early return must not leave MEETING + paused main mic.
            self._finalize_meeting_session()

    def _meeting_hotkey_only(self, recorder: MeetingRecorder) -> None:
        last_print = 0.0
        while self.session.mode is Mode.MEETING and recorder.active:
            if self._stopped():
                break
            ui = self._poll_ui_command()
            if ui == "quit":
                if self._stop_event is not None:
                    self._stop_event.set()
                break
            if ui == "meeting" or (
                self.meeting_hk is not None and self.meeting_hk.consume_edge_down()
            ):
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


def run_loop(
    config: VoiceClientConfig | None = None,
    *,
    stop_event: threading.Event | None = None,
    command_queue: queue.Queue[str] | None = None,
    status_bus: StatusBus | None = None,
    config_path: str | Path | None = None,
) -> int:
    cfg = config or VoiceClientConfig.load()
    bus = status_bus if status_bus is not None else get_status_bus()
    if bus is not None:
        cfg_path = ""
        if config_path is not None:
            cfg_path = str(Path(config_path).expanduser())
        else:
            from krabobot_voice.config import app_config_candidates, ensure_app_config

            ensured = ensure_app_config()
            cfg_path = str(ensured) if ensured else str(app_config_candidates()[0])
        bus.set_paths(
            config_path=cfg_path,
            meetings_dir=str(resolve_meetings_save_dir(cfg)),
        )
        bus.set_mode("idle")
        bus.set_activity("")
        bus.set_status("starting…")

    cpu_n = max(1, int(os.cpu_count() or 1))
    # load() already resolves; re-resolve so bare VoiceClientConfig() (0) is safe.
    stt_threads = resolve_stt_num_threads(cfg.stt_num_threads, cpu_count=cpu_n)
    cfg.stt_num_threads = stt_threads
    wake_energy = effective_wake_energy(
        cfg.wake_energy_threshold, listen_source=cfg.audio_listen_source
    )
    _log(f"krabobot-voice → {cfg.base_url}")
    _log(f"device_id={cfg.device_id}")
    _log(f"wake_mode={cfg.wake_mode}, ptt={'on' if cfg.ptt_enabled else 'off'}")
    _log(f"audio.listen_source={cfg.audio_listen_source}")
    _log(f"meetings dir: {resolve_meetings_save_dir(cfg)}")

    # onnxruntime MUST load before WinRT (mic permission). Reverse order =
    # native ACCESS_VIOLATION; PyInstaller console shows no Python traceback.
    if sys.platform == "win32" and (cfg.vad_backend or "silero").strip().lower() == "silero":
        try:
            from krabobot_voice.silero_vad import preload_onnxruntime

            preload_onnxruntime()
        except Exception as e:
            _log(f"WARNING: onnxruntime preload failed ({e})")

    needs_mic = cfg.audio_listen_source == "mic" or cfg.meeting_enabled
    if needs_mic and sys.platform == "win32":
        from krabobot_voice.mic_permission import ensure_microphone_access

        _log("Запрос доступа к микрофону (системное окно Windows)…")
        mic_access = ensure_microphone_access(interactive=True)
        _log(f"mic access: {mic_access.status}" + (f" ({mic_access.detail})" if mic_access.detail else ""))
        if mic_access.status == "denied":
            _log(
                "ERROR: нет доступа к микрофону. "
                "Разрешите в системном окне / Параметрах и запустите снова."
            )
            return 4

    if stop_event is not None and stop_event.is_set():
        return 0

    auto_threads = max(1, cpu_n // 2)
    if stt_threads == auto_threads:
        _log(f"ASR threads: {stt_threads} (50% of {cpu_n} CPUs)")
    else:
        _log(f"ASR threads: {stt_threads} (config override; {cpu_n} CPUs)")
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
            asr = LocalWakeAsr(model_dir, num_threads=stt_threads)
        except Exception as e:
            _log(f"ERROR: не удалось загрузить sherpa-onnx: {e}")
            return 3

    # Local commands during talk/meeting need sherpa even when wake_mode=kws.
    if asr is None and cfg.wake_mode == "kws":
        try:
            from krabobot_voice.local_asr import LocalWakeAsr, resolve_stt_model_dir

            model_dir = resolve_stt_model_dir(cfg.stt_model_dir)
            asr = LocalWakeAsr(model_dir, num_threads=stt_threads)
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
            f"early={cfg.wake_early_asr_s:.1f}s "
            f"silence_end={cfg.wake_silence_end_s:.2f}s "
            f"backend={vad_label}"
        )
    elif cfg.wake_mode == "asr":
        hints.append(f'скажите «{wake_label}» (ASR)')
        _log(
            f"wake ASR: VAD max={cfg.wake_max_s:.1f}s "
            f"early={cfg.wake_early_asr_s:.1f}s "
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
        stop_event=stop_event,
        command_queue=command_queue,
        status_bus=bus,
    )
    talk_enabled = cfg.wake_mode != "off" or ptt is not None

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                _log("stopped.")
                return 0
            if session.mode is Mode.MEETING:
                if driver._recorder is None:
                    driver._start_meeting()
                driver.run_meeting_loop()
                continue

            if meeting_hk is not None and meeting_hk.consume_edge_down():
                driver._apply(session.on_event(Hotkey("meeting")))
                continue
            ui = driver._poll_ui_command()
            if ui == "quit":
                if stop_event is not None:
                    stop_event.set()
                _log("stopped.")
                return 0
            if ui == "meeting":
                driver._apply(session.on_event(Hotkey("meeting")))
                continue
            if ui == "ptt":
                driver._apply(session.on_event(Hotkey("ptt")))
                # Fall through to open talk stream if needed.

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
        if bus is not None:
            bus.set_activity("")
            bus.set_mode("idle")
            bus.set_status("stopped")


def _configure_stdio_utf8() -> None:
    """Avoid UnicodeEncodeError on Windows consoles (cp1251) / frozen exe."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _pause_on_fatal_if_frozen() -> None:
    """Keep the console open after a fatal error in a double-clicked frozen exe."""
    from krabobot_voice.config import is_frozen

    if not is_frozen():
        return
    try:
        input("\nНажмите Enter для выхода… ")
    except Exception:
        time.sleep(8.0)


def main(argv: list[str] | None = None) -> int:
    _configure_stdio_utf8()
    try:
        return _main_inner(argv)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        _log("stopped.")
        return 0
    except BaseException as e:
        import traceback

        traceback.print_exc()
        try:
            sys.stderr.flush()
            sys.stdout.flush()
        except Exception:
            pass
        _log(f"FATAL: {type(e).__name__}: {e}")
        _pause_on_fatal_if_frozen()
        return 1


def _main_inner(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--list-devices" in args:
        from krabobot_voice.audio_io import (
            format_input_devices_lines,
            list_input_devices,
            mic_privacy_hint,
            require_sounddevice,
        )

        try:
            require_sounddevice()
        except RuntimeError as e:
            _log(f"ERROR: {e}")
            return 2
        devices = list_input_devices()
        _log("Input devices (PortAudio / sounddevice):")
        _log(format_input_devices_lines(devices))
        _log(mic_privacy_hint())
        _log("Задайте audio.input_device подстрокой имени или индексом, либо оставьте пустым.")
        return 0
    if "--test-loopback" in args:
        args = [a for a in args if a != "--test-loopback"]
        path = args[0] if args and not args[0].startswith("-") else None
        cfg = VoiceClientConfig.load(path)
        return run_test_loopback(cfg, duration_s=3.0)

    console_only = "--console" in args
    # After one-shot flags (--list-devices, --test-loopback), which must not lock.
    if not ensure_single_instance(console=console_only):
        return ALREADY_RUNNING_EXIT_CODE
    # CLI forces this run only; persistent default is ui.start_minimized in YAML.
    cli_minimized = "--minimized" in args or "--start-minimized" in args
    args = [
        a
        for a in args
        if a not in {"--console", "--ui", "--minimized", "--start-minimized"}
    ]
    path = None
    if args and not args[0].startswith("-"):
        path = args[0]
    cfg = VoiceClientConfig.load(path)

    if console_only:
        return run_loop(cfg, config_path=path)

    # Default: tray + window UI (Architecture A — one process).
    try:
        from krabobot_voice.ui.app_ui import run_ui
    except ImportError as e:
        _log(
            f"WARNING: UI deps missing ({e}). "
            'Install: pip install -e ".\\clients\\krabobot-voice[ui]" '
            "or run with --console"
        )
        return run_loop(cfg, config_path=path)
    start_minimized = bool(cli_minimized or cfg.ui_start_minimized)
    return run_ui(cfg, config_path=path, start_minimized=start_minimized)
