"""Client configuration for krabobot-voice."""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _default_device_id() -> str:
    host = (socket.gethostname() or "pc").strip().lower()
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in host)[:40]
    return f"local-{safe or 'pc'}"


def local_appdata_config_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "krabobot-voice" / "config.yaml"


def read_admin_token_from_krabobot() -> str:
    """Read api.auth.adminToken from ~/.krabobot/config.json if present."""
    path = Path.home() / ".krabobot" / "config.json"
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    auth = data.get("api", {}) if isinstance(data, dict) else {}
    if not isinstance(auth, dict):
        return ""
    auth = auth.get("auth") or {}
    if not isinstance(auth, dict):
        return ""
    return str(auth.get("adminToken") or "").strip()


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        val = os.getenv(name)
        if val is not None and str(val).strip():
            return str(val).strip()
    return default


def _as_bool(val: Any, default: bool = False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default


def _as_str_list(val: Any, default: list[str]) -> list[str]:
    if val is None:
        return list(default)
    if isinstance(val, str):
        parts = [p.strip() for p in val.split(",")]
        return [p for p in parts if p] or list(default)
    if isinstance(val, (list, tuple)):
        out = [str(x).strip() for x in val if str(x).strip()]
        return out or list(default)
    return list(default)


def _resolve_wake_phrases(
    wake: dict[str, Any],
    data: dict[str, Any],
    default: list[str],
) -> list[str]:
    """Prefer ``wake.phrase`` (one string), else ``wake.phrases`` (list), else default.

    ``wake.phrase: "Эй, Арнольд"`` is always one phrase (comma is punctuation).
    If both ``phrase`` and ``phrases`` are set, ``phrase`` is prepended (deduped).
    """
    primary = wake.get("phrase")
    if primary is None or primary == "":
        flat = data.get("wake_phrase")
        if flat not in (None, ""):
            primary = flat

    listed = wake.get("phrases")
    if listed is None or listed == "":
        listed = data.get("wake_phrases")

    phrases: list[str] = []
    if isinstance(primary, str) and primary.strip():
        phrases.append(primary.strip())
    elif isinstance(primary, (list, tuple)):
        phrases.extend(_as_str_list(primary, []))

    if listed is not None and listed != "":
        phrases.extend(_as_str_list(listed, []))

    seen: set[str] = set()
    out: list[str] = []
    for p in phrases:
        key = " ".join(p.lower().replace("ё", "е").split())
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out or list(default)


def _resolve_wake_greetings(
    wake: dict[str, Any],
    data: dict[str, Any],
    phrases: list[str],
    default: list[str],
) -> list[str]:
    """Explicit ``wake.greetings``, else derive from multi-word phrases, else default."""
    if "greetings" in wake:
        raw = wake.get("greetings")
        if isinstance(raw, (list, tuple)):
            # Allow explicit empty list (disable greeting+name path extras).
            return [str(x).strip() for x in raw if str(x).strip()]
        return _as_str_list(raw, default)
    if "wake_greetings" in data:
        raw = data.get("wake_greetings")
        if isinstance(raw, (list, tuple)):
            return [str(x).strip() for x in raw if str(x).strip()]
        return _as_str_list(raw, default)
    # Derive leading tokens from configured phrases (e.g. "Ок, Бот" → ["ок"]).
    derived: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        toks = " ".join(
            phrase.lower().replace("ё", "е").replace(",", " ").split()
        ).split()
        if len(toks) < 2:
            continue
        for g in toks[:-1]:
            if g and g not in seen:
                seen.add(g)
                derived.append(g)
    return derived if derived else list(default)


def _resolve_listen_timeout(
    talk: dict[str, Any],
    data: dict[str, Any],
    default: float,
) -> float:
    """``talk.listen_timeout_s``, else explicit ``no_speech_timeout_s``, else default."""
    if "listen_timeout_s" in talk and talk["listen_timeout_s"] not in (None, ""):
        return float(talk["listen_timeout_s"])
    if "talk_listen_timeout_s" in data and data["talk_listen_timeout_s"] not in (None, ""):
        return float(data["talk_listen_timeout_s"])
    if "no_speech_timeout_s" in talk and talk["no_speech_timeout_s"] not in (None, ""):
        return float(talk["no_speech_timeout_s"])
    if "no_speech_timeout_s" in data and data["no_speech_timeout_s"] not in (None, ""):
        return float(data["no_speech_timeout_s"])
    return float(default)


@dataclass
class VoiceClientConfig:
    """HTTP + local wake/listen / PTT settings."""

    base_url: str = "http://127.0.0.1:8900"
    device_id: str = field(default_factory=_default_device_id)
    token: str = ""
    timeout_s: float = 120.0
    sample_rate: int = 16000
    # Wake: asr (default, sherpa) | kws (optional MFCC) | off (PTT only)
    # Primary: wake.phrase / wake.phrases in config.yaml (user-settable).
    # Flow: VAD segment → one ASR/KWS score per closed utterance (no sliding window).
    wake_mode: str = "asr"
    # Wake VAD: shorter trailing silence than talk so wake feels snappy.
    wake_silence_end_s: float = 0.55
    # Cap one wake utterance (filler + phrase); avoids endless buffer.
    wake_max_s: float = 9.0
    # Short wake phrases need a lower voiced-floor than talk commands.
    wake_min_speech_s: float = 0.45
    # Lower gate so quiet mics / loopback still reach local ASR (preprocess lifts level).
    wake_energy_threshold: float = 0.006
    wake_phrases: list[str] = field(
        default_factory=lambda: [
            "эй арнольд",
            "ей арнольд",
            "привет арнольд",
            "hey arnold",
        ]
    )
    wake_greetings: list[str] = field(
        default_factory=lambda: ["эй", "ей", "привет", "hey"]
    )
    kws_threshold: float = 0.90
    kws_energy_threshold: float = 0.018
    kws_refs_dir: str = ""
    kws_auto_enroll_tts: bool = True
    # Local sherpa wake (wake_mode=asr)
    stt_model_dir: str = ""
    stt_num_threads: int = 2
    # PTT
    ptt_enabled: bool = True
    ptt_hotkey: str = "ctrl+alt+space"
    # Utterance capture (post-wake VAD)
    utterance_max_s: float = 15.0
    # After speech started: close Talk/LISTEN/DIALOG segment on this much silence.
    silence_end_s: float = 2.0
    speech_start_s: float = 0.25
    min_speech_s: float = 1.2
    settle_s: float = 0.35
    preroll_s: float = 0.4
    no_speech_timeout_s: float = 5.0
    energy_threshold: float = 0.008
    # VAD: silero (default, ONNX speech vs music) | energy (RMS-only, no onnxruntime)
    vad_backend: str = "silero"
    vad_threshold: float = 0.5  # Silero speech probability threshold
    # Skip Silero when frame RMS is below this (pure silence CPU save). Music is louder.
    vad_energy_pregate: float = 0.0008
    vad_model_path: str = ""  # empty = %LOCALAPPDATA%/krabobot-voice/models/silero_vad.onnx
    # Empty listen window (no speech yet): keep ~10s before idle / follow-up end
    talk_listen_timeout_s: float = 10.0  # post-wake window; falls back from no_speech_timeout_s
    talk_follow_up_s: float = 10.0  # 0 = disabled; typical 6–10

    talk_follow_up_beep: bool = False  # legacy; prefer talk.beeps
    talk_beeps: bool = True  # 2 beeps on wake, 1 beep on return to idle
    # Local voice commands (sherpa ASR on utterance before server upload)
    cmd_meeting_start: list[str] = field(
        default_factory=lambda: [
            "начать совещание",
            "начать запись",
            "запиши совещание",
            "начни совещание",
            "начни запись",
        ]
    )
    cmd_meeting_stop: list[str] = field(
        default_factory=lambda: [
            "закончить совещание",
            "завершить запись",
            "стоп запись",
            "закончи совещание",
            "останови запись",
            "стоп совещание",
        ]
    )
    cmd_run_test: list[str] = field(
        default_factory=lambda: [
            "выполни тест",
            "выполнить тест",
            "сделай тест",
            "сделать тест",
            "запусти тест",
            "запустить тест",
            "проведи тест",
            "провести тест",
        ]
    )
    cmd_exit: list[str] = field(
        default_factory=lambda: [
            "хватит",
            "выход",
            "спокойной ночи",
            "отмена",
            "закончили",
            "пока",
        ]
    )
    # Optional device hints (name substring or PortAudio index as string)
    audio_input_device: str = ""
    audio_output_device: str = ""
    # Wake listen path: mic (default) | loopback (WASAPI system audio — wake testing)
    audio_listen_source: str = "mic"
    # Meeting recording (toggle hotkey → background capture → upload+instruct)
    meeting_enabled: bool = True
    meeting_capture: str = "mix"  # mic | loopback | mix
    meeting_hotkey: str = "ctrl+alt+m"
    meeting_loopback_device: str = ""
    meeting_instruct: str = (
        "Это запись встречи (Teams/звонок). Сделай краткое резюме, "
        "ключевые решения и список action items."
    )
    meeting_max_s: float = 7200.0

    @classmethod
    def from_env(cls) -> VoiceClientConfig:
        device = _env_first("KRABOBOT_DEVICE_ID", "KRABOBOT_VOICE_DEVICE_ID")
        listen = (
            _env_first(
                "KRABOBOT_VOICE_LISTEN_SOURCE",
                "VOICE_LISTEN_SOURCE",
            )
            or "mic"
        ).strip().lower()
        if listen not in {"mic", "loopback"}:
            listen = "mic"
        kwargs: dict[str, Any] = {
            "base_url": _env_first(
                "KRABOBOT_URL",
                "KRABOBOT_VOICE_URL",
                default="http://127.0.0.1:8900",
            ).rstrip("/"),
            "device_id": device or _default_device_id(),
            "token": _env_first(
                "KRABOBOT_TOKEN",
                "KRABOBOT_VOICE_TOKEN",
                "KRABOBOT_ADMIN_TOKEN",
            ),
            "timeout_s": float(
                _env_first("KRABOBOT_VOICE_TIMEOUT", default="120") or 120
            ),
            "wake_mode": _env_first("KRABOBOT_VOICE_WAKE_MODE", default="asr")
            or "asr",
            "ptt_hotkey": _env_first(
                "KRABOBOT_VOICE_PTT_HOTKEY",
                default="ctrl+alt+space",
            )
            or "ctrl+alt+space",
            "audio_listen_source": listen,
        }
        wake_phrase = _env_first(
            "KRABOBOT_VOICE_WAKE_PHRASE",
            "VOICE_WAKE_PHRASE",
        )
        if wake_phrase:
            kwargs["wake_phrases"] = [wake_phrase]
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path | None = None) -> VoiceClientConfig:
        """Load YAML/JSON config, then fill gaps from env and ~/.krabobot."""
        cfg = cls.from_env()
        candidates: list[Path] = []
        if path is not None:
            candidates.append(Path(path).expanduser())
        else:
            candidates.append(local_appdata_config_path())
            candidates.append(Path.home() / ".krabobot" / "voice-client.json")
            candidates.append(Path.home() / ".krabobot" / "voice-client.yaml")

        data: dict[str, Any] = {}
        for p in candidates:
            if not p.is_file():
                continue
            text = p.read_text(encoding="utf-8")
            if p.suffix.lower() in {".yaml", ".yml"}:
                try:
                    import yaml  # type: ignore[import-untyped]
                except ImportError as e:
                    raise RuntimeError("PyYAML required for config.yaml") from e
                loaded = yaml.safe_load(text) or {}
            else:
                loaded = json.loads(text)
            if isinstance(loaded, dict):
                data = loaded
                break

        wake = data.get("wake") if isinstance(data.get("wake"), dict) else {}
        ptt = data.get("ptt") if isinstance(data.get("ptt"), dict) else {}
        talk = data.get("talk") if isinstance(data.get("talk"), dict) else {}
        meeting = data.get("meeting") if isinstance(data.get("meeting"), dict) else {}
        audio = data.get("audio") if isinstance(data.get("audio"), dict) else {}
        vad = data.get("vad") if isinstance(data.get("vad"), dict) else {}

        def pick(key: str, *alts: str, default: Any = None) -> Any:
            for k in (key, *alts):
                if k in data and data[k] not in (None, ""):
                    return data[k]
            return default

        def pick_nested(section: dict[str, Any], key: str, flat: str, default: Any) -> Any:
            if key in section and section[key] not in (None, ""):
                return section[key]
            return pick(flat, default=default)

        token = str(pick("token", "admin_token", default=cfg.token) or "").strip()
        if not token:
            token = read_admin_token_from_krabobot()

        wake_mode = str(
            pick_nested(wake, "mode", "wake_mode", cfg.wake_mode) or cfg.wake_mode
        ).strip().lower()
        if wake_mode not in {"kws", "asr", "off"}:
            wake_mode = "asr"

        meeting_capture = str(
            pick_nested(meeting, "capture", "meeting_capture", cfg.meeting_capture)
            or cfg.meeting_capture
        ).strip().lower()
        if meeting_capture not in {"mic", "loopback", "mix"}:
            meeting_capture = "mix"

        vad_backend = str(
            pick_nested(vad, "backend", "vad_backend", cfg.vad_backend) or cfg.vad_backend
        ).strip().lower()
        if vad_backend not in {"silero", "energy"}:
            vad_backend = "silero"

        # Prefer audio.listen_source; also accept wake.input / wake.listen_source.
        listen_raw = pick_nested(
            audio, "listen_source", "audio_listen_source", None
        )
        if listen_raw in (None, ""):
            listen_raw = pick_nested(wake, "input", "wake_input", None)
        if listen_raw in (None, ""):
            listen_raw = pick_nested(wake, "listen_source", "wake_listen_source", None)
        if listen_raw in (None, ""):
            listen_raw = cfg.audio_listen_source
        audio_listen_source = str(listen_raw or "mic").strip().lower()
        if audio_listen_source not in {"mic", "loopback"}:
            audio_listen_source = "mic"
        # Env wins when set (override YAML).
        env_listen = _env_first(
            "KRABOBOT_VOICE_LISTEN_SOURCE",
            "VOICE_LISTEN_SOURCE",
        )
        if env_listen:
            env_listen = env_listen.strip().lower()
            if env_listen in {"mic", "loopback"}:
                audio_listen_source = env_listen

        wake_phrases = _resolve_wake_phrases(wake, data, cfg.wake_phrases)
        env_wake = _env_first(
            "KRABOBOT_VOICE_WAKE_PHRASE",
            "VOICE_WAKE_PHRASE",
        )
        if env_wake:
            # Env phrase is primary; keep YAML aliases after it (deduped).
            wake_phrases = _resolve_wake_phrases(
                {"phrase": env_wake, "phrases": wake_phrases},
                {},
                wake_phrases,
            )
        wake_greetings = _resolve_wake_greetings(
            wake, data, wake_phrases, cfg.wake_greetings
        )

        return cls(
            base_url=str(pick("base_url", "url", default=cfg.base_url) or cfg.base_url).rstrip(
                "/"
            ),
            device_id=str(pick("device_id", default=cfg.device_id) or cfg.device_id).strip(),
            token=token,
            timeout_s=float(pick("timeout_s", default=cfg.timeout_s) or cfg.timeout_s),
            sample_rate=int(pick("sample_rate", default=cfg.sample_rate) or cfg.sample_rate),
            wake_mode=wake_mode,
            wake_silence_end_s=float(
                pick_nested(
                    wake,
                    "silence_end_s",
                    "wake_silence_end_s",
                    cfg.wake_silence_end_s,
                )
            ),
            wake_max_s=float(
                pick_nested(
                    wake,
                    "max_s",
                    "wake_max_s",
                    # Legacy alias: old sliding-window length → max utterance.
                    pick_nested(wake, "window_s", "wake_window_s", cfg.wake_max_s),
                )
            ),
            wake_min_speech_s=float(
                pick_nested(
                    wake,
                    "min_speech_s",
                    "wake_min_speech_s",
                    cfg.wake_min_speech_s,
                )
            ),
            wake_energy_threshold=float(
                pick_nested(
                    wake,
                    "energy_threshold",
                    "wake_energy_threshold",
                    cfg.wake_energy_threshold,
                )
            ),
            wake_phrases=wake_phrases,
            wake_greetings=wake_greetings,
            kws_threshold=float(
                pick_nested(wake, "threshold", "kws_threshold", cfg.kws_threshold)
            ),
            kws_energy_threshold=float(
                pick_nested(
                    wake,
                    "kws_energy_threshold",
                    "energy_threshold",
                    cfg.kws_energy_threshold,
                )
            ),
            kws_refs_dir=str(
                pick_nested(wake, "refs_dir", "kws_refs_dir", cfg.kws_refs_dir) or ""
            ).strip(),
            kws_auto_enroll_tts=_as_bool(
                pick_nested(
                    wake,
                    "auto_enroll_tts",
                    "kws_auto_enroll_tts",
                    cfg.kws_auto_enroll_tts,
                ),
                cfg.kws_auto_enroll_tts,
            ),
            stt_model_dir=str(pick("stt_model_dir", default=cfg.stt_model_dir) or "").strip(),
            stt_num_threads=int(pick("stt_num_threads", default=cfg.stt_num_threads) or 2),
            ptt_enabled=_as_bool(
                pick_nested(ptt, "enabled", "ptt_enabled", cfg.ptt_enabled),
                cfg.ptt_enabled,
            ),
            ptt_hotkey=str(
                pick_nested(ptt, "hotkey", "ptt_hotkey", cfg.ptt_hotkey) or cfg.ptt_hotkey
            ).strip(),
            utterance_max_s=float(pick("utterance_max_s", default=cfg.utterance_max_s)),
            silence_end_s=float(pick("silence_end_s", default=cfg.silence_end_s)),
            speech_start_s=float(pick("speech_start_s", default=cfg.speech_start_s)),
            min_speech_s=float(pick("min_speech_s", default=cfg.min_speech_s)),
            settle_s=float(pick("settle_s", default=cfg.settle_s)),
            preroll_s=float(pick("preroll_s", default=cfg.preroll_s)),
            no_speech_timeout_s=float(
                pick("no_speech_timeout_s", default=cfg.no_speech_timeout_s)
            ),
            energy_threshold=float(pick("energy_threshold", default=cfg.energy_threshold)),
            vad_backend=vad_backend,
            vad_threshold=float(
                pick_nested(vad, "threshold", "vad_threshold", cfg.vad_threshold)
            ),
            vad_energy_pregate=float(
                pick_nested(
                    vad,
                    "energy_pregate",
                    "vad_energy_pregate",
                    cfg.vad_energy_pregate,
                )
            ),
            vad_model_path=str(
                pick_nested(vad, "model_path", "vad_model_path", cfg.vad_model_path) or ""
            ).strip(),
            talk_listen_timeout_s=_resolve_listen_timeout(
                talk, data, cfg.talk_listen_timeout_s
            ),
            talk_follow_up_s=float(
                pick_nested(talk, "follow_up_s", "talk_follow_up_s", cfg.talk_follow_up_s)
            ),
            talk_follow_up_beep=_as_bool(
                pick_nested(
                    talk,
                    "follow_up_beep",
                    "talk_follow_up_beep",
                    cfg.talk_follow_up_beep,
                ),
                cfg.talk_follow_up_beep,
            ),
            talk_beeps=_as_bool(
                pick_nested(talk, "beeps", "talk_beeps", cfg.talk_beeps),
                cfg.talk_beeps,
            ),
            cmd_meeting_start=_as_str_list(
                pick_nested(
                    talk,
                    "meeting_start",
                    "cmd_meeting_start",
                    meeting.get("start_phrases", cfg.cmd_meeting_start),
                ),
                cfg.cmd_meeting_start,
            ),
            cmd_meeting_stop=_as_str_list(
                pick_nested(
                    talk,
                    "meeting_stop",
                    "cmd_meeting_stop",
                    meeting.get("stop_phrases", cfg.cmd_meeting_stop),
                ),
                cfg.cmd_meeting_stop,
            ),
            cmd_run_test=_as_str_list(
                pick_nested(talk, "run_test", "cmd_run_test", cfg.cmd_run_test),
                cfg.cmd_run_test,
            ),
            cmd_exit=_as_str_list(
                pick_nested(talk, "exit", "cmd_exit", cfg.cmd_exit),
                cfg.cmd_exit,
            ),
            audio_input_device=str(
                pick_nested(audio, "input_device", "audio_input_device", cfg.audio_input_device)
                or ""
            ).strip(),
            audio_output_device=str(
                pick_nested(
                    audio, "output_device", "audio_output_device", cfg.audio_output_device
                )
                or ""
            ).strip(),
            audio_listen_source=audio_listen_source,
            meeting_enabled=_as_bool(
                pick_nested(meeting, "enabled", "meeting_enabled", cfg.meeting_enabled),
                cfg.meeting_enabled,
            ),
            meeting_capture=meeting_capture,
            meeting_hotkey=str(
                pick_nested(meeting, "hotkey", "meeting_hotkey", cfg.meeting_hotkey)
                or cfg.meeting_hotkey
            ).strip(),
            meeting_loopback_device=str(
                pick_nested(
                    meeting,
                    "loopback_device",
                    "meeting_loopback_device",
                    cfg.meeting_loopback_device,
                )
                or ""
            ).strip(),
            meeting_instruct=str(
                pick_nested(meeting, "instruct", "meeting_instruct", cfg.meeting_instruct)
                or cfg.meeting_instruct
            ),
            meeting_max_s=float(
                pick_nested(meeting, "max_s", "meeting_max_s", cfg.meeting_max_s)
                or cfg.meeting_max_s
            ),
        )
