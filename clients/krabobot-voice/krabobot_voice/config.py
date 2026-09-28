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


@dataclass
class VoiceClientConfig:
    """HTTP + local wake/listen / PTT settings."""

    base_url: str = "http://127.0.0.1:8900"
    device_id: str = field(default_factory=_default_device_id)
    token: str = ""
    timeout_s: float = 120.0
    sample_rate: int = 16000
    # Wake: kws (default) | asr (legacy sherpa) | off (PTT only)
    wake_mode: str = "kws"
    wake_window_s: float = 1.6
    wake_hop_s: float = 0.3
    kws_threshold: float = 0.82
    kws_energy_threshold: float = 0.012
    kws_refs_dir: str = ""
    kws_auto_enroll_tts: bool = True
    # Legacy sherpa wake (only if wake_mode=asr)
    stt_model_dir: str = ""
    stt_num_threads: int = 2
    # PTT
    ptt_enabled: bool = True
    ptt_hotkey: str = "ctrl+alt+space"
    # Utterance capture (post-wake VAD)
    utterance_max_s: float = 15.0
    silence_end_s: float = 1.2
    speech_start_s: float = 0.25
    min_speech_s: float = 1.2
    settle_s: float = 0.35
    preroll_s: float = 0.4
    no_speech_timeout_s: float = 5.0
    energy_threshold: float = 0.008

    @classmethod
    def from_env(cls) -> VoiceClientConfig:
        device = _env_first("KRABOBOT_DEVICE_ID", "KRABOBOT_VOICE_DEVICE_ID")
        return cls(
            base_url=_env_first(
                "KRABOBOT_URL",
                "KRABOBOT_VOICE_URL",
                default="http://127.0.0.1:8900",
            ).rstrip("/"),
            device_id=device or _default_device_id(),
            token=_env_first(
                "KRABOBOT_TOKEN",
                "KRABOBOT_VOICE_TOKEN",
                "KRABOBOT_ADMIN_TOKEN",
            ),
            timeout_s=float(_env_first("KRABOBOT_VOICE_TIMEOUT", default="120") or 120),
            wake_mode=_env_first("KRABOBOT_VOICE_WAKE_MODE", default="kws") or "kws",
            ptt_hotkey=_env_first(
                "KRABOBOT_VOICE_PTT_HOTKEY",
                default="ctrl+alt+space",
            )
            or "ctrl+alt+space",
        )

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
            wake_mode = "kws"

        return cls(
            base_url=str(pick("base_url", "url", default=cfg.base_url) or cfg.base_url).rstrip(
                "/"
            ),
            device_id=str(pick("device_id", default=cfg.device_id) or cfg.device_id).strip(),
            token=token,
            timeout_s=float(pick("timeout_s", default=cfg.timeout_s) or cfg.timeout_s),
            sample_rate=int(pick("sample_rate", default=cfg.sample_rate) or cfg.sample_rate),
            wake_mode=wake_mode,
            wake_window_s=float(
                pick_nested(wake, "window_s", "wake_window_s", cfg.wake_window_s)
            ),
            wake_hop_s=float(pick_nested(wake, "hop_s", "wake_hop_s", cfg.wake_hop_s)),
            kws_threshold=float(
                pick_nested(wake, "threshold", "kws_threshold", cfg.kws_threshold)
            ),
            kws_energy_threshold=float(
                pick_nested(
                    wake,
                    "energy_threshold",
                    "kws_energy_threshold",
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
        )
