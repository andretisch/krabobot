"""Web settings: redacted GET payload, editable PUT with backups / restore."""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from krabobot.config.loader import get_config_path, load_config, save_config
from krabobot.config.schema import Config
from krabobot.providers.registry import PROVIDERS

# Known channel-wide flags (camelCase aliases from model_dump by_alias=True)
_CHANNEL_COMMON_KEYS: frozenset[str] = frozenset({"sendProgress", "sendToolHints", "sendMaxRetries"})
_OTHER_SECTION_KEYS_UI: frozenset[str] = frozenset({"api", "gateway", "tools", "tts", "stt"})
SECRET_DISPLAY_MASK = "••••••••"
_BACKUP_LEGACY = re.compile(r"^config\.backup\.[0-9]{8}-[0-9]{6}\.json$")
# Microseconds avoid colliding with a restore source backup created same wall second.
_BACKUP_WITH_MICROS = re.compile(r"^config\.backup\.[0-9]{8}-[0-9]{6}-[0-9]{6}\.json$")


def _should_redact_key(key: str) -> bool:
    """Detect config field names likely holding credentials or opaque secrets."""
    n = "".join(c for c in key if c.isalnum()).lower()
    if len(n) < 2:
        return False
    if n in {"apikey", "password", "passwd", "secret"}:
        return True
    if n.endswith("apikey"):
        return True
    if n.endswith("password") or n.endswith("passwd"):
        return True
    if n.endswith("secret"):
        return True
    if n.endswith("tokens"):
        return False
    if n.endswith("token"):
        return True
    return False


def redact_secrets(obj: Any) -> None:
    """Recursively mask sensitive string values in-place (JSON tree from model_dump)."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, (dict, list)):
                redact_secrets(v)
            elif isinstance(v, str) and v and _should_redact_key(k):
                obj[k] = SECRET_DISPLAY_MASK
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                redact_secrets(item)


def _is_masked_ui_value(val: Any) -> bool:
    """Client kept secret unchanged (masked) or omitted meaning keep."""
    if not isinstance(val, str):
        return False
    s = val.strip()
    if not s:
        return True
    if s == SECRET_DISPLAY_MASK:
        return True
    return bool(s and all(ch == "•" or ch == "·" for ch in s))


def _deep_merge_preserving_masked_secrets(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge incoming UI tree onto existing subtree; masked/empty secrets keep existing."""
    out: dict[str, Any] = dict(existing)
    for k, iv in incoming.items():
        ev = existing.get(k)
        if isinstance(iv, dict):
            base = ev if isinstance(ev, dict) else {}
            out[k] = _deep_merge_preserving_masked_secrets(base, iv)
        elif isinstance(iv, list):
            out[k] = iv
        elif isinstance(iv, str) and _should_redact_key(k) and _is_masked_ui_value(iv):
            continue  # unchanged secret in UI → keep stored value
        else:
            out[k] = iv
    return out


def _sections_to_partial_top_level(sections: dict[str, Any]) -> dict[str, Any]:
    """Same shape as GET /v1/web/config → flat top-level keys for Config."""
    patch: dict[str, Any] = {}
    core = sections.get("core") or {}
    if isinstance(core.get("agents"), dict):
        patch["agents"] = core["agents"]
    if isinstance(core.get("providers"), dict):
        patch["providers"] = core["providers"]

    merged_ch: dict[str, Any] = {}
    ch = sections.get("channels") or {}
    if isinstance(ch.get("common"), dict):
        merged_ch.update(ch["common"])
    if isinstance(ch.get("named"), dict):
        merged_ch.update(ch["named"])
    if merged_ch:
        patch["channels"] = merged_ch

    other = sections.get("other") or {}
    if isinstance(other, dict):
        for ok, ov in other.items():
            if ok in _OTHER_SECTION_KEYS_UI:
                patch[ok] = ov

    return patch


def sections_payload_merge_into_config(existing: Config, sections: dict[str, Any]) -> Config:
    """Apply UI sections blob onto current validated config."""
    partial = _sections_to_partial_top_level(sections)
    base: dict[str, Any] = existing.model_dump(mode="json", by_alias=True)

    merged: dict[str, Any] = dict(base)
    for k, pv in partial.items():
        cur = merged.get(k)
        if isinstance(pv, dict) and isinstance(cur, dict):
            merged[k] = _deep_merge_preserving_masked_secrets(cur, pv)
        elif isinstance(pv, dict):
            merged[k] = _deep_merge_preserving_masked_secrets({}, pv)
        else:
            merged[k] = pv

    return Config.model_validate(merged)


def build_web_config_payload() -> dict[str, Any]:
    """
    Load config from get_config_path(), return JSON-safe structure for the UI.

    Raises:
        FileNotFoundError: if the config file is missing.
    """
    path = get_config_path()
    if not path.is_file():
        raise FileNotFoundError(str(path.resolve()))

    cfg = load_config(path)
    data: dict[str, Any] = cfg.model_dump(mode="json", by_alias=True)
    redact_secrets(data)

    raw_channels: dict[str, Any] = dict(data.get("channels") or {})
    ch_common = {k: raw_channels[k] for k in _CHANNEL_COMMON_KEYS if k in raw_channels}
    ch_named = {k: v for k, v in sorted(raw_channels.items()) if k not in _CHANNEL_COMMON_KEYS}

    other: dict[str, Any] = {}
    for key in ("api", "gateway", "tools", "tts", "stt"):
        if key in data:
            other[key] = data[key]

    provider_choices: list[dict[str, str]] = [
        {"value": "auto", "label": "auto (определить по модели)"}
    ]
    for spec in PROVIDERS:
        provider_choices.append({"value": spec.name, "label": spec.label})

    core = {
        "agents": data.get("agents") or {},
        "providers": data.get("providers") or {},
    }

    return {
        "path": str(path.resolve()),
        "providerChoices": provider_choices,
        "core": core,
        "channels": {"common": ch_common, "named": ch_named},
        "other": other,
    }


def _is_backup_filename(name: str) -> bool:
    return bool(_BACKUP_LEGACY.match(name) or _BACKUP_WITH_MICROS.match(name))


def backup_config_now(path: Path | None = None) -> Path:
    """Copy config.json to config.backup.<timestamp>-<microseconds>.json."""
    cfg = path or get_config_path()
    if not cfg.is_file():
        raise FileNotFoundError(str(cfg.resolve()))
    parent = cfg.parent
    suffix = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    dup = parent / f"config.backup.{suffix}.json"
    shutil.copy2(cfg, dup)
    return dup


def save_web_config_sections(sections: dict[str, Any]) -> tuple[Path, Path]:
    """
    Validate merged config, backup current file, write new JSON.

    Returns:
        (config_path, backup_path)
    """
    path = get_config_path()
    if not path.is_file():
        raise FileNotFoundError(str(path.resolve()))

    cfg = load_config(path)
    merged = sections_payload_merge_into_config(cfg, sections)
    bak = backup_config_now(path)
    save_config(merged, path)
    return path.resolve(), bak.resolve()


def list_config_backups(limit: int = 40) -> list[dict[str, Any]]:
    """Newest backups first."""
    path = get_config_path()
    parent = path.parent
    out: list[tuple[float, Path]] = []
    for p in parent.iterdir():
        if p.is_file() and _is_backup_filename(p.name):
            out.append((p.stat().st_mtime, p))
    out.sort(reverse=True, key=lambda t: t[0])
    return [{"name": p.name, "path": str(p.resolve())} for _, p in out[:limit]]


def restore_backup(basename: str) -> tuple[Path, Path]:
    """
    Restore config.json from backup (basename must match config.backup.*.json).

    Returns:
        (config_path, backup_used)
    """
    if not basename or "/" in basename or "\\" in basename or ".." in basename:
        raise ValueError("invalid backup name")
    if not _is_backup_filename(basename):
        raise ValueError("invalid backup filename")
    dest = get_config_path()
    parent = dest.parent
    src = (parent / basename).resolve()
    if not src.is_file():
        raise FileNotFoundError(str(src))
    if parent != src.parent or src.name != basename:
        raise ValueError("backup path escapes config directory")

    bak = backup_config_now(dest)
    shutil.copy2(src, dest)
    return dest.resolve(), bak.resolve()
