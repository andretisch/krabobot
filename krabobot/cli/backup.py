"""Backup and restore krabobot config and data directories."""

from __future__ import annotations

import io
from typing import Any
import json
import shutil
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from krabobot import __version__

MANIFEST_FORMAT = 1
MANIFEST_NAME = "manifest.json"
PAYLOAD = "payload"


def _data_dir(config_path: Path) -> Path:
    return config_path.expanduser().resolve().parent


def _build_manifest(entries: list[str]) -> dict:
    return {
        "format": MANIFEST_FORMAT,
        "createdUtc": datetime.now(UTC).isoformat(),
        "krabobotVersion": __version__,
        "entries": entries,
    }


def _safe_member_name(name: str) -> bool:
    if not name or name.startswith(("/", "\\")):
        return False
    if ".." in Path(name).parts:
        return False
    return name == MANIFEST_NAME or name.startswith(f"{PAYLOAD}/")


def create_archive(
    archive_path: Path,
    *,
    config_path: Path,
    include_workspace: bool = True,
    include_media: bool = False,
    include_models: bool = False,
    include_history: bool = False,
) -> tuple[Path, dict]:
    """
    Pack config (+ optional dirs) into a gzip tar archive.

    Returns (archive_path, manifest dict).
    """
    from krabobot.config.loader import load_config

    cfg_path = config_path.expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"No config file: {cfg_path}")

    archive_path = archive_path.expanduser().resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cfg_path)
    workspace = cfg.workspace_path.expanduser().resolve()
    runtime = _data_dir(cfg_path)

    entries: list[str] = []

    with tarfile.open(archive_path, "w:gz", format=tarfile.PAX_FORMAT) as tf:
        tf.add(str(cfg_path), arcname=f"{PAYLOAD}/config.json")
        entries.append("config")

        if include_workspace and workspace.is_dir():
            tf.add(str(workspace), arcname=f"{PAYLOAD}/workspace", recursive=True)
            entries.append("workspace")

        if include_media:
            media = runtime / "media"
            if media.is_dir():
                tf.add(str(media), arcname=f"{PAYLOAD}/media", recursive=True)
                entries.append("media")

        if include_models:
            models = runtime / "models"
            if models.is_dir():
                tf.add(str(models), arcname=f"{PAYLOAD}/models", recursive=True)
                entries.append("models")

        if include_history:
            hist = runtime / "history"
            if hist.is_dir():
                tf.add(str(hist), arcname=f"{PAYLOAD}/history", recursive=True)
                entries.append("history")

        manifest = _build_manifest(entries)
        m_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
        ti = tarfile.TarInfo(MANIFEST_NAME)
        ti.size = len(m_bytes)
        ti.mtime = int(datetime.now(UTC).timestamp())
        tf.addfile(ti, io.BytesIO(m_bytes))

    return archive_path, manifest


def _extract_manifest(tf: tarfile.TarFile) -> dict:
    try:
        m = tf.getmember(MANIFEST_NAME)
    except KeyError as e:
        raise ValueError(f'Backup archive missing «{MANIFEST_NAME}»: not a krabobot backup') from e
    fobj = tf.extractfile(m)
    if not fobj:
        raise ValueError("Cannot read manifest from backup")
    return json.loads(fobj.read().decode("utf-8"))


def _extract_member(tf: tarfile.TarFile, member: tarfile.TarInfo, dest: Path) -> None:
    kwargs: dict[str, Any] = {"set_attrs": False}
    if sys.version_info >= (3, 12):
        kwargs["filter"] = "data"
    tf.extract(member, path=dest, **kwargs)


def _extract_known_members(tf: tarfile.TarFile, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for m in tf.getmembers():
        if not _safe_member_name(m.name):
            continue
        _extract_member(tf, m, dest)


@contextmanager
def _temp_extract_dir():
    with tempfile.TemporaryDirectory(prefix="krabobot-restore-") as tmp:
        yield Path(tmp)


def restore_archive(
    archive_path: Path,
    *,
    config_path: Path,
    dry_run: bool = False,
    apply: bool = False,
    config_only: bool = False,
    replace_workspace: bool = False,
    full: bool = False,
) -> list[str]:
    """
    Restore from a backup archive.

    ``apply`` must be True to write files (unless ``dry_run``).

    ``full``: before unpacking each archived tree (workspace, media, models, history),
    remove the destination directory entirely if it exists — mirrors ``backup --full``.
    Implies destructive replace for all components present in the archive.
    """
    archive_path = archive_path.expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"No archive: {archive_path}")

    cfg_target = config_path.expanduser().resolve()
    actions: list[str] = []

    with tarfile.open(archive_path, "r:gz") as tf:
        manifest = _extract_manifest(tf)
        if manifest.get("format") != MANIFEST_FORMAT:
            raise ValueError(f"Unsupported backup format: {manifest.get('format')}")
        entries: list[str] = list(manifest.get("entries") or [])

        try:
            tf.getmember(f"{PAYLOAD}/config.json")
        except KeyError as e:
            raise ValueError("Backup has no payload/config.json") from e

        with _temp_extract_dir() as tmp_path:
            _extract_known_members(tf, tmp_path)

            payload = tmp_path / PAYLOAD
            src_cfg = payload / "config.json"
            if not src_cfg.is_file():
                raise ValueError("Extracted payload/config.json missing")

            actions.append(f"Config → {cfg_target}")
            for e in entries:
                actions.append(f"contains: {e}")

            if not dry_run and apply:
                cfg_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_cfg, cfg_target)

            from krabobot.config.loader import load_config
            from krabobot.config.schema import Config

            if apply and not dry_run:
                cfg = load_config(cfg_target)
            else:
                cfg = Config.model_validate(json.loads(src_cfg.read_text(encoding="utf-8")))

            if config_only:
                return actions

            ws_target = cfg.workspace_path.expanduser().resolve()
            runtime = _data_dir(cfg_target)

            def _would_replace(name: str) -> bool:
                if full and name in ("workspace", "media", "models", "history"):
                    return True
                return bool(replace_workspace and name == "workspace")

            def _restore_tree(name: str, dest: Path) -> None:
                src = payload / name
                if not src.is_dir():
                    return
                if _would_replace(name) and dest.exists():
                    shutil.rmtree(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dest, dirs_exist_ok=True)

            plan: list[tuple[str, Path]] = []
            if "workspace" in entries:
                plan.append(("workspace", ws_target))
            if "media" in entries:
                plan.append(("media", runtime / "media"))
            if "models" in entries:
                plan.append(("models", runtime / "models"))
            if "history" in entries:
                plan.append(("history", runtime / "history"))

            for name, dest in plan:
                how = "(replace)" if _would_replace(name) else "(merge)"
                actions.append(f"Restore {name} → {dest} {how}")

            if not dry_run and apply:
                for name, dest in plan:
                    _restore_tree(name, dest)

    return actions
