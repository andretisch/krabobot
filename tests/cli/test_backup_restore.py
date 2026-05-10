"""Tests for krabobot backup / restore archives."""

import shutil
from pathlib import Path

from krabobot.cli.backup import create_archive, restore_archive
from krabobot.config.loader import load_config, save_config
from krabobot.config.schema import Config


def test_backup_restore_roundtrip_workspace(tmp_path: Path) -> None:
    krabot = tmp_path / ".krabobot"
    krabot.mkdir(parents=True)
    cfg_path = krabot / "config.json"

    ws = tmp_path / "my-workspace"
    ws.mkdir(parents=True)
    (ws / "state.txt").write_text("saved", encoding="utf-8")

    cfg = Config()
    cfg.agents.defaults.workspace = str(ws)
    save_config(cfg, cfg_path)

    arch = tmp_path / "bk.tar.gz"
    create_archive(arch, config_path=cfg_path, include_workspace=True)

    cfg_path.unlink()
    shutil.rmtree(ws)

    actions = restore_archive(
        arch,
        config_path=cfg_path,
        apply=True,
        replace_workspace=True,
    )
    assert any("workspace" in a.lower() for a in actions)

    restored = load_config(cfg_path)
    rp = Path(restored.agents.defaults.workspace).expanduser().resolve()
    assert (rp / "state.txt").read_text(encoding="utf-8") == "saved"


def test_restore_full_wipes_workspace_extras_before_unpack(tmp_path: Path) -> None:
    krabot = tmp_path / ".krabobot"
    krabot.mkdir(parents=True)
    cfg_path = krabot / "config.json"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    (ws / "good.txt").write_text("ok", encoding="utf-8")

    cfg = Config()
    cfg.agents.defaults.workspace = str(ws)
    save_config(cfg, cfg_path)
    arch = tmp_path / "bk.tar.gz"
    create_archive(arch, config_path=cfg_path, include_workspace=True)

    (ws / "extra.txt").write_text("junk", encoding="utf-8")

    restore_archive(
        arch,
        config_path=cfg_path,
        apply=True,
        full=True,
    )

    assert not (ws / "extra.txt").exists()
    assert (ws / "good.txt").read_text(encoding="utf-8") == "ok"


def test_restore_merge_action_label(tmp_path: Path) -> None:
    krabot = tmp_path / ".krabobot"
    krabot.mkdir(parents=True)
    cfg_path = krabot / "config.json"
    ws = tmp_path / "ws2"
    ws.mkdir(parents=True)
    (ws / "a").write_text("1", encoding="utf-8")
    cfg = Config()
    cfg.agents.defaults.workspace = str(ws)
    save_config(cfg, cfg_path)
    arch = tmp_path / "bk2.tar.gz"
    create_archive(arch, config_path=cfg_path, include_workspace=True)
    lines = restore_archive(arch, config_path=cfg_path, dry_run=True, apply=False, full=False)
    assert any("(merge)" in line for line in lines)


def test_restore_rejects_non_backup(tmp_path: Path) -> None:
    bad = tmp_path / "empty.tar.gz"
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text("{}", encoding="utf-8")

    import tarfile

    with tarfile.open(bad, "w:gz") as tf:
        ti = tarfile.TarInfo("junk.txt")
        ti.size = 3
        import io

        tf.addfile(ti, io.BytesIO(b"x" * 3))

    try:
        restore_archive(bad, config_path=cfg_path, apply=False, dry_run=False)
    except ValueError as e:
        assert "manifest" in str(e).lower() or "payload" in str(e).lower()
        return
    raise AssertionError("expected ValueError")
