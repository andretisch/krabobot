"""App-directory config path resolution (frozen and python -m)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.config import (  # noqa: E402
    VoiceClientConfig,
    app_base_dir,
    app_config_candidates,
    default_config_candidates,
    ensure_app_config,
    env_config_path,
    local_appdata_config_path,
    package_root_dir,
)


def _clear_device_env(monkeypatch) -> None:
    for key in (
        "KRABOBOT_DEVICE_ID",
        "KRABOBOT_VOICE_DEVICE_ID",
        "KRABOBOT_TOKEN",
        "KRABOBOT_VOICE_TOKEN",
        "KRABOBOT_ADMIN_TOKEN",
        "KRABOBOT_VOICE_CONFIG",
        "KRABOBOT_VOICE_CONFIG_DIR",
    ):
        monkeypatch.delenv(key, raising=False)


def test_package_root_is_parent_of_package() -> None:
    assert package_root_dir() == ROOT.resolve()
    assert (package_root_dir() / "krabobot_voice").is_dir()


def test_default_candidates_dev_uses_package_root_not_localappdata(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    _clear_device_env(monkeypatch)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    cands = default_config_candidates()
    assert cands[0] == ROOT.resolve() / "config.yaml"
    assert cands[1] == ROOT.resolve() / "config.yml"
    assert local_appdata_config_path() not in cands
    assert app_base_dir() == ROOT.resolve()


def test_app_base_dir_frozen_uses_exe_parent(
    monkeypatch, tmp_path: Path
) -> None:
    fake_exe = tmp_path / "krabobot-voice.exe"
    fake_exe.write_bytes(b"")
    _clear_device_env(monkeypatch)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe))

    assert app_base_dir() == tmp_path.resolve()
    assert app_config_candidates() == [
        tmp_path.resolve() / "config.yaml",
        tmp_path.resolve() / "config.yml",
    ]


def test_config_dir_env_overrides_app_dir(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "custom"
    override.mkdir()
    fake_exe = tmp_path / "krabobot-voice.exe"
    fake_exe.write_bytes(b"")
    monkeypatch.setenv("KRABOBOT_VOICE_CONFIG_DIR", str(override))
    monkeypatch.delenv("KRABOBOT_VOICE_CONFIG", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe))

    assert app_base_dir() == override.resolve()
    cands = default_config_candidates()
    assert cands[0] == override.resolve() / "config.yaml"


def test_config_file_env_overrides_app_dir(monkeypatch, tmp_path: Path) -> None:
    cfg = tmp_path / "elsewhere.yaml"
    cfg.write_text("device_id: from-env\n", encoding="utf-8")
    monkeypatch.setenv("KRABOBOT_VOICE_CONFIG", str(cfg))
    monkeypatch.delenv("KRABOBOT_VOICE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(sys, "frozen", False, raising=False)

    assert env_config_path() == cfg.resolve()
    cands = default_config_candidates()
    assert cands == [cfg.resolve()]


def test_ensure_app_config_copies_example(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "krabobot-voice.exe"))
    _clear_device_env(monkeypatch)
    example = tmp_path / "config.example.yaml"
    example.write_text(
        "base_url: http://127.0.0.1:8900\ndevice_id: portable-test\n",
        encoding="utf-8",
    )

    created = ensure_app_config(tmp_path)
    assert created == tmp_path / "config.yaml"
    assert created is not None and created.is_file()
    assert "portable-test" in created.read_text(encoding="utf-8")

    # Second call must not overwrite an existing config.
    created.write_text("device_id: kept\n", encoding="utf-8")
    again = ensure_app_config(tmp_path)
    assert again == created
    assert created.read_text(encoding="utf-8") == "device_id: kept\n"


def test_load_frozen_prefers_exe_dir_not_localappdata(
    monkeypatch, tmp_path: Path
) -> None:
    """Frozen load reads exe-dir config; LocalAppData is not in search order."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "krabobot-voice.exe"))
    _clear_device_env(monkeypatch)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData"))

    (tmp_path / "config.yaml").write_text(
        "device_id: exe-only\nbase_url: http://127.0.0.1:8900\n",
        encoding="utf-8",
    )
    appdata = tmp_path / "AppData" / "krabobot-voice"
    appdata.mkdir(parents=True)
    (appdata / "config.yaml").write_text(
        "device_id: should-not-load\n", encoding="utf-8"
    )

    cands = default_config_candidates()
    assert cands[0] == tmp_path.resolve() / "config.yaml"
    assert local_appdata_config_path() not in cands

    cfg = VoiceClientConfig.load()
    assert cfg.device_id == "exe-only"


def test_load_dev_uses_package_root_not_localappdata(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    _clear_device_env(monkeypatch)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData"))
    monkeypatch.setenv("KRABOBOT_VOICE_CONFIG_DIR", str(tmp_path))

    (tmp_path / "config.yaml").write_text(
        "device_id: pkg-root\nbase_url: http://127.0.0.1:8900\n",
        encoding="utf-8",
    )
    appdata = tmp_path / "AppData" / "krabobot-voice"
    appdata.mkdir(parents=True)
    (appdata / "config.yaml").write_text(
        "device_id: should-not-load\n", encoding="utf-8"
    )

    cfg = VoiceClientConfig.load()
    assert cfg.device_id == "pkg-root"


def test_load_prefers_yml_when_yaml_missing(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "krabobot-voice.exe"))
    _clear_device_env(monkeypatch)

    (tmp_path / "config.yml").write_text(
        "device_id: from-yml\nbase_url: http://127.0.0.1:8900\n",
        encoding="utf-8",
    )
    cfg = VoiceClientConfig.load()
    assert cfg.device_id == "from-yml"


def test_load_auto_copies_example_on_first_run(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "krabobot-voice.exe"))
    _clear_device_env(monkeypatch)

    (tmp_path / "config.example.yaml").write_text(
        "device_id: from-example\nbase_url: http://127.0.0.1:8900\n",
        encoding="utf-8",
    )
    assert not (tmp_path / "config.yaml").exists()

    cfg = VoiceClientConfig.load()
    assert (tmp_path / "config.yaml").is_file()
    assert cfg.device_id == "from-example"


def test_load_argv_path_wins_over_env_and_app_dir(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    _clear_device_env(monkeypatch)
    monkeypatch.setenv("KRABOBOT_VOICE_CONFIG_DIR", str(tmp_path / "app"))
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "config.yaml").write_text(
        "device_id: app-dir\n", encoding="utf-8"
    )
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(
        "device_id: argv\nbase_url: http://127.0.0.1:8900\n",
        encoding="utf-8",
    )
    env_file = tmp_path / "env.yaml"
    env_file.write_text("device_id: env\n", encoding="utf-8")
    monkeypatch.setenv("KRABOBOT_VOICE_CONFIG", str(env_file))

    cfg = VoiceClientConfig.load(explicit)
    assert cfg.device_id == "argv"
