"""resolve_stt_model_dir: app models/ first, then legacy ~/.krabobot."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.local_asr import (  # noqa: E402
    PREFERRED_STT_FOLDER,
    resolve_stt_model_dir,
)


def _make_stt(dir_path: Path) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "tokens.txt").write_text("a\n", encoding="utf-8")
    return dir_path


def test_resolve_stt_explicit(tmp_path: Path) -> None:
    model = _make_stt(tmp_path / "custom-stt")
    assert resolve_stt_model_dir(str(model)) == model.resolve()


def test_resolve_stt_prefers_app_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from krabobot_voice import config as cfg
    from krabobot_voice import local_asr as la

    app = tmp_path / "app"
    app_model = _make_stt(app / "models" / "stt" / PREFERRED_STT_FOLDER)
    legacy = _make_stt(tmp_path / "legacy" / PREFERRED_STT_FOLDER)
    monkeypatch.setattr(cfg, "app_base_dir", lambda: app)
    monkeypatch.setattr(cfg, "is_frozen", lambda: False)
    monkeypatch.setattr(la, "legacy_stt_base", lambda: legacy.parent)
    assert resolve_stt_model_dir() == app_model.resolve()


def test_resolve_stt_frozen_requires_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from krabobot_voice import config as cfg
    from krabobot_voice import local_asr as la

    app = tmp_path / "app"
    app.mkdir()
    legacy = _make_stt(tmp_path / "legacy" / PREFERRED_STT_FOLDER)
    monkeypatch.setattr(cfg, "app_base_dir", lambda: app)
    monkeypatch.setattr(cfg, "is_frozen", lambda: True)
    monkeypatch.setattr(la, "legacy_stt_base", lambda: legacy.parent)
    with pytest.raises(FileNotFoundError, match="Положите модель"):
        resolve_stt_model_dir()


def test_resolve_stt_falls_back_to_legacy_when_not_frozen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from krabobot_voice import config as cfg
    from krabobot_voice import local_asr as la

    app = tmp_path / "app"
    app.mkdir()
    legacy = _make_stt(tmp_path / "legacy" / PREFERRED_STT_FOLDER)
    monkeypatch.setattr(cfg, "app_base_dir", lambda: app)
    monkeypatch.setattr(cfg, "is_frozen", lambda: False)
    monkeypatch.setattr(la, "legacy_stt_base", lambda: legacy.parent)
    assert resolve_stt_model_dir() == legacy.resolve()
