"""Meetings directory defaults (portable app-dir, not LocalAppData)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.config import app_base_dir  # noqa: E402
from krabobot_voice.meeting import (  # noqa: E402
    default_meetings_dir,
    legacy_meetings_dir,
    new_meeting_wav_path,
)


def test_default_meetings_dir_is_app_meetings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setenv("KRABOBOT_VOICE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData"))

    assert default_meetings_dir() == tmp_path.resolve() / "meetings"
    assert default_meetings_dir() == app_base_dir() / "meetings"
    assert legacy_meetings_dir() == tmp_path / "AppData" / "krabobot-voice" / "meetings"
    assert default_meetings_dir() != legacy_meetings_dir()


def test_default_meetings_dir_frozen_next_to_exe(monkeypatch, tmp_path: Path) -> None:
    fake_exe = tmp_path / "krabobot-voice.exe"
    fake_exe.write_bytes(b"")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe))
    monkeypatch.delenv("KRABOBOT_VOICE_CONFIG_DIR", raising=False)

    assert default_meetings_dir() == tmp_path.resolve() / "meetings"


def test_new_meeting_wav_path_uses_default_when_empty(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KRABOBOT_VOICE_CONFIG_DIR", str(tmp_path))
    path = new_meeting_wav_path(None)
    assert path.parent == tmp_path.resolve() / "meetings"
    assert path.suffix == ".wav"
    assert path.parent.is_dir()
