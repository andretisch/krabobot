"""Unit tests for mic device listing / preferred resolution (no real PortAudio)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice import audio_io  # noqa: E402


class _FakeSd:
    def __init__(self, devices: list[dict[str, Any]], hostapis: list[dict[str, Any]]) -> None:
        self._devices = devices
        self._hostapis = hostapis
        self.default = SimpleNamespace(device=[1, 4])
        self.WasapiSettings = lambda **kw: SimpleNamespace(**kw)

    def query_devices(self, index: int | None = None, kind: str | None = None):  # noqa: ARG002
        if index is None:
            return list(self._devices)
        return self._devices[int(index)]

    def query_hostapis(self):
        return list(self._hostapis)


@pytest.fixture()
def fake_sd(monkeypatch: pytest.MonkeyPatch) -> _FakeSd:
    devices = [
        {
            "name": "Переназначение звуковых устр. - Input",
            "max_input_channels": 2,
            "default_samplerate": 44100,
            "hostapi": 0,
        },
        {
            "name": "Микрофон (Lenovo USB Headset)",
            "max_input_channels": 2,
            "default_samplerate": 48000,
            "hostapi": 2,
        },
        {
            "name": "Speakers Only",
            "max_input_channels": 0,
            "default_samplerate": 48000,
            "hostapi": 2,
        },
        {
            "name": "Mic WDM",
            "max_input_channels": 2,
            "default_samplerate": 48000,
            "hostapi": 3,
        },
    ]
    hostapis = [
        {"name": "MME", "default_input_device": 0},
        {"name": "Windows DirectSound", "default_input_device": 1},
        {"name": "Windows WASAPI", "default_input_device": 1},
        {"name": "Windows WDM-KS", "default_input_device": 3},
    ]
    fake = _FakeSd(devices, hostapis)
    monkeypatch.setattr(audio_io, "sd", fake)
    return fake


def test_list_input_devices_skips_wdm_ks(fake_sd: _FakeSd) -> None:
    rows = audio_io.list_input_devices()
    names = [r["name"] for r in rows]
    assert "Микрофон (Lenovo USB Headset)" in names
    assert "Mic WDM" not in names
    assert any(r.get("virtual_mapper") for r in rows)


def test_resolve_preferred_by_substring(fake_sd: _FakeSd) -> None:
    assert audio_io.resolve_preferred_input_device("Lenovo") == 1
    assert audio_io.resolve_preferred_input_device("1") == 1


def test_resolve_preferred_stale_index_and_name(fake_sd: _FakeSd) -> None:
    assert audio_io.resolve_preferred_input_device("99") is None
    assert audio_io.resolve_preferred_input_device("no-such-mic") is None
    assert audio_io.resolve_preferred_input_device("") is None


def test_candidates_include_none_and_skip_mapper(fake_sd: _FakeSd) -> None:
    cands = audio_io._candidate_input_devices(None)
    assert any(c[0] is None for c in cands)
    # Mapper index 0 should not appear as a concrete candidate.
    assert all(c[0] != 0 for c in cands)


def test_mic_privacy_hint_mentions_exe_when_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_io, "_is_frozen", lambda: True)
    text = audio_io.mic_privacy_hint()
    assert "krabobot-voice.exe" in text
    assert "ms-settings:privacy-microphone" in text
    monkeypatch.setattr(audio_io, "_is_frozen", lambda: False)
    text2 = audio_io.mic_privacy_hint()
    assert "python.exe" in text2.lower() or "Python" in text2


def test_format_input_devices_lines(fake_sd: _FakeSd) -> None:
    text = audio_io.format_input_devices_lines()
    assert "[1]" in text
    assert "Lenovo" in text
