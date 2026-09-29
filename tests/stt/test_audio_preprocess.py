"""Unit tests for server-side STT audio alignment."""

from __future__ import annotations

import numpy as np

from krabobot.stt.audio_preprocess import (
    TARGET_SR,
    align_waveform_float,
    frame_rms,
    normalize_level,
    remove_dc_offset,
    trim_silence,
)


def test_align_raises_quiet_and_removes_dc() -> None:
    n = TARGET_SR
    t = np.arange(n, dtype=np.float32) / float(TARGET_SR)
    x = 0.02 * np.sin(2 * np.pi * 200.0 * t) + 0.15  # DC + quiet
    y = align_waveform_float(x, sample_rate=TARGET_SR, trim=False)
    assert abs(float(y.mean())) < 0.02
    assert frame_rms(y) > frame_rms(x - float(x.mean()))


def test_trim_and_normalize_pipeline() -> None:
    speech = 0.05 * np.sin(2 * np.pi * 180.0 * np.arange(int(0.7 * TARGET_SR)) / TARGET_SR)
    pad = np.zeros(int(0.4 * TARGET_SR), dtype=np.float32)
    x = np.concatenate([pad, speech.astype(np.float32), pad])
    y = align_waveform_float(x, sample_rate=TARGET_SR, energy_threshold=0.01)
    assert y.size < x.size
    assert float(np.max(np.abs(y))) > 0.1
    assert frame_rms(y) > 0.05


def test_normalize_does_not_clip_hard() -> None:
    x = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    y = normalize_level(x)
    assert float(np.max(np.abs(y))) <= 0.95


def test_trim_all_silence_keeps_original() -> None:
    x = np.zeros(TARGET_SR, dtype=np.float32)
    y = trim_silence(x, sample_rate=TARGET_SR)
    assert y.size == x.size


def test_remove_dc_offset_zero_mean() -> None:
    x = np.ones(500, dtype=np.float32) * 0.3
    assert abs(float(remove_dc_offset(x).mean())) < 1e-6
