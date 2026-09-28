"""Unit tests for MFCC embedding KWS (no mic / no server)."""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.kws import (  # noqa: E402
    EmbeddingKws,
    cosine_similarity,
    mfcc_embedding,
    score_phrase_against_refs,
)

SR = 16000


def _tone(duration_s: float, freq: float = 220.0, amp: float = 0.25) -> np.ndarray:
    n = int(duration_s * SR)
    t = np.arange(n, dtype=np.float32) / float(SR)
    x = amp * np.sin(2 * np.pi * freq * t)
    # Mild AM to look less like pure silence-gated noise
    x *= 0.7 + 0.3 * np.sin(2 * np.pi * 3.0 * t)
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)


def _noise(duration_s: float, amp: float = 0.05) -> np.ndarray:
    n = int(duration_s * SR)
    x = amp * np.random.default_rng(0).standard_normal(n).astype(np.float32)
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)


def test_mfcc_self_similarity_high() -> None:
    clip = _tone(1.2, freq=180.0)
    emb_a = mfcc_embedding(clip, sample_rate=SR)
    emb_b = mfcc_embedding(clip, sample_rate=SR)
    assert cosine_similarity(emb_a, emb_b) > 0.99


def test_mfcc_different_signals_lower() -> None:
    a = _tone(1.2, freq=180.0)
    b = _tone(1.2, freq=880.0)
    sa = score_phrase_against_refs(a, [mfcc_embedding(a, sample_rate=SR)], sample_rate=SR)
    sb = score_phrase_against_refs(b, [mfcc_embedding(a, sample_rate=SR)], sample_rate=SR)
    assert sa > 0.95
    assert sb < sa


def test_embedding_kws_detects_matching_window(tmp_path: Path) -> None:
    ref = _tone(1.4, freq=200.0, amp=0.3)
    wav_path = tmp_path / "ref.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(ref.tobytes())

    kws = EmbeddingKws.from_refs_dir(
        tmp_path,
        sample_rate=SR,
        threshold=0.90,
        window_s=1.2,
        hop_s=0.3,
        energy_threshold=0.01,
        auto_tts=False,
        block=480,
    )
    assert kws.ready

    # Feed silence then the same phrase
    block = 480
    hit = None
    silence = np.zeros(block, dtype=np.int16)
    for _ in range(10):
        hit = kws.process(silence) or hit
    assert hit is None

    # Stream the reference audio in blocks
    for i in range(0, ref.size - block + 1, block):
        hit = kws.process(ref[i : i + block])
        if hit is not None:
            break
    assert hit is not None
    assert hit.score >= 0.90


def test_noise_does_not_match_high_threshold(tmp_path: Path) -> None:
    ref = _tone(1.4, freq=200.0, amp=0.3)
    wav_path = tmp_path / "ref.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(ref.tobytes())

    kws = EmbeddingKws.from_refs_dir(
        tmp_path,
        sample_rate=SR,
        threshold=0.92,
        window_s=1.2,
        hop_s=0.3,
        energy_threshold=0.01,
        auto_tts=False,
        block=480,
    )
    noise = _noise(2.0, amp=0.2)
    block = 480
    for i in range(0, noise.size - block + 1, block):
        assert kws.process(noise[i : i + block]) is None
