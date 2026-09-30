"""Silero VAD (ONNX) — speech vs non-speech without PyTorch.

Uses the official ``silero_vad.onnx`` (snakers4/silero-vad) via onnxruntime.

Resolution order (empty ``model_path``):
1. ``<app>/models/silero_vad.onnx`` (exe dir or package root)
2. Legacy ``%LOCALAPPDATA%/krabobot-voice/models/silero_vad.onnx`` if present
3. Prefer app path as the canonical location

Portable / frozen builds never download: missing model → clear error to place
the file under ``models/``. Non-frozen may download once into the app models dir.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

# Silero v5/v6 @ 16 kHz: 512-sample windows + 64-sample context prepend.
_WINDOW_16K = 512
_CONTEXT_16K = 64
_STATE_SHAPE = (2, 1, 128)
_SAMPLE_RATE = 16000

# Upstream ONNX (opset 16). Downloaded only for non-frozen runs when missing.
_DEFAULT_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/"
    "src/silero_vad/data/silero_vad.onnx"
)

_MISSING_HINT = (
    "Положите модель в models/silero_vad.onnx рядом с приложением "
    "(или укажите vad.model_path)."
)


def legacy_silero_model_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "krabobot-voice" / "models" / "silero_vad.onnx"


def default_silero_model_path() -> Path:
    """Canonical Silero path: app ``models/``, else legacy LocalAppData if present."""
    from krabobot_voice.config import app_silero_model_path, is_frozen

    app_path = app_silero_model_path()
    if app_path.is_file() and app_path.stat().st_size > 10_000:
        return app_path
    if is_frozen():
        return app_path
    legacy = legacy_silero_model_path()
    if legacy.is_file() and legacy.stat().st_size > 10_000:
        return legacy
    return app_path


def resolve_silero_model_path(path: str | Path | None = None) -> Path:
    if path is not None and str(path).strip():
        return Path(path).expanduser()
    return default_silero_model_path()


def ensure_silero_model(
    path: str | Path | None = None,
    *,
    url: str = _DEFAULT_MODEL_URL,
    allow_download: bool | None = None,
) -> Path:
    """Return a local ONNX path.

    Downloads only when ``allow_download`` is true (default: not frozen) and the
    file is missing. Portable builds never download.
    """
    from krabobot_voice.config import is_frozen

    dest = resolve_silero_model_path(path)
    if dest.is_file() and dest.stat().st_size > 10_000:
        return dest
    if allow_download is None:
        allow_download = not is_frozen()
    if not allow_download:
        raise FileNotFoundError(f"Silero ONNX not found: {dest}. {_MISSING_HINT}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".onnx.partial")
    try:
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 — fixed upstream URL
        tmp.replace(dest)
    except Exception:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise
    return dest


class SileroOnnxVad:
    """Streaming Silero VAD: feed arbitrary PCM16 blocks, get speech probability."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        sample_rate: int = _SAMPLE_RATE,
        threshold: float = 0.5,
        force_cpu: bool = True,
    ) -> None:
        import onnxruntime as ort  # type: ignore[import-not-found]

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Silero ONNX not found: {path}. {_MISSING_HINT}")
        if int(sample_rate) != _SAMPLE_RATE:
            raise ValueError(f"Silero VAD requires {_SAMPLE_RATE} Hz (got {sample_rate})")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        providers = (
            ["CPUExecutionProvider"]
            if force_cpu and "CPUExecutionProvider" in ort.get_available_providers()
            else None
        )
        self._session = ort.InferenceSession(
            str(path),
            sess_options=opts,
            providers=providers,
        )
        self.sample_rate = _SAMPLE_RATE
        self.threshold = float(threshold)
        self._window = _WINDOW_16K
        self._context_n = _CONTEXT_16K
        self._pending = np.zeros(0, dtype=np.float32)
        self._context = np.zeros((1, self._context_n), dtype=np.float32)
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._last_prob = 0.0

    def reset(self) -> None:
        """Clear LSTM state / context (call on new capture stream)."""
        self._pending = np.zeros(0, dtype=np.float32)
        self._context = np.zeros((1, self._context_n), dtype=np.float32)
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._last_prob = 0.0

    @property
    def last_prob(self) -> float:
        return float(self._last_prob)

    def _run_window(self, window: np.ndarray) -> float:
        # window: (512,) float32 in [-1, 1]
        audio = window.reshape(1, -1).astype(np.float32, copy=False)
        x = np.concatenate([self._context, audio], axis=1)  # (1, 576)
        sr = np.array(_SAMPLE_RATE, dtype=np.int64)
        out, state = self._session.run(
            None,
            {"input": x, "state": self._state, "sr": sr},
        )
        self._state = state
        self._context = audio[:, -self._context_n :]
        prob = float(np.asarray(out).reshape(-1)[0])
        self._last_prob = prob
        return prob

    def push_pcm16(self, pcm16: np.ndarray) -> float:
        """Append int16 mono samples; run complete 512-windows; return latest prob."""
        arr = np.asarray(pcm16, dtype=np.int16).reshape(-1)
        if arr.size == 0:
            return self._last_prob
        samples = (arr.astype(np.float32) / 32768.0).astype(np.float32, copy=False)
        if self._pending.size:
            samples = np.concatenate([self._pending, samples])
        # Process as many full windows as possible.
        n = samples.size
        pos = 0
        while pos + self._window <= n:
            self._run_window(samples[pos : pos + self._window])
            pos += self._window
        self._pending = samples[pos:].copy() if pos < n else np.zeros(0, dtype=np.float32)
        return self._last_prob

    def is_speech(self, pcm16: np.ndarray) -> bool:
        """True if latest Silero probability ≥ threshold after consuming ``pcm16``."""
        return self.push_pcm16(pcm16) >= self.threshold


def load_silero_vad(
    *,
    model_path: str | Path | None = None,
    threshold: float = 0.5,
    sample_rate: int = _SAMPLE_RATE,
    download: bool | None = None,
) -> SileroOnnxVad:
    """Load Silero ONNX. Portable builds never download; see ``ensure_silero_model``."""
    from krabobot_voice.config import is_frozen

    if download is None:
        download = not is_frozen()
    path = resolve_silero_model_path(model_path)
    if not path.is_file():
        if not download:
            raise FileNotFoundError(f"Silero ONNX not found: {path}. {_MISSING_HINT}")
        path = ensure_silero_model(path, allow_download=True)
    return SileroOnnxVad(path, sample_rate=sample_rate, threshold=threshold)


def silero_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def preload_onnxruntime() -> bool:
    """Import ``onnxruntime`` before any WinRT (``winrt``) native modules.

    On Windows, loading ``winrt`` first then ``onnxruntime`` causes a hard
    ACCESS_VIOLATION (exit ``0xC0000005``) that Python cannot catch — the
    portable exe exits silently after mic-permission success. Reverse order
    is safe. Returns True if onnxruntime imported.
    """
    try:
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False
