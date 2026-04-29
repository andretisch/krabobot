"""Local STT backend based on GigaAM ONNX runtime."""

from __future__ import annotations

from pathlib import Path
import threading


class GigaamOnnxTranscriber:
    """Thread-safe singleton-like loader for ONNX sessions."""

    _lock = threading.Lock()
    _cache: dict[tuple[str, str], tuple[object, object]] = {}
    _prepare_locks: dict[str, threading.Lock] = {}

    @classmethod
    def _prepare_lock(cls, key: str) -> threading.Lock:
        lock = cls._prepare_locks.get(key)
        if lock is not None:
            return lock
        with cls._lock:
            lock = cls._prepare_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                cls._prepare_locks[key] = lock
            return lock

    @classmethod
    def ensure_onnx_dir(cls, *, onnx_dir: str, model_version: str) -> str:
        """Ensure ONNX export exists; export from gigaam model if missing."""
        out_dir = Path(onnx_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        if any(out_dir.glob("*.onnx")):
            return str(out_dir)

        lock = cls._prepare_lock(str(out_dir))
        with lock:
            if any(out_dir.glob("*.onnx")):
                return str(out_dir)
            import gigaam

            model = gigaam.load_model(model_version)
            model.to_onnx(dir_path=str(out_dir))
        return str(out_dir)

    @classmethod
    def _load_sessions(cls, onnx_dir: str, model_version: str) -> tuple[object, object]:
        key = (onnx_dir, model_version)
        cached = cls._cache.get(key)
        if cached is not None:
            return cached
        with cls._lock:
            cached = cls._cache.get(key)
            if cached is not None:
                return cached
            from gigaam.onnx_utils import load_onnx

            sessions, model_cfg = load_onnx(onnx_dir, model_version)
            cls._cache[key] = (sessions, model_cfg)
            return sessions, model_cfg

    @classmethod
    def transcribe(cls, file_path: str | Path, *, onnx_dir: str, model_version: str) -> str:
        from gigaam.onnx_utils import infer_onnx

        audio_path = str(Path(file_path).expanduser().resolve())
        prepared_dir = cls.ensure_onnx_dir(onnx_dir=onnx_dir, model_version=model_version)
        sessions, model_cfg = cls._load_sessions(prepared_dir, model_version)
        result = infer_onnx(audio_path, model_cfg, sessions)
        return str(result or "").strip()

