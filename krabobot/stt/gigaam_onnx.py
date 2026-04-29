"""Local STT backend based on GigaAM ONNX runtime."""

from __future__ import annotations

from pathlib import Path
import threading


class GigaamOnnxTranscriber:
    """Thread-safe singleton-like loader for ONNX sessions."""

    _lock = threading.Lock()
    _cache: dict[tuple[str, str], tuple[str, object]] = {}
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
    @staticmethod
    def _split_model_version(model_version: str) -> tuple[str, str]:
        mv = (model_version or "v2_ctc").strip().lower()
        if "_" in mv:
            version, model_type = mv.split("_", 1)
        else:
            version, model_type = "v2", "ctc"
        if model_type not in {"ctc", "rnnt"}:
            model_type = "ctc"
        return version, model_type

    @classmethod
    def _load_sessions(cls, onnx_dir: str, model_version: str) -> tuple[str, object]:
        key = (onnx_dir, model_version)
        cached = cls._cache.get(key)
        if cached is not None:
            return cached
        with cls._lock:
            cached = cls._cache.get(key)
            if cached is not None:
                return cached
            from gigaam.onnx_utils import load_onnx_sessions

            version, model_type = cls._split_model_version(model_version)
            sessions = load_onnx_sessions(onnx_dir, model_type=model_type, model_version=version)
            cls._cache[key] = (model_type, sessions)
            return model_type, sessions

    @classmethod
    def transcribe(cls, file_path: str | Path, *, onnx_dir: str, model_version: str) -> str:
        from gigaam.onnx_utils import transcribe_sample

        audio_path = str(Path(file_path).expanduser().resolve())
        prepared_dir = cls.ensure_onnx_dir(onnx_dir=onnx_dir, model_version=model_version)
        model_type, sessions = cls._load_sessions(prepared_dir, model_version)
        result = transcribe_sample(audio_path, model_type=model_type, sessions=sessions)
        return str(result or "").strip()

