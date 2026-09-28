"""Microphone capture, beep, and WAV playback (Windows-friendly)."""

from __future__ import annotations

import io
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except ImportError as e:  # pragma: no cover
    sd = None  # type: ignore[assignment]
    _SD_ERR = e
else:
    _SD_ERR = None


def require_sounddevice() -> None:
    if sd is None:
        raise RuntimeError(
            "sounddevice is required for mic/playback. "
            f"Install with: pip install sounddevice  ({_SD_ERR})"
        )


def pcm16_to_wav_bytes(pcm16: bytes | np.ndarray, *, sample_rate: int = 16000) -> bytes:
    if isinstance(pcm16, np.ndarray):
        data = np.asarray(pcm16, dtype=np.int16).tobytes()
    else:
        data = bytes(pcm16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data)
    return buf.getvalue()


def play_beep(*, freq: int = 1000, duration_ms: int = 150) -> None:
    """Short acknowledgment beep (winsound on Windows, tone fallback elsewhere)."""
    if sys.platform == "win32":
        try:
            import winsound

            winsound.Beep(int(freq), int(duration_ms))
            return
        except Exception:
            pass
    require_sounddevice()
    sr = 16000
    t = np.linspace(0, duration_ms / 1000.0, int(sr * duration_ms / 1000.0), endpoint=False)
    tone = (0.25 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sd.play(tone, sr)
    sd.wait()


def play_wav_bytes(wav_bytes: bytes) -> None:
    """Play a WAV blob via winsound (Windows) or sounddevice."""
    if not wav_bytes:
        return
    if sys.platform == "win32":
        try:
            import winsound

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                path = tmp.name
            try:
                winsound.PlaySound(path, winsound.SND_FILENAME)
                return
            finally:
                Path(path).unlink(missing_ok=True)
        except Exception:
            pass

    require_sounddevice()
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
        width = wf.getsampwidth()
    if width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    if channels > 1:
        audio = audio.reshape(-1, channels)
    sd.play(audio, rate)
    sd.wait()


def _resample_mono(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or x.size == 0:
        return x
    n_dst = max(1, int(round(x.size * float(dst_rate) / float(src_rate))))
    xp = np.linspace(0.0, 1.0, num=x.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
    return np.interp(x_new, xp, x.astype(np.float64)).astype(np.float32)


def _candidate_input_devices() -> list[tuple[int | None, int, int]]:
    """Return (device_id|None, sample_rate, channels) candidates for open attempts."""
    require_sounddevice()
    out: list[tuple[int | None, int, int]] = []
    seen: set[tuple[int | None, int, int]] = set()

    def add(dev: int | None, rate: int, ch: int) -> None:
        key = (dev, int(rate), int(ch))
        if key not in seen and rate > 0 and ch > 0:
            seen.add(key)
            out.append(key)

    # Prefer WASAPI / DirectSound; skip WDM-KS (no blocking PortAudio API).
    prefer = ("WASAPI", "DirectSound", "MME")
    try:
        apis = list(sd.query_hostapis())
        for tag in prefer:
            for api in apis:
                name = str(api.get("name") or "")
                if tag not in name:
                    continue
                din = api.get("default_input_device")
                if din is None or int(din) < 0:
                    continue
                info = sd.query_devices(int(din))
                ch = min(2, int(info.get("max_input_channels") or 1))
                rate = int(info.get("default_samplerate") or 48000)
                add(int(din), rate, ch)
                add(int(din), rate, 1)
    except Exception:
        pass

    try:
        default_in = sd.default.device[0] if sd.default.device else None
        if default_in is not None and int(default_in) >= 0:
            info = sd.query_devices(int(default_in))
            rate = int(info.get("default_samplerate") or 48000)
            ch = min(2, int(info.get("max_input_channels") or 1))
            add(int(default_in), rate, ch)
    except Exception:
        pass

    add(None, 48000, 1)
    add(None, 44100, 1)
    add(None, 16000, 1)

    # Explicit scan of a few input devices (skip WDM-KS)
    try:
        apis = sd.query_hostapis()
        for i, info in enumerate(sd.query_devices()):
            if int(info.get("max_input_channels") or 0) < 1:
                continue
            ha = int(info.get("hostapi") or 0)
            ha_name = str(apis[ha].get("name") or "") if ha < len(apis) else ""
            if "WDM-KS" in ha_name:
                continue
            rate = int(info.get("default_samplerate") or 48000)
            ch = min(2, int(info.get("max_input_channels") or 1))
            add(i, rate, ch)
            if len(out) > 24:
                break
    except Exception:
        pass
    return out


class MicStream:
    """Blocking mic reader producing int16 mono @ target sample_rate."""

    def __init__(self, *, sample_rate: int = 16000, block_ms: int = 30) -> None:
        require_sounddevice()
        self.sample_rate = sample_rate
        self.block = max(1, int(sample_rate * block_ms / 1000))
        self._stream = None
        self._capture_rate = sample_rate
        self._channels = 1
        self._carry = np.zeros(0, dtype=np.float32)

    def __enter__(self) -> MicStream:
        last_err: Exception | None = None
        for device, rate, channels in _candidate_input_devices():
            try:
                stream = sd.InputStream(
                    device=device,
                    samplerate=rate,
                    channels=channels,
                    dtype="float32",
                    blocksize=0,
                    latency="high",
                )
                stream.start()
                # Smoke-read one chunk
                n = max(256, int(rate * 0.02))
                stream.read(n)
                self._stream = stream
                self._capture_rate = rate
                self._channels = channels
                return self
            except Exception as e:
                last_err = e
                try:
                    if "stream" in locals():
                        stream.close()  # type: ignore[name-defined]
                except Exception:
                    pass
        hint = (
            "Не удалось открыть микрофон. "
            "Проверьте: Параметры Windows → Конфиденциальность → Микрофон "
            "(доступ для классических приложений / Python / терминала). "
            "Закройте другие программы, занявшие микрофон."
        )
        if last_err is not None:
            raise RuntimeError(f"{hint} Детали: {last_err}") from last_err
        raise RuntimeError(hint)

    def __exit__(self, *exc: object) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def read_block(self) -> np.ndarray:
        assert self._stream is not None
        # Read enough raw frames so after resample we cover ~one output block
        need_out = self.block - self._carry.size
        if need_out < 1:
            need_out = self.block
        raw_n = max(
            256,
            int(np.ceil(need_out * float(self._capture_rate) / float(self.sample_rate))),
        )
        data, _overflowed = self._stream.read(raw_n)
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        arr = arr.reshape(-1)
        if self._capture_rate != self.sample_rate:
            arr = _resample_mono(arr, self._capture_rate, self.sample_rate)
        if self._carry.size:
            arr = np.concatenate([self._carry, arr])
        if arr.size >= self.block:
            out = arr[: self.block]
            self._carry = arr[self.block :]
        else:
            # pad rare short reads
            out = np.zeros(self.block, dtype=np.float32)
            out[: arr.size] = arr
            self._carry = np.zeros(0, dtype=np.float32)
        pcm = np.clip(out * 32767.0, -32768, 32767).astype(np.int16)
        return pcm
