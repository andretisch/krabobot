"""Microphone / WASAPI loopback capture, beep, and WAV playback (Windows-friendly)."""

from __future__ import annotations

import io
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

try:
    import sounddevice as sd
except ImportError as e:  # pragma: no cover
    sd = None  # type: ignore[assignment]
    _SD_ERR = e
else:
    _SD_ERR = None

try:
    import pyaudiowpatch as pyaudio  # type: ignore[import-untyped]
except ImportError as e:  # pragma: no cover
    pyaudio = None  # type: ignore[assignment]
    _PA_ERR = e
else:
    _PA_ERR = None


def require_sounddevice() -> None:
    if sd is None:
        raise RuntimeError(
            "sounddevice is required for mic/playback. "
            f"Install with: pip install sounddevice  ({_SD_ERR})"
        )


def require_pyaudiowpatch() -> None:
    if pyaudio is None:
        raise RuntimeError(
            "PyAudioWPatch is required for WASAPI loopback (meeting.capture=loopback|mix). "
            f"Install with: pip install PyAudioWPatch  ({_PA_ERR})"
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


def resample_mono(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-resample a 1-D float/int array to ``dst_rate`` (pure numpy)."""
    arr = np.asarray(x).reshape(-1)
    if src_rate == dst_rate or arr.size == 0:
        return arr.astype(np.float32, copy=False)
    n_dst = max(1, int(round(arr.size * float(dst_rate) / float(src_rate))))
    xp = np.linspace(0.0, 1.0, num=arr.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
    return np.interp(x_new, xp, arr.astype(np.float64)).astype(np.float32)


def downmix_to_mono_float(arr: np.ndarray) -> np.ndarray:
    """Channel-average to mono float32 in roughly [-1, 1] if int16, else pass-through."""
    x = np.asarray(arr)
    is_int = np.issubdtype(x.dtype, np.integer)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.reshape(-1).astype(np.float32, copy=False)
    if is_int:
        return x / 32768.0
    return x


def float_to_pcm16(x: np.ndarray) -> np.ndarray:
    """Clip float [-1, 1] to int16 PCM."""
    return np.clip(np.asarray(x, dtype=np.float64) * 32767.0, -32768, 32767).astype(np.int16)


def mix_pcm16_average(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Best-effort sync mix: truncate to min length, average, clip-protect.

    Echo risk: if speakers play the local mic loudly, ``mix`` can double that voice.
    Prefer headphones for Teams / meeting capture.
    """
    aa = np.asarray(a, dtype=np.int16).reshape(-1)
    bb = np.asarray(b, dtype=np.int16).reshape(-1)
    n = min(aa.size, bb.size)
    if n == 0:
        return np.zeros(0, dtype=np.int16)
    mixed = (aa[:n].astype(np.int32) + bb[:n].astype(np.int32)) // 2
    return np.clip(mixed, -32768, 32767).astype(np.int16)


# Back-compat alias used by MicStream
_resample_mono = resample_mono


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


def _match_device_name(name: str, needle: str) -> bool:
    n = (needle or "").strip().lower()
    if not n:
        return False
    return n in (name or "").strip().lower()


def _candidate_input_devices(
    preferred: str | int | None = None,
) -> list[tuple[int | None, int, int]]:
    """Return (device_id|None, sample_rate, channels) candidates for open attempts."""
    require_sounddevice()
    out: list[tuple[int | None, int, int]] = []
    seen: set[tuple[int | None, int, int]] = set()

    def add(dev: int | None, rate: int, ch: int) -> None:
        key = (dev, int(rate), int(ch))
        if key not in seen and rate > 0 and ch > 0:
            seen.add(key)
            out.append(key)

    # Explicit preferred device first (index or name substring).
    if preferred is not None and str(preferred).strip() != "":
        try:
            if isinstance(preferred, int) or str(preferred).strip().isdigit():
                idx = int(preferred)
                info = sd.query_devices(idx)
                if int(info.get("max_input_channels") or 0) >= 1:
                    rate = int(info.get("default_samplerate") or 48000)
                    ch = min(2, int(info.get("max_input_channels") or 1))
                    add(idx, rate, ch)
                    add(idx, rate, 1)
            else:
                needle = str(preferred).strip()
                for i, info in enumerate(sd.query_devices()):
                    if int(info.get("max_input_channels") or 0) < 1:
                        continue
                    if _match_device_name(str(info.get("name") or ""), needle):
                        rate = int(info.get("default_samplerate") or 48000)
                        ch = min(2, int(info.get("max_input_channels") or 1))
                        add(i, rate, ch)
                        add(i, rate, 1)
        except Exception:
            pass

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


def resolve_loopback_device_info(
    *,
    device: str | int | None = None,
    output_device: str | int | None = None,
) -> dict[str, Any]:
    """Pick a WASAPI loopback device via PyAudioWPatch.

    Priority:
    1. ``device`` / ``meeting.loopback_device`` (index or name substring)
    2. Loopback matching ``output_device`` / ``audio.output_device``
    3. Loopback for the default WASAPI output device
    """
    require_pyaudiowpatch()
    assert pyaudio is not None
    pa = pyaudio.PyAudio()
    try:
        loopbacks = list(pa.get_loopback_device_info_generator())
        if not loopbacks:
            raise RuntimeError(
                "WASAPI loopback devices not found. "
                "Проверьте устройство воспроизведения Windows и pip install PyAudioWPatch."
            )

        def by_pref(pref: str | int | None) -> dict[str, Any] | None:
            if pref is None or str(pref).strip() == "":
                return None
            if isinstance(pref, int) or str(pref).strip().isdigit():
                idx = int(pref)
                for lb in loopbacks:
                    if int(lb["index"]) == idx:
                        return lb
                # Also allow selecting by output device index → matching loopback name
                try:
                    out_info = pa.get_device_info_by_index(idx)
                    out_name = str(out_info.get("name") or "")
                    for lb in loopbacks:
                        if out_name and out_name in str(lb.get("name") or ""):
                            return lb
                except Exception:
                    pass
                return None
            needle = str(pref).strip()
            for lb in loopbacks:
                if _match_device_name(str(lb.get("name") or ""), needle):
                    return lb
            return None

        chosen = by_pref(device)
        if chosen is None:
            chosen = by_pref(output_device)
        if chosen is None:
            try:
                wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_out = pa.get_device_info_by_index(int(wasapi["defaultOutputDevice"]))
                out_name = str(default_out.get("name") or "")
                for lb in loopbacks:
                    if out_name and out_name in str(lb.get("name") or ""):
                        chosen = lb
                        break
            except Exception:
                chosen = None
        if chosen is None:
            chosen = loopbacks[0]
        return dict(chosen)
    finally:
        pa.terminate()


class MicStream:
    """Blocking mic reader producing int16 mono @ target sample_rate."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        block_ms: int = 30,
        device: str | int | None = None,
    ) -> None:
        require_sounddevice()
        self.sample_rate = sample_rate
        self.block = max(1, int(sample_rate * block_ms / 1000))
        self._preferred = device
        self._stream = None
        self._capture_rate = sample_rate
        self._channels = 1
        self._carry = np.zeros(0, dtype=np.float32)

    def __enter__(self) -> MicStream:
        last_err: Exception | None = None
        for device, rate, channels in _candidate_input_devices(self._preferred):
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
        arr = downmix_to_mono_float(np.asarray(data, dtype=np.float32))
        if self._capture_rate != self.sample_rate:
            arr = resample_mono(arr, self._capture_rate, self.sample_rate)
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
        return float_to_pcm16(out)


class LoopbackStream:
    """WASAPI loopback reader (system / Teams playback) → int16 mono @ sample_rate.

    Uses PyAudioWPatch; stock sounddevice PortAudio wheels lack loopback.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        block_ms: int = 30,
        device: str | int | None = None,
        output_device: str | int | None = None,
    ) -> None:
        require_pyaudiowpatch()
        self.sample_rate = sample_rate
        self.block = max(1, int(sample_rate * block_ms / 1000))
        self._device_pref = device
        self._output_pref = output_device
        self._pa = None
        self._stream = None
        self._capture_rate = sample_rate
        self._channels = 1
        self._carry = np.zeros(0, dtype=np.float32)
        self._device_name = ""

    def __enter__(self) -> LoopbackStream:
        assert pyaudio is not None
        info = resolve_loopback_device_info(
            device=self._device_pref,
            output_device=self._output_pref,
        )
        self._pa = pyaudio.PyAudio()
        idx = int(info["index"])
        channels = max(1, int(info.get("maxInputChannels") or 1))
        rate = int(info.get("defaultSampleRate") or 48000)
        self._device_name = str(info.get("name") or "")
        try:
            stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=idx,
                frames_per_buffer=max(256, int(rate * 0.03)),
            )
        except Exception:
            # Some devices prefer int16
            stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=idx,
                frames_per_buffer=max(256, int(rate * 0.03)),
            )
            self._fmt = "int16"
        else:
            self._fmt = "float32"
        self._stream = stream
        self._capture_rate = rate
        self._channels = channels
        return self

    def __exit__(self, *exc: object) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    @property
    def device_name(self) -> str:
        return self._device_name

    def read_block(self) -> np.ndarray:
        assert self._stream is not None
        need_out = self.block - self._carry.size
        if need_out < 1:
            need_out = self.block
        raw_n = max(
            256,
            int(np.ceil(need_out * float(self._capture_rate) / float(self.sample_rate))),
        )
        raw = self._stream.read(raw_n, exception_on_overflow=False)
        if self._fmt == "float32":
            data = np.frombuffer(raw, dtype=np.float32)
        else:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if self._channels > 1:
            usable = (data.size // self._channels) * self._channels
            data = data[:usable].reshape(-1, self._channels)
        arr = downmix_to_mono_float(data)
        if self._capture_rate != self.sample_rate:
            arr = resample_mono(arr, self._capture_rate, self.sample_rate)
        if self._carry.size:
            arr = np.concatenate([self._carry, arr])
        if arr.size >= self.block:
            out = arr[: self.block]
            self._carry = arr[self.block :]
        else:
            out = np.zeros(self.block, dtype=np.float32)
            out[: arr.size] = arr
            self._carry = np.zeros(0, dtype=np.float32)
        return float_to_pcm16(out)
