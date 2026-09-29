"""Microphone / WASAPI loopback capture, beep, and WAV playback (Windows-friendly)."""

from __future__ import annotations

import io
import sys
import tempfile
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np

_T = TypeVar("_T")

# Max time a single PortAudio/sounddevice read may spend in C without returning
# to Python (so Ctrl+C / wall-clock VAD timeouts can fire).
_DEFAULT_READ_TIMEOUT_S = 0.35

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
            "PyAudioWPatch is required for WASAPI loopback "
            "(audio.listen_source=loopback or meeting.capture=loopback|mix). "
            f"Install with: pip install PyAudioWPatch  ({_PA_ERR})"
        )


@runtime_checkable
class PcmBlockSource(Protocol):
    """MicStream / LoopbackStream duck type: blocking int16 mono blocks."""

    sample_rate: int
    block: int

    def read_block(self) -> np.ndarray: ...

    def __enter__(self) -> PcmBlockSource: ...

    def __exit__(self, *exc: object) -> None: ...


def with_stream_keepalive(
    mic: object,
    fn: Callable[[], _T],
    sink: Any | None = None,
) -> _T:
    """Run ``fn`` while continuously draining ``mic`` so PortAudio does not overflow.

    While sherpa ASR (or any slow work) runs, the capture thread must keep
    calling ``read_block`` / ``drain_available``. Otherwise WASAPI loopback
    buffers fill and the *next* ``stream.read`` can hang forever in C — where
    Ctrl+C cannot interrupt.

    When ``sink`` is given (typically a bounded ``deque``), drained blocks are
    appended instead of discarded so the segmenter can ``feed_backlog`` them.
    Reads stay nonblocking / Ctrl+C-friendly.
    """
    stop = threading.Event()

    def _drain() -> None:
        while not stop.is_set():
            try:
                if sink is not None:
                    block = getattr(mic, "read_block")()
                    try:
                        sink.append(block)
                    except Exception:
                        pass
                else:
                    drain = getattr(mic, "drain_available", None)
                    if callable(drain):
                        drain()
                    else:
                        getattr(mic, "read_block")()
            except Exception:
                break
            # Short sleep so Ctrl+C / stop can land between drains.
            if stop.wait(0.005):
                break

    t = threading.Thread(target=_drain, name="krabobot-voice-keepalive", daemon=True)
    t.start()
    try:
        return fn()
    finally:
        stop.set()
        t.join(timeout=1.5)


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


def play_beeps(
    count: int,
    *,
    freq: int = 1000,
    duration_ms: int = 120,
    gap_ms: int = 80,
) -> None:
    """Play ``count`` acknowledgment beeps (wake=2, idle-return=1)."""
    n = max(0, int(count))
    for i in range(n):
        play_beep(freq=freq, duration_ms=duration_ms)
        if i + 1 < n and gap_ms > 0:
            time.sleep(max(0.0, float(gap_ms) / 1000.0))


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
    """Pick a WASAPI loopback device via PyAudioWPatch (same path as meeting).

    Priority:
    1. ``device`` / ``meeting.loopback_device`` (index or name substring)
    2. Loopback matching ``output_device`` / ``audio.output_device``
    3. ``get_default_wasapi_loopback()`` / loopback for default WASAPI output
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
                        return dict(lb)
                # Output device index → matching loopback analogue
                try:
                    return dict(pa.get_wasapi_loopback_analogue_by_index(idx))
                except Exception:
                    try:
                        out_info = pa.get_device_info_by_index(idx)
                        out_name = str(out_info.get("name") or "")
                        for lb in loopbacks:
                            if out_name and out_name in str(lb.get("name") or ""):
                                return dict(lb)
                    except Exception:
                        pass
                return None
            needle = str(pref).strip()
            for lb in loopbacks:
                if _match_device_name(str(lb.get("name") or ""), needle):
                    return dict(lb)
            # Name may refer to a playback device — map to its loopback analogue.
            try:
                for info in pa.get_device_info_generator_by_host_api(
                    host_api_type=pyaudio.paWASAPI
                ):
                    if int(info.get("maxOutputChannels") or 0) < 1:
                        continue
                    if info.get("isLoopbackDevice"):
                        continue
                    if _match_device_name(str(info.get("name") or ""), needle):
                        try:
                            return dict(pa.get_wasapi_loopback_analogue_by_dict(info))
                        except Exception:
                            pass
            except Exception:
                pass
            return None

        chosen = by_pref(device)
        if chosen is None:
            chosen = by_pref(output_device)
        if chosen is None:
            try:
                chosen = dict(pa.get_default_wasapi_loopback())
            except Exception:
                chosen = None
        if chosen is None:
            try:
                wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_out = pa.get_device_info_by_index(int(wasapi["defaultOutputDevice"]))
                chosen = dict(pa.get_wasapi_loopback_analogue_by_dict(default_out))
            except Exception:
                chosen = None
        if chosen is None:
            chosen = dict(loopbacks[0])
        if not chosen.get("isLoopbackDevice", True):
            raise RuntimeError(
                f"Resolved device is not WASAPI loopback: {chosen.get('name')!r}. "
                "Задайте meeting.loopback_device / audio.output_device."
            )
        return chosen
    finally:
        pa.terminate()


# Soft boost for WASAPI loopback (often quieter than a close mic after downmix).
DEFAULT_LOOPBACK_GAIN = 2.5


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
        self._device_name = ""
        self._device_index: int | None = None

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
                self._device_index = int(device) if device is not None else None
                try:
                    info = sd.query_devices(device) if device is not None else sd.query_devices(
                        kind="input"
                    )
                    self._device_name = str(info.get("name") or "")
                except Exception:
                    self._device_name = ""
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

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def device_index(self) -> int | None:
        return self._device_index

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
        data = self._read_raw_interruptible(raw_n, timeout_s=_DEFAULT_READ_TIMEOUT_S)
        arr = downmix_to_mono_float(np.asarray(data, dtype=np.float32))
        if self._capture_rate != self.sample_rate:
            arr = resample_mono(arr, self._capture_rate, self.sample_rate)
        if self._carry.size:
            arr = np.concatenate([self._carry, arr])
        if arr.size >= self.block:
            out = arr[: self.block]
            self._carry = arr[self.block :]
        else:
            # pad rare short reads / timeouts
            out = np.zeros(self.block, dtype=np.float32)
            out[: arr.size] = arr
            self._carry = np.zeros(0, dtype=np.float32)
        return float_to_pcm16(out)

    def _read_raw_interruptible(self, raw_n: int, *, timeout_s: float) -> np.ndarray:
        """Read via read_available; sleep in Python so Ctrl+C works."""
        assert self._stream is not None
        deadline = time.monotonic() + max(0.05, float(timeout_s))
        parts: list[np.ndarray] = []
        got = 0
        while got < raw_n:
            try:
                avail = int(getattr(self._stream, "read_available", 0) or 0)
            except Exception:
                avail = 0
            if avail > 0:
                n = min(avail, raw_n - got)
                try:
                    chunk, _overflowed = self._stream.read(n)
                except Exception:
                    break
                parts.append(np.asarray(chunk, dtype=np.float32))
                got += int(np.asarray(chunk).shape[0])
                continue
            if time.monotonic() >= deadline:
                break
            time.sleep(0.005)
        if not parts:
            return np.zeros((0, max(1, self._channels)), dtype=np.float32)
        return np.concatenate(parts, axis=0)

    def drain_available(self) -> None:
        """Discard buffered mic frames without blocking long (ASR keepalive)."""
        assert self._stream is not None
        try:
            avail = int(getattr(self._stream, "read_available", 0) or 0)
        except Exception:
            avail = 0
        if avail <= 0:
            return
        n = min(avail, max(256, int(self._capture_rate * 0.05)))
        try:
            self._stream.read(n)
        except Exception:
            return
        self._carry = np.zeros(0, dtype=np.float32)


class LoopbackStream:
    """WASAPI loopback reader (system / Teams playback) → int16 mono @ sample_rate.

    Uses PyAudioWPatch; stock sounddevice PortAudio wheels lack loopback.
    Same capture path as ``meeting.capture=loopback|mix``.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        block_ms: int = 30,
        device: str | int | None = None,
        output_device: str | int | None = None,
        gain: float = DEFAULT_LOOPBACK_GAIN,
    ) -> None:
        require_pyaudiowpatch()
        self.sample_rate = sample_rate
        self.block = max(1, int(sample_rate * block_ms / 1000))
        self._device_pref = device
        self._output_pref = output_device
        self._gain = float(gain) if gain and gain > 0 else 1.0
        self._pa = None
        self._stream = None
        self._capture_rate = sample_rate
        self._channels = 1
        self._carry = np.zeros(0, dtype=np.float32)
        self._device_name = ""
        self._device_index: int | None = None
        self._fmt = "int16"

    def __enter__(self) -> LoopbackStream:
        assert pyaudio is not None
        info = resolve_loopback_device_info(
            device=self._device_pref,
            output_device=self._output_pref,
        )
        self._pa = pyaudio.PyAudio()
        idx = int(info["index"])
        channels = max(1, int(info.get("maxInputChannels") or 2))
        rate = int(info.get("defaultSampleRate") or 48000)
        self._device_name = str(info.get("name") or "")
        self._device_index = idx
        # Official PyAudioWPatch examples prefer int16 + exact device channel count.
        last_err: Exception | None = None
        stream = None
        for fmt, fmt_name in (
            (pyaudio.paInt16, "int16"),
            (pyaudio.paFloat32, "float32"),
        ):
            for ch in (channels, 2, 1):
                if ch < 1:
                    continue
                try:
                    stream = self._pa.open(
                        format=fmt,
                        channels=ch,
                        rate=rate,
                        input=True,
                        input_device_index=idx,
                        frames_per_buffer=max(512, int(rate * 0.03)),
                    )
                    self._fmt = fmt_name
                    self._channels = ch
                    break
                except Exception as e:
                    last_err = e
                    stream = None
            if stream is not None:
                break
        if stream is None:
            raise RuntimeError(
                f"Не удалось открыть WASAPI loopback "
                f"idx={idx} name={self._device_name!r}. "
                f"Проверьте Privacy->Microphone и устройство воспроизведения. "
                f"Детали: {last_err}"
            ) from last_err
        self._stream = stream
        self._capture_rate = rate
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

    @property
    def device_index(self) -> int | None:
        return self._device_index

    def drain_available(self) -> None:
        """Discard currently buffered loopback frames without blocking forever.

        Used as ASR keepalive: while sherpa runs, keep PortAudio drained so the
        next talk ``read`` does not hang after an overflow.
        """
        assert self._stream is not None
        try:
            avail = int(self._stream.get_read_available())
        except Exception:
            avail = 0
        if avail <= 0:
            return
        # Cap one drain burst so keepalive stays responsive to stop events.
        n = min(avail, max(512, int(self._capture_rate * 0.05)))
        try:
            self._stream.read(n, exception_on_overflow=False)
        except Exception:
            return
        self._carry = np.zeros(0, dtype=np.float32)

    def _read_raw_interruptible(self, raw_n: int, *, timeout_s: float) -> bytes:
        """Read ``raw_n`` frames using get_read_available; never block forever in C.

        Sleeps in Python while waiting so KeyboardInterrupt can fire. On timeout
        returns whatever was collected (caller pads with silence).
        """
        assert self._stream is not None
        deadline = time.monotonic() + max(0.05, float(timeout_s))
        chunks: list[bytes] = []
        got = 0
        sample_width = 4 if self._fmt == "float32" else 2
        while got < raw_n:
            try:
                avail = int(self._stream.get_read_available())
            except Exception:
                avail = 0
            if avail > 0:
                n = min(avail, raw_n - got)
                try:
                    raw = self._stream.read(n, exception_on_overflow=False)
                except Exception:
                    break
                chunks.append(raw)
                # Frame count from byte length (stereo = channels samples per frame).
                frame_bytes = sample_width * self._channels
                if frame_bytes > 0:
                    got += len(raw) // frame_bytes
                continue
            if time.monotonic() >= deadline:
                break
            # Yield to interpreter — Ctrl+C lands here, not inside PortAudio.
            time.sleep(0.005)
        return b"".join(chunks)

    def read_block(self) -> np.ndarray:
        assert self._stream is not None
        need_out = self.block - self._carry.size
        if need_out < 1:
            need_out = self.block
        raw_n = max(
            256,
            int(np.ceil(need_out * float(self._capture_rate) / float(self.sample_rate))),
        )
        raw = self._read_raw_interruptible(raw_n, timeout_s=_DEFAULT_READ_TIMEOUT_S)
        if not raw:
            # Timed out with no frames — return silence so VAD/timeouts advance.
            return np.zeros(self.block, dtype=np.int16)
        if self._fmt == "float32":
            data = np.frombuffer(raw, dtype=np.float32)
        else:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if self._channels > 1:
            usable = (data.size // self._channels) * self._channels
            data = data[:usable].reshape(-1, self._channels)
        arr = downmix_to_mono_float(data)
        if self._gain != 1.0:
            arr = arr * self._gain
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


def probe_capture_rms(
    source: str = "loopback",
    *,
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    input_device: str | int | None = None,
    loopback_device: str | int | None = None,
    output_device: str | int | None = None,
) -> dict[str, Any]:
    """Open capture briefly and return device info + RMS stats (for diagnostics)."""
    import time

    from krabobot_voice.vad import frame_rms

    with open_capture_stream(
        source,
        sample_rate=sample_rate,
        input_device=input_device,
        loopback_device=loopback_device,
        output_device=output_device,
    ) as stream:
        name = str(getattr(stream, "device_name", "") or "")
        idx = getattr(stream, "device_index", None)
        rmss: list[float] = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < max(0.2, float(duration_s)):
            pcm = stream.read_block()
            rmss.append(frame_rms(pcm))
    peak = max(rmss) if rmss else 0.0
    mean = (sum(rmss) / len(rmss)) if rmss else 0.0
    return {
        "source": (source or "mic").strip().lower(),
        "device_name": name,
        "device_index": idx,
        "blocks": len(rmss),
        "rms_mean": mean,
        "rms_max": peak,
        "ok": peak >= 0.002,
    }


def open_capture_stream(
    source: str = "mic",
    *,
    sample_rate: int = 16000,
    block_ms: int = 30,
    input_device: str | int | None = None,
    loopback_device: str | int | None = None,
    output_device: str | int | None = None,
    loopback_gain: float = DEFAULT_LOOPBACK_GAIN,
) -> MicStream | LoopbackStream:
    """Open mic or WASAPI loopback with the same ``read_block`` contract.

    ``source``: ``mic`` (default) | ``loopback``.
    Loopback reuses the same PyAudioWPatch path as meeting capture.
    """
    mode = (source or "mic").strip().lower()
    if mode == "loopback":
        return LoopbackStream(
            sample_rate=sample_rate,
            block_ms=block_ms,
            device=loopback_device,
            output_device=output_device,
            gain=loopback_gain,
        )
    return MicStream(
        sample_rate=sample_rate,
        block_ms=block_ms,
        device=input_device,
    )
