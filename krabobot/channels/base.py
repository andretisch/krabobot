"""Base channel interface for chat platforms."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
import tempfile
import os
from pathlib import Path
from typing import Any

from loguru import logger

from krabobot.bus.events import InboundMessage, OutboundMessage
from krabobot.bus.queue import MessageBus
from krabobot.config.schema import STTConfig, TTSConfig


class BaseChannel(ABC):
    """
    Abstract base class for chat channel implementations.

    Each channel (Telegram, Discord, etc.) should implement this interface
    to integrate with the krabobot message bus.
    """

    name: str = "base"
    display_name: str = "Base"

    def __init__(self, config: Any, bus: MessageBus):
        """
        Initialize the channel.

        Args:
            config: Channel-specific configuration.
            bus: The message bus for communication.
        """
        self.config = config
        self.bus = bus
        self._running = False
        self._last_stt_error: str | None = None
        self._tts_config: TTSConfig | None = None
        self._stt_config: STTConfig | None = None

    def set_tts_config(self, cfg: TTSConfig) -> None:
        """Attach global TTS settings from root config."""
        self._tts_config = cfg

    def set_stt_config(self, cfg: STTConfig) -> None:
        """Attach global STT settings from root config."""
        self._stt_config = cfg

    async def transcribe_audio_with_error(self, file_path: str | Path) -> tuple[str, str | None]:
        """Transcribe audio and return ``(text, error)`` for user-facing diagnostics."""
        self._last_stt_error = None
        path = Path(file_path)
        try:
            from krabobot.stt.sherpa_onnx_stt import SherpaOnnxTranscriber

            model_dir = self._resolve_sherpa_stt_model_dir()
            num_threads = self._resolve_sherpa_stt_num_threads()
            backend_provider = self._resolve_sherpa_stt_provider()
            text = await asyncio.to_thread(
                SherpaOnnxTranscriber.transcribe,
                path,
                model_dir=model_dir,
                num_threads=num_threads,
                provider=backend_provider,
            )
            if text:
                return text, None
            self._last_stt_error = "sherpa_onnx returned empty transcription"
            return "", self._last_stt_error
        except Exception as e:
            logger.warning("{}: sherpa-onnx transcription failed: {}", self.name, e)
            self._last_stt_error = f"sherpa_onnx failed: {e}"
            return "", self._last_stt_error

    def _resolve_sherpa_stt_model_dir(self) -> str:
        """Resolve sherpa-onnx STT model directory from config/env."""
        if self._stt_config:
            base = Path(self._stt_config.sherpa_models_dir).expanduser().resolve()
            model_id = (self._stt_config.sherpa_model_id or "").strip()
            model_name = model_id.split("/", 1)[-1] if "/" in model_id else model_id
            if model_name:
                return str((base / model_name).resolve())
            return str(base.resolve())
        default_dir = str((Path.home() / ".krabobot" / "models" / "stt" / "sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16").resolve())
        return (os.getenv("SHERPA_STT_MODEL_DIR", default_dir) or "").strip()

    def _resolve_sherpa_stt_num_threads(self) -> int:
        if self._stt_config:
            return int(max(1, self._stt_config.sherpa_num_threads))
        raw = (os.getenv("SHERPA_STT_NUM_THREADS", "2") or "2").strip()
        try:
            return max(1, int(raw))
        except ValueError:
            return 2

    def _resolve_sherpa_stt_provider(self) -> str:
        if self._stt_config and self._stt_config.sherpa_provider:
            return str(self._stt_config.sherpa_provider).strip() or "cpu"
        return (os.getenv("SHERPA_STT_PROVIDER", "cpu") or "cpu").strip()

    async def transcribe_audio(self, file_path: str | Path) -> str:
        """Backward-compatible wrapper that returns only text."""
        text, _ = await self.transcribe_audio_with_error(file_path)
        return text

    def consume_last_stt_error(self) -> str | None:
        """Read and clear the last STT error message."""
        err = self._last_stt_error
        self._last_stt_error = None
        return err

    async def synthesize_speech(
        self, text: str
    ) -> str | None:
        """Synthesize speech with configured backend and return local audio path."""
        clean_text = (text or "").strip()
        if not clean_text:
            return None
        provider = (
            (self._tts_config.provider if self._tts_config else os.getenv("TTS_PROVIDER", "gtts"))
            or "gtts"
        ).strip().lower()
        if provider in {"sherpa", "sherpa_onnx", "sherpa-onnx"}:
            path = await self._synthesize_sherpa_onnx(clean_text)
            if path:
                return path
            logger.warning("{}: sherpa-onnx TTS failed, falling back to gTTS", self.name)
        try:
            from gtts import gTTS

            lang = (
                self._tts_config.language
                if self._tts_config and self._tts_config.language
                else os.getenv("TTS_LANG", "ru")
            )
            lang = (lang or "ru").strip()
            fd, out_path = tempfile.mkstemp(prefix=f"{self.name}_tts_", suffix=".mp3")
            os.close(fd)
            await asyncio.to_thread(gTTS(clean_text, lang=lang).save, out_path)
            return out_path
        except ImportError:
            logger.warning("{}: gTTS is not installed", self.name)
            return None
        except Exception as e:
            logger.warning("{}: gTTS synthesis failed: {}", self.name, e)
            return None

    async def _synthesize_sherpa_onnx(self, clean_text: str) -> str | None:
        """Synthesize speech via sherpa-onnx VITS model directory."""
        from krabobot.tts.sherpa_onnx_tts import SherpaOnnxTTS

        model_dir = self._resolve_sherpa_tts_model_dir()
        if not model_dir:
            logger.warning("{}: sherpa-onnx model dir is not configured", self.name)
            return None
        speed = self._sherpa_tts_speed()
        fd, out_path = tempfile.mkstemp(prefix=f"{self.name}_tts_", suffix=".wav")
        os.close(fd)
        try:
            await asyncio.to_thread(
                SherpaOnnxTTS.synthesize_to_wav,
                text=clean_text,
                model_dir=model_dir,
                out_path=out_path,
                speed=speed,
                sid=0,
            )
            return out_path
        except ImportError:
            logger.warning("{}: sherpa-onnx is not installed", self.name)
            return None
        except Exception as e:
            logger.warning("{}: sherpa-onnx TTS failed: {}", self.name, e)
            return None

    def _resolve_sherpa_tts_model_dir(self) -> str:
        """Resolve sherpa-onnx TTS model directory from config/env."""
        if self._tts_config:
            base = Path(self._tts_config.sherpa_models_dir).expanduser().resolve()
            model_id = (self._tts_config.sherpa_model_id or "").strip()
            model_name = model_id.split("/", 1)[-1] if "/" in model_id else model_id
            if model_name:
                return str((base / model_name).resolve())
            return str(base.resolve())
        default_dir = str((Path.home() / ".krabobot" / "models" / "tts" / "vits-piper-ru_RU-irina-medium").resolve())
        return (os.getenv("SHERPA_TTS_MODEL_DIR", default_dir) or "").strip()

    def _sherpa_tts_speed(self) -> float:
        if self._tts_config:
            return max(0.5, min(2.0, float(self._tts_config.sherpa_speed)))
        raw = (os.getenv("SHERPA_TTS_SPEED", "1.0") or "1.0").strip()
        try:
            val = float(raw)
        except ValueError:
            return 1.0
        return max(0.5, min(2.0, val))

    async def login(self, force: bool = False) -> bool:
        """
        Perform channel-specific interactive login (e.g. QR code scan).

        Args:
            force: If True, ignore existing credentials and force re-authentication.

        Returns True if already authenticated or login succeeds.
        Override in subclasses that support interactive login.
        """
        return True

    @abstractmethod
    async def start(self) -> None:
        """
        Start the channel and begin listening for messages.

        This should be a long-running async task that:
        1. Connects to the chat platform
        2. Listens for incoming messages
        3. Forwards messages to the bus via _handle_message()
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel and clean up resources."""
        pass

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """
        Send a message through this channel.

        Args:
            msg: The message to send.

        Implementations should raise on delivery failure so the channel manager
        can apply any retry policy in one place.
        """
        pass

    async def send_delta(self, chat_id: str, delta: str, metadata: dict[str, Any] | None = None) -> None:
        """Deliver a streaming text chunk.

        Override in subclasses to enable streaming. Implementations should
        raise on delivery failure so the channel manager can retry.

        Streaming contract: ``_stream_delta`` is a chunk, ``_stream_end`` ends
        the current segment, and stateful implementations must key buffers by
        ``_stream_id`` rather than only by ``chat_id``.
        """
        pass

    @property
    def supports_streaming(self) -> bool:
        """True when config enables streaming AND this subclass implements send_delta."""
        cfg = self.config
        streaming = cfg.get("streaming", False) if isinstance(cfg, dict) else getattr(cfg, "streaming", False)
        return bool(streaming) and type(self).send_delta is not BaseChannel.send_delta

    def is_allowed(self, sender_id: str) -> bool:
        """Check if *sender_id* is permitted.  Empty list → deny all; ``"*"`` → allow all."""
        allow_list = getattr(self.config, "allow_from", [])
        if not allow_list:
            logger.warning("{}: allow_from is empty — all access denied", self.name)
            return False
        if "*" in allow_list:
            return True
        return str(sender_id) in allow_list

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
    ) -> None:
        """
        Handle an incoming message from the chat platform.

        This method checks permissions and forwards to the bus.

        Args:
            sender_id: The sender's identifier.
            chat_id: The chat/channel identifier.
            content: Message text content.
            media: Optional list of media URLs.
            metadata: Optional channel-specific metadata.
            session_key: Optional session key override (e.g. thread-scoped sessions).
        """
        if not self.is_allowed(sender_id):
            logger.warning(
                "Access denied for sender {} on channel {}. "
                "Add them to allowFrom list in config to grant access.",
                sender_id, self.name,
            )
            return

        meta = metadata or {}
        if self.supports_streaming:
            meta = {**meta, "_wants_stream": True}

        msg = InboundMessage(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=content,
            media=media or [],
            metadata=meta,
            session_key_override=session_key,
        )

        await self.bus.publish_inbound(msg)

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """Return default config for onboard auto-population in config.json."""
        return {"enabled": False}

    @property
    def is_running(self) -> bool:
        """Check if the channel is running."""
        return self._running
