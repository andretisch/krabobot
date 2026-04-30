"""VK (VKontakte) channel implementation using vkbottle."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from pydantic import Field

from krabobot.bus.events import OutboundMessage
from krabobot.bus.queue import MessageBus
from krabobot.channels.base import BaseChannel
from krabobot.config.paths import get_media_dir
from krabobot.config.schema import Base
from krabobot.utils.ffmpeg import resolve_ffmpeg_exe

VKBOTTLE_AVAILABLE = importlib.util.find_spec("vkbottle") is not None
if VKBOTTLE_AVAILABLE:
    from vkbottle.bot import Bot, Message


class VKConfig(Base):
    """VK channel configuration."""

    enabled: bool = False
    token: str = ""
    allow_from: list[str] = Field(default_factory=list, alias="allowFrom")
    reaction_id: int = Field(default=10, alias="reactionId")
    access_denied_message: str = Field(
        default="Ваш ID: {id}. Этот пользователь не в доверенных. Обратитесь к администратору бота.",
        alias="accessDeniedMessage",
    )
    tts_enabled: bool = False
    transcribe_voice: bool = True
    transcribe_audio: bool = False


class VKChannel(BaseChannel):
    """VK long-poll channel."""

    name = "vk"
    display_name = "VK"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return VKConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = VKConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: VKConfig = config
        self.bot: Bot | None = None

    async def _download_media(self, url: str, ext: str = ".bin") -> str | None:
        """Download media and store it under media/vk."""
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                media_dir = get_media_dir("vk")
                fd, path = tempfile.mkstemp(suffix=ext, prefix="vk_media_", dir=str(media_dir))
                with os.fdopen(fd, "wb") as f:
                    f.write(resp.content)
                return path
        except Exception as e:
            logger.warning("VK media download failed: {}", e)
            return None

    async def _extract_attachments(self, message: Message) -> tuple[list[str], list[str], list[str]]:
        """Extract attachments and split voice vs generic audio paths."""
        media: list[str] = []
        voice_paths: list[str] = []
        audio_paths: list[str] = []
        for att in getattr(message, "attachments", []) or []:
            photo = getattr(att, "photo", None)
            if photo and getattr(photo, "sizes", None):
                sizes = sorted(photo.sizes, key=lambda s: (getattr(s, "width", 0) * getattr(s, "height", 0)))
                if sizes:
                    path = await self._download_media(getattr(sizes[-1], "url", ""), ext=".jpg")
                    if path:
                        media.append(path)
                continue

            doc = getattr(att, "doc", None)
            doc_url = getattr(doc, "url", None) if doc else None
            if doc_url:
                title = getattr(doc, "title", "") or ""
                ext = os.path.splitext(title)[1] or ".bin"
                path = await self._download_media(doc_url, ext=ext)
                if path:
                    media.append(path)
                    if ext.lower() in {".ogg", ".mp3", ".wav", ".m4a", ".aac"}:
                        audio_paths.append(path)
                continue

            # VK voice messages are delivered as attachment type "audio_message".
            audio_message = getattr(att, "audio_message", None)
            if audio_message:
                ogg_url = getattr(audio_message, "link_ogg", None)
                mp3_url = getattr(audio_message, "link_mp3", None)
                audio_url = ogg_url or mp3_url
                ext = ".ogg" if ogg_url else ".mp3"
                if audio_url:
                    path = await self._download_media(audio_url, ext=ext)
                    if path:
                        media.append(path)
                        voice_paths.append(path)
        return media, voice_paths, audio_paths

    async def _upload_doc_attachment(self, peer_id: int, file_path: str) -> str | None:
        """Upload file as VK document and return attachment token."""
        if not self.bot:
            return None
        try:
            upload_server = await self.bot.api.request(
                "docs.getMessagesUploadServer",
                {"peer_id": peer_id, "type": "doc"},
            )
            upload_url = self._extract_upload_url(upload_server)
            if not upload_url:
                return None

            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
                async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                    upload_resp = await client.post(upload_url, files=files)
                    upload_resp.raise_for_status()
            uploaded = upload_resp.json()
            file_token = uploaded.get("file") if isinstance(uploaded, dict) else None
            if not file_token:
                return None

            saved = await self.bot.api.request(
                "docs.save",
                {"file": file_token, "title": os.path.basename(file_path)},
            )
            doc = self._extract_saved_doc(saved)
            if not doc:
                return None
            owner_id = doc.get("owner_id")
            doc_id = doc.get("id")
            access_key = doc.get("access_key")
            if owner_id is None or doc_id is None:
                return None
            return f"doc{owner_id}_{doc_id}" + (f"_{access_key}" if access_key else "")
        except Exception as e:
            logger.warning("VK doc upload failed ({}): {}", file_path, e)
            return None

    async def _upload_voice_attachment(self, peer_id: int, file_path: str) -> tuple[str | None, str | None]:
        """Upload file as VK audio_message and return attachment token + error."""
        if not self.bot:
            return None, "bot not initialized"
        try:
            upload_server = await self.bot.api.request(
                "docs.getMessagesUploadServer",
                {"peer_id": peer_id, "type": "audio_message"},
            )
            upload_url = self._extract_upload_url(upload_server)
            if not upload_url:
                return None, "docs.getMessagesUploadServer(type=audio_message) returned no upload_url"

            voice_path = await self._ensure_voice_ogg(file_path, speed=1.0)
            with open(voice_path, "rb") as f:
                files = {"file": (os.path.basename(voice_path), f, "audio/ogg")}
                async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                    upload_resp = await client.post(upload_url, files=files)
                    upload_resp.raise_for_status()
            uploaded = upload_resp.json()
            file_token = uploaded.get("file") if isinstance(uploaded, dict) else None
            if not file_token:
                return None, f"audio_message upload response missing file token: {uploaded}"

            saved = await self.bot.api.request("docs.save", {"file": file_token})
            doc = self._extract_saved_doc(saved)
            if not doc:
                return None, f"docs.save returned empty payload: {saved}"
            owner_id = doc.get("owner_id")
            doc_id = doc.get("id")
            access_key = doc.get("access_key")
            if owner_id is None or doc_id is None:
                return None, f"docs.save returned incomplete identifiers: {doc}"
            return f"doc{owner_id}_{doc_id}" + (f"_{access_key}" if access_key else ""), None
        except Exception as e:
            logger.warning("VK voice upload failed ({}): {}", file_path, e)
            return None, str(e)

    @staticmethod
    def _extract_saved_doc(saved: Any) -> dict[str, Any] | None:
        """Normalize docs.save response into a single doc payload."""
        if isinstance(saved, dict) and "response" in saved:
            saved = saved.get("response")
        if isinstance(saved, list) and saved:
            first = saved[0]
            if isinstance(first, dict):
                return first
        if isinstance(saved, dict):
            if isinstance(saved.get("doc"), dict):
                return saved["doc"]
            if isinstance(saved.get("audio_message"), dict):
                return saved["audio_message"]
        return None

    @staticmethod
    def _extract_upload_url(payload: Any) -> str | None:
        """Extract upload_url from VK API response shapes."""
        if isinstance(payload, dict):
            if isinstance(payload.get("upload_url"), str):
                return payload["upload_url"]
            response = payload.get("response")
            if isinstance(response, dict) and isinstance(response.get("upload_url"), str):
                return response["upload_url"]
        return None

    async def _ensure_voice_ogg(self, file_path: str, *, speed: float = 1.0) -> str:
        """Convert audio file to ogg/opus for VK voice notes, with optional speedup."""
        in_path = Path(file_path)
        if in_path.suffix.lower() == ".ogg" and abs(speed - 1.0) < 0.001:
            return file_path
        ffmpeg = resolve_ffmpeg_exe()
        if not ffmpeg:
            logger.warning("ffmpeg not found, trying to upload non-ogg voice file: {}", file_path)
            return file_path
        fd, out_path = tempfile.mkstemp(prefix="vk_voice_", suffix=".ogg")
        os.close(fd)

        def _convert() -> None:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(in_path),
                    "-ac",
                    "1",
                    "-c:a",
                    "libopus",
                    "-b:a",
                    "32k",
                    "-filter:a",
                    f"atempo={max(0.5, min(2.0, float(speed))):.2f}",
                    out_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        try:
            await asyncio.to_thread(_convert)
            return out_path
        except Exception as exc:
            logger.warning("Failed to convert {} to ogg/opus for VK voice: {}", file_path, exc)
            try:
                os.unlink(out_path)
            except Exception:
                pass
            return file_path

    @staticmethod
    def _vk_plain_text(text: str) -> str:
        """Best-effort markdown cleanup for VK plain-text rendering."""
        out = text or ""
        out = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r"\1 (\2)", out)
        out = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).replace("```", ""), out)
        out = re.sub(r"`([^`]+)`", r"\1", out)
        out = re.sub(r"(\*\*|__)(.*?)\1", r"\2", out)
        out = re.sub(r"(\*|_)(.*?)\1", r"\2", out)
        out = re.sub(r"~~(.*?)~~", r"\1", out)
        return out.strip()

    @staticmethod
    def _vk_commands_keyboard() -> str:
        """Build VK keyboard with common slash commands."""
        payload = {
            "one_time": False,
            "inline": False,
            "buttons": [
                [
                    {"action": {"type": "text", "label": "/help", "payload": "{}"}, "color": "primary"},
                    {"action": {"type": "text", "label": "/id", "payload": "{}"}, "color": "secondary"},
                ],
                [
                    {"action": {"type": "text", "label": "/stop", "payload": "{}"}, "color": "negative"},
                    {"action": {"type": "text", "label": "/restart", "payload": "{}"}, "color": "negative"},
                ],
                [
                    {"action": {"type": "text", "label": "/link", "payload": "{}"}, "color": "primary"},
                    {"action": {"type": "text", "label": "/status", "payload": "{}"}, "color": "secondary"},
                ],
                [
                    {"action": {"type": "text", "label": "/tts status", "payload": "{}"}, "color": "secondary"},
                    {"action": {"type": "text", "label": "/new", "payload": "{}"}, "color": "negative"},
                ],
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    async def _upload_photo_attachment(self, peer_id: int, file_path: str) -> str | None:
        """Upload file as VK message photo and return attachment token."""
        if not self.bot:
            return None
        try:
            upload_server = await self.bot.api.request(
                "photos.getMessagesUploadServer",
                {"peer_id": peer_id},
            )
            upload_url = (upload_server or {}).get("upload_url")
            if not upload_url:
                return None

            with open(file_path, "rb") as f:
                files = {"photo": (os.path.basename(file_path), f, "application/octet-stream")}
                async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                    upload_resp = await client.post(upload_url, files=files)
                    upload_resp.raise_for_status()
            uploaded = upload_resp.json()
            if not isinstance(uploaded, dict):
                return None

            saved = await self.bot.api.request(
                "photos.saveMessagesPhoto",
                {
                    "server": uploaded.get("server"),
                    "photo": uploaded.get("photo"),
                    "hash": uploaded.get("hash"),
                },
            )
            photos = saved if isinstance(saved, list) else []
            if not photos:
                return None
            photo = photos[0]
            owner_id = photo.get("owner_id")
            photo_id = photo.get("id")
            access_key = photo.get("access_key")
            if owner_id is None or photo_id is None:
                return None
            return f"photo{owner_id}_{photo_id}" + (f"_{access_key}" if access_key else "")
        except Exception as e:
            logger.warning("VK photo upload failed ({}): {}", file_path, e)
            return None

    async def start(self) -> None:
        if not VKBOTTLE_AVAILABLE:
            logger.error("vkbottle not installed. Run: pip install vkbottle")
            return
        if not self.config.token:
            logger.error("VK token not configured")
            return

        self._running = True
        self.bot = Bot(token=self.config.token)

        @self.bot.on.message()
        async def _on_message(message: Message) -> None:
            if not self._running:
                return

            sender_id = str(getattr(message, "from_id", ""))
            chat_id = str(getattr(message, "peer_id", ""))

            if not self.is_allowed(sender_id):
                try:
                    deny_text = self.config.access_denied_message
                    if "{id}" in deny_text:
                        deny_text = deny_text.replace("{id}", sender_id)
                    await self.bot.api.messages.send(
                        peer_id=int(chat_id),
                        message=deny_text,
                        random_id=0,
                    )
                except Exception:
                    pass
                return

            content = getattr(message, "text", "") or ""
            media, voice_paths, audio_paths = await self._extract_attachments(message)
            selected_for_stt: str | None = None
            if self.config.transcribe_voice and voice_paths:
                selected_for_stt = voice_paths[0]
            elif self.config.transcribe_audio and audio_paths:
                selected_for_stt = audio_paths[0]
            if selected_for_stt:
                transcription = await self.transcribe_audio(selected_for_stt)
                if transcription:
                    content = (f"{content}\n" if content else "") + f"[transcription: {transcription}]"
                else:
                    stt_error = self.consume_last_stt_error()
                    if stt_error:
                        content = (f"{content}\n" if content else "") + f"[transcription_error: {stt_error}]"
                    else:
                        content = (f"{content}\n" if content else "") + f"[voice: {selected_for_stt}]"

            reply = getattr(message, "reply_message", None)
            reply_text = (getattr(reply, "text", "") or "").strip() if reply else ""
            if reply_text:
                short = reply_text[:100] + ("..." if len(reply_text) > 100 else "")
                content = f"[Reply to: {short}]\n{content}" if content else f"[Reply to: {short}]"

            if not content and not media:
                content = "[empty message]"

            async def _typing_and_reaction() -> None:
                try:
                    if self.config.reaction_id > 0 and getattr(message, "conversation_message_id", None):
                        await self.bot.api.request(
                            "messages.sendReaction",
                            {
                                "peer_id": int(chat_id),
                                "cmid": getattr(message, "conversation_message_id"),
                                "reaction_id": self.config.reaction_id,
                            },
                        )
                except Exception:
                    pass
                try:
                    await self.bot.api.messages.set_activity(peer_id=int(chat_id), type="typing")
                except Exception:
                    pass

            asyncio.create_task(_typing_and_reaction())

            await self._handle_message(
                sender_id=sender_id,
                chat_id=chat_id,
                content=content,
                media=media,
                metadata={
                    "message_id": getattr(message, "id", None),
                    "conversation_message_id": getattr(message, "conversation_message_id", None),
                },
            )

        # vkbottle requires awaiting run_polling() inside an active event loop.
        await self.bot.run_polling()

    async def stop(self) -> None:
        self._running = False
        if self.bot and getattr(self.bot, "polling", None):
            try:
                self.bot.polling.stop()
            except Exception:
                pass

    async def send(self, msg: OutboundMessage) -> None:
        if not self._running or not self.bot:
            return
        peer_id = int(msg.chat_id)
        attachment_tokens: list[str] = []
        failed_media: list[str] = []
        failed_details: list[str] = []

        if not (msg.media or []):
            wants_tts = bool(msg.metadata.get("_tts_enabled_for_user", self.config.tts_enabled))
            if wants_tts and not bool(msg.metadata.get("_skip_tts")) and msg.content and msg.content != "[empty message]":
                tts_path = await self.synthesize_speech(self._vk_plain_text(msg.content))
                if tts_path:
                    msg.media = [*msg.media, tts_path]

        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
        for media_path in msg.media or []:
            path = media_path
            if media_path.startswith(("http://", "https://")):
                guessed_ext = os.path.splitext(media_path.split("?", 1)[0])[1] or ".bin"
                downloaded = await self._download_media(media_path, ext=guessed_ext)
                if not downloaded:
                    failed_media.append(os.path.basename(media_path))
                    failed_details.append(f"{os.path.basename(media_path)}: download failed")
                    continue
                path = downloaded
            if not os.path.exists(path):
                logger.warning("VK send: media path does not exist: {}", path)
                failed_media.append(os.path.basename(path))
                failed_details.append(f"{os.path.basename(path)}: file does not exist")
                continue

            ext = os.path.splitext(path)[1].lower()
            if ext in image_exts:
                token = await self._upload_photo_attachment(peer_id, path)
                if not token:
                    failed_details.append(f"{os.path.basename(path)}: photos upload failed")
            else:
                token = None
                # Try true voice-note upload first for audio files.
                if ext in {".ogg", ".mp3", ".wav", ".m4a", ".aac"}:
                    token, err = await self._upload_voice_attachment(peer_id, path)
                    if not token and err:
                        failed_details.append(f"{os.path.basename(path)}: audio_message failed ({err})")
                if not token:
                    token = await self._upload_doc_attachment(peer_id, path)
                    if not token:
                        failed_details.append(f"{os.path.basename(path)}: docs upload failed")
            if token:
                attachment_tokens.append(token)
            else:
                failed_media.append(os.path.basename(path))

        text = self._vk_plain_text(msg.content or (" " if attachment_tokens else ""))
        if failed_media:
            failed_list = ", ".join(dict.fromkeys(failed_media))
            details = "; ".join(dict.fromkeys(failed_details))
            hint = (
                f"\n\n[VK] Не удалось прикрепить файл(ы): {failed_list}. "
                "Проверьте права community token (docs/files) и доступ к audio_message."
            )
            if details:
                hint += f"\n[VK debug] {details}"
            text = (text or "").strip() + hint

        await self.bot.api.messages.send(
            peer_id=peer_id,
            message=text,
            attachment=",".join(attachment_tokens) if attachment_tokens else None,
            keyboard=self._vk_commands_keyboard(),
            random_id=0,
        )
