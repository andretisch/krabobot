"""HTTP client for POST /v1/voice/turn."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx

from krabobot_voice.config import VoiceClientConfig


@dataclass
class VoiceTurnResult:
    """Response from a voice turn."""

    status_code: int
    audio_wav: bytes | None
    transcript: str
    reply: str
    device_id: str
    json_body: dict[str, Any] | None = None
    error_message: str = ""


class VoiceHttpClient:
    """Calls krabobot ``POST /v1/voice/turn`` (server STT/TTS)."""

    def __init__(self, config: VoiceClientConfig | None = None) -> None:
        self.config = config or VoiceClientConfig.from_env()

    def turn(
        self,
        *,
        audio_path: str | Path | None = None,
        audio_bytes: bytes | None = None,
        audio_filename: str = "clip.wav",
        instruct: str | None = None,
        files: list[str | Path] | None = None,
        device_id: str | None = None,
    ) -> VoiceTurnResult:
        """Send one turn. Provide audio and/or instruct (and optional files)."""
        did = (device_id or self.config.device_id).strip()
        if not did:
            raise ValueError("device_id is required")

        multipart: list[tuple[str, Any]] = [("device_id", (None, did))]
        if instruct:
            multipart.append(("instruct", (None, instruct)))

        if audio_bytes is not None:
            multipart.append(("audio", (audio_filename, audio_bytes, "audio/wav")))
        elif audio_path is not None:
            path = Path(audio_path)
            multipart.append(
                ("audio", (path.name or audio_filename, path.read_bytes(), "audio/wav"))
            )

        for f in files or []:
            fp = Path(f)
            multipart.append(("files", (fp.name, fp.read_bytes(), "application/octet-stream")))

        headers = {}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"

        url = f"{self.config.base_url}/v1/voice/turn"
        with httpx.Client(timeout=self.config.timeout_s) as client:
            resp = client.post(url, files=multipart, headers=headers)

        transcript = unquote(resp.headers.get("X-Krabobot-Transcript") or "")
        reply = unquote(resp.headers.get("X-Krabobot-Reply") or "")
        ctype = (resp.headers.get("Content-Type") or "").lower()
        json_body = None
        audio_wav = None
        err = ""
        if "audio/wav" in ctype or "audio/wave" in ctype:
            audio_wav = resp.content
        else:
            try:
                json_body = resp.json()
            except Exception:
                json_body = None
            if isinstance(json_body, dict):
                transcript = str(json_body.get("transcript") or transcript)
                reply = str(json_body.get("reply") or reply)
                err_obj = json_body.get("error")
                if isinstance(err_obj, dict):
                    err = str(err_obj.get("message") or "")
                elif resp.status_code >= 400:
                    err = str(json_body.get("message") or resp.text[:300])
            elif resp.status_code >= 400:
                err = resp.text[:300]

        return VoiceTurnResult(
            status_code=resp.status_code,
            audio_wav=audio_wav,
            transcript=transcript,
            reply=reply,
            device_id=did,
            json_body=json_body,
            error_message=err,
        )
