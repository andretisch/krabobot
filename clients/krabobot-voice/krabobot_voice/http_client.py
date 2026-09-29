"""HTTP client for POST /v1/voice/turn."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx

from krabobot_voice.config import VoiceClientConfig
from krabobot_voice.protocol import ClientState, parse_actions


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
    actions: list[str] = field(default_factory=list)
    queued: bool = False


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
        client_state: ClientState | str | dict[str, Any] | None = None,
        async_meeting: bool = False,
        timeout_s: float | None = None,
    ) -> VoiceTurnResult:
        """Send one turn. Provide audio and/or instruct (and optional files)."""
        did = (device_id or self.config.device_id).strip()
        if not did:
            raise ValueError("device_id is required")

        multipart: list[tuple[str, Any]] = [("device_id", (None, did))]
        if instruct:
            multipart.append(("instruct", (None, instruct)))

        state_json = _client_state_json(client_state)
        if state_json:
            multipart.append(("client_state", (None, state_json)))
        if async_meeting:
            multipart.append(("async", (None, "1")))

        if audio_bytes is not None:
            multipart.append(("audio", (audio_filename, audio_bytes, "audio/wav")))
        elif audio_path is not None:
            path = Path(audio_path)
            multipart.append(
                ("audio", (path.name or audio_filename, path.read_bytes(), "audio/wav"))
            )

        for f in files or []:
            fp = Path(f)
            mime = "application/octet-stream"
            suffix = fp.suffix.lower()
            if suffix in {".wav"}:
                mime = "audio/wav"
            elif suffix in {".mp3", ".ogg", ".m4a", ".flac", ".opus"}:
                mime = f"audio/{suffix.lstrip('.')}"
            elif suffix in {".mp4", ".webm", ".mov", ".mkv"}:
                mime = "video/mp4" if suffix == ".mp4" else f"video/{suffix.lstrip('.')}"
            multipart.append(("files", (fp.name, fp.read_bytes(), mime)))

        headers = {}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"

        url = f"{self.config.base_url}/v1/voice/turn"
        req_timeout = self.config.timeout_s if timeout_s is None else float(timeout_s)
        with httpx.Client(timeout=req_timeout) as client:
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

        actions = parse_actions(resp.headers, json_body if isinstance(json_body, dict) else None)
        queued = resp.status_code == 202
        if isinstance(json_body, dict) and json_body.get("status") == "queued":
            queued = True

        return VoiceTurnResult(
            status_code=resp.status_code,
            audio_wav=audio_wav,
            transcript=transcript,
            reply=reply,
            device_id=did,
            json_body=json_body,
            error_message=err,
            actions=actions,
            queued=queued,
        )


def _client_state_json(
    client_state: ClientState | str | dict[str, Any] | None,
) -> str:
    if client_state is None:
        return ""
    if isinstance(client_state, ClientState):
        return client_state.to_json()
    if isinstance(client_state, str):
        return client_state.strip()
    if isinstance(client_state, dict):
        import json

        return json.dumps(client_state, ensure_ascii=False, separators=(",", ":"))
    return ""
