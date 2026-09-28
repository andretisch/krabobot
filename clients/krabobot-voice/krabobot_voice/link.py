"""Ensure voice device_id is linked to the owner via admin API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from krabobot_voice.config import VoiceClientConfig


@dataclass
class LinkResult:
    ok: bool
    user_id: str
    device_id: str
    already_linked: bool
    message: str


def ensure_device_linked(config: VoiceClientConfig) -> LinkResult:
    """GET users, find owner, POST link channel=voice if needed."""
    did = (config.device_id or "").strip()
    if not did:
        return LinkResult(False, "", "", False, "device_id is empty")
    if not config.token:
        return LinkResult(
            False,
            "",
            did,
            False,
            "Нет токена: задайте KRABOBOT_TOKEN / token в config.yaml "
            "или api.auth.adminToken в ~/.krabobot/config.json",
        )

    headers = {"Authorization": f"Bearer {config.token}"}
    url_users = f"{config.base_url}/v1/web/users"

    try:
        with httpx.Client(timeout=min(30.0, config.timeout_s)) as client:
            resp = client.get(url_users, headers=headers)
            if resp.status_code == 401:
                return LinkResult(
                    False,
                    "",
                    did,
                    False,
                    "401 Unauthorized — проверьте adminToken / Bearer token",
                )
            if resp.status_code >= 400:
                return LinkResult(
                    False,
                    "",
                    did,
                    False,
                    f"GET /v1/web/users failed: HTTP {resp.status_code} {resp.text[:200]}",
                )
            body = resp.json()
            users = body.get("data") if isinstance(body, dict) else None
            if not isinstance(users, list) or not users:
                return LinkResult(
                    False,
                    "",
                    did,
                    False,
                    "Нет пользователей на сервере — сначала создайте owner "
                    "(вход в веб-UI или любой канал).",
                )

            owner = next((u for u in users if u.get("is_owner")), None)
            target: dict[str, Any] = owner if isinstance(owner, dict) else users[0]
            user_id = str(target.get("user_id") or "").strip()
            if not user_id:
                return LinkResult(False, "", did, False, "Не удалось определить user_id владельца")

            accounts = [str(a) for a in (target.get("accounts") or [])]
            key = f"voice:{did}"
            if key in accounts:
                return LinkResult(
                    True,
                    user_id,
                    did,
                    True,
                    f"device_id={did} уже привязан к {user_id}",
                )

            link_url = f"{config.base_url}/v1/web/users/{user_id}/links"
            link_resp = client.post(
                link_url,
                headers={**headers, "Content-Type": "application/json"},
                json={"channel": "voice", "sender_id": did},
            )
            if link_resp.status_code >= 400:
                return LinkResult(
                    False,
                    user_id,
                    did,
                    False,
                    f"POST link failed: HTTP {link_resp.status_code} {link_resp.text[:200]}",
                )
            return LinkResult(
                True,
                user_id,
                did,
                False,
                f"device_id={did} привязан к owner {user_id}",
            )
    except httpx.ConnectError as e:
        return LinkResult(
            False,
            "",
            did,
            False,
            f"Не удалось подключиться к {config.base_url}: {e}. "
            "Запустите `krabobot serve` (порт 8900).",
        )
    except Exception as e:
        return LinkResult(False, "", did, False, f"Ошибка привязки: {e}")
