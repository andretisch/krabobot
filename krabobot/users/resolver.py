"""Resolve internal user identities across channels."""

from __future__ import annotations

import asyncio
import json
import secrets
import string
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from krabobot.utils.helpers import ensure_dir


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _account_key(channel: str, sender_id: str) -> str:
    return f"{channel}:{sender_id}"


@dataclass(slots=True)
class LinkConsumeResult:
    """Outcome of consuming a one-time link code."""

    ok: bool
    user_id: str | None = None
    error: str | None = None


@dataclass(slots=True)
class RegistrationRequest:
    """Pending self-registration request."""

    request_id: str
    channel: str
    sender_id: str
    note: str
    created_at: str


@dataclass(slots=True)
class RegistrationDecision:
    """Outcome of registration approval/rejection."""

    ok: bool
    user_id: str | None = None
    error: str | None = None


class UserResolver:
    """Thread-safe account resolver + link code storage."""

    def __init__(
        self,
        storage_dir: Path,
        *,
        code_ttl_seconds: int = 600,
        code_attempt_limit: int = 5,
    ) -> None:
        self._storage_dir = ensure_dir(storage_dir)
        self._path = self._storage_dir / "user_links.json"
        self._code_ttl_seconds = max(60, int(code_ttl_seconds))
        self._code_attempt_limit = max(1, int(code_attempt_limit))
        self._lock = asyncio.Lock()

    def user_workspace(self, users_root: Path, user_id: str) -> Path:
        """Return the workspace path for a resolved user id."""
        return ensure_dir(users_root / user_id)

    async def lookup(self, channel: str, sender_id: str) -> str | None:
        """Lookup mapped user id for a channel account."""
        account = _account_key(channel, sender_id)
        async with self._lock:
            db = self._load()
            return db.get("accounts", {}).get(account)

    async def resolve_or_create(self, channel: str, sender_id: str) -> str:
        """Resolve existing user id, or create a new one."""
        account = _account_key(channel, sender_id)
        async with self._lock:
            db = self._load()
            accounts = db.setdefault("accounts", {})
            if account in accounts:
                return str(accounts[account])
            user_id = uuid.uuid4().hex
            accounts[account] = user_id
            self._save(db)
            logger.info("Created user mapping {} -> {}", account, user_id)
            return user_id

    async def link_account(self, user_id: str, channel: str, sender_id: str) -> None:
        """Force-link a channel account to the specified internal user."""
        account = _account_key(channel, sender_id)
        async with self._lock:
            db = self._load()
            db.setdefault("accounts", {})[account] = user_id
            self._save(db)
            logger.info("Linked account {} to user {}", account, user_id)

    async def unlink_account(self, channel: str, sender_id: str) -> bool:
        """Remove account mapping if present (e.g. web chat session deleted)."""
        account = _account_key(channel, sender_id)
        async with self._lock:
            db = self._load()
            accounts = db.get("accounts", {}) or {}
            if account not in accounts:
                return False
            del accounts[account]
            db["accounts"] = accounts
            self._save(db)
            logger.info("Unlinked account {}", account)
            return True

    async def accounts_for_user(self, user_id: str) -> list[str]:
        """Return all account keys linked to a user id."""
        if not user_id:
            return []
        async with self._lock:
            db = self._load()
            accounts = db.get("accounts", {}) or {}
            linked = [k for k, v in accounts.items() if str(v) == str(user_id)]
        linked.sort()
        return linked

    async def ensure_owner(self, user_id: str) -> str:
        """Persist and return owner user id, assigning the first user as owner."""
        if not user_id:
            return ""
        async with self._lock:
            db = self._load()
            owner = str(db.get("owner_user_id") or "").strip()
            if owner:
                return owner
            db["owner_user_id"] = str(user_id)
            self._save(db)
            logger.info("Assigned first user as owner: {}", user_id)
            return str(user_id)

    async def is_registered(self, channel: str, sender_id: str) -> bool:
        """Check whether account is approved and linked to a user."""
        account = _account_key(channel, sender_id)
        async with self._lock:
            db = self._load()
            return bool((db.get("accounts", {}) or {}).get(account))

    async def create_registration_request(
        self,
        channel: str,
        sender_id: str,
        *,
        note: str = "",
    ) -> RegistrationRequest:
        """Create or replace pending registration request for account."""
        async with self._lock:
            db = self._load()
            reqs = db.setdefault("registration_requests", {})
            # Reuse existing pending request id for the same account.
            existing_id = ""
            for rid, payload in reqs.items():
                if (
                    isinstance(payload, dict)
                    and str(payload.get("channel", "")) == channel
                    and str(payload.get("sender_id", "")) == sender_id
                ):
                    existing_id = rid
                    break
            req_id = existing_id or uuid.uuid4().hex[:10].upper()
            reqs[req_id] = {
                "channel": channel,
                "sender_id": sender_id,
                "note": (note or "").strip(),
                "created_at": _now().isoformat(),
            }
            self._save(db)
            payload = reqs[req_id]
            return RegistrationRequest(
                request_id=req_id,
                channel=str(payload["channel"]),
                sender_id=str(payload["sender_id"]),
                note=str(payload.get("note", "")),
                created_at=str(payload["created_at"]),
            )

    async def list_registration_requests(self) -> list[RegistrationRequest]:
        """Return pending registration requests sorted by creation time."""
        async with self._lock:
            db = self._load()
            reqs = db.get("registration_requests", {}) or {}
        items: list[RegistrationRequest] = []
        for req_id, payload in reqs.items():
            if not isinstance(payload, dict):
                continue
            items.append(
                RegistrationRequest(
                    request_id=str(req_id),
                    channel=str(payload.get("channel", "")),
                    sender_id=str(payload.get("sender_id", "")),
                    note=str(payload.get("note", "")),
                    created_at=str(payload.get("created_at", "")),
                )
            )
        items.sort(key=lambda x: x.created_at)
        return items

    async def approve_registration(self, request_id: str) -> RegistrationDecision:
        """Approve pending request and create linked user account."""
        rid = (request_id or "").strip().upper()
        if not rid:
            return RegistrationDecision(ok=False, error="empty_request_id")
        async with self._lock:
            db = self._load()
            reqs = db.setdefault("registration_requests", {})
            payload = reqs.get(rid)
            if not isinstance(payload, dict):
                return RegistrationDecision(ok=False, error="request_not_found")
            channel = str(payload.get("channel", "")).strip()
            sender_id = str(payload.get("sender_id", "")).strip()
            if not channel or not sender_id:
                reqs.pop(rid, None)
                self._save(db)
                return RegistrationDecision(ok=False, error="invalid_request_payload")
            account = _account_key(channel, sender_id)
            accounts = db.setdefault("accounts", {})
            existing = accounts.get(account)
            if existing:
                reqs.pop(rid, None)
                self._save(db)
                return RegistrationDecision(ok=True, user_id=str(existing))
            user_id = uuid.uuid4().hex
            accounts[account] = user_id
            reqs.pop(rid, None)
            self._save(db)
            return RegistrationDecision(ok=True, user_id=user_id)

    async def reject_registration(self, request_id: str) -> RegistrationDecision:
        """Reject pending request and remove it from queue."""
        rid = (request_id or "").strip().upper()
        if not rid:
            return RegistrationDecision(ok=False, error="empty_request_id")
        async with self._lock:
            db = self._load()
            reqs = db.setdefault("registration_requests", {})
            if rid not in reqs:
                return RegistrationDecision(ok=False, error="request_not_found")
            reqs.pop(rid, None)
            self._save(db)
            return RegistrationDecision(ok=True)

    async def create_registration_code(self, owner_user_id: str, *, ttl_seconds: int = 3600) -> str:
        """Create one-time registration code for auto-approval."""
        if not owner_user_id:
            raise RuntimeError("owner_user_id is required")
        alphabet = string.ascii_uppercase + string.digits
        expires_at = (_now() + timedelta(seconds=max(60, int(ttl_seconds)))).isoformat()
        async with self._lock:
            db = self._load()
            codes = db.setdefault("registration_codes", {})
            for _ in range(20):
                code = "".join(secrets.choice(alphabet) for _ in range(8))
                if code in codes:
                    continue
                codes[code] = {"owner_user_id": owner_user_id, "expires_at": expires_at}
                self._save(db)
                return code
        raise RuntimeError("Failed to generate registration code")

    async def consume_registration_code(
        self,
        code: str,
        channel: str,
        sender_id: str,
    ) -> RegistrationDecision:
        """Consume one-time registration code and auto-approve account."""
        normalized = (code or "").strip().upper()
        if not normalized:
            return RegistrationDecision(ok=False, error="empty_code")
        account = _account_key(channel, sender_id)
        async with self._lock:
            db = self._load()
            codes = db.setdefault("registration_codes", {})
            payload = codes.get(normalized)
            if not isinstance(payload, dict):
                return RegistrationDecision(ok=False, error="invalid_code")
            expires_at = self._parse_ts(payload.get("expires_at"))
            if expires_at is None or expires_at <= _now():
                codes.pop(normalized, None)
                self._save(db)
                return RegistrationDecision(ok=False, error="expired_code")
            accounts = db.setdefault("accounts", {})
            user_id = str(accounts.get(account) or "").strip()
            if not user_id:
                user_id = uuid.uuid4().hex
                accounts[account] = user_id
            codes.pop(normalized, None)
            self._save(db)
            return RegistrationDecision(ok=True, user_id=user_id)

    async def get_owner_user_id(self) -> str | None:
        """Return current owner user id if present."""
        async with self._lock:
            db = self._load()
            owner = str(db.get("owner_user_id") or "").strip()
            return owner or None

    async def is_owner(self, user_id: str | None) -> bool:
        """Check whether the given user id is the configured owner."""
        if not user_id:
            return False
        async with self._lock:
            db = self._load()
            owner = str(db.get("owner_user_id") or "").strip()
            return bool(owner) and owner == str(user_id)

    async def get_tts_enabled(self, user_id: str, *, default: bool = False) -> bool:
        """Return per-user TTS preference."""
        if not user_id:
            return bool(default)
        async with self._lock:
            db = self._load()
            prefs = db.get("user_prefs", {}) or {}
            user_pref = prefs.get(str(user_id), {}) or {}
            raw = user_pref.get("tts_enabled")
            if raw is None:
                return bool(default)
            return bool(raw)

    async def set_tts_enabled(self, user_id: str, enabled: bool) -> None:
        """Persist per-user TTS preference."""
        if not user_id:
            return
        async with self._lock:
            db = self._load()
            prefs = db.setdefault("user_prefs", {})
            user_pref = prefs.setdefault(str(user_id), {})
            user_pref["tts_enabled"] = bool(enabled)
            self._save(db)

    async def get_display_name(self, user_id: str) -> str:
        """Return stored display name for a user (empty if unset)."""
        if not user_id:
            return ""
        async with self._lock:
            db = self._load()
            prefs = db.get("user_prefs", {}) or {}
            user_pref = prefs.get(str(user_id), {}) or {}
            return str(user_pref.get("display_name") or "").strip()

    async def set_display_name(self, user_id: str, display_name: str) -> None:
        """Persist display name in user_prefs."""
        if not user_id:
            return
        async with self._lock:
            db = self._load()
            prefs = db.setdefault("user_prefs", {})
            user_pref = prefs.setdefault(str(user_id), {})
            user_pref["display_name"] = (display_name or "").strip()
            self._save(db)

    async def list_users(self) -> list[dict[str, Any]]:
        """Aggregate users from accounts + prefs (sorted: owner first, then id)."""
        async with self._lock:
            db = self._load()
            accounts = db.get("accounts", {}) or {}
            prefs = db.get("user_prefs", {}) or {}
            owner = str(db.get("owner_user_id") or "").strip()

        by_id: dict[str, dict[str, Any]] = {}
        if owner:
            by_id[owner] = {
                "user_id": owner,
                "display_name": "",
                "accounts": [],
                "tts_enabled": False,
                "is_owner": True,
            }
        for account_key, uid in accounts.items():
            user_id = str(uid)
            if not user_id:
                continue
            entry = by_id.setdefault(
                user_id,
                {
                    "user_id": user_id,
                    "display_name": "",
                    "accounts": [],
                    "tts_enabled": False,
                    "is_owner": False,
                },
            )
            entry["accounts"].append(str(account_key))

        for user_id, pref in prefs.items():
            uid = str(user_id)
            if not uid:
                continue
            entry = by_id.setdefault(
                uid,
                {
                    "user_id": uid,
                    "display_name": "",
                    "accounts": [],
                    "tts_enabled": False,
                    "is_owner": False,
                },
            )
            if isinstance(pref, dict):
                entry["display_name"] = str(pref.get("display_name") or "").strip()
                if pref.get("tts_enabled") is not None:
                    entry["tts_enabled"] = bool(pref.get("tts_enabled"))

        for entry in by_id.values():
            entry["is_owner"] = bool(owner) and entry["user_id"] == owner
            entry["accounts"].sort()

        users = list(by_id.values())
        users.sort(key=lambda u: (not u["is_owner"], u["user_id"]))
        return users

    async def create_user(
        self,
        *,
        display_name: str = "",
        channel: str = "",
        sender_id: str = "",
    ) -> str:
        """Create a new user id; optionally link one channel account."""
        channel = (channel or "").strip()
        sender_id = (sender_id or "").strip()
        name = (display_name or "").strip()
        async with self._lock:
            db = self._load()
            accounts = db.setdefault("accounts", {})
            if channel and sender_id:
                account = _account_key(channel, sender_id)
                existing = accounts.get(account)
                if existing:
                    user_id = str(existing)
                else:
                    user_id = uuid.uuid4().hex
                    accounts[account] = user_id
            else:
                user_id = uuid.uuid4().hex
            prefs = db.setdefault("user_prefs", {})
            user_pref = prefs.setdefault(user_id, {})
            if name:
                user_pref["display_name"] = name
            if not db.get("owner_user_id"):
                db["owner_user_id"] = user_id
            self._save(db)
            logger.info("Created user {} (display_name={!r})", user_id, name)
            return user_id

    async def delete_user(self, user_id: str) -> tuple[bool, str | None]:
        """Remove account links and prefs for *user_id*. Blocks deleting the owner.

        Returns (ok, error_code). Does not wipe the workspace directory — caller does.
        """
        uid = str(user_id or "").strip()
        if not uid:
            return False, "empty_user_id"
        async with self._lock:
            db = self._load()
            owner = str(db.get("owner_user_id") or "").strip()
            if owner and owner == uid:
                return False, "cannot_delete_owner"
            accounts = db.get("accounts", {}) or {}
            to_drop = [k for k, v in accounts.items() if str(v) == uid]
            for k in to_drop:
                del accounts[k]
            db["accounts"] = accounts
            prefs = db.get("user_prefs", {}) or {}
            prefs.pop(uid, None)
            db["user_prefs"] = prefs
            self._save(db)
            logger.info("Deleted user {} ({} account links)", uid, len(to_drop))
            return True, None

    async def user_exists(self, user_id: str) -> bool:
        """True if user appears in accounts or prefs (or is owner)."""
        uid = str(user_id or "").strip()
        if not uid:
            return False
        async with self._lock:
            db = self._load()
            if str(db.get("owner_user_id") or "").strip() == uid:
                return True
            accounts = db.get("accounts", {}) or {}
            if any(str(v) == uid for v in accounts.values()):
                return True
            prefs = db.get("user_prefs", {}) or {}
            return uid in prefs

    async def create_link_code(self, user_id: str) -> str:
        """Create a one-time code that can link another account to *user_id*."""
        alphabet = string.ascii_uppercase + string.digits
        for _ in range(20):
            code = "".join(secrets.choice(alphabet) for _ in range(8))
            async with self._lock:
                db = self._load()
                links = db.setdefault("pending_links", {})
                if code in links:
                    continue
                links[code] = {
                    "user_id": user_id,
                    "expires_at": (_now() + timedelta(seconds=self._code_ttl_seconds)).isoformat(),
                    "remaining_attempts": self._code_attempt_limit,
                }
                self._save(db)
                return code
        raise RuntimeError("Failed to generate unique link code")

    async def consume_link_code(self, code: str, channel: str, sender_id: str) -> LinkConsumeResult:
        """Consume and apply a link code for the current channel account."""
        normalized = (code or "").strip().upper()
        if not normalized:
            return LinkConsumeResult(ok=False, error="empty_code")
        account = _account_key(channel, sender_id)
        async with self._lock:
            db = self._load()
            links = db.setdefault("pending_links", {})
            payload = links.get(normalized)
            if not payload:
                return LinkConsumeResult(ok=False, error="invalid_code")

            expires_at = self._parse_ts(payload.get("expires_at"))
            if expires_at is None or expires_at <= _now():
                links.pop(normalized, None)
                self._save(db)
                return LinkConsumeResult(ok=False, error="expired_code")

            attempts = int(payload.get("remaining_attempts", 0))
            if attempts <= 0:
                links.pop(normalized, None)
                self._save(db)
                return LinkConsumeResult(ok=False, error="attempts_exhausted")

            user_id = str(payload.get("user_id") or "")
            if not user_id:
                links.pop(normalized, None)
                self._save(db)
                return LinkConsumeResult(ok=False, error="invalid_payload")

            db.setdefault("accounts", {})[account] = user_id
            links.pop(normalized, None)
            self._save(db)
            return LinkConsumeResult(ok=True, user_id=user_id)

    async def register_failed_link_attempt(self, code: str) -> None:
        """Decrease remaining attempts for a code when validation fails."""
        normalized = (code or "").strip().upper()
        if not normalized:
            return
        async with self._lock:
            db = self._load()
            links = db.setdefault("pending_links", {})
            payload = links.get(normalized)
            if not payload:
                return
            attempts = int(payload.get("remaining_attempts", 0))
            attempts -= 1
            if attempts <= 0:
                links.pop(normalized, None)
            else:
                payload["remaining_attempts"] = attempts
            self._save(db)

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {
                "accounts": {},
                "pending_links": {},
                "user_prefs": {},
                "owner_user_id": None,
                "registration_requests": {},
                "registration_codes": {},
            }
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("accounts", {})
                data.setdefault("pending_links", {})
                data.setdefault("user_prefs", {})
                data.setdefault("owner_user_id", None)
                data.setdefault("registration_requests", {})
                data.setdefault("registration_codes", {})
                return data
        except Exception:
            logger.exception("Failed to load user links from {}", self._path)
        return {
            "accounts": {},
            "pending_links": {},
            "user_prefs": {},
            "owner_user_id": None,
            "registration_requests": {},
            "registration_codes": {},
        }

    def _save(self, payload: dict[str, Any]) -> None:
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _parse_ts(raw: Any) -> datetime | None:
        if not isinstance(raw, str) or not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
