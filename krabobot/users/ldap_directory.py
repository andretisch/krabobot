"""ldap3 search-then-bind against Active Directory or another LDAP server.

Imported only when a real directory login runs. Passwords are never logged.
"""

from __future__ import annotations

import ssl

from loguru import logger

from krabobot.config.schema import LdapConfig
from krabobot.users.ldap_auth import (
    LdapDirectoryUser,
    build_user_filter,
    normalize_ldap_username,
    pick_display_name,
)


def _safe_error(exc: BaseException, password: str) -> str:
    text = str(exc)
    if password and password in text:
        text = text.replace(password, "***")
    return text


def _entry_attr(entry: object, name: str) -> str:
    if not name or name not in entry:  # type: ignore[operator]
        return ""
    try:
        value = entry[name].value  # type: ignore[index]
    except Exception:
        return ""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value or "").strip()


class Ldap3Directory:
    """Service-account search, then bind as the user DN with the typed password."""

    def __init__(self, cfg: LdapConfig) -> None:
        self._cfg = cfg

    def authenticate(self, username: str, password: str) -> LdapDirectoryUser | None:
        cfg = self._cfg
        login = (username or "").strip()
        if not login or not password:
            return None
        if not cfg.enabled or not cfg.host.strip() or not cfg.base_dn.strip():
            return None
        try:
            from ldap3 import NONE, SUBTREE, Connection, Server, Tls
            from ldap3.core.exceptions import LDAPException
        except ImportError:
            logger.error("ldap3 is not installed; LDAP web login is unavailable")
            return None

        filt = build_user_filter(cfg.user_filter, cfg.username_attribute, login)
        attrs = [
            a
            for a in {
                cfg.username_attribute,
                cfg.display_name_attribute,
                cfg.email_attribute,
                "cn",
            }
            if a
        ]
        tls_config = None
        if cfg.use_ssl or cfg.start_tls:
            tls_config = Tls(
                validate=ssl.CERT_REQUIRED if cfg.verify_certificates else ssl.CERT_NONE
            )
        server = Server(
            cfg.host.strip(),
            port=int(cfg.port),
            use_ssl=bool(cfg.use_ssl),
            tls=tls_config,
            get_info=NONE,
            connect_timeout=8,
        )
        conn = Connection(
            server,
            user=cfg.bind_dn.strip() or None,
            password=cfg.bind_password if cfg.bind_dn.strip() else None,
            auto_bind=False,
            receive_timeout=8,
            raise_exceptions=False,
        )
        user_dn = ""
        raw_username = ""
        display = ""
        email = ""
        try:
            if not conn.open():
                logger.warning("LDAP connect failed host={}", cfg.host)
                return None
            if cfg.start_tls and not cfg.use_ssl and not conn.start_tls():
                logger.warning("LDAP StartTLS failed host={}", cfg.host)
                return None
            if cfg.bind_dn.strip() and not conn.bind():
                logger.warning("LDAP service bind failed")
                return None
            found = conn.search(
                search_base=cfg.base_dn.strip(),
                search_filter=filt,
                search_scope=SUBTREE,
                attributes=attrs,
                size_limit=2,
            )
            entries = list(conn.entries)
            if not found or len(entries) != 1:
                logger.info("LDAP search for {} matched {} entries", login, len(entries))
                return None
            entry = entries[0]
            user_dn = str(getattr(entry, "entry_dn", "") or "")
            raw_username = _entry_attr(entry, cfg.username_attribute) or login
            display = pick_display_name(
                _entry_attr(entry, cfg.display_name_attribute),
                _entry_attr(entry, "cn"),
                fallback=raw_username,
            )
            email = _entry_attr(entry, cfg.email_attribute)
        except (LDAPException, OSError) as exc:
            logger.warning("LDAP search error for {}: {}", login, _safe_error(exc, password))
            return None
        finally:
            try:
                conn.unbind()
            except Exception:
                pass

        key = normalize_ldap_username(raw_username)
        if not key or not user_dn:
            logger.info("LDAP entry for {} has no usable username or DN", login)
            return None

        user_conn = Connection(
            server,
            user=user_dn,
            password=password,
            auto_bind=False,
            receive_timeout=8,
            raise_exceptions=False,
        )
        try:
            if not user_conn.open():
                return None
            if cfg.start_tls and not cfg.use_ssl and not user_conn.start_tls():
                return None
            if not user_conn.bind():
                logger.info("LDAP user bind failed for {}", key)
                return None
        except (LDAPException, OSError) as exc:
            logger.warning("LDAP user bind error for {}: {}", key, _safe_error(exc, password))
            return None
        finally:
            try:
                user_conn.unbind()
            except Exception:
                pass

        return LdapDirectoryUser(
            username=key,
            display_name=pick_display_name(display, fallback=key),
            email=email,
            dn=user_dn,
        )
