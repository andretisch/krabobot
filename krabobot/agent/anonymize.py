"""Turn-local PII anonymization for LLM-bound messages.

Masks sensitive spans toward the model only. Session/history/UI keep originals;
token maps are never persisted.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from loguru import logger

_DEFAULT_RULES_PATH = Path(__file__).with_name("anonymize_rules.json")
_TOKEN_RE = re.compile(r"\[[A-Z]+-\d{5}\]")
_LINKED_ACCOUNTS_TAG = "[Linked Accounts]"
_TEXT_PART_TYPES = frozenset({"text", "input_text"})


@dataclass(slots=True)
class RegexRule:
    """One regex anonymization rule loaded from JSON."""

    pattern: re.Pattern[str]
    category: str
    description: str = ""
    enabled: bool = True


@dataclass
class TokenMap:
    """Turn-local token ↔ original mapping (not persisted)."""

    token_to_original: dict[str, str] = field(default_factory=dict)
    original_to_token: dict[str, str] = field(default_factory=dict)
    _counters: dict[str, int] = field(default_factory=dict)

    def token_for(self, category: str, original: str) -> str:
        """Return existing or new token for *original* within this turn."""
        existing = self.original_to_token.get(original)
        if existing is not None:
            return existing
        n = self._counters.get(category, 0) + 1
        self._counters[category] = n
        token = f"[{category}-{n:05d}]"
        self.original_to_token[original] = token
        self.token_to_original[token] = original
        return token

    def encode(self, text: str, rules: list[RegexRule] | None = None) -> str:
        """Replace PII spans in *text* with tokens."""
        if not text:
            return text
        active = rules if rules is not None else load_default_rules()
        if not active:
            return text
        return _encode_text(text, self, active)

    def decode(self, text: str) -> str:
        """Restore originals for known tokens in *text*."""
        if not text or not self.token_to_original:
            return text
        result = text
        # Longer tokens first is unnecessary for fixed [CAT-#####] form, but stable.
        for token, original in sorted(
            self.token_to_original.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            if token in result:
                result = result.replace(token, original)
        return result


@lru_cache(maxsize=1)
def load_default_rules() -> tuple[RegexRule, ...]:
    """Load and compile packaged regex rules (cached)."""
    return tuple(_load_rules_from_path(_DEFAULT_RULES_PATH))


def _load_rules_from_path(path: Path) -> list[RegexRule]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        logger.warning("Failed to read anonymize rules from {}: {}", path, exc)
        return []
    except json.JSONDecodeError as exc:
        logger.warning("Invalid anonymize rules JSON {}: {}", path, exc)
        return []

    raw_rules = data.get("regex_rules") if isinstance(data, dict) else None
    if not isinstance(raw_rules, list):
        logger.warning("anonymize rules missing regex_rules array: {}", path)
        return []

    compiled: list[RegexRule] = []
    for item in raw_rules:
        if not isinstance(item, dict):
            continue
        if not item.get("enabled", True):
            continue
        pattern = item.get("pattern")
        category = item.get("category")
        if not isinstance(pattern, str) or not isinstance(category, str) or not category:
            continue
        try:
            compiled.append(
                RegexRule(
                    pattern=re.compile(pattern),
                    category=category.upper(),
                    description=str(item.get("description") or ""),
                    enabled=True,
                )
            )
        except re.error as exc:
            logger.warning("Skipping bad anonymize pattern {!r}: {}", pattern, exc)
    return compiled


def _protected_spans(text: str) -> list[tuple[int, int]]:
    """Spans that must not be re-masked: existing tokens and Linked Accounts blocks."""
    spans: list[tuple[int, int]] = []
    for m in _TOKEN_RE.finditer(text):
        spans.append((m.start(), m.end()))

    start = 0
    while True:
        idx = text.find(_LINKED_ACCOUNTS_TAG, start)
        if idx < 0:
            break
        end = text.find("\n\n", idx)
        if end < 0:
            end = len(text)
        else:
            end += 2  # keep the blank-line separator protected with the block
        spans.append((idx, end))
        start = end
    return spans


def _is_protected(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    for ps, pe in spans:
        if start < pe and end > ps:
            return True
    return False


@dataclass(slots=True, order=True)
class _Match:
    start: int
    end: int
    category: str
    text: str

    @property
    def length(self) -> int:
        return self.end - self.start


def _collect_matches(text: str, rules: list[RegexRule], protected: list[tuple[int, int]]) -> list[_Match]:
    found: list[_Match] = []
    for rule in rules:
        for m in rule.pattern.finditer(text):
            if m.start() == m.end():
                continue
            if _is_protected(protected, m.start(), m.end()):
                continue
            found.append(
                _Match(start=m.start(), end=m.end(), category=rule.category, text=m.group(0))
            )
    # Prefer earlier start, then longer span (stable overlap resolution).
    found.sort(key=lambda m: (m.start, -m.length))
    selected: list[_Match] = []
    cursor = 0
    for m in found:
        if m.start < cursor:
            continue
        selected.append(m)
        cursor = m.end
    return selected


def _encode_text(text: str, token_map: TokenMap, rules: list[RegexRule]) -> str:
    protected = _protected_spans(text)
    matches = _collect_matches(text, rules, protected)
    if not matches:
        return text
    parts: list[str] = []
    last = 0
    for m in matches:
        parts.append(text[last:m.start])
        parts.append(token_map.token_for(m.category, m.text))
        last = m.end
    parts.append(text[last:])
    return "".join(parts)


def encode_content(content: Any, token_map: TokenMap, rules: list[RegexRule] | None = None) -> Any:
    """Anonymize string content or multimodal text parts."""
    active = rules if rules is not None else list(load_default_rules())
    if isinstance(content, str):
        return token_map.encode(content, active)
    if isinstance(content, list):
        out: list[Any] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in _TEXT_PART_TYPES:
                text = block.get("text")
                if isinstance(text, str):
                    out.append({**block, "text": token_map.encode(text, active)})
                    continue
            out.append(block)
        return out
    return content


def decode_content(content: Any, token_map: TokenMap) -> Any:
    """Restore originals in string content or multimodal text parts."""
    if isinstance(content, str):
        return token_map.decode(content)
    if isinstance(content, list):
        out: list[Any] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in _TEXT_PART_TYPES:
                text = block.get("text")
                if isinstance(text, str):
                    out.append({**block, "text": token_map.decode(text)})
                    continue
            out.append(block)
        return out
    return content


def anonymize_messages(
    messages: list[dict[str, Any]],
    token_map: TokenMap,
    rules: list[RegexRule] | None = None,
) -> list[dict[str, Any]]:
    """Deep-copy *messages* and anonymize non-system content toward the LLM."""
    active = rules if rules is not None else list(load_default_rules())
    out: list[dict[str, Any]] = []
    for msg in messages:
        cloned = copy.deepcopy(msg)
        role = cloned.get("role")
        if role != "system" and "content" in cloned:
            cloned["content"] = encode_content(cloned["content"], token_map, active)
        out.append(cloned)
    return out


def decode_messages(
    messages: list[dict[str, Any]],
    token_map: TokenMap,
) -> list[dict[str, Any]]:
    """Decode tokenized content (and tool-call argument strings) for persistence/UI."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        cloned = copy.deepcopy(msg)
        if "content" in cloned:
            cloned["content"] = decode_content(cloned["content"], token_map)
        tool_calls = cloned.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                    fn["arguments"] = token_map.decode(fn["arguments"])
        out.append(cloned)
    return out
