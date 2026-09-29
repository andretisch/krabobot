"""Local wake-phrase matching («Эй, Арнольд» / «Привет, Арнольд» / hey arnold)."""

from __future__ import annotations

import re
import unicodedata
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")

# Common sherpa / gigaam mishears for «Арнольд»
_ARNOLD_RE = re.compile(
    r"арн+о+л+ь?д|arnold|арнолд|арнольд|арнольт|arnol+d",
    re.IGNORECASE,
)

DEFAULT_WAKE_PHRASES: tuple[str, ...] = (
    "эй арнольд",
    "ей арнольд",
    "привет арнольд",
    "hey arnold",
)

# Greetings that count when «Арнольд» is also present (fuzzy name match).
DEFAULT_WAKE_GREETINGS: tuple[str, ...] = (
    "эй",
    "ей",
    "привет",
    "hey",
)


def normalize_text(text: str) -> str:
    """Lowercase, ё→е, strip punctuation, collapse spaces."""
    s = unicodedata.normalize("NFKC", text or "")
    s = s.lower().replace("ё", "е")
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return s


def _has_arnold(norm: str) -> bool:
    tokens = set(norm.split())
    return bool(_ARNOLD_RE.search(norm)) or "арнольд" in tokens or "arnold" in tokens


def _phrase_hits(norm: str, phrase: str) -> bool:
    """True if normalized utterance contains phrase as whole words."""
    p = normalize_text(phrase)
    if not p or not norm:
        return False
    return f" {p} " in f" {norm} "


def matches_wake_phrase(
    text: str,
    *,
    phrases: Sequence[str] | None = None,
    greetings: Sequence[str] | None = None,
) -> bool:
    """True if normalized ASR text matches a wake variant.

    Defaults: configured phrases («эй/привет арнольд», hey arnold) plus
    fuzzy «Арнольд» with a greeting token nearby.
    """
    norm = normalize_text(text)
    if not norm:
        return False

    phrase_list = tuple(phrases) if phrases is not None else DEFAULT_WAKE_PHRASES
    for phrase in sorted(phrase_list, key=lambda s: len(normalize_text(s)), reverse=True):
        if _phrase_hits(norm, phrase):
            return True

    # Compact glue of configured phrases: «эйарнольд» without space
    compact = norm.replace(" ", "")
    for phrase in phrase_list:
        needle = normalize_text(phrase).replace(" ", "")
        if needle and needle in compact:
            return True

    has_arnold = _has_arnold(norm)
    if not has_arnold:
        return False

    greet_list = tuple(greetings) if greetings is not None else DEFAULT_WAKE_GREETINGS
    tokens = set(norm.split())
    for g in greet_list:
        g_norm = normalize_text(g)
        if not g_norm:
            continue
        if g_norm in tokens or bool(re.search(rf"(?:^|\s){re.escape(g_norm)}(?:\s|$)", norm)):
            return True
    return False


@dataclass(frozen=True)
class WakeDecodeDecision:
    """What to do after one wake-window ASR decode."""

    matched: bool
    print_text: str | None  # None = skip console print (debounce)
    consume_samples: int  # drop this many samples from the ring left
    cooldown_samples: int  # ignore decode for this many incoming samples


def decide_after_wake_decode(
    text: str,
    *,
    last_printed: str,
    window_n: int,
    sample_rate: int,
    phrases: Sequence[str] | None = None,
    greetings: Sequence[str] | None = None,
    cooldown_s: float = 0.8,
) -> WakeDecodeDecision:
    """Advance past the scored audio so the same utterance is not rescored forever.

    - On match: consume the window, short cooldown (caller leaves the wake loop).
    - On miss: consume the window + cooldown so overlapping hops cannot spam.
    - Print only when non-empty text differs from the last printed line.
    """
    stripped = (text or "").strip()
    matched = matches_wake_phrase(stripped, phrases=phrases, greetings=greetings)
    should_print = bool(stripped) and stripped != last_printed
    consume = max(window_n, int(0.5 * sample_rate))
    cooldown = max(0, int(cooldown_s * sample_rate))
    return WakeDecodeDecision(
        matched=matched,
        print_text=stripped if should_print else None,
        consume_samples=consume,
        cooldown_samples=cooldown,
    )


def drop_ring_samples(ring: deque, total: int, n: int) -> int:
    """Popleft from a ring until ``n`` samples dropped (or empty).

    Returns the new ``total`` sample count.
    """
    remaining = n
    while remaining > 0 and ring:
        chunk = ring.popleft()
        size = int(getattr(chunk, "size", len(chunk)))
        total -= size
        remaining -= size
    return max(0, total)
