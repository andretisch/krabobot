"""Local wake-phrase matching («Эй, Арнольд» / hey arnold)."""

from __future__ import annotations

import re
import unicodedata

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, ё→е, strip punctuation, collapse spaces."""
    s = unicodedata.normalize("NFKC", text or "")
    s = s.lower().replace("ё", "е")
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return s


def matches_wake_phrase(text: str) -> bool:
    """True if normalized ASR text contains wake variants.

    Primary: «эй» + «арнольд». Optional Latin: hey + arnold.
    """
    norm = normalize_text(text)
    if not norm:
        return False
    tokens = set(norm.split())
    if "эй" in tokens and "арнольд" in tokens:
        return True
    if "эй" in norm and "арнольд" in norm:
        return True
    if "hey" in tokens and "arnold" in tokens:
        return True
    if "hey arnold" in norm:
        return True
    # Soft variants without exact tokenization (ASR glitches)
    if "арнольд" in norm and ("эй" in norm or "hey" in norm):
        return True
    return False
