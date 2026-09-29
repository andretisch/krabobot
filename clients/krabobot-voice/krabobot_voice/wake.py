"""Local wake-phrase matching from user-configured phrase(s).

Defaults remain «Эй/Привет Арнольд» / hey arnold, but the primary source is
``wake.phrase`` / ``wake.phrases`` in ``config.yaml`` next to the app.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")

# Common sherpa / gigaam mishears for «Арнольд» (used when that name is configured).
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

# Greetings that count when the configured name token is also present.
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


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _max_edits_for(token: str) -> int:
    """Looser tolerance for longer tokens (ASR glitches on names)."""
    n = len(token)
    if n <= 2:
        return 0
    if n <= 4:
        return 1
    if n <= 7:
        return 1
    return 2


def _token_matches(hay_token: str, needle: str, *, max_edits: int | None = None) -> bool:
    """Exact, prefix/suffix, or small edit-distance match for one token."""
    if not needle or not hay_token:
        return False
    if hay_token == needle:
        return True
    # Short greetings: exact only (avoid «эй»↔«ей» false friends handled via phrase list).
    if len(needle) <= 2:
        return hay_token == needle
    budget = _max_edits_for(needle) if max_edits is None else max_edits
    if abs(len(hay_token) - len(needle)) > budget:
        return False
    if needle in hay_token or hay_token in needle:
        # Containment only when lengths are close (avoid «ар» in «арнольд»).
        if min(len(hay_token), len(needle)) >= max(3, len(needle) - budget):
            return True
    return _levenshtein(hay_token, needle) <= budget


def _tokens_cover(norm: str, needed: Sequence[str]) -> bool:
    """True if every needed token has a fuzzy match among utterance tokens."""
    if not needed:
        return False
    hay = norm.split()
    if not hay:
        return False
    used: set[int] = set()
    for needle in needed:
        found = False
        for i, ht in enumerate(hay):
            if i in used:
                continue
            if _token_matches(ht, needle):
                used.add(i)
                found = True
                break
        if not found:
            return False
    return True


def phrase_tokens(phrase: str) -> tuple[str, ...]:
    """Normalized whitespace-split tokens of a wake phrase."""
    p = normalize_text(phrase)
    return tuple(t for t in p.split() if t)


def derive_name_tokens(phrases: Sequence[str]) -> tuple[str, ...]:
    """Last token of each multi/single-word phrase (the 'name' to spot fuzzily)."""
    names: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        toks = phrase_tokens(phrase)
        if not toks:
            continue
        name = toks[-1]
        if name not in seen:
            seen.add(name)
            names.append(name)
    return tuple(names)


def derive_greetings(phrases: Sequence[str]) -> tuple[str, ...]:
    """Leading tokens of multi-word phrases (everything except the last)."""
    greets: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        toks = phrase_tokens(phrase)
        if len(toks) < 2:
            continue
        for g in toks[:-1]:
            if g not in seen:
                seen.add(g)
                greets.append(g)
    return tuple(greets)


def _has_configured_name(norm: str, names: Sequence[str]) -> bool:
    tokens = norm.split()
    for name in names:
        if not name:
            continue
        # Special-case Arnold ASR glitches when that name is configured.
        if name in {"арнольд", "arnold"} and (
            bool(_ARNOLD_RE.search(norm)) or "арнольд" in tokens or "arnold" in tokens
        ):
            return True
        for ht in tokens:
            if _token_matches(ht, name):
                return True
        # Compact: name glued without spaces («эйарнольд» already handled elsewhere).
        compact = norm.replace(" ", "")
        if name in compact:
            return True
    return False


def _phrase_hits(norm: str, phrase: str) -> bool:
    """True if normalized utterance contains phrase as whole words."""
    p = normalize_text(phrase)
    if not p or not norm:
        return False
    return f" {p} " in f" {norm} "


def _resolve_greetings(
    phrase_list: Sequence[str],
    greetings: Sequence[str] | None,
) -> tuple[str, ...]:
    if greetings is not None:
        return tuple(greetings)
    derived = derive_greetings(phrase_list)
    if derived:
        return derived
    if tuple(phrase_list) == DEFAULT_WAKE_PHRASES:
        return DEFAULT_WAKE_GREETINGS
    return derived


def _contiguous_phrase_end(tokens: Sequence[str], needed: Sequence[str]) -> int | None:
    """Index of last token of the earliest fuzzy contiguous match, or None."""
    if not needed or not tokens:
        return None
    n = len(needed)
    for start in range(0, len(tokens) - n + 1):
        ok = True
        for j, needle in enumerate(needed):
            if not _token_matches(tokens[start + j], needle):
                ok = False
                break
        if ok:
            return start + n - 1
    return None


def _wake_end_token_index(
    tokens: Sequence[str],
    *,
    phrases: Sequence[str],
    greetings: Sequence[str],
) -> int | None:
    """Token index where the first wake match ends (inclusive), or None."""
    if not tokens:
        return None

    # 1) Contiguous configured phrases (longest first → better «эй арнольд»).
    best: int | None = None
    for phrase in sorted(phrases, key=lambda s: len(normalize_text(s)), reverse=True):
        needed = phrase_tokens(phrase)
        if not needed:
            continue
        end = _contiguous_phrase_end(tokens, needed)
        if end is not None and (best is None or end < best):
            best = end
    if best is not None:
        return best

    # 2) Compact glue: whole utterance is one glued phrase → end at last token.
    compact = "".join(tokens)
    for phrase in phrases:
        needle = normalize_text(phrase).replace(" ", "")
        if needle and needle in compact:
            return len(tokens) - 1

    # 3) Greeting + name: end at the matched name token (after a greeting).
    names = derive_name_tokens(phrases)
    greet_norms = [normalize_text(g) for g in greetings if normalize_text(g)]
    has_greet = False
    for ht in tokens:
        for g in greet_norms:
            if _token_matches(ht, g):
                has_greet = True
                break
        if has_greet:
            break
    if has_greet and names:
        for i, ht in enumerate(tokens):
            for name in names:
                if not name:
                    continue
                if name in {"арнольд", "arnold"} and (
                    bool(_ARNOLD_RE.fullmatch(ht)) or _token_matches(ht, name)
                ):
                    return i
                if _token_matches(ht, name):
                    return i

    # 4) Order-independent token cover: end at the latest matched needed token.
    for phrase in phrases:
        needed = phrase_tokens(phrase)
        if not needed:
            continue
        used: set[int] = set()
        last = -1
        ok = True
        for needle in needed:
            found = False
            for i, ht in enumerate(tokens):
                if i in used:
                    continue
                if _token_matches(ht, needle):
                    used.add(i)
                    last = max(last, i)
                    found = True
                    break
            if not found:
                ok = False
                break
        if ok and last >= 0:
            if best is None or last < best:
                best = last
    return best


def matches_wake_phrase(
    text: str,
    *,
    phrases: Sequence[str] | None = None,
    greetings: Sequence[str] | None = None,
) -> bool:
    """True if normalized ASR text matches a configured wake variant.

    Matching order:
    1. Full configured phrase as contiguous words (or glued without spaces).
    2. All tokens of a multi-word phrase present (fuzzy, order-independent).
    3. Configured/derived greeting + fuzzy name token nearby.
    """
    norm = normalize_text(text)
    if not norm:
        return False

    phrase_list = tuple(phrases) if phrases is not None else DEFAULT_WAKE_PHRASES
    if not phrase_list:
        return False

    for phrase in sorted(phrase_list, key=lambda s: len(normalize_text(s)), reverse=True):
        if _phrase_hits(norm, phrase):
            return True

    # Compact glue of configured phrases: «эйарнольд» without space
    compact = norm.replace(" ", "")
    for phrase in phrase_list:
        needle = normalize_text(phrase).replace(" ", "")
        if needle and needle in compact:
            return True

    # Looser: every token of a configured phrase appears (fuzzy).
    for phrase in phrase_list:
        toks = phrase_tokens(phrase)
        if len(toks) >= 2 and _tokens_cover(norm, toks):
            return True
        if len(toks) == 1 and _tokens_cover(norm, toks):
            # Single-token wake: require the token itself (already fuzzy).
            return True

    names = derive_name_tokens(phrase_list)
    if not _has_configured_name(norm, names):
        return False

    greet_list = _resolve_greetings(phrase_list, greetings)

    if not greet_list:
        # Name alone is not enough unless the configured phrase is a single token.
        return any(len(phrase_tokens(p)) == 1 for p in phrase_list)

    for g in greet_list:
        g_norm = normalize_text(g)
        if not g_norm:
            continue
        # Fuzzy greet match on tokens (short greetings stay exact via _token_matches).
        for ht in norm.split():
            if _token_matches(ht, g_norm):
                return True
        if bool(re.search(rf"(?:^|\s){re.escape(g_norm)}(?:\s|$)", norm)):
            return True
    return False


def command_after_wake(
    text: str,
    *,
    phrases: Sequence[str] | None = None,
    greetings: Sequence[str] | None = None,
) -> str:
    """Return text *after* the wake phrase; empty if wake-only or wake at the end.

    Prefers the remainder after the first wake match. Leading filler before the
    wake phrase is ignored (not treated as a command).
    """
    if not matches_wake_phrase(text, phrases=phrases, greetings=greetings):
        return ""

    phrase_list = tuple(phrases) if phrases is not None else DEFAULT_WAKE_PHRASES
    greet_list = _resolve_greetings(phrase_list, greetings)
    tokens = normalize_text(text).split()
    end = _wake_end_token_index(tokens, phrases=phrase_list, greetings=greet_list)
    if end is None or end >= len(tokens) - 1:
        return ""
    return " ".join(tokens[end + 1 :]).strip()
