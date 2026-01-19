"""Shared normalization + aliasing for player name matching across data sources."""

from __future__ import annotations

import re
import unicodedata
from typing import Final


_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_NON_ALNUM_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")

# Common name suffixes to strip for matching (generational, honorifics).
_NAME_SUFFIXES: Final[frozenset[str]] = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})

# Aliases are keyed by a *compact* normalized string (no spaces/punctuation) and map to a
# canonical normalized name with spaces (as returned by `normalize_player_name`).
# Only needed for cases that can't be handled by suffix stripping or other rules.
PLAYER_NAME_ALIASES: Final[dict[str, str]] = {
    # GG Jackson (MEM) is sometimes emitted as "G.G. Jackson" or full name.
    "ggjackson": "gg jackson",
    "gregoryjackson": "gg jackson",
    # Alexandre Sarr -> Alex Sarr
    "alexandresarr": "alex sarr",
}


def normalize_player_name(value: object) -> str:
    """Normalize a player name for matching: fold accents, strip punctuation, lowercase.

    Automatically strips common suffixes (Jr, Sr, II, III, IV, V) and applies alias rewrites.
    Returns a space-separated normalized name.
    """
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKD", text)
    ascii_folded = normalized.encode("ascii", "ignore").decode("ascii").lower()
    # For matching, treat apostrophes and periods as intra-token punctuation (remove without spacing),
    # while other punctuation becomes a token boundary.
    ascii_folded = ascii_folded.replace("'", "").replace(".", "")
    spaced = _NON_ALNUM_RE.sub(" ", ascii_folded).strip()
    spaced = _SPACE_RE.sub(" ", spaced)
    if not spaced:
        return ""

    tokens = spaced.split()
    # Collapse sequences of single-letter tokens: "r j barrett" -> "rj barrett".
    collapsed: list[str] = []
    buf = ""
    for tok in tokens:
        if len(tok) == 1 and tok.isalnum():
            buf += tok
            continue
        if buf:
            collapsed.append(buf)
            buf = ""
        collapsed.append(tok)
    if buf:
        collapsed.append(buf)

    # Strip trailing suffixes (Jr, Sr, II, III, IV, V).
    while collapsed and collapsed[-1] in _NAME_SUFFIXES:
        collapsed.pop()

    if not collapsed:
        return ""

    canonical = " ".join(collapsed)
    compact = canonical.replace(" ", "")
    return PLAYER_NAME_ALIASES.get(compact, canonical)
