"""Shared normalization + aliasing for player name matching across data sources."""

from __future__ import annotations

import re
import unicodedata
from typing import Final


_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_NON_ALNUM_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


# Aliases are keyed by a *compact* normalized string (no spaces/punctuation) and map to a
# canonical normalized name with spaces (as returned by `normalize_player_name`).
PLAYER_NAME_ALIASES: Final[dict[str, str]] = {
    # Existing special cases.
    "alexandresarr": "alex sarr",
    "jimmybutleriii": "jimmy butler",
    # GG Jackson (MEM) is sometimes emitted as "G.G. Jackson II".
    "ggjacksonii": "gg jackson",
    # Some sources use the player's full given name.
    "gregoryjacksonii": "gg jackson",
}


def normalize_player_name(value: object) -> str:
    """Normalize a player name for matching: fold accents, strip punctuation, lowercase.

    Returns a space-separated normalized name, with optional alias rewrites applied.
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

    canonical = " ".join(collapsed)
    compact = canonical.replace(" ", "")
    return PLAYER_NAME_ALIASES.get(compact, canonical)
