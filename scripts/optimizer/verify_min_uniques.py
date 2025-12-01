from __future__ import annotations

import argparse
import sys
from typing import List, Tuple, Sequence

import pandas as pd


DK_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]


def _extract_token(cell: object) -> str:
    """Return a stable identifier from a lineup cell.

    Accepts values like "Jalen Brunson (BRUNSJA01)" or plain names/IDs.
    If parentheses are present, returns the innermost (...) content; otherwise the stripped string.
    """
    if cell is None:
        return ""
    s = str(cell).strip()
    if not s:
        return ""
    # Prefer PID between the last '(' and ')', if present
    if ")" in s and "(" in s and s.rfind(")") > s.rfind("("):
        try:
            return s[s.rfind("(") + 1 : s.rfind(")")].strip()
        except Exception:
            pass
    return s


def _load_lineups_from_csv(path: str, slot_cols: Sequence[str] | None = None) -> List[List[str]]:
    df = pd.read_csv(path)
    cols = [c for c in (slot_cols or DK_SLOTS) if c in df.columns]
    if not cols:
        raise SystemExit(f"No DK slot columns found in CSV at {path}")
    lineups: List[List[str]] = []
    for _, row in df.iterrows():
        toks: List[str] = []
        for c in cols:
            toks.append(_extract_token(row.get(c)))
        # Guard against partial rows
        toks = [t for t in toks if t]
        if toks:
            lineups.append(toks)
    return lineups


def verify_consecutive_min_uniques(
    lineups: Sequence[Sequence[str]],
    min_uniques: int,
    lineup_size: int | None = None,
) -> Tuple[int, int, List[Tuple[int, int, int]]]:
    """Check that each lineup differs from the previous by at least `min_uniques` players.

    Returns (n_checked, min_changes, violations), where violations is a list of
    tuples (index, changes, overlap) for each failed consecutive pair (index is the second lineup's index).
    """
    n = len(lineups)
    if n <= 1:
        return 0, 0, []
    viols: List[Tuple[int, int, int]] = []
    min_changes = 1_000_000
    for i in range(1, n):
        a = set(lineups[i - 1])
        b = set(lineups[i])
        lsz = lineup_size or max(len(a), len(b))
        overlap = len(a & b)
        changes = lsz - overlap
        min_changes = min(min_changes, changes)
        if changes < min_uniques:
            viols.append((i, changes, overlap))
    return n - 1, (min_changes if min_changes != 1_000_000 else 0), viols


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify min-uniques across consecutive lineups from a CSV export.")
    ap.add_argument("--csv", required=True, help="Path to lineups CSV (with DK slot columns)")
    ap.add_argument("--min-uniques", type=int, required=True, help="Required minimum unique players between consecutive lineups")
    ap.add_argument("--lineup-size", type=int, default=8, help="Lineup size to assume when computing changes (default: 8 for DK)")
    ap.add_argument(
        "--slots",
        type=str,
        default=",".join(DK_SLOTS),
        help="Comma-separated slot column names present in the CSV (default: DK slots)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    slot_cols = [s.strip() for s in str(args.slots).split(",") if s.strip()]
    lineups = _load_lineups_from_csv(args.csv, slot_cols)
    if not lineups:
        print("No lineups found in CSV.")
        return 2

    n_checked, min_changes, viols = verify_consecutive_min_uniques(
        lineups, min_uniques=max(0, int(args.min_uniques)), lineup_size=max(1, int(args.lineup_size))
    )

    print(f"Checked {n_checked} consecutive pairs (lineups={len(lineups)}), min_uniques={args.min_uniques}, lineup_size={args.lineup_size}.")
    print(f"Minimum changes across consecutive pairs: {min_changes}")
    if not viols:
        print("OK: All consecutive pairs satisfy the min-uniques constraint.")
        return 0

    print(f"FAIL: {len(viols)} violation(s) found. First 10 shown:")
    for idx, (i, changes, overlap) in enumerate(viols[:10], start=1):
        print(f" {idx:2d}. pair=({i-1},{i}) changes={changes} overlap={overlap}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

