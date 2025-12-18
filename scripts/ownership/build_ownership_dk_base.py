"""
Build ownership training base from DK actual ownership data.

Joins DK contest ownership with Linestar projected data for features.
Uses player name matching to combine the datasets.

Inputs:
    bronze/dk_contests/ownership_by_slate/*.parquet  (DK actual ownership)
    gold/ownership_training_base/ownership_training_base.parquet (Linestar features)

Output:
    gold/ownership_dk_base/ownership_dk_base.parquet
"""

from __future__ import annotations

import argparse
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from projections.paths import data_path


def normalize_name(val: object) -> str:
    """Normalize player name for matching: strip accents, lowercase, trim."""
    if val is None or pd.isna(val):
        return ""
    normalized = unicodedata.normalize("NFKD", str(val))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_only.strip().lower()


def load_dk_ownership(dk_ownership_path: Path) -> pd.DataFrame:
    """Load all DK ownership parquet files."""
    all_files = sorted(dk_ownership_path.glob("*.parquet"))

    # Exclude the combined file if present
    all_files = [f for f in all_files if not f.name.startswith("all_")]

    if not all_files:
        raise FileNotFoundError(f"No ownership files found in {dk_ownership_path}")

    print(f"Loading {len(all_files)} DK ownership files...")

    dfs = []
    for f in all_files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: failed to read {f.name}: {e}")

    result = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(result):,} rows from DK ownership")

    return result


def load_linestar_features(training_base_path: Path) -> pd.DataFrame:
    """Load Linestar training base for feature matching."""
    if not training_base_path.exists():
        raise FileNotFoundError(f"Training base not found: {training_base_path}")

    df = pd.read_parquet(training_base_path)
    print(f"Loaded {len(df):,} rows from Linestar training base")

    return df


@dataclass(frozen=True)
class SlateMatch:
    """Best-effort mapping between a DK slate and a LineStar slate."""

    dk_slate_id: str
    dk_game_date: str
    linestar_slate_id: str | None
    linestar_game_date: str | None
    dk_players: int
    linestar_players: int
    intersection: int
    recall_dk: float
    recall_linestar: float
    overlap_coeff: float
    jaccard: float
    date_offset_days: int | None


def _candidate_game_dates(game_date_str: str, *, max_day_offset: int = 1) -> list[str]:
    """Return candidate date strings allowing for timezone/late-slate drift."""

    base = date.fromisoformat(game_date_str)
    candidates = [base + timedelta(days=d) for d in range(-max_day_offset, max_day_offset + 1)]
    return [d.isoformat() for d in candidates]


def _build_linestar_slate_index(linestar: pd.DataFrame) -> pd.DataFrame:
    """Build per-slate player sets for slate-aware matching."""

    required = {"slate_id", "game_date", "player_name_norm"}
    missing = sorted(required - set(linestar.columns))
    if missing:
        raise KeyError(f"Linestar features missing required columns: {missing}")

    working = linestar[["slate_id", "game_date", "player_name_norm"]].copy()
    working["slate_id"] = working["slate_id"].astype(str)
    working["game_date"] = working["game_date"].astype(str)
    working["player_name_norm"] = working["player_name_norm"].astype(str)
    working = working[working["player_name_norm"].ne("")].copy()

    def _to_set(s: pd.Series) -> set[str]:
        return set(s.tolist())

    idx = (
        working.groupby("slate_id", sort=False)
        .agg(game_date=("game_date", "first"), player_set=("player_name_norm", _to_set), n_players=("player_name_norm", "size"))
        .reset_index()
    )
    return idx


def match_dk_slates_to_linestar(
    dk_own: pd.DataFrame,
    linestar: pd.DataFrame,
    *,
    max_day_offset: int = 1,
    min_overlap_coeff: float = 0.85,
    min_intersection: int = 20,
) -> tuple[pd.DataFrame, list[SlateMatch]]:
    """Map each DK slate_id to the best matching LineStar slate_id via player overlap.

    Avoids ambiguous `(game_date, player)` joins and handles common game_date drift between
    DK contest dates and LineStar game_date (e.g. late slates rolling over to next UTC day).
    """

    required = {"slate_id", "game_date", "player_name_norm"}
    missing = sorted(required - set(dk_own.columns))
    if missing:
        raise KeyError(f"DK ownership missing required columns: {missing}")

    linestar_idx = _build_linestar_slate_index(linestar)
    by_date: dict[str, list[tuple[str, set[str], int]]] = {}
    for _, row in linestar_idx.iterrows():
        game_date = str(row["game_date"])
        if not game_date or game_date.lower() == "nan":
            continue
        by_date.setdefault(game_date, []).append(
            (str(row["slate_id"]), set(row["player_set"]), int(row["n_players"]))
        )

    dk = dk_own[["slate_id", "game_date", "player_name_norm"]].copy()
    dk["slate_id"] = dk["slate_id"].astype(str)
    dk["game_date"] = dk["game_date"].astype(str)
    dk["player_name_norm"] = dk["player_name_norm"].astype(str)
    dk = dk[dk["player_name_norm"].ne("")].copy()

    mapping: dict[str, str] = {}
    matches: list[SlateMatch] = []

    for dk_slate_id, g in dk.groupby("slate_id", sort=False):
        dk_date = str(g["game_date"].iloc[0])
        try:
            _ = date.fromisoformat(dk_date)
        except ValueError:
            matches.append(
                SlateMatch(
                    dk_slate_id=dk_slate_id,
                    dk_game_date=dk_date,
                    linestar_slate_id=None,
                    linestar_game_date=None,
                    dk_players=int(g["player_name_norm"].nunique()),
                    linestar_players=0,
                    intersection=0,
                    recall_dk=0.0,
                    recall_linestar=0.0,
                    overlap_coeff=0.0,
                    jaccard=0.0,
                    date_offset_days=None,
                )
            )
            continue

        dk_set = set(g["player_name_norm"].tolist())
        dk_n = len(dk_set)

        best: tuple[float, int, int, float, str, str, int, float, float, int] | None = None
        # (
        #   overlap_coeff, intersection, neg_abs_offset, jaccard,
        #   linestar_slate_id, linestar_date, linestar_n,
        #   recall_dk, recall_linestar, date_offset
        # )

        for cand_date in _candidate_game_dates(dk_date, max_day_offset=max_day_offset):
            candidates = by_date.get(cand_date, [])
            for ls_slate_id, ls_set, ls_n in candidates:
                inter = len(dk_set & ls_set)
                if inter == 0:
                    continue
                union = len(dk_set | ls_set)
                jacc = inter / union if union else 0.0
                rec_dk = inter / dk_n if dk_n else 0.0
                rec_ls = inter / ls_n if ls_n else 0.0
                denom = min(dk_n, ls_n)
                overlap = inter / denom if denom else 0.0
                offset = (date.fromisoformat(cand_date) - date.fromisoformat(dk_date)).days

                cand = (overlap, inter, -abs(offset), jacc, ls_slate_id, cand_date, ls_n, rec_dk, rec_ls, offset)
                if best is None:
                    best = cand
                    continue

                # Prefer higher overlap coefficient (subset-aware), then higher intersection,
                # then closer date alignment, then higher Jaccard.
                if cand[:4] > best[:4]:
                    best = cand

        if best is None:
            matches.append(
                SlateMatch(
                    dk_slate_id=dk_slate_id,
                    dk_game_date=dk_date,
                    linestar_slate_id=None,
                    linestar_game_date=None,
                    dk_players=dk_n,
                    linestar_players=0,
                    intersection=0,
                    recall_dk=0.0,
                    recall_linestar=0.0,
                    overlap_coeff=0.0,
                    jaccard=0.0,
                    date_offset_days=None,
                )
            )
            continue

        overlap, inter, _, jacc, ls_slate_id, ls_date, ls_n, rec_dk, rec_ls, offset = best
        if overlap < min_overlap_coeff or inter < min_intersection:
            matches.append(
                SlateMatch(
                    dk_slate_id=dk_slate_id,
                    dk_game_date=dk_date,
                    linestar_slate_id=None,
                    linestar_game_date=None,
                    dk_players=dk_n,
                    linestar_players=ls_n,
                    intersection=inter,
                    recall_dk=float(rec_dk),
                    recall_linestar=float(rec_ls),
                    overlap_coeff=float(overlap),
                    jaccard=float(jacc),
                    date_offset_days=int(offset),
                )
            )
            continue

        mapping[dk_slate_id] = ls_slate_id
        matches.append(
            SlateMatch(
                dk_slate_id=dk_slate_id,
                dk_game_date=dk_date,
                linestar_slate_id=ls_slate_id,
                linestar_game_date=ls_date,
                dk_players=dk_n,
                linestar_players=ls_n,
                intersection=inter,
                recall_dk=float(rec_dk),
                recall_linestar=float(rec_ls),
                overlap_coeff=float(overlap),
                jaccard=float(jacc),
                date_offset_days=int(offset),
            )
        )

    # Resolve collisions: avoid mapping multiple DK slates to the same LineStar slate.
    # Keep only the best-overlap DK slate per LineStar slate_id.
    best_for_ls: dict[str, SlateMatch] = {}
    for m in matches:
        if m.linestar_slate_id is None:
            continue
        prev = best_for_ls.get(m.linestar_slate_id)
        if prev is None:
            best_for_ls[m.linestar_slate_id] = m
            continue
        score = (m.overlap_coeff, m.intersection, -(abs(m.date_offset_days or 0)), m.jaccard)
        prev_score = (prev.overlap_coeff, prev.intersection, -(abs(prev.date_offset_days or 0)), prev.jaccard)
        if score > prev_score:
            best_for_ls[m.linestar_slate_id] = m

    allowed_dk = {m.dk_slate_id for m in best_for_ls.values()}
    for dk_slate_id in list(mapping.keys()):
        if dk_slate_id not in allowed_dk:
            mapping.pop(dk_slate_id, None)

    out = dk_own.copy()
    out["linestar_slate_id"] = out["slate_id"].astype(str).map(mapping)
    final_matches: list[SlateMatch] = []
    for m in matches:
        if m.linestar_slate_id is None or m.dk_slate_id in allowed_dk:
            final_matches.append(m)
            continue
        final_matches.append(
            SlateMatch(
                dk_slate_id=m.dk_slate_id,
                dk_game_date=m.dk_game_date,
                linestar_slate_id=None,
                linestar_game_date=None,
                dk_players=m.dk_players,
                linestar_players=m.linestar_players,
                intersection=m.intersection,
                recall_dk=m.recall_dk,
                recall_linestar=m.recall_linestar,
                overlap_coeff=m.overlap_coeff,
                jaccard=m.jaccard,
                date_offset_days=m.date_offset_days,
            )
        )
    return out, final_matches


def build_dk_base(
    dk_ownership_path: Path,
    linestar_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Build training base from DK ownership + Linestar features.

    Strategy:
    1. Load DK ownership (actual ownership from DK contests)
    2. Load Linestar features (salary, proj_fpts, etc.)
    3. Match DK slate -> LineStar slate by player-pool overlap (handles multi-slate dates + timezone drift)
    4. Match players within the mapped slate by normalized name
    4. Use DK actual ownership as label, Linestar for features
    """
    # Load data
    dk_own = load_dk_ownership(dk_ownership_path)
    linestar = load_linestar_features(linestar_path)

    # Normalize names
    dk_own["player_name_norm"] = dk_own["Player"].apply(normalize_name)

    # Ensure linestar has normalized names
    if "player_name_norm" not in linestar.columns:
        linestar["player_name_norm"] = linestar["player_name"].apply(normalize_name)

    # Slate-aware matching to avoid ambiguous (date, player) joins and handle date drift.
    dk_own, slate_matches = match_dk_slates_to_linestar(
        dk_own,
        linestar,
        max_day_offset=1,
        min_overlap_coeff=0.85,
        min_intersection=20,
    )

    matched_slates = [m for m in slate_matches if m.linestar_slate_id is not None]
    unmatched_slates = [m for m in slate_matches if m.linestar_slate_id is None]

    print("\nSlate matching (DK -> LineStar):")
    print(f"  DK slates: {len(slate_matches):,}")
    print(f"  Matched slates: {len(matched_slates):,}")
    print(f"  Unmatched slates: {len(unmatched_slates):,}")
    if matched_slates:
        offsets = [m.date_offset_days or 0 for m in matched_slates]
        print(f"  Date offsets (days) among matches: {pd.Series(offsets).value_counts().to_dict()}")

    # Feature columns to bring from LineStar (join keys handled separately).
    linestar_feature_cols = [
        "season",
        "player_id",
        "player_name",
        "team",
        "pos",
        "salary",
        "proj_fpts",
        "floor_fpts",
        "ceil_fpts",
        "conf",
        "value_per_k",
        "ppg",
        "matchup",
        "home_team",
        "away_team",
        "opp_rank",
        "opp_total",
        "player_is_out",
        "player_is_questionable",
        "team_outs_count",
        "proj_own_pct",  # Linestar projection for comparison
    ]

    # Only include columns that exist
    linestar_feature_cols = [c for c in linestar_feature_cols if c in linestar.columns]

    linestar_for_join = linestar.copy()
    linestar_for_join["linestar_slate_id"] = linestar_for_join["slate_id"].astype(str)

    # Join
    before = len(dk_own)
    dk_own = dk_own[dk_own["linestar_slate_id"].notna()].copy()

    join_cols = ["linestar_slate_id", "player_name_norm"] + linestar_feature_cols
    merged = dk_own.merge(
        linestar_for_join[join_cols],
        on=["linestar_slate_id", "player_name_norm"],
        how="inner",  # Only keep matches (features required for training)
    )

    print("\nMatching data:")
    print(f"  DK ownership rows: {before:,}")
    print(f"  DK rows in matched slates: {len(dk_own):,}")
    print(f"  Matched rows (features joined): {len(merged):,} ({len(merged)/before*100:.1f}% of DK data)")

    # Rename DK ownership columns (keep slate_id for compatibility)
    merged = merged.rename(columns={
        "own_pct": "actual_own_pct",
        "slate_size": "dk_slate_size",
        "FPTS": "scored_fpts_dk",
    })

    # Add data source marker
    merged["data_source"] = "dk"

    # Select output columns
    output_cols = [
        # Identifiers
        "season",
        "slate_id",
        "game_date",
        "player_id",
        "player_name",
        "player_name_norm",
        "team",
        "pos",
        # Salaries
        "salary",
        # Projections (from Linestar)
        "proj_fpts",
        "floor_fpts",
        "ceil_fpts",
        "conf",
        "value_per_k",
        "ppg",
        # Vegas context
        "matchup",
        "home_team",
        "away_team",
        "opp_rank",
        "opp_total",
        # Injury context
        "player_is_out",
        "player_is_questionable",
        "team_outs_count",
        # Ownership
        "proj_own_pct",
        "actual_own_pct",
        # DK-specific
        "dk_slate_size",
        "scored_fpts_dk",
        "entries",
        "num_contests",
        # Source
        "data_source",
    ]

    output_cols = [c for c in output_cols if c in merged.columns]
    result = merged[output_cols].copy()

    # Filter out broken data
    # Remove rows with extreme ownership (>98% usually indicates data issues)
    before_filter = len(result)
    result = result[result["actual_own_pct"] <= 98.0].copy()
    print(f"\nFiltered: {before_filter - len(result):,} rows with ownership > 98%")

    # Summary
    print("\n--- DK Base Summary ---")
    print(f"Total rows: {len(result):,}")
    print(f"Unique slates: {result['slate_id'].nunique():,}")
    print(f"Unique players: {result['player_id'].nunique():,}")
    print(f"Date range: {result['game_date'].min()} to {result['game_date'].max()}")
    print(f"Mean ownership: {result['actual_own_pct'].mean():.2f}%")
    if "dk_slate_size" in result.columns:
        print(f"Slate size: mean={result['dk_slate_size'].mean():.0f}, median={result['dk_slate_size'].median():.0f}")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"\nWrote {len(result):,} rows to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Build DK ownership training base")
    parser.add_argument(
        "--dk-ownership-path",
        type=Path,
        default=None,
        help="Path to DK ownership parquet files",
    )
    parser.add_argument(
        "--linestar-path",
        type=Path,
        default=None,
        help="Path to Linestar training base parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path",
    )

    args = parser.parse_args()

    # Defaults
    if args.dk_ownership_path is None:
        args.dk_ownership_path = data_path() / "bronze" / "dk_contests" / "ownership_by_slate"

    if args.linestar_path is None:
        args.linestar_path = data_path() / "gold" / "ownership_training_base" / "ownership_training_base.parquet"

    if args.output is None:
        args.output = data_path() / "gold" / "ownership_dk_base" / "ownership_dk_base.parquet"

    build_dk_base(
        dk_ownership_path=args.dk_ownership_path,
        linestar_path=args.linestar_path,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
