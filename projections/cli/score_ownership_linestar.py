"""
CLI for fetching ownership projections from LineStar.

Replaces score_ownership_live.py when using LineStar as the ownership source.
Fetches LineStar projected ownership, matches to DK salaries, and writes
output in the same format expected by downstream consumers.

Usage:
    uv run python -m projections.cli.score_ownership_linestar --date 2026-01-18 --run-id 20260118T120000Z
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from projections.runtime_safety import configure_runtime_safety

configure_runtime_safety()

import pandas as pd

from projections.names import normalize_player_name
from projections.paths import data_path

# Add scrapers to path for import
SCRAPERS_PATH = Path(__file__).parent.parent.parent / "scrapers"
if str(SCRAPERS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRAPERS_PATH))


def _normalize_name(name: str | None) -> str:
    """Normalize player name for matching."""
    return normalize_player_name(name) if name else ""


def _load_dk_salaries(
    game_date: date,
    data_root: Path,
) -> dict[str, pd.DataFrame]:
    """
    Load DK salaries for all draft groups on a date.

    Returns dict of {draft_group_id: DataFrame}.
    """
    base = data_root / "gold" / "dk_salaries" / "site=dk" / f"game_date={game_date}"

    if not base.exists():
        print(f"[linestar-own] No DK salary data at {base}")
        return {}

    draft_group_dirs = sorted(base.glob("draft_group_id=*"))
    if not draft_group_dirs:
        print(f"[linestar-own] No draft groups found for {game_date}")
        return {}

    slates = {}
    for dg_dir in draft_group_dirs:
        parquet_path = dg_dir / "salaries.parquet"
        if not parquet_path.exists():
            continue

        df = pd.read_parquet(parquet_path)
        dg_id = dg_dir.name.split("=")[1]
        df["draft_group_id"] = dg_id

        # Normalize columns for matching
        if "display_name" in df.columns:
            df["player_name"] = df["display_name"]
        if "team_abbrev" in df.columns:
            df["team"] = df["team_abbrev"]
        if "dk_player_id" in df.columns:
            df["player_id"] = df["dk_player_id"]
        if "positions" in df.columns:
            def _format_positions(x):
                if x is None:
                    return ""
                if isinstance(x, list):
                    return "/".join(str(p) for p in x)
                if hasattr(x, "tolist"):  # numpy array
                    return "/".join(str(p) for p in x.tolist())
                return str(x) if x else ""
            df["pos"] = df["positions"].apply(_format_positions)
        if "salary" in df.columns:
            df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)

        # Add normalized name for matching
        df["_name_norm"] = df["player_name"].apply(_normalize_name)

        slates[dg_id] = df

    print(f"[linestar-own] Loaded {len(slates)} DK slates for {game_date}")
    for dg_id, df in slates.items():
        teams = df["team"].unique() if "team" in df.columns else []
        print(f"  - {dg_id}: {len(df)} players, teams: {sorted(teams)[:6]}{'...' if len(teams) > 6 else ''}")

    return slates


def _select_main_dk_slate(slates: dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    """Select the DK main slate using salary coverage + size heuristics."""
    if not slates:
        raise RuntimeError("No DK slates available for selection")

    candidates: list[tuple[str, int, int]] = []
    for dg_id, df in slates.items():
        n_rows = int(len(df))
        n_salary = int(df["salary"].notna().sum()) if "salary" in df.columns else 0
        candidates.append((dg_id, n_rows, n_salary))

    has_salary = any(n_salary > 0 for _, _, n_salary in candidates)
    filtered = [c for c in candidates if c[2] > 0] if has_salary else candidates

    def _dg_key(val: str) -> int:
        try:
            return int(val)
        except ValueError:
            return 0

    filtered.sort(key=lambda x: (-x[2], -x[1], _dg_key(x[0])))
    main_id = filtered[0][0]
    return main_id, slates[main_id]


def _select_main_linestar_slate(
    ls_df: pd.DataFrame,
    dk_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    """Filter LineStar data to the slate that best matches the DK main slate."""
    if ls_df.empty or "ls_slate_id" not in ls_df.columns:
        return ls_df, None

    ls_subset = ls_df
    if "team" in ls_df.columns and "team" in dk_df.columns:
        dk_teams = set(dk_df["team"].dropna().astype(str).str.strip())
        if dk_teams:
            ls_subset = ls_df[ls_df["team"].isin(dk_teams)].copy()

    if ls_subset.empty:
        ls_subset = ls_df

    counts = ls_subset.groupby("ls_slate_id").size()
    if counts.empty:
        return ls_df, None

    main_slate = counts.idxmax()
    return ls_df[ls_df["ls_slate_id"] == main_slate].copy(), str(main_slate)


def _map_linestar_slates_to_dk(
    ls_df: pd.DataFrame,
    dk_slates: dict[str, pd.DataFrame],
) -> dict[str, str]:
    """Map each DK draft group to the best matching LineStar slate."""
    if ls_df.empty or "ls_slate_id" not in ls_df.columns or not dk_slates:
        return {}

    ls = ls_df.copy()
    ls["_name_norm"] = ls["player_name"].apply(_normalize_name)
    ls_groups: list[tuple[str, set[str], set[str]]] = []
    for slate_id, group in ls.groupby("ls_slate_id", sort=False):
        player_set = set(group["_name_norm"].dropna().astype(str))
        team_set = set(group["team"].dropna().astype(str).str.strip()) if "team" in group.columns else set()
        if player_set:
            ls_groups.append((str(slate_id), player_set, team_set))

    matches: dict[str, tuple[float, float, int, int, float, str]] = {}
    # (team_match, overlap, intersection, team_intersection, jaccard, ls_slate_id)
    for dg_id, dk_df in dk_slates.items():
        dk_set = set(dk_df["_name_norm"].dropna().astype(str))
        dk_teams = set(dk_df["team"].dropna().astype(str).str.strip()) if "team" in dk_df.columns else set()
        if not dk_set:
            continue

        best: tuple[float, float, int, int, float, str] | None = None
        for ls_slate_id, ls_set, ls_teams in ls_groups:
            inter = len(dk_set & ls_set)
            if inter == 0:
                continue
            union = len(dk_set | ls_set)
            overlap = inter / min(len(dk_set), len(ls_set))
            jaccard = inter / union if union else 0.0
            team_match = 1.0 if dk_teams and dk_teams == ls_teams else 0.0
            team_inter = len(dk_teams & ls_teams)
            score = (team_match, overlap, inter, team_inter, jaccard, ls_slate_id)
            if best is None or score[:5] > best[:5]:
                best = score

        if best is None:
            continue

        _, overlap, inter, _, jaccard, ls_slate_id = best
        required_intersection = min(20, max(2, len(dk_set) // 2))
        if overlap < 0.85 or inter < required_intersection:
            continue
        matches[dg_id] = best

    # Avoid assigning the same LineStar slate to multiple DK draft groups.
    best_for_ls: dict[str, tuple[str, tuple[float, float, int, int, float, str]]] = {}
    for dg_id, score in matches.items():
        ls_slate_id = score[5]
        prev = best_for_ls.get(ls_slate_id)
        if prev is None or score[:5] > prev[1][:5]:
            best_for_ls[ls_slate_id] = (dg_id, score)

    mapping = {dg_id: score[5] for _, (dg_id, score) in best_for_ls.items()}
    if mapping:
        print("[linestar-own] DK → LineStar slate mapping:")
        for dg_id, ls_slate_id in sorted(mapping.items()):
            score = matches[dg_id]
            print(
                f"  - DK {dg_id} -> LS {ls_slate_id} "
                f"(team_match={int(score[0])}, overlap={score[1]:.3f}, inter={score[2]}, jacc={score[4]:.3f})"
            )
    else:
        print("[linestar-own] No DK slates matched to LineStar slates")

    return mapping


def _match_linestar_to_dk(
    ls_df: pd.DataFrame,
    dk_df: pd.DataFrame,
    draft_group_id: str,
) -> pd.DataFrame:
    """
    Match LineStar ownership data to DK salaries.

    Uses player_name (normalized) + salary for robust matching.
    Falls back to name-only matching for any unmatched players.

    Returns DataFrame in ownership output format with DK player_ids.
    """
    # Normalize LineStar names
    ls_df = ls_df.copy()
    ls_df["_name_norm"] = ls_df["player_name"].apply(_normalize_name)
    ls_df["_salary"] = pd.to_numeric(ls_df["salary"], errors="coerce").fillna(0).astype(int)

    dk_df = dk_df.copy()
    dk_df["_salary"] = pd.to_numeric(dk_df["salary"], errors="coerce").fillna(0).astype(int)

    # Get teams in this DK slate
    dk_teams = set(dk_df["team"].unique()) if "team" in dk_df.columns else set()

    # Filter LineStar to players on teams in this slate
    if dk_teams and "team" in ls_df.columns:
        ls_slate = ls_df[ls_df["team"].isin(dk_teams)].copy()
    else:
        ls_slate = ls_df.copy()

    if ls_slate.empty:
        print(f"    No LineStar players match DK slate teams: {sorted(dk_teams)}")
        return pd.DataFrame()

    # Deduplicate LineStar data - same player may appear in multiple LS slates
    # Prefer the entry with the higher ownership (more relevant slate)
    ls_slate = (
        ls_slate.sort_values("proj_own_pct", ascending=False)
        .drop_duplicates(subset=["_name_norm", "_salary"], keep="first")
    )

    # Strategy 1: Match by normalized name + salary (most robust)
    merged = dk_df.merge(
        ls_slate[["_name_norm", "_salary", "proj_own_pct", "ls_proj_fpts", "ls_floor_fpts", "ls_ceil_fpts"]],
        on=["_name_norm", "_salary"],
        how="left",
    )

    matched_count = merged["proj_own_pct"].notna().sum()
    print(f"    Matched {matched_count}/{len(dk_df)} by name+salary")

    # Strategy 2: For unmatched, try name-only (handles salary differences)
    unmatched_mask = merged["proj_own_pct"].isna()
    if unmatched_mask.any():
        # Build name -> ownership map from LineStar (prefer lower salary if dupe names)
        ls_by_name = ls_slate.sort_values("_salary").groupby("_name_norm").first()

        for idx in merged[unmatched_mask].index:
            name = merged.loc[idx, "_name_norm"]
            if name in ls_by_name.index:
                merged.loc[idx, "proj_own_pct"] = ls_by_name.loc[name, "proj_own_pct"]
                merged.loc[idx, "ls_proj_fpts"] = ls_by_name.loc[name, "ls_proj_fpts"]
                merged.loc[idx, "ls_floor_fpts"] = ls_by_name.loc[name, "ls_floor_fpts"]
                merged.loc[idx, "ls_ceil_fpts"] = ls_by_name.loc[name, "ls_ceil_fpts"]

        final_matched = merged["proj_own_pct"].notna().sum()
        print(f"    After name-only fallback: {final_matched}/{len(dk_df)} matched")

    # Build output DataFrame in expected format
    output = pd.DataFrame({
        "player_id": merged["player_id"],
        "player_name": merged["player_name"],
        "salary": merged["salary"],
        "pos": merged["pos"],
        "team": merged["team"],
        "pred_own_pct": merged["proj_own_pct"],
        "draft_group_id": draft_group_id,
        # Include LineStar projections as bonus columns
        "ls_proj_fpts": merged.get("ls_proj_fpts"),
        "ls_floor_fpts": merged.get("ls_floor_fpts"),
        "ls_ceil_fpts": merged.get("ls_ceil_fpts"),
    })

    # Fill missing ownership with 0 (unmatched players get 0% ownership)
    output["pred_own_pct"] = output["pred_own_pct"].fillna(0.0)

    return output


def fetch_and_match_ownership(
    game_date: date,
    data_root: Path,
    *,
    run_id: str,
) -> dict[str, pd.DataFrame]:
    """
    Fetch LineStar ownership and match to all DK slates.

    Returns dict of {draft_group_id: ownership_df}.
    """
    from linestar.fetch_live_ownership import fetch_linestar_ownership

    # Load DK salaries
    dk_slates = _load_dk_salaries(game_date, data_root)
    if not dk_slates:
        return {}

    # Fetch LineStar ownership
    print(f"[linestar-own] Fetching LineStar ownership for {game_date}...")
    try:
        ls_df = fetch_linestar_ownership(site="dk", sport="nba", target_date=game_date)
    except Exception as e:
        print(f"[linestar-own] Failed to fetch LineStar data: {e}")
        return {}

    if ls_df.empty:
        print("[linestar-own] No LineStar ownership data returned")
        return {}

    print(f"[linestar-own] Got {len(ls_df)} LineStar rows across {ls_df['ls_slate_id'].nunique()} slates")

    slate_mapping = _map_linestar_slates_to_dk(ls_df, dk_slates)
    if not slate_mapping:
        return {}

    results: dict[str, pd.DataFrame] = {}
    scraped_at = datetime.now(UTC).isoformat()
    for dg_id, dk_df in dk_slates.items():
        ls_slate_id = slate_mapping.get(dg_id)
        if not ls_slate_id:
            print(f"[linestar-own] Skipping unmatched DK slate {dg_id}")
            continue

        ls_slate = ls_df[ls_df["ls_slate_id"].astype(str) == str(ls_slate_id)].copy()
        if ls_slate.empty:
            print(f"[linestar-own] LineStar slate {ls_slate_id} empty for DK slate {dg_id}")
            continue

        print(f"[linestar-own] Matching DK slate {dg_id} with LineStar slate {ls_slate_id}...")
        matched = _match_linestar_to_dk(ls_slate, dk_df, dg_id)
        if matched.empty:
            print(f"    Skipping empty slate {dg_id}")
            continue

        matched["game_date"] = game_date
        matched["run_id"] = run_id
        matched["model_run"] = "linestar"
        matched["is_locked"] = False
        matched["source"] = "linestar"
        matched["scraped_at"] = scraped_at
        matched["ls_slate_id"] = str(ls_slate_id)
        results[dg_id] = matched

        top = matched.nlargest(3, "pred_own_pct")[["player_name", "salary", "pred_own_pct"]]
        print(f"    Top 3: {top.values.tolist()}")

    return results


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Atomically write JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}.json")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch LineStar ownership projections")
    parser.add_argument("--date", required=True, help="Game date (YYYY-MM-DD)")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--data-root", default=None, help="Data root path")
    args = parser.parse_args()

    game_date = date.fromisoformat(args.date)
    root = Path(args.data_root) if args.data_root else data_path()

    # Fetch and match ownership
    results = fetch_and_match_ownership(game_date, root, run_id=args.run_id)

    if not results:
        print("[linestar-own] No ownership predictions generated")
        return 1

    # Save in same format as score_ownership_live.py
    out_dir = root / "silver" / "ownership_predictions" / str(game_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / f"run={args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    allow_legacy_flat = os.environ.get(
        "PROJECTIONS_ALLOW_LEGACY_OWNERSHIP_FLAT_WRITES", ""
    ).strip().lower() in {"1", "true", "yes"}

    for dg_id, df in results.items():
        out_path = run_dir / f"{dg_id}.parquet"
        df.to_parquet(out_path, index=False)
        if allow_legacy_flat:
            legacy_path = out_dir / f"{dg_id}.parquet"
            df.to_parquet(legacy_path, index=False)
        print(f"[linestar-own] Saved slate {dg_id}: {len(df)} predictions -> {out_path}")

    # Write latest_run.json pointer
    skip_pointers = os.environ.get(
        "PROJECTIONS_SKIP_POINTER_WRITES", ""
    ).strip().lower() in {"1", "true", "yes"}

    if not skip_pointers:
        from projections.pipeline import writer_guard

        writer_guard.assert_can_write_pointers(
            purpose=f"score_ownership_linestar promote {out_dir}"
        )
        latest_payload = {
            "run_id": args.run_id,
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "source": "linestar",
        }
        _atomic_write_json(out_dir / "latest_run.json", latest_payload)

    # Write slates.json metadata
    slates_meta = {}
    for dg_id, df in results.items():
        teams = list(df["team"].unique()) if "team" in df.columns else []
        slates_meta[dg_id] = {
            "player_count": len(df),
            "teams": sorted(teams),
            "source": "linestar",
            "is_locked": False,
        }
    _atomic_write_json(run_dir / "slates.json", slates_meta)

    # Print summary
    main_dg = max(results.keys(), key=lambda k: len(results[k]))
    main_df = results[main_dg]
    print(f"\n[linestar-own] Main slate ({main_dg}) top 5 by ownership:")
    print(main_df.nlargest(5, "pred_own_pct")[["player_name", "salary", "pred_own_pct"]].to_string())

    total_players = sum(len(df) for df in results.values())
    print(f"\n[linestar-own] Done: {len(results)} slates, {total_players} total players")

    return 0


if __name__ == "__main__":
    sys.exit(main())
