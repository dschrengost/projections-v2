"""
Build usage shares training base for learning within-team opportunity allocation.

Inputs:
  --start-date YYYY-MM-DD
  --end-date YYYY-MM-DD
  --data-root /path/to/projections-data

Source:
  gold/rates_training_base/season=*/game_date=*/rates_training_base.parquet

Labels (derived from rates_training_base):
  fga = (fga2_per_min + fga3_per_min) * minutes_actual
  fta = fta_per_min * minutes_actual
  tov = tov_per_min * minutes_actual
  share_x = x / sum_team(x)  for x in {fga, fta, tov}

Features (MVP, pregame-safe):
  - minutes_pred_p50, minutes_pred_play_prob, is_starter
  - position flags (position_flags_PG, ..., position_flags_C)
  - vegas total/spread (total_close, spread_close) - from rates_training_base
  - vacancy aggregates (vac_min_szn, vac_fga_szn, ...)
  - season-to-date rates (season_fga_per_min, season_3pa_per_min, ...)
  - tracking role features (track_role_cluster, etc.)

Anti-leak enforcement:
  - tip_ts is joined from silver/schedule
  - All features in rates_training_base are already pregame-safe (as_of_ts <= tip_ts)
  - Leak-safe flag computed per row

Output:
  gold/usage_shares_training_base/season=YYYY/game_date=YYYY-MM-DD/usage_shares_training_base.parquet

Usage:
  uv run python -m scripts.usage_shares_v1.build_training_base \\
      --start-date 2024-10-01 \\
      --end-date 2025-12-01 \\
      --data-root /home/daniel/projections-data
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.paths import data_path

app = typer.Typer(add_completion=False, help=__doc__)


# Minimum minutes to include a player in training data
MIN_MINUTES_ACTUAL = 4.0

# Overwrite existing partitions
OVERWRITE_PARTITIONS = True


def _season_from_date(d: pd.Timestamp) -> int:
    """NBA season start year (Aug-Jul)."""
    return d.year if d.month >= 8 else d.year - 1


def _iter_days(start: pd.Timestamp, end: pd.Timestamp):
    """Iterate over days in [start, end] inclusive."""
    day = start.normalize()
    end_day = end.normalize()
    while day <= end_day:
        yield day
        day += pd.Timedelta(days=1)


def _load_rates_training_base(
    data_root: Path, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Load rates_training_base partitions for the date range."""
    root = data_root / "gold" / "rates_training_base"
    frames: list[pd.DataFrame] = []
    for season_dir in root.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if day < start or day > end:
                continue
            path = day_dir / "rates_training_base.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(
            f"No rates_training_base partitions found for {start.date()}..{end.date()}"
        )
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


def _load_odds_from_minutes_artifacts(
    data_root: Path, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """
    Load odds from minutes artifacts as fallback when rates_training_base has gaps.
    
    TECH DEBT: This exists because silver/odds_snapshot has gaps for Dec 2025.
    The proper fix is to backfill odds_snapshot and ensure the daily ETL writes
    all odds to silver. See docs/pipeline/tech_debt.md for details.
    
    This scans all runs for each day and picks the LATEST pre-tip odds per game,
    ensuring we get the most accurate closing lines (odds move throughout the day
    as injuries and other news come out).
    """
    # Sources to check (in priority order)
    sources = [
        data_root / "gold" / "projections_minutes_v1",
        Path("/home/daniel/projects/projections-v2/artifacts/minutes_v1/daily"),
    ]
    
    # Track best odds per game: game_id -> (odds_as_of_ts, spread, total, team_itt, opp_itt)
    best_odds: dict[int, tuple] = {}
    
    for day in pd.date_range(start, end, freq="D"):
        day_str = day.date().isoformat()
        
        for source in sources:
            if source.name == "projections_minutes_v1":
                day_dir = source / f"game_date={day_str}"
            else:
                day_dir = source / day_str
            
            if not day_dir.exists():
                continue
            
            # Scan ALL runs to find the latest pre-tip odds per game
            for run_dir in day_dir.glob("run=*"):
                minutes_path = run_dir / "minutes.parquet"
                if not minutes_path.exists():
                    continue
                
                try:
                    df = pd.read_parquet(minutes_path)
                    if "spread_home" not in df.columns or "total" not in df.columns:
                        continue
                    
                    # Parse timestamps
                    if "odds_as_of_ts" in df.columns:
                        df["odds_as_of_ts"] = pd.to_datetime(df["odds_as_of_ts"], utc=True, errors="coerce")
                    else:
                        df["odds_as_of_ts"] = pd.NaT
                    
                    if "tip_ts" in df.columns:
                        df["tip_ts"] = pd.to_datetime(df["tip_ts"], utc=True, errors="coerce")
                    else:
                        df["tip_ts"] = pd.NaT
                    
                    # Check each game
                    for game_id in df["game_id"].unique():
                        game_df = df[df["game_id"] == game_id]
                        row = game_df.iloc[0]
                        
                        if pd.isna(row.get("spread_home")) or pd.isna(row.get("total")):
                            continue
                        
                        # Prefer pre-tip odds (odds_as_of_ts < tip_ts)
                        odds_ts = row.get("odds_as_of_ts")
                        tip_ts = row.get("tip_ts")
                        is_pretip = pd.notna(odds_ts) and pd.notna(tip_ts) and odds_ts < tip_ts
                        
                        game_id_int = int(game_id)
                        current = best_odds.get(game_id_int)
                        
                        # Update if:
                        # 1. We don't have this game yet
                        # 2. Current is not pre-tip but new one is
                        # 3. Both are pre-tip and new one is more recent
                        should_update = (
                            current is None
                            or (is_pretip and (current[5] is False or (pd.notna(odds_ts) and odds_ts > current[0])))
                        )
                        
                        if should_update:
                            best_odds[game_id_int] = (
                                odds_ts,
                                float(row["spread_home"]),
                                float(row["total"]),
                                float(row["team_implied_total"]) if "team_implied_total" in row.index and pd.notna(row.get("team_implied_total")) else np.nan,
                                float(row["opponent_implied_total"]) if "opponent_implied_total" in row.index and pd.notna(row.get("opponent_implied_total")) else np.nan,
                                is_pretip,
                            )
                except Exception:
                    continue
    
    if not best_odds:
        return pd.DataFrame()
    
    records = [
        {
            "game_id": gid,
            "spread_close": vals[1],
            "total_close": vals[2],
            "team_itt": vals[3],
            "opp_itt": vals[4],
            "odds_as_of_ts": vals[0],  # For leak-safety verification
        }
        for gid, vals in best_odds.items()
    ]
    return pd.DataFrame(records)


def _load_schedule_tips(data_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Load schedule to get tip_ts per game_id."""
    schedule_root = data_root / "silver" / "schedule"
    frames: list[pd.DataFrame] = []
    for season_dir in schedule_root.glob("season=*"):
        for month_dir in season_dir.glob("month=*"):
            path = month_dir / "schedule.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=["game_id", "tip_ts"])
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    # Filter to date range with some buffer for games that span dates
    df = df[(df["game_date"] >= start - pd.Timedelta(days=1)) & (df["game_date"] <= end + pd.Timedelta(days=1))]
    # Ensure tip_ts is UTC
    if "tip_ts" in df.columns:
        df["tip_ts"] = pd.to_datetime(df["tip_ts"], utc=True, errors="coerce")
    return df[["game_id", "tip_ts"]].drop_duplicates(subset=["game_id"])


def _safe_divide(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Safe division with 0/NaN handling."""
    denom_safe = denom.replace(0, np.nan)
    out = numer / denom_safe
    return out.fillna(0.0)


def build_usage_shares_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute opportunity totals and within-team shares.

    Labels:
      fga = (fga2_per_min + fga3_per_min) * minutes_actual
      fta = fta_per_min * minutes_actual
      tov = tov_per_min * minutes_actual
      share_x = x / sum_team(x)
    """
    required = {
        "game_id",
        "team_id",
        "player_id",
        "minutes_actual",
        "fga2_per_min",
        "fga3_per_min",
        "fta_per_min",
        "tov_per_min",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    working = df.copy()

    # Coerce to numeric
    for col in ("minutes_actual", "fga2_per_min", "fga3_per_min", "fta_per_min", "tov_per_min"):
        working[col] = pd.to_numeric(working[col], errors="coerce").fillna(0.0)

    # Compute opportunity totals
    working["fga2"] = (working["fga2_per_min"] * working["minutes_actual"]).clip(lower=0.0)
    working["fga3"] = (working["fga3_per_min"] * working["minutes_actual"]).clip(lower=0.0)
    working["fga"] = (working["fga2"] + working["fga3"]).clip(lower=0.0)
    working["fta"] = (working["fta_per_min"] * working["minutes_actual"]).clip(lower=0.0)
    working["tov"] = (working["tov_per_min"] * working["minutes_actual"]).clip(lower=0.0)

    # Compute team totals
    team_keys = ["game_id", "team_id"]
    team_totals = (
        working.groupby(team_keys, as_index=False)[["fga", "fta", "tov", "minutes_actual"]]
        .sum()
        .rename(
            columns={
                "fga": "team_fga",
                "fta": "team_fta",
                "tov": "team_tov",
                "minutes_actual": "team_minutes",
            }
        )
    )
    working = working.merge(team_totals, on=team_keys, how="left")

    # Compute shares with validity flags
    # When team total is 0 or NaN, share is invalid (set to 0 with valid=False)
    eps = 1e-9
    
    # FGA shares
    fga_valid = working["team_fga"] > eps
    working["share_fga"] = np.where(fga_valid, working["fga"] / working["team_fga"], 0.0)
    working["share_fga_valid"] = fga_valid
    
    # FTA shares
    fta_valid = working["team_fta"] > eps
    working["share_fta"] = np.where(fta_valid, working["fta"] / working["team_fta"], 0.0)
    working["share_fta_valid"] = fta_valid
    
    # TOV shares
    tov_valid = working["team_tov"] > eps
    working["share_tov"] = np.where(tov_valid, working["tov"] / working["team_tov"], 0.0)
    working["share_tov_valid"] = tov_valid

    return working


def build_training_base(
    rates_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the usage shares training base.

    1. Filter to MIN_MINUTES_ACTUAL
    2. Compute labels (opportunity totals + shares)
    3. Join tip_ts from schedule
    4. Select features + labels + metadata
    """
    # Filter to minimum minutes
    df = rates_df[rates_df["minutes_actual"] >= MIN_MINUTES_ACTUAL].copy()
    if df.empty:
        return pd.DataFrame()

    # Compute labels
    df = build_usage_shares_labels(df)

    # Join tip_ts from schedule
    df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
    schedule_df["game_id"] = pd.to_numeric(schedule_df["game_id"], errors="coerce").astype("Int64")
    df = df.merge(schedule_df[["game_id", "tip_ts"]], on="game_id", how="left")

    # Leak-safe flag: tip_ts is present (features are derived from as_of <= tip_ts sources)
    df["tip_ts_present"] = df["tip_ts"].notna()
    # Mark rows where we have tip_ts as leak_safe
    # All features in rates_training_base are already pregame-safe by construction
    df["leak_safe"] = df["tip_ts_present"]

    # Select output columns
    # Keys
    key_cols = [
        "season",
        "game_id",
        "game_date",
        "team_id",
        "opponent_id",
        "home_flag",
        "player_id",
    ]

    # Features (pregame-safe)
    feature_cols = [
        # Minutes predictions
        "minutes_pred_p10",
        "minutes_pred_p50",
        "minutes_pred_p90",
        "minutes_pred_play_prob",
        # Role
        "is_starter",
        # Position
        "position_primary",
        "position_flags_PG",
        "position_flags_SG",
        "position_flags_SF",
        "position_flags_PF",
        "position_flags_C",
        # Vegas
        "spread_close",
        "total_close",
        "team_itt",
        "opp_itt",
        "has_odds",
        # Season-to-date rates (prior to this game)
        "season_fga_per_min",
        "season_3pa_per_min",
        "season_fta_per_min",
        "season_ast_per_min",
        "season_tov_per_min",
        "season_reb_per_min",
        "season_stl_per_min",
        "season_blk_per_min",
        # Vacancy features (team-level injury impacts)
        "vac_min_szn",
        "vac_fga_szn",
        "vac_ast_szn",
        "vac_min_guard_szn",
        "vac_min_wing_szn",
        "vac_min_big_szn",
        "team_minutes_vacated",
        "team_usage_vacated",
        # Team context
        "team_pace_szn",
        "team_off_rtg_szn",
        "team_def_rtg_szn",
        "opp_pace_szn",
        "opp_def_rtg_szn",
        # Days rest
        "days_rest",
        # Tracking role features (if available)
        "track_touches_per_min_szn",
        "track_sec_per_touch_szn",
        "track_pot_ast_per_min_szn",
        "track_drives_per_min_szn",
        "track_role_cluster",
        "track_role_is_low_minutes",
    ]

    # Labels (actual outcomes)
    label_cols = [
        "minutes_actual",
        # Per-minute rates (labels for rates model, useful as features for shares)
        "fga2_per_min",
        "fga3_per_min",
        "fta_per_min",
        "tov_per_min",
        "ast_per_min",
        "oreb_per_min",
        "dreb_per_min",
        "stl_per_min",
        "blk_per_min",
        # Opportunity totals
        "fga2",
        "fga3",
        "fga",
        "fta",
        "tov",
        # Team totals
        "team_fga",
        "team_fta",
        "team_tov",
        "team_minutes",
        # Shares (target labels for share model)
        "share_fga",
        "share_fta",
        "share_tov",
    ]

    # Metadata
    meta_cols = [
        "tip_ts",
        "tip_ts_present",
        "leak_safe",
        "odds_as_of_ts",  # For leak-safety verification
        # Share validity flags (False when team total = 0)
        "share_fga_valid",
        "share_fta_valid",
        "share_tov_valid",
    ]

    # Filter to columns that exist
    all_cols = key_cols + feature_cols + label_cols + meta_cols
    existing_cols = [c for c in all_cols if c in df.columns]

    return df[existing_cols]


def _write_partitions(df: pd.DataFrame, output_root: Path) -> int:
    """Write partitioned parquet files. Returns count of partitions written."""
    output_root.mkdir(parents=True, exist_ok=True)
    df["season"] = df["season"].astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()

    partitions_written = 0
    grouped = df.groupby(["season", "game_date"])
    for (season, game_date), frame in grouped:
        partition_dir = (
            output_root
            / f"season={int(season)}"
            / f"game_date={pd.Timestamp(game_date).date().isoformat()}"
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        output_path = partition_dir / "usage_shares_training_base.parquet"
        if output_path.exists() and not OVERWRITE_PARTITIONS:
            typer.echo(f"[usage_shares] skipping existing partition {output_path}")
            continue
        frame.to_parquet(output_path, index=False)
        partitions_written += 1

    return partitions_written


@app.command()
def main(
    start_date: str = typer.Option(..., help="Start date (YYYY-MM-DD) inclusive."),
    end_date: str = typer.Option(..., help="End date (YYYY-MM-DD) inclusive."),
    data_root: Optional[Path] = typer.Option(
        None,
        help="Root containing bronze/silver/gold data (defaults to PROJECTIONS_DATA_ROOT).",
    ),
    output_root: Optional[Path] = typer.Option(
        None,
        help="Output root for gold/usage_shares_training_base (defaults under data_root/gold).",
    ),
) -> None:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    root = data_root or data_path()
    out_root = output_root or (root / "gold" / "usage_shares_training_base")

    typer.echo(f"[usage_shares] building training base from {start.date()} to {end.date()}")
    typer.echo(f"[usage_shares] data_root={root}")
    typer.echo(f"[usage_shares] output_root={out_root}")

    # Load rates_training_base
    typer.echo("[usage_shares] loading rates_training_base...")
    rates_df = _load_rates_training_base(root, start, end)
    typer.echo(f"[usage_shares] rates_training_base: {len(rates_df):,} rows")

    # Check for missing vegas features and apply fallback if needed
    vegas_cols = ["spread_close", "total_close"]
    vegas_missing_rate = 0.0
    for col in vegas_cols:
        if col in rates_df.columns:
            vegas_missing_rate = max(vegas_missing_rate, rates_df[col].isna().mean())
        else:
            vegas_missing_rate = 1.0
    
    if vegas_missing_rate > 0.5:
        typer.echo(f"[usage_shares] WARNING: vegas features {vegas_missing_rate*100:.1f}% missing in rates_training_base")
        typer.echo("[usage_shares] attempting fallback to minutes artifacts for odds...")
        odds_fallback = _load_odds_from_minutes_artifacts(root, start, end)
        if not odds_fallback.empty:
            typer.echo(f"[usage_shares] loaded {len(odds_fallback):,} games with odds from artifacts")
            # Merge fallback odds where missing
            rates_df["game_id"] = pd.to_numeric(rates_df["game_id"], errors="coerce").astype("Int64")
            odds_fallback["game_id"] = pd.to_numeric(odds_fallback["game_id"], errors="coerce").astype("Int64")
            for col in ["spread_close", "total_close", "team_itt", "opp_itt"]:
                if col not in rates_df.columns:
                    rates_df[col] = np.nan
            if "odds_as_of_ts" not in rates_df.columns:
                rates_df["odds_as_of_ts"] = pd.Series(pd.NaT, index=rates_df.index, dtype="datetime64[ns, UTC]")
            # Create a mapping from fallback
            fallback_map = odds_fallback.set_index("game_id")
            # Capture mask BEFORE filling any columns (rows where spread_close is missing)
            needs_odds_mask = rates_df["spread_close"].isna()
            for col in ["spread_close", "total_close", "team_itt", "opp_itt", "odds_as_of_ts"]:
                if col in fallback_map.columns:
                    rates_df.loc[needs_odds_mask, col] = rates_df.loc[needs_odds_mask, "game_id"].map(fallback_map[col])
            # Update has_odds
            rates_df["has_odds"] = rates_df["spread_close"].notna() & rates_df["total_close"].notna()
            new_coverage = rates_df["spread_close"].notna().mean() * 100
            typer.echo(f"[usage_shares] after fallback: spread_close coverage={new_coverage:.1f}%")
        else:
            typer.echo("[usage_shares] WARNING: no odds found in minutes artifacts either")

    # Load schedule for tip_ts
    typer.echo("[usage_shares] loading schedule for tip_ts...")
    schedule_df = _load_schedule_tips(root, start, end)
    typer.echo(f"[usage_shares] schedule: {len(schedule_df):,} games with tip_ts")

    # Build training base
    typer.echo("[usage_shares] building training base...")
    training_df = build_training_base(rates_df, schedule_df)
    if training_df.empty:
        typer.echo("[usage_shares] WARNING: no rows after filtering")
        raise typer.Exit(1)

    # Diagnostics
    n_total = len(training_df)
    n_leak_safe = training_df["leak_safe"].sum() if "leak_safe" in training_df.columns else 0
    n_missing_tip = n_total - n_leak_safe
    typer.echo(f"[usage_shares] total rows: {n_total:,}")
    typer.echo(f"[usage_shares] leak_safe rows: {n_leak_safe:,} ({100*n_leak_safe/n_total:.1f}%)")
    typer.echo(f"[usage_shares] missing tip_ts: {n_missing_tip:,}")

    # Share validity reporting
    for target, valid_col in [("fga", "share_fga_valid"), ("fta", "share_fta_valid"), ("tov", "share_tov_valid")]:
        if valid_col in training_df.columns:
            n_invalid = training_df.groupby(["game_id", "team_id"])[valid_col].first()
            n_invalid_groups = (~n_invalid).sum()
            if n_invalid_groups > 0:
                typer.echo(f"[usage_shares] WARNING: {n_invalid_groups} team-games with team_{target}=0 (share_{target}_valid=False)")
                # Print top offenders
                invalid_df = training_df[~training_df[valid_col]].drop_duplicates(subset=["game_id", "team_id"])
                team_col = f"team_{target}"
                if team_col in training_df.columns and len(invalid_df) > 0:
                    for _, row in invalid_df.head(5).iterrows():
                        typer.echo(f"    game_id={row['game_id']} team_id={row['team_id']} {team_col}={row.get(team_col, 'N/A')}")
    
    # Share sum checks (ONLY for valid groups)
    for col, valid_col in [("share_fga", "share_fga_valid"), ("share_fta", "share_fta_valid"), ("share_tov", "share_tov_valid")]:
        if col in training_df.columns and valid_col in training_df.columns:
            valid_rows = training_df[training_df[valid_col]]
            if len(valid_rows) > 0:
                share_sums = valid_rows.groupby(["game_id", "team_id"])[col].sum()
                max_err = (share_sums - 1.0).abs().max()
                typer.echo(f"[usage_shares] {col} sum max_err_vs_1: {max_err:.6f} (valid groups only)")
            else:
                typer.echo(f"[usage_shares] {col} sum: no valid groups")
    
    # Odds leak-safety check
    if "odds_as_of_ts" in training_df.columns and "tip_ts" in training_df.columns:
        training_df["_odds_as_of_ts"] = pd.to_datetime(training_df["odds_as_of_ts"], utc=True, errors="coerce")
        training_df["_tip_ts"] = pd.to_datetime(training_df["tip_ts"], utc=True, errors="coerce")
        both_present = training_df["_odds_as_of_ts"].notna() & training_df["_tip_ts"].notna()
        if both_present.any():
            delta = (training_df.loc[both_present, "_odds_as_of_ts"] - training_df.loc[both_present, "_tip_ts"]).dt.total_seconds() / 3600
            max_delta = delta.max()
            typer.echo(f"[usage_shares] odds leak-safety: max(odds_as_of_ts - tip_ts) = {max_delta:.2f} hours")
            if max_delta > 0:
                typer.echo("[usage_shares] WARNING: odds_as_of_ts > tip_ts for some rows (potential leak!)")
            else:
                p05, p50, p95 = delta.quantile([0.05, 0.50, 0.95])
                typer.echo(f"[usage_shares] odds lead time (tip_ts - odds_as_of_ts): p05={-p05:.1f}h p50={-p50:.1f}h p95={-p95:.1f}h")
        else:
            typer.echo("[usage_shares] odds leak-safety: no rows with both odds_as_of_ts and tip_ts")
        training_df.drop(columns=["_odds_as_of_ts", "_tip_ts"], inplace=True)
    else:
        typer.echo("[usage_shares] odds leak-safety: odds_as_of_ts or tip_ts not present")

    # Feature missingness
    feature_cols = [
        "minutes_pred_p50",
        "minutes_pred_play_prob",
        "is_starter",
        "spread_close",
        "total_close",
        "vac_min_szn",
        "track_role_cluster",
    ]
    typer.echo("[usage_shares] feature availability (% non-null):")
    for col in feature_cols:
        if col in training_df.columns:
            pct = 100.0 * (1.0 - training_df[col].isna().mean())
            typer.echo(f"  - {col}: {pct:.1f}%")
        else:
            typer.echo(f"  - {col}: NOT PRESENT")

    # Herfindahl diagnostic (only for valid FGA groups)
    if "share_fga" in training_df.columns and "share_fga_valid" in training_df.columns:
        valid_fga = training_df[training_df["share_fga_valid"]]
        if len(valid_fga) > 0:
            valid_fga = valid_fga.copy()
            valid_fga["share_fga_sq"] = valid_fga["share_fga"] ** 2
            herfindahl = valid_fga.groupby(["game_id", "team_id"])["share_fga_sq"].sum()
            typer.echo(
                f"[usage_shares] FGA Herfindahl: mean={herfindahl.mean():.4f} "
                f"p10={herfindahl.quantile(0.1):.4f} p50={herfindahl.quantile(0.5):.4f} "
                f"p90={herfindahl.quantile(0.9):.4f}"
            )

    # Top-1 share (only for valid FGA groups)
    if "share_fga" in training_df.columns and "share_fga_valid" in training_df.columns:
        valid_fga = training_df[training_df["share_fga_valid"]]
        if len(valid_fga) > 0:
            top1 = valid_fga.groupby(["game_id", "team_id"])["share_fga"].max()
            typer.echo(
                f"[usage_shares] FGA top-1 share: mean={top1.mean():.4f} "
                f"p10={top1.quantile(0.1):.4f} p50={top1.quantile(0.5):.4f} "
                f"p90={top1.quantile(0.9):.4f}"
            )

    # Write partitions
    typer.echo("[usage_shares] writing partitions...")
    n_partitions = _write_partitions(training_df, out_root)
    typer.echo(f"[usage_shares] wrote {n_partitions} partitions to {out_root}")

    # Summary
    min_date = pd.to_datetime(training_df["game_date"]).min()
    max_date = pd.to_datetime(training_df["game_date"]).max()
    n_games = training_df["game_id"].nunique()
    n_players = training_df["player_id"].nunique()
    typer.echo(
        f"[usage_shares] DONE: {n_total:,} rows, {n_games:,} games, {n_players:,} players "
        f"from {min_date.date()} to {max_date.date()}"
    )


if __name__ == "__main__":
    app()
