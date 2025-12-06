"""Build rates_v1 features for live scoring.

This CLI mirrors the training base feature construction from
scripts/rates/build_training_base.py but works with live data sources
(minutes predictions, season aggregates, tracking, Vegas, injuries).

Output: live/features_rates_v1/{date}/run={id}/features.parquet
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections import paths
from projections.pipeline.status import JobStatus, write_status
from projections.rates_v1.features import FEATURES_STAGE3_CONTEXT
from projections.rates_v1.schemas import (
    FEATURES_RATES_V1_SCHEMA,
    FeatureSchemaMismatchError,
    enforce_schema,
    validate_rates_features,
    validate_with_pandera,
)

app = typer.Typer(help=__doc__)

DEFAULT_DATA_ROOT = paths.get_data_root()
DEFAULT_MINUTES_FEATURES_ROOT = paths.data_path("live", "features_minutes_v1")
DEFAULT_OUTPUT_ROOT = paths.data_path("live", "features_rates_v1")
FEATURE_FILENAME = "features.parquet"
SUMMARY_FILENAME = "summary.json"
LATEST_POINTER = "latest_run.json"


def _normalize_day(value: datetime | None) -> date:
    if value is None:
        return datetime.now(tz=UTC).date()
    return value.date()


def _read_latest_run_id(features_dir: Path) -> str | None:
    """Read the latest run ID from a features directory."""
    pointer = features_dir / LATEST_POINTER
    if pointer.exists():
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            return payload.get("run_id")
        except json.JSONDecodeError:
            pass
    return None


def _load_minutes_predictions(features_path: Path) -> pd.DataFrame:
    """Load minutes predictions from the live features parquet.

    Expected columns from minutes_v1 features that map to rates schema:
    - minutes_pred_p50 (from scoring) or prior_play_prob, etc.
    """
    if not features_path.exists():
        raise FileNotFoundError(f"Minutes features not found at {features_path}")

    df = pd.read_parquet(features_path)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()

    # Normalize key columns
    for col in ("game_id", "player_id", "team_id"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


def _load_season_aggregates(
    data_root: Path,
    game_date: date,
    player_ids: list[int],
) -> pd.DataFrame:
    """Load season-to-date per-minute aggregates for players.

    Sources from gold/rates_training_base or computed from boxscores.
    """
    # Try loading from pre-computed rates training base
    training_base_root = data_root / "gold" / "rates_training_base"

    # Find the most recent available data before game_date
    season_year = game_date.year if game_date.month >= 8 else game_date.year - 1

    # Collect all available game dates for this season
    season_dir = training_base_root / f"season={season_year}"
    if not season_dir.exists():
        typer.echo(f"[rates-live] Warning: No training base for season {season_year}", err=True)
        return pd.DataFrame()

    frames = []
    for day_dir in sorted(season_dir.glob("game_date=*")):
        try:
            day = pd.Timestamp(day_dir.name.split("=", 1)[1]).date()
        except (ValueError, IndexError):
            continue
        if day >= game_date:
            continue
        parquet_path = day_dir / "rates_training_base.parquet"
        if parquet_path.exists():
            frames.append(pd.read_parquet(parquet_path))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()

    # Get the latest row per player (most recent prior game)
    df = df.sort_values("game_date")
    latest = df.groupby("player_id", as_index=False).tail(1)

    # Filter to requested players
    latest = latest[latest["player_id"].isin(player_ids)]

    # Select columns we need for season aggregates
    agg_cols = [
        "player_id",
        "season_fga_per_min",
        "season_3pa_per_min",
        "season_fta_per_min",
        "season_ast_per_min",
        "season_tov_per_min",
        "season_reb_per_min",
        "season_stl_per_min",
        "season_blk_per_min",
        "season_fg2_pct",
        "season_fg3_pct",
        "season_ft_pct",
    ]
    for col in agg_cols:
        if col not in latest.columns:
            latest[col] = 0.0
    return latest[agg_cols].copy()


def _load_tracking_features(
    data_root: Path,
    game_date: date,
    player_ids: list[int],
) -> pd.DataFrame:
    """Load tracking role features for players."""
    tracking_root = data_root / "gold" / "tracking_roles"
    if not tracking_root.exists():
        return pd.DataFrame()

    season_year = game_date.year if game_date.month >= 8 else game_date.year - 1
    season_dir = tracking_root / f"season={season_year}"
    if not season_dir.exists():
        return pd.DataFrame()

    frames = []
    for day_dir in sorted(season_dir.glob("game_date=*")):
        try:
            day = pd.Timestamp(day_dir.name.split("=", 1)[1]).date()
        except (ValueError, IndexError):
            continue
        if day >= game_date:
            continue
        parquet_path = day_dir / "tracking_roles.parquet"
        if parquet_path.exists():
            frames.append(pd.read_parquet(parquet_path))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    df = df.sort_values("game_date")
    latest = df.groupby("player_id", as_index=False).tail(1)
    latest = latest[latest["player_id"].isin(player_ids)]

    track_cols = [
        "player_id",
        "track_touches_per_min_szn",
        "track_sec_per_touch_szn",
        "track_pot_ast_per_min_szn",
        "track_drives_per_min_szn",
        "track_role_cluster",
        "track_role_is_low_minutes",
    ]
    available = [c for c in track_cols if c in latest.columns]
    return latest[available].copy()


def _load_vacancy_features(
    data_root: Path,
    game_date: date,
    team_ids: list[int],
) -> pd.DataFrame:
    """Load vacancy features from injuries.

    For live scoring, we compute these from the most recent training base
    that has vacated team features.
    """
    training_base_root = data_root / "gold" / "rates_training_base"
    season_year = game_date.year if game_date.month >= 8 else game_date.year - 1
    season_dir = training_base_root / f"season={season_year}"

    if not season_dir.exists():
        return pd.DataFrame()

    # Get the most recent day's vacancy features per team
    frames = []
    for day_dir in sorted(season_dir.glob("game_date=*"), reverse=True):
        try:
            day = pd.Timestamp(day_dir.name.split("=", 1)[1]).date()
        except (ValueError, IndexError):
            continue
        if day >= game_date:
            continue
        parquet_path = day_dir / "rates_training_base.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            if not frames:
                frames.append(df)
            break  # Only need the most recent

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    vac_cols = [
        "team_id",
        "vac_min_szn",
        "vac_fga_szn",
        "vac_ast_szn",
        "vac_min_guard_szn",
        "vac_min_wing_szn",
        "vac_min_big_szn",
    ]
    available = [c for c in vac_cols if c in df.columns]
    if not available:
        return pd.DataFrame()

    # Get team-level averages from the most recent games
    team_vac = df.groupby("team_id")[available[1:]].mean().reset_index()
    team_vac = team_vac[team_vac["team_id"].isin(team_ids)]
    return team_vac


def _load_team_context(
    data_root: Path,
    game_date: date,
    team_ids: list[int],
) -> pd.DataFrame:
    """Load team pace/rating context features."""
    training_base_root = data_root / "gold" / "rates_training_base"
    season_year = game_date.year if game_date.month >= 8 else game_date.year - 1
    season_dir = training_base_root / f"season={season_year}"

    if not season_dir.exists():
        return pd.DataFrame()

    frames = []
    for day_dir in sorted(season_dir.glob("game_date=*"), reverse=True):
        try:
            day = pd.Timestamp(day_dir.name.split("=", 1)[1]).date()
        except (ValueError, IndexError):
            continue
        if day >= game_date:
            continue
        parquet_path = day_dir / "rates_training_base.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            if not frames:
                frames.append(df)
            break

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    ctx_cols = [
        "team_id",
        "team_pace_szn",
        "team_off_rtg_szn",
        "team_def_rtg_szn",
    ]
    available = [c for c in ctx_cols if c in df.columns]
    if len(available) < 2:
        return pd.DataFrame()

    team_ctx = df.groupby("team_id")[available[1:]].mean().reset_index()
    team_ctx = team_ctx[team_ctx["team_id"].isin(team_ids)]
    return team_ctx


def build_rates_features(
    minutes_features: pd.DataFrame,
    season_aggs: pd.DataFrame,
    tracking: pd.DataFrame,
    vacancy: pd.DataFrame,
    team_context: pd.DataFrame,
    game_date: date,
) -> pd.DataFrame:
    """Assemble rates_v1 features from component data sources."""

    df = minutes_features.copy()

    # Extract key columns we need
    key_cols = ["game_id", "player_id", "team_id", "game_date"]

    # Map minutes_v1 features to rates feature names
    # minutes_v1 columns that map to stage1 features:
    feature_mapping = {
        # From minutes predictions (if scored)
        "minutes_p50": "minutes_pred_p50",
        "minutes_p10": "minutes_pred_p10",
        "minutes_p90": "minutes_pred_p90",
        # From minutes_v1 features
        "prior_play_prob": "minutes_pred_play_prob",
        "is_projected_starter": "is_starter",
        "is_confirmed_starter": "is_starter_confirmed",
        "home_flag": "home_flag",
        "days_since_last": "days_rest",
        "spread_home": "spread_close",
        "total": "total_close",
    }

    # Apply mapping
    for src, dst in feature_mapping.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # Derive is_starter from confirmation/projection
    if "is_starter" not in df.columns:
        df["is_starter"] = (
            df.get("is_confirmed_starter", False).astype(bool)
            | df.get("is_projected_starter", False).astype(bool)
        ).astype(int)
    else:
        df["is_starter"] = df["is_starter"].fillna(0).astype(int)

    # Compute spread/play_prob if missing
    if "minutes_pred_p50" not in df.columns and "roll_mean_5" in df.columns:
        df["minutes_pred_p50"] = df["roll_mean_5"]
    if "minutes_pred_spread" not in df.columns:
        p90 = df.get("minutes_pred_p90", df.get("minutes_p90", pd.NA))
        p10 = df.get("minutes_pred_p10", df.get("minutes_p10", pd.NA))
        if p90 is not pd.NA and p10 is not pd.NA:
            df["minutes_pred_spread"] = p90 - p10
        else:
            df["minutes_pred_spread"] = 10.0  # Default spread
    if "minutes_pred_play_prob" not in df.columns:
        df["minutes_pred_play_prob"] = df.get("prior_play_prob", 1.0)

    # Normalize days_rest
    if "days_rest" not in df.columns:
        df["days_rest"] = df.get("days_since_last", 1).clip(0, 3)
    df["days_rest"] = df["days_rest"].fillna(1).clip(0, 3).astype(int)

    # Position flags (from pos_bucket if available)
    if "pos_bucket" in df.columns:
        pos = df["pos_bucket"].fillna("UNK")
        for p in ("PG", "SG", "SF", "PF", "C"):
            df[f"position_flags_{p}"] = (pos == p).astype(int)
    else:
        for p in ("PG", "SG", "SF", "PF", "C"):
            if f"position_flags_{p}" not in df.columns:
                df[f"position_flags_{p}"] = 0

    # Join season aggregates
    if not season_aggs.empty:
        df = df.merge(season_aggs, on="player_id", how="left", suffixes=("", "_szn"))

    # Fill missing season stats with zeros (indicating new/limited data)
    season_cols = [
        "season_fga_per_min",
        "season_3pa_per_min",
        "season_fta_per_min",
        "season_ast_per_min",
        "season_tov_per_min",
        "season_reb_per_min",
        "season_stl_per_min",
        "season_blk_per_min",
    ]
    for col in season_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    # Vegas context
    if "spread_close" not in df.columns:
        df["spread_close"] = df.get("spread_home", np.nan)
    if "total_close" not in df.columns:
        df["total_close"] = df.get("total", np.nan)

    # Compute implied totals
    df["has_odds"] = (~df["spread_close"].isna()) & (~df["total_close"].isna())
    home_flag = df["home_flag"].fillna(0).astype(int)
    total = df["total_close"].fillna(220.0)
    spread = df["spread_close"].fillna(0.0)
    home_itt = total / 2 - spread / 2
    away_itt = total - home_itt
    df["team_itt"] = np.where(home_flag == 1, home_itt, away_itt)
    df["opp_itt"] = np.where(home_flag == 1, away_itt, home_itt)
    df["has_odds"] = df["has_odds"].astype(int)

    # Join tracking features
    if not tracking.empty:
        df = df.merge(tracking, on="player_id", how="left", suffixes=("", "_track"))

    track_cols = [
        "track_touches_per_min_szn",
        "track_sec_per_touch_szn",
        "track_pot_ast_per_min_szn",
        "track_drives_per_min_szn",
        "track_role_cluster",
        "track_role_is_low_minutes",
    ]
    for col in track_cols:
        if col not in df.columns:
            df[col] = 0.0 if "role" not in col else 0
        df[col] = df[col].fillna(0.0 if "role" not in col else 0)
        # Ensure role columns are int to avoid mixed bool/int dtype
        if "role" in col:
            df[col] = df[col].astype(int)

    # Join vacancy features
    if not vacancy.empty:
        df = df.merge(vacancy, on="team_id", how="left", suffixes=("", "_vac"))

    vac_cols = [
        "vac_min_szn",
        "vac_fga_szn",
        "vac_ast_szn",
        "vac_min_guard_szn",
        "vac_min_wing_szn",
        "vac_min_big_szn",
    ]
    for col in vac_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    # Join team context
    if not team_context.empty:
        df = df.merge(team_context, on="team_id", how="left", suffixes=("", "_ctx"))

    # Also need opponent context
    if "opponent_team_id" in df.columns and not team_context.empty:
        opp_ctx = team_context.rename(columns={
            "team_id": "opponent_team_id",
            "team_pace_szn": "opp_pace_szn",
            "team_def_rtg_szn": "opp_def_rtg_szn",
        })
        opp_cols = ["opponent_team_id", "opp_pace_szn", "opp_def_rtg_szn"]
        opp_cols = [c for c in opp_cols if c in opp_ctx.columns]
        if len(opp_cols) > 1:
            df = df.merge(opp_ctx[opp_cols], on="opponent_team_id", how="left", suffixes=("", "_opp"))

    ctx_cols = [
        "team_pace_szn",
        "team_off_rtg_szn",
        "team_def_rtg_szn",
        "opp_pace_szn",
        "opp_def_rtg_szn",
    ]
    for col in ctx_cols:
        if col not in df.columns:
            df[col] = 100.0 if "rtg" in col else 100.0  # Default pace/rating
        df[col] = df[col].fillna(100.0)

    # Ensure game_date is present
    df["game_date"] = pd.Timestamp(game_date).normalize()

    return df


def _write_output(
    df: pd.DataFrame,
    output_root: Path,
    game_date: date,
    run_id: str,
) -> Path:
    """Write features to output directory."""
    day_dir = output_root / game_date.isoformat()
    run_dir = day_dir / f"run={run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    out_path = run_dir / FEATURE_FILENAME
    df.to_parquet(out_path, index=False)

    # Write summary
    summary = {
        "date": game_date.isoformat(),
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "run_id": run_id,
        "counts": {
            "rows": len(df),
            "players": int(df["player_id"].nunique()),
            "games": int(df["game_id"].nunique()),
        },
        "feature_columns": list(df.columns),
    }
    (run_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Update latest pointer
    pointer = {"run_id": run_id, "generated_at": datetime.now(tz=UTC).isoformat()}
    (day_dir / LATEST_POINTER).write_text(
        json.dumps(pointer, indent=2), encoding="utf-8"
    )

    return out_path


@app.command()
def main(
    date_value: datetime = typer.Option(..., "--date", help="Slate date (YYYY-MM-DD)"),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Run ID (defaults to minutes run ID)"
    ),
    minutes_features_path: Optional[Path] = typer.Option(
        None,
        "--minutes-features-path",
        help="Explicit path to minutes features parquet (overrides auto-discovery)",
    ),
    minutes_features_root: Path = typer.Option(
        DEFAULT_MINUTES_FEATURES_ROOT,
        "--minutes-features-root",
        help="Root containing live minutes features",
    ),
    data_root: Path = typer.Option(
        DEFAULT_DATA_ROOT,
        "--data-root",
        help="Root containing training base and other data",
    ),
    output_root: Path = typer.Option(
        DEFAULT_OUTPUT_ROOT,
        "--output-root",
        help="Output root for live rates features",
    ),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="Raise error if schema validation fails (default: True)",
    ),
) -> None:
    """Build live rates features for a slate."""

    game_date = _normalize_day(date_value)
    run_ts_iso = datetime.now(tz=UTC).isoformat()

    try:
        # Resolve minutes features path
        if minutes_features_path is None:
            day_dir = minutes_features_root / game_date.isoformat()
            resolved_run = run_id or _read_latest_run_id(day_dir)
            if resolved_run is None:
                raise FileNotFoundError(
                    f"No minutes features found for {game_date}; pass --run-id or --minutes-features-path"
                )
            minutes_features_path = day_dir / f"run={resolved_run}" / FEATURE_FILENAME
        else:
            resolved_run = run_id or datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")

        typer.echo(f"[rates-live] Loading minutes features from {minutes_features_path}")
        minutes_df = _load_minutes_predictions(minutes_features_path)
        if minutes_df.empty:
            raise ValueError("Minutes features are empty")

        player_ids = minutes_df["player_id"].dropna().astype(int).unique().tolist()
        team_ids = minutes_df["team_id"].dropna().astype(int).unique().tolist()

        typer.echo(
            f"[rates-live] Found {len(player_ids)} players, {len(team_ids)} teams"
        )

        # Load component data
        season_aggs = _load_season_aggregates(data_root, game_date, player_ids)
        typer.echo(f"[rates-live] Season aggregates: {len(season_aggs)} players")

        tracking = _load_tracking_features(data_root, game_date, player_ids)
        typer.echo(f"[rates-live] Tracking features: {len(tracking)} players")

        vacancy = _load_vacancy_features(data_root, game_date, team_ids)
        typer.echo(f"[rates-live] Vacancy features: {len(vacancy)} teams")

        team_context = _load_team_context(data_root, game_date, team_ids)
        typer.echo(f"[rates-live] Team context: {len(team_context)} teams")

        # Build features
        features = build_rates_features(
            minutes_df,
            season_aggs,
            tracking,
            vacancy,
            team_context,
            game_date,
        )

        # Validate schema
        missing = validate_rates_features(features, strict=strict)
        if missing:
            typer.echo(
                f"[rates-live] Warning: missing columns (non-strict): {missing}",
                err=True,
            )

        # Write output
        out_path = _write_output(features, output_root, game_date, resolved_run)
        typer.echo(
            f"[rates-live] Wrote {len(features)} rows -> {out_path}"
        )

        write_status(
            JobStatus(
                job_name="build_rates_features_live",
                stage="features",
                target_date=game_date.isoformat(),
                run_ts=run_ts_iso,
                status="success",
                rows_written=len(features),
                expected_rows=len(minutes_df),
            )
        )

    except Exception as exc:
        write_status(
            JobStatus(
                job_name="build_rates_features_live",
                stage="features",
                target_date=game_date.isoformat(),
                run_ts=run_ts_iso,
                status="error",
                rows_written=0,
                message=str(exc),
            )
        )
        raise


if __name__ == "__main__":
    app()
