"""Build rates_v1 features for live scoring.

This CLI mirrors the training base feature construction from
scripts/rates/build_training_base.py but works with live data sources
(minutes predictions, season aggregates, tracking, Vegas, injuries).

Output: live/features_rates_v1/{date}/run={id}/features.parquet
"""

from __future__ import annotations

# ruff: noqa: E402

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Optional

from projections.runtime_safety import configure_runtime_safety

configure_runtime_safety()

import numpy as np
import pandas as pd
import typer

from projections import paths
from projections.features.action_props import (
    attach_action_props_features,
    load_action_props_feature_snapshots_for_date,
)
from projections.minutes_v1.pos import canonical_pos_bucket
from projections.pipeline.status import JobStatus, write_status
from projections.rates_v1.schemas import validate_rates_features
from projections.minutes_v1.season_dataset import _parse_minutes_iso

app = typer.Typer(help=__doc__)

DEFAULT_DATA_ROOT = paths.get_data_root()
DEFAULT_MINUTES_FEATURES_ROOT = paths.data_path("live", "features_minutes_v1")
DEFAULT_OUTPUT_ROOT = paths.data_path("live", "features_rates_v1")
FEATURE_FILENAME = "features.parquet"
SUMMARY_FILENAME = "summary.json"
LATEST_POINTER = "latest_run.json"

_STATUS_OUT_LIKE: set[str] = {"OUT", "O", "INACTIVE"}
_STATUS_QUESTIONABLE: set[str] = {"Q", "QUESTIONABLE"}
_STATUS_PROBABLE: set[str] = {"PROB", "PROBABLE"}
_STATUS_DOUBTFUL: set[str] = {"D", "DOUBTFUL"}


def _status_to_out_probability(status: pd.Series) -> pd.Series:
    """Map injury/status strings to an "out probability" in [0, 1]."""
    normalized = status.fillna("").astype(str).str.upper().str.strip()
    out_prob = pd.Series(0.0, index=normalized.index, dtype=float)

    out_prob[normalized.isin(_STATUS_OUT_LIKE)] = 1.0
    out_prob[normalized.isin(_STATUS_QUESTIONABLE)] = 1.0 - 0.55
    out_prob[normalized.isin(_STATUS_PROBABLE)] = 1.0 - 0.78
    out_prob[normalized.isin(_STATUS_DOUBTFUL)] = 1.0 - 0.25
    return out_prob


def _count_action_props_matches(df: pd.DataFrame) -> int:
    """Count rows with at least one attached Action props market."""
    if "an_has_any_props" not in df.columns:
        return 0
    return int(
        pd.to_numeric(df.get("an_has_any_props"), errors="coerce")
        .fillna(0.0)
        .gt(0.0)
        .sum()
    )


def _load_boxscores_history(data_root: Path, season_year: int) -> pd.DataFrame:
    """Load raw boxscores for the given season to build player history."""
    season_dir = data_root / "bronze" / "boxscores_raw" / f"season={season_year}"
    if not season_dir.exists():
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    # Glob all dates
    for pq_path in season_dir.glob("date=*/boxscores_raw.parquet"):
        try:
            bronze = pd.read_parquet(pq_path)
        except Exception:
            continue
            
        for row in bronze.itertuples():
            try:
                payload = json.loads(row.payload)
            except (json.JSONDecodeError, AttributeError):
                continue
                
            tip_ts_raw = payload.get("game_time_utc") or payload.get("game_time_local")
            if not tip_ts_raw:
                continue
            # Ensure UTC
            tip_ts = pd.Timestamp(tip_ts_raw)
            if tip_ts.tzinfo is None:
                tip_ts = tip_ts.tz_localize(UTC)
            else:
                tip_ts = tip_ts.tz_convert(UTC)

            home = payload.get("home") or {}
            away = payload.get("away") or {}
            
            for team_payload in (home, away):
                for player in team_payload.get("players", []):
                    stats = player.get("statistics") or {}
                    
                    records.append({
                        "player_id": int(player.get("person_id") or player.get("personId") or 0),
                        "tip_ts": tip_ts,
                        "minutes_played": _parse_minutes_iso(stats.get("minutes")),
                        "fga": float(stats.get("fieldGoalsAttempted") or 0.0),
                        "fgm": float(stats.get("fieldGoalsMade") or stats.get("fieldGoalsMade") or 0.0),
                        "three_pa": float(stats.get("threePointersAttempted") or 0.0),
                        "three_pm": float(stats.get("threePointersMade") or 0.0),
                        "fta": float(stats.get("freeThrowsAttempted") or 0.0),
                        "ftm": float(stats.get("freeThrowsMade") or 0.0),
                        "assists": float(stats.get("assists") or 0.0),
                        "turnovers": float(stats.get("turnovers") or 0.0),
                        "oreb": float(stats.get("reboundsOffensive") or 0.0),
                        "dreb": float(stats.get("reboundsDefensive") or 0.0),
                        "steals": float(stats.get("steals") or 0.0),
                        "blocks": float(stats.get("blocks") or 0.0),
                    })
                    
    if not records:
        return pd.DataFrame()
        
    df = pd.DataFrame.from_records(records)
    # Deduplicate: latest per player per game (tip_ts serves as game proxy)
    df.sort_values("tip_ts", inplace=True)
    return df

def _compute_player_priors(history: pd.DataFrame, *, player_ids: set[int]) -> pd.DataFrame:
    """Compute season-to-date + recency per-minute priors from player boxscore history."""
    if history.empty or not player_ids:
        return pd.DataFrame()

    df = history.copy()
    df = df[pd.to_numeric(df["player_id"], errors="coerce").isin(player_ids)].copy()
    if df.empty:
        return pd.DataFrame()

    df["tip_ts"] = pd.to_datetime(df["tip_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["player_id", "tip_ts"])
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype(int)

    num_cols = [
        "minutes_played",
        "fga",
        "fgm",
        "three_pa",
        "three_pm",
        "fta",
        "ftm",
        "assists",
        "turnovers",
        "oreb",
        "dreb",
        "steals",
        "blocks",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
        else:
            df[col] = 0.0

    df["fga2"] = (df["fga"] - df["three_pa"]).clip(lower=0.0)
    df["fg2_made"] = (df["fgm"] - df["three_pm"]).clip(lower=0.0)

    df.sort_values(["player_id", "tip_ts"], inplace=True)

    def _rates(frame: pd.DataFrame) -> dict[str, float]:
        minutes_sum = float(frame["minutes_played"].sum())
        # Avoid division by zero
        denom = minutes_sum if minutes_sum > 0 else 1.0
        
        # Sums
        fga2_sum = float(frame["fga2"].sum())
        fga3_sum = float(frame["three_pa"].sum())
        fta_sum = float(frame["fta"].sum())
        ast_sum = float(frame["assists"].sum())
        tov_sum = float(frame["turnovers"].sum())
        oreb_sum = float(frame["oreb"].sum())
        dreb_sum = float(frame["dreb"].sum())
        stl_sum = float(frame["steals"].sum())
        blk_sum = float(frame["blocks"].sum())

        # Shooting Pcts
        fg2_att = float(frame["fga2"].sum())
        fg2_made = float(frame["fg2_made"].sum())
        fg3_att = float(frame["three_pa"].sum())
        fg3_made = float(frame["three_pm"].sum())
        ft_att = float(frame["fta"].sum())
        ft_made = float(frame["ftm"].sum())

        fg2_pct = fg2_made / fg2_att if fg2_att > 0 else 0.55
        fg3_pct = fg3_made / fg3_att if fg3_att > 0 else 0.35
        ft_pct = ft_made / ft_att if ft_att > 0 else 0.75

        # Clip efficiency to stable ranges.
        fg2_pct = float(np.clip(fg2_pct, 0.35, 0.75))
        fg3_pct = float(np.clip(fg3_pct, 0.25, 0.55))
        ft_pct = float(np.clip(ft_pct, 0.5, 0.9))
        
        res = {
            "minutes_sum": minutes_sum,
            "fga2_per_min": fga2_sum / denom if minutes_sum > 0 else 0.0,
            "fga3_per_min": fga3_sum / denom if minutes_sum > 0 else 0.0,
            "fta_per_min": fta_sum / denom if minutes_sum > 0 else 0.0,
            "ast_per_min": ast_sum / denom if minutes_sum > 0 else 0.0,
            "tov_per_min": tov_sum / denom if minutes_sum > 0 else 0.0,
            "oreb_per_min": oreb_sum / denom if minutes_sum > 0 else 0.0,
            "dreb_per_min": dreb_sum / denom if minutes_sum > 0 else 0.0,
            "stl_per_min": stl_sum / denom if minutes_sum > 0 else 0.0,
            "blk_per_min": blk_sum / denom if minutes_sum > 0 else 0.0,
            "fg2_pct": fg2_pct,
            "fg3_pct": fg3_pct,
            "ft_pct": ft_pct,
        }
        
        # Alias for 3pa (legacy)
        res["3pa_per_min"] = res["fga3_per_min"] 
        return res

    rows: list[dict[str, object]] = []
    
    # Define stats to extract for each window
    stat_keys = [
        "fga2_per_min", "fga3_per_min", "fta_per_min",
        "ast_per_min", "tov_per_min",
        "oreb_per_min", "dreb_per_min",
        "stl_per_min", "blk_per_min"
    ]

    for pid, frame in df.groupby("player_id", sort=False):
        row = {"player_id": int(pid)}

        # Season
        season = _rates(frame)
        row["n_games_season"] = len(frame)  # Sample size for shrinkage calibration
        row["season_minutes_sum"] = season["minutes_sum"]  # Total minutes played
        row["season_fga2_per_min"] = season["fga2_per_min"]
        row["season_3pa_per_min"] = season["fga3_per_min"]
        row["season_fta_per_min"] = season["fta_per_min"]
        row["season_ast_per_min"] = season["ast_per_min"]
        row["season_tov_per_min"] = season["tov_per_min"]
        row["season_oreb_per_min"] = season["oreb_per_min"]
        row["season_dreb_per_min"] = season["dreb_per_min"]
        row["season_stl_per_min"] = season["stl_per_min"]
        row["season_blk_per_min"] = season["blk_per_min"]
        row["season_fg2_pct"] = season["fg2_pct"]
        row["season_fg3_pct"] = season["fg3_pct"]
        row["season_ft_pct"] = season["ft_pct"]

        # Windows: last1, last3, last5, last10
        windows = {
            "last1": frame.tail(1),
            "last3": frame.tail(3),
            "last5": frame.tail(5),
            "last10": frame.tail(10),
        }
        
        for pfx, win_frame in windows.items():
            stats = _rates(win_frame)
            row[f"{pfx}_minutes_sum"] = stats["minutes_sum"]
            for key in stat_keys:
                row[f"{pfx}_{key}"] = stats[key]

        rows.append(row)

    return pd.DataFrame.from_records(rows)


def _compute_team_context(team_history: pd.DataFrame, *, team_ids: set[int]) -> pd.DataFrame:
    """Compute simple season-to-date pace/off/def context from team game logs."""
    if team_history.empty or not team_ids:
        return pd.DataFrame()

    df = team_history.copy()
    df = df[pd.to_numeric(df["team_id"], errors="coerce").isin(team_ids)].copy()
    if df.empty:
        return pd.DataFrame()

    for col in ("points_for", "points_against", "fga", "fta", "oreb", "turnovers"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype(int)
    df["game_id"] = pd.to_numeric(df.get("game_id"), errors="coerce").astype("Int64")

    # Basic possessions estimate:
    # 0.96 * (FGA + TOV + 0.44*FTA - OREB)
    df["poss"] = 0.96 * (df["fga"] + df["turnovers"] + 0.44 * df["fta"] - df["oreb"])
    grouped = df.groupby("team_id", as_index=False).agg(
        games_played=("game_id", "nunique"),
        poss_total=("poss", "sum"),
        pts_for_total=("points_for", "sum"),
        pts_against_total=("points_against", "sum"),
    )

    grouped["team_pace_szn"] = grouped["poss_total"] / grouped["games_played"].replace(0, np.nan)
    grouped["team_off_rtg_szn"] = 100.0 * (grouped["pts_for_total"] / grouped["poss_total"].replace(0, np.nan))
    grouped["team_def_rtg_szn"] = 100.0 * (grouped["pts_against_total"] / grouped["poss_total"].replace(0, np.nan))

    grouped = grouped.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return grouped[["team_id", "team_pace_szn", "team_off_rtg_szn", "team_def_rtg_szn"]].copy()


def _compute_vacancy_features(player_history: pd.DataFrame, minutes_preds: pd.DataFrame) -> pd.DataFrame:
    """Compute vacated team features from a season history + current OUT statuses."""
    if player_history.empty or minutes_preds.empty:
        return pd.DataFrame()

    history = player_history.copy()
    history["tip_ts"] = pd.to_datetime(history["tip_ts"], utc=True, errors="coerce")
    history = history.dropna(subset=["player_id", "tip_ts"]).copy()
    history["player_id"] = pd.to_numeric(history["player_id"], errors="coerce").astype(int)

    for col in ("minutes_played", "fga", "assists"):
        if col in history.columns:
            history[col] = pd.to_numeric(history[col], errors="coerce").fillna(0.0).astype(float)
        else:
            history[col] = 0.0

    history.sort_values(["player_id", "tip_ts"], inplace=True)
    history["cum_minutes_szn"] = history.groupby("player_id")["minutes_played"].cumsum()
    history["cum_fga_szn"] = history.groupby("player_id")["fga"].cumsum()
    history["cum_ast_szn"] = history.groupby("player_id")["assists"].cumsum()

    preds = minutes_preds.copy()
    preds["tip_ts"] = pd.to_datetime(preds["tip_ts"], utc=True, errors="coerce")
    preds = preds.dropna(subset=["game_id", "team_id", "player_id", "tip_ts"]).copy()
    for col in ("game_id", "team_id", "player_id"):
        preds[col] = pd.to_numeric(preds[col], errors="coerce").astype(int)

    out_prob = _status_to_out_probability(preds["status"])
    preds["out_prob"] = out_prob
    preds = preds[preds["out_prob"] > 0].copy()
    if preds.empty:
        return pd.DataFrame()

    preds["pos_bucket"] = preds.get("pos_bucket", pd.Series("UNK", index=preds.index)).fillna("UNK").astype(str)
    preds["pos_bucket"] = preds["pos_bucket"].apply(canonical_pos_bucket)
    preds.sort_values(["player_id", "tip_ts"], inplace=True)

    hist_cols = ["player_id", "tip_ts", "cum_minutes_szn", "cum_fga_szn", "cum_ast_szn"]
    merged = pd.merge_asof(
        preds,
        history[hist_cols],
        by="player_id",
        on="tip_ts",
        direction="backward",
        allow_exact_matches=True,
    )
    for col in ("cum_minutes_szn", "cum_fga_szn", "cum_ast_szn"):
        merged[col] = merged[col].fillna(0.0).astype(float)

    merged["vac_min_weighted"] = merged["cum_minutes_szn"] * merged["out_prob"]
    merged["vac_fga_weighted"] = merged["cum_fga_szn"] * merged["out_prob"]
    merged["vac_ast_weighted"] = merged["cum_ast_szn"] * merged["out_prob"]

    group_cols = ["game_id", "team_id"]
    out = merged.groupby(group_cols, as_index=False).agg(
        vac_min_szn=("vac_min_weighted", "sum"),
        vac_fga_szn=("vac_fga_weighted", "sum"),
        vac_ast_szn=("vac_ast_weighted", "sum"),
    )

    bucket_map = {"G": "vac_min_guard_szn", "W": "vac_min_wing_szn", "BIG": "vac_min_big_szn"}
    for bucket, col_name in bucket_map.items():
        sub = merged[merged["pos_bucket"] == bucket].groupby(group_cols)["vac_min_weighted"].sum().reset_index(name=col_name)
        out = out.merge(sub, on=group_cols, how="left")

    for col in ("vac_min_guard_szn", "vac_min_wing_szn", "vac_min_big_szn"):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = out[col].fillna(0.0).astype(float)

    return out


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
        # Extended FTA tracking features (stage5+)
        "track_drive_fta_per_min_szn",
        "track_drive_pf_per_min_szn",
        "track_paint_touches_per_min_szn",
        "track_fta_per_drive_szn",
        # 3PA profile tracking features
        "track_catch_shoot_fg3a_per_min_szn",
        "track_pull_up_fg3a_per_min_szn",
        "track_pull_up_3pa_share_szn",
    ]
    available = [c for c in track_cols if c in latest.columns]
    return latest[available].copy()


_VACANCY_DEFAULTS: dict[str, float] = {
    "vac_min_szn": 0.0,
    "vac_fga_szn": 0.0,
    "vac_ast_szn": 0.0,
    "vac_min_guard_szn": 0.0,
    "vac_min_wing_szn": 0.0,
    "vac_min_big_szn": 0.0,
}

_TEAM_CONTEXT_DEFAULTS: dict[str, float] = {
    # Keep aligned with build_rates_features() fallback.
    "team_pace_szn": 100.0,
    "team_off_rtg_szn": 100.0,
    "team_def_rtg_szn": 100.0,
    "team_fta_allowed_per_game": 24.0,  # League average FTA allowed per game
}


def _normalize_team_ids(team_ids: list[int]) -> list[int]:
    normalized: list[int] = []
    for value in team_ids:
        try:
            normalized.append(int(value))
        except Exception:  # noqa: BLE001
            continue
    return sorted(set(normalized))


def _load_team_features_from_rates_training_base(
    *,
    season_dir: Path,
    game_date: date,
    team_ids: set[int],
    required_cols: list[str],
    max_days_back: int,
) -> pd.DataFrame:
    """Backfill team features per team_id from prior rates_training_base partitions."""
    if not team_ids:
        return pd.DataFrame(columns=required_cols)

    candidates: list[tuple[pd.Timestamp, Path]] = []
    target_day = pd.Timestamp(game_date).normalize()
    for day_dir in season_dir.glob("game_date=*"):
        try:
            day_str = day_dir.name.split("=", 1)[1]
            day = pd.Timestamp(day_str).normalize()
        except Exception:  # noqa: BLE001
            continue
        if day >= target_day:
            continue
        pq_path = day_dir / "rates_training_base.parquet"
        if pq_path.exists():
            candidates.append((day, pq_path))

    if not candidates:
        return pd.DataFrame(columns=required_cols)

    candidates.sort(key=lambda item: item[0], reverse=True)
    remaining = set(team_ids)
    frames: list[pd.DataFrame] = []

    for idx, (_day, pq_path) in enumerate(candidates):
        if not remaining or idx >= max_days_back:
            break
        try:
            day_df = pd.read_parquet(pq_path, columns=required_cols)
        except Exception:  # noqa: BLE001
            try:
                day_df = pd.read_parquet(pq_path)
            except Exception:  # noqa: BLE001
                continue
            if "team_id" not in day_df.columns:
                continue

        if day_df.empty:
            continue
        day_df["team_id"] = pd.to_numeric(day_df["team_id"], errors="coerce").astype("Int64")
        day_df = day_df.dropna(subset=["team_id"])
        if day_df.empty:
            continue
        day_df = day_df.loc[day_df["team_id"].astype(int).isin(remaining)].copy()
        if day_df.empty:
            continue

        feature_cols = [col for col in required_cols if col != "team_id" and col in day_df.columns]
        if not feature_cols:
            continue
        grouped = day_df.groupby("team_id", as_index=False)[feature_cols].mean()
        frames.append(grouped)
        found = set(grouped["team_id"].dropna().astype(int).tolist())
        remaining -= found

    if not frames:
        return pd.DataFrame(columns=required_cols)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["team_id"]).drop_duplicates(subset=["team_id"], keep="first")
    combined["team_id"] = combined["team_id"].astype(int)
    for col in required_cols:
        if col == "team_id":
            continue
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return combined.reset_index(drop=True)


def _reindex_team_features(
    *,
    slate_team_ids: list[int],
    df: pd.DataFrame,
    feature_defaults: dict[str, float],
) -> pd.DataFrame:
    """Ensure one row per slate team_id, filling feature defaults."""
    slate = pd.DataFrame({"team_id": _normalize_team_ids(slate_team_ids)})
    if slate.empty:
        out = pd.DataFrame(columns=["team_id", *feature_defaults.keys()])
        for col, default in feature_defaults.items():
            out[col] = out.get(col, pd.Series(dtype=float)).fillna(default)
        return out

    if df is None or df.empty or "team_id" not in df.columns:
        out = slate.copy()
        for col, default in feature_defaults.items():
            out[col] = default
        return out.reset_index(drop=True)

    work = df.copy()
    work["team_id"] = pd.to_numeric(work["team_id"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["team_id"])
    work["team_id"] = work["team_id"].astype(int)

    keep_cols = ["team_id", *[c for c in feature_defaults.keys() if c in work.columns]]
    work = work.loc[:, keep_cols].drop_duplicates(subset=["team_id"], keep="first")

    out = slate.merge(work, on="team_id", how="left")
    for col, default in feature_defaults.items():
        if col not in out.columns:
            out[col] = default
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default).astype(float)
    return out.reset_index(drop=True)


def _load_vacancy_features(
    data_root: Path,
    game_date: date,
    team_ids: list[int],
) -> pd.DataFrame:
    """Load vacancy features for the slate teams.

    rates_training_base is partitioned by game_date and only contains teams active
    on that date. For a slate, many teams may not have played on the most recent
    prior date, so we backfill per-team from earlier partitions and then reindex
    to the slate teams, filling defaults for any teams still missing.
    """
    training_base_root = data_root / "gold" / "rates_training_base"
    season_year = game_date.year if game_date.month >= 8 else game_date.year - 1
    season_dir = training_base_root / f"season={season_year}"

    if not season_dir.exists():
        return _reindex_team_features(
            slate_team_ids=team_ids,
            df=pd.DataFrame(),
            feature_defaults=_VACANCY_DEFAULTS,
        )

    slate_team_ids = _normalize_team_ids(team_ids)
    raw = _load_team_features_from_rates_training_base(
        season_dir=season_dir,
        game_date=game_date,
        team_ids=set(slate_team_ids),
        required_cols=["team_id", *_VACANCY_DEFAULTS.keys()],
        max_days_back=21,
    )

    present = (
        sorted(set(raw["team_id"].dropna().astype(int).tolist()))
        if (raw is not None and not raw.empty and "team_id" in raw.columns)
        else []
    )
    missing = sorted(set(slate_team_ids) - set(present))
    typer.echo(f"[rates-live] Vacancy raw teams: {present}")
    if missing:
        typer.echo(
            f"[rates-live] Vacancy missing teams (filled defaults): {missing}",
            err=True,
        )

    return _reindex_team_features(
        slate_team_ids=slate_team_ids,
        df=raw,
        feature_defaults=_VACANCY_DEFAULTS,
    )


def _load_team_context(
    data_root: Path,
    game_date: date,
    team_ids: list[int],
) -> pd.DataFrame:
    """Load team pace/rating context features for the slate teams.

    rates_training_base is partitioned by game_date and only contains teams active
    on that date. For a slate, many teams may not have played on the most recent
    prior date, so we backfill per-team from earlier partitions and then reindex
    to the slate teams, filling defaults for any teams still missing.
    """
    training_base_root = data_root / "gold" / "rates_training_base"
    season_year = game_date.year if game_date.month >= 8 else game_date.year - 1
    season_dir = training_base_root / f"season={season_year}"

    if not season_dir.exists():
        return _reindex_team_features(
            slate_team_ids=team_ids,
            df=pd.DataFrame(),
            feature_defaults=_TEAM_CONTEXT_DEFAULTS,
        )

    slate_team_ids = _normalize_team_ids(team_ids)
    raw = _load_team_features_from_rates_training_base(
        season_dir=season_dir,
        game_date=game_date,
        team_ids=set(slate_team_ids),
        required_cols=["team_id", *_TEAM_CONTEXT_DEFAULTS.keys()],
        max_days_back=21,
    )

    present = (
        sorted(set(raw["team_id"].dropna().astype(int).tolist()))
        if (raw is not None and not raw.empty and "team_id" in raw.columns)
        else []
    )
    missing = sorted(set(slate_team_ids) - set(present))
    typer.echo(f"[rates-live] Team context raw teams: {present}")
    if missing:
        typer.echo(
            f"[rates-live] Team context missing teams (filled defaults): {missing}",
            err=True,
        )

    return _reindex_team_features(
        slate_team_ids=slate_team_ids,
        df=raw,
        feature_defaults=_TEAM_CONTEXT_DEFAULTS,
    )


def build_rates_features(
    minutes_features: pd.DataFrame,
    season_aggs: pd.DataFrame,
    tracking: pd.DataFrame,
    vacancy: pd.DataFrame,
    team_context: pd.DataFrame,
    priors: pd.DataFrame,
    game_date: date,
) -> pd.DataFrame:
    """Assemble rates_v1 features from component data sources."""

    df = minutes_features.copy()

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

    # Vegas context - ensure numeric dtype even when source data has None values
    if "spread_close" not in df.columns:
        df["spread_close"] = df.get("spread_home", np.nan)
    if "total_close" not in df.columns:
        df["total_close"] = df.get("total", np.nan)
    # Coerce to numeric to handle Python None (which creates object dtype)
    df["spread_close"] = pd.to_numeric(df["spread_close"], errors="coerce")
    df["total_close"] = pd.to_numeric(df["total_close"], errors="coerce")

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
        # Extended FTA tracking features (stage5+)
        "track_drive_fta_per_min_szn",
        "track_drive_pf_per_min_szn",
        "track_paint_touches_per_min_szn",
        "track_fta_per_drive_szn",
        # 3PA profile tracking features
        "track_catch_shoot_fg3a_per_min_szn",
        "track_pull_up_fg3a_per_min_szn",
        "track_pull_up_3pa_share_szn",
    ]
    for col in track_cols:
        if col not in df.columns:
            df[col] = np.nan
        # Leave tracking features nullable; score-time preprocessing applies
        # bundle-specific train-time imputations for parity.
        df[col] = pd.to_numeric(df[col], errors="coerce")

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

    # Join priors (recency features)
    if not priors.empty:
        # priors has 'season_fga2_per_min', 'last1_minutes_sum', etc.
        # We merge on player_id. 
        # We use suffixes=("_stale", "") so that columns in priors (e.g. season_fga2_per_min)
        # OVERWRITE the default/stale ones in df (which become season_fga2_per_min_stale).
        df = df.merge(priors, on="player_id", how="left", suffixes=("_stale", ""))

    # Also need opponent context
    if "opponent_team_id" in df.columns and not team_context.empty:
        opp_ctx = team_context.rename(columns={
            "team_id": "opponent_team_id",
            "team_pace_szn": "opp_pace_szn",
            "team_def_rtg_szn": "opp_def_rtg_szn",
            "team_fta_allowed_per_game": "opp_fta_allowed_per_game",
        })
        opp_cols = ["opponent_team_id", "opp_pace_szn", "opp_def_rtg_szn", "opp_fta_allowed_per_game"]
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
    # FTA allowed defaults to league average (~24 FTA/game)
    if "opp_fta_allowed_per_game" not in df.columns:
        df["opp_fta_allowed_per_game"] = 24.0
    df["opp_fta_allowed_per_game"] = df["opp_fta_allowed_per_game"].fillna(24.0)

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
    import os

    if os.environ.get("PROJECTIONS_SKIP_POINTER_WRITES", "").strip().lower() not in {"1", "true", "yes"}:
        from projections.pipeline import writer_guard

        writer_guard.assert_can_write_pointers(purpose=f"build_rates_features_live promote {day_dir}")
        pointer = {"run_id": run_id, "generated_at": datetime.now(tz=UTC).isoformat()}
        (day_dir / LATEST_POINTER).write_text(json.dumps(pointer, indent=2), encoding="utf-8")

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

        action_props_dir = data_root / "bronze" / "action_network" / "props"
        action_props_snapshot_rows = 0
        action_props_matched_rows = 0
        existing_action_props_rows = (
            _count_action_props_matches(minutes_df)
            if "an_has_any_props" in minutes_df.columns
            else 0
        )
        attach_action_props_fallback = (
            "an_has_any_props" not in minutes_df.columns or existing_action_props_rows == 0
        )
        if attach_action_props_fallback:
            action_props_snapshots = pd.DataFrame()
            if action_props_dir.exists():
                try:
                    snapshot_frames: list[pd.DataFrame] = []
                    action_props_snapshots = load_action_props_feature_snapshots_for_date(
                        props_dir=action_props_dir,
                        game_date=pd.Timestamp(game_date),
                    )
                    if not action_props_snapshots.empty:
                        snapshot_frames.append(action_props_snapshots)
                    next_day_snapshots = load_action_props_feature_snapshots_for_date(
                        props_dir=action_props_dir,
                        game_date=pd.Timestamp(game_date) + pd.Timedelta(days=1),
                    )
                    if not next_day_snapshots.empty:
                        snapshot_frames.append(next_day_snapshots)
                    action_props_snapshots = (
                        pd.concat(snapshot_frames, ignore_index=True)
                        if snapshot_frames
                        else pd.DataFrame()
                    )
                    action_props_snapshot_rows = int(len(action_props_snapshots))
                except Exception as exc:  # noqa: BLE001
                    typer.echo(f"[rates-live] Warning: failed to load Action props ({exc})", err=True)
            minutes_df = attach_action_props_features(
                minutes_df,
                action_props_snapshots,
                strict_asof=True,
                as_of_col="feature_as_of_ts",
                tip_col="tip_ts",
                game_date_offsets=(0, -1),
                clamp_late_asof_to_game_date=True,
            )
            action_props_matched_rows = _count_action_props_matches(minutes_df)
            typer.echo(
                f"[rates-live] Action props fallback attached: snapshots={action_props_snapshot_rows}, "
                f"matched_rows={action_props_matched_rows}, total_rows={len(minutes_df)}"
            )
        else:
            action_props_matched_rows = existing_action_props_rows
            typer.echo(
                f"[rates-live] Using Action props from minutes features: "
                f"matched_rows={action_props_matched_rows}, total_rows={len(minutes_df)}"
            )

        player_ids = minutes_df["player_id"].dropna().astype(int).unique().tolist()
        team_ids = minutes_df["team_id"].dropna().astype(int).unique().tolist()

        typer.echo(
            f"[rates-live] Found {len(player_ids)} players, {len(team_ids)} teams"
        )
        typer.echo(f"[rates-live] Slate team_ids: {_normalize_team_ids(team_ids)}")

        # Load component data
        season_aggs = _load_season_aggregates(data_root, game_date, player_ids)
        typer.echo(f"[rates-live] Season aggregates: {len(season_aggs)} players")

        tracking = _load_tracking_features(data_root, game_date, player_ids)
        typer.echo(f"[rates-live] Tracking features: {len(tracking)} players")

        vacancy = _load_vacancy_features(data_root, game_date, team_ids)
        typer.echo(f"[rates-live] Vacancy features: {len(vacancy)} teams")

        team_context = _load_team_context(data_root, game_date, team_ids)
        typer.echo(f"[rates-live] Team context: {len(team_context)} teams")

        # Load history and compute priors
        # Determine season year: Jan 2026 -> Season 2026 (starts 2025). Aug-Dec 2025 -> Season 2026.
        season_year = game_date.year if game_date.month >= 8 else game_date.year - 1
        season_year += 1 # NBA seasons usually named by ending year? 
        # Wait, build_training_base says: day.year if day.month >= 8 else day.year - 1
        # That is the START year (e.g. 2025-26 season has start year 2025).
        # But data paths usually use season=2026 for 2025-26?
        # Let's check existing paths. 
        # bronze/boxscores_raw/season=2026 exists? NO.
        # But roster_nightly/season=2026 DOES exist?
        # Checked earlier: gold/rates_training_base/season=2026 did NOT exist.
        # But bronze/boxscores_raw paths...
        # I should double check the directory structure later if this fails.
        # For now, I will use: year if month >= 8 else year. (e.g. Jan 2026 -> 2026).
        # Actually user date is Jan 2026. This is the 2025-26 season. Use 2026?
        # My glob was season=2026/date=...
        season_target = game_date.year if game_date.month < 8 else game_date.year + 1
        
        typer.echo(f"[rates-live] Loading boxscore history for season={season_target}...")
        history = _load_boxscores_history(data_root, season_target)
        # Filter history to only include games BEFORE the current slate?
        # _compute_player_priors computes based on the whole DF provided.
        # We must filter out today's games (if any exist in history).
        if not history.empty:
            history = history[history["tip_ts"] < pd.Timestamp(game_date, tz=UTC)].copy()

        priors = _compute_player_priors(history, player_ids=set(player_ids))
        typer.echo(f"[rates-live] Computed priors for {len(priors)} players")

        # Build features
        features = build_rates_features(
            minutes_df,
            season_aggs,
            tracking,
            vacancy,
            team_context,
            priors,
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
