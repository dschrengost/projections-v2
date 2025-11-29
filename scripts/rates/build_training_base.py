"""
Build the stage-0 training base for rates_v1 GBM models.

Inputs:
- Gold minutes labels: gold/labels_minutes_v1/season=YYYY/game_date=YYYY-MM-DD/labels.parquet
- Bronze boxscores: bronze/boxscores_raw/season=YYYY/date=YYYY-MM-DD/boxscores_raw.parquet
- Silver odds snapshots: silver/odds_snapshot/season=YYYY/month=MM/odds_snapshot.parquet
- Silver roster nightly: silver/roster_nightly/season=YYYY/month=MM/roster.parquet
- Silver injuries snapshot: silver/injuries_snapshot/... (schema only, features TODO)

Output:
- Gold rates training base: gold/rates_training_base/season=YYYY/game_date=YYYY-MM-DD/rates_training_base.parquet

Schema (per (season, game_id, team_id, player_id) with minutes_actual >= 4):
- identifiers: season, game_id, game_date, team_id, opponent_id, home_flag, player_id
- labels: minutes_actual, fga2_per_min, fga3_per_min, fta_per_min, ast_per_min, tov_per_min,
          oreb_per_min, dreb_per_min, stl_per_min, blk_per_min
- position: position_primary, position_flags_PG/SG/SF/PF/C
- season-to-date rates: season_fga_per_min, season_3pa_per_min, season_fta_per_min,
                        season_ast_per_min, season_tov_per_min, season_reb_per_min,
                        season_stl_per_min, season_blk_per_min
- minutes/opportunity: is_starter (minutes_expected/play_prob TODO hooks)
- game context: days_rest, spread_close, total_close, team_itt, opp_itt (pace/def TODO hooks)
- injury/vacated usage placeholders: num_rotation_players_out, team_minutes_vacated,
                                     team_usage_vacated, star_scorer_out_flag,
                                     primary_ballhandler_out_flag, starting_center_out_flag

Usage examples:
- Build a multi-season base (will overwrite any existing partitions):
    uv run python -m scripts.rates.build_training_base \
        --start-date 2023-10-01 \
        --end-date   2025-11-26 \
        --data-root  /home/daniel/projections-data \
        --output-root /home/daniel/projections-data/gold/rates_training_base

- Build while dropping feature-desert dates (from find_feature_desert_dates):
    uv run python -m scripts.rates.build_training_base \
        --start-date 2023-10-01 \
        --end-date   2025-11-26 \
        --data-root  /home/daniel/projections-data \
        --output-root /home/daniel/projections-data/gold/rates_training_base \
        --desert-csv /home/daniel/projections-data/artifacts/minutes_v1/feature_deserts.csv
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import typer

from projections.paths import data_path
from projections.minutes_v1.pos import canonical_pos_bucket
from projections.fpts_v1.datasets import (
    _coerce_ts,
    _iter_days,
    _parse_minutes_iso,
    _season_from_day,
)

app = typer.Typer(add_completion=False)


MIN_MINUTES = 4.0
OVERWRITE_PARTITIONS = True  # If True, existing day partitions are replaced; otherwise skipped.


@dataclass
class OddsSnapshot:
    game_id: int
    spread_home: float | None
    total: float | None
    as_of_ts: pd.Timestamp | None


def _season_start_from_day(day: pd.Timestamp) -> int:
    """Return NBA season start year (Aug–Jul)."""
    return day.year if day.month >= 8 else day.year - 1


def _read_parquet_tree(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_parquet(path)
    files = sorted(path.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_minutes_labels(data_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in _iter_days(start, end):
        season = _season_start_from_day(day)
        path = (
            data_root
            / "gold"
            / "labels_minutes_v1"
            / f"season={season}"
            / f"game_date={day.date().isoformat()}"
            / "labels.parquet"
        )
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"No minutes labels found between {start.date()} and {end.date()}")
    labels = pd.concat(frames, ignore_index=True)
    labels["game_date"] = pd.to_datetime(labels["game_date"]).dt.normalize()
    labels = labels[(labels["game_date"] >= start) & (labels["game_date"] <= end)].copy()
    labels["season"] = labels["season"].astype(int)
    for key in ("game_id", "team_id", "player_id"):
        if key in labels.columns:
            labels[key] = pd.to_numeric(labels[key], errors="coerce").astype("Int64")
    # Coerce minutes to float
    labels["minutes"] = pd.to_numeric(labels["minutes"], errors="coerce")
    mask_nan = labels["minutes"].isna()
    if mask_nan.any():
        labels.loc[mask_nan, "minutes"] = labels.loc[mask_nan, "minutes"].apply(_parse_minutes_iso)
    labels.rename(columns={"minutes": "minutes_actual"}, inplace=True)
    labels["minutes_actual"] = labels["minutes_actual"].astype(float)
    return labels


def _bronze_paths(data_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> Iterable[Path]:
    seen: set[tuple[int, str]] = set()
    for day in _iter_days(start, end):
        season = _season_start_from_day(day)
        token = day.date().isoformat()
        key = (season, token)
        if key in seen:
            continue
        seen.add(key)
        candidate = (
            data_root
            / "bronze"
            / "boxscores_raw"
            / f"season={season}"
            / f"date={token}"
            / "boxscores_raw.parquet"
        )
        if candidate.exists():
            yield candidate


def load_boxscores(data_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for path in _bronze_paths(data_root, start, end):
        bronze = pd.read_parquet(path)
        for row in bronze.itertuples():
            payload = json.loads(row.payload)
            tip_ts = _coerce_ts(payload.get("game_time_utc") or payload.get("game_time_local"))
            if tip_ts is None:
                continue
            game_date = tip_ts.tz_convert("America/New_York").tz_localize(None).normalize()
            season = _season_start_from_day(game_date)
            game_id = int(str(payload.get("game_id") or row.game_id).zfill(10))
            home = payload.get("home") or {}
            away = payload.get("away") or {}
            for side, team_payload, opp_payload in (("home", home, away), ("away", away, home)):
                team_id = int(team_payload.get("team_id") or team_payload.get("teamId") or 0)
                opponent_id = int(opp_payload.get("team_id") or opp_payload.get("teamId") or 0)
                home_flag = 1 if side == "home" else 0
                for player in team_payload.get("players", []):
                    stats = player.get("statistics") or {}
                    minutes = _parse_minutes_iso(stats.get("minutes"))
                    points = float(stats.get("points") or 0.0)
                    fga = float(stats.get("fieldGoalsAttempted") or 0.0)
                    fta = float(stats.get("freeThrowsAttempted") or 0.0)
                    three_pa = float(stats.get("threePointersAttempted") or 0.0)
                    assists = float(stats.get("assists") or 0.0)
                    turnovers = float(stats.get("turnovers") or 0.0)
                    oreb = float(stats.get("reboundsOffensive") or 0.0)
                    dreb = float(stats.get("reboundsDefensive") or 0.0)
                    steals = float(stats.get("steals") or 0.0)
                    blocks = float(stats.get("blocks") or 0.0)
                    records.append(
                        {
                            "game_id": game_id,
                            "player_id": int(
                                player.get("person_id") or player.get("personId") or 0
                            ),
                            "team_id": team_id,
                            "opponent_id": opponent_id,
                            "home_flag": home_flag,
                            "game_date": game_date,
                            "tip_ts": tip_ts,
                            "season": season,
                            "minutes_played": minutes,
                            "points": points,
                            "fga": fga,
                            "three_pa": three_pa,
                            "fta": fta,
                            "assists": assists,
                            "turnovers": turnovers,
                            "oreb": oreb,
                            "dreb": dreb,
                            "steals": steals,
                            "blocks": blocks,
                            "starter_flag_box": bool(player.get("starter") or player.get("starter_flag")),
                        }
                    )
    if not records:
        raise FileNotFoundError(f"No boxscore stat lines found between {start.date()} and {end.date()}")
    stats = pd.DataFrame.from_records(records)
    stats.sort_values(["player_id", "tip_ts"], inplace=True)
    return stats


def load_roster(data_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    roster_dir = data_root / "silver" / "roster_nightly"
    roster = _read_parquet_tree(roster_dir)
    if roster.empty:
        return roster
    roster["game_date"] = pd.to_datetime(roster["game_date"]).dt.normalize()
    mask = (roster["game_date"] >= start) & (roster["game_date"] <= end)
    return roster.loc[mask].copy()


def _latest_snapshot_leq_tip(df: pd.DataFrame, tip_ts: pd.Timestamp) -> pd.Series | None:
    if df.empty:
        return None
    df["as_of_ts"] = pd.to_datetime(df["as_of_ts"], utc=True, errors="coerce")
    candidates = df[df["as_of_ts"] <= tip_ts]
    if candidates.empty:
        return None
    return candidates.sort_values("as_of_ts").iloc[-1]


def load_odds(
    data_root: Path, start: pd.Timestamp, end: pd.Timestamp, tips: pd.Series
) -> pd.DataFrame:
    odds_dir = data_root / "silver" / "odds_snapshot"
    raw = _read_parquet_tree(odds_dir)
    if raw.empty:
        return pd.DataFrame()
    raw["as_of_ts"] = pd.to_datetime(raw["as_of_ts"], utc=True, errors="coerce")
    raw["game_id"] = pd.to_numeric(raw["game_id"], errors="coerce").astype("Int64")
    raw = raw.dropna(subset=["game_id"])
    raw["game_id"] = raw["game_id"].astype(int)
    odds_rows: list[dict] = []
    tip_map = tips.dropna().to_dict()
    for game_id, tip_ts in tip_map.items():
        subset = raw[raw["game_id"] == game_id]
        row = _latest_snapshot_leq_tip(subset.copy(), tip_ts)
        if row is None:
            # TODO: enforce tip-aware odds selection once tip_ts coverage is guaranteed.
            continue
        odds_rows.append(
            {
                "game_id": game_id,
                "spread_home": float(row.get("spread_home")) if pd.notna(row.get("spread_home")) else np.nan,
                "total_close": float(row.get("total")) if pd.notna(row.get("total")) else np.nan,
                "as_of_ts": row.get("as_of_ts"),
            }
        )
    return pd.DataFrame(odds_rows)


def load_injuries(data_root: Path) -> pd.DataFrame:
    injuries_dir = data_root / "silver" / "injuries_snapshot"
    return _read_parquet_tree(injuries_dir)


def load_minutes_predictions(data_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Optional precomputed minutes predictions for rates (minutes_for_rates).
    Expected columns: season, game_id, game_date, team_id, player_id,
    minutes_pred_p10, minutes_pred_p50, minutes_pred_p90, minutes_pred_play_prob.
    """
    root = data_root / "gold" / "minutes_for_rates"
    if not root.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for season_dir in root.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if start <= day <= end:
                path = day_dir / "minutes_for_rates.parquet"
                if path.exists():
                    frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


def load_tracking_roles(data_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Optional tracking role features built from player tracking data.
    Expected columns: season, game_date, game_id, team_id, player_id, track_* features.
    """

    root = data_root / "gold" / "tracking_roles"
    if not root.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for season_dir in root.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if start <= day <= end:
                path = day_dir / "tracking_roles.parquet"
                if path.exists():
                    frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


def _seasonal_cumulative_avg(
    values: pd.Series, weights: pd.Series, player_ids: pd.Series, seasons: pd.Series, *, min_weight: float
) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    weighted = numeric_values * numeric_weights
    shifted_weighted = weighted.groupby([player_ids, seasons]).shift(1)
    shifted_weights = numeric_weights.groupby([player_ids, seasons]).shift(1)
    cum_weighted = shifted_weighted.groupby([player_ids, seasons]).cumsum()
    cum_weights = shifted_weights.groupby([player_ids, seasons]).cumsum()
    denom = cum_weights.replace(0.0, np.nan)
    avg = cum_weighted / denom
    return avg.where(cum_weights >= min_weight)


def _prepare_roster_latest(roster: pd.DataFrame) -> pd.DataFrame:
    roster_cols = [
        "game_id",
        "team_id",
        "player_id",
        "starter_flag",
        "is_confirmed_starter",
        "is_projected_starter",
        "listed_pos",
        "as_of_ts",
        "lineup_role",
        "game_date",
    ]
    roster_cols = [c for c in roster_cols if c in roster.columns]
    if not roster_cols:
        return pd.DataFrame(columns=["game_id", "team_id", "player_id"])
    roster_sub = roster[roster_cols].copy()
    if "as_of_ts" in roster_sub.columns:
        roster_sub["as_of_ts"] = pd.to_datetime(roster_sub["as_of_ts"], errors="coerce", utc=True)
        roster_sub.sort_values("as_of_ts", inplace=True)
    return roster_sub.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last")


def _player_history_totals(stats: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date cumulative totals per player (excluding current game)."""

    label_minutes = labels[["game_id", "player_id", "minutes_actual"]] if "minutes_actual" in labels.columns else pd.DataFrame()
    hist = stats.merge(label_minutes, on=["game_id", "player_id"], how="left", suffixes=("", "_label"))
    hist["minutes_hist"] = hist["minutes_actual"].fillna(hist["minutes_played"]).astype(float)
    hist.sort_values(["season", "player_id", "tip_ts"], inplace=True)

    hist["hist_minutes_szn"] = hist.groupby(["season", "player_id"])["minutes_hist"].cumsum().shift(1)
    hist["hist_fga_szn"] = hist.groupby(["season", "player_id"])["fga"].cumsum().shift(1)
    hist["hist_3pa_szn"] = hist.groupby(["season", "player_id"])["three_pa"].cumsum().shift(1)
    hist["hist_fta_szn"] = hist.groupby(["season", "player_id"])["fta"].cumsum().shift(1)
    hist["hist_ast_szn"] = hist.groupby(["season", "player_id"])["assists"].cumsum().shift(1)
    hist_cols = ["hist_minutes_szn", "hist_fga_szn", "hist_3pa_szn", "hist_fta_szn", "hist_ast_szn"]
    hist[hist_cols] = hist[hist_cols].fillna(0.0)
    return hist[
        [
            "season",
            "game_id",
            "team_id",
            "player_id",
            "game_date",
            "tip_ts",
            "hist_minutes_szn",
            "hist_fga_szn",
            "hist_3pa_szn",
            "hist_fta_szn",
            "hist_ast_szn",
        ]
    ]


def _compute_vacated_team_features(
    stats: pd.DataFrame, labels: pd.DataFrame, roster_sub: pd.DataFrame, injuries: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate season-to-date totals for players flagged OUT-like for each team/game."""

    if injuries.empty or stats.empty:
        return pd.DataFrame()

    tips = (
        stats.drop_duplicates(subset=["game_id"])[["game_id", "tip_ts", "season", "game_date"]]
        .dropna(subset=["tip_ts"])
        .copy()
    )
    injuries_norm = injuries.copy()
    injuries_norm["as_of_ts"] = pd.to_datetime(injuries_norm["as_of_ts"], errors="coerce", utc=True)
    injuries_norm["game_id"] = pd.to_numeric(injuries_norm["game_id"], errors="coerce").astype("Int64")
    injuries_norm["player_id"] = pd.to_numeric(injuries_norm["player_id"], errors="coerce").astype("Int64")
    if "team_id" in injuries_norm.columns:
        injuries_norm["team_id"] = pd.to_numeric(injuries_norm["team_id"], errors="coerce").astype("Int64")
    injuries_norm = injuries_norm.dropna(subset=["game_id", "player_id", "as_of_ts"])
    injuries_norm["game_id"] = injuries_norm["game_id"].astype(int)
    injuries_norm["player_id"] = injuries_norm["player_id"].astype(int)
    injuries_norm = injuries_norm.merge(tips, on="game_id", how="inner")
    injuries_norm = injuries_norm[injuries_norm["as_of_ts"] <= injuries_norm["tip_ts"]]
    if injuries_norm.empty:
        return pd.DataFrame()

    injuries_norm.sort_values(["game_id", "player_id", "as_of_ts"], inplace=True)
    latest = injuries_norm.groupby(["game_id", "player_id"], as_index=False).tail(1)

    # Attach team + position from roster (if not already present)
    roster_map_cols = [c for c in ["game_id", "team_id", "player_id", "listed_pos"] if c in roster_sub.columns]
    roster_map = roster_sub[roster_map_cols].drop_duplicates(subset=["game_id", "player_id"]) if roster_map_cols else pd.DataFrame()
    if "team_id" not in latest.columns or latest["team_id"].isna().any():
        latest = latest.merge(roster_map, on=["game_id", "player_id"], how="left", suffixes=("", "_roster"))
        if "team_id" not in latest.columns and "team_id_roster" in latest.columns:
            latest.rename(columns={"team_id_roster": "team_id"}, inplace=True)
        elif "team_id_roster" in latest.columns:
            latest["team_id"] = latest["team_id"].fillna(latest["team_id_roster"])
        latest.drop(columns=[c for c in ["team_id_roster"] if c in latest.columns], inplace=True)
    if "listed_pos" not in latest.columns and not roster_map.empty:
        latest = latest.merge(roster_map[["game_id", "player_id", "listed_pos"]], on=["game_id", "player_id"], how="left")
    latest = latest.dropna(subset=["team_id"])
    if latest.empty:
        return pd.DataFrame()
    latest["team_id"] = latest["team_id"].astype(int)

    latest["status_norm"] = latest["status"].astype(str).str.upper().str.strip()
    out_like_values = {"OUT", "DOUBTFUL", "QUESTIONABLE", "INACTIVE"}
    latest["is_out_like"] = latest["status_norm"].isin(out_like_values)
    latest = latest[latest["is_out_like"]]
    if latest.empty:
        return pd.DataFrame()

    player_hist = _player_history_totals(stats, labels)
    hist_cols = [
        "hist_minutes_szn",
        "hist_fga_szn",
        "hist_3pa_szn",
        "hist_fta_szn",
        "hist_ast_szn",
    ]
    player_hist = player_hist.sort_values(["tip_ts", "season", "player_id"]).reset_index(drop=True)
    latest = latest.sort_values(["tip_ts", "season", "player_id"]).reset_index(drop=True)
    latest = pd.merge_asof(
        latest,
        player_hist[["season", "player_id", "tip_ts"] + hist_cols],
        by=["season", "player_id"],
        left_on="tip_ts",
        right_on="tip_ts",
        direction="backward",
        allow_exact_matches=True,
    )
    for col in hist_cols:
        latest[col] = latest[col].fillna(0.0)

    if "listed_pos" in latest.columns:
        latest["pos_bucket"] = latest["listed_pos"].fillna("UNK").apply(canonical_pos_bucket)
    else:
        latest["pos_bucket"] = "UNK"
    latest["pos_group"] = latest["pos_bucket"].map({"G": "G", "W": "W", "BIG": "B"}).fillna("UNK")

    group_cols = ["season", "game_id", "team_id"]
    grouped = latest.groupby(group_cols).agg(
        vac_min_szn=("hist_minutes_szn", "sum"),
        vac_fga_szn=("hist_fga_szn", "sum"),
        vac_ast_szn=("hist_ast_szn", "sum"),
    )
    grouped = grouped.reset_index()

    for bucket, col_name in [("G", "vac_min_guard_szn"), ("W", "vac_min_wing_szn"), ("B", "vac_min_big_szn")]:
        pos_agg = (
            latest[latest["pos_group"] == bucket]
            .groupby(group_cols)["hist_minutes_szn"]
            .sum()
            .reset_index(name=col_name)
        )
        grouped = grouped.merge(pos_agg, on=group_cols, how="left")

    for col in ["vac_min_szn", "vac_fga_szn", "vac_ast_szn", "vac_min_guard_szn", "vac_min_wing_szn", "vac_min_big_szn"]:
        if col not in grouped.columns:
            grouped[col] = 0.0
        grouped[col] = grouped[col].fillna(0.0)

    return grouped


def _compute_team_context(stats: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date pace/off/def context per team/game (excluding current game)."""

    if stats.empty:
        return pd.DataFrame()

    agg_cols = ["season", "game_id", "team_id", "opponent_id", "game_date", "tip_ts"]
    team_game = (
        stats.groupby(agg_cols, as_index=False)
        .agg(points=("points", "sum"), fga=("fga", "sum"), fta=("fta", "sum"), tov=("turnovers", "sum"))
        .rename(columns={"points": "pts_for", "tov": "tov"})
    )
    team_game["poss"] = team_game["fga"] + 0.44 * team_game["fta"] + team_game["tov"]

    opp_map = team_game[["season", "game_id", "team_id", "pts_for", "poss"]].rename(
        columns={"team_id": "opponent_id", "pts_for": "pts_against", "poss": "opp_poss"}
    )
    team_game = team_game.merge(opp_map, on=["season", "game_id", "opponent_id"], how="left")

    team_game.sort_values(["season", "team_id", "tip_ts"], inplace=True)
    team_game["games_played_prior"] = team_game.groupby(["season", "team_id"]).cumcount()
    team_game["cum_poss"] = team_game.groupby(["season", "team_id"])["poss"].cumsum().shift(1)
    team_game["cum_pts_for"] = team_game.groupby(["season", "team_id"])["pts_for"].cumsum().shift(1)
    team_game["cum_pts_against"] = team_game.groupby(["season", "team_id"])["pts_against"].cumsum().shift(1)

    games = team_game["games_played_prior"].replace(0, np.nan)
    poss_denom = team_game["cum_poss"].replace(0.0, np.nan)
    team_game["team_pace_szn"] = team_game["cum_poss"] / games
    team_game["team_off_rtg_szn"] = 100.0 * (team_game["cum_pts_for"] / poss_denom)
    team_game["team_def_rtg_szn"] = 100.0 * (team_game["cum_pts_against"] / poss_denom)

    return team_game[["season", "game_id", "team_id", "team_pace_szn", "team_off_rtg_szn", "team_def_rtg_szn"]]


def build_features(
    labels: pd.DataFrame,
    stats: pd.DataFrame,
    roster: pd.DataFrame,
    odds: pd.DataFrame,
    minutes_preds: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    roster_sub = _prepare_roster_latest(roster)
    label_cols = [c for c in ["game_id", "player_id", "team_id", "minutes_actual", "starter_flag", "listed_pos"] if c in labels.columns]
    df = stats.merge(
        labels[label_cols],
        how="left",
        on=["game_id", "player_id", "team_id"],
    )
    # Use label minutes if available, otherwise fall back to boxscore minutes.
    df.rename(columns={"minutes_actual": "minutes_actual_label"}, inplace=True)
    df["minutes_actual"] = df["minutes_actual_label"].fillna(df["minutes_played"]).astype(float)
    # Starter flag resolution:
    # Prefer roster starter_flag/is_confirmed_starter/is_projected_starter; fall back to label starter_flag.
    # Missing roster rows default to bench (0).
    if "starter_flag" in df.columns:
        df.rename(columns={"starter_flag": "starter_flag_label"}, inplace=True)
    df = df.merge(roster_sub, on=["game_id", "team_id", "player_id"], how="left")
    if "lineup_role" in df.columns:
        df["starter_flag_lineup"] = df["lineup_role"].isin({"confirmed_starter", "projected_starter"}).astype("Int64")

    starter_sources = [
        col
        for col in [
            "starter_flag_lineup",
            "starter_flag_box",
            "starter_flag_label",
            "is_confirmed_starter",
            "is_projected_starter",
            "starter_flag",
        ]
        if col in df
    ]
    starter_sources = list(dict.fromkeys(starter_sources))  # de-duplicate while preserving order
    if starter_sources:
        starter_frame = df[starter_sources].copy()
        starter_series = starter_frame.bfill(axis=1).iloc[:, 0].fillna(0)
        df["is_starter"] = starter_series.astype(int)
    else:
        df["is_starter"] = 0

    df = df[df["minutes_actual"] >= MIN_MINUTES].copy()

    # Position features
    if "listed_pos" in df.columns:
        df["position_primary"] = df["listed_pos"].fillna("UNK").apply(canonical_pos_bucket)
    elif "listed_pos" in roster_sub.columns:
        df = df.merge(
            roster_sub[["game_id", "player_id", "listed_pos"]],
            on=["game_id", "player_id"],
            how="left",
        )
        df["position_primary"] = df["listed_pos"].fillna("UNK").apply(canonical_pos_bucket)
    else:
        df["position_primary"] = "UNK"
    for pos in ("PG", "SG", "SF", "PF", "C"):
        df[f"position_flags_{pos}"] = (df["position_primary"] == pos).astype(int)

    # Labels (per-minute rates)
    minutes = df["minutes_actual"]
    df["fga2_per_min"] = (df["fga"] - df["three_pa"]) / minutes
    df["fga3_per_min"] = df["three_pa"] / minutes
    df["fta_per_min"] = df["fta"] / minutes
    df["ast_per_min"] = df["assists"] / minutes
    df["tov_per_min"] = df["turnovers"] / minutes
    df["oreb_per_min"] = df["oreb"] / minutes
    df["dreb_per_min"] = df["dreb"] / minutes
    df["stl_per_min"] = df["steals"] / minutes
    df["blk_per_min"] = df["blocks"] / minutes

    # Season-to-date per-minute averages (exclude current game via shift)
    cumulative_specs = [
        ("fga", "season_fga_per_min"),
        ("three_pa", "season_3pa_per_min"),
        ("fta", "season_fta_per_min"),
        ("assists", "season_ast_per_min"),
        ("turnovers", "season_tov_per_min"),
        ("rebounds_total", "season_reb_per_min"),
        ("steals", "season_stl_per_min"),
        ("blocks", "season_blk_per_min"),
    ]
    for col, out_col in cumulative_specs:
        values = df["oreb"] + df["dreb"] if col == "rebounds_total" else df[col]
        df[out_col] = _seasonal_cumulative_avg(
            values=values,
            weights=df["minutes_actual"],
            player_ids=df["player_id"],
            seasons=df["season"],
            min_weight=10.0,
        )

    # Days rest (clipped 0–3+)
    df["days_rest"] = (
        df.groupby("player_id")["tip_ts"]
        .diff()
        .dt.total_seconds()
        .div(86400)
        .round()
        .clip(lower=0, upper=3)
    )
    df["days_rest"] = df["days_rest"].astype("Int64")

    # Odds join (tip-aware snapshots)
    if not odds.empty:
        df = df.merge(odds[["game_id", "spread_home", "total_close"]], on="game_id", how="left")
    else:
        df["spread_home"] = np.nan
        df["total_close"] = np.nan
    df["spread_close"] = df["spread_home"]
    # Persist has_odds so training/scoring share the same odds-availability logic.
    df["has_odds"] = (~df["spread_close"].isna()) & (~df["total_close"].isna())
    # Implied team totals
    def _implied_totals(row: pd.Series) -> tuple[float | np.nan, float | np.nan]:
        total = row.get("total_close")
        spread_home = row.get("spread_home")
        if pd.isna(total) or pd.isna(spread_home):
            return np.nan, np.nan
        home_itt = total / 2 - spread_home / 2
        away_itt = total - home_itt
        return (home_itt, away_itt)

    implied = df.apply(_implied_totals, axis=1, result_type="expand")
    df["team_itt"] = np.where(df["home_flag"] == 1, implied[0], implied[1])
    df["opp_itt"] = np.where(df["home_flag"] == 1, implied[1], implied[0])

    # Team-level vacated minutes/usage from injuries
    vacated_team = _compute_vacated_team_features(stats, labels, roster_sub, injuries)
    if vacated_team.empty:
        df["vac_min_szn"] = 0.0
        df["vac_fga_szn"] = 0.0
        df["vac_ast_szn"] = 0.0
        df["vac_min_guard_szn"] = 0.0
        df["vac_min_wing_szn"] = 0.0
        df["vac_min_big_szn"] = 0.0
    else:
        df = df.merge(vacated_team, on=["season", "game_id", "team_id"], how="left")
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
        # Legacy aliases for backward compatibility
        df["team_minutes_vacated"] = df["vac_min_szn"]
        df["team_usage_vacated"] = df["vac_fga_szn"]
    if "team_minutes_vacated" not in df.columns:
        df["team_minutes_vacated"] = df["vac_min_szn"]
    if "team_usage_vacated" not in df.columns:
        df["team_usage_vacated"] = df["vac_fga_szn"]

    # Pace and defensive context
    team_context = _compute_team_context(stats)
    if team_context.empty:
        df["team_pace_szn"] = np.nan
        df["team_off_rtg_szn"] = np.nan
        df["team_def_rtg_szn"] = np.nan
        df["opp_pace_szn"] = np.nan
        df["opp_def_rtg_szn"] = np.nan
        pace_non_null = 0.0
        def_non_null = 0.0
    else:
        df = df.merge(team_context, on=["season", "game_id", "team_id"], how="left")
        opp_context = team_context.rename(
            columns={
                "team_id": "opponent_id",
                "team_pace_szn": "opp_pace_szn",
                "team_off_rtg_szn": "opp_off_rtg_szn",
                "team_def_rtg_szn": "opp_def_rtg_szn",
            }
        )
        df = df.merge(opp_context[["season", "game_id", "opponent_id", "opp_pace_szn", "opp_def_rtg_szn"]], on=["season", "game_id", "opponent_id"], how="left")
        pace_non_null = 1.0 - df["team_pace_szn"].isna().mean()
        def_non_null = 1.0 - df["team_def_rtg_szn"].isna().mean()
    typer.echo(f"[rates_base] team_pace_szn coverage: {pace_non_null:.3%}; team_def_rtg_szn coverage: {def_non_null:.3%}")
    pace_fill_cols = ["team_pace_szn", "team_off_rtg_szn", "team_def_rtg_szn", "opp_pace_szn", "opp_def_rtg_szn"]
    for col in pace_fill_cols:
        if col not in df.columns:
            df[col] = np.nan
        mean_val = df[col].mean(skipna=True)
        fill_val = 0.0 if pd.isna(mean_val) else mean_val
        df[col] = df[col].fillna(fill_val)
    # Legacy placeholders retained for compatibility
    df["season_pace_team"] = df["team_pace_szn"]
    df["season_pace_opp"] = df["opp_pace_szn"]
    df["opp_def_rating"] = df["opp_def_rtg_szn"]
    df["opp_def_3pa_allowed"] = pd.NA
    df["opp_def_reb_rate"] = pd.NA
    df["opp_def_ast_rate"] = pd.NA
    df["num_rotation_players_out"] = pd.NA
    df["star_scorer_out_flag"] = pd.NA
    df["primary_ballhandler_out_flag"] = pd.NA
    df["starting_center_out_flag"] = pd.NA

    # Optional minutes predictions (future Stage 1). Keep minutes_actual for Stage 0.
    if not minutes_preds.empty:
        cols = [
            "season",
            "game_id",
            "game_date",
            "team_id",
            "player_id",
            "minutes_pred_p10",
            "minutes_pred_p50",
            "minutes_pred_p90",
            "minutes_pred_play_prob",
        ]
        cols = [c for c in cols if c in minutes_preds.columns]
        join_keys = ["season", "game_id", "team_id", "player_id"]
        if "game_date" in minutes_preds.columns and "game_date" in df.columns:
            join_keys.append("game_date")
        df = df.merge(minutes_preds[cols], on=join_keys, how="left")
    else:
        df["minutes_pred_p10"] = pd.NA
        df["minutes_pred_p50"] = pd.NA
        df["minutes_pred_p90"] = pd.NA
        df["minutes_pred_play_prob"] = pd.NA

    # Minutes/opportunity features (only is_starter + minutes_actual for now)
    # TODO: add minutes_expected_p50, minutes_spread, play_prob from historical minutes scoring bundle.

    # Final column selection
    columns = [
        "season",
        "game_id",
        "game_date",
        "team_id",
        "opponent_id",
        "home_flag",
        "player_id",
        "minutes_actual",
        "fga2_per_min",
        "fga3_per_min",
        "fta_per_min",
        "ast_per_min",
        "tov_per_min",
        "oreb_per_min",
        "dreb_per_min",
        "stl_per_min",
        "blk_per_min",
        "position_primary",
        "position_flags_PG",
        "position_flags_SG",
        "position_flags_SF",
        "position_flags_PF",
        "position_flags_C",
        "season_fga_per_min",
        "season_3pa_per_min",
        "season_fta_per_min",
        "season_ast_per_min",
        "season_tov_per_min",
        "season_reb_per_min",
        "season_stl_per_min",
        "season_blk_per_min",
        "is_starter",
        "days_rest",
        "spread_close",
        "total_close",
        "team_itt",
        "opp_itt",
        "has_odds",
        "team_pace_szn",
        "team_off_rtg_szn",
        "team_def_rtg_szn",
        "opp_pace_szn",
        "opp_def_rtg_szn",
        "vac_min_szn",
        "vac_fga_szn",
        "vac_ast_szn",
        "vac_min_guard_szn",
        "vac_min_wing_szn",
        "vac_min_big_szn",
        "team_minutes_vacated",
        "team_usage_vacated",
        "season_pace_team",
        "season_pace_opp",
        "opp_def_rating",
        "opp_def_3pa_allowed",
        "opp_def_reb_rate",
        "opp_def_ast_rate",
        "num_rotation_players_out",
        "star_scorer_out_flag",
        "primary_ballhandler_out_flag",
        "starting_center_out_flag",
        "minutes_pred_p10",
        "minutes_pred_p50",
        "minutes_pred_p90",
        "minutes_pred_play_prob",
    ]
    return df[columns]


def _write_partitions(df: pd.DataFrame, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    grouped = df.groupby(["season", "game_date"])
    for (season, game_date), frame in grouped:
        partition_dir = (
            output_root
            / f"season={int(season)}"
            / f"game_date={pd.Timestamp(game_date).date().isoformat()}"
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        output_path = partition_dir / "rates_training_base.parquet"
        if output_path.exists() and not OVERWRITE_PARTITIONS:
            typer.echo(f"[rates] skipping existing partition {output_path}")
            continue
        frame.to_parquet(output_path, index=False)


@app.command()
def main(
    start_date: str = typer.Option(..., help="Start date (YYYY-MM-DD) inclusive."),
    end_date: str = typer.Option(..., help="End date (YYYY-MM-DD) inclusive."),
    data_root: Optional[Path] = typer.Option(
        None,
        help="Root containing bronze/silver/gold data (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    output_root: Optional[Path] = typer.Option(
        None,
        help="Output root for gold/rates_training_base (defaults under data_root/gold).",
    ),
    desert_csv: Optional[Path] = typer.Option(
        None,
        help="Optional CSV of feature-desert dates; rows for those game_dates will be dropped.",
    ),
) -> None:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    root = data_root or data_path()
    out_root = output_root or (root / "gold" / "rates_training_base")

    typer.echo(f"[rates] building training base from {start.date()} to {end.date()} using data_root={root}")
    if OVERWRITE_PARTITIONS:
        typer.echo("[rates] existing partitions will be overwritten.")
    else:
        typer.echo("[rates] existing partitions will be skipped.")

    labels = load_minutes_labels(root, start, end)
    stats = load_boxscores(root, start, end)
    roster = load_roster(root, start, end)
    odds = load_odds(
        root,
        start,
        end,
        stats.drop_duplicates(subset=["game_id"]).set_index("game_id")["tip_ts"],
    )
    minutes_preds = load_minutes_predictions(root, start, end)
    tracking_roles = load_tracking_roles(root, start, end)
    injuries = load_injuries(root)  # reserved for future vacated usage features
    if injuries.empty:
        typer.echo("[rates] injuries snapshot empty; vacated usage features will remain placeholders.")

    features = build_features(labels, stats, roster, odds, minutes_preds, injuries)
    if desert_csv:
        desert_df = pd.read_csv(desert_csv)
        if "game_date" in desert_df.columns:
            desert_df["game_date"] = pd.to_datetime(desert_df["game_date"]).dt.normalize()
            desert_col = desert_df.get("is_desert")
            partial_col = desert_df.get("is_partial_desert")
            mask = (
                (desert_col.fillna(False).astype(bool) if desert_col is not None else False)
                | (partial_col.fillna(False).astype(bool) if partial_col is not None else False)
            )
            desert_dates = set(desert_df.loc[mask, "game_date"].dt.date.tolist())
            if desert_dates:
                before = len(features)
                features["game_date"] = pd.to_datetime(features["game_date"]).dt.normalize()
                drop_mask = features["game_date"].dt.date.isin(desert_dates)
                features = features[~drop_mask].copy()
                dropped = before - len(features)
                typer.echo(
                    f"[rates] dropped {dropped} rows from feature-desert dates ({len(desert_dates)} dates)"
                )
    # Ensure datetime dtype before joins
    features["game_date"] = pd.to_datetime(features["game_date"]).dt.normalize()
    if not tracking_roles.empty:
        track_cols = [
            "season",
            "game_date",
            "game_id",
            "team_id",
            "player_id",
            "track_touches_per_min_szn",
            "track_sec_per_touch_szn",
            "track_pot_ast_per_min_szn",
            "track_drives_per_min_szn",
            "track_role_cluster",
            "track_role_is_low_minutes",
        ]
        track_cols = [c for c in track_cols if c in tracking_roles.columns]
        features = features.merge(
            tracking_roles[track_cols],
            on=["season", "game_date", "game_id", "team_id", "player_id"],
            how="left",
        )
    else:
        features["track_touches_per_min_szn"] = np.nan
        features["track_sec_per_touch_szn"] = np.nan
        features["track_pot_ast_per_min_szn"] = np.nan
        features["track_drives_per_min_szn"] = np.nan
        features["track_role_cluster"] = np.nan
        features["track_role_is_low_minutes"] = np.nan
    n_total = len(features)
    if "minutes_pred_p50" in features.columns:
        n_missing_pred = features["minutes_pred_p50"].isna().sum()
    else:
        n_missing_pred = n_total
    typer.echo(f"[rates_base] minutes_pred_p50 missing for {n_missing_pred}/{n_total} rows")
    track_missing = (
        features["track_touches_per_min_szn"].isna().sum()
        if "track_touches_per_min_szn" in features.columns
        else len(features)
    )
    typer.echo(f"[rates_base] tracking features missing for {track_missing}/{len(features)} rows")
    track_fill_cols = [
        "track_touches_per_min_szn",
        "track_sec_per_touch_szn",
        "track_pot_ast_per_min_szn",
        "track_drives_per_min_szn",
    ]
    for col in track_fill_cols:
        if col not in features.columns:
            features[col] = np.nan
        mean_val = features[col].mean(skipna=True)
        fill_val = 0.0 if pd.isna(mean_val) else mean_val
        features[col] = features[col].fillna(fill_val)
    if "track_role_cluster" in features.columns:
        features["track_role_cluster"] = features["track_role_cluster"].fillna(-1).astype(int)
    else:
        features["track_role_cluster"] = -1
    if "track_role_is_low_minutes" in features.columns:
        features["track_role_is_low_minutes"] = (
            features["track_role_is_low_minutes"].fillna(True).astype(bool)
        )
    else:
        features["track_role_is_low_minutes"] = True
    vac_frac = (features["vac_min_szn"] > 0).mean() if "vac_min_szn" in features.columns else 0.0
    typer.echo(f"[rates_base] vacated_minutes>0 for {vac_frac:.3%} of rows")
    _write_partitions(features, out_root)

    min_date = pd.to_datetime(features["game_date"]).min()
    max_date = pd.to_datetime(features["game_date"]).max()
    typer.echo(
        f"[rates] wrote {len(features):,} rows across {min_date.date()}–{max_date.date()}"
    )
    sample_cols = [
        "season",
        "game_id",
        "team_id",
        "player_id",
        "minutes_actual",
        "fga2_per_min",
        "fga3_per_min",
        "ast_per_min",
        "is_starter",
        "team_itt",
        "opp_itt",
    ]
    typer.echo(features[sample_cols].head().to_string(index=False))


if __name__ == "__main__":
    app()
