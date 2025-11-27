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


def build_features(
    labels: pd.DataFrame,
    stats: pd.DataFrame,
    roster: pd.DataFrame,
    odds: pd.DataFrame,
    minutes_preds: pd.DataFrame,
) -> pd.DataFrame:
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
    if not roster.empty:
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
        ]
        roster_cols = [c for c in roster_cols if c in roster.columns]
        roster_sub = roster[roster_cols].copy()
        if "as_of_ts" in roster_sub.columns:
            roster_sub["as_of_ts"] = pd.to_datetime(roster_sub["as_of_ts"], errors="coerce", utc=True)
            roster_sub.sort_values("as_of_ts", inplace=True)
        roster_sub = roster_sub.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last")
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
    elif "listed_pos" in roster.columns:
        df = df.merge(
            roster[["game_id", "player_id", "listed_pos"]],
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

    # Injury/vacated placeholders (feature columns exist; logic TODO)
    df["num_rotation_players_out"] = pd.NA
    df["team_minutes_vacated"] = pd.NA
    df["team_usage_vacated"] = pd.NA
    df["star_scorer_out_flag"] = pd.NA
    df["primary_ballhandler_out_flag"] = pd.NA
    df["starting_center_out_flag"] = pd.NA

    # Pace/defensive placeholders (TODO: join team-level metrics)
    df["season_pace_team"] = pd.NA
    df["season_pace_opp"] = pd.NA
    df["opp_def_rating"] = pd.NA
    df["opp_def_3pa_allowed"] = pd.NA
    df["opp_def_reb_rate"] = pd.NA
    df["opp_def_ast_rate"] = pd.NA

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
        df = df.merge(minutes_preds[cols], on=["season", "game_id", "game_date", "team_id", "player_id"], how="left")
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
        "season_pace_team",
        "season_pace_opp",
        "opp_def_rating",
        "opp_def_3pa_allowed",
        "opp_def_reb_rate",
        "opp_def_ast_rate",
        "num_rotation_players_out",
        "team_minutes_vacated",
        "team_usage_vacated",
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
    injuries = load_injuries(root)  # reserved for future vacated usage features
    if injuries.empty:
        typer.echo("[rates] injuries snapshot empty; vacated usage features will remain placeholders.")

    features = build_features(labels, stats, roster, odds, minutes_preds)
    _write_partitions(features, out_root)

    typer.echo(
        f"[rates] wrote {len(features):,} rows across "
        f"{features['game_date'].min().date()}–{features['game_date'].max().date()}"
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
