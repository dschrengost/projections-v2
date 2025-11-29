from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import typer

from projections.fpts_v1.datasets import _coerce_ts, _iter_days, _parse_minutes_iso, _season_from_day
from projections.fpts_v2.scoring import compute_dk_fpts
from projections.rates_v1.current import load_current_rates_bundle
from projections.rates_v1.features import get_rates_feature_sets
from projections.rates_v1.score import predict_rates

app = typer.Typer(add_completion=False)

DEFAULT_START = "2023-10-01"
DEFAULT_END = "2025-11-26"
DEFAULT_DATA_ROOT = Path(os.environ.get("PROJECTIONS_DATA_ROOT") or "/home/daniel/projections-data")


def _iter_boxscore_paths(data_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> Iterable[Path]:
    for day in _iter_days(start, end):
        season = _season_from_day(day)
        path = (
            data_root
            / "bronze"
            / "boxscores_raw"
            / f"season={season}"
            / f"date={day.date().isoformat()}"
            / "boxscores_raw.parquet"
        )
        if path.exists():
            yield path


def load_boxscores(data_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for path in _iter_boxscore_paths(data_root, start, end):
        bronze = pd.read_parquet(path)
        for row in bronze.itertuples():
            payload = json.loads(row.payload)
            tip_ts = _coerce_ts(payload.get("game_time_utc") or payload.get("game_time_local"))
            if tip_ts is None:
                continue
            game_date = tip_ts.tz_convert("America/New_York").tz_localize(None).normalize()
            season = _season_from_day(game_date)
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
                    oreb = float(stats.get("reboundsOffensive") or 0.0)
                    dreb = float(stats.get("reboundsDefensive") or 0.0)
                    reb_total = float(stats.get("reboundsTotal") or (oreb + dreb))
                    record = {
                        "season": season,
                        "game_id": game_id,
                        "team_id": team_id,
                        "opponent_id": opponent_id,
                        "player_id": int(player.get("person_id") or player.get("personId") or 0),
                        "game_date": game_date,
                        "tip_ts": tip_ts,
                        "home_flag": home_flag,
                        "minutes_actual": minutes,
                        "pts": float(stats.get("points") or 0.0),
                        "fgm": float(stats.get("fieldGoalsMade") or 0.0),
                        "fga": float(stats.get("fieldGoalsAttempted") or 0.0),
                        "fg3m": float(stats.get("threePointersMade") or 0.0),
                        "fg3a": float(stats.get("threePointersAttempted") or 0.0),
                        "ftm": float(stats.get("freeThrowsMade") or 0.0),
                        "fta": float(stats.get("freeThrowsAttempted") or 0.0),
                        "reb": reb_total,
                        "oreb": oreb,
                        "dreb": dreb,
                        "ast": float(stats.get("assists") or 0.0),
                        "stl": float(stats.get("steals") or 0.0),
                        "blk": float(stats.get("blocks") or 0.0),
                        "tov": float(stats.get("turnovers") or 0.0),
                        "pf": float(stats.get("foulsPersonal") or 0.0),
                        "plus_minus": float(
                            stats.get("plusMinusPoints")
                            or (stats.get("plus") or 0.0) - (stats.get("minus") or 0.0)
                        ),
                    }
                    records.append(record)
    if not records:
        raise FileNotFoundError(f"No boxscore stat lines found between {start.date()} and {end.date()}")
    stats_df = pd.DataFrame.from_records(records)
    stats_df.sort_values(["player_id", "tip_ts"], inplace=True)
    stats_df["game_date"] = pd.to_datetime(stats_df["game_date"]).dt.normalize()
    return stats_df


def load_minutes_predictions(data_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    root = data_root / "gold" / "minutes_for_rates"
    frames: list[pd.DataFrame] = []
    for season_dir in root.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if day < start or day > end:
                continue
            path = day_dir / "minutes_for_rates.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    # Ensure numeric ids
    for key in ("season", "game_id", "team_id", "player_id"):
        if key in df.columns:
            df[key] = pd.to_numeric(df[key], errors="coerce").astype("Int64")
    if {"minutes_pred_p90", "minutes_pred_p10"}.issubset(df.columns) and "minutes_pred_spread" not in df.columns:
        df["minutes_pred_spread"] = pd.to_numeric(df["minutes_pred_p90"], errors="coerce") - pd.to_numeric(
            df["minutes_pred_p10"], errors="coerce"
        )
    # Provide alias columns for downstream output convenience
    df["minutes_p10"] = df.get("minutes_pred_p10")
    df["minutes_p50"] = df.get("minutes_pred_p50")
    df["minutes_p90"] = df.get("minutes_pred_p90")
    df["play_prob"] = df.get("minutes_pred_play_prob")
    keep_cols = [
        "season",
        "game_id",
        "team_id",
        "player_id",
        "game_date",
        "minutes_pred_p10",
        "minutes_pred_p50",
        "minutes_pred_p90",
        "minutes_pred_spread",
        "minutes_pred_play_prob",
        "minutes_p10",
        "minutes_p50",
        "minutes_p90",
        "play_prob",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


def load_minutes_labels(data_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in _iter_days(start, end):
        season = _season_from_day(day)
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
        return pd.DataFrame()
    labels = pd.concat(frames, ignore_index=True)
    labels["game_date"] = pd.to_datetime(labels["game_date"]).dt.normalize()
    labels = labels[(labels["game_date"] >= start) & (labels["game_date"] <= end)].copy()
    labels["minutes_actual"] = pd.to_numeric(labels["minutes"], errors="coerce")
    labels = labels.rename(columns={"minutes": "minutes_label"})
    for key in ("season", "game_id", "team_id", "player_id"):
        if key in labels.columns:
            labels[key] = pd.to_numeric(labels[key], errors="coerce").astype("Int64")
    return labels[["season", "game_id", "team_id", "player_id", "game_date", "minutes_actual"]]


def load_rates_base(data_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    base = data_root / "gold" / "rates_training_base"
    partitions: list[Path] = []
    for season_dir in base.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if day < start or day > end:
                continue
            candidate = day_dir / "rates_training_base.parquet"
            if candidate.exists():
                partitions.append(candidate)
    if not partitions:
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in partitions]
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


def _apply_desert_filter(df: pd.DataFrame, desert_csv: Path) -> pd.DataFrame:
    desert_df = pd.read_csv(desert_csv)
    if "game_date" not in desert_df.columns:
        return df
    desert_df["game_date"] = pd.to_datetime(desert_df["game_date"]).dt.normalize()
    desert_col = desert_df.get("is_desert")
    partial_col = desert_df.get("is_partial_desert")
    mask = (
        (desert_col.fillna(False).astype(bool) if desert_col is not None else False)
        | (partial_col.fillna(False).astype(bool) if partial_col is not None else False)
    )
    desert_dates = set(desert_df.loc[mask, "game_date"].dt.date.tolist())
    if not desert_dates:
        return df
    before = len(df)
    drop_mask = df["game_date"].dt.date.isin(desert_dates)
    filtered = df[~drop_mask].copy()
    typer.echo(f"[fpts_base] dropped {before - len(filtered)} rows from feature-desert dates")
    return filtered


def _warn_missing_dates(label: str, start: pd.Timestamp, end: pd.Timestamp, present_dates: set[pd.Timestamp]) -> None:
    requested = {day.normalize() for day in _iter_days(start, end)}
    missing = sorted(date for date in requested if date not in present_dates)
    if missing:
        typer.echo(f"[fpts_base] warning: missing {label} for {len(missing)} dates (e.g., {missing[:3]})")


def _select_output_columns(df: pd.DataFrame, pred_cols: list[str], context_cols: list[str]) -> pd.DataFrame:
    base_cols = [
        "season",
        "game_id",
        "team_id",
        "player_id",
        "game_date",
        "dk_fpts_actual",
        "minutes_p10",
        "minutes_p50",
        "minutes_p90",
        "play_prob",
        "is_starter",
        "minutes_actual",
        "pts",
        "reb",
        "ast",
        "stl",
        "blk",
        "tov",
        "oreb",
        "dreb",
        "fg3m",
        "pf",
        "plus_minus",
    ]
    ordered: list[str] = []
    seen: set[str] = set()
    for col in base_cols + pred_cols + context_cols:
        if col in df.columns and col not in seen:
            ordered.append(col)
            seen.add(col)
    return df[ordered]


@app.command()
def main(
    data_root: Optional[Path] = typer.Option(None, help="Root containing gold/ and bronze/ tiers."),
    output_root: Optional[Path] = typer.Option(
        None, help="Output root (defaults to <data_root>/gold/fpts_training_base)."
    ),
    start_date: str = typer.Option(DEFAULT_START, help="Start date (YYYY-MM-DD)."),
    end_date: str = typer.Option(DEFAULT_END, help="End date (YYYY-MM-DD)."),
    minutes_desert_csv: Optional[Path] = typer.Option(
        None, help="Optional CSV of feature desert dates; matching game_dates will be dropped."
    ),
) -> None:
    root = data_root or DEFAULT_DATA_ROOT
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    out_root = output_root or (root / "gold" / "fpts_training_base")

    typer.echo(
        f"[fpts_base] data_root={root} output_root={out_root} "
        f"window=({start.date()} to {end.date()})"
    )

    minutes_preds = load_minutes_predictions(root, start, end)
    rates_base = load_rates_base(root, start, end)
    stats = load_boxscores(root, start, end)
    minutes_labels = load_minutes_labels(root, start, end)

    if rates_base.empty:
        typer.echo("[fpts_base] no rates_training_base partitions found; exiting.")
        raise typer.Exit(code=1)
    if minutes_preds.empty:
        typer.echo("[fpts_base] no minutes predictions found; nothing to build.")
        raise typer.Exit(code=1)

    _warn_missing_dates("minutes_for_rates", start, end, set(minutes_preds["game_date"].unique()))

    bundle = load_current_rates_bundle()
    if "minutes_pred_spread" not in rates_base.columns:
        if {"minutes_pred_p90", "minutes_pred_p10"}.issubset(rates_base.columns):
            rates_base["minutes_pred_spread"] = pd.to_numeric(
                rates_base["minutes_pred_p90"], errors="coerce"
            ) - pd.to_numeric(rates_base["minutes_pred_p10"], errors="coerce")
    feature_missing = [c for c in bundle.feature_cols if c not in rates_base.columns]
    if feature_missing:
        raise KeyError(f"rates_training_base missing required columns: {feature_missing}")

    merged = rates_base.merge(
        minutes_preds,
        on=["season", "game_id", "team_id", "player_id", "game_date"],
        how="left",
    )

    # Resolve minutes prediction columns to the expected base names (prefer minutes_preds over rates_base).
    minutes_pred_cols = [
        "minutes_pred_p10",
        "minutes_pred_p50",
        "minutes_pred_p90",
        "minutes_pred_spread",
        "minutes_pred_play_prob",
    ]
    for col in minutes_pred_cols:
        sources = [f"{col}_y", f"{col}_x", col]
        for src in sources:
            if src in merged.columns:
                merged[col] = merged[src]
                break
    # Clean up suffix columns to avoid confusion.
    drop_cols = [c for c in merged.columns if c.endswith("_x") or c.endswith("_y")]
    merged.drop(columns=drop_cols, inplace=True, errors="ignore")

    feature_frame = merged.copy()
    for col in bundle.feature_cols:
        if col in feature_frame.columns:
            feature_frame[col] = pd.to_numeric(feature_frame[col], errors="coerce")
    preds = predict_rates(feature_frame[bundle.feature_cols], bundle)
    pred_cols = preds.columns.tolist()
    preds = preds.add_prefix("pred_")
    merged = pd.concat([merged.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

    stats["dk_fpts_actual"] = compute_dk_fpts(stats)
    label_cols = [
        "season",
        "game_id",
        "team_id",
        "player_id",
        "game_date",
        "dk_fpts_actual",
        "minutes_actual",
        "pts",
        "reb",
        "ast",
        "stl",
        "blk",
        "tov",
        "oreb",
        "dreb",
        "fg3m",
        "pf",
        "plus_minus",
    ]
    merged = merged.merge(stats[label_cols], on=["season", "game_id", "team_id", "player_id", "game_date"], how="left")
    if not minutes_labels.empty:
        merged = merged.merge(
            minutes_labels, on=["season", "game_id", "team_id", "player_id", "game_date"], how="left", suffixes=("", "_label")
        )
        if "minutes_actual_label" in merged.columns:
            merged["minutes_actual"] = merged["minutes_actual"].fillna(merged["minutes_actual_label"])
        merged.drop(columns=[c for c in merged.columns if c.endswith("_label")], inplace=True)

    if "minutes_p50" not in merged.columns and "minutes_pred_p50" in merged.columns:
        merged["minutes_p50"] = merged["minutes_pred_p50"]
    if "minutes_p10" not in merged.columns and "minutes_pred_p10" in merged.columns:
        merged["minutes_p10"] = merged["minutes_pred_p10"]
    if "minutes_p90" not in merged.columns and "minutes_pred_p90" in merged.columns:
        merged["minutes_p90"] = merged["minutes_pred_p90"]
    if "play_prob" not in merged.columns and "minutes_pred_play_prob" in merged.columns:
        merged["play_prob"] = merged["minutes_pred_play_prob"]

    merged["game_date"] = pd.to_datetime(merged["game_date"]).dt.normalize()
    merged = merged[(merged["game_date"] >= start) & (merged["game_date"] <= end)].copy()

    required = ["dk_fpts_actual", "minutes_p50"] + [f"pred_{c}" for c in pred_cols]
    before_filter = len(merged)
    merged = merged.dropna(subset=required)
    dropped = before_filter - len(merged)
    if dropped:
        typer.echo(f"[fpts_base] dropped {dropped} rows missing label/minutes/predictions")

    if minutes_desert_csv:
        merged = _apply_desert_filter(merged, minutes_desert_csv)

    rates_feature_sets = get_rates_feature_sets()
    context_candidates = {c for c in rates_feature_sets.get("stage3_context", []) if not c.startswith("minutes_pred_")}
    context_cols = [c for c in merged.columns if c in context_candidates]
    if context_cols:
        typer.echo(f"[fpts_base] context_cols added: {context_cols}")
        ctx_nan = merged[context_cols].isna().mean().sort_values(ascending=False)
        if not ctx_nan.empty:
            head = ctx_nan.head(10)
            msg = ", ".join([f"{col}={val:.3f}" for col, val in head.items()])
            typer.echo(f"[fpts_base] context NaN fraction (top 10): {msg}")

    output_cols = _select_output_columns(merged, [f"pred_{c}" for c in pred_cols], context_cols)
    if output_cols.empty:
        typer.echo("[fpts_base] no rows to write after filtering.")
        raise typer.Exit(code=1)

    out_root.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for game_date, frame in output_cols.groupby("game_date"):
        season = frame["season"].iloc[0]
        target_dir = out_root / f"season={season}" / f"game_date={game_date.date().isoformat()}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / "fpts_training_base.parquet"
        frame.to_parquet(target_path, index=False)
        total_rows += len(frame)
        typer.echo(f"[fpts_base] wrote {len(frame)} rows -> {target_path}")

    nan_label = output_cols["dk_fpts_actual"].isna().sum()
    nan_minutes = output_cols["minutes_p50"].isna().sum()
    nan_preds = output_cols[[c for c in output_cols.columns if c.startswith("pred_")]].isna().any(axis=1).sum()
    typer.echo(
        f"[fpts_base] completed. rows={total_rows} "
        f"nan label={nan_label} minutes={nan_minutes} pred_rows={nan_preds}"
    )


if __name__ == "__main__":
    app()
