"""Debug the fpts_v2 training dataset for a given run.

Builds an evaluation frame that joins the training target with boxscore-derived
actuals (from the same training base) plus minutes/rates features, then prints
sanity stats and per-player FPPM summaries.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import typer

from projections.fpts_v2.scoring import compute_dk_fpts
from projections.paths import get_data_root

DEFAULT_RUN_ID = "fpts_v2_stage0_20251129_062655"
DEFAULT_OUT_PATH = (
    Path("/home/daniel/projections-data/artifacts/fpts_v2/debug/fpts_v2_stage0_eval_base.parquet")
)

app = typer.Typer(add_completion=False)


def _parse_date(value: str | None) -> Optional[pd.Timestamp]:
    if not value:
        return None
    try:
        return pd.to_datetime(value).normalize()
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Invalid date: {value}") from exc


def _iter_partitions(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    base = root / "gold" / "fpts_training_base"
    partitions: list[Path] = []
    for season_dir in base.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if day < start or day > end:
                continue
            candidate = day_dir / "fpts_training_base.parquet"
            if candidate.exists():
                partitions.append(candidate)
    return sorted(partitions)


def _load_training_base(
    root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    columns: Iterable[str],
) -> pd.DataFrame:
    paths = _iter_partitions(root, start, end)
    if not paths:
        raise FileNotFoundError("No fpts_training_base partitions matched the requested window.")
    frames: list[pd.DataFrame] = []
    for p in paths:
        schema_cols = set(pq.read_schema(p).names)
        cols_to_read = [c for c in columns if c in schema_cols]
        frames.append(pd.read_parquet(p, columns=cols_to_read))
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


def _resolve_meta(run_id: str, data_root: Path) -> dict:
    meta_path = data_root / "artifacts" / "fpts_v2" / "runs" / run_id / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json missing at {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _baseline_from_rates(df: pd.DataFrame, minutes_col: str = "minutes_p50") -> pd.Series:
    rate_cols = [c for c in df.columns if c.startswith("pred_")]
    if not rate_cols or minutes_col not in df.columns:
        return pd.Series([np.nan] * len(df))
    minutes = pd.to_numeric(df[minutes_col], errors="coerce").fillna(0.0)

    def _get(name: str) -> np.ndarray:
        return pd.to_numeric(df.get(f"pred_{name}_per_min"), errors="coerce").fillna(0.0).to_numpy()

    fga2 = _get("fga2") * minutes.to_numpy()
    fga3 = _get("fga3") * minutes.to_numpy()
    fta = _get("fta") * minutes.to_numpy()
    ast = _get("ast") * minutes.to_numpy()
    tov = _get("tov") * minutes.to_numpy()
    oreb = _get("oreb") * minutes.to_numpy()
    dreb = _get("dreb") * minutes.to_numpy()
    stl = _get("stl") * minutes.to_numpy()
    blk = _get("blk") * minutes.to_numpy()

    reb = oreb + dreb
    ftm = 0.75 * fta
    pts = 2.0 * fga2 + 3.0 * fga3 + ftm
    fgm = fga2 + fga3
    fga = fga2 + fga3
    fg3m = fga3
    fg3a = fga3

    payload = pd.DataFrame(
        {
            "pts": pts,
            "reb": reb,
            "ast": ast,
            "stl": stl,
            "blk": blk,
            "tov": tov,
            "fgm": fgm,
            "fga": fga,
            "fg3m": fg3m,
            "fg3a": fg3a,
            "ftm": ftm,
            "fta": fta,
            "oreb": oreb,
            "dreb": dreb,
            "pf": np.zeros_like(pts),
            "plus_minus": np.zeros_like(pts),
        }
    )
    return compute_dk_fpts(payload)


def _print_describe(label: str, series: pd.Series) -> None:
    desc = series.describe()
    typer.echo(f"\n{label} describe():")
    typer.echo(desc.to_string())


def _top_players(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if "player_name" in df.columns:
        key = "player_name"
    else:
        key = "player_id"
    grouped = df.groupby(key)
    stats = grouped.agg(
        rows=("actual_fppm", "count"),
        mean_actual_fppm=("actual_fppm", "mean"),
        mean_target_fppm=("target_fppm", "mean"),
        mean_baseline_fppm=("baseline_fppm", "mean"),
    )
    stats = stats.sort_values("mean_actual_fppm", ascending=False).head(n)
    return stats.reset_index()


@app.command()
def main(
    run_id: str = typer.Option(DEFAULT_RUN_ID, "--run-id"),
    start_date: Optional[str] = typer.Option(None, "--start-date"),
    end_date: Optional[str] = typer.Option(None, "--end-date"),
    data_root: Path = typer.Option(None, "--data-root", help="Defaults to PROJECTIONS_DATA_ROOT."),
    write_eval: bool = typer.Option(True, "--write-eval/--no-write-eval"),
    out_path: Path = typer.Option(DEFAULT_OUT_PATH, "--out-path"),
) -> None:
    root = data_root or get_data_root()
    meta = _resolve_meta(run_id, root)
    window = meta.get("date_window") or {}
    start = _parse_date(start_date) or _parse_date(window.get("start"))
    end = _parse_date(end_date) or _parse_date(window.get("end"))
    if start is None or end is None:
        raise RuntimeError("Missing start/end dates; provide --start-date/--end-date or ensure meta.date_window is set.")

    cols_needed = [
        "season",
        "game_id",
        "team_id",
        "player_id",
        "game_date",
        "player_name",
        "dk_fpts_actual",
        "minutes_actual",
        "minutes",
        "minutes_p50",
        "minutes_p50_cond",
        "play_prob",
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
    # Include any pred_* columns if present
    sample_path = _iter_partitions(root, start, end)
    extra_pred_cols: list[str] = []
    if sample_path:
        sample_cols = pq.read_schema(sample_path[0]).names
        extra_pred_cols = [c for c in sample_cols if c.startswith("pred_")]
    cols_needed += extra_pred_cols

    df = _load_training_base(root, start, end, columns=cols_needed)
    df = df.dropna(subset=["dk_fpts_actual"])
    df.rename(columns={"dk_fpts_actual": "target_fpts"}, inplace=True)
    df["minutes_actual"] = pd.to_numeric(df.get("minutes_actual"), errors="coerce")
    df["minutes_p50"] = pd.to_numeric(df.get("minutes_p50"), errors="coerce")
    df["minutes_p50_cond"] = pd.to_numeric(df.get("minutes_p50_cond"), errors="coerce")
    df["target_fppm"] = df["target_fpts"] / df["minutes_actual"].replace(0, np.nan)

    # Actuals from in-frame stats
    for col in ("pts", "reb", "ast", "stl", "blk", "tov", "oreb", "dreb", "fg3m", "pf", "plus_minus"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0
    df["fgm"] = df["fg3m"]  # minimal placeholder; not used in scoring formula
    df["fga"] = df["fg3m"]
    df["fg3a"] = df["fg3m"]
    df["ftm"] = 0.0
    df["fta"] = 0.0
    actual_payload = df[
        [
            "pts",
            "fgm",
            "fga",
            "fg3m",
            "fg3a",
            "ftm",
            "fta",
            "reb",
            "oreb",
            "dreb",
            "ast",
            "stl",
            "blk",
            "tov",
            "pf",
            "plus_minus",
        ]
    ]
    df["actual_fpts"] = compute_dk_fpts(actual_payload)
    df["actual_fppm"] = df["actual_fpts"] / df["minutes_actual"].replace(0, np.nan)

    # Baseline from rates preds if available
    baseline = _baseline_from_rates(df, minutes_col="minutes_p50")
    df["baseline_fpts"] = baseline
    df["baseline_fppm"] = baseline / df["minutes_p50"].replace(0, np.nan)

    # Stats
    _print_describe("target_fpts", df["target_fpts"])
    _print_describe("target_fppm", df["target_fppm"].dropna())
    _print_describe("actual_fpts", df["actual_fpts"])
    _print_describe("actual_fppm", df["actual_fppm"].dropna())
    if df["baseline_fpts"].notna().any():
        _print_describe("baseline_fpts", df["baseline_fpts"].dropna())
        _print_describe("baseline_fppm", df["baseline_fppm"].dropna())

    # Player-level top list
    top_players = _top_players(df, n=10)
    typer.echo("\nTop players by mean actual FPPM:")
    typer.echo(top_players.to_string(index=False))

    # Jokic / studs
    if "player_name" in df.columns:
        studs = ["JOK", "LUKA", "GIANN", "GILGEOUS", "EMBIID", "CURRY"]
        mask = df["player_name"].str.upper().fillna("")
        stud_mask = mask.apply(lambda x: any(s in x for s in studs))
        if stud_mask.any():
            stud_stats = df.loc[stud_mask].groupby("player_name").agg(
                rows=("actual_fppm", "count"),
                mean_actual_fppm=("actual_fppm", "mean"),
                mean_target_fppm=("target_fppm", "mean"),
                mean_baseline_fppm=("baseline_fppm", "mean"),
            )
            typer.echo("\nStuds (name contains Jok/Luka/Giannis/SGA/Embiid/Curry):")
            typer.echo(stud_stats.sort_values("mean_actual_fppm", ascending=False).to_string())

    if write_eval:
        out_path = out_path.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        typer.echo(f"\nWrote eval frame to {out_path}")


if __name__ == "__main__":
    app()
