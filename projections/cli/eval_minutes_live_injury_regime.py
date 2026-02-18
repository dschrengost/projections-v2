"""Evaluate close-to-tip minutes realism with injury-regime slices from live snapshots.

This is the "close to tip" counterpart to `eval_minutes_injury_regime.py`. Instead of
evaluating the final per-day `minutes.parquet`, it evaluates the *last snapshot before
tip-off* for each game using the prediction logs in
`$PROJECTIONS_DATA_ROOT/gold/prediction_logs_minutes/`.

Why:
- DFS pain is dominated by the final pre-tip injury regime.
- We want to diagnose the two failure modes:
  1) zeroing too much bench -> core/rotation gets overprojected
  2) smearing too much to the tail -> bench promotions get underprojected

Slices (team-games):
- `injury_regime`: >= min_starters_out previous-game starters OUT OR >= min_team_out total OUT
- `non_injury`: starter_out_count == 0 AND team_out_count == 0
- `all_games`: unfiltered

Supports comparing "current" (from logs) vs a candidate minutes root (e.g. RMH outputs)
that is partitioned by `date/run=<run_id>/minutes.parquet`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import typer

from projections import paths
from projections.labels import derive_starter_flag_labels
from projections.minutes_v1.eval_live import MinutesLiveEvalDatasetBuilder

# Reuse the established injury-regime metrics + tables to keep experiments comparable.
from projections.cli import eval_minutes_injury_regime as injury_eval

app = typer.Typer(help=__doc__)


def _iter_days(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _normalize_day(value: str | date | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return ts.tz_localize(None).normalize()


@dataclass(frozen=True)
class CandidateCoverage:
    required_pairs: int
    found_pairs: int
    missing_pairs: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "required_date_runs": int(self.required_pairs),
            "found_date_runs": int(self.found_pairs),
            "missing_date_runs": list(self.missing_pairs),
        }


def _clean_json_value(value: object) -> object:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta,)):
        return str(value)
    # pandas Int64 / numpy scalars
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _df_records_for_json(df: pd.DataFrame, *, cols: list[str], n: int) -> list[dict[str, Any]]:
    if df.empty:
        return []
    view = df.loc[:, [c for c in cols if c in df.columns]].head(n).copy()
    records: list[dict[str, Any]] = []
    for row in view.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            clean[key] = _clean_json_value(value)
        records.append(clean)
    return records


def _catastrophic_minutes_metrics(
    df: pd.DataFrame,
    *,
    pred_col: str,
    actual_col: str,
    ghost_pred_min: float,
    ghost_actual_max: float,
    missed_actual_min: float,
    missed_pred_max: float,
    top_n: int,
) -> dict[str, Any]:
    if df.empty:
        return {
            "ghost_dnp": {"n_pred_ge": 0, "n_ghost": 0, "rate": None, "top": []},
            "missed_run": {"n_actual_ge": 0, "n_missed": 0, "rate": None, "top": []},
        }

    actual = pd.to_numeric(df.get(actual_col), errors="coerce").fillna(0.0).astype(float)
    pred = pd.to_numeric(df.get(pred_col), errors="coerce").fillna(0.0).astype(float)

    ghost_pred_min = float(ghost_pred_min)
    ghost_actual_max = float(ghost_actual_max)
    missed_actual_min = float(missed_actual_min)
    missed_pred_max = float(missed_pred_max)

    pred_ge = pred >= ghost_pred_min
    ghost_mask = pred_ge & (actual <= ghost_actual_max)
    n_pred_ge = int(pred_ge.sum())
    n_ghost = int(ghost_mask.sum())
    ghost_rate = float(n_ghost / n_pred_ge) if n_pred_ge > 0 else None

    actual_ge = actual >= missed_actual_min
    missed_mask = actual_ge & (pred <= missed_pred_max)
    n_actual_ge = int(actual_ge.sum())
    n_missed = int(missed_mask.sum())
    missed_rate = float(n_missed / n_actual_ge) if n_actual_ge > 0 else None

    cols = [
        "game_date",
        "run_id",
        "game_id",
        "team_id",
        "team_tricode",
        "player_id",
        "player_name",
        "status",
        "starter_flag_label",
        actual_col,
        pred_col,
    ]

    ghost_top = (
        df.loc[ghost_mask]
        .assign(**{pred_col: pred.loc[ghost_mask], actual_col: actual.loc[ghost_mask]})
        .sort_values(pred_col, ascending=False, kind="mergesort")
    )
    missed_top = (
        df.loc[missed_mask]
        .assign(**{pred_col: pred.loc[missed_mask], actual_col: actual.loc[missed_mask]})
        .sort_values(actual_col, ascending=False, kind="mergesort")
    )

    return {
        "config": {
            "actual_col": actual_col,
            "pred_col": pred_col,
            "ghost_pred_min": ghost_pred_min,
            "ghost_actual_max": ghost_actual_max,
            "missed_actual_min": missed_actual_min,
            "missed_pred_max": missed_pred_max,
            "top_n": int(top_n),
        },
        "ghost_dnp": {
            "n_pred_ge": n_pred_ge,
            "n_ghost": n_ghost,
            "rate": ghost_rate,
            "top": _df_records_for_json(ghost_top, cols=cols, n=top_n),
        },
        "missed_run": {
            "n_actual_ge": n_actual_ge,
            "n_missed": n_missed,
            "rate": missed_rate,
            "top": _df_records_for_json(missed_top, cols=cols, n=top_n),
        },
    }


def _resolve_minutes_path(root: Path, *, day: date, run_id: str) -> Path | None:
    iso = day.isoformat()
    candidates = [
        root / iso / f"run={run_id}" / "minutes.parquet",
        root / f"game_date={iso}" / f"run={run_id}" / "minutes.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_candidate_minutes(
    *,
    root: Path,
    required: pd.DataFrame,
    minutes_col: str,
    require_full_coverage: bool,
) -> tuple[pd.DataFrame, CandidateCoverage]:
    """Load candidate minutes.parquet for each (game_date, run_id) pair."""

    empty_cols = ["run_id", "game_id", "team_id", "player_id", "_pred_minutes"]
    if required.empty:
        return pd.DataFrame(columns=empty_cols), CandidateCoverage(0, 0, [])

    frames: list[pd.DataFrame] = []
    missing: list[dict[str, Any]] = []
    for rec in required.to_dict(orient="records"):
        day = rec["game_date"]
        run_id = rec["run_id"]
        path = _resolve_minutes_path(root, day=day, run_id=run_id)
        if path is None:
            missing.append({"game_date": day.isoformat(), "run_id": str(run_id), "reason": "missing_minutes_parquet"})
            continue
        df = pd.read_parquet(path)
        if df.empty:
            missing.append({"game_date": day.isoformat(), "run_id": str(run_id), "reason": "empty_minutes_parquet"})
            continue
        df = df.copy()
        df["game_date"] = pd.to_datetime(df.get("game_date", day)).dt.normalize()
        df["run_id"] = str(run_id)
        frames.append(df)

    coverage = CandidateCoverage(
        required_pairs=int(len(required)),
        found_pairs=int(len(required) - len(missing)),
        missing_pairs=missing,
    )
    if require_full_coverage and missing:
        missing_preview = ", ".join([f"{m['game_date']}:{m['run_id']}" for m in missing[:5]])
        raise FileNotFoundError(
            f"Candidate minutes coverage incomplete: missing {len(missing)}/{len(required)} date-runs "
            f"(sample: {missing_preview}) under {root}"
        )
    if not frames:
        return pd.DataFrame(columns=empty_cols), coverage

    combined = pd.concat(frames, ignore_index=True)
    for col in ("game_id", "player_id", "team_id"):
        combined[col] = pd.to_numeric(combined.get(col), errors="coerce").astype("Int64")
    combined = combined.dropna(subset=["game_id", "player_id", "team_id"]).copy()
    combined["game_id"] = combined["game_id"].astype(int)
    combined["player_id"] = combined["player_id"].astype(int)
    combined["team_id"] = combined["team_id"].astype(int)

    # Prefer explicit requested minutes column.
    if minutes_col not in combined.columns:
        # Common fallbacks across scorers.
        fallbacks = ["effective_minutes", "minutes_p50", "minutes_mean", "minutes_p50_cond"]
        found = next((c for c in fallbacks if c in combined.columns), None)
        if found is None:
            raise ValueError(
                f"Candidate minutes missing requested col '{minutes_col}' and no known fallback present. "
                f"Available cols sample: {sorted(combined.columns)[:25]}"
            )
        minutes_col = found

    combined["_pred_minutes"] = pd.to_numeric(combined[minutes_col], errors="coerce").fillna(0.0).astype(float)
    combined = combined.sort_values(["game_date", "run_id", "game_id", "team_id", "player_id"], kind="mergesort")
    combined = combined.drop_duplicates(subset=["run_id", "game_id", "team_id", "player_id"], keep="last")
    return combined[["run_id", "game_id", "team_id", "player_id", "_pred_minutes"]], coverage


def _load_schedule_slice(
    *,
    schedule_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if not schedule_root.exists():
        raise FileNotFoundError(f"Schedule root missing: {schedule_root}")

    frames: list[pd.DataFrame] = []
    for season_dir in sorted(schedule_root.glob("season=*")):
        for month_dir in sorted(season_dir.glob("month=*")):
            path = month_dir / "schedule.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            if df.empty:
                continue
            if not {"game_id", "game_date"}.issubset(df.columns):
                continue

            tip_col = "tip_ts" if "tip_ts" in df.columns else None
            if tip_col is None:
                if "game_ts" in df.columns:
                    tip_col = "game_ts"
                elif "game_time_utc" in df.columns:
                    tip_col = "game_time_utc"
            if tip_col is None or tip_col not in df.columns:
                continue

            keep_cols = [c for c in ("game_id", "game_date", tip_col, "home_team_id", "away_team_id") if c in df.columns]
            work = df.loc[:, keep_cols].copy()
            work.rename(columns={tip_col: "tip_ts"}, inplace=True)

            work["game_id"] = pd.to_numeric(work["game_id"], errors="coerce")
            work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce")
            work["tip_ts"] = pd.to_datetime(work["tip_ts"], utc=True, errors="coerce")
            work = work.dropna(subset=["game_id", "game_date", "tip_ts"])
            if work.empty:
                continue
            work["game_id"] = work["game_id"].astype(int)
            work["game_date"] = work["game_date"].dt.tz_localize(None).dt.normalize()
            frames.append(work)

    if not frames:
        return pd.DataFrame()

    sched = pd.concat(frames, ignore_index=True)
    sched = sched[(sched["game_date"] >= start) & (sched["game_date"] <= end)].copy()
    if sched.empty:
        return pd.DataFrame()

    sched.sort_values(["game_date", "game_id"], inplace=True)
    sched = sched.drop_duplicates(subset=["game_id"], keep="last").reset_index(drop=True)
    return sched


def _season_start(day: date) -> int:
    return day.year if day.month >= 8 else day.year - 1


def _load_gold_labels_minutes_v1(
    *,
    labels_root: Path,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Load minutes labels from gold day partitions and derive starter_flag_label."""

    frames: list[pd.DataFrame] = []
    for day in _iter_days(start, end):
        season = _season_start(day)
        path = labels_root / f"season={season}" / f"game_date={day.isoformat()}" / "labels.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        if not {"game_id", "player_id", "team_id", "game_date"}.issubset(df.columns):
            continue

        work = df.copy()
        work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce").dt.normalize()
        if "actual_minutes" in work.columns:
            work["minutes"] = pd.to_numeric(work["actual_minutes"], errors="coerce")
        elif "minutes" in work.columns:
            work["minutes"] = pd.to_numeric(work["minutes"], errors="coerce")
        else:
            continue
        work["minutes"] = work["minutes"].fillna(0.0).astype(float)
        for col in ("game_id", "player_id", "team_id"):
            work[col] = pd.to_numeric(work[col], errors="coerce").astype("Int64")
        work = work.dropna(subset=["game_id", "player_id", "team_id"]).copy()
        work["game_id"] = work["game_id"].astype(int)
        work["player_id"] = work["player_id"].astype(int)
        work["team_id"] = work["team_id"].astype(int)

        frames.append(work[["game_date", "game_id", "team_id", "player_id", "minutes"]])

    if not frames:
        raise FileNotFoundError(f"No gold labels found under {labels_root} for {start} → {end}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["game_date", "game_id", "team_id", "player_id"], kind="mergesort")
    combined = combined.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last").copy()
    combined = derive_starter_flag_labels(
        combined,
        minutes_col="minutes",
        game_col="game_id",
        team_col="team_id",
        player_col="player_id",
        output_col="starter_flag_label",
    )
    combined["starter_flag_label"] = (
        pd.to_numeric(combined["starter_flag_label"], errors="coerce").fillna(0).astype(int)
    )
    return combined.reset_index(drop=True)


def _build_team_game_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    if schedule.empty:
        return pd.DataFrame(columns=["game_id", "team_id", "tip_ts", "prev_game_id"])

    required = {"game_id", "tip_ts", "home_team_id", "away_team_id"}
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError(f"Schedule missing required columns for team-game ordering: {', '.join(sorted(missing))}")

    home = schedule[["game_id", "tip_ts", "home_team_id"]].rename(columns={"home_team_id": "team_id"})
    away = schedule[["game_id", "tip_ts", "away_team_id"]].rename(columns={"away_team_id": "team_id"})
    team_games = pd.concat([home, away], ignore_index=True)
    team_games["team_id"] = pd.to_numeric(team_games["team_id"], errors="coerce").astype("Int64")
    team_games = team_games.dropna(subset=["team_id"]).copy()
    team_games["team_id"] = team_games["team_id"].astype(int)
    team_games["tip_ts"] = pd.to_datetime(team_games["tip_ts"], utc=True, errors="coerce")
    team_games = team_games.dropna(subset=["tip_ts"]).copy()
    team_games.sort_values(["team_id", "tip_ts", "game_id"], inplace=True, kind="mergesort")
    team_games["prev_game_id"] = team_games.groupby("team_id", sort=False)["game_id"].shift(1)
    return team_games.reset_index(drop=True)


def _compute_out_flag(snapshot: pd.DataFrame) -> pd.Series:
    if snapshot.empty:
        return pd.Series(dtype=bool)
    out = pd.Series(False, index=snapshot.index, dtype=bool)
    if "is_out" in snapshot.columns:
        out |= pd.to_numeric(snapshot["is_out"], errors="coerce").fillna(0).astype(int) == 1
    if "status" in snapshot.columns:
        status = snapshot["status"].astype(str).str.upper()
        out |= status.str.contains("OUT", na=False)
    if "lineup_role" in snapshot.columns:
        role = snapshot["lineup_role"].astype(str).str.lower()
        out |= role.eq("out")
    return out


def _active_only_slice(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = _compute_out_flag(df).astype(bool)
    return df.loc[~out].copy()


def _coerce_utc_ts_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NaT, index=df.index)
    return pd.to_datetime(df[column], utc=True, errors="coerce")


def _build_injury_regime_table_from_snapshots(
    *,
    team_games: pd.DataFrame,
    labels: pd.DataFrame,
    snapshot: pd.DataFrame,
    min_starters_out: int,
    min_team_out: int,
) -> pd.DataFrame:
    if team_games.empty or labels.empty or snapshot.empty:
        return pd.DataFrame(columns=["game_id", "team_id", "starter_out_count", "team_out_count", "injury_regime"])

    required_labels = {"game_id", "team_id", "player_id", "starter_flag_label"}
    missing_labels = required_labels - set(labels.columns)
    if missing_labels:
        raise ValueError(f"Labels missing required columns: {', '.join(sorted(missing_labels))}")

    out_flags = snapshot[["game_id", "team_id", "player_id"]].copy()
    out_flags["_out_flag"] = _compute_out_flag(snapshot).astype(bool)

    team_out = (
        out_flags.groupby(["game_id", "team_id"], sort=False)["_out_flag"].sum().reset_index(name="team_out_count")
    )

    prev_starters = labels.loc[
        pd.to_numeric(labels["starter_flag_label"], errors="coerce").fillna(0).astype(int) == 1,
        ["game_id", "team_id", "player_id"],
    ].rename(columns={"game_id": "prev_game_id"})

    starter_out = (
        team_games[["game_id", "team_id", "prev_game_id"]]
        .merge(prev_starters, on=["team_id", "prev_game_id"], how="left")
        .merge(out_flags, on=["game_id", "team_id", "player_id"], how="left")
    )
    starter_out["_out_flag"] = starter_out["_out_flag"].astype("boolean").fillna(False).astype(bool)
    starter_out_count = (
        starter_out.groupby(["game_id", "team_id"], sort=False)["_out_flag"].sum().reset_index(name="starter_out_count")
    )

    merged = team_out.merge(starter_out_count, on=["game_id", "team_id"], how="left")
    merged["starter_out_count"] = merged["starter_out_count"].fillna(0).astype(int)
    merged["team_out_count"] = merged["team_out_count"].fillna(0).astype(int)
    merged["injury_regime"] = (merged["starter_out_count"] >= int(min_starters_out)) | (
        merged["team_out_count"] >= int(min_team_out)
    )
    return merged[["game_id", "team_id", "starter_out_count", "team_out_count", "injury_regime"]]


@app.command()
def main(
    *,
    start_date: str = typer.Option(..., "--start-date", help="Start game_date (YYYY-MM-DD, inclusive)."),
    end_date: str = typer.Option(..., "--end-date", help="End game_date (YYYY-MM-DD, inclusive)."),
    data_root: Path = typer.Option(paths.get_data_root(), "--data-root", help="Data root (PROJECTIONS_DATA_ROOT)."),
    logs_root: Path | None = typer.Option(None, "--logs-root", help="Override prediction logs root."),
    labels_root: Path | None = typer.Option(None, "--labels-root", help="Override labels root."),
    schedule_root: Path | None = typer.Option(None, "--schedule-root", help="Override schedule root."),
    snapshot_mode: str = typer.Option(
        "last_before_tip",
        "--snapshot-mode",
        help="Snapshot selection mode (currently only last_before_tip).",
    ),
    pred_col: str = typer.Option(
        "minutes_p50",
        "--pred-col",
        help="Prediction column to evaluate from the snapshot logs (e.g. minutes_p50, minutes_p50_raw).",
    ),
    min_starters_out: int = typer.Option(
        1,
        "--min-starters-out",
        help="Injury regime if >= this many previous-game starters are OUT (as-of snapshot).",
    ),
    min_team_out: int = typer.Option(
        2,
        "--min-team-out",
        help="Injury regime if >= this many total OUT players on team (as-of snapshot).",
    ),
    baseline_top_k: int = typer.Option(8, "--baseline-top-k", help="Baseline compress-to-top-K heuristic."),
    lookback_days: int = typer.Option(
        30, "--lookback-days", help="Lookback window for previous-game starters (days before start_date)."
    ),
    cat_ghost_pred_min: float = typer.Option(
        15.0, "--cat-ghost-pred-min", help="Catastrophic ghost: pred>=this but actual<=cat-ghost-actual-max."
    ),
    cat_ghost_actual_max: float = typer.Option(
        0.0, "--cat-ghost-actual-max", help="Catastrophic ghost: treat actual minutes <= this as DNP."
    ),
    cat_missed_actual_min: float = typer.Option(
        15.0, "--cat-missed-actual-min", help="Catastrophic missed: actual>=this but pred<=cat-missed-pred-max."
    ),
    cat_missed_pred_max: float = typer.Option(
        5.0, "--cat-missed-pred-max", help="Catastrophic missed: pred minutes <= this."
    ),
    cat_top_n: int = typer.Option(25, "--cat-top-n", help="Number of top catastrophic examples to include."),
    cat_active_only: bool = typer.Option(
        True,
        "--cat-active-only/--cat-all-status",
        help="Also report catastrophic metrics on active-only rows (excludes OUT-like rows).",
    ),
    max_minutes_to_tip: float | None = typer.Option(
        None,
        "--max-minutes-to-tip",
        help="Optional close-to-tip filter. Keep rows with 0 <= minutes_to_tip <= threshold.",
    ),
    candidate_root: Path | None = typer.Option(
        None,
        "--candidate-root",
        help="Optional candidate minutes root (expects <date>/run=<run_id>/minutes.parquet).",
    ),
    candidate_minutes_col: str = typer.Option(
        "effective_minutes",
        "--candidate-minutes-col",
        help="Candidate minutes column to use (fallbacks applied if missing).",
    ),
    require_candidate_coverage: bool = typer.Option(
        True,
        "--require-candidate-coverage/--allow-missing-candidate",
        help="Fail if any required candidate date-run minutes file is missing.",
    ),
    out: Path | None = typer.Option(None, "--out", help="Optional JSON output path."),
) -> None:
    start = _normalize_day(start_date).date()
    end = _normalize_day(end_date).date()
    if end < start:
        raise typer.BadParameter("--end-date must be on or after --start-date")

    data_root = data_root.expanduser().resolve()
    logs_root = logs_root.expanduser().resolve() if logs_root else None
    labels_root = labels_root.expanduser().resolve() if labels_root else None
    schedule_root = schedule_root.expanduser().resolve() if schedule_root else None

    start_lb = start - timedelta(days=int(lookback_days))

    typer.echo(f"[load] snapshots {start} → {end} (lookback={start_lb} → {end})")
    builder = MinutesLiveEvalDatasetBuilder(
        data_root=data_root,
        logs_root=logs_root,
        labels_root=labels_root,
        schedule_root=schedule_root,
        snapshot_mode=snapshot_mode,
    )
    snapshot_lb = builder.build(start_lb, end)
    if snapshot_lb.empty:
        raise FileNotFoundError("No snapshot rows found; check prediction logs coverage and date window.")

    snapshot_eval = snapshot_lb[
        (pd.to_datetime(snapshot_lb["game_date"]).dt.date >= start)
        & (pd.to_datetime(snapshot_lb["game_date"]).dt.date <= end)
    ].copy()
    if snapshot_eval.empty:
        raise FileNotFoundError("No snapshot rows in requested evaluation window.")

    if pred_col not in snapshot_eval.columns:
        raise ValueError(f"Snapshot missing requested pred_col '{pred_col}'.")

    # Labels (for bench-core / role buckets / prev-game starters).
    labels_root_effective = labels_root or paths.data_path("gold", "labels_minutes_v1")
    typer.echo(f"[load] gold labels {start_lb} → {end}")
    labels = _load_gold_labels_minutes_v1(labels_root=labels_root_effective, start=start_lb, end=end)
    labels_eval = labels[
        (labels["game_date"] >= pd.to_datetime(start)) & (labels["game_date"] <= pd.to_datetime(end))
    ].copy()

    # Team-game schedule for prev_game mapping (use full schedule to avoid gaps where snapshots are missing).
    schedule_root_effective = schedule_root or paths.data_path("silver", "schedule")
    typer.echo(f"[load] schedule {start_lb} → {end}")
    sched = _load_schedule_slice(
        schedule_root=schedule_root_effective,
        start=pd.Timestamp(start_lb),
        end=pd.Timestamp(end),
    )

    # Annotate selected snapshot age relative to tip. Optionally keep only close-to-tip rows.
    tip_lookup = sched.loc[:, ["game_id", "tip_ts"]].rename(columns={"tip_ts": "tip_ts_schedule"})
    tip_lookup = tip_lookup.drop_duplicates(subset=["game_id"], keep="last").copy()
    snapshot_eval = snapshot_eval.merge(tip_lookup, on="game_id", how="left")
    snapshot_eval["run_as_of_ts"] = _coerce_utc_ts_series(snapshot_eval, "run_as_of_ts")
    tip_ts = _coerce_utc_ts_series(snapshot_eval, "tip_ts").fillna(
        _coerce_utc_ts_series(snapshot_eval, "tip_ts_schedule")
    )
    snapshot_eval["tip_ts"] = tip_ts
    snapshot_eval["minutes_to_tip"] = (
        (snapshot_eval["tip_ts"] - snapshot_eval["run_as_of_ts"]).dt.total_seconds() / 60.0
    )
    tip_rows_before = int(len(snapshot_eval))
    if max_minutes_to_tip is not None:
        limit = float(max_minutes_to_tip)
        close_mask = snapshot_eval["minutes_to_tip"].notna() & snapshot_eval["minutes_to_tip"].between(0.0, limit)
        snapshot_eval = snapshot_eval.loc[close_mask].copy()
        if snapshot_eval.empty:
            raise ValueError(
                f"No snapshot rows within --max-minutes-to-tip={limit:g}. "
                "Widen the threshold or date window."
            )
    tip_rows_after = int(len(snapshot_eval))
    tip_minutes = pd.to_numeric(snapshot_eval["minutes_to_tip"], errors="coerce")
    tip_summary = {
        "rows_before_filter": tip_rows_before,
        "rows_after_filter": tip_rows_after,
        "max_minutes_to_tip": None if max_minutes_to_tip is None else float(max_minutes_to_tip),
        "minutes_to_tip_p50": None if tip_minutes.dropna().empty else float(tip_minutes.median()),
        "minutes_to_tip_p90": None if tip_minutes.dropna().empty else float(tip_minutes.quantile(0.9)),
    }
    team_games = _build_team_game_schedule(sched)

    # Injury regime table defined on snapshot OUT flags.
    typer.echo("[slice] injury_regime")
    injury_table = _build_injury_regime_table_from_snapshots(
        team_games=team_games,
        labels=labels,
        snapshot=snapshot_lb,
        min_starters_out=min_starters_out,
        min_team_out=min_team_out,
    )

    # Current model = snapshot logs.
    spine_cols = ["run_id", "game_id", "team_id", "player_id"]
    for extra in ("player_name", "team_name", "team_tricode", "status"):
        if extra in snapshot_eval.columns:
            spine_cols.append(extra)
    spine = snapshot_eval.loc[:, spine_cols].copy()
    spine["run_id"] = spine["run_id"].astype(str)
    eval_base = spine.merge(
        labels_eval,
        on=["game_id", "team_id", "player_id"],
        how="inner",
    )

    current_preds = snapshot_eval[["run_id", "game_id", "team_id", "player_id", pred_col]].copy()
    current_preds["run_id"] = current_preds["run_id"].astype(str)
    current_preds["_pred_minutes"] = pd.to_numeric(
        current_preds[pred_col], errors="coerce"
    ).fillna(0.0).astype(float)
    current_preds = current_preds.drop(columns=[pred_col])

    eval_current = eval_base.merge(
        current_preds,
        on=["run_id", "game_id", "team_id", "player_id"],
        how="left",
    )
    eval_current["_pred_minutes"] = eval_current["_pred_minutes"].fillna(0.0)
    eval_current = eval_current.merge(injury_table, on=["game_id", "team_id"], how="inner")
    eval_current["injury_regime"] = eval_current["injury_regime"].astype("boolean").fillna(False).astype(bool)

    slices = injury_eval._build_eval_slices(eval_current)
    if slices["injury_regime"].empty:
        raise ValueError(
            "No injury-regime rows found for the requested window; widen range or relax thresholds."
        )

    results: dict[str, Any] = {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "snapshot": {
            "snapshot_mode": builder.snapshot_mode,
            "pred_col": pred_col,
            "snapshot_summary": builder.last_snapshot_summary,
            "tip_window": tip_summary,
        },
        "injury_regime": {
            "min_starters_out": int(min_starters_out),
            "min_team_out": int(min_team_out),
            "lookback_days": int(lookback_days),
        },
        "catastrophic": {
            "ghost_pred_min": float(cat_ghost_pred_min),
            "ghost_actual_max": float(cat_ghost_actual_max),
            "missed_actual_min": float(cat_missed_actual_min),
            "missed_pred_max": float(cat_missed_pred_max),
            "top_n": int(cat_top_n),
        },
        "slices": {
            "injury_regime": {
                "team_games": int(slices["injury_regime"].groupby(["game_id", "team_id"]).ngroups),
                "player_rows": int(len(slices["injury_regime"])),
            },
            "non_injury": {
                "team_games": int(slices["non_injury"].groupby(["game_id", "team_id"]).ngroups),
                "player_rows": int(len(slices["non_injury"])),
            },
            "all_games": {
                "team_games": int(slices["all_games"].groupby(["game_id", "team_id"]).ngroups),
                "player_rows": int(len(slices["all_games"])),
            },
        },
        "models": {},
    }

    baseline_all = slices["all_games"].copy()
    baseline_all["_pred_baseline"] = injury_eval._compress_to_top_k(
        baseline_all, pred_col="_pred_minutes", k=int(baseline_top_k)
    )

    models: dict[str, dict[str, pd.DataFrame]] = {
        "current": {
            "injury_regime": slices["injury_regime"],
            "non_injury": slices["non_injury"],
            "all_games": slices["all_games"],
        },
        f"baseline_top{baseline_top_k}": {
            "injury_regime": baseline_all.loc[baseline_all["injury_regime"]].copy(),
            "non_injury": baseline_all.loc[slices["non_injury"].index].copy(),
            "all_games": baseline_all,
        },
    }

    candidate_coverage: CandidateCoverage | None = None
    if candidate_root is not None:
        candidate_root = candidate_root.expanduser().resolve()
        required_pairs = (
            snapshot_eval[["game_date", "run_id"]]
            .drop_duplicates()
            .assign(game_date=lambda df: pd.to_datetime(df["game_date"]).dt.date)
            .sort_values(["game_date", "run_id"], kind="mergesort")
            .reset_index(drop=True)
        )
        typer.echo(f"[load] candidate minutes ({len(required_pairs)} date-runs) from {candidate_root}")
        cand_preds, candidate_coverage = _load_candidate_minutes(
            root=candidate_root,
            required=required_pairs,
            minutes_col=candidate_minutes_col,
            require_full_coverage=require_candidate_coverage,
        )
        results["candidate_coverage"] = candidate_coverage.as_dict()

        # Join candidate predictions by (run_id, game_id, team_id, player_id) to
        # ensure we compare against the *same* last-before-tip snapshot chosen by
        # the live logs evaluation.
        eval_cand = eval_base.merge(
            cand_preds,
            on=["run_id", "game_id", "team_id", "player_id"],
            how="left",
        )
        eval_cand["_pred_minutes"] = eval_cand["_pred_minutes"].fillna(0.0)
        eval_cand = eval_cand.merge(injury_table, on=["game_id", "team_id"], how="inner")
        eval_cand["injury_regime"] = eval_cand["injury_regime"].astype("boolean").fillna(False).astype(bool)
        cand_slices = injury_eval._build_eval_slices(eval_cand)
        models["candidate"] = {
            "injury_regime": cand_slices["injury_regime"],
            "non_injury": cand_slices["non_injury"],
            "all_games": cand_slices["all_games"],
        }

    # Per-team breakdown: top 10 worst teams by bench_core MAE.
    for slice_name in ("injury_regime", "non_injury", "all_games"):
        current_team = injury_eval._bench_core_team_mae_table(
            models["current"][slice_name], pred_col="_pred_minutes"
        ).rename(
            columns={
                "bench_core_mae": "bench_core_mae_current",
                "bench_core_bias": "bench_core_bias_current",
                "bench_core_team_games": "bench_core_team_games_current",
            }
        )
        merged = current_team.copy()
        sort_key = "bench_core_mae_current"
        if "candidate" in models:
            cand_team = injury_eval._bench_core_team_mae_table(
                models["candidate"][slice_name], pred_col="_pred_minutes"
            ).rename(
                columns={
                    "bench_core_mae": "bench_core_mae_candidate",
                    "bench_core_bias": "bench_core_bias_candidate",
                    "bench_core_team_games": "bench_core_team_games_candidate",
                }
            )
            merged = merged.merge(cand_team, on="team_id", how="outer")
            merged["bench_core_mae_delta_candidate_minus_current"] = (
                merged["bench_core_mae_candidate"] - merged["bench_core_mae_current"]
            )
            sort_key = "bench_core_mae_candidate"
        merged = merged.sort_values(sort_key, ascending=False, kind="mergesort").head(10).copy()

        records: list[dict[str, Any]] = []
        for row in merged.to_dict(orient="records"):
            clean: dict[str, Any] = {}
            for key, value in row.items():
                if value is None or pd.isna(value):
                    clean[key] = None
                elif key == "team_id":
                    clean[key] = int(value)
                elif key.endswith("_team_games_current") or key.endswith("_team_games_candidate"):
                    clean[key] = int(value)
                else:
                    clean[key] = float(value)
            records.append(clean)
        results["slices"][slice_name]["worst_teams_by_bench_core_mae"] = records

    # Compute metrics per model per slice.
    for model_name, model_slices in models.items():
        results["models"][model_name] = {}
        for slice_name, slice_df in model_slices.items():
            pred_name = "_pred_minutes" if model_name != f"baseline_top{baseline_top_k}" else "_pred_baseline"
            metrics = injury_eval._compute_metrics(slice_df, pred_col=pred_name).to_dict()
            metrics["catastrophic"] = _catastrophic_minutes_metrics(
                slice_df,
                pred_col=pred_name,
                actual_col="minutes",
                ghost_pred_min=cat_ghost_pred_min,
                ghost_actual_max=cat_ghost_actual_max,
                missed_actual_min=cat_missed_actual_min,
                missed_pred_max=cat_missed_pred_max,
                top_n=cat_top_n,
            )
            if cat_active_only:
                active_slice = _active_only_slice(slice_df)
                active_cat = _catastrophic_minutes_metrics(
                    active_slice,
                    pred_col=pred_name,
                    actual_col="minutes",
                    ghost_pred_min=cat_ghost_pred_min,
                    ghost_actual_max=cat_ghost_actual_max,
                    missed_actual_min=cat_missed_actual_min,
                    missed_pred_max=cat_missed_pred_max,
                    top_n=cat_top_n,
                )
                active_cat["scope"] = {
                    "rows_total": int(len(slice_df)),
                    "rows_active_only": int(len(active_slice)),
                    "rows_out_like": int(len(slice_df) - len(active_slice)),
                }
                metrics["catastrophic_active_only"] = active_cat
            results["models"][model_name][slice_name] = metrics

    payload = json.dumps(results, indent=2, sort_keys=True)
    typer.echo(payload)
    if out is not None:
        out_path = out.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        typer.echo(f"[write] {out_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
