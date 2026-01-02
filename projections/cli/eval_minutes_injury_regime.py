"""Evaluate minutes "next man up" realism with injury-regime slices.

This script is intended to diagnose the failure mode where bench minutes are
too smeared when starters are OUT.

Slices:
- `injury_regime`: team-games meeting the configured OUT thresholds
- `non_injury`: strictly healthy team-games (no starters out, no team outs)
- `all_games`: full evaluation window (unfiltered)

Metrics:
- Team-game: rotation depth (>=10 min), top-7 sum, bench concentration (Gini/HHI)
- Bench-core: errors on bench players who actually played real minutes
  * bench_core = non-starters with actual minutes >= 18
  * top-2 bench by actual minutes, and the single largest bench-minute player
- Buckets: player error tables by actual minutes bins and by starters_out count buckets

Comparison:
- Current predictions (from a predictions root)
- Baseline heuristic: compress-to-top-K (default K=8)
- Optional candidate predictions root (e.g., experimental model output)

Example:
  uv run python -m projections.cli.eval_minutes_injury_regime \\
    --start-date 2025-11-01 --end-date 2025-11-30 \\
    --out reports/minutes_injury_regime/2025-11.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import typer

from projections import paths
from projections.labels import derive_starter_flag_labels

app = typer.Typer(help=__doc__)

TEAM_TOTAL_MINUTES = 240.0
BENCH_CORE_MINUTES = 18.0
ROTATION_MINUTES_THRESHOLD = 10.0
TOP_K = 7
CORE_TOP_K = 9

ACTUAL_MINUTE_BINS = (-1e-9, 5.0, 15.0, 25.0, 38.0, float("inf"))
ACTUAL_MINUTE_BIN_LABELS = ("0-5", "5-15", "15-25", "25-38", "38+")
STARTERS_OUT_BUCKET_LABELS = ("0", "1", "2", "3+")


def _iter_days(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _season_from_date(day: date) -> int:
    return day.year if day.month >= 8 else day.year - 1


def _resolve_preds_path(preds_root: Path, day: date, *, run_id: str | None) -> Path | None:
    candidates = [
        preds_root / f"game_date={day.isoformat()}",
        preds_root / day.isoformat(),
    ]
    for base in candidates:
        if not base.exists():
            continue
        if run_id is not None:
            run_dir = base / f"run={run_id}"
            run_path = run_dir / "minutes.parquet"
            if run_path.exists():
                return run_path
            return None
        direct = base / "minutes.parquet"
        if direct.exists():
            return direct
    return None


def _load_predictions(
    *,
    preds_root: Path,
    start: date,
    end: date,
    run_id: str | None,
    name: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in _iter_days(start, end):
        path = _resolve_preds_path(preds_root, day, run_id=run_id)
        if path is None:
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
        else:
            df["game_date"] = pd.Timestamp(day).normalize()
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No prediction files found under {preds_root} for {start} → {end}")
    combined = pd.concat(frames, ignore_index=True)
    required = {"game_id", "player_id", "team_id"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"Predictions missing required columns: {', '.join(sorted(missing))}")
    combined["game_id"] = pd.to_numeric(combined["game_id"], errors="coerce").astype("Int64")
    combined["player_id"] = pd.to_numeric(combined["player_id"], errors="coerce").astype("Int64")
    combined["team_id"] = pd.to_numeric(combined["team_id"], errors="coerce").astype("Int64")
    combined = combined.dropna(subset=["game_id", "player_id", "team_id"]).copy()
    combined["game_id"] = combined["game_id"].astype(int)
    combined["player_id"] = combined["player_id"].astype(int)
    combined["team_id"] = combined["team_id"].astype(int)

    # Prefer p50; fall back to minutes_mean or minutes_p50_cond if needed.
    minutes_col = None
    for candidate in ("minutes_p50", "minutes_mean", "minutes_p50_cond"):
        if candidate in combined.columns:
            minutes_col = candidate
            break
    if minutes_col is None:
        raise ValueError("Predictions missing minutes column (expected minutes_p50 or minutes_mean).")
    combined["_pred_minutes"] = pd.to_numeric(combined[minutes_col], errors="coerce").fillna(0.0).astype(float)
    combined["_pred_name"] = name

    # Ensure uniqueness; keep last (some roots store multiple runs concatenated).
    combined = combined.sort_values(["game_date", "game_id", "team_id", "player_id"], kind="mergesort")
    combined = combined.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last")
    return combined[["game_id", "team_id", "player_id", "_pred_minutes", "_pred_name"]]


def _coerce_minutes_to_float(series: pd.Series) -> pd.Series:
    raw = series
    if raw.dtype == object:
        values = raw.astype(str)
        out = pd.to_numeric(raw, errors="coerce")

        # Legacy label sources store minutes as ISO8601 durations like "PT38M34.00S".
        # Some seasons store numeric floats; windows spanning multiple seasons can be mixed,
        # so parse per-row instead of using a majority-format heuristic.
        pt_mask = values.str.startswith("PT")
        if pt_mask.any():
            parts = values[pt_mask].str.extract(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?")
            hours = pd.to_numeric(parts[0], errors="coerce").fillna(0.0).astype(float)
            minutes = pd.to_numeric(parts[1], errors="coerce").fillna(0.0).astype(float)
            seconds = pd.to_numeric(parts[2], errors="coerce").fillna(0.0).astype(float)
            out.loc[pt_mask] = hours * 60.0 + minutes + seconds / 60.0

        return out.fillna(0.0).astype(float)

    return pd.to_numeric(raw, errors="coerce").fillna(0.0).astype(float)


def _load_labels(*, data_root: Path, start: date, end: date) -> pd.DataFrame:
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    seasons: list[int] = sorted({_season_from_date(day) for day in _iter_days(start, end)})
    frames: list[pd.DataFrame] = []

    # Prefer legacy season-level boxscore labels (full slates for historical seasons).
    legacy_root = data_root / "labels"
    gold_root = data_root / "gold" / "labels_minutes_v1"

    for season in seasons:
        legacy_path = legacy_root / f"season={season}" / "boxscore_labels.parquet"
        if legacy_path.exists():
            season_df = pd.read_parquet(legacy_path)
            season_df["game_date"] = pd.to_datetime(season_df["game_date"]).dt.normalize()
            season_df = season_df[(season_df["game_date"] >= start_ts) & (season_df["game_date"] <= end_ts)].copy()
            if not season_df.empty:
                frames.append(season_df)
            continue

        # Fallback: gold day partitions (primarily for current season).
        for day in _iter_days(start, end):
            if _season_from_date(day) != season:
                continue
            path = gold_root / f"season={season}" / f"game_date={day.isoformat()}" / "labels.parquet"
            if not path.exists():
                continue
            frames.append(pd.read_parquet(path))

    if not frames:
        raise FileNotFoundError(
            f"No labels found under {legacy_root} or {gold_root} for {start} → {end}"
        )

    df = pd.concat(frames, ignore_index=True)
    for col in ("game_id", "player_id", "team_id"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["game_id", "player_id", "team_id"]).copy()
    df["game_id"] = df["game_id"].astype(int)
    df["player_id"] = df["player_id"].astype(int)
    df["team_id"] = df["team_id"].astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    df["minutes"] = _coerce_minutes_to_float(df["minutes"])

    # Legacy season boxscore labels can contain duplicate snapshots; keep the last copy.
    df = df.sort_values(["game_date", "game_id", "team_id", "player_id"], kind="mergesort")
    df = df.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last").copy()

    # Ensure we have exactly five starters per team-game (stable across label sources).
    df = derive_starter_flag_labels(
        df,
        minutes_col="minutes",
        game_col="game_id",
        team_col="team_id",
        player_col="player_id",
        output_col="starter_flag_label",
    )
    df["starter_flag_label"] = pd.to_numeric(df["starter_flag_label"], errors="coerce").fillna(0).astype(int)
    return df[["game_date", "game_id", "team_id", "player_id", "minutes", "starter_flag_label"]]


def _load_features_for_injury(*, data_root: Path, start: date, end: date) -> pd.DataFrame:
    features_root = data_root / "gold" / "features_minutes_v1"
    month_keys: set[tuple[int, int]] = set()
    for day in _iter_days(start, end):
        month_keys.add((_season_from_date(day), day.month))

    frames: list[pd.DataFrame] = []
    cols = {"game_id", "team_id", "player_id", "game_date", "tip_ts", "is_out", "status", "lineup_role"}
    for season, month in sorted(month_keys):
        path = features_root / f"season={season}" / f"month={month:02d}" / "features.parquet"
        if not path.exists():
            continue
        df_full = pd.read_parquet(path)
        keep = [c for c in df_full.columns if c in cols]
        if not keep:
            continue
        df = df_full[keep].copy()
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No features found under {features_root} for {start} → {end}")
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    df = df[(df["game_date"] >= pd.to_datetime(start)) & (df["game_date"] <= pd.to_datetime(end))].copy()
    for col in ("game_id", "player_id", "team_id"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["game_id", "player_id", "team_id"]).copy()
    df["game_id"] = df["game_id"].astype(int)
    df["player_id"] = df["player_id"].astype(int)
    df["team_id"] = df["team_id"].astype(int)
    if "tip_ts" in df.columns:
        df["tip_ts"] = pd.to_datetime(df["tip_ts"], utc=True, errors="coerce")
    else:
        df["tip_ts"] = pd.NaT
    df["is_out"] = pd.to_numeric(df.get("is_out", 0), errors="coerce").fillna(0).astype(int)
    status = df.get("status", pd.Series("", index=df.index)).astype(str).str.upper()
    lineup_role = df.get("lineup_role", pd.Series("", index=df.index)).astype(str).str.lower()
    df["_out_flag"] = (df["is_out"] == 1) | status.str.contains("OUT") | lineup_role.eq("out")
    return df[["game_date", "game_id", "team_id", "player_id", "tip_ts", "_out_flag"]]


def _gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr >= 0.0)]
    if arr.size < 2:
        return 0.0
    total = float(arr.sum())
    if total <= 0.0:
        return 0.0
    arr = np.sort(arr)
    cumsum = np.cumsum(arr)
    n = float(arr.size)
    gini = (n + 1.0 - 2.0 * float(cumsum.sum()) / total) / n
    return float(max(0.0, min(1.0, gini)))


def _hhi(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr >= 0.0)]
    total = float(arr.sum())
    if total <= 0.0:
        return 0.0
    shares = arr / total
    return float(np.sum(shares**2))


def _topk_sum(values: np.ndarray, k: int) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    ordered = np.sort(arr)[::-1]
    return float(ordered[:k].sum())


def _actual_minutes_bin_series(minutes: pd.Series) -> pd.Series:
    values = pd.to_numeric(minutes, errors="coerce").fillna(0.0).astype(float)
    # Include an explicit 38+ bucket to avoid dropping high-minute stars / OT.
    binned = pd.cut(
        values,
        bins=list(ACTUAL_MINUTE_BINS),
        labels=list(ACTUAL_MINUTE_BIN_LABELS),
        include_lowest=True,
        right=True,
    )
    return binned.astype(str).fillna(ACTUAL_MINUTE_BIN_LABELS[0])


def _starters_out_bucket_series(starters_out_count: pd.Series) -> pd.Series:
    values = pd.to_numeric(starters_out_count, errors="coerce").fillna(0).astype(int)
    bucket = pd.Series("0", index=values.index, dtype=object)
    bucket = bucket.where(values != 1, "1")
    bucket = bucket.where(values != 2, "2")
    bucket = bucket.where(values < 3, "3+")
    return bucket


def _build_eval_slices(eval_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    required = {"injury_regime", "starter_out_count", "team_out_count"}
    missing = required - set(eval_frame.columns)
    if missing:
        raise ValueError(f"Eval frame missing required columns for slicing: {', '.join(sorted(missing))}")

    non_injury_mask = (eval_frame["starter_out_count"] == 0) & (eval_frame["team_out_count"] == 0)
    slices = {
        "injury_regime": eval_frame.loc[eval_frame["injury_regime"]].copy(),
        "non_injury": eval_frame.loc[non_injury_mask].copy(),
        "all_games": eval_frame.copy(),
    }
    if not slices["non_injury"].empty:
        max_starters_out = int(pd.to_numeric(slices["non_injury"]["starter_out_count"], errors="coerce").fillna(0).max())
        max_team_out = int(pd.to_numeric(slices["non_injury"]["team_out_count"], errors="coerce").fillna(0).max())
        assert max_starters_out == 0, "non_injury slice contains starter_out_count > 0"
        assert max_team_out == 0, "non_injury slice contains team_out_count > 0"
    return slices


def _bucket_error_table(
    df: pd.DataFrame,
    *,
    bucket: pd.Series,
    expected_order: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    if df.empty:
        return {label: {"n": 0, "mae": float("nan"), "bias": float("nan")} for label in expected_order}

    working = df.copy()
    working["_bucket"] = bucket.astype(str)
    grouped = (
        working.groupby("_bucket", sort=False)
        .agg(
            n=("_abs_err", "size"),
            mae=("_abs_err", "mean"),
            bias=("_err", "mean"),
        )
        .reset_index()
    )
    payload: dict[str, dict[str, float]] = {
        label: {"n": 0, "mae": float("nan"), "bias": float("nan")} for label in expected_order
    }
    for row in grouped.to_dict(orient="records"):
        label = str(row.get("_bucket"))
        if label not in payload:
            payload[label] = {"n": 0, "mae": float("nan"), "bias": float("nan")}
        payload[label] = {
            "n": int(row.get("n", 0)),
            "mae": float(row.get("mae", float("nan"))),
            "bias": float(row.get("bias", float("nan"))),
        }
    return payload


def _quantiles(values: np.ndarray, qs: tuple[float, ...]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{int(q*100)}": float("nan") for q in qs}
    out: dict[str, float] = {}
    for q in qs:
        out[f"p{int(q*100)}"] = float(np.quantile(arr, q))
    return out


def _role_bucket(minutes: float, starter_flag_label: int) -> str:
    if minutes <= 0.0:
        return "dnp"
    if int(starter_flag_label) == 1:
        return "starter"
    if minutes >= 10.0:
        return "bench"
    return "deep"


def _compress_to_top_k(
    df: pd.DataFrame,
    *,
    pred_col: str,
    k: int,
    cap: float = 48.0,
) -> pd.Series:
    """Compress per-team minutes to top-K and rescale to 240 with a hard cap."""

    if df.empty:
        return pd.Series(dtype=float)

    out = pd.Series(0.0, index=df.index, dtype=float)

    for (game_id, team_id), g in df.groupby(["game_id", "team_id"], sort=False):
        mins = pd.to_numeric(g[pred_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if mins.size == 0:
            continue
        order = np.argsort(-mins, kind="mergesort")
        keep_local = order[: min(k, len(order))]
        keep_idx = g.index.to_numpy()[keep_local]
        kept = mins[keep_local].copy()
        if float(kept.sum()) <= 0.0:
            kept[:] = TEAM_TOTAL_MINUTES / float(len(kept))
        else:
            kept *= TEAM_TOTAL_MINUTES / float(kept.sum())

        kept = np.minimum(kept, cap)
        # Redistribute leftover minutes to non-capped players.
        for _ in range(10):
            total = float(kept.sum())
            gap = TEAM_TOTAL_MINUTES - total
            if abs(gap) <= 1e-6:
                break
            room = cap - kept
            eligible = room > 1e-6
            if not eligible.any():
                break
            if gap > 0:
                add = gap * (room[eligible] / float(room[eligible].sum()))
                kept[eligible] += np.minimum(add, room[eligible])
            else:
                # Remove proportionally from kept minutes.
                pos = kept > 1e-6
                if not pos.any():
                    break
                remove = (-gap) * (kept[pos] / float(kept[pos].sum()))
                kept[pos] = np.maximum(0.0, kept[pos] - remove)

        out.loc[keep_idx] = kept

    return out


@dataclass(frozen=True)
class ModelSliceMetrics:
    n_team_games: int
    n_player_rows: int
    team_total_minutes_actual_mean: float
    team_total_minutes_actual_dev_max_abs: float
    team_total_minutes_pred_dev_mean_abs: float
    team_total_minutes_pred_dev_max_abs: float
    player_mae: float
    player_bias: float
    player_mae_by_role: dict[str, float]
    player_error_by_actual_minutes: dict[str, dict[str, float]]
    player_error_by_starters_out: dict[str, dict[str, float]]
    rotation_depth_mae: float
    top7_sum_actual_mean: float
    top7_sum_pred_mean: float
    top7_sum_mae: float
    top7_sum_bias: float
    top7_sum_abs_err_quantiles: dict[str, float]
    top7_sum_err_quantiles: dict[str, float]
    top9_player_mae: float
    top9_sum_actual_mean: float
    top9_sum_pred_mean: float
    top9_sum_mae: float
    top9_sum_bias: float
    top9_player_mae_team240: float
    top9_sum_pred_mean_team240: float
    top9_sum_mae_team240: float
    top9_sum_bias_team240: float
    tail_minutes_actual_mean: float
    tail_minutes_pred_mean: float
    tail_minutes_mae: float
    tail_minutes_bias: float
    tail_minutes_actual_mean_team240: float
    tail_minutes_pred_mean_team240: float
    tail_minutes_mae_team240: float
    tail_minutes_bias_team240: float
    bench_gini_mae: float
    bench_hhi_mae: float
    bench_core_mae: float
    bench_core_bias: float
    bench_core_team_games: int
    bench_core_player_rows: int
    top2_bench_mae: float
    top2_bench_bias: float
    bench_max_mae: float
    bench_max_bias: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_team_games": self.n_team_games,
            "n_player_rows": self.n_player_rows,
            "team_total_minutes_actual_mean": self.team_total_minutes_actual_mean,
            "team_total_minutes_actual_dev_max_abs": self.team_total_minutes_actual_dev_max_abs,
            "team_total_minutes_pred_dev_mean_abs": self.team_total_minutes_pred_dev_mean_abs,
            "team_total_minutes_pred_dev_max_abs": self.team_total_minutes_pred_dev_max_abs,
            "player_mae": self.player_mae,
            "player_bias": self.player_bias,
            "player_mae_by_role": self.player_mae_by_role,
            "player_error_by_actual_minutes": self.player_error_by_actual_minutes,
            "player_error_by_starters_out": self.player_error_by_starters_out,
            "rotation_depth_mae": self.rotation_depth_mae,
            "top7_sum_actual_mean": self.top7_sum_actual_mean,
            "top7_sum_pred_mean": self.top7_sum_pred_mean,
            "top7_sum_mae": self.top7_sum_mae,
            "top7_sum_bias": self.top7_sum_bias,
            "top7_sum_abs_err_quantiles": self.top7_sum_abs_err_quantiles,
            "top7_sum_err_quantiles": self.top7_sum_err_quantiles,
            "top9_player_mae": self.top9_player_mae,
            "top9_sum_actual_mean": self.top9_sum_actual_mean,
            "top9_sum_pred_mean": self.top9_sum_pred_mean,
            "top9_sum_mae": self.top9_sum_mae,
            "top9_sum_bias": self.top9_sum_bias,
            "top9_player_mae_team240": self.top9_player_mae_team240,
            "top9_sum_pred_mean_team240": self.top9_sum_pred_mean_team240,
            "top9_sum_mae_team240": self.top9_sum_mae_team240,
            "top9_sum_bias_team240": self.top9_sum_bias_team240,
            "tail_minutes_actual_mean": self.tail_minutes_actual_mean,
            "tail_minutes_pred_mean": self.tail_minutes_pred_mean,
            "tail_minutes_mae": self.tail_minutes_mae,
            "tail_minutes_bias": self.tail_minutes_bias,
            "tail_minutes_actual_mean_team240": self.tail_minutes_actual_mean_team240,
            "tail_minutes_pred_mean_team240": self.tail_minutes_pred_mean_team240,
            "tail_minutes_mae_team240": self.tail_minutes_mae_team240,
            "tail_minutes_bias_team240": self.tail_minutes_bias_team240,
            "bench_gini_mae": self.bench_gini_mae,
            "bench_hhi_mae": self.bench_hhi_mae,
            "bench_core_mae": self.bench_core_mae,
            "bench_core_bias": self.bench_core_bias,
            "bench_core_team_games": self.bench_core_team_games,
            "bench_core_player_rows": self.bench_core_player_rows,
            "top2_bench_mae": self.top2_bench_mae,
            "top2_bench_bias": self.top2_bench_bias,
            "bench_max_mae": self.bench_max_mae,
            "bench_max_bias": self.bench_max_bias,
        }


def _compute_metrics(df: pd.DataFrame, *, pred_col: str) -> ModelSliceMetrics:
    if df.empty:
        return ModelSliceMetrics(
            n_team_games=0,
            n_player_rows=0,
            team_total_minutes_actual_mean=float("nan"),
            team_total_minutes_actual_dev_max_abs=float("nan"),
            team_total_minutes_pred_dev_mean_abs=float("nan"),
            team_total_minutes_pred_dev_max_abs=float("nan"),
            player_mae=float("nan"),
            player_bias=float("nan"),
            player_mae_by_role={},
            player_error_by_actual_minutes={label: {"n": 0, "mae": float("nan"), "bias": float("nan")} for label in ACTUAL_MINUTE_BIN_LABELS},
            player_error_by_starters_out={label: {"n": 0, "mae": float("nan"), "bias": float("nan")} for label in STARTERS_OUT_BUCKET_LABELS},
            rotation_depth_mae=float("nan"),
            top7_sum_actual_mean=float("nan"),
            top7_sum_pred_mean=float("nan"),
            top7_sum_mae=float("nan"),
            top7_sum_bias=float("nan"),
            top7_sum_abs_err_quantiles={"p50": float("nan"), "p90": float("nan")},
            top7_sum_err_quantiles={"p50": float("nan"), "p90": float("nan")},
            top9_player_mae=float("nan"),
            top9_sum_actual_mean=float("nan"),
            top9_sum_pred_mean=float("nan"),
            top9_sum_mae=float("nan"),
            top9_sum_bias=float("nan"),
            top9_player_mae_team240=float("nan"),
            top9_sum_pred_mean_team240=float("nan"),
            top9_sum_mae_team240=float("nan"),
            top9_sum_bias_team240=float("nan"),
            tail_minutes_actual_mean=float("nan"),
            tail_minutes_pred_mean=float("nan"),
            tail_minutes_mae=float("nan"),
            tail_minutes_bias=float("nan"),
            tail_minutes_actual_mean_team240=float("nan"),
            tail_minutes_pred_mean_team240=float("nan"),
            tail_minutes_mae_team240=float("nan"),
            tail_minutes_bias_team240=float("nan"),
            bench_gini_mae=float("nan"),
            bench_hhi_mae=float("nan"),
            bench_core_mae=float("nan"),
            bench_core_bias=float("nan"),
            bench_core_team_games=0,
            bench_core_player_rows=0,
            top2_bench_mae=float("nan"),
            top2_bench_bias=float("nan"),
            bench_max_mae=float("nan"),
            bench_max_bias=float("nan"),
        )

    working = df.copy()
    working["_pred"] = pd.to_numeric(working[pred_col], errors="coerce").fillna(0.0).astype(float)
    working["_err"] = working["_pred"] - working["minutes"]
    working["_abs_err"] = working["_err"].abs()
    working["_role"] = [
        _role_bucket(m, s)
        for m, s in zip(working["minutes"].to_numpy(dtype=float), working["starter_flag_label"].to_numpy(dtype=int))
    ]

    player_mae = float(working["_abs_err"].mean())
    player_bias = float(working["_err"].mean())
    by_role = (
        working.groupby("_role")["_abs_err"].mean().to_dict()
        if working["_role"].nunique() > 0
        else {}
    )
    by_role = {str(k): float(v) for k, v in by_role.items()}

    by_minutes = _bucket_error_table(
        working,
        bucket=_actual_minutes_bin_series(working["minutes"]),
        expected_order=ACTUAL_MINUTE_BIN_LABELS,
    )
    starters_out = working["starter_out_count"] if "starter_out_count" in working.columns else pd.Series(0, index=working.index)
    by_starters_out = _bucket_error_table(
        working,
        bucket=_starters_out_bucket_series(starters_out),
        expected_order=STARTERS_OUT_BUCKET_LABELS,
    )

    # Bench-core diagnostics (per team-game).
    starters_mask = working["starter_flag_label"].to_numpy(dtype=int) == 1
    bench = working.loc[~starters_mask].copy()
    bench_rank = (
        bench.groupby(["game_id", "team_id"], sort=False)["minutes"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    bench["_bench_rank"] = bench_rank

    bench_max = bench.loc[bench["_bench_rank"] == 1].copy()
    bench_max_mae = float(bench_max["_abs_err"].mean()) if not bench_max.empty else float("nan")
    bench_max_bias = float(bench_max["_err"].mean()) if not bench_max.empty else float("nan")

    bench_top2 = bench.loc[bench["_bench_rank"] <= 2].copy()
    top2_team = (
        bench_top2.groupby(["game_id", "team_id"], sort=False)
        .agg(mae=("_abs_err", "mean"), bias=("_err", "mean"))
        .reset_index()
    )
    top2_bench_mae = float(top2_team["mae"].mean()) if not top2_team.empty else float("nan")
    top2_bench_bias = float(top2_team["bias"].mean()) if not top2_team.empty else float("nan")

    bench_core = bench.loc[bench["minutes"] >= float(BENCH_CORE_MINUTES)].copy()
    core_team = (
        bench_core.groupby(["game_id", "team_id"], sort=False)
        .agg(mae=("_abs_err", "mean"), bias=("_err", "mean"), n=("player_id", "size"))
        .reset_index()
    )
    bench_core_mae = float(core_team["mae"].mean()) if not core_team.empty else float("nan")
    bench_core_bias = float(core_team["bias"].mean()) if not core_team.empty else float("nan")
    bench_core_team_games = int(len(core_team))
    bench_core_player_rows = int(bench_core.shape[0])

    team_rows: list[dict[str, float]] = []
    actual_totals: list[float] = []
    pred_totals: list[float] = []
    top7_actual_list: list[float] = []
    top7_pred_list: list[float] = []
    top7_err: list[float] = []
    top7_abs_err: list[float] = []
    top9_actual_list: list[float] = []
    top9_pred_list: list[float] = []
    top9_err: list[float] = []
    top9_abs_err: list[float] = []
    top9_player_abs_err: list[float] = []
    top9_pred_team240_list: list[float] = []
    top9_err_team240: list[float] = []
    top9_abs_err_team240: list[float] = []
    top9_player_abs_err_team240: list[float] = []
    tail_actual_list: list[float] = []
    tail_pred_list: list[float] = []
    tail_err: list[float] = []
    tail_abs_err: list[float] = []
    tail_pred_team240_list: list[float] = []
    tail_err_team240: list[float] = []
    tail_abs_err_team240: list[float] = []
    for (_, _), g in working.groupby(["game_id", "team_id"], sort=False):
        actual = g["minutes"].to_numpy(dtype=float)
        pred = g["_pred"].to_numpy(dtype=float)
        starters = g["starter_flag_label"].to_numpy(dtype=int) == 1
        bench_actual = actual[~starters]
        bench_pred = pred[~starters]

        actual_totals.append(float(np.sum(actual)))
        pred_totals.append(float(np.sum(pred)))

        top7_actual = _topk_sum(actual, TOP_K)
        top7_pred = _topk_sum(pred, TOP_K)
        top7_actual_list.append(top7_actual)
        top7_pred_list.append(top7_pred)
        err = float(top7_pred - top7_actual)
        top7_err.append(err)
        top7_abs_err.append(abs(err))

        # Core/top-9 metrics: define the core set by actual minutes, evaluate predicted minutes on that set.
        order = np.argsort(-actual, kind="mergesort")
        keep = order[: min(CORE_TOP_K, len(order))]
        top9_actual = float(np.sum(actual[keep])) if keep.size else 0.0
        top9_pred = float(np.sum(pred[keep])) if keep.size else 0.0
        top9_actual_list.append(top9_actual)
        top9_pred_list.append(top9_pred)
        err9 = float(top9_pred - top9_actual)
        top9_err.append(err9)
        top9_abs_err.append(abs(err9))
        if keep.size:
            top9_player_abs_err.extend(np.abs(pred[keep] - actual[keep]).astype(float).tolist())

        pred_sum = float(np.sum(pred))
        if pred_sum <= 0.0:
            pred_team240 = np.full_like(pred, TEAM_TOTAL_MINUTES / float(len(pred)))
        else:
            pred_team240 = pred * (TEAM_TOTAL_MINUTES / pred_sum)
        top9_pred_team240 = float(np.sum(pred_team240[keep])) if keep.size else 0.0
        top9_pred_team240_list.append(top9_pred_team240)
        err9_240 = float(top9_pred_team240 - top9_actual)
        top9_err_team240.append(err9_240)
        top9_abs_err_team240.append(abs(err9_240))
        if keep.size:
            top9_player_abs_err_team240.extend(np.abs(pred_team240[keep] - actual[keep]).astype(float).tolist())

        # Tail minutes are defined relative to a 240-minute target (aligned with sim constraint).
        tail_actual = float(TEAM_TOTAL_MINUTES - top9_actual)
        tail_pred = float(TEAM_TOTAL_MINUTES - top9_pred)
        tail_actual_list.append(tail_actual)
        tail_pred_list.append(tail_pred)
        tail_e = float(tail_pred - tail_actual)
        tail_err.append(tail_e)
        tail_abs_err.append(abs(tail_e))

        tail_pred_team240 = float(TEAM_TOTAL_MINUTES - top9_pred_team240)
        tail_pred_team240_list.append(tail_pred_team240)
        tail_e_240 = float(tail_pred_team240 - tail_actual)
        tail_err_team240.append(tail_e_240)
        tail_abs_err_team240.append(abs(tail_e_240))

        team_rows.append(
            {
                "rotation_depth_err": float(abs((actual >= ROTATION_MINUTES_THRESHOLD).sum() - (pred >= ROTATION_MINUTES_THRESHOLD).sum())),
                "bench_gini_err": float(abs(_gini(bench_actual) - _gini(bench_pred))),
                "bench_hhi_err": float(abs(_hhi(bench_actual) - _hhi(bench_pred))),
            }
        )
    team_df = pd.DataFrame(team_rows)
    actual_totals_arr = np.asarray(actual_totals, dtype=float)
    pred_totals_arr = np.asarray(pred_totals, dtype=float)
    top7_actual_arr = np.asarray(top7_actual_list, dtype=float)
    top7_pred_arr = np.asarray(top7_pred_list, dtype=float)
    top7_err_arr = np.asarray(top7_err, dtype=float)
    top7_abs_arr = np.asarray(top7_abs_err, dtype=float)
    top9_actual_arr = np.asarray(top9_actual_list, dtype=float)
    top9_pred_arr = np.asarray(top9_pred_list, dtype=float)
    top9_pred_team240_arr = np.asarray(top9_pred_team240_list, dtype=float)
    top9_err_arr = np.asarray(top9_err, dtype=float)
    top9_abs_arr = np.asarray(top9_abs_err, dtype=float)
    top9_player_abs_arr = np.asarray(top9_player_abs_err, dtype=float)
    top9_err_team240_arr = np.asarray(top9_err_team240, dtype=float)
    top9_abs_team240_arr = np.asarray(top9_abs_err_team240, dtype=float)
    top9_player_abs_team240_arr = np.asarray(top9_player_abs_err_team240, dtype=float)
    tail_actual_arr = np.asarray(tail_actual_list, dtype=float)
    tail_pred_arr = np.asarray(tail_pred_list, dtype=float)
    tail_err_arr = np.asarray(tail_err, dtype=float)
    tail_abs_arr = np.asarray(tail_abs_err, dtype=float)
    tail_pred_team240_arr = np.asarray(tail_pred_team240_list, dtype=float)
    tail_err_team240_arr = np.asarray(tail_err_team240, dtype=float)
    tail_abs_team240_arr = np.asarray(tail_abs_err_team240, dtype=float)
    return ModelSliceMetrics(
        n_team_games=int(working.groupby(["game_id", "team_id"]).ngroups),
        n_player_rows=int(len(working)),
        team_total_minutes_actual_mean=float(np.mean(actual_totals_arr)) if actual_totals_arr.size else float("nan"),
        team_total_minutes_actual_dev_max_abs=float(np.max(np.abs(actual_totals_arr - TEAM_TOTAL_MINUTES))) if actual_totals_arr.size else float("nan"),
        team_total_minutes_pred_dev_mean_abs=float(np.mean(np.abs(pred_totals_arr - TEAM_TOTAL_MINUTES))) if pred_totals_arr.size else float("nan"),
        team_total_minutes_pred_dev_max_abs=float(np.max(np.abs(pred_totals_arr - TEAM_TOTAL_MINUTES))) if pred_totals_arr.size else float("nan"),
        player_mae=player_mae,
        player_bias=player_bias,
        player_mae_by_role=by_role,
        player_error_by_actual_minutes=by_minutes,
        player_error_by_starters_out=by_starters_out,
        rotation_depth_mae=float(team_df["rotation_depth_err"].mean()) if not team_df.empty else float("nan"),
        top7_sum_actual_mean=float(np.mean(top7_actual_arr)) if top7_actual_arr.size else float("nan"),
        top7_sum_pred_mean=float(np.mean(top7_pred_arr)) if top7_pred_arr.size else float("nan"),
        top7_sum_mae=float(np.mean(top7_abs_arr)) if top7_abs_arr.size else float("nan"),
        top7_sum_bias=float(np.mean(top7_err_arr)) if top7_err_arr.size else float("nan"),
        top7_sum_abs_err_quantiles=_quantiles(top7_abs_arr, (0.50, 0.90)),
        top7_sum_err_quantiles=_quantiles(top7_err_arr, (0.50, 0.90)),
        top9_player_mae=float(np.mean(top9_player_abs_arr)) if top9_player_abs_arr.size else float("nan"),
        top9_sum_actual_mean=float(np.mean(top9_actual_arr)) if top9_actual_arr.size else float("nan"),
        top9_sum_pred_mean=float(np.mean(top9_pred_arr)) if top9_pred_arr.size else float("nan"),
        top9_sum_mae=float(np.mean(top9_abs_arr)) if top9_abs_arr.size else float("nan"),
        top9_sum_bias=float(np.mean(top9_err_arr)) if top9_err_arr.size else float("nan"),
        top9_player_mae_team240=float(np.mean(top9_player_abs_team240_arr)) if top9_player_abs_team240_arr.size else float("nan"),
        top9_sum_pred_mean_team240=float(np.mean(top9_pred_team240_arr)) if top9_pred_team240_arr.size else float("nan"),
        top9_sum_mae_team240=float(np.mean(top9_abs_team240_arr)) if top9_abs_team240_arr.size else float("nan"),
        top9_sum_bias_team240=float(np.mean(top9_err_team240_arr)) if top9_err_team240_arr.size else float("nan"),
        tail_minutes_actual_mean=float(np.mean(tail_actual_arr)) if tail_actual_arr.size else float("nan"),
        tail_minutes_pred_mean=float(np.mean(tail_pred_arr)) if tail_pred_arr.size else float("nan"),
        tail_minutes_mae=float(np.mean(tail_abs_arr)) if tail_abs_arr.size else float("nan"),
        tail_minutes_bias=float(np.mean(tail_err_arr)) if tail_err_arr.size else float("nan"),
        tail_minutes_actual_mean_team240=float(np.mean(tail_actual_arr)) if tail_actual_arr.size else float("nan"),
        tail_minutes_pred_mean_team240=float(np.mean(tail_pred_team240_arr)) if tail_pred_team240_arr.size else float("nan"),
        tail_minutes_mae_team240=float(np.mean(tail_abs_team240_arr)) if tail_abs_team240_arr.size else float("nan"),
        tail_minutes_bias_team240=float(np.mean(tail_err_team240_arr)) if tail_err_team240_arr.size else float("nan"),
        bench_gini_mae=float(team_df["bench_gini_err"].mean()) if not team_df.empty else float("nan"),
        bench_hhi_mae=float(team_df["bench_hhi_err"].mean()) if not team_df.empty else float("nan"),
        bench_core_mae=bench_core_mae,
        bench_core_bias=bench_core_bias,
        bench_core_team_games=bench_core_team_games,
        bench_core_player_rows=bench_core_player_rows,
        top2_bench_mae=top2_bench_mae,
        top2_bench_bias=top2_bench_bias,
        bench_max_mae=bench_max_mae,
        bench_max_bias=bench_max_bias,
    )


def _bench_core_team_mae_table(df: pd.DataFrame, *, pred_col: str) -> pd.DataFrame:
    """Per-team bench_core MAE table (bench_core is defined per team-game)."""

    if df.empty:
        return pd.DataFrame(columns=["team_id", "bench_core_mae", "bench_core_bias", "bench_core_team_games"])

    working = df.copy()
    working["_pred"] = pd.to_numeric(working[pred_col], errors="coerce").fillna(0.0).astype(float)
    working["_err"] = working["_pred"] - pd.to_numeric(working["minutes"], errors="coerce").fillna(0.0).astype(float)
    working["_abs_err"] = working["_err"].abs()

    starters = pd.to_numeric(working["starter_flag_label"], errors="coerce").fillna(0).astype(int) == 1
    bench_core = working.loc[~starters & (working["minutes"] >= float(BENCH_CORE_MINUTES))].copy()
    if bench_core.empty:
        return pd.DataFrame(columns=["team_id", "bench_core_mae", "bench_core_bias", "bench_core_team_games"])

    team_games = (
        bench_core.groupby(["game_id", "team_id"], sort=False)
        .agg(teamgame_mae=("_abs_err", "mean"), teamgame_bias=("_err", "mean"))
        .reset_index()
    )
    per_team = (
        team_games.groupby("team_id", sort=False)
        .agg(
            bench_core_mae=("teamgame_mae", "mean"),
            bench_core_bias=("teamgame_bias", "mean"),
            bench_core_team_games=("teamgame_mae", "size"),
        )
        .reset_index()
    )
    per_team["team_id"] = pd.to_numeric(per_team["team_id"], errors="coerce").fillna(-1).astype(int)
    per_team = per_team[per_team["team_id"] >= 0].copy()
    return per_team.sort_values("bench_core_mae", ascending=False, kind="mergesort").reset_index(drop=True)


def _build_injury_regime_table(
    *,
    labels: pd.DataFrame,
    features: pd.DataFrame,
    min_starters_out: int,
    min_team_out: int,
) -> pd.DataFrame:
    if labels.empty or features.empty:
        return pd.DataFrame(columns=["game_id", "team_id", "starter_out_count", "team_out_count", "injury_regime"])

    # Build per-team-game ordering from features (tip_ts is available pre-lock).
    games = features[["game_id", "team_id", "tip_ts"]].drop_duplicates().copy()
    games["tip_ts"] = pd.to_datetime(games["tip_ts"], utc=True, errors="coerce")
    games = games.dropna(subset=["tip_ts"])
    games = games.sort_values(["team_id", "tip_ts", "game_id"], kind="mergesort")
    games["prev_game_id"] = games.groupby("team_id")["game_id"].shift(1)

    prev_starters = labels[labels["starter_flag_label"] == 1][["game_id", "team_id", "player_id"]].copy()
    prev_starters = prev_starters.rename(columns={"game_id": "prev_game_id"})

    # Team OUT count from the features slice (includes all rostered players).
    team_out = (
        features.groupby(["game_id", "team_id"], sort=False)["_out_flag"].sum().reset_index(name="team_out_count")
    )

    starter_out = (
        games.merge(prev_starters, on=["team_id", "prev_game_id"], how="left")
        .merge(
            features[["game_id", "team_id", "player_id", "_out_flag"]],
            on=["game_id", "team_id", "player_id"],
            how="left",
        )
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
    start_date: str = typer.Option(..., help="Start game_date (YYYY-MM-DD, inclusive)."),
    end_date: str = typer.Option(..., help="End game_date (YYYY-MM-DD, inclusive)."),
    data_root: Path = typer.Option(paths.get_data_root(), help="Data root (defaults to PROJECTIONS_DATA_ROOT)."),
    preds_root: Path = typer.Option(
        paths.data_path("gold", "projections_minutes_v1"),
        help="Root containing per-day minutes.parquet predictions.",
    ),
    preds_run_id: str | None = typer.Option(None, help="Optional run id under day/run=<id>/minutes.parquet."),
    candidate_root: Path | None = typer.Option(None, help="Optional second predictions root to compare."),
    candidate_run_id: str | None = typer.Option(None, help="Optional candidate run id (under candidate root)."),
    min_starters_out: int = typer.Option(1, help="Injury regime if >= this many previous-game starters are OUT."),
    min_team_out: int = typer.Option(2, help="Injury regime if >= this many total OUT players on team."),
    baseline_top_k: int = typer.Option(8, help="Baseline compress-to-top-K heuristic."),
    out: Path | None = typer.Option(None, help="Optional JSON output path."),
    lookback_days: int = typer.Option(
        30, help="Lookback window for computing previous-game starters (days before start_date)."
    ),
) -> None:
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    if end < start:
        raise typer.BadParameter("end_date must be on or after start_date", param_name="end_date")

    data_root = data_root.expanduser().resolve()
    preds_root = preds_root.expanduser().resolve()
    start_lb = start - timedelta(days=int(lookback_days))

    typer.echo(f"[load] labels {start_lb} → {end}")
    labels = _load_labels(data_root=data_root, start=start_lb, end=end)
    labels_eval = labels[(labels["game_date"] >= pd.to_datetime(start)) & (labels["game_date"] <= pd.to_datetime(end))].copy()

    typer.echo(f"[load] features {start_lb} → {end}")
    features = _load_features_for_injury(data_root=data_root, start=start_lb, end=end)

    injury_table = _build_injury_regime_table(
        labels=labels,
        features=features,
        min_starters_out=min_starters_out,
        min_team_out=min_team_out,
    )
    injury_eval = injury_table.merge(
        labels_eval[["game_id", "team_id"]].drop_duplicates(),
        on=["game_id", "team_id"],
        how="inner",
    )

    typer.echo(f"[load] current preds {start} → {end}")
    current_preds = _load_predictions(
        preds_root=preds_root, start=start, end=end, run_id=preds_run_id, name="current"
    )

    eval_current = labels_eval.merge(
        current_preds.drop(columns=["_pred_name"]),
        on=["game_id", "team_id", "player_id"],
        how="left",
    )
    eval_current["_pred_minutes"] = eval_current["_pred_minutes"].fillna(0.0)
    # Restrict evaluation to team-games for which the injury regime table is defined (features present).
    eval_current = eval_current.merge(injury_eval, on=["game_id", "team_id"], how="inner")
    eval_current["injury_regime"] = eval_current["injury_regime"].astype("boolean").fillna(False).astype(bool)

    # Slices:
    # - injury_regime: configured thresholds
    # - non_injury: strict healthy teams (no starters out, no team outs)
    # - all_games: full eval window (unfiltered)
    slices = _build_eval_slices(eval_current)
    if slices["injury_regime"].empty:
        raise ValueError("No injury-regime rows found for the requested window; widen the date range or relax thresholds.")

    results: dict[str, Any] = {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "injury_regime": {
            "min_starters_out": int(min_starters_out),
            "min_team_out": int(min_team_out),
            "lookback_days": int(lookback_days),
        },
        "slices": {
            "injury_regime": {
                "team_games": int(slices["injury_regime"].groupby(["game_id", "team_id"]).ngroups),
                "player_rows": int(len(slices["injury_regime"])),
                "min_starters_out": int(min_starters_out),
                "min_team_out": int(min_team_out),
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

    # Baseline is derived from current predictions.
    baseline_all = slices["all_games"].copy()
    baseline_all["_pred_baseline"] = _compress_to_top_k(baseline_all, pred_col="_pred_minutes", k=int(baseline_top_k))

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

    if candidate_root is not None:
        typer.echo(f"[load] candidate preds {start} → {end}")
        cand_preds = _load_predictions(
            preds_root=candidate_root.expanduser().resolve(),
            start=start,
            end=end,
            run_id=candidate_run_id,
            name="candidate",
        )
        eval_cand = labels_eval.merge(
            cand_preds.drop(columns=["_pred_name"]),
            on=["game_id", "team_id", "player_id"],
            how="left",
        )
        eval_cand["_pred_minutes"] = eval_cand["_pred_minutes"].fillna(0.0)
        eval_cand = eval_cand.merge(injury_eval, on=["game_id", "team_id"], how="inner")
        eval_cand["injury_regime"] = eval_cand["injury_regime"].astype("boolean").fillna(False).astype(bool)
        cand_slices = _build_eval_slices(eval_cand)
        models["candidate"] = {
            "injury_regime": cand_slices["injury_regime"],
            "non_injury": cand_slices["non_injury"],
            "all_games": cand_slices["all_games"],
        }

    # Per-team breakdown: top 10 worst teams by bench_core MAE.
    for slice_name in ("injury_regime", "non_injury", "all_games"):
        current_team = _bench_core_team_mae_table(models["current"][slice_name], pred_col="_pred_minutes").rename(
            columns={
                "bench_core_mae": "bench_core_mae_current",
                "bench_core_bias": "bench_core_bias_current",
                "bench_core_team_games": "bench_core_team_games_current",
            }
        )
        merged = current_team.copy()
        sort_key = "bench_core_mae_current"
        if "candidate" in models:
            cand_team = _bench_core_team_mae_table(models["candidate"][slice_name], pred_col="_pred_minutes").rename(
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
            pred_col = "_pred_minutes" if model_name != f"baseline_top{baseline_top_k}" else "_pred_baseline"
            metrics = _compute_metrics(slice_df, pred_col=pred_col).to_dict()
            if slice_name == "non_injury" and not slice_df.empty:
                by_out = metrics.get("player_error_by_starters_out", {})
                for label, row in by_out.items():
                    if label != "0" and int(row.get("n", 0)) > 0:
                        raise AssertionError("non_injury slice contains starters_out > 0")
            results["models"][model_name][slice_name] = metrics

    payload = json.dumps(results, indent=2, sort_keys=True)
    typer.echo(payload)
    if out is not None:
        out_path = out.expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        typer.echo(f"[write] {out_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
