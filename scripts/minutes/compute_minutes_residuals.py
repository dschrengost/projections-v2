"""Compute minutes residual summaries for the current minutes head.

Residual definition:
    r = minutes_actual - minutes_pred_center

Buckets: (starter_flag, status_bucket, minutes_p50_bin)
Output: $DATA_ROOT/artifacts/minutes_v1/residuals/<run_id>_minutes_residuals.json
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import typer

from projections.minutes_v1.production import load_production_minutes_bundle
from projections.paths import data_path
from projections.sim_v2.minutes_noise import (
    MINUTES_P50_BIN_EDGES,
    bin_bounds,
    minutes_bin_indices,
    status_bucket_from_raw,
)

Split = Literal["train", "cal", "val", "all"]

app = typer.Typer(add_completion=False)

STATUS_CANDIDATES: tuple[str, ...] = (
    "status_pred",
    "status",
    "injury_status_pred",
    "injury_status",
    "availability_status_pred",
    "availability_status",
)


def _season_from_day(day: pd.Timestamp) -> int:
    return day.year if day.month >= 8 else day.year - 1


def _iter_days(start: pd.Timestamp, end: pd.Timestamp):
    cur = start
    while cur <= end:
        yield cur
        cur += pd.Timedelta(days=1)


def _window_from_meta(meta: dict, split: Split) -> tuple[pd.Timestamp, pd.Timestamp]:
    windows = meta.get("windows") or {}
    if split != "all":
        window = windows.get(split)
        if not window:
            raise RuntimeError(f"Missing date window for split={split}")
        start = pd.Timestamp(window["start"]).tz_localize(None).normalize()
        end = pd.Timestamp(window["end"]).tz_localize(None).normalize()
        return start, end

    if not windows:
        raise RuntimeError("No date windows found in minutes bundle meta")
    starts = []
    ends = []
    for win in windows.values():
        try:
            starts.append(pd.Timestamp(win["start"]).tz_localize(None).normalize())
            ends.append(pd.Timestamp(win["end"]).tz_localize(None).normalize())
        except Exception:
            continue
    if not starts or not ends:
        raise RuntimeError("Unable to derive aggregate window from bundle meta")
    return min(starts), max(ends)


def _load_proj_partition(root: Path, day: date) -> pd.DataFrame | None:
    iso = day.isoformat()
    candidates = [
        root / "gold" / "projections_minutes_v1" / iso / "minutes.parquet",
        root / "gold" / "projections_minutes_v1" / f"game_date={iso}" / "minutes.parquet",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
        df["season"] = df["game_date"].apply(_season_from_day)
        return df
    return None


def _load_labels_partition(root: Path, day: date) -> pd.DataFrame | None:
    ts = pd.Timestamp(day)
    season = _season_from_day(ts)
    iso = day.isoformat()
    labels_path = root / "gold" / "labels_minutes_v1" / f"season={season}" / f"game_date={iso}" / "labels.parquet"
    if not labels_path.exists():
        return None
    df = pd.read_parquet(labels_path)
    if df.empty:
        return None
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    df["season"] = season
    if "minutes_actual" not in df.columns:
        rename_candidates = [c for c in ("minutes", "actual_minutes") if c in df.columns]
        if rename_candidates:
            df.rename(columns={rename_candidates[0]: "minutes_actual"}, inplace=True)
    return df


def _resolve_minutes_center(df: pd.DataFrame) -> pd.Series:
    candidates = [
        "minutes_p50_cond",
        "minutes_p50",
        "minutes_pred_p50",
        "minutes_p50_pred",
    ]
    for col in candidates:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if not series.isna().all():
                return series
    raise RuntimeError("No usable minutes prediction column found (checked minutes_p50_cond/p50/pred_p50)")


def _resolve_status_series(df: pd.DataFrame) -> pd.Series:
    for col in STATUS_CANDIDATES:
        if col in df.columns:
            return df[col]
    return pd.Series("healthy", index=df.index, dtype=object)


def _resolve_starter(df: pd.DataFrame) -> pd.Series:
    candidates = ["starter_flag", "is_starter", "starter_flag_label", "is_starter_pred"]
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return pd.Series(0, index=df.index, dtype=int)


def _collect_residuals(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in _iter_days(start, end):
        proj = _load_proj_partition(root, day.date())
        labels = _load_labels_partition(root, day.date())
        if proj is None or labels is None:
            continue
        for col in ("game_id", "player_id", "team_id"):
            if col in proj.columns:
                proj[col] = pd.to_numeric(proj[col], errors="coerce")
            if col in labels.columns:
                labels[col] = pd.to_numeric(labels[col], errors="coerce")
        join_cols = [c for c in ("season", "game_id", "team_id", "player_id") if c in proj.columns and c in labels.columns]
        if len(join_cols) < 3:
            continue
        merged = proj.merge(labels, on=join_cols, how="inner", suffixes=("_pred", "_actual"))
        if merged.empty:
            continue
        if "minutes_actual" not in merged.columns:
            continue
        merged["minutes_actual"] = pd.to_numeric(merged["minutes_actual"], errors="coerce")
        merged["minutes_pred_center"] = _resolve_minutes_center(merged)
        merged["starter_flag"] = _resolve_starter(merged)
        status_series = _resolve_status_series(merged)
        merged["status_bucket"] = status_series.apply(status_bucket_from_raw)

        merged = merged.dropna(subset=["minutes_pred_center", "minutes_actual"])
        merged = merged[merged["status_bucket"].str.lower() != "out"]
        if merged.empty:
            continue

        merged["p50_bin_idx"] = minutes_bin_indices(
            merged["minutes_pred_center"].to_numpy(dtype=float), edges=MINUTES_P50_BIN_EDGES
        )
        merged["residual"] = merged["minutes_actual"] - merged["minutes_pred_center"]
        frames.append(merged[["starter_flag", "status_bucket", "p50_bin_idx", "residual"]])

    if not frames:
        return pd.DataFrame(columns=["starter_flag", "status_bucket", "p50_bin_idx", "residual"])
    return pd.concat(frames, ignore_index=True)


@app.command()
def main(
    split: Split = typer.Option(
        "all",
        "--split",
        case_sensitive=False,
        help="Split to evaluate: train|cal|val|all (default all date windows)",
    ),
    data_root: Optional[Path] = typer.Option(
        None,
        "--data-root",
        help="Data root (defaults to PROJECTIONS_DATA_ROOT or ./data)",
    ),
) -> None:
    split_norm: Split = split.lower()  # type: ignore[assignment]
    bundle = load_production_minutes_bundle()
    run_id = bundle.get("run_id") or bundle.get("meta", {}).get("run_id")
    if not run_id:
        run_dir = bundle.get("run_dir")
        if run_dir:
            run_id = Path(str(run_dir)).name
    run_id = str(run_id or "unknown_run")
    meta = bundle.get("meta", {})
    start, end = _window_from_meta(meta, split_norm)
    root = data_root or data_path()

    typer.echo(
        f"[minutes_residuals] run_id={run_id} split={split_norm} window={start.date()}→{end.date()} root={root}"
    )
    residual_df = _collect_residuals(root, start, end)
    if residual_df.empty:
        raise RuntimeError("No residual rows collected; ensure projections and labels exist for the window")

    grouped = residual_df.groupby(["starter_flag", "status_bucket", "p50_bin_idx"])
    buckets: list[dict[str, object]] = []
    for (starter_flag, status_bucket, bin_idx), group in grouped:
        residuals = group["residual"].to_numpy(dtype=float)
        buckets.append(
            {
                "starter_flag": int(starter_flag),
                "status_bucket": str(status_bucket),
                "p50_bin_idx": int(bin_idx),
                "p50_bin": list(bin_bounds(int(bin_idx), edges=MINUTES_P50_BIN_EDGES)),
                "count": int(residuals.size),
                "mean_r": float(np.mean(residuals)),
                "std_r": float(np.std(residuals, ddof=0)),
                "p10": float(np.quantile(residuals, 0.10)),
                "p50": float(np.quantile(residuals, 0.50)),
                "p90": float(np.quantile(residuals, 0.90)),
            }
        )

    payload = {
        "run_id": run_id,
        "split": split_norm,
        "date_window": {"start": start.isoformat(), "end": end.isoformat()},
        "bin_edges": list(MINUTES_P50_BIN_EDGES),
        "n_rows": int(len(residual_df)),
        "n_buckets": int(len(buckets)),
        "buckets": buckets,
    }

    out_root = root / "artifacts" / "minutes_v1" / "residuals"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{run_id}_minutes_residuals.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    typer.echo(
        f"[minutes_residuals] run_id={run_id} split={split_norm} rows={len(residual_df)} "
        f"buckets={len(buckets)} written={out_path}"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
