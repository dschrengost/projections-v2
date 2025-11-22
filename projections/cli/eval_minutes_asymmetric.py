"""Evaluate minutes coverage with a conditional mask and asymmetric tail scaling."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import typer

from projections.minutes_v1.calibration.asymmetric_scaling import (
    AsymmetricK,
    apply_asymmetric_k,
    compute_coverage,
    fit_global_asymmetric_k,
)
from projections.minutes_v1.datasets import KEY_COLUMNS, deduplicate_latest
from projections.models import minutes_lgbm as ml


app = typer.Typer(help=__doc__)


def _load_features(features_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(features_root.rglob("features.parquet")):
        frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"No features.parquet files under {features_root}")
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    df["feature_as_of_ts"] = pd.to_datetime(df["feature_as_of_ts"])
    df["starter_flag"] = (
        df.get("is_projected_starter", False).astype(bool) | df.get("is_confirmed_starter", False).astype(bool)
    )
    return df


def _conditional_mask(df: pd.DataFrame) -> pd.Series:
    """Return mask for conditional minutes evaluation (in-rotation, not OUT)."""

    status_series = df.get("status")
    if status_series is None:
        status_upper = pd.Series(["UNK"] * len(df), index=df.index)
    else:
        status_upper = status_series.astype(str).str.upper()

    bad_prefixes = ("OUT", "INACTIVE", "DNP", "G-LEAGUE", "G_LEAGUE", "TWO-WAY", "TWO_WAY")
    keep_status = ~status_upper.str.startswith(bad_prefixes)

    # Simple rotation mask: starter OR any recent playing time signal.
    starter = df.get("starter_flag", False).astype(bool)
    recent_start = df.get("recent_start_pct_10", 0).fillna(0) > 0
    recent_minutes = df.get("min_last5", 0).fillna(0) > 0
    rotation = starter | recent_start | recent_minutes

    return keep_status & rotation


def _score_slice(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    feature_cols: list[str] = bundle["feature_columns"]
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Feature frame missing columns: {', '.join(sorted(missing))}")

    df = deduplicate_latest(df, key_cols=KEY_COLUMNS, order_cols=["feature_as_of_ts"])

    preds = ml.modeling.predict_quantiles(bundle["quantiles"], df[feature_cols])
    p10_raw = np.minimum(preds[0.1], preds[0.5])
    p90_raw = np.maximum(preds[0.9], preds[0.5])
    cal = bundle["calibrator"]
    p10_cal, p90_cal = cal.calibrate(p10_raw, p90_raw)

    working = df.copy()
    working["minutes_p10"] = p10_cal
    working["minutes_p50"] = preds[0.5]
    working["minutes_p90"] = p90_cal
    working["p10_pred"] = p10_cal
    working["p50_pred"] = preds[0.5]
    working["p90_pred"] = p90_cal
    working = ml.apply_conformal(
        working,
        bundle["bucket_offsets"],
        mode=bundle["conformal_mode"],
        bucket_mode=bundle["bucket_mode"],
    )
    working["minutes_p10"] = working["p10_adj"]
    working["minutes_p50"] = working["p50_adj"]
    working["minutes_p90"] = working["p90_adj"]
    return working


def _coverage(df: pd.DataFrame) -> tuple[float, float]:
    return compute_coverage(df["minutes"].to_numpy(dtype=float), df["minutes_p10"].to_numpy(), df["minutes_p90"].to_numpy())


def _stratified(df: pd.DataFrame, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    df = df.copy()
    df["starter"] = df["starter_flag"].astype(bool)
    spread = np.abs(df.get("spread_home", np.nan))
    df["spread_bin"] = pd.cut(spread, bins=[-np.inf, 5, 10, np.inf], labels=["<5", "5-10", ">10"])
    status_series = df.get("status", pd.Series(["UNK"] * len(df), index=df.index)).astype(str).str.upper()

    def add_row(key: str, subset: pd.DataFrame) -> None:
        if subset.empty:
            return
        cov10, cov90 = _coverage(subset)
        rows.append({"window": label, "group": key, "p10": cov10, "p90": cov90, "n": len(subset)})

    add_row("starter", df[df["starter"]])
    add_row("bench", df[~df["starter"]])
    for b in df["spread_bin"].dropna().unique():
        add_row(f"spread_{b}", df[df["spread_bin"] == b])
    for st in status_series.dropna().unique():
        add_row(f"status_{st}", df[status_series == st])
    return rows


def _apply_asymmetric(df: pd.DataFrame, k: AsymmetricK) -> pd.DataFrame:
    q10p, q90p = apply_asymmetric_k(
        df["minutes_p10"].to_numpy(dtype=float),
        df["minutes_p50"].to_numpy(dtype=float),
        df["minutes_p90"].to_numpy(dtype=float),
        k,
    )
    df = df.copy()
    df["minutes_p10_asym"] = q10p
    df["minutes_p90_asym"] = q90p
    return df


@app.command()
def main(
    run_id: str = typer.Option(..., help="Trained run_id to evaluate."),
    features_root: Path = typer.Option(
        Path("/home/daniel/projections-data/gold/features_minutes_v1"),
        help="Root containing season=*/month=*/features.parquet",
    ),
    artifact_root: Path = typer.Option(Path("artifacts/minutes_v1"), help="Where run artifacts live."),
    train_start: str = typer.Option("2022-10-01"),
    train_end: str = typer.Option("2025-02-01"),
    cal_start: str = typer.Option("2025-02-02"),
    cal_end: str = typer.Option("2025-06-30"),
    val_start: str = typer.Option("2025-10-01"),
    val_end: str = typer.Option("2025-11-20"),
) -> None:
    run_dir = artifact_root / run_id
    bundle_path = run_dir / "lgbm_quantiles.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found at {bundle_path}")
    bundle = ml.joblib.load(bundle_path)

    features = _load_features(features_root)

    def slice_window(start: str, end: str) -> pd.DataFrame:
        mask = (features["game_date"] >= pd.to_datetime(start)) & (features["game_date"] <= pd.to_datetime(end))
        return features.loc[mask].copy()

    train_df = slice_window(train_start, train_end)
    cal_df = slice_window(cal_start, cal_end)
    val_df = slice_window(val_start, val_end)

    scored_train = _score_slice(train_df, bundle)
    scored_cal = _score_slice(cal_df, bundle)
    scored_val = _score_slice(val_df, bundle)

    mask_train = _conditional_mask(scored_train)
    mask_cal = _conditional_mask(scored_cal)
    mask_val = _conditional_mask(scored_val)

    def cover(df: pd.DataFrame) -> dict[str, object]:
        c10, c90 = _coverage(df)
        return {"p10": c10, "p90": c90, "n": len(df)}
    summary = {
        "base": {
            "train": cover(scored_train),
            "cal": cover(scored_cal),
            "val": cover(scored_val),
        },
        "conditional": {
            "train": cover(scored_train[mask_train]),
            "cal": cover(scored_cal[mask_cal]),
            "val": cover(scored_val[mask_val]),
        },
    }

    # Stratified coverage on conditional pool
    strata_rows: list[dict[str, object]] = []
    strata_rows += _stratified(scored_cal[mask_cal], "cal")
    strata_rows += _stratified(scored_val[mask_val], "val")

    # Fit asymmetric k on conditional cal
    k_opt = fit_global_asymmetric_k(scored_cal[mask_cal], minutes_col="minutes", q10_col="minutes_p10", q50_col="minutes_p50", q90_col="minutes_p90")
    scaled = {
        "k": asdict(k_opt),
        "cal": cover(_apply_asymmetric(scored_cal[mask_cal], k_opt).assign(minutes_p10=lambda d: d["minutes_p10_asym"], minutes_p90=lambda d: d["minutes_p90_asym"])),
        "val": cover(_apply_asymmetric(scored_val[mask_val], k_opt).assign(minutes_p10=lambda d: d["minutes_p10_asym"], minutes_p90=lambda d: d["minutes_p90_asym"])),
    }

    output = {
        "run_id": run_id,
        "windows": {
            "train": {"start": train_start, "end": train_end},
            "cal": {"start": cal_start, "end": cal_end},
            "val": {"start": val_start, "end": val_end},
        },
        "coverage": summary,
        "strata": strata_rows,
        "asymmetric": scaled,
    }
    typer.echo(json.dumps(output, indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()
