from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ResidualBucket:
    name: str
    min_minutes: float
    max_minutes: float | None  # None = open-ended
    is_starter: int | None  # 1, 0, or None for both
    sigma: float  # scale parameter
    nu: int  # Student-t degrees of freedom
    n: int  # number of samples used


@dataclass
class ResidualModel:
    buckets: List[ResidualBucket]
    sigma_default: float
    nu_default: int


def default_buckets() -> list[dict]:
    """
    Default residual buckets keyed by expected minutes and starter flag.

    Ranges are half-open on the right: [min_minutes, max_minutes).
    """

    return [
        {"name": "starter_32_plus", "min_minutes": 32.0, "max_minutes": None, "is_starter": 1},
        {"name": "starter_24_32", "min_minutes": 24.0, "max_minutes": 32.0, "is_starter": 1},
        {"name": "starter_16_24", "min_minutes": 16.0, "max_minutes": 24.0, "is_starter": 1},
        {"name": "starter_under_16", "min_minutes": 0.0, "max_minutes": 16.0, "is_starter": 1},
        {"name": "bench_28_plus", "min_minutes": 28.0, "max_minutes": None, "is_starter": 0},
        {"name": "bench_20_28", "min_minutes": 20.0, "max_minutes": 28.0, "is_starter": 0},
        {"name": "bench_12_20", "min_minutes": 12.0, "max_minutes": 20.0, "is_starter": 0},
        {"name": "bench_under_12", "min_minutes": 0.0, "max_minutes": 12.0, "is_starter": 0},
    ]


def _coerce_minutes(row: pd.Series) -> float | None:
    for key in ("minutes_pred_p50", "minutes_p50", "minutes_actual"):
        if key in row:
            value = row.get(key)
            if pd.isna(value):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def assign_bucket(row: pd.Series, buckets: list[dict]) -> str | None:
    """
    Given a row with minutes_pred_p50 (or minutes_p50) and is_starter,
    return bucket name or None if no bucket matches.
    """

    minutes_val = _coerce_minutes(row)
    if minutes_val is None:
        return None

    starter_value = row.get("is_starter")
    if pd.isna(starter_value):
        starter_flag: int | None = None
    else:
        try:
            starter_flag = int(starter_value)
        except (TypeError, ValueError):
            starter_flag = None

    for bucket in buckets:
        bucket_starter = bucket.get("is_starter")
        if bucket_starter is not None and starter_flag is not None and int(bucket_starter) != int(starter_flag):
            continue
        if bucket_starter is not None and starter_flag is None:
            continue
        if minutes_val < float(bucket["min_minutes"]):
            continue
        max_minutes = bucket.get("max_minutes")
        if max_minutes is not None and minutes_val >= float(max_minutes):
            continue
        return str(bucket["name"])

    return None


def fit_residual_model(
    df: pd.DataFrame,
    fpts_pred_col: str = "dk_fpts_pred",
    fpts_label_col: str = "dk_fpts",
    minutes_col: str = "minutes_pred_p50",
    is_starter_col: str = "is_starter",
    min_rows_per_bucket: int = 200,
    nu_default: int = 5,
) -> ResidualModel:
    """
    Fit a residual scale model split by expected minutes and starter status.
    """

    working = df.copy()

    if minutes_col not in working.columns:
        fallback_col = None
        for candidate in ("minutes_actual", "minutes_p50"):
            if candidate in working.columns:
                fallback_col = candidate
                break
        if fallback_col:
            logger.warning("[sim_residuals] minutes column %s missing; using %s", minutes_col, fallback_col)
            working[minutes_col] = working[fallback_col]
        else:
            logger.warning("[sim_residuals] missing minutes column; defaulting to NaN")
            working[minutes_col] = np.nan
    working[minutes_col] = pd.to_numeric(working[minutes_col], errors="coerce")

    if is_starter_col not in working.columns:
        logger.warning("[sim_residuals] is_starter missing; treating all players as bench (0)")
        working[is_starter_col] = 0
    working[is_starter_col] = working[is_starter_col].apply(
        lambda v: None if pd.isna(v) else int(v)
    )

    if fpts_label_col not in working.columns:
        alt = next((c for c in ("dk_fpts_actual", "fpts_dk_label") if c in working.columns), None)
        if alt is None:
            raise KeyError(f"Missing label column {fpts_label_col}")
        logger.warning("[sim_residuals] label column %s missing; using %s", fpts_label_col, alt)
        fpts_label_col = alt

    if fpts_pred_col not in working.columns:
        raise KeyError(f"Missing prediction column {fpts_pred_col}")

    labels = pd.to_numeric(working[fpts_label_col], errors="coerce")
    preds = pd.to_numeric(working[fpts_pred_col], errors="coerce")
    residuals = labels - preds
    working["_residual"] = residuals

    buckets: list[ResidualBucket] = []
    bucket_defs = default_buckets()
    working["_bucket"] = working.apply(lambda row: assign_bucket(row, bucket_defs), axis=1)

    for bucket_def in bucket_defs:
        mask = working["_bucket"] == bucket_def["name"]
        bucket_resid = working.loc[mask, "_residual"].dropna()
        n_rows = int(bucket_resid.shape[0])
        if n_rows < min_rows_per_bucket:
            continue
        sigma = float(np.std(bucket_resid.values))
        buckets.append(
            ResidualBucket(
                name=bucket_def["name"],
                min_minutes=float(bucket_def["min_minutes"]),
                max_minutes=float(bucket_def["max_minutes"]) if bucket_def["max_minutes"] is not None else None,
                is_starter=bucket_def["is_starter"],
                sigma=sigma,
                nu=nu_default,
                n=n_rows,
            )
        )

    sigma_default = float(np.std(working["_residual"].dropna().values)) if working["_residual"].notna().any() else 0.0
    return ResidualModel(buckets=buckets, sigma_default=sigma_default, nu_default=nu_default)


def to_json(model: ResidualModel) -> dict:
    """Return a JSON-serializable dict."""

    return {
        "buckets": [
            {
                "name": bucket.name,
                "min_minutes": bucket.min_minutes,
                "max_minutes": bucket.max_minutes,
                "is_starter": bucket.is_starter,
                "sigma": bucket.sigma,
                "nu": bucket.nu,
                "n": bucket.n,
            }
            for bucket in model.buckets
        ],
        "sigma_default": model.sigma_default,
        "nu_default": model.nu_default,
    }


def from_json(data: dict) -> ResidualModel:
    """Inverse of to_json."""

    buckets = [
        ResidualBucket(
            name=item["name"],
            min_minutes=float(item["min_minutes"]),
            max_minutes=float(item["max_minutes"]) if item.get("max_minutes") is not None else None,
            is_starter=item.get("is_starter"),
            sigma=float(item["sigma"]),
            nu=int(item.get("nu", 0)),
            n=int(item.get("n", 0)),
        )
        for item in data.get("buckets", [])
    ]
    return ResidualModel(
        buckets=buckets,
        sigma_default=float(data.get("sigma_default", 0.0)),
        nu_default=int(data.get("nu_default", 0)),
    )


__all__ = [
    "ResidualBucket",
    "ResidualModel",
    "assign_bucket",
    "default_buckets",
    "fit_residual_model",
    "to_json",
    "from_json",
]
