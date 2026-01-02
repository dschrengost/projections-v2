#!/usr/bin/env python3
"""
Calibrate minutes-rates covariance for sim_v2.

This script measures the empirical correlation between minutes residuals and
per-minute rate residuals to determine if/how to model their joint distribution.

Uses gold/rates_training_base which contains:
- minutes_actual, minutes_pred_p50 (observed vs predicted minutes)
- fga2_per_min, ast_per_min, etc. (actual per-minute rates as labels)
- season_* priors (used as rate predictions baseline)

Output: Correlation matrix and recommendations for sim noise model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.paths import data_path

app = typer.Typer()

# Rate targets we care about
RATE_TARGETS = [
    "fga2_per_min",
    "fga3_per_min",
    "fta_per_min",
    "ast_per_min",
    "tov_per_min",
    "oreb_per_min",
    "dreb_per_min",
    "stl_per_min",
    "blk_per_min",
]

# Corresponding season priors (naive baseline for rate prediction)
RATE_PRIORS = {
    "fga2_per_min": "season_fga2_per_min",
    "fga3_per_min": "season_3pa_per_min",  # Note: 3pa not fga3
    "fta_per_min": "season_fta_per_min",
    "ast_per_min": "season_ast_per_min",
    "tov_per_min": "season_tov_per_min",
    "oreb_per_min": "season_oreb_per_min",
    "dreb_per_min": "season_dreb_per_min",
    "stl_per_min": "season_stl_per_min",
    "blk_per_min": "season_blk_per_min",
}


@dataclass
class CovarianceResult:
    """Results from covariance calibration."""

    n_samples: int
    minutes_resid_std: float

    # Correlations: minutes_resid vs rate_resid
    corr_by_rate: dict[str, float] = field(default_factory=dict)

    # By starter/bench bucket
    corr_by_rate_starter: dict[str, float] = field(default_factory=dict)
    corr_by_rate_bench: dict[str, float] = field(default_factory=dict)

    # By minutes bucket
    corr_by_rate_high_minutes: dict[str, float] = field(default_factory=dict)
    corr_by_rate_low_minutes: dict[str, float] = field(default_factory=dict)

    # Rate residual stats
    rate_resid_std: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "minutes_resid_std": self.minutes_resid_std,
            "corr_by_rate": self.corr_by_rate,
            "corr_by_rate_starter": self.corr_by_rate_starter,
            "corr_by_rate_bench": self.corr_by_rate_bench,
            "corr_by_rate_high_minutes": self.corr_by_rate_high_minutes,
            "corr_by_rate_low_minutes": self.corr_by_rate_low_minutes,
            "rate_resid_std": self.rate_resid_std,
        }


def load_rates_training_base(
    data_root: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load rates training base for date range."""

    base_path = data_root / "gold" / "rates_training_base"
    if not base_path.exists():
        raise FileNotFoundError(f"Rates training base not found at {base_path}")

    dfs = []
    for season_dir in sorted(base_path.glob("season=*")):
        for date_dir in sorted(season_dir.glob("game_date=*")):
            try:
                date_str = date_dir.name.split("=")[1]
                game_date = pd.Timestamp(date_str).normalize()
            except Exception:
                continue

            if game_date < start_date or game_date > end_date:
                continue

            parquet_path = date_dir / "rates_training_base.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                df["game_date"] = game_date
                dfs.append(df)

    if not dfs:
        raise ValueError(f"No data found in range {start_date.date()} to {end_date.date()}")

    return pd.concat(dfs, ignore_index=True)


def compute_covariance(
    df: pd.DataFrame,
    min_minutes: float = 8.0,
) -> CovarianceResult:
    """Compute minutes-rates covariance from training data."""

    # Filter to players with enough minutes for stable rate estimates
    df = df[df["minutes_actual"] >= min_minutes].copy()

    # Compute minutes residual
    if "minutes_pred_p50" not in df.columns:
        typer.echo("[warn] minutes_pred_p50 not found, using season prior as baseline")
        # Fallback: use rolling mean as proxy
        if "roll_mean_10" in df.columns:
            df["minutes_pred"] = df["roll_mean_10"]
        else:
            df["minutes_pred"] = df.groupby("player_id")["minutes_actual"].transform("mean")
    else:
        df["minutes_pred"] = df["minutes_pred_p50"]

    df["minutes_resid"] = df["minutes_actual"] - df["minutes_pred"]

    # Compute rate residuals (actual - season prior baseline)
    rate_resids = {}
    for target in RATE_TARGETS:
        if target not in df.columns:
            continue
        prior_col = RATE_PRIORS.get(target)
        if prior_col and prior_col in df.columns:
            df[f"{target}_resid"] = df[target] - df[prior_col]
            rate_resids[target] = f"{target}_resid"
        else:
            # No prior available, skip
            continue

    if not rate_resids:
        raise ValueError("No rate residuals could be computed")

    # Overall correlations
    result = CovarianceResult(
        n_samples=len(df),
        minutes_resid_std=float(df["minutes_resid"].std()),
    )

    for target, resid_col in rate_resids.items():
        valid = df[["minutes_resid", resid_col]].dropna()
        if len(valid) < 100:
            continue

        corr = valid["minutes_resid"].corr(valid[resid_col])
        result.corr_by_rate[target] = round(float(corr), 4)
        result.rate_resid_std[target] = round(float(valid[resid_col].std()), 6)

    # By starter/bench
    if "is_starter" in df.columns:
        starters = df[df["is_starter"] == 1]
        bench = df[df["is_starter"] == 0]

        for target, resid_col in rate_resids.items():
            # Starters
            valid_s = starters[["minutes_resid", resid_col]].dropna()
            if len(valid_s) >= 100:
                result.corr_by_rate_starter[target] = round(
                    float(valid_s["minutes_resid"].corr(valid_s[resid_col])), 4
                )

            # Bench
            valid_b = bench[["minutes_resid", resid_col]].dropna()
            if len(valid_b) >= 100:
                result.corr_by_rate_bench[target] = round(
                    float(valid_b["minutes_resid"].corr(valid_b[resid_col])), 4
                )

    # By minutes bucket (high vs low)
    median_minutes = df["minutes_pred"].median()
    high_min = df[df["minutes_pred"] >= median_minutes]
    low_min = df[df["minutes_pred"] < median_minutes]

    for target, resid_col in rate_resids.items():
        valid_h = high_min[["minutes_resid", resid_col]].dropna()
        if len(valid_h) >= 100:
            result.corr_by_rate_high_minutes[target] = round(
                float(valid_h["minutes_resid"].corr(valid_h[resid_col])), 4
            )

        valid_l = low_min[["minutes_resid", resid_col]].dropna()
        if len(valid_l) >= 100:
            result.corr_by_rate_low_minutes[target] = round(
                float(valid_l["minutes_resid"].corr(valid_l[resid_col])), 4
            )

    return result


def print_report(result: CovarianceResult) -> None:
    """Print human-readable report."""

    print("\n" + "=" * 70)
    print("MINUTES-RATES COVARIANCE CALIBRATION REPORT")
    print("=" * 70)
    print(f"\nSamples: {result.n_samples:,}")
    print(f"Minutes residual std: {result.minutes_resid_std:.2f}")

    print("\n" + "-" * 70)
    print("OVERALL CORRELATIONS: Corr(minutes_resid, rate_resid)")
    print("-" * 70)
    print(f"{'Target':<20} {'Corr':>10} {'Rate Resid Std':>15}")
    print("-" * 50)

    for target in RATE_TARGETS:
        if target in result.corr_by_rate:
            corr = result.corr_by_rate[target]
            std = result.rate_resid_std.get(target, 0)
            flag = " ***" if abs(corr) > 0.15 else " *" if abs(corr) > 0.10 else ""
            print(f"{target:<20} {corr:>10.4f} {std:>15.6f}{flag}")

    print("\n" + "-" * 70)
    print("BY STARTER STATUS")
    print("-" * 70)
    print(f"{'Target':<20} {'Starter':>10} {'Bench':>10}")
    print("-" * 50)

    for target in RATE_TARGETS:
        s = result.corr_by_rate_starter.get(target, float("nan"))
        b = result.corr_by_rate_bench.get(target, float("nan"))
        if not (np.isnan(s) and np.isnan(b)):
            print(f"{target:<20} {s:>10.4f} {b:>10.4f}")

    print("\n" + "-" * 70)
    print("BY MINUTES BUCKET (high vs low predicted minutes)")
    print("-" * 70)
    print(f"{'Target':<20} {'High Min':>10} {'Low Min':>10}")
    print("-" * 50)

    for target in RATE_TARGETS:
        h = result.corr_by_rate_high_minutes.get(target, float("nan"))
        l = result.corr_by_rate_low_minutes.get(target, float("nan"))
        if not (np.isnan(h) and np.isnan(l)):
            print(f"{target:<20} {h:>10.4f} {l:>10.4f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Summarize
    avg_corr = np.mean([v for v in result.corr_by_rate.values() if np.isfinite(v)])
    max_corr = max([abs(v) for v in result.corr_by_rate.values() if np.isfinite(v)], default=0)

    print(f"\nAverage |correlation|: {abs(avg_corr):.4f}")
    print(f"Max |correlation|: {max_corr:.4f}")

    if max_corr < 0.05:
        print("\n→ RECOMMENDATION: Correlations are negligible (<0.05).")
        print("  Minutes and rates can be treated as independent in the sim.")
    elif max_corr < 0.10:
        print("\n→ RECOMMENDATION: Correlations are small (0.05-0.10).")
        print("  Probably not worth modeling, but could add minor improvement.")
    elif max_corr < 0.20:
        print("\n→ RECOMMENDATION: Correlations are moderate (0.10-0.20).")
        print("  Consider adding a shared latent factor or conditional adjustment.")
    else:
        print("\n→ RECOMMENDATION: Correlations are significant (>0.20).")
        print("  Should definitely model minutes-rates covariance in sim.")


@app.command()
def main(
    start_date: str = typer.Option("2024-10-01", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option("2025-04-01", help="End date (YYYY-MM-DD)"),
    data_root: Optional[Path] = typer.Option(None, help="Data root path"),
    output_json: Optional[Path] = typer.Option(None, help="Output JSON path"),
    min_minutes: float = typer.Option(8.0, help="Min actual minutes for inclusion"),
) -> None:
    """Calibrate minutes-rates covariance from historical data."""

    root = data_root or data_path()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    typer.echo(f"[calibrate] Loading rates_training_base from {start.date()} to {end.date()}")
    df = load_rates_training_base(root, start, end)
    typer.echo(f"[calibrate] Loaded {len(df):,} rows")

    result = compute_covariance(df, min_minutes=min_minutes)
    print_report(result)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result.to_dict(), indent=2))
        typer.echo(f"\n[calibrate] Wrote results to {output_json}")


if __name__ == "__main__":
    app()
