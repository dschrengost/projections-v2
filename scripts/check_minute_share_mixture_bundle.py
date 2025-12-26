#!/usr/bin/env python
"""Sanity check for minute share mixture bundle.

Validates:
1. Required files exist
2. Bundle loads successfully
3. predict_expected_minutes() works on sample data
4. Output statistics are reasonable

Example usage:
    uv run python scripts/check_minute_share_mixture_bundle.py \
        --bundle artifacts/minute_share_mixture/v0 \
        --features data/training/datasets/full_contract_v2/features.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projections.models.minute_share_mixture import (
    MixtureBundle,
    predict_expected_minutes,
)

app = typer.Typer(help=__doc__)

REQUIRED_FILES = [
    "models/state_classifier.joblib",
    "models/regressor_state_1.joblib",
    "models/regressor_state_2.joblib",
    "models/regressor_state_3.joblib",
    "models/regressor_state_4.joblib",
    "feature_columns.json",
    "train_meta.json",
    "imputer.joblib",
]


@app.command()
def main(
    bundle: Path = typer.Option(
        Path("artifacts/minute_share_mixture/v0"),
        "--bundle",
        help="Path to mixture bundle directory.",
    ),
    features: Path = typer.Option(
        Path("data/training/datasets/full_contract_v2/features.parquet"),
        "--features",
        help="Path to features parquet for testing predictions.",
    ),
    n_rows: int = typer.Option(
        50,
        "--n-rows",
        help="Number of rows to sample for prediction test.",
    ),
) -> None:
    """Check mixture bundle integrity and run prediction sanity check."""

    typer.echo(f"[check] Checking bundle: {bundle}")
    typer.echo("-" * 60)

    # Step 1: Check required files exist
    typer.echo("\n1. Checking required files...")
    missing = []
    for f in REQUIRED_FILES:
        path = bundle / f
        if path.exists():
            size = path.stat().st_size
            typer.echo(f"   ✓ {f} ({size:,} bytes)")
        else:
            typer.echo(f"   ✗ {f} MISSING")
            missing.append(f)

    if missing:
        typer.echo(f"\n❌ Bundle incomplete: {len(missing)} files missing")
        raise typer.Exit(1)

    typer.echo(f"   All {len(REQUIRED_FILES)} required files present")

    # Step 2: Load bundle
    typer.echo("\n2. Loading bundle...")
    try:
        mb = MixtureBundle.load(bundle)
        typer.echo(f"   ✓ Bundle loaded successfully")
        typer.echo(f"   - {len(mb.feature_columns)} feature columns")
        typer.echo(f"   - {len(mb.regressors)} state regressors (S1-S4)")
        typer.echo(f"   - State means: {mb.state_means}")
    except Exception as e:
        typer.echo(f"   ✗ Failed to load bundle: {e}")
        raise typer.Exit(1)

    # Step 3: Load features and run prediction
    typer.echo(f"\n3. Testing predict_expected_minutes on {n_rows} rows...")

    if not features.exists():
        typer.echo(f"   ⚠ Features file not found: {features}")
        typer.echo("   Skipping prediction test")
        typer.echo("\n✓ Bundle structure validated (prediction test skipped)")
        return

    try:
        df = pd.read_parquet(features)
        typer.echo(f"   Loaded {len(df):,} rows from {features.name}")

        # Sample rows
        if len(df) > n_rows:
            df_sample = df.sample(n=n_rows, random_state=42)
        else:
            df_sample = df

        # Check feature columns
        missing_cols = [c for c in mb.feature_columns if c not in df_sample.columns]
        if missing_cols:
            typer.echo(f"   ⚠ Missing {len(missing_cols)} feature columns: {missing_cols[:5]}...")
            raise typer.Exit(1)

        X = df_sample[mb.feature_columns]
        typer.echo(f"   Feature matrix: {X.shape}")

        # Run prediction
        preds = predict_expected_minutes(X, mb)

        # Compute stats
        nan_frac = np.isnan(preds).mean()
        mean = np.nanmean(preds)
        p50 = np.nanpercentile(preds, 50)
        p95 = np.nanpercentile(preds, 95)
        max_val = np.nanmax(preds)

        typer.echo(f"\n   📊 Prediction Summary:")
        typer.echo(f"      nan_frac: {nan_frac:.4f}")
        typer.echo(f"      mean:     {mean:.2f} min")
        typer.echo(f"      p50:      {p50:.2f} min")
        typer.echo(f"      p95:      {p95:.2f} min")
        typer.echo(f"      max:      {max_val:.2f} min")

        # Sanity checks
        issues = []
        if nan_frac > 0.01:
            issues.append(f"High NaN fraction: {nan_frac:.2%}")
        if mean < 5 or mean > 40:
            issues.append(f"Mean outside expected range [5,40]: {mean:.2f}")
        if max_val > 48:
            issues.append(f"Max exceeds 48: {max_val:.2f}")

        if issues:
            typer.echo("\n   ⚠ Warnings:")
            for issue in issues:
                typer.echo(f"      - {issue}")

    except Exception as e:
        typer.echo(f"   ✗ Prediction failed: {e}")
        raise typer.Exit(1)

    typer.echo("\n" + "=" * 60)
    typer.echo("✓ Bundle sanity check PASSED")
    typer.echo("=" * 60)


if __name__ == "__main__":
    app()
