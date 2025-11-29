from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

CATEGORICAL_FEATURES_DEFAULT = {"track_role_cluster", "track_role_is_low_minutes"}


def build_fpts_design_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    categorical_cols: Iterable[str] | None = None,
    fill_missing_with_zero: bool = False,
    warn_missing: bool = False,
) -> pd.DataFrame:
    """
    Prepare an FPTS feature matrix consistent with training.

    - Ensures all feature_cols exist (optionally filling missing with zeros).
    - Casts numeric columns with to_numeric then fillna(0).
    - Categorical columns: fillna(-1) and cast to int.
    """

    cat_cols = set(categorical_cols or CATEGORICAL_FEATURES_DEFAULT)
    working = df.copy()

    missing = [c for c in feature_cols if c not in working.columns]
    if missing:
        if not fill_missing_with_zero:
            raise RuntimeError(f"Missing feature columns: {missing}")
        if warn_missing:
            import typer

            typer.echo(f"[fpts_features] warning: filling missing feature columns with 0: {missing}")
        for col in missing:
            working[col] = 0.0

    present_cols = [c for c in feature_cols if c in working.columns]
    design = working[present_cols].copy()

    cat_present = [c for c in present_cols if c in cat_cols]
    num_present = [c for c in present_cols if c not in cat_cols]

    for col in num_present:
        design[col] = pd.to_numeric(design[col], errors="coerce").fillna(0.0)
    for col in cat_present:
        design[col] = pd.to_numeric(design[col], errors="coerce").fillna(-1).astype(int)

    design = design.fillna(0.0)
    return design


__all__ = ["build_fpts_design_matrix", "CATEGORICAL_FEATURES_DEFAULT"]
