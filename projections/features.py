"""Feature engineering utilities."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


def add_rolling_features(
    df: pd.DataFrame,
    *,
    group_cols: Iterable[str],
    target_col: str,
    windows: Iterable[int] = (3, 5, 10),
) -> pd.DataFrame:
    """Create rolling aggregation features for each player/team bucket."""

    feat_df = df.sort_values(["game_date"]).copy()
    for window in windows:
        rolling = (
            feat_df.groupby(list(group_cols))[target_col]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=list(range(len(group_cols))), drop=True)
        )
        feat_df[f"{target_col}_rolling_{window}"] = rolling
    return feat_df


def build_feature_target_split(
    df: pd.DataFrame, *, target_col: str, drop_cols: Iterable[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and target vector y."""

    target = df[target_col]
    features = df.drop(columns=list(drop_cols))
    return features, target


def stratified_split(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and targets into train/test partitions."""

    return train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=None
    )
