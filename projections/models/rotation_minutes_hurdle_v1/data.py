from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_FEATURE_COLS = {"game_id", "team_id", "player_id", "game_date", "tip_ts", "status", "injury_as_of_ts"}
REQUIRED_LABEL_COLS = {"game_id", "team_id", "player_id", "game_date", "minutes"}


def load_labeled_frame(dataset_dir: Path) -> pd.DataFrame:
    """Load (features, labels) parquet pair and return a merged dataframe.

    The training datasets currently duplicate `minutes`/`starter_flag_label` in features.parquet;
    we explicitly drop those from the feature frame to avoid accidental leakage.
    """

    features_path = dataset_dir / "features.parquet"
    labels_path = dataset_dir / "labels.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features.parquet: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.parquet: {labels_path}")

    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)

    missing_feat = REQUIRED_FEATURE_COLS.difference(features.columns)
    if missing_feat:
        raise ValueError(f"features.parquet missing required columns: {sorted(missing_feat)}")

    missing_lab = REQUIRED_LABEL_COLS.difference(labels.columns)
    if missing_lab:
        raise ValueError(f"labels.parquet missing required columns: {sorted(missing_lab)}")

    features = features.drop(columns=[c for c in ("minutes", "starter_flag_label") if c in features.columns])
    labels = labels[list(REQUIRED_LABEL_COLS)].copy()

    merged = features.merge(
        labels,
        on=["game_id", "team_id", "player_id", "game_date"],
        how="inner",
        validate="one_to_one",
    )
    return merged


def add_has_injury_row(df: pd.DataFrame) -> pd.DataFrame:
    """Add `has_injury_row` per RMH_v1 spec.

    has_injury_row = (injury_as_of_ts not null) OR (status != 'Ava')
    """

    out = df.copy()
    # NOTE: status is a string dtype in this dataset; treat missing as non-Ava.
    status = out["status"].astype(str)
    out["has_injury_row"] = ((out["injury_as_of_ts"].notna()) | (status != "Ava")).astype("int8")
    return out


def build_y_play(minutes: pd.Series, *, play_threshold: float) -> np.ndarray:
    return (minutes.astype(float) >= float(play_threshold)).astype(np.float32).to_numpy()


def compute_recency_weights(
    tip_ts: pd.Series,
    *,
    half_life_days: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute exp(-Δt/half_life) weights where Δt is measured from the most recent tip_ts."""

    tips = pd.to_datetime(tip_ts, utc=True)
    ref = tips.max()
    delta_days = (ref - tips).dt.total_seconds() / 86400.0
    delta_days = delta_days.clip(lower=0.0).astype(float)
    # Ensure we return a plain ndarray (pandas ufuncs would otherwise return Series).
    delta_arr = delta_days.to_numpy(dtype=np.float64, copy=False)
    weights = np.exp(-delta_arr / float(half_life_days)).astype(np.float32, copy=False)
    summary = {
        "ref_tip_ts": ref.isoformat(),
        "delta_days_min": float(delta_arr.min()) if len(delta_arr) else float("nan"),
        "delta_days_p50": float(np.percentile(delta_arr, 50)) if len(delta_arr) else float("nan"),
        "delta_days_p90": float(np.percentile(delta_arr, 90)) if len(delta_arr) else float("nan"),
        "weight_min": float(weights.min()) if len(weights) else float("nan"),
        "weight_p50": float(np.percentile(weights, 50)) if len(weights) else float("nan"),
        "weight_p90": float(np.percentile(weights, 90)) if len(weights) else float("nan"),
        "weight_max": float(weights.max()) if len(weights) else float("nan"),
    }
    return weights, summary


@dataclass(frozen=True)
class DatasetSummary:
    n_rows: int
    played_rate: float
    status_counts: dict[str, int]


def dataset_summary(df: pd.DataFrame, *, y_play: np.ndarray) -> DatasetSummary:
    status_counts = df["status"].astype(str).value_counts().to_dict()
    return DatasetSummary(
        n_rows=int(len(df)),
        played_rate=float(np.mean(y_play)) if len(y_play) else float("nan"),
        status_counts={str(k): int(v) for k, v in status_counts.items()},
    )


__all__ = [
    "DatasetSummary",
    "add_has_injury_row",
    "build_y_play",
    "compute_recency_weights",
    "dataset_summary",
    "load_labeled_frame",
]
