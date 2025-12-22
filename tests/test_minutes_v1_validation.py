"""Validation guardrail tests for Minutes V1."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from projections.minutes_v1.validation import (
    hash_season_labels,
    reconciliation_sanity_check,
    sample_anti_leak_check,
    validate_label_hashes,
)


def test_sample_anti_leak_check_detects_violations():
    base_ts = pd.Timestamp("2024-11-01T00:00:00Z")
    df = pd.DataFrame(
        {
            "tip_ts": [base_ts + timedelta(hours=i % 24) for i in range(1200)],
        }
    )
    df["feature_as_of_ts"] = df["tip_ts"] - pd.Timedelta(minutes=30)
    df["injury_as_of_ts"] = df["tip_ts"] - pd.Timedelta(hours=2)
    df["odds_as_of_ts"] = df["tip_ts"] - pd.Timedelta(hours=1)

    sample_anti_leak_check(df, sample_size=1000)

    df.loc[0, "feature_as_of_ts"] = df.loc[0, "tip_ts"] + pd.Timedelta(minutes=5)
    with pytest.raises(AssertionError):
        sample_anti_leak_check(df, sample_size=1000)


def test_label_hash_validation_catches_drift(tmp_path: Path):
    season_dir = tmp_path / "season=2024-25"
    season_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "game_id": [1],
            "player_id": [10],
            "minutes": [30.0],
            "starter_flag": [1],
            "team_id": [100],
            "season": ["2024-25"],
            "game_date": ["2024-10-21"],
            "source": ["test"],
        }
    )
    df.to_parquet(season_dir / "boxscore_labels.parquet", index=False)

    hashes = hash_season_labels(tmp_path)
    validate_label_hashes(tmp_path, hashes)

    df.loc[0, "minutes"] = 33.0
    df.to_parquet(season_dir / "boxscore_labels.parquet", index=False)
    with pytest.raises(AssertionError):
        validate_label_hashes(tmp_path, hashes)


def test_reconciliation_sanity_check_runs_clean():
    df = pd.DataFrame(
        {
            "game_id": [1] * 8 + [2] * 8,
            "team_id": [10] * 8 + [20] * 8,
            "player_id": list(range(16)),
            "p50": [40, 38, 36, 34, 28, 24, 22, 18, 39, 37, 35, 33, 27, 25, 23, 21],
            "starter_prev_game_asof": [1, 1, 1, 1, 0, 0, 0, 0] * 2,
            "ramp_flag": [0, 0, 0, 0, 0, 0, 0, 1] * 2,
            "quantile_width": [6, 5, 7, 4, 8, 9, 7, 10] * 2,
            "blowout_index": [1.0] * 16,
            # Construct a reconciled set that exactly sums to 240 and respects caps.
            "minutes_reconciled": [35.0, 35.0, 35.0, 35.0, 26.0, 26.0, 26.0, 22.0] * 2,
        }
    )
    report = reconciliation_sanity_check(df, minutes_col="p50", reconciled_col="minutes_reconciled")
    assert report.teams_checked == 2
    assert report.cap_violations == 0
