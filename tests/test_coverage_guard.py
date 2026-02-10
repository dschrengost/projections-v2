from __future__ import annotations

import pandas as pd
import pytest

from projections.etl.coverage_guard import (
    compute_game_coverage,
    enforce_game_coverage,
    format_game_coverage,
)


def test_compute_game_coverage_counts_overlap() -> None:
    schedule = pd.DataFrame({"game_id": [1, 2, 3]})
    observed = pd.DataFrame({"game_id": [2, 3, 99]})
    stats = compute_game_coverage(schedule_df=schedule, observed_df=observed)
    assert stats.scheduled_games == 3
    assert stats.observed_games == 3
    assert stats.overlap_games == 2
    assert stats.off_schedule_games == 1
    assert stats.coverage_rate == pytest.approx(2 / 3)


def test_enforce_game_coverage_no_games_no_error() -> None:
    stats = compute_game_coverage(
        schedule_df=pd.DataFrame(columns=["game_id"]),
        observed_df=pd.DataFrame(columns=["game_id"]),
    )
    enforce_game_coverage(dataset_name="odds", stats=stats, strict=True)


def test_enforce_game_coverage_raises_when_no_overlap() -> None:
    schedule = pd.DataFrame({"game_id": [1, 2]})
    observed = pd.DataFrame({"game_id": [99]})
    stats = compute_game_coverage(schedule_df=schedule, observed_df=observed)
    with pytest.raises(RuntimeError, match="no overlapping games"):
        enforce_game_coverage(dataset_name="injuries", stats=stats, strict=True)


def test_format_game_coverage_reports_no_game_day() -> None:
    stats = compute_game_coverage(
        schedule_df=pd.DataFrame(columns=["game_id"]),
        observed_df=pd.DataFrame(columns=["game_id"]),
    )
    msg = format_game_coverage("injuries", stats)
    assert "no scheduled games" in msg
