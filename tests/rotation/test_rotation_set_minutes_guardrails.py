from __future__ import annotations

import pandas as pd
import pytest

from projections.rotation.guardrails import apply_rotation_minutes_guardrails


def test_guardrail_row_fallback_overrides_rotation() -> None:
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "team_id": [10, 10, 10],
            "player_id": [101, 102, 103],
            "rotation_p50": [20.0, 100.0, 110.0],
            "baseline_p50": [30.0, 100.0, 110.0],
            "minutes_features_row_missing": [1, 0, 0],
            # Keep team blend off (lineup present, good injury coverage).
            "lineup_timestamp": [pd.Timestamp("2026-01-01T00:00:00Z"), pd.NaT, pd.NaT],
            "lineup_status": [pd.NA, pd.NA, pd.NA],
            "lineup_roster_status": [pd.NA, pd.NA, pd.NA],
            "injury_snapshot_missing": [0, 0, 0],
        }
    )

    result = apply_rotation_minutes_guardrails(
        df,
        rotation_p50_col="rotation_p50",
        baseline_p50_col="baseline_p50",
        blend_weight=0.2,
        injury_coverage_threshold=0.5,
        pathology_max_minutes_threshold=1e9,
        pathology_n_gt_40_threshold=999,
        pathology_top5_sum_threshold=1e9,
        team_target_minutes=240.0,
    )
    out = result.minutes_p50

    assert float(out.iloc[0]) == 30.0
    assert float(out.iloc[1]) == 100.0
    assert float(out.iloc[2]) == 110.0


def test_guardrail_team_blend_triggers_on_missing_lineups_and_low_injury_coverage() -> None:
    df = pd.DataFrame(
        {
            "game_id": [1, 1],
            "team_id": [10, 10],
            "player_id": [201, 202],
            "rotation_p50": [30.0, 210.0],
            "baseline_p50": [40.0, 200.0],
            "minutes_features_row_missing": [0, 0],
            # Lineup fields missing for all rows.
            "lineup_timestamp": [pd.NaT, pd.NaT],
            "lineup_status": [pd.NA, pd.NA],
            "lineup_roster_status": [pd.NA, pd.NA],
            # injury_snapshot_missing=1 => injury_coverage_rate=0.0
            "injury_snapshot_missing": [1, 1],
        }
    )

    result = apply_rotation_minutes_guardrails(
        df,
        rotation_p50_col="rotation_p50",
        baseline_p50_col="baseline_p50",
        blend_weight=0.2,
        injury_coverage_threshold=0.5,
        pathology_max_minutes_threshold=1e9,
        pathology_n_gt_40_threshold=999,
        pathology_top5_sum_threshold=1e9,
        team_target_minutes=240.0,
    )
    out = result.minutes_p50

    # 0.2*rot + 0.8*base
    assert float(out.iloc[0]) == 38.0
    assert float(out.iloc[1]) == 202.0


def test_guardrail_scales_to_240_even_without_row_fallbacks() -> None:
    df = pd.DataFrame(
        {
            "game_id": [1, 1],
            "team_id": [10, 10],
            "player_id": [301, 302],
            "rotation_p50": [200.0, 40.0],
            "baseline_p50": [120.0, 120.0],
            "minutes_features_row_missing": [0, 0],
            # Keep team blend off.
            "lineup_timestamp": [pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")],
            "lineup_status": ["Confirmed", "Confirmed"],
            "lineup_roster_status": [pd.NA, pd.NA],
            "injury_snapshot_missing": [0, 0],
            # OUT rows should be hard-zeroed, and the remaining rows should scale back to 240.
            "status": ["OUT", "Ava"],
        }
    )

    result = apply_rotation_minutes_guardrails(
        df,
        rotation_p50_col="rotation_p50",
        baseline_p50_col="baseline_p50",
        dnp_tail_minutes_threshold=8.0,
        pathology_max_minutes_threshold=1e9,
        pathology_n_gt_40_threshold=999,
        pathology_top5_sum_threshold=1e9,
        team_target_minutes=240.0,
    )

    out = result.minutes_p50
    assert float(out.sum()) == pytest.approx(240.0, abs=1e-6)
    assert float(out.iloc[0]) == 0.0
    assert float(out.iloc[1]) == pytest.approx(240.0, abs=1e-6)


def test_guardrail_cap_max_minutes_caps_and_redistributes() -> None:
    df = pd.DataFrame(
        {
            "game_id": [1] * 6,
            "team_id": [10] * 6,
            "player_id": [1, 2, 3, 4, 5, 6],
            # Sums to 240, max is > cap but <= OT threshold (48) so we should cap, not fall back.
            "rotation_p50": [46.0, 46.0, 40.0, 40.0, 34.0, 34.0],
            "baseline_p50": [40.0, 40.0, 40.0, 40.0, 40.0, 40.0],
            "minutes_features_row_missing": [0] * 6,
            # Keep team blend off.
            "lineup_timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")] * 6,
            "lineup_status": [pd.NA] * 6,
            "lineup_roster_status": [pd.NA] * 6,
            "injury_snapshot_missing": [0] * 6,
            "status": ["Ava"] * 6,
        }
    )

    result = apply_rotation_minutes_guardrails(
        df,
        rotation_p50_col="rotation_p50",
        baseline_p50_col="baseline_p50",
        cap_max_minutes=44.0,
        team_target_minutes=240.0,
    )

    out = result.minutes_p50
    assert float(out.sum()) == pytest.approx(240.0, abs=1e-6)
    assert float(out.max()) <= 44.0 + 1e-6
    assert float(out.iloc[0]) == pytest.approx(44.0, abs=1e-6)
    assert float(out.iloc[1]) == pytest.approx(44.0, abs=1e-6)


def test_guardrail_pathology_fallback_reverts_team_game_to_baseline() -> None:
    df = pd.DataFrame(
        {
            "game_id": [1] * 6,
            "team_id": [10] * 6,
            "player_id": [1, 2, 3, 4, 5, 6],
            # OT-level max minutes triggers pathology fallback (default threshold 48).
            "rotation_p50": [65.0, 50.0, 45.0, 40.0, 20.0, 20.0],
            "baseline_p50": [40.0, 38.0, 36.0, 34.0, 46.0, 46.0],  # sums to 240
            "minutes_features_row_missing": [0] * 6,
            "lineup_timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")] * 6,
            "lineup_status": [pd.NA] * 6,
            "lineup_roster_status": [pd.NA] * 6,
            "injury_snapshot_missing": [0] * 6,
            "status": ["Ava"] * 6,
        }
    )

    result = apply_rotation_minutes_guardrails(
        df,
        rotation_p50_col="rotation_p50",
        baseline_p50_col="baseline_p50",
        cap_max_minutes=44.0,
        team_target_minutes=240.0,
    )

    out = result.minutes_p50
    assert float(out.sum()) == pytest.approx(240.0, abs=1e-6)
    assert out.to_numpy(dtype=float) == pytest.approx(df["baseline_p50"].to_numpy(dtype=float), abs=1e-6)
    assert bool(result.team_game_summary.loc[0, "pathology_fallback"]) is True


def test_guardrail_flatness_does_not_fallback_on_low_max_if_top5_ok() -> None:
    """Regression: avoid over-triggering flatness fallback on reasonable rotations."""

    df = pd.DataFrame(
        {
            "game_id": [1] * 9,
            "team_id": [10] * 9,
            "player_id": list(range(1, 10)),
            # Sums to 240, max < 30, but top5_sum is still healthy (>125).
            "rotation_p50": [27.0] * 6 + [26.0] * 3,
            "baseline_p50": [35.0, 34.0, 33.0, 30.0, 28.0, 25.0, 20.0, 18.0, 17.0],
            "minutes_features_row_missing": [0] * 9,
            "lineup_timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")] * 9,
            "lineup_status": [pd.NA] * 9,
            "lineup_roster_status": [pd.NA] * 9,
            "injury_snapshot_missing": [0] * 9,
            "status": ["Ava"] * 9,
        }
    )

    result = apply_rotation_minutes_guardrails(
        df,
        rotation_p50_col="rotation_p50",
        baseline_p50_col="baseline_p50",
        pathology_max_minutes_threshold=1e9,
        pathology_n_gt_40_threshold=999,
        pathology_top5_sum_threshold=1e9,
        pathology_min_max_minutes_threshold=30.0,
        pathology_min_top5_sum_threshold=125.0,
        team_target_minutes=240.0,
    )

    out = result.minutes_p50
    assert float(out.sum()) == pytest.approx(240.0, abs=1e-6)
    assert out.to_numpy(dtype=float) == pytest.approx(df["rotation_p50"].to_numpy(dtype=float), abs=1e-6)
    assert bool(result.team_game_summary.loc[0, "pathology_fallback"]) is False


def test_guardrail_flatness_fallback_triggers_when_max_and_top5_both_low() -> None:
    df = pd.DataFrame(
        {
            "game_id": [1] * 10,
            "team_id": [10] * 10,
            "player_id": list(range(1, 11)),
            # Uniform 10-man split: max < 30 AND top5_sum < 125 -> treat as flat overlay.
            "rotation_p50": [24.0] * 10,
            "baseline_p50": [35.0, 34.0, 33.0, 30.0, 28.0, 25.0, 20.0, 18.0, 10.0, 7.0],
            "minutes_features_row_missing": [0] * 10,
            "lineup_timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")] * 10,
            "lineup_status": [pd.NA] * 10,
            "lineup_roster_status": [pd.NA] * 10,
            "injury_snapshot_missing": [0] * 10,
            "status": ["Ava"] * 10,
        }
    )

    result = apply_rotation_minutes_guardrails(
        df,
        rotation_p50_col="rotation_p50",
        baseline_p50_col="baseline_p50",
        pathology_max_minutes_threshold=1e9,
        pathology_n_gt_40_threshold=999,
        pathology_top5_sum_threshold=1e9,
        pathology_min_max_minutes_threshold=30.0,
        pathology_min_top5_sum_threshold=125.0,
        team_target_minutes=240.0,
    )

    out = result.minutes_p50
    assert float(out.sum()) == pytest.approx(240.0, abs=1e-6)
    assert out.to_numpy(dtype=float) == pytest.approx(df["baseline_p50"].to_numpy(dtype=float), abs=1e-6)
    assert bool(result.team_game_summary.loc[0, "pathology_fallback"]) is True
