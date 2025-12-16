import pandas as pd
import pandas as pd
import pytest

from projections.minutes_v1 import calibration


def _cfg(**overrides: object) -> calibration.RollingCalibrationConfig:
    defaults: dict[str, object] = {
        "window_days": 2,
        "min_n": 1,
        "tau": 1.0,
        "cold_start_min_n": 0,
        "p10_floor_guard": 0.0,
        "p90_floor_guard": 0.0,
        "use_global_p10_delta": True,
        "use_global_p90_delta": True,
        "warmup_days_p10": 0,
        "warmup_days_p90": 0,
        "bucket_floor_relief": 0.0,
        "max_abs_delta_p10": None,
        "max_abs_delta_p90": None,
        "recent_target_rows": 10_000,
        "recency_window_days": 365,
        "recency_half_life_days": 365.0,
        "global_recent_target_rows": 10_000,
        "global_recent_target_rows_p10": 10_000,
        "global_recent_target_rows_p90": 10_000,
        "global_min_n": 10_000,
        "global_baseline_months": 0,
        "global_delta_cap": 10.0,
        "bucket_min_effective_sample_size": 0.0,
        "bucket_inherit_if_low_n": True,
        "cap_guardrail_rate": None,
        "max_width_delta_pct": None,
        "monotonicity_repair_threshold": None,
        "apply_center_width_projection": True,
        "min_quantile_half_width": 0.01,
        "one_sided_days": 0,
        "hysteresis_lower": 0.0,
        "hysteresis_upper": 1.0,
        "delta_smoothing_alpha_p10": 0.0,
        "delta_smoothing_alpha_p90": 0.0,
    }
    defaults.update(overrides)
    return calibration.RollingCalibrationConfig(**defaults)


def _sample_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2, 3],
            "player_id": [10, 20, 10, 10, 30],
            "team_id": [5, 5, 5, 5, 5],
            "feature_as_of_ts": [
                "2024-11-30T15:00:00Z",
                "2024-11-30T15:05:00Z",
                "2024-12-01T15:00:00Z",
                "2024-12-01T16:00:00Z",
                "2024-12-02T15:00:00Z",
            ],
            "tip_ts": [
                "2024-12-01T00:00:00Z",
                "2024-12-01T00:00:00Z",
                "2024-12-02T00:00:00Z",
                "2024-12-02T00:00:00Z",
                "2024-12-03T00:00:00Z",
            ],
            "game_date": [
                "2024-12-01",
                "2024-12-01",
                "2024-12-02",
                "2024-12-02",
                "2024-12-03",
            ],
            "minutes": [30.0, 18.0, 28.0, 29.0, 26.0],
            "p10_raw": [20.0, 14.0, 21.0, 21.5, 19.0],
            "p90_raw": [40.0, 32.0, 39.0, 39.5, 36.0],
            "starter_prev_game_asof": [1, 0, 1, 1, 0],
            "ramp_flag": [0, 0, 0, 0, 0],
        }
    )


def test_compute_rolling_offsets_respects_history():
    df = _sample_predictions()
    cfg = _cfg(max_abs_delta_p10=0.0, max_abs_delta_p90=0.0)
    offsets = calibration.compute_rolling_offsets(
        df,
        global_delta_p10=0.5,
        global_delta_p90=-0.5,
        label_col="minutes",
        config=cfg,
    )

    assert offsets["score_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-12-01",
        "2024-12-02",
        "2024-12-03",
    ]

    day_two = offsets.loc[offsets["score_date"] == pd.Timestamp("2024-12-02")].iloc[0]
    assert day_two["window_start"] == pd.Timestamp("2024-11-30")
    assert day_two["window_end"] == pd.Timestamp("2024-12-02")
    assert day_two["n"] == 2  # Only the two 12-01 rows contribute.
    assert day_two["shrinkage"] == pytest.approx(2 / 3)
    # Local deltas: Q0.10([10, 4]) = 4.6, Q0.90([-10, -14]) = -10.4
    assert day_two["delta_p10"] == pytest.approx((2 / 3) * 4.6 + (1 / 3) * 0.5, rel=1e-4)
    assert day_two["delta_p90"] == pytest.approx((2 / 3) * -10.4 + (1 / 3) * -0.5, abs=1e-3)

    day_three = offsets.loc[offsets["score_date"] == pd.Timestamp("2024-12-03")].iloc[0]
    assert day_three["n"] == 3  # Adds the deduplicated 12-02 row.
    assert day_three["shrinkage"] == pytest.approx(3 / 4)


def test_compute_rolling_offsets_respects_minimum_history():
    df = _sample_predictions()
    cfg = _cfg(min_n=5, max_abs_delta_p10=0.0, max_abs_delta_p90=0.0)
    offsets = calibration.compute_rolling_offsets(
        df,
        global_delta_p10=1.25,
        global_delta_p90=-2.0,
        label_col="minutes",
        config=cfg,
    )
    day_two = offsets.loc[offsets["score_date"] == pd.Timestamp("2024-12-02")].iloc[0]
    assert day_two["n"] == 2
    assert day_two["shrinkage"] == 0.0  # below min_n
    assert day_two["delta_p10"] == pytest.approx(1.25, abs=1e-3)
    assert day_two["delta_p90"] == pytest.approx(-2.0, abs=1e-3)


def test_apply_rolling_offsets_updates_quantiles():
    df = _sample_predictions().loc[[0, 2]].copy()
    df["p10_raw"] = [15.0, 16.0]
    df["p90_raw"] = [28.0, 30.0]
    offsets = pd.DataFrame(
        {
            "score_date": [pd.Timestamp("2024-12-01"), pd.Timestamp("2024-12-02")],
            "delta_p10": [2.0, -1.0],
            "delta_p90": [3.0, 4.0],
        }
    )
    adjusted = calibration.apply_rolling_offsets(df, offsets)
    assert (adjusted["p10"] == pd.Series([17.0, 15.0])).all()
    assert (adjusted["p90"] == pd.Series([31.0, 34.0])).all()


def test_apply_rolling_offsets_requires_offsets():
    df = _sample_predictions().loc[[0, 2]].copy()
    offsets = pd.DataFrame(
        {
            "score_date": [pd.Timestamp("2024-12-01")],
            "delta_p10": [0.0],
            "delta_p90": [0.0],
        }
    )
    with pytest.raises(ValueError):
        calibration.apply_rolling_offsets(df, offsets)


def test_compute_rolling_offsets_with_buckets():
    df = pd.DataFrame(
        {
            "game_id": [101, 102, 201, 202],
            "player_id": [1, 2, 3, 4],
            "team_id": [10, 10, 10, 10],
            "feature_as_of_ts": [
                "2024-11-29T12:00:00Z",
                "2024-11-30T12:00:00Z",
                "2024-12-01T12:00:00Z",
                "2024-12-01T12:05:00Z",
            ],
            "tip_ts": [
                "2024-11-29T22:00:00Z",
                "2024-11-30T22:00:00Z",
                "2024-12-02T00:00:00Z",
                "2024-12-02T00:00:00Z",
            ],
            "game_date": ["2024-11-29", "2024-11-30", "2024-12-02", "2024-12-02"],
            "minutes": [35.0, 28.0, 30.0, 26.0],
            "p10_raw": [25.0, 20.0, 24.0, 22.0],
            "p90_raw": [45.0, 40.0, 44.0, 42.0],
            "injury_snapshot_missing": [0, 0, 0, 1],
            "spread_home": [1.0, 3.0, 2.0, 2.5],
        }
    )
    cfg = _cfg(
        window_days=5,
        tau=0.0,
        use_buckets=True,
        min_n_bucket=1,
    )
    offsets = calibration.compute_rolling_offsets(
        df,
        global_delta_p10=0.0,
        global_delta_p90=0.0,
        label_col="minutes",
        config=cfg,
    )
    day_two = offsets.loc[offsets["score_date"] == pd.Timestamp("2024-12-02")]
    assert len(day_two) == 2
    bucket_names = sorted(day_two["bucket_name"].tolist())
    assert bucket_names == ["injury_snapshot_missing=0", "injury_snapshot_missing=1"]

    bucket_zero = day_two.loc[day_two["injury_snapshot_missing"] == 0].iloc[0]
    assert bucket_zero["bucket_source"] == "bucket"
    assert bucket_zero["n_bucket"] == 2

    bucket_one = day_two.loc[day_two["injury_snapshot_missing"] == 1].iloc[0]
    assert bucket_one["bucket_source"] == "inherit"
    assert bucket_one["n_bucket"] == 0
    # Fallback bucket should reuse the window-level delta values.
    fallback_cfg = _cfg(window_days=5, tau=0.0, use_buckets=False)
    base_offsets = calibration.compute_rolling_offsets(
        df.drop(columns=["injury_snapshot_missing", "spread_home"]),
        global_delta_p10=0.0,
        global_delta_p90=0.0,
        label_col="minutes",
        config=fallback_cfg,
    )
    expected = base_offsets.loc[base_offsets["score_date"] == pd.Timestamp("2024-12-02")].iloc[0]
    assert bucket_one["delta_p10"] == pytest.approx(expected["delta_p10"])
    assert bucket_one["delta_p90"] == pytest.approx(expected["delta_p90"])


def test_apply_rolling_offsets_with_buckets_enforces_monotonicity():
    df = pd.DataFrame(
        {
            "game_id": [1001, 1002],
            "player_id": [1, 2],
            "team_id": [10, 20],
            "game_date": ["2024-12-01", "2024-12-01"],
            "p10_raw": [10.0, 5.0],
            "p50_raw": [15.0, 12.0],
            "p90_raw": [20.0, 18.0],
            "injury_snapshot_missing": [0, 1],
            "spread_home": [2.0, 8.0],
        }
    )
    offsets = pd.DataFrame(
        {
            "score_date": [pd.Timestamp("2024-12-01"), pd.Timestamp("2024-12-01")],
            "injury_snapshot_missing": [0, 1],
            "spread_bucket": ["abs_spread_0_4.5", "abs_spread_5_9.5"],
            "delta_p10": [10.0, 5.0],
            "delta_p90": [-10.0, -5.0],
        }
    )
    cfg = _cfg(use_buckets=True, use_spread_buckets=True)
    adjusted = calibration.apply_rolling_offsets(df, offsets, config=cfg)

    lower_ok = (adjusted["p10"] <= adjusted["p50_raw"]).all()
    upper_ok = (adjusted["p90"] >= adjusted["p50_raw"]).all()
    assert lower_ok and upper_ok


def test_compute_rolling_offsets_cold_start_zeroes_p10():
    df = _sample_predictions()
    cfg = _cfg(cold_start_min_n=10, max_abs_delta_p10=0.0, max_abs_delta_p90=0.0)
    offsets = calibration.compute_rolling_offsets(
        df,
        global_delta_p10=-2.0,
        global_delta_p90=3.0,
        label_col="minutes",
        config=cfg,
    )
    day_one = offsets.loc[offsets["score_date"] == pd.Timestamp("2024-12-01")].iloc[0]
    assert day_one["cold_start_used"] == 1
    assert day_one["delta_p10"] == pytest.approx(-2.0)
    assert day_one["delta_p90"] == pytest.approx(3.0)


def test_solve_p10_floor_target_hits_threshold():
    frame = pd.DataFrame(
        {
            "minutes": [28.0, 27.0, 26.0, 25.0],
            "p10_raw": [35.0, 35.0, 35.0, 35.0],
            "p90_raw": [45.0, 45.0, 45.0, 45.0],
        }
    )
    cfg = _cfg(p10_floor_guard=0.5)
    solved, guard_hit, before, after = calibration._solve_coverage_target(
        frame,
        delta=-10.0,
        label_col="minutes",
        quantile_col="p10_raw",
        target=cfg.p10_floor_guard,
    )
    assert guard_hit
    assert before < cfg.p10_floor_guard
    assert pytest.approx(after, rel=1e-3) == pytest.approx(cfg.p10_floor_guard, rel=1e-3)
    assert solved > -10.0


def test_max_abs_delta_p10_clamps_shift():
    df = pd.DataFrame(
        {
            "game_id": [1, 2],
            "player_id": [10, 10],
            "team_id": [5, 5],
            "feature_as_of_ts": ["2024-11-29T12:00:00Z", "2024-11-30T12:00:00Z"],
            "tip_ts": ["2024-11-29T22:00:00Z", "2024-11-30T22:00:00Z"],
            "game_date": ["2024-11-29", "2024-11-30"],
            "minutes": [0.0, 0.0],
            "p10_raw": [20.0, 20.0],
            "p90_raw": [35.0, 35.0],
            "injury_snapshot_missing": [0, 0],
        }
    )
    cfg = _cfg(tau=0.0, max_abs_delta_p10=0.25, use_buckets=True)
    offsets = calibration.compute_rolling_offsets(
        df,
        global_delta_p10=-1.0,
        global_delta_p90=1.0,
        label_col="minutes",
        config=cfg,
    )
    day_two = offsets.loc[offsets["score_date"] == pd.Timestamp("2024-11-30")].iloc[0]
    assert day_two["delta_p10"] == pytest.approx(-0.25)


def test_bucket_delta_cap_limits_deviation():
    df = pd.DataFrame(
        {
            "game_id": [101, 102, 201, 202],
            "player_id": [1, 2, 3, 4],
            "team_id": [10, 10, 10, 10],
            "feature_as_of_ts": [
                "2024-11-29T12:00:00Z",
                "2024-11-30T12:00:00Z",
                "2024-12-01T12:00:00Z",
                "2024-12-01T12:05:00Z",
            ],
            "tip_ts": [
                "2024-11-29T22:00:00Z",
                "2024-11-30T22:00:00Z",
                "2024-12-02T00:00:00Z",
                "2024-12-02T00:30:00Z",
            ],
            "game_date": ["2024-11-29", "2024-11-30", "2024-12-02", "2024-12-02"],
            "minutes": [30.0, 5.0, 28.0, 10.0],
            "p10_raw": [30.0, 30.0, 20.0, 20.0],
            "p90_raw": [45.0, 45.0, 35.0, 35.0],
            "injury_snapshot_missing": [0, 1, 0, 1],
        }
    )
    cfg = _cfg(
        window_days=5,
        tau=0.0,
        tau_bucket=0.0,
        use_buckets=True,
        min_n_bucket=1,
        bucket_delta_cap=0.15,
        max_abs_delta_p10=10.0,
    )
    offsets = calibration.compute_rolling_offsets(
        df,
        global_delta_p10=0.0,
        global_delta_p90=0.0,
        label_col="minutes",
        config=cfg,
    )
    day_rows = offsets.loc[offsets["score_date"] == pd.Timestamp("2024-12-02")]
    bucket_zero = day_rows.loc[day_rows["injury_snapshot_missing"] == 0].iloc[0]
    bucket_one = day_rows.loc[day_rows["injury_snapshot_missing"] == 1].iloc[0]
    assert bucket_one["bucket_source"] == "bucket"
    assert bucket_zero["bucket_source"] == "bucket"
    assert abs(bucket_one["delta_p10"] - bucket_zero["delta_p10"]) <= 0.15 + 1e-6


def test_bucket_inheritance_when_effective_sample_small():
    df = pd.DataFrame(
        {
            "game_id": [101, 102, 201, 202],
            "player_id": [1, 2, 3, 4],
            "team_id": [10, 10, 10, 10],
            "feature_as_of_ts": [
                "2024-11-30T12:00:00Z",
                "2024-11-30T12:05:00Z",
                "2024-12-01T12:00:00Z",
                "2024-12-01T12:05:00Z",
            ],
            "tip_ts": [
                "2024-11-30T22:00:00Z",
                "2024-11-30T22:05:00Z",
                "2024-12-02T00:00:00Z",
                "2024-12-02T00:05:00Z",
            ],
            "game_date": ["2024-12-01", "2024-12-01", "2024-12-02", "2024-12-02"],
            "minutes": [30.0, 29.0, 28.0, 10.0],
            "p10_raw": [21.0, 21.0, 20.0, 18.0],
            "p90_raw": [39.0, 39.0, 38.0, 36.0],
            "injury_snapshot_missing": [0, 0, 0, 1],
        }
    )
    cfg = _cfg(
        window_days=5,
        tau=0.0,
        use_buckets=True,
        min_n_bucket=1,
        bucket_min_effective_sample_size=5.0,
    )
    offsets = calibration.compute_rolling_offsets(
        df,
        global_delta_p10=0.0,
        global_delta_p90=0.0,
        label_col="minutes",
        config=cfg,
    )
    bucket_row = offsets.loc[
        (offsets["score_date"] == pd.Timestamp("2024-12-02"))
        & (offsets["bucket_name"] == "injury_snapshot_missing=1")
    ].iloc[0]
    assert bucket_row["bucket_source"] == "inherit"
    assert bucket_row["bucket_inherited"] == 1


def test_global_recent_weight_varies_by_stratum():
    df = pd.DataFrame(
        {
            "game_id": [101, 102, 201, 202],
            "player_id": [1, 2, 3, 4],
            "team_id": [10, 10, 10, 10],
            "feature_as_of_ts": [
                "2024-11-30T12:00:00Z",
                "2024-11-30T12:05:00Z",
                "2024-12-01T12:00:00Z",
                "2024-12-01T12:05:00Z",
            ],
            "tip_ts": [
                "2024-12-01T00:00:00Z",
                "2024-12-01T00:05:00Z",
                "2024-12-02T00:00:00Z",
                "2024-12-02T00:05:00Z",
            ],
            "game_date": ["2024-12-01", "2024-12-01", "2024-12-02", "2024-12-02"],
            "minutes": [30.0, 29.0, 15.0, 18.0],
            "p10_raw": [20.0, 20.0, 18.0, 19.0],
            "p90_raw": [40.0, 40.0, 36.0, 37.0],
            "injury_snapshot_missing": [0, 0, 1, 0],
        }
    )
    cfg = _cfg(
        window_days=5,
        tau=0.0,
        use_buckets=True,
        min_n_bucket=1,
        global_min_n=1,
        global_recent_target_rows=2,
        global_recent_target_rows_p10=2,
        global_recent_target_rows_p90=2,
        global_baseline_months=0,
    )
    offsets = calibration.compute_rolling_offsets(
        df,
        global_delta_p10=0.0,
        global_delta_p90=0.0,
        label_col="minutes",
        config=cfg,
        score_dates=[pd.Timestamp("2024-12-02")],
    )
    late_rows = offsets.loc[offsets["score_date"] == pd.Timestamp("2024-12-02")]
    zero_bucket = late_rows.loc[late_rows["bucket_name"] == "injury_snapshot_missing=0"].iloc[0]
    one_bucket = late_rows.loc[late_rows["bucket_name"] == "injury_snapshot_missing=1"].iloc[0]
    assert zero_bucket["global_recent_rows"] == 2
    assert one_bucket["global_recent_rows"] == 0
    assert zero_bucket["global_recent_weight_p10"] == pytest.approx(1.0)
    assert one_bucket["global_recent_weight_p10"] == pytest.approx(0.0)


def test_history_months_is_metadata_only():
    df = pd.DataFrame(
        {
            "game_id": [1001, 1002],
            "player_id": [1, 1],
            "team_id": [10, 10],
            "feature_as_of_ts": ["2024-11-01T12:00:00Z", "2024-12-01T12:00:00Z"],
            "tip_ts": ["2024-11-02T00:00:00Z", "2024-12-02T00:00:00Z"],
            "game_date": ["2024-11-02", "2024-12-02"],
            "minutes": [20.0, 25.0],
            "p10_raw": [18.0, 19.0],
            "p90_raw": [32.0, 33.0],
        }
    )
    base_cfg = _cfg(window_days=5, tau=0.0, history_months=0)
    base_offsets = calibration.compute_rolling_offsets(
        df,
        global_delta_p10=0.0,
        global_delta_p90=0.0,
        label_col="minutes",
        config=base_cfg,
        score_dates=[pd.Timestamp("2024-12-03")],
    )
    hist_cfg = _cfg(window_days=5, tau=0.0, history_months=2)
    hist_offsets = calibration.compute_rolling_offsets(
        df,
        global_delta_p10=0.0,
        global_delta_p90=0.0,
        label_col="minutes",
        config=hist_cfg,
        score_dates=[pd.Timestamp("2024-12-03")],
    )
    assert int(base_offsets.iloc[0]["n"]) == int(hist_offsets.iloc[0]["n"]) == 1


def test_warmup_days_blend_toward_raw():
    df = _sample_predictions()
    cfg = _cfg(tau=0.0, warmup_days_p10=10)
    offsets = calibration.compute_rolling_offsets(
        df,
        global_delta_p10=-1.0,
        global_delta_p90=1.0,
        label_col="minutes",
        config=cfg,
        score_dates=[pd.Timestamp("2024-12-01"), pd.Timestamp("2024-12-15")],
    )
    day_one = offsets.loc[offsets["score_date"] == pd.Timestamp("2024-12-01")].iloc[0]
    assert day_one["delta_p10"] == pytest.approx(0.0)
    day_mid = offsets.loc[offsets["score_date"] == pd.Timestamp("2024-12-15")].iloc[0]
    assert day_mid["delta_p10"] != pytest.approx(0.0)


def test_apply_rolling_offsets_width_guardrail():
    df = pd.DataFrame(
        {
            "game_id": [1],
            "player_id": [10],
            "team_id": [5],
            "game_date": ["2024-12-01"],
            "p10_raw": [10.0],
            "p50_raw": [12.0],
            "p90_raw": [14.0],
        }
    )
    offsets = pd.DataFrame(
        {
            "score_date": [pd.Timestamp("2024-12-01")],
            "delta_p10": [-5.0],
            "delta_p90": [10.0],
        }
    )
    cfg = _cfg(max_width_delta_pct=0.1)
    with pytest.raises(RuntimeError):
        calibration.apply_rolling_offsets(df, offsets, config=cfg)


def test_apply_rolling_offsets_monotonicity_guardrail():
    df = pd.DataFrame(
        {
            "game_id": [1, 2],
            "player_id": [10, 11],
            "team_id": [5, 5],
            "game_date": ["2024-12-01", "2024-12-01"],
            "p10_raw": [10.0, 11.0],
            "p50_raw": [12.0, 12.0],
            "p90_raw": [13.0, 13.0],
        }
    )
    offsets = pd.DataFrame(
        {
            "score_date": [pd.Timestamp("2024-12-01"), pd.Timestamp("2024-12-01")],
            "delta_p10": [5.0, 5.0],
            "delta_p90": [-5.0, -5.0],
        }
    )
    cfg = _cfg(monotonicity_repair_threshold=0.0, apply_center_width_projection=False)
    with pytest.raises(RuntimeError):
        calibration.apply_rolling_offsets(df, offsets, config=cfg)


def test_center_width_projection_symmetrizes_tails():
    df = pd.DataFrame(
        {
            "game_id": [1],
            "player_id": [10],
            "team_id": [5],
            "game_date": ["2024-12-01"],
            "p10_raw": [10.0],
            "p50_raw": [15.0],
            "p90_raw": [20.0],
        }
    )
    offsets = pd.DataFrame(
        {
            "score_date": [pd.Timestamp("2024-12-01")],
            "delta_p10": [8.0],
            "delta_p90": [-10.0],
        }
    )
    cfg = _cfg(apply_center_width_projection=True, min_quantile_half_width=0.5)
    adjusted = calibration.apply_rolling_offsets(df, offsets, config=cfg)
    row = adjusted.iloc[0]
    assert row["p10"] < row["p50_raw"] < row["p90"]
    assert pytest.approx(row["p50_raw"] - row["p10"], rel=1e-6) == pytest.approx(
        row["p90"] - row["p50_raw"], rel=1e-6
    )


def test_one_sided_delta_direction():
    # Coverage below target should block negative p10 deltas
    adjusted = calibration._apply_one_sided_delta(
        -0.5,
        direction="p10",
        enforce=True,
        coverage_before=0.05,
        lower_threshold=0.09,
        upper_threshold=0.11,
    )
    assert adjusted == 0.0

    # Coverage above target should block positive p10 deltas
    adjusted = calibration._apply_one_sided_delta(
        0.5,
        direction="p10",
        enforce=True,
        coverage_before=0.2,
        lower_threshold=0.09,
        upper_threshold=0.11,
    )
    assert adjusted == 0.0
