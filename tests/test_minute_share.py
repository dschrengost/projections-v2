"""Tests for minute share prediction model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.minutes_v1.minute_share import (
    DNP_SHARE_THRESHOLD,
    MINUTE_BUCKETS,
    ROTATION_MINUTES_THRESHOLD,
    TEAM_TOTAL_MINUTES,
    MinuteLabelMode,
    MinuteShareArtifacts,
    MinuteShareEvaluation,
    compute_minute_share,
    evaluate_minute_share_model,
    normalize_shares_per_team,
    predict_minutes,
    predict_raw_shares,
    train_minute_share_model,
    validate_share_predictions,
)


@pytest.fixture
def sample_team_data() -> pd.DataFrame:
    """Create a sample team with realistic minute distributions."""
    np.random.seed(42)
    n_players = 12  # Typical roster

    # Realistic minute distribution: starters get more
    minutes = np.array([
        36, 34, 32, 28, 26,  # 5 starters
        18, 14, 8, 4, 0, 0, 0,  # 7 bench players (some DNP)
    ])
    assert minutes.sum() == 200  # Typical for one team if some players don't play
    # Scale to sum to 240 for players who play
    active_minutes = minutes[minutes > 0]
    minutes_scaled = np.zeros(n_players)
    minutes_scaled[:len(active_minutes)] = (active_minutes / active_minutes.sum()) * 240

    return pd.DataFrame({
        "game_id": [1] * n_players,  # Added game_id
        "player_id": range(1, n_players + 1),
        "team_id": ["TEAM_A"] * n_players,
        "feature_1": np.random.randn(n_players),
        "feature_2": np.random.randn(n_players),
        "feature_3": np.random.randn(n_players),
        "minutes": minutes_scaled,
    })


@pytest.fixture
def multi_team_data() -> pd.DataFrame:
    """Create sample data for multiple teams."""
    np.random.seed(42)
    teams = ["TEAM_A", "TEAM_B", "TEAM_C"]
    dfs = []

    for i, team in enumerate(teams):
        n_players = 10
        # Vary distribution per team
        base_minutes = np.array([40, 35, 30, 28, 25, 22, 18, 12, 6, 0])
        # Scale so active players sum to 240
        active = base_minutes[base_minutes > 0]
        scaled = np.zeros(n_players)
        scaled[:len(active)] = (active / active.sum()) * 240

        team_df = pd.DataFrame({
            "game_id": [i + 1] * n_players,  # Each team in different game
            "player_id": range(i * 100 + 1, i * 100 + n_players + 1),
            "team_id": [team] * n_players,
            "feature_1": np.random.randn(n_players),
            "feature_2": np.random.randn(n_players),
            "feature_3": np.random.randn(n_players),
            "minutes": scaled,
        })
        dfs.append(team_df)

    return pd.concat(dfs, ignore_index=True)


class TestComputeMinuteShare:
    """Tests for compute_minute_share function."""

    def test_basic_conversion(self):
        """Test basic minute to share conversion with REG240 mode."""
        minutes = np.array([48, 36, 24, 12, 0])
        game_ids = np.array([1, 1, 1, 1, 1])
        team_ids = np.array([1, 1, 1, 1, 1])
        shares = compute_minute_share(minutes, game_ids, team_ids, mode=MinuteLabelMode.REG240)

        assert shares[0] == pytest.approx(0.2)  # 48/240
        assert shares[1] == pytest.approx(0.15)  # 36/240
        assert shares[2] == pytest.approx(0.1)  # 24/240
        assert shares[3] == pytest.approx(0.05)  # 12/240
        assert shares[4] == pytest.approx(0.0)  # 0/240

    def test_star_player_share(self):
        """Test that star player shares are in expected range."""
        # Max realistic minutes ~42 (rare case)
        max_minutes = 42
        game_ids = np.array([1])
        team_ids = np.array([1])
        share = compute_minute_share(np.array([max_minutes]), game_ids, team_ids, mode=MinuteLabelMode.REG240)[0]
        assert 0.15 <= share <= 0.20  # ~0.175

    def test_sum_of_shares_reg240(self):
        """Test that team shares sum to 1.0 with REG240 mode when team total is 240."""
        # Full team that sums to 240
        minutes = np.array([36, 34, 32, 28, 26, 22, 18, 14, 10, 8, 8, 4])
        assert minutes.sum() == 240
        game_ids = np.array([1] * 12)
        team_ids = np.array([1] * 12)

        shares = compute_minute_share(minutes, game_ids, team_ids, mode=MinuteLabelMode.REG240)
        assert shares.sum() == pytest.approx(1.0)

    def test_custom_team_total(self):
        """Test with custom team total override."""
        minutes = np.array([30])
        game_ids = np.array([1])
        team_ids = np.array([1])
        share = compute_minute_share(minutes, game_ids, team_ids, team_total=300)
        assert share[0] == pytest.approx(0.1)
    
    def test_team_total_actual_mode(self):
        """Test TEAM_TOTAL_ACTUAL mode sums to 1.0."""
        minutes = np.array([36, 34, 32, 28, 26, 22, 18, 14])  # sum=210
        game_ids = np.array([1] * 8)
        team_ids = np.array([1] * 8)
        
        shares = compute_minute_share(minutes, game_ids, team_ids, mode=MinuteLabelMode.TEAM_TOTAL_ACTUAL)
        assert shares.sum() == pytest.approx(1.0)
    
    def test_ot_game_reg240_exceeds_one(self):
        """Test that OT games with REG240 mode can have shares sum >1.0."""
        # OT game: 265 total minutes
        minutes = np.array([45, 42, 40, 38, 36, 30, 20, 14])  # sum=265
        game_ids = np.array([1] * 8)
        team_ids = np.array([1] * 8)
        
        shares = compute_minute_share(minutes, game_ids, team_ids, mode=MinuteLabelMode.REG240)
        # With REG240, sum = 265/240 > 1.0
        assert shares.sum() > 1.0
        assert shares.sum() == pytest.approx(265.0 / 240.0)


class TestNormalizeSharesPerTeam:
    """Tests for normalize_shares_per_team function."""

    def test_single_team_normalization(self):
        """Test normalization for a single team."""
        raw_shares = np.array([0.3, 0.25, 0.20, 0.15, 0.10])
        game_ids = pd.Series([1, 1, 1, 1, 1])
        team_ids = pd.Series(["A", "A", "A", "A", "A"])

        normalized = normalize_shares_per_team(raw_shares, game_ids, team_ids)

        assert normalized.sum() == pytest.approx(1.0)

    def test_multi_team_normalization(self):
        """Test normalization across multiple teams in same game."""
        raw_shares = np.array([
            0.4, 0.3, 0.2,  # Team A: sum = 0.9
            0.5, 0.4, 0.3,  # Team B: sum = 1.2
        ])
        game_ids = pd.Series([1, 1, 1, 1, 1, 1])
        team_ids = pd.Series(["A", "A", "A", "B", "B", "B"])

        normalized = normalize_shares_per_team(raw_shares, game_ids, team_ids)

        # Each team should sum to 1.0
        team_a_sum = normalized[:3].sum()
        team_b_sum = normalized[3:].sum()

        assert team_a_sum == pytest.approx(1.0)
        assert team_b_sum == pytest.approx(1.0)

    def test_handles_near_zero_sum(self):
        """Test that epsilon prevents division by zero."""
        raw_shares = np.array([0.0, 0.0, 0.0])
        game_ids = pd.Series([1, 1, 1])
        team_ids = pd.Series(["A", "A", "A"])

        # Should not raise
        normalized = normalize_shares_per_team(raw_shares, game_ids, team_ids)
        assert np.all(np.isfinite(normalized))


class TestTrainMinuteShareModel:
    """Tests for model training."""

    def test_training_returns_artifacts(self, sample_team_data):
        """Test that training produces valid artifacts."""
        # Filter to players who played
        df = sample_team_data[sample_team_data["minutes"] > 0].copy()
        df["game_id"] = 1  # Add game_id
        X = df[["feature_1", "feature_2", "feature_3"]]
        y = df["minutes"]
        game_ids = df["game_id"] if "game_id" in df.columns else pd.Series([1] * len(df))
        team_ids = df["team_id"]

        artifacts = train_minute_share_model(X, y, game_ids, team_ids, random_state=42)

        assert isinstance(artifacts, MinuteShareArtifacts)
        assert artifacts.model is not None
        assert artifacts.imputer is not None
        assert artifacts.feature_columns == ["feature_1", "feature_2", "feature_3"]

    def test_training_with_custom_params(self, sample_team_data):
        """Test training with custom LightGBM parameters."""
        df = sample_team_data[sample_team_data["minutes"] > 0].copy()
        df["game_id"] = 1
        X = df[["feature_1", "feature_2", "feature_3"]]
        y = df["minutes"]
        game_ids = df["game_id"]
        team_ids = df["team_id"]

        custom_params = {"n_estimators": 50, "learning_rate": 0.1}
        artifacts = train_minute_share_model(X, y, game_ids, team_ids, params=custom_params)

        assert artifacts.params["n_estimators"] == 50
        assert artifacts.params["learning_rate"] == 0.1


class TestPredictMinutes:
    """Tests for inference pipeline."""

    def test_predictions_sum_to_240(self, multi_team_data):
        """Test that predicted minutes sum to 240 per team."""
        # Filter to active players
        df = multi_team_data[multi_team_data["minutes"] > 0]

        X = df[["feature_1", "feature_2", "feature_3"]]
        y = df["minutes"]
        team_ids = df["team_id"]

        # Train model
        artifacts = train_minute_share_model(X, y, random_state=42)

        # Predict
        predictions = predict_minutes(artifacts, X, team_ids)

        # Check each team sums to 240
        predictions["team_id"] = team_ids.values
        team_sums = predictions.groupby("team_id")["predicted_minutes"].sum()

        for team, total in team_sums.items():
            assert total == pytest.approx(TEAM_TOTAL_MINUTES, abs=0.01), \
                f"Team {team} total {total} != 240"

    def test_normalized_shares_sum_to_one(self, multi_team_data):
        """Test that normalized shares sum to 1.0 per team."""
        df = multi_team_data[multi_team_data["minutes"] > 0]

        X = df[["feature_1", "feature_2", "feature_3"]]
        y = df["minutes"]
        team_ids = df["team_id"]

        artifacts = train_minute_share_model(X, y, random_state=42)
        predictions = predict_minutes(artifacts, X, team_ids)

        predictions["team_id"] = team_ids.values
        share_sums = predictions.groupby("team_id")["normalized_share"].sum()

        for team, total in share_sums.items():
            assert total == pytest.approx(1.0, abs=1e-6), \
                f"Team {team} shares sum to {total} != 1.0"

    def test_raw_share_sum_diagnostic(self, multi_team_data):
        """Test that raw_share_sum is computed correctly."""
        df = multi_team_data[multi_team_data["minutes"] > 0]

        X = df[["feature_1", "feature_2", "feature_3"]]
        y = df["minutes"]
        team_ids = df["team_id"]

        artifacts = train_minute_share_model(X, y, random_state=42)
        predictions = predict_minutes(artifacts, X, team_ids)

        # raw_share_sum should be consistent within a team
        predictions["team_id"] = team_ids.values
        for team_id, group in predictions.groupby("team_id"):
            assert group["raw_share_sum"].nunique() == 1, \
                f"Team {team_id} has inconsistent raw_share_sum values"

    def test_non_negative_predictions(self, multi_team_data):
        """Test that all predictions are non-negative."""
        df = multi_team_data[multi_team_data["minutes"] > 0]

        X = df[["feature_1", "feature_2", "feature_3"]]
        y = df["minutes"]
        team_ids = df["team_id"]

        artifacts = train_minute_share_model(X, y, random_state=42)
        predictions = predict_minutes(artifacts, X, team_ids)

        assert (predictions["raw_share"] >= 0).all()
        assert (predictions["normalized_share"] >= 0).all()
        assert (predictions["predicted_minutes"] >= 0).all()


class TestValidateSharePredictions:
    """Tests for validation functions."""

    def test_passes_valid_predictions(self, multi_team_data):
        """Test that valid predictions pass validation."""
        df = multi_team_data[multi_team_data["minutes"] > 0]

        X = df[["feature_1", "feature_2", "feature_3"]]
        y = df["minutes"]
        team_ids = df["team_id"]

        artifacts = train_minute_share_model(X, y, random_state=42)
        predictions = predict_minutes(artifacts, X, team_ids)
        predictions["team_id"] = team_ids.values

        report = validate_share_predictions(predictions, verbose=False)

        assert report.team_sums_normalized

    def test_reports_raw_share_sum_stats(self, multi_team_data):
        """Test that validation reports raw share sum statistics."""
        df = multi_team_data[multi_team_data["minutes"] > 0]

        X = df[["feature_1", "feature_2", "feature_3"]]
        y = df["minutes"]
        team_ids = df["team_id"]

        artifacts = train_minute_share_model(X, y, random_state=42)
        predictions = predict_minutes(artifacts, X, team_ids)
        predictions["team_id"] = team_ids.values

        report = validate_share_predictions(predictions, verbose=False)

        # Should have valid statistics
        assert report.mean_raw_share_sum > 0
        assert report.min_raw_share_sum <= report.mean_raw_share_sum
        assert report.max_raw_share_sum >= report.mean_raw_share_sum


class TestPredictRawShares:
    """Tests for predict_raw_shares function."""

    def test_raises_on_missing_features(self, sample_team_data):
        """Test that missing features raise ValueError."""
        df = sample_team_data[sample_team_data["minutes"] > 0]
        X = df[["feature_1", "feature_2", "feature_3"]]
        y = df["minutes"]

        artifacts = train_minute_share_model(X, y, random_state=42)

        # Try to predict with missing feature
        X_bad = df[["feature_1", "feature_2"]]  # Missing feature_3

        with pytest.raises(ValueError, match="Missing required feature columns"):
            predict_raw_shares(artifacts, X_bad)

    def test_handles_nan_features(self, sample_team_data):
        """Test that NaN features are imputed."""
        df = sample_team_data[sample_team_data["minutes"] > 0].copy()
        X = df[["feature_1", "feature_2", "feature_3"]]
        y = df["minutes"]

        artifacts = train_minute_share_model(X, y, random_state=42)

        # Introduce NaN
        X_with_nan = X.copy()
        X_with_nan.iloc[0, 0] = np.nan

        # Should not raise
        shares = predict_raw_shares(artifacts, X_with_nan)
        assert np.all(np.isfinite(shares))


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline(self, multi_team_data):
        """Test full train -> predict -> validate pipeline."""
        # Prepare data
        df = multi_team_data[multi_team_data["minutes"] > 0]

        # Split train/test by team (simple approach)
        train_df = df[df["team_id"].isin(["TEAM_A", "TEAM_B"])]
        test_df = df[df["team_id"] == "TEAM_C"]

        feature_cols = ["feature_1", "feature_2", "feature_3"]

        # Train
        artifacts = train_minute_share_model(
            train_df[feature_cols],
            train_df["minutes"],
            random_state=42,
        )

        # Predict on test set
        predictions = predict_minutes(
            artifacts,
            test_df[feature_cols],
            test_df["team_id"],
        )
        predictions["team_id"] = test_df["team_id"].values

        # Validate
        report = validate_share_predictions(predictions, verbose=False)

        # Core assertions
        assert report.team_sums_normalized
        assert predictions["predicted_minutes"].sum() == pytest.approx(
            TEAM_TOTAL_MINUTES,  # Only one team in test
            abs=0.1
        )

    def test_realistic_share_ranges(self, multi_team_data):
        """Test that predicted shares fall in realistic ranges."""
        df = multi_team_data[multi_team_data["minutes"] > 0]

        X = df[["feature_1", "feature_2", "feature_3"]]
        y = df["minutes"]
        team_ids = df["team_id"]

        artifacts = train_minute_share_model(X, y, random_state=42)
        predictions = predict_minutes(artifacts, X, team_ids)

        # Max share should be <= ~0.20 (48/240)
        # In practice with noise, allow some tolerance
        max_share = predictions["normalized_share"].max()
        assert max_share <= 0.25, f"Max share {max_share} is unrealistically high"

        # Min share should be >= 0
        min_share = predictions["normalized_share"].min()
        assert min_share >= 0, f"Min share {min_share} is negative"


# =============================================================================
# EVALUATION FUNCTION TESTS
# =============================================================================


@pytest.fixture
def evaluation_data() -> dict:
    """Create sample data for evaluation tests.

    Returns data for 3 teams with 10 players each, including:
    - Some DNP players (actual_minutes = 0)
    - Some rotation players (>= 15 min)
    - Various minute buckets covered
    """
    np.random.seed(42)
    teams = ["TEAM_A", "TEAM_B", "TEAM_C"]
    data_rows = []

    for team in teams:
        # Realistic minute distribution per team
        actual_minutes = np.array([
            35.0,  # star (core bucket)
            32.0,  # star (core bucket)
            28.0,  # rotation (rotation bucket)
            25.0,  # rotation (rotation bucket)
            22.0,  # rotation (rotation bucket)
            16.0,  # rotation (bench bucket)
            10.0,  # bench (garbage_time bucket)
            6.0,   # bench (garbage_time bucket)
            0.0,   # DNP
            0.0,   # DNP
        ])
        # Scale active players to sum to 240
        active_mask = actual_minutes > 0
        active_total = actual_minutes[active_mask].sum()
        actual_minutes[active_mask] = actual_minutes[active_mask] / active_total * 240

        # Predicted shares with some noise
        predicted_shares = actual_minutes / 240 + np.random.randn(10) * 0.02
        predicted_shares = np.maximum(0, predicted_shares)  # Clamp to non-negative

        # Normalize predicted shares per team
        predicted_shares = predicted_shares / predicted_shares.sum()

        # Predicted minutes (after normalization)
        predicted_minutes = predicted_shares * 240

        for i in range(10):
            data_rows.append({
                "team_id": team,
                "actual_minutes": actual_minutes[i],
                "predicted_shares": predicted_shares[i],
                "predicted_minutes": predicted_minutes[i],
            })

    return {
        "actual_minutes": np.array([r["actual_minutes"] for r in data_rows]),
        "predicted_shares": np.array([r["predicted_shares"] for r in data_rows]),
        "predicted_minutes": np.array([r["predicted_minutes"] for r in data_rows]),
        "team_ids": np.array([r["team_id"] for r in data_rows]),
    }


class TestMinuteShareEvaluation:
    """Tests for MinuteShareEvaluation dataclass."""

    def test_to_dict_contains_all_fields(self, evaluation_data):
        """Test that to_dict() returns all expected keys."""
        result = evaluate_minute_share_model(
            actual_minutes=evaluation_data["actual_minutes"],
            predicted_shares=evaluation_data["predicted_shares"],
            predicted_minutes=evaluation_data["predicted_minutes"],
            team_ids=evaluation_data["team_ids"],
            verbose=False,
        )

        d = result.to_dict()

        # Check core metrics
        assert "mae_shares" in d
        assert "mae_minutes" in d
        assert "rmse_shares" in d
        assert "rmse_minutes" in d

        # Check DNP classification
        assert "dnp_auc" in d
        assert "dnp_accuracy" in d
        assert "dnp_precision" in d
        assert "dnp_recall" in d
        assert "dnp_threshold" in d

        # Check rotation classification
        assert "rotation_accuracy" in d
        assert "rotation_precision" in d
        assert "rotation_recall" in d

        # Check distribution diagnostics
        assert "team_share_sum_mean" in d
        assert "team_share_sum_std" in d
        assert "max_share_mean" in d
        assert "max_share_std" in d

        # Check sample sizes
        assert "n_samples" in d
        assert "n_teams" in d
        assert "n_dnp_actual" in d
        assert "n_rotation_actual" in d

        # Check bucket metrics
        for bucket_name in MINUTE_BUCKETS:
            assert f"mae_{bucket_name}" in d
            assert f"count_{bucket_name}" in d

    def test_to_dict_values_are_serializable(self, evaluation_data):
        """Test that to_dict() values are JSON-serializable types."""
        result = evaluate_minute_share_model(
            actual_minutes=evaluation_data["actual_minutes"],
            predicted_shares=evaluation_data["predicted_shares"],
            predicted_minutes=evaluation_data["predicted_minutes"],
            team_ids=evaluation_data["team_ids"],
            verbose=False,
        )

        d = result.to_dict()

        # All values should be basic Python types
        for key, value in d.items():
            if value is not None:
                assert isinstance(value, (int, float, str, bool)), \
                    f"Key {key} has non-serializable type {type(value)}"


class TestEvaluateMinuteShareModel:
    """Tests for evaluate_minute_share_model function."""

    def test_returns_evaluation_object(self, evaluation_data):
        """Test that function returns MinuteShareEvaluation."""
        result = evaluate_minute_share_model(
            actual_minutes=evaluation_data["actual_minutes"],
            predicted_shares=evaluation_data["predicted_shares"],
            predicted_minutes=evaluation_data["predicted_minutes"],
            team_ids=evaluation_data["team_ids"],
            verbose=False,
        )

        assert isinstance(result, MinuteShareEvaluation)

    def test_mae_is_non_negative(self, evaluation_data):
        """Test that MAE values are non-negative."""
        result = evaluate_minute_share_model(
            actual_minutes=evaluation_data["actual_minutes"],
            predicted_shares=evaluation_data["predicted_shares"],
            predicted_minutes=evaluation_data["predicted_minutes"],
            team_ids=evaluation_data["team_ids"],
            verbose=False,
        )

        assert result.mae_shares >= 0
        assert result.mae_minutes >= 0
        assert result.rmse_shares >= 0
        assert result.rmse_minutes >= 0

    def test_mae_shares_is_reasonable(self, evaluation_data):
        """Test that MAE in shares is in reasonable range."""
        result = evaluate_minute_share_model(
            actual_minutes=evaluation_data["actual_minutes"],
            predicted_shares=evaluation_data["predicted_shares"],
            predicted_minutes=evaluation_data["predicted_minutes"],
            team_ids=evaluation_data["team_ids"],
            verbose=False,
        )

        # With small noise added, MAE should be small
        assert result.mae_shares < 0.1  # Less than 10% share error on average

    def test_sample_counts_are_correct(self, evaluation_data):
        """Test that sample counts are accurate."""
        result = evaluate_minute_share_model(
            actual_minutes=evaluation_data["actual_minutes"],
            predicted_shares=evaluation_data["predicted_shares"],
            predicted_minutes=evaluation_data["predicted_minutes"],
            team_ids=evaluation_data["team_ids"],
            verbose=False,
        )

        # 3 teams, 10 players each = 30 total
        assert result.n_samples == 30
        assert result.n_teams == 3

        # 2 DNP players per team = 6 total
        assert result.n_dnp_actual == 6

    def test_dnp_threshold_is_used(self, evaluation_data):
        """Test that custom DNP threshold is used."""
        custom_threshold = 0.05

        result = evaluate_minute_share_model(
            actual_minutes=evaluation_data["actual_minutes"],
            predicted_shares=evaluation_data["predicted_shares"],
            predicted_minutes=evaluation_data["predicted_minutes"],
            team_ids=evaluation_data["team_ids"],
            dnp_threshold=custom_threshold,
            verbose=False,
        )

        assert result.dnp_threshold == custom_threshold

    def test_accuracy_is_bounded(self, evaluation_data):
        """Test that accuracy values are in [0, 1]."""
        result = evaluate_minute_share_model(
            actual_minutes=evaluation_data["actual_minutes"],
            predicted_shares=evaluation_data["predicted_shares"],
            predicted_minutes=evaluation_data["predicted_minutes"],
            team_ids=evaluation_data["team_ids"],
            verbose=False,
        )

        assert 0 <= result.dnp_accuracy <= 1
        assert 0 <= result.rotation_accuracy <= 1

        if result.dnp_precision is not None:
            assert 0 <= result.dnp_precision <= 1
        if result.dnp_recall is not None:
            assert 0 <= result.dnp_recall <= 1
        if result.rotation_precision is not None:
            assert 0 <= result.rotation_precision <= 1
        if result.rotation_recall is not None:
            assert 0 <= result.rotation_recall <= 1
        if result.dnp_auc is not None:
            assert 0 <= result.dnp_auc <= 1

    def test_bucket_mae_values(self, evaluation_data):
        """Test that bucket MAE values are computed for non-empty buckets."""
        result = evaluate_minute_share_model(
            actual_minutes=evaluation_data["actual_minutes"],
            predicted_shares=evaluation_data["predicted_shares"],
            predicted_minutes=evaluation_data["predicted_minutes"],
            team_ids=evaluation_data["team_ids"],
            verbose=False,
        )

        # DNP bucket should have some players
        assert result.count_by_bucket["dnp"] > 0
        assert result.mae_by_bucket["dnp"] is not None
        assert result.mae_by_bucket["dnp"] >= 0

    def test_bucket_counts_sum_to_total(self, evaluation_data):
        """Test that bucket counts are reasonable (may not cover all samples if >48 min)."""
        result = evaluate_minute_share_model(
            actual_minutes=evaluation_data["actual_minutes"],
            predicted_shares=evaluation_data["predicted_shares"],
            predicted_minutes=evaluation_data["predicted_minutes"],
            team_ids=evaluation_data["team_ids"],
            verbose=False,
        )

        total_bucket_count = sum(result.count_by_bucket.values())
        # Bucket counts may be less than n_samples if some players have >48 minutes
        # (after scaling to 240, stars can exceed the stars bucket upper bound)
        assert total_bucket_count <= result.n_samples
        # But should cover a reasonable portion
        assert total_bucket_count >= result.n_samples * 0.8

    def test_team_share_sum_diagnostics(self, evaluation_data):
        """Test that team share sum diagnostics are computed."""
        result = evaluate_minute_share_model(
            actual_minutes=evaluation_data["actual_minutes"],
            predicted_shares=evaluation_data["predicted_shares"],
            predicted_minutes=evaluation_data["predicted_minutes"],
            team_ids=evaluation_data["team_ids"],
            verbose=False,
        )

        # Since predicted shares were normalized per team, sum should be ~1.0
        assert result.team_share_sum_mean == pytest.approx(1.0, abs=0.01)
        assert result.team_share_sum_std >= 0

    def test_accepts_pandas_series(self, evaluation_data):
        """Test that function accepts pandas Series inputs."""
        result = evaluate_minute_share_model(
            actual_minutes=pd.Series(evaluation_data["actual_minutes"]),
            predicted_shares=pd.Series(evaluation_data["predicted_shares"]),
            predicted_minutes=pd.Series(evaluation_data["predicted_minutes"]),
            team_ids=pd.Series(evaluation_data["team_ids"]),
            verbose=False,
        )

        assert isinstance(result, MinuteShareEvaluation)
        assert result.n_samples == 30


class TestMaeBuckets:
    """Tests for minute bucket MAE computation."""

    def test_dnp_bucket_is_exact_zero(self):
        """Test that DNP bucket only includes exactly 0 minutes."""
        actual = np.array([0.0, 0.1, 5.0, 15.0, 25.0, 35.0])
        predicted = np.array([0.0, 1.0, 6.0, 16.0, 26.0, 36.0])
        team_ids = np.array(["A", "A", "A", "A", "A", "A"])

        result = evaluate_minute_share_model(
            actual_minutes=actual,
            predicted_shares=predicted / 240,  # Convert to shares
            predicted_minutes=predicted,
            team_ids=team_ids,
            verbose=False,
        )

        # Only one player at exactly 0 minutes
        assert result.count_by_bucket["dnp"] == 1
        assert result.mae_by_bucket["dnp"] == pytest.approx(0.0, abs=0.01)

    def test_garbage_time_bucket_range(self):
        """Test that garbage_time bucket covers (0, 10] minutes."""
        # Players at 5 and 8 minutes should be in garbage_time
        actual = np.array([5.0, 8.0, 12.0, 25.0])
        predicted = np.array([6.0, 7.0, 13.0, 24.0])
        team_ids = np.array(["A", "A", "A", "A"])

        result = evaluate_minute_share_model(
            actual_minutes=actual,
            predicted_shares=predicted / 240,
            predicted_minutes=predicted,
            team_ids=team_ids,
            verbose=False,
        )

        assert result.count_by_bucket["garbage_time"] == 2

    def test_empty_bucket_returns_none(self):
        """Test that empty buckets return None for MAE."""
        # Only rotation-level players, no stars
        actual = np.array([25.0, 26.0, 27.0])
        predicted = np.array([24.0, 25.0, 26.0])
        team_ids = np.array(["A", "A", "A"])

        result = evaluate_minute_share_model(
            actual_minutes=actual,
            predicted_shares=predicted / 240,
            predicted_minutes=predicted,
            team_ids=team_ids,
            verbose=False,
        )

        # Stars bucket (38-48) should be empty
        assert result.count_by_bucket["stars"] == 0
        assert result.mae_by_bucket["stars"] is None


class TestDNPClassification:
    """Tests for play/DNP classification metrics."""

    def test_perfect_dnp_classification(self):
        """Test metrics with perfect DNP classification."""
        actual = np.array([0.0, 0.0, 30.0, 30.0])
        # DNP players have low predicted shares, active have high
        predicted_shares = np.array([0.01, 0.01, 0.15, 0.15])
        predicted_minutes = predicted_shares * 240
        team_ids = np.array(["A", "A", "A", "A"])

        result = evaluate_minute_share_model(
            actual_minutes=actual,
            predicted_shares=predicted_shares,
            predicted_minutes=predicted_minutes,
            team_ids=team_ids,
            verbose=False,
        )

        # With default threshold of 0.02, first two are predicted DNP
        assert result.dnp_accuracy == 1.0
        assert result.dnp_precision == 1.0
        assert result.dnp_recall == 1.0

    def test_dnp_auc_with_mixed_results(self):
        """Test that AUC is computed for mixed predictions."""
        # Mix of DNP and active players
        actual = np.array([0.0, 0.0, 10.0, 20.0, 30.0, 35.0])
        predicted_shares = np.array([0.01, 0.03, 0.05, 0.10, 0.12, 0.14])
        predicted_minutes = predicted_shares * 240
        team_ids = np.array(["A", "A", "A", "A", "A", "A"])

        result = evaluate_minute_share_model(
            actual_minutes=actual,
            predicted_shares=predicted_shares,
            predicted_minutes=predicted_minutes,
            team_ids=team_ids,
            verbose=False,
        )

        # Should have valid AUC since both classes present
        assert result.dnp_auc is not None
        assert 0 <= result.dnp_auc <= 1


class TestRotationClassification:
    """Tests for rotation player classification metrics."""

    def test_rotation_threshold_default(self, evaluation_data):
        """Test that default rotation threshold is 15 minutes."""
        result = evaluate_minute_share_model(
            actual_minutes=evaluation_data["actual_minutes"],
            predicted_shares=evaluation_data["predicted_shares"],
            predicted_minutes=evaluation_data["predicted_minutes"],
            team_ids=evaluation_data["team_ids"],
            verbose=False,
        )

        # Rotation count should match players >= 15 min
        n_rotation_expected = (evaluation_data["actual_minutes"] >= 15.0).sum()
        assert result.n_rotation_actual == n_rotation_expected

    def test_rotation_classification_with_custom_threshold(self, evaluation_data):
        """Test rotation classification with custom threshold."""
        result = evaluate_minute_share_model(
            actual_minutes=evaluation_data["actual_minutes"],
            predicted_shares=evaluation_data["predicted_shares"],
            predicted_minutes=evaluation_data["predicted_minutes"],
            team_ids=evaluation_data["team_ids"],
            rotation_threshold=20.0,
            verbose=False,
        )

        # With higher threshold, fewer rotation players
        n_rotation_expected = (evaluation_data["actual_minutes"] >= 20.0).sum()
        assert result.n_rotation_actual == n_rotation_expected


class TestShareDistributionDiagnostics:
    """Tests for share distribution diagnostics."""

    def test_normalized_shares_sum_to_one(self):
        """Test diagnostics when shares are already normalized."""
        # Shares that already sum to 1.0 per team
        predicted_shares = np.array([0.4, 0.3, 0.2, 0.1])  # Team A sums to 1.0
        actual = np.array([35.0, 30.0, 20.0, 10.0])
        team_ids = np.array(["A", "A", "A", "A"])

        result = evaluate_minute_share_model(
            actual_minutes=actual,
            predicted_shares=predicted_shares,
            predicted_minutes=predicted_shares * 240,
            team_ids=team_ids,
            verbose=False,
        )

        assert result.team_share_sum_mean == pytest.approx(1.0, abs=0.001)
        # With single team, std should be 0 or NaN-like
        # (pandas std of single value is NaN, gets converted to 0 or ignored)

    def test_max_share_statistics(self, evaluation_data):
        """Test that max share statistics are computed per team."""
        result = evaluate_minute_share_model(
            actual_minutes=evaluation_data["actual_minutes"],
            predicted_shares=evaluation_data["predicted_shares"],
            predicted_minutes=evaluation_data["predicted_minutes"],
            team_ids=evaluation_data["team_ids"],
            verbose=False,
        )

        # Max share should be reasonable (star player ~0.15-0.20)
        assert 0.05 <= result.max_share_mean <= 0.40
        assert result.max_share_std >= 0


class TestEvaluationWithRealModel:
    """Integration tests using actual trained model."""

    def test_evaluation_after_train_predict(self, multi_team_data):
        """Test evaluation with actual model predictions."""
        df = multi_team_data[multi_team_data["minutes"] > 0].copy()

        X = df[["feature_1", "feature_2", "feature_3"]]
        y = df["minutes"]
        team_ids = df["team_id"]

        # Train model
        artifacts = train_minute_share_model(X, y, random_state=42)

        # Get predictions
        predictions = predict_minutes(artifacts, X, team_ids)

        # Evaluate
        result = evaluate_minute_share_model(
            actual_minutes=y.values,
            predicted_shares=predictions["raw_share"].values,
            predicted_minutes=predictions["predicted_minutes"].values,
            team_ids=team_ids.values,
            verbose=False,
        )

        # Basic sanity checks
        assert isinstance(result, MinuteShareEvaluation)
        assert result.n_samples == len(df)
        assert result.n_teams == 3
        assert result.mae_minutes >= 0
        assert result.mae_shares >= 0

        # Since model trains on this data, should have reasonable accuracy
        assert result.mae_minutes < 20  # Less than 20 min average error

    def test_evaluation_verbose_does_not_crash(self, multi_team_data, capsys):
        """Test that verbose mode runs without error."""
        df = multi_team_data[multi_team_data["minutes"] > 0].copy()

        X = df[["feature_1", "feature_2", "feature_3"]]
        y = df["minutes"]
        team_ids = df["team_id"]

        artifacts = train_minute_share_model(X, y, random_state=42)
        predictions = predict_minutes(artifacts, X, team_ids)

        # This should not raise
        result = evaluate_minute_share_model(
            actual_minutes=y.values,
            predicted_shares=predictions["raw_share"].values,
            predicted_minutes=predictions["predicted_minutes"].values,
            team_ids=team_ids.values,
            verbose=True,
        )

        assert isinstance(result, MinuteShareEvaluation)
