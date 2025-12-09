"""Tests for ownership calibration module."""

import numpy as np
import pandas as pd
import pytest

from projections.ownership_v1.calibration import (
    CalibrationParams,
    SoftmaxCalibrator,
    apply_calibration,
    fit_calibration,
)


class TestApplyCalibration:
    """Tests for apply_calibration function."""
    
    def test_sum_to_r_constraint(self):
        """Calibrated outputs should sum to R."""
        scores = np.array([10.0, 20.0, 30.0, 15.0, 25.0])
        params = CalibrationParams(a=0.1, b=0.0, R=8.0)
        
        calibrated = apply_calibration(scores, params)
        
        assert np.isclose(calibrated.sum(), 8.0), \
            f"Expected sum=8.0, got {calibrated.sum()}"
    
    def test_sum_to_r_various_params(self):
        """Sum-to-R should hold for various parameter values."""
        scores = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 12.0, 18.0])
        
        for a in [0.01, 0.1, 0.5, 1.0, 2.0]:
            for b in [-10.0, 0.0, 10.0]:
                for R in [8.0, 9.0, 6.0]:
                    params = CalibrationParams(a=a, b=b, R=R)
                    calibrated = apply_calibration(scores, params)
                    assert np.isclose(calibrated.sum(), R, rtol=1e-6), \
                        f"Failed for a={a}, b={b}, R={R}: sum={calibrated.sum()}"
    
    def test_ranking_preservation(self):
        """Ranking of outputs should match ranking of inputs."""
        scores = np.array([10.0, 30.0, 20.0, 5.0, 25.0])
        params = CalibrationParams(a=0.1, b=0.0, R=8.0)
        
        calibrated = apply_calibration(scores, params)
        
        # Argsort should be the same
        assert np.array_equal(np.argsort(scores), np.argsort(calibrated))
    
    def test_ranking_preservation_negative_a(self):
        """Ranking should be inverted when a < 0."""
        scores = np.array([10.0, 30.0, 20.0, 5.0, 25.0])
        params = CalibrationParams(a=-0.1, b=0.0, R=8.0)
        
        calibrated = apply_calibration(scores, params)
        
        # With negative a, higher scores get LOWER ownership
        assert np.array_equal(np.argsort(scores), np.argsort(-calibrated))
    
    def test_tail_stretching(self):
        """Higher 'a' should create more contrast (chalk gets more)."""
        scores = np.array([10.0, 20.0, 50.0])  # Clear top player
        
        # Low a (compressed)
        params_low = CalibrationParams(a=0.01, b=0.0, R=8.0)
        calibrated_low = apply_calibration(scores, params_low)
        
        # High a (stretched)
        params_high = CalibrationParams(a=0.2, b=0.0, R=8.0)
        calibrated_high = apply_calibration(scores, params_high)
        
        # Top player should get more ownership with high a
        assert calibrated_high[2] > calibrated_low[2], \
            f"Expected more chalk with high a: {calibrated_high[2]} > {calibrated_low[2]}"
        
        # Bottom player should get less with high a
        assert calibrated_high[0] < calibrated_low[0], \
            f"Expected less value with high a: {calibrated_high[0]} < {calibrated_low[0]}"
    
    def test_vs_naive_scaling(self):
        """Calibration should stretch tails vs naive linear scaling."""
        scores = np.array([5.0, 10.0, 15.0, 20.0, 50.0])  # One clear chalk
        R = 8.0
        
        # Naive scaling (just sum-normalize)
        naive = R * scores / scores.sum()
        
        # Calibration with positive a
        params = CalibrationParams(a=0.1, b=0.0, R=R)
        calibrated = apply_calibration(scores, params)
        
        # Both should sum to R
        assert np.isclose(naive.sum(), R)
        assert np.isclose(calibrated.sum(), R)
        
        # Top player should get MORE with calibration than naive
        assert calibrated[4] > naive[4], \
            f"Expected calibration to boost chalk: {calibrated[4]} > {naive[4]}"
    
    def test_empty_input(self):
        """Empty input should return empty output."""
        scores = np.array([])
        params = CalibrationParams(a=0.1, b=0.0, R=8.0)
        
        calibrated = apply_calibration(scores, params)
        
        assert len(calibrated) == 0


class TestFitCalibration:
    """Tests for fit_calibration function."""
    
    def test_basic_fitting(self):
        """Fitting should reduce MSE vs initial parameters."""
        # Create synthetic data where true a=0.15
        np.random.seed(42)
        
        slates = []
        for i in range(20):
            n_players = np.random.randint(30, 60)
            scores = np.random.uniform(5, 50, n_players)
            
            # Generate "true" ownership with known a=0.15, b=0
            true_params = CalibrationParams(a=0.15, b=0.0, R=8.0)
            targets = apply_calibration(scores, true_params)
            # Add some noise
            targets = targets * (1 + np.random.normal(0, 0.1, n_players))
            targets = np.clip(targets, 0.001, 2.0)
            
            slate_df = pd.DataFrame({
                "pred_own_pct": scores,
                "own_pct": targets * 100,  # Convert to percent
                "slate_id": f"slate_{i}",
            })
            slates.append(slate_df)
        
        df = pd.concat(slates, ignore_index=True)
        
        # Fit
        params = fit_calibration(
            df,
            score_col="pred_own_pct",
            target_col="own_pct",
            R=8.0,
            verbose=False,
        )
        
        # Fitted 'a' should be close to true value
        assert 0.05 < params.a < 0.3, f"Expected a near 0.15, got {params.a}"


class TestSoftmaxCalibrator:
    """Tests for SoftmaxCalibrator class."""
    
    def test_apply_df(self):
        """apply_df should add calibrated column to DataFrame."""
        df = pd.DataFrame({
            "player": ["A", "B", "C"],
            "pred_own_pct": [10.0, 20.0, 30.0],
        })
        
        calibrator = SoftmaxCalibrator(
            params=CalibrationParams(a=0.1, b=0.0, R=8.0)
        )
        
        result = calibrator.apply_df(df, score_col="pred_own_pct")
        
        assert "calibrated_own_pct" in result.columns
        assert np.isclose(result["calibrated_own_pct"].sum(), 800.0)  # In percent space
    
    def test_save_load(self, tmp_path):
        """Calibrator should serialize and deserialize correctly."""
        calibrator = SoftmaxCalibrator(
            params=CalibrationParams(a=0.123, b=-5.67, R=8.0),
            fitted=True,
        )
        
        save_path = tmp_path / "calibrator.json"
        calibrator.save(save_path)
        
        loaded = SoftmaxCalibrator.load(save_path)
        
        assert loaded.a == calibrator.a
        assert loaded.b == calibrator.b
        assert loaded.R == calibrator.R
        assert loaded.fitted == calibrator.fitted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
