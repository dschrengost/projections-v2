"""End-to-end smoke tests for Minutes V1 CLI entrypoints."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from typer.testing import CliRunner

from projections.cli.score_minutes_v1 import app as score_v1_app
from projections.cli.sequential_backtest import app as backtest_app
from projections.metrics.minutes_metrics import app as metrics_app
from projections.minutes_v1 import modeling
from projections.minutes_v1.artifacts import compute_feature_hash


runner = CliRunner()


class _FakeModel:
    """Simple constant predictor used to stub LightGBM in tests."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, df: pd.DataFrame) -> np.ndarray:  # pragma: no cover - trivial
        return np.full(len(df), self.value, dtype=float)


def _write_feature_set(tmp_path: Path, *, season: int = 2024, month: int = 12) -> Path:
    data_root = tmp_path / "data"
    feature_dir = (
        data_root
        / "gold"
        / "features_minutes_v1"
        / f"season={season}"
        / f"month={month:02d}"
    )
    feature_dir.mkdir(parents=True)
    base_dates = [
        f"{season}-{month:02d}-01",
        f"{season}-{month:02d}-02",
    ]
    df = pd.DataFrame(
        {
            "game_id": [1001, 1002],
            "player_id": [1, 2],
            "team_id": [10, 20],
            "opponent_team_id": [30, 40],
            "season": [str(season), str(season)],
            "player_name": ["Player 1", "Player 2"],
            "team_name": ["Team 10", "Team 20"],
            "team_tricode": ["T10", "T20"],
            "opponent_team_name": ["Team 30", "Team 40"],
            "opponent_team_tricode": ["T30", "T40"],
            "minutes": [30.0, 18.0],
            "game_date": base_dates,
            "starter_prev_game_asof": [1, 0],
            "ramp_flag": [0, 0],
            "tip_ts": [f"{season}-{month:02d}-01T23:00:00Z", f"{season}-{month:02d}-02T23:00:00Z"],
            "feature_as_of_ts": [
                f"{season}-{month:02d}-01T18:00:00Z",
                f"{season}-{month:02d}-02T18:00:00Z",
            ],
            "injury_as_of_ts": [
                f"{season}-{month:02d}-01T17:00:00Z",
                f"{season}-{month:02d}-02T17:00:00Z",
            ],
            "odds_as_of_ts": [
                f"{season}-{month:02d}-01T19:00:00Z",
                f"{season}-{month:02d}-02T19:00:00Z",
            ],
            "blowout_index": [1.0, 1.0],
            "feat_a": [0.2, 0.8],
            "feat_b": [1.5, 0.3],
        }
    )
    df.to_parquet(feature_dir / "features.parquet", index=False)
    return data_root


def _write_stub_artifacts(
    tmp_path: Path,
    data_root: Path,
    run_id: str = "test_run",
    *,
    season: int = 2024,
    month: int = 12,
) -> Path:
    artifact_root = tmp_path / "artifacts" / "minutes_v1" / run_id
    artifact_root.mkdir(parents=True)
    feature_cols = ["feat_a", "feat_b"]
    imputer = SimpleImputer(strategy="mean")
    features_path = (
        data_root
        / "gold"
        / "features_minutes_v1"
        / f"season={season}"
        / f"month={month:02d}"
        / "features.parquet"
    )
    features_df = pd.read_parquet(features_path)
    imputer.fit(features_df[feature_cols])
    models = {q: _FakeModel(20 + 5 * q) for q in (0.1, 0.5, 0.9)}
    quantiles = modeling.QuantileArtifacts(models=models, imputer=imputer)
    calibrator = modeling.ConformalIntervalCalibrator(alpha_low=0.1, alpha_high=0.1)
    calibrator.fit(np.array([10.0, 12.0]), np.array([9.0, 11.0]), np.array([13.0, 15.0]))
    bundle = {
        "feature_columns": feature_cols,
        "quantiles": quantiles,
        "calibrator": calibrator,
        "bucket_offsets": {"__global__": {"d10": 0.0, "d90": 0.0, "n": len(features_df)}},
        "bucket_mode": "none",
        "conformal_mode": "tail-deltas",
        "play_probability": None,
    }
    joblib.dump(bundle, artifact_root / "lgbm_quantiles.joblib")
    meta = {"feature_hash": compute_feature_hash(feature_cols)}
    (artifact_root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return artifact_root


def test_score_minutes_v1_cli_writes_daily_artifacts(tmp_path):
    data_root = _write_feature_set(tmp_path, season=2025)
    bundle_dir = _write_stub_artifacts(tmp_path, data_root, season=2025)
    features_root = data_root / "gold" / "features_minutes_v1"
    daily_root = tmp_path / "artifacts" / "minutes_v1" / "daily"
    injuries_root = tmp_path / "data" / "bronze" / "injuries_raw"
    schedule_root = tmp_path / "data" / "silver" / "schedule"

    season_dir = injuries_root / "season=2025"
    season_dir.mkdir(parents=True, exist_ok=True)
    injuries_df = pd.DataFrame(
        {
            "report_date": ["2025-12-01", "2025-12-01"],
            "as_of_ts": ["2025-12-01T00:00:00Z"] * 2,
            "team_id": [10, 20],
            "player_name": ["Player 1", "Player 2"],
            "player_id": [1, 2],
            "status_raw": ["", ""],
            "notes_raw": ["", ""],
            "game_id": [1001, 1002],
            "ingested_ts": ["2025-12-01T00:00:00Z"] * 2,
            "source": ["test", "test"],
            "source_row_id": ["a", "b"],
            "status": ["OUT", "OK"],
            "restriction_flag": [False, False],
            "ramp_flag": [False, False],
            "games_since_return": [0, 0],
            "days_since_return": [0, 0],
        }
    )
    injuries_df.to_parquet(season_dir / "injuries.parquet", index=False)

    month_dir = schedule_root / "season=2025" / "month=12"
    month_dir.mkdir(parents=True, exist_ok=True)
    schedule_df = pd.DataFrame(
        {
            "home_team_id": [10],
            "home_team_name": ["Team Ten"],
            "home_team_tricode": ["TEN"],
            "away_team_id": [20],
            "away_team_name": ["Team Twenty"],
            "away_team_tricode": ["TTW"],
        }
    )
    schedule_df.to_parquet(month_dir / "schedule.parquet", index=False)

    result = runner.invoke(
        score_v1_app,
        [
            "--date",
            "2025-12-01",
            "--features-root",
            str(features_root),
            "--bundle-dir",
            str(bundle_dir),
            "--artifact-root",
            str(daily_root),
            "--injuries-root",
            str(injuries_root),
            "--schedule-root",
            str(schedule_root),
            "--disable-promotion-prior",
        ],
    )
    assert result.exit_code == 0, result.output

    parquet_path = daily_root / "2025-12-01" / "minutes.parquet"
    summary_path = parquet_path.with_name("summary.json")
    assert parquet_path.exists()
    assert summary_path.exists()
    frame = pd.read_parquet(parquet_path)
    assert not frame.empty
    assert {"minutes_p10", "minutes_p50", "minutes_p90", "player_name", "team_tricode"}.issubset(
        frame.columns
    )
    assert frame["player_name"].notna().all()
    with summary_path.open() as handle:
        summary = json.load(handle)
    assert summary["counts"]["rows"] == len(frame)


def test_minutes_metrics_emits_reports(tmp_path):
    report_root = tmp_path / "reports"
    data_root = tmp_path / "data"
    preds_dir = data_root / "preds" / "minutes_v1" / "2024-12"
    preds_dir.mkdir(parents=True)
    minutes = [35, 34, 33, 32, 31, 36, 34, 30, 28, 5]
    minutes_reconciled = [m - 1 for m in minutes]
    p10 = [40, 5, 5, 5, 5, 0, 0, 0, 0, 0]
    p90 = [40] * 9 + [4]
    spread_home = [2.0, 2.0, 2.0, 7.0, 7.0, 7.0, 12.0, 12.0, 12.0, 12.0]
    home_flag = [1] * 5 + [0] * 5
    injury_flags = [0] * 5 + [1] * 5
    df = pd.DataFrame(
        {
            "game_date": ["2024-12-01"] * 10,
            "game_id": ["2001"] * 10,
            "team_id": [10] * 5 + [20] * 5,
            "player_id": list(range(1, 11)),
            "minutes": minutes,
            "minutes_reconciled": minutes_reconciled,
            "p50_raw": [m + 0.5 for m in minutes],
            "p10_raw": p10,
            "p90_raw": p90,
            "p10": p10,
            "p90": p90,
            "tip_ts": ["2024-12-01T23:00:00Z"] * 10,
            "feature_as_of_ts": ["2024-12-01T19:00:00Z"] * 10,
            "starter_prev_game_asof": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "ramp_flag": [0] * 10,
            "injury_snapshot_missing": injury_flags,
            "spread_home": spread_home,
            "home_flag": home_flag,
            "home_team_tricode": ["ATL"] * 10,
            "away_team_tricode": ["BOS"] * 10,
        }
    )
    preds_path = preds_dir / "minutes_pred.parquet"
    df.to_parquet(preds_path, index=False)

    result = runner.invoke(
        metrics_app,
        [
            "--month",
            "2024-12",
            "--data-root",
            str(data_root),
            "--reports-root",
            str(report_root),
        ],
    )
    assert result.exit_code == 0, result.output
    metrics_path = report_root / "2024-12" / "metrics.csv"
    html_path = report_root / "2024-12" / "summary.html"
    assert metrics_path.exists()
    assert html_path.exists()
    report_dir = report_root / "2024-12"
    segment_files = [
        "segment_coverage_injury_snapshot_missing.csv",
        "segment_coverage_daily_injury_snapshot_missing.csv",
        "segment_coverage_spread_bucket.csv",
        "segment_coverage_daily_spread_bucket.csv",
        "segment_coverage_home_flag.csv",
        "segment_coverage_daily_home_flag.csv",
        "segment_coverage_team.csv",
        "segment_coverage_daily_team.csv",
    ]
    for filename in segment_files:
        assert (report_dir / filename).exists()

    injury_segments = pd.read_csv(report_dir / "segment_coverage_injury_snapshot_missing.csv")
    assert {"segment", "p10_ci_low", "warning"}.issubset(injury_segments.columns)
    assert set(injury_segments["segment"]) == {"injury_snapshot_missing=0", "injury_snapshot_missing=1"}
    warnings = injury_segments["warning"]
    if warnings.dtype != bool:
        warnings = warnings.astype(str).str.lower() == "true"
    assert warnings.any()

    html_text = html_path.read_text()
    assert "Segmented Coverage" in html_text
    assert "Coverage by injury_snapshot_missing" in html_text
    assert "segment_coverage_daily_injury_snapshot_missing.csv" in html_text
    assert "badge warning" in html_text


def test_minutes_metrics_enforces_coverage_thresholds(tmp_path):
    report_root = tmp_path / "reports"
    data_root = tmp_path / "data"
    preds_dir = data_root / "preds" / "minutes_v1" / "2024-12"
    preds_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "game_date": ["2024-12-01"] * 10,
            "game_id": ["2100"] * 10,
            "team_id": [10] * 10,
            "player_id": list(range(10)),
            "minutes": [20] * 10,
            "minutes_reconciled": [20] * 10,
            "p50_raw": [20] * 10,
            "p10_raw": [5] * 10,
            "p90_raw": [25] * 10,
            "p10": [5] * 10,
            "p90": [25] * 10,
            "tip_ts": ["2024-12-01T23:00:00Z"] * 10,
            "feature_as_of_ts": ["2024-12-01T21:00:00Z"] * 10,
            "injury_snapshot_missing": [0] * 10,
            "spread_home": [3.5] * 10,
            "home_flag": [1] * 10,
            "home_team_tricode": ["ATL"] * 10,
            "away_team_tricode": ["BOS"] * 10,
        }
    )
    preds_path = preds_dir / "minutes_pred.parquet"
    df.to_parquet(preds_path, index=False)

    result = runner.invoke(
        metrics_app,
        [
            "--month",
            "2024-12",
            "--data-root",
            str(data_root),
            "--reports-root",
            str(report_root),
        ],
    )
    assert result.exit_code != 0


def test_sequential_backtest_emits_artifacts(tmp_path):
    data_root = _write_feature_set(tmp_path)
    artifact_root = _write_stub_artifacts(tmp_path, data_root)
    reports_root = tmp_path / "reports" / "minutes_v1"

    result = runner.invoke(
        backtest_app,
        [
            "--start",
            "2024-12-01",
            "--end",
            "2024-12-02",
            "--run-id",
            "test_run",
            "--data-root",
            str(data_root),
            "--artifact-root",
            str(artifact_root.parent),
            "--season",
            "2024",
            "--month",
            "12",
            "--reports-root",
            str(reports_root),
            "--window-days",
            "1",
            "--min-n",
            "1",
            "--tau",
            "1.0",
        ],
    )
    assert result.exit_code == 0, result.output

    report_dir = reports_root / "2024-12"
    offsets_path = report_dir / "rolling_offsets.csv"
    p10_path = report_dir / "p10_coverage_daily.csv"
    p90_path = report_dir / "p90_coverage_daily.csv"
    summary_path = report_dir / "rolling_backtest_summary.json"

    assert offsets_path.exists()
    assert p10_path.exists()
    assert p90_path.exists()
    assert summary_path.exists()

    offsets = pd.read_csv(offsets_path)
    assert {"score_date", "window_start", "window_end", "delta_p10", "delta_p90"}.issubset(offsets.columns)

    p10_daily = pd.read_csv(p10_path)
    assert {"coverage_rolling", "coverage_global", "coverage_raw"}.issubset(p10_daily.columns)

    p90_daily = pd.read_csv(p90_path)
    assert {"coverage_rolling", "coverage_global", "coverage_raw"}.issubset(p90_daily.columns)

    summary = json.loads(summary_path.read_text())
    assert summary["run_id"] == "test_run"
    assert "p10" in summary and "p90" in summary
