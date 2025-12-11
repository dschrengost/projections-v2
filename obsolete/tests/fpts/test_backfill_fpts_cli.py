from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import joblib
import numpy as np
from typer.testing import CliRunner

from projections.cli import backfill_fpts_v1
from projections.cli import score_fpts_v1
from projections.cli.score_fpts_v1 import OUTPUT_FILENAME

from tests.fpts.conftest import TEST_MINUTES_RUN_ID


class ConstantRegressor:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.full(matrix.shape[0], self.value, dtype=float)


class ZeroFillImputer:
    def fit(self, matrix: np.ndarray) -> "ZeroFillImputer":
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return np.nan_to_num(matrix, nan=0.0)


def _stub_dataset(game_date: str) -> pd.DataFrame:
    day = pd.Timestamp(game_date)
    tip_ts = pd.Timestamp(f"{game_date}T23:00:00Z")
    feature_ts = pd.Timestamp(f"{game_date}T18:00:00Z")
    return pd.DataFrame(
        [
            {
                "game_date": day,
                "tip_ts": tip_ts,
                "game_id": 1001,
                "player_id": 10,
                "team_id": 200,
                "starter_flag": 1,
                "pos_bucket": "G",
                "status": "available",
                "lineup_role": "starter",
                "play_prob_pred": 0.95,
                "minutes_p10_pred": 30.0,
                "minutes_p50_pred": 33.0,
                "minutes_p90_pred": 37.0,
                "minutes_p10": 30.0,
                "minutes_p50": 33.0,
                "minutes_p90": 37.0,
                "feature_as_of_ts": feature_ts,
                "total": 225.0,
                "spread_home": -5.5,
                "home_flag": 1,
            },
            {
                "game_date": day,
                "tip_ts": tip_ts,
                "game_id": 1001,
                "player_id": 11,
                "team_id": 200,
                "starter_flag": 0,
                "pos_bucket": "W",
                "status": "questionable - ankle",
                "lineup_role": "bench",
                "play_prob_pred": 0.25,
                "minutes_p10_pred": 14.0,
                "minutes_p50_pred": 18.0,
                "minutes_p90_pred": 23.0,
                "minutes_p10": 14.0,
                "minutes_p50": 18.0,
                "minutes_p90": 23.0,
                "feature_as_of_ts": feature_ts,
                "total": 225.0,
                "spread_home": -5.5,
                "home_flag": 1,
            },
        ]
    )


def _build_minutes_output(dataset: pd.DataFrame) -> pd.DataFrame:
    player_names = {10: "Alpha Guard", 11: "Wing Bench"}
    records: list[dict] = []
    for row in dataset.to_dict(orient="records"):
        records.append(
            {
                "game_date": row["game_date"],
                "tip_ts": row["tip_ts"],
                "game_id": row["game_id"],
                "player_id": row["player_id"],
                "player_name": player_names.get(row["player_id"], f"P{row['player_id']}"),
                "team_id": row["team_id"],
                "team_name": "Home",
                "team_tricode": "HOM",
                "opponent_team_id": 300,
                "opponent_team_name": "Away",
                "opponent_team_tricode": "AWY",
                "starter_flag": row.get("starter_flag", 0),
                "pos_bucket": row.get("pos_bucket", "G"),
                "status": row.get("status"),
                "play_prob": row.get("play_prob_pred", 1.0),
                "minutes_p10": row.get("minutes_p10_pred", 0.0),
                "minutes_p50": row.get("minutes_p50_pred", 0.0),
                "minutes_p90": row.get("minutes_p90_pred", 0.0),
                "minutes_p10_cond": row.get("minutes_p10_pred", 0.0),
                "minutes_p50_cond": row.get("minutes_p50_pred", 0.0),
                "minutes_p90_cond": row.get("minutes_p90_pred", 0.0),
                "minutes_p10_uncond": row.get("minutes_p10_pred", 0.0),
                "minutes_p50_uncond": row.get("minutes_p50_pred", 0.0),
                "minutes_p90_uncond": row.get("minutes_p90_pred", 0.0),
            }
        )
    return pd.DataFrame(records)


def _write_minutes_artifacts(
    *,
    base_dir: Path,
    features_root: Path,
    run_name: str,
    dataset: pd.DataFrame,
) -> None:
    day = pd.Timestamp(dataset["game_date"].iloc[0]).date()
    day_dir = base_dir / day.isoformat()
    run_dir = day_dir / f"run={run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    minutes_df = _build_minutes_output(dataset)
    minutes_df.to_parquet(run_dir / score_fpts_v1.MINUTES_FILENAME, index=False)
    summary = {
        "run_id": run_name,
        "model_run_id": TEST_MINUTES_RUN_ID,
        "run_as_of_ts": "2024-10-01T12:00:00Z",
    }
    (run_dir / score_fpts_v1.MINUTES_SUMMARY).write_text(json.dumps(summary), encoding="utf-8")
    pointer = day_dir / score_fpts_v1.LATEST_POINTER
    pointer.write_text(json.dumps({"run_id": run_name}), encoding="utf-8")

    feature_dir = features_root / day.isoformat() / f"run={run_name}"
    feature_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(feature_dir / score_fpts_v1.FEATURE_FILENAME, index=False)


def test_backfill_cli_runs_and_skips(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    dataset_day1 = _stub_dataset("2024-10-01")
    dataset_day2 = _stub_dataset("2024-10-02")
    feature_columns = ["minutes_p50_pred"]
    feature_matrix = dataset_day1[feature_columns].fillna(0.0).to_numpy(dtype=float)
    model = ConstantRegressor(0.75)
    imputer = ZeroFillImputer().fit(feature_matrix)
    run_dir = artifact_root / "testrun"
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "imputer": imputer,
            "feature_columns": feature_columns,
            "metadata": {"run_id": "testrun", "scoring_system": "dk"},
        },
        run_dir / "model.joblib",
    )
    minutes_root = tmp_path / "minutes_daily"
    features_root = tmp_path / "live_features"
    features_root.mkdir(parents=True, exist_ok=True)
    out_root = tmp_path / "fpts_out"

    datasets = [dataset_day1, dataset_day2]

    for idx, dataset in enumerate(datasets):
        run_name = f"{TEST_MINUTES_RUN_ID}_{idx}"
        _write_minutes_artifacts(
            base_dir=minutes_root,
            features_root=features_root,
            run_name=run_name,
            dataset=dataset,
        )

    # Pre-create an output for the first day to test skip logic.
    existing = out_root / "2024-10-01" / f"run={TEST_MINUTES_RUN_ID}_0"
    existing.mkdir(parents=True, exist_ok=True)
    (existing / OUTPUT_FILENAME).write_text("existing", encoding="utf-8")

    result = runner.invoke(
        backfill_fpts_v1.app,
        [
            "--start-date",
            "2024-10-01",
            "--end-date",
            "2024-10-02",
            "--minutes-run-id",
            TEST_MINUTES_RUN_ID,
            "--fpts-run-id",
            "testrun",
            "--fpts-artifact-root",
            str(artifact_root),
            "--minutes-root",
            str(minutes_root),
            "--live-features-root",
            str(features_root),
            "--out-root",
            str(out_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (existing / OUTPUT_FILENAME).read_text(encoding="utf-8") == "existing"

    new_output = out_root / "2024-10-02" / f"run={TEST_MINUTES_RUN_ID}_1" / OUTPUT_FILENAME
    assert new_output.exists()
    scored = pd.read_parquet(new_output)
    assert {"proj_fpts", "fpts_per_min_pred"}.issubset(scored.columns)
