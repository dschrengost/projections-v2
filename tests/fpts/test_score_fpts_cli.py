from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from projections.cli import score_fpts_v1
from projections.fpts_v1.datasets import FptsDatasetBuilder
from projections.models import fpts_lgbm

from tests.fpts.conftest import TEST_MINUTES_RUN_ID


def _build_minutes_output(dataset: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    player_names = {10: "Alpha Guard", 11: "Wing Bench"}
    for row in dataset.to_dict(orient="records"):
        game_date = pd.Timestamp(row["game_date"])
        records.append(
            {
                "game_date": game_date,
                "tip_ts": row["tip_ts"],
                "game_id": row["game_id"],
                "player_id": row["player_id"],
                "player_name": player_names.get(row["player_id"], f"P{row['player_id']}"),
                "status": row.get("status"),
                "team_id": row.get("team_id", 200),
                "team_name": "Home",
                "team_tricode": "HOM",
                "opponent_team_id": 300,
                "opponent_team_name": "Away",
                "opponent_team_tricode": "AWY",
                "starter_flag": row.get("starter_flag", 0),
                "pos_bucket": row.get("pos_bucket", "G"),
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


def test_score_fpts_cli_writes_outputs(sample_data_root: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    run_dir_root = tmp_path / "artifacts"
    result = runner.invoke(
        fpts_lgbm.app,
        [
            "--run-id",
            "testrun",
            "--data-root",
            str(sample_data_root),
            "--artifact-root",
            str(run_dir_root),
            "--train-start",
            "2024-10-01",
            "--train-end",
            "2024-10-01",
            "--cal-start",
            "2024-10-01",
            "--cal-end",
            "2024-10-01",
            "--minutes-run-id",
            TEST_MINUTES_RUN_ID,
        ],
    )
    assert result.exit_code == 0, result.output
    bundle_dir = run_dir_root / "testrun"

    builder = FptsDatasetBuilder(
        data_root=sample_data_root,
        history_days=1,
        minutes_source="predicted",
        minutes_run_id=TEST_MINUTES_RUN_ID,
    )
    dataset = builder.build(datetime(2024, 10, 1), datetime(2024, 10, 1))
    run_id = "testrun-live"
    date_text = "2024-10-01"

    live_features_dir = sample_data_root / "live" / "features_minutes_v1" / date_text / f"run={run_id}"
    live_features_dir.mkdir(parents=True, exist_ok=True)
    drop_cols = [
        col
        for col in dataset.columns
        if "prior" in col or col.startswith("actual_") or col.startswith("fpts_baseline")
    ]
    live_features = dataset.drop(columns=drop_cols)
    live_features.to_parquet(live_features_dir / "features.parquet", index=False)

    minutes_root = tmp_path / "minutes_daily"
    minutes_run_dir = minutes_root / date_text / f"run={run_id}"
    minutes_run_dir.mkdir(parents=True, exist_ok=True)
    minutes_df = _build_minutes_output(dataset)
    minutes_df.to_parquet(minutes_run_dir / "minutes.parquet", index=False)
    (minutes_run_dir / "summary.json").write_text(
        json.dumps({"run_id": run_id, "date": date_text}), encoding="utf-8"
    )

    out_root = tmp_path / "fpts_out"
    result = runner.invoke(
        score_fpts_v1.app,
        [
            "--date",
            date_text,
            "--run-id",
            run_id,
            "--minutes-root",
            str(minutes_root),
            "--live-features-root",
            str(sample_data_root / "live" / "features_minutes_v1"),
            "--out-root",
            str(out_root),
            "--data-root",
            str(sample_data_root),
        ],
        env={
            "FPTS_PRODUCTION_DIR": str(bundle_dir),
            "FPTS_PRODUCTION_RUN_ID": "testrun",
        },
    )
    assert result.exit_code == 0, result.output
    output_parquet = out_root / date_text / f"run={run_id}" / "fpts.parquet"
    assert output_parquet.exists()
    scored = pd.read_parquet(output_parquet)
    assert {"fpts_per_min_pred", "proj_fpts", "scoring_system"}.issubset(scored.columns)
    assert (out_root / date_text / "latest_run.json").exists()
