from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from projections.pipeline.parity_manifest import (
    build_parity_manifest,
    write_parity_manifest,
)
from projections.pipeline.v3_postflight import V3PostflightError, run_postflight_gate
from projections.pipeline.v3_preflight import V3PreflightError, run_preflight_gate
from projections.pipeline import writer_guard


def _features_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": pd.Series([1, 1], dtype="int64"),
            "team_id": pd.Series([100, 200], dtype="int64"),
            "player_id": pd.Series([11, 22], dtype="int64"),
            "x": pd.Series([0.1, 0.2], dtype="float64"),
        }
    )


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_v3_preflight_pass_and_fail_cases(tmp_path: Path) -> None:
    features = _features_df()
    features_path = (
        tmp_path
        / "live"
        / "features_gtv2_v1"
        / "2026-01-18"
        / "run=test"
        / "features.parquet"
    )
    _write_parquet(features_path, features)

    manifest = build_parity_manifest(
        model_id="gtv2_test",
        features_df=features,
        transform_manifest={"builder": "unit", "scale": "none"},
        output_manifest={
            "projection_columns": ["game_id", "team_id", "player_id", "dk_fpts_mean"]
        },
        integrity={"git_sha": "abc", "config_hash": "cfg", "artifact_hash": "art"},
    )
    parity_manifest_path = tmp_path / "bundle" / "parity_manifest.json"
    write_parity_manifest(parity_manifest_path, manifest)

    required_input = (
        tmp_path
        / "bronze"
        / "v3_core_inputs"
        / "date=2026-01-18"
        / "core_inputs_ready.json"
    )
    required_input.parent.mkdir(parents=True, exist_ok=True)
    required_input.write_text("{}", encoding="utf-8")

    score_run_dir = (
        tmp_path / "artifacts" / "gtv2_scores" / "game_date=2026-01-18" / "run=test"
    )
    worlds_run_dir = (
        tmp_path / "artifacts" / "gtv2_worlds" / "game_date=2026-01-18" / "run=test"
    )

    as_of_ts = pd.Timestamp.utcnow().isoformat()

    report = run_preflight_gate(
        as_of_ts=as_of_ts,
        required_inputs={"core_inputs": required_input, "features": features_path},
        run_dirs=[score_run_dir, worlds_run_dir],
        features_path=features_path,
        parity_manifest_path=parity_manifest_path,
        observed_transform_manifest={"builder": "unit", "scale": "none"},
        observed_integrity={
            "git_sha": "abc",
            "config_hash": "cfg",
            "artifact_hash": "art",
        },
        input_max_age_minutes=1_000_000.0,
    )
    assert "feature_report" in report

    # Dirty run dir should fail.
    dirty_dir = tmp_path / "artifacts" / "dirty"
    dirty_dir.mkdir(parents=True, exist_ok=True)
    (dirty_dir / "old.parquet").write_text("x", encoding="utf-8")
    with pytest.raises(V3PreflightError):
        run_preflight_gate(
            as_of_ts=as_of_ts,
            required_inputs={"core_inputs": required_input, "features": features_path},
            run_dirs=[dirty_dir],
            features_path=features_path,
            parity_manifest_path=parity_manifest_path,
            observed_transform_manifest={"builder": "unit", "scale": "none"},
            observed_integrity={
                "git_sha": "abc",
                "config_hash": "cfg",
                "artifact_hash": "art",
            },
            input_max_age_minutes=1_000_000.0,
        )

    with pytest.raises(V3PreflightError):
        run_preflight_gate(
            as_of_ts=as_of_ts,
            required_inputs={"core_inputs": required_input, "features": features_path},
            run_dirs=[score_run_dir, worlds_run_dir],
            features_path=features_path,
            parity_manifest_path=parity_manifest_path,
            observed_transform_manifest={"builder": "unit", "scale": "none"},
            observed_integrity={
                "git_sha": "abc",
                "config_hash": "cfg",
                "artifact_hash": "art",
            },
            input_max_age_minutes=1_000_000.0,
            frozen_freshness_gates={
                "lock_window": {
                    "ok": False,
                    "failures": [{"game_id": 1, "window": "last_30"}],
                }
            },
        )


def test_v3_postflight_pass_and_fail_cases(tmp_path: Path) -> None:
    projections = pd.DataFrame(
        {
            "game_date": ["2026-01-18", "2026-01-18"],
            "game_id": [1, 1],
            "team_id": [100, 200],
            "player_id": [11, 22],
            "dk_fpts_mean": [30.0, 32.0],
        }
    )
    projections_path = (
        tmp_path
        / "artifacts"
        / "projections"
        / "2026-01-18"
        / "run=test"
        / "projections.parquet"
    )
    _write_parquet(projections_path, projections)

    feature_df = pd.DataFrame(
        {
            "game_id": [1],
            "team_id": [100],
            "player_id": [11],
        }
    )
    parity_manifest = build_parity_manifest(
        model_id="gtv2_test",
        features_df=feature_df,
        transform_manifest={"builder": "unit"},
        output_manifest={
            "projection_columns": [
                "game_date",
                "game_id",
                "team_id",
                "player_id",
                "dk_fpts_mean",
            ]
        },
        integrity={"git_sha": "abc", "config_hash": "cfg", "artifact_hash": "art"},
    )
    parity_manifest_path = tmp_path / "bundle" / "parity_manifest.json"
    write_parity_manifest(parity_manifest_path, parity_manifest)

    world_summary_path = (
        tmp_path
        / "artifacts"
        / "gtv2_worlds"
        / "game_date=2026-01-18"
        / "run=test"
        / "world_contracts_summary.json"
    )
    world_summary_path.parent.mkdir(parents=True, exist_ok=True)
    world_summary_path.write_text(
        json.dumps(
            {
                "contract_checks": {
                    "team_minutes_not_240": 0,
                    "minutes_negative": 0,
                    "minutes_over_48": 0,
                    "negative_stats": 0,
                    "fg2m_gt_fga2": 0,
                    "fg3m_gt_fga3": 0,
                    "ftm_gt_fta": 0,
                    "inactive_nonzero_stats": 0,
                    "inactive_nonzero_fpts_proxy": 0,
                }
            }
        ),
        encoding="utf-8",
    )

    with writer_guard.PipelineWriterLock(data_root=tmp_path, run_id="test"):
        report = run_postflight_gate(
            projections_path=projections_path,
            parity_manifest_path=parity_manifest_path,
            world_contract_summary_path=world_summary_path,
            key_columns=("game_id", "team_id", "player_id"),
            min_rows=2,
        )
    assert report["projection_row_report"]["row_count"] == 2

    world_summary_path.write_text(
        json.dumps({"contract_checks": {"team_minutes_not_240": 1}}),
        encoding="utf-8",
    )
    with writer_guard.PipelineWriterLock(data_root=tmp_path, run_id="test2"):
        with pytest.raises(V3PostflightError):
            run_postflight_gate(
                projections_path=projections_path,
                parity_manifest_path=parity_manifest_path,
                world_contract_summary_path=world_summary_path,
                key_columns=("game_id", "team_id", "player_id"),
                min_rows=2,
            )
