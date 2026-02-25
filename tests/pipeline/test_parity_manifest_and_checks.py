from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from projections.pipeline.parity_checks import (
    ParityCheckError,
    validate_feature_distribution_contract,
    validate_feature_frame_against_manifest,
    validate_integrity_manifest,
    validate_projection_output_columns,
    validate_transform_manifest,
)
from projections.pipeline.parity_manifest import (
    build_parity_manifest,
    load_parity_manifest,
    resolve_parity_manifest_path,
    write_parity_manifest,
)


def _feature_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": pd.Series([1, 1], dtype="int64"),
            "team_id": pd.Series([100, 200], dtype="int64"),
            "player_id": pd.Series([10, 20], dtype="int64"),
            "x_usage": pd.Series([0.21, 0.19], dtype="float64"),
        }
    )


def test_parity_manifest_roundtrip_and_feature_validation(tmp_path: Path) -> None:
    features = _feature_df()
    manifest = build_parity_manifest(
        model_id="gtv2_test",
        features_df=features,
        transform_manifest={"builder": "unit_test", "scale": "none"},
        output_manifest={"projection_columns": ["game_id", "team_id", "player_id", "dk_fpts_mean"]},
        integrity={"git_sha": "abc123", "config_hash": "cfg", "artifact_hash": "art"},
    )

    path = resolve_parity_manifest_path(tmp_path)
    write_parity_manifest(path, manifest)
    loaded = load_parity_manifest(path)

    report = validate_feature_frame_against_manifest(features, loaded)
    assert report["row_count"] == 2
    assert report["column_count"] == 4

    t_report = validate_transform_manifest(loaded, {"builder": "unit_test", "scale": "none"})
    assert "transform_keys" in t_report

    i_report = validate_integrity_manifest(
        loaded,
        {"git_sha": "abc123", "config_hash": "cfg", "artifact_hash": "art"},
    )
    assert set(i_report["validated_keys"]) == {"git_sha", "config_hash", "artifact_hash"}


def test_parity_checks_fail_on_schema_order_dtype_and_transform_mismatch(tmp_path: Path) -> None:
    features = _feature_df()
    manifest = build_parity_manifest(
        model_id="gtv2_test",
        features_df=features,
        transform_manifest={"builder": "unit_test", "scale": "none"},
        output_manifest={"projection_columns": ["game_id", "team_id", "player_id", "dk_fpts_mean"]},
        integrity={"git_sha": "abc123", "config_hash": "cfg", "artifact_hash": "art"},
    )
    manifest_path = resolve_parity_manifest_path(tmp_path)
    write_parity_manifest(manifest_path, manifest)
    loaded = load_parity_manifest(manifest_path)

    reordered = features[["team_id", "game_id", "player_id", "x_usage"]]
    with pytest.raises(ParityCheckError):
        validate_feature_frame_against_manifest(reordered, loaded)

    bad_dtype = features.copy()
    bad_dtype["x_usage"] = bad_dtype["x_usage"].astype("float32")
    with pytest.raises(ParityCheckError):
        validate_feature_frame_against_manifest(bad_dtype, loaded)

    with pytest.raises(ParityCheckError):
        validate_transform_manifest(loaded, {"builder": "different"})


def test_projection_output_contract_validation() -> None:
    manifest = {
        "output_manifest": {
            "projection_columns": ["game_id", "team_id", "player_id", "dk_fpts_mean"],
        }
    }

    ok_df = pd.DataFrame(
        {
            "game_id": [1],
            "team_id": [100],
            "player_id": [10],
            "dk_fpts_mean": [34.2],
            "extra_col": [1],
        }
    )
    report = validate_projection_output_columns(ok_df, manifest)
    assert report["projection_row_count"] == 1

    missing_col = ok_df.drop(columns=["dk_fpts_mean"])
    with pytest.raises(ParityCheckError):
        validate_projection_output_columns(missing_col, manifest)


def test_feature_distribution_contract_pass_and_fail(tmp_path: Path) -> None:
    features = _feature_df()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "feature_columns": ["game_id", "team_id", "player_id", "x_usage"],
                "feature_mean": [1.0, 150.0, 15.0, 0.2],
                "feature_std": [1.0, 60.0, 8.0, 0.1],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    manifest = build_parity_manifest(
        model_id="gtv2_test",
        features_df=features,
        transform_manifest={"builder": "unit_test", "scale": "none"},
        output_manifest={"projection_columns": ["game_id", "team_id", "player_id", "dk_fpts_mean"]},
        integrity={"git_sha": "abc123", "config_hash": "cfg", "artifact_hash": "art"},
        missing_value_policy={
            "distribution_contract": {
                "enabled": True,
                "feature_limits": {
                    "x_usage": {"max_abs_mean_z": 2.0, "max_p95_abs_z": 3.0},
                },
                "conditional_limits": [
                    {
                        "name": "out_missing_rate",
                        "condition_col": "team_id",
                        "condition_eq": 100,
                        "metric_col": "x_usage",
                        "max_rate": 0.5,
                    }
                ],
            }
        },
    )

    # Pass case.
    report = validate_feature_distribution_contract(features, manifest, bundle_config_path=config_path)
    assert report["enabled"] is True
    assert "x_usage" in report["feature_report"]

    # Fail case on z-score gate.
    bad = features.copy()
    bad["x_usage"] = [8.0, 8.0]
    with pytest.raises(ParityCheckError):
        validate_feature_distribution_contract(bad, manifest, bundle_config_path=config_path)
