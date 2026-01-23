from __future__ import annotations

from pathlib import Path

from projections.models.minutes_features import MinutesFeatureSpec
from projections.models.minutes_nn import MinutesPreprocessorState
from projections.models.rotation_minutes_hurdle_v1.bundle import (
    load_bundle,
    save_bundle,
)
from projections.models.rotation_minutes_hurdle_v1.model import RotationMinutesHurdleMLP


def _write_rmh_bundle(tmp_path: Path, *, delta_out: int) -> Path:
    run_dir = tmp_path / f"rmh_delta_{delta_out}"
    feature_spec = MinutesFeatureSpec(continuous=["feat1"], categorical=[])
    preprocessor = MinutesPreprocessorState(
        continuous={"feat1": {"mean": 0.0, "std": 1.0}},
        categorical={},
    )
    model = RotationMinutesHurdleMLP(
        n_continuous=1,
        cat_cardinalities=[],
        emb_dim=8,
        hidden_dims=[256],
        dropout=0.0,
        delta_out=delta_out,
    )
    config = {"emb_dim": 8, "hidden_dims": [256], "dropout": 0.0}
    save_bundle(
        run_dir,
        model=model,
        feature_spec=feature_spec,
        preprocessor=preprocessor,
        config=config,
        metrics={},
    )
    return run_dir


def test_load_bundle_infers_delta_out_2(tmp_path: Path) -> None:
    run_dir = _write_rmh_bundle(tmp_path, delta_out=2)
    bundle = load_bundle(run_dir)
    assert bundle.model.delta_head.out_features == 2


def test_load_bundle_infers_delta_out_6(tmp_path: Path) -> None:
    run_dir = _write_rmh_bundle(tmp_path, delta_out=6)
    bundle = load_bundle(run_dir)
    assert bundle.model.delta_head.out_features == 6
