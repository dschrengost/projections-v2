from __future__ import annotations

from pathlib import Path

import torch

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


def test_delta_out_2_fill_mapping() -> None:
    model = RotationMinutesHurdleMLP(
        n_continuous=1,
        cat_cardinalities=[],
        emb_dim=8,
        hidden_dims=[],
        dropout=0.0,
        delta_out=2,
    )
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.q50_head.bias.fill_(2.0)
        model.delta_head.bias[:] = torch.tensor([0.1, 0.2])

    x_cont = torch.zeros((1, 1))
    x_cat = torch.zeros((1, 0), dtype=torch.long)
    out = model(x_cont, x_cat)

    q10 = out.q10_cond
    q50 = out.q50_cond
    q90 = out.q90_cond

    expected_q05 = q10 - (q50 - q10) * 0.125
    expected_q25 = q10 + (q50 - q10) * 0.375
    expected_q75 = q50 + (q90 - q50) * 0.625
    expected_q95 = q90 + (q90 - q50) * 0.125

    assert torch.allclose(out.q05_cond, expected_q05, atol=1e-6)
    assert torch.allclose(out.q25_cond, expected_q25, atol=1e-6)
    assert torch.allclose(out.q75_cond, expected_q75, atol=1e-6)
    assert torch.allclose(out.q95_cond, expected_q95, atol=1e-6)
