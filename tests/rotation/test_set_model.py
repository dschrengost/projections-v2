from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from projections.rotation.set_model import (
    RotationSetModelConfig,
    build_model,
    predict_minutes,
    zfill_game_id_series,
)


def _write_toy_model_dir(tmp_path: Path, *, model_type: str) -> Path:
    feature_cols = ["f1", "f2", "f3"]
    config = RotationSetModelConfig(
        model=model_type,  # type: ignore[arg-type]
        feature_columns=feature_cols,
        feature_mean=[0.0, 0.0, 0.0],
        feature_std=[1.0, 1.0, 1.0],
        embed_dim=8,
        hidden_dim=16,
        dropout=0.0,
        num_transformer_layers=1,
        num_attention_heads=2,
    )
    model = build_model(config)
    for param in model.parameters():
        torch.nn.init.constant_(param, 0.0)

    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")
    config.save(model_dir / "config.json")
    return model_dir


def _toy_features_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1, 2, 2, 2, 2],
            "team_id": [10, 10, 10, 20, 20, 20, 20],
            "player_id": [101, 102, 103, 201, 202, 203, 204],
            "f1": [0.1, -0.2, 0.3, 0.0, 1.0, -1.0, 0.5],
            "f2": [2.0, 1.0, 0.5, 0.25, -0.25, 0.75, 1.25],
            "f3": [0, 1, 0, 1, 0, 1, 0],
        }
    )


@pytest.mark.parametrize("model_type", ["deepsets", "settransformer"])
def test_team_sums_equal_240(tmp_path: Path, model_type: str) -> None:
    model_dir = _write_toy_model_dir(tmp_path / model_type, model_type=model_type)
    df = _toy_features_df()

    scored = predict_minutes(df, model_dir=model_dir, device="cpu", batch_size=2)
    assert len(scored) == len(df)
    assert scored["pred_minutes"].notna().all()

    scored_norm = scored.assign(game_id_norm=zfill_game_id_series(scored["game_id"]))
    sums = scored_norm.groupby(["game_id_norm", "team_id"])["pred_minutes"].sum()
    np.testing.assert_allclose(sums.to_numpy(), np.full(len(sums), 240.0), rtol=0.0, atol=1e-4)


def test_output_rows_align_with_input_rows(tmp_path: Path) -> None:
    model_dir = _write_toy_model_dir(tmp_path, model_type="deepsets")
    df = _toy_features_df().sample(frac=1.0, random_state=0).reset_index(drop=True)

    scored = predict_minutes(df, model_dir=model_dir, device="cpu", batch_size=2)
    pd.testing.assert_series_equal(scored["game_id"], df["game_id"], check_dtype=False)
    pd.testing.assert_series_equal(scored["team_id"], df["team_id"], check_dtype=False)
    pd.testing.assert_series_equal(scored["player_id"], df["player_id"], check_dtype=False)

