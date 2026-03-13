from __future__ import annotations

import numpy as np
import pandas as pd

from projections.ownership_v2 import (
    OwnershipSlateDataset,
    OwnershipSlateTransformer,
    OwnershipSlateTransformerConfig,
    collate_ownership_slates,
    merge_gtv2_embeddings,
)


def test_ownership_slate_dataset_builds_normalized_targets() -> None:
    df = pd.DataFrame(
        {
            "slate_id": ["s1", "s1", "s2"],
            "game_date": ["2026-01-01", "2026-01-01", "2026-01-02"],
            "player_id": [1, 2, 3],
            "player_name": ["A", "B", "C"],
            "team": ["X", "Y", "Z"],
            "pos": ["PG", "SG", "C"],
            "actual_own_pct": [60.0, 20.0, 40.0],
            "salary": [9000, 5000, 7000],
            "proj_fpts": [50.0, 25.0, 35.0],
        }
    )
    ds = OwnershipSlateDataset(df, feature_columns=["salary", "proj_fpts"])

    assert len(ds) == 2
    first = ds[0]
    assert np.isclose(first["target_share"].sum(), 1.0)
    assert first["features"].shape == (2, 2)


def test_collate_and_model_preserve_slate_sum_constraint() -> None:
    df = pd.DataFrame(
        {
            "slate_id": ["s1", "s1", "s1", "s2", "s2"],
            "game_date": ["2026-01-01"] * 3 + ["2026-01-02"] * 2,
            "player_id": [1, 2, 3, 4, 5],
            "player_name": ["A", "B", "C", "D", "E"],
            "team": ["X", "X", "Y", "Z", "Z"],
            "pos": ["PG", "SG", "PF", "C", "SF"],
            "actual_own_pct": [40.0, 20.0, 10.0, 30.0, 10.0],
            "salary": [9000, 7000, 5000, 8000, 4500],
            "proj_fpts": [45.0, 35.0, 22.0, 38.0, 18.0],
        }
    )
    ds = OwnershipSlateDataset(df, feature_columns=["salary", "proj_fpts"])
    batch = collate_ownership_slates([ds[0], ds[1]])

    config = OwnershipSlateTransformerConfig(
        feature_columns=["salary", "proj_fpts"],
        feature_mean=[0.0, 0.0],
        feature_std=[1.0, 1.0],
        d_model=16,
        num_heads=4,
        num_layers=1,
        hidden_dim=32,
        dropout=0.0,
        target_sum_pct=800.0,
        max_players=8,
    )
    model = OwnershipSlateTransformer(config)
    out = model(batch["features"], batch["valid_mask"])

    pred = out["pred_pct"].detach().numpy()
    valid = batch["valid_mask"].numpy()
    for i in range(pred.shape[0]):
        assert np.isclose(pred[i][valid[i]].sum(), 800.0, atol=1e-4)
        assert np.all(pred[i][~valid[i]] == 0.0)


def test_merge_gtv2_embeddings_zero_fills_missing_rows() -> None:
    base = pd.DataFrame(
        {
            "game_date": ["2026-01-01", "2026-01-01", "2026-01-02"],
            "player_id": [1, 2, 3],
            "value_per_k": [4.0, 5.0, 6.0],
        }
    )
    emb = pd.DataFrame(
        {
            "game_date": ["2026-01-01", "2026-01-02"],
            "player_id": [1, 3],
            "gtv2_state_000": [0.5, -0.2],
            "gtv2_minutes_deterministic": [34.0, 28.0],
        }
    )

    merged, cols, coverage = merge_gtv2_embeddings(base, emb)

    assert cols == ["gtv2_minutes_deterministic", "gtv2_state_000"]
    assert np.isclose(coverage, 2 / 3)
    assert np.isclose(float(merged.loc[merged["player_id"] == 2, "gtv2_state_000"].iloc[0]), 0.0)
