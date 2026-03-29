from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from projections.rotation.tree_rate_bundle import (
    load_tree_rate_bundle,
    score_tree_rate_bundle_features,
    train_tree_rate_bundle,
)


def test_train_and_score_tree_rate_bundle_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    labels = []
    for day_idx in range(12):
        game_date = f"2026-02-{day_idx + 1:02d}"
        for player_id in range(4):
            game_id = 1000 + day_idx
            team_id = 10 if player_id < 2 else 20
            feat_a = float(player_id + 1)
            feat_b = float(day_idx + 1)
            an_ast_line = float(4 + player_id)
            an_reb_line = float(6 + player_id)
            rows.append(
                {
                    "game_date": game_date,
                    "game_id": game_id,
                    "team_id": team_id,
                    "player_id": player_id + 100,
                    "feat_a": feat_a,
                    "feat_b": feat_b,
                    "an_ast_line": an_ast_line,
                    "an_reb_line": an_reb_line,
                    "player_name": f"Player {player_id}",
                }
            )
            labels.append(
                {
                    "game_date": game_date,
                    "game_id": game_id,
                    "team_id": team_id,
                    "player_id": player_id + 100,
                    "ast_per_min": 0.05 * feat_a + 0.002 * feat_b + 0.01 * rng.random(),
                    "oreb_per_min": 0.02 * feat_a + 0.005 * rng.random(),
                    "dreb_per_min": 0.04 * feat_a + 0.003 * feat_b + 0.005 * rng.random(),
                }
            )

    pd.DataFrame(rows).to_parquet(dataset_dir / "features.parquet", index=False)
    pd.DataFrame(labels).to_parquet(dataset_dir / "labels_rates.parquet", index=False)

    bundle_dir = tmp_path / "bundle"
    metadata = train_tree_rate_bundle(
        dataset_dir=dataset_dir,
        bundle_dir=bundle_dir,
        model_type="lgbm",
        target_set="astreb",
        cal_days=3,
        num_boost_round=20,
        ast_weight_mult=1.5,
        reb_weight_mult=1.5,
        live_feature_columns={"game_date", "game_id", "team_id", "player_id", "feat_a", "feat_b", "an_ast_line", "an_reb_line", "player_name"},
    )

    assert metadata["model_type"] == "lgbm"
    assert (bundle_dir / "bundle_meta.json").exists()

    bundle = load_tree_rate_bundle(bundle_dir)
    live_features = pd.DataFrame(
        {
            "game_date": ["2026-03-29", "2026-03-29"],
            "game_id": [2001, 2001],
            "team_id": [10, 20],
            "player_id": [100, 102],
            "feat_a": [1.0, 3.0],
            "feat_b": [20.0, 20.0],
            "an_ast_line": [8.0, 6.0],
            "an_reb_line": [11.0, 9.0],
            "player_name": ["Alpha", "Bravo"],
        }
    )
    scored = score_tree_rate_bundle_features(
        features_df=live_features,
        bundle=bundle,
        include_extra_cols=["player_name"],
    )

    assert list(scored["player_name"]) == ["Alpha", "Bravo"]
    assert {"pred_ast_per_min", "pred_oreb_per_min", "pred_dreb_per_min"}.issubset(scored.columns)
    assert (scored[["pred_ast_per_min", "pred_oreb_per_min", "pred_dreb_per_min"]] >= 0.0).all().all()
