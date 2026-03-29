from __future__ import annotations

import pandas as pd
import torch

from projections.pipeline.gtv2_inference_runtime import score_gtv2_features_df
from projections.rotation.game_transformer_v2 import GameTransformerV2Config


def test_score_gtv2_features_df_hard_zeros_out_players() -> None:
    class _Out:
        def __init__(self, bsz: int) -> None:
            minutes = torch.zeros((bsz, 30), dtype=torch.float32)
            minutes[:, :2] = 24.0
            minutes[:, 15:17] = 24.0
            active_mask = torch.zeros((bsz, 30), dtype=torch.bool)
            active_mask[:, :2] = True
            active_mask[:, 15:17] = True
            player_logits = torch.full((bsz, 30), 8.0, dtype=torch.float32)
            self.minutes = type("Minutes", (), {"minutes": minutes})()
            self.active = type(
                "Active",
                (),
                {"active_mask": active_mask, "player_logits": player_logits},
            )()

    class _Model:
        def __call__(
            self,
            player_features: torch.Tensor,
            player_valid_mask: torch.Tensor,
            **_: object,
        ) -> _Out:
            return _Out(int(player_features.shape[0]))

    features_df = pd.DataFrame(
        [
            {"game_id": 1, "team_id": 10, "player_id": 101, "is_out": 1, "lineup_available": 1},
            {"game_id": 1, "team_id": 10, "player_id": 102, "is_out": 0, "lineup_available": 1},
            {"game_id": 1, "team_id": 20, "player_id": 201, "is_out": 0, "lineup_available": 1},
            {"game_id": 1, "team_id": 20, "player_id": 202, "is_out": 0, "lineup_available": 1},
        ]
    )
    config = GameTransformerV2Config(
        feature_columns=["is_out"],
        feature_mean=[0.0],
        feature_std=[1.0],
        game_feature_columns=[],
        team_feature_columns=[],
        min_active_count=1,
    )

    scores = score_gtv2_features_df(
        features_df=features_df,
        game_date="2026-03-27",
        config=config,
        model=_Model(),
        device=torch.device("cpu"),
        batch_size=2,
    )

    out_row = scores.loc[scores["player_id"] == 101].iloc[0]
    assert float(out_row["minutes_deterministic"]) == 0.0
    assert int(out_row["active_deterministic"]) == 0
    assert float(out_row["active_prob_proxy"]) == 0.0

    in_row = scores.loc[scores["player_id"] == 102].iloc[0]
    assert float(in_row["minutes_deterministic"]) > 0.0
    assert int(in_row["active_deterministic"]) == 1
