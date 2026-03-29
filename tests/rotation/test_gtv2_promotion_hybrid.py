from __future__ import annotations

import pytest
import torch

from projections.rotation.game_transformer_v2 import FLOW_TARGET_COLUMNS_V1, GameTransformerV2Config
from projections.rotation.gtv2_promotion_hybrid import (
    BenchRiserHybridConfig,
    PromotionHybridConfig,
    assert_promotion_hybrid_compatible,
    blend_expert_predictions,
    blend_promotion_predictions,
    compute_bench_riser_candidate_mask,
    compute_starter_promotion_candidate_mask,
)
from projections.rotation.sample_worlds_v2 import sample_worlds_for_batch


def _team_index(num_rows: int) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros((num_rows, 15), dtype=torch.long),
            torch.ones((num_rows, 15), dtype=torch.long),
        ],
        dim=1,
    )


def test_compute_starter_promotion_candidate_mask_uses_decoded_raw_thresholds() -> None:
    cfg = GameTransformerV2Config(
        feature_columns=[
            "minutes_from_stints_prior_20",
            "recent_start_pct_10",
            "started_proxy_rate_prior_10",
            "started_proxy_rate_prior_20",
        ],
        feature_mean=[10.0, 0.30, 0.20, 0.25],
        feature_std=[2.0, 0.10, 0.10, 0.10],
        game_feature_columns=[],
        team_feature_columns=[],
    )
    hybrid_cfg = PromotionHybridConfig.from_model_config(cfg, prior_minutes_max=12.0, hist_start_rate_max=0.20)

    player_features = torch.zeros((1, 2, 15, 4), dtype=torch.float32)
    player_valid_mask = torch.zeros((1, 2, 15), dtype=torch.bool)
    starter_hint_mask = torch.zeros((1, 2, 15), dtype=torch.bool)
    player_valid_mask[:, 0, :3] = True
    starter_hint_mask[:, 0, :3] = True

    # Candidate 0 decodes to prior_minutes=11 and hist_start_rate=max(0.15,0.10,0.05)=0.15.
    player_features[0, 0, 0, 0] = 0.5
    player_features[0, 0, 0, 1] = -1.5
    player_features[0, 0, 0, 2] = -1.0
    player_features[0, 0, 0, 3] = -2.0
    # Candidate 1 exceeds prior-minutes threshold (14).
    player_features[0, 0, 1, 0] = 2.0
    player_features[0, 0, 1, 1] = -1.0
    # Candidate 2 exceeds hist-start threshold (0.35).
    player_features[0, 0, 2, 0] = 0.0
    player_features[0, 0, 2, 1] = 0.5

    mask = compute_starter_promotion_candidate_mask(
        player_features=player_features,
        player_valid_mask=player_valid_mask,
        starter_hint_mask=starter_hint_mask,
        config=hybrid_cfg,
    )
    assert bool(mask[0, 0, 0]) is True
    assert bool(mask[0, 0, 1]) is False
    assert bool(mask[0, 0, 2]) is False


def test_blend_promotion_predictions_uplift_only_preserves_non_candidates() -> None:
    baseline_minutes = torch.tensor([[6.0, 20.0, 18.0]], dtype=torch.float32)
    baseline_active = torch.tensor([[False, True, True]])
    expert_minutes = torch.tensor([[28.0, 8.0, 22.0]], dtype=torch.float32)
    expert_active = torch.tensor([[True, False, True]])
    candidate_mask = torch.tensor([[True, False, False]])

    blended_minutes, blended_active = blend_promotion_predictions(
        baseline_minutes=baseline_minutes,
        baseline_active_mask=baseline_active,
        expert_minutes=expert_minutes,
        expert_active_mask=expert_active,
        promotion_candidate_mask=candidate_mask,
        uplift_only=True,
    )
    assert torch.allclose(blended_minutes, torch.tensor([[28.0, 20.0, 18.0]], dtype=torch.float32))
    assert torch.equal(blended_active, torch.tensor([[True, True, True]]))


def test_blend_promotion_predictions_can_force_candidates_active() -> None:
    blended_minutes, blended_active = blend_promotion_predictions(
        baseline_minutes=torch.tensor([[6.0, 20.0]], dtype=torch.float32),
        baseline_active_mask=torch.tensor([[False, True]]),
        expert_minutes=torch.tensor([[8.0, 20.0]], dtype=torch.float32),
        expert_active_mask=torch.tensor([[False, True]]),
        promotion_candidate_mask=torch.tensor([[True, False]]),
        uplift_only=True,
        force_active_candidates=True,
    )
    assert torch.allclose(blended_minutes, torch.tensor([[8.0, 20.0]], dtype=torch.float32))
    assert torch.equal(blended_active, torch.tensor([[True, True]]))


def test_compute_bench_riser_candidate_mask_uses_decoded_raw_thresholds() -> None:
    cfg = GameTransformerV2Config(
        feature_columns=[
            "minutes_from_stints_prior_20",
            "prior_play_prob",
            "an_implied_minutes",
            "recent_start_pct_10",
            "started_proxy_rate_prior_10",
            "started_proxy_rate_prior_20",
        ],
        feature_mean=[10.0, 0.70, 11.0, 0.30, 0.20, 0.25],
        feature_std=[2.0, 0.10, 2.0, 0.10, 0.10, 0.10],
        game_feature_columns=[],
        team_feature_columns=[],
    )
    hybrid_cfg = BenchRiserHybridConfig.from_model_config(
        cfg,
        prior_minutes_min=12.0,
        prior_play_prob_min=0.80,
        implied_minutes_min=12.0,
        hist_start_rate_max=0.35,
    )

    player_features = torch.zeros((1, 2, 15, 6), dtype=torch.float32)
    player_valid_mask = torch.zeros((1, 2, 15), dtype=torch.bool)
    starter_hint_mask = torch.zeros((1, 2, 15), dtype=torch.bool)
    player_valid_mask[:, 0, :4] = True

    player_features[0, 0, 0, 0] = 2.0
    player_features[0, 0, 0, 1] = 1.5
    player_features[0, 0, 0, 2] = 1.0
    player_features[0, 0, 0, 3:] = -1.0

    player_features[0, 0, 1, 0] = 0.0
    player_features[0, 0, 1, 1] = 1.5
    player_features[0, 0, 1, 2] = 1.0
    player_features[0, 0, 1, 3:] = -1.0

    player_features[0, 0, 2, 0] = 2.0
    player_features[0, 0, 2, 1] = 0.0
    player_features[0, 0, 2, 2] = 1.0
    player_features[0, 0, 2, 3:] = -1.0

    player_features[0, 0, 3, 0] = 2.0
    player_features[0, 0, 3, 1] = 1.5
    player_features[0, 0, 3, 2] = 1.0
    player_features[0, 0, 3, 3] = 1.0

    mask = compute_bench_riser_candidate_mask(
        player_features=player_features,
        player_valid_mask=player_valid_mask,
        starter_hint_mask=starter_hint_mask,
        config=hybrid_cfg,
    )
    assert bool(mask[0, 0, 0]) is True
    assert bool(mask[0, 0, 1]) is False
    assert bool(mask[0, 0, 2]) is False
    assert bool(mask[0, 0, 3]) is False


def test_blend_expert_predictions_uplift_only() -> None:
    blended_minutes, blended_active = blend_expert_predictions(
        baseline_minutes=torch.tensor([[10.0, 18.0]], dtype=torch.float32),
        baseline_active_mask=torch.tensor([[True, True]]),
        expert_minutes=torch.tensor([[22.0, 8.0]], dtype=torch.float32),
        expert_active_mask=torch.tensor([[True, False]]),
        candidate_mask=torch.tensor([[True, False]]),
        uplift_only=True,
    )
    assert torch.allclose(blended_minutes, torch.tensor([[22.0, 18.0]], dtype=torch.float32))
    assert torch.equal(blended_active, torch.tensor([[True, True]]))


def test_assert_promotion_hybrid_compatible_rejects_feature_mismatch() -> None:
    primary = GameTransformerV2Config(
        feature_columns=["minutes_from_stints_prior_20", "recent_start_pct_10"],
        feature_mean=[0.0, 0.0],
        feature_std=[1.0, 1.0],
        game_feature_columns=[],
        team_feature_columns=[],
    )
    expert = GameTransformerV2Config(
        feature_columns=["minutes_from_stints_prior_20", "started_proxy_rate_prior_10"],
        feature_mean=[0.0, 0.0],
        feature_std=[1.0, 1.0],
        game_feature_columns=[],
        team_feature_columns=[],
    )
    with pytest.raises(ValueError, match="feature_columns"):
        assert_promotion_hybrid_compatible(primary, expert)


def test_sample_worlds_for_batch_promotion_hybrid_uses_blended_minutes_context() -> None:
    cols = list(FLOW_TARGET_COLUMNS_V1)

    class _FlowHead:
        def __init__(self) -> None:
            self.last_minutes_context: torch.Tensor | None = None

        def sample(self, z: torch.Tensor, **kwargs: object) -> torch.Tensor:
            minutes_context = kwargs.get("minutes_context")
            assert isinstance(minutes_context, torch.Tensor)
            self.last_minutes_context = minutes_context.detach().cpu()
            return torch.zeros_like(z, dtype=torch.float32)

    class _Out:
        def __init__(
            self,
            valid_flat: torch.Tensor,
            active_flat: torch.Tensor,
            minutes_flat: torch.Tensor,
            team_idx: torch.Tensor,
        ) -> None:
            self.player_states = torch.zeros((valid_flat.shape[0], 30, 4), dtype=torch.float32)
            self.team_states = torch.zeros((valid_flat.shape[0], 2, 4), dtype=torch.float32)
            self.game_state = torch.zeros((valid_flat.shape[0], 4), dtype=torch.float32)
            self.player_valid_mask = valid_flat
            self.player_team_index = team_idx
            self.active = type("Active", (), {"active_mask": active_flat})()
            self.minutes = type("Minutes", (), {"minutes": minutes_flat})()
            self.flow = None
            self.backbone = None
            self.usage_share = None
            self.efficiency = None

    class _Model:
        def __init__(self, *, home_candidate_minutes: float, home_candidate_active: bool, flow_head: _FlowHead | None = None) -> None:
            self.flow_target_columns = cols
            self.enable_possession_backbone = False
            self.flow_head = flow_head if flow_head is not None else _FlowHead()
            self._home_candidate_minutes = float(home_candidate_minutes)
            self._home_candidate_active = bool(home_candidate_active)

        def __call__(self, pf: torch.Tensor, pvm: torch.Tensor, **kwargs: object) -> _Out:
            assert kwargs.get("starter_hint_mask") is not None
            bsz = int(pf.shape[0])
            valid_flat = pvm.reshape(bsz, -1).to(dtype=torch.bool)
            active_flat = torch.zeros((bsz, 30), dtype=torch.bool)
            minutes_flat = torch.zeros((bsz, 30), dtype=torch.float32)
            minutes_flat[:, 0] = self._home_candidate_minutes
            minutes_flat[:, 1:6] = (240.0 - self._home_candidate_minutes) / 5.0
            minutes_flat[:, 15:20] = 48.0
            active_flat[:, 1:6] = True
            active_flat[:, 15:20] = True
            active_flat[:, 0] = self._home_candidate_active
            return _Out(
                valid_flat=valid_flat,
                active_flat=active_flat,
                minutes_flat=minutes_flat,
                team_idx=_team_index(bsz),
            )

    feature_columns = [
        "minutes_from_stints_prior_20",
        "recent_start_pct_10",
        "started_proxy_rate_prior_10",
        "started_proxy_rate_prior_20",
    ]
    hybrid_cfg = PromotionHybridConfig.from_model_config(
        GameTransformerV2Config(
            feature_columns=feature_columns,
            feature_mean=[0.0, 0.0, 0.0, 0.0],
            feature_std=[1.0, 1.0, 1.0, 1.0],
            game_feature_columns=[],
            team_feature_columns=[],
        ),
        prior_minutes_max=12.0,
        hist_start_rate_max=0.20,
        uplift_only=True,
    )

    player_features = torch.zeros((1, 2, 15, len(feature_columns)), dtype=torch.float32)
    player_features[:, 0, :, 0] = 24.0
    player_features[:, 1, :, 0] = 24.0
    player_features[:, 0, :, 1:] = 0.8
    player_features[:, 1, :, 1:] = 0.8
    player_features[0, 0, 0, 0] = 5.0
    player_features[0, 0, 0, 1:] = 0.1

    player_valid_mask = torch.zeros((1, 2, 15), dtype=torch.bool)
    player_valid_mask[:, 0, :6] = True
    player_valid_mask[:, 1, :5] = True
    starter_force_active_worlds = torch.zeros((1, 2, 15), dtype=torch.bool)
    starter_force_active_worlds[:, 0, 0] = True

    batch: dict[str, torch.Tensor | list[str]] = {
        "player_features": player_features,
        "player_valid_mask": player_valid_mask,
        "force_active_worlds": torch.zeros((1, 2, 15), dtype=torch.bool),
        "starter_force_active_worlds": starter_force_active_worlds,
        "force_active_minutes_anchor": torch.zeros((1, 2, 15), dtype=torch.float32),
        "game_features": torch.zeros((1, 0), dtype=torch.float32),
        "team_features": torch.zeros((1, 2, 0), dtype=torch.float32),
        "player_ids": torch.arange(1001, 1031, dtype=torch.long).reshape(1, 2, 15),
        "team_ids": torch.tensor([[10, 20]], dtype=torch.long),
        "game_id_norm": ["1001"],
        "game_date": ["2026-01-18"],
    }

    flow_head = _FlowHead()
    baseline_model = _Model(home_candidate_minutes=6.0, home_candidate_active=False, flow_head=flow_head)
    expert_model = _Model(home_candidate_minutes=30.0, home_candidate_active=True)

    worlds_df, checks = sample_worlds_for_batch(
        baseline_model,
        batch,
        device=torch.device("cpu"),
        num_worlds=2,
        chunk_size=2,
        active_temperature=1.0,
        strict_contracts=True,
        promotion_expert_model=expert_model,
        promotion_hybrid_config=hybrid_cfg,
    )

    assert checks["total_violations"] == 0
    candidate_rows = worlds_df.loc[worlds_df["player_id"] == 1001].copy()
    assert len(candidate_rows) == 2
    assert int(candidate_rows["active"].min()) == 1
    assert float(candidate_rows["minutes"].mean()) > 6.0
    assert flow_head.last_minutes_context is not None
    assert float(flow_head.last_minutes_context[0, 0]) > 6.0
    team_minutes = worlds_df.groupby(["world_idx", "team_id"], as_index=False)["minutes"].sum()
    assert float((team_minutes["minutes"] - 240.0).abs().max()) <= 1e-3


def test_sample_worlds_for_batch_promotion_and_bench_hybrids_interact_safely() -> None:
    cols = [
        "minutes_from_stints_prior_20",
        "prior_play_prob",
        "an_implied_minutes",
        "recent_start_pct_10",
        "started_proxy_rate_prior_10",
        "started_proxy_rate_prior_20",
    ]

    class _FlowHead:
        def __init__(self) -> None:
            self.last_minutes_context: torch.Tensor | None = None

        def sample(self, z: torch.Tensor, **kwargs: object) -> torch.Tensor:
            minutes_context = kwargs.get("minutes_context")
            assert isinstance(minutes_context, torch.Tensor)
            self.last_minutes_context = minutes_context.detach().cpu()
            return torch.zeros_like(z, dtype=torch.float32)

    class _Out:
        def __init__(self, valid_flat: torch.Tensor, active_flat: torch.Tensor, minutes_flat: torch.Tensor, team_idx: torch.Tensor) -> None:
            self.player_states = torch.zeros((valid_flat.shape[0], 30, 4), dtype=torch.float32)
            self.team_states = torch.zeros((valid_flat.shape[0], 2, 4), dtype=torch.float32)
            self.game_state = torch.zeros((valid_flat.shape[0], 4), dtype=torch.float32)
            self.player_valid_mask = valid_flat
            self.player_team_index = team_idx
            self.active = type("Active", (), {"active_mask": active_flat})()
            self.minutes = type("Minutes", (), {"minutes": minutes_flat})()
            self.flow = None
            self.backbone = None
            self.usage_share = None
            self.efficiency = None

    class _Model:
        def __init__(
            self,
            *,
            starter_minutes: float,
            starter_active: bool,
            bench_minutes: float,
            bench_active: bool,
            flow_head: _FlowHead | None = None,
        ) -> None:
            self.flow_target_columns = list(FLOW_TARGET_COLUMNS_V1)
            self.enable_possession_backbone = False
            self.flow_head = flow_head if flow_head is not None else _FlowHead()
            self._starter_minutes = float(starter_minutes)
            self._starter_active = bool(starter_active)
            self._bench_minutes = float(bench_minutes)
            self._bench_active = bool(bench_active)

        def __call__(self, pf: torch.Tensor, pvm: torch.Tensor, **kwargs: object) -> _Out:
            bsz = int(pf.shape[0])
            valid_flat = pvm.reshape(bsz, -1).to(dtype=torch.bool)
            active_flat = torch.zeros((bsz, 30), dtype=torch.bool)
            minutes_flat = torch.zeros((bsz, 30), dtype=torch.float32)
            minutes_flat[:, 0] = self._starter_minutes
            minutes_flat[:, 1] = self._bench_minutes
            minutes_flat[:, 2:6] = (240.0 - self._starter_minutes - self._bench_minutes) / 4.0
            minutes_flat[:, 15:20] = 48.0
            active_flat[:, 0] = self._starter_active
            active_flat[:, 1] = self._bench_active
            active_flat[:, 2:6] = True
            active_flat[:, 15:20] = True
            return _Out(valid_flat=valid_flat, active_flat=active_flat, minutes_flat=minutes_flat, team_idx=_team_index(bsz))

    cfg = GameTransformerV2Config(
        feature_columns=cols,
        feature_mean=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        feature_std=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        game_feature_columns=[],
        team_feature_columns=[],
    )
    promotion_cfg = PromotionHybridConfig.from_model_config(cfg, prior_minutes_max=12.0, hist_start_rate_max=0.20)
    bench_cfg = BenchRiserHybridConfig.from_model_config(
        cfg,
        prior_minutes_min=12.0,
        prior_play_prob_min=0.80,
        implied_minutes_min=12.0,
        hist_start_rate_max=0.35,
    )

    player_features = torch.zeros((1, 2, 15, len(cols)), dtype=torch.float32)
    player_valid_mask = torch.zeros((1, 2, 15), dtype=torch.bool)
    starter_force_active_worlds = torch.zeros((1, 2, 15), dtype=torch.bool)
    player_valid_mask[:, 0, :6] = True
    player_valid_mask[:, 1, :5] = True
    starter_force_active_worlds[:, 0, 0] = True

    player_features[0, 0, 0, 0] = 5.0
    player_features[0, 0, 0, 3:] = 0.1

    player_features[0, 0, 1, 0] = 16.0
    player_features[0, 0, 1, 1] = 0.90
    player_features[0, 0, 1, 2] = 18.0
    player_features[0, 0, 1, 3:] = 0.10

    flow_head = _FlowHead()
    baseline_model = _Model(starter_minutes=6.0, starter_active=False, bench_minutes=14.0, bench_active=True, flow_head=flow_head)
    starter_expert = _Model(starter_minutes=30.0, starter_active=True, bench_minutes=14.0, bench_active=True)
    bench_expert = _Model(starter_minutes=6.0, starter_active=False, bench_minutes=28.0, bench_active=True)

    batch: dict[str, torch.Tensor | list[str]] = {
        "player_features": player_features,
        "player_valid_mask": player_valid_mask,
        "force_active_worlds": torch.zeros((1, 2, 15), dtype=torch.bool),
        "starter_force_active_worlds": starter_force_active_worlds,
        "force_active_minutes_anchor": torch.zeros((1, 2, 15), dtype=torch.float32),
        "game_features": torch.zeros((1, 0), dtype=torch.float32),
        "team_features": torch.zeros((1, 2, 0), dtype=torch.float32),
        "player_ids": torch.arange(1001, 1031, dtype=torch.long).reshape(1, 2, 15),
        "team_ids": torch.tensor([[10, 20]], dtype=torch.long),
        "game_id_norm": ["1001"],
        "game_date": ["2026-01-18"],
    }

    worlds_df, checks = sample_worlds_for_batch(
        baseline_model,
        batch,
        device=torch.device("cpu"),
        num_worlds=2,
        chunk_size=2,
        active_temperature=1.0,
        strict_contracts=True,
        promotion_expert_model=starter_expert,
        promotion_hybrid_config=promotion_cfg,
        bench_expert_model=bench_expert,
        bench_hybrid_config=bench_cfg,
    )

    assert checks["total_violations"] == 0
    starter_rows = worlds_df.loc[worlds_df["player_id"] == 1001].copy()
    bench_rows = worlds_df.loc[worlds_df["player_id"] == 1002].copy()
    assert float(starter_rows["minutes"].mean()) > 6.0
    assert float(bench_rows["minutes"].mean()) > 14.0
    assert flow_head.last_minutes_context is not None
    assert float(flow_head.last_minutes_context[0, 0]) > 6.0
    assert float(flow_head.last_minutes_context[0, 1]) > 14.0
