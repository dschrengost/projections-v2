"""Tests for projected-starter boost logic (rotation_set_minutes inference)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projections.rotation.projected_starter_boost import apply_projected_starter_boost
from projections.rotation.set_model import minutes_from_gate_and_share_logits


def _write_model_config(tmp_path: Path) -> Path:
    model_dir = tmp_path / "rot_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                # Keep the allocator simple/deterministic for unit tests.
                "alloc_activation": "softmax",
                "entmax_alpha": 1.5,
                "share_temperature": 1.0,
                "total_minutes": 240.0,
                "eps": 1e-6,
                "use_prior_head": True,
                "prior_weight_col": "prior_w",
                "prior_weight_floor": 1.0,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return model_dir


def test_projected_starter_boost_recomputes_only_when_gate_low(tmp_path: Path) -> None:
    model_dir = _write_model_config(tmp_path)

    # Single team-game with 5 eligible players.
    keys = {
        "game_id": ["g"] * 5,
        "team_id": [1] * 5,
        "player_id": [10, 11, 12, 13, 14],
    }

    # Two projected starters: one low gate_prob (needs boost), one already high gate_prob (no boost).
    rot_features = pd.DataFrame(
        {
            **keys,
            "is_projected_starter": [1, 1, 0, 0, 0],
            "is_confirmed_starter": [0, 0, 0, 0, 0],
            "is_out": [0, 0, 0, 0, 0],
            "prior_play_prob": [0.95] * 5,
            "prior_w": [40.0, 35.0, 30.0, 10.0, 5.0],
        }
    )

    gate_logits = np.array([0.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)  # p=[0.5, ~0.99, ...]
    share_logits = np.array([1.0, 0.5, 0.2, -0.2, -1.0], dtype=np.float32)
    mask = np.ones(5, dtype=bool)

    base_w = torch.as_tensor(rot_features["prior_w"].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32)
    base_minutes_t = minutes_from_gate_and_share_logits(
        torch.as_tensor(gate_logits[None, :], dtype=torch.float32),
        torch.as_tensor(share_logits[None, :], dtype=torch.float32),
        torch.as_tensor(mask[None, :], dtype=torch.bool),
        total_minutes=240.0,
        eps=1e-6,
        base_weights=base_w,
        base_floor=1.0,
        alloc_activation="softmax",
        entmax_alpha=1.5,
        share_temperature=1.0,
    )
    base_minutes = base_minutes_t.detach().cpu().numpy()[0]

    rot_pred = pd.DataFrame(
        {
            **keys,
            "rotation_minutes_p50": base_minutes.astype(float),
            "gate_logit": gate_logits.astype(float),
            "share_logit": share_logits.astype(float),
            "gate_prob": (1.0 / (1.0 + np.exp(-gate_logits))).astype(float),
        }
    )

    boosted, stats = apply_projected_starter_boost(
        rot_pred,
        rot_features=rot_features,
        model_dir=model_dir,
        alloc_mask_mode="not_out",
        alloc_min_eligible=9,
        alloc_prior_play_prob_threshold=0.2,
        alloc_baseline_minutes_threshold=4.0,
        gate_prob_threshold=0.90,
        gate_logit_boost=20.0,
    )

    assert stats.enabled is True
    assert stats.skipped_reason is None
    assert stats.boosted_players == 1
    assert stats.recomputed_team_games == 1

    # Only player_id=10 (gate_prob=0.5) should have its gate_logit boosted.
    boosted_gate = boosted.set_index("player_id")["gate_logit"].to_dict()
    assert boosted_gate[10] == 20.0
    assert boosted_gate[11] == 5.0

    # New minutes should match a direct recompute with boosted gate_logits.
    gate_logits_boosted = gate_logits.copy()
    gate_logits_boosted[0] = 20.0
    expected_minutes_t = minutes_from_gate_and_share_logits(
        torch.as_tensor(gate_logits_boosted[None, :], dtype=torch.float32),
        torch.as_tensor(share_logits[None, :], dtype=torch.float32),
        torch.as_tensor(mask[None, :], dtype=torch.bool),
        total_minutes=240.0,
        eps=1e-6,
        base_weights=base_w,
        base_floor=1.0,
        alloc_activation="softmax",
        entmax_alpha=1.5,
        share_temperature=1.0,
    )
    expected_minutes = expected_minutes_t.detach().cpu().numpy()[0].astype(float)

    got_minutes = boosted.sort_values("player_id")["rotation_minutes_p50"].to_numpy(dtype=float)
    np.testing.assert_allclose(got_minutes, expected_minutes, rtol=1e-6, atol=1e-5)

    # Sanity: sums to 240.
    assert abs(float(got_minutes.sum()) - 240.0) < 1e-3

