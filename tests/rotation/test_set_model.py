from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from projections.rotation.set_model import (
    RotationSetAuxOutputs,
    RotationSetMinutesPredictor,
    RotationSetModelConfig,
    build_model,
    minutes_from_gate_and_share_logits,
    minutes_from_logits,
    minutes_from_moe_gate_and_share_logits,
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


@pytest.mark.parametrize("model_type", ["deepsets", "settransformer"])
def test_is_out_rows_receive_zero_minutes(tmp_path: Path, model_type: str) -> None:
    model_dir = _write_toy_model_dir(tmp_path / model_type, model_type=model_type)
    df = _toy_features_df()
    df = df.assign(is_out=0)
    # Mark a single player as OUT in game 1 / team 10.
    df.loc[(df["game_id"] == 1) & (df["team_id"] == 10) & (df["player_id"] == 101), "is_out"] = 1

    scored = predict_minutes(df, model_dir=model_dir, device="cpu", batch_size=2)
    out_row = scored[(scored["game_id"] == 1) & (scored["team_id"] == 10) & (scored["player_id"] == 101)].iloc[0]
    assert float(out_row["pred_minutes"]) == pytest.approx(0.0, abs=1e-6)

    scored_norm = scored.assign(game_id_norm=zfill_game_id_series(scored["game_id"]))
    sums = scored_norm.groupby(["game_id_norm", "team_id"])["pred_minutes"].sum()
    np.testing.assert_allclose(sums.to_numpy(), np.full(len(sums), 240.0), rtol=0.0, atol=1e-4)


def test_minutes_from_logits_softplus_matches_legacy_mapping() -> None:
    logits = torch.tensor([[2.0, -1.0, 0.0, 5.0]], dtype=torch.float32)
    mask = torch.tensor([[1, 1, 0, 1]], dtype=torch.bool)
    base = torch.tensor([[1.0, 2.0, 3.0, 0.5]], dtype=torch.float32)
    eps = 1e-6

    got = minutes_from_logits(
        logits,
        mask,
        total_minutes=240.0,
        eps=eps,
        base_weights=base,
        base_floor=1.0,
        alloc_activation="softplus",
    )

    positive = torch.nn.functional.softplus(logits) * mask.to(dtype=torch.float32)
    legacy_base = (base.clamp(min=0) + 1.0) * mask.to(dtype=torch.float32)
    positive = positive * legacy_base
    denom = positive.sum(dim=1, keepdim=True).clamp(min=eps)
    expected = positive / denom * 240.0
    expected = expected * mask.to(dtype=expected.dtype)

    torch.testing.assert_close(got, expected, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("activation", ["sparsemax", "entmax15"])
def test_sparse_allocations_produce_exact_zeros(activation: str) -> None:
    logits = torch.tensor([[5.0, 1.0, -2.0, -10.0, -10.0]], dtype=torch.float32)
    mask = torch.ones_like(logits, dtype=torch.bool)

    minutes = minutes_from_logits(
        logits,
        mask,
        total_minutes=240.0,
        eps=1e-8,
        alloc_activation=activation,
    )

    # Nonnegativity + exact-sum constraint.
    assert torch.all(minutes >= 0)
    torch.testing.assert_close(minutes.sum(dim=1), torch.tensor([240.0]), rtol=0.0, atol=1e-4)

    # Sparsity: low logits should get exactly 0 minutes.
    assert float(minutes[0, 3].item()) == pytest.approx(0.0, abs=1e-9)
    assert float(minutes[0, 4].item()) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("activation", ["sparsemax", "entmax15"])
def test_sparse_allocations_respect_mask(activation: str) -> None:
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]], dtype=torch.float32)
    mask = torch.tensor([[1, 0, 1, 1]], dtype=torch.bool)

    minutes = minutes_from_logits(
        logits,
        mask,
        total_minutes=240.0,
        eps=1e-8,
        alloc_activation=activation,
    )

    # Masked-out player gets exactly 0; sums still hit 240 over eligible.
    assert float(minutes[0, 1].item()) == pytest.approx(0.0, abs=1e-9)
    torch.testing.assert_close(minutes.sum(dim=1), torch.tensor([240.0]), rtol=0.0, atol=1e-4)


def test_minutes_from_gate_and_share_logits_sums_to_240_and_respects_mask() -> None:
    gate_logits = torch.tensor([[10.0, -10.0, 0.0, 5.0]], dtype=torch.float32)
    share_logits = torch.tensor([[1.0, 2.0, -1.0, 0.0]], dtype=torch.float32)
    mask = torch.tensor([[1, 0, 1, 1]], dtype=torch.bool)

    minutes = minutes_from_gate_and_share_logits(
        gate_logits,
        share_logits,
        mask,
        total_minutes=240.0,
        eps=1e-8,
        alloc_activation="entmax",
        entmax_alpha=1.5,
        share_temperature=1.0,
    )

    assert float(minutes[0, 1].item()) == pytest.approx(0.0, abs=1e-9)
    assert torch.all(minutes >= 0)
    torch.testing.assert_close(minutes.sum(dim=1), torch.tensor([240.0]), rtol=0.0, atol=1e-4)


def test_minutes_from_moe_gate_and_share_logits_sums_to_240_and_uses_router() -> None:
    gate_logits = torch.tensor(
        [
            [
                [10.0, -10.0, -10.0, 0.0],  # expert 0 gates player 0
                [-10.0, 10.0, -10.0, 0.0],  # expert 1 gates player 1
            ]
        ],
        dtype=torch.float32,
    )
    share_logits = torch.tensor(
        [
            [
                [10.0, 0.0, -10.0, -10.0],  # expert 0 allocates to player 0
                [0.0, 10.0, -10.0, -10.0],  # expert 1 allocates to player 1
            ]
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)

    minutes0 = minutes_from_moe_gate_and_share_logits(
        gate_logits,
        share_logits,
        router_pi=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        mask=mask,
        total_minutes=240.0,
        eps=1e-8,
        alloc_activation="entmax",
        entmax_alpha=1.5,
        share_temperature=1.0,
    )
    minutes1 = minutes_from_moe_gate_and_share_logits(
        gate_logits,
        share_logits,
        router_pi=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        mask=mask,
        total_minutes=240.0,
        eps=1e-8,
        alloc_activation="entmax",
        entmax_alpha=1.5,
        share_temperature=1.0,
    )

    assert float(minutes0[0, 3].item()) == pytest.approx(0.0, abs=1e-9)
    assert float(minutes1[0, 3].item()) == pytest.approx(0.0, abs=1e-9)
    torch.testing.assert_close(minutes0.sum(dim=1), torch.tensor([240.0]), rtol=0.0, atol=1e-4)
    torch.testing.assert_close(minutes1.sum(dim=1), torch.tensor([240.0]), rtol=0.0, atol=1e-4)

    # Router should change which player gets the bulk of minutes.
    assert float(minutes0[0, 0].item()) > float(minutes0[0, 1].item())
    assert float(minutes1[0, 1].item()) > float(minutes1[0, 0].item())


@pytest.mark.parametrize(
    ("router_mode", "expected_nonzero"),
    [
        ("top1_st", 1),
        ("top2_st", 2),
    ],
)
def test_moe_hard_router_produces_sparse_expert_weights(router_mode: str, expected_nonzero: int) -> None:
    config = RotationSetModelConfig(
        model="settransformer",  # type: ignore[arg-type]
        feature_columns=["f1", "f2"],
        feature_mean=[0.0, 0.0],
        feature_std=[1.0, 1.0],
        embed_dim=8,
        hidden_dim=16,
        dropout=0.0,
        num_transformer_layers=1,
        num_attention_heads=2,
        use_gate_head=True,
        num_experts=4,
        router_feature_dim=0,
        router_temperature=1.0,
        router_mode=router_mode,
        router_gumbel=False,
    )
    model = build_model(config)
    for param in model.parameters():
        torch.nn.init.constant_(param, 0.0)
    model.eval()

    x = torch.zeros((2, 3, 2), dtype=torch.float32)
    team_idx = torch.zeros((2, 3), dtype=torch.long)
    opp_idx = torch.zeros((2, 3), dtype=torch.long)
    mask = torch.ones((2, 3), dtype=torch.bool)

    minutes, _, _, router_pi_soft, router_pi_used = model.forward_with_aux(  # type: ignore[attr-defined]
        x, team_idx, opp_idx, mask, alloc_mask=mask
    )
    assert router_pi_soft is not None
    assert router_pi_used is not None

    torch.testing.assert_close(router_pi_soft.sum(dim=1), torch.ones((2,)), rtol=0.0, atol=1e-6)
    torch.testing.assert_close(router_pi_used.sum(dim=1), torch.ones((2,)), rtol=0.0, atol=1e-6)

    nonzero = (router_pi_used > 0).sum(dim=1)
    assert int(nonzero.min().item()) == expected_nonzero
    assert int(nonzero.max().item()) == expected_nonzero

    # Forward pass remains a valid allocator.
    torch.testing.assert_close(minutes.sum(dim=1), torch.tensor([240.0, 240.0]), rtol=0.0, atol=1e-4)


# -----------------------------------------------------------------------------
# Tests for return_aux functionality (PR5 task 1)
# -----------------------------------------------------------------------------


def _write_toy_model_dir_with_gate(tmp_path: Path, *, model_type: str, num_experts: int = 1) -> Path:
    """Create a toy model with gate head for aux output testing."""
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
        use_gate_head=True,
        num_experts=num_experts,
        router_feature_dim=0,
        router_temperature=1.0,
        router_mode="soft" if num_experts > 1 else "soft",
    )
    model = build_model(config)
    # Initialize to non-zero for meaningful outputs
    torch.manual_seed(42)
    for param in model.parameters():
        torch.nn.init.uniform_(param, -0.1, 0.1)

    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")
    config.save(model_dir / "config.json")
    return model_dir


@pytest.mark.parametrize("model_type", ["deepsets", "settransformer"])
def test_return_aux_false_returns_dataframe(tmp_path: Path, model_type: str) -> None:
    """When return_aux=False (default), predict returns a plain DataFrame."""
    model_dir = _write_toy_model_dir_with_gate(tmp_path / model_type, model_type=model_type)
    predictor = RotationSetMinutesPredictor.load(model_dir, device="cpu")
    df = _toy_features_df()

    result = predictor.predict(df, return_aux=False)

    assert isinstance(result, pd.DataFrame)
    assert "pred_minutes" in result.columns
    assert "gate_logit" not in result.columns
    assert "gate_prob" not in result.columns
    assert "share_logit" not in result.columns


@pytest.mark.parametrize("model_type", ["deepsets", "settransformer"])
def test_return_aux_true_returns_aux_outputs(tmp_path: Path, model_type: str) -> None:
    """When return_aux=True, predict returns RotationSetAuxOutputs."""
    model_dir = _write_toy_model_dir_with_gate(tmp_path / model_type, model_type=model_type)
    predictor = RotationSetMinutesPredictor.load(model_dir, device="cpu")
    df = _toy_features_df()

    result = predictor.predict(df, return_aux=True)

    assert isinstance(result, RotationSetAuxOutputs)
    assert isinstance(result.player_df, pd.DataFrame)
    assert "pred_minutes" in result.player_df.columns
    assert "gate_logit" in result.player_df.columns
    assert "gate_prob" in result.player_df.columns
    assert "share_logit" in result.player_df.columns


@pytest.mark.parametrize("model_type", ["deepsets", "settransformer"])
def test_return_aux_pred_minutes_match(tmp_path: Path, model_type: str) -> None:
    """pred_minutes should be identical whether return_aux is True or False."""
    model_dir = _write_toy_model_dir_with_gate(tmp_path / model_type, model_type=model_type)
    predictor = RotationSetMinutesPredictor.load(model_dir, device="cpu")
    df = _toy_features_df()

    result_no_aux = predictor.predict(df, return_aux=False)
    result_aux = predictor.predict(df, return_aux=True)

    np.testing.assert_allclose(
        result_no_aux["pred_minutes"].values,
        result_aux.player_df["pred_minutes"].values,
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize("model_type", ["deepsets", "settransformer"])
def test_return_aux_gate_prob_is_sigmoid_of_gate_logit(tmp_path: Path, model_type: str) -> None:
    """gate_prob should be sigmoid(gate_logit)."""
    model_dir = _write_toy_model_dir_with_gate(tmp_path / model_type, model_type=model_type)
    predictor = RotationSetMinutesPredictor.load(model_dir, device="cpu")
    df = _toy_features_df()

    result = predictor.predict(df, return_aux=True)

    gate_logit = result.player_df["gate_logit"].values
    gate_prob = result.player_df["gate_prob"].values
    expected_prob = 1.0 / (1.0 + np.exp(-gate_logit))

    np.testing.assert_allclose(gate_prob, expected_prob, rtol=0.0, atol=1e-6)


def test_return_aux_with_moe_includes_router_df(tmp_path: Path) -> None:
    """MoE models should include router_df with per-group router probabilities."""
    model_dir = _write_toy_model_dir_with_gate(
        tmp_path / "moe", model_type="settransformer", num_experts=4
    )
    predictor = RotationSetMinutesPredictor.load(model_dir, device="cpu")
    df = _toy_features_df()

    result = predictor.predict(df, return_aux=True)

    assert result.router_df is not None
    assert "game_id_norm" in result.router_df.columns
    assert "team_id" in result.router_df.columns
    # Check router_pi columns (one per expert)
    for i in range(4):
        assert f"router_pi_{i}" in result.router_df.columns

    # Should have one row per (game_id, team_id) group
    expected_groups = df.groupby(["game_id", "team_id"]).ngroups
    assert len(result.router_df) == expected_groups

    # Router probs should sum to ~1 per row
    router_cols = [f"router_pi_{i}" for i in range(4)]
    router_sums = result.router_df[router_cols].sum(axis=1)
    np.testing.assert_allclose(router_sums.values, np.ones(len(result.router_df)), rtol=0.0, atol=1e-5)


def test_return_aux_without_moe_router_df_is_none(tmp_path: Path) -> None:
    """Non-MoE models (num_experts=1) should have router_df=None."""
    model_dir = _write_toy_model_dir_with_gate(
        tmp_path / "no_moe", model_type="settransformer", num_experts=1
    )
    predictor = RotationSetMinutesPredictor.load(model_dir, device="cpu")
    df = _toy_features_df()

    result = predictor.predict(df, return_aux=True)

    assert result.router_df is None
