"""Expert-overlay helpers for GameTransformerV2 inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from projections.rotation.game_transformer_v2 import GameTransformerV2Config

PRIOR_MINUTES_FEATURE_NAME = "minutes_from_stints_prior_20"
PRIOR_PLAY_PROB_FEATURE_NAME = "prior_play_prob"
IMPLIED_MINUTES_FEATURE_NAME = "an_implied_minutes"
HIST_START_RATE_FEATURE_NAMES = (
    "recent_start_pct_10",
    "started_proxy_rate_prior_10",
    "started_proxy_rate_prior_20",
)


@dataclass(frozen=True)
class PromotionHybridConfig:
    prior_minutes_idx: int
    prior_minutes_mean: float
    prior_minutes_std: float
    hist_start_rate_idxs: tuple[int, ...]
    hist_start_rate_means: tuple[float, ...]
    hist_start_rate_stds: tuple[float, ...]
    prior_minutes_max: float = 12.0
    hist_start_rate_max: float = 0.20
    uplift_only: bool = True
    force_active_candidates: bool = False

    @classmethod
    def from_model_config(
        cls,
        config: GameTransformerV2Config,
        *,
        prior_minutes_max: float = 12.0,
        hist_start_rate_max: float = 0.20,
        uplift_only: bool = True,
        force_active_candidates: bool = False,
    ) -> "PromotionHybridConfig":
        feature_columns, feature_mean, feature_std, idx_by_name = _feature_arrays(config)
        del feature_columns

        prior_minutes_idx = idx_by_name.get(PRIOR_MINUTES_FEATURE_NAME)
        if prior_minutes_idx is None:
            raise ValueError(f"Missing required promotion feature: {PRIOR_MINUTES_FEATURE_NAME}")
        hist_idxs, hist_means, hist_stds = _resolve_hist_start_bundle(
            idx_by_name=idx_by_name,
            feature_mean=feature_mean,
            feature_std=feature_std,
            label="promotion",
        )
        return cls(
            prior_minutes_idx=int(prior_minutes_idx),
            prior_minutes_mean=float(feature_mean[int(prior_minutes_idx)]),
            prior_minutes_std=float(feature_std[int(prior_minutes_idx)]),
            hist_start_rate_idxs=hist_idxs,
            hist_start_rate_means=hist_means,
            hist_start_rate_stds=hist_stds,
            prior_minutes_max=float(prior_minutes_max),
            hist_start_rate_max=float(hist_start_rate_max),
            uplift_only=bool(uplift_only),
            force_active_candidates=bool(force_active_candidates),
        )


@dataclass(frozen=True)
class BenchRiserHybridConfig:
    prior_minutes_idx: int
    prior_minutes_mean: float
    prior_minutes_std: float
    prior_play_prob_idx: int
    prior_play_prob_mean: float
    prior_play_prob_std: float
    implied_minutes_idx: int
    implied_minutes_mean: float
    implied_minutes_std: float
    hist_start_rate_idxs: tuple[int, ...]
    hist_start_rate_means: tuple[float, ...]
    hist_start_rate_stds: tuple[float, ...]
    prior_minutes_min: float = 12.0
    prior_play_prob_min: float = 0.80
    implied_minutes_min: float = 12.0
    hist_start_rate_max: float = 0.35
    uplift_only: bool = True
    force_active_candidates: bool = False

    @classmethod
    def from_model_config(
        cls,
        config: GameTransformerV2Config,
        *,
        prior_minutes_min: float = 12.0,
        prior_play_prob_min: float = 0.80,
        implied_minutes_min: float = 12.0,
        hist_start_rate_max: float = 0.35,
        uplift_only: bool = True,
        force_active_candidates: bool = False,
    ) -> "BenchRiserHybridConfig":
        _, feature_mean, feature_std, idx_by_name = _feature_arrays(config)

        prior_minutes_idx = idx_by_name.get(PRIOR_MINUTES_FEATURE_NAME)
        if prior_minutes_idx is None:
            raise ValueError(f"Missing required bench-riser feature: {PRIOR_MINUTES_FEATURE_NAME}")
        prior_play_prob_idx = idx_by_name.get(PRIOR_PLAY_PROB_FEATURE_NAME)
        if prior_play_prob_idx is None:
            raise ValueError(f"Missing required bench-riser feature: {PRIOR_PLAY_PROB_FEATURE_NAME}")
        implied_minutes_idx = idx_by_name.get(IMPLIED_MINUTES_FEATURE_NAME)
        if implied_minutes_idx is None:
            raise ValueError(f"Missing required bench-riser feature: {IMPLIED_MINUTES_FEATURE_NAME}")
        hist_idxs, hist_means, hist_stds = _resolve_hist_start_bundle(
            idx_by_name=idx_by_name,
            feature_mean=feature_mean,
            feature_std=feature_std,
            label="bench-riser",
        )
        return cls(
            prior_minutes_idx=int(prior_minutes_idx),
            prior_minutes_mean=float(feature_mean[int(prior_minutes_idx)]),
            prior_minutes_std=float(feature_std[int(prior_minutes_idx)]),
            prior_play_prob_idx=int(prior_play_prob_idx),
            prior_play_prob_mean=float(feature_mean[int(prior_play_prob_idx)]),
            prior_play_prob_std=float(feature_std[int(prior_play_prob_idx)]),
            implied_minutes_idx=int(implied_minutes_idx),
            implied_minutes_mean=float(feature_mean[int(implied_minutes_idx)]),
            implied_minutes_std=float(feature_std[int(implied_minutes_idx)]),
            hist_start_rate_idxs=hist_idxs,
            hist_start_rate_means=hist_means,
            hist_start_rate_stds=hist_stds,
            prior_minutes_min=float(prior_minutes_min),
            prior_play_prob_min=float(prior_play_prob_min),
            implied_minutes_min=float(implied_minutes_min),
            hist_start_rate_max=float(hist_start_rate_max),
            uplift_only=bool(uplift_only),
            force_active_candidates=bool(force_active_candidates),
        )


def _feature_arrays(
    config: GameTransformerV2Config,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, int]]:
    feature_columns = list(config.feature_columns)
    feature_mean = np.asarray(config.feature_mean, dtype=np.float32)
    feature_std = np.asarray(config.feature_std, dtype=np.float32)
    idx_by_name = {name: idx for idx, name in enumerate(feature_columns)}
    return feature_columns, feature_mean, feature_std, idx_by_name


def _resolve_hist_start_bundle(
    *,
    idx_by_name: dict[str, int],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    label: str,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    hist_idxs = tuple(int(idx_by_name[name]) for name in HIST_START_RATE_FEATURE_NAMES if name in idx_by_name)
    if not hist_idxs:
        raise ValueError(
            f"Missing {label} history start-rate features; expected at least one of {HIST_START_RATE_FEATURE_NAMES}"
        )
    return (
        hist_idxs,
        tuple(float(feature_mean[idx]) for idx in hist_idxs),
        tuple(float(feature_std[idx]) for idx in hist_idxs),
    )


def assert_promotion_hybrid_compatible(
    primary_config: GameTransformerV2Config,
    expert_config: GameTransformerV2Config,
) -> None:
    if list(primary_config.feature_columns) != list(expert_config.feature_columns):
        raise ValueError("Promotion expert feature_columns must exactly match primary model feature_columns")
    if list(primary_config.game_feature_columns) != list(expert_config.game_feature_columns):
        raise ValueError(
            "Promotion expert game_feature_columns must exactly match primary model game_feature_columns"
        )
    if list(primary_config.team_feature_columns) != list(expert_config.team_feature_columns):
        raise ValueError(
            "Promotion expert team_feature_columns must exactly match primary model team_feature_columns"
        )
    primary_mean = np.asarray(primary_config.feature_mean, dtype=np.float32)
    expert_mean = np.asarray(expert_config.feature_mean, dtype=np.float32)
    primary_std = np.asarray(primary_config.feature_std, dtype=np.float32)
    expert_std = np.asarray(expert_config.feature_std, dtype=np.float32)
    if primary_mean.shape != expert_mean.shape or not np.allclose(primary_mean, expert_mean, atol=1e-6):
        raise ValueError("Promotion expert feature_mean must exactly match primary model feature_mean")
    if primary_std.shape != expert_std.shape or not np.allclose(primary_std, expert_std, atol=1e-6):
        raise ValueError("Promotion expert feature_std must exactly match primary model feature_std")


def _decode_feature(
    player_features: torch.Tensor,
    *,
    idx: int,
    mean: float,
    std: float,
) -> torch.Tensor:
    raw = player_features[..., int(idx)].to(dtype=torch.float32)
    scale = float(std) if abs(float(std)) > 1e-6 else 1.0
    return raw * scale + float(mean)


def _decode_hist_start_rate(
    player_features: torch.Tensor,
    *,
    idxs: tuple[int, ...],
    means: tuple[float, ...],
    stds: tuple[float, ...],
) -> torch.Tensor:
    hist_rate_parts = [
        _decode_feature(player_features, idx=idx, mean=mean, std=std)
        for idx, mean, std in zip(idxs, means, stds, strict=True)
    ]
    return torch.stack(hist_rate_parts, dim=0).amax(dim=0)


def compute_starter_promotion_candidate_mask(
    *,
    player_features: torch.Tensor,
    player_valid_mask: torch.Tensor,
    starter_hint_mask: torch.Tensor,
    config: PromotionHybridConfig,
) -> torch.Tensor:
    if player_features.ndim != 4:
        raise ValueError("player_features must have shape (B,2,15,F)")
    if player_valid_mask.shape != player_features.shape[:3]:
        raise ValueError("player_valid_mask must have shape (B,2,15)")
    if starter_hint_mask.shape != player_features.shape[:3]:
        raise ValueError("starter_hint_mask must have shape (B,2,15)")

    prior_minutes = _decode_feature(
        player_features,
        idx=int(config.prior_minutes_idx),
        mean=float(config.prior_minutes_mean),
        std=float(config.prior_minutes_std),
    )
    hist_start_rate = _decode_hist_start_rate(
        player_features,
        idxs=config.hist_start_rate_idxs,
        means=config.hist_start_rate_means,
        stds=config.hist_start_rate_stds,
    )
    return (
        starter_hint_mask.to(dtype=torch.bool)
        & player_valid_mask.to(dtype=torch.bool)
        & prior_minutes.le(float(config.prior_minutes_max))
        & hist_start_rate.le(float(config.hist_start_rate_max))
    )


def compute_bench_riser_candidate_mask(
    *,
    player_features: torch.Tensor,
    player_valid_mask: torch.Tensor,
    starter_hint_mask: torch.Tensor,
    config: BenchRiserHybridConfig,
) -> torch.Tensor:
    if player_features.ndim != 4:
        raise ValueError("player_features must have shape (B,2,15,F)")
    if player_valid_mask.shape != player_features.shape[:3]:
        raise ValueError("player_valid_mask must have shape (B,2,15)")
    if starter_hint_mask.shape != player_features.shape[:3]:
        raise ValueError("starter_hint_mask must have shape (B,2,15)")

    prior_minutes = _decode_feature(
        player_features,
        idx=int(config.prior_minutes_idx),
        mean=float(config.prior_minutes_mean),
        std=float(config.prior_minutes_std),
    )
    prior_play_prob = _decode_feature(
        player_features,
        idx=int(config.prior_play_prob_idx),
        mean=float(config.prior_play_prob_mean),
        std=float(config.prior_play_prob_std),
    )
    implied_minutes = _decode_feature(
        player_features,
        idx=int(config.implied_minutes_idx),
        mean=float(config.implied_minutes_mean),
        std=float(config.implied_minutes_std),
    )
    hist_start_rate = _decode_hist_start_rate(
        player_features,
        idxs=config.hist_start_rate_idxs,
        means=config.hist_start_rate_means,
        stds=config.hist_start_rate_stds,
    )
    return (
        (~starter_hint_mask.to(dtype=torch.bool))
        & player_valid_mask.to(dtype=torch.bool)
        & prior_minutes.ge(float(config.prior_minutes_min))
        & prior_play_prob.ge(float(config.prior_play_prob_min))
        & implied_minutes.ge(float(config.implied_minutes_min))
        & hist_start_rate.le(float(config.hist_start_rate_max))
    )


def blend_expert_predictions(
    *,
    baseline_minutes: torch.Tensor,
    baseline_active_mask: torch.Tensor,
    expert_minutes: torch.Tensor,
    expert_active_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    uplift_only: bool,
    force_active_candidates: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if baseline_minutes.shape != expert_minutes.shape:
        raise ValueError("baseline_minutes and expert_minutes must have matching shapes")
    if baseline_active_mask.shape != expert_active_mask.shape:
        raise ValueError("baseline_active_mask and expert_active_mask must have matching shapes")
    if candidate_mask.shape != baseline_minutes.shape:
        raise ValueError("candidate_mask must match minutes shape")

    candidate = candidate_mask.to(dtype=torch.bool)
    baseline_active = baseline_active_mask.to(dtype=torch.bool)
    expert_active = expert_active_mask.to(dtype=torch.bool)
    blended_active = torch.where(candidate, baseline_active | expert_active, baseline_active)
    if force_active_candidates:
        blended_active = blended_active | candidate
    if bool(uplift_only):
        blended_minutes = torch.where(candidate, torch.maximum(baseline_minutes, expert_minutes), baseline_minutes)
    else:
        blended_minutes = torch.where(candidate, expert_minutes, baseline_minutes)
    return blended_minutes, blended_active


def blend_promotion_predictions(
    *,
    baseline_minutes: torch.Tensor,
    baseline_active_mask: torch.Tensor,
    expert_minutes: torch.Tensor,
    expert_active_mask: torch.Tensor,
    promotion_candidate_mask: torch.Tensor,
    uplift_only: bool,
    force_active_candidates: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return blend_expert_predictions(
        baseline_minutes=baseline_minutes,
        baseline_active_mask=baseline_active_mask,
        expert_minutes=expert_minutes,
        expert_active_mask=expert_active_mask,
        candidate_mask=promotion_candidate_mask,
        uplift_only=uplift_only,
        force_active_candidates=force_active_candidates,
    )
