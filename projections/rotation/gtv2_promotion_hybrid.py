"""Expert-overlay helpers for GameTransformerV2 inference."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

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
HAS_ANY_PROPS_FEATURE_NAME = "an_has_any_props"


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
    blend_alpha: float = 1.0

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
        blend_alpha: float = 1.0,
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
            blend_alpha=float(blend_alpha),
        )


@dataclass(frozen=True)
class SparseEmergencyHybridConfig:
    prior_minutes_idx: int
    prior_minutes_mean: float
    prior_minutes_std: float
    prior_play_prob_idx: int
    prior_play_prob_mean: float
    prior_play_prob_std: float
    has_any_props_idx: int | None = None
    has_any_props_mean: float = 0.0
    has_any_props_std: float = 1.0
    prior_minutes_max: float = 12.0
    prior_play_prob_max: float = 0.50
    uplift_only: bool = True
    force_active_candidates: bool = False
    blend_alpha: float = 1.0
    require_no_props: bool = False

    @classmethod
    def from_model_config(
        cls,
        config: GameTransformerV2Config,
        *,
        prior_minutes_max: float = 12.0,
        prior_play_prob_max: float = 0.50,
        uplift_only: bool = True,
        force_active_candidates: bool = False,
        blend_alpha: float = 1.0,
        require_no_props: bool = False,
    ) -> "SparseEmergencyHybridConfig":
        _, feature_mean, feature_std, idx_by_name = _feature_arrays(config)

        prior_minutes_idx = idx_by_name.get(PRIOR_MINUTES_FEATURE_NAME)
        if prior_minutes_idx is None:
            raise ValueError(f"Missing required sparse-emergency feature: {PRIOR_MINUTES_FEATURE_NAME}")
        prior_play_prob_idx = idx_by_name.get(PRIOR_PLAY_PROB_FEATURE_NAME)
        if prior_play_prob_idx is None:
            raise ValueError(f"Missing required sparse-emergency feature: {PRIOR_PLAY_PROB_FEATURE_NAME}")
        has_any_props_idx = idx_by_name.get(HAS_ANY_PROPS_FEATURE_NAME)
        if require_no_props and has_any_props_idx is None:
            raise ValueError(f"Missing required sparse-emergency feature: {HAS_ANY_PROPS_FEATURE_NAME}")
        return cls(
            prior_minutes_idx=int(prior_minutes_idx),
            prior_minutes_mean=float(feature_mean[int(prior_minutes_idx)]),
            prior_minutes_std=float(feature_std[int(prior_minutes_idx)]),
            prior_play_prob_idx=int(prior_play_prob_idx),
            prior_play_prob_mean=float(feature_mean[int(prior_play_prob_idx)]),
            prior_play_prob_std=float(feature_std[int(prior_play_prob_idx)]),
            has_any_props_idx=int(has_any_props_idx) if has_any_props_idx is not None else None,
            has_any_props_mean=float(feature_mean[int(has_any_props_idx)]) if has_any_props_idx is not None else 0.0,
            has_any_props_std=float(feature_std[int(has_any_props_idx)]) if has_any_props_idx is not None else 1.0,
            prior_minutes_max=float(prior_minutes_max),
            prior_play_prob_max=float(prior_play_prob_max),
            uplift_only=bool(uplift_only),
            force_active_candidates=bool(force_active_candidates),
            blend_alpha=float(blend_alpha),
            require_no_props=bool(require_no_props),
        )


@dataclass(frozen=True)
class SparseEmergencyGateConfig:
    feature_indices: tuple[int, ...]
    decode_means: tuple[float, ...]
    decode_stds: tuple[float, ...]
    feature_names: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_stds: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    prob_threshold: float

    @classmethod
    def from_artifact(
        cls,
        config: GameTransformerV2Config,
        artifact_path: str | Path,
    ) -> "SparseEmergencyGateConfig":
        payload = json.loads(Path(artifact_path).expanduser().read_text(encoding="utf-8"))
        feature_names = tuple(str(x) for x in payload["feature_names"])
        feature_columns, feature_mean, feature_std, idx_by_name = _feature_arrays(config)
        del feature_columns
        feature_indices: list[int] = []
        decode_means: list[float] = []
        decode_stds: list[float] = []
        for name in feature_names:
            idx = idx_by_name.get(name)
            if idx is None:
                raise ValueError(f"Missing sparse gate feature in model config: {name}")
            feature_indices.append(int(idx))
            decode_means.append(float(feature_mean[int(idx)]))
            decode_stds.append(float(feature_std[int(idx)]))
        feature_means = tuple(float(x) for x in payload["feature_means"])
        feature_stds = tuple(float(x) for x in payload["feature_stds"])
        coefficients = tuple(float(x) for x in payload["coefficients"])
        if not (
            len(feature_names)
            == len(feature_means)
            == len(feature_stds)
            == len(coefficients)
            == len(feature_indices)
        ):
            raise ValueError("Sparse gate artifact dimensions do not match")
        return cls(
            feature_indices=tuple(feature_indices),
            decode_means=tuple(decode_means),
            decode_stds=tuple(decode_stds),
            feature_names=feature_names,
            feature_means=feature_means,
            feature_stds=feature_stds,
            coefficients=coefficients,
            intercept=float(payload["intercept"]),
            prob_threshold=float(payload.get("prob_threshold", 0.5)),
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
    if primary_mean.shape != expert_mean.shape:
        raise ValueError("Promotion expert feature_mean shape must match primary model feature_mean shape")
    if primary_std.shape != expert_std.shape:
        raise ValueError("Promotion expert feature_std shape must match primary model feature_std shape")


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


def compute_sparse_emergency_candidate_mask(
    *,
    player_features: torch.Tensor,
    player_valid_mask: torch.Tensor,
    config: SparseEmergencyHybridConfig,
) -> torch.Tensor:
    if player_features.ndim != 4:
        raise ValueError("player_features must have shape (B,2,15,F)")
    if player_valid_mask.shape != player_features.shape[:3]:
        raise ValueError("player_valid_mask must have shape (B,2,15)")

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
    sparse_signal = prior_minutes.le(float(config.prior_minutes_max)) | prior_play_prob.le(
        float(config.prior_play_prob_max)
    )
    if bool(config.require_no_props) and config.has_any_props_idx is not None:
        has_any_props = _decode_feature(
            player_features,
            idx=int(config.has_any_props_idx),
            mean=float(config.has_any_props_mean),
            std=float(config.has_any_props_std),
        )
        sparse_signal = sparse_signal & has_any_props.lt(0.5)
    return player_valid_mask.to(dtype=torch.bool) & sparse_signal


def compute_sparse_emergency_gate_probability(
    *,
    player_features: torch.Tensor,
    config: SparseEmergencyGateConfig,
) -> torch.Tensor:
    if player_features.ndim != 4:
        raise ValueError("player_features must have shape (B,2,15,F)")
    cols: list[torch.Tensor] = []
    for idx, decode_mean, decode_std, feat_mean, feat_std in zip(
        config.feature_indices,
        config.decode_means,
        config.decode_stds,
        config.feature_means,
        config.feature_stds,
        strict=True,
    ):
        raw = _decode_feature(
            player_features,
            idx=int(idx),
            mean=float(decode_mean),
            std=float(decode_std),
        )
        scale = float(feat_std)
        if abs(scale) < 1e-8:
            scale = 1.0
        cols.append((raw - float(feat_mean)) / scale)
    x = torch.stack(cols, dim=-1)
    coef = torch.as_tensor(config.coefficients, dtype=x.dtype, device=x.device)
    logits = (x * coef).sum(dim=-1) + float(config.intercept)
    return torch.sigmoid(logits)


def blend_expert_predictions(
    *,
    baseline_minutes: torch.Tensor,
    baseline_active_mask: torch.Tensor,
    expert_minutes: torch.Tensor,
    expert_active_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    uplift_only: bool,
    force_active_candidates: bool = False,
    blend_alpha: float = 1.0,
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
    alpha = float(blend_alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("blend_alpha must be in [0, 1]")
    if bool(uplift_only):
        uplift = (expert_minutes - baseline_minutes).clamp(min=0.0)
        target_minutes = baseline_minutes + alpha * uplift
        blended_minutes = torch.where(candidate, target_minutes, baseline_minutes)
    else:
        target_minutes = baseline_minutes + alpha * (expert_minutes - baseline_minutes)
        blended_minutes = torch.where(candidate, target_minutes, baseline_minutes)
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
    blend_alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return blend_expert_predictions(
        baseline_minutes=baseline_minutes,
        baseline_active_mask=baseline_active_mask,
        expert_minutes=expert_minutes,
        expert_active_mask=expert_active_mask,
        candidate_mask=promotion_candidate_mask,
        uplift_only=uplift_only,
        force_active_candidates=force_active_candidates,
        blend_alpha=blend_alpha,
    )
