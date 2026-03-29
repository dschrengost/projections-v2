"""Shared GTV2 inference runtime helpers for local and server backends."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from projections.rotation.game_transformer_v2 import (
    GameLevelDataset,
    GameTransformerV2Config,
    build_game_level_examples,
    build_game_transformer_v2,
    collate_game_level_examples,
)

_TORCH_RUNTIME_CONFIGURED = False
_MODEL_CACHE: dict[
    tuple[str, str, float | None],
    tuple[GameTransformerV2Config, torch.nn.Module],
] = {}


def configure_torch_runtime_for_inference() -> None:
    """Apply conservative torch settings for stable long-running inference."""
    global _TORCH_RUNTIME_CONFIGURED
    if _TORCH_RUNTIME_CONFIGURED:
        return

    num_threads = int(os.environ.get("PROJECTIONS_TORCH_NUM_THREADS", "1"))
    interop_threads = int(
        os.environ.get("PROJECTIONS_TORCH_NUM_INTEROP_THREADS", "1")
    )
    disable_mkldnn = (
        str(os.environ.get("PROJECTIONS_TORCH_DISABLE_MKLDNN", "1"))
        .strip()
        .lower()
        in {"1", "true", "yes"}
    )

    try:
        torch.set_num_threads(max(1, int(num_threads)))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, int(interop_threads)))
    except Exception:
        pass
    if disable_mkldnn:
        try:
            torch.backends.mkldnn.enabled = False
        except Exception:
            pass
    _TORCH_RUNTIME_CONFIGURED = True


def set_inference_seed(seed: int) -> None:
    configure_torch_runtime_for_inference()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def resolve_torch_device(device: str | None) -> torch.device:
    if device:
        return torch.device(str(device))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_gtv2_model(
    bundle_dir: Path,
    *,
    device: torch.device,
    flow_scale_clip_override: float | None = None,
) -> tuple[GameTransformerV2Config, torch.nn.Module]:
    config_path = Path(bundle_dir) / "config.json"
    model_path = Path(bundle_dir) / "model.pt"
    if not config_path.exists():
        raise RuntimeError(f"missing bundle config: {config_path}")
    if not model_path.exists():
        raise RuntimeError(f"missing bundle model: {model_path}")

    cache_key = (
        str(Path(bundle_dir).resolve()),
        str(device),
        float(flow_scale_clip_override) if flow_scale_clip_override is not None else None,
    )
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    config = GameTransformerV2Config.load(config_path)
    model = build_game_transformer_v2(config)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    setattr(model, "gtv2_config", config)
    if flow_scale_clip_override is not None:
        model.flow_head.set_scale_clip(float(flow_scale_clip_override))
    model = model.to(device=device)
    model.eval()
    _MODEL_CACHE[cache_key] = (config, model)
    return config, model


def build_gtv2_inference_examples(
    *,
    features_df: pd.DataFrame,
    game_date: str,
    config: GameTransformerV2Config,
) -> list[Any]:
    frame = features_df.copy()
    frame["game_date"] = pd.Timestamp(game_date).normalize()
    frame["minutes"] = 0.0
    examples = build_game_level_examples(
        frame,
        feature_columns=list(config.feature_columns),
        feature_mean=np.asarray(config.feature_mean, dtype=np.float32),
        feature_std=np.asarray(config.feature_std, dtype=np.float32),
        game_feature_columns=list(config.game_feature_columns),
        team_feature_columns=list(config.team_feature_columns),
        flow_label_columns=[],
        minutes_label_col="minutes",
        min_valid_players_per_team=max(1, int(config.min_active_count)),
        overflow_protected_prior_play_prob_floor=float(
            config.overflow_protected_prior_play_prob_floor
        ),
        overflow_protected_prior_minutes_floor=float(
            config.overflow_protected_prior_minutes_floor
        ),
        overflow_risk_weight_consecutive_active_dnp=float(
            config.overflow_risk_weight_consecutive_active_dnp
        ),
        overflow_risk_weight_active_but_dnp_rate_last10=float(
            config.overflow_risk_weight_active_but_dnp_rate_last10
        ),
        overflow_risk_weight_inactive_streak_len=float(
            config.overflow_risk_weight_inactive_streak_len
        ),
        overflow_keep_weight_prior_play_prob=float(
            config.overflow_keep_weight_prior_play_prob
        ),
        overflow_keep_weight_prior_minutes=float(
            config.overflow_keep_weight_prior_minutes
        ),
    )
    if not examples:
        raise RuntimeError("no valid game examples produced for GTV2 inference")
    return examples


def _build_out_player_lookup(features_df: pd.DataFrame) -> pd.DataFrame:
    lookup = features_df.loc[:, [c for c in ("game_id", "team_id", "player_id", "is_out", "status") if c in features_df.columns]].copy()
    if lookup.empty:
        return pd.DataFrame(columns=["game_id", "team_id", "player_id", "out_player"])
    for col in ("game_id", "team_id", "player_id"):
        if col not in lookup.columns:
            return pd.DataFrame(columns=["game_id", "team_id", "player_id", "out_player"])
        lookup[col] = pd.to_numeric(lookup[col], errors="coerce").astype("Int64")
    status_out = pd.Series(False, index=lookup.index)
    if "status" in lookup.columns:
        status = lookup["status"].fillna("").astype(str).str.upper().str.strip()
        status_out = (
            status.isin({"OUT", "O", "INACTIVE", "D", "DOUBTFUL", "SUSPENDED"})
            | status.str.contains("DOUBT", na=False)
        )
    is_out = pd.Series(False, index=lookup.index)
    if "is_out" in lookup.columns:
        is_out = pd.to_numeric(lookup["is_out"], errors="coerce").fillna(0.0).ge(1.0)
    lookup["out_player"] = (status_out | is_out).astype(bool)
    return lookup.loc[:, ["game_id", "team_id", "player_id", "out_player"]].drop_duplicates(
        subset=["game_id", "team_id", "player_id"], keep="last"
    )


def score_gtv2_features_df(
    *,
    features_df: pd.DataFrame,
    game_date: str,
    config: GameTransformerV2Config,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 4,
) -> pd.DataFrame:
    out_lookup = _build_out_player_lookup(features_df)
    examples = build_gtv2_inference_examples(
        features_df=features_df,
        game_date=game_date,
        config=config,
    )
    loader = DataLoader(
        GameLevelDataset(examples),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_game_level_examples,
    )

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            player_features = batch["player_features"].to(device=device)
            player_valid_mask = batch["player_valid_mask"].to(device=device)
            game_features = batch["game_features"].to(device=device)
            team_features = batch["team_features"].to(device=device)

            out = model(
                player_features,
                player_valid_mask,
                game_features=game_features,
                team_features=team_features,
                sample_active=False,
                run_flow=False,
            )

            valid = batch["player_valid_mask"].cpu().numpy().astype(bool)
            player_ids = batch["player_ids"].cpu().numpy().astype(np.int64)
            team_ids = batch["team_ids"].cpu().numpy().astype(np.int64)
            game_ids = [int(v) for v in batch["game_id_norm"]]
            game_dates = [str(v) for v in batch["game_date"]]

            minutes = out.minutes.minutes.detach().cpu().numpy()
            active_mask = out.active.active_mask.detach().cpu().numpy().astype(bool)
            player_logits = out.active.player_logits.detach().cpu().numpy()
            active_prob = 1.0 / (1.0 + np.exp(-np.clip(player_logits, -40.0, 40.0)))

            for b_idx in range(minutes.shape[0]):
                valid_flat = np.concatenate([valid[b_idx, 0], valid[b_idx, 1]], axis=0)
                player_flat = np.concatenate(
                    [player_ids[b_idx, 0], player_ids[b_idx, 1]], axis=0
                )
                team_flat = np.concatenate(
                    [
                        np.full((15,), int(team_ids[b_idx, 0]), dtype=np.int64),
                        np.full((15,), int(team_ids[b_idx, 1]), dtype=np.int64),
                    ],
                    axis=0,
                )
                for idx in np.where(valid_flat)[0]:
                    rows.append(
                        {
                            "game_date": game_dates[b_idx],
                            "game_id": game_ids[b_idx],
                            "team_id": int(team_flat[idx]),
                            "player_id": int(player_flat[idx]),
                            "minutes_deterministic": float(minutes[b_idx, idx]),
                            "active_deterministic": int(bool(active_mask[b_idx, idx])),
                            "active_logit": float(player_logits[b_idx, idx]),
                            "active_prob_proxy": float(active_prob[b_idx, idx]),
                        }
                    )

    if not rows:
        raise RuntimeError("GTV2 scoring produced zero rows")
    scores = pd.DataFrame(rows)
    if not out_lookup.empty:
        scores = scores.merge(
            out_lookup,
            on=["game_id", "team_id", "player_id"],
            how="left",
        )
        out_mask = scores["out_player"].fillna(False).astype(bool)
        if bool(out_mask.any()):
            scores.loc[out_mask, "minutes_deterministic"] = 0.0
            scores.loc[out_mask, "active_deterministic"] = 0
            scores.loc[out_mask, "active_logit"] = -40.0
            scores.loc[out_mask, "active_prob_proxy"] = 0.0
        scores = scores.drop(columns=["out_player"], errors="ignore")
    scores = scores.sort_values(
        ["game_date", "game_id", "team_id", "player_id"]
    ).reset_index(drop=True)
    return scores
