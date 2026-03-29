"""Compare GTv2 variants under production-aligned post-processing."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from prefect_flows.live_nba_pipeline_v3 import (
    _apply_props_uplift_calibration_to_worlds,
    _apply_world_realism_controls_to_worlds,
    _repair_world_frame_contract_fields,
)
from projections.rotation.game_transformer_v2 import (
    GameLevelDataset,
    GameTransformerV2Config,
    build_game_level_examples,
    build_game_transformer_v2,
    collate_game_level_examples,
)
from projections.rotation.gtv2_promotion_hybrid import (
    BenchRiserHybridConfig,
    PromotionHybridConfig,
    assert_promotion_hybrid_compatible,
)
from projections.rotation.sample_worlds_v2 import (
    JOIN_KEYS,
    MakeModelConfig,
    MinutesUncertaintyConfig,
    _coerce_join_keys,
    _resolve_dataset_dir,
    _resolve_run_dir,
    _split_val,
    sample_worlds_for_batch,
    summarize_worlds_to_projections,
)
from projections.rotation.set_model import zfill_game_id_series
from scripts.rotation.eval_make_rate_calibration import (
    _actual_player_metrics,
    _actual_team_concentration,
    _actual_team_metrics,
    _build_game_meta,
    _build_team_meta,
    _invariant_checks,
    _mae_bias,
    _pred_player_metrics,
    _pred_team_concentration,
    _pred_team_metrics,
    _segment_metrics,
    _span_ratio,
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    run_dir: str
    active_temperature: float
    promotion_expert_run_dir: str | None = None
    promotion_prior_minutes_max: float = 12.0
    promotion_hist_start_rate_max: float = 0.20
    promotion_blend_mode: str = "uplift_only"
    promotion_force_active_candidates: bool = False
    bench_expert_run_dir: str | None = None
    bench_prior_minutes_min: float = 12.0
    bench_prior_play_prob_min: float = 0.80
    bench_implied_minutes_min: float = 12.0
    bench_hist_start_rate_max: float = 0.35
    bench_blend_mode: str = "uplift_only"
    bench_force_active_candidates: bool = False
    make_model: str = "beta_binomial_all"
    allocation_source: str = "emergent"
    allocation_blend_alpha: float = 0.5
    apply_props_uplift: bool = False
    props_uplift_scope: str = "stars_only"
    props_uplift_confidence_weighted: bool = True
    apply_world_realism_controls: bool = True
    minutes_uncertainty_enabled: bool = False
    minutes_uncertainty_mode: str = "gaussian"
    minutes_uncertainty_gaussian_scale: float = 1.0
    minutes_uncertainty_min_sigma: float = 0.75
    minutes_uncertainty_max_sigma: float = 6.0
    minutes_uncertainty_fallback_sigma: float = 1.5
    minutes_uncertainty_use_hurdle_sigma: bool = True
    minutes_uncertainty_use_prior_std: bool = True
    minutes_uncertainty_preserve_top_k_per_team: int = 3
    minutes_uncertainty_full_sigma_at_minutes_or_below: float = 24.0
    minutes_uncertainty_zero_sigma_at_minutes_or_above: float = 32.0
    minutes_uncertainty_dirichlet_base_concentration: float = 24.0


@dataclass(frozen=True)
class EvalContext:
    dataset_dir: Path
    selected_val_df: pd.DataFrame
    selected_features_df: pd.DataFrame
    selected_features_meta: pd.DataFrame
    selected_labels_minutes_df: pd.DataFrame
    selected_labels_counts_df: pd.DataFrame
    selected_game_keys: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-file", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--game-keys-csv",
        type=str,
        default=None,
        help="Optional CSV with columns game_date,game_id to override the default selected validation games.",
    )
    parser.add_argument("--val-days", type=int, default=60)
    parser.add_argument("--num-games", type=int, default=60)
    parser.add_argument("--num-worlds", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--baseline-name", type=str, default="prod_live_exact")
    return parser.parse_args()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_worlds_df(worlds_df: pd.DataFrame) -> pd.DataFrame:
    out = worlds_df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date.astype(str)
    for col in ["game_id", "team_id", "player_id", "world_idx"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(int)
    return out


def _normalize_game_date_str(df: pd.DataFrame, *, col: str = "game_date") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.date.astype(str)
    return out


def _load_variant_specs(path: Path) -> list[VariantSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"variant file must be a list of dicts: {path}")
    specs = [VariantSpec(**item) for item in payload]
    if not specs:
        raise ValueError(f"variant file is empty: {path}")
    return specs


def _load_eval_context(
    dataset_dir: Path,
    *,
    val_days: int,
    num_games: int,
    game_keys_csv: Path | None = None,
) -> EvalContext:
    features_df = _coerce_join_keys(pd.read_parquet(dataset_dir / "features.parquet"), name="features")
    labels_minutes_df = _coerce_join_keys(pd.read_parquet(dataset_dir / "labels_minutes.parquet"), name="labels_minutes")
    labels_counts_df = _coerce_join_keys(pd.read_parquet(dataset_dir / "labels_boxscore_counts.parquet"), name="labels_boxscore_counts")

    label_overlap = [c for c in labels_minutes_df.columns if c in features_df.columns and c not in JOIN_KEYS]
    labels_for_merge = labels_minutes_df.drop(columns=label_overlap)
    merged = features_df.merge(labels_for_merge, on=JOIN_KEYS, how="left", validate="one_to_one")
    merged["game_id_norm"] = zfill_game_id_series(merged["game_id"])
    val_df = _split_val(merged, val_days=int(val_days))

    if game_keys_csv is not None:
        selected_game_keys = pd.read_csv(game_keys_csv)
        missing = {"game_date", "game_id"} - set(selected_game_keys.columns)
        if missing:
            raise ValueError(f"game-keys csv missing required columns {sorted(missing)}: {game_keys_csv}")
        selected_game_keys = selected_game_keys.loc[:, ["game_date", "game_id"]].copy()
        selected_game_keys["game_date"] = pd.to_datetime(selected_game_keys["game_date"], errors="coerce")
        selected_game_keys["game_id"] = pd.to_numeric(selected_game_keys["game_id"], errors="coerce").astype("Int64")
        selected_game_keys = selected_game_keys.dropna(subset=["game_date", "game_id"]).copy()
        selected_game_keys["game_id"] = selected_game_keys["game_id"].astype(int)
        selected_game_keys = selected_game_keys.drop_duplicates().reset_index(drop=True)
    else:
        selected_game_keys = (
            val_df.loc[:, ["game_date", "game_id"]]
            .drop_duplicates()
            .sort_values(["game_date", "game_id"], kind="stable")
            .head(max(1, int(num_games)))
            .reset_index(drop=True)
        )
    selected_val_df = val_df.merge(selected_game_keys, on=["game_date", "game_id"], how="inner")
    selected_features_df = features_df.merge(selected_game_keys, on=["game_date", "game_id"], how="inner")
    selected_labels_minutes_df = labels_minutes_df.merge(selected_game_keys, on=["game_date", "game_id"], how="inner")
    selected_labels_counts_df = labels_counts_df.merge(selected_game_keys, on=["game_date", "game_id"], how="inner")

    feature_meta_cols = [
        "game_date",
        "game_id",
        "team_id",
        "home_flag",
        "is_home",
        "vegas_total",
        "vegas_spread",
        "spread_home",
        "estimated_possessions",
    ]
    use_meta_cols = [c for c in feature_meta_cols if c in selected_features_df.columns]
    selected_features_meta = _normalize_game_date_str(selected_features_df.loc[:, use_meta_cols].copy())

    return EvalContext(
        dataset_dir=dataset_dir,
        selected_val_df=selected_val_df,
        selected_features_df=selected_features_df,
        selected_features_meta=selected_features_meta,
        selected_labels_minutes_df=_normalize_game_date_str(selected_labels_minutes_df),
        selected_labels_counts_df=_normalize_game_date_str(selected_labels_counts_df),
        selected_game_keys=_normalize_game_date_str(selected_game_keys),
    )


def _resolve_device(value: str) -> torch.device:
    requested = str(value).strip().lower()
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _generate_raw_worlds(
    ctx: EvalContext,
    *,
    spec: VariantSpec,
    num_worlds: int,
    batch_size: int,
    chunk_size: int,
    seed: int,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    run_dir = _resolve_run_dir(spec.run_dir)
    config = GameTransformerV2Config.load(run_dir / "config.json")
    model = build_game_transformer_v2(config)
    setattr(model, "gtv2_config", config)
    state = torch.load(run_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device=device)
    model.eval()
    promotion_expert_model: torch.nn.Module | None = None
    promotion_hybrid_config: PromotionHybridConfig | None = None
    promotion_expert_run_dir: Path | None = None
    if spec.promotion_expert_run_dir:
        promotion_expert_run_dir = _resolve_run_dir(str(spec.promotion_expert_run_dir))
        promotion_expert_cfg = GameTransformerV2Config.load(promotion_expert_run_dir / "config.json")
        assert_promotion_hybrid_compatible(config, promotion_expert_cfg)
        promotion_expert_model = build_game_transformer_v2(promotion_expert_cfg)
        setattr(promotion_expert_model, "gtv2_config", promotion_expert_cfg)
        expert_state = torch.load(promotion_expert_run_dir / "model.pt", map_location="cpu")
        promotion_expert_model.load_state_dict(expert_state)
        promotion_expert_model = promotion_expert_model.to(device=device)
        promotion_expert_model.eval()
        promotion_hybrid_config = PromotionHybridConfig.from_model_config(
            config,
            prior_minutes_max=float(spec.promotion_prior_minutes_max),
            hist_start_rate_max=float(spec.promotion_hist_start_rate_max),
            uplift_only=(str(spec.promotion_blend_mode).strip().lower() == "uplift_only"),
            force_active_candidates=bool(spec.promotion_force_active_candidates),
        )
    bench_expert_model: torch.nn.Module | None = None
    bench_hybrid_config: BenchRiserHybridConfig | None = None
    bench_expert_run_dir: Path | None = None
    if spec.bench_expert_run_dir:
        bench_expert_run_dir = _resolve_run_dir(str(spec.bench_expert_run_dir))
        bench_expert_cfg = GameTransformerV2Config.load(bench_expert_run_dir / "config.json")
        if list(config.feature_columns) != list(bench_expert_cfg.feature_columns):
            raise ValueError("Bench expert feature_columns must exactly match primary model feature_columns")
        if list(config.game_feature_columns) != list(bench_expert_cfg.game_feature_columns):
            raise ValueError("Bench expert game_feature_columns must exactly match primary model game_feature_columns")
        if list(config.team_feature_columns) != list(bench_expert_cfg.team_feature_columns):
            raise ValueError("Bench expert team_feature_columns must exactly match primary model team_feature_columns")
        bench_expert_model = build_game_transformer_v2(bench_expert_cfg)
        setattr(bench_expert_model, "gtv2_config", bench_expert_cfg)
        expert_state = torch.load(bench_expert_run_dir / "model.pt", map_location="cpu")
        bench_expert_model.load_state_dict(expert_state)
        bench_expert_model = bench_expert_model.to(device=device)
        bench_expert_model.eval()
        bench_hybrid_config = BenchRiserHybridConfig.from_model_config(
            config,
            prior_minutes_min=float(spec.bench_prior_minutes_min),
            prior_play_prob_min=float(spec.bench_prior_play_prob_min),
            implied_minutes_min=float(spec.bench_implied_minutes_min),
            hist_start_rate_max=float(spec.bench_hist_start_rate_max),
            uplift_only=(str(spec.bench_blend_mode).strip().lower() == "uplift_only"),
            force_active_candidates=bool(spec.bench_force_active_candidates),
        )

    examples = build_game_level_examples(
        ctx.selected_val_df,
        feature_columns=list(config.feature_columns),
        feature_mean=np.asarray(config.feature_mean, dtype=np.float32),
        feature_std=np.asarray(config.feature_std, dtype=np.float32),
        game_feature_columns=list(config.game_feature_columns),
        team_feature_columns=list(config.team_feature_columns),
        flow_label_columns=None,
        minutes_label_col="minutes_label" if "minutes_label" in ctx.selected_val_df.columns else "minutes",
        overflow_protected_prior_play_prob_floor=float(config.overflow_protected_prior_play_prob_floor),
        overflow_protected_prior_minutes_floor=float(config.overflow_protected_prior_minutes_floor),
        overflow_risk_weight_consecutive_active_dnp=float(config.overflow_risk_weight_consecutive_active_dnp),
        overflow_risk_weight_active_but_dnp_rate_last10=float(config.overflow_risk_weight_active_but_dnp_rate_last10),
        overflow_risk_weight_inactive_streak_len=float(config.overflow_risk_weight_inactive_streak_len),
        overflow_keep_weight_prior_play_prob=float(config.overflow_keep_weight_prior_play_prob),
        overflow_keep_weight_prior_minutes=float(config.overflow_keep_weight_prior_minutes),
    )
    loader = DataLoader(
        GameLevelDataset(examples),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_game_level_examples,
    )
    make_model_config = MakeModelConfig(mode=str(spec.make_model))
    minutes_uncertainty_config = MinutesUncertaintyConfig(
        enabled=bool(spec.minutes_uncertainty_enabled),
        mode=str(spec.minutes_uncertainty_mode),
        gaussian_scale=float(spec.minutes_uncertainty_gaussian_scale),
        min_sigma=float(spec.minutes_uncertainty_min_sigma),
        max_sigma=float(spec.minutes_uncertainty_max_sigma),
        fallback_sigma=float(spec.minutes_uncertainty_fallback_sigma),
        use_hurdle_sigma=bool(spec.minutes_uncertainty_use_hurdle_sigma),
        use_prior_std=bool(spec.minutes_uncertainty_use_prior_std),
        preserve_top_k_per_team=int(spec.minutes_uncertainty_preserve_top_k_per_team),
        full_sigma_at_minutes_or_below=float(spec.minutes_uncertainty_full_sigma_at_minutes_or_below),
        zero_sigma_at_minutes_or_above=float(spec.minutes_uncertainty_zero_sigma_at_minutes_or_above),
        dirichlet_base_concentration=float(spec.minutes_uncertainty_dirichlet_base_concentration),
    )

    frames: list[pd.DataFrame] = []
    contract_counter: dict[str, int] = {}
    for batch in loader:
        df_batch, checks = sample_worlds_for_batch(
            model,
            batch,
            device=device,
            num_worlds=int(num_worlds),
            chunk_size=max(1, int(chunk_size)),
            active_temperature=float(spec.active_temperature),
            strict_contracts=True,
            attempt_conditioning_mode="predicted_attempts",
            make_model_config=make_model_config,
            allocation_source=str(spec.allocation_source),
            allocation_blend_alpha=float(spec.allocation_blend_alpha),
            minutes_uncertainty_config=minutes_uncertainty_config,
            promotion_expert_model=promotion_expert_model,
            promotion_hybrid_config=promotion_hybrid_config,
            bench_expert_model=bench_expert_model,
            bench_hybrid_config=bench_hybrid_config,
        )
        frames.append(df_batch)
        for key, value in checks.items():
            contract_counter[str(key)] = int(contract_counter.get(str(key), 0) + int(value))

    raw_worlds = _normalize_worlds_df(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())

    del model
    if promotion_expert_model is not None:
        del promotion_expert_model
    if bench_expert_model is not None:
        del bench_expert_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return raw_worlds, {
        "run_dir": str(run_dir),
        "promotion_expert_run_dir": str(promotion_expert_run_dir) if promotion_expert_run_dir is not None else None,
        "bench_expert_run_dir": str(bench_expert_run_dir) if bench_expert_run_dir is not None else None,
        "promotion_hybrid": (
            {
                "prior_minutes_max": float(promotion_hybrid_config.prior_minutes_max),
                "hist_start_rate_max": float(promotion_hybrid_config.hist_start_rate_max),
                "blend_mode": str(spec.promotion_blend_mode),
            }
            if promotion_hybrid_config is not None
            else None
        ),
        "bench_hybrid": (
            {
                "prior_minutes_min": float(bench_hybrid_config.prior_minutes_min),
                "prior_play_prob_min": float(bench_hybrid_config.prior_play_prob_min),
                "implied_minutes_min": float(bench_hybrid_config.implied_minutes_min),
                "hist_start_rate_max": float(bench_hybrid_config.hist_start_rate_max),
                "blend_mode": str(spec.bench_blend_mode),
            }
            if bench_hybrid_config is not None
            else None
        ),
        "contract_checks": contract_counter,
        "num_examples": int(len(examples)),
        "minutes_uncertainty": {
            "enabled": bool(spec.minutes_uncertainty_enabled),
            "mode": str(spec.minutes_uncertainty_mode),
            "gaussian_scale": float(spec.minutes_uncertainty_gaussian_scale),
            "use_hurdle_sigma": bool(spec.minutes_uncertainty_use_hurdle_sigma),
            "use_prior_std": bool(spec.minutes_uncertainty_use_prior_std),
        },
    }


def _apply_live_postprocessing(
    raw_worlds: pd.DataFrame,
    *,
    ctx: EvalContext,
    spec: VariantSpec,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    worlds = raw_worlds.copy()
    if bool(spec.apply_props_uplift):
        worlds, props_report = _apply_props_uplift_calibration_to_worlds(
            worlds,
            features_df=ctx.selected_features_df.copy(),
            scope=str(spec.props_uplift_scope),
            confidence_weighted=bool(spec.props_uplift_confidence_weighted),
        )
    else:
        props_report = {"applied": False, "reason": "disabled"}

    worlds, realism_report = _apply_world_realism_controls_to_worlds(
        worlds,
        enabled=bool(spec.apply_world_realism_controls),
        random_seed=int(seed),
        low_minutes_tail_damping_enabled=True,
        low_minutes_tail_minutes_threshold=12.0,
        low_minutes_tail_min_scale=0.55,
        outlier_resample_enabled=True,
        outlier_resample_max_passes=1,
        target_game_ids=None,
    )
    worlds, repair_report = _repair_world_frame_contract_fields(worlds)
    return worlds, {
        "props_uplift": props_report,
        "world_realism_controls": realism_report,
        "world_contract_repair": repair_report,
    }


def _calibration_payload(
    worlds: pd.DataFrame,
    *,
    ctx: EvalContext,
    name: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = ctx.selected_labels_counts_df.copy()
    keys = worlds[["game_date", "game_id", "team_id", "player_id"]].drop_duplicates()
    labels = labels.merge(keys, on=["game_date", "game_id", "team_id", "player_id"], how="inner")
    required_actual_cols = [
        "fga2",
        "fg2m",
        "fga3",
        "fg3m",
        "fta",
        "ftm",
        "oreb",
        "dreb",
        "ast",
        "stl",
        "blk",
        "tov",
    ]
    labels = labels.dropna(subset=[col for col in required_actual_cols if col in labels.columns]).copy()
    if labels.empty:
        raise ValueError(f"no overlap between worlds and labels for {name}")

    pred_team, poss_diag = _pred_team_metrics(worlds)
    act_team = _actual_team_metrics(labels)
    team_eval = pred_team.merge(act_team, on=["game_date", "game_id", "team_id"], how="inner")

    pred_player = _pred_player_metrics(worlds)
    act_player = _actual_player_metrics(labels)
    player_eval = pred_player.merge(act_player, on=["game_date", "game_id", "team_id", "player_id"], how="inner")
    player_eval = player_eval.merge(
        pred_team[["game_date", "game_id", "team_id", "pred_fga"]],
        on=["game_date", "game_id", "team_id"],
        how="left",
    ).merge(
        act_team[["game_date", "game_id", "team_id", "act_fga"]],
        on=["game_date", "game_id", "team_id"],
        how="left",
    )
    player_eval["act_fga_share"] = np.where(
        pd.to_numeric(player_eval["act_fga"], errors="coerce").fillna(0.0) > 0.0,
        pd.to_numeric(player_eval["actual_fga"], errors="coerce").fillna(0.0)
        / pd.to_numeric(player_eval["act_fga"], errors="coerce").fillna(1.0),
        np.nan,
    )
    player_eval["pred_fga_share"] = np.where(
        pd.to_numeric(player_eval["pred_fga"], errors="coerce").fillna(0.0) > 0.0,
        pd.to_numeric(player_eval["pred_fga_mean"], errors="coerce").fillna(0.0)
        / pd.to_numeric(player_eval["pred_fga"], errors="coerce").fillna(1.0),
        np.nan,
    )

    team_meta = _build_team_meta(ctx.selected_features_meta.copy())
    game_meta = _build_game_meta(team_meta)

    spread_eval = None
    if not game_meta.empty:
        pred_home = pred_team.merge(
            game_meta[["game_date", "game_id", "home_team_id"]],
            left_on=["game_date", "game_id", "team_id"],
            right_on=["game_date", "game_id", "home_team_id"],
            how="inner",
        ).rename(columns={"pred_pts": "pred_home_pts", "pred_poss": "pred_home_poss"})
        pred_away = pred_team.merge(
            game_meta[["game_date", "game_id", "away_team_id"]],
            left_on=["game_date", "game_id", "team_id"],
            right_on=["game_date", "game_id", "away_team_id"],
            how="inner",
        ).rename(columns={"pred_pts": "pred_away_pts", "pred_poss": "pred_away_poss"})
        spread_eval = pred_home.merge(
            pred_away,
            on=["game_date", "game_id"],
            how="inner",
            suffixes=("_home", "_away"),
        ).merge(game_meta, on=["game_date", "game_id"], how="left")
        spread_eval["pred_spread"] = spread_eval["pred_home_pts"] - spread_eval["pred_away_pts"]
        spread_eval["pred_total"] = spread_eval["pred_home_pts"] + spread_eval["pred_away_pts"]
        spread_eval["pred_game_poss"] = 0.5 * (
            spread_eval["pred_home_poss"] + spread_eval["pred_away_poss"]
        )

        act_home = act_team.merge(
            game_meta[["game_date", "game_id", "home_team_id"]],
            left_on=["game_date", "game_id", "team_id"],
            right_on=["game_date", "game_id", "home_team_id"],
            how="inner",
        ).rename(columns={"act_pts": "act_home_pts"})
        act_away = act_team.merge(
            game_meta[["game_date", "game_id", "away_team_id"]],
            left_on=["game_date", "game_id", "team_id"],
            right_on=["game_date", "game_id", "away_team_id"],
            how="inner",
        ).rename(columns={"act_pts": "act_away_pts"})
        act_spread = act_home.merge(act_away, on=["game_date", "game_id"], how="inner")
        spread_eval = spread_eval.merge(
            act_spread[["game_date", "game_id", "act_home_pts", "act_away_pts"]],
            on=["game_date", "game_id"],
            how="left",
        )
        spread_eval["act_spread"] = spread_eval["act_home_pts"] - spread_eval["act_away_pts"]
        spread_eval["act_total"] = spread_eval["act_home_pts"] + spread_eval["act_away_pts"]

    pred_conc = _pred_team_concentration(worlds)
    act_conc = _actual_team_concentration(_actual_player_metrics(labels))
    conc_eval = pred_conc.merge(act_conc, on=["game_date", "game_id", "team_id"], how="inner")

    poss_bias, poss_mae = _mae_bias(team_eval["pred_poss"], team_eval["act_poss"])
    fga_bias, fga_mae = _mae_bias(team_eval["pred_fga"], team_eval["act_fga"])
    fta_bias, fta_mae = _mae_bias(team_eval["pred_fta"], team_eval["act_fta"])
    pts_bias, pts_mae = _mae_bias(team_eval["pred_pts"], team_eval["act_pts"])
    fg_pct_bias, fg_pct_mae = _mae_bias(team_eval["pred_fg_pct"], team_eval["act_fg_pct"])
    ft_pct_bias, ft_pct_mae = _mae_bias(team_eval["pred_ft_pct"], team_eval["act_ft_pct"])

    cov90 = (player_eval["actual_dk_fpts"] <= player_eval["pred_dk_p90"]).astype(float)
    cov95 = (player_eval["actual_dk_fpts"] <= player_eval["pred_dk_p95"]).astype(float)
    p90_cov = float(cov90.mean()) if len(cov90) else float("nan")
    p95_cov = float(cov95.mean()) if len(cov95) else float("nan")

    elite_mask = player_eval["actual_pts"] >= 35.0
    star_mask = (player_eval["actual_pts"] >= 25.0) & (player_eval["actual_pts"] < 35.0)
    high_usage_mask = player_eval["actual_fga"] >= 18.0
    ultra_usage_mask = player_eval["actual_fga"] >= 22.0
    elite_bias, elite_mae = _mae_bias(
        player_eval.loc[elite_mask, "pred_pts_mean"],
        player_eval.loc[elite_mask, "actual_pts"],
    )
    star_bias, star_mae = _mae_bias(
        player_eval.loc[star_mask, "pred_pts_mean"],
        player_eval.loc[star_mask, "actual_pts"],
    )
    star_seg = _segment_metrics(player_eval, star_mask)
    elite_seg = _segment_metrics(player_eval, elite_mask)
    high_usage_seg = _segment_metrics(player_eval, high_usage_mask)
    ultra_usage_seg = _segment_metrics(player_eval, ultra_usage_mask)

    top1_bias, top1_mae = _mae_bias(conc_eval["pred_top1_share_pts"], conc_eval["act_top1_share_pts"])
    top2_bias, top2_mae = _mae_bias(conc_eval["pred_top2_share_pts"], conc_eval["act_top2_share_pts"])

    spread_bias_vs_vegas = spread_mae_vs_vegas = spread_corr_vs_vegas = float("nan")
    total_bias_vs_vegas = total_mae_vs_vegas = total_corr_vs_vegas = float("nan")
    spread_bias_vs_actual = spread_mae_vs_actual = float("nan")
    total_bias_vs_actual = total_mae_vs_actual = float("nan")
    game_poss_bias_vs_est = game_poss_mae_vs_est = poss_corr_vs_est = float("nan")
    spread_span_ratio = total_span_ratio = float("nan")
    if spread_eval is not None and not spread_eval.empty:
        if "vegas_spread" in spread_eval.columns:
            spread_eval["vegas_spread_home_margin"] = -spread_eval["vegas_spread"]
            spread_bias_vs_vegas, spread_mae_vs_vegas = _mae_bias(
                spread_eval["pred_spread"],
                spread_eval["vegas_spread_home_margin"],
            )
            if spread_eval["vegas_spread_home_margin"].notna().any():
                spread_corr_vs_vegas = float(
                    spread_eval[["pred_spread", "vegas_spread_home_margin"]].corr().iloc[0, 1]
                )
            spread_span_ratio = _span_ratio(
                spread_eval["pred_spread"],
                spread_eval["vegas_spread_home_margin"],
            )
        if "vegas_total" in spread_eval.columns:
            total_bias_vs_vegas, total_mae_vs_vegas = _mae_bias(
                spread_eval["pred_total"],
                spread_eval["vegas_total"],
            )
            if spread_eval["vegas_total"].notna().any():
                total_corr_vs_vegas = float(spread_eval[["pred_total", "vegas_total"]].corr().iloc[0, 1])
            total_span_ratio = _span_ratio(spread_eval["pred_total"], spread_eval["vegas_total"])
        if "act_spread" in spread_eval.columns:
            spread_bias_vs_actual, spread_mae_vs_actual = _mae_bias(
                spread_eval["pred_spread"],
                spread_eval["act_spread"],
            )
        if "act_total" in spread_eval.columns:
            total_bias_vs_actual, total_mae_vs_actual = _mae_bias(
                spread_eval["pred_total"],
                spread_eval["act_total"],
            )
        if "estimated_possessions" in spread_eval.columns:
            game_poss_bias_vs_est, game_poss_mae_vs_est = _mae_bias(
                spread_eval["pred_game_poss"],
                spread_eval["estimated_possessions"],
            )
            if spread_eval["estimated_possessions"].notna().any():
                poss_corr_vs_est = float(
                    spread_eval[["pred_game_poss", "estimated_possessions"]].corr().iloc[0, 1]
                )

    payload: dict[str, Any] = {
        "created_at": _utc_now_iso(),
        "name": str(name),
        "dataset_dir": str(ctx.dataset_dir),
        "counts": {
            "n_team_games": int(len(team_eval)),
            "n_player_games": int(len(player_eval)),
            "n_world_rows": int(len(worlds)),
        },
        "invariants": _invariant_checks(worlds),
        "metrics": {
            "poss_bias_mean": poss_bias,
            "poss_mae": poss_mae,
            "fga_bias_mean": fga_bias,
            "fga_mae": fga_mae,
            "fta_bias_mean": fta_bias,
            "fta_mae": fta_mae,
            "pts_bias_mean": pts_bias,
            "pts_mae": pts_mae,
            "pred_fg_pct_mean": float(team_eval["pred_fg_pct"].mean()) if len(team_eval) else float("nan"),
            "act_fg_pct_mean": float(team_eval["act_fg_pct"].mean()) if len(team_eval) else float("nan"),
            "fg_pct_bias_mean": fg_pct_bias,
            "fg_pct_mae": fg_pct_mae,
            "pred_ft_pct_mean": float(team_eval["pred_ft_pct"].mean()) if len(team_eval) else float("nan"),
            "act_ft_pct_mean": float(team_eval["act_ft_pct"].mean()) if len(team_eval) else float("nan"),
            "ft_pct_bias_mean": ft_pct_bias,
            "ft_pct_mae": ft_pct_mae,
            "p90_coverage": p90_cov,
            "p90_calibration_error_abs": float(abs(p90_cov - 0.90)) if np.isfinite(p90_cov) else float("nan"),
            "p95_coverage": p95_cov,
            "p95_calibration_error_abs": float(abs(p95_cov - 0.95)) if np.isfinite(p95_cov) else float("nan"),
            "elite_bias_pts_35plus": elite_bias,
            "elite_mae_pts_35plus": elite_mae,
            "elite_n_35plus": int(elite_mask.sum()),
            "star_bias_pts_25_34": star_bias,
            "star_mae_pts_25_34": star_mae,
            "star_n_25_34": int(star_mask.sum()),
            "star_fga_bias_25_34": star_seg["fga_bias"],
            "star_fga_mae_25_34": star_seg["fga_mae"],
            "star_fga_share_bias_25_34": star_seg["fga_share_bias"],
            "star_fga_share_mae_25_34": star_seg["fga_share_mae"],
            "elite_fga_bias_35plus": elite_seg["fga_bias"],
            "elite_fga_mae_35plus": elite_seg["fga_mae"],
            "elite_fga_share_bias_35plus": elite_seg["fga_share_bias"],
            "elite_fga_share_mae_35plus": elite_seg["fga_share_mae"],
            "high_usage_bias_pts_18plus": high_usage_seg["pts_bias"],
            "high_usage_mae_pts_18plus": high_usage_seg["pts_mae"],
            "high_usage_n_18plus": int(high_usage_seg["n"]),
            "high_usage_fga_bias_18plus": high_usage_seg["fga_bias"],
            "high_usage_fga_mae_18plus": high_usage_seg["fga_mae"],
            "high_usage_fga_share_bias_18plus": high_usage_seg["fga_share_bias"],
            "high_usage_fga_share_mae_18plus": high_usage_seg["fga_share_mae"],
            "ultra_usage_bias_pts_22plus": ultra_usage_seg["pts_bias"],
            "ultra_usage_mae_pts_22plus": ultra_usage_seg["pts_mae"],
            "ultra_usage_n_22plus": int(ultra_usage_seg["n"]),
            "ultra_usage_fga_bias_22plus": ultra_usage_seg["fga_bias"],
            "ultra_usage_fga_mae_22plus": ultra_usage_seg["fga_mae"],
            "ultra_usage_fga_share_bias_22plus": ultra_usage_seg["fga_share_bias"],
            "ultra_usage_fga_share_mae_22plus": ultra_usage_seg["fga_share_mae"],
            "spread_bias_vs_vegas": spread_bias_vs_vegas,
            "spread_mae_vs_vegas": spread_mae_vs_vegas,
            "spread_corr_vs_vegas": spread_corr_vs_vegas,
            "spread_bias_vs_actual": spread_bias_vs_actual,
            "spread_mae_vs_actual": spread_mae_vs_actual,
            "total_bias_vs_vegas": total_bias_vs_vegas,
            "total_mae_vs_vegas": total_mae_vs_vegas,
            "total_corr_vs_vegas": total_corr_vs_vegas,
            "total_bias_vs_actual": total_bias_vs_actual,
            "total_mae_vs_actual": total_mae_vs_actual,
            "game_poss_bias_vs_est": game_poss_bias_vs_est,
            "game_poss_mae_vs_est": game_poss_mae_vs_est,
            "game_poss_corr_vs_est": poss_corr_vs_est,
            "spread_span_ratio": spread_span_ratio,
            "total_span_ratio": total_span_ratio,
            "top1_share_bias_pts": top1_bias,
            "top1_share_mae_pts": top1_mae,
            "top2_share_bias_pts": top2_bias,
            "top2_share_mae_pts": top2_mae,
            "top1_share_pred_mean": float(conc_eval["pred_top1_share_pts"].mean()) if len(conc_eval) else float("nan"),
            "top1_share_act_mean": float(conc_eval["act_top1_share_pts"].mean()) if len(conc_eval) else float("nan"),
            "top2_share_pred_mean": float(conc_eval["pred_top2_share_pts"].mean()) if len(conc_eval) else float("nan"),
            "top2_share_act_mean": float(conc_eval["act_top2_share_pts"].mean()) if len(conc_eval) else float("nan"),
            **poss_diag,
        },
    }
    return payload, team_eval, player_eval, spread_eval if spread_eval is not None else pd.DataFrame()


def _actual_dk_fpts(df: pd.DataFrame) -> pd.Series:
    actual_pts = 2.0 * pd.to_numeric(df["fg2m"], errors="coerce").fillna(0.0)
    actual_pts += 3.0 * pd.to_numeric(df["fg3m"], errors="coerce").fillna(0.0)
    actual_pts += pd.to_numeric(df["ftm"], errors="coerce").fillna(0.0)
    actual_reb = pd.to_numeric(df["oreb"], errors="coerce").fillna(0.0) + pd.to_numeric(
        df["dreb"], errors="coerce"
    ).fillna(0.0)
    ast = pd.to_numeric(df["ast"], errors="coerce").fillna(0.0)
    stl = pd.to_numeric(df["stl"], errors="coerce").fillna(0.0)
    blk = pd.to_numeric(df["blk"], errors="coerce").fillna(0.0)
    tov = pd.to_numeric(df["tov"], errors="coerce").fillna(0.0)
    base = actual_pts + 1.25 * actual_reb + 1.5 * ast + 2.0 * stl + 2.0 * blk - 0.5 * tov
    qualifiers = pd.concat(
        [
            (actual_pts >= 10.0).astype(int),
            (actual_reb >= 10.0).astype(int),
            (ast >= 10.0).astype(int),
            (stl >= 10.0).astype(int),
            (blk >= 10.0).astype(int),
        ],
        axis=1,
    ).sum(axis=1)
    return base + np.where(qualifiers == 2, 1.5, 0.0) + np.where(qualifiers >= 3, 3.0, 0.0)


def _player_summary_payload(
    worlds: pd.DataFrame,
    *,
    ctx: EvalContext,
    spec: VariantSpec,
) -> tuple[dict[str, Any], pd.DataFrame]:
    work = worlds.copy()
    work["pred_reb"] = pd.to_numeric(work["oreb"], errors="coerce").fillna(0.0) + pd.to_numeric(
        work["dreb"], errors="coerce"
    ).fillna(0.0)
    pred = (
        work.groupby(["game_date", "game_id", "team_id", "player_id"], as_index=False)
        .agg(
            pred_minutes=("minutes", "mean"),
            pred_pts=("pts", "mean"),
            pred_reb=("pred_reb", "mean"),
            pred_ast=("ast", "mean"),
            pred_stl=("stl", "mean"),
            pred_blk=("blk", "mean"),
            pred_tov=("tov", "mean"),
            pred_dk_fpts=("dk_fpts", "mean"),
            pred_active_prob=("minutes", lambda x: float(np.mean(np.asarray(x, dtype=float) > 0.0))),
            pred_active_prob_4=("minutes", lambda x: float(np.mean(np.asarray(x, dtype=float) >= 4.0))),
        )
        .copy()
    )

    minutes_actual = ctx.selected_labels_minutes_df.copy()
    minutes_col = "minutes_label" if "minutes_label" in minutes_actual.columns else "minutes"
    minutes_actual = minutes_actual.loc[:, ["game_date", "game_id", "team_id", "player_id", minutes_col]].rename(
        columns={minutes_col: "actual_minutes"}
    )
    counts_actual = ctx.selected_labels_counts_df.copy()
    counts_actual["actual_pts"] = (
        2.0 * pd.to_numeric(counts_actual["fg2m"], errors="coerce").fillna(0.0)
        + 3.0 * pd.to_numeric(counts_actual["fg3m"], errors="coerce").fillna(0.0)
        + pd.to_numeric(counts_actual["ftm"], errors="coerce").fillna(0.0)
    )
    counts_actual["actual_reb"] = pd.to_numeric(counts_actual["oreb"], errors="coerce").fillna(0.0) + pd.to_numeric(
        counts_actual["dreb"], errors="coerce"
    ).fillna(0.0)
    counts_actual["actual_dk_fpts"] = _actual_dk_fpts(counts_actual)
    actual = counts_actual.merge(
        minutes_actual,
        on=["game_date", "game_id", "team_id", "player_id"],
        how="left",
        validate="one_to_one",
    )
    actual = actual.loc[
        :,
        [
            "game_date",
            "game_id",
            "team_id",
            "player_id",
            "actual_minutes",
            "actual_pts",
            "actual_reb",
            "ast",
            "stl",
            "blk",
            "tov",
            "actual_dk_fpts",
        ],
    ].rename(
        columns={
            "ast": "actual_ast",
            "stl": "actual_stl",
            "blk": "actual_blk",
            "tov": "actual_tov",
        }
    )
    merged = pred.merge(
        actual,
        on=["game_date", "game_id", "team_id", "player_id"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("player summary merge is empty")

    metrics: dict[str, float] = {}
    for stat in ["minutes", "pts", "reb", "ast", "stl", "blk", "tov", "dk_fpts"]:
        pred_col = f"pred_{stat}"
        actual_col = f"actual_{stat}"
        bias, mae = _mae_bias(merged[pred_col], merged[actual_col])
        metrics[f"{stat}_bias_mean"] = bias
        metrics[f"{stat}_mae"] = mae
    pred_active_mask = pd.to_numeric(merged["pred_minutes"], errors="coerce").fillna(0.0) >= 4.0
    actual_active_mask = pd.to_numeric(merged["actual_minutes"], errors="coerce").fillna(0.0) >= 4.0
    metrics["active_acc_at4"] = float((pred_active_mask == actual_active_mask).mean())
    metrics["pred_active_rate_at4"] = float(pred_active_mask.mean())
    metrics["actual_active_rate_at4"] = float(actual_active_mask.mean())
    metrics["pred_active_prob_mean"] = float(pd.to_numeric(merged["pred_active_prob"], errors="coerce").mean())
    metrics["pred_active_prob_4_mean"] = float(pd.to_numeric(merged["pred_active_prob_4"], errors="coerce").mean())

    feature_cols = [
        "game_date",
        "game_id",
        "team_id",
        "player_id",
        "lineup_starter_announced",
        "is_projected_starter",
        "is_confirmed_starter",
        "minutes_from_stints_prior_20",
        "prior_play_prob",
        "an_implied_minutes",
        "recent_start_pct_10",
        "started_proxy_rate_prior_10",
        "started_proxy_rate_prior_20",
    ]
    available_feature_cols = [col for col in feature_cols if col in ctx.selected_features_df.columns]
    feature_frame = ctx.selected_features_df.loc[:, available_feature_cols].copy()
    for col in feature_cols:
        if col not in feature_frame.columns:
            feature_frame[col] = 0.0
    feature_frame = _normalize_game_date_str(feature_frame)
    promo = merged.merge(
        feature_frame,
        on=["game_date", "game_id", "team_id", "player_id"],
        how="left",
        validate="one_to_one",
    )
    starter_signal = np.zeros(len(promo), dtype=bool)
    for col in ("lineup_starter_announced", "is_projected_starter", "is_confirmed_starter"):
        starter_signal |= pd.to_numeric(promo[col], errors="coerce").fillna(0.0).to_numpy(dtype=float) >= 0.5
    hist_start_rate = np.maximum.reduce(
        [
            pd.to_numeric(promo["recent_start_pct_10"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            pd.to_numeric(promo["started_proxy_rate_prior_10"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            pd.to_numeric(promo["started_proxy_rate_prior_20"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        ]
    )
    prior_minutes = pd.to_numeric(promo["minutes_from_stints_prior_20"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    promo_mask = (
        starter_signal
        & (prior_minutes <= float(spec.promotion_prior_minutes_max))
        & (hist_start_rate <= float(spec.promotion_hist_start_rate_max))
    )
    promo_next_up_mask = promo_mask & (
        pd.to_numeric(promo["actual_minutes"], errors="coerce").fillna(0.0).to_numpy(dtype=float) >= 20.0
    )
    if bool(promo_mask.any()):
        promo_minutes_bias, promo_minutes_mae = _mae_bias(
            promo.loc[promo_mask, "pred_minutes"],
            promo.loc[promo_mask, "actual_minutes"],
        )
    else:
        promo_minutes_bias, promo_minutes_mae = float("nan"), float("nan")
    metrics["starter_promotion_slice_n"] = float(int(promo_mask.sum()))
    metrics["starter_promotion_next_up_n"] = float(int(promo_next_up_mask.sum()))
    metrics["starter_promotion_pred_minutes_mean"] = (
        float(pd.to_numeric(promo.loc[promo_mask, "pred_minutes"], errors="coerce").mean())
        if bool(promo_mask.any())
        else float("nan")
    )
    metrics["starter_promotion_minutes_bias_mean"] = promo_minutes_bias
    metrics["starter_promotion_minutes_mae"] = promo_minutes_mae
    metrics["starter_promotion_active_recall_at4"] = (
        float((pd.to_numeric(promo.loc[promo_next_up_mask, "pred_minutes"], errors="coerce").fillna(0.0) >= 4.0).mean())
        if bool(promo_next_up_mask.any())
        else float("nan")
    )
    metrics["starter_promotion_under10_rate"] = (
        float((pd.to_numeric(promo.loc[promo_next_up_mask, "pred_minutes"], errors="coerce").fillna(0.0) < 10.0).mean())
        if bool(promo_next_up_mask.any())
        else float("nan")
    )
    metrics["starter_promotion_low8_rate"] = (
        float((pd.to_numeric(promo.loc[promo_next_up_mask, "pred_minutes"], errors="coerce").fillna(0.0) < 8.0).mean())
        if bool(promo_next_up_mask.any())
        else float("nan")
    )

    nonstarter_mask = ~starter_signal
    prior_play_prob = pd.to_numeric(promo["prior_play_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    implied_minutes = pd.to_numeric(promo["an_implied_minutes"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    bench_mask = (
        nonstarter_mask
        & (prior_minutes >= float(spec.bench_prior_minutes_min))
        & (prior_play_prob >= float(spec.bench_prior_play_prob_min))
        & (implied_minutes >= float(spec.bench_implied_minutes_min))
        & (hist_start_rate <= float(spec.bench_hist_start_rate_max))
    )
    bench_next_up_mask = bench_mask & (
        pd.to_numeric(promo["actual_minutes"], errors="coerce").fillna(0.0).to_numpy(dtype=float) >= 20.0
    )
    bench_core_mask = bench_mask & (
        pd.to_numeric(promo["actual_minutes"], errors="coerce").fillna(0.0).to_numpy(dtype=float) >= 32.0
    )
    if bool(bench_mask.any()):
        bench_minutes_bias, bench_minutes_mae = _mae_bias(
            promo.loc[bench_mask, "pred_minutes"],
            promo.loc[bench_mask, "actual_minutes"],
        )
    else:
        bench_minutes_bias, bench_minutes_mae = float("nan"), float("nan")
    metrics["bench_riser_slice_n"] = float(int(bench_mask.sum()))
    metrics["bench_riser_next_up_n"] = float(int(bench_next_up_mask.sum()))
    metrics["bench_core_next_up_n"] = float(int(bench_core_mask.sum()))
    metrics["bench_riser_pred_minutes_mean"] = (
        float(pd.to_numeric(promo.loc[bench_mask, "pred_minutes"], errors="coerce").mean())
        if bool(bench_mask.any())
        else float("nan")
    )
    metrics["bench_riser_minutes_bias_mean"] = bench_minutes_bias
    metrics["bench_riser_minutes_mae"] = bench_minutes_mae
    metrics["bench_riser_active_recall_at4"] = (
        float((pd.to_numeric(promo.loc[bench_next_up_mask, "pred_minutes"], errors="coerce").fillna(0.0) >= 4.0).mean())
        if bool(bench_next_up_mask.any())
        else float("nan")
    )
    metrics["bench_riser_under16_rate"] = (
        float((pd.to_numeric(promo.loc[bench_next_up_mask, "pred_minutes"], errors="coerce").fillna(0.0) < 16.0).mean())
        if bool(bench_next_up_mask.any())
        else float("nan")
    )
    metrics["bench_riser_low8_rate"] = (
        float((pd.to_numeric(promo.loc[bench_next_up_mask, "pred_minutes"], errors="coerce").fillna(0.0) < 8.0).mean())
        if bool(bench_next_up_mask.any())
        else float("nan")
    )

    return {
        "counts": {
            "n_player_games": int(len(merged)),
        },
        "metrics": metrics,
    }, merged


def _flatten_variant_row(
    spec: VariantSpec,
    *,
    raw_worlds: pd.DataFrame,
    post_worlds: pd.DataFrame,
    generation_meta: dict[str, Any],
    postprocess_meta: dict[str, Any],
    calibration_payload: dict[str, Any],
    player_summary_payload: dict[str, Any],
) -> dict[str, Any]:
    cal_metrics = calibration_payload["metrics"]
    player_metrics = player_summary_payload["metrics"]
    props_meta = postprocess_meta["props_uplift"]
    realism_meta = postprocess_meta["world_realism_controls"]
    repair_meta = postprocess_meta["world_contract_repair"]
    return {
        "variant": spec.name,
        "run_dir": spec.run_dir,
        "promotion_expert_run_dir": str(spec.promotion_expert_run_dir) if spec.promotion_expert_run_dir else None,
        "bench_expert_run_dir": str(spec.bench_expert_run_dir) if spec.bench_expert_run_dir else None,
        "promotion_blend_mode": str(spec.promotion_blend_mode),
        "promotion_force_active_candidates": bool(spec.promotion_force_active_candidates),
        "bench_blend_mode": str(spec.bench_blend_mode),
        "bench_force_active_candidates": bool(spec.bench_force_active_candidates),
        "active_temperature": float(spec.active_temperature),
        "make_model": str(spec.make_model),
        "allocation_source": str(spec.allocation_source),
        "apply_props_uplift": bool(spec.apply_props_uplift),
        "props_uplift_scope": str(spec.props_uplift_scope),
        "apply_world_realism_controls": bool(spec.apply_world_realism_controls),
        "minutes_uncertainty_enabled": bool(spec.minutes_uncertainty_enabled),
        "minutes_uncertainty_mode": str(spec.minutes_uncertainty_mode),
        "n_games": int(len(post_worlds[["game_date", "game_id"]].drop_duplicates())),
        "raw_world_rows": int(len(raw_worlds)),
        "post_world_rows": int(len(post_worlds)),
        "props_uplift_applied": bool(props_meta.get("applied", False)),
        "props_total_adjusted_players": int(props_meta.get("total_adjusted_players", 0) or 0),
        "world_realism_applied": bool(realism_meta.get("applied", False)),
        "contract_repair_applied": bool(repair_meta.get("applied", False)),
        "dk_fpts_mae": player_metrics["dk_fpts_mae"],
        "dk_fpts_bias_mean": player_metrics["dk_fpts_bias_mean"],
        "minutes_mae": player_metrics["minutes_mae"],
        "minutes_bias_mean": player_metrics["minutes_bias_mean"],
        "active_acc_at4": player_metrics["active_acc_at4"],
        "starter_promotion_slice_n": player_metrics["starter_promotion_slice_n"],
        "starter_promotion_next_up_n": player_metrics["starter_promotion_next_up_n"],
        "starter_promotion_pred_minutes_mean": player_metrics["starter_promotion_pred_minutes_mean"],
        "starter_promotion_minutes_mae": player_metrics["starter_promotion_minutes_mae"],
        "starter_promotion_active_recall_at4": player_metrics["starter_promotion_active_recall_at4"],
        "starter_promotion_under10_rate": player_metrics["starter_promotion_under10_rate"],
        "starter_promotion_low8_rate": player_metrics["starter_promotion_low8_rate"],
        "bench_riser_slice_n": player_metrics["bench_riser_slice_n"],
        "bench_riser_next_up_n": player_metrics["bench_riser_next_up_n"],
        "bench_core_next_up_n": player_metrics["bench_core_next_up_n"],
        "bench_riser_pred_minutes_mean": player_metrics["bench_riser_pred_minutes_mean"],
        "bench_riser_minutes_mae": player_metrics["bench_riser_minutes_mae"],
        "bench_riser_active_recall_at4": player_metrics["bench_riser_active_recall_at4"],
        "bench_riser_under16_rate": player_metrics["bench_riser_under16_rate"],
        "bench_riser_low8_rate": player_metrics["bench_riser_low8_rate"],
        "pts_mae_player": player_metrics["pts_mae"],
        "reb_mae_player": player_metrics["reb_mae"],
        "ast_mae_player": player_metrics["ast_mae"],
        "stl_mae_player": player_metrics["stl_mae"],
        "blk_mae_player": player_metrics["blk_mae"],
        "pts_bias_mean_player": player_metrics["pts_bias_mean"],
        "reb_bias_mean_player": player_metrics["reb_bias_mean"],
        "ast_bias_mean_player": player_metrics["ast_bias_mean"],
        "stl_bias_mean_player": player_metrics["stl_bias_mean"],
        "blk_bias_mean_player": player_metrics["blk_bias_mean"],
        "pts_mae_team": cal_metrics["pts_mae"],
        "pts_bias_mean_team": cal_metrics["pts_bias_mean"],
        "spread_mae_vs_vegas": cal_metrics["spread_mae_vs_vegas"],
        "spread_corr_vs_vegas": cal_metrics["spread_corr_vs_vegas"],
        "total_mae_vs_vegas": cal_metrics["total_mae_vs_vegas"],
        "total_corr_vs_vegas": cal_metrics["total_corr_vs_vegas"],
        "p90_calibration_error_abs": cal_metrics["p90_calibration_error_abs"],
        "p95_calibration_error_abs": cal_metrics["p95_calibration_error_abs"],
        "top1_share_bias_pts": cal_metrics["top1_share_bias_pts"],
        "top2_share_bias_pts": cal_metrics["top2_share_bias_pts"],
        "poss_mae": cal_metrics["poss_mae"],
        "poss_sym_abs_p95": cal_metrics["poss_sym_abs_p95"],
        "generation_contract_checks": json.dumps(generation_meta["contract_checks"], sort_keys=True),
    }


def _compare_vs_baseline(summary_df: pd.DataFrame, *, baseline_name: str) -> pd.DataFrame:
    baseline = summary_df.loc[summary_df["variant"] == baseline_name]
    if baseline.empty:
        raise ValueError(f"baseline variant not found: {baseline_name}")
    base_row = baseline.iloc[0]
    delta_rows: list[dict[str, Any]] = []
    compare_cols = [
        "dk_fpts_mae",
        "minutes_mae",
        "active_acc_at4",
        "starter_promotion_pred_minutes_mean",
        "starter_promotion_minutes_mae",
        "starter_promotion_active_recall_at4",
        "starter_promotion_under10_rate",
        "starter_promotion_low8_rate",
        "bench_riser_pred_minutes_mean",
        "bench_riser_minutes_mae",
        "bench_riser_active_recall_at4",
        "bench_riser_under16_rate",
        "bench_riser_low8_rate",
        "pts_mae_player",
        "reb_mae_player",
        "ast_mae_player",
        "stl_mae_player",
        "blk_mae_player",
        "spread_mae_vs_vegas",
        "spread_corr_vs_vegas",
        "total_mae_vs_vegas",
        "total_corr_vs_vegas",
        "p90_calibration_error_abs",
        "p95_calibration_error_abs",
        "top1_share_bias_pts",
        "top2_share_bias_pts",
        "poss_mae",
    ]
    for _, row in summary_df.iterrows():
        item = {"variant": row["variant"], "baseline_variant": baseline_name}
        for col in compare_cols:
            item[f"delta_{col}"] = float(row[col]) - float(base_row[col])
        delta_rows.append(item)
    return pd.DataFrame(delta_rows)


def main() -> None:
    args = parse_args()
    variant_file = Path(args.variant_file).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = _resolve_dataset_dir(args.dataset_dir)
    specs = _load_variant_specs(variant_file)
    game_keys_csv = Path(args.game_keys_csv).expanduser().resolve() if args.game_keys_csv else None
    ctx = _load_eval_context(
        dataset_dir,
        val_days=int(args.val_days),
        num_games=int(args.num_games),
        game_keys_csv=game_keys_csv,
    )
    ctx.selected_game_keys.to_csv(out_dir / "selected_games.csv", index=False)

    device = _resolve_device(args.device)
    summary_rows: list[dict[str, Any]] = []
    for spec in specs:
        variant_dir = out_dir / spec.name
        variant_dir.mkdir(parents=True, exist_ok=True)

        raw_worlds, generation_meta = _generate_raw_worlds(
            ctx,
            spec=spec,
            num_worlds=int(args.num_worlds),
            batch_size=int(args.batch_size),
            chunk_size=int(args.chunk_size),
            seed=int(args.seed),
            device=device,
        )
        raw_worlds.to_parquet(variant_dir / "raw_worlds.parquet", index=False)

        post_worlds, postprocess_meta = _apply_live_postprocessing(
            raw_worlds,
            ctx=ctx,
            spec=spec,
            seed=int(args.seed),
        )
        post_worlds.to_parquet(variant_dir / "worlds.parquet", index=False)
        projections_df = summarize_worlds_to_projections(post_worlds, sim_profile="game_transformer_v2")
        projections_df.to_parquet(variant_dir / "projections.parquet", index=False)

        calibration_payload, team_eval, player_eval, spread_eval = _calibration_payload(
            post_worlds,
            ctx=ctx,
            name=spec.name,
        )
        player_summary_payload, player_summary_df = _player_summary_payload(post_worlds, ctx=ctx, spec=spec)

        (variant_dir / "generation_meta.json").write_text(
            json.dumps(generation_meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (variant_dir / "postprocess_meta.json").write_text(
            json.dumps(postprocess_meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (variant_dir / "calibration.json").write_text(
            json.dumps(calibration_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (variant_dir / "player_summary.json").write_text(
            json.dumps(player_summary_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        team_eval.to_csv(variant_dir / "team_eval.csv", index=False)
        player_eval.to_csv(variant_dir / "calibration_player_eval.csv", index=False)
        player_summary_df.to_csv(variant_dir / "player_summary_eval.csv", index=False)
        if not spread_eval.empty:
            spread_eval.to_csv(variant_dir / "spread_eval.csv", index=False)

        summary_rows.append(
            _flatten_variant_row(
                spec,
                raw_worlds=raw_worlds,
                post_worlds=post_worlds,
                generation_meta=generation_meta,
                postprocess_meta=postprocess_meta,
                calibration_payload=calibration_payload,
                player_summary_payload=player_summary_payload,
            )
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["variant"], kind="stable").reset_index(drop=True)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    delta_df = _compare_vs_baseline(summary_df, baseline_name=str(args.baseline_name))
    delta_df.to_csv(out_dir / "compare_vs_baseline.csv", index=False)

    manifest = {
        "created_at": _utc_now_iso(),
        "variant_file": str(variant_file),
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "val_days": int(args.val_days),
        "num_games": int(args.num_games),
        "num_worlds": int(args.num_worlds),
        "batch_size": int(args.batch_size),
        "chunk_size": int(args.chunk_size),
        "seed": int(args.seed),
        "device": str(device),
        "baseline_name": str(args.baseline_name),
        "variants": [asdict(spec) for spec in specs],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(summary_df.to_csv(index=False))
    print(f"saved_summary={out_dir / 'summary.csv'}")
    print(f"saved_delta={out_dir / 'compare_vs_baseline.csv'}")


if __name__ == "__main__":
    main()
