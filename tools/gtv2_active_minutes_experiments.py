#!/usr/bin/env python3
"""Run targeted GTV2 active/minutes diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
from projections.pipeline.gtv2_inference_runtime import (
    build_gtv2_inference_examples,
    load_gtv2_model,
    resolve_torch_device,
    score_gtv2_features_df,
    set_inference_seed,
)
from projections.rotation.game_transformer_v2 import (
    GameLevelDataset,
    build_game_level_examples,
    collate_game_level_examples,
)
from projections.rotation.joint_active_set import build_active_set_labels
from projections.rotation.sample_worlds_v2 import (
    JOIN_KEYS,
    MakeModelConfig,
    _coerce_join_keys,
    _split_val,
    sample_worlds_for_batch,
)
from projections.rotation.set_model import zfill_game_id_series


@dataclass(frozen=True)
class WorldVariant:
    name: str
    bundle_dir: str
    active_temperature: float
    make_model: str = "beta_binomial_all"
    allocation_source: str = "emergent"
    allocation_blend_alpha: float = 0.5
    apply_props_uplift: bool = True
    props_uplift_scope: str = "stars_only"
    props_uplift_confidence_weighted: bool = True
    apply_world_realism_controls: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-features", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--old-bundle", type=Path, required=True)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-worlds", type=int, default=10000)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _normalize_features_snapshot(features_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(features_path)
    if "game_date" not in df.columns:
        try:
            inferred_game_date = str(features_path.parents[1].name)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"snapshot features missing game_date and path is not date-scoped: {features_path}") from exc
        df["game_date"] = inferred_game_date
    df = _coerce_join_keys(df, name="snapshot_features")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date.astype(str)
    return df


def _player_meta(features_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "game_date",
        "game_id",
        "team_id",
        "player_id",
        "player_name",
        "an_pts_line",
        "an_implied_minutes",
        "lineup_starter_announced",
        "an_has_any_props",
        "prior_play_prob",
        "minutes_from_stints_prior_20",
    ]
    use_cols = [c for c in cols if c in features_df.columns]
    meta = features_df.loc[:, use_cols].copy()
    for col in ("an_pts_line", "an_implied_minutes", "prior_play_prob", "minutes_from_stints_prior_20"):
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce")
    if "lineup_starter_announced" in meta.columns:
        meta["lineup_starter_announced"] = (
            pd.to_numeric(meta["lineup_starter_announced"], errors="coerce").fillna(0.0).astype(float)
        )
    if "an_has_any_props" in meta.columns:
        meta["an_has_any_props"] = pd.to_numeric(meta["an_has_any_props"], errors="coerce").fillna(0.0).astype(float)
    return meta.drop_duplicates(subset=["game_date", "game_id", "team_id", "player_id"])


def _game_meta(features_df: pd.DataFrame) -> pd.DataFrame:
    df = features_df.copy()
    if "home_flag" not in df.columns and "is_home" in df.columns:
        df["home_flag"] = pd.to_numeric(df["is_home"], errors="coerce").fillna(0.0)
    if "home_flag" not in df.columns:
        raise ValueError("snapshot features missing home_flag/is_home")
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    df["home_flag"] = pd.to_numeric(df["home_flag"], errors="coerce").fillna(0.0)
    grouped = []
    for (game_date, game_id), grp in df.groupby(["game_date", "game_id"], sort=False):
        home_rows = grp.loc[grp["home_flag"] >= 0.5]
        away_rows = grp.loc[grp["home_flag"] < 0.5]
        home_team_id = int(pd.to_numeric(home_rows["team_id"], errors="coerce").dropna().iloc[0]) if not home_rows.empty else -1
        away_team_id = int(pd.to_numeric(away_rows["team_id"], errors="coerce").dropna().iloc[0]) if not away_rows.empty else -1
        vegas_total = float(pd.to_numeric(grp.get("vegas_total"), errors="coerce").dropna().iloc[0]) if "vegas_total" in grp.columns and not pd.to_numeric(grp.get("vegas_total"), errors="coerce").dropna().empty else float("nan")
        spread_home = None
        for col in ("spread_home", "vegas_spread"):
            if col in grp.columns:
                vals = pd.to_numeric(grp[col], errors="coerce").dropna()
                if not vals.empty:
                    spread_home = float(vals.iloc[0])
                    break
        grouped.append(
            {
                "game_date": str(game_date),
                "game_id": int(game_id),
                "home_team_id": int(home_team_id),
                "away_team_id": int(away_team_id),
                "vegas_total": float(vegas_total),
                "spread_home_market": float(spread_home) if spread_home is not None else float("nan"),
            }
        )
    return pd.DataFrame(grouped)


def _build_world_variants(old_bundle: Path, candidate_bundle: Path) -> list[WorldVariant]:
    variants = [
        WorldVariant(
            name="old_prod_t1_live_exact",
            bundle_dir=str(old_bundle),
            active_temperature=1.0,
        )
    ]
    for temp in (0.35, 0.45, 0.55, 0.70, 1.0):
        variants.append(
            WorldVariant(
                name=f"cand_t{str(temp).replace('.', '')}_live_exact",
                bundle_dir=str(candidate_bundle),
                active_temperature=float(temp),
            )
        )
    return variants


def _build_snapshot_loader(
    *,
    features_df: pd.DataFrame,
    config: Any,
    batch_size: int,
) -> DataLoader:
    examples = build_gtv2_inference_examples(
        features_df=features_df,
        game_date=str(features_df["game_date"].iloc[0]),
        config=config,
    )
    return DataLoader(
        GameLevelDataset(examples),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_game_level_examples,
    )


def _generate_worlds_for_variant(
    *,
    variant: WorldVariant,
    features_df: pd.DataFrame,
    device: torch.device,
    num_worlds: int,
    chunk_size: int,
    batch_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    set_inference_seed(int(seed))
    config, model = load_gtv2_model(Path(variant.bundle_dir), device=device)
    loader = _build_snapshot_loader(features_df=features_df, config=config, batch_size=batch_size)
    make_model_config = MakeModelConfig(mode=str(variant.make_model))

    raw_frames: list[pd.DataFrame] = []
    checks_total: dict[str, int] = {}
    for batch in loader:
        df_batch, checks = sample_worlds_for_batch(
            model,
            batch,
            device=device,
            num_worlds=int(num_worlds),
            chunk_size=max(1, int(chunk_size)),
            active_temperature=float(variant.active_temperature),
            strict_contracts=True,
            attempt_conditioning_mode="predicted_attempts",
            make_model_config=make_model_config,
            allocation_source=str(variant.allocation_source),
            allocation_blend_alpha=float(variant.allocation_blend_alpha),
        )
        raw_frames.append(df_batch)
        for key, value in checks.items():
            checks_total[str(key)] = int(checks_total.get(str(key), 0) + int(value))
    raw_worlds = pd.concat(raw_frames, ignore_index=True)

    player_meta = _player_meta(features_df)
    final_worlds = raw_worlds.copy()
    post_report: dict[str, Any] = {
        "props_uplift_calibration": {"applied": False, "reason": "disabled"},
        "world_realism_controls": {"applied": False, "reason": "disabled"},
        "world_contract_repair": {"applied": False, "reason": "disabled"},
    }
    if bool(variant.apply_props_uplift):
        final_worlds, uplift_report = _apply_props_uplift_calibration_to_worlds(
            final_worlds,
            features_df=features_df,
            scope=str(variant.props_uplift_scope),
            confidence_weighted=bool(variant.props_uplift_confidence_weighted),
        )
        post_report["props_uplift_calibration"] = uplift_report
    if bool(variant.apply_world_realism_controls):
        final_worlds, realism_report = _apply_world_realism_controls_to_worlds(
            final_worlds,
            enabled=True,
            random_seed=int(seed),
            low_minutes_tail_damping_enabled=True,
            low_minutes_tail_minutes_threshold=12.0,
            low_minutes_tail_min_scale=0.55,
            outlier_resample_enabled=True,
            outlier_resample_max_passes=1,
            target_game_ids=None,
        )
        post_report["world_realism_controls"] = realism_report
    final_worlds, repair_report = _repair_world_frame_contract_fields(final_worlds)
    post_report["world_contract_repair"] = repair_report
    return raw_worlds, final_worlds, post_report, player_meta


def _world_team_game_metrics(worlds_df: pd.DataFrame, game_meta: pd.DataFrame) -> pd.DataFrame:
    team_world = (
        worlds_df.groupby(["game_date", "game_id", "world_idx", "team_id"], sort=False, observed=True)["pts"]
        .sum()
        .reset_index(name="team_pts")
    )
    merged = team_world.merge(game_meta, on=["game_date", "game_id"], how="left", validate="many_to_one")
    merged["is_home"] = merged["team_id"].astype(int).eq(merged["home_team_id"].astype(int))
    home = (
        merged.loc[merged["is_home"], ["game_date", "game_id", "world_idx", "team_pts"]]
        .rename(columns={"team_pts": "home_pts"})
    )
    away = (
        merged.loc[~merged["is_home"], ["game_date", "game_id", "world_idx", "team_pts"]]
        .rename(columns={"team_pts": "away_pts"})
    )
    game_world = home.merge(away, on=["game_date", "game_id", "world_idx"], how="inner", validate="one_to_one")
    game_world["total_pts"] = game_world["home_pts"] + game_world["away_pts"]
    game_world["spread_home"] = game_world["home_pts"] - game_world["away_pts"]
    return game_world.merge(game_meta, on=["game_date", "game_id"], how="left", validate="many_to_one")


def _player_means(worlds_df: pd.DataFrame, player_meta: pd.DataFrame) -> pd.DataFrame:
    means = (
        worlds_df.groupby(["game_date", "game_id", "team_id", "player_id"], sort=False, observed=True)[
            ["minutes", "pts", "dk_fpts"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"minutes": "minutes_mean", "pts": "pts_mean", "dk_fpts": "dk_fpts_mean"})
    )
    return means.merge(player_meta, on=["game_date", "game_id", "team_id", "player_id"], how="left")


def _concentration_metrics(player_means: pd.DataFrame) -> dict[str, float]:
    rows: list[tuple[float, float]] = []
    for _, grp in player_means.groupby(["game_date", "game_id", "team_id"], sort=False):
        pts = pd.to_numeric(grp["pts_mean"], errors="coerce").fillna(0.0).sort_values(ascending=False).to_numpy()
        total = float(np.sum(pts))
        if total <= 0.0:
            continue
        top1 = float(pts[0] / total) if pts.size >= 1 else 0.0
        top2 = float(np.sum(pts[:2]) / total) if pts.size >= 2 else top1
        rows.append((top1, top2))
    if not rows:
        return {"top1_share_mean": float("nan"), "top2_share_mean": float("nan")}
    arr = np.asarray(rows, dtype=float)
    return {
        "top1_share_mean": float(np.mean(arr[:, 0])),
        "top2_share_mean": float(np.mean(arr[:, 1])),
    }


def _world_summary_metrics(
    *,
    worlds_df: pd.DataFrame,
    player_meta: pd.DataFrame,
    game_meta: pd.DataFrame,
    post_report: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    game_world = _world_team_game_metrics(worlds_df, game_meta)
    game_summary = (
        game_world.groupby(["game_date", "game_id", "vegas_total", "spread_home_market"], sort=False, observed=True)[
            ["total_pts", "spread_home"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"total_pts": "total_mean", "spread_home": "spread_home_mean"})
    )
    game_summary["total_bias_vs_vegas"] = game_summary["total_mean"] - pd.to_numeric(
        game_summary["vegas_total"], errors="coerce"
    ).fillna(0.0)
    game_summary["spread_bias_vs_vegas"] = game_summary["spread_home_mean"] - pd.to_numeric(
        game_summary["spread_home_market"], errors="coerce"
    ).fillna(0.0)

    player_means = _player_means(worlds_df, player_meta)
    concentration = _concentration_metrics(player_means)

    prop_star_mask = pd.Series(False, index=player_means.index)
    if "an_pts_line" in player_means.columns:
        prop_star_mask |= pd.to_numeric(player_means["an_pts_line"], errors="coerce").fillna(0.0).ge(20.0)
    if "an_implied_minutes" in player_means.columns:
        prop_star_mask |= pd.to_numeric(player_means["an_implied_minutes"], errors="coerce").fillna(0.0).ge(30.0)
    star_minutes_mean = float(
        pd.to_numeric(player_means.loc[prop_star_mask, "minutes_mean"], errors="coerce").mean()
    ) if bool(prop_star_mask.any()) else float("nan")

    outlier_report = (post_report.get("world_realism_controls") or {}).get("outlier_resample") or {}
    passes = outlier_report.get("passes") or []
    first_pass = passes[0] if passes else {}
    summary = {
        "games": int(game_summary.shape[0]),
        "players": int(player_means.shape[0]),
        "total_bias_vs_vegas": float(pd.to_numeric(game_summary["total_bias_vs_vegas"], errors="coerce").mean()),
        "total_mae_vs_vegas": float(pd.to_numeric(game_summary["total_bias_vs_vegas"], errors="coerce").abs().mean()),
        "spread_bias_vs_vegas": float(pd.to_numeric(game_summary["spread_bias_vs_vegas"], errors="coerce").mean()),
        "spread_mae_vs_vegas": float(pd.to_numeric(game_summary["spread_bias_vs_vegas"], errors="coerce").abs().mean()),
        "prop_star_minutes_mean": float(star_minutes_mean),
        "bad_pair_count": int(first_pass.get("bad_pair_count", 0)),
        "replaced_pair_count": int(first_pass.get("replaced_pair_count", 0)),
        "low_minutes_affected_rows": int(
            ((post_report.get("world_realism_controls") or {}).get("low_minutes_tail_damping") or {}).get(
                "affected_rows",
                0,
            )
        ),
        **concentration,
    }
    return summary, game_summary, player_means


def _compute_same_snapshot_sweep(
    *,
    variants: list[WorldVariant],
    features_df: pd.DataFrame,
    device: torch.device,
    out_dir: Path,
    num_worlds: int,
    chunk_size: int,
    batch_size: int,
    seed: int,
) -> None:
    game_meta = _game_meta(features_df)
    summary_rows: list[dict[str, Any]] = []
    deterministic_rows: list[dict[str, Any]] = []
    top_prop_minutes_rows: list[pd.DataFrame] = []
    for variant in variants:
        variant_dir = out_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)

        config, model = load_gtv2_model(Path(variant.bundle_dir), device=device)
        scores = score_gtv2_features_df(
            features_df=features_df,
            game_date=str(features_df["game_date"].iloc[0]),
            config=config,
            model=model,
            device=device,
            batch_size=max(1, int(batch_size)),
        )
        score_meta = scores.merge(_player_meta(features_df), on=["game_date", "game_id", "team_id", "player_id"], how="left")
        star_mask = pd.Series(False, index=score_meta.index)
        if "an_pts_line" in score_meta.columns:
            star_mask |= pd.to_numeric(score_meta["an_pts_line"], errors="coerce").fillna(0.0).ge(20.0)
        if "an_implied_minutes" in score_meta.columns:
            star_mask |= pd.to_numeric(score_meta["an_implied_minutes"], errors="coerce").fillna(0.0).ge(30.0)
        deterministic_rows.append(
            {
                "variant": variant.name,
                "bundle_dir": variant.bundle_dir,
                "active_temperature": float(variant.active_temperature),
                "det_minutes_mean": float(pd.to_numeric(score_meta["minutes_deterministic"], errors="coerce").mean()),
                "det_active_prob_mean": float(pd.to_numeric(score_meta["active_prob_proxy"], errors="coerce").mean()),
                "det_prop_star_minutes_mean": float(
                    pd.to_numeric(score_meta.loc[star_mask, "minutes_deterministic"], errors="coerce").mean()
                ) if bool(star_mask.any()) else float("nan"),
            }
        )

        raw_worlds, final_worlds, post_report, player_meta = _generate_worlds_for_variant(
            variant=variant,
            features_df=features_df,
            device=device,
            num_worlds=num_worlds,
            chunk_size=chunk_size,
            batch_size=batch_size,
            seed=seed,
        )
        raw_summary, raw_games, raw_players = _world_summary_metrics(
            worlds_df=raw_worlds,
            player_meta=player_meta,
            game_meta=game_meta,
            post_report={},
        )
        final_summary, final_games, final_players = _world_summary_metrics(
            worlds_df=final_worlds,
            player_meta=player_meta,
            game_meta=game_meta,
            post_report=post_report,
        )
        summary_rows.append(
            {
                "variant": variant.name,
                "bundle_dir": variant.bundle_dir,
                "active_temperature": float(variant.active_temperature),
                "raw_total_bias_vs_vegas": raw_summary["total_bias_vs_vegas"],
                "raw_total_mae_vs_vegas": raw_summary["total_mae_vs_vegas"],
                "raw_spread_mae_vs_vegas": raw_summary["spread_mae_vs_vegas"],
                "raw_prop_star_minutes_mean": raw_summary["prop_star_minutes_mean"],
                "raw_top1_share_mean": raw_summary["top1_share_mean"],
                "raw_top2_share_mean": raw_summary["top2_share_mean"],
                "final_total_bias_vs_vegas": final_summary["total_bias_vs_vegas"],
                "final_total_mae_vs_vegas": final_summary["total_mae_vs_vegas"],
                "final_spread_mae_vs_vegas": final_summary["spread_mae_vs_vegas"],
                "final_prop_star_minutes_mean": final_summary["prop_star_minutes_mean"],
                "final_top1_share_mean": final_summary["top1_share_mean"],
                "final_top2_share_mean": final_summary["top2_share_mean"],
                "bad_pair_count": final_summary["bad_pair_count"],
                "replaced_pair_count": final_summary["replaced_pair_count"],
                "low_minutes_affected_rows": final_summary["low_minutes_affected_rows"],
                "props_adjusted_players": int((post_report.get("props_uplift_calibration") or {}).get("total_adjusted_players", 0)),
            }
        )

        raw_games.to_csv(variant_dir / "game_metrics_raw.csv", index=False)
        final_games.to_csv(variant_dir / "game_metrics_final.csv", index=False)
        raw_players.to_csv(variant_dir / "player_means_raw.csv", index=False)
        final_players.to_csv(variant_dir / "player_means_final.csv", index=False)
        score_meta.to_csv(variant_dir / "deterministic_scores.csv", index=False)
        (variant_dir / "postprocess_report.json").write_text(json.dumps(post_report, indent=2), encoding="utf-8")

        top_prop = final_players.copy()
        if "an_pts_line" in top_prop.columns:
            top_prop["an_pts_line"] = pd.to_numeric(top_prop["an_pts_line"], errors="coerce")
            top_prop = top_prop.sort_values(["an_pts_line", "minutes_mean"], ascending=[False, False], kind="stable")
        else:
            top_prop = top_prop.sort_values(["minutes_mean"], ascending=False, kind="stable")
        top_prop = top_prop.head(20).copy()
        top_prop.insert(0, "variant", variant.name)
        top_prop_minutes_rows.append(top_prop)

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(deterministic_rows).to_csv(out_dir / "deterministic_summary.csv", index=False)
    if top_prop_minutes_rows:
        pd.concat(top_prop_minutes_rows, ignore_index=True).to_csv(out_dir / "top_prop_minutes.csv", index=False)


def _load_val_frame(dataset_dir: Path, *, val_days: int) -> pd.DataFrame:
    features_df = _coerce_join_keys(pd.read_parquet(dataset_dir / "features.parquet"), name="features")
    labels_minutes_df = _coerce_join_keys(pd.read_parquet(dataset_dir / "labels_minutes.parquet"), name="labels_minutes")
    label_overlap = [c for c in labels_minutes_df.columns if c in features_df.columns and c not in JOIN_KEYS]
    labels_for_merge = labels_minutes_df.drop(columns=label_overlap)
    merged = features_df.merge(labels_for_merge, on=JOIN_KEYS, how="left", validate="one_to_one")
    merged["game_id_norm"] = zfill_game_id_series(merged["game_id"])
    return _split_val(merged, val_days=int(val_days))


def _team_index(batch_size: int, n_team_slots: int, device: torch.device) -> torch.Tensor:
    home = torch.zeros((batch_size, n_team_slots), dtype=torch.long, device=device)
    away = torch.ones((batch_size, n_team_slots), dtype=torch.long, device=device)
    return torch.cat([home, away], dim=1)


def _active_metrics(pred_active: np.ndarray, target_active: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    mask = valid_mask.astype(bool)
    pred = pred_active.astype(bool) & mask
    target = target_active.astype(bool) & mask
    tp = int(np.count_nonzero(pred & target))
    fp = int(np.count_nonzero(pred & (~target)))
    fn = int(np.count_nonzero((~pred) & target))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    return {
        "active_precision": precision,
        "active_recall": recall,
        "active_fp": fp,
        "active_fn": fn,
    }


def _run_forced_active_ablation(
    *,
    bundle_dir: Path,
    dataset_dir: Path,
    device: torch.device,
    val_days: int,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config, model = load_gtv2_model(bundle_dir, device=device)
    val_df = _load_val_frame(dataset_dir, val_days=val_days)
    examples = build_game_level_examples(
        val_df,
        feature_columns=list(config.feature_columns),
        feature_mean=np.asarray(config.feature_mean, dtype=np.float32),
        feature_std=np.asarray(config.feature_std, dtype=np.float32),
        game_feature_columns=list(config.game_feature_columns),
        team_feature_columns=list(config.team_feature_columns),
        flow_label_columns=[],
        minutes_label_col="minutes_label" if "minutes_label" in val_df.columns else "minutes",
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
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_game_level_examples,
    )

    rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            player_features = batch["player_features"].to(device=device)
            player_valid_mask = batch["player_valid_mask"].to(device=device)
            game_features = batch["game_features"].to(device=device)
            team_features = batch["team_features"].to(device=device)
            y_minutes = batch["y_minutes"].to(device=device)

            y_flat = torch.cat([y_minutes[:, 0], y_minutes[:, 1]], dim=1)
            valid_flat = torch.cat([player_valid_mask[:, 0], player_valid_mask[:, 1]], dim=1)
            team_index = _team_index(batch_size=y_minutes.shape[0], n_team_slots=y_minutes.shape[2], device=device)
            labels = build_active_set_labels(
                y_flat,
                valid_flat,
                team_index,
                threshold=float(config.active_threshold_minutes),
                min_active_count=int(config.min_active_count),
                max_active_count=int(config.max_active_count),
            )
            target_active_mask_2d = labels.active_mask.view(y_minutes.shape[0], 2, y_minutes.shape[2])

            kwargs: dict[str, Any] = {
                "game_features": game_features,
                "team_features": team_features,
                "sample_active": False,
                "active_temperature": 1.0,
                "run_flow": False,
            }
            if mode in {"minutes_forced", "full_forced"}:
                kwargs["target_active_mask"] = target_active_mask_2d
                kwargs["minutes_use_target_active"] = True
            if mode == "full_forced":
                kwargs["use_target_active_mask"] = True
            out = model(player_features, player_valid_mask, **kwargs)

            pred_minutes = out.minutes.minutes.detach().cpu().numpy()
            pred_active = out.active.active_mask.detach().cpu().numpy().astype(bool)
            target_active = labels.active_mask.detach().cpu().numpy().astype(bool)
            valid_np = valid_flat.detach().cpu().numpy().astype(bool)
            actual_minutes = y_flat.detach().cpu().numpy()
            active_prob = 1.0 / (1.0 + np.exp(-np.clip(out.active.player_logits.detach().cpu().numpy(), -40.0, 40.0)))

            player_ids = batch["player_ids"].cpu().numpy().astype(np.int64)
            team_ids = batch["team_ids"].cpu().numpy().astype(np.int64)
            lineup_available = batch["lineup_available"].cpu().numpy().astype(bool)
            game_ids = [int(v) for v in batch["game_id_norm"]]
            game_dates = [str(v) for v in batch["game_date"]]
            pred_counts = out.active.sampled_counts.detach().cpu().numpy().astype(int)
            target_counts = labels.count_targets.detach().cpu().numpy().astype(int)

            for b_idx in range(pred_minutes.shape[0]):
                flat_player_ids = np.concatenate([player_ids[b_idx, 0], player_ids[b_idx, 1]], axis=0)
                flat_team_ids = np.concatenate(
                    [
                        np.full((player_ids.shape[2],), int(team_ids[b_idx, 0]), dtype=np.int64),
                        np.full((player_ids.shape[2],), int(team_ids[b_idx, 1]), dtype=np.int64),
                    ],
                    axis=0,
                )
                flat_lineup = np.concatenate([lineup_available[b_idx, 0], lineup_available[b_idx, 1]], axis=0)
                for idx in np.where(valid_np[b_idx])[0]:
                    rows.append(
                        {
                            "game_date": game_dates[b_idx],
                            "game_id": game_ids[b_idx],
                            "team_id": int(flat_team_ids[idx]),
                            "player_id": int(flat_player_ids[idx]),
                            "actual_minutes": float(actual_minutes[b_idx, idx]),
                            "pred_minutes": float(pred_minutes[b_idx, idx]),
                            "pred_active": int(bool(pred_active[b_idx, idx])),
                            "target_active": int(bool(target_active[b_idx, idx])),
                            "active_prob_proxy": float(active_prob[b_idx, idx]),
                            "lineup_available": int(bool(flat_lineup[idx])),
                        }
                    )
                for team_idx in (0, 1):
                    team_rows.append(
                        {
                            "game_date": game_dates[b_idx],
                            "game_id": game_ids[b_idx],
                            "team_side": int(team_idx),
                            "pred_count": int(pred_counts[b_idx, team_idx]),
                            "target_count": int(target_counts[b_idx, team_idx]),
                        }
                    )

    row_df = pd.DataFrame(rows)
    team_df = pd.DataFrame(team_rows)
    row_df["abs_minutes_err"] = (pd.to_numeric(row_df["pred_minutes"], errors="coerce") - pd.to_numeric(row_df["actual_minutes"], errors="coerce")).abs()
    row_df["minutes_err"] = pd.to_numeric(row_df["pred_minutes"], errors="coerce") - pd.to_numeric(row_df["actual_minutes"], errors="coerce")
    high28 = pd.to_numeric(row_df["actual_minutes"], errors="coerce").ge(28.0)
    high32 = pd.to_numeric(row_df["actual_minutes"], errors="coerce").ge(32.0)
    active_stats = _active_metrics(
        pred_active=row_df["pred_active"].to_numpy(dtype=bool, copy=False),
        target_active=row_df["target_active"].to_numpy(dtype=bool, copy=False),
        valid_mask=np.ones(len(row_df), dtype=bool),
    )
    summary = {
        "rows": int(len(row_df)),
        "games": int(row_df.loc[:, ["game_date", "game_id"]].drop_duplicates().shape[0]),
        "minutes_mae": float(row_df["abs_minutes_err"].mean()),
        "minutes_bias": float(row_df["minutes_err"].mean()),
        "minutes_mae_lineup1": float(row_df.loc[row_df["lineup_available"] == 1, "abs_minutes_err"].mean()),
        "minutes_mae_lineup0": float(row_df.loc[row_df["lineup_available"] == 0, "abs_minutes_err"].mean()),
        "minutes_mae_actual_ge28": float(row_df.loc[high28, "abs_minutes_err"].mean()),
        "minutes_mae_actual_ge32": float(row_df.loc[high32, "abs_minutes_err"].mean()),
        "mean_pred_minutes_actual_ge28": float(row_df.loc[high28, "pred_minutes"].mean()),
        "mean_actual_minutes_actual_ge28": float(row_df.loc[high28, "actual_minutes"].mean()),
        "mean_pred_minutes_actual_ge32": float(row_df.loc[high32, "pred_minutes"].mean()),
        "mean_actual_minutes_actual_ge32": float(row_df.loc[high32, "actual_minutes"].mean()),
        "active_count_mae": float((team_df["pred_count"] - team_df["target_count"]).abs().mean()),
        **active_stats,
    }
    return row_df, summary


def _compute_forced_active_diagnostics(
    *,
    old_bundle: Path,
    candidate_bundle: Path,
    dataset_dir: Path,
    device: torch.device,
    out_dir: Path,
) -> None:
    summary_rows: list[dict[str, Any]] = []
    recovery_frames: list[pd.DataFrame] = []
    for val_days in (14, 60):
        for bundle_name, bundle_dir in (("old_prod", old_bundle), ("candidate", candidate_bundle)):
            base_df, base_summary = _run_forced_active_ablation(
                bundle_dir=bundle_dir,
                dataset_dir=dataset_dir,
                device=device,
                val_days=val_days,
                mode="normal",
            )
            forced_df, forced_summary = _run_forced_active_ablation(
                bundle_dir=bundle_dir,
                dataset_dir=dataset_dir,
                device=device,
                val_days=val_days,
                mode="minutes_forced",
            )
            full_df, full_summary = _run_forced_active_ablation(
                bundle_dir=bundle_dir,
                dataset_dir=dataset_dir,
                device=device,
                val_days=val_days,
                mode="full_forced",
            )
            for mode_name, summary in (
                ("normal", base_summary),
                ("minutes_forced", forced_summary),
                ("full_forced", full_summary),
            ):
                row = {"bundle": bundle_name, "val_days": int(val_days), "mode": mode_name}
                row.update(summary)
                summary_rows.append(row)

            merged = base_df.merge(
                forced_df[
                    ["game_date", "game_id", "team_id", "player_id", "pred_minutes"]
                ].rename(columns={"pred_minutes": "pred_minutes_forced"}),
                on=["game_date", "game_id", "team_id", "player_id"],
                how="inner",
                validate="one_to_one",
            )
            merged["minutes_recovery"] = merged["pred_minutes_forced"] - merged["pred_minutes"]
            merged["actual_ge_28"] = pd.to_numeric(merged["actual_minutes"], errors="coerce").ge(28.0)
            recovered = (
                merged.loc[merged["actual_ge_28"]]
                .groupby(["team_id", "player_id"], sort=False, observed=True)[
                    ["actual_minutes", "pred_minutes", "pred_minutes_forced", "minutes_recovery"]
                ]
                .mean()
                .reset_index()
                .sort_values(["minutes_recovery"], ascending=False, kind="stable")
                .head(30)
            )
            recovered.insert(0, "bundle", bundle_name)
            recovered.insert(1, "val_days", int(val_days))
            recovery_frames.append(recovered)

    pd.DataFrame(summary_rows).to_csv(out_dir / "forced_active_summary.csv", index=False)
    if recovery_frames:
        pd.concat(recovery_frames, ignore_index=True).to_csv(out_dir / "forced_active_recovery_top30.csv", index=False)


def _compare_tensorized_player_selection(
    *,
    old_bundle: Path,
    candidate_bundle: Path,
    features_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    results: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    for label, bundle_dir in (("old_prod", old_bundle), ("candidate", candidate_bundle)):
        config, _ = load_gtv2_model(bundle_dir, device=torch.device("cpu"))
        examples = build_gtv2_inference_examples(
            features_df=features_df,
            game_date=str(features_df["game_date"].iloc[0]),
            config=config,
        )
        for ex in examples:
            for side_idx, team_id in enumerate(ex.team_ids.tolist()):
                valid_mask = ex.player_valid_mask[side_idx]
                player_ids = ex.player_ids[side_idx][valid_mask]
                selected = pd.DataFrame(
                    {
                        "bundle": label,
                        "game_date": ex.game_date,
                        "game_id": int(ex.game_id_norm),
                        "team_id": int(team_id),
                        "slot_idx": np.arange(player_ids.shape[0], dtype=int),
                        "player_id": player_ids.astype(int),
                    }
                )
                selected_frames.append(selected)
    all_selected = pd.concat(selected_frames, ignore_index=True)
    old_sel = all_selected.loc[all_selected["bundle"] == "old_prod"].drop(columns=["bundle"])
    new_sel = all_selected.loc[all_selected["bundle"] == "candidate"].drop(columns=["bundle"])
    compare = old_sel.merge(
        new_sel,
        on=["game_date", "game_id", "team_id", "slot_idx"],
        how="outer",
        suffixes=("_old", "_candidate"),
        validate="one_to_one",
    )
    compare["same_player"] = compare["player_id_old"].fillna(-1).astype(int).eq(compare["player_id_candidate"].fillna(-1).astype(int))
    team_compare = (
        compare.groupby(["game_date", "game_id", "team_id"], sort=False, observed=True)["same_player"]
        .agg(["all", "mean"])
        .reset_index()
        .rename(columns={"all": "all_slots_match", "mean": "slot_match_rate"})
    )
    results.append(
        {
            "teams_compared": int(team_compare.shape[0]),
            "all_teams_exact_match": bool(team_compare["all_slots_match"].all()),
            "mean_slot_match_rate": float(team_compare["slot_match_rate"].mean()),
        }
    )
    compare.to_csv(out_dir / "tensor_player_selection_compare.csv", index=False)
    team_compare.to_csv(out_dir / "tensor_player_selection_team_summary.csv", index=False)
    (out_dir / "tensor_player_selection_summary.json").write_text(json.dumps(results[0], indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(args.device)
    features_df = _normalize_features_snapshot(args.snapshot_features)

    sweep_dir = args.out_dir / "same_snapshot_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    variants = _build_world_variants(args.old_bundle, args.candidate_bundle)
    _compute_same_snapshot_sweep(
        variants=variants,
        features_df=features_df,
        device=device,
        out_dir=sweep_dir,
        num_worlds=int(args.num_worlds),
        chunk_size=int(args.chunk_size),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )

    forced_dir = args.out_dir / "forced_active_ablation"
    forced_dir.mkdir(parents=True, exist_ok=True)
    _compute_forced_active_diagnostics(
        old_bundle=args.old_bundle,
        candidate_bundle=args.candidate_bundle,
        dataset_dir=args.dataset_dir,
        device=device,
        out_dir=forced_dir,
    )

    overflow_dir = args.out_dir / "tensor_selection_audit"
    overflow_dir.mkdir(parents=True, exist_ok=True)
    _compare_tensorized_player_selection(
        old_bundle=args.old_bundle,
        candidate_bundle=args.candidate_bundle,
        features_df=features_df,
        out_dir=overflow_dir,
    )


if __name__ == "__main__":
    main()
