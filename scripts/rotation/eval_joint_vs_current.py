#!/usr/bin/env python3
"""Evaluate joint model vs current models on joint val; optionally prepare sim inputs.

Produces:
1) raw-vs-raw metrics on the same validation slice.
2) optional sim-ready minutes/rates parquet runs for both systems, so sim-vs-sim
   can be run through scripts.sim_v2.generate_worlds_fpts_v2 with explicit run IDs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from projections import paths
from projections.pipeline.effective_inputs import (
    EFFECTIVE_MINUTES_FILENAME,
    EFFECTIVE_RATES_FILENAME,
)
from projections.rates_v1.loader import load_rates_bundle
from projections.rates_v1.score import predict_rates
from projections.rotation.joint_set_model_v1 import JointRotationRatesModelConfig, build_joint_model
from projections.rotation.set_model import RotationSetMinutesPredictor, zfill_game_id_series
from scripts.rotation.train_joint_rotation_rates_model_v1 import (
    EFFICIENCY_TARGETS,
    RATE_TARGETS,
    _add_player_embedding_indices,
    _add_player_team_hash_indices,
    _add_team_embedding_indices,
    _build_player_id_vocab,
    _build_team_id_vocab,
    _coerce_join_keys,
    _numeric_frame,
)

JOIN_KEYS = ["game_id", "team_id", "player_id", "game_date"]


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _mae(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, int]:
    mask = y_true.notna() & y_pred.notna()
    if not mask.any():
        return float("nan"), 0
    err = (y_true.loc[mask] - y_pred.loc[mask]).abs()
    return float(err.mean()), int(mask.sum())


def _predict_joint(
    merged: pd.DataFrame,
    *,
    run_dir: Path,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    config = JointRotationRatesModelConfig.load(run_dir / "config.json")
    model = build_joint_model(config).to(torch.device(device))
    state = torch.load(run_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    df = merged.copy()
    team_vocab: list[int] | None = None
    if bool(config.use_team_embeddings):
        team_vocab = list(config.team_id_vocab or _build_team_id_vocab(df))
        df = _add_team_embedding_indices(df, team_id_vocab=team_vocab)
    else:
        df["team_id_idx"] = 0
        df["opp_id_idx"] = 0
    player_vocab: list[int] | None = None
    if bool(config.use_player_embeddings):
        player_vocab = list(config.player_id_vocab or _build_player_id_vocab(df))
        df = _add_player_embedding_indices(df, player_id_vocab=player_vocab)
    else:
        df["player_id_idx"] = 0
    if bool(config.use_player_team_embeddings):
        df = _add_player_team_hash_indices(df, buckets=int(config.player_team_hash_buckets))
    else:
        df["player_team_hash_idx"] = 0

    feature_cols = list(config.feature_columns)
    feature_mean = np.asarray(config.feature_mean, dtype=np.float32)
    feature_std = np.asarray(config.feature_std, dtype=np.float32)
    feature_std = np.where(feature_std <= 1e-6, 1.0, feature_std)

    out_minutes = np.full(len(df), np.nan, dtype=np.float32)
    out_rates = np.full((len(df), len(RATE_TARGETS)), np.nan, dtype=np.float32)
    out_eff = np.full((len(df), len(EFFICIENCY_TARGETS)), np.nan, dtype=np.float32)
    feats = _numeric_frame(df, feature_cols).to_numpy(dtype="float32", copy=False)
    feats = np.nan_to_num(feats, nan=feature_mean[None, :], posinf=feature_mean[None, :], neginf=feature_mean[None, :])
    feats = (feats - feature_mean[None, :]) / feature_std[None, :]
    team_idx_all = pd.to_numeric(df.get("team_id_idx", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    opp_idx_all = pd.to_numeric(df.get("opp_id_idx", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    player_idx_all = pd.to_numeric(df.get("player_id_idx", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    player_team_idx_all = pd.to_numeric(df.get("player_team_hash_idx", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    alloc_mask_all = ~(pd.to_numeric(df.get("is_out", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int64) > 0)
    prior_w_all = (
        pd.to_numeric(df.get(str(config.prior_weight_col), 0.0), errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float32)
    )
    prior20_all = (
        pd.to_numeric(df.get("minutes_from_stints_prior_20", 0.0), errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float32)
    )
    groups = list(df.groupby(["game_id_norm", "team_id"], sort=False).indices.values())
    with torch.no_grad():
        for start in range(0, len(groups), max(1, int(batch_size))):
            batch_groups = groups[start : start + int(batch_size)]
            max_n = max(len(g) for g in batch_groups)
            bsz = len(batch_groups)
            x = torch.zeros((bsz, max_n, len(feature_cols)), dtype=torch.float32, device=device)
            team_idx = torch.zeros((bsz, max_n), dtype=torch.long, device=device)
            opp_idx = torch.zeros((bsz, max_n), dtype=torch.long, device=device)
            player_idx = torch.zeros((bsz, max_n), dtype=torch.long, device=device)
            player_team_idx = torch.zeros((bsz, max_n), dtype=torch.long, device=device)
            alloc_mask = torch.zeros((bsz, max_n), dtype=torch.bool, device=device)
            prior_w = torch.zeros((bsz, max_n), dtype=torch.float32, device=device)
            router_features = torch.zeros((bsz, 3), dtype=torch.float32, device=device)
            mask = torch.zeros((bsz, max_n), dtype=torch.bool, device=device)

            for i, idx in enumerate(batch_groups):
                idx_arr = np.asarray(idx, dtype=np.int64)
                n = len(idx_arr)
                x[i, :n] = torch.from_numpy(feats[idx_arr]).to(device=device)
                team_idx[i, :n] = torch.from_numpy(team_idx_all[idx_arr]).to(device=device)
                opp_idx[i, :n] = torch.from_numpy(opp_idx_all[idx_arr]).to(device=device)
                player_idx[i, :n] = torch.from_numpy(player_idx_all[idx_arr]).to(device=device)
                player_team_idx[i, :n] = torch.from_numpy(player_team_idx_all[idx_arr]).to(device=device)
                alloc_mask[i, :n] = torch.from_numpy(alloc_mask_all[idx_arr]).to(device=device)
                prior_w[i, :n] = torch.from_numpy(prior_w_all[idx_arr]).to(device=device)
                mask[i, :n] = True

                alloc_np = alloc_mask_all[idx_arr]
                prior20 = prior20_all[idx_arr].astype(np.float64, copy=False)
                out_mask = ~alloc_np
                vacated = float(np.sum(prior20[out_mask]))
                k = int(min(5, prior20.shape[0]))
                starters_out_proxy = int(np.sum(out_mask[np.argsort(prior20)[::-1][:k]].astype(np.int64))) if k > 0 else 0
                team_out_count = int(np.sum(out_mask.astype(np.int64)))
                router_features[i] = torch.tensor(
                    [
                        float(np.clip(vacated / 60.0, 0.0, 1.0)),
                        float(np.clip(team_out_count / 10.0, 0.0, 1.0)),
                        float(np.clip(starters_out_proxy / 5.0, 0.0, 1.0)),
                    ],
                    dtype=torch.float32,
                    device=device,
                )

            pred_minutes, _gate_logits, _share_logits, pred_rates, pred_eff, _r1, _r2 = model(
                x,
                team_idx,
                opp_idx,
                mask,
                player_idx=player_idx,
                player_team_idx=player_team_idx,
                alloc_mask=alloc_mask,
                prior_weights=prior_w,
                router_features=router_features,
            )
            pm = pred_minutes.cpu().numpy()
            pr = pred_rates.cpu().numpy()
            pe = pred_eff.cpu().numpy()
            m = mask.cpu().numpy()
            for i, idx in enumerate(batch_groups):
                idx_arr = np.asarray(idx, dtype=np.int64)
                n = len(idx_arr)
                valid = m[i, :n].astype(bool)
                out_minutes[idx_arr[valid]] = pm[i, :n][valid]
                out_rates[idx_arr[valid], :] = pr[i, :n, :][valid]
                out_eff[idx_arr[valid], :] = pe[i, :n, :][valid]

    pred = merged.loc[:, JOIN_KEYS].copy()
    pred["minutes_pred_joint"] = out_minutes
    for j, col in enumerate(RATE_TARGETS):
        pred[f"{col}_pred_joint"] = out_rates[:, j]
    for j, col in enumerate(EFFICIENCY_TARGETS):
        pred[f"{col}_pred_joint"] = out_eff[:, j]
    return pred


def _prepare_sim_minutes_frame(df: pd.DataFrame, minutes_col: str) -> pd.DataFrame:
    out = df.loc[:, JOIN_KEYS].copy()
    out["minutes_final"] = pd.to_numeric(df[minutes_col], errors="coerce")
    out["minutes_p50"] = out["minutes_final"]
    out["minutes_pred_p50"] = out["minutes_final"]
    out["play_prob"] = pd.to_numeric(df.get("play_prob", df.get("prior_play_prob", 1.0)), errors="coerce").fillna(1.0)
    out["is_starter"] = pd.to_numeric(
        df.get("is_starter", df.get("is_projected_starter", df.get("is_confirmed_starter", 0))),
        errors="coerce",
    ).fillna(0.0)
    out["is_out"] = pd.to_numeric(df.get("is_out", 0), errors="coerce").fillna(0).astype(int)
    out["prior_play_prob"] = pd.to_numeric(df.get("prior_play_prob", out["play_prob"]), errors="coerce").fillna(1.0)
    if "status" in df.columns:
        out["status"] = df["status"].astype(str)
    if "spread_home" in df.columns:
        out["spread_home"] = pd.to_numeric(df["spread_home"], errors="coerce")
    if "total" in df.columns:
        out["total"] = pd.to_numeric(df["total"], errors="coerce")
    return out


def _prepare_sim_rates_frame(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.loc[:, JOIN_KEYS].copy()
    for col in RATE_TARGETS:
        out[col] = (
            pd.to_numeric(df[f"{col}_{prefix}"], errors="coerce")
            .fillna(pd.to_numeric(df.get(f"{col}_pred_joint"), errors="coerce"))
            .fillna(0.0)
        )
    out["fg2_pct"] = (
        pd.to_numeric(df[f"fg2_pct_label_{prefix}"], errors="coerce")
        .fillna(pd.to_numeric(df.get("fg2_pct_label_pred_joint"), errors="coerce"))
        .fillna(0.52)
    )
    out["fg3_pct"] = (
        pd.to_numeric(df[f"fg3_pct_label_{prefix}"], errors="coerce")
        .fillna(pd.to_numeric(df.get("fg3_pct_label_pred_joint"), errors="coerce"))
        .fillna(0.36)
    )
    out["ft_pct"] = (
        pd.to_numeric(df[f"ft_pct_label_{prefix}"], errors="coerce")
        .fillna(pd.to_numeric(df.get("ft_pct_label_pred_joint"), errors="coerce"))
        .fillna(0.78)
    )
    return out


def _load_current_rates_live_for_dates(
    *,
    data_root: Path,
    run_id: str,
    dates: list[pd.Timestamp],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in dates:
        date_token = pd.Timestamp(d).date().isoformat()
        p_run = data_root / "gold" / "rates_v1_live" / date_token / f"run={run_id}" / "rates.parquet"
        p_flat = data_root / "gold" / "rates_v1_live" / date_token / "rates.parquet"
        path = p_run if p_run.exists() else p_flat
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "game_date" not in df.columns:
            df["game_date"] = pd.to_datetime(date_token)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=JOIN_KEYS)
    out = pd.concat(frames, ignore_index=True)
    out = _coerce_join_keys(out, name="current_rates_live", require_game_date=True)
    out = out.drop_duplicates(subset=JOIN_KEYS, keep="last")
    return out


def _write_sim_input_runs(
    df: pd.DataFrame,
    *,
    data_root: Path,
    minutes_col_joint: str,
    minutes_col_current: str,
    joint_run_id: str,
    current_run_id: str,
) -> dict[str, Any]:
    minutes_joint = _prepare_sim_minutes_frame(df, minutes_col_joint)
    minutes_current = _prepare_sim_minutes_frame(df, minutes_col_current)
    rates_joint = _prepare_sim_rates_frame(df, "pred_joint")
    rates_current = _prepare_sim_rates_frame(df, "pred_current")

    dates = sorted(pd.to_datetime(df["game_date"]).dt.date.unique().tolist())
    for day in dates:
        date_token = pd.Timestamp(day).date().isoformat()
        dj = minutes_joint[pd.to_datetime(minutes_joint["game_date"]).dt.date == day].copy()
        dc = minutes_current[pd.to_datetime(minutes_current["game_date"]).dt.date == day].copy()
        rj = rates_joint[pd.to_datetime(rates_joint["game_date"]).dt.date == day].copy()
        rc = rates_current[pd.to_datetime(rates_current["game_date"]).dt.date == day].copy()

        j_min_dir = data_root / "artifacts" / "minutes_v1" / "daily" / date_token / f"run={joint_run_id}"
        c_min_dir = data_root / "artifacts" / "minutes_v1" / "daily" / date_token / f"run={current_run_id}"
        j_rate_dir = data_root / "gold" / "rates_v1_live" / date_token / f"run={joint_run_id}"
        c_rate_dir = data_root / "gold" / "rates_v1_live" / date_token / f"run={current_run_id}"
        for d in (j_min_dir, c_min_dir, j_rate_dir, c_rate_dir):
            d.mkdir(parents=True, exist_ok=True)

        dj.to_parquet(j_min_dir / "minutes.parquet", index=False)
        dj.to_parquet(j_min_dir / EFFECTIVE_MINUTES_FILENAME, index=False)
        dc.to_parquet(c_min_dir / "minutes.parquet", index=False)
        dc.to_parquet(c_min_dir / EFFECTIVE_MINUTES_FILENAME, index=False)

        rj.to_parquet(j_rate_dir / "rates.parquet", index=False)
        rj.to_parquet(j_rate_dir / EFFECTIVE_RATES_FILENAME, index=False)
        rc.to_parquet(c_rate_dir / "rates.parquet", index=False)
        rc.to_parquet(c_rate_dir / EFFECTIVE_RATES_FILENAME, index=False)

    return {
        "dates": [str(d) for d in dates],
        "joint_minutes_run_id": joint_run_id,
        "joint_rates_run_id": joint_run_id,
        "current_minutes_run_id": current_run_id,
        "current_rates_run_id": current_run_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--joint-run-dir", type=str, required=True, help="Path to trained joint run dir.")
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(paths.get_project_root() / "reports" / "joint_rotation_rates_v1" / f"compare_joint_vs_current_{_utc_now_compact()}.json"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prepare-sim-inputs", action="store_true")
    parser.add_argument("--joint-sim-run-id", type=str, default=f"joint_eval_{_utc_now_compact()}")
    parser.add_argument("--current-sim-run-id", type=str, default=f"current_eval_{_utc_now_compact()}")
    parser.add_argument("--sim-profile", type=str, default="sim_v3")
    parser.add_argument("--sim-worlds", type=int, default=2000)
    args = parser.parse_args()

    data_root = paths.get_data_root()
    project_root = paths.get_project_root()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    joint_run_dir = Path(args.joint_run_dir).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()

    features = _coerce_join_keys(pd.read_parquet(dataset_dir / "features.parquet"), name="features", require_game_date=True)
    labels_minutes = _coerce_join_keys(pd.read_parquet(dataset_dir / "labels_minutes.parquet"), name="labels_minutes", require_game_date=True)
    labels_rates = _coerce_join_keys(pd.read_parquet(dataset_dir / "labels_rates.parquet"), name="labels_rates", require_game_date=True)
    minutes_keep = labels_minutes.loc[:, JOIN_KEYS + ["minutes_label"]]
    rate_keep_cols = [c for c in RATE_TARGETS + EFFICIENCY_TARGETS + ["rates_loss_eligible", "minutes_actual"] if c in labels_rates.columns]
    rates_keep = labels_rates.loc[:, JOIN_KEYS + rate_keep_cols]
    merged = features.merge(minutes_keep, on=JOIN_KEYS, how="left", validate="one_to_one")
    merged = merged.merge(rates_keep, on=JOIN_KEYS, how="left", validate="one_to_one")
    merged["game_id_norm"] = zfill_game_id_series(merged["game_id"])

    run_manifest = _load_json(joint_run_dir / "manifest.json")
    split = run_manifest.get("split", {})
    val_min_date = pd.Timestamp(split["val_min_date"]).normalize()
    val_max_date = pd.Timestamp(split["val_max_date"]).normalize()
    val_df = merged[(merged["game_date"] >= val_min_date) & (merged["game_date"] <= val_max_date)].copy()
    val_df = val_df.reset_index(drop=True)

    pred_joint = _predict_joint(val_df, run_dir=joint_run_dir, batch_size=int(args.batch_size), device=str(args.device))
    eval_df = val_df.merge(pred_joint, on=JOIN_KEYS, how="left", validate="one_to_one")

    rot_cfg = _load_json(project_root / "config" / "rotation_set_minutes_live.json")
    model_dir = Path(str(rot_cfg["model_dir"]))
    if not model_dir.is_absolute():
        model_dir = (project_root / model_dir).resolve()
    rot_predictor = RotationSetMinutesPredictor.load(model_dir, device=str(args.device))
    rot_pred_df = rot_predictor.predict(
        val_df,
        batch_size=int(args.batch_size),
        alloc_mask_mode=rot_cfg.get("alloc_mask_mode"),
        alloc_min_eligible=int(rot_cfg.get("alloc_min_eligible", 9)),
        alloc_prior_play_prob_threshold=float(rot_cfg.get("alloc_prior_play_prob_threshold", 0.2)),
        alloc_baseline_minutes_threshold=float(rot_cfg.get("alloc_baseline_minutes_threshold", 4.0)),
    )
    eval_df = eval_df.merge(
        rot_pred_df.loc[:, JOIN_KEYS + ["pred_minutes"]].rename(columns={"pred_minutes": "minutes_pred_current"}),
        on=JOIN_KEYS,
        how="left",
        validate="one_to_one",
    )

    rates_cfg = _load_json(project_root / "config" / "rates_current_run.json")
    rates_run_id = str(rates_cfg["run_id"])
    val_dates = sorted(pd.to_datetime(val_df["game_date"]).dropna().dt.normalize().unique().tolist())
    live_rates = _load_current_rates_live_for_dates(data_root=data_root, run_id=rates_run_id, dates=val_dates)
    if not live_rates.empty:
        eval_df = eval_df.merge(
            live_rates,
            on=JOIN_KEYS,
            how="left",
            validate="one_to_one",
            suffixes=("", "_current_live"),
        )
        for col in RATE_TARGETS:
            src = col if col in eval_df.columns else (f"pred_{col}" if f"pred_{col}" in eval_df.columns else None)
            eval_df[f"{col}_pred_current"] = pd.to_numeric(eval_df[src], errors="coerce") if src else np.nan
        eff_map = {
            "fg2_pct_label": ["fg2_pct", "pred_fg2_pct", "fg2_pct_label"],
            "fg3_pct_label": ["fg3_pct", "pred_fg3_pct", "fg3_pct_label"],
            "ft_pct_label": ["ft_pct", "pred_ft_pct", "ft_pct_label"],
        }
        for tgt, candidates in eff_map.items():
            src = next((c for c in candidates if c in eval_df.columns), None)
            eval_df[f"{tgt}_pred_current"] = pd.to_numeric(eval_df[src], errors="coerce") if src else np.nan

    # Always run a full-model fallback pass for missing current predictions.
    rates_bundle = load_rates_bundle(run_id=rates_run_id, base_artifacts_root=data_root)
    val_for_rates = val_df.copy()
    for col in rates_bundle.feature_cols:
        if col not in val_for_rates.columns:
            val_for_rates[col] = np.nan
    rates_pred_fallback = predict_rates(val_for_rates, rates_bundle).reindex(val_df.index)
    for col in RATE_TARGETS + EFFICIENCY_TARGETS:
        pred_col = f"{col}_pred_current"
        if pred_col not in eval_df.columns:
            eval_df[pred_col] = np.nan
        if col in rates_pred_fallback.columns:
            eval_df[pred_col] = pd.to_numeric(eval_df[pred_col], errors="coerce").fillna(
                pd.to_numeric(rates_pred_fallback[col], errors="coerce")
            )

    minutes_joint_mae, minutes_joint_n = _mae(eval_df["minutes_label"], eval_df["minutes_pred_joint"])
    minutes_current_mae, minutes_current_n = _mae(eval_df["minutes_label"], eval_df["minutes_pred_current"])

    rates_eligible = pd.to_numeric(eval_df.get("rates_loss_eligible", 0), errors="coerce").fillna(0).astype(int) > 0
    matched_current = pd.to_numeric(eval_df[f"{RATE_TARGETS[0]}_pred_current"], errors="coerce").notna().values & rates_eligible.values
    matched_joint = rates_eligible.values
    rates_joint_rows = int(matched_joint.sum())
    rates_current_rows = int(matched_current.sum())

    joint_rate_maes: dict[str, float] = {}
    current_rate_maes: dict[str, float] = {}
    for col in RATE_TARGETS:
        y = pd.to_numeric(eval_df[col], errors="coerce")
        pj = pd.to_numeric(eval_df[f"{col}_pred_joint"], errors="coerce")
        pc = pd.to_numeric(eval_df[f"{col}_pred_current"], errors="coerce")
        joint_rate_maes[col] = float((y[matched_joint] - pj[matched_joint]).abs().mean()) if rates_joint_rows else float("nan")
        current_rate_maes[col] = float((y[matched_current] - pc[matched_current]).abs().mean()) if rates_current_rows else float("nan")

    joint_eff_maes: dict[str, float] = {}
    current_eff_maes: dict[str, float] = {}
    for col in EFFICIENCY_TARGETS:
        y = pd.to_numeric(eval_df[col], errors="coerce")
        pj = pd.to_numeric(eval_df[f"{col}_pred_joint"], errors="coerce")
        pc = pd.to_numeric(eval_df[f"{col}_pred_current"], errors="coerce")
        joint_eff_maes[col] = float((y[matched_joint] - pj[matched_joint]).abs().mean()) if rates_joint_rows else float("nan")
        current_eff_maes[col] = float((y[matched_current] - pc[matched_current]).abs().mean()) if rates_current_rows else float("nan")

    sim_setup: dict[str, Any] | None = None
    if args.prepare_sim_inputs:
        sim_setup = _write_sim_input_runs(
            eval_df,
            data_root=data_root,
            minutes_col_joint="minutes_pred_joint",
            minutes_col_current="minutes_pred_current",
            joint_run_id=str(args.joint_sim_run_id),
            current_run_id=str(args.current_sim_run_id),
        )
        sim_setup["sim_commands"] = {
            "joint": (
                "uv run python -m scripts.sim_v2.generate_worlds_fpts_v2 "
                f"--start-date {split['val_min_date']} --end-date {split['val_max_date']} "
                f"--profile {args.sim_profile} --n-worlds {int(args.sim_worlds)} "
                f"--minutes-run-id {args.joint_sim_run_id} --rates-run-id {args.joint_sim_run_id} "
                f"--run-id sim_joint_eval_{_utc_now_compact()}"
            ),
            "current": (
                "uv run python -m scripts.sim_v2.generate_worlds_fpts_v2 "
                f"--start-date {split['val_min_date']} --end-date {split['val_max_date']} "
                f"--profile {args.sim_profile} --n-worlds {int(args.sim_worlds)} "
                f"--minutes-run-id {args.current_sim_run_id} --rates-run-id {args.current_sim_run_id} "
                f"--run-id sim_current_eval_{_utc_now_compact()}"
            ),
        }

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "joint_run_dir": str(joint_run_dir),
        "val_slice": {
            "val_min_date": str(split["val_min_date"]),
            "val_max_date": str(split["val_max_date"]),
            "rows": int(len(eval_df)),
            "team_games": int(eval_df.groupby(["game_id_norm", "team_id"], sort=False).ngroups),
        },
        "raw_vs_raw": {
            "minutes": {
                "joint_mae": minutes_joint_mae,
                "joint_count": minutes_joint_n,
                "current_mae": minutes_current_mae,
                "current_count": minutes_current_n,
            },
            "rates": {
                "joint_rows_eligible": rates_joint_rows,
                "current_rows_eligible_and_feature_complete": rates_current_rows,
                "joint_mae_by_target": joint_rate_maes,
                "current_mae_by_target": current_rate_maes,
                "joint_mae_9targets": float(np.nanmean(list(joint_rate_maes.values()))),
                "current_mae_9targets": float(np.nanmean(list(current_rate_maes.values()))),
            },
            "efficiency": {
                "joint_mae_by_target": joint_eff_maes,
                "current_mae_by_target": current_eff_maes,
                "joint_mae_3targets": float(np.nanmean(list(joint_eff_maes.values()))),
                "current_mae_3targets": float(np.nanmean(list(current_eff_maes.values()))),
            },
        },
        "sim_setup": sim_setup,
    }
    _save_json(out_json, payload)
    print(f"[joint_eval] wrote report -> {out_json}")
    if sim_setup is not None:
        print(f"[joint_eval] wrote sim input run_ids: joint={sim_setup['joint_minutes_run_id']} current={sim_setup['current_minutes_run_id']}")


if __name__ == "__main__":
    main()
