#!/usr/bin/env python3
"""Run a 60-day rebound baseline on a GTv2 run and write broad-slice artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from projections import paths
from projections.rotation.game_transformer_v2 import (
    GameLevelDataset,
    GameTransformerV2Config,
    build_game_level_examples,
    build_game_transformer_v2,
    collate_game_level_examples,
)
from projections.rotation.sample_worlds_v2 import (
    JOIN_KEYS,
    MakeModelConfig,
    _coerce_join_keys,
    _resolve_run_dir,
    _split_val,
    sample_worlds_for_batch,
    summarize_worlds_to_projections,
)
from projections.rotation.set_model import zfill_game_id_series


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_compact() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _resolve_device(value: str) -> torch.device:
    requested = str(value).strip().lower()
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_run_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_dataset_dir(value: str | None, *, run_dir: Path) -> Path:
    if value:
        p = Path(value).expanduser()
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Dataset directory not found: {p}")

    summary = _load_run_summary(run_dir)
    dataset_dir = (
        summary.get("args", {}).get("dataset_dir")
        if isinstance(summary, dict)
        else None
    )
    if dataset_dir:
        p = Path(str(dataset_dir)).expanduser()
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        "--dataset-dir was not provided and run summary.json does not expose a valid args.dataset_dir",
    )


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()
    for col in ["game_id", "team_id", "player_id"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out


def _load_eval_frames(
    dataset_dir: Path,
    *,
    val_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features_df = _coerce_join_keys(pd.read_parquet(dataset_dir / "features.parquet"), name="features")
    labels_minutes_df = _coerce_join_keys(pd.read_parquet(dataset_dir / "labels_minutes.parquet"), name="labels_minutes")
    labels_counts_df = _coerce_join_keys(pd.read_parquet(dataset_dir / "labels_boxscore_counts.parquet"), name="labels_boxscore_counts")

    label_overlap = [c for c in labels_minutes_df.columns if c in features_df.columns and c not in JOIN_KEYS]
    labels_for_merge = labels_minutes_df.drop(columns=label_overlap)
    merged = features_df.merge(labels_for_merge, on=JOIN_KEYS, how="left", validate="one_to_one")
    merged["game_id_norm"] = zfill_game_id_series(merged["game_id"])
    val_df = _split_val(merged, val_days=int(val_days))
    if val_df.empty:
        raise ValueError(f"No validation rows found for val_days={val_days}")

    selected_game_keys = (
        val_df.loc[:, ["game_date", "game_id"]]
        .drop_duplicates()
        .sort_values(["game_date", "game_id"], kind="stable")
        .reset_index(drop=True)
    )
    selected_features_df = features_df.merge(selected_game_keys, on=["game_date", "game_id"], how="inner")
    selected_labels_minutes_df = labels_minutes_df.merge(selected_game_keys, on=["game_date", "game_id"], how="inner")
    selected_labels_counts_df = labels_counts_df.merge(selected_game_keys, on=["game_date", "game_id"], how="inner")
    selected_val_df = val_df.merge(selected_game_keys, on=["game_date", "game_id"], how="inner")
    return selected_val_df, selected_features_df, selected_labels_minutes_df, selected_labels_counts_df


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    df = pd.DataFrame(
        {
            "left": pd.to_numeric(left, errors="coerce"),
            "right": pd.to_numeric(right, errors="coerce"),
        }
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 2:
        return float("nan")
    if float(df["left"].std(ddof=0)) <= 1e-12 or float(df["right"].std(ddof=0)) <= 1e-12:
        return float("nan")
    return float(df["left"].corr(df["right"]))


def _safe_rate(num: pd.Series, den: pd.Series, *, max_rate: float = 1.0) -> pd.Series:
    num_s = pd.to_numeric(num, errors="coerce")
    den_s = pd.to_numeric(den, errors="coerce")
    rate = num_s / den_s.replace(0.0, np.nan)
    rate = rate.replace([np.inf, -np.inf], np.nan)
    return rate.clip(lower=0.0, upper=max_rate)


def _metric_block(df: pd.DataFrame, *, pred_col: str, actual_col: str) -> dict[str, float]:
    pred = pd.to_numeric(df[pred_col], errors="coerce")
    actual = pd.to_numeric(df[actual_col], errors="coerce")
    err = pred - actual
    return {
        "n": int(len(df)),
        "bias": float(err.mean()),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "corr": _safe_corr(pred, actual),
        "actual_mean": float(actual.mean()),
        "pred_mean": float(pred.mean()),
    }


def _ordered_pair_corr_pts_to_teammate_ast(worlds: pd.DataFrame) -> tuple[float, int]:
    pair_corrs: list[float] = []
    for _, grp in worlds.groupby(["game_date", "game_id", "team_id"], sort=False):
        pts_pivot = grp.pivot(index="world_idx", columns="player_id", values="pts").sort_index(axis=0).sort_index(axis=1)
        ast_pivot = grp.pivot(index="world_idx", columns="player_id", values="ast").sort_index(axis=0).sort_index(axis=1)
        common = [col for col in pts_pivot.columns if col in ast_pivot.columns]
        if len(common) < 2:
            continue
        pts_mat = pts_pivot[common].to_numpy(dtype=np.float64, copy=False)
        ast_mat = ast_pivot[common].to_numpy(dtype=np.float64, copy=False)
        for i in range(len(common)):
            pts_i = pts_mat[:, i]
            if float(np.std(pts_i)) <= 1e-12:
                continue
            for j in range(len(common)):
                if i == j:
                    continue
                ast_j = ast_mat[:, j]
                if float(np.std(ast_j)) <= 1e-12:
                    continue
                v = np.corrcoef(pts_i, ast_j)[0, 1]
                if np.isfinite(v):
                    pair_corrs.append(float(v))
    if not pair_corrs:
        return float("nan"), 0
    arr = np.asarray(pair_corrs, dtype=np.float64)
    return float(arr.mean()), int(arr.size)


def _build_structure_report(
    *,
    projections_df: pd.DataFrame,
    worlds_df: pd.DataFrame,
    labels_counts_df: pd.DataFrame,
) -> dict[str, Any]:
    proj = projections_df.loc[:, ["game_date", "game_id", "team_id", "player_id", "minutes_mean", "pts_mean", "reb_mean", "ast_mean"]].copy()
    proj = _normalize_keys(proj)

    labels = labels_counts_df.loc[
        :,
        [
            "game_date",
            "game_id",
            "team_id",
            "player_id",
            "fga2",
            "fg2m",
            "fga3",
            "fg3m",
            "fta",
            "ftm",
            "oreb",
            "dreb",
            "ast",
            "minutes",
            "played",
        ],
    ].copy()
    labels = _normalize_keys(labels)
    for col in ["fga2", "fg2m", "fga3", "fg3m", "fta", "ftm", "oreb", "dreb", "ast", "minutes", "played"]:
        labels[col] = pd.to_numeric(labels[col], errors="coerce").fillna(0.0)
    labels["reb_actual"] = labels["oreb"] + labels["dreb"]
    labels["pts_actual"] = 2.0 * labels["fg2m"] + 3.0 * labels["fg3m"] + labels["ftm"]
    labels["fgm_actual"] = labels["fg2m"] + labels["fg3m"]
    labels["missed_fg_actual"] = (labels["fga2"] + labels["fga3"]) - labels["fgm_actual"]

    merged = proj.merge(labels, on=["game_date", "game_id", "team_id", "player_id"], how="inner", validate="one_to_one")
    team_actual = labels.groupby(["game_date", "game_id", "team_id"], as_index=False).agg(
        reb_actual=("reb_actual", "sum"),
        ast_actual=("ast", "sum"),
        pts_actual=("pts_actual", "sum"),
        fgm_actual=("fgm_actual", "sum"),
        oreb_actual=("oreb", "sum"),
        dreb_actual=("dreb", "sum"),
        missed_fg_actual=("missed_fg_actual", "sum"),
    )
    opp = team_actual[["game_date", "game_id", "team_id", "missed_fg_actual"]].rename(
        columns={"team_id": "opp_team_id", "missed_fg_actual": "opp_missed_fg_actual"}
    )
    team_actual = team_actual.merge(opp, on=["game_date", "game_id"], how="left")
    team_actual = team_actual.loc[team_actual["team_id"] != team_actual["opp_team_id"]].copy()
    team_actual = team_actual.drop(columns=["opp_team_id"])

    team_pred = merged.groupby(["game_date", "game_id", "team_id"], as_index=False).agg(
        reb_pred=("reb_mean", "sum"),
        ast_pred=("ast_mean", "sum"),
        pts_pred=("pts_mean", "sum"),
    )
    team_merged = team_pred.merge(team_actual, on=["game_date", "game_id", "team_id"], how="inner", validate="one_to_one")

    player_blocks = {
        "all_rows": merged,
        "played_ge_1m": merged.loc[merged["minutes"] >= 1.0].copy(),
        "rotation_ge_20m": merged.loc[merged["minutes"] >= 20.0].copy(),
    }
    player_metrics = {
        key: {
            "reb": _metric_block(df, pred_col="reb_mean", actual_col="reb_actual"),
            "ast": _metric_block(df, pred_col="ast_mean", actual_col="ast"),
        }
        for key, df in player_blocks.items()
    }
    team_metrics = {
        "reb": _metric_block(team_merged, pred_col="reb_pred", actual_col="reb_actual"),
        "ast": _metric_block(team_merged, pred_col="ast_pred", actual_col="ast_actual"),
    }

    assist_df = merged.merge(
        team_actual[["game_date", "game_id", "team_id", "fgm_actual", "pts_actual"]].rename(
            columns={"fgm_actual": "team_fgm_actual", "pts_actual": "team_pts_actual"}
        ),
        on=["game_date", "game_id", "team_id"],
        how="left",
    ).merge(
        team_pred[["game_date", "game_id", "team_id", "pts_pred"]].rename(columns={"pts_pred": "team_pts_pred"}),
        on=["game_date", "game_id", "team_id"],
        how="left",
    )
    assist_df["teammate_fgm_actual"] = assist_df["team_fgm_actual"] - assist_df["fgm_actual"]
    assist_df["teammate_pts_actual"] = assist_df["team_pts_actual"] - assist_df["pts_actual"]
    assist_df["teammate_pts_pred"] = assist_df["team_pts_pred"] - assist_df["pts_mean"]
    assist_cross_section = {}
    for key, df in {
        "played_ge_1m": assist_df.loc[assist_df["minutes"] >= 1.0].copy(),
        "rotation_ge_20m": assist_df.loc[assist_df["minutes"] >= 20.0].copy(),
    }.items():
        assist_cross_section[key] = {
            "n": int(len(df)),
            "actual_corr_ast_vs_teammate_fgm": _safe_corr(df["ast"], df["teammate_fgm_actual"]),
            "actual_corr_ast_vs_teammate_pts": _safe_corr(df["ast"], df["teammate_pts_actual"]),
            "actual_corr_ast_vs_team_pts": _safe_corr(df["ast"], df["team_pts_actual"]),
            "pred_mean_corr_ast_vs_teammate_pts": _safe_corr(df["ast_mean"], df["teammate_pts_pred"]),
            "pred_mean_corr_ast_vs_team_pts": _safe_corr(df["ast_mean"], df["team_pts_pred"]),
        }

    worlds = worlds_df.loc[
        :,
        [
            "world_idx",
            "game_date",
            "game_id",
            "team_id",
            "player_id",
            "active",
            "minutes",
            "pts",
            "reb",
            "ast",
            "oreb",
            "dreb",
            "fga2",
            "fg2m",
            "fga3",
            "fg3m",
        ],
    ].copy()
    worlds = _normalize_keys(worlds)
    worlds["world_idx"] = pd.to_numeric(worlds["world_idx"], errors="coerce").astype("Int64")
    for col in ["active", "minutes", "pts", "reb", "ast", "oreb", "dreb", "fga2", "fg2m", "fga3", "fg3m"]:
        worlds[col] = pd.to_numeric(worlds[col], errors="coerce").fillna(0.0)
    worlds["fgm"] = worlds["fg2m"] + worlds["fg3m"]
    worlds["missed_fg"] = (worlds["fga2"] + worlds["fga3"]) - worlds["fgm"]

    team_world = worlds.groupby(["game_date", "game_id", "team_id", "world_idx"], as_index=False).agg(
        team_pts=("pts", "sum"),
        team_reb=("reb", "sum"),
        team_ast=("ast", "sum"),
        team_fgm=("fgm", "sum"),
        team_oreb=("oreb", "sum"),
        team_dreb=("dreb", "sum"),
        own_missed_fg=("missed_fg", "sum"),
    )
    opp_team_world = team_world[["game_date", "game_id", "team_id", "world_idx", "own_missed_fg"]].rename(
        columns={"team_id": "opp_team_id", "own_missed_fg": "opp_missed_fg"}
    )
    team_world = team_world.merge(opp_team_world, on=["game_date", "game_id", "world_idx"], how="left")
    team_world = team_world.loc[team_world["team_id"] != team_world["opp_team_id"]].copy()
    team_world = team_world.drop(columns=["opp_team_id"])
    team_world["game_total_missed_fg"] = team_world["own_missed_fg"] + team_world["opp_missed_fg"]
    team_world["oreb_capture_rate"] = _safe_rate(team_world["team_oreb"], team_world["own_missed_fg"])
    team_world["dreb_capture_rate"] = _safe_rate(team_world["team_dreb"], team_world["opp_missed_fg"])
    team_world["ast_on_fgm_rate"] = _safe_rate(team_world["team_ast"], team_world["team_fgm"])

    team_actual_rates = team_actual.copy()
    team_actual_rates["oreb_capture_rate_actual"] = _safe_rate(team_actual_rates["oreb_actual"], team_actual_rates["missed_fg_actual"])
    team_actual_rates["dreb_capture_rate_actual"] = _safe_rate(team_actual_rates["dreb_actual"], team_actual_rates["opp_missed_fg_actual"])
    team_actual_rates["ast_on_fgm_rate_actual"] = _safe_rate(team_actual_rates["ast_actual"], team_actual_rates["fgm_actual"])

    rebound_correlation = {
        "pred_world_corr_team_oreb_vs_own_missed_fg": _safe_corr(team_world["team_oreb"], team_world["own_missed_fg"]),
        "pred_world_corr_team_dreb_vs_opp_missed_fg": _safe_corr(team_world["team_dreb"], team_world["opp_missed_fg"]),
        "pred_world_corr_team_reb_vs_game_total_missed_fg": _safe_corr(team_world["team_reb"], team_world["game_total_missed_fg"]),
        "actual_teamgame_corr_oreb_vs_own_missed_fg": _safe_corr(team_actual["oreb_actual"], team_actual["missed_fg_actual"]),
        "actual_teamgame_corr_dreb_vs_opp_missed_fg": _safe_corr(team_actual["dreb_actual"], team_actual["opp_missed_fg_actual"]),
        "actual_teamgame_corr_reb_vs_game_total_missed_fg": _safe_corr(
            team_actual["reb_actual"],
            team_actual["missed_fg_actual"] + team_actual["opp_missed_fg_actual"],
        ),
    }
    rebound_rate_alignment = {
        "pred_world_mean_oreb_capture_rate": float(team_world["oreb_capture_rate"].mean()),
        "actual_teamgame_mean_oreb_capture_rate": float(team_actual_rates["oreb_capture_rate_actual"].mean()),
        "pred_world_mean_dreb_capture_rate": float(team_world["dreb_capture_rate"].mean()),
        "actual_teamgame_mean_dreb_capture_rate": float(team_actual_rates["dreb_capture_rate_actual"].mean()),
    }

    assist_correlation = {
        "pred_world_corr_team_ast_vs_team_fgm": _safe_corr(team_world["team_ast"], team_world["team_fgm"]),
        "pred_world_corr_team_ast_vs_team_pts": _safe_corr(team_world["team_ast"], team_world["team_pts"]),
        "actual_teamgame_corr_team_ast_vs_team_fgm": _safe_corr(team_actual["ast_actual"], team_actual["fgm_actual"]),
        "actual_teamgame_corr_team_ast_vs_team_pts": _safe_corr(team_actual["ast_actual"], team_actual["pts_actual"]),
    }
    assist_rate_alignment = {
        "pred_world_mean_ast_on_fgm_rate": float(team_world["ast_on_fgm_rate"].mean()),
        "actual_teamgame_mean_ast_on_fgm_rate": float(team_actual_rates["ast_on_fgm_rate_actual"].mean()),
    }

    player_world = worlds.merge(
        team_world[["game_date", "game_id", "team_id", "world_idx", "team_pts", "team_fgm"]],
        on=["game_date", "game_id", "team_id", "world_idx"],
        how="left",
    )
    player_world["teammate_pts"] = player_world["team_pts"] - player_world["pts"]
    player_world["teammate_fgm"] = player_world["team_fgm"] - player_world["fgm"]
    for key, df in {
        "active": player_world.loc[player_world["active"] > 0].copy(),
        "rotation20": player_world.loc[player_world["minutes"] >= 20.0].copy(),
    }.items():
        assist_correlation[f"pred_world_corr_ast_vs_teammate_fgm_{key}"] = _safe_corr(df["ast"], df["teammate_fgm"])
        assist_correlation[f"pred_world_corr_ast_vs_teammate_pts_{key}"] = _safe_corr(df["ast"], df["teammate_pts"])
        assist_correlation[f"pred_world_corr_ast_vs_team_fgm_{key}"] = _safe_corr(df["ast"], df["team_fgm"])
        assist_correlation[f"pred_world_corr_ast_vs_team_pts_{key}"] = _safe_corr(df["ast"], df["team_pts"])

    ordered_pair_mean, ordered_pair_n = _ordered_pair_corr_pts_to_teammate_ast(worlds)
    assist_correlation["pred_world_mean_ordered_pair_corr_pts_i_vs_ast_j_teammate"] = ordered_pair_mean
    assist_correlation["n_ordered_pairs"] = ordered_pair_n

    return {
        "created_at": _utc_now().isoformat(),
        "window": {
            "start_date": str(proj["game_date"].min().date()),
            "end_date": str(proj["game_date"].max().date()),
            "n_game_dates": int(proj["game_date"].nunique()),
            "n_projection_rows": int(len(proj)),
            "n_joined_rows": int(len(merged)),
            "n_world_rows": int(len(worlds)),
            "n_team_games": int(len(team_merged)),
        },
        "player_metrics": player_metrics,
        "team_metrics": team_metrics,
        "assist_cross_section": assist_cross_section,
        "rebound_correlation": rebound_correlation,
        "rebound_rate_alignment": rebound_rate_alignment,
        "assist_correlation": assist_correlation,
        "assist_rate_alignment": assist_rate_alignment,
    }


def _slice_summary(
    df: pd.DataFrame,
    *,
    pred_col: str,
    actual_col: str,
    line_col: str,
) -> dict[str, float]:
    pred = pd.to_numeric(df[pred_col], errors="coerce")
    actual = pd.to_numeric(df[actual_col], errors="coerce")
    line = pd.to_numeric(df[line_col], errors="coerce")
    return {
        "rows": int(len(df)),
        "games": int(df[["game_date", "game_id"]].drop_duplicates().shape[0]),
        "players": int(df["player_id"].nunique()),
        "pred_mean": float(pred.mean()),
        "line_mean": float(line.mean()),
        "actual_mean": float(actual.mean()),
        "pred_minus_line_mean": float((pred - line).mean()),
        "pred_minus_actual_mean": float((pred - actual).mean()),
        "line_minus_actual_mean": float((line - actual).mean()),
        "pred_mae_actual": float((pred - actual).abs().mean()),
        "line_mae_actual": float((line - actual).abs().mean()),
        "pred_mae_line": float((pred - line).abs().mean()),
        "pred_corr_actual": _safe_corr(pred, actual),
        "line_corr_actual": _safe_corr(line, actual),
        "over_line_rate": float((pred > line).mean()),
        "over_actual_rate": float((pred > actual).mean()),
    }


def _top_player_gaps(
    df: pd.DataFrame,
    *,
    player_name_col: str,
    pred_col: str,
    actual_col: str,
    line_col: str,
    value_col: str,
    limit: int,
) -> list[dict[str, Any]]:
    if df.empty:
        return []
    work = df.copy()
    work[player_name_col] = work[player_name_col].fillna("unknown")
    grouped = (
        work.groupby(["player_id", player_name_col], as_index=False)
        .agg(
            games=("game_id", "count"),
            pred=(pred_col, "mean"),
            line=(line_col, "mean"),
            actual=(actual_col, "mean"),
            value=(value_col, "mean"),
        )
        .sort_values("value", ascending=False, kind="stable")
        .head(max(0, int(limit)))
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    for row in grouped.itertuples(index=False):
        rows.append(
            {
                "player_id": int(row.player_id),
                "player_name": str(getattr(row, player_name_col)),
                "games": int(row.games),
                "pred_reb": float(row.pred),
                "line": float(row.line),
                "actual": float(row.actual),
                "value": float(row.value),
            }
        )
    return rows


def _build_reb_base_frame(
    *,
    features_df: pd.DataFrame,
    labels_counts_df: pd.DataFrame,
) -> pd.DataFrame:
    feature_cols = ["game_date", "game_id", "team_id", "player_id", "player_name", "an_has_reb", "an_reb_line"]
    feature_cols = [c for c in feature_cols if c in features_df.columns]
    base = features_df.loc[:, feature_cols].copy()
    base = _normalize_keys(base)
    if "player_name" not in base.columns:
        base["player_name"] = "unknown"
    if "an_has_reb" not in base.columns:
        base["an_has_reb"] = 0.0
    if "an_reb_line" not in base.columns:
        base["an_reb_line"] = np.nan

    counts = labels_counts_df.loc[:, ["game_date", "game_id", "team_id", "player_id", "oreb", "dreb"]].copy()
    counts = _normalize_keys(counts)
    counts["actual_reb"] = pd.to_numeric(counts["oreb"], errors="coerce").fillna(0.0) + pd.to_numeric(
        counts["dreb"], errors="coerce"
    ).fillna(0.0)
    counts = counts.drop(columns=["oreb", "dreb"])
    base = base.merge(counts, on=["game_date", "game_id", "team_id", "player_id"], how="inner", validate="one_to_one")
    base["an_has_reb"] = pd.to_numeric(base["an_has_reb"], errors="coerce").fillna(0.0)
    base["an_reb_line"] = pd.to_numeric(base["an_reb_line"], errors="coerce")
    return base


def _build_examples_and_worlds(
    *,
    run_dir: Path,
    selected_val_df: pd.DataFrame,
    num_worlds: int,
    batch_size: int,
    chunk_size: int,
    seed: int,
    device: torch.device,
    active_temperature: float,
    make_model: str,
    allocation_source: str,
    allocation_blend_alpha: float,
    bb_use_learned_efficiency: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    config = GameTransformerV2Config.load(run_dir / "config.json")
    model = build_game_transformer_v2(config)
    setattr(model, "gtv2_config", config)
    state = torch.load(run_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device=device)
    model.eval()

    examples = build_game_level_examples(
        selected_val_df,
        feature_columns=list(config.feature_columns),
        feature_mean=np.asarray(config.feature_mean, dtype=np.float32),
        feature_std=np.asarray(config.feature_std, dtype=np.float32),
        game_feature_columns=list(config.game_feature_columns),
        team_feature_columns=list(config.team_feature_columns),
        flow_label_columns=None,
        minutes_label_col="minutes_label" if "minutes_label" in selected_val_df.columns else "minutes",
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

    make_model_config = MakeModelConfig(
        mode=str(make_model),
        use_learned_efficiency=bool(bb_use_learned_efficiency),
    )
    frames: list[pd.DataFrame] = []
    contract_checks: dict[str, int] = {}
    for batch in loader:
        df_batch, checks = sample_worlds_for_batch(
            model,
            batch,
            device=device,
            num_worlds=int(num_worlds),
            chunk_size=max(1, int(chunk_size)),
            active_temperature=float(active_temperature),
            strict_contracts=True,
            attempt_conditioning_mode="predicted_attempts",
            make_model_config=make_model_config,
            allocation_source=str(allocation_source),
            allocation_blend_alpha=float(allocation_blend_alpha),
        )
        frames.append(df_batch)
        for key, value in checks.items():
            contract_checks[str(key)] = int(contract_checks.get(str(key), 0) + int(value))

    worlds_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    projections_df = summarize_worlds_to_projections(worlds_df, sim_profile="game_transformer_v2")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return projections_df, worlds_df, {
        "num_examples": int(len(examples)),
        "score_rows": int(len(projections_df)),
        "world_rows": int(len(worlds_df)),
        "contract_checks": contract_checks,
        "num_worlds": int(num_worlds),
        "make_model": str(make_model),
        "allocation_source": str(allocation_source),
        "allocation_blend_alpha": float(allocation_blend_alpha),
        "bb_use_learned_efficiency": bool(bb_use_learned_efficiency),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--val-days", type=int, default=60)
    parser.add_argument("--num-worlds", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--active-temperature", type=float, default=1.0)
    parser.add_argument("--make-model", type=str, default="beta_binomial_all")
    parser.add_argument("--allocation-source", type=str, default="emergent")
    parser.add_argument("--allocation-blend-alpha", type=float, default=0.5)
    parser.add_argument("--bb-use-learned-efficiency", type=int, default=1, choices=[0, 1])
    parser.add_argument("--high-reb-threshold", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = _resolve_run_dir(args.run_dir)
    dataset_dir = _resolve_dataset_dir(args.dataset_dir, run_dir=run_dir)
    device = _resolve_device(args.device)

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (paths.get_data_root() / "training" / "runs" / f"reb_60d_eval_{_utc_now_compact()}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_val_df, selected_features_df, _, selected_labels_counts_df = _load_eval_frames(
        dataset_dir,
        val_days=int(args.val_days),
    )

    projections_df, worlds_df, model_meta = _build_examples_and_worlds(
        run_dir=run_dir,
        selected_val_df=selected_val_df,
        num_worlds=int(args.num_worlds),
        batch_size=int(args.batch_size),
        chunk_size=int(args.chunk_size),
        seed=int(args.seed),
        device=device,
        active_temperature=float(args.active_temperature),
        make_model=str(args.make_model),
        allocation_source=str(args.allocation_source),
        allocation_blend_alpha=float(args.allocation_blend_alpha),
        bb_use_learned_efficiency=bool(int(args.bb_use_learned_efficiency)),
    )

    projections_out = out_dir / "projections_60d.parquet"
    worlds_out = out_dir / "worlds_60d.parquet"
    _atomic_write_parquet(projections_df, projections_out)
    _atomic_write_parquet(worlds_df, worlds_out)

    base_df = _build_reb_base_frame(
        features_df=selected_features_df,
        labels_counts_df=selected_labels_counts_df,
    )
    base_df = base_df.loc[base_df["an_has_reb"] > 0.0].copy()
    base_df = base_df.loc[base_df["an_reb_line"].notna()].copy()
    base_df = base_df.sort_values(["game_date", "game_id", "team_id", "player_id"], kind="stable").reset_index(drop=True)
    _atomic_write_parquet(base_df, out_dir / "window_player_eval_base.parquet")

    proj_keys = projections_df.loc[:, ["game_date", "game_id", "team_id", "player_id", "reb_mean"]].copy()
    proj_keys = _normalize_keys(proj_keys)
    proj_keys["pred_reb"] = pd.to_numeric(proj_keys["reb_mean"], errors="coerce")
    player_eval = base_df.merge(
        proj_keys.loc[:, ["game_date", "game_id", "team_id", "player_id", "pred_reb"]],
        on=["game_date", "game_id", "team_id", "player_id"],
        how="inner",
        validate="one_to_one",
    )
    player_eval["line_minus_pred"] = player_eval["an_reb_line"] - player_eval["pred_reb"]
    player_eval["actual_minus_pred"] = player_eval["actual_reb"] - player_eval["pred_reb"]
    player_eval["actual_minus_line"] = player_eval["actual_reb"] - player_eval["an_reb_line"]
    player_eval = player_eval.sort_values(["game_date", "game_id", "team_id", "player_id"], kind="stable").reset_index(drop=True)
    _atomic_write_parquet(player_eval, out_dir / "player_eval.parquet")

    high_threshold = float(args.high_reb_threshold)
    high_slice = player_eval.loc[player_eval["an_reb_line"] >= high_threshold].copy()

    market_baseline = {
        "overall": {
            "rows": int(len(base_df)),
            "games": int(base_df[["game_date", "game_id"]].drop_duplicates().shape[0]),
            "players": int(base_df["player_id"].nunique()),
            "line_mean": float(pd.to_numeric(base_df["an_reb_line"], errors="coerce").mean()),
            "actual_mean": float(pd.to_numeric(base_df["actual_reb"], errors="coerce").mean()),
            "line_minus_actual_mean": float((pd.to_numeric(base_df["an_reb_line"], errors="coerce") - pd.to_numeric(base_df["actual_reb"], errors="coerce")).mean()),
            "line_mae_actual": float((pd.to_numeric(base_df["an_reb_line"], errors="coerce") - pd.to_numeric(base_df["actual_reb"], errors="coerce")).abs().mean()),
            "line_corr_actual": _safe_corr(base_df["an_reb_line"], base_df["actual_reb"]),
        },
        f"reb_line_ge_{int(high_threshold)}": {
            "rows": int(len(high_slice)),
            "line_minus_actual_mean": float((pd.to_numeric(high_slice["an_reb_line"], errors="coerce") - pd.to_numeric(high_slice["actual_reb"], errors="coerce")).mean()) if not high_slice.empty else float("nan"),
            "line_mae_actual": float((pd.to_numeric(high_slice["an_reb_line"], errors="coerce") - pd.to_numeric(high_slice["actual_reb"], errors="coerce")).abs().mean()) if not high_slice.empty else float("nan"),
            "line_corr_actual": _safe_corr(high_slice["an_reb_line"], high_slice["actual_reb"]),
        },
    }

    structure_report = _build_structure_report(
        projections_df=projections_df,
        worlds_df=worlds_df,
        labels_counts_df=selected_labels_counts_df,
    )
    structure_path = out_dir / "reb_structure_report.json"
    _write_json(structure_path, structure_report)

    summary = {
        "created_at": _utc_now().isoformat(),
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "val_window": {
            "start": str(pd.to_datetime(selected_val_df["game_date"]).min().date()),
            "end": str(pd.to_datetime(selected_val_df["game_date"]).max().date()),
            "n_dates": int(pd.to_datetime(selected_val_df["game_date"]).dt.normalize().nunique()),
            "val_rows": int(len(selected_val_df)),
            "prop_rows": int(len(base_df)),
            f"high_reb_rows_ge_{int(high_threshold)}": int(len(high_slice)),
        },
        "generation": model_meta,
        "market_baseline": market_baseline,
        "model": {
            "meta": {
                "run_dir": str(run_dir),
                **model_meta,
            },
            "overall": _slice_summary(
                player_eval,
                pred_col="pred_reb",
                actual_col="actual_reb",
                line_col="an_reb_line",
            ),
            f"reb_line_ge_{int(high_threshold)}": _slice_summary(
                high_slice,
                pred_col="pred_reb",
                actual_col="actual_reb",
                line_col="an_reb_line",
            ) if not high_slice.empty else {
                "rows": 0,
                "games": 0,
                "players": 0,
                "pred_mean": float("nan"),
                "line_mean": float("nan"),
                "actual_mean": float("nan"),
                "pred_minus_line_mean": float("nan"),
                "pred_minus_actual_mean": float("nan"),
                "line_minus_actual_mean": float("nan"),
                "pred_mae_actual": float("nan"),
                "line_mae_actual": float("nan"),
                "pred_mae_line": float("nan"),
                "pred_corr_actual": float("nan"),
                "line_corr_actual": float("nan"),
                "over_line_rate": float("nan"),
                "over_actual_rate": float("nan"),
            },
            f"top_line_minus_pred_ge_{int(high_threshold)}": _top_player_gaps(
                high_slice.assign(value=pd.to_numeric(high_slice["an_reb_line"], errors="coerce") - pd.to_numeric(high_slice["pred_reb"], errors="coerce")),
                player_name_col="player_name",
                pred_col="pred_reb",
                actual_col="actual_reb",
                line_col="an_reb_line",
                value_col="value",
                limit=int(args.top_k),
            ),
            f"top_actual_minus_pred_ge_{int(high_threshold)}": _top_player_gaps(
                high_slice.assign(value=pd.to_numeric(high_slice["actual_reb"], errors="coerce") - pd.to_numeric(high_slice["pred_reb"], errors="coerce")),
                player_name_col="player_name",
                pred_col="pred_reb",
                actual_col="actual_reb",
                line_col="an_reb_line",
                value_col="value",
                limit=int(args.top_k),
            ),
        },
        "structure_report_path": str(structure_path),
        "structure_report_excerpt": {
            "team_reb": structure_report.get("team_metrics", {}).get("reb", {}),
            "player_reb_rotation20": structure_report.get("player_metrics", {}).get("rotation_ge_20m", {}).get("reb", {}),
            "rebound_correlation": structure_report.get("rebound_correlation", {}),
            "rebound_rate_alignment": structure_report.get("rebound_rate_alignment", {}),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
