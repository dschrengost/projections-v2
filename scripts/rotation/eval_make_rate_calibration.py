#!/usr/bin/env python3
"""Evaluate section 15.25 make-rate calibration metrics for GTv2 worlds."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    out = pd.Series(np.zeros(len(num), dtype=float), index=num.index, dtype=float)
    mask = pd.to_numeric(den, errors="coerce").fillna(0.0) > 0
    if mask.any():
        out.loc[mask] = (
            pd.to_numeric(num.loc[mask], errors="coerce").fillna(0.0)
            / pd.to_numeric(den.loc[mask], errors="coerce").fillna(1.0)
        )
    return out


def _load_worlds(path: Path) -> pd.DataFrame:
    cols = [
        "world_idx",
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
        "stl",
        "blk",
        "tov",
        "pts",
        "dk_fpts",
    ]
    df = pd.read_parquet(path, columns=cols)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date.astype(str)
    for col in [
        "game_id",
        "team_id",
        "player_id",
        "world_idx",
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
        "pts",
        "dk_fpts",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["game_id", "team_id", "player_id", "world_idx"]).copy()
    df["game_id"] = df["game_id"].astype(int)
    df["team_id"] = df["team_id"].astype(int)
    df["player_id"] = df["player_id"].astype(int)
    df["world_idx"] = df["world_idx"].astype(int)
    return df


def _load_labels(dataset_dir: Path) -> pd.DataFrame:
    labels = pd.read_parquet(dataset_dir / "labels_boxscore_counts.parquet")
    labels["game_date"] = pd.to_datetime(labels["game_date"], errors="coerce").dt.date.astype(str)
    for col in [
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
        "stl",
        "blk",
        "tov",
    ]:
        labels[col] = pd.to_numeric(labels[col], errors="coerce").fillna(0.0)
    labels["game_id"] = labels["game_id"].astype(int)
    labels["team_id"] = labels["team_id"].astype(int)
    labels["player_id"] = labels["player_id"].astype(int)
    return labels


def _load_features(dataset_dir: Path) -> pd.DataFrame:
    path = dataset_dir / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"features parquet not found: {path}")
    schema_cols = pq.ParquetFile(path).schema.names
    want = [
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
    use = [c for c in want if c in schema_cols]
    df = pd.read_parquet(path, columns=use)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date.astype(str)
    for col in ["game_id", "team_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def _resolve_feature_columns(df: pd.DataFrame) -> dict[str, str | None]:
    cols = set(df.columns)
    home_flag = "home_flag" if "home_flag" in cols else ("is_home" if "is_home" in cols else None)
    vegas_total = "vegas_total" if "vegas_total" in cols else None
    vegas_spread = "vegas_spread" if "vegas_spread" in cols else ("spread_home" if "spread_home" in cols else None)
    est_poss = "estimated_possessions" if "estimated_possessions" in cols else None
    return {
        "home_flag": home_flag,
        "vegas_total": vegas_total,
        "vegas_spread": vegas_spread,
        "estimated_possessions": est_poss,
    }


def _build_team_meta(features: pd.DataFrame) -> pd.DataFrame:
    cols = _resolve_feature_columns(features)
    home_col = cols["home_flag"]
    vegas_total_col = cols["vegas_total"]
    vegas_spread_col = cols["vegas_spread"]
    est_poss_col = cols["estimated_possessions"]

    if home_col is None:
        raise ValueError("features missing home_flag/is_home column")

    agg: dict[str, tuple[str, str]] = {
        "home_flag": (home_col, "max"),
    }
    if vegas_total_col:
        agg["vegas_total"] = (vegas_total_col, "first")
    if vegas_spread_col:
        agg["vegas_spread"] = (vegas_spread_col, "first")
    if est_poss_col:
        agg["estimated_possessions"] = (est_poss_col, "first")

    return (
        features.groupby(["game_date", "game_id", "team_id"], as_index=False)
        .agg(**agg)
        .copy()
    )


def _build_game_meta(team_meta: pd.DataFrame) -> pd.DataFrame:
    home = team_meta.loc[team_meta["home_flag"] == 1].copy()
    away = team_meta.loc[team_meta["home_flag"] == 0].copy()
    keep_cols = [c for c in ["vegas_total", "vegas_spread", "estimated_possessions"] if c in team_meta.columns]

    home = home.rename(columns={"team_id": "home_team_id"})
    away = away.rename(columns={"team_id": "away_team_id"})

    merge_cols = ["game_date", "game_id"]
    home_small = home[merge_cols + ["home_team_id"] + keep_cols]
    away_small = away[merge_cols + ["away_team_id"]]

    out = home_small.merge(away_small, on=merge_cols, how="inner")
    return out


def _actual_team_metrics(labels: pd.DataFrame) -> pd.DataFrame:
    team = (
        labels.groupby(["game_date", "game_id", "team_id"], as_index=False)
        .agg(
            act_fga2=("fga2", "sum"),
            act_fg2m=("fg2m", "sum"),
            act_fga3=("fga3", "sum"),
            act_fg3m=("fg3m", "sum"),
            act_fta=("fta", "sum"),
            act_ftm=("ftm", "sum"),
            act_oreb=("oreb", "sum"),
            act_tov=("tov", "sum"),
        )
        .copy()
    )
    team["act_fga"] = team["act_fga2"] + team["act_fga3"]
    team["act_fgm"] = team["act_fg2m"] + team["act_fg3m"]
    team["act_pts"] = 2.0 * team["act_fg2m"] + 3.0 * team["act_fg3m"] + team["act_ftm"]
    team["act_poss"] = team["act_fga"] - team["act_oreb"] + team["act_tov"] + 0.44 * team["act_fta"]
    team["act_fg_pct"] = _safe_div(team["act_fgm"], team["act_fga"])
    team["act_ft_pct"] = _safe_div(team["act_ftm"], team["act_fta"])
    return team


def _pred_team_metrics(worlds: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    world_team = (
        worlds.groupby(["game_date", "game_id", "team_id", "world_idx"], as_index=False)
        .agg(
            fga2=("fga2", "sum"),
            fg2m=("fg2m", "sum"),
            fga3=("fga3", "sum"),
            fg3m=("fg3m", "sum"),
            fta=("fta", "sum"),
            ftm=("ftm", "sum"),
            oreb=("oreb", "sum"),
            tov=("tov", "sum"),
        )
        .copy()
    )
    world_team["fga"] = world_team["fga2"] + world_team["fga3"]
    world_team["fgm"] = world_team["fg2m"] + world_team["fg3m"]
    world_team["pts"] = 2.0 * world_team["fg2m"] + 3.0 * world_team["fg3m"] + world_team["ftm"]
    world_team["poss"] = world_team["fga"] - world_team["oreb"] + world_team["tov"] + 0.44 * world_team["fta"]

    pred = (
        world_team.groupby(["game_date", "game_id", "team_id"], as_index=False)
        .agg(
            pred_fga=("fga", "mean"),
            pred_fgm=("fgm", "mean"),
            pred_fta=("fta", "mean"),
            pred_ftm=("ftm", "mean"),
            pred_pts=("pts", "mean"),
            pred_poss=("poss", "mean"),
        )
        .copy()
    )
    pred["pred_fg_pct"] = _safe_div(pred["pred_fgm"], pred["pred_fga"])
    pred["pred_ft_pct"] = _safe_div(pred["pred_ftm"], pred["pred_fta"])

    poss_delta = (
        world_team.groupby(["game_date", "game_id", "world_idx"], as_index=False)
        .agg(poss_min=("poss", "min"), poss_max=("poss", "max"), n_teams=("team_id", "nunique"))
        .copy()
    )
    poss_delta = poss_delta.loc[poss_delta["n_teams"] >= 2].copy()
    poss_delta["poss_abs_delta"] = (poss_delta["poss_max"] - poss_delta["poss_min"]).abs()
    poss_diag = {
        "poss_sym_abs_mean": float(poss_delta["poss_abs_delta"].mean()) if len(poss_delta) else float("nan"),
        "poss_sym_abs_p95": float(poss_delta["poss_abs_delta"].quantile(0.95)) if len(poss_delta) else float("nan"),
        "poss_sym_abs_max": float(poss_delta["poss_abs_delta"].max()) if len(poss_delta) else float("nan"),
    }
    return pred, poss_diag


def _actual_player_metrics(labels: pd.DataFrame) -> pd.DataFrame:
    out = labels.copy()
    out["actual_fga"] = out["fga2"] + out["fga3"]
    out["actual_pts"] = 2.0 * out["fg2m"] + 3.0 * out["fg3m"] + out["ftm"]
    out["actual_reb"] = out["oreb"] + out["dreb"]
    base = (
        out["actual_pts"]
        + 1.25 * out["actual_reb"]
        + 1.5 * out["ast"]
        + 2.0 * out["stl"]
        + 2.0 * out["blk"]
        - 0.5 * out["tov"]
    )
    qualifiers = pd.concat(
        [
            (out["actual_pts"] >= 10.0).astype(int),
            (out["actual_reb"] >= 10.0).astype(int),
            (out["ast"] >= 10.0).astype(int),
            (out["stl"] >= 10.0).astype(int),
            (out["blk"] >= 10.0).astype(int),
        ],
        axis=1,
    ).sum(axis=1)
    out["actual_dk_fpts"] = base + np.where(qualifiers == 2, 1.5, 0.0) + np.where(qualifiers >= 3, 3.0, 0.0)
    return out[["game_date", "game_id", "team_id", "player_id", "actual_fga", "actual_pts", "actual_dk_fpts"]]


def _pred_player_metrics(worlds: pd.DataFrame) -> pd.DataFrame:
    x = worlds.copy()
    x["fga"] = pd.to_numeric(x["fga2"], errors="coerce").fillna(0.0) + pd.to_numeric(x["fga3"], errors="coerce").fillna(0.0)
    pred = (
        x.groupby(["game_date", "game_id", "team_id", "player_id"], as_index=False)
        .agg(
            pred_fga_mean=("fga", "mean"),
            pred_pts_mean=("pts", "mean"),
            pred_dk_mean=("dk_fpts", "mean"),
            pred_dk_p90=("dk_fpts", lambda x: float(np.quantile(np.asarray(x, dtype=float), 0.90))),
            pred_dk_p95=("dk_fpts", lambda x: float(np.quantile(np.asarray(x, dtype=float), 0.95))),
        )
        .copy()
    )
    return pred


def _actual_team_concentration(labels: pd.DataFrame) -> pd.DataFrame:
    grp = labels.groupby(["game_date", "game_id", "team_id"], as_index=False)
    rows: list[dict[str, Any]] = []
    for keys, g in grp:
        pts = pd.to_numeric(g["actual_pts"], errors="coerce").fillna(0.0).to_numpy()
        total = float(pts.sum())
        if total <= 0.0:
            share1 = float("nan")
            share2 = float("nan")
        else:
            if pts.size == 0:
                share1 = float("nan")
                share2 = float("nan")
            else:
                top = np.partition(pts, -min(2, pts.size))[-min(2, pts.size):]
                share1 = float(top.max() / total) if pts.size >= 1 else float("nan")
                share2 = float(top.sum() / total) if pts.size >= 1 else float("nan")
        rows.append({
            "game_date": str(keys[0]),
            "game_id": int(keys[1]),
            "team_id": int(keys[2]),
            "act_top1_share_pts": share1,
            "act_top2_share_pts": share2,
        })
    return pd.DataFrame(rows)


def _pred_team_concentration(worlds: pd.DataFrame) -> pd.DataFrame:
    grp = worlds.groupby(["game_date", "game_id", "team_id", "world_idx"], as_index=False)
    rows: list[dict[str, Any]] = []
    for keys, g in grp:
        pts = pd.to_numeric(g["pts"], errors="coerce").fillna(0.0).to_numpy()
        total = float(pts.sum())
        if total <= 0.0:
            share1 = float("nan")
            share2 = float("nan")
        else:
            if pts.size == 0:
                share1 = float("nan")
                share2 = float("nan")
            else:
                top = np.partition(pts, -min(2, pts.size))[-min(2, pts.size):]
                share1 = float(top.max() / total) if pts.size >= 1 else float("nan")
                share2 = float(top.sum() / total) if pts.size >= 1 else float("nan")
        rows.append({
            "game_date": str(keys[0]),
            "game_id": int(keys[1]),
            "team_id": int(keys[2]),
            "world_idx": int(keys[3]),
            "pred_top1_share_pts": share1,
            "pred_top2_share_pts": share2,
        })
    per_world = pd.DataFrame(rows)
    if per_world.empty:
        return per_world
    return (
        per_world.groupby(["game_date", "game_id", "team_id"], as_index=False)
        .agg(
            pred_top1_share_pts=("pred_top1_share_pts", "mean"),
            pred_top2_share_pts=("pred_top2_share_pts", "mean"),
        )
        .copy()
    )


def _span_ratio(pred: pd.Series, ref: pd.Series) -> float:
    pred = pd.to_numeric(pred, errors="coerce").dropna()
    ref = pd.to_numeric(ref, errors="coerce").dropna()
    if pred.empty or ref.empty:
        return float("nan")
    denom = float(ref.max() - ref.min())
    if denom <= 0.0:
        return float("nan")
    return float((pred.max() - pred.min()) / denom)


def _mae_bias(pred: pd.Series, actual: pd.Series) -> tuple[float, float]:
    err = pd.to_numeric(pred, errors="coerce") - pd.to_numeric(actual, errors="coerce")
    err = err.replace([np.inf, -np.inf], np.nan).dropna()
    if err.empty:
        return float("nan"), float("nan")
    return float(err.mean()), float(err.abs().mean())


def _segment_metrics(player_eval: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    if len(player_eval) == 0:
        return {
            "n": 0,
            "pts_bias": float("nan"),
            "pts_mae": float("nan"),
            "fga_bias": float("nan"),
            "fga_mae": float("nan"),
            "fga_share_bias": float("nan"),
            "fga_share_mae": float("nan"),
        }
    sel = player_eval.loc[mask].copy()
    if sel.empty:
        return {
            "n": 0,
            "pts_bias": float("nan"),
            "pts_mae": float("nan"),
            "fga_bias": float("nan"),
            "fga_mae": float("nan"),
            "fga_share_bias": float("nan"),
            "fga_share_mae": float("nan"),
        }
    pts_bias, pts_mae = _mae_bias(sel["pred_pts_mean"], sel["actual_pts"])
    fga_bias, fga_mae = _mae_bias(sel["pred_fga_mean"], sel["actual_fga"])
    fga_share_bias, fga_share_mae = _mae_bias(sel["pred_fga_share"], sel["act_fga_share"])
    return {
        "n": int(len(sel)),
        "pts_bias": pts_bias,
        "pts_mae": pts_mae,
        "fga_bias": fga_bias,
        "fga_mae": fga_mae,
        "fga_share_bias": fga_share_bias,
        "fga_share_mae": fga_share_mae,
    }


def _invariant_checks(worlds: pd.DataFrame, *, tol: float = 1e-6) -> dict[str, Any]:
    fg2_over = pd.to_numeric(worlds["fg2m"], errors="coerce") - pd.to_numeric(worlds["fga2"], errors="coerce")
    fg3_over = pd.to_numeric(worlds["fg3m"], errors="coerce") - pd.to_numeric(worlds["fga3"], errors="coerce")
    ft_over = pd.to_numeric(worlds["ftm"], errors="coerce") - pd.to_numeric(worlds["fta"], errors="coerce")
    return {
        "n_rows": int(len(worlds)),
        "fg2m_gt_fga2_rows": int((fg2_over > float(tol)).sum()),
        "fg3m_gt_fga3_rows": int((fg3_over > float(tol)).sum()),
        "ftm_gt_fta_rows": int((ft_over > float(tol)).sum()),
        "max_fg2m_minus_fga2": float(max(0.0, fg2_over.max())) if len(fg2_over) else 0.0,
        "max_fg3m_minus_fga3": float(max(0.0, fg3_over.max())) if len(fg3_over) else 0.0,
        "max_ftm_minus_fta": float(max(0.0, ft_over.max())) if len(ft_over) else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--worlds-parquet", type=str, required=True)
    parser.add_argument("--name", type=str, default="variant")
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--out-team-csv", type=str, default=None)
    parser.add_argument("--out-player-csv", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    worlds_path = Path(args.worlds_parquet).expanduser().resolve()
    if not worlds_path.exists():
        raise FileNotFoundError(f"worlds parquet not found: {worlds_path}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset dir not found: {dataset_dir}")

    worlds = _load_worlds(worlds_path)
    labels = _load_labels(dataset_dir)
    features = _load_features(dataset_dir)
    if worlds.empty:
        raise ValueError("worlds dataframe is empty")

    keys = worlds[["game_date", "game_id", "team_id", "player_id"]].drop_duplicates()
    labels = labels.merge(keys, on=["game_date", "game_id", "team_id", "player_id"], how="inner")
    if labels.empty:
        raise ValueError("no overlap between worlds and labels")

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

    # Game-level context metadata (vegas/spread/home flag)
    team_meta = _build_team_meta(features)
    game_meta = _build_game_meta(team_meta)

    # Spread/total evaluation
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
        )
        spread_eval = spread_eval.merge(game_meta, on=["game_date", "game_id"], how="left")
        spread_eval["pred_spread"] = spread_eval["pred_home_pts"] - spread_eval["pred_away_pts"]
        spread_eval["pred_total"] = spread_eval["pred_home_pts"] + spread_eval["pred_away_pts"]
        spread_eval["pred_game_poss"] = 0.5 * (
            spread_eval["pred_home_poss"] + spread_eval["pred_away_poss"]
        )

        # Actuals for spread
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

    # Concentration diagnostics
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
            # vegas_spread uses sportsbook line sign convention where home favorites are often negative.
            # Normalize to home-margin convention to match pred_spread = home_pts - away_pts.
            spread_eval["vegas_spread_home_margin"] = -spread_eval["vegas_spread"]
            spread_bias_vs_vegas, spread_mae_vs_vegas = _mae_bias(
                spread_eval["pred_spread"], spread_eval["vegas_spread_home_margin"]
            )
            if spread_eval["vegas_spread_home_margin"].notna().any():
                spread_corr_vs_vegas = float(
                    spread_eval[["pred_spread", "vegas_spread_home_margin"]].corr().iloc[0, 1],
                )
            spread_span_ratio = _span_ratio(spread_eval["pred_spread"], spread_eval["vegas_spread_home_margin"])
        if "vegas_total" in spread_eval.columns:
            total_bias_vs_vegas, total_mae_vs_vegas = _mae_bias(
                spread_eval["pred_total"], spread_eval["vegas_total"]
            )
            if spread_eval["vegas_total"].notna().any():
                total_corr_vs_vegas = float(spread_eval[["pred_total", "vegas_total"]].corr().iloc[0, 1])
            total_span_ratio = _span_ratio(spread_eval["pred_total"], spread_eval["vegas_total"])
        if "act_spread" in spread_eval.columns:
            spread_bias_vs_actual, spread_mae_vs_actual = _mae_bias(
                spread_eval["pred_spread"], spread_eval["act_spread"]
            )
        if "act_total" in spread_eval.columns:
            total_bias_vs_actual, total_mae_vs_actual = _mae_bias(
                spread_eval["pred_total"], spread_eval["act_total"]
            )
        if "estimated_possessions" in spread_eval.columns:
            game_poss_bias_vs_est, game_poss_mae_vs_est = _mae_bias(
                spread_eval["pred_game_poss"], spread_eval["estimated_possessions"]
            )
            if spread_eval["estimated_possessions"].notna().any():
                poss_corr_vs_est = float(
                    spread_eval[["pred_game_poss", "estimated_possessions"]].corr().iloc[0, 1]
                )

    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": str(args.name),
        "dataset_dir": str(dataset_dir),
        "worlds_parquet": str(worlds_path),
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

    if args.out_team_csv:
        out_team = Path(args.out_team_csv).expanduser().resolve()
        out_team.parent.mkdir(parents=True, exist_ok=True)
        team_eval.to_csv(out_team, index=False)
    if args.out_player_csv:
        out_player = Path(args.out_player_csv).expanduser().resolve()
        out_player.parent.mkdir(parents=True, exist_ok=True)
        player_eval.to_csv(out_player, index=False)

    out_json = (
        Path(args.out_json).expanduser().resolve()
        if args.out_json
        else worlds_path.with_name(f"{worlds_path.stem}_make_rate_calibration.json")
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"saved_json={out_json}")


if __name__ == "__main__":
    main()
