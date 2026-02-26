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
    return out[["game_date", "game_id", "team_id", "player_id", "actual_pts", "actual_dk_fpts"]]


def _pred_player_metrics(worlds: pd.DataFrame) -> pd.DataFrame:
    pred = (
        worlds.groupby(["game_date", "game_id", "team_id", "player_id"], as_index=False)
        .agg(
            pred_pts_mean=("pts", "mean"),
            pred_dk_mean=("dk_fpts", "mean"),
            pred_dk_p90=("dk_fpts", lambda x: float(np.quantile(np.asarray(x, dtype=float), 0.90))),
            pred_dk_p95=("dk_fpts", lambda x: float(np.quantile(np.asarray(x, dtype=float), 0.95))),
        )
        .copy()
    )
    return pred


def _mae_bias(pred: pd.Series, actual: pd.Series) -> tuple[float, float]:
    err = pd.to_numeric(pred, errors="coerce") - pd.to_numeric(actual, errors="coerce")
    err = err.replace([np.inf, -np.inf], np.nan).dropna()
    if err.empty:
        return float("nan"), float("nan")
    return float(err.mean()), float(err.abs().mean())


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
    elite_bias, elite_mae = _mae_bias(
        player_eval.loc[elite_mask, "pred_pts_mean"],
        player_eval.loc[elite_mask, "actual_pts"],
    )
    star_bias, star_mae = _mae_bias(
        player_eval.loc[star_mask, "pred_pts_mean"],
        player_eval.loc[star_mask, "actual_pts"],
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
