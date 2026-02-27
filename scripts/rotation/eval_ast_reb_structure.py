#!/usr/bin/env python3
"""Audit GTv2 assist/rebound bias and structural coupling against holdout labels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    df = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 2:
        return float("nan")
    if float(df["left"].std(ddof=0)) <= 1e-12 or float(df["right"].std(ddof=0)) <= 1e-12:
        return float("nan")
    return float(df["left"].corr(df["right"]))


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


def _safe_rate(num: pd.Series, den: pd.Series, *, max_rate: float = 1.0) -> pd.Series:
    num_s = pd.to_numeric(num, errors="coerce")
    den_s = pd.to_numeric(den, errors="coerce")
    rate = num_s / den_s.replace(0.0, np.nan)
    rate = rate.replace([np.inf, -np.inf], np.nan)
    return rate.clip(lower=0.0, upper=max_rate)


def _load_projection_frame(path: Path) -> pd.DataFrame:
    cols = ["game_date", "game_id", "team_id", "player_id", "minutes_mean", "pts_mean", "reb_mean", "ast_mean"]
    df = pd.read_parquet(path, columns=cols)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    for col in ["game_id", "team_id", "player_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def _load_labels_frame(path: Path, *, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    cols = [
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
    ]
    df = pd.read_parquet(path, columns=cols)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    if start_date is not None:
        df = df.loc[df["game_date"] >= pd.Timestamp(start_date)].copy()
    if end_date is not None:
        df = df.loc[df["game_date"] <= pd.Timestamp(end_date)].copy()
    for col in ["game_id", "team_id", "player_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["fga2", "fg2m", "fga3", "fg3m", "fta", "ftm", "oreb", "dreb", "ast", "minutes", "played"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["reb_actual"] = df["oreb"] + df["dreb"]
    df["pts_actual"] = 2.0 * df["fg2m"] + 3.0 * df["fg3m"] + df["ftm"]
    df["fgm_actual"] = df["fg2m"] + df["fg3m"]
    df["missed_fg_actual"] = (df["fga2"] + df["fga3"]) - df["fgm_actual"]
    return df


def _team_actual_frame(labels: pd.DataFrame) -> pd.DataFrame:
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
    return team_actual


def _load_worlds_frame(path: Path, *, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    cols = [
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
    ]
    df = pd.read_parquet(path, columns=cols)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    if start_date is not None:
        df = df.loc[df["game_date"] >= pd.Timestamp(start_date)].copy()
    if end_date is not None:
        df = df.loc[df["game_date"] <= pd.Timestamp(end_date)].copy()
    for col in ["world_idx", "game_id", "team_id", "player_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["active", "minutes", "pts", "reb", "ast", "oreb", "dreb", "fga2", "fg2m", "fga3", "fg3m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["fgm"] = df["fg2m"] + df["fg3m"]
    df["missed_fg"] = (df["fga2"] + df["fga3"]) - df["fgm"]
    return df


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


def build_report(
    *,
    projections_path: Path,
    worlds_path: Path,
    labels_path: Path,
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    proj = _load_projection_frame(projections_path)
    if start_date is not None:
        proj = proj.loc[proj["game_date"] >= pd.Timestamp(start_date)].copy()
    if end_date is not None:
        proj = proj.loc[proj["game_date"] <= pd.Timestamp(end_date)].copy()

    if proj.empty:
        raise ValueError("No projection rows found for requested window")

    labels = _load_labels_frame(labels_path, start_date=start_date, end_date=end_date)
    labels = labels.loc[labels["game_date"].isin(proj["game_date"].unique())].copy()
    merged = proj.merge(labels, on=["game_date", "game_id", "team_id", "player_id"], how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No joined projection/label rows found for requested window")

    team_actual = _team_actual_frame(labels)
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

    worlds = _load_worlds_frame(worlds_path, start_date=start_date, end_date=end_date)
    if worlds.empty:
        raise ValueError("No world rows found for requested window")

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
    team_actual_rates["oreb_capture_rate_actual"] = _safe_rate(
        team_actual_rates["oreb_actual"],
        team_actual_rates["missed_fg_actual"],
    )
    team_actual_rates["dreb_capture_rate_actual"] = _safe_rate(
        team_actual_rates["dreb_actual"],
        team_actual_rates["opp_missed_fg_actual"],
    )
    team_actual_rates["ast_on_fgm_rate_actual"] = _safe_rate(
        team_actual_rates["ast_actual"],
        team_actual_rates["fgm_actual"],
    )

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
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "projections_parquet": str(projections_path),
            "worlds_parquet": str(worlds_path),
            "labels_parquet": str(labels_path),
        },
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=str, required=True, help="Dataset dir containing labels_boxscore_counts.parquet.")
    parser.add_argument("--projections-parquet", type=str, required=True)
    parser.add_argument("--worlds-parquet", type=str, required=True)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--out-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    projections_path = Path(args.projections_parquet).expanduser().resolve()
    worlds_path = Path(args.worlds_parquet).expanduser().resolve()
    labels_path = (dataset_dir / "labels_boxscore_counts.parquet").resolve()

    report = build_report(
        projections_path=projections_path,
        worlds_path=worlds_path,
        labels_path=labels_path,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    out_json = (
        Path(args.out_json).expanduser().resolve()
        if args.out_json
        else worlds_path.with_name(f"{worlds_path.stem}_ast_reb_structure.json")
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
