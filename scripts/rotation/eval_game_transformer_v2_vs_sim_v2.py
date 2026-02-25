#!/usr/bin/env python3
"""Offline eval suite: GameTransformerV2 worlds vs sim_v2 worlds."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from projections import paths

PLAYER_KEY = tuple[str, int, int, int]
TEAM_KEY = tuple[str, int, int]
PAIR_KEY = tuple[str, int, int, int, int]


@dataclass
class SampleBundle:
    player_samples: dict[PLAYER_KEY, np.ndarray]
    team_samples: dict[TEAM_KEY, np.ndarray]
    pair_corr: dict[PAIR_KEY, float]
    source_rows: int


def _date_range(start: str, end: str) -> list[date]:
    s = datetime.fromisoformat(str(start)).date()
    e = datetime.fromisoformat(str(end)).date()
    if e < s:
        raise ValueError("end_date must be >= start_date")
    out: list[date] = []
    d = s
    while d <= e:
        out.append(d)
        d = d + timedelta(days=1)
    return out


def _compute_actual_fpts(labels_counts: pd.DataFrame, *, start_date: str, end_date: str) -> tuple[dict[PLAYER_KEY, float], dict[TEAM_KEY, float]]:
    req = {"game_date", "game_id", "team_id", "player_id", "fg2m", "fg3m", "ftm", "oreb", "dreb", "ast", "stl", "blk", "tov"}
    missing = sorted(req - set(labels_counts.columns))
    if missing:
        raise ValueError(f"labels_boxscore_counts missing columns: {missing}")

    df = labels_counts.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    s = datetime.fromisoformat(str(start_date)).date()
    e = datetime.fromisoformat(str(end_date)).date()
    df = df.loc[(df["game_date"] >= s) & (df["game_date"] <= e)].copy()

    for col in ["fg2m", "fg3m", "ftm", "oreb", "dreb", "ast", "stl", "blk", "tov"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    pts = 2.0 * df["fg2m"] + 3.0 * df["fg3m"] + df["ftm"]
    reb = df["oreb"] + df["dreb"]
    base = pts + 1.25 * reb + 1.5 * df["ast"] + 2.0 * df["stl"] + 2.0 * df["blk"] - 0.5 * df["tov"]
    qualifying = pd.concat(
        [
            (pts >= 10.0).astype(int),
            (reb >= 10.0).astype(int),
            (df["ast"] >= 10.0).astype(int),
            (df["stl"] >= 10.0).astype(int),
            (df["blk"] >= 10.0).astype(int),
        ],
        axis=1,
    ).sum(axis=1)
    df["dk_fpts_actual"] = base + np.where(qualifying == 2, 1.5, 0.0) + np.where(qualifying >= 3, 3.0, 0.0)

    player_actual: dict[PLAYER_KEY, float] = {}
    for row in df.itertuples(index=False):
        key = (str(row.game_date), int(row.game_id), int(row.team_id), int(row.player_id))
        player_actual[key] = float(row.dk_fpts_actual)

    team_actual_df = (
        df.groupby(["game_date", "game_id", "team_id"], as_index=False)
        .agg(dk_fpts_actual=("dk_fpts_actual", "sum"))
    )
    team_actual: dict[TEAM_KEY, float] = {}
    for row in team_actual_df.itertuples(index=False):
        team_actual[(str(row.game_date), int(row.game_id), int(row.team_id))] = float(row.dk_fpts_actual)

    return player_actual, team_actual


def _build_pair_corr_from_matrix(matrix: np.ndarray, player_ids: list[int], *, date_key: str, game_id: int, team_id: int) -> dict[PAIR_KEY, float]:
    out: dict[PAIR_KEY, float] = {}
    if matrix.shape[1] < 2:
        return out
    corr = np.corrcoef(matrix, rowvar=False)
    for i in range(len(player_ids)):
        for j in range(i + 1, len(player_ids)):
            v = float(corr[i, j])
            if not np.isfinite(v):
                continue
            p1 = int(min(player_ids[i], player_ids[j]))
            p2 = int(max(player_ids[i], player_ids[j]))
            out[(date_key, int(game_id), int(team_id), p1, p2)] = v
    return out


def _load_game_transformer_worlds(path: Path, *, start_date: str, end_date: str) -> SampleBundle:
    cols = ["world_idx", "game_date", "game_id", "team_id", "player_id", "dk_fpts"]
    df = pd.read_parquet(path, columns=cols)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date.astype(str)
    s = str(datetime.fromisoformat(str(start_date)).date())
    e = str(datetime.fromisoformat(str(end_date)).date())
    df = df.loc[(df["game_date"] >= s) & (df["game_date"] <= e)].copy()
    df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["dk_fpts"] = pd.to_numeric(df["dk_fpts"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["game_id", "team_id", "player_id"]).copy()

    player_samples: dict[PLAYER_KEY, np.ndarray] = {}
    for key, grp in df.groupby(["game_date", "game_id", "team_id", "player_id"], sort=False):
        arr = grp.sort_values("world_idx")["dk_fpts"].to_numpy(dtype=np.float32, copy=False)
        player_samples[(str(key[0]), int(key[1]), int(key[2]), int(key[3]))] = arr

    team_samples: dict[TEAM_KEY, np.ndarray] = {}
    team_world = (
        df.groupby(["game_date", "game_id", "team_id", "world_idx"], as_index=False)
        .agg(team_fpts=("dk_fpts", "sum"))
    )
    for key, grp in team_world.groupby(["game_date", "game_id", "team_id"], sort=False):
        arr = grp.sort_values("world_idx")["team_fpts"].to_numpy(dtype=np.float32, copy=False)
        team_samples[(str(key[0]), int(key[1]), int(key[2]))] = arr

    pair_corr: dict[PAIR_KEY, float] = {}
    for key, grp in df.groupby(["game_date", "game_id", "team_id"], sort=False):
        pivot = grp.pivot(index="world_idx", columns="player_id", values="dk_fpts").sort_index()
        if pivot.empty:
            continue
        mat = pivot.to_numpy(dtype=np.float32, copy=False)
        pids = [int(v) for v in pivot.columns.tolist()]
        pair_corr.update(
            _build_pair_corr_from_matrix(
                mat,
                pids,
                date_key=str(key[0]),
                game_id=int(key[1]),
                team_id=int(key[2]),
            )
        )

    return SampleBundle(
        player_samples=player_samples,
        team_samples=team_samples,
        pair_corr=pair_corr,
        source_rows=int(len(df)),
    )


def _load_sim_v2_worlds(
    *,
    worlds_root: Path,
    run_id: str,
    start_date: str,
    end_date: str,
) -> SampleBundle:
    player_samples: dict[PLAYER_KEY, np.ndarray] = {}
    team_samples: dict[TEAM_KEY, np.ndarray] = {}
    pair_corr: dict[PAIR_KEY, float] = {}
    rows_scanned = 0

    for day in _date_range(start_date, end_date):
        date_key = day.isoformat()
        base = worlds_root / f"game_date={date_key}" / f"run={run_id}"
        worlds_path = base / "worlds_matrix.parquet"
        proj_path = base / "projections.parquet"
        if not worlds_path.exists() or not proj_path.exists():
            continue

        worlds_df = pd.read_parquet(worlds_path)
        proj_df = pd.read_parquet(proj_path, columns=["game_date", "game_id", "team_id", "player_id"])
        if worlds_df.empty or proj_df.empty:
            continue
        rows_scanned += int(len(proj_df))

        proj_df["game_date"] = pd.to_datetime(proj_df["game_date"], errors="coerce").dt.date.astype(str)
        proj_df["game_id"] = pd.to_numeric(proj_df["game_id"], errors="coerce").astype("Int64")
        proj_df["team_id"] = pd.to_numeric(proj_df["team_id"], errors="coerce").astype("Int64")
        proj_df["player_id"] = pd.to_numeric(proj_df["player_id"], errors="coerce").astype("Int64")
        proj_df = proj_df.dropna(subset=["game_id", "team_id", "player_id"]).copy()

        for key, grp in proj_df.groupby(["game_date", "game_id", "team_id"], sort=False):
            pids = [int(v) for v in grp["player_id"].tolist()]
            cols: list[str] = []
            keep_pids: list[int] = []
            for pid in pids:
                c = str(int(pid))
                if c in worlds_df.columns:
                    cols.append(c)
                    keep_pids.append(int(pid))
            if not cols:
                continue

            mat = worlds_df.loc[:, cols].to_numpy(dtype=np.float32, copy=False)
            for idx, pid in enumerate(keep_pids):
                player_samples[(str(key[0]), int(key[1]), int(key[2]), int(pid))] = mat[:, idx]
            team_samples[(str(key[0]), int(key[1]), int(key[2]))] = mat.sum(axis=1)
            pair_corr.update(
                _build_pair_corr_from_matrix(
                    mat,
                    keep_pids,
                    date_key=str(key[0]),
                    game_id=int(key[1]),
                    team_id=int(key[2]),
                )
            )

    return SampleBundle(
        player_samples=player_samples,
        team_samples=team_samples,
        pair_corr=pair_corr,
        source_rows=rows_scanned,
    )


def _crps(samples: np.ndarray, y: float) -> float:
    x = np.asarray(samples, dtype=np.float64)
    if x.size <= 0:
        return float("nan")
    term_obs = float(np.mean(np.abs(x - float(y))))
    pair = np.abs(x[:, None] - x[None, :])
    term_pair = 0.5 * float(np.mean(pair))
    return term_obs - term_pair


def _safe_mean(values: list[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size <= 0:
        return float("nan")
    return float(arr.mean())


def _compute_metrics(
    bundle: SampleBundle,
    *,
    player_actual: dict[PLAYER_KEY, float],
    team_actual: dict[TEAM_KEY, float],
) -> dict[str, Any]:
    crps_vals: list[float] = []
    cov90: list[float] = []
    cov95: list[float] = []

    for key, y in player_actual.items():
        samples = bundle.player_samples.get(key)
        if samples is None or samples.size <= 0:
            continue
        crps_vals.append(_crps(samples, y))
        cov90.append(float(y <= float(np.quantile(samples, 0.90))))
        cov95.append(float(y <= float(np.quantile(samples, 0.95))))

    team_mae: list[float] = []
    team_std_score: list[float] = []
    team_var: list[float] = []
    for key, y in team_actual.items():
        samples = bundle.team_samples.get(key)
        if samples is None or samples.size <= 0:
            continue
        mean = float(np.mean(samples))
        var = float(np.var(samples))
        team_mae.append(abs(mean - float(y)))
        team_var.append(var)
        team_std_score.append(((float(y) - mean) ** 2) / max(var, 1e-6))

    pair_vals = [v for v in bundle.pair_corr.values() if np.isfinite(v)]
    out = {
        "n_player_actual": int(len(player_actual)),
        "n_player_scored": int(len(crps_vals)),
        "n_team_actual": int(len(team_actual)),
        "n_team_scored": int(len(team_mae)),
        "n_pair_corr": int(len(pair_vals)),
        "crps_mean": _safe_mean(crps_vals),
        "p90_coverage": _safe_mean(cov90),
        "p95_coverage": _safe_mean(cov95),
        "p90_calibration_error_abs": float(abs(_safe_mean(cov90) - 0.90)) if cov90 else float("nan"),
        "p95_calibration_error_abs": float(abs(_safe_mean(cov95) - 0.95)) if cov95 else float("nan"),
        "team_total_mae": _safe_mean(team_mae),
        "team_variance_calibration_mse_norm": _safe_mean(team_std_score),
        "team_variance_mean": _safe_mean(team_var),
        "same_team_pair_corr_mean": _safe_mean(pair_vals),
        "source_rows": int(bundle.source_rows),
    }
    return out


def _delta(gt: dict[str, Any], sim: dict[str, Any], keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in keys:
        gv = float(gt.get(k, float("nan")))
        sv = float(sim.get(k, float("nan")))
        out[k] = float(gv - sv) if np.isfinite(gv) and np.isfinite(sv) else float("nan")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--gt-worlds-parquet", type=str, required=True)
    parser.add_argument("--sim-v2-run-id", type=str, required=True)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--sim-v2-worlds-root", type=str, default=None)
    parser.add_argument("--out-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    gt_worlds = Path(args.gt_worlds_parquet).expanduser().resolve()
    if not gt_worlds.exists():
        raise FileNotFoundError(f"gt worlds parquet not found: {gt_worlds}")

    labels_counts = pd.read_parquet(dataset_dir / "labels_boxscore_counts.parquet")
    player_actual, team_actual = _compute_actual_fpts(
        labels_counts,
        start_date=str(args.start_date),
        end_date=str(args.end_date),
    )

    gt_bundle = _load_game_transformer_worlds(
        gt_worlds,
        start_date=str(args.start_date),
        end_date=str(args.end_date),
    )
    sim_root = (
        Path(args.sim_v2_worlds_root).expanduser().resolve()
        if args.sim_v2_worlds_root
        else (paths.get_data_root() / "artifacts" / "sim_v2" / "worlds_fpts_v2").resolve()
    )
    sim_bundle = _load_sim_v2_worlds(
        worlds_root=sim_root,
        run_id=str(args.sim_v2_run_id),
        start_date=str(args.start_date),
        end_date=str(args.end_date),
    )

    gt_metrics = _compute_metrics(gt_bundle, player_actual=player_actual, team_actual=team_actual)
    sim_metrics = _compute_metrics(sim_bundle, player_actual=player_actual, team_actual=team_actual)

    common_pair_keys = set(gt_bundle.pair_corr.keys()) & set(sim_bundle.pair_corr.keys())
    pair_rmse_vs_sim = float("nan")
    if common_pair_keys:
        diffs = [
            float(gt_bundle.pair_corr[k] - sim_bundle.pair_corr[k])
            for k in common_pair_keys
            if np.isfinite(gt_bundle.pair_corr[k]) and np.isfinite(sim_bundle.pair_corr[k])
        ]
        if diffs:
            pair_rmse_vs_sim = float(np.sqrt(np.mean(np.square(np.asarray(diffs, dtype=np.float64)))))

    gt_metrics["pair_corr_rmse_vs_sim_v2"] = pair_rmse_vs_sim
    sim_metrics["pair_corr_rmse_vs_sim_v2"] = 0.0

    delta_metrics = _delta(
        gt_metrics,
        sim_metrics,
        [
            "crps_mean",
            "p90_calibration_error_abs",
            "p95_calibration_error_abs",
            "team_total_mae",
            "team_variance_calibration_mse_norm",
            "same_team_pair_corr_mean",
            "pair_corr_rmse_vs_sim_v2",
        ],
    )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "gt_worlds_parquet": str(gt_worlds),
        "sim_v2_worlds_root": str(sim_root),
        "sim_v2_run_id": str(args.sim_v2_run_id),
        "evaluation_semantics": {
            "default_view": "unconditional_dnp_zero",
            "explanation": "Metrics are computed from full worlds samples; inactive worlds contribute zero.",
        },
        "window": {
            "start_date": str(args.start_date),
            "end_date": str(args.end_date),
        },
        "actual_counts": {
            "player_rows": int(len(player_actual)),
            "team_rows": int(len(team_actual)),
        },
        "game_transformer_v2": gt_metrics,
        "sim_v2": sim_metrics,
        "delta_gt_minus_sim_v2": delta_metrics,
        "go_no_go_checks": {
            "crps_improved_vs_sim_v2": bool(
                np.isfinite(gt_metrics["crps_mean"])
                and np.isfinite(sim_metrics["crps_mean"])
                and float(gt_metrics["crps_mean"]) < float(sim_metrics["crps_mean"])
            ),
            "tail_p90_error_leq_0_03": bool(
                np.isfinite(gt_metrics["p90_calibration_error_abs"])
                and float(gt_metrics["p90_calibration_error_abs"]) <= 0.03
            ),
            "tail_p95_error_leq_0_03": bool(
                np.isfinite(gt_metrics["p95_calibration_error_abs"])
                and float(gt_metrics["p95_calibration_error_abs"]) <= 0.03
            ),
        },
    }

    out_json = (
        Path(args.out_json).expanduser().resolve()
        if args.out_json
        else (gt_worlds.parent / f"offline_eval_vs_sim_v2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"saved_eval_json={out_json}")


if __name__ == "__main__":
    main()
