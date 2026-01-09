#!/usr/bin/env python3
"""Diagnose rotation_set_minutes_v1 live overlay pathologies.

This script is meant to be deterministic for a given (game_date, run_id). It:
  1) Finds a "worst" team-game in the overlay minutes output (max minutes).
  2) Prints the top minutes and basic concentration stats (sum, max, count > 40).
  3) Optionally recomputes baseline minutes_v1 to a debug artifact root and prints
     baseline vs overlay side-by-side for the chosen team-game.
  4) Validates rotation_set_minutes feature contract alignment and surfaces
     suspicious missing/zero patterns + training-distribution drift for key features.

Examples:
  # Just inspect the published overlay minutes + rotation features:
  PROJECTIONS_DATA_ROOT=/home/daniel/projections-data \
  uv run python scripts/diagnostics/rotation_set_minutes_live_pathology.py \
    --game-date 2026-01-05 \
    --run-id 20260105T050145Z

  # Also recompute baseline minutes to a debug root for side-by-side comparison:
  PROJECTIONS_DATA_ROOT=/home/daniel/projections-data \
  uv run python scripts/diagnostics/rotation_set_minutes_live_pathology.py \
    --game-date 2026-01-05 \
    --run-id 20260105T050145Z \
    --compute-baseline \
    --baseline-artifact-root /home/daniel/projections-data/debug/baseline_minutes_v1/daily
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _data_root() -> Path:
    env = os.environ.get("PROJECTIONS_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path("data").resolve()


def _read_feature_columns(model_dir: Path) -> list[str]:
    payload = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))
    cols = payload.get("columns")
    if not isinstance(cols, list) or not cols:
        raise ValueError(f"Invalid feature_columns.json at {model_dir / 'feature_columns.json'}")
    return [str(c) for c in cols]


def _resolve_model_dir(config_path: Path) -> Path:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    model_dir = payload.get("model_dir")
    if not model_dir:
        raise ValueError(f"Missing model_dir in {config_path}")
    return Path(str(model_dir)).expanduser().resolve()


def _team_game_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby(["game_id", "team_id", "team_tricode", "opponent_team_tricode"], dropna=False)
        .agg(
            team_sum=("minutes_p50_cond", "sum"),
            max_p50=("minutes_p50_cond", "max"),
            n_gt_40=("minutes_p50_cond", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > 40.0).sum())),
            n_gt_44=("minutes_p50_cond", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > 44.0).sum())),
            n_gt_48=("minutes_p50_cond", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > 48.0).sum())),
            n_players=("player_id", "nunique"),
        )
        .reset_index()
        .sort_values(["max_p50", "n_gt_48", "n_gt_44", "n_gt_40"], ascending=False)
    )


def _print_top_minutes(df_team: pd.DataFrame, *, title: str, k: int = 12) -> None:
    cols = [
        "player_name",
        "status",
        "is_projected_starter",
        "is_confirmed_starter",
        "play_prob",
        "rotation_prob",
        "minutes_p50_cond",
        "minutes_p90_cond",
    ]
    cols = [c for c in cols if c in df_team.columns]
    print(f"\n[{title}] top {k} minutes_p50_cond")
    print(df_team.sort_values("minutes_p50_cond", ascending=False).loc[:, cols].head(k).to_string(index=False))
    mins = pd.to_numeric(df_team["minutes_p50_cond"], errors="coerce").fillna(0.0)
    print(
        f"[{title}] team_sum={float(mins.sum()):.3f} max={float(mins.max()):.3f} "
        f"count>40={int((mins>40).sum())} count>44={int((mins>44).sum())} count>48={int((mins>48).sum())}"
    )


def _run_baseline_minutes(
    *,
    game_date: str,
    run_id: str,
    artifact_root: Path,
    reconcile_mode: str,
) -> Path:
    env = os.environ.copy()
    env.setdefault("PROJECTIONS_SKIP_POINTER_WRITES", "1")
    cmd = [
        sys.executable,
        "-m",
        "projections.cli.score_minutes_v1",
        "--date",
        game_date,
        "--mode",
        "live",
        "--run-id",
        run_id,
        "--reconcile-team-minutes",
        str(reconcile_mode),
        "--minutes-output",
        "conditional",
        "--artifact-root",
        str(artifact_root),
    ]
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True, check=False)
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Baseline minutes_v1 scorer failed with exit_code={proc.returncode}")
    out = artifact_root / game_date / f"run={run_id}" / "minutes.parquet"
    if not out.exists():
        raise FileNotFoundError(f"Expected baseline minutes.parquet at {out}")
    return out


def _drift_report(
    *,
    model_dir: Path,
    live_rows: pd.DataFrame,
    features: list[str],
) -> None:
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset_dir = Path(str(manifest["dataset_dir"])).expanduser().resolve()
    train_path = dataset_dir / "features.parquet"
    train = pd.read_parquet(train_path, columns=features)

    stats = pd.DataFrame({"feature": features})
    train_vals = train[features]
    stats["train_mean"] = train_vals.mean(numeric_only=True).to_numpy()
    stats["train_std"] = train_vals.std(numeric_only=True).replace(0, np.nan).to_numpy()

    qs = [0.01, 0.5, 0.99]
    qdf = train_vals.quantile(qs, numeric_only=True).T
    qdf.columns = [f"p{int(q * 100):02d}" for q in qs]
    stats = stats.merge(qdf, left_on="feature", right_index=True, how="left")

    print(f"\n[drift] training_features={train_path}")
    print(f"[drift] train_rows={len(train)}")
    for _, row in live_rows.iterrows():
        name = row.get("player_name") or row.get("player_id")
        print(f"\n[drift] player={name} player_id={int(row['player_id'])}")
        out_rows: list[dict[str, object]] = []
        for feat in features:
            live_val = float(row[feat])
            ref = stats[stats["feature"] == feat].iloc[0]
            mean = float(ref["train_mean"])
            std = float(ref["train_std"]) if pd.notna(ref["train_std"]) else float("nan")
            z = (live_val - mean) / std if std and np.isfinite(std) else float("nan")
            out_rows.append(
                {
                    "feature": feat,
                    "live": live_val,
                    "mean": mean,
                    "std": std,
                    "z": z,
                    "p01": float(ref["p01"]),
                    "p50": float(ref["p50"]),
                    "p99": float(ref["p99"]),
                }
            )
        out = pd.DataFrame(out_rows)
        flagged = out[(out["z"].abs() > 5) | (out["live"] <= out["p01"]) | (out["live"] >= out["p99"])].copy()
        if flagged.empty:
            print("[drift] no extreme flags (|z|>5 or outside [p01,p99]).")
        else:
            print("[drift] extreme flags:")
            print(flagged.sort_values("z", key=lambda s: s.abs(), ascending=False).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--game-date", required=True, type=str)
    parser.add_argument("--run-id", required=True, type=str)
    parser.add_argument("--game-id", type=int, default=None, help="Optional override of bad game_id.")
    parser.add_argument("--team-id", type=int, default=None, help="Optional override of bad team_id.")
    parser.add_argument(
        "--minutes-path",
        type=str,
        default=None,
        help="Override path to minutes.parquet (defaults to artifacts/minutes_v1/daily/<date>/run=<id>/minutes.parquet).",
    )
    parser.add_argument(
        "--rotation-features-path",
        type=str,
        default=None,
        help="Override path to rotation features.parquet (defaults to live/features_rotation_set_minutes_v1/<date>/run=<id>/features.parquet).",
    )
    parser.add_argument(
        "--rotation-config",
        type=str,
        default="config/rotation_set_minutes_live.json",
        help="Used only to resolve the model_dir unless --model-dir is provided.",
    )
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--compute-baseline", action="store_true")
    parser.add_argument("--baseline-artifact-root", type=str, default=None)
    parser.add_argument("--reconcile-mode", type=str, default="p50")
    args = parser.parse_args()

    data_root = _data_root()
    game_date = args.game_date
    run_id = args.run_id

    minutes_path = (
        Path(args.minutes_path).expanduser().resolve()
        if args.minutes_path
        else data_root / "artifacts" / "minutes_v1" / "daily" / game_date / f"run={run_id}" / "minutes.parquet"
    )
    if not minutes_path.exists():
        raise FileNotFoundError(f"Missing overlay minutes.parquet at {minutes_path}")

    minutes = pd.read_parquet(minutes_path)
    summary = _team_game_summary(minutes)
    print(f"[inputs] data_root={data_root}")
    print(f"[inputs] minutes_path={minutes_path}")
    print("\n[overlay] worst team-games:")
    print(summary.head(10).to_string(index=False))

    if args.game_id is None or args.team_id is None:
        bad = summary.iloc[0]
        game_id = int(bad["game_id"])
        team_id = int(bad["team_id"])
        team = str(bad["team_tricode"])
        opp = str(bad["opponent_team_tricode"])
    else:
        game_id = int(args.game_id)
        team_id = int(args.team_id)
        team = "?"
        opp = "?"

    print(f"\n[overlay] selected team-game: game_id={game_id} team_id={team_id} team={team} opp={opp}")
    overlay_team = minutes[(minutes["game_id"].astype(int) == game_id) & (minutes["team_id"].astype(int) == team_id)].copy()
    _print_top_minutes(overlay_team, title="overlay")

    # Optional: recompute baseline for side-by-side comparison.
    if args.compute_baseline:
        baseline_root = (
            Path(args.baseline_artifact_root).expanduser().resolve()
            if args.baseline_artifact_root
            else (data_root / "debug" / "baseline_minutes_v1" / "daily")
        )
        baseline_path = _run_baseline_minutes(
            game_date=game_date,
            run_id=run_id,
            artifact_root=baseline_root,
            reconcile_mode=args.reconcile_mode,
        )
        baseline = pd.read_parquet(baseline_path)
        baseline_team = baseline[(baseline["game_id"].astype(int) == game_id) & (baseline["team_id"].astype(int) == team_id)].copy()
        _print_top_minutes(baseline_team, title="baseline")

        merged = baseline_team.merge(
            overlay_team,
            on=["game_id", "team_id", "player_id"],
            how="outer",
            suffixes=("_baseline", "_overlay"),
        )
        merged["delta_p50"] = merged["minutes_p50_cond_overlay"] - merged["minutes_p50_cond_baseline"]
        merged = merged.sort_values("minutes_p50_cond_overlay", ascending=False)
        cols = [
            "player_name_baseline",
            "status_baseline",
            "minutes_p50_cond_baseline",
            "minutes_p50_cond_overlay",
            "delta_p50",
        ]
        cols = [c for c in cols if c in merged.columns]
        print("\n[delta] baseline vs overlay (all players):")
        print(merged.loc[:, cols].to_string(index=False))

    # Feature contract alignment.
    rotation_features_path = (
        Path(args.rotation_features_path).expanduser().resolve()
        if args.rotation_features_path
        else data_root / "live" / "features_rotation_set_minutes_v1" / game_date / f"run={run_id}" / "features.parquet"
    )
    if not rotation_features_path.exists():
        raise FileNotFoundError(f"Missing rotation features.parquet at {rotation_features_path}")
    rotation_features = pd.read_parquet(rotation_features_path)

    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else _resolve_model_dir(Path(args.rotation_config))
    feature_cols = _read_feature_columns(model_dir)
    keep_cols = ["opponent_team_id"] if "opponent_team_id" in rotation_features.columns else []
    allowed = set(["game_id", "team_id", "player_id", *keep_cols, *feature_cols])
    extras = [c for c in rotation_features.columns if c not in allowed]
    missing = [c for c in feature_cols if c not in rotation_features.columns]

    print(f"\n[features] rotation_features_path={rotation_features_path}")
    print(f"[features] rows={len(rotation_features)} cols={len(rotation_features.columns)}")
    print(f"[features] model_dir={model_dir}")
    print(f"[features] manifest_feature_cols={len(feature_cols)}")
    print(f"[features] extras_in_live_not_manifest={extras}")
    print(f"[features] missing_from_live_vs_manifest={missing}")

    # Missing rates will usually be 0 due to training-consistent fill-to-zero.
    miss_rate = rotation_features[feature_cols].isna().mean().sort_values(ascending=False)
    nonzero = miss_rate[miss_rate > 0]
    print(f"[features] nonzero_missing_features={len(nonzero)}")
    if not nonzero.empty:
        print(nonzero.head(30).to_string())

    # Bad team-game rows, joined with overlay minutes for readability.
    rot_team = rotation_features[(rotation_features["game_id"].astype(int) == game_id) & (rotation_features["team_id"].astype(int) == team_id)].copy()
    rot_team = rot_team.merge(
        overlay_team.loc[:, ["player_id", "player_name", "minutes_p50_cond"]],
        on="player_id",
        how="left",
    ).sort_values("minutes_p50_cond", ascending=False)

    print("\n[features] bad team-game quick-view (selected columns):")
    focus = [
        "player_name",
        "player_id",
        "minutes_p50_cond",
        "prior_play_prob",
        "is_out",
        "is_q",
        "is_prob",
        "is_projected_starter",
        "is_confirmed_starter",
        "minutes_features_row_missing",
        "roll_mean_10",
        "sum_min_7d",
        "days_since_last",
        "spread_home",
        "total",
        "blowout_index",
        "available_B",
        "available_G",
        "available_W",
    ]
    focus = [c for c in focus if c in rot_team.columns]
    print(rot_team.loc[:, focus].head(12).to_string(index=False))

    # Training-distribution drift (top 8 by overlay minutes).
    drift_features = [
        c
        for c in [
            "prior_play_prob",
            "is_out",
            "is_q",
            "is_prob",
            "is_projected_starter",
            "is_confirmed_starter",
            "minutes_features_row_missing",
            "roll_mean_10",
            "sum_min_7d",
            "days_since_last",
            "spread_home",
            "total",
            "blowout_index",
            "available_B",
            "available_G",
            "available_W",
            "bench_pool_minutes_prior_5",
            "bench_conc_top1_prior_5",
            "starter_pool_minutes_prior_5",
            "team_minutes_dispersion_prior",
        ]
        if c in feature_cols
    ]
    if drift_features:
        top8 = rot_team.head(8).copy()
        _drift_report(model_dir=model_dir, live_rows=top8.assign(player_name=top8.get("player_name")), features=drift_features)


if __name__ == "__main__":
    main()

