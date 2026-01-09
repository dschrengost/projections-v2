#!/usr/bin/env python
"""Audit a single sim_v2 run for correctness and observability.

Usage:
    uv run python -m scripts.sim_v2.audit_one_run --date 2026-01-08 --run-id 20260108T152500Z
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

from projections.paths import data_path

app = typer.Typer(add_completion=False)


def _load_run_artifacts(
    data_root: Path, game_date: str, run_id: str
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, Any]]:
    """Load projections.parquet and worlds_matrix.parquet for a run."""
    run_dir = (
        data_root
        / "artifacts"
        / "sim_v2"
        / "worlds_fpts_v2"
        / f"game_date={game_date}"
        / f"run={run_id}"
    )
    proj_path = run_dir / "projections.parquet"
    matrix_path = run_dir / "worlds_matrix.parquet"
    metrics_path = run_dir / "metrics.json"
    diag_path = run_dir / "sim_diagnostics.json"

    proj_df = pd.read_parquet(proj_path) if proj_path.exists() else None
    worlds_df = pd.read_parquet(matrix_path) if matrix_path.exists() else None

    meta: dict[str, Any] = {"run_dir": str(run_dir)}
    if metrics_path.exists():
        with open(metrics_path) as f:
            meta["metrics"] = json.load(f)
    if diag_path.exists():
        with open(diag_path) as f:
            meta["diagnostics"] = json.load(f)

    return proj_df, worlds_df, meta


def _compute_dk_fpts_stats(
    worlds_df: pd.DataFrame,
) -> dict[str, float]:
    """Recompute dk_fpts_world stats from raw matrix."""
    all_vals = worlds_df.values.astype(float).flatten()
    # Filter out zeros (inactive players)
    active_vals = all_vals[all_vals > 0]

    return {
        "all_min": float(np.nanmin(all_vals)),
        "all_median": float(np.nanmedian(all_vals)),
        "all_max": float(np.nanmax(all_vals)),
        "all_mean": float(np.nanmean(all_vals)),
        "active_min": float(np.nanmin(active_vals)) if len(active_vals) else np.nan,
        "active_median": float(np.nanmedian(active_vals)) if len(active_vals) else np.nan,
        "active_max": float(np.nanmax(active_vals)) if len(active_vals) else np.nan,
        "active_mean": float(np.nanmean(active_vals)) if len(active_vals) else np.nan,
    }


def _compute_per_player_max(worlds_df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """Get top N players by max FPTS across all worlds."""
    per_player_max = worlds_df.max(axis=0)
    return per_player_max.nlargest(top_n)


def _compute_minutes_team_sums(
    proj_df: pd.DataFrame,
) -> dict[str, float]:
    """Check team minutes sums from projections."""
    if "minutes_sim_mean" not in proj_df.columns:
        return {}

    team_sums = proj_df.groupby(["game_id", "team_id"])["minutes_sim_mean"].sum()
    return {
        "min_team_sum": float(team_sums.min()),
        "max_team_sum": float(team_sums.max()),
        "mean_team_sum": float(team_sums.mean()),
        "teams_off_240": int((np.abs(team_sums - 240.0) > 1.0).sum()),
    }


def _compute_same_team_correlation(
    worlds_df: pd.DataFrame, proj_df: pd.DataFrame, n_pairs: int = 50
) -> dict[str, float]:
    """Sample same-team player pairs and compute FPTS correlation."""
    if proj_df is None or "team_id" not in proj_df.columns:
        return {}

    # Build player_id -> team_id mapping
    player_team = dict(zip(proj_df["player_id"].astype(str), proj_df["team_id"]))

    # Get player columns in worlds matrix
    player_cols = [c for c in worlds_df.columns if c in player_team]
    if len(player_cols) < 2:
        return {}

    # Find same-team pairs
    team_players: dict[int, list[str]] = {}
    for p in player_cols:
        tid = player_team.get(p)
        if tid is not None:
            team_players.setdefault(tid, []).append(p)

    # Sample pairs
    same_team_corrs = []
    diff_team_corrs = []

    rng = np.random.default_rng(42)
    for tid, players in team_players.items():
        if len(players) >= 2:
            # Sample up to 3 pairs per team
            for _ in range(min(3, len(players) // 2)):
                p1, p2 = rng.choice(players, 2, replace=False)
                corr = np.corrcoef(worlds_df[p1].values, worlds_df[p2].values)[0, 1]
                if np.isfinite(corr):
                    same_team_corrs.append(corr)

    # Sample some cross-team pairs for comparison
    all_teams = list(team_players.keys())
    if len(all_teams) >= 2:
        for _ in range(min(20, n_pairs)):
            t1, t2 = rng.choice(all_teams, 2, replace=False)
            if team_players[t1] and team_players[t2]:
                p1 = rng.choice(team_players[t1])
                p2 = rng.choice(team_players[t2])
                corr = np.corrcoef(worlds_df[p1].values, worlds_df[p2].values)[0, 1]
                if np.isfinite(corr):
                    diff_team_corrs.append(corr)

    result = {}
    if same_team_corrs:
        result["same_team_corr_mean"] = float(np.mean(same_team_corrs))
        result["same_team_corr_std"] = float(np.std(same_team_corrs))
        result["same_team_corr_n"] = len(same_team_corrs)
    if diff_team_corrs:
        result["diff_team_corr_mean"] = float(np.mean(diff_team_corrs))
        result["diff_team_corr_std"] = float(np.std(diff_team_corrs))
        result["diff_team_corr_n"] = len(diff_team_corrs)

    return result


def _check_extreme_values(
    worlds_df: pd.DataFrame, threshold: float = 120.0
) -> list[dict[str, Any]]:
    """Find player-worlds with FPTS above threshold."""
    extreme = []
    per_player_max = worlds_df.max(axis=0)
    for player_id, max_val in per_player_max.items():
        if max_val > threshold:
            n_above = int((worlds_df[player_id] > threshold).sum())
            extreme.append({
                "player_id": player_id,
                "max_fpts": float(max_val),
                "n_worlds_above_threshold": n_above,
            })
    return sorted(extreme, key=lambda x: -x["max_fpts"])


@app.command()
def main(
    date: str = typer.Option(..., "--date", help="Game date (YYYY-MM-DD)"),
    run_id: str = typer.Option(..., "--run-id", help="Run ID (e.g., 20260108T152500Z)"),
    data_root: Path | None = typer.Option(None, "--data-root", help="Data root override"),
    extreme_threshold: float = typer.Option(120.0, "--extreme-threshold", help="FPTS threshold for extreme value warnings"),
    output_json: Path | None = typer.Option(None, "--output-json", help="Write audit report to JSON file"),
) -> None:
    """Audit a sim_v2 run: schema, stats, correlations, guardrails."""
    root = data_root or data_path()

    typer.echo(f"[audit] Loading run: date={date} run_id={run_id}")
    proj_df, worlds_df, meta = _load_run_artifacts(root, date, run_id)

    report: dict[str, Any] = {
        "date": date,
        "run_id": run_id,
        "run_dir": meta.get("run_dir"),
    }

    # 1. Schema summary
    if proj_df is not None:
        report["projections"] = {
            "rows": len(proj_df),
            "columns": list(proj_df.columns),
        }
        typer.echo(f"[audit] projections.parquet: {len(proj_df)} rows, {len(proj_df.columns)} columns")
    else:
        typer.echo("[audit] WARNING: projections.parquet not found", err=True)

    if worlds_df is not None:
        n_worlds, n_players = worlds_df.shape
        report["worlds_matrix"] = {
            "n_worlds": n_worlds,
            "n_players": n_players,
        }
        typer.echo(f"[audit] worlds_matrix.parquet: {n_worlds} worlds × {n_players} players")
    else:
        typer.echo("[audit] WARNING: worlds_matrix.parquet not found", err=True)
        return

    # 2. dk_fpts_world stats (recomputed)
    fpts_stats = _compute_dk_fpts_stats(worlds_df)
    report["dk_fpts_world"] = fpts_stats
    typer.echo(
        f"[audit] dk_fpts_world: min={fpts_stats['all_min']:.2f} "
        f"med={fpts_stats['all_median']:.2f} max={fpts_stats['all_max']:.2f} "
        f"mean={fpts_stats['all_mean']:.2f}"
    )

    # 3. Per-player max top 10
    top_players = _compute_per_player_max(worlds_df, top_n=10)
    report["top_players_by_max"] = [
        {"player_id": str(pid), "max_fpts": float(val)}
        for pid, val in top_players.items()
    ]
    typer.echo("\n[audit] Top 10 players by max FPTS across worlds:")
    for pid, val in top_players.items():
        typer.echo(f"  {pid}: {val:.2f}")

    # 4. Extreme values check
    extremes = _check_extreme_values(worlds_df, threshold=extreme_threshold)
    report["extreme_values"] = {
        "threshold": extreme_threshold,
        "count": len(extremes),
        "players": extremes[:10],  # Top 10
    }
    if extremes:
        typer.echo(f"\n[audit] WARNING: {len(extremes)} players have worlds > {extreme_threshold} FPTS", err=True)
        for ext in extremes[:5]:
            typer.echo(
                f"  player_id={ext['player_id']} max={ext['max_fpts']:.2f} "
                f"n_worlds_above={ext['n_worlds_above_threshold']}",
                err=True,
            )

    # 5. Minutes team sums
    if proj_df is not None:
        minutes_stats = _compute_minutes_team_sums(proj_df)
        report["minutes_team_sums"] = minutes_stats
        if minutes_stats:
            typer.echo(
                f"\n[audit] Team minutes sums: min={minutes_stats['min_team_sum']:.1f} "
                f"max={minutes_stats['max_team_sum']:.1f} teams_off_240={minutes_stats['teams_off_240']}"
            )

    # 6. Same-team correlation
    if proj_df is not None:
        corr_stats = _compute_same_team_correlation(worlds_df, proj_df)
        report["correlation"] = corr_stats
        if "same_team_corr_mean" in corr_stats:
            typer.echo(
                f"\n[audit] Same-team correlation: mean={corr_stats['same_team_corr_mean']:.3f} "
                f"std={corr_stats['same_team_corr_std']:.3f} (n={corr_stats['same_team_corr_n']})"
            )
        if "diff_team_corr_mean" in corr_stats:
            typer.echo(
                f"[audit] Diff-team correlation: mean={corr_stats['diff_team_corr_mean']:.3f} "
                f"std={corr_stats['diff_team_corr_std']:.3f} (n={corr_stats['diff_team_corr_n']})"
            )

    # 7. Include existing metrics/diagnostics
    if "metrics" in meta:
        report["run_metrics"] = meta["metrics"]
    if "diagnostics" in meta:
        report["run_diagnostics"] = meta["diagnostics"]

    # Output
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        typer.echo(f"\n[audit] Report written to {output_json}")

    typer.echo("\n[audit] Done.")


if __name__ == "__main__":
    app()
