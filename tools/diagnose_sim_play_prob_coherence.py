"""Diagnose coherence between play_prob inputs and simulated "played" rates.

Example:
  uv run python tools/diagnose_sim_play_prob_coherence.py --date 2026-01-28
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from projections.paths import get_data_root


app = typer.Typer(add_completion=False)


def _read_latest_run_id(day_dir: Path) -> str | None:
    pointer = day_dir / "latest_run.json"
    if not pointer.exists():
        return None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    run_id = payload.get("run_id")
    return str(run_id) if run_id else None


def _resolve_run_dir(day_dir: Path, run_id: str | None) -> Path:
    if run_id:
        run_dir = day_dir / f"run={run_id}"
        if run_dir.exists():
            return run_dir
    # Fallback: newest run directory.
    run_dirs = sorted([p for p in day_dir.glob("run=*") if p.is_dir()], reverse=True)
    if run_dirs:
        return run_dirs[0]
    return day_dir


def _as_str_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df))
    return df[col].astype(str)


@app.command()
def main(
    *,
    date: str = typer.Option(..., "--date", help="Slate date (YYYY-MM-DD)."),
    sim_run_id: str | None = typer.Option(None, "--sim-run-id", help="Optional sim run_id override."),
    projections_run_id: str | None = typer.Option(None, "--projections-run-id", help="Optional unified projections run_id override."),
    data_root: Path | None = typer.Option(None, "--data-root", help="Optional PROJECTIONS_DATA_ROOT override."),
    n_players: int = typer.Option(20, "--n-players", help="How many players to print (evenly spaced by uncond minutes)."),
    play_threshold_minutes: float = typer.Option(1.0, "--play-threshold-minutes", help="Minutes >= this => 'played'."),
    rotation_threshold_minutes: float = typer.Option(5.0, "--rotation-threshold-minutes", help="Minutes >= this => 'meaningful rotation'."),
) -> None:
    root = (Path(data_root).expanduser().resolve() if data_root is not None else get_data_root())
    day = str(date).strip()

    sim_day_dir = root / "artifacts" / "sim_v2" / "worlds_fpts_v2" / f"game_date={day}"
    if not sim_day_dir.exists():
        raise typer.Exit(code=2)

    resolved_sim_run_id = sim_run_id or _read_latest_run_id(sim_day_dir)
    sim_run_dir = _resolve_run_dir(sim_day_dir, resolved_sim_run_id)
    sim_proj_path = sim_run_dir / "projections.parquet"
    minutes_path = sim_run_dir / "minutes_matrix.parquet"

    if not sim_proj_path.exists():
        raise FileNotFoundError(f"Missing {sim_proj_path}")
    if not minutes_path.exists():
        raise FileNotFoundError(f"Missing {minutes_path}")

    sim_df = pd.read_parquet(sim_proj_path)
    minutes_df = pd.read_parquet(minutes_path)
    minutes_df.columns = [str(c) for c in minutes_df.columns]

    # Try to load unified projections for names/status (best-effort).
    proj_day_dir = root / "artifacts" / "projections" / day
    resolved_proj_run_id = projections_run_id or resolved_sim_run_id or _read_latest_run_id(proj_day_dir)
    unified_df = None
    if proj_day_dir.exists():
        run_dir = _resolve_run_dir(proj_day_dir, resolved_proj_run_id)
        parquet = run_dir / "projections.parquet"
        if parquet.exists():
            unified_df = pd.read_parquet(parquet)

    sim_df["player_id"] = sim_df["player_id"].astype(str)
    minutes_players = list(minutes_df.columns)
    sim_df = sim_df[sim_df["player_id"].isin(minutes_players)].copy()
    sim_df = sim_df.drop_duplicates("player_id", keep="last").reset_index(drop=True)

    mins = minutes_df.loc[:, sim_df["player_id"].tolist()].to_numpy(dtype=float, copy=False)  # (W, P)
    played = mins >= float(play_threshold_minutes)
    in_rot = mins >= float(rotation_threshold_minutes)

    n_worlds = mins.shape[0]
    played_counts = played.sum(axis=0).astype(float)
    rot_counts = in_rot.sum(axis=0).astype(float)

    p_played = played_counts / float(max(1, n_worlds))
    p_rot = rot_counts / float(max(1, n_worlds))

    mean_uncond = mins.mean(axis=0, dtype=float)
    sum_cond = (mins * played).sum(axis=0, dtype=float)
    mean_cond = np.where(played_counts > 0, sum_cond / played_counts, 0.0)

    # Pull inputs / diagnostics.
    play_prob_display = pd.to_numeric(sim_df.get("play_prob"), errors="coerce").fillna(np.nan).to_numpy(dtype=float)
    play_prob_used = pd.to_numeric(sim_df.get("play_prob_eff"), errors="coerce").fillna(np.nan).to_numpy(dtype=float)
    bench_zero_p_zero = pd.to_numeric(sim_df.get("bench_zero_p_zero"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    is_core = sim_df.get("rotation_lock")
    if is_core is None:
        is_core_arr = np.zeros(len(sim_df), dtype=bool)
    else:
        is_core_arr = is_core.astype(bool).to_numpy()

    p_play_expected = np.clip(play_prob_used * (1.0 - bench_zero_p_zero), 0.0, 1.0)
    check_expected = p_play_expected * mean_cond
    check_realized = p_played * mean_cond

    # Sample across the distribution by uncond minutes.
    order = np.argsort(mean_uncond)
    take = int(max(1, min(int(n_players), len(order))))
    pick = np.unique(np.round(np.linspace(0, len(order) - 1, take)).astype(int))
    idxs = order[pick]

    # Names (best-effort).
    name_map = {}
    if unified_df is not None and "player_id" in unified_df.columns:
        unified_df = unified_df.copy()
        unified_df["player_id"] = unified_df["player_id"].astype(str)
        if "player_name" in unified_df.columns:
            name_map = dict(zip(unified_df["player_id"], _as_str_series(unified_df, "player_name")))

    out = pd.DataFrame(
        {
            "player_id": sim_df["player_id"].to_numpy(dtype=object)[idxs],
            "name": [name_map.get(pid, "") for pid in sim_df["player_id"].to_numpy(dtype=object)[idxs]],
            "is_core": is_core_arr[idxs],
            "play_prob_display": play_prob_display[idxs],
            "play_prob_used": play_prob_used[idxs],
            "bench_zero_p_zero": bench_zero_p_zero[idxs],
            "p_play_expected": p_play_expected[idxs],
            "share_minutes_ge_playT": p_played[idxs],
            "share_minutes_ge_rotT": p_rot[idxs],
            "mean_minutes_uncond": mean_uncond[idxs],
            "mean_minutes_cond": mean_cond[idxs],
            "check_expected": check_expected[idxs],
            "check_realized": check_realized[idxs],
            "err_expected": (mean_uncond - check_expected)[idxs],
            "err_realized": (mean_uncond - check_realized)[idxs],
        }
    )

    with pd.option_context("display.max_rows", 200, "display.width", 220):
        typer.echo(f"[sim] run_dir={sim_run_dir}")
        typer.echo(f"[sim] worlds={n_worlds} players={len(sim_df)}")
        typer.echo(
            f"[thresholds] play_minutes>={float(play_threshold_minutes):.1f} "
            f"rotation_minutes>={float(rotation_threshold_minutes):.1f}"
        )
        typer.echo(out.to_string(index=False, float_format=lambda x: f"{x:0.3f}" if np.isfinite(x) else "nan"))


if __name__ == "__main__":
    app()
