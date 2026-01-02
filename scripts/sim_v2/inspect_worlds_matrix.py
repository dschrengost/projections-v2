"""CLI helpers for inspecting tail outcomes from worlds_matrix.parquet.

Usage examples:

  # Show a player's top tail worlds and correlated teammates/opponents
  uv run python scripts/sim_v2/inspect_worlds_matrix.py tails 2025-12-26 203081 --top-k 25

  # Inspect a specific world (slice within the player's game)
  uv run python scripts/sim_v2/inspect_worlds_matrix.py world 2025-12-26 1234 --player-id 203081 --top-n 20
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from projections.paths import get_data_root

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class SimRunPaths:
    base_dir: Path
    run_dir: Path
    worlds_matrix_path: Path
    projections_path: Path | None


def _read_latest_run_id(day_dir: Path) -> str | None:
    pointer = day_dir / "latest_run.json"
    if not pointer.exists():
        return None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except Exception:
        return None
    run_id = payload.get("run_id")
    if not run_id:
        return None
    return str(run_id)


def _resolve_run_dir(day_dir: Path, run_id: str | None) -> Path:
    if run_id:
        candidate = day_dir / f"run={run_id}"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"run_id={run_id!r} not found under {day_dir}")

    latest = _read_latest_run_id(day_dir)
    if latest:
        candidate = day_dir / f"run={latest}"
        if candidate.exists():
            return candidate

    run_dirs = sorted(
        [p for p in day_dir.iterdir() if p.is_dir() and p.name.startswith("run=")],
        reverse=True,
    )
    if run_dirs:
        return run_dirs[0]
    return day_dir


def resolve_sim_paths(
    game_date: str,
    *,
    data_root: Path | None,
    worlds_root: Path | None,
    run_id: str | None,
) -> SimRunPaths:
    root = data_root or get_data_root()
    day_dir = (worlds_root or (root / "artifacts" / "sim_v2" / "worlds_fpts_v2")) / f"game_date={game_date}"
    if not day_dir.exists():
        raise FileNotFoundError(f"No sim worlds found for {game_date} under {day_dir}")

    run_dir = _resolve_run_dir(day_dir, run_id)
    worlds_matrix_path = run_dir / "worlds_matrix.parquet"
    if not worlds_matrix_path.exists():
        raise FileNotFoundError(f"Missing {worlds_matrix_path}")

    projections_path = run_dir / "projections.parquet"
    if not projections_path.exists():
        projections_path = None

    return SimRunPaths(
        base_dir=day_dir,
        run_dir=run_dir,
        worlds_matrix_path=worlds_matrix_path,
        projections_path=projections_path,
    )


def _load_projections(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    df = pd.read_parquet(path)
    if "player_id" in df.columns:
        df["player_id"] = df["player_id"].astype(str)
    for col in ("game_id", "team_id"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fmt_float(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.2f}"


def _corrcoef(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size != y.size:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_var = float(np.var(x))
    y_var = float(np.var(y))
    if x_var <= 0.0 or y_var <= 0.0:
        return None
    cov = float(np.mean((x - x.mean()) * (y - y.mean())))
    return cov / float(np.sqrt(x_var * y_var))


@app.command("tails")
def tails(
    game_date: str = typer.Argument(..., help="YYYY-MM-DD"),
    player_id: str = typer.Argument(..., help="Player id (matches worlds_matrix column name)."),
    top_k: int = typer.Option(20, "--top-k", help="Number of top worlds to print."),
    corr_top_n: int = typer.Option(8, "--corr-top-n", help="How many correlations to show per sign."),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional sim run_id (defaults to latest_run.json)."),
    data_root: Path | None = typer.Option(None, "--data-root", help="Optional override for data root."),
    worlds_root: Path | None = typer.Option(
        None,
        "--worlds-root",
        help="Optional override for worlds root (dir containing game_date=... folders).",
    ),
) -> None:
    paths = resolve_sim_paths(game_date, data_root=data_root, worlds_root=worlds_root, run_id=run_id)
    typer.echo(f"[inspect] run_dir={paths.run_dir}")
    worlds_df = pd.read_parquet(paths.worlds_matrix_path)

    pid = str(player_id)
    if pid not in worlds_df.columns:
        raise typer.BadParameter(f"player_id={pid} not present in worlds_matrix columns")

    values = pd.to_numeric(worlds_df[pid], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n_worlds = int(values.size)
    quantiles = {
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }
    typer.echo(
        "[inspect] "
        + " ".join([f"{k}={_fmt_float(v)}" for k, v in quantiles.items()])
        + f" n_worlds={n_worlds}"
    )

    proj_df = _load_projections(paths.projections_path)
    game_id = team_id = None
    teammates: list[str] = []
    opponents: list[str] = []
    if proj_df is not None and {"player_id", "game_id", "team_id"}.issubset(proj_df.columns):
        row = proj_df.loc[proj_df["player_id"] == pid]
        if not row.empty:
            game_id = int(row["game_id"].iloc[0])
            team_id = int(row["team_id"].iloc[0])
            game_df = proj_df.loc[proj_df["game_id"] == game_id].copy()
            team_df = game_df.loc[game_df["team_id"] == team_id]
            opp_df = game_df.loc[game_df["team_id"] != team_id]
            teammates = [p for p in team_df["player_id"].astype(str).tolist() if p in worlds_df.columns]
            opponents = [p for p in opp_df["player_id"].astype(str).tolist() if p in worlds_df.columns]

    top_k_eff = max(1, min(int(top_k), n_worlds))
    top = worlds_df[pid].nlargest(top_k_eff)
    world_idxs = top.index.to_numpy(dtype=int)

    tail_payload: dict[str, np.ndarray] = {
        "world_idx": world_idxs,
        "player_fpts": top.to_numpy(dtype=float),
    }
    if teammates and opponents:
        team_sum = worlds_df.loc[world_idxs, teammates].sum(axis=1).to_numpy(dtype=float)
        opp_sum = worlds_df.loc[world_idxs, opponents].sum(axis=1).to_numpy(dtype=float)
        tail_payload["team_fpts"] = team_sum
        tail_payload["opp_fpts"] = opp_sum
        tail_payload["game_fpts"] = team_sum + opp_sum

    tail_df = pd.DataFrame(tail_payload).sort_values("player_fpts", ascending=False)
    typer.echo(tail_df.to_string(index=False))

    if game_id is None or team_id is None:
        typer.echo("[inspect] projections.parquet missing or player not found; skipping correlations.")
        return

    game_cols = teammates + opponents
    x = pd.to_numeric(worlds_df[pid], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    corrs: list[tuple[str, float]] = []
    for other in game_cols:
        if other == pid:
            continue
        y = pd.to_numeric(worlds_df[other], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        c = _corrcoef(x, y)
        if c is None:
            continue
        corrs.append((other, float(c)))

    if not corrs:
        typer.echo("[inspect] correlations: none (zero variance columns?)")
        return

    corr_df = pd.DataFrame(corrs, columns=["player_id", "corr"]).sort_values("corr", ascending=False)
    pos = corr_df.head(max(0, int(corr_top_n)))
    neg = corr_df.tail(max(0, int(corr_top_n))).sort_values("corr", ascending=True)
    typer.echo("[inspect] correlations (same game):")
    typer.echo("  most positive:\n" + pos.to_string(index=False))
    typer.echo("  most negative:\n" + neg.to_string(index=False))


@app.command("world")
def world(
    game_date: str = typer.Argument(..., help="YYYY-MM-DD"),
    world_idx: int = typer.Argument(..., help="0-based row index into worlds_matrix"),
    player_id: str | None = typer.Option(
        None, "--player-id", help="Optional player id; if set, slices the world to that player's game."
    ),
    top_n: int = typer.Option(20, "--top-n", help="How many players to show in this world slice."),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional sim run_id (defaults to latest_run.json)."),
    data_root: Path | None = typer.Option(None, "--data-root", help="Optional override for data root."),
    worlds_root: Path | None = typer.Option(
        None,
        "--worlds-root",
        help="Optional override for worlds root (dir containing game_date=... folders).",
    ),
) -> None:
    paths = resolve_sim_paths(game_date, data_root=data_root, worlds_root=worlds_root, run_id=run_id)
    typer.echo(f"[inspect] run_dir={paths.run_dir}")
    worlds_df = pd.read_parquet(paths.worlds_matrix_path)
    if world_idx < 0 or world_idx >= len(worlds_df):
        raise typer.BadParameter(f"world_idx out of range (0..{len(worlds_df)-1})")

    proj_df = _load_projections(paths.projections_path)
    row = worlds_df.iloc[int(world_idx)]

    if player_id is None or proj_df is None or {"player_id", "game_id"}.issubset(proj_df.columns) is False:
        top = row.sort_values(ascending=False).head(int(top_n))
        out_df = pd.DataFrame({"player_id": top.index.astype(str), "dk_fpts_world": top.to_numpy(dtype=float)})
        typer.echo(out_df.to_string(index=False))
        return

    pid = str(player_id)
    sel = proj_df.loc[proj_df["player_id"] == pid]
    if sel.empty:
        raise typer.BadParameter(f"player_id={pid} not present in projections.parquet")
    game_id = int(sel["game_id"].iloc[0])

    game_df = proj_df.loc[proj_df["game_id"] == game_id].copy()
    game_pids = [p for p in game_df["player_id"].astype(str).tolist() if p in worlds_df.columns]
    if not game_pids:
        typer.echo("[inspect] no players in that game found in worlds_matrix columns")
        return

    slice_df = game_df.loc[game_df["player_id"].isin(game_pids), ["player_id"] + [c for c in ("team_id",) if c in game_df.columns]]
    slice_df = slice_df.copy()
    slice_df["dk_fpts_world"] = slice_df["player_id"].map(lambda p: float(row.get(str(p), 0.0)))
    slice_df = slice_df.sort_values("dk_fpts_world", ascending=False).head(int(top_n))

    typer.echo(f"[inspect] world_idx={world_idx} game_id={game_id}")
    typer.echo(slice_df.to_string(index=False))


if __name__ == "__main__":
    app()

