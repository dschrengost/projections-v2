"""Audit that dashboard/GameView projections match optimizer inputs.

This script compares:
- the `/api/minutes` payload (what the UI sees)
- the projections DataFrame used to build the optimizer pool

For a given (date, run_id), the canonical decision fields must match exactly.

Usage
-----
uv run python scripts/diagnostics/audit_projection_consistency.py --date 2026-01-29 --run-id 20260129T204959Z
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

from projections.api.minutes_api import create_app
from projections.api.optimizer_service import load_projections_for_date
from projections.projections_bundle import resolve_unified_projections_run

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class AuditResult:
    run_id: str
    n_api: int
    n_opt: int
    n_join: int
    max_abs_diffs: dict[str, float]
    mismatches: pd.DataFrame


def _safe_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").astype(float)


def _load_api_minutes_payload(
    *,
    game_date: str,
    run_id: str,
) -> pd.DataFrame:
    # In-process call keeps this runnable even when the systemd service isn't running.
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    resp = client.get("/api/minutes", params={"date": game_date, "run_id": run_id})
    resp.raise_for_status()
    payload: dict[str, Any] = resp.json()
    players = payload.get("players") or []
    df = pd.DataFrame(players)
    if df.empty:
        return df

    # Normalize ids for joining.
    if "player_id" in df.columns:
        df["player_id"] = df["player_id"].astype(str)
    return df


def audit_projection_consistency(
    *,
    game_date: str,
    run_id: str,
    data_root: Path | None,
    n_players: int | None,
) -> AuditResult:
    api_df = _load_api_minutes_payload(game_date=game_date, run_id=run_id)

    opt_df = load_projections_for_date(game_date, run_id=run_id, data_root=data_root)
    if "player_id" in opt_df.columns:
        opt_df = opt_df.copy()
        opt_df["player_id"] = opt_df["player_id"].astype(str)

    # Canonical fields that must match between UI and downstream consumers.
    fields = [
        "minutes_sim_p_active",
        "minutes_sim_uncond_mean",
        "minutes_sim_uncond_p50",
        "fpts_sim_uncond_mean",
    ]

    left = api_df[[c for c in ["player_id", *fields] if c in api_df.columns]].copy()
    right = opt_df[[c for c in ["player_id", *fields] if c in opt_df.columns]].copy()

    merged = left.merge(right, on="player_id", how="inner", suffixes=("__api", "__opt"))

    max_abs: dict[str, float] = {}
    rows = []
    for field in fields:
        a = _safe_float_series(merged, f"{field}__api")
        b = _safe_float_series(merged, f"{field}__opt")
        diff = (a - b).abs()
        max_abs[field] = float(diff.max()) if len(diff) else 0.0
        merged[f"absdiff__{field}"] = diff
        rows.append(diff)

    merged["absdiff__total"] = np.nan
    if rows:
        total = np.zeros(len(merged), dtype=float)
        for d in rows:
            total += np.nan_to_num(d.to_numpy(dtype=float), nan=0.0)
        merged["absdiff__total"] = total

    mismatches = merged.loc[merged["absdiff__total"] > 0].copy()
    mismatches = mismatches.sort_values("absdiff__total", ascending=False)
    if n_players is not None:
        mismatches = mismatches.head(int(max(0, n_players)))

    return AuditResult(
        run_id=run_id,
        n_api=int(len(api_df)),
        n_opt=int(len(opt_df)),
        n_join=int(len(merged)),
        max_abs_diffs=max_abs,
        mismatches=mismatches,
    )


@app.command()
def main(
    date: str = typer.Option(..., "--date", help="Slate date YYYY-MM-DD"),
    run_id: str | None = typer.Option(None, "--run-id", help="Unified projections run_id"),
    n_players: int | None = typer.Option(None, "--n-players", help="Show top N mismatches"),
    data_root: Path | None = typer.Option(None, "--data-root", help="Override PROJECTIONS_DATA_ROOT"),
) -> None:
    resolved = resolve_unified_projections_run(date, run_id=run_id, data_root=data_root)
    if resolved.run_id is None:
        raise typer.Exit(code=2)

    result = audit_projection_consistency(
        game_date=date,
        run_id=resolved.run_id,
        data_root=data_root,
        n_players=n_players,
    )

    typer.echo(f"[audit] date={date} run_id={result.run_id}")
    typer.echo(f"[audit] rows api={result.n_api} optimizer={result.n_opt} joined={result.n_join}")

    for field, val in result.max_abs_diffs.items():
        typer.echo(f"[audit] max_abs_diff {field} = {val:.10f}")

    if not result.mismatches.empty:
        cols = [
            "player_id",
            *[f"absdiff__{f}" for f in result.max_abs_diffs.keys()],
            "absdiff__total",
        ]
        cols = [c for c in cols if c in result.mismatches.columns]
        typer.echo("[audit] mismatches (top):")
        typer.echo(result.mismatches[cols].to_string(index=False))
        raise typer.Exit(code=1)

    typer.echo("[audit] OK: all canonical fields match")


if __name__ == "__main__":
    app()
