"""Audit that sim/selection wiring uses DNP=0 semantics in production artifacts.

This script is intentionally lightweight and read-only. It verifies, for a given slate date:
- Which sim profile was used (from sim_manifest.json).
- Whether play_prob masking (Bernoulli active sampling) is enabled.
- Whether contest sim will load worlds_matrix.parquet (preferred) vs world=*.parquet fallback.
- Whether unified projections contain *_uncond columns and which columns optimizer would pick.

Usage:
  uv run python scripts/diagnostics/audit_production_wiring.py --date 2026-01-16
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from projections.api.optimizer_service import load_projections_for_date
from projections.paths import data_path
from projections.pipeline import control_plane
from projections.sim_v2.config import load_sim_v2_profile

app = typer.Typer(add_completion=False)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_run_dir(base_dir: Path, *, run_id: str | None) -> tuple[str | None, Path | None]:
    if run_id:
        candidate = base_dir / f"run={run_id}"
        if candidate.exists():
            return run_id, candidate
        return run_id, None

    promoted = control_plane.read_promoted_run_id(base_dir)
    if promoted:
        candidate = base_dir / f"run={promoted}"
        if candidate.exists():
            return promoted, candidate

    if control_plane.allow_unpromoted_run_reads():
        run_dirs = sorted(
            [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run=")],
            reverse=True,
        )
        if run_dirs:
            rid = run_dirs[0].name.split("=", 1)[1]
            return rid, run_dirs[0]

    return None, None


def _parquet_shape_fast(path: Path) -> tuple[int | None, int | None]:
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        meta = pf.metadata
        if meta is None:
            return None, None
        return int(meta.num_rows), int(meta.num_columns)
    except Exception:
        return None, None


@app.command()
def main(
    date: str = typer.Option(..., "--date", help="Slate date (YYYY-MM-DD)."),
    data_root: Path | None = typer.Option(None, "--data-root", help="Defaults to PROJECTIONS_DATA_ROOT."),
    sim_run_id: str | None = typer.Option(None, "--sim-run-id", help="Optional sim run_id override."),
    projections_run_id: str | None = typer.Option(
        None,
        "--projections-run-id",
        help="Optional unified projections run_id override (defaults to promoted pointer).",
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    root = data_root or data_path()

    typer.echo(f"[audit] data_root={root}")
    typer.echo(f"[audit] date={date}")

    # -------------------- sim_v2 worlds --------------------
    sim_base = root / "artifacts" / "sim_v2" / "worlds_fpts_v2" / f"game_date={date}"
    if not sim_base.exists():
        raise typer.Exit(code=2)

    sim_rid, sim_dir = _resolve_run_dir(sim_base, run_id=sim_run_id)
    if sim_dir is None:
        typer.echo(f"[audit] sim_v2: could not resolve run dir under {sim_base}", err=True)
        raise typer.Exit(code=2)
    typer.echo(f"[audit] sim_v2: run_id={sim_rid} dir={sim_dir}")

    sim_manifest_path = sim_dir / "sim_manifest.json"
    if sim_manifest_path.exists():
        manifest = _read_json(sim_manifest_path)
        profile = str(manifest.get("profile") or manifest.get("sim_profile") or "unknown")
        play_prob_masking = manifest.get("play_prob_masking")
        typer.echo(f"[audit] sim_v2: profile={profile} play_prob_masking={play_prob_masking}")

        try:
            profile_cfg = load_sim_v2_profile(profile=profile)
            typer.echo(
                f"[audit] sim_v2: config.use_play_prob_masking={profile_cfg.use_play_prob_masking} "
                f"(min_play_prob={profile_cfg.min_play_prob})"
            )
        except Exception as exc:
            typer.echo(f"[audit] sim_v2: warning: failed to load profile config ({exc})", err=True)
    else:
        typer.echo(f"[audit] sim_v2: missing {sim_manifest_path}", err=True)

    matrix_path = sim_dir / "worlds_matrix.parquet"
    if matrix_path.exists():
        n_worlds, n_players = _parquet_shape_fast(matrix_path)
        size_mb = matrix_path.stat().st_size / (1024 * 1024)
        typer.echo(
            f"[audit] sim_v2: worlds_matrix.parquet present ({size_mb:.1f} MB, shape={n_worlds}x{n_players})"
        )
    else:
        world_files = sorted(sim_dir.glob("world=*.parquet"))
        typer.echo(
            f"[audit] sim_v2: worlds_matrix.parquet missing; fallback world_files={len(world_files)}",
            err=True,
        )

    # -------------------- unified projections --------------------
    proj_base = root / "artifacts" / "projections" / date
    proj_rid, proj_dir = _resolve_run_dir(proj_base, run_id=projections_run_id)
    if proj_dir is None:
        typer.echo(f"[audit] projections: could not resolve run dir under {proj_base}", err=True)
        raise typer.Exit(code=2)
    typer.echo(f"[audit] projections: run_id={proj_rid} dir={proj_dir}")

    proj_path = proj_dir / "projections.parquet"
    if proj_path.exists():
        df = pd.read_parquet(proj_path)
        uncond_cols = sorted([c for c in df.columns if c.endswith("_uncond")])
        typer.echo(
            f"[audit] projections: rows={len(df)} cols={len(df.columns)} uncond_cols={len(uncond_cols)}"
        )
        for key in ("dk_fpts_mean_uncond", "dk_fpts_std_uncond", "dk_fpts_p90_uncond", "minutes_sim_mean_uncond"):
            typer.echo(f"[audit] projections: has {key}={key in df.columns}")
    else:
        typer.echo(f"[audit] projections: missing {proj_path}", err=True)

    # -------------------- optimizer column selection (static) --------------------
    loaded = load_projections_for_date(date, run_id=proj_rid, data_root=root)
    proj_candidates = [
        "sim_dk_fpts_mean_uncond",
        "dk_fpts_mean_uncond",
        "sim_dk_fpts_mean",
        "dk_fpts_mean",
        "proj_fpts",
        "fpts_mean",
        "proj",
    ]
    minutes_candidates = [
        "sim_minutes_sim_mean_uncond",
        "minutes_sim_mean_uncond",
        "sim_minutes_sim_mean",
        "minutes_sim_mean",
        "sim_minutes_sim_p50_uncond",
        "minutes_sim_p50_uncond",
        "sim_minutes_sim_p50",
        "minutes_sim_p50",
        "minutes_p50",
        "minutes",
        "minutes_pred",
    ]
    stddev_candidates = [
        "sim_dk_fpts_std_uncond",
        "dk_fpts_std_uncond",
        "sim_dk_fpts_std",
        "stddev",
        "fpts_std",
    ]
    p90_candidates = [
        "sim_dk_fpts_p90_uncond",
        "dk_fpts_p90_uncond",
        "sim_dk_fpts_p90",
        "dk_fpts_p90",
        "fpts_p90",
    ]

    def _pick(candidates: list[str]) -> str | None:
        for c in candidates:
            if c in loaded.columns:
                return c
        return None

    typer.echo(f"[audit] optimizer: proj_col={_pick(proj_candidates)}")
    typer.echo(f"[audit] optimizer: minutes_col={_pick(minutes_candidates)}")
    typer.echo(f"[audit] optimizer: stddev_col={_pick(stddev_candidates)}")
    typer.echo(f"[audit] optimizer: p90_col={_pick(p90_candidates)}")


if __name__ == "__main__":  # pragma: no cover
    app()

