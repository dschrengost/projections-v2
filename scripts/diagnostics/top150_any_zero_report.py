"""Report P(lineup has >=1 zero-score player) for top-K lineups per slate.

This is a lightweight diagnostic for "fragility" risk after enabling DNP=0 worlds.
It uses saved contest sim builds under:
  <DATA_ROOT>/builds/contest_sim/<DATE>/*.json

For each date, it picks a build (default: largest file), ranks lineups by select_score
(fallback: expected_value), and computes across worlds:
  p_any_zero(lineup) = mean_w[ any(player_score(w)==0) ]

Usage:
  uv run python scripts/diagnostics/top150_any_zero_report.py --start 2026-01-08 --end 2026-01-10
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from projections.contest_sim.contest_sim_service import load_worlds_matrix
from projections.paths import data_path

app = typer.Typer(add_completion=False)


def _iter_dates(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    dates: list[str] = []
    cur = start.normalize()
    end_n = end.normalize()
    while cur <= end_n:
        dates.append(cur.date().isoformat())
        cur += pd.Timedelta(days=1)
    return dates


def _select_build(build_dir: Path, *, mode: str) -> Path | None:
    files = [p for p in build_dir.glob("*.json") if p.is_file()]
    if not files:
        return None
    if mode == "latest":
        return max(files, key=lambda p: p.stat().st_mtime)
    if mode == "largest":
        return max(files, key=lambda p: p.stat().st_size)
    raise ValueError(f"Unknown build select mode: {mode}")


@dataclass(frozen=True)
class SlateRow:
    date: str
    build_id: str
    lineups_count: int
    sort_key: str
    p_any_zero_mean: float
    p_any_zero_p50: float
    p_any_zero_p90: float
    missing_player_lineups: int


def _compute_top_k_any_zero(
    *,
    worlds_matrix: np.ndarray,
    player_index: dict[str, int],
    results: list[dict],
    top_k: int,
) -> tuple[str, dict[str, float] | None, int]:
    if not results or worlds_matrix.size == 0:
        return "none", None, 0

    select_scores = np.asarray(
        [r.get("select_score") if r.get("select_score") is not None else float("-inf") for r in results],
        dtype=np.float64,
    )
    if np.isfinite(select_scores).any():
        order = np.argsort(-select_scores)
        sort_key = "select_score"
    else:
        evs = np.asarray([r.get("expected_value", float("-inf")) for r in results], dtype=np.float64)
        order = np.argsort(-evs)
        sort_key = "expected_value"

    k = min(int(top_k), int(len(results)))
    top_idx = order[:k]

    rates: list[float] = []
    missing_player_lineups = 0
    for i in top_idx:
        r = results[int(i)]
        pids = [str(pid).strip() for pid in (r.get("player_ids") or []) if str(pid).strip()]
        cols = [player_index.get(pid) for pid in pids]
        if any(c is None for c in cols):
            missing_player_lineups += 1
        cols_i = np.asarray([int(c) for c in cols if c is not None], dtype=np.int64)
        if cols_i.size == 0:
            continue
        sub = np.take(worlds_matrix, cols_i, axis=1)
        any_zero = np.any(sub == 0.0, axis=1)
        rates.append(float(np.mean(any_zero)))

    if not rates:
        return sort_key, None, missing_player_lineups
    payload = {
        "mean": float(np.mean(rates)),
        "p50": float(np.median(rates)),
        "p90": float(np.percentile(rates, 90)),
    }
    return sort_key, payload, missing_player_lineups


@app.command()
def main(
    start: str | None = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)."),
    end: str | None = typer.Option(None, "--end", help="End date (YYYY-MM-DD)."),
    date: list[str] = typer.Option([], "--date", help="Specific date(s); can repeat."),
    data_root: Path | None = typer.Option(None, "--data-root", help="Defaults to PROJECTIONS_DATA_ROOT."),
    top_k: int = typer.Option(150, "--top-k", help="Top K lineups to evaluate."),
    build_select: str = typer.Option(
        "largest",
        "--build-select",
        help="Which saved build to use per date: 'largest' or 'latest'.",
    ),
    use_saved_debug: bool = typer.Option(True, "--use-saved-debug/--no-saved-debug"),
) -> None:
    root = data_root or data_path()
    builds_root = root / "builds" / "contest_sim"

    if date:
        dates = [pd.Timestamp(d).date().isoformat() for d in date]
    else:
        if not start or not end:
            raise typer.BadParameter("Provide --date or both --start and --end.")
        dates = _iter_dates(pd.Timestamp(start), pd.Timestamp(end))

    rows: list[SlateRow] = []
    for d in dates:
        build_dir = builds_root / d
        build_path = _select_build(build_dir, mode=build_select) if build_dir.exists() else None
        if build_path is None:
            typer.echo(f"[any_zero] {d}: no contest_sim builds under {build_dir}; skipping", err=True)
            continue

        payload = json.loads(build_path.read_text(encoding="utf-8"))
        build_id = str(payload.get("build_id") or build_path.stem)
        results = payload.get("results") or []
        if not isinstance(results, list) or not results:
            typer.echo(f"[any_zero] {d}: build {build_id} has no results; skipping", err=True)
            continue

        # Fast path: use saved debug stat if present (newer builds).
        debug = ((payload.get("stats") or {}).get("debug") or {}) if isinstance(payload.get("stats"), dict) else {}
        saved = debug.get("top_k_p_any_zero_player") if use_saved_debug else None
        if isinstance(saved, dict) and {"mean", "p50", "p90"}.issubset(saved.keys()):
            rows.append(
                SlateRow(
                    date=d,
                    build_id=build_id,
                    lineups_count=int(payload.get("lineups_count") or len(results)),
                    sort_key=str(debug.get("top_k_sort_key") or "select_score"),
                    p_any_zero_mean=float(saved.get("mean", 0.0)),
                    p_any_zero_p50=float(saved.get("p50", 0.0)),
                    p_any_zero_p90=float(saved.get("p90", 0.0)),
                    missing_player_lineups=int(saved.get("missing_player_lineups") or 0),
                )
            )
            continue

        worlds, player_index = load_worlds_matrix(d, data_root=root)
        sort_key, stats, missing = _compute_top_k_any_zero(
            worlds_matrix=worlds,
            player_index=player_index,
            results=results,
            top_k=top_k,
        )
        if stats is None:
            typer.echo(f"[any_zero] {d}: failed to compute stats; skipping", err=True)
            continue

        rows.append(
            SlateRow(
                date=d,
                build_id=build_id,
                lineups_count=int(payload.get("lineups_count") or len(results)),
                sort_key=sort_key,
                p_any_zero_mean=float(stats["mean"]),
                p_any_zero_p50=float(stats["p50"]),
                p_any_zero_p90=float(stats["p90"]),
                missing_player_lineups=int(missing),
            )
        )

    if not rows:
        raise typer.Exit(code=2)

    df = pd.DataFrame([r.__dict__ for r in rows]).sort_values("date")
    typer.echo("\n## Top-K P(any zero-score player) per slate")
    typer.echo(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":  # pragma: no cover
    app()

