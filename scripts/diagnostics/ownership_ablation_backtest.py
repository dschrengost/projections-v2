"""Ownership + dupe penalty + select_score sensitivity backtest (read-only).

This script loads an existing slate's worlds_matrix.parquet and a saved contest_sim
build (candidate lineups). It recomputes lineup distribution metrics from worlds,
then re-ranks the same lineup set under different:
  - ownership modes: off | dupe_only | field_only | full
  - rank modes: current | tail_only | tail_times_dupe

It outputs per-date summaries + an aggregate summary across the date range, and
reports overlap (Jaccard) of selected top-K sets versus the baseline
ownership_mode=full + rank_mode=current.

Usage:
  uv run python scripts/diagnostics/ownership_ablation_backtest.py \
    --start 2026-01-01 --end 2026-01-10 --data-root /home/daniel/projections-data
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import typer

from projections.contest_sim.dupe_penalty import compute_batch_dupe_penalties
from projections.paths import data_path
from projections.pipeline import control_plane

app = typer.Typer(add_completion=False)


TAIL_WEIGHT_P90 = 0.6
TAIL_WEIGHT_UCV = 0.4


def _iter_dates(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    dates: list[str] = []
    cur = start.normalize()
    end_n = end.normalize()
    while cur <= end_n:
        dates.append(cur.date().isoformat())
        cur += pd.Timedelta(days=1)
    return dates


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_build(build_dir: Path, *, mode: str) -> Path | None:
    files = [p for p in build_dir.glob("*.json") if p.is_file()]
    if not files:
        return None
    mode_n = str(mode).strip().lower()
    if mode_n in {"latest", "most_recent"}:
        return max(files, key=lambda p: p.stat().st_mtime)
    if mode_n == "largest":
        return max(files, key=lambda p: p.stat().st_size)
    raise ValueError(f"Unknown build select mode: {mode}")


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


def _load_player_ownership(game_date: str, *, data_root: Path) -> dict[str, float]:
    """Load player_id -> pred_own_pct (percent) mapping. Empty dict if unavailable."""
    unified_root = data_root / "artifacts" / "projections" / game_date
    if unified_root.exists():
        rid, run_dir = _resolve_run_dir(unified_root, run_id=None)
        if run_dir is not None:
            proj_path = run_dir / "projections.parquet"
            if proj_path.exists():
                try:
                    df = pd.read_parquet(proj_path, columns=["player_id", "pred_own_pct"])
                except Exception:  # noqa: BLE001
                    df = pd.read_parquet(proj_path)
                if {"player_id", "pred_own_pct"}.issubset(df.columns):
                    own = df.dropna(subset=["pred_own_pct"]).copy()
                    own["player_id"] = own["player_id"].astype(str)
                    own["pred_own_pct"] = pd.to_numeric(own["pred_own_pct"], errors="coerce")
                    own = own.dropna(subset=["pred_own_pct"])
                    return dict(zip(own["player_id"], own["pred_own_pct"], strict=False))

    # Fall back to silver/ownership_predictions
    slate_dir = data_root / "silver" / "ownership_predictions" / game_date
    if slate_dir.exists():
        slate_files = [p for p in slate_dir.glob("*.parquet") if not p.name.endswith("_locked.parquet")]
        if slate_files:
            own_path = max(slate_files, key=lambda p: p.stat().st_size)
            df = pd.read_parquet(own_path)
            if {"player_id", "pred_own_pct"}.issubset(df.columns):
                own = df.dropna(subset=["pred_own_pct"]).copy()
                own["player_id"] = own["player_id"].astype(str)
                own["pred_own_pct"] = pd.to_numeric(own["pred_own_pct"], errors="coerce")
                own = own.dropna(subset=["pred_own_pct"])
                return dict(zip(own["player_id"], own["pred_own_pct"], strict=False))

    flat = data_root / "silver" / "ownership_predictions" / f"{game_date}.parquet"
    if flat.exists():
        df = pd.read_parquet(flat)
        if {"player_id", "pred_own_pct"}.issubset(df.columns):
            own = df.dropna(subset=["pred_own_pct"]).copy()
            own["player_id"] = own["player_id"].astype(str)
            own["pred_own_pct"] = pd.to_numeric(own["pred_own_pct"], errors="coerce")
            own = own.dropna(subset=["pred_own_pct"])
            return dict(zip(own["player_id"], own["pred_own_pct"], strict=False))

    return {}


def _load_worlds_matrix_parquet_only(
    game_date: str,
    *,
    data_root: Path,
    sim_run_id: str | None,
    required_player_ids: Iterable[str],
) -> tuple[np.ndarray, dict[str, int], str | None, Path]:
    sim_base = data_root / "artifacts" / "sim_v2" / "worlds_fpts_v2" / f"game_date={game_date}"
    if not sim_base.exists():
        raise FileNotFoundError(f"Missing sim_v2 base dir: {sim_base}")

    resolved_run_id, run_dir = _resolve_run_dir(sim_base, run_id=sim_run_id)
    matrix_path = (run_dir / "worlds_matrix.parquet") if run_dir is not None else None

    if matrix_path is None or not matrix_path.exists():
        # Some promoted nightly runs persist only projections.parquet (no matrix). For
        # ablations we require worlds_matrix.parquet, so fall back to another run dir
        # that has it when sim_run_id is not explicitly pinned.
        if sim_run_id:
            raise FileNotFoundError(
                f"worlds_matrix.parquet missing under explicit sim run_id={sim_run_id} at {sim_base}"
            )

        candidates: list[tuple[str, Path]] = []
        for rd in sorted([p for p in sim_base.glob("run=*") if p.is_dir()], reverse=True):
            mp = rd / "worlds_matrix.parquet"
            if mp.exists():
                rid = rd.name.split("=", 1)[1]
                candidates.append((rid, rd))
        if not candidates:
            raise FileNotFoundError(f"worlds_matrix.parquet missing under {sim_base} (no run dirs contain it)")
        resolved_run_id, run_dir = candidates[0]
        matrix_path = run_dir / "worlds_matrix.parquet"

    required = sorted({str(pid).strip() for pid in required_player_ids if str(pid).strip()})
    if not required:
        raise ValueError("required_player_ids is empty")

    # Read only needed columns; silently skip missing player ids (tracked by caller).
    try:
        import pyarrow.parquet as pq

        cols = set(pq.ParquetFile(matrix_path).schema.names)
    except Exception:  # noqa: BLE001
        cols = set(pd.read_parquet(matrix_path, engine="pyarrow").columns)

    keep = [pid for pid in required if pid in cols]
    if not keep:
        raise RuntimeError(f"None of the required player_ids are present in {matrix_path}")

    df = pd.read_parquet(matrix_path, columns=keep)
    player_index = {str(pid): i for i, pid in enumerate(df.columns)}
    worlds = df.to_numpy(dtype=np.float64, copy=False)
    return worlds, player_index, resolved_run_id, matrix_path


@dataclass(frozen=True)
class LineupMetrics:
    mean: float
    p90: float
    ucv90: float
    tail_score: float
    p_any_zero: float


def _compute_metrics_for_lineups(
    *,
    worlds_matrix: np.ndarray,
    player_index: dict[str, int],
    lineups: list[list[str]],
) -> tuple[list[LineupMetrics], int]:
    if worlds_matrix.size == 0:
        raise ValueError("worlds_matrix is empty")
    W = int(worlds_matrix.shape[0])
    missing_lineups = 0
    metrics: list[LineupMetrics] = []
    for lu in lineups:
        pids = [str(pid).strip() for pid in lu if str(pid).strip()]
        cols = [player_index.get(pid) for pid in pids]
        if any(c is None for c in cols):
            missing_lineups += 1
        cols_i = np.asarray([int(c) for c in cols if c is not None], dtype=np.int64)
        if cols_i.size == 0:
            metrics.append(LineupMetrics(mean=float("nan"), p90=float("nan"), ucv90=float("nan"), tail_score=float("nan"), p_any_zero=float("nan")))
            continue
        sub = np.take(worlds_matrix, cols_i, axis=1)
        totals = sub.sum(axis=1)
        mean = float(np.mean(totals))
        p90 = float(np.percentile(totals, 90))
        tail_mask = totals >= p90
        ucv90 = float(np.mean(totals[tail_mask])) if bool(np.any(tail_mask)) else p90
        tail_score = float(TAIL_WEIGHT_P90 * p90 + TAIL_WEIGHT_UCV * ucv90)
        p_any_zero = float(np.mean(np.any(sub == 0.0, axis=1)))
        metrics.append(LineupMetrics(mean=mean, p90=p90, ucv90=ucv90, tail_score=tail_score, p_any_zero=p_any_zero))
    if len(metrics) != len(lineups):
        raise RuntimeError("metrics length mismatch")
    if W < 1:
        raise RuntimeError("worlds must have W>=1")
    return metrics, int(missing_lineups)


def _compute_select_scores(
    *,
    metrics: list[LineupMetrics],
    dupe_penalties: np.ndarray,
    rank_mode: str,
) -> np.ndarray:
    mode = str(rank_mode).strip().lower()
    tail = np.asarray([m.tail_score for m in metrics], dtype=np.float64)
    mean = np.asarray([m.mean for m in metrics], dtype=np.float64)
    dupe = np.asarray(dupe_penalties, dtype=np.float64)
    if mode == "tail_only":
        return tail
    if mode == "tail_times_dupe":
        return tail * dupe
    if mode == "current":
        return tail - (1.0 - dupe) * mean
    raise ValueError(f"Invalid rank_mode: {rank_mode!r}")


def _canonical_lineup_key(lineup: list[str]) -> tuple[str, ...]:
    return tuple(sorted(str(pid).strip() for pid in lineup if str(pid).strip()))


def _jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return 0.0 if union == 0 else float(inter / union)


def _normalize_ownership_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    allowed = {"off", "dupe_only", "field_only", "full"}
    if m not in allowed:
        raise ValueError(f"Invalid ownership_mode: {mode!r} (allowed: {sorted(allowed)})")
    return m


def _parse_csv_list(raw: str) -> list[str]:
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    return items


@app.command()
def main(
    start: str = typer.Option(..., "--start", help="Start date (YYYY-MM-DD)."),
    end: str = typer.Option(..., "--end", help="End date (YYYY-MM-DD)."),
    data_root: Path | None = typer.Option(None, "--data-root", help="Defaults to PROJECTIONS_DATA_ROOT."),
    ownership_modes: str = typer.Option(
        "off,dupe_only,field_only,full",
        "--ownership-modes",
        help="Comma-separated: off,dupe_only,field_only,full",
    ),
    rank_modes: str = typer.Option(
        "current,tail_only,tail_times_dupe",
        "--rank-modes",
        help="Comma-separated: current,tail_only,tail_times_dupe",
    ),
    top_k: int = typer.Option(150, "--top-k", min=1, help="Top K lineups to select."),
    build_select: str = typer.Option(
        "largest",
        "--build-select",
        help="Which saved contest_sim build to use per date: largest|latest|most_recent",
    ),
    sim_run_id: str | None = typer.Option(None, "--sim-run-id", help="Optional sim run_id override."),
    worlds_sample: int | None = typer.Option(
        None,
        "--worlds-sample",
        min=100,
        help="Optional subsample of worlds for speed (uses fixed seed).",
    ),
    seed: int = typer.Option(42, "--seed", help="RNG seed for worlds subsample."),
    output_csv: Path | None = typer.Option(None, "--output-csv", help="Write per-date results CSV."),
    output_md: Path | None = typer.Option(None, "--output-md", help="Write aggregate markdown summary."),
) -> None:
    root = data_root or data_path()
    dates = _iter_dates(pd.Timestamp(start), pd.Timestamp(end))
    builds_root = root / "builds" / "contest_sim"

    own_modes = [_normalize_ownership_mode(m) for m in _parse_csv_list(ownership_modes)]
    rk_modes = _parse_csv_list(rank_modes)

    rows: list[dict[str, object]] = []
    skipped = 0

    for day in dates:
        build_dir = builds_root / day
        build_path = _select_build(build_dir, mode=build_select) if build_dir.exists() else None
        if build_path is None:
            skipped += 1
            typer.echo(f"[ablation] {day}: no contest_sim builds under {build_dir}; skipping", err=True)
            continue

        payload = _read_json(build_path)
        request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
        cfg = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        field_size = int(cfg.get("field_size") or request.get("field_size_override") or 5000)
        entry_max = int(request.get("entry_max") or 150)

        lineups = payload.get("lineups") if isinstance(payload.get("lineups"), list) else None
        if not lineups:
            # Fallback: derive from results list.
            results = payload.get("results") if isinstance(payload.get("results"), list) else []
            lineups = [r.get("player_ids") for r in results if isinstance(r, dict) and isinstance(r.get("player_ids"), list)]
        lineups = [[str(pid).strip() for pid in lu if str(pid).strip()] for lu in (lineups or [])]
        if not lineups:
            skipped += 1
            typer.echo(f"[ablation] {day}: build {build_path.name} has no lineups; skipping", err=True)
            continue

        union_pids = sorted({pid for lu in lineups for pid in lu})
        try:
            worlds, player_index, resolved_sim_run, matrix_path = _load_worlds_matrix_parquet_only(
                day, data_root=root, sim_run_id=sim_run_id, required_player_ids=union_pids
            )
        except Exception as exc:
            skipped += 1
            typer.echo(f"[ablation] {day}: worlds load failed ({exc}); skipping", err=True)
            continue

        if worlds_sample is not None and int(worlds_sample) < int(worlds.shape[0]):
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(worlds.shape[0], size=int(worlds_sample), replace=False)
            idx.sort()
            worlds = worlds[idx, :]

        metrics, missing_lineups = _compute_metrics_for_lineups(worlds_matrix=worlds, player_index=player_index, lineups=lineups)

        own_map = _load_player_ownership(day, data_root=root)
        has_own = bool(own_map)

        dupe_full = np.ones(len(lineups), dtype=np.float64)
        if has_own:
            dupe_full = np.asarray(
                compute_batch_dupe_penalties(lineups=lineups, player_ownership=own_map, field_size=field_size, entry_max=entry_max),
                dtype=np.float64,
            )

        # Baseline selection set for overlap comparisons.
        baseline_key = ("full", "current")
        selected_sets: dict[tuple[str, str], set[tuple[str, ...]]] = {}

        for om in own_modes:
            dupe = dupe_full if (om in {"full", "dupe_only"} and has_own) else np.ones(len(lineups), dtype=np.float64)
            for rm in rk_modes:
                scores = _compute_select_scores(metrics=metrics, dupe_penalties=dupe, rank_mode=rm)
                order = np.argsort(-scores)
                k = min(int(top_k), int(len(lineups)))
                top_idx = order[:k]
                selected = { _canonical_lineup_key(lineups[int(i)]) for i in top_idx }
                selected_sets[(om, rm)] = selected

        baseline = selected_sets.get(baseline_key, set())

        for om in own_modes:
            dupe = dupe_full if (om in {"full", "dupe_only"} and has_own) else np.ones(len(lineups), dtype=np.float64)
            for rm in rk_modes:
                scores = _compute_select_scores(metrics=metrics, dupe_penalties=dupe, rank_mode=rm)
                order = np.argsort(-scores)
                k = min(int(top_k), int(len(lineups)))
                top_idx = order[:k]

                sel_metrics = [metrics[int(i)] for i in top_idx]
                sel_dupe = dupe[top_idx]
                p_any_zero = np.asarray([m.p_any_zero for m in sel_metrics], dtype=np.float64)
                means = np.asarray([m.mean for m in sel_metrics], dtype=np.float64)
                tails = np.asarray([m.tail_score for m in sel_metrics], dtype=np.float64)

                row: dict[str, object] = {
                    "date": day,
                    "build": build_path.name,
                    "sim_run_id": resolved_sim_run,
                    "worlds_path": str(matrix_path),
                    "lineups_n": int(len(lineups)),
                    "missing_lineups": int(missing_lineups),
                    "worlds_n": int(worlds.shape[0]),
                    "ownership_mode": om,
                    "rank_mode": rm,
                    "top_k": int(k),
                    "has_ownership": bool(has_own),
                    "dupe_penalty_mean": float(np.mean(sel_dupe)) if sel_dupe.size else float("nan"),
                    "dupe_penalty_p10": float(np.percentile(sel_dupe, 10)) if sel_dupe.size else float("nan"),
                    "dupe_penalty_p50": float(np.percentile(sel_dupe, 50)) if sel_dupe.size else float("nan"),
                    "dupe_penalty_p90": float(np.percentile(sel_dupe, 90)) if sel_dupe.size else float("nan"),
                    "mean_mean": float(np.mean(means)) if means.size else float("nan"),
                    "tail_score_mean": float(np.mean(tails)) if tails.size else float("nan"),
                    "p_any_zero_mean": float(np.mean(p_any_zero)) if p_any_zero.size else float("nan"),
                    "p_any_zero_p50": float(np.percentile(p_any_zero, 50)) if p_any_zero.size else float("nan"),
                    "p_any_zero_p90": float(np.percentile(p_any_zero, 90)) if p_any_zero.size else float("nan"),
                    "jaccard_vs_full_current": float(_jaccard(selected_sets.get((om, rm), set()), baseline)),
                }
                rows.append(row)

    if not rows:
        raise typer.Exit(code=2)

    df = pd.DataFrame(rows).sort_values(["date", "ownership_mode", "rank_mode"])

    out_csv = output_csv or Path(f"/tmp/ownership_ablation_backtest_{start}_to_{end}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    typer.echo(f"\n[ablation] wrote per-date CSV: {out_csv}")
    typer.echo(f"[ablation] dates={len(dates)} rows={len(df)} skipped_dates={skipped}")

    # Aggregate summary (mean over dates)
    agg = (
        df.groupby(["ownership_mode", "rank_mode"], as_index=False)
        .agg(
            dates=("date", "nunique"),
            mean_mean=("mean_mean", "mean"),
            tail_score_mean=("tail_score_mean", "mean"),
            dupe_penalty_mean=("dupe_penalty_mean", "mean"),
            p_any_zero_mean=("p_any_zero_mean", "mean"),
            jaccard_vs_full_current=("jaccard_vs_full_current", "mean"),
        )
        .sort_values(["ownership_mode", "rank_mode"])
    )

    out_md = output_md or Path(f"/tmp/ownership_ablation_backtest_{start}_to_{end}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        "\n".join(
            [
                "# Ownership ablation summary",
                "",
                f"- Dates: {start} to {end} (skipped {skipped})",
                f"- Top-K: {int(top_k)}",
                "",
                "## Aggregate (mean over dates)",
                "",
                "```",
                agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    typer.echo(f"[ablation] wrote aggregate markdown: {out_md}")


if __name__ == "__main__":  # pragma: no cover
    app()
