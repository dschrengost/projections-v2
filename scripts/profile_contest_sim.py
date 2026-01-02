#!/usr/bin/env python3
"""Profiling harness for contest simulation payouts."""

from __future__ import annotations

import argparse
import cProfile
import datetime as dt
import io
import os
import platform
import pstats
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from projections.contest_sim.payout_generator import generate_payout_tiers, get_field_size, load_config
from projections.contest_sim.payouts import compute_expected_user_payouts_vectorized
from projections.contest_sim.weights import scale_integer_weights_to_target


def _format_bytes(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def _safe_mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return float(sum(items)) / float(len(items))


def _resolve_profile_path(profile_out: str | None, timestamp: str) -> Path | None:
    if not profile_out:
        return None
    path = Path(profile_out)
    if path.suffix:
        return path
    if profile_out.endswith(os.sep) or (path.exists() and path.is_dir()):
        return path / f"contest_sim_{timestamp}.prof"
    return path.with_suffix(".prof")


def _resolve_report_path(report_out: str | None, timestamp: str) -> Path:
    if report_out:
        return Path(report_out)
    report_dir = Path("artifacts") / "profiling"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"contest_sim_{timestamp}.md"


def _collect_rss() -> tuple[int | None, int | None]:
    rss_now = None
    rss_peak = None
    try:
        import psutil  # type: ignore

        process = psutil.Process()
        rss_now = int(process.memory_info().rss)
    except Exception:
        pass

    try:
        import resource

        ru_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            rss_peak = int(ru_max)
        else:
            rss_peak = int(ru_max) * 1024
    except Exception:
        pass

    return rss_now, rss_peak


def _build_inputs(
    *,
    rng: np.random.Generator,
    worlds: int,
    field_size: int,
    user_lineups: int,
    field_lineups: int | None,
    entry_fee: float,
    archetype: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[Any]]:
    user_weights = [1] * int(user_lineups)
    user_total_entries = int(sum(user_weights))

    target_field_entries = int(field_size - user_total_entries)
    if target_field_entries <= 0:
        raise ValueError("field_size must exceed total user entries")

    if field_lineups is None:
        default_field = max(user_lineups * 4, 1000)
        field_lineups = min(target_field_entries, default_field)

    field_lineups = int(field_lineups)
    if field_lineups <= 0:
        raise ValueError("field_lineups must be positive")

    base_field_weights = rng.integers(1, 4, size=field_lineups).tolist()
    scaled_field_weights = scale_integer_weights_to_target(
        base_field_weights,
        target_field_entries,
        min_weight=1 if target_field_entries >= field_lineups else 0,
    )

    user_scores = rng.normal(loc=100.0, scale=15.0, size=(user_lineups, worlds)).astype(np.float64)
    field_scores = rng.normal(loc=100.0, scale=15.0, size=(field_lineups, worlds)).astype(np.float64)

    payout_tiers = generate_payout_tiers(
        archetype_name=archetype,
        field_size=field_size,
        entry_fee=entry_fee,
        config=config,
    )

    return user_scores, field_scores, user_weights, scaled_field_weights, payout_tiers


def _summarize_profile(profile_path: Path, limit: int = 15) -> list[str]:
    stats = pstats.Stats(str(profile_path))
    stats.strip_dirs().sort_stats("cumulative")
    sio = io.StringIO()
    stats.stream = sio
    stats.print_stats(limit)
    return [line.rstrip("\n") for line in sio.getvalue().splitlines() if line.strip()]


def _line_profile(run_fn: Any) -> tuple[Any, str]:
    try:
        from line_profiler import LineProfiler  # type: ignore
    except Exception:
        return run_fn(), "line_profiler not installed"

    profiler = LineProfiler()
    profiler.add_function(compute_expected_user_payouts_vectorized)
    result = profiler.runcall(run_fn)
    sio = io.StringIO()
    profiler.print_stats(stream=sio)
    return result, sio.getvalue()


def main() -> int:
    config = load_config()
    defaults = config.get("defaults", {})
    sim_defaults = config.get("simulation", {})

    parser = argparse.ArgumentParser(description="Profile contest sim payout performance.")
    parser.add_argument("--worlds", type=int, default=int(sim_defaults.get("n_worlds", 10000)))
    parser.add_argument("--field-size", type=int, default=int(get_field_size(defaults.get("field_size_bucket", "5000"), config)))
    parser.add_argument("--user-lineups", type=int, default=150)
    parser.add_argument("--field-lineups", type=int, default=None)
    parser.add_argument("--entry-fee", type=float, default=float(defaults.get("entry_fee", 3.0)))
    parser.add_argument("--archetype", type=str, default=str(defaults.get("archetype", "GPP Standard (20% paid)")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--world-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--rank-mode",
        type=str,
        choices=("baseline", "combined_sort"),
        default=os.getenv("CONTEST_SIM_RANK_MODE", "baseline"),
        help="Ranking implementation (also configurable via CONTEST_SIM_RANK_MODE).",
    )
    parser.add_argument("--profile-out", type=str, default=None)
    parser.add_argument("--report-out", type=str, default=None)
    parser.add_argument("--trace-malloc", action="store_true", default=False)
    parser.add_argument("--fastpath", action="store_true", default=False)
    parser.add_argument("--line-profile", action="store_true", default=False)

    args = parser.parse_args()

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_path = _resolve_profile_path(args.profile_out, timestamp)
    report_path = _resolve_report_path(args.report_out, timestamp)

    rng = np.random.default_rng(args.seed)
    user_scores, field_scores, user_weights, field_weights, payout_tiers = _build_inputs(
        rng=rng,
        worlds=args.worlds,
        field_size=args.field_size,
        user_lineups=args.user_lineups,
        field_lineups=args.field_lineups,
        entry_fee=args.entry_fee,
        archetype=args.archetype,
        config=config,
    )

    timing: dict[str, float] | None = {}

    def _run_once() -> Any:
        return compute_expected_user_payouts_vectorized(
            user_scores=user_scores,
            field_scores=field_scores,
            user_weights=user_weights,
            field_weights=field_weights,
            payout_tiers=payout_tiers,
            workers=args.workers,
            compute_field_side=True,
            world_chunk_size=args.world_chunk_size,
            rank_mode=args.rank_mode,
            fastpath=args.fastpath,
            timing=timing,
        )

    def _run_all() -> Any:
        result = None
        for _ in range(int(args.repeat)):
            result = _run_once()
        return result

    tracemalloc_stats: list[str] = []
    if args.trace_malloc:
        import tracemalloc

        tracemalloc.start()

    run_times: list[float] = []
    result = None
    if profile_path is not None:
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(int(args.repeat)):
            t0 = perf_counter()
            result = _run_once()
            run_times.append(perf_counter() - t0)
        profiler.disable()
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(profile_path))
    else:
        for _ in range(int(args.repeat)):
            t0 = perf_counter()
            result = _run_once()
            run_times.append(perf_counter() - t0)
    if args.line_profile:
        _, line_stats = _line_profile(_run_all)
    else:
        line_stats = ""

    if args.trace_malloc:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")[:20]
        for stat in top_stats:
            tracemalloc_stats.append(str(stat))
        tracemalloc.stop()

    rss_now, rss_peak = _collect_rss()

    timing_total = float(timing.get("total", 0.0))
    argsort_time = float(timing.get("argsort", 0.0))
    searchsorted_time = float(timing.get("searchsorted", 0.0))
    payout_time = float(timing.get("payout_mapping", 0.0))

    denom = timing_total if timing_total > 0 else 1.0
    argsort_pct = 100.0 * argsort_time / denom
    searchsorted_pct = 100.0 * searchsorted_time / denom
    payout_pct = 100.0 * payout_time / denom
    other_pct = max(0.0, 100.0 - argsort_pct - searchsorted_pct - payout_pct)

    print("\nContest sim profiling summary")
    print("============================")
    print("metric | value")
    print(f"runs | {args.repeat}")
    print(f"total_runtime_s | {sum(run_times):.6f}")
    print(f"avg_runtime_s | {_safe_mean(run_times):.6f}")
    print(f"time_per_world_s | {_safe_mean(run_times) / max(args.worlds, 1):.9f}")
    print(f"argsort_pct | {argsort_pct:.2f}")
    print(f"searchsorted_pct | {searchsorted_pct:.2f}")
    print(f"payout_mapping_pct | {payout_pct:.2f}")
    print(f"other_pct | {other_pct:.2f}")
    print(f"rank_mode | {args.rank_mode}")
    if rss_peak is not None:
        print(f"peak_rss | {_format_bytes(rss_peak)}")
    elif rss_now is not None:
        print(f"rss_now | {_format_bytes(rss_now)}")

    profile_summary: list[str] = []
    if profile_path is not None:
        profile_summary = _summarize_profile(profile_path)

    report_lines = [
        "# Contest sim profiling report",
        "",
        f"timestamp: {timestamp}",
        "",
        "## Environment",
        f"- python: {platform.python_version()} ({platform.python_implementation()})",
        f"- numpy: {np.__version__}",
        "",
        "## Inputs",
        f"- worlds: {args.worlds}",
        f"- user_lineups: {args.user_lineups}",
        f"- field_lineups: {field_scores.shape[0]}",
        f"- field_size: {args.field_size}",
        f"- entry_fee: {args.entry_fee}",
        f"- archetype: {args.archetype}",
        f"- workers: {args.workers}",
        f"- world_chunk_size: {args.world_chunk_size}",
        f"- fastpath: {args.fastpath}",
        f"- rank_mode: {args.rank_mode}",
        f"- user_scores shape/dtype: {user_scores.shape} / {user_scores.dtype}",
        f"- field_scores shape/dtype: {field_scores.shape} / {field_scores.dtype}",
        f"- user_weights sum: {int(sum(user_weights))}",
        f"- field_weights sum: {int(sum(field_weights))}",
        "",
        "## Timing summary",
        f"- runs: {args.repeat}",
        f"- total_time: {sum(run_times):.6f}s",
        f"- avg_time: {_safe_mean(run_times):.6f}s",
        f"- time_per_world: {_safe_mean(run_times) / max(args.worlds, 1):.9f}s",
        "",
        "### update() breakdown (percent of update total)",
        f"- argsort: {argsort_pct:.2f}%",
        f"- searchsorted: {searchsorted_pct:.2f}%",
        f"- payout_mapping: {payout_pct:.2f}%",
        f"- other: {other_pct:.2f}%",
        "",
        "## Memory",
        f"- rss_now: {_format_bytes(rss_now)}",
        f"- rss_peak: {_format_bytes(rss_peak)}",
        "",
        "## cProfile hotspots",
    ]

    if profile_summary:
        report_lines.extend(["```", *profile_summary, "```"])
    else:
        report_lines.append("(cProfile not collected)")

    report_lines.extend([
        "",
        "## Allocation summary (tracemalloc)",
    ])
    if tracemalloc_stats:
        report_lines.extend(["```", *tracemalloc_stats, "```"])
    else:
        report_lines.append("(tracemalloc not collected)")

    report_lines.extend([
        "",
        "## py-spy (optional)",
        "Install py-spy separately, then run:",
        "```",
        "py-spy record -o artifacts/profiling/contest_sim.svg -- ",
        "  uv run python scripts/profile_contest_sim.py --worlds 10000 --field-size 5000 --user-lineups 150",
        "```",
    ])

    if line_stats:
        report_lines.extend([
            "",
            "## line_profiler",
            "```",
            line_stats.strip(),
            "```",
        ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"\nreport written: {report_path}")
    if profile_path is not None:
        print(f"profile written: {profile_path}")

    _ = result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
