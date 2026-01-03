from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class AcceptanceMetrics:
    max_team_sum_dev: float
    nz_count_dist: dict[int, int]
    nz_distinct: int
    nz_modal_frac: float
    non_uniform_frac: float
    starter_std_p50_p10: float
    starter_std_p50_p50: float
    starter_std_p50_p90: float
    bench_std_p50_p10: float
    bench_std_p50_p50: float
    bench_std_p50_p90: float
    top5_sum_std: float
    top8_sum_std: float


def compute_team_game_metrics(
    df: pd.DataFrame,
    *,
    minutes_col: str = "minutes_p50",
    eps: float = 0.01,
) -> pd.DataFrame:
    """Compute per-team-game allocation diagnostics for a minutes parquet."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "team_id",
                "nz_count",
                "starter_nz",
                "bench_nz",
                "starter_std",
                "bench_std",
                "top5_sum",
                "top8_sum",
            ]
        )

    if minutes_col not in df.columns:
        raise ValueError(f"Missing minutes column: {minutes_col}")
    if not {"game_id", "team_id"}.issubset(df.columns):
        raise ValueError("Expected columns game_id and team_id")

    mins = pd.to_numeric(df[minutes_col], errors="coerce").fillna(0.0).astype(float)
    starter_flag = pd.to_numeric(df.get("starter_flag", 0), errors="coerce").fillna(0).astype(int)

    records: list[dict[str, float | int | str]] = []
    for (game_id, team_id), g in df.assign(_mins=mins, _starter=starter_flag).groupby(["game_id", "team_id"], sort=False):
        g_mins = pd.to_numeric(g["_mins"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        nz_mask = g_mins > float(eps)
        starter_mask = pd.to_numeric(g["_starter"], errors="coerce").fillna(0).astype(int).to_numpy() > 0

        starters = g_mins[nz_mask & starter_mask]
        bench = g_mins[nz_mask & ~starter_mask]

        starter_std = float(np.std(starters, ddof=0)) if starters.size >= 2 else 0.0
        bench_std = float(np.std(bench, ddof=0)) if bench.size >= 2 else 0.0

        sorted_mins = np.sort(g_mins)[::-1]
        records.append(
            {
                "game_id": int(game_id) if str(game_id).isdigit() else str(game_id),
                "team_id": int(team_id) if str(team_id).isdigit() else str(team_id),
                "nz_count": int(np.sum(nz_mask)),
                "starter_nz": int(np.sum(nz_mask & starter_mask)),
                "bench_nz": int(np.sum(nz_mask & ~starter_mask)),
                "starter_std": starter_std,
                "bench_std": bench_std,
                "top5_sum": float(sorted_mins[:5].sum()) if sorted_mins.size else 0.0,
                "top8_sum": float(sorted_mins[:8].sum()) if sorted_mins.size else 0.0,
            }
        )

    return pd.DataFrame.from_records(records)


def compute_acceptance_metrics(
    df: pd.DataFrame,
    *,
    minutes_col: str = "minutes_p50",
    eps: float = 0.01,
    std_threshold: float = 0.25,
) -> AcceptanceMetrics:
    if df.empty:
        return AcceptanceMetrics(
            max_team_sum_dev=float("nan"),
            nz_count_dist={},
            nz_distinct=0,
            nz_modal_frac=float("nan"),
            non_uniform_frac=float("nan"),
            starter_std_p50_p10=float("nan"),
            starter_std_p50_p50=float("nan"),
            starter_std_p50_p90=float("nan"),
            bench_std_p50_p10=float("nan"),
            bench_std_p50_p50=float("nan"),
            bench_std_p50_p90=float("nan"),
            top5_sum_std=float("nan"),
            top8_sum_std=float("nan"),
        )

    if minutes_col not in df.columns:
        raise ValueError(f"Missing minutes column: {minutes_col}")
    if not {"game_id", "team_id"}.issubset(df.columns):
        raise ValueError("Expected columns game_id and team_id")

    p50 = pd.to_numeric(df[minutes_col], errors="coerce").fillna(0.0)
    team_sums = p50.groupby([df["game_id"], df["team_id"]]).sum()
    max_team_sum_dev = float((team_sums - 240.0).abs().max())

    rows: list[dict[str, float]] = []
    nz_counts: list[int] = []
    top5_sums: list[float] = []
    top8_sums: list[float] = []

    for (_, _), g in df.groupby(["game_id", "team_id"], sort=False):
        mins = pd.to_numeric(g[minutes_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        nz_mask = mins > float(eps)
        nz_counts.append(int(np.sum(nz_mask)))
        top5_sums.append(float(np.sort(mins)[::-1][:5].sum()))
        top8_sums.append(float(np.sort(mins)[::-1][:8].sum()))

        starter_flag = pd.to_numeric(g.get("starter_flag", 0), errors="coerce").fillna(0).astype(int).to_numpy()
        starter_mask = starter_flag > 0
        starters = mins[nz_mask & starter_mask]
        bench = mins[nz_mask & ~starter_mask]
        starter_std = float(np.std(starters, ddof=0)) if starters.size >= 2 else 0.0
        bench_std = float(np.std(bench, ddof=0)) if bench.size >= 2 else 0.0
        rows.append({"starter_std": starter_std, "bench_std": bench_std})

    dist = Counter(nz_counts)
    nz_count_dist = {int(k): int(v) for k, v in sorted(dist.items())}
    nz_distinct = int(len(nz_count_dist))
    nz_modal_frac = float(max(dist.values()) / len(nz_counts)) if nz_counts else float("nan")

    std_df = pd.DataFrame(rows)
    non_uniform = (std_df["bench_std"] > float(std_threshold)) | (std_df["starter_std"] > float(std_threshold))
    non_uniform_frac = float(non_uniform.mean()) if not std_df.empty else float("nan")

    starter_q = std_df["starter_std"].quantile([0.1, 0.5, 0.9]).to_dict()
    bench_q = std_df["bench_std"].quantile([0.1, 0.5, 0.9]).to_dict()

    return AcceptanceMetrics(
        max_team_sum_dev=max_team_sum_dev,
        nz_count_dist=nz_count_dist,
        nz_distinct=nz_distinct,
        nz_modal_frac=nz_modal_frac,
        non_uniform_frac=non_uniform_frac,
        starter_std_p50_p10=float(starter_q.get(0.1, 0.0)),
        starter_std_p50_p50=float(starter_q.get(0.5, 0.0)),
        starter_std_p50_p90=float(starter_q.get(0.9, 0.0)),
        bench_std_p50_p10=float(bench_q.get(0.1, 0.0)),
        bench_std_p50_p50=float(bench_q.get(0.5, 0.0)),
        bench_std_p50_p90=float(bench_q.get(0.9, 0.0)),
        top5_sum_std=float(np.std(top5_sums, ddof=0)) if top5_sums else float("nan"),
        top8_sum_std=float(np.std(top8_sums, ddof=0)) if top8_sums else float("nan"),
    )


def _print_metrics(label: str, metrics: AcceptanceMetrics) -> None:
    payload = asdict(metrics)
    print(f"\n[{label}]")
    for key, value in payload.items():
        print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute minutes allocation acceptance metrics.")
    parser.add_argument("--path", type=Path, help="Path to minutes.parquet")
    parser.add_argument("--before", type=Path, help="Before minutes.parquet (optional)")
    parser.add_argument("--after", type=Path, help="After minutes.parquet (optional)")
    parser.add_argument("--minutes-col", default="minutes_p50", help="Minutes column (default: minutes_p50)")
    parser.add_argument("--eps", type=float, default=0.01, help="Zero threshold for nz counts (default: 0.01)")
    parser.add_argument("--std-threshold", type=float, default=0.25, help="Std threshold for non-uniform check (default: 0.25)")
    parser.add_argument(
        "--per-team-game",
        action="store_true",
        help="Print per (game_id, team_id) diagnostics table.",
    )
    args = parser.parse_args()

    if args.before and args.after:
        before_df = pd.read_parquet(args.before)
        after_df = pd.read_parquet(args.after)
        if args.per_team_game:
            print(f"\n[before team-games] {args.before}")
            before_table = compute_team_game_metrics(before_df, minutes_col=args.minutes_col, eps=args.eps)
            print(before_table.sort_values(["game_id", "team_id"]).to_string(index=False))
            print(f"\n[after team-games] {args.after}")
            after_table = compute_team_game_metrics(after_df, minutes_col=args.minutes_col, eps=args.eps)
            print(after_table.sort_values(["game_id", "team_id"]).to_string(index=False))
        before_metrics = compute_acceptance_metrics(
            before_df,
            minutes_col=args.minutes_col,
            eps=args.eps,
            std_threshold=args.std_threshold,
        )
        after_metrics = compute_acceptance_metrics(
            after_df,
            minutes_col=args.minutes_col,
            eps=args.eps,
            std_threshold=args.std_threshold,
        )
        _print_metrics("before", before_metrics)
        _print_metrics("after", after_metrics)
        return

    if args.path is None:
        raise SystemExit("Provide --path or (--before and --after).")

    df = pd.read_parquet(args.path)
    if args.per_team_game:
        table = compute_team_game_metrics(df, minutes_col=args.minutes_col, eps=args.eps)
        print(table.sort_values(["game_id", "team_id"]).to_string(index=False))
    metrics = compute_acceptance_metrics(df, minutes_col=args.minutes_col, eps=args.eps, std_threshold=args.std_threshold)
    _print_metrics(str(args.path), metrics)


if __name__ == "__main__":
    main()
