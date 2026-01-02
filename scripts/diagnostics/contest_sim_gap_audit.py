"""Diagnostics for understanding EV vs projection gaps in contest simulation.

This script loads a saved contest-sim build, re-scores the user/field lineups
against sim worlds, and reports how the "top EV" lineups differ from the "top
mean" lineups under different payout curves and dupe-penalty settings.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from projections.artifacts.unified_projections import resolve_unified_run_dir
from projections.contest_sim.contest_sim_service import load_worlds_matrix, score_lineups
from projections.contest_sim.dupe_penalty import compute_batch_dupe_penalties
from projections.contest_sim.field_library import load_field_library
from projections.contest_sim.payout_generator import generate_payout_tiers, load_config
from projections.contest_sim.payouts import compute_expected_user_payouts_vectorized
from projections.contest_sim.scoring_models import PayoutTier
from projections.contest_sim.weights import scale_integer_weights_to_target
from projections.paths import data_path


@dataclass(frozen=True)
class _BuildInputs:
    build_path: Path
    game_date: str
    user_lineups: list[list[str]]
    user_weights: list[int]
    field_size: int
    entry_fee: float
    rake: float
    field_library_path: Path | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-date", required=True, help="Slate date YYYY-MM-DD")
    parser.add_argument("--build-id", help="Saved contest-sim build UUID (under builds/contest_sim/<date>/)")
    parser.add_argument("--build-path", type=Path, help="Path to saved contest-sim build JSON")
    parser.add_argument("--run-id", default=None, help="Optional run_id for sim_v2 worlds and unified projections")
    parser.add_argument("--worlds-sample", type=int, default=5000, help="Worlds to sample for diagnostics (0=all)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for world sampling")
    parser.add_argument(
        "--payout",
        action="append",
        default=[],
        help=(
            "Payout archetype name from contest_sim.yaml, or 'FLAT20' for a custom flat top-20% payout. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        action="append",
        default=[1, 10, 50],
        help="Report gaps for these top-N values (may be passed multiple times).",
    )
    return parser.parse_args()


def _resolve_build_path(game_date: str, build_id: str) -> Path:
    return data_path("builds", "contest_sim", game_date, f"{build_id}.json")


def _load_build_inputs(args: argparse.Namespace) -> _BuildInputs:
    if args.build_path is None:
        if not args.build_id:
            raise ValueError("Provide --build-path or --build-id")
        build_path = _resolve_build_path(args.game_date, str(args.build_id))
    else:
        build_path = Path(args.build_path)

    payload = json.loads(build_path.read_text(encoding="utf-8"))
    game_date = str(payload.get("game_date") or args.game_date)

    lineups = payload.get("lineups")
    if not isinstance(lineups, list) or not lineups:
        raise ValueError("Build is missing non-empty 'lineups'")
    user_lineups = [[str(p) for p in lu] for lu in lineups]

    req = payload.get("request") if isinstance(payload.get("request"), dict) else {}
    weights = req.get("weights")
    if weights is None:
        user_weights = [1] * len(user_lineups)
    else:
        user_weights = [int(w) for w in weights]
        if len(user_weights) != len(user_lineups):
            raise ValueError("request.weights length must match build lineups length")

    cfg = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    field_size = int(cfg.get("field_size") or 0)
    entry_fee = float(cfg.get("entry_fee") or 0.0)
    rake = float(cfg.get("rake") or load_config().get("defaults", {}).get("rake", 0.15))
    if field_size <= 0 or entry_fee <= 0:
        raise ValueError("Build is missing config.field_size/entry_fee")

    stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
    debug = stats.get("debug") if isinstance(stats.get("debug"), dict) else {}
    field_library_path = debug.get("field_library_path")
    field_library_path = Path(field_library_path) if field_library_path else None

    return _BuildInputs(
        build_path=build_path,
        game_date=game_date,
        user_lineups=user_lineups,
        user_weights=user_weights,
        field_size=field_size,
        entry_fee=entry_fee,
        rake=rake,
        field_library_path=field_library_path,
    )


def _canonical_key(lineup: Iterable[object]) -> tuple[str, ...]:
    return tuple(sorted(str(p).strip() for p in lineup if str(p).strip()))


def _flat_top_pct_tiers(*, field_size: int, entry_fee: float, rake: float, paid_pct: float) -> list[PayoutTier]:
    if not (0.0 < paid_pct <= 1.0):
        raise ValueError("paid_pct must be in (0, 1]")
    itm = max(1, int(math.ceil(field_size * paid_pct)))
    prize_pool = field_size * entry_fee * (1.0 - rake)
    payout = prize_pool / float(itm)
    return [PayoutTier(start_place=1, end_place=itm, payout=float(payout))]


def _report_gap(
    *,
    label: str,
    user_mean: np.ndarray,
    ev: np.ndarray,
    top_ns: list[int],
) -> None:
    order_ev = np.argsort(ev)
    order_mean = np.argsort(user_mean)
    for n in top_ns:
        n_eff = min(int(n), int(ev.size))
        if n_eff <= 0:
            continue
        top_ev = order_ev[-n_eff:]
        top_mean = order_mean[-n_eff:]
        mean_top_ev = float(user_mean[top_ev].mean())
        mean_top_mean = float(user_mean[top_mean].mean())
        gap_pct = 0.0 if mean_top_mean == 0 else (mean_top_ev / mean_top_mean - 1.0) * 100.0
        print(
            f"{label} N={n_eff:>4d} mean(topEV)={mean_top_ev:>7.2f} mean(topMean)={mean_top_mean:>7.2f} gap={gap_pct:>6.2f}%"
        )


def main() -> None:
    args = _parse_args()
    build = _load_build_inputs(args)

    payouts = args.payout or [
        "GPP Standard (20% paid)",
        "GPP Top-Heavy (15% paid)",
        "Mini GPP Flat (25% paid)",
        "FLAT20",
    ]
    top_ns = sorted({int(n) for n in args.top_n if int(n) > 0})
    if not top_ns:
        top_ns = [1, 10, 50]

    user_total_entries = int(sum(max(int(w), 0) for w in build.user_weights))
    if user_total_entries <= 0:
        raise ValueError("Sum of user weights must be positive")
    target_field_entries = int(build.field_size - user_total_entries)
    if target_field_entries <= 0:
        raise ValueError("User entries must be less than field_size")

    if build.field_library_path and build.field_library_path.exists():
        library = load_field_library(build.field_library_path)
        field_lineups = library.lineups
        base_field_weights = library.weights
        field_source = str(build.field_library_path)
    else:
        field_lineups = build.user_lineups
        base_field_weights = build.user_weights
        field_source = "self_play"

    field_weights = scale_integer_weights_to_target(
        base_field_weights,
        target_field_entries,
        min_weight=1 if target_field_entries >= len(field_lineups) else 0,
    )

    worlds_matrix, player_index = load_worlds_matrix(build.game_date, run_id=args.run_id)
    total_worlds = int(worlds_matrix.shape[0])
    worlds_sample = int(args.worlds_sample)
    if worlds_sample <= 0 or worlds_sample >= total_worlds:
        worlds_idx = np.arange(total_worlds, dtype=np.int64)
    else:
        rng = np.random.default_rng(int(args.seed))
        worlds_idx = rng.choice(total_worlds, size=worlds_sample, replace=False)
        worlds_idx.sort()
    worlds_s = worlds_matrix[worlds_idx, :]

    user_scores = score_lineups(build.user_lineups, worlds_s, player_index)
    field_scores = score_lineups(field_lineups, worlds_s, player_index)
    user_mean = user_scores.mean(axis=1)

    # Ownership for dupe penalties (pred_own_pct from unified projections artifact).
    slate_day = pd.Timestamp(build.game_date).date()
    run_dir, _ctx = resolve_unified_run_dir(data_path(), slate_day, run_id=args.run_id)
    if run_dir is None:
        raise FileNotFoundError(f"Unified projections run dir not found for {build.game_date}")
    proj = pd.read_parquet(run_dir / "projections.parquet", columns=["player_id", "pred_own_pct"])
    ownership = dict(zip(proj["player_id"].astype(str), proj["pred_own_pct"]))

    dupe_penalties = np.array(
        compute_batch_dupe_penalties(
            build.user_lineups,
            ownership,
            field_size=build.field_size,
            entry_max=150,
        ),
        dtype=np.float64,
    )
    # Match contest_sim_service behavior: if the lineup is present in the modeled
    # field, tie-splitting already captures duplication, so disable penalty.
    field_key_to_weight: dict[tuple[str, ...], int] = {}
    for lu, w in zip(field_lineups, field_weights, strict=True):
        key = _canonical_key(lu)
        if not key:
            continue
        field_key_to_weight[key] = field_key_to_weight.get(key, 0) + int(w)
    dupe_penalties_disabled = dupe_penalties.copy()
    match_count = 0
    for i, lu in enumerate(build.user_lineups):
        key = _canonical_key(lu)
        if field_key_to_weight.get(key, 0) > 0:
            dupe_penalties_disabled[i] = 1.0
            match_count += 1

    print(f"build: {build.build_path}")
    print(f"game_date: {build.game_date} field_size: {build.field_size} entry_fee: {build.entry_fee} rake: {build.rake}")
    print(f"user_lineups: {len(build.user_lineups)} user_entries: {user_total_entries}")
    print(f"field_source: {field_source} field_unique_k: {len(field_lineups)} field_entries: {int(sum(field_weights))}")
    print(f"worlds: total={total_worlds} used={int(worlds_idx.size)} (seed={args.seed})")
    print(f"dupe_penalty: min={float(dupe_penalties.min()):.4f} mean={float(dupe_penalties.mean()):.4f} max={float(dupe_penalties.max()):.4f}")
    print(f"dupe_penalty disabled for {match_count} field matches")

    cfg = load_config()
    for payout_name in payouts:
        if payout_name.upper() == "FLAT20":
            tiers = _flat_top_pct_tiers(
                field_size=build.field_size,
                entry_fee=build.entry_fee,
                rake=build.rake,
                paid_pct=0.20,
            )
        else:
            tiers = generate_payout_tiers(payout_name, build.field_size, build.entry_fee, cfg)

        payout_res = compute_expected_user_payouts_vectorized(
            user_scores=user_scores,
            field_scores=field_scores,
            user_weights=build.user_weights,
            field_weights=field_weights,
            payout_tiers=tiers,
            workers=1,
            compute_field_side=False,
            world_chunk_size=256,
        )
        unadj_payout = payout_res.expected_payouts
        unadj_ev = unadj_payout - build.entry_fee

        print()
        print(f"=== {payout_name} ===")
        _report_gap(label="unadjusted", user_mean=user_mean, ev=unadj_ev, top_ns=top_ns)
        _report_gap(
            label="dupe_adjusted",
            user_mean=user_mean,
            ev=(unadj_payout * dupe_penalties_disabled) - build.entry_fee,
            top_ns=top_ns,
        )


if __name__ == "__main__":
    main()

