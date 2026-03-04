"""Portfolio selection utilities for multi-entry DFS.

This module is the authoritative selection layer *after* contest simulation has
already produced lineup-level metrics. It supports:

- deterministic greedy selection over a chosen contest-sim metric
- hard min-uniques and player max-exposure constraints
- covariance-aware EV retention selection using player/world outcomes
- portfolio summaries used by the API and UI diagnostics
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import ceil, floor
from math import isfinite
from typing import Iterable, Literal, Mapping, Sequence

SortDir = Literal["asc", "desc"]
SortKey = Literal[
    "lineup_id",
    "mean",
    "p90",
    "p95",
    "expected_value",
    "roi",
    "win_rate",
    "top_5pct_rate",
    "top_10pct_rate",
    "top_1pct_rate",
    "cash_rate",
    "total_own",
    "ucv90",
    "tail_score",
    "select_score",
    "score_lcb95",
    "score_cvar10",
    "robust_floor",
]


@dataclass(frozen=True)
class ExposureBoundsPct:
    """Player exposure bounds expressed in percentages (0..100)."""

    min: float | None = None
    max: float | None = None


@dataclass(frozen=True)
class PortfolioCandidate:
    """A candidate lineup with metrics used for ranking/filters."""

    lineup_id: int
    player_ids: tuple[str, ...]

    mean: float | None = None
    p90: float | None = None
    p95: float | None = None

    expected_value: float | None = None
    roi: float | None = None

    win_rate: float | None = None
    top_1pct_rate: float | None = None
    top_5pct_rate: float | None = None
    top_10pct_rate: float | None = None
    cash_rate: float | None = None

    total_own: float | None = None
    ucv90: float | None = None
    tail_score: float | None = None
    select_score: float | None = None
    score_lcb95: float | None = None
    score_cvar10: float | None = None
    robust_floor: float | None = None


@dataclass(frozen=True)
class PortfolioSelection:
    """Result of selecting a portfolio from candidate lineups."""

    selected: list[PortfolioCandidate]
    min_uniques_checked: int


@dataclass(frozen=True)
class PortfolioDiagnosticsSummary:
    """Exposure and overlap summaries for a selected lineup set."""

    exposure_summary: dict[str, object]
    overlap_summary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "exposure_summary": dict(self.exposure_summary),
            "overlap_summary": dict(self.overlap_summary),
        }


@dataclass(frozen=True)
class DecorrelatedPortfolioConfig:
    """Configuration for covariance-penalized portfolio selection."""

    ev_retention: float = 0.99
    worlds_sample: int = 5000
    seed: int = 42
    # Optional guardrail: restrict risk modeling to a deterministic train-only
    # subset of the provided worlds (split permutation uses `seed`).
    # When unset (default), behavior is unchanged (all worlds eligible).
    worlds_train_frac: float | None = None
    max_passes: int = 3
    max_swaps: int = 50


@dataclass(frozen=True)
class DecorrelatedPortfolioDiagnostics:
    """Diagnostics for a decorrelated portfolio build."""

    ev_best: float
    ev_target: float
    ev_selected: float
    worlds_used: int
    players_used: int
    passes: int
    swaps_made: int
    risk_var_total_baseline: float
    risk_var_total_selected: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "ev_best": float(self.ev_best),
            "ev_target": float(self.ev_target),
            "ev_selected": float(self.ev_selected),
            "worlds_used": int(self.worlds_used),
            "players_used": int(self.players_used),
            "passes": int(self.passes),
            "swaps_made": int(self.swaps_made),
            "risk_var_total_baseline": float(self.risk_var_total_baseline),
            "risk_var_total_selected": float(self.risk_var_total_selected),
            "risk_var_total_reduction_pct": (
                0.0
                if self.risk_var_total_baseline <= 0
                else float(
                    100.0
                    * (self.risk_var_total_baseline - self.risk_var_total_selected)
                    / self.risk_var_total_baseline
                )
            ),
        }


def _as_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _validate_exposure_bounds(
    exposure_bounds: Mapping[str, ExposureBoundsPct] | None,
) -> Mapping[str, ExposureBoundsPct]:
    bounds = exposure_bounds or {}
    for pid, bound in bounds.items():
        if bound.min is not None and not (0.0 <= float(bound.min) <= 100.0):
            raise ValueError(
                f"Exposure min for player {pid!r} must be between 0 and 100"
            )
        if bound.max is not None and not (0.0 <= float(bound.max) <= 100.0):
            raise ValueError(
                f"Exposure max for player {pid!r} must be between 0 and 100"
            )
        if (
            bound.min is not None
            and bound.max is not None
            and float(bound.min) > float(bound.max)
        ):
            raise ValueError(
                f"Exposure min for player {pid!r} cannot exceed max exposure"
            )
    return bounds


def _pct_to_min_count(pct: float, target_count: int) -> int:
    return max(0, min(target_count, int(ceil((float(pct) / 100.0) * float(target_count) - 1e-12))))


def _pct_to_max_count(pct: float, target_count: int) -> int:
    return max(0, min(target_count, int(floor((float(pct) / 100.0) * float(target_count) + 1e-12))))


def _build_exposure_count_bounds(
    *,
    bounds: Mapping[str, ExposureBoundsPct],
    portfolio_size: int,
) -> tuple[dict[str, int], dict[str, int]]:
    min_by_pid: dict[str, int] = {}
    max_by_pid: dict[str, int] = {}
    for pid, bound in bounds.items():
        if bound.min is not None:
            min_by_pid[str(pid)] = _pct_to_min_count(float(bound.min), portfolio_size)
        if bound.max is not None:
            max_by_pid[str(pid)] = _pct_to_max_count(float(bound.max), portfolio_size)
    for pid, min_count in min_by_pid.items():
        max_count = max_by_pid.get(pid)
        if max_count is not None and min_count > max_count:
            raise ValueError(
                f"Exposure constraints infeasible for player {pid!r}: "
                f"min_count={min_count} exceeds max_count={max_count}"
            )
    return min_by_pid, max_by_pid


def _count_min_uniques_passes(
    candidates: Sequence[PortfolioCandidate],
    *,
    min_uniques: int,
) -> int:
    if min_uniques <= 0:
        return len(candidates)
    selected_sets: list[set[str]] = []
    pass_count = 0
    for candidate in candidates:
        cand_set = set(candidate.player_ids)
        compatible = True
        for existing_set in selected_sets:
            if _count_uniques_between_lineups(candidate.player_ids, existing_set) < min_uniques:
                compatible = False
                break
        if not compatible:
            continue
        selected_sets.append(cand_set)
        pass_count += 1
    return pass_count


def _select_constrained_indices(
    candidates: Sequence[PortfolioCandidate],
    *,
    target_count: int,
    min_uniques: int,
    min_by_pid: Mapping[str, int],
    max_by_pid: Mapping[str, int],
) -> list[int]:
    lineup_sets = [set(c.player_ids) for c in candidates]
    constrained_pids = set(min_by_pid.keys()).union(max_by_pid.keys())
    lineup_constrained_pid_set = [
        set(pid for pid in c.player_ids if pid in constrained_pids)
        for c in candidates
    ]
    rank_score = [len(candidates) - idx for idx in range(len(candidates))]
    min_entries = [(pid, int(min_count)) for pid, min_count in min_by_pid.items() if int(min_count) > 0]

    for pid, min_count in min_entries:
        coverage = sum(1 for pid_set in lineup_constrained_pid_set if pid in pid_set)
        if coverage < min_count:
            raise ValueError(
                f"Exposure minimum infeasible for player {pid!r}: "
                f"needs {min_count}/{target_count} lineups, only {coverage} available"
            )

    selected_indices: list[int] = []
    selected_index_set: set[int] = set()
    selected_counts: dict[str, int] = {}

    def _add_index(idx: int) -> None:
        selected_indices.append(idx)
        selected_index_set.add(idx)
        for pid in lineup_constrained_pid_set[idx]:
            selected_counts[pid] = selected_counts.get(pid, 0) + 1

    def _can_add_index(idx: int) -> bool:
        for pid in lineup_constrained_pid_set[idx]:
            max_count = max_by_pid.get(pid)
            if max_count is not None and (selected_counts.get(pid, 0) + 1) > max_count:
                return False
        if min_uniques <= 0:
            return True
        for existing_idx in selected_indices:
            if _count_uniques_between_lineups(
                candidates[idx].player_ids,
                lineup_sets[existing_idx],
            ) < min_uniques:
                return False
        return True

    def _compute_total_deficit() -> int:
        total = 0
        for pid, min_count in min_entries:
            total += max(0, int(min_count) - int(selected_counts.get(pid, 0)))
        return total

    def _compute_deficits() -> list[tuple[str, int]]:
        deficits: list[tuple[str, int]] = []
        for pid, min_count in min_entries:
            deficit = int(min_count) - int(selected_counts.get(pid, 0))
            if deficit > 0:
                deficits.append((pid, deficit))
        return deficits

    while len(selected_indices) < target_count:
        deficits = _compute_deficits()
        best_idx = -1
        best_score = float("-inf")
        for idx in range(len(candidates)):
            if idx in selected_index_set or not _can_add_index(idx):
                continue
            gain = 0
            if deficits:
                for pid, _ in deficits:
                    if pid in lineup_constrained_pid_set[idx]:
                        gain += 1
                if gain == 0:
                    continue
            score = float(gain) * 1_000_000.0 + float(rank_score[idx])
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx < 0:
            break
        _add_index(best_idx)

    if len(selected_indices) < target_count:
        for idx in range(len(candidates)):
            if len(selected_indices) >= target_count:
                break
            if idx in selected_index_set or not _can_add_index(idx):
                continue
            _add_index(idx)

    def _can_swap(add_idx: int, remove_idx: int) -> bool:
        for pid, max_count in max_by_pid.items():
            next_count = int(selected_counts.get(pid, 0))
            if pid in lineup_constrained_pid_set[remove_idx]:
                next_count -= 1
            if pid in lineup_constrained_pid_set[add_idx]:
                next_count += 1
            if next_count > int(max_count):
                return False
        if min_uniques > 0:
            for existing_idx in selected_indices:
                if existing_idx == remove_idx:
                    continue
                if _count_uniques_between_lineups(
                    candidates[add_idx].player_ids,
                    lineup_sets[existing_idx],
                ) < min_uniques:
                    return False
        return True

    def _deficit_after_swap(add_idx: int, remove_idx: int) -> int:
        deficit = 0
        for pid, min_count in min_entries:
            next_count = int(selected_counts.get(pid, 0))
            if pid in lineup_constrained_pid_set[remove_idx]:
                next_count -= 1
            if pid in lineup_constrained_pid_set[add_idx]:
                next_count += 1
            deficit += max(0, int(min_count) - next_count)
        return deficit

    repair_iterations = 0
    while _compute_total_deficit() > 0 and repair_iterations < 200:
        repair_iterations += 1
        deficits = _compute_deficits()
        best_swap: tuple[int, int, int, int] | None = None
        for add_idx in range(len(candidates)):
            if add_idx in selected_index_set:
                continue
            if deficits and not any(pid in lineup_constrained_pid_set[add_idx] for pid, _ in deficits):
                continue
            for remove_idx in selected_indices:
                if not _can_swap(add_idx, remove_idx):
                    continue
                deficit = _deficit_after_swap(add_idx, remove_idx)
                score_delta = int(rank_score[add_idx]) - int(rank_score[remove_idx])
                if best_swap is None:
                    best_swap = (add_idx, remove_idx, deficit, score_delta)
                    continue
                _, _, best_deficit, best_score_delta = best_swap
                if deficit < best_deficit or (deficit == best_deficit and score_delta > best_score_delta):
                    best_swap = (add_idx, remove_idx, deficit, score_delta)

        if best_swap is None:
            break
        current_deficit = _compute_total_deficit()
        add_idx, remove_idx, deficit, score_delta = best_swap
        if deficit > current_deficit:
            break
        if deficit == current_deficit and score_delta <= 0:
            break

        remove_pos = selected_indices.index(remove_idx)
        selected_indices[remove_pos] = add_idx
        selected_index_set.remove(remove_idx)
        selected_index_set.add(add_idx)
        for pid in lineup_constrained_pid_set[remove_idx]:
            selected_counts[pid] = int(selected_counts.get(pid, 0)) - 1
        for pid in lineup_constrained_pid_set[add_idx]:
            selected_counts[pid] = int(selected_counts.get(pid, 0)) + 1

    if len(selected_indices) < target_count:
        raise ValueError(
            f"Only {len(selected_indices)} of {target_count} lineups meet constraints (pool exhausted)"
        )

    unmet = [
        f"{pid!r} ({selected_counts.get(pid, 0)}/{min_count})"
        for pid, min_count in min_entries
        if int(selected_counts.get(pid, 0)) < int(min_count)
    ]
    if unmet:
        raise ValueError(
            "Min exposure constraints not met (pool exhausted): " + ", ".join(unmet)
        )

    selected_indices.sort()
    return selected_indices


def _get_candidate_metric_value(
    candidate: PortfolioCandidate,
    *,
    sort_key: SortKey,
) -> float | None:
    if sort_key == "lineup_id":
        return float(candidate.lineup_id)
    value = getattr(candidate, sort_key, None)
    value_f = _as_float(value)
    if value_f is None or not isfinite(value_f):
        return None
    return float(value_f)


def get_candidate_metric_value(
    candidate: PortfolioCandidate,
    *,
    sort_key: SortKey,
) -> float | None:
    """Public helper for API-level candidate filtering/diagnostics."""
    return _get_candidate_metric_value(candidate, sort_key=sort_key)


def _sort_candidates(
    candidates: Sequence[PortfolioCandidate],
    *,
    sort_key: SortKey,
    sort_dir: SortDir,
) -> list[PortfolioCandidate]:
    multiplier = -1.0 if sort_dir == "desc" else 1.0

    def _key(candidate: PortfolioCandidate) -> tuple[int, float, int]:
        metric_value = _get_candidate_metric_value(candidate, sort_key=sort_key)
        if metric_value is None:
            return (1, 0.0, candidate.lineup_id)
        return (0, multiplier * float(metric_value), candidate.lineup_id)

    return sorted(
        candidates,
        key=_key,
    )


def _count_uniques_between_lineups(
    candidate_player_ids: Sequence[str],
    existing_player_ids: set[str],
) -> int:
    shared = 0
    for pid in candidate_player_ids:
        if pid in existing_player_ids:
            shared += 1
    return len(candidate_player_ids) - shared


def _passes_min_uniques(
    candidate: PortfolioCandidate,
    *,
    selected_sets: Sequence[set[str]],
    min_uniques: int,
) -> bool:
    if min_uniques <= 0:
        return True
    for existing in selected_sets:
        if _count_uniques_between_lineups(candidate.player_ids, existing) < min_uniques:
            return False
    return True


def build_portfolio(
    candidates: Sequence[PortfolioCandidate],
    portfolio_size: int,
    *,
    sort_key: SortKey = "expected_value",
    sort_dir: SortDir = "desc",
    filter_positive_ev: bool = False,
    max_total_own: float | None = None,
    min_uniques: int = 0,
    exposure_bounds: Mapping[str, ExposureBoundsPct] | None = None,
    seed_lineup_ids: Sequence[int] | None = None,
) -> PortfolioSelection:
    """Select a portfolio of lineups with simple diversification constraints.

    Parameters
    ----------
    candidates:
        Candidate lineups with EV metrics from contest sim.
    portfolio_size:
        Target number of entries to select.
    sort_key / sort_dir:
        Ordering used for greedy selection.
    filter_positive_ev:
        If True, exclude candidates with EV < 0 (or missing EV).
    max_total_own:
        If provided, exclude candidates whose summed ownership exceeds this value.
        Candidates missing ``total_own`` are excluded when this filter is active.
    min_uniques:
        Minimum unique players between any pair of selected lineups.
    exposure_bounds:
        Optional player exposure min/max percentage bounds; selection enforces
        min/max bounds during greedy construction/repair.
    seed_lineup_ids:
        Optional previously selected lineup ids. When provided, selection
        prioritizes these lineups (highest ranked first) if they still satisfy
        active constraints, then fills remaining slots from the ranked pool.

    Returns
    -------
    PortfolioSelection
        Selected lineups in the order they were chosen.
    """
    if portfolio_size < 1:
        raise ValueError("portfolio_size must be >= 1")
    if not candidates:
        raise ValueError("candidates must be non-empty")
    if min_uniques < 0:
        raise ValueError("min_uniques must be >= 0")

    bounds = _validate_exposure_bounds(exposure_bounds)

    filtered: list[PortfolioCandidate] = []
    for c in candidates:
        if filter_positive_ev:
            ev = _as_float(c.expected_value)
            if ev is None or ev < 0:
                continue
        if max_total_own is not None:
            own = _as_float(c.total_own)
            if own is None or own > float(max_total_own):
                continue
        filtered.append(c)

    if not filtered:
        raise ValueError("No candidates remain after filtering")
    if portfolio_size > len(filtered):
        raise ValueError(
            f"portfolio_size={portfolio_size} exceeds candidate count={len(filtered)}"
        )

    ordered = _sort_candidates(filtered, sort_key=sort_key, sort_dir=sort_dir)
    if seed_lineup_ids:
        rank_by_lineup_id = {c.lineup_id: idx for idx, c in enumerate(ordered)}
        seed_ids_ranked: list[int] = []
        seen_seed_ids: set[int] = set()
        for lineup_id_raw in seed_lineup_ids:
            try:
                lineup_id = int(lineup_id_raw)
            except Exception:
                continue
            if lineup_id in seen_seed_ids:
                continue
            if lineup_id not in rank_by_lineup_id:
                continue
            seen_seed_ids.add(lineup_id)
            seed_ids_ranked.append(lineup_id)
        seed_ids_ranked.sort(key=lambda lineup_id: rank_by_lineup_id[lineup_id])
        if seed_ids_ranked:
            seed_set = set(seed_ids_ranked)
            seed_front = [c for c in ordered if c.lineup_id in seed_set]
            ordered = seed_front + [c for c in ordered if c.lineup_id not in seed_set]

    min_by_pid, max_by_pid = _build_exposure_count_bounds(
        bounds=bounds,
        portfolio_size=portfolio_size,
    )
    selected_indices = _select_constrained_indices(
        ordered,
        target_count=portfolio_size,
        min_uniques=min_uniques,
        min_by_pid=min_by_pid,
        max_by_pid=max_by_pid,
    )
    min_uniques_checked = _count_min_uniques_passes(
        ordered,
        min_uniques=min_uniques,
    )
    selected = [ordered[idx] for idx in selected_indices]

    return PortfolioSelection(
        selected=selected,
        min_uniques_checked=min_uniques_checked,
    )


def compute_total_own(player_ids: Iterable[str], ownership: Mapping[str, float]) -> float:
    """Compute total ownership as sum of per-player ownership percentages."""
    total = 0.0
    for pid in player_ids:
        try:
            total += float(ownership.get(str(pid), 0.0))
        except Exception:
            continue
    return total


def summarize_portfolio(
    selected: Sequence[PortfolioCandidate],
) -> PortfolioDiagnosticsSummary:
    """Build exposure and overlap summaries for diagnostics."""
    if not selected:
        return PortfolioDiagnosticsSummary(
            exposure_summary={
                "unique_players": 0,
                "top_players": [],
                "max_exposure_pct": 0.0,
            },
            overlap_summary={
                "pairs": 0,
                "mean_shared_players": 0.0,
                "max_shared_players": 0,
                "mean_uniques": 0.0,
                "min_uniques": 0,
            },
        )

    exposure_counts: dict[str, int] = {}
    selected_sets = [set(c.player_ids) for c in selected]
    for candidate in selected:
        for pid in candidate.player_ids:
            exposure_counts[pid] = exposure_counts.get(pid, 0) + 1

    lineup_count = len(selected)
    top_players = sorted(
        (
            {
                "player_id": pid,
                "count": count,
                "exposure_pct": round((count / lineup_count) * 100.0, 2),
            }
            for pid, count in exposure_counts.items()
        ),
        key=lambda item: (-float(item["exposure_pct"]), str(item["player_id"])),
    )[:15]

    shared_counts: list[int] = []
    unique_counts: list[int] = []
    lineup_size = max((len(c.player_ids) for c in selected), default=0)
    for i in range(len(selected_sets)):
        for j in range(i + 1, len(selected_sets)):
            shared = len(selected_sets[i].intersection(selected_sets[j]))
            shared_counts.append(shared)
            unique_counts.append(max(0, lineup_size - shared))

    overlap_summary = {
        "pairs": len(shared_counts),
        "mean_shared_players": (
            0.0 if not shared_counts else round(sum(shared_counts) / len(shared_counts), 4)
        ),
        "max_shared_players": max(shared_counts, default=0),
        "mean_uniques": (
            0.0 if not unique_counts else round(sum(unique_counts) / len(unique_counts), 4)
        ),
        "min_uniques": min(unique_counts, default=0),
    }
    exposure_summary = {
        "unique_players": len(exposure_counts),
        "max_exposure_pct": max(
            (float(item["exposure_pct"]) for item in top_players),
            default=0.0,
        ),
        "top_players": top_players,
    }
    return PortfolioDiagnosticsSummary(
        exposure_summary=exposure_summary,
        overlap_summary=overlap_summary,
    )


def build_decorrelated_portfolio(
    candidates: Sequence[PortfolioCandidate],
    portfolio_size: int,
    *,
    worlds_matrix,
    player_index: Mapping[str, int],
    world_indices: Sequence[int] | None = None,
    config: DecorrelatedPortfolioConfig | None = None,
    exposure_bounds: Mapping[str, ExposureBoundsPct] | None = None,
    min_uniques: int = 0,
    seed_lineup_ids: Sequence[int] | None = None,
) -> tuple[PortfolioSelection, DecorrelatedPortfolioDiagnostics]:
    """Select a diversified portfolio by penalizing covariance between lineups.

    This builds a player-level covariance matrix from `worlds_matrix` and uses a
    greedy algorithm that trades off EV vs incremental portfolio variance.

    The key property: diversification works *without* requiring explicit min-uniques
    or exposure caps. (Caps can still be used as optional guardrails.)
    """
    import numpy as np

    if portfolio_size < 1:
        raise ValueError("portfolio_size must be >= 1")
    if not candidates:
        raise ValueError("candidates must be non-empty")
    if portfolio_size > len(candidates):
        raise ValueError(
            f"portfolio_size={portfolio_size} exceeds candidate count={len(candidates)}"
        )
    cfg = config or DecorrelatedPortfolioConfig()
    if (
        world_indices is None
        and cfg.worlds_train_frac is None
        and os.environ.get("PROJECTIONS_SELECTION_REQUIRE_WORLD_IDX", "").strip().lower() in {"1", "true", "yes", "y"}
    ):
        raise ValueError(
            "Selection guardrail active: pass world_indices=... or set DecorrelatedPortfolioConfig(worlds_train_frac=...)."
        )
    if not (0.0 < float(cfg.ev_retention) <= 1.0):
        raise ValueError("ev_retention must be in (0, 1]")
    if int(cfg.worlds_sample) <= 0:
        raise ValueError("worlds_sample must be positive")
    if int(cfg.max_passes) < 1:
        raise ValueError("max_passes must be >= 1")
    if int(cfg.max_swaps) < 0:
        raise ValueError("max_swaps must be >= 0")
    if min_uniques < 0:
        raise ValueError("min_uniques must be >= 0")

    bounds = _validate_exposure_bounds(exposure_bounds)

    # Build player universe from candidates.
    player_ids: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        for pid in cand.player_ids:
            pid_s = str(pid).strip()
            if not pid_s or pid_s in seen:
                continue
            if pid_s not in player_index:
                continue
            seen.add(pid_s)
            player_ids.append(pid_s)

    if not player_ids:
        raise ValueError("No candidate players found in worlds player_index")

    pid_to_local = {pid: i for i, pid in enumerate(player_ids)}
    cols = np.asarray([int(player_index[pid]) for pid in player_ids], dtype=np.int64)
    worlds = np.asarray(worlds_matrix, dtype=np.float32)
    worlds_sub = np.take(worlds, cols, axis=1)

    W = int(worlds_sub.shape[0])
    if W < 2:
        raise ValueError("Need at least 2 worlds to compute covariance")

    # Optional: restrict to an explicit subset of worlds, or a deterministic train split.
    if world_indices is not None:
        idx = np.asarray(world_indices, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError("world_indices must be a 1D array of row indices")
        if idx.size < 2:
            raise ValueError("world_indices must include at least 2 worlds")
        worlds_sub = worlds_sub[idx, :]
        W = int(worlds_sub.shape[0])
    elif cfg.worlds_train_frac is not None:
        train_frac = float(cfg.worlds_train_frac)
        if not (0.0 < train_frac < 1.0):
            raise ValueError("worlds_train_frac must be in (0, 1)")
        rng_split = np.random.default_rng(int(cfg.seed))
        perm = rng_split.permutation(int(W))
        n_train = int(np.floor(train_frac * float(W)))
        n_train = max(1, min(n_train, int(W) - 1))
        rows = np.sort(perm[:n_train])
        worlds_sub = worlds_sub[rows, :]
        W = int(worlds_sub.shape[0])
    target_W = min(W, int(cfg.worlds_sample))
    if target_W < W:
        rng = np.random.default_rng(int(cfg.seed))
        rows = rng.choice(W, size=target_W, replace=False)
        worlds_sub = worlds_sub[rows, :]
        W = target_W

    # Compute player covariance Σ = X^T X / (W-1).
    mu = worlds_sub.mean(axis=0, dtype=np.float32)
    X = worlds_sub - mu  # (W, P)
    sigma = (X.T @ X) / float(max(1, W - 1))  # (P, P)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Build lineup index matrix (K, L) and EV array.
    lineup_idxs: list[list[int]] = []
    lineup_evs: list[float] = []
    for cand in candidates:
        ev = _as_float(cand.expected_value)
        if ev is None:
            ev = float("-inf")
        idxs = [pid_to_local[str(pid)] for pid in cand.player_ids if str(pid) in pid_to_local]
        if not idxs:
            # No mapped players -> cannot risk-model; treat as unusable.
            ev = float("-inf")
        lineup_idxs.append(idxs)
        lineup_evs.append(float(ev))

    ev_arr = np.asarray(lineup_evs, dtype=np.float64)

    # Keep only candidates with finite EV and at least 1 mapped player.
    keep_mask = np.isfinite(ev_arr) & (np.asarray([len(x) for x in lineup_idxs]) > 0)
    if int(np.sum(keep_mask)) < portfolio_size:
        raise ValueError(
            f"Not enough candidates with finite EV ({int(np.sum(keep_mask))}) for portfolio_size={portfolio_size}"
        )

    kept_candidates = [c for c, keep in zip(candidates, keep_mask) if keep]
    kept_idxs = [idxs for idxs, keep in zip(lineup_idxs, keep_mask) if keep]
    kept_ev = ev_arr[keep_mask]
    kept_player_sets = [set(c.player_ids) for c in kept_candidates]

    # Re-pack to dense (K2, Lmax) with padding for fast vectorized penalty sums.
    K2 = len(kept_candidates)
    Lmax = max(len(idxs) for idxs in kept_idxs)
    idx_mat = np.full((K2, Lmax), -1, dtype=np.int32)
    for i, idxs in enumerate(kept_idxs):
        idx_mat[i, : len(idxs)] = np.asarray(idxs, dtype=np.int32)
    idx_safe = idx_mat.copy()
    idx_safe[idx_safe < 0] = 0
    pad_mask = (idx_mat >= 0).astype(np.float64)

    # Precompute lineup variances.
    var_arr = np.zeros((K2,), dtype=np.float64)
    for i in range(K2):
        idxs = idx_mat[i]
        idxs = idxs[idxs >= 0]
        sub = sigma[np.ix_(idxs, idxs)]
        var_arr[i] = float(np.sum(sub))

    # Precompute Σ @ a_i for fast swap updates and covariance terms.
    sigma_sum = np.zeros((K2, len(player_ids)), dtype=np.float32)
    for i in range(K2):
        idxs = idx_mat[i]
        idxs = idxs[idxs >= 0]
        sigma_sum[i] = np.sum(sigma[:, idxs], axis=1, dtype=np.float64).astype(np.float32)

    # Exposure bounds (optional): convert pct bounds to count bounds.
    max_counts = np.full((len(player_ids),), np.inf, dtype=np.float64)
    min_counts = np.zeros((len(player_ids),), dtype=np.float64)
    min_locs = np.asarray([], dtype=np.int64)
    min_targets = np.asarray([], dtype=np.float64)
    min_hit = None
    if bounds:
        for pid, b in bounds.items():
            loc = pid_to_local.get(str(pid))
            if loc is None:
                continue
            if b.max is not None:
                max_counts[loc] = float(_pct_to_max_count(float(b.max), portfolio_size))
            if b.min is not None:
                min_counts[loc] = float(_pct_to_min_count(float(b.min), portfolio_size))
        finite_max = np.isfinite(max_counts)
        infeasible = np.flatnonzero((min_counts > 0.0) & finite_max & (min_counts > max_counts + 1e-9))
        if infeasible.size > 0:
            pid = player_ids[int(infeasible[0])]
            raise ValueError(
                f"Exposure constraints infeasible for player {pid!r}: "
                f"min_count={int(min_counts[int(infeasible[0])])} exceeds "
                f"max_count={int(max_counts[int(infeasible[0])])}"
            )
        min_locs = np.flatnonzero(min_counts > 0.0).astype(np.int64)
        if min_locs.size > 0:
            min_targets = min_counts[min_locs].astype(np.float64)
            min_hit = np.zeros((K2, int(min_locs.size)), dtype=np.float64)
            loc_to_col = {int(loc): j for j, loc in enumerate(min_locs.tolist())}
            for i, idxs in enumerate(kept_idxs):
                for loc in idxs:
                    col = loc_to_col.get(int(loc))
                    if col is not None:
                        min_hit[i, col] = 1.0
            coverage = np.sum(min_hit > 0.0, axis=0)
            for j, cov in enumerate(coverage.tolist()):
                needed = int(min_targets[j])
                if int(cov) < needed:
                    pid = player_ids[int(min_locs[j])]
                    raise ValueError(
                        f"Exposure minimum infeasible for player {pid!r}: "
                        f"needs {needed}/{portfolio_size} lineups, only {int(cov)} available"
                    )

    kept_lineup_idx_by_id = {c.lineup_id: idx for idx, c in enumerate(kept_candidates)}
    seed_idxs: list[int] = []
    if seed_lineup_ids:
        seen_seed_idxs: set[int] = set()
        for lineup_id_raw in seed_lineup_ids:
            try:
                lineup_id = int(lineup_id_raw)
            except Exception:
                continue
            idx = kept_lineup_idx_by_id.get(lineup_id)
            if idx is None or idx in seen_seed_idxs:
                continue
            seen_seed_idxs.add(idx)
            seed_idxs.append(idx)

    def _select_best_ev_feasible(initial_seed_idxs: Sequence[int] | None = None) -> list[int]:
        """Best-EV greedy baseline under active hard constraints."""
        order = np.argsort(-kept_ev)
        if initial_seed_idxs:
            dedup_seed_idxs: list[int] = []
            seen_seed_idxs: set[int] = set()
            for idx in initial_seed_idxs:
                idx_i = int(idx)
                if idx_i < 0 or idx_i >= K2 or idx_i in seen_seed_idxs:
                    continue
                seen_seed_idxs.add(idx_i)
                dedup_seed_idxs.append(idx_i)
            dedup_seed_idxs.sort(
                key=lambda idx_i: (-float(kept_ev[idx_i]), kept_candidates[idx_i].lineup_id)
            )
            if dedup_seed_idxs:
                order = np.asarray(
                    dedup_seed_idxs + [int(i) for i in order if int(i) not in seen_seed_idxs],
                    dtype=np.int64,
                )

        selected: list[int] = []
        counts = np.zeros((len(player_ids),), dtype=np.float64)
        for i in order:
            idxs = idx_mat[int(i)]
            idxs = idxs[idxs >= 0]
            if idxs.size == 0:
                continue
            if np.any((counts[idxs] + 1.0) > (max_counts[idxs] + 1e-9)):
                continue
            if min_uniques > 0:
                cand_set = kept_player_sets[int(i)]
                violates = False
                for selected_idx in selected:
                    if _count_uniques_between_lineups(
                        tuple(cand_set),
                        kept_player_sets[int(selected_idx)],
                    ) < min_uniques:
                        violates = True
                        break
                if violates:
                    continue
            if min_locs.size > 0 and min_hit is not None:
                remaining_slots = int(portfolio_size) - (len(selected) + 1)
                post_counts = counts[min_locs] + min_hit[int(i)]
                if np.any((post_counts + float(remaining_slots)) < (min_targets - 1e-9)):
                    continue
            selected.append(int(i))
            counts[idxs] += 1.0
            if len(selected) >= portfolio_size:
                break

        if len(selected) < portfolio_size:
            raise ValueError("No feasible portfolio under active constraints (pool exhausted)")
        if min_locs.size > 0 and np.any(counts[min_locs] + 1e-9 < min_targets):
            unmet: list[str] = []
            for j, loc in enumerate(min_locs.tolist()):
                got = int(counts[int(loc)])
                need = int(min_targets[j])
                if got < need:
                    unmet.append(f"{player_ids[int(loc)]!r} ({got}/{need})")
            raise ValueError(
                "No feasible portfolio under active constraints (min exposures unmet): "
                + ", ".join(unmet)
            )
        return selected

    # Baseline: best EV portfolio (under optional caps).
    best_ev_selected = _select_best_ev_feasible()
    ev_best = float(np.sum(kept_ev[best_ev_selected]))
    ev_target = float(cfg.ev_retention) * ev_best

    selected = list(best_ev_selected)
    if seed_idxs:
        seeded_selected = _select_best_ev_feasible(seed_idxs)
        seeded_ev = float(np.sum(kept_ev[seeded_selected]))
        # Preserve seeded portfolios when they already satisfy EV retention.
        if seeded_ev + 1e-9 >= ev_target:
            selected = list(seeded_selected)

    selected_mask = np.zeros((K2,), dtype=bool)
    selected_mask[selected] = True

    counts = np.zeros((len(player_ids),), dtype=np.float64)
    for i in selected:
        idxs = idx_mat[int(i)]
        idxs = idxs[idxs >= 0]
        counts[idxs] += 1.0

    v = sigma @ counts  # Σ @ counts
    risk_base = float(np.dot(counts, v))

    ev_selected = float(np.sum(kept_ev[selected]))
    risk_selected = float(risk_base)

    passes = 0
    swaps_made = 0
    tol = 1e-9

    for _ in range(int(cfg.max_passes)):
        if swaps_made >= int(cfg.max_swaps):
            break
        passes += 1
        improved = False

        penalty_all = np.sum(v[idx_safe] * pad_mask, axis=1)  # a_i^T Σ counts
        slack = ev_selected - ev_target

        for pos in range(portfolio_size):
            if swaps_made >= int(cfg.max_swaps):
                break

            idx_remove = int(selected[pos])
            idxs_remove = idx_mat[idx_remove]
            idxs_remove = idxs_remove[idxs_remove >= 0]
            if idxs_remove.size == 0:
                continue

            min_ev_allowed = float(kept_ev[idx_remove] - slack)

            feasible = (~selected_mask) & (kept_ev >= min_ev_allowed)
            if not feasible.any():
                continue

            # Apply max exposure caps for this swap: counts' = counts - a_i + a_j.
            if not np.isinf(max_counts).all():
                counts_wo = counts.copy()
                counts_wo[idxs_remove] -= 1.0
                over = np.any(
                    ((counts_wo[idx_safe] + 1.0) * pad_mask) > (max_counts[idx_safe] + 1e-9),
                    axis=1,
                )
                feasible &= ~over
                if not feasible.any():
                    continue
            else:
                counts_wo = counts.copy()
                counts_wo[idxs_remove] -= 1.0

            if min_locs.size > 0 and min_hit is not None:
                post_counts = counts_wo[min_locs][None, :] + min_hit
                meets_min = np.all(post_counts + 1e-9 >= min_targets[None, :], axis=1)
                feasible &= meets_min
                if not feasible.any():
                    continue

            if min_uniques > 0:
                feasible_uniques = np.ones((K2,), dtype=bool)
                current_sets = [
                    kept_player_sets[int(selected_idx)]
                    for selected_idx in selected
                    if int(selected_idx) != idx_remove
                ]
                for candidate_idx in np.flatnonzero(feasible):
                    cand_set = kept_player_sets[int(candidate_idx)]
                    for existing_set in current_sets:
                        if _count_uniques_between_lineups(
                            tuple(cand_set),
                            existing_set,
                        ) < min_uniques:
                            feasible_uniques[int(candidate_idx)] = False
                            break
                feasible &= feasible_uniques
                if not feasible.any():
                    continue

            # Risk objective:
            #   R(counts) = countsᵀ Σ counts
            # where counts is the player exposure-count vector across selected lineups.
            #
            # For swapping out lineup i and swapping in lineup j:
            #   counts' = counts - a_i + a_j
            # where a_k is the 0/1 indicator vector for lineup k. (Note: counts here
            # includes a_i at evaluation time; we do *not* pre-subtract it.)
            #
            # Closed form:
            #   ΔR = R(counts') - R(counts)
            #      = (var_i - 2 * countsᵀΣ a_i) + (var_j + 2 * countsᵀΣ a_j) - 2 * cov_ij
            # where:
            #   var_k = a_kᵀ Σ a_k          (precomputed in var_arr)
            #   cov_ij = a_iᵀ Σ a_j         (computed via Σ a_j summed over i's players)
            #   countsᵀΣ a_k = a_kᵀ Σ counts (computed via v = Σ counts, summed over k's players)
            #
            # We compute ΔR for all candidate j efficiently and pick the most-negative
            # (largest risk reduction) swap that maintains EV >= target.
            covs = np.sum(sigma_sum[:, idxs_remove], axis=1, dtype=np.float64)
            base_removal = float(var_arr[idx_remove] - 2.0 * penalty_all[idx_remove])
            delta = base_removal + (var_arr + 2.0 * penalty_all) - 2.0 * covs
            delta[~feasible] = float("inf")

            idx_add = int(np.argmin(delta))
            best_delta = float(delta[idx_add])
            if not np.isfinite(best_delta) or best_delta >= -tol:
                continue

            idxs_add = idx_mat[idx_add]
            idxs_add = idxs_add[idxs_add >= 0]
            if idxs_add.size == 0:
                continue

            # Execute swap.
            selected_mask[idx_remove] = False
            selected_mask[idx_add] = True
            selected[pos] = idx_add

            ev_selected = float(ev_selected - kept_ev[idx_remove] + kept_ev[idx_add])

            counts[idxs_remove] -= 1.0
            counts[idxs_add] += 1.0

            v = v - sigma_sum[idx_remove].astype(np.float64) + sigma_sum[idx_add].astype(np.float64)
            risk_selected = float(risk_selected + best_delta)

            swaps_made += 1
            improved = True

            # Recompute penalties for subsequent swaps with updated counts/v.
            penalty_all = np.sum(v[idx_safe] * pad_mask, axis=1)
            slack = ev_selected - ev_target

        if not improved:
            break

    # Recompute risk exactly for reporting.
    risk_selected = float(np.dot(counts, sigma @ counts))
    if min_locs.size > 0 and np.any(counts[min_locs] + 1e-9 < min_targets):
        unmet: list[str] = []
        for j, loc in enumerate(min_locs.tolist()):
            got = int(counts[int(loc)])
            need = int(min_targets[j])
            if got < need:
                unmet.append(f"{player_ids[int(loc)]!r} ({got}/{need})")
        raise ValueError(
            "No feasible portfolio under active constraints (min exposures unmet): "
            + ", ".join(unmet)
        )

    selection = PortfolioSelection(
        selected=[kept_candidates[i] for i in selected],
        min_uniques_checked=int(min_uniques),
    )
    diag = DecorrelatedPortfolioDiagnostics(
        ev_best=ev_best,
        ev_target=ev_target,
        ev_selected=float(ev_selected),
        worlds_used=W,
        players_used=len(player_ids),
        passes=int(passes),
        swaps_made=int(swaps_made),
        risk_var_total_baseline=risk_base,
        risk_var_total_selected=float(risk_selected),
    )
    return selection, diag
