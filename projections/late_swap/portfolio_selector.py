"""Grouped portfolio selector for Late Swap V2."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from projections.late_swap.models import (
    LateSwapCandidate,
    LateSwapPolicy,
    pct_to_max_count,
    pct_to_min_count,
)


@dataclass(frozen=True)
class SelectorInput:
    candidates_by_entry_id: dict[str, list[LateSwapCandidate]]
    policy: LateSwapPolicy
    locked_floor_count_by_player: dict[str, int]
    target_count_by_player: dict[str, int]
    pinned_candidates_by_entry_id: dict[str, str]
    team_keys_by_candidate_id: dict[str, set[str]]
    game_keys_by_candidate_id: dict[str, set[str]]


@dataclass(frozen=True)
class SelectorResult:
    status: str
    selected_candidate_ids_by_entry_id: dict[str, str]
    objective_value: float | None
    warnings: list[str]
    notes: list[str]
    infeasible: bool


def _count_uniques(left: list[str], right: list[str]) -> int:
    return len(set(left).symmetric_difference(set(right)))


def _fallback_hold_selection(
    candidates_by_entry_id: dict[str, list[LateSwapCandidate]],
) -> dict[str, str]:
    selected: dict[str, str] = {}
    for entry_id, candidates in candidates_by_entry_id.items():
        hold = next((candidate for candidate in candidates if candidate.generated_by == "hold"), None)
        selected[entry_id] = (hold or candidates[0]).candidate_id
    return selected


def select_grouped_portfolio(payload: SelectorInput) -> SelectorResult:
    # Import lazily to avoid import-time hard dependency in tooling paths.
    from ortools.sat.python import cp_model

    warnings: list[str] = []
    notes: list[str] = []

    entries = sorted(payload.candidates_by_entry_id.keys())
    if not entries:
        return SelectorResult(
            status="infeasible",
            selected_candidate_ids_by_entry_id={},
            objective_value=None,
            warnings=["No entries available for grouped selection"],
            notes=[],
            infeasible=True,
        )

    model = cp_model.CpModel()
    x: dict[tuple[str, str], cp_model.IntVar] = {}
    candidate_by_key: dict[tuple[str, str], LateSwapCandidate] = {}
    candidate_by_id: dict[str, LateSwapCandidate] = {}

    for entry_id in entries:
        entry_candidates = payload.candidates_by_entry_id.get(entry_id, [])
        if not entry_candidates:
            warnings.append(f"Entry {entry_id} has zero candidates; falling back to hold.")
            return SelectorResult(
                status="fallback_hold",
                selected_candidate_ids_by_entry_id=_fallback_hold_selection(payload.candidates_by_entry_id),
                objective_value=None,
                warnings=warnings,
                notes=notes,
                infeasible=True,
            )
        for candidate in entry_candidates:
            key = (entry_id, candidate.candidate_id)
            x[key] = model.NewBoolVar(f"x_{entry_id}_{candidate.candidate_id}")
            candidate_by_key[key] = candidate
            candidate_by_id[candidate.candidate_id] = candidate

    # Exactly one candidate per entry.
    for entry_id in entries:
        vars_for_entry = [var for (eid, _cid), var in x.items() if eid == entry_id]
        model.Add(sum(vars_for_entry) == 1)

    # Pin constraints.
    for entry_id, candidate_id in payload.pinned_candidates_by_entry_id.items():
        if (entry_id, candidate_id) not in x:
            warnings.append(f"Pinned candidate missing: entry={entry_id} candidate={candidate_id}")
            continue
        model.Add(x[(entry_id, candidate_id)] == 1)

    # Duplicate lineup cap.
    by_hash: dict[str, list[cp_model.IntVar]] = {}
    for (entry_id, candidate_id), var in x.items():
        lineup_hash = candidate_by_key[(entry_id, candidate_id)].lineup_hash
        by_hash.setdefault(lineup_hash, []).append(var)
    max_dupes = max(1, int(payload.policy.max_duplicate_lineups))
    for lineup_hash, vars_for_hash in by_hash.items():
        if len(vars_for_hash) > 1:
            model.Add(sum(vars_for_hash) <= max_dupes)

    # Player exposure constraints.
    n_entries = len(entries)
    all_bound_pids = sorted(payload.policy.exposure_bounds.keys())
    for pid in all_bound_pids:
        bound = payload.policy.exposure_bounds[pid]
        vars_with_pid: list[cp_model.IntVar] = []
        for key, var in x.items():
            candidate = candidate_by_key[key]
            if str(pid) in set(candidate.player_ids):
                vars_with_pid.append(var)
        count_expr = sum(vars_with_pid)
        floor_count = int(payload.locked_floor_count_by_player.get(str(pid), 0))
        if bound.max is not None:
            raw_cap = pct_to_max_count(float(bound.max), n_entries)
            if floor_count > raw_cap:
                warnings.append(
                    f"Player {pid} cap infeasible from locks (floor={floor_count}, cap={raw_cap}); enforcing no-add rule."
                )
                model.Add(count_expr <= floor_count)
            else:
                model.Add(count_expr <= raw_cap)
        if bound.min is not None:
            model.Add(count_expr >= pct_to_min_count(float(bound.min), n_entries))

    # Team exposure constraints.
    for team_key, bound in payload.policy.team_exposure_bounds.items():
        vars_with_team = [
            var
            for (entry_id, candidate_id), var in x.items()
            if str(team_key) in payload.team_keys_by_candidate_id.get(candidate_id, set())
        ]
        if not vars_with_team:
            continue
        count_expr = sum(vars_with_team)
        if bound.max is not None:
            model.Add(count_expr <= pct_to_max_count(float(bound.max), n_entries))
        if bound.min is not None:
            model.Add(count_expr >= pct_to_min_count(float(bound.min), n_entries))

    # Game exposure constraints.
    for game_key, bound in payload.policy.game_exposure_bounds.items():
        vars_with_game = [
            var
            for (entry_id, candidate_id), var in x.items()
            if str(game_key) in payload.game_keys_by_candidate_id.get(candidate_id, set())
        ]
        if not vars_with_game:
            continue
        count_expr = sum(vars_with_game)
        if bound.max is not None:
            model.Add(count_expr <= pct_to_max_count(float(bound.max), n_entries))
        if bound.min is not None:
            model.Add(count_expr >= pct_to_min_count(float(bound.min), n_entries))

    # Max total swaps.
    if payload.policy.max_total_swaps is not None:
        model.Add(
            sum(
                int(candidate_by_key[key].swap_count) * var
                for key, var in x.items()
            )
            <= int(payload.policy.max_total_swaps)
        )

    # Max swaps per entry.
    if payload.policy.max_swaps_per_entry is not None:
        limit = int(payload.policy.max_swaps_per_entry)
        for (entry_id, candidate_id), var in x.items():
            candidate = candidate_by_key[(entry_id, candidate_id)]
            if candidate.swap_count > limit:
                model.Add(var == 0)

    # Minimum uniques constraints (pairwise; skipped at high scale).
    if payload.policy.min_uniques > 0:
        pair_count = 0
        max_pairwise_constraints = 250_000
        for left_entry, right_entry in combinations(entries, 2):
            left_candidates = payload.candidates_by_entry_id[left_entry]
            right_candidates = payload.candidates_by_entry_id[right_entry]
            for left in left_candidates:
                for right in right_candidates:
                    if _count_uniques(left.player_ids, right.player_ids) < payload.policy.min_uniques:
                        model.Add(x[(left_entry, left.candidate_id)] + x[(right_entry, right.candidate_id)] <= 1)
                        pair_count += 1
                        if pair_count >= max_pairwise_constraints:
                            warnings.append(
                                "Skipped additional min-uniques pairwise constraints due to scale cap."
                            )
                            break
                if pair_count >= max_pairwise_constraints:
                    break
            if pair_count >= max_pairwise_constraints:
                break
        if pair_count > 0:
            notes.append(f"Applied {pair_count} min-uniques pairwise constraints.")

    # Soft target deviations.
    target_lambda = float(payload.policy.target_deviation_lambda)
    deviation_terms: list[cp_model.IntVar] = []
    if target_lambda > 0.0 and payload.target_count_by_player:
        for pid, target_count in payload.target_count_by_player.items():
            vars_with_pid = [
                var
                for key, var in x.items()
                if str(pid) in set(candidate_by_key[key].player_ids)
            ]
            if not vars_with_pid:
                continue
            count_var = model.NewIntVar(0, n_entries, f"count_{pid}")
            model.Add(count_var == sum(vars_with_pid))
            dev = model.NewIntVar(0, n_entries, f"dev_{pid}")
            model.Add(dev >= count_var - int(target_count))
            model.Add(dev >= int(target_count) - count_var)
            deviation_terms.append(dev)

    score_scale = 1000
    objective_terms = [
        int(round(float(candidate_by_key[key].objective_score) * score_scale)) * var
        for key, var in x.items()
    ]
    if deviation_terms:
        deviation_penalty = int(round(target_lambda * score_scale))
        objective_terms.extend([-deviation_penalty * dev for dev in deviation_terms])
    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 8.0
    solver.parameters.relative_gap_limit = 0.001
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        warnings.append("Selector infeasible; reverted to hold candidates.")
        return SelectorResult(
            status="infeasible",
            selected_candidate_ids_by_entry_id=_fallback_hold_selection(payload.candidates_by_entry_id),
            objective_value=None,
            warnings=warnings,
            notes=notes,
            infeasible=True,
        )

    selected: dict[str, str] = {}
    for entry_id in entries:
        for candidate in payload.candidates_by_entry_id[entry_id]:
            key = (entry_id, candidate.candidate_id)
            if solver.Value(x[key]) == 1:
                selected[entry_id] = candidate.candidate_id
                break

    return SelectorResult(
        status="optimal" if status == cp_model.OPTIMAL else "feasible",
        selected_candidate_ids_by_entry_id=selected,
        objective_value=float(solver.ObjectiveValue()),
        warnings=warnings,
        notes=notes,
        infeasible=False,
    )
