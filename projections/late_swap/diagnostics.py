"""Diagnostics and exposure-state helpers for late swap."""

from __future__ import annotations

from collections import Counter
from typing import Callable

from projections.late_swap.models import (
    ExposureStateRow,
    LateSwapCandidate,
    LateSwapDiagnostics,
    LateSwapPolicy,
    LateSwapSelectionSummary,
    pct_to_max_count,
    pct_to_min_count,
)


def _exposure_pct(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * float(count) / float(total)


def exposure_counts_from_entries(
    *,
    entries_by_entry_id: dict[str, dict[str, str]],
    dk_slots: list[str],
    extract_draftable_id: Callable[[str], int | None],
    draftable_to_internal: dict[int, str],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for entry in entries_by_entry_id.values():
        seen_entry: set[str] = set()
        for slot in dk_slots:
            raw = str(entry.get(slot, "")).strip()
            draftable_id = extract_draftable_id(raw)
            if draftable_id is None:
                continue
            pid = draftable_to_internal.get(draftable_id)
            if not pid:
                continue
            seen_entry.add(str(pid))
        for pid in seen_entry:
            counts[pid] += 1
    return dict(counts)


def exposure_counts_from_selection(
    *,
    selected_candidate_ids_by_entry_id: dict[str, str],
    candidates_by_entry_id: dict[str, list[LateSwapCandidate]],
) -> dict[str, int]:
    by_candidate_id: dict[str, LateSwapCandidate] = {}
    for candidates in candidates_by_entry_id.values():
        for candidate in candidates:
            by_candidate_id[candidate.candidate_id] = candidate

    counts: Counter[str] = Counter()
    for candidate_id in selected_candidate_ids_by_entry_id.values():
        candidate = by_candidate_id.get(candidate_id)
        if candidate is None:
            continue
        for pid in set(candidate.player_ids):
            counts[str(pid)] += 1
    return dict(counts)


def derive_target_count_by_player(
    *,
    policy: LateSwapPolicy,
    total_entries: int,
    source_target_count_by_player: dict[str, int] | None,
    current_committed_count_by_player: dict[str, int],
) -> dict[str, int]:
    targets: dict[str, int] = {}
    if policy.target_source in {"source_portfolio", "explicit"} and source_target_count_by_player:
        targets = {str(pid): int(count) for pid, count in source_target_count_by_player.items()}
    elif policy.target_source == "current_entries":
        targets = {str(pid): int(count) for pid, count in current_committed_count_by_player.items()}

    if policy.target_source == "explicit":
        for pid, bounds in policy.exposure_bounds.items():
            if bounds.min is None and bounds.max is None:
                continue
            if bounds.min is not None and bounds.max is not None:
                pct = 0.5 * (float(bounds.min) + float(bounds.max))
            elif bounds.min is not None:
                pct = float(bounds.min)
            else:
                pct = float(bounds.max or 0.0)
            targets[str(pid)] = int(round((pct / 100.0) * float(total_entries)))

    return targets


def build_exposure_diagnostics(
    *,
    policy: LateSwapPolicy,
    total_entries: int,
    player_name_by_id: dict[str, str],
    source_target_count_by_player: dict[str, int],
    locked_floor_count_by_player: dict[str, int],
    current_committed_count_by_player: dict[str, int],
    proposed_final_count_by_player: dict[str, int],
) -> LateSwapDiagnostics:
    rows: list[ExposureStateRow] = []
    warnings: list[str] = []
    forced_over: list[str] = []
    constrained_pids = set(policy.exposure_bounds.keys())
    all_players = (
        set(source_target_count_by_player)
        | set(locked_floor_count_by_player)
        | set(current_committed_count_by_player)
        | set(proposed_final_count_by_player)
        | constrained_pids
    )

    for pid in sorted(all_players):
        source_target_count = int(source_target_count_by_player.get(pid, 0))
        locked_floor_count = int(locked_floor_count_by_player.get(pid, 0))
        current_count = int(current_committed_count_by_player.get(pid, 0))
        proposed_count = int(proposed_final_count_by_player.get(pid, 0))
        bounds = policy.exposure_bounds.get(pid)

        source_target_pct = _exposure_pct(source_target_count, total_entries) if source_target_count else None
        locked_floor_pct = _exposure_pct(locked_floor_count, total_entries)
        current_pct = _exposure_pct(current_count, total_entries)
        proposed_pct = _exposure_pct(proposed_count, total_entries)
        delta_vs_target_pct = (
            (proposed_pct - source_target_pct) if source_target_pct is not None else None
        )

        status = "within_target"
        forced_by_locks = False
        hard_min = float(bounds.min) if bounds and bounds.min is not None else None
        hard_max = float(bounds.max) if bounds and bounds.max is not None else None
        min_achievable = locked_floor_pct
        if hard_max is not None:
            cap_count = pct_to_max_count(hard_max, total_entries)
            if locked_floor_count > cap_count:
                forced_by_locks = True
                forced_over.append(pid)
                status = "forced_over_by_locks"
                warnings.append(
                    f"Player {pid} requested max {hard_max:.1f}% is below lock floor {locked_floor_pct:.1f}%."
                )
            elif proposed_pct > hard_max + 1e-6:
                status = "over_target"
        if status == "within_target" and hard_min is not None and proposed_pct + 1e-6 < hard_min:
            status = "under_target"
        if (
            status == "within_target"
            and source_target_pct is not None
            and delta_vs_target_pct is not None
        ):
            if delta_vs_target_pct > 3.0:
                status = "added_post_lock"
            elif delta_vs_target_pct < -3.0:
                status = "removed_post_lock"

        rows.append(
            ExposureStateRow(
                player_id=str(pid),
                player_name=player_name_by_id.get(str(pid), str(pid)),
                source_target_pct=source_target_pct,
                locked_floor_pct=locked_floor_pct,
                current_committed_pct=current_pct,
                proposed_final_pct=proposed_pct,
                delta_vs_target_pct=delta_vs_target_pct,
                status=status,
                hard_min_pct=hard_min,
                hard_max_pct=hard_max,
                forced_over_cap_by_locks=forced_by_locks,
                min_achievable_pct=min_achievable,
            )
        )

    return LateSwapDiagnostics(
        warnings=warnings,
        forced_over_cap_players=sorted(set(forced_over)),
        exposure_states=rows,
    )


def summarize_selection(
    *,
    selected_candidate_ids_by_entry_id: dict[str, str],
    candidates_by_entry_id: dict[str, list[LateSwapCandidate]],
    objective_value: float | None,
    status: str,
    current_committed_count_by_player: dict[str, int],
    proposed_final_count_by_player: dict[str, int],
) -> LateSwapSelectionSummary:
    by_candidate_id: dict[str, LateSwapCandidate] = {}
    for candidates in candidates_by_entry_id.values():
        for candidate in candidates:
            by_candidate_id[candidate.candidate_id] = candidate

    swapped = 0
    total_swaps = 0
    projected_delta_total = 0.0
    duplicates_before = max(0, len(current_committed_count_by_player) - len(set(current_committed_count_by_player)))
    duplicates_after = max(0, len(proposed_final_count_by_player) - len(set(proposed_final_count_by_player)))

    for entry_id, candidate_id in selected_candidate_ids_by_entry_id.items():
        candidate = by_candidate_id.get(candidate_id)
        if candidate is None:
            continue
        total_swaps += int(candidate.swap_count)
        if candidate.swap_count > 0:
            swapped += 1
        if candidate.projected_score is not None:
            hold = next(
                (
                    item
                    for item in candidates_by_entry_id.get(entry_id, [])
                    if item.generated_by == "hold"
                ),
                None,
            )
            if hold and hold.projected_score is not None:
                projected_delta_total += float(candidate.projected_score) - float(hold.projected_score)

    entries_total = len(selected_candidate_ids_by_entry_id)
    held = max(0, entries_total - swapped)
    return LateSwapSelectionSummary(
        status="optimal" if status == "optimal" else "feasible" if status == "feasible" else "fallback_hold",
        entries_total=entries_total,
        entries_swapped=swapped,
        entries_held=held,
        total_swaps=total_swaps,
        average_swaps_per_changed_entry=(float(total_swaps) / float(swapped)) if swapped > 0 else 0.0,
        objective_value=objective_value,
        projected_delta_total=projected_delta_total,
        projected_delta_avg=(projected_delta_total / float(entries_total)) if entries_total else 0.0,
        max_exposure_before_pct=max((_exposure_pct(v, entries_total) for v in current_committed_count_by_player.values()), default=0.0),
        max_exposure_after_pct=max((_exposure_pct(v, entries_total) for v in proposed_final_count_by_player.values()), default=0.0),
        duplicate_count_before=duplicates_before,
        duplicate_count_after=duplicates_after,
        infeasibility_count=0,
    )


def validate_policy_feasibility(
    *,
    policy: LateSwapPolicy,
    total_entries: int,
    locked_floor_count_by_player: dict[str, int],
    candidate_coverage_count_by_player: dict[str, int],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for pid, bound in policy.exposure_bounds.items():
        floor_count = int(locked_floor_count_by_player.get(str(pid), 0))
        coverage = int(candidate_coverage_count_by_player.get(str(pid), 0))
        if bound.max is not None:
            cap = pct_to_max_count(float(bound.max), total_entries)
            if floor_count > cap:
                warnings.append(
                    f"Player {pid} cap {bound.max:.1f}% is below lock floor {100.0 * floor_count / max(1, total_entries):.1f}%."
                )
        if bound.min is not None:
            min_count = pct_to_min_count(float(bound.min), total_entries)
            if coverage < min_count:
                errors.append(
                    f"Player {pid} min {bound.min:.1f}% infeasible: only {coverage}/{total_entries} entries have candidate coverage."
                )
    return errors, warnings
