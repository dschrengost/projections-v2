"""Candidate generation for Late Swap V2."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

from projections.late_swap.lock_state import EntryLockState
from projections.late_swap.models import (
    LateSwapCandidate,
    LateSwapCandidateSummary,
    LateSwapPolicy,
)
from projections.optimizer.cpsat_solver import solve_cpsat_iterative_counts
from projections.optimizer.optimizer_types import Constraints, OwnershipPenaltySettings


@dataclass(frozen=True)
class CandidateGenerationInput:
    entries_by_entry_id: dict[str, dict[str, str]]
    contest_by_entry_id: dict[str, str]
    lock_state_by_entry_id: dict[str, EntryLockState]
    policy: LateSwapPolicy
    dk_slots: list[str]
    player_pool: list[dict[str, Any]]
    internal_to_dk_player_id: dict[str, int]
    internal_to_name: dict[str, str]
    draftable_ids_by_player: dict[int, dict[str, int]]
    dk_names_by_player: dict[int, str]
    draftable_to_internal: dict[int, str]
    extract_draftable_id: Callable[[str], int | None]
    banned_ids_global: set[str]
    out_ids: set[str]
    only_out_lineups: bool


def _lineup_hash(player_ids: list[str]) -> str:
    token = ",".join(sorted(str(pid) for pid in player_ids))
    return hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]


def _entry_player_ids(
    *,
    entry: dict[str, str],
    dk_slots: list[str],
    extract_draftable_id: Callable[[str], int | None],
    draftable_to_internal: dict[int, str],
) -> list[str]:
    out: list[str] = []
    for slot in dk_slots:
        draftable_id = extract_draftable_id(str(entry.get(slot, "")).strip())
        if draftable_id is None:
            continue
        pid = draftable_to_internal.get(draftable_id)
        if pid:
            out.append(str(pid))
    return list(dict.fromkeys(out))


def _slot_values_from_lineup_players(
    *,
    lineup_players: list[object],
    dk_slots: list[str],
    internal_to_dk_player_id: dict[str, int],
    internal_to_name: dict[str, str],
    draftable_ids_by_player: dict[int, dict[str, int]],
    dk_names_by_player: dict[int, str],
) -> dict[str, str]:
    slot_values: dict[str, str] = {slot: "" for slot in dk_slots}
    for player in lineup_players:
        pid = str(getattr(player, "player_id", ""))
        slot = str(getattr(player, "pos", ""))
        if slot not in slot_values:
            return {}
        dk_pid = internal_to_dk_player_id.get(pid)
        if dk_pid is None:
            return {}
        draftable_id = draftable_ids_by_player.get(dk_pid, {}).get(slot)
        if draftable_id is None:
            return {}
        name = dk_names_by_player.get(dk_pid) or internal_to_name.get(pid) or pid
        slot_values[slot] = f"{name} ({draftable_id})"
    if any(not slot_values.get(slot, "") for slot in dk_slots):
        return {}
    return slot_values


def _build_hold_candidate(
    *,
    entry_id: str,
    contest_id: str,
    entry: dict[str, str],
    lock_state: EntryLockState,
    dk_slots: list[str],
    extract_draftable_id: Callable[[str], int | None],
    draftable_to_internal: dict[int, str],
    proj_by_internal: dict[str, float],
    own_by_internal: dict[str, float],
) -> LateSwapCandidate:
    player_ids = _entry_player_ids(
        entry=entry,
        dk_slots=dk_slots,
        extract_draftable_id=extract_draftable_id,
        draftable_to_internal=draftable_to_internal,
    )
    projected_score = float(sum(proj_by_internal.get(pid, 0.0) for pid in player_ids))
    total_own = float(sum(own_by_internal.get(pid, 0.0) for pid in player_ids))
    unlocked = [pid for pid in player_ids if pid not in set(lock_state.locked_player_ids)]
    return LateSwapCandidate(
        candidate_id=f"{entry_id}:hold",
        entry_id=entry_id,
        contest_id=contest_id,
        locked_slots=list(lock_state.locked_slots),
        slot_values={slot: str(entry.get(slot, "")) for slot in dk_slots},
        player_ids=player_ids,
        unlocked_player_ids=unlocked,
        generated_by="hold",
        projected_score=projected_score,
        remaining_score_mean=projected_score,
        total_own=total_own,
        swap_count=0,
        lineup_hash=_lineup_hash(player_ids),
        reason_codes=["hold_current_lineup"],
    )


def _make_candidate_from_lineup(
    *,
    entry_id: str,
    contest_id: str,
    dk_slots: list[str],
    generated_by: str,
    lineup: object,
    hold_candidate: LateSwapCandidate,
    lock_state: EntryLockState,
    internal_to_dk_player_id: dict[str, int],
    internal_to_name: dict[str, str],
    draftable_ids_by_player: dict[int, dict[str, int]],
    dk_names_by_player: dict[int, str],
    own_by_internal: dict[str, float],
) -> LateSwapCandidate | None:
    lineup_players = list(getattr(lineup, "players", []) or [])
    slot_values = _slot_values_from_lineup_players(
        lineup_players=lineup_players,
        dk_slots=dk_slots,
        internal_to_dk_player_id=internal_to_dk_player_id,
        internal_to_name=internal_to_name,
        draftable_ids_by_player=draftable_ids_by_player,
        dk_names_by_player=dk_names_by_player,
    )
    if not slot_values:
        return None
    for slot in lock_state.locked_slots:
        slot_values[slot] = hold_candidate.slot_values.get(slot, "")

    player_ids = [str(getattr(player, "player_id", "")) for player in lineup_players]
    player_ids = [pid for pid in dict.fromkeys(player_ids) if pid]
    hold_set = set(hold_candidate.player_ids)
    cand_set = set(player_ids)
    added = sorted(cand_set - hold_set)
    removed = sorted(hold_set - cand_set)
    swap_count = len(added)
    unlocked = [pid for pid in player_ids if pid not in set(lock_state.locked_player_ids)]
    projected = float(getattr(lineup, "total_proj", 0.0))
    total_own = float(sum(own_by_internal.get(pid, 0.0) for pid in player_ids))
    base_id = f"{entry_id}:{generated_by}:{_lineup_hash(player_ids)}"
    return LateSwapCandidate(
        candidate_id=base_id,
        entry_id=entry_id,
        contest_id=contest_id,
        locked_slots=list(lock_state.locked_slots),
        slot_values=slot_values,
        player_ids=player_ids,
        unlocked_player_ids=unlocked,
        generated_by=generated_by,  # type: ignore[arg-type]
        projected_score=projected,
        remaining_score_mean=projected,
        total_own=total_own,
        swap_count=swap_count,
        added_player_ids=added,
        removed_player_ids=removed,
        lineup_hash=_lineup_hash(player_ids),
        reason_codes=[f"generated_{generated_by}"],
    )


def generate_candidates_for_entries(
    payload: CandidateGenerationInput,
) -> tuple[dict[str, list[LateSwapCandidate]], LateSwapCandidateSummary]:
    requested = int(payload.policy.candidate_count_per_entry)
    stddev_available = any("stddev" in row for row in payload.player_pool)
    proj_by_internal = {
        str(player.get("player_id")): float(player.get("proj", 0.0))
        for player in payload.player_pool
    }
    own_by_internal = {
        str(player.get("player_id")): float(player.get("pred_own_pct", 0.0) or 0.0)
        for player in payload.player_pool
    }
    constrained_cap_pids = {
        str(pid)
        for pid, bounds in payload.policy.exposure_bounds.items()
        if bounds.max is not None
    }
    target_min_pids = [
        str(pid)
        for pid, bounds in payload.policy.exposure_bounds.items()
        if bounds.min is not None and float(bounds.min) > 0.0
    ]

    summary = LateSwapCandidateSummary()
    out: dict[str, list[LateSwapCandidate]] = {}

    for entry_id, entry in payload.entries_by_entry_id.items():
        summary.entries_total += 1
        summary.requested_total += requested
        lock_state = payload.lock_state_by_entry_id[entry_id]
        contest_id = payload.contest_by_entry_id[entry_id]
        hold = _build_hold_candidate(
            entry_id=entry_id,
            contest_id=contest_id,
            entry=entry,
            lock_state=lock_state,
            dk_slots=payload.dk_slots,
            extract_draftable_id=payload.extract_draftable_id,
            draftable_to_internal=payload.draftable_to_internal,
            proj_by_internal=proj_by_internal,
            own_by_internal=own_by_internal,
        )
        candidates: list[LateSwapCandidate] = [hold]
        seen_hashes: set[str] = {hold.lineup_hash}

        entry_player_ids = set(hold.player_ids)
        if payload.only_out_lineups and not (entry_player_ids & payload.out_ids):
            hold.reason_codes.append("skipped_only_out_filter")
            out[entry_id] = candidates
            summary.entries_with_candidates += 1
            summary.generated_total += len(candidates)
            summary.pass_counts["hold"] = summary.pass_counts.get("hold", 0) + 1
            continue

        if lock_state.is_fully_locked or lock_state.is_degraded:
            reason = "all_slots_locked" if lock_state.is_fully_locked else "degraded_unmapped_locked_slot"
            hold.reason_codes.append(reason)
            out[entry_id] = candidates
            summary.entries_with_candidates += 1
            summary.generated_total += len(candidates)
            summary.pass_counts["hold"] = summary.pass_counts.get("hold", 0) + 1
            continue

        locked_slot_map: dict[str, str] = {}
        for slot in lock_state.locked_slots:
            raw = str(entry.get(slot, "")).strip()
            draftable_id = payload.extract_draftable_id(raw)
            if draftable_id is None:
                continue
            pid = payload.draftable_to_internal.get(draftable_id)
            if pid:
                locked_slot_map[slot] = str(pid)

        started_or_banned = set(payload.banned_ids_global)
        started_or_banned -= set(lock_state.locked_player_ids)

        pass_defs: list[tuple[str, int, float, set[str], list[str], bool]] = []
        random_n = max(1, requested - 4)
        pass_defs.append(("best_ev", 1, 0.0, set(), [], False))
        pass_defs.append(("randomized", random_n, 20.0 if stddev_available else 0.0, set(), [], False))

        cap_relief_bans = set(lock_state.locked_player_ids)
        cap_relief_bans = (entry_player_ids & constrained_cap_pids) - set(lock_state.locked_player_ids)
        if cap_relief_bans:
            pass_defs.append(("cap_relief", 1, 0.0, cap_relief_bans, [], False))

        if target_min_pids:
            required = [pid for pid in target_min_pids if pid not in set(lock_state.locked_player_ids)]
            if required:
                pass_defs.append(("target_recovery", 1, 0.0, set(), [required[0]], False))

        if payload.policy.mode in {"catch_up", "decorrelated_ev"}:
            pass_defs.append(("low_own", 1, 0.0, set(), [], True))

        for pass_name, n_lineups, randomness, extra_bans, extra_locks, own_penalty in pass_defs:
            if len(candidates) >= requested:
                break
            if n_lineups <= 0:
                continue
            constraints = Constraints(
                N_lineups=int(n_lineups),
                unique_players=1,
                randomness_pct=float(randomness if stddev_available else 0.0),
                min_salary=0,
            )
            constraints.lock_slots = dict(locked_slot_map)
            constraints.lock_ids = sorted(set(lock_state.locked_player_ids).union(extra_locks))
            constraints.ban_ids = sorted(started_or_banned.union(extra_bans))
            constraints.ownership_penalty = OwnershipPenaltySettings(enabled=own_penalty, mode="by_percent")
            constraints.validate("dk", stddev_available=stddev_available)

            solved_lineups, _diagnostics = solve_cpsat_iterative_counts(
                payload.player_pool,
                constraints,
                seed=0,
                site="dk",
            )
            summary.pass_counts[pass_name] = summary.pass_counts.get(pass_name, 0) + 1
            for lineup in solved_lineups:
                candidate = _make_candidate_from_lineup(
                    entry_id=entry_id,
                    contest_id=contest_id,
                    dk_slots=payload.dk_slots,
                    generated_by=pass_name,
                    lineup=lineup,
                    hold_candidate=hold,
                    lock_state=lock_state,
                    internal_to_dk_player_id=payload.internal_to_dk_player_id,
                    internal_to_name=payload.internal_to_name,
                    draftable_ids_by_player=payload.draftable_ids_by_player,
                    dk_names_by_player=payload.dk_names_by_player,
                    own_by_internal=own_by_internal,
                )
                if candidate is None:
                    summary.rejected_unassignable_total += 1
                    continue
                if payload.policy.max_swaps_per_entry is not None and candidate.swap_count > int(
                    payload.policy.max_swaps_per_entry
                ):
                    summary.rejected_swap_limit_total += 1
                    continue
                if candidate.lineup_hash in seen_hashes:
                    summary.deduped_total += 1
                    continue
                seen_hashes.add(candidate.lineup_hash)
                candidates.append(candidate)
                if len(candidates) >= requested:
                    break

        out[entry_id] = candidates
        summary.entries_with_candidates += 1
        summary.generated_total += len(candidates)

    return out, summary
