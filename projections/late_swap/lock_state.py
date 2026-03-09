"""Lock-state computation for late swap."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from projections.late_swap.models import LateSwapLockStateSummary


@dataclass(frozen=True)
class EntryLockState:
    entry_id: str
    locked_slots: list[str]
    unlocked_slots: list[str]
    locked_player_ids: list[str]
    unmapped_locked_slots: list[str]

    @property
    def is_fully_locked(self) -> bool:
        return len(self.unlocked_slots) == 0

    @property
    def is_degraded(self) -> bool:
        return len(self.unmapped_locked_slots) > 0


def build_lock_state(
    *,
    entries: list[dict[str, str]],
    dk_slots: list[str],
    entry_id_resolver: Callable[[dict[str, str], int], str],
    extract_draftable_id: Callable[[str], int | None],
    is_dk_locked: Callable[[str], bool],
    draftable_to_internal: dict[int, str],
    draftable_start_times: dict[int, datetime],
    internal_start_times: dict[str, datetime | None],
    now_utc: datetime,
) -> tuple[list[EntryLockState], LateSwapLockStateSummary]:
    states: list[EntryLockState] = []
    player_floor_count: dict[str, int] = {}
    locked_slots_by_entry: dict[str, list[str]] = {}
    unmapped_locked_by_entry: dict[str, list[str]] = {}
    fully_locked = 0
    locked_slots_total = 0
    degraded = 0

    for idx, entry in enumerate(entries):
        entry_id = entry_id_resolver(entry, idx)
        locked_slots: list[str] = []
        unlocked_slots: list[str] = []
        locked_player_ids: list[str] = []
        unmapped_locked_slots: list[str] = []
        for slot in dk_slots:
            raw_value = str(entry.get(slot, "")).strip()
            if not raw_value:
                unlocked_slots.append(slot)
                continue
            draftable_id = extract_draftable_id(raw_value)
            internal_id = draftable_to_internal.get(draftable_id) if draftable_id is not None else None
            slot_start = draftable_start_times.get(draftable_id) if draftable_id is not None else None
            if slot_start is None and internal_id:
                slot_start = internal_start_times.get(internal_id)
            locked = bool(is_dk_locked(raw_value) or (slot_start and slot_start <= now_utc))
            if not locked:
                unlocked_slots.append(slot)
                continue
            locked_slots.append(slot)
            if internal_id:
                locked_player_ids.append(str(internal_id))
                player_floor_count[str(internal_id)] = player_floor_count.get(str(internal_id), 0) + 1
            else:
                unmapped_locked_slots.append(slot)

        if len(unlocked_slots) == 0:
            fully_locked += 1
        if unmapped_locked_slots:
            degraded += 1
            unmapped_locked_by_entry[entry_id] = list(unmapped_locked_slots)
        locked_slots_total += len(locked_slots)
        locked_slots_by_entry[entry_id] = list(locked_slots)

        states.append(
            EntryLockState(
                entry_id=entry_id,
                locked_slots=locked_slots,
                unlocked_slots=unlocked_slots,
                locked_player_ids=list(dict.fromkeys(locked_player_ids)),
                unmapped_locked_slots=unmapped_locked_slots,
            )
        )

    entries_total = len(entries)
    locked_pct = (100.0 * float(locked_slots_total) / float(max(1, entries_total * len(dk_slots))))
    summary = LateSwapLockStateSummary(
        entries_total=entries_total,
        locked_slots_total=locked_slots_total,
        locked_slots_pct=locked_pct,
        entries_fully_locked=fully_locked,
        entries_with_unmapped_locked_slots=degraded,
        locked_slots_by_entry_id=locked_slots_by_entry,
        unmapped_locked_by_entry_id=unmapped_locked_by_entry,
        player_locked_floor_count=player_floor_count,
    )
    return states, summary
