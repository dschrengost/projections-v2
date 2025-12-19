## Late Swap Implementation Plan (DK NBA Classic)

### Goals
- Support late swap for DraftKings NBA Classic using the existing optimizer.
- Start with per-entry optimization (independent entries, no exposure constraints).
- Use FPTS-only objective for remaining slots, with locked players as constraints.
- Persist entry state so users can re-run swap later without re-uploading.

### Inputs
- DK entry CSV (format like `projections/DK_entries_fixed.csv`).
- Schedule/game start times from existing scraped schedule data.
- Current projections/ownership from optimizer player pool.

### Storage (projections-data)
Proposed location:
```
/home/daniel/projections-data/entries/{date}/dk/{contest_id}.json
```
Contents:
- Original CSV metadata (contest, entry ids, entry fee).
- Parsed entries with slot -> player mapping (player id, name, dk_id).
- Lock status per slot (started vs future).
- Last updated timestamp and revision.

### Parsing DK Entry CSV
DK columns:
```
Entry ID, Contest Name, Contest ID, Entry Fee, PG, SG, SF, PF, C, G, F, UTIL
```
Each slot value is `"Player Name (DK_ID)"`.

Parser tasks:
- Extract `dk_id` from parentheses.
- Keep display name as-is.
- Map slot names to DK roster slots.
- Preserve entry-level metadata (contest id/name, entry id, fee).

### Data Model (Python)
```python
@dataclass
class EntrySlot:
    slot: str
    dk_id: str
    name: str
    player_id: Optional[str]
    team: Optional[str]
    game_start_utc: Optional[str]
    is_locked: bool

@dataclass
class Entry:
    entry_id: str
    contest_id: str
    contest_name: str
    entry_fee: Optional[float]
    slots: Dict[str, EntrySlot]  # PG, SG, SF, PF, C, G, F, UTIL

@dataclass
class EntryFileState:
    game_date: str
    site: str = "dk"
    draft_group_id: int
    updated_at: str
    client_revision: int
    entries: List[Entry]
```

### Late Swap Logic (Phase 1)
For each entry:
1. Determine lock status:
   - Slot is locked if player’s game has started.
   - Use schedule start times mapped by `team` or `game_id`.
2. Build optimizer request:
   - Lock players in started slots.
   - Restrict remaining slots to future games only (optional).
3. Run optimizer with FPTS-only objective.
4. Return new lineup for remaining slots + preserve locked players.

we should return 5 candidate lineups per entry and show the diff in some way (who was swapped and for whom)

### Optimizer Integration
Changes in optimizer service:
- Add a late-swap endpoint to accept entry file state (or reference).
- Use `build_player_pool` with `use_user_overrides` if enabled.
- For each entry:
  - Build per-entry lock list from locked slots.
  - Call optimizer with `lock_ids`.

### API Endpoints (Phase 1)
1. `POST /api/optimizer/entries/upload`
   - Upload DK CSV, parse, persist state.
2. `GET /api/optimizer/entries`
   - List saved entry files for a date.
3. `GET /api/optimizer/entries/{contest_id}`
   - Load entry file state.
4. `POST /api/optimizer/late-swap`
   - Input: date, draft_group_id, contest_id (or inline entry state), site.
   - Output: updated entries + exportable CSV.

### UI (Phase 1)
Minimal “Late Swap” panel under Optimizer:
- Upload CSV → show parsed entries.
- Show locked vs unlockable slots.
- Run “Late Swap” button → downloads updated DK CSV.

### Future Phases
Phase 2:
- Add global exposure limits across entries.
- Add “swap only within existing teams/games” constraints.

Phase 3:
- Contest-aware swap (in-game standings, win-prob objective).
- Pull current contest files from DK/3rd-party if accessible.

### Open Questions
- Where to persist schedule start times for the swap engine (reuse existing schedule artifacts).

use best practice

- Whether to require mapping DK IDs to internal `player_id` or use DK IDs only.

DK IDs only if possible

- Expected scale (number of entries per swap run) for performance tuning.

a maxiumum of around 300 entries. this shouldn't be too much of an issue.