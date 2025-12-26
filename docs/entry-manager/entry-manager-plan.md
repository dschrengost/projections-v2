## Entry Manager Implementation Plan (DK NBA Classic)

### Goals
- Upload and persist DraftKings entry CSV files.
- Provide a durable state model for lineups, locks, and slot metadata.
- Support editing/updating lineups and exporting updated CSV.
- Serve as the foundation for late swap and contest-aware enhancements.

### Inputs
- DK entry CSV (headers: Entry ID, Contest Name, Contest ID, Entry Fee, PG, SG, SF, PF, C, G, F, UTIL).
- Optimizer player pool (for mapping dk_id → player_id, team, game start).
- Schedule data for start times (already scraped).

### Storage (projections-data)
```
/home/daniel/projections-data/entries/{date}/dk/{contest_id}.json
```
Fields:
- game_date, draft_group_id, site, updated_at, client_revision
- contests (name, id, fee)
- entries (entry_id + slots)
- original_csv_checksum (for provenance)

### Data Model (Python)
```python
@dataclass
class EntrySlot:
    slot: str                # PG, SG, SF, PF, C, G, F, UTIL
    dk_id: str               # DK player id
    name: str                # "Player Name"
    player_id: Optional[str] # internal id if resolved
    team: Optional[str]
    game_start_utc: Optional[str]
    is_locked: bool

@dataclass
class Entry:
    entry_id: str
    contest_id: str
    contest_name: str
    entry_fee: Optional[float]
    slots: Dict[str, EntrySlot]

@dataclass
class EntryFileState:
    game_date: str
    draft_group_id: int
    site: str = "dk"
    updated_at: str
    client_revision: int
    entries: List[Entry]
```

### Parsing Workflow
1. Accept CSV upload.
2. Parse header row and normalize slot columns.
3. Extract dk_id from `Name (DK_ID)` format.
4. Resolve dk_id → player_id/team/start via player pool/schedule.
5. Persist as EntryFileState.

### API Endpoints (Phase 1)
1. `POST /api/optimizer/entries/upload`
   - Input: date, draft_group_id, CSV file
   - Output: parsed entry state
2. `GET /api/optimizer/entries?date=YYYY-MM-DD`
   - List available entry files
3. `GET /api/optimizer/entries/{contest_id}?date=YYYY-MM-DD`
   - Load entry file state
4. `PUT /api/optimizer/entries/{contest_id}`
   - Update entry state (manual edits or overrides)
5. `POST /api/optimizer/entries/{contest_id}/export`
   - Return DK CSV with updated lineups
6. `POST /api/optimizer/entries/{contest_id}/apply-build`
   - Input: build source (optimizer or contest-sim) + build id
   - Output: updated entry state with lineups applied

### UI (Phase 1)
- New “Entry Manager” panel or page:
  - Upload CSV
  - List saved contests
  - View entries (locked vs editable)
  - Export updated CSV
  - Apply saved build lineups (optimizer or contest sim)

### Phase 2 Enhancements
- Per-entry edits in UI (slot-level swaps).
- Global exposure view across entries.
- Versioning and change history.

### Saved Build Integration
- Load lineups from optimizer saved builds (`/api/optimizer/saved-builds`).
- Load lineups from contest sim saved builds (`/api/contest-sim/saved-builds`).
- Entry manager applies selected build lineups to entries (primary population path).

### Decisions
- Store date + slate in metadata, but use contest_id as the primary key.
- Enforce one slate per CSV for now.
- Persist original DK CSV row order for export fidelity.

### Late Swap Notes (In-Slate)
- DK late swap locks are **slot-level**: once a game starts, that roster slot cannot change.
- Lock detection should use either DK's `LOCKED` marker (if present in the CSV) or `game_start_utc <= server time`.
- Optimization should enforce slot locks in the solver (not via post-processing) so exported lineups remain valid and truly optimized under constraints.
- Late swap pool building should keep salary-only / zero-projection rows available so locked players can always be represented, then ban inactive/out players unless they are locked.
