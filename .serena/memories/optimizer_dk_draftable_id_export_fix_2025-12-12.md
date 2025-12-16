Fixed DraftKings ID mismatch in optimizer CSV exports.

Root cause:
- DK upload expects slot-specific `draftableId` (roster IDs), not the per-player `playerId`.
- Existing exporter used `dk_player_id` (playerId), so CSV IDs didn’t match DraftKings.

Backend changes:
- `projections/api/optimizer_api.py`: CSV export now loads bronze draftables (`projections-data/bronze/dk/draftables/draftables_raw_{draft_group_id}.json`) and uses `draftableId` per slot (PG/SG/SF/PF/C/G/F/UTIL).
- Added `POST /api/optimizer/export` endpoint to export arbitrary lineups (used by UI for selected/group export).
- `projections/api/optimizer_service.py`: fixed pool-building merge bug where `salary` from projections (NaN) incorrectly overrode salary parquet due to `or` truthiness; now prefers `salary_sal` etc.

Frontend changes:
- `web/minutes-dashboard/src/pages/OptimizerPage.tsx`: selected/group CSV export now calls backend `POST /api/optimizer/export` instead of building CSV client-side.
- `web/minutes-dashboard/src/api/optimizer.ts`: added `exportCustomLineupsCSV()` helper.

Docs:
- `docs/optimizer/README.md`: updated export endpoint notes.

Validation:
- Generated CSV from a real saved build and verified IDs are `411xxxxx`-style `draftableId` values per slot.