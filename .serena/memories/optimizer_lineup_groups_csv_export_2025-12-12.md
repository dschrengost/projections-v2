Added lineup selection + grouping + CSV export in `web/minutes-dashboard/src/pages/OptimizerPage.tsx`.

UI (Lineups toolbar):
- Per-lineup checkbox selection.
- Buttons: `Select showing`, `Select filtered`, `Clear selection`, `Export selected CSV`.
- Group controls: group dropdown, `Save group` (from current selection), `Select group` (load group into selection), `Export group CSV`, `Delete group`.

Notes:
- Groups are in-memory for the current loaded lineups; selection/groups reset when `lineups` changes (new build/load/join).
- CSV output matches DK upload format: header `PG,SG,SF,PF,C,G,F,UTIL` and cells `Player Name (DK_ID)` using bipartite slot assignment.

Styling updates in `web/minutes-dashboard/src/App.css`:
- Added wrapping for toolbar/filter row and styles for the new action buttons + selected lineup card state.