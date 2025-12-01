Initial DraftKings NBA scraper scaffolding added in projections-v2 (feature/dk-scraper branch).

Key modules:
- projections/dk/api.py: requests wrappers for lobby contests and draftables endpoints with default headers/timeouts.
- projections/dk/slates.py: normalizes contests payload, parses start times from `sd` (/Date(ms)/) to America/New_York, infers slate type from contest name, aggregates draft groups per date.

CLIs:
- scripts/dk/list_slates.py: lists draft groups for a given game date/slate type; writes raw contests JSON to <data_root>/bronze/dk/contests/contests_raw_<date>.json.
- scripts/dk/fetch_draftables.py: resolves draft group by id or date+slate-type, fetches draftables, writes raw JSON to <data_root>/bronze/dk/draftables/draftables_raw_<dg>.json, optional CSV flatten.

Notes:
- SlateType supports main/night/turbo/early/showdown/all.
- Uses projections.paths.get_data_root to resolve PROJECTIONS_DATA_ROOT; default bronze path under that root.
- Error handling raises RuntimeError when expected keys missing (e.g., Contests list, draftables list).
- Smoke tests not run in-session due to missing requests/pandas in environment.
