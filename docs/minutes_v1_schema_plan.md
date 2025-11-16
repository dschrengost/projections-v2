# Minutes V1 — Shared Schema Objects Plan

Owner: data eng  •  Status: ready to implement  •  Updated: 2025-02-15

> Goal: move the informal schema definitions in `minutes_v1_spec.md` into reusable Python objects that builders, validations, and tests all consume. This ensures we never drift from the data contract and makes CI guardrails easier to extend.

---

## 1. Scope & Target Datasets

Authoritative schemas live in §1 of `minutes_v1_spec.md`. For this project we will cover all tables that touch the Minutes V1 pipeline end-to-end:

| Tier | Table | Notes |
| --- | --- | --- |
| Bronze | `injuries_raw`, `odds_raw`, `roster_nightly` | Built by `projections.minutes_v1.smoke_dataset.SmokeDatasetBuilder`. |
| Silver | `injuries_snapshot`, `odds_snapshot`, `roster_nightly` snapshot outputs | Same columns as bronze + snapshot metadata. |
| Gold | `boxscore_labels`, `features_minutes_v1` | Labels already frozen; features schema inferred from current parquet. |
| Static | `schedule`, `coach_tenure`, optional arena mapping | Provide read-only schemas to keep everything centralized. |

Future tables (CI metrics, monitoring, t‑15 snapshots, etc.) should hook into the same pattern once defined.

---

## 2. Deliverables

1. **Shared schema module:** `projections/minutes_v1/schemas.py` exporting a `Schema` object per dataset (detailed in §3).
2. **Helper utilities:** type aliases and convenience functions (`as_pandas_dtypes`, `enforce_schema(df, schema)`, etc.).
3. **Integration into builders:** `smoke_dataset.py`, `pipelines/build_features_minutes_v1.py`, and any CLI that writes parquet must call the enforcement helpers before persisting.
4. **Contract tests:** add or update tests under `tests/minutes_v1/test_schema_contract.py` (or create it) to validate a sample parquet/df against the shared schemas.
5. **Docs update:** reference the new module in `minutes_v1_spec.md` (short note near §1) so future agents know the implementation lives in code.

---

## 3. Schema Representation Requirements

Use a thin dataclass wrapper so we can express:

```python
@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: tuple[str, ...]        # ordered for parquet writes
    pandas_dtypes: dict[str, str]   # e.g., {"game_id": "string[pyarrow]"}
    primary_key: tuple[str, ...]
    defaults: dict[str, Any] = field(default_factory=dict)
```

Key rules:
- Dtypes should be Arrow-friendly (`string[pyarrow]`, `Int64`, `boolean`) so `.astype` works cross-platform.
- Provide constants like `INJURIES_RAW_SCHEMA = TableSchema(...)`.
- Add helper `def enforce(df: pd.DataFrame, schema: TableSchema, *, allow_missing_optional=False) -> pd.DataFrame` that:
  1. Ensures all required columns exist (fail fast with descriptive error).
  2. Orders columns according to `schema.columns`.
  3. Casts dtypes using `df.astype(schema.pandas_dtypes)`.
  4. Optionally fills defaults for missing optional columns.
- Store PK info for future uniqueness checks (tests can call `assert df.duplicated(list(schema.primary_key)).sum() == 0`).

---

## 4. Implementation Steps

1. **Create module skeleton**
   - File: `projections/minutes_v1/schemas.py`
   - Include `TableSchema` dataclass, dtype helper constants (`UTC_TS = "datetime64[ns, UTC]"`, etc.), and one schema per dataset.
   - Source column lists from `minutes_v1_spec.md` and current parquet headers (for `features_minutes_v1` use `pd.read_parquet` on a recent file, or maintain a curated list).

2. **Wire builders**
   - In `smoke_dataset.py`, right before writing each bronze/silver parquet, call `enforce()` for the relevant schema.
   - In `pipelines/build_features_minutes_v1.py`, enforce `FEATURES_MINUTES_V1_SCHEMA` before writing gold features.
   - Add `TODO/NOTE` near `coach_tenure` warning to point at schema once that CSV is available.

3. **Testing**
   - Create/extend `tests/minutes_v1/test_schema_contract.py`.
   - For each schema, load a small fixture frame (existing parquet slice or generated data), run `enforce()`, and assert:
     - Columns match exactly.
     - Dtypes match declared `pandas_dtypes`.
     - PK uniqueness holds via `.duplicated`.
   - Add regression test ensuring new schemas stay in sync with spec (e.g., `assert set(INJURIES_RAW_SCHEMA.columns) == {...}` referencing spec constants).

4. **Documentation touchups**
   - Insert a short sentence into `minutes_v1_spec.md §1` linking to the schema module (“See `projections/minutes_v1/schemas.py` for the live definitions used by builders and tests.”).
   - If desired, add a `README` snippet in `docs/` describing how to add new schemas.

5. **(Optional) Pandera integration**
   - If time permits, create `pandera.DataFrameSchema` objects using the same metadata to enable richer validation in CI. This can be a follow-up ticket.

---

## 5. Acceptance Criteria

- Every parquet writer invoked by `projections.cli.build_month` runs through the schema enforcement helper without raising.
- Running `uv run pytest tests/minutes_v1/test_schema_contract.py -q` passes and would fail if any column list/dtype drifts.
- `minutes_v1_spec.md` references the shared schema module so spec + code stay aligned.
- Future agents can import `from projections.minutes_v1.schemas import INJURIES_RAW_SCHEMA` and get authoritative metadata without digging through markdown.

---

## 6. Suggested Command Sequence

```bash
# 1. Generate column lists (optional exploratory)
uv run python scripts/inspect_features_schema.py

# 2. Run unit tests focused on schema enforcement
uv run pytest tests/minutes_v1/test_schema_contract.py -q

# 3. Build a month to ensure writers obey schemas
uv run python -m projections.cli.build_month 2023 12 --season 2023 --skip-bronze
```

*(Adjust commands as needed; include `--skip-bronze` if only gold enforcement changed.)*

---

## 7. Open Questions / Follow-Ups

1. Should we formalize schema versioning (e.g., embed `schema_version` column)?—not required for first pass.
2. Do we want automatic PK uniqueness tests in CI for every parquet write?—recommended but can be separate PR.
3. Once `coach_tenure.csv` lands, add its schema to this module to keep everything centralized.

---

**Hand-off Note:** The next agent should start by implementing the `TableSchema` class and the schema constants. After that, wiring enforcement into the builders and adding the tests should be incremental. Reach out if parquet column order differs between months; we may need a helper to union columns safely.
