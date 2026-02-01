from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.dataset as ds

from projections import paths
from projections.pbp.identity import normalize_name

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlayerIdMapResult:
    """Season-scoped mapping from NBA personId -> internal player_id."""

    season_start_year: int
    person_id_to_internal_id: dict[int, int]
    mapping: pd.DataFrame
    unmapped_person_ids: pd.DataFrame


def _season_start_year_from_season_id(value: object) -> Optional[int]:
    """Parse 'YYYY-YY' or 'YYYY-YYYY' into a start year."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Common format: 2024-25
    if "-" in s:
        left = s.split("-", 1)[0].strip()
        if left.isdigit():
            return int(left)
    # Fallback: '2024'
    if s.isdigit() and len(s) == 4:
        return int(s)
    return None


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _load_overrides(*, data_root: Path) -> dict[int, str]:
    overrides_path = (
        data_root
        / "bronze"
        / "pbp_vendor"
        / "player_overrides"
        / "player_id_map_overrides.csv"
    )
    if not overrides_path.exists():
        return {}
    df = pd.read_csv(overrides_path)
    needed = {"normalized_name", "nba_person_id"}
    missing = [c for c in sorted(needed) if c not in df.columns]
    if missing:
        raise ValueError(f"Overrides file missing columns {missing}: {overrides_path}")
    df = df.copy()
    df["nba_person_id"] = pd.to_numeric(df["nba_person_id"], errors="coerce").astype("Int64")
    df["normalized_name"] = df["normalized_name"].astype("string").fillna("").str.strip()
    df = df.dropna(subset=["nba_person_id"]).copy()
    df["nba_person_id"] = df["nba_person_id"].astype(int)
    df = df[df["normalized_name"] != ""].copy()
    dupes = df.groupby("nba_person_id")["normalized_name"].nunique()
    bad = dupes[dupes > 1]
    if not bad.empty:
        raise ValueError(
            "Overrides file has ambiguous nba_person_id -> normalized_name mappings: "
            f"{bad.index.tolist()[:20]} (and {max(0, len(bad) - 20)} more). Path={overrides_path}"
        )
    return dict(zip(df["nba_person_id"].tolist(), df["normalized_name"].tolist()))


def _load_person_source_table(
    *,
    season_start_year: int,
    data_root: Path,
    person_source: Path | None,
) -> pd.DataFrame:
    if person_source is None:
        base = data_root / "silver" / "nba_daily_lineups" / f"season={int(season_start_year)}"
        if not base.exists():
            raise FileNotFoundError(f"Missing expected personId source directory: {base}")
        dataset = ds.dataset(str(base), format="parquet")
        cols = ["player_id", "player_name", "season_start"]
        table = dataset.to_table(columns=cols)
        df = table.to_pandas()
    else:
        p = Path(person_source).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--person-source not found: {p}")
        if p.is_dir():
            dataset = ds.dataset(str(p), format="parquet")
            table = dataset.to_table()
            df = table.to_pandas()
        else:
            if p.suffix.lower() == ".parquet":
                df = pd.read_parquet(p)
            elif p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
            else:
                raise ValueError(f"Unsupported --person-source file type: {p}")

    # Normalize to expected columns: nba_person_id + player_name + season_start.
    df = df.copy()
    if "nba_person_id" not in df.columns:
        if "player_id" in df.columns:
            df = df.rename(columns={"player_id": "nba_person_id"})
        elif "person_id" in df.columns:
            df = df.rename(columns={"person_id": "nba_person_id"})

    if "player_name" not in df.columns:
        for cand in ("name", "full_name", "canonical_name"):
            if cand in df.columns:
                df = df.rename(columns={cand: "player_name"})
                break

    if "season_start" not in df.columns:
        if "season" in df.columns:
            df = df.rename(columns={"season": "season_start"})
        else:
            df["season_start"] = int(season_start_year)

    missing = [c for c in ["nba_person_id", "player_name", "season_start"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"person source missing required columns {missing}. "
            f"Need at least nba_person_id, player_name, season_start. Got columns={list(df.columns)}"
        )

    df["season_start"] = pd.to_numeric(df["season_start"], errors="coerce").astype("Int64")
    df = df[df["season_start"] == int(season_start_year)].copy()
    df["nba_person_id"] = pd.to_numeric(df["nba_person_id"], errors="coerce").astype("Int64")
    df["player_name"] = df["player_name"].astype("string").fillna("").str.strip()
    df = df.dropna(subset=["nba_person_id"]).copy()
    df["nba_person_id"] = df["nba_person_id"].astype(int)
    df = df[df["player_name"] != ""].copy()
    return df[["nba_person_id", "player_name", "season_start"]].copy()


def _load_internal_players_dim(*, data_root: Path) -> pd.DataFrame:
    pbp_root = data_root / "artifacts" / "pbp_v1"
    latest_path = pbp_root / "LATEST_PUBLISHED"
    if not latest_path.exists():
        raise FileNotFoundError(f"Missing pbp_v1 LATEST_PUBLISHED pointer: {latest_path}")
    run_id = latest_path.read_text(encoding="utf-8").strip()
    if not run_id:
        raise ValueError(f"Empty pbp_v1 LATEST_PUBLISHED pointer: {latest_path}")
    dim_path = pbp_root / run_id / "players_dim.parquet"
    if not dim_path.exists():
        raise FileNotFoundError(f"Missing pbp_v1 players_dim.parquet: {dim_path}")

    cols = ["player_id", "canonical_name", "normalized_name", "season_first_seen", "season_last_seen"]
    df = pd.read_parquet(dim_path, columns=cols).copy()
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["normalized_name"] = df["normalized_name"].astype("string").fillna("").str.strip()
    df["season_first_seen"] = df["season_first_seen"].astype("string").fillna("").str.strip()
    df["season_last_seen"] = df["season_last_seen"].astype("string").fillna("").str.strip()
    df = df.dropna(subset=["player_id"]).copy()
    df["player_id"] = df["player_id"].astype(int)
    df = df[df["normalized_name"] != ""].copy()

    # Hard-fail if internal dim itself has collisions.
    dupes = df.groupby("normalized_name")["player_id"].nunique()
    bad = dupes[dupes > 1]
    if not bad.empty:
        examples = (
            df[df["normalized_name"].isin(bad.index)]
            .sort_values(["normalized_name", "player_id"], kind="mergesort")
            .head(50)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Internal players_dim has normalized_name collisions (cannot build deterministic mapping). "
            f"n_bad={len(bad)} examples={examples}"
        )

    return df[cols].copy()


def build_person_id_to_internal_id_map(
    *,
    season_start_year: int,
    person_source: Path | None = None,
    data_root: Path | None = None,
    diagnostics_dir: Path | None = None,
) -> PlayerIdMapResult:
    """Build a deterministic NBA personId -> internal_id map for a season.

    Matching rules:
    - Use `normalize_name()` on person-source names.
    - Match to internal `players_dim.normalized_name` where seasons overlap.
    - HARD-FAIL on ambiguous personId->name inputs, and on collisions where multiple personIds map to one internal_id.
    - Emit diagnostics CSVs in diagnostics_dir when provided.
    """
    season_start_year = int(season_start_year)
    root = (data_root or paths.get_data_root()).expanduser().resolve()

    internal_dim = _load_internal_players_dim(data_root=root)
    internal_dim = internal_dim.copy()
    internal_dim["first_year"] = internal_dim["season_first_seen"].map(_season_start_year_from_season_id)
    internal_dim["last_year"] = internal_dim["season_last_seen"].map(_season_start_year_from_season_id)
    internal_dim["first_year"] = internal_dim["first_year"].fillna(season_start_year).astype(int)
    internal_dim["last_year"] = internal_dim["last_year"].fillna(season_start_year).astype(int)
    internal_dim = internal_dim[
        (internal_dim["first_year"] <= season_start_year) & (internal_dim["last_year"] >= season_start_year)
    ].copy()
    internal_dim = internal_dim.sort_values(["normalized_name", "player_id"], kind="mergesort").reset_index(drop=True)
    internal_names_set = set(internal_dim["normalized_name"].tolist())

    overrides = _load_overrides(data_root=root)
    person_df = _load_person_source_table(
        season_start_year=season_start_year,
        data_root=root,
        person_source=person_source,
    )
    person_df["normalized_name"] = person_df["player_name"].map(normalize_name).astype("string").fillna("")
    if overrides:
        override_series = person_df["nba_person_id"].map(overrides)
        override_mask = override_series.notna()
        if override_mask.any():
            person_df.loc[override_mask, "normalized_name"] = override_series.loc[override_mask].astype("string")
    person_df["normalized_name"] = person_df["normalized_name"].astype("string").fillna("").str.strip()
    person_df = person_df[person_df["normalized_name"] != ""].copy()

    # Canonicalize to one name per personId (mode; ties broken lexicographically).
    name_counts = (
        person_df.groupby(["nba_person_id", "normalized_name"], sort=True)
        .size()
        .reset_index(name="n_rows")
        .sort_values(["nba_person_id", "n_rows", "normalized_name"], ascending=[True, False, True], kind="mergesort")
    )
    ambiguous_rows = pd.DataFrame(
        columns=[
            "nba_person_id",
            "chosen_normalized_name",
            "candidate_normalized_names",
            "n_candidates",
            "n_candidates_matching_internal",
        ]
    )
    unresolved_ambiguous_person_ids: list[int] = []

    chosen_rows: list[dict[str, object]] = []
    for pid, grp in name_counts.groupby("nba_person_id", sort=True):
        candidates = grp["normalized_name"].tolist()
        if len(candidates) == 1:
            chosen = candidates[0]
            chosen_rows.append({"nba_person_id": int(pid), "normalized_name": str(chosen)})
            continue

        matching = [c for c in candidates if c in internal_names_set]
        chosen = None
        if len(matching) == 1:
            chosen = matching[0]
        else:
            # Deterministic fallback: pick the mode (already sorted by n_rows desc, then name asc).
            # If multiple candidates match internal IDs, this is a true ambiguity that would change mapping -> hard fail.
            if len(matching) > 1:
                unresolved_ambiguous_person_ids.append(int(pid))
            chosen = candidates[0]

        chosen_rows.append({"nba_person_id": int(pid), "normalized_name": str(chosen)})
        ambiguous_rows.loc[len(ambiguous_rows)] = {
            "nba_person_id": int(pid),
            "chosen_normalized_name": str(chosen),
            "candidate_normalized_names": "|".join(str(x) for x in candidates),
            "n_candidates": int(len(candidates)),
            "n_candidates_matching_internal": int(len(matching)),
        }

    if unresolved_ambiguous_person_ids:
        ambiguous_rows = ambiguous_rows.sort_values(["nba_person_id"], kind="mergesort").reset_index(drop=True)
        if diagnostics_dir is not None:
            _write_csv(ambiguous_rows, Path(diagnostics_dir) / "ambiguous_matches.csv")
        raise ValueError(
            f"Ambiguous personId->internal mapping (multiple name variants match internal dim) for season_start_year={season_start_year}: "
            f"{unresolved_ambiguous_person_ids[:20]} (and {max(0, len(unresolved_ambiguous_person_ids) - 20)} more). "
            "See ambiguous_matches.csv for details."
        )

    person_unique = pd.DataFrame(chosen_rows, columns=["nba_person_id", "normalized_name"])
    person_unique = person_unique.sort_values(["nba_person_id"], kind="mergesort").reset_index(drop=True)

    # Hard-fail if the person source would collapse multiple distinct personIds into the same normalized name.
    # This would be a deterministic but incorrect mapping (name collision in personId space).
    name_dupes = person_unique.groupby("normalized_name", dropna=False)["nba_person_id"].nunique()
    bad_names = name_dupes[name_dupes > 1]
    collision_rows = pd.DataFrame(columns=["normalized_name", "nba_person_id"])
    if not bad_names.empty:
        collision_rows = (
            person_unique[person_unique["normalized_name"].isin(bad_names.index)]
            .sort_values(["normalized_name", "nba_person_id"], kind="mergesort")
            .reset_index(drop=True)
        )
        if diagnostics_dir is not None:
            _write_csv(collision_rows, Path(diagnostics_dir) / "collisions.csv")
        raise ValueError(
            "Name collisions in person source (multiple nba_person_id share normalized_name). "
            f"n_bad_names={len(bad_names)} examples={collision_rows.head(20).to_dict(orient='records')}"
        )

    merged = person_unique.merge(
        internal_dim[["player_id", "normalized_name"]],
        on="normalized_name",
        how="left",
    )
    merged["player_id"] = pd.to_numeric(merged["player_id"], errors="coerce").astype("Int64")

    # For completeness, assign synthetic internal IDs to personIds that don't exist in pbp_v1 players_dim
    # (e.g. players on lineups/rosters who never appear in pbp/rotation labels).
    # This preserves existing internal IDs and deterministically extends the ID space for the season.
    unmapped = merged[merged["player_id"].isna()].copy()
    unmapped = unmapped.sort_values(["normalized_name", "nba_person_id"], kind="mergesort").reset_index(drop=True)
    next_id = int(internal_dim["player_id"].max()) + 1 if len(internal_dim) else 1
    if not unmapped.empty:
        assigned = list(range(next_id, next_id + len(unmapped)))
        unmapped = unmapped.assign(assigned_internal_id=pd.Series(assigned, dtype="int64"))
        merged = merged.copy()
        merged.loc[merged["player_id"].isna(), "player_id"] = unmapped["assigned_internal_id"].to_numpy()

    mapped = merged.dropna(subset=["player_id"]).copy()
    mapped["player_id"] = mapped["player_id"].astype(int)

    # Detect collisions: multiple personIds map to same internal_id (hard fail).
    collisions = mapped.groupby("player_id", dropna=False)["nba_person_id"].nunique()
    bad_internal_ids = collisions[collisions > 1].index.tolist()
    collision_rows2 = pd.DataFrame(columns=["player_id", "nba_person_id", "normalized_name"])
    if bad_internal_ids:
        collision_rows2 = (
            mapped[mapped["player_id"].isin(bad_internal_ids)][
                ["player_id", "nba_person_id", "normalized_name"]
            ]
            .drop_duplicates()
            .sort_values(["player_id", "nba_person_id"], kind="mergesort")
            .reset_index(drop=True)
        )
        if diagnostics_dir is not None:
            _write_csv(collision_rows2, Path(diagnostics_dir) / "collisions.csv")
        raise ValueError(
            "personId->internal_id collisions detected "
            f"(season_start_year={season_start_year}, n_bad_internal_ids={len(bad_internal_ids)}). "
            "See collisions.csv for details."
        )

    if diagnostics_dir is not None:
        diagnostics_dir = Path(diagnostics_dir)
        _write_csv(unmapped, diagnostics_dir / "unmapped_person_ids.csv")
        # Always emit empty placeholders for debugging convenience.
        ambiguous_rows = ambiguous_rows.sort_values(["nba_person_id"], kind="mergesort").reset_index(drop=True)
        _write_csv(ambiguous_rows, diagnostics_dir / "ambiguous_matches.csv")
        if not (diagnostics_dir / "collisions.csv").exists():
            _write_csv(collision_rows, diagnostics_dir / "collisions.csv")

    mapping = (
        mapped[["nba_person_id", "player_id", "normalized_name"]]
        .sort_values(["nba_person_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    person_id_to_internal_id = dict(zip(mapping["nba_person_id"].tolist(), mapping["player_id"].tolist()))

    logger.info(
        "Built personId->internal_id map for season_start_year=%s: mapped=%s unmapped=%s",
        season_start_year,
        len(person_id_to_internal_id),
        len(unmapped),
    )

    return PlayerIdMapResult(
        season_start_year=season_start_year,
        person_id_to_internal_id=person_id_to_internal_id,
        mapping=mapping,
        unmapped_person_ids=unmapped,
    )
