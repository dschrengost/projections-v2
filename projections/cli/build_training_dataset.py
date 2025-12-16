"""Build a versioned Minutes V1 training dataset from immutable gold slates + labels."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import typer

from projections import paths
from projections.etl.storage import iter_days
from projections.minutes_v1.datasets import KEY_COLUMNS


UTC = timezone.utc
LABELS_FILENAME = "labels.parquet"
SLATES_FILENAME_TEMPLATE = "{snapshot_type}.parquet"
RATES_FILENAME = "rates_training_base.parquet"

# Columns to enrich from rates_training_base
ENRICHMENT_COLUMNS_VACANCY = [
    "vac_min_szn",
    "vac_min_guard_szn",
    "vac_min_wing_szn",
    "vac_min_big_szn",
]
ENRICHMENT_COLUMNS_PACE = [
    "team_pace_szn",
    "opp_pace_szn",
]
ENRICHMENT_COLUMNS_TEAM_STRENGTH = [
    "team_off_rtg_szn",
    "team_def_rtg_szn",
    "opp_def_rtg_szn",
]
ENRICHMENT_COLUMNS = (
    ENRICHMENT_COLUMNS_VACANCY + ENRICHMENT_COLUMNS_PACE + ENRICHMENT_COLUMNS_TEAM_STRENGTH
)

app = typer.Typer(help=__doc__)


def _normalize_date(value: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _git_rev_parse_head() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S603, S607
            or None
        )
    except Exception:  # noqa: BLE001
        return None


def _season_start_from_day(day: pd.Timestamp) -> int:
    """Return NBA season start year for an ET-domain date."""

    return int(day.year if day.month >= 8 else day.year - 1)


def _coerce_int_series(series: pd.Series) -> pd.Series:
    coerced = pd.to_numeric(series, errors="coerce").astype("Int64")
    return coerced


def _coerce_game_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], *, label: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"{label} missing required columns: {missing_cols}")


def _discover_slate_snapshot_paths(
    slates_root: Path,
    *,
    snapshot_type: str,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
) -> list[Path]:
    """Return slate parquet paths for the requested date window (inclusive)."""

    normalized_snapshot = snapshot_type.strip().lower()
    if normalized_snapshot not in {"lock", "pretip"}:
        raise ValueError("--snapshot-type must be 'lock' or 'pretip'")

    slate_paths: list[Path] = []
    if not slates_root.exists():
        return slate_paths

    for season_dir in sorted(slates_root.glob("season=*")):
        if not season_dir.is_dir():
            continue
        for date_dir in sorted(season_dir.glob("game_date=*")):
            if not date_dir.is_dir():
                continue
            try:
                date_value = pd.Timestamp(date_dir.name.split("=", 1)[1]).normalize()
            except Exception:
                continue
            if date_value < start_day or date_value > end_day:
                continue
            for game_dir in sorted(date_dir.glob("game_id=*")):
                if not game_dir.is_dir():
                    continue
                path = game_dir / SLATES_FILENAME_TEMPLATE.format(snapshot_type=normalized_snapshot)
                if path.exists():
                    slate_paths.append(path)
    return slate_paths


def _load_slate_snapshots(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class LabelsDiscovery:
    paths: list[Path]
    missing_days: list[str]


def _discover_label_paths(
    labels_root: Path,
    *,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
) -> LabelsDiscovery:
    """Return per-day gold label partitions for the window (inclusive)."""

    found_paths: list[Path] = []
    missing: list[str] = []
    for day in iter_days(start_day, end_day):
        season_start = _season_start_from_day(day)
        path = (
            labels_root
            / f"season={season_start}"
            / f"game_date={day.date().isoformat()}"
            / LABELS_FILENAME
        )
        if path.exists():
            found_paths.append(path)
        else:
            missing.append(day.date().isoformat())
    return LabelsDiscovery(paths=found_paths, missing_days=missing)


def _load_labels(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class EnrichmentDiscovery:
    """Discovery result for rates_training_base enrichment files."""

    paths: list[Path]
    missing_days: list[str]


def _discover_enrichment_paths(
    rates_root: Path,
    *,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
) -> EnrichmentDiscovery:
    """Return rates_training_base parquet paths for the requested date window."""
    found_paths: list[Path] = []
    missing: list[str] = []

    if not rates_root.exists():
        # All days missing if root doesn't exist
        for day in iter_days(start_day, end_day):
            missing.append(day.date().isoformat())
        return EnrichmentDiscovery(paths=[], missing_days=missing)

    for day in iter_days(start_day, end_day):
        season_start = _season_start_from_day(day)
        path = (
            rates_root
            / f"season={season_start}"
            / f"game_date={day.date().isoformat()}"
            / RATES_FILENAME
        )
        if path.exists():
            found_paths.append(path)
        else:
            missing.append(day.date().isoformat())

    return EnrichmentDiscovery(paths=found_paths, missing_days=missing)


def _load_enrichment(
    paths: list[Path],
    *,
    enrichment_columns: list[str],
) -> pd.DataFrame:
    """Load enrichment features from rates_training_base partitions.

    Returns deduplicated DataFrame with game_id, player_id, team_id + enrichment columns.
    """
    if not paths:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    join_keys = ["game_id", "player_id", "team_id"]
    cols_to_load = join_keys + [c for c in enrichment_columns if c not in join_keys]

    for path in paths:
        try:
            df = pd.read_parquet(path, columns=cols_to_load)
            frames.append(df)
        except Exception:  # noqa: BLE001
            # Skip files missing required columns
            continue

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # Normalize join keys
    for col in join_keys:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").astype("Int64")
    # Deduplicate by join keys (keep last in case of multiple entries)
    combined = combined.dropna(subset=join_keys).drop_duplicates(subset=join_keys, keep="last")
    return combined.reset_index(drop=True)


def _apply_enrichment(
    joined: pd.DataFrame,
    enrichment: pd.DataFrame,
    *,
    enrichment_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Merge enrichment features into joined dataset and fill missing values.

    Returns (enriched DataFrame, coverage stats dict).
    """
    if enrichment.empty:
        # Add empty enrichment columns with appropriate fill values
        for col in ENRICHMENT_COLUMNS_VACANCY:
            joined[col] = 0.0
        for col in ENRICHMENT_COLUMNS_PACE + ENRICHMENT_COLUMNS_TEAM_STRENGTH:
            joined[col] = np.nan
        coverage = {col: 0.0 for col in enrichment_columns}
        return joined, coverage

    join_keys = ["game_id", "player_id", "team_id"]
    # Ensure join keys are compatible types
    for col in join_keys:
        if col in joined.columns:
            joined[col] = pd.to_numeric(joined[col], errors="coerce").astype("Int64")

    # Merge enrichment
    enriched = joined.merge(
        enrichment,
        on=join_keys,
        how="left",
        suffixes=("", "_enrich"),
    )

    # Calculate coverage and fill missing values
    coverage: dict[str, float] = {}
    n_rows = len(enriched)

    for col in ENRICHMENT_COLUMNS_VACANCY:
        if col in enriched.columns:
            n_present = enriched[col].notna().sum()
            coverage[col] = n_present / n_rows if n_rows > 0 else 0.0
            enriched[col] = enriched[col].fillna(0.0)
        else:
            enriched[col] = 0.0
            coverage[col] = 0.0

    for col in ENRICHMENT_COLUMNS_PACE + ENRICHMENT_COLUMNS_TEAM_STRENGTH:
        if col in enriched.columns:
            n_present = enriched[col].notna().sum()
            coverage[col] = n_present / n_rows if n_rows > 0 else 0.0
            # Fill with global mean (or fallback)
            mean_val = enriched[col].mean(skipna=True)
            fill_val = mean_val if pd.notna(mean_val) else 100.0  # reasonable default pace/rating
            enriched[col] = enriched[col].fillna(fill_val)
        else:
            enriched[col] = 100.0  # default pace/rating
            coverage[col] = 0.0

    return enriched, coverage


def _normalize_slates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    required = [*KEY_COLUMNS, "game_date"]
    _ensure_columns(df, required, label="Slate features")
    working = df.copy()
    for col in KEY_COLUMNS:
        working[col] = _coerce_int_series(working[col])
    working["game_date"] = _coerce_game_date(working["game_date"])
    working = working.dropna(subset=[*KEY_COLUMNS, "game_date"]).copy()
    if "minutes" in working.columns:
        working = working.drop(columns=["minutes"])

    key_cols = [*KEY_COLUMNS, "game_date"]
    order_cols = [col for col in ("snapshot_ts", "frozen_at", "feature_as_of_ts") if col in working.columns]
    if order_cols:
        for col in order_cols:
            working[col] = pd.to_datetime(working[col], utc=True, errors="coerce")
        working = working.sort_values(key_cols + order_cols, kind="mergesort")
    else:
        working = working.sort_values(key_cols, kind="mergesort")
    working = working.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    return working


def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    required = [*KEY_COLUMNS, "game_date", "minutes"]
    _ensure_columns(df, required, label="Minutes labels")
    working = df.copy()
    for col in KEY_COLUMNS:
        working[col] = _coerce_int_series(working[col])
    working["game_date"] = _coerce_game_date(working["game_date"])
    working["minutes"] = pd.to_numeric(working["minutes"], errors="coerce")
    working = working.dropna(subset=[*KEY_COLUMNS, "game_date"]).copy()

    key_cols = [*KEY_COLUMNS, "game_date"]
    if "label_frozen_ts" in working.columns:
        working["label_frozen_ts"] = pd.to_datetime(working["label_frozen_ts"], utc=True, errors="coerce")
        working = working.sort_values(key_cols + ["label_frozen_ts"], kind="mergesort")
    else:
        working = working.sort_values(key_cols, kind="mergesort")
    working = working.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    return working


def _compute_missing_rates(
    *,
    slate_rows: int,
    label_rows: int,
    joined_rows: int,
    joined_minutes_missing: int,
) -> dict[str, float]:
    slate_missing = float("nan") if slate_rows == 0 else (slate_rows - joined_rows) / slate_rows
    label_missing = float("nan") if label_rows == 0 else (label_rows - joined_rows) / label_rows
    joined_minutes_missing_rate = (
        float("nan") if joined_rows == 0 else joined_minutes_missing / joined_rows
    )
    return {
        "slate_rows_missing_labels": float(slate_missing),
        "label_rows_missing_slates": float(label_missing),
        "joined_minutes_missing": float(joined_minutes_missing_rate),
    }


def _build_manifest(
    *,
    version: str,
    snapshot_type: str,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    slates_paths: list[Path],
    labels_discovery: LabelsDiscovery,
    slates: pd.DataFrame,
    labels: pd.DataFrame,
    joined: pd.DataFrame,
) -> dict[str, Any]:
    joined_minutes_missing = int(pd.to_numeric(joined.get("minutes"), errors="coerce").isna().sum())
    missing_rates = _compute_missing_rates(
        slate_rows=int(len(slates)),
        label_rows=int(len(labels)),
        joined_rows=int(len(joined)),
        joined_minutes_missing=joined_minutes_missing,
    )
    return {
        "version": version,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "git_sha": _git_rev_parse_head(),
        "snapshot_type": snapshot_type,
        "date_range": {
            "start": start_day.strftime("%Y-%m-%d"),
            "end": end_day.strftime("%Y-%m-%d"),
        },
        "input_counts": {
            "slate_files": int(len(slates_paths)),
            "slate_games": int(slates["game_id"].nunique()) if not slates.empty else 0,
            "slate_rows": int(len(slates)),
            "label_partitions": int(len(labels_discovery.paths)),
            "label_missing_days": int(len(labels_discovery.missing_days)),
            "label_games": int(labels["game_id"].nunique()) if not labels.empty else 0,
            "label_rows": int(len(labels)),
            "joined_games": int(joined["game_id"].nunique()) if not joined.empty else 0,
            "joined_rows": int(len(joined)),
        },
        "missing_rates": missing_rates,
    }


@app.command()
def main(
    version: str = typer.Option(..., help="Dataset version (e.g., v1_20251208)."),
    start_date: datetime = typer.Option(..., "--start-date", "--start", help="Start date (inclusive)."),
    end_date: datetime = typer.Option(..., "--end-date", "--end", help="End date (inclusive)."),
    snapshot_type: str = typer.Option(
        "pretip",
        "--snapshot-type",
        help="Gold slate snapshot type to use ('pretip' recommended for no-leak training).",
    ),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Root directory containing gold/ partitions (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    out_root: Path | None = typer.Option(
        None,
        "--out-root",
        help="Optional override for training/datasets root (defaults to <data_root>/training/datasets).",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite outputs for an existing dataset version."),
    enable_enrichment: bool = typer.Option(
        True,
        "--enable-enrichment/--disable-enrichment",
        help="Enrich dataset with vacancy, pace, and team context features from rates_training_base.",
    ),
) -> None:
    """Build a reproducible training dataset from frozen gold slate features + minutes labels."""

    data_root = data_root.expanduser().resolve()
    start_day = _normalize_date(start_date)
    end_day = _normalize_date(end_date)
    if end_day < start_day:
        raise typer.BadParameter("--end-date must be on/after --start-date.")

    normalized_snapshot = snapshot_type.strip().lower()
    if normalized_snapshot not in {"lock", "pretip"}:
        raise typer.BadParameter("--snapshot-type must be 'lock' or 'pretip'.")

    datasets_root = (out_root or (data_root / "training" / "datasets")).expanduser().resolve()
    out_dir = datasets_root / version
    if out_dir.exists() and not force:
        existing = [p.name for p in out_dir.glob("*.json")] + [p.name for p in out_dir.glob("*.parquet")]
        if existing:
            raise typer.BadParameter(
                f"Dataset output dir already exists with files ({', '.join(sorted(existing))}); "
                "choose a new --version or pass --force to overwrite."
            )

    slates_root = data_root / "gold" / "slates"
    labels_root = data_root / "gold" / "labels_minutes_v1"

    slates_paths = _discover_slate_snapshot_paths(
        slates_root,
        snapshot_type=normalized_snapshot,
        start_day=start_day,
        end_day=end_day,
    )
    if not slates_paths:
        raise typer.BadParameter(
            f"No gold slate snapshots found under {slates_root} for {start_day.date()} → {end_day.date()} "
            f"(snapshot_type={normalized_snapshot})."
        )

    slates_raw = _load_slate_snapshots(slates_paths)
    slates = _normalize_slates(slates_raw)
    if slates.empty:
        raise RuntimeError("Loaded slate snapshots but produced zero usable rows after normalization.")

    labels_discovery = _discover_label_paths(labels_root, start_day=start_day, end_day=end_day)
    if not labels_discovery.paths:
        raise typer.BadParameter(
            f"No gold labels found under {labels_root} for {start_day.date()} → {end_day.date()}."
        )

    labels_raw = _load_labels(labels_discovery.paths)
    labels = _normalize_labels(labels_raw)
    if labels.empty:
        raise RuntimeError("Loaded labels but produced zero usable rows after normalization.")

    key_cols = [*KEY_COLUMNS, "game_date"]

    # Drop columns from labels that already exist in slates (except key cols and minutes).
    # This avoids suffix collisions like starter_flag -> starter_flag_label conflicting
    # with an existing starter_flag_label column in labels.
    label_keep_cols = [
        *key_cols,
        "minutes",
        "starter_flag_label",  # derived label column
        "label_frozen_ts",
        "source",
    ]
    label_keep_cols = [c for c in label_keep_cols if c in labels.columns]
    labels_for_merge = labels[label_keep_cols].copy()

    joined = slates.merge(labels_for_merge, on=key_cols, how="inner", suffixes=("", "_label"))
    if joined.empty:
        raise RuntimeError("Slate/label merge produced zero rows — verify inputs overlap.")

    # Ensure minutes column exists (from labels) and is numeric.
    joined["minutes"] = pd.to_numeric(joined["minutes"], errors="coerce")

    # Enrichment from rates_training_base (vacancy, pace, team context)
    enrichment_discovery: EnrichmentDiscovery | None = None
    enrichment_coverage: dict[str, float] = {}

    if enable_enrichment:
        rates_root = data_root / "gold" / "rates_training_base"
        enrichment_discovery = _discover_enrichment_paths(
            rates_root, start_day=start_day, end_day=end_day
        )

        if enrichment_discovery.paths:
            typer.echo(
                f"[training-dataset] loading enrichment from {len(enrichment_discovery.paths)} "
                f"rates_training_base partitions..."
            )
            enrichment_df = _load_enrichment(
                enrichment_discovery.paths,
                enrichment_columns=ENRICHMENT_COLUMNS,
            )
            joined, enrichment_coverage = _apply_enrichment(
                joined,
                enrichment_df,
                enrichment_columns=ENRICHMENT_COLUMNS,
            )
            # Log enrichment coverage
            avg_coverage = sum(enrichment_coverage.values()) / len(enrichment_coverage) if enrichment_coverage else 0.0
            typer.echo(f"[training-dataset] enrichment coverage: {avg_coverage:.1%} average")
            for col, cov in sorted(enrichment_coverage.items()):
                typer.echo(f"  {col}: {cov:.1%}")
        else:
            typer.echo(
                f"[training-dataset] WARNING: no rates_training_base partitions found "
                f"({len(enrichment_discovery.missing_days)} days missing); skipping enrichment."
            )
            # Add empty enrichment columns
            for col in ENRICHMENT_COLUMNS_VACANCY:
                joined[col] = 0.0
            for col in ENRICHMENT_COLUMNS_PACE + ENRICHMENT_COLUMNS_TEAM_STRENGTH:
                joined[col] = np.nan
            enrichment_coverage = {col: 0.0 for col in ENRICHMENT_COLUMNS}
    else:
        typer.echo("[training-dataset] enrichment disabled; skipping rates_training_base features.")

    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "features.parquet"
    labels_path = out_dir / "labels.parquet"
    manifest_path = out_dir / "manifest.json"

    joined.to_parquet(features_path, index=False)
    labels.to_parquet(labels_path, index=False)

    manifest = _build_manifest(
        version=version,
        snapshot_type=normalized_snapshot,
        start_day=start_day,
        end_day=end_day,
        slates_paths=slates_paths,
        labels_discovery=labels_discovery,
        slates=slates,
        labels=labels,
        joined=joined,
    )

    # Add enrichment metadata to manifest
    if enable_enrichment:
        manifest["enrichment"] = {
            "enabled": True,
            "source": "rates_training_base",
            "columns": ENRICHMENT_COLUMNS,
            "partitions_found": len(enrichment_discovery.paths) if enrichment_discovery else 0,
            "partitions_missing_days": len(enrichment_discovery.missing_days) if enrichment_discovery else 0,
            "coverage": {col: float(cov) for col, cov in enrichment_coverage.items()},
        }
    else:
        manifest["enrichment"] = {"enabled": False}

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    typer.echo(f"[training-dataset] wrote {features_path}")
    typer.echo(f"[training-dataset] wrote {labels_path}")
    typer.echo(f"[training-dataset] wrote {manifest_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
