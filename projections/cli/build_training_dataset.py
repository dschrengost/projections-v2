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

# Smoke test (manual):
#   uv run python -m projections.cli.build_training_dataset \
#     --version v1_enriched_smoke_20251204 \
#     --start-date 2025-12-01 --end-date 2025-12-04 --snapshot-type pretip --force
# Expect: opp_ctx_missing < 1.0 and vac_missing < 1.0 when rates_training_base has coverage.


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

    Returns deduplicated DataFrame with game_id, team_id + enrichment columns.
    """
    if not paths:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    join_keys = ["game_id", "team_id"]
    cols_to_load = join_keys + list(enrichment_columns)
    skipped_missing_cols = 0
    skipped_read_errors = 0

    for path in paths:
        try:
            df = pd.read_parquet(path, columns=cols_to_load)
        except Exception as exc:  # noqa: BLE001
            skipped_read_errors += 1
            typer.echo(f"[training-dataset] enrichment read failed: {path} ({exc})")
            continue
        missing = [col for col in cols_to_load if col not in df.columns]
        if missing:
            skipped_missing_cols += 1
            typer.echo(f"[training-dataset] enrichment missing columns: {path} -> {missing}")
            continue
        frames.append(df)
    if skipped_missing_cols or skipped_read_errors:
        typer.echo(
            "[training-dataset] enrichment load summary: "
            f"files={len(paths)} loaded={len(frames)} "
            f"skipped_missing_cols={skipped_missing_cols} skipped_read_errors={skipped_read_errors}"
        )

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # Normalize join keys
    for col in [*join_keys, "opponent_id"]:
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
    """Merge enrichment features into joined dataset and compute coverage stats.

    Returns (enriched DataFrame, coverage stats dict).
    """
    if enrichment.empty:
        for col in enrichment_columns:
            joined[col] = np.nan
        joined["team_ctx_missing"] = 1
        joined["opp_ctx_missing"] = 1
        joined["vac_missing"] = 1
        coverage = {col: 0.0 for col in enrichment_columns}
        return joined, coverage

    working = joined.copy()
    had_opponent_id = "opponent_id" in working.columns
    for col in ("game_id", "team_id", "player_id"):
        if col in working.columns:
            working[col] = _coerce_int_series(working[col])

    opponent_id: pd.Series
    if "opponent_id" in working.columns:
        opponent_id = _coerce_int_series(working["opponent_id"])
    elif "opponent_team_id" in working.columns:
        opponent_id = _coerce_int_series(working["opponent_team_id"])
    elif {"home_team_id", "away_team_id", "team_id"}.issubset(working.columns):
        home = _coerce_int_series(working["home_team_id"])
        away = _coerce_int_series(working["away_team_id"])
        team = _coerce_int_series(working["team_id"])

        opponent_id = pd.Series(pd.NA, index=working.index, dtype="Int64")
        is_home = team == home
        is_away = team == away
        opponent_id.loc[is_home] = away.loc[is_home]
        opponent_id.loc[is_away] = home.loc[is_away]
    else:
        opponent_id = pd.Series(pd.NA, index=working.index, dtype="Int64")

    working["opponent_id"] = opponent_id

    enriched_source = enrichment.copy()
    for col in ("game_id", "team_id", "opponent_id", "player_id"):
        if col in enriched_source.columns:
            enriched_source[col] = _coerce_int_series(enriched_source[col])

    missing_cols = [col for col in enrichment_columns if col not in enriched_source.columns]
    if missing_cols:
        debug_cols = ["game_id", "team_id"] + [
            c for c in enrichment_columns if c in enriched_source.columns
        ]
        debug_head = (
            enriched_source.loc[:, debug_cols].head(5).to_string(index=False)
            if debug_cols
            else "<no columns>"
        )
        raise ValueError(
            "rates_training_base missing enrichment columns: "
            f"{missing_cols}. enrichment_source_cols={sorted(enriched_source.columns)} "
            f"enrichment_source_head=\n{debug_head}"
        )

    team_ctx_cols = [c for c in ("team_pace_szn", "team_off_rtg_szn", "team_def_rtg_szn") if c in enriched_source.columns]

    spine_game_teams = (
        working.loc[:, ["game_id", "team_id"]].dropna().drop_duplicates().copy()
        if {"game_id", "team_id"}.issubset(working.columns)
        else pd.DataFrame(columns=["game_id", "team_id"])
    )
    enrich_game_teams = (
        enriched_source.loc[:, ["game_id", "team_id"]].dropna().drop_duplicates().copy()
        if {"game_id", "team_id"}.issubset(enriched_source.columns)
        else pd.DataFrame(columns=["game_id", "team_id"])
    )
    overlap_games = int(
        len(set(spine_game_teams["game_id"].tolist()) & set(enrich_game_teams["game_id"].tolist()))
        if not spine_game_teams.empty and not enrich_game_teams.empty
        else 0
    )
    overlap_game_team = int(
        len(spine_game_teams.merge(enrich_game_teams, on=["game_id", "team_id"], how="inner"))
        if not spine_game_teams.empty and not enrich_game_teams.empty
        else 0
    )
    typer.echo(
        f"[training-dataset] enrichment overlap: overlap_games={overlap_games} overlap_game_team={overlap_game_team}"
    )
    if overlap_games == 0:
        spine_sample = spine_game_teams["game_id"].dropna().astype(int).astype(str).head(10).tolist()
        enrich_sample = enrich_game_teams["game_id"].dropna().astype(int).astype(str).head(10).tolist()
        typer.echo(f"[training-dataset] sample spine game_ids: {spine_sample}")
        typer.echo(f"[training-dataset] sample enrich game_ids: {enrich_sample}")
    if overlap_game_team == 0:
        spine_pairs_df = spine_game_teams.dropna().head(10)
        enrich_pairs_df = enrich_game_teams.dropna().head(10)
        spine_pairs = [
            f"({int(gid)},{int(tid)})" for gid, tid in spine_pairs_df.itertuples(index=False, name=None)
        ]
        enrich_pairs = [
            f"({int(gid)},{int(tid)})" for gid, tid in enrich_pairs_df.itertuples(index=False, name=None)
        ]
        typer.echo(f"[training-dataset] sample spine (game_id, team_id): {spine_pairs}")
        typer.echo(f"[training-dataset] sample enrich (game_id, team_id): {enrich_pairs}")

    def _enrichment_merge_debug(label: str, df: pd.DataFrame) -> None:
        expected = set(enrichment_columns)
        present = sorted(expected & set(df.columns))
        missing = sorted(expected - set(df.columns))
        xy_cols = sorted(
            [
                c
                for c in df.columns
                if (c.endswith("_x") or c.endswith("_y")) and c.rsplit("_", 1)[0] in expected
            ]
        )
        nn_vac = int(df["vac_min_szn"].notna().sum()) if "vac_min_szn" in df.columns else 0
        nn_opp_pace = int(df["opp_pace_szn"].notna().sum()) if "opp_pace_szn" in df.columns else 0
        nn_opp_def = int(df["opp_def_rtg_szn"].notna().sum()) if "opp_def_rtg_szn" in df.columns else 0
        typer.echo(
            "[training-dataset] enrichment merge "
            f"{label}: present={len(present)}/{len(expected)} "
            f"nn(vac_min_szn)={nn_vac} nn(opp_pace_szn)={nn_opp_pace} nn(opp_def_rtg_szn)={nn_opp_def} "
            f"suffixed_xy={len(xy_cols)}"
        )
        if missing:
            typer.echo(f"[training-dataset] enrichment merge {label}: missing_cols={missing}")
        if xy_cols:
            typer.echo(f"[training-dataset] enrichment merge {label}: suffixed_xy_cols={xy_cols}")

    def _coalesce_xy(df: pd.DataFrame, base_col: str) -> None:
        col_x = f"{base_col}_x"
        col_y = f"{base_col}_y"
        if col_x not in df.columns and col_y not in df.columns:
            return
        if base_col not in df.columns:
            df[base_col] = np.nan
        if col_y in df.columns:
            df[base_col] = df[base_col].fillna(df[col_y])
        if col_x in df.columns:
            df[base_col] = df[base_col].fillna(df[col_x])
        df.drop(columns=[c for c in (col_x, col_y) if c in df.columns], inplace=True)

    # TEAM enrichment: join on (game_id, team_id)
    team_features_cols = ["game_id", "team_id", *enrichment_columns]
    team_features = enriched_source.loc[:, [c for c in team_features_cols if c in enriched_source.columns]].copy()
    team_features = team_features.dropna(subset=["game_id", "team_id"]).drop_duplicates(
        subset=["game_id", "team_id"], keep="last"
    )
    if overlap_game_team > 0:
        check_cols = ["opp_pace_szn", "opp_def_rtg_szn", *ENRICHMENT_COLUMNS_VACANCY]
        zero_non_null = [
            col for col in check_cols
            if col in team_features.columns and int(team_features[col].notna().sum()) == 0
        ]
        missing_in_team = [col for col in check_cols if col not in team_features.columns]
        if zero_non_null or missing_in_team:
            debug_cols = ["game_id", "team_id"] + [
                col for col in check_cols if col in team_features.columns or col in enriched_source.columns
            ]
            team_head = (
                team_features.loc[:, debug_cols].head(5).to_string(index=False)
                if debug_cols
                else "<no columns>"
            )
            source_head = (
                enriched_source.loc[:, debug_cols].head(5).to_string(index=False)
                if debug_cols
                else "<no columns>"
            )
            raise ValueError(
                "Team enrichment missing expected columns: "
                f"missing={missing_in_team} zero_non_null={zero_non_null} "
                f"team_features_head=\n{team_head}\n"
                f"enrichment_source_head=\n{source_head}"
            )

    team_suffix = "__rtb_team"
    rename_team = {col: f"{col}{team_suffix}" for col in enrichment_columns if col in team_features.columns}
    team_features = team_features.rename(columns=rename_team)
    enriched = working.merge(team_features, on=["game_id", "team_id"], how="left", validate="m:1")
    for col in enrichment_columns:
        rtb_col = f"{col}{team_suffix}"
        if rtb_col not in enriched.columns:
            continue
        if col in enriched.columns:
            enriched[col] = enriched[rtb_col].where(enriched[rtb_col].notna(), enriched[col])
        else:
            enriched[col] = enriched[rtb_col]
        enriched.drop(columns=[rtb_col], inplace=True)
    for col in enrichment_columns:
        _coalesce_xy(enriched, col)
    _enrichment_merge_debug("team", enriched)

    # OPPONENT enrichment: join opponent_id to rates team context.
    opp_features_cols = ["game_id", "team_id", "team_pace_szn", "team_def_rtg_szn"]
    opp_features_cols = [c for c in opp_features_cols if c in enriched_source.columns]
    opp_features = enriched_source.loc[:, opp_features_cols].copy()
    opp_features = opp_features.dropna(subset=["game_id", "team_id"]).drop_duplicates(
        subset=["game_id", "team_id"], keep="last"
    )
    opp_suffix = "__rtb_opp"
    rename_opp = {
        "team_id": "opponent_id",
        "team_pace_szn": f"opp_pace_szn{opp_suffix}",
        "team_def_rtg_szn": f"opp_def_rtg_szn{opp_suffix}",
    }
    opp_features = opp_features.rename(columns={k: v for k, v in rename_opp.items() if k in opp_features.columns})
    enriched = enriched.merge(opp_features, on=["game_id", "opponent_id"], how="left", validate="m:1")

    opp_pace_from_team = f"opp_pace_szn{opp_suffix}"
    if opp_pace_from_team in enriched.columns:
        if "opp_pace_szn" in enriched.columns:
            enriched["opp_pace_szn"] = enriched["opp_pace_szn"].fillna(enriched[opp_pace_from_team])
        else:
            enriched["opp_pace_szn"] = enriched[opp_pace_from_team]
        enriched.drop(columns=[opp_pace_from_team], inplace=True)

    opp_def_from_team = f"opp_def_rtg_szn{opp_suffix}"
    if opp_def_from_team in enriched.columns:
        if "opp_def_rtg_szn" in enriched.columns:
            enriched["opp_def_rtg_szn"] = enriched["opp_def_rtg_szn"].fillna(enriched[opp_def_from_team])
        else:
            enriched["opp_def_rtg_szn"] = enriched[opp_def_from_team]
        enriched.drop(columns=[opp_def_from_team], inplace=True)

    for col in enrichment_columns:
        _coalesce_xy(enriched, col)
    _enrichment_merge_debug("opp", enriched)

    if overlap_game_team > 0:
        zero_cols: list[str] = []
        for col in [*ENRICHMENT_COLUMNS_VACANCY, "opp_pace_szn", "opp_def_rtg_szn"]:
            if col in enriched.columns and int(enriched[col].notna().sum()) == 0:
                zero_cols.append(col)
        if zero_cols:
            debug_cols = ["game_id", "team_id"] + [
                c for c in enrichment_columns if c in enriched_source.columns
            ]
            debug_head = (
                enriched_source.loc[:, debug_cols].head(5).to_string(index=False)
                if debug_cols
                else "<no columns>"
            )
            raise ValueError(
                "Enrichment merge produced 0% non-null columns: "
                f"{zero_cols}. enrichment_source_cols={sorted(enriched_source.columns)} "
                f"enrichment_source_head=\n{debug_head}"
            )

    for col in enrichment_columns:
        if col not in enriched.columns:
            enriched[col] = np.nan
        enriched[col] = pd.to_numeric(enriched[col], errors="coerce")

    enriched["team_ctx_missing"] = (
        enriched[team_ctx_cols].isna().all(axis=1).astype(int) if team_ctx_cols else 1
    )
    enriched["opp_ctx_missing"] = (
        enriched[["opp_pace_szn", "opp_def_rtg_szn"]].isna().all(axis=1).astype(int)
    )
    enriched["vac_missing"] = enriched[ENRICHMENT_COLUMNS_VACANCY].isna().all(axis=1).astype(int)
    if overlap_game_team > 0:
        opp_missing_rate = float(pd.to_numeric(enriched["opp_ctx_missing"], errors="coerce").fillna(1).mean())
        vac_missing_rate = float(pd.to_numeric(enriched["vac_missing"], errors="coerce").fillna(1).mean())
        if opp_missing_rate >= 1.0:
            typer.echo(
                "[training-dataset] WARNING: opp_ctx_missing=100% despite overlap_game_team > 0."
            )
        if vac_missing_rate >= 1.0:
            typer.echo(
                "[training-dataset] WARNING: vac_missing=100% despite overlap_game_team > 0."
            )

    if overlap_game_team > 0 and not spine_game_teams.empty and not enrich_game_teams.empty:
        overlap_keys = spine_game_teams.merge(enrich_game_teams, on=["game_id", "team_id"], how="inner")
        source_overlap = enriched_source.merge(overlap_keys, on=["game_id", "team_id"], how="inner")
        enriched_overlap = enriched.merge(overlap_keys, on=["game_id", "team_id"], how="inner")

        for col in ("vac_min_szn", "opp_pace_szn", "opp_def_rtg_szn"):
            source_non_null = int(source_overlap[col].notna().sum()) if col in source_overlap.columns else 0
            output_non_null = int(enriched_overlap[col].notna().sum()) if col in enriched_overlap.columns else 0
            if source_non_null > 0 and output_non_null == 0:
                raise ValueError(
                    "Enrichment invariant failed: "
                    f"overlap_game_team={overlap_game_team} source_non_null({col})={source_non_null} "
                    f"output_non_null({col})={output_non_null}"
                )

    # Calculate coverage (no default fills)
    coverage: dict[str, float] = {}
    n_rows = len(enriched)
    for col in enrichment_columns:
        n_present = int(enriched[col].notna().sum()) if col in enriched.columns else 0
        coverage[col] = n_present / n_rows if n_rows > 0 else 0.0

    if not had_opponent_id:
        enriched.drop(columns=["opponent_id"], inplace=True, errors="ignore")

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


def _compute_odds_coverage(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "game_id" not in df.columns:
        return {"odds_columns_present": False}
    if "spread_home" not in df.columns or "total" not in df.columns:
        return {"odds_columns_present": False}

    working = df.copy()
    working["game_id_norm"] = working["game_id"].astype(str).str.zfill(10)
    has_odds = working["spread_home"].notna() & working["total"].notna()

    coverage_rate = float(has_odds.mean()) if len(working) > 0 else float("nan")
    slate_games = int(working["game_id_norm"].nunique())
    odds_games = int(working.loc[has_odds, "game_id_norm"].nunique())

    coverage_by_prefix: dict[str, float] = {}
    for prefix in ("00222", "00223", "00224", "00225"):
        mask = working["game_id_norm"].astype(str).str.startswith(prefix)
        if not mask.any():
            continue
        coverage_by_prefix[prefix] = float(has_odds[mask].mean())

    return {
        "odds_columns_present": True,
        "odds_coverage_rate": coverage_rate,
        "slate_games": slate_games,
        "odds_games": odds_games,
        "overlap_games": odds_games,
        "coverage_by_prefix": coverage_by_prefix,
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
    odds_coverage: dict[str, Any] | None = None,
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
        "odds_join": {
            "game_id_norm_zfill10": True,
            **(odds_coverage or {}),
        },
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
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        help="Optional explicit output directory (overrides --out-root/--version).",
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
    out_dir = (out_dir or (datasets_root / version)).expanduser().resolve()
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

    odds_coverage = _compute_odds_coverage(joined)
    if odds_coverage.get("odds_columns_present"):
        typer.echo(
            f"[training-dataset] odds coverage: {odds_coverage.get('odds_coverage_rate', float('nan')):.1%} "
            f"({odds_coverage.get('odds_games', 0)}/{odds_coverage.get('slate_games', 0)} games)"
        )
        typer.echo(
            "[training-dataset] odds games: "
            f"slates={odds_coverage.get('slate_games', 0)} "
            f"odds={odds_coverage.get('odds_games', 0)} "
            f"overlap={odds_coverage.get('overlap_games', 0)}"
        )
        for prefix, rate in sorted(odds_coverage.get("coverage_by_prefix", {}).items()):
            typer.echo(f"[training-dataset] odds coverage {prefix}*: {rate:.1%}")

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
            typer.echo("[training-dataset] enrichment stats:")
            for col in ENRICHMENT_COLUMNS:
                non_null_rate = float(joined[col].notna().mean()) if col in joined.columns else 0.0
                nunique = int(joined[col].nunique(dropna=True)) if col in joined.columns else 0
                typer.echo(f"  {col}: non_null={non_null_rate:.1%} nunique={nunique}")
                if non_null_rate > 0.95 and nunique <= 1 and col not in ENRICHMENT_COLUMNS_VACANCY:
                    raise ValueError(
                        f"Enrichment sanity check failed: column '{col}' appears constant "
                        f"(nunique={nunique}, non_null={non_null_rate:.1%})."
                    )
                if non_null_rate > 0.95 and nunique <= 1 and col in ENRICHMENT_COLUMNS_VACANCY:
                    typer.echo(
                        f"[training-dataset] WARNING: enrichment column '{col}' appears constant "
                        f"(nunique={nunique}, non_null={non_null_rate:.1%})."
                    )
            for flag in ("team_ctx_missing", "opp_ctx_missing", "vac_missing"):
                if flag not in joined.columns:
                    continue
                rate = float(pd.to_numeric(joined[flag], errors="coerce").fillna(1).mean()) if len(joined) else 1.0
                typer.echo(f"  {flag}: {rate:.1%}")
        else:
            typer.echo(
                f"[training-dataset] WARNING: no rates_training_base partitions found "
                f"({len(enrichment_discovery.missing_days)} days missing); skipping enrichment."
            )
            for col in ENRICHMENT_COLUMNS:
                joined[col] = np.nan
            joined["team_ctx_missing"] = 1
            joined["opp_ctx_missing"] = 1
            joined["vac_missing"] = 1
            enrichment_coverage = {col: 0.0 for col in ENRICHMENT_COLUMNS}
    else:
        typer.echo("[training-dataset] enrichment disabled; skipping rates_training_base features.")
        for col in ENRICHMENT_COLUMNS:
            joined[col] = np.nan
        joined["team_ctx_missing"] = 1
        joined["opp_ctx_missing"] = 1
        joined["vac_missing"] = 1

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
        odds_coverage=odds_coverage,
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
