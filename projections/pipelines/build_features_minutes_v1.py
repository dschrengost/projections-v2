"""Build Minutes V1 gold features for a given date range."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Any

import pandas as pd
import typer

from projections import paths
from projections.minutes_v1.datasets import (
    KEY_COLUMNS,
    deduplicate_latest,
    schedule_game_ids_in_range,
    write_ids_csv,
)
from projections.minutes_v1.features import MinutesFeatureBuilder
from projections.minutes_v1.smoke_dataset import _parse_minutes_iso
from projections.minutes_v1.schemas import (
    FEATURES_MINUTES_V1_SCHEMA,
    enforce_schema,
    validate_with_pandera,
)

app = typer.Typer(help=__doc__)


def _normalize_date(value: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _read_parquet_tree(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset at {path}")
    if path.is_file():
        return pd.read_parquet(path)
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    frames = [pd.read_parquet(file) for file in files]
    return pd.concat(frames, ignore_index=True)


def _normalize_game_id(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = int(str(value))
    except (TypeError, ValueError):
        return None
    return f"{number:010d}"


def _is_regular_season_game(game_id: Any) -> bool:
    normalized = _normalize_game_id(game_id)
    if normalized is None:
        return False
    return normalized.startswith("002")


def _coerce_minutes_column(df: pd.DataFrame) -> None:
    if "minutes" not in df.columns:
        return
    series = df["minutes"]
    if pd.api.types.is_numeric_dtype(series):
        df["minutes"] = pd.to_numeric(series, errors="coerce")
        return

    def _convert_minutes_value(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                return None
            if trimmed.upper().startswith("PT"):
                return _parse_minutes_iso(trimmed)
            try:
                return float(trimmed)
            except ValueError:
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    df["minutes"] = pd.Series(
        (_convert_minutes_value(value) for value in series),
        index=series.index,
        dtype="float64",
    )


def _load_labels(data_root: Path, season: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    labels_path = data_root / "labels" / f"season={season}" / "boxscore_labels.parquet"
    labels = pd.read_parquet(labels_path)
    labels["game_date"] = pd.to_datetime(labels["game_date"]).dt.normalize()
    mask = (labels["game_date"] >= start) & (labels["game_date"] <= end)
    sliced = labels.loc[mask].copy()
    if "label_frozen_ts" in sliced:
        sliced["label_frozen_ts"] = pd.to_datetime(sliced["label_frozen_ts"], utc=True, errors="coerce")
        sliced.sort_values(list(KEY_COLUMNS) + ["label_frozen_ts"], inplace=True)
        sliced.drop(columns=["label_frozen_ts"], inplace=True)
    sliced = sliced.drop_duplicates(subset=list(KEY_COLUMNS), keep="last")
    _coerce_minutes_column(sliced)
    if sliced.empty:
        raise ValueError(
            f"No label rows found between {start.date()} and {end.date()} for season={season}."
        )
    return sliced


def _load_schedule(data_root: Path, game_ids: Iterable) -> pd.DataFrame:
    schedule_dir = data_root / "silver" / "schedule"
    schedule = _read_parquet_tree(schedule_dir)
    return schedule[schedule["game_id"].isin(game_ids)].copy()


def _load_snapshot(data_root: Path, name: str, game_ids: Iterable) -> pd.DataFrame:
    dataset_dir = data_root / "silver" / name
    df = _read_parquet_tree(dataset_dir)
    if "game_id" in df.columns:
        df = df[df["game_id"].isin(game_ids)]
    return df.copy()


def _filter_games_with_snapshot(
    dataset: pd.DataFrame,
    game_ids: list[int],
    *,
    label: str,
) -> tuple[pd.DataFrame, list[int]]:
    if dataset.empty or "game_id" not in dataset.columns:
        return dataset, game_ids
    available = set(dataset["game_id"].dropna().astype(int))
    missing = [gid for gid in game_ids if gid not in available]
    if missing:
        typer.echo(
            f"[features] warning: dropping {len(missing)} game_ids missing {label} rows"
        )
        filtered_ids = [gid for gid in game_ids if gid in available]
        dataset = dataset[dataset["game_id"].isin(filtered_ids)].copy()
        return dataset, filtered_ids
    return dataset, game_ids


def _load_roster(data_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    roster_dir = data_root / "silver" / "roster_nightly"
    roster = _read_parquet_tree(roster_dir)
    roster["game_date"] = pd.to_datetime(roster["game_date"]).dt.normalize()
    mask = (roster["game_date"] >= start) & (roster["game_date"] <= end)
    return roster.loc[mask].copy()


def _load_coach_static(data_root: Path) -> pd.DataFrame | None:
    coach_path = data_root / "static" / "coach_tenure.csv"
    if not coach_path.exists():
        typer.echo("coach_tenure.csv not found — skipping coach features.")
        return None
    return pd.read_csv(coach_path)


def _ensure_output_path(
    data_root: Path, season: int, month: int, output: Path | None
) -> Path:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        return output
    dest_dir = (
        data_root
        / "gold"
        / "features_minutes_v1"
        / f"season={season}"
        / f"month={month:02d}"
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / "features.parquet"


def _ids_sidecar_path(parquet_path: Path) -> Path:
    return parquet_path.with_name(parquet_path.stem + ".ids.csv")


def _season_partition_candidates(season: int | str) -> list[str]:
    label = str(season)
    parts = [label]
    if "-" in label:
        start = label.split("-", 1)[0]
        if start and start not in parts:
            parts.append(start)
    return parts


def _load_roles_artifact(root: Path | None, season: int | str) -> pd.DataFrame | None:
    if root is None or not root.exists():
        return None
    for candidate in _season_partition_candidates(season):
        target = root / f"season={candidate}" / "roles.parquet"
        if target.exists():
            return pd.read_parquet(target)
    return None


def _load_archetype_deltas(root: Path | None, season: int | str) -> pd.DataFrame | None:
    if root is None or not root.exists():
        return None
    for candidate in _season_partition_candidates(season):
        target = root / f"season={candidate}" / "archetype_deltas.parquet"
        if target.exists():
            return pd.read_parquet(target)
    return None


@app.command()
def main(
    start: datetime = typer.Option(..., help="Start date (inclusive) in YYYY-MM-DD"),
    end: datetime = typer.Option(..., help="End date (inclusive) in YYYY-MM-DD"),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Root directory containing silver/gold/label data (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    roles_root: Path | None = typer.Option(
        None,
        "--roles-root",
        help="Optional root for minutes roles artifacts (season=YYYY/roles.parquet).",
    ),
    archetype_root: Path | None = typer.Option(
        None,
        "--archetype-root",
        help="Optional root for archetype delta artifacts (season=YYYY/archetype_deltas.parquet).",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Optional explicit output parquet path. Defaults to gold/features_minutes_v1/season=YYYY/month=MM.",
    ),
    season: int | None = typer.Option(
        None,
        "--season",
        help="Season year (e.g., 2024). Defaults to the start date year.",
    ),
    month: int | None = typer.Option(
        None,
        "--month",
        help="Month partition (1-12). Defaults to the start date month.",
    ),
) -> None:
    """Entry point executed via `python -m projections.pipelines.build_features_minutes_v1`."""

    start_date = _normalize_date(start)
    end_date = _normalize_date(end)
    if start_date > end_date:
        raise typer.BadParameter("start date must be on/before end date.")

    season_value = season or start_date.year
    month_value = month or start_date.month

    typer.echo(
        f"Building Minutes V1 features for {start_date.date()} → {end_date.date()} (season={season_value}, month={month_value:02d})"
    )

    labels = _load_labels(data_root, season_value, start_date, end_date)
    game_ids = labels["game_id"].unique().tolist()
    # Keep only regular-season game_ids (prefix 002) so we ignore preseason (001) and playoffs (004).
    regular_game_ids = [gid for gid in game_ids if _is_regular_season_game(gid)]
    if not regular_game_ids:
        raise ValueError("No remaining games after filtering by injuries/odds coverage.")

    labels = labels[labels["game_id"].isin(regular_game_ids)].copy()
    schedule = _load_schedule(data_root, regular_game_ids)
    if schedule.empty:
        raise ValueError("Schedule slice is empty after filtering by game_id.")
    injuries = _load_snapshot(data_root, "injuries_snapshot", game_ids)
    injuries, regular_game_ids = _filter_games_with_snapshot(
        injuries, regular_game_ids, label="injuries"
    )
    if injuries.empty:
        raise ValueError("Injuries snapshot slice is empty for requested games.")
    odds = _load_snapshot(data_root, "odds_snapshot", game_ids)
    odds, regular_game_ids = _filter_games_with_snapshot(
        odds, regular_game_ids, label="odds"
    )
    if odds.empty:
        raise ValueError("Odds snapshot slice is empty for requested games.")
    roster = _load_roster(data_root, start_date, end_date)
    if roster.empty:
        raise ValueError("Roster nightly slice is empty for requested window.")
    coach = _load_coach_static(data_root)
    roles_df = _load_roles_artifact(roles_root, season_value)
    if roles_root is not None and roles_df is None:
        typer.echo(f"[features] warning: missing roles.parquet for season={season_value} under {roles_root}")
    archetype_deltas = _load_archetype_deltas(archetype_root, season_value)
    if archetype_root is not None and archetype_deltas is None:
        typer.echo(
            f"[features] warning: missing archetype_deltas.parquet for season={season_value} under {archetype_root}"
        )

    labels = labels[labels["game_id"].isin(regular_game_ids)].copy()
    schedule = schedule[schedule["game_id"].isin(regular_game_ids)].copy()
    injuries = injuries[injuries["game_id"].isin(regular_game_ids)].copy()
    odds = odds[odds["game_id"].isin(regular_game_ids)].copy()

    builder = MinutesFeatureBuilder(
        schedule=schedule,
        injuries_snapshot=injuries,
        odds_snapshot=odds,
        roster_nightly=roster,
        coach_tenure=coach,
        archetype_roles=roles_df,
        archetype_deltas=archetype_deltas,
    )
    features = builder.build(labels)

    features = deduplicate_latest(features, key_cols=KEY_COLUMNS, order_cols=["feature_as_of_ts"])
    features = enforce_schema(features, FEATURES_MINUTES_V1_SCHEMA)
    validate_with_pandera(features, FEATURES_MINUTES_V1_SCHEMA)

    expected_games = set(
        schedule_game_ids_in_range(
            schedule,
            start=start_date,
            end=end_date,
        ).astype(str)
    )
    actual_games = set(features["game_id"].astype(str).str.zfill(10).unique())
    missing_games = expected_games - actual_games
    if missing_games:
        typer.echo(
            "[features] warning: missing schedule games: "
            + ", ".join(sorted(missing_games))
        )

    if (features["feature_as_of_ts"] > features["tip_ts"]).any():
        raise RuntimeError("Detected feature_as_of_ts > tip_ts rows; aborting write.")

    output_path = _ensure_output_path(data_root, season_value, month_value, out)
    features.to_parquet(output_path, index=False)
    write_ids_csv(features, _ids_sidecar_path(output_path))
    typer.echo(f"Wrote {len(features):,} feature rows to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
