"""Label freezing utilities for Minutes V1 quick start."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from projections.labels import derive_starter_flag_labels
from projections.minutes_v1.schemas import (
    BOX_SCORE_LABELS_SCHEMA,
    enforce_schema,
    validate_with_pandera,
)
from projections.utils import ensure_directory

REQUIRED_LABEL_COLUMNS = {
    "game_id",
    "player_id",
    "minutes",
    "starter_flag",
    "team_id",
    "season",
    "game_date",
    "source",
}


def _season_path(root: Path, season: str | int) -> Path:
    return ensure_directory(root / f"season={season}")


def freeze_boxscore_labels(
    df: pd.DataFrame,
    root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Persist immutable box score labels grouped by season."""

    missing = REQUIRED_LABEL_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Label dataframe missing required columns: {', '.join(sorted(missing))}")

    frozen_ts = pd.Timestamp(datetime.now(tz=timezone.utc))
    working = df.copy()
    working = derive_starter_flag_labels(working)
    numeric_columns = ("game_id", "player_id", "team_id", "starter_flag", "starter_flag_label")
    for column in numeric_columns:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    if "label_frozen_ts" in working:
        working["label_frozen_ts"] = (
            pd.to_datetime(working["label_frozen_ts"], utc=True, errors="coerce")
            .fillna(frozen_ts)
        )
    else:
        working["label_frozen_ts"] = frozen_ts
    written: dict[str, Path] = {}
    for season, season_df in working.groupby("season"):
        season_dir = _season_path(root, season)
        target = season_dir / "boxscore_labels.parquet"
        if target.exists() and not overwrite:
            raise FileExistsError(f"Labels already frozen for season {season} at {target}")
        season_df = season_df.sort_values(list(BOX_SCORE_LABELS_SCHEMA.primary_key) + ["label_frozen_ts"])
        season_df = season_df.drop_duplicates(subset=list(BOX_SCORE_LABELS_SCHEMA.primary_key), keep="last")
        prepared = enforce_schema(season_df, BOX_SCORE_LABELS_SCHEMA)
        validate_with_pandera(prepared, BOX_SCORE_LABELS_SCHEMA)
        prepared.to_parquet(target, index=False)
        written[str(season)] = target
    return written


def load_frozen_labels(root: Path, seasons: Iterable[str | int] | None = None) -> pd.DataFrame:
    """Load one or more frozen label parquets."""

    if seasons is None:
        files = sorted(root.glob("season=*/boxscore_labels.parquet"))
    else:
        files = []
        for season in seasons:
            candidate = root / f"season={season}" / "boxscore_labels.parquet"
            if not candidate.exists():
                raise FileNotFoundError(f"No frozen labels found at {candidate}")
            files.append(candidate)
    if not files:
        raise FileNotFoundError(f"No frozen label files found under {root}")
    frames = [pd.read_parquet(path) for path in files]
    return pd.concat(frames, ignore_index=True)
