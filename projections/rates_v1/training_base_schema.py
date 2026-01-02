"""Schema guardrails for gold/rates_training_base partitions.

This is intentionally minimal: it exists to prevent regressions where critical
timestamp columns disappear (which would break point-in-time evaluation and
recency weighting).
"""

from __future__ import annotations

import json
from pathlib import Path

from projections.minutes_v1.schemas import (
    BOOL_DTYPE,
    FLOAT_DTYPE,
    INT_DTYPE,
    NAIVE_TS,
    UTC_TS,
    TableSchema,
)


_SCHEMA_JSON_PATH = Path(__file__).with_name("rates_training_base_schema.json")
_SCHEMA_JSON = json.loads(_SCHEMA_JSON_PATH.read_text(encoding="utf-8"))

RATES_TRAINING_BASE_REQUIRED_COLUMNS = tuple(_SCHEMA_JSON["required_columns"])


_DTYPES = {
    "season": INT_DTYPE,
    "game_id": INT_DTYPE,
    "game_date": NAIVE_TS,
    "tip_ts": UTC_TS,
    "feature_as_of_ts": UTC_TS,
    "team_id": INT_DTYPE,
    "opponent_id": INT_DTYPE,
    "home_flag": INT_DTYPE,
    "player_id": INT_DTYPE,
    "minutes_actual": FLOAT_DTYPE,
    "fga2_per_min": FLOAT_DTYPE,
    "fga3_per_min": FLOAT_DTYPE,
    "fta_per_min": FLOAT_DTYPE,
    "ast_per_min": FLOAT_DTYPE,
    "tov_per_min": FLOAT_DTYPE,
    "oreb_per_min": FLOAT_DTYPE,
    "dreb_per_min": FLOAT_DTYPE,
    "stl_per_min": FLOAT_DTYPE,
    "blk_per_min": FLOAT_DTYPE,
    "fg2_pct_label": FLOAT_DTYPE,
    "fg3_pct_label": FLOAT_DTYPE,
    "ft_pct_label": FLOAT_DTYPE,
    # Optional booleans are allowed; kept for consistency with TableSchema typing.
    "has_odds": BOOL_DTYPE,
}


RATES_TRAINING_BASE_SCHEMA = TableSchema(
    name="rates_training_base",
    columns=RATES_TRAINING_BASE_REQUIRED_COLUMNS,
    pandas_dtypes=_DTYPES,
    primary_key=("game_id", "player_id", "team_id"),
)


class TrainingBaseSchemaMismatchError(ValueError):
    """Raised when rates_training_base is missing required columns."""


def validate_rates_training_base(df, *, strict: bool = True) -> list[str]:
    expected = set(RATES_TRAINING_BASE_SCHEMA.columns)
    actual = set(getattr(df, "columns", []))
    missing = sorted(expected - actual)
    if missing and strict:
        raise TrainingBaseSchemaMismatchError(
            f"{RATES_TRAINING_BASE_SCHEMA.name} missing required columns: {missing}"
        )
    return missing


__all__ = [
    "RATES_TRAINING_BASE_REQUIRED_COLUMNS",
    "RATES_TRAINING_BASE_SCHEMA",
    "TrainingBaseSchemaMismatchError",
    "validate_rates_training_base",
]
