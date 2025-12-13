from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

STRING_DTYPE = "string[pyarrow]"
INT_DTYPE = "Int64"
FLOAT_DTYPE = "Float64"
BOOL_DTYPE = "boolean"
UTC_TS = "datetime64[ns, UTC]"
NAIVE_TS = "datetime64[ns]"


class SchemaError(ValueError):
    """Raised when a dataframe cannot be aligned to a declared schema."""


@dataclass(frozen=True)
class TableSchema:
    """Authoritative schema definition for a parquet output."""

    name: str
    columns: tuple[str, ...]
    pandas_dtypes: Mapping[str, str]
    primary_key: tuple[str, ...]
    defaults: Mapping[str, Any] = field(default_factory=dict)
    enforce_primary_key: bool = True

    @property
    def optional_columns(self) -> set[str]:
        return set(self.defaults.keys())


def as_pandas_dtypes(schema: TableSchema) -> Mapping[str, str]:
    """Return a shallow copy of the dtype mapping to avoid accidental mutation."""

    return dict(schema.pandas_dtypes)


def _import_pandera():
    """Import pandera components on demand to avoid a hard runtime dependency."""

    try:
        from pandera.pandas import Check, Column, DataFrameSchema
    except ImportError as exc:  # pragma: no cover - exercised indirectly in tests
        raise RuntimeError(
            "Pandera is required for schema validation. Install it with `pip install pandera`."
        ) from exc
    return Check, Column, DataFrameSchema


def to_pandera_schema(schema: TableSchema):
    """Build a Pandera DataFrameSchema from a TableSchema definition."""

    Check, Column, DataFrameSchema = _import_pandera()
    columns = {}
    for column in schema.columns:
        dtype = schema.pandas_dtypes.get(column, STRING_DTYPE)
        columns[column] = Column(dtype, nullable=True, required=True, coerce=True)
    checks = []
    if schema.primary_key and schema.enforce_primary_key:
        pk_cols = list(schema.primary_key)

        def _pk_unique(df: pd.DataFrame) -> bool:
            return not df.duplicated(pk_cols).any()

        checks.append(
            Check(
                _pk_unique,
                error=f"{schema.name} primary key {schema.primary_key} contains duplicates",
            )
        )
    return DataFrameSchema(
        columns,
        checks=checks,
        coerce=True,
        strict=True,
        ordered=True,
        name=schema.name,
    )


def validate_with_pandera(
    df: pd.DataFrame,
    schema: TableSchema,
    *,
    lazy: bool = True,
) -> pd.DataFrame:
    """Validate a dataframe using the Pandera representation of the schema."""

    pandera_schema = to_pandera_schema(schema)
    return pandera_schema.validate(df, lazy=lazy)


def _cast_column(series: pd.Series, dtype: str) -> pd.Series:
    """Cast a column into the requested dtype while handling datetime nuances."""

    if dtype == NAIVE_TS:
        converted = pd.to_datetime(series, errors="coerce")
        if getattr(converted.dt, "tz", None) is not None:
            converted = converted.dt.tz_convert(None)
        return converted
    if dtype == UTC_TS:
        converted = pd.to_datetime(series, errors="coerce", utc=True)
        return converted
    return series.astype(dtype)


def _ensure_optional_column(
    df: pd.DataFrame,
    *,
    column: str,
    default: Any,
    dtype: str,
) -> None:
    """Add an optional column filled with the provided default."""

    if column in df.columns:
        return
    if df.empty:
        df[column] = pd.Series([], dtype=dtype)
        return
    df[column] = pd.Series(default, index=df.index)


def enforce_schema(
    df: pd.DataFrame,
    schema: TableSchema,
    *,
    allow_missing_optional: bool = False,
) -> pd.DataFrame:
    """Align a dataframe to the declared schema.

    Parameters
    ----------
    df:
        Input dataframe that may require reordering, dtype coercion, or default fills.
    schema:
        Target TableSchema definition.
    allow_missing_optional:
        When True, optional columns (those listed in ``schema.defaults``) may be missing
        from ``df`` and will be created with their default values. When False, missing
        optional columns raise SchemaError.
    """

    working = df.copy()
    missing = [col for col in schema.columns if col not in working.columns]
    optional_missing = [col for col in missing if col in schema.optional_columns]
    required_missing = [col for col in missing if col not in schema.optional_columns]
    if required_missing:
        raise SchemaError(
            f"{schema.name} missing required columns: {', '.join(sorted(required_missing))}"
        )
    if optional_missing:
        if not allow_missing_optional:
            raise SchemaError(
                f"{schema.name} missing optional columns: {', '.join(sorted(optional_missing))}"
            )
        for column in optional_missing:
            default_value = schema.defaults[column]
            dtype = schema.pandas_dtypes.get(column, STRING_DTYPE)
            _ensure_optional_column(working, column=column, default=default_value, dtype=dtype)

    for column, default_value in schema.defaults.items():
        if column in working.columns:
            working[column] = working[column].fillna(default_value)

    dtype_map = schema.pandas_dtypes
    for column, dtype in dtype_map.items():
        if column not in working.columns:
            continue
        working[column] = _cast_column(working[column], dtype)

    ordered_columns = list(schema.columns)
    return working.loc[:, ordered_columns]


INJURIES_RAW_SCHEMA = TableSchema(
    name="injuries_raw",
    columns=(
        "report_date",
        "as_of_ts",
        "team_id",
        "player_name",
        "player_id",
        "status_raw",
        "notes_raw",
        "game_id",
        "ingested_ts",
        "source",
        "source_row_id",
        "status",
        "restriction_flag",
        "ramp_flag",
        "games_since_return",
        "days_since_return",
    ),
    pandas_dtypes={
        "report_date": NAIVE_TS,
        "as_of_ts": UTC_TS,
        "team_id": INT_DTYPE,
        "player_name": STRING_DTYPE,
        "player_id": INT_DTYPE,
        "status_raw": STRING_DTYPE,
        "notes_raw": STRING_DTYPE,
        "game_id": INT_DTYPE,
        "ingested_ts": UTC_TS,
        "source": STRING_DTYPE,
        "source_row_id": STRING_DTYPE,
        "status": STRING_DTYPE,
        "restriction_flag": BOOL_DTYPE,
        "ramp_flag": BOOL_DTYPE,
        "games_since_return": INT_DTYPE,
        "days_since_return": INT_DTYPE,
    },
    primary_key=(
        "report_date",
        "as_of_ts",
        "team_id",
        "player_name",
        "source_row_id",
    ),
    defaults={
        "games_since_return": pd.NA,
        "days_since_return": pd.NA,
    },
)

INJURIES_SNAPSHOT_SCHEMA = TableSchema(
    name="injuries_snapshot",
    columns=(
        "game_id",
        "player_id",
        "as_of_ts",
        "status",
        "restriction_flag",
        "ramp_flag",
        "games_since_return",
        "days_since_return",
        "ingested_ts",
        "source",
        "selection_rule",
        "snapshot_missing",
    ),
    pandas_dtypes={
        "game_id": INT_DTYPE,
        "player_id": INT_DTYPE,
        "as_of_ts": UTC_TS,
        "status": STRING_DTYPE,
        "restriction_flag": BOOL_DTYPE,
        "ramp_flag": BOOL_DTYPE,
        "games_since_return": INT_DTYPE,
        "days_since_return": INT_DTYPE,
        "ingested_ts": UTC_TS,
        "source": STRING_DTYPE,
        "selection_rule": STRING_DTYPE,
        "snapshot_missing": INT_DTYPE,
    },
    primary_key=("game_id", "player_id"),
)

ODDS_RAW_SCHEMA = TableSchema(
    name="odds_raw",
    columns=(
        "game_id",
        "as_of_ts",
        "book",
        "market",
        "home_team_id",
        "away_team_id",
        "spread_home",
        "total",
        "ingested_ts",
        "source",
    ),
    pandas_dtypes={
        "game_id": INT_DTYPE,
        "as_of_ts": UTC_TS,
        "book": STRING_DTYPE,
        "market": STRING_DTYPE,
        "home_team_id": INT_DTYPE,
        "away_team_id": INT_DTYPE,
        "spread_home": FLOAT_DTYPE,
        "total": FLOAT_DTYPE,
        "ingested_ts": UTC_TS,
        "source": STRING_DTYPE,
    },
    primary_key=("game_id", "as_of_ts", "book", "market"),
)

ODDS_SNAPSHOT_SCHEMA = TableSchema(
    name="odds_snapshot",
    columns=(
        "game_id",
        "as_of_ts",
        "spread_home",
        "total",
        "book",
        "book_pref",
        "ingested_ts",
        "source",
    ),
    pandas_dtypes={
        "game_id": INT_DTYPE,
        "as_of_ts": UTC_TS,
        "spread_home": FLOAT_DTYPE,
        "total": FLOAT_DTYPE,
        "book": STRING_DTYPE,
        "book_pref": STRING_DTYPE,
        "ingested_ts": UTC_TS,
        "source": STRING_DTYPE,
    },
    primary_key=("game_id",),
    defaults={
        "book_pref": pd.NA,
    },
)

ROSTER_NIGHTLY_RAW_SCHEMA = TableSchema(
    name="roster_nightly_raw",
    columns=(
        "game_id",
        "team_id",
        "game_date",
        "player_id",
        "player_name",
        "active_flag",
        "starter_flag",
        "listed_pos",
        "ingested_ts",
        "source",
        "as_of_ts",
    ),
    pandas_dtypes={
        "game_id": INT_DTYPE,
        "team_id": INT_DTYPE,
        "game_date": NAIVE_TS,
        "player_id": INT_DTYPE,
        "player_name": STRING_DTYPE,
        "active_flag": BOOL_DTYPE,
        "starter_flag": BOOL_DTYPE,
        "listed_pos": STRING_DTYPE,
        "ingested_ts": UTC_TS,
        "source": STRING_DTYPE,
        "as_of_ts": UTC_TS,
    },
    primary_key=("team_id", "game_date", "player_id"),
    enforce_primary_key=False,
)

ROSTER_NIGHTLY_SCHEMA = TableSchema(
    name="roster_nightly",
    columns=(
        "game_id",
        "team_id",
        "player_id",
        "player_name",
        "game_date",
        "as_of_ts",
        "active_flag",
        "starter_flag",
        "lineup_role",
        "lineup_status",
        "lineup_roster_status",
        "lineup_timestamp",
        "is_projected_starter",
        "is_confirmed_starter",
        "listed_pos",
        "height_in",
        "weight_lb",
        "age",
        "ingested_ts",
        "source",
    ),
    pandas_dtypes={
        "game_id": INT_DTYPE,
        "team_id": INT_DTYPE,
        "player_id": INT_DTYPE,
        "player_name": STRING_DTYPE,
        "game_date": NAIVE_TS,
        "as_of_ts": UTC_TS,
        "active_flag": BOOL_DTYPE,
        "starter_flag": BOOL_DTYPE,
        "lineup_role": STRING_DTYPE,
        "lineup_status": STRING_DTYPE,
        "lineup_roster_status": STRING_DTYPE,
        "lineup_timestamp": UTC_TS,
        "is_projected_starter": BOOL_DTYPE,
        "is_confirmed_starter": BOOL_DTYPE,
        "listed_pos": STRING_DTYPE,
        "height_in": INT_DTYPE,
        "weight_lb": INT_DTYPE,
        "age": FLOAT_DTYPE,
        "ingested_ts": UTC_TS,
        "source": STRING_DTYPE,
    },
    primary_key=("team_id", "game_date", "player_id"),
    defaults={
        "height_in": pd.NA,
        "weight_lb": pd.NA,
        "age": pd.NA,
        "lineup_role": pd.NA,
        "lineup_status": pd.NA,
        "lineup_roster_status": pd.NA,
        "lineup_timestamp": pd.NaT,
        "is_projected_starter": False,
        "is_confirmed_starter": False,
    },
    enforce_primary_key=False,
)

SCHEDULE_SCHEMA = TableSchema(
    name="schedule",
    columns=(
        "game_id",
        "game_code",
        "season",
        "game_date",
        "tip_day",
        "tip_ts",
        "home_team_id",
        "home_team_name",
        "home_team_city",
        "home_team_tricode",
        "away_team_id",
        "away_team_name",
        "away_team_city",
        "away_team_tricode",
        "arena_id",
        "arena_name",
        "arena_city",
        "arena_state",
    ),
    pandas_dtypes={
        "game_id": INT_DTYPE,
        "game_code": STRING_DTYPE,
        "season": STRING_DTYPE,
        "game_date": NAIVE_TS,
        "tip_day": NAIVE_TS,
        "tip_ts": UTC_TS,
        "home_team_id": INT_DTYPE,
        "home_team_name": STRING_DTYPE,
        "home_team_city": STRING_DTYPE,
        "home_team_tricode": STRING_DTYPE,
        "away_team_id": INT_DTYPE,
        "away_team_name": STRING_DTYPE,
        "away_team_city": STRING_DTYPE,
        "away_team_tricode": STRING_DTYPE,
        "arena_id": STRING_DTYPE,
        "arena_name": STRING_DTYPE,
        "arena_city": STRING_DTYPE,
        "arena_state": STRING_DTYPE,
    },
    primary_key=("game_id",),
    defaults={
        "arena_id": pd.NA,
    },
)

BOX_SCORE_LABELS_SCHEMA = TableSchema(
    name="boxscore_labels",
    columns=(
        "game_id",
        "player_id",
        "team_id",
        "player_name",
        "season",
        "game_date",
        "minutes",
        "starter_flag",
        "starter_flag_label",
        "source",
        "label_frozen_ts",
    ),
    pandas_dtypes={
        "game_id": INT_DTYPE,
        "player_id": INT_DTYPE,
        "team_id": INT_DTYPE,
        "player_name": STRING_DTYPE,
        "season": STRING_DTYPE,
        "game_date": NAIVE_TS,
        "minutes": FLOAT_DTYPE,
        "starter_flag": INT_DTYPE,
        "starter_flag_label": INT_DTYPE,
        "source": STRING_DTYPE,
        "label_frozen_ts": UTC_TS,
    },
    primary_key=("game_id", "player_id"),
)

FEATURES_MINUTES_V1_SCHEMA = TableSchema(
    name="features_minutes_v1",
    columns=(
        "game_id",
        "player_id",
        "team_id",
        "player_name",
        "team_name",
        "team_tricode",
        "season",
        "game_date",
        "minutes",
        "tip_ts",
        "home_team_id",
        "away_team_id",
        "home_flag",
        "opponent_team_id",
        "opponent_team_name",
        "opponent_team_tricode",
        "status",
        "restriction_flag",
        "ramp_flag",
        "games_since_return",
        "days_since_return",
        "injury_as_of_ts",
        "prior_play_prob",
        "is_out",
        "is_q",
        "is_prob",
        "injury_snapshot_missing",
        "spread_home",
        "total",
        "odds_as_of_ts",
        "blowout_index",
        "blowout_risk_score",
        "close_game_score",
        "available_B",
        "available_G",
        "available_W",
        "depth_same_pos_active",
        "archetype",
        "pos_bucket",
        "roster_as_of_ts",
        "lineup_role",
        "lineup_status",
        "lineup_roster_status",
        "lineup_timestamp",
        "is_projected_starter",
        "is_confirmed_starter",
        "same_archetype_overlap",
        "min_last1",
        "min_last3",
        "min_last5",
        "sum_min_7d",
        "roll_mean_3",
        "roll_mean_5",
        "roll_mean_10",
        "roll_iqr_5",
        "rotation_minutes_std_5g",
        "z_vs_10",
        "role_change_rate_10g",
        "season_phase",
        "starter_flag",
        "starter_prev_game_asof",
        "recent_start_pct_10",
        "days_since_last",
        "is_b2b",
        "is_3in4",
        "is_4in6",
        "team_minutes_dispersion_prior",
        "arch_delta_sum",
        "arch_delta_same_pos",
        "arch_delta_max_role",
        "arch_delta_min_role",
        "arch_missing_same_pos_count",
        "arch_missing_total_count",
        "feature_as_of_ts",
    ),
    pandas_dtypes={
        "game_id": INT_DTYPE,
        "player_id": INT_DTYPE,
        "team_id": INT_DTYPE,
        "player_name": STRING_DTYPE,
        "team_name": STRING_DTYPE,
        "team_tricode": STRING_DTYPE,
        "season": STRING_DTYPE,
        "game_date": NAIVE_TS,
        "minutes": FLOAT_DTYPE,
        "tip_ts": UTC_TS,
        "home_team_id": INT_DTYPE,
        "away_team_id": INT_DTYPE,
        "home_flag": INT_DTYPE,
        "opponent_team_id": INT_DTYPE,
        "opponent_team_name": STRING_DTYPE,
        "opponent_team_tricode": STRING_DTYPE,
        "status": STRING_DTYPE,
        "restriction_flag": BOOL_DTYPE,
        "ramp_flag": BOOL_DTYPE,
        "games_since_return": INT_DTYPE,
        "days_since_return": INT_DTYPE,
        "injury_as_of_ts": UTC_TS,
        "prior_play_prob": FLOAT_DTYPE,
        "is_out": INT_DTYPE,
        "is_q": INT_DTYPE,
        "is_prob": INT_DTYPE,
        "injury_snapshot_missing": INT_DTYPE,
        "spread_home": FLOAT_DTYPE,
        "total": FLOAT_DTYPE,
        "odds_as_of_ts": UTC_TS,
        "blowout_index": FLOAT_DTYPE,
        "blowout_risk_score": FLOAT_DTYPE,
        "close_game_score": FLOAT_DTYPE,
        "available_B": INT_DTYPE,
        "available_G": INT_DTYPE,
        "available_W": INT_DTYPE,
        "depth_same_pos_active": INT_DTYPE,
        "archetype": STRING_DTYPE,
        "pos_bucket": STRING_DTYPE,
        "roster_as_of_ts": UTC_TS,
        "lineup_role": STRING_DTYPE,
        "lineup_status": STRING_DTYPE,
        "lineup_roster_status": STRING_DTYPE,
        "lineup_timestamp": UTC_TS,
        "is_projected_starter": BOOL_DTYPE,
        "is_confirmed_starter": BOOL_DTYPE,
        "same_archetype_overlap": INT_DTYPE,
        "min_last1": FLOAT_DTYPE,
        "min_last3": FLOAT_DTYPE,
        "min_last5": FLOAT_DTYPE,
        "sum_min_7d": FLOAT_DTYPE,
        "roll_mean_3": FLOAT_DTYPE,
        "roll_mean_5": FLOAT_DTYPE,
        "roll_mean_10": FLOAT_DTYPE,
        "roll_iqr_5": FLOAT_DTYPE,
        "rotation_minutes_std_5g": FLOAT_DTYPE,
        "z_vs_10": FLOAT_DTYPE,
        "role_change_rate_10g": FLOAT_DTYPE,
        "season_phase": FLOAT_DTYPE,
        "starter_flag": INT_DTYPE,
        "starter_prev_game_asof": FLOAT_DTYPE,
        "recent_start_pct_10": FLOAT_DTYPE,
        "days_since_last": INT_DTYPE,
        "is_b2b": INT_DTYPE,
        "is_3in4": INT_DTYPE,
        "is_4in6": INT_DTYPE,
        "team_minutes_dispersion_prior": FLOAT_DTYPE,
        "arch_delta_sum": FLOAT_DTYPE,
        "arch_delta_same_pos": FLOAT_DTYPE,
        "arch_delta_max_role": FLOAT_DTYPE,
        "arch_delta_min_role": FLOAT_DTYPE,
        "arch_missing_same_pos_count": INT_DTYPE,
        "arch_missing_total_count": INT_DTYPE,
        "feature_as_of_ts": UTC_TS,
    },
    primary_key=("game_id", "player_id", "team_id"),
    defaults={
        "archetype": pd.NA,
        "pos_bucket": pd.NA,
        "arch_delta_sum": 0.0,
        "arch_delta_same_pos": 0.0,
        "arch_delta_max_role": 0.0,
        "arch_delta_min_role": 0.0,
        "arch_missing_same_pos_count": 0,
        "arch_missing_total_count": 0,
        "depth_same_pos_active": 0,
        "starter_flag": 0,
    },
)

SLATE_FEATURES_MINUTES_V1_SCHEMA = TableSchema(
    name="slate_features_minutes_v1",
    columns=FEATURES_MINUTES_V1_SCHEMA.columns
    + (
        "snapshot_type",
        "snapshot_ts",
        "frozen_at",
    ),
    pandas_dtypes={
        **FEATURES_MINUTES_V1_SCHEMA.pandas_dtypes,
        "snapshot_type": STRING_DTYPE,
        "snapshot_ts": UTC_TS,
        "frozen_at": UTC_TS,
    },
    primary_key=("game_id", "player_id", "team_id", "snapshot_type"),
)

SCHEDULE_STATIC_SCHEMA = TableSchema(
    name="schedule_static",
    columns=(
        "game_id",
        "season",
        "game_date",
        "tip_ts",
        "home_team_id",
        "away_team_id",
        "arena_id",
        "home_city",
        "away_city",
    ),
    pandas_dtypes={
        "game_id": INT_DTYPE,
        "season": STRING_DTYPE,
        "game_date": NAIVE_TS,
        "tip_ts": UTC_TS,
        "home_team_id": INT_DTYPE,
        "away_team_id": INT_DTYPE,
        "arena_id": STRING_DTYPE,
        "home_city": STRING_DTYPE,
        "away_city": STRING_DTYPE,
    },
    primary_key=("game_id",),
    defaults={
        "arena_id": pd.NA,
    },
)

COACH_TENURE_SCHEMA = TableSchema(
    name="coach_tenure",
    columns=(
        "coach_id",
        "coach_name",
        "team_id",
        "start_date",
        "end_date",
    ),
    pandas_dtypes={
        "coach_id": INT_DTYPE,
        "coach_name": STRING_DTYPE,
        "team_id": INT_DTYPE,
        "start_date": NAIVE_TS,
        "end_date": NAIVE_TS,
    },
    primary_key=("coach_id", "team_id", "start_date"),
    defaults={
        "end_date": pd.NaT,
    },
)

ARENA_TIMEZONE_SCHEMA = TableSchema(
    name="arena_tz_map",
    columns=(
        "arena_id",
        "arena_name",
        "tz",
        "city",
        "state",
    ),
    pandas_dtypes={
        "arena_id": STRING_DTYPE,
        "arena_name": STRING_DTYPE,
        "tz": STRING_DTYPE,
        "city": STRING_DTYPE,
        "state": STRING_DTYPE,
    },
    primary_key=("arena_id",),
)

__all__ = [
    "ARENA_TIMEZONE_SCHEMA",
    "BOX_SCORE_LABELS_SCHEMA",
    "COACH_TENURE_SCHEMA",
    "FEATURES_MINUTES_V1_SCHEMA",
    "SLATE_FEATURES_MINUTES_V1_SCHEMA",
    "INJURIES_RAW_SCHEMA",
    "INJURIES_SNAPSHOT_SCHEMA",
    "ODDS_RAW_SCHEMA",
    "ODDS_SNAPSHOT_SCHEMA",
    "ROSTER_NIGHTLY_RAW_SCHEMA",
    "ROSTER_NIGHTLY_SCHEMA",
    "SCHEDULE_SCHEMA",
    "SCHEDULE_STATIC_SCHEMA",
    "SchemaError",
    "TableSchema",
    "as_pandas_dtypes",
    "enforce_schema",
    "to_pandera_schema",
    "validate_with_pandera",
]
