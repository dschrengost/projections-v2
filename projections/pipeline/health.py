"""Lightweight pipeline health checks that hard-fail bad runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


class DataContractError(RuntimeError):
    """Raised when a stage violates an expected data contract."""


@dataclass(frozen=True, slots=True)
class HealthReport:
    label: str
    passed: bool
    metrics: dict[str, float]
    errors: list[str]


def _coerce_ts(value: str | datetime | None) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        ts = pd.Timestamp(value)
    else:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def require_file(path: Path, *, label: str) -> None:
    if not path.exists():
        raise DataContractError(f"[health] missing {label}: {path}")


def require_columns(df: pd.DataFrame, *, required: Iterable[str], label: str) -> None:
    required_set = set(required)
    missing = sorted([c for c in required_set if c not in df.columns])
    if missing:
        raise DataContractError(f"[health] {label}: missing required columns: {missing}")


def require_non_null(df: pd.DataFrame, *, cols: Iterable[str], label: str) -> None:
    cols_list = [c for c in cols if c in df.columns]
    if not cols_list:
        return
    nulls = {c: int(df[c].isna().sum()) for c in cols_list}
    bad = {c: n for c, n in nulls.items() if n > 0}
    if bad:
        raise DataContractError(f"[health] {label}: nulls in key columns: {bad}")


def require_row_count(df: pd.DataFrame, *, min_rows: int, max_rows: int | None, label: str) -> None:
    n = int(len(df))
    if n < min_rows:
        raise DataContractError(f"[health] {label}: row_count={n} below min_rows={min_rows}")
    if max_rows is not None and n > max_rows:
        raise DataContractError(f"[health] {label}: row_count={n} above max_rows={max_rows}")


def require_game_date(df: pd.DataFrame, *, game_date: str, label: str) -> None:
    if "game_date" not in df.columns or df.empty:
        return
    normalized = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    target = pd.Timestamp(game_date).date()
    bad = normalized.notna() & (normalized != target)
    if bad.any():
        sample = sorted(set(normalized.loc[bad].astype(str).head(5).tolist()))
        raise DataContractError(
            f"[health] {label}: game_date mismatch (target={target}, sample_bad={sample})"
        )


def require_freshness(
    df: pd.DataFrame,
    *,
    ts_cols: Iterable[str],
    reference_ts: str | datetime,
    max_age_minutes: float,
    label: str,
) -> HealthReport:
    errors: list[str] = []
    metrics: dict[str, float] = {}

    ref = _coerce_ts(reference_ts)
    if ref is None:
        raise DataContractError(f"[health] {label}: invalid reference_ts={reference_ts!r}")

    present_cols = [c for c in ts_cols if c in df.columns]
    if not present_cols:
        return HealthReport(label=label, passed=True, metrics={}, errors=[])

    newest: pd.Timestamp | None = None
    nulls = 0
    for col in present_cols:
        series = pd.to_datetime(df[col], utc=True, errors="coerce")
        nulls += int(series.isna().sum())
        col_newest = series.max()
        if pd.notna(col_newest):
            newest = col_newest if newest is None else max(newest, col_newest)

    _require(newest is not None, f"{label}: could not parse any timestamps from {present_cols}", errors)
    if newest is not None:
        age_minutes = float((ref - newest).total_seconds() / 60.0)
        metrics["freshness_age_minutes"] = age_minutes
        _require(
            age_minutes <= max_age_minutes,
            f"{label}: newest_ts too old (age_minutes={age_minutes:.1f} > {max_age_minutes:.1f})",
            errors,
        )

    if nulls:
        metrics["null_ts_cells"] = float(nulls)

    if errors:
        raise DataContractError("[health] " + "; ".join(errors))
    return HealthReport(label=label, passed=True, metrics=metrics, errors=[])


def require_minutes_sanity(
    df: pd.DataFrame,
    *,
    label: str,
    minutes_col_candidates: Iterable[str] = ("effective_minutes", "minutes_final", "minutes_p50_cond", "minutes_p50", "minutes_pred_p50"),
    team_target: float = 240.0,
    team_tolerance: float = 15.0,
    max_active_players: int = 20,
) -> HealthReport:
    errors: list[str] = []
    metrics: dict[str, float] = {}

    if df.empty:
        raise DataContractError(f"[health] {label}: empty dataframe")

    minutes_col = next((c for c in minutes_col_candidates if c in df.columns), None)
    if minutes_col is None:
        raise DataContractError(f"[health] {label}: missing minutes column (candidates={list(minutes_col_candidates)})")

    required = {"game_id", "team_id", minutes_col}
    require_columns(df, required=required, label=label)
    require_non_null(df, cols=["game_id", "team_id"], label=label)

    work = df.copy()
    work[minutes_col] = pd.to_numeric(work[minutes_col], errors="coerce").fillna(0.0).astype(float)
    active_mask = work[minutes_col] > 0.5

    # Team totals near 240.
    team_totals = work.loc[active_mask].groupby(["game_id", "team_id"])[minutes_col].sum()
    if team_totals.empty:
        raise DataContractError(f"[health] {label}: no active minutes found")
    deviations = (team_totals - team_target).abs()
    metrics["max_team_minutes_deviation"] = float(deviations.max())
    metrics["teams_checked"] = float(len(team_totals))
    too_far = deviations > team_tolerance
    if too_far.any():
        worst = deviations.idxmax()
        worst_val = float(deviations.max())
        errors.append(f"team minutes off target: team={worst} dev={worst_val:.1f} (tol={team_tolerance})")

    # Rotation sanity: too many active players per team.
    active_counts = work.loc[active_mask].groupby(["game_id", "team_id"])["player_id"].nunique()
    metrics["max_active_players_per_team"] = float(active_counts.max())
    if (active_counts > max_active_players).any():
        worst_team = active_counts.idxmax()
        worst_n = int(active_counts.max())
        errors.append(f"too many active players: team={worst_team} active_players={worst_n} (max={max_active_players})")

    if errors:
        raise DataContractError("[health] " + "; ".join(errors))
    return HealthReport(label=label, passed=True, metrics=metrics, errors=[])
