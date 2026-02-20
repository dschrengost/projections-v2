"""Prop-derived minutes features.

This module derives a rough "implied minutes" signal from Action Network prop lines by
dividing an activity proxy (PRA line or PTS+REB+AST sum) by an observed PRA-per-minute
prior computed from historical boxscores.

Design constraints:
- Leakage-safe: priors use only games strictly before the target game date.
- No dependency on rates model predictions (avoids circular scoring graphs).
- Best-effort: if history/props are missing, fill conservative defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FPTS_TRAINING_BASE_FILENAME = "fpts_training_base.parquet"

DEFAULT_BASELINE_PRA_PER_MIN: float = 1.0
DEFAULT_ALPHA_MINUTES: float = 50.0
DEFAULT_MIN_RATE: float = 0.25
DEFAULT_MAX_MINUTES: float = 48.0
DEFAULT_LOOKBACK_DAYS: int = 365


def _to_date(value: object) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, errors="coerce")
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def _safe_float_series(series: pd.Series, *, fill: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(float(fill)).astype(float)


def _safe_int_series(series: pd.Series, *, fill: int = 0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(int(fill)).astype(int)


def _resolve_season_from_date(day: pd.Timestamp) -> int:
    day = pd.Timestamp(day).normalize()
    return int(day.year if day.month >= 8 else day.year - 1)


def _season_start_day(season: int) -> pd.Timestamp:
    return pd.Timestamp(f"{int(season)}-08-01")


def _season_end_day(season: int) -> pd.Timestamp:
    return pd.Timestamp(f"{int(season) + 1}-07-31")


def _discover_partition_paths(season_root: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    paths: list[Path] = []
    for day_dir in sorted(season_root.glob("game_date=*")):
        try:
            day_str = day_dir.name.split("=", 1)[1]
        except IndexError:
            continue
        day = _to_date(day_str)
        if day is None:
            continue
        if day < start or day > end:
            continue
        candidate = day_dir / FPTS_TRAINING_BASE_FILENAME
        if candidate.exists():
            paths.append(candidate)
    return paths


def load_fpts_training_base_history(
    *,
    data_root: Path,
    season: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    player_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Load minimal boxscore stats history from gold/fpts_training_base partitions."""

    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end).normalize()
    if start_day > end_day:
        return pd.DataFrame()

    season_root = data_root / "gold" / "fpts_training_base" / f"season={int(season)}"
    if not season_root.exists():
        return pd.DataFrame()

    paths = _discover_partition_paths(season_root, start=start_day, end=end_day)
    if not paths:
        return pd.DataFrame()

    cols = ["game_id", "team_id", "player_id", "game_date", "minutes_actual", "pts", "reb", "ast"]
    ids_set = set(int(x) for x in player_ids) if player_ids is not None else None

    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_parquet(path, columns=cols)
        except Exception:
            continue
        if ids_set is not None and "player_id" in frame.columns:
            pid = pd.to_numeric(frame["player_id"], errors="coerce")
            frame = frame.loc[pid.isin(ids_set)].copy()
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()
    for col in ("game_id", "team_id", "player_id"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    for col in ("minutes_actual", "pts", "reb", "ast"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["game_id", "player_id", "game_date"], how="any").copy()
    return out.reset_index(drop=True)


def load_fpts_training_base_history_multi_season(
    *,
    data_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    player_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Load fpts_training_base history spanning multiple seasons by game_date range."""

    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end).normalize()
    if start_day > end_day:
        return pd.DataFrame()

    start_season = _resolve_season_from_date(start_day)
    end_season = _resolve_season_from_date(end_day)
    frames: list[pd.DataFrame] = []
    for season in range(int(start_season), int(end_season) + 1):
        season_start = _season_start_day(season)
        season_end = _season_end_day(season)
        slice_start = max(start_day, season_start)
        slice_end = min(end_day, season_end)
        if slice_start > slice_end:
            continue
        frame = load_fpts_training_base_history(
            data_root=data_root,
            season=season,
            start=slice_start,
            end=slice_end,
            player_ids=player_ids,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class PraPriorConfig:
    baseline_pra_per_min: float = DEFAULT_BASELINE_PRA_PER_MIN
    alpha_minutes: float = DEFAULT_ALPHA_MINUTES
    min_rate: float = DEFAULT_MIN_RATE
    max_minutes: float = DEFAULT_MAX_MINUTES


def compute_player_game_pra_priors(
    history: pd.DataFrame,
    *,
    config: PraPriorConfig = PraPriorConfig(),
) -> pd.DataFrame:
    """Compute per-(game_id, player_id) priors using only prior games for each player."""

    if history.empty:
        return pd.DataFrame(columns=["game_id", "player_id", "an_pra_per_min_prior", "an_pra_prior_minutes_sum", "an_pra_prior_games"])

    work = history.copy()
    work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce").dt.normalize()
    work["game_id"] = pd.to_numeric(work["game_id"], errors="coerce").astype("Int64")
    work["player_id"] = pd.to_numeric(work["player_id"], errors="coerce").astype("Int64")
    minutes = _safe_float_series(work.get("minutes_actual", pd.Series(0.0, index=work.index)))
    pts = _safe_float_series(work.get("pts", pd.Series(0.0, index=work.index)))
    reb = _safe_float_series(work.get("reb", pd.Series(0.0, index=work.index)))
    ast = _safe_float_series(work.get("ast", pd.Series(0.0, index=work.index)))

    minutes = minutes.clip(lower=0.0)
    pra = (pts + reb + ast).clip(lower=0.0)
    played = (minutes > 0.0).astype(int)

    work["_pra"] = np.where(minutes > 0.0, pra, 0.0)
    work["_min"] = np.where(minutes > 0.0, minutes, 0.0)
    work["_played"] = played

    work = work.sort_values(["player_id", "game_date", "game_id"], kind="mergesort").copy()
    grp = work.groupby("player_id", sort=False)

    pra_sum_prev = grp["_pra"].cumsum().shift(1).fillna(0.0)
    min_sum_prev = grp["_min"].cumsum().shift(1).fillna(0.0)
    games_prev = grp["_played"].cumsum().shift(1).fillna(0).astype(int)

    alpha = float(config.alpha_minutes)
    baseline = float(config.baseline_pra_per_min)
    prior_rate = (pra_sum_prev + alpha * baseline) / (min_sum_prev + alpha)

    out = pd.DataFrame(
        {
            "game_id": work["game_id"],
            "player_id": work["player_id"],
            "an_pra_per_min_prior": prior_rate.astype(float),
            "an_pra_prior_minutes_sum": min_sum_prev.astype(float),
            "an_pra_prior_games": games_prev.astype(int),
        }
    )
    out = out.dropna(subset=["game_id", "player_id"]).copy()
    out = out.drop_duplicates(subset=["game_id", "player_id"], keep="last")
    return out.reset_index(drop=True)


def compute_player_pra_priors_asof(
    history: pd.DataFrame,
    *,
    config: PraPriorConfig = PraPriorConfig(),
) -> pd.DataFrame:
    """Compute per-player priors from a slice of historical games (already filtered to <= as_of)."""

    if history.empty:
        return pd.DataFrame(columns=["player_id", "an_pra_per_min_prior", "an_pra_prior_minutes_sum", "an_pra_prior_games"])

    work = history.copy()
    work["player_id"] = pd.to_numeric(work["player_id"], errors="coerce").astype("Int64")
    minutes = _safe_float_series(work.get("minutes_actual", pd.Series(0.0, index=work.index))).clip(lower=0.0)
    pts = _safe_float_series(work.get("pts", pd.Series(0.0, index=work.index)))
    reb = _safe_float_series(work.get("reb", pd.Series(0.0, index=work.index)))
    ast = _safe_float_series(work.get("ast", pd.Series(0.0, index=work.index)))

    pra = (pts + reb + ast).clip(lower=0.0)
    played = (minutes > 0.0).astype(int)

    work["_pra"] = np.where(minutes > 0.0, pra, 0.0)
    work["_min"] = np.where(minutes > 0.0, minutes, 0.0)
    work["_played"] = played

    agg = (
        work.groupby("player_id", dropna=True, sort=False)
        .agg(
            pra_sum=("_pra", "sum"),
            min_sum=("_min", "sum"),
            games=("_played", "sum"),
        )
        .reset_index()
    )
    alpha = float(config.alpha_minutes)
    baseline = float(config.baseline_pra_per_min)
    rate = (pd.to_numeric(agg["pra_sum"], errors="coerce").fillna(0.0) + alpha * baseline) / (
        pd.to_numeric(agg["min_sum"], errors="coerce").fillna(0.0) + alpha
    )
    out = pd.DataFrame(
        {
            "player_id": agg["player_id"].astype("Int64"),
            "an_pra_per_min_prior": rate.astype(float),
            "an_pra_prior_minutes_sum": pd.to_numeric(agg["min_sum"], errors="coerce").fillna(0.0).astype(float),
            "an_pra_prior_games": pd.to_numeric(agg["games"], errors="coerce").fillna(0).astype(int),
        }
    )
    out = out.dropna(subset=["player_id"]).copy()
    return out.reset_index(drop=True)


def attach_prop_implied_minutes(
    df: pd.DataFrame,
    *,
    priors: pd.DataFrame,
    join_keys: tuple[str, ...],
    config: PraPriorConfig = PraPriorConfig(),
    line_col: str = "an_implied_activity_line",
    has_props_col: str = "an_has_any_props",
    is_out_col: str = "is_out",
) -> pd.DataFrame:
    """Attach implied minutes features to a frame containing Action props features."""

    if df.empty:
        out = df.copy()
        out["an_pra_per_min_prior"] = float(config.baseline_pra_per_min)
        out["an_pra_prior_minutes_sum"] = 0.0
        out["an_pra_prior_games"] = 0
        out["an_implied_minutes"] = 0.0
        out["an_has_implied_minutes"] = 0
        out["an_implied_minutes_missing"] = 1
        return out

    work = df.copy()
    pri = priors.copy()

    for key in join_keys:
        if key in work.columns:
            work[key] = pd.to_numeric(work[key], errors="coerce").astype("Int64")
        if key in pri.columns:
            pri[key] = pd.to_numeric(pri[key], errors="coerce").astype("Int64")

    pri = pri.drop_duplicates(subset=list(join_keys), keep="last").copy()
    merged = work.merge(pri, on=list(join_keys), how="left", suffixes=("", "_prior"))

    baseline = float(config.baseline_pra_per_min)
    merged["an_pra_per_min_prior"] = _safe_float_series(merged.get("an_pra_per_min_prior", pd.Series(baseline, index=merged.index)), fill=baseline)
    merged["an_pra_prior_minutes_sum"] = _safe_float_series(merged.get("an_pra_prior_minutes_sum", pd.Series(0.0, index=merged.index)), fill=0.0)
    merged["an_pra_prior_games"] = _safe_int_series(merged.get("an_pra_prior_games", pd.Series(0, index=merged.index)), fill=0)

    line = _safe_float_series(merged.get(line_col, pd.Series(0.0, index=merged.index)), fill=0.0).clip(lower=0.0)
    if has_props_col in merged.columns:
        has_props = pd.to_numeric(merged[has_props_col], errors="coerce").fillna(0.0).astype(float) > 0.0
    else:
        has_props = line > 0.0

    rate = pd.to_numeric(merged["an_pra_per_min_prior"], errors="coerce").fillna(baseline).astype(float)
    rate = rate.clip(lower=float(config.min_rate))

    implied = pd.Series(0.0, index=merged.index, dtype=float)
    implied.loc[has_props] = (line.loc[has_props] / rate.loc[has_props]).astype(float)
    implied = implied.clip(lower=0.0, upper=float(config.max_minutes))

    has_implied = has_props & (line > 0.0)
    implied_missing = ~has_implied

    if is_out_col in merged.columns:
        is_out = pd.to_numeric(merged[is_out_col], errors="coerce").fillna(0.0).astype(float) > 0.0
        implied = implied.where(~is_out, 0.0)
        has_implied = has_implied & (~is_out)
        implied_missing = implied_missing | is_out

    merged["an_implied_minutes"] = implied.astype(float)
    merged["an_has_implied_minutes"] = has_implied.astype(int)
    merged["an_implied_minutes_missing"] = implied_missing.astype(int)
    return merged


__all__ = [
    "FPTS_TRAINING_BASE_FILENAME",
    "DEFAULT_LOOKBACK_DAYS",
    "PraPriorConfig",
    "attach_prop_implied_minutes",
    "compute_player_game_pra_priors",
    "compute_player_pra_priors_asof",
    "load_fpts_training_base_history",
    "load_fpts_training_base_history_multi_season",
]
