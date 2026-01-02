"""DFS slate-day readiness checks (inputs, freshness, invariants).

This module is intentionally pragmatic: it is meant to catch the failure modes
that wreck real DFS slates (stale inputs, run_id mismatches, NaNs, broken team
minute constraints, and projection semantics issues).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


UTC = timezone.utc


@dataclass(frozen=True)
class ReadinessReport:
    game_date: str
    as_of_ts: str
    run_id: str | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_run_id_pointer(day_dir: Path) -> str | None:
    payload = _read_json(day_dir / "latest_run.json")
    run_id = payload.get("run_id") if payload else None
    return str(run_id) if run_id else None


def _newest_parquet_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    if path.is_file() and path.suffix == ".parquet":
        return path
    parquet_files = list(path.glob("**/*.parquet")) if path.is_dir() else []
    if not parquet_files:
        return None
    return max(parquet_files, key=lambda p: p.stat().st_mtime)


def _artifact_max_ts(path: Path, *, time_col: str) -> pd.Timestamp | None:
    newest = _newest_parquet_file(path)
    if newest is None:
        return None
    try:
        df = pd.read_parquet(newest, columns=[time_col])
    except Exception:
        return None
    if time_col not in df.columns or df.empty:
        return None
    ts = pd.to_datetime(df[time_col], utc=True, errors="coerce").dropna()
    return ts.max() if not ts.empty else None


def _age_minutes(now: pd.Timestamp, ts: pd.Timestamp | None) -> float | None:
    if ts is None or pd.isna(ts):
        return None
    delta = now - ts
    return float(delta.total_seconds() / 60.0)


def _resolve_season_start(day: pd.Timestamp) -> int:
    return int(day.year) if int(day.month) >= 8 else int(day.year) - 1


def _load_schedule_for_date(
    *,
    data_root: Path,
    game_day: pd.Timestamp,
    season_start: int,
) -> tuple[pd.DataFrame | None, str | None]:
    month = int(game_day.month)
    candidates = [
        data_root / "silver" / "schedule" / f"season={season_start}" / f"month={month:02d}" / "schedule.parquet",
        data_root / "silver" / "schedule" / f"season={int(game_day.year)}" / f"month={month:02d}" / "schedule.parquet",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if "game_date" in df.columns:
            df = df.copy()
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
            df = df[df["game_date"] == game_day.date()]
        return df, str(path)
    return None, None


def _parse_run_id_timestamp(run_id: str | None) -> pd.Timestamp | None:
    if not run_id:
        return None
    try:
        dt = datetime.strptime(str(run_id), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return None
    return pd.Timestamp(dt)


def run_dfs_readiness(
    *,
    game_date: str,
    data_root: Path,
    run_id: str | None = None,
    as_of_ts: datetime | None = None,
    strict: bool = True,
) -> ReadinessReport:
    now = pd.Timestamp(as_of_ts or datetime.now(tz=UTC)).tz_convert("UTC")
    day = pd.Timestamp(game_date).normalize()
    season_start = _resolve_season_start(day)

    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {
        "season_start": season_start,
    }

    schedule_df, schedule_path = _load_schedule_for_date(
        data_root=data_root,
        game_day=day,
        season_start=season_start,
    )
    metrics["schedule_path"] = schedule_path
    first_tip = None
    if schedule_df is not None and not schedule_df.empty and "tip_ts" in schedule_df.columns:
        tips = pd.to_datetime(schedule_df["tip_ts"], utc=True, errors="coerce").dropna()
        if not tips.empty:
            first_tip = tips.min()
            metrics["first_tip_ts"] = first_tip.isoformat()
            metrics["minutes_to_first_tip"] = float((first_tip - now).total_seconds() / 60.0)
    else:
        warnings.append("schedule: missing or empty (cannot compute time-to-lock windows).")

    # Resolve run_id from pointers when not provided.
    resolved_run_id = run_id
    if resolved_run_id is None:
        projections_day = data_root / "artifacts" / "projections" / game_date
        resolved_run_id = _read_run_id_pointer(projections_day)
        if resolved_run_id is None:
            sim_day = data_root / "artifacts" / "sim_v2" / "worlds_fpts_v2" / f"game_date={game_date}"
            resolved_run_id = _read_run_id_pointer(sim_day)
        if resolved_run_id is None:
            minutes_day = data_root / "artifacts" / "minutes_v1" / "daily" / game_date
            resolved_run_id = _read_run_id_pointer(minutes_day)

    if resolved_run_id is None:
        errors.append(f"run_id: unable to resolve from pointers for game_date={game_date}")

    metrics["run_id"] = resolved_run_id

    # Input freshness (best-effort; checks newest parquet file under each artifact path).
    injuries_path = data_root / "silver" / "injuries_snapshot" / f"season={season_start}"
    espn_path = data_root / "silver" / "espn_injuries" / f"date={game_date}" / "injuries.parquet"
    rotowire_path = data_root / "silver" / "rotowire_lineups" / f"date={game_date}" / "lineups.parquet"
    odds_path = data_root / "silver" / "odds_snapshot" / f"season={season_start}"
    dk_salaries_path = data_root / "gold" / "dk_salaries" / "site=dk" / f"game_date={game_date}"

    input_specs = [
        ("injuries_snapshot", injuries_path, "as_of_ts"),
        ("espn_injuries", espn_path, "as_of_ts"),
        ("rotowire_lineups", rotowire_path, "ingested_ts"),
        ("odds_snapshot", odds_path, "as_of_ts"),
    ]
    input_freshness: dict[str, Any] = {}
    for label, path, ts_col in input_specs:
        max_ts = _artifact_max_ts(path, time_col=ts_col)
        age = _age_minutes(now, max_ts)
        input_freshness[label] = {
            "path": str(path),
            "exists": bool(path.exists()),
            "max_ts": max_ts.isoformat() if max_ts is not None else None,
            "age_minutes": round(age, 2) if age is not None else None,
        }
        if not path.exists():
            warnings.append(f"{label}: missing at {path}")
    metrics["input_freshness"] = input_freshness

    # DK salaries: ensure at least one salaries.parquet exists.
    salary_files = list(dk_salaries_path.glob("draft_group_id=*/salaries.parquet")) if dk_salaries_path.exists() else []
    metrics["dk_salaries"] = {
        "path": str(dk_salaries_path),
        "n_slates": int(len(salary_files)),
    }
    if not salary_files:
        errors.append(f"dk_salaries: missing salaries.parquet under {dk_salaries_path}")

    # Tighten freshness requirements in the last 60/30 minutes before lock.
    minutes_to_tip = metrics.get("minutes_to_first_tip")
    if isinstance(minutes_to_tip, (int, float)):
        window = "normal"
        if 0 < minutes_to_tip <= 30:
            window = "last_30"
            thresholds = {"odds_snapshot": 15.0, "injuries_snapshot": 30.0, "espn_injuries": 30.0, "rotowire_lineups": 30.0}
        elif 0 < minutes_to_tip <= 60:
            window = "last_60"
            thresholds = {"odds_snapshot": 30.0, "injuries_snapshot": 60.0, "espn_injuries": 60.0, "rotowire_lineups": 60.0}
        else:
            thresholds = {}
        metrics["freshness_window"] = window

        for label, max_age in thresholds.items():
            age = input_freshness.get(label, {}).get("age_minutes")
            if age is None:
                if strict:
                    errors.append(f"freshness[{window}]: {label} missing timestamp (cannot verify <= {max_age}m)")
                else:
                    warnings.append(f"freshness[{window}]: {label} missing timestamp (cannot verify <= {max_age}m)")
                continue
            if float(age) > float(max_age):
                msg = f"freshness[{window}]: {label} age={age:.1f}m > {max_age:.1f}m"
                (errors if strict else warnings).append(msg)

    # Output invariants (prefer unified projections; that's what optimizer consumes).
    if resolved_run_id is not None:
        projections_path = data_root / "artifacts" / "projections" / game_date / f"run={resolved_run_id}" / "projections.parquet"
        if not projections_path.exists():
            errors.append(f"projections: missing {projections_path}")
        else:
            df = pd.read_parquet(projections_path)
            metrics["projections_rows"] = int(len(df))

            if "run_as_of_ts" in df.columns:
                parsed = pd.to_datetime(df["run_as_of_ts"], utc=True, errors="coerce").dropna()
                if parsed.empty:
                    warnings.append("projections: run_as_of_ts present but not parseable")
                else:
                    file_asof = parsed.max()
                    metrics["projections_run_as_of_ts"] = file_asof.isoformat()
                    expected_asof = _parse_run_id_timestamp(resolved_run_id)
                    if expected_asof is not None and abs(file_asof - expected_asof) > pd.Timedelta(minutes=2):
                        warnings.append(
                            f"projections: run_as_of_ts {file_asof.isoformat()} mismatches run_id {resolved_run_id}"
                        )

            # Run-id lineage checks.
            if "projections_run_id" in df.columns:
                uniq = [v for v in df["projections_run_id"].dropna().unique().tolist() if v]
                if uniq and str(uniq[0]) != str(resolved_run_id):
                    errors.append(f"projections_run_id mismatch: file={resolved_run_id} col={uniq[0]}")

            has_row_source = "row_source_run_id" in df.columns and "row_source_reason" in df.columns
            if has_row_source and "tip_ts" in df.columns and "is_locked" in df.columns:
                tip_ts = pd.to_datetime(df["tip_ts"], utc=True, errors="coerce")
                src_ts = pd.to_datetime(
                    df["row_source_run_id"].astype(str),
                    utc=True,
                    format="%Y%m%dT%H%M%SZ",
                    errors="coerce",
                )
                locked_mask = df["is_locked"].astype(bool) & tip_ts.notna()
                bad_src = locked_mask & src_ts.notna() & (src_ts > tip_ts + pd.Timedelta(seconds=1))
                bad_count = int(bad_src.sum())
                metrics["locked_rows"] = int(locked_mask.sum())
                metrics["locked_rows_bad_row_source_ts"] = bad_count
                if bad_count:
                    errors.append(
                        f"locked_rows: {bad_count} rows have row_source_run_id after tip_ts (post-lock leakage)"
                    )
                missing_src = locked_mask & src_ts.isna()
                missing_count = int(missing_src.sum())
                if missing_count:
                    warnings.append(
                        f"locked_rows: {missing_count} rows missing parseable row_source_run_id timestamp"
                    )

                # For unlocked rows, minutes/sim run IDs should normally match the resolved run.
                unlocked_mask = (~df["is_locked"].astype(bool)) & df["tip_ts"].notna()
                for col in ("minutes_run_id", "sim_run_id"):
                    if col not in df.columns or resolved_run_id is None:
                        continue
                    values = df.loc[unlocked_mask, col].dropna().astype(str).unique().tolist()
                    values = [v for v in values if v]
                    if values and (len(values) > 1 or values[0] != str(resolved_run_id)):
                        warnings.append(f"{col}: unlocked rows use {values} (expected {resolved_run_id})")
            else:
                for col in ("minutes_run_id", "sim_run_id", "rates_run_id"):
                    if col in df.columns:
                        uniq = [v for v in df[col].dropna().unique().tolist() if v]
                        if uniq and str(uniq[0]) != str(resolved_run_id):
                            warnings.append(
                                f"{col} != run_id (expected same run-scoped id): {uniq[0]} vs {resolved_run_id}"
                            )

            # Basic numeric sanity.
            for col in ("minutes_p50", "dk_fpts_mean", "salary"):
                if col in df.columns:
                    bad = pd.to_numeric(df[col], errors="coerce").isna().sum()
                    if bad:
                        errors.append(f"{col}: found {int(bad)} NaNs")

            if "minutes_p50" in df.columns:
                mins = pd.to_numeric(df["minutes_p50"], errors="coerce")
                if (mins < -1e-6).any() or (mins > 48.0 + 1e-6).any():
                    errors.append("minutes_p50: outside [0,48] range")

                if {"game_id", "team_id"}.issubset(df.columns):
                    active = df[mins > 0].copy()
                    team_sums = active.groupby(["game_id", "team_id"])["minutes_p50"].sum()
                    dev = (team_sums - 240.0).abs()
                    max_dev = float(dev.max()) if len(dev) else 0.0
                    metrics["minutes_team_sum_max_dev"] = max_dev
                    if max_dev > 2.0:
                        errors.append(f"minutes_p50: team sum max deviation {max_dev:.2f} > 2.0")

            if "play_prob" in df.columns:
                p = pd.to_numeric(df["play_prob"], errors="coerce")
                if (p < -1e-6).any() or (p > 1.0 + 1e-6).any():
                    errors.append("play_prob: outside [0,1] range")

            # Conditional vs unconditional semantics checks (when present).
            if {"dk_fpts_mean", "dk_fpts_mean_uncond", "play_prob"}.issubset(df.columns):
                f_cond = pd.to_numeric(df["dk_fpts_mean"], errors="coerce")
                f_uncond = pd.to_numeric(df["dk_fpts_mean_uncond"], errors="coerce")
                p = pd.to_numeric(df["play_prob"], errors="coerce").clip(lower=0.0, upper=1.0)
                bad = (f_uncond > f_cond + 1e-6).sum()
                if bad:
                    errors.append(f"dk_fpts_mean_uncond > dk_fpts_mean for {int(bad)} rows (should not happen)")
                # For near-certain players, conditional and unconditional should match closely.
                near_one = (p >= 0.99) & f_cond.notna() & f_uncond.notna()
                if int(near_one.sum()) > 0:
                    max_gap = float((f_cond[near_one] - f_uncond[near_one]).abs().max())
                    metrics["fpts_uncond_gap_p>=0.99_max"] = max_gap
                    if max_gap > 1e-3:
                        warnings.append(f"dk_fpts_mean_uncond deviates for p>=0.99 (max_gap={max_gap:.4f})")

            # Ownership checks (if present).
            if "pred_own_pct" in df.columns:
                own = pd.to_numeric(df["pred_own_pct"], errors="coerce")
                if own.notna().any():
                    if (own < -1e-6).any() or (own > 100.0 + 1e-6).any():
                        errors.append("pred_own_pct: outside [0,100] range")
                    own_sum = float(own.fillna(0.0).sum())
                    metrics["pred_own_pct_sum"] = own_sum
                    if own_sum < 650.0 or own_sum > 950.0:
                        warnings.append(f"pred_own_pct sum {own_sum:.1f}% outside [650,950] (check calibration)")

    return ReadinessReport(
        game_date=game_date,
        as_of_ts=now.isoformat(),
        run_id=resolved_run_id,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )
