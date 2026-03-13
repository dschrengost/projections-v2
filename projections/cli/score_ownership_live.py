"""CLI for scoring ownership predictions on live slates."""

from __future__ import annotations

import argparse
import json
import os
import re
from functools import lru_cache
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from projections.names import normalize_player_name
from projections.ownership_v2 import (
    load_ownership_transformer_bundle,
    predict_ownership_transformer_slate,
)
from projections.ownership_v1.calibration import (
    PowerCalibrator,
    SoftmaxCalibrator,
    apply_calibration_with_mask,
)
from projections.ownership_v1.loader import load_ownership_bundle
from projections.ownership_v1.schemas import (
    fill_optional_columns,
    prepare_model_input,
    validate_raw_input,
)
from projections.ownership_v1.score import (
    compute_ownership_features,
    normalize_ownership_to_target_sum,
    predict_ownership,
)
from projections.paths import data_path

OwnershipModelFamily = Literal["ownership_v1", "ownership_v2"]


def _normalize_name(value: str | None) -> str:
    return normalize_player_name(value)


def _load_calibration_config() -> dict:
    """Load ownership calibration config from YAML."""
    config_path = Path(__file__).parent.parent.parent / "config" / "ownership_calibration.yaml"
    if not config_path.exists():
        return {"calibration": {"enabled": False}}
    
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def _apply_postprocessing(
    output: pd.DataFrame,
    salaries: pd.DataFrame,
    *,
    config: dict,
    data_root: Path,
) -> pd.DataFrame:
    """Apply playable filter, optional calibration, and normalization.

    Important production invariant: the final `pred_own_pct` should always be
    normalized to the configured target sum (unless normalization is disabled),
    even if calibration is enabled but missing/invalid.
    """

    result = output.copy()

    # Apply playable filter: zero out unplayable players.
    play_cfg = config.get("playable_filter", {})
    unplayable_mask = pd.Series(False, index=result.index)
    if play_cfg.get("enabled", False):
        min_fpts = float(play_cfg.get("min_proj_fpts", 8.0))
        if play_cfg.get("slate_aware", False):
            baseline = int(play_cfg.get("baseline_slate_size", 80))
            scale = float(play_cfg.get("scale_per_player", 0.05))
            min_fpts = min_fpts + max(0.0, (len(result) - baseline) * scale)

        unplayable_mask = result["proj_fpts"].astype(float) < min_fpts
        if unplayable_mask.any():
            result.loc[unplayable_mask, "pred_own_pct"] = 0.0

    cal_cfg = config.get("calibration", {})
    norm_cfg = config.get("normalization", {})

    slots = float(cal_cfg.get("R", 8.0))
    target_sum_pct = float(norm_cfg.get("target_sum_pct", slots * 100.0))
    cap_pct = float(norm_cfg.get("cap_pct", 100.0))

    # Apply calibration if enabled.
    if cal_cfg.get("enabled", False):
        method = str(cal_cfg.get("method", "softmax")).lower()
        print(
            f"[ownership] Applying calibration method={method} (sum before: {result['pred_own_pct'].sum():.1f}%)"
        )

        calibrator_path = data_root / cal_cfg.get("calibrator_path", "artifacts/ownership_v1/calibrator.json")
        if not calibrator_path.exists():
            print(f"[ownership] Calibrator not found at {calibrator_path}, falling back to scaling")
        else:
            try:
                calibrator: SoftmaxCalibrator | PowerCalibrator
                if method == "softmax":
                    calibrator = SoftmaxCalibrator.load(calibrator_path)
                elif method == "power":
                    calibrator = PowerCalibrator.load(calibrator_path)
                else:
                    raise ValueError(f"Unknown calibration method: {method}")

                # Build structural zero mask.
                # True = include in calibration, False = structural zero (set to 0)
                struct_cfg = cal_cfg.get("structural_zeros", {})
                mask = pd.Series(True, index=result.index)

                # Exclude OUT players (already filtered earlier, but double-check).
                if struct_cfg.get("exclude_out", True) and "_injury_status" in salaries.columns:
                    injury_status = (
                        salaries["_injury_status"]
                        .astype("string")
                        .fillna("")
                        .str.upper()
                    )
                    mask &= injury_status.ne("OUT")

                # Exclude zero-minute players - check if we have this info.
                if struct_cfg.get("exclude_zero_minutes", True) and "proj_minutes" in result.columns:
                    proj_minutes = pd.to_numeric(result["proj_minutes"], errors="coerce").fillna(0.0)
                    mask &= proj_minutes.gt(0.0)

                # Exclude zero prediction (optional).
                if struct_cfg.get("exclude_zero_prediction", False):
                    pred_vals = pd.to_numeric(result["pred_own_pct"], errors="coerce").fillna(0.0)
                    mask &= pred_vals.gt(0.0)

                # Exclude unplayable players from calibration allocation.
                mask &= ~unplayable_mask
                mask = mask.fillna(False).astype(bool)

                scores = result["pred_own_pct_raw"].values
                if method == "softmax":
                    calibrated = apply_calibration_with_mask(scores, mask.values, calibrator.params)
                    result["pred_own_pct"] = calibrated * 100.0  # Convert to percent
                elif method == "power":
                    calibrated = calibrator.apply(scores, mask=mask.values)
                    result["pred_own_pct"] = calibrated * 100.0  # Convert to percent
                else:  # pragma: no cover
                    raise ValueError(f"Unknown calibration method: {method}")

                log_cfg = config.get("logging", {})
                if log_cfg.get("log_metrics", True):
                    n_zeros = (~mask).sum()
                    print(
                        f"[ownership] Calibration: {n_zeros} structural zeros, sum after: {result['pred_own_pct'].sum():.1f}%"
                    )
            except Exception as e:
                print(f"[ownership] Calibration failed: {e}, falling back to scaling")

    # Always normalize unless explicitly disabled.
    if norm_cfg.get("enabled", True):
        result["pred_own_pct"] = normalize_ownership_to_target_sum(
            result["pred_own_pct"],
            target_sum_pct=target_sum_pct,
            cap_pct=cap_pct,
        )

    return result


def _load_schedule_with_times(game_date: date, data_root: Path) -> pd.DataFrame:
    """Load schedule with game times for lock detection."""
    month = game_date.month
    year = game_date.year if game_date.month >= 10 else game_date.year  # Season year
    
    schedule_path = data_root / "silver" / "schedule" / f"season={year}" / f"month={month:02d}" / "schedule.parquet"
    if not schedule_path.exists():
        return pd.DataFrame()
    
    df = pd.read_parquet(schedule_path)
    df = df.copy()
    df["_game_date"] = pd.to_datetime(df.get("game_date"), errors="coerce").dt.date
    df = df[df["_game_date"] == game_date].copy()
    df = df.drop(columns=["_game_date"], errors="ignore")
    
    # Parse tip_ts (UTC) to datetime for consistent gating / leak safety.
    if "tip_ts" in df.columns:
        df["game_start"] = pd.to_datetime(df["tip_ts"], utc=True, errors="coerce")
    
    return df


def _load_dk_draft_group_lock_ts(
    *,
    draft_group_id: str,
    data_root: Path,
) -> datetime | None:
    """Best-effort first-tip timestamp for a DK draft group from bronze draftables.

    This is more reliable for backtests than the schedule parquet (which may only
    contain recently scraped dates).
    """

    draftables_path = data_root / "bronze" / "dk" / "draftables" / f"draftables_raw_{draft_group_id}.json"
    if not draftables_path.exists():
        return None

    try:
        import json

        payload = json.loads(draftables_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    comps = payload.get("competitions") if isinstance(payload, dict) else None
    if not isinstance(comps, list) or not comps:
        return None

    starts = [c.get("startTime") for c in comps if isinstance(c, dict) and c.get("startTime")]
    if not starts:
        return None

    parsed = pd.to_datetime(pd.Series(starts), utc=True, errors="coerce").dropna()
    if parsed.empty:
        return None
    lock_ts = parsed.min()
    return lock_ts.to_pydatetime() if hasattr(lock_ts, "to_pydatetime") else lock_ts


def _get_locked_teams(schedule: pd.DataFrame, current_time: datetime) -> set:
    """Get set of team tricodes whose games have already started."""
    if schedule.empty or 'game_start' not in schedule.columns:
        return set()
    
    locked_teams = set()
    for _, row in schedule.iterrows():
        game_start = row['game_start']
        if pd.isna(game_start):
            continue
        if current_time >= game_start:
            locked_teams.add(row['home_team_tricode'])
            locked_teams.add(row['away_team_tricode'])
    
    return locked_teams


def _sanitize_lock_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return token.strip("._") or "default"


def _lock_cache_paths(
    *,
    game_date: date,
    draft_group_id: str,
    data_root: Path,
    model_family: OwnershipModelFamily,
    model_run: str,
) -> tuple[Path, Path]:
    out_dir = data_root / "silver" / "ownership_predictions" / str(game_date)
    model_run_token = _sanitize_lock_token(model_run)
    scoped = out_dir / f"{draft_group_id}_locked__{model_family}__{model_run_token}.parquet"
    legacy = out_dir / f"{draft_group_id}_locked.parquet"
    return scoped, legacy


def _load_locked_predictions(
    game_date: date,
    draft_group_id: str,
    data_root: Path,
    *,
    model_family: OwnershipModelFamily,
    model_run: str,
) -> Optional[pd.DataFrame]:
    """Load previously locked ownership predictions for a specific slate."""

    scoped_path, legacy_path = _lock_cache_paths(
        game_date=game_date,
        draft_group_id=draft_group_id,
        data_root=data_root,
        model_family=model_family,
        model_run=model_run,
    )
    if scoped_path.exists():
        return pd.read_parquet(scoped_path)

    # Backwards compatibility for old v1 lock cache files.
    if model_family != "ownership_v1" or not legacy_path.exists():
        return None

    legacy = pd.read_parquet(legacy_path)
    if legacy.empty:
        return None
    legacy_model_run = str(legacy.get("model_run", pd.Series([""])).iloc[0]).strip()
    legacy_model_family = str(legacy.get("model_family", pd.Series(["ownership_v1"])).iloc[0]).strip()
    if legacy_model_run != str(model_run).strip():
        return None
    if legacy_model_family not in {"", "ownership_v1"}:
        return None
    return legacy


def _save_locked_predictions(
    df: pd.DataFrame,
    game_date: date,
    draft_group_id: str,
    data_root: Path,
    *,
    model_family: OwnershipModelFamily,
    model_run: str,
    overwrite: bool = False,
) -> None:
    """Save predictions to locked file for a specific slate."""

    out_dir = data_root / "silver" / "ownership_predictions" / str(game_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    scoped_path, legacy_path = _lock_cache_paths(
        game_date=game_date,
        draft_group_id=draft_group_id,
        data_root=data_root,
        model_family=model_family,
        model_run=model_run,
    )
    if overwrite or not scoped_path.exists():
        df.to_parquet(scoped_path, index=False)
        print(f"[ownership] Saved locked predictions for slate {draft_group_id}: {len(df)} players")

    # Keep legacy v1 cache writes for backward compatibility with older tooling.
    if model_family == "ownership_v1" and (overwrite or not legacy_path.exists()):
        df.to_parquet(legacy_path, index=False)


PRODUCTION_MODEL_RUN = "dk_only_v6_logit_chalk5_cleanbase_seed1337"
PRODUCTION_MODEL_RUN_V2 = "ownership_xfmr_v1_12ep_big"


def _load_all_slates(
    game_date: date,
    data_root: Path,
) -> dict[str, pd.DataFrame]:
    """
    Load all DK slates (draft groups) for a date.
    
    Returns dict of {draft_group_id: DataFrame with normalized columns}.
    """
    base = data_root / "gold" / "dk_salaries" / "site=dk" / f"game_date={game_date}"
    
    if not base.exists():
        print(f"[ownership] No salary data at {base}")
        return {}
    
    draft_group_dirs = sorted(base.glob("draft_group_id=*"))
    if not draft_group_dirs:
        print(f"[ownership] No draft groups found for {game_date}")
        return {}
    
    slates = {}
    for dg_dir in draft_group_dirs:
        parquet_path = dg_dir / "salaries.parquet"
        if not parquet_path.exists():
            continue
        
        df = pd.read_parquet(parquet_path)
        dg_id = dg_dir.name.split("=")[1]
        df["draft_group_id"] = dg_id
        
        # Normalize column names for ownership model
        if "display_name" in df.columns and "player_name" not in df.columns:
            df["player_name"] = df["display_name"]
        if "positions" in df.columns and "pos" not in df.columns:
            df["pos"] = df["positions"].apply(lambda x: "/".join(x) if isinstance(x, list) else str(x))
        if "team_abbrev" in df.columns and "team" not in df.columns:
            df["team"] = df["team_abbrev"]
        if "dk_player_id" in df.columns and "player_id" not in df.columns:
            df["player_id"] = df["dk_player_id"]
        if "salary" in df.columns:
            df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)
        
        slates[dg_id] = df
    
    print(f"[ownership] Loaded {len(slates)} slates for {game_date}")
    for dg_id, df in slates.items():
        teams = df["team"].unique() if "team" in df.columns else []
        print(f"  - {dg_id}: {len(df)} players, teams: {list(teams)[:6]}{'...' if len(teams) > 6 else ''}")
    
    return slates


def _get_slate_first_game_time(
    slate_teams: set[str],
    schedule: pd.DataFrame,
) -> Optional[datetime]:
    """Get earliest game start time for teams in this slate."""
    if schedule.empty or "game_start" not in schedule.columns:
        return None
    
    earliest = None
    for _, row in schedule.iterrows():
        home = row.get("home_team_tricode")
        away = row.get("away_team_tricode")
        if home in slate_teams or away in slate_teams:
            game_start = row["game_start"]
            if pd.notna(game_start):
                if earliest is None or game_start < earliest:
                    earliest = game_start
    
    return earliest


def _is_slate_locked(
    slate_teams: set[str],
    schedule: pd.DataFrame,
    current_time: datetime,
) -> bool:
    """Check if a slate's first game has already started."""
    first_game = _get_slate_first_game_time(slate_teams, schedule)
    if first_game is None:
        return False
    return current_time >= first_game


def _load_fpts_predictions(
    game_date: date,
    run_id: str,
    data_root: Path,
    *,
    cutoff_ts: datetime | None = None,
) -> Optional[pd.DataFrame]:
    """
    Load FPTS predictions from live GTV2 projections outputs.

    Source priority:
    1) artifacts/gtv2_worlds/.../projections.parquet (authoritative pre-finalize path)
    2) artifacts/projections/.../projections.parquet (unified projections)
    """
    def _coerce_cutoff_ts_utc(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        cutoff_dt = value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
        if not isinstance(cutoff_dt, datetime):
            return None
        cutoff_dt = cutoff_dt if cutoff_dt.tzinfo is not None else cutoff_dt.replace(tzinfo=UTC)
        return cutoff_dt.astimezone(UTC)

    def _read_pointer_run_id(base_dir: Path) -> str | None:
        for pointer in (base_dir / "LATEST" / "current.json", base_dir / "latest_run.json"):
            if not pointer.exists():
                continue
            try:
                payload = json.loads(pointer.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                run_id_value = str(payload.get("run_id", "")).strip()
                if run_id_value:
                    return run_id_value
        return None

    def _resolve_projection_run_dir(
        base_dir: Path,
        desired_run_id: str | None,
        *,
        cutoff_ts_utc: datetime | None,
    ) -> Path | None:
        # Prefer explicit run id from CLI when available.
        if desired_run_id:
            candidate = base_dir / f"run={desired_run_id}"
            if candidate.exists():
                return candidate
            return None

        # Backtest safety: if we have a cutoff timestamp (e.g., slate lock),
        # prefer the latest run at or before that cutoff.
        if desired_run_id is None and cutoff_ts_utc is not None and base_dir.is_dir():
            best_dt: datetime | None = None
            best_dir: Path | None = None
            for p in base_dir.iterdir():
                if not p.is_dir() or not p.name.startswith("run="):
                    continue
                rid = p.name.split("run=", 1)[1]
                try:
                    dt = datetime.strptime(rid, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
                except ValueError:
                    continue
                if dt <= cutoff_ts_utc and (best_dt is None or dt > best_dt):
                    best_dt = dt
                    best_dir = p
            if best_dir is not None:
                return best_dir

        if desired_run_id is None:
            promoted_run_id = _read_pointer_run_id(base_dir)
            if promoted_run_id:
                candidate = base_dir / f"run={promoted_run_id}"
                if candidate.exists():
                    return candidate

        if desired_run_id is None:
            run_dirs = sorted(
                [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run=")],
                reverse=True,
            )
            if run_dirs:
                return run_dirs[0]
        if desired_run_id is None and (base_dir / "projections.parquet").exists():
            return base_dir
        return None

    def _iter_projection_bases(day: date) -> list[tuple[str, Path]]:
        day_iso = day.isoformat()
        gtv2_root = data_root / "artifacts" / "gtv2_worlds"
        unified_root = data_root / "artifacts" / "projections"
        return [
            ("gtv2_worlds", gtv2_root / f"game_date={day_iso}"),
            ("gtv2_worlds", gtv2_root / f"date={day_iso}"),
            ("gtv2_worlds", gtv2_root / day_iso),
            ("unified_projections", unified_root / day_iso),
            ("unified_projections", unified_root / f"game_date={day_iso}"),
            ("unified_projections", unified_root / f"date={day_iso}"),
        ]

    def _load_live_projections(
        day: date,
        desired_run_id: str | None,
    ) -> tuple[pd.DataFrame | None, str | None, Path | None]:
        cutoff_ts_utc = _coerce_cutoff_ts_utc(cutoff_ts)
        for source_label, base in _iter_projection_bases(day):
            if not base.exists():
                continue
            if base.is_file() and base.suffix == ".parquet":
                try:
                    return pd.read_parquet(base), source_label, base
                except Exception:
                    continue

            run_dir = None
            if desired_run_id and base.is_dir():
                explicit = base / f"run={desired_run_id}"
                if explicit.exists():
                    run_dir = explicit
            if run_dir is None:
                if desired_run_id is None:
                    direct = base / "projections.parquet"
                    if direct.exists():
                        try:
                            return pd.read_parquet(direct), source_label, direct
                        except Exception:
                            pass

            if run_dir is None:
                run_dir = (
                    _resolve_projection_run_dir(base, desired_run_id, cutoff_ts_utc=cutoff_ts_utc)
                    if base.is_dir()
                    else None
                )
                if run_dir is None:
                    continue
            candidates: list[Path] = []
            if run_dir.is_file() and run_dir.suffix == ".parquet":
                candidates.append(run_dir)
            else:
                candidates.append(run_dir / "projections.parquet")
            for candidate in candidates:
                if not candidate.exists():
                    continue
                try:
                    return pd.read_parquet(candidate), source_label, candidate
                except Exception:
                    continue
        return None, None, None

    df, source, source_path = _load_live_projections(game_date, run_id)
    if df is None or df.empty:
        print(
            "[ownership] No live projections found under "
            f"{data_root / 'artifacts' / 'gtv2_worlds'} or {data_root / 'artifacts' / 'projections'} "
            f"for {game_date} (run_id={run_id})"
        )
        return None

    # Use dk_fpts_mean from live GTV2/unified projections.
    if "dk_fpts_mean" not in df.columns:
        print(
            "[ownership] live projections missing dk_fpts_mean column "
            f"(source={source}, path={source_path})"
        )
        return None

    df = df.copy()
    df["dk_fpts_mean"] = pd.to_numeric(df["dk_fpts_mean"], errors="coerce")
    valid_fpts = df["dk_fpts_mean"].notna() & df["dk_fpts_mean"].between(0.0, 300.0)
    dropped = int((~valid_fpts).sum())
    if dropped:
        print(
            "[ownership] Dropping rows with invalid dk_fpts_mean "
            f"(source={source}, dropped={dropped})"
        )
        df = df.loc[valid_fpts].copy()
    if df.empty:
        print("[ownership] No valid dk_fpts_mean rows in live projections after filtering")
        return None

    print(
        f"[ownership] Loaded {len(df)} players from live projections "
        f"(source={source}, path={source_path})"
    )

    # Return core + optional distributional features when available.
    # Note: projections use NBA player_id; we map to DK names later.
    optional_cols = [
        "minutes_mean",
        "dk_fpts_p90",
        "dk_fpts_p50",
        "minutes_sim_mean",
        "sim_p_active",
        "play_prob_eff",
    ]
    cols = ["player_id", "dk_fpts_mean", *[c for c in optional_cols if c in df.columns]]
    return df[cols].rename(columns={"dk_fpts_mean": "pred_fpts"})


def _ensure_v2_base_feature_defaults(salaries: pd.DataFrame) -> pd.DataFrame:
    """Ensure transformer baseline feature columns exist with safe defaults."""

    working = salaries.copy()
    proj_fpts = pd.to_numeric(working.get("proj_fpts"), errors="coerce").fillna(0.0)

    def _numeric_series_or_na(column: str) -> pd.Series:
        if column in working.columns:
            return pd.to_numeric(working[column], errors="coerce")
        return pd.Series(float("nan"), index=working.index, dtype=float)

    if "proj_minutes" in working.columns:
        minutes_proxy = pd.to_numeric(working["proj_minutes"], errors="coerce")
    else:
        # Backfill when minutes aren't present in joined live frame.
        minutes_proxy = proj_fpts / 1.1
    minutes_proxy = pd.to_numeric(minutes_proxy, errors="coerce").fillna(proj_fpts / 1.1)
    minutes_proxy = minutes_proxy.clip(lower=0.0)

    working["minutes_mean"] = (
        _numeric_series_or_na("minutes_mean")
        .fillna(minutes_proxy)
        .clip(lower=0.0)
    )
    working["minutes_sim_mean"] = (
        _numeric_series_or_na("minutes_sim_mean")
        .fillna(working["minutes_mean"])
        .clip(lower=0.0)
    )
    working["dk_fpts_p50"] = _numeric_series_or_na("dk_fpts_p50").fillna(proj_fpts)
    working["dk_fpts_p90"] = (
        _numeric_series_or_na("dk_fpts_p90")
        .fillna(working["dk_fpts_p50"] * 1.25)
        .clip(lower=working["dk_fpts_p50"])
    )
    working["sim_p_active"] = (
        _numeric_series_or_na("sim_p_active")
        .fillna(1.0)
        .clip(0.0, 1.0)
    )
    working["play_prob_eff"] = (
        _numeric_series_or_na("play_prob_eff")
        .fillna(working["sim_p_active"])
        .clip(0.0, 1.0)
    )
    return working


def _load_injuries(
    game_date: date,
    data_root: Path,
    *,
    cutoff_ts: datetime | None = None,
) -> pd.DataFrame:
    """Load injury data for the date."""
    season = game_date.year if game_date.month >= 10 else game_date.year
    month = game_date.month
    
    inj_path = (
        data_root / "silver" / "injuries_snapshot"
        / f"season={season}" / f"month={month:02d}" / "injuries_snapshot.parquet"
    )
    
    if inj_path.exists():
        df = pd.read_parquet(inj_path)
        # injuries_snapshot is keyed by report_date (not game_date).
        if "report_date" in df.columns:
            df["_report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date
            df = df[df["_report_date"] == game_date].copy()
            df = df.drop(columns=["_report_date"], errors="ignore")
        if cutoff_ts is not None and "as_of_ts" in df.columns and not df.empty:
            as_of = pd.to_datetime(df["as_of_ts"], utc=True, errors="coerce")
            df = df[as_of.notna() & (as_of <= cutoff_ts)].copy()
        if not df.empty:
            return df
    
    return pd.DataFrame()


@lru_cache(maxsize=32)
def _historical_ownership_feature_maps(
    *, game_date_iso: str, data_root_str: str, window: int = 10
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    float,
    float,
    float,
    float,
]:
    """Compute historical ownership features per player before game_date.

    Returns:
        (avg10_map, median_map, std_map, chalk_rate_map,
         overall_avg10, overall_median, overall_std, overall_chalk_rate)
    """

    data_root = Path(data_root_str)
    path = (
        data_root
        / "bronze"
        / "dk_contests"
        / "ownership_by_slate"
        / "all_ownership.parquet"
    )
    if not path.exists():
        return {}, {}, {}, {}, 0.0, 0.0, 0.0, 0.0

    df = pd.read_parquet(path)
    if df.empty or "Player" not in df.columns or "own_pct" not in df.columns:
        return {}, {}, {}, {}, 0.0, 0.0, 0.0, 0.0

    df = df.copy()
    df["game_date"] = pd.to_datetime(df.get("game_date"), errors="coerce").dt.date
    cutoff = date.fromisoformat(game_date_iso)
    df = df[df["game_date"].notna() & (df["game_date"] < cutoff)].copy()
    if df.empty:
        return {}, {}, {}, {}, 0.0, 0.0, 0.0, 0.0

    df["_name_norm"] = df["Player"].astype(str).apply(_normalize_name)
    df["_own"] = pd.to_numeric(df["own_pct"], errors="coerce")
    df = df[df["_name_norm"].ne("") & df["_own"].notna()].copy()
    if df.empty:
        return {}, {}, {}, {}, 0.0, 0.0, 0.0, 0.0

    # Ensure stable ordering within date.
    sort_cols = ["game_date"]
    if "slate_id" in df.columns:
        sort_cols.append("slate_id")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    def _tail_mean(g: pd.DataFrame) -> float:
        return float(g.tail(window)["_own"].mean())

    by_player = df.groupby("_name_norm", sort=False)

    avg10 = by_player.apply(_tail_mean, include_groups=False)
    median = by_player["_own"].median()
    std = by_player["_own"].std()
    chalk_rate = by_player.apply(lambda g: float((g["_own"] > 30.0).mean()), include_groups=False)

    overall_avg10 = float(pd.Series(avg10.values).mean()) if len(avg10) else 0.0
    overall_median = float(df["_own"].median())
    overall_std = float(df["_own"].std()) if df["_own"].std() == df["_own"].std() else 0.0
    overall_chalk_rate = float((df["_own"] > 30.0).mean())

    return (
        {str(k): float(v) for k, v in avg10.to_dict().items()},
        {str(k): float(v) for k, v in median.to_dict().items()},
        {str(k): float(v) for k, v in std.fillna(0.0).to_dict().items()},
        {str(k): float(v) for k, v in chalk_rate.to_dict().items()},
        overall_avg10,
        overall_median,
        overall_std,
        overall_chalk_rate,
    )


def _attach_live_ownership_enrichment(
    salaries: pd.DataFrame,
    *,
    game_date: date,
    data_root: Path,
    minutes_team_map: dict[int, str] | None,
    nba_player_ids: pd.Series | None,
    injuries_cutoff_ts: datetime | None,
) -> pd.DataFrame:
    """Attach player_own_avg_10, player_is_questionable, team_outs_count."""

    working = salaries.copy()
    working["_name_norm"] = working["player_name"].astype(str).apply(_normalize_name)

    # Historical ownership features.
    (
        avg10_map,
        median_map,
        std_map,
        chalk_rate_map,
        overall_avg10,
        overall_median,
        overall_std,
        overall_chalk_rate,
    ) = _historical_ownership_feature_maps(
        game_date_iso=game_date.isoformat(),
        data_root_str=str(data_root),
        window=10,
    )
    if avg10_map:
        working["player_own_avg_10"] = working["_name_norm"].map(avg10_map).fillna(overall_avg10).astype(float)
        working["player_own_median"] = working["_name_norm"].map(median_map).fillna(overall_median).astype(float)
        working["player_own_variance"] = working["_name_norm"].map(std_map).fillna(overall_std).astype(float)
        working["player_chalk_rate"] = working["_name_norm"].map(chalk_rate_map).fillna(overall_chalk_rate).astype(float)
    else:
        working["player_own_avg_10"] = 0.0
        working["player_own_median"] = 0.0
        working["player_own_variance"] = 0.0
        working["player_chalk_rate"] = 0.0

    # Injury enrichment.
    inj = _load_injuries(game_date, data_root, cutoff_ts=injuries_cutoff_ts)
    if inj.empty:
        working["player_is_questionable"] = 0
        working["team_outs_count"] = 0
        working = working.drop(columns=["_name_norm"], errors="ignore")
        return working

    inj = inj.copy()
    # Keep latest snapshot per player when multiple as_of_ts rows exist.
    if "as_of_ts" in inj.columns:
        inj["_as_of_ts"] = pd.to_datetime(inj["as_of_ts"], utc=True, errors="coerce")
        inj = inj.sort_values("_as_of_ts").dropna(subset=["player_id"]).drop_duplicates("player_id", keep="last")
        inj = inj.drop(columns=["_as_of_ts"], errors="ignore")

    status = inj.get("status")
    status_raw = inj.get("status_raw")
    status_u = status.astype(str).str.upper() if status is not None else pd.Series("", index=inj.index)
    status_raw_u = status_raw.astype(str).str.upper() if status_raw is not None else pd.Series("", index=inj.index)

    is_out = status_u.isin(["OUT", "D", "DOUBTFUL"]) | status_raw_u.isin(["OUT", "DOUBTFUL"])
    is_q = status_u.isin(["Q", "PROB"]) | status_raw_u.isin(["QUESTIONABLE", "PROBABLE"])
    inj["_is_q"] = is_q.astype(int)

    # Team outs by tricode (DK salary table uses tricodes).
    team_outs_count: dict[str, int] = {}
    if "team_id" in inj.columns and minutes_team_map:
        team_ids = pd.to_numeric(inj["team_id"], errors="coerce")
        inj["_team_tricode"] = team_ids.map(lambda v: minutes_team_map.get(int(v)) if pd.notna(v) else None)
        outs_by_team = inj.loc[is_out & inj["_team_tricode"].notna()].groupby("_team_tricode")["player_id"].count()
        team_outs_count = {str(k): int(v) for k, v in outs_by_team.to_dict().items()}
    working["team_outs_count"] = working.get("team", pd.Series("", index=working.index)).map(team_outs_count).fillna(0).astype(int)

    # Player questionable by NBA player_id when available, else by name_norm.
    q_by_pid: dict[int, int] = {}
    if "player_id" in inj.columns:
        pid_series = pd.to_numeric(inj["player_id"], errors="coerce").astype("Int64")
        q_by_pid = {int(pid): int(flag) for pid, flag in zip(pid_series.dropna().astype(int), is_q.astype(int))}

    if nba_player_ids is not None:
        pid_norm = pd.to_numeric(nba_player_ids, errors="coerce").astype("Int64")
        working["player_is_questionable"] = pid_norm.map(lambda v: q_by_pid.get(int(v), 0) if pd.notna(v) else 0).astype(int)
    else:
        inj["_name_norm"] = inj["player_name"].astype(str).apply(_normalize_name)
        q_by_name = inj.loc[inj["_name_norm"].ne("")].groupby("_name_norm")["_is_q"].max()
        working["player_is_questionable"] = working["_name_norm"].map(q_by_name.to_dict()).fillna(0).astype(int)

    working = working.drop(columns=["_name_norm"], errors="ignore")
    return working


def _required_gtv2_feature_columns(feature_columns: list[str]) -> list[str]:
    return sorted(
        [
            c
            for c in feature_columns
            if c.startswith("gtv2_")
        ]
    )


def _read_gtv2_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _load_live_gtv2_feature_frame(
    *,
    game_date: date,
    run_id: str,
    data_root: Path,
    gtv2_features_path: Path | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    """Best-effort load of live GTV2-derived features for ownership enrichment."""

    candidates: list[Path] = []
    if gtv2_features_path is not None:
        user_path = Path(gtv2_features_path).expanduser().resolve()
        if user_path.is_dir():
            candidates.extend([user_path / "scores.parquet", user_path / "features.parquet"])
        else:
            candidates.append(user_path)

    base = data_root / "artifacts" / "scores_gtv2" / f"game_date={game_date.isoformat()}"
    if base.exists():
        explicit = base / f"run={run_id}" / "scores.parquet"
        candidates.append(explicit)

        latest_pointer = base / "latest_run.json"
        if latest_pointer.exists():
            try:
                latest_payload = json.loads(latest_pointer.read_text(encoding="utf-8"))
                latest_run = str(latest_payload.get("run_id", "")).strip()
                if latest_run:
                    candidates.append(base / f"run={latest_run}" / "scores.parquet")
            except json.JSONDecodeError:
                pass

    for path in candidates:
        if not path.exists():
            continue
        try:
            frame = _read_gtv2_frame(path)
            if frame.empty:
                continue
            work = frame.copy()
            rename_map = {
                "minutes_deterministic": "gtv2_minutes_deterministic",
                "active_logit": "gtv2_active_logit",
                "active_prob_proxy": "gtv2_active_prob_proxy",
            }
            work = work.rename(columns=rename_map)
            if "game_date" in work.columns:
                work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            if "player_id" in work.columns:
                work["player_id"] = pd.to_numeric(work["player_id"], errors="coerce").astype("Int64")
            if "player_name" in work.columns:
                work["_name_norm"] = work["player_name"].astype(str).apply(_normalize_name)
            return work, str(path)
        except Exception as exc:
            print(f"[ownership] Failed to load GTV2 live features from {path}: {exc}")

    return None, None


def _attach_live_gtv2_enrichment(
    *,
    feature_frame: pd.DataFrame,
    salaries: pd.DataFrame,
    required_gtv2_cols: list[str],
    game_date: date,
    run_id: str,
    data_root: Path,
    gtv2_features_path: Path | None,
) -> pd.DataFrame:
    """Attach optional GTV2 features to ownership feature frame with safe fallback."""

    if not required_gtv2_cols:
        return feature_frame

    result = feature_frame.copy()
    gtv2_frame, source = _load_live_gtv2_feature_frame(
        game_date=game_date,
        run_id=run_id,
        data_root=data_root,
        gtv2_features_path=gtv2_features_path,
    )

    if gtv2_frame is None or gtv2_frame.empty:
        for col in required_gtv2_cols:
            if col not in result.columns:
                result[col] = 0.0
        print(
            "[ownership] GTV2 live enrichment unavailable; zero-filled "
            f"{len(required_gtv2_cols)} gtv2_* feature columns"
        )
        return result

    available_gtv2_cols = [c for c in required_gtv2_cols if c in gtv2_frame.columns]
    if not available_gtv2_cols:
        for col in required_gtv2_cols:
            if col not in result.columns:
                result[col] = 0.0
        print(
            "[ownership] GTV2 source loaded but no required columns present; zero-filled "
            f"{len(required_gtv2_cols)} gtv2_* feature columns"
        )
        return result

    matched_any = pd.Series(False, index=result.index)

    if "player_id" in gtv2_frame.columns and "nba_player_id" in salaries.columns:
        id_map_df = (
            gtv2_frame.loc[gtv2_frame["player_id"].notna(), ["player_id", *available_gtv2_cols]]
            .drop_duplicates(subset=["player_id"], keep="last")
            .set_index("player_id")
        )
        nba_ids = pd.to_numeric(salaries["nba_player_id"], errors="coerce").astype("Int64")
        for col in available_gtv2_cols:
            mapped = nba_ids.map(id_map_df[col]) if col in id_map_df.columns else pd.Series(index=result.index, dtype=float)
            result[col] = mapped
            matched_any |= mapped.notna()

    if "_name_norm" in gtv2_frame.columns:
        name_map_df = (
            gtv2_frame.loc[gtv2_frame["_name_norm"].ne(""), ["_name_norm", *available_gtv2_cols]]
            .drop_duplicates(subset=["_name_norm"], keep="last")
            .set_index("_name_norm")
        )
        name_norm = salaries["player_name"].astype(str).apply(_normalize_name)
        for col in available_gtv2_cols:
            mapped = name_norm.map(name_map_df[col]) if col in name_map_df.columns else pd.Series(index=result.index, dtype=float)
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")
                result[col] = result[col].where(result[col].notna(), mapped)
            else:
                result[col] = mapped
            matched_any |= mapped.notna()

    for col in required_gtv2_cols:
        if col in result.columns:
            series = result[col]
        else:
            series = pd.Series(0.0, index=result.index)
        result[col] = pd.to_numeric(series, errors="coerce").fillna(0.0)

    coverage = float(matched_any.mean()) if len(matched_any) else 0.0
    print(
        "[ownership] Attached GTV2 live enrichment "
        f"(source={source}, available_cols={len(available_gtv2_cols)}, "
        f"required_cols={len(required_gtv2_cols)}, row_coverage={coverage:.3f})"
    )
    return result


def score_ownership(
    slate_df: pd.DataFrame,
    draft_group_id: str,
    game_date: date,
    run_id: str,
    data_root: Path,
    model_run: str = PRODUCTION_MODEL_RUN,
    model_family: OwnershipModelFamily = "ownership_v1",
    injuries_cutoff_ts: datetime | None = None,
    gtv2_features_path: Path | None = None,
) -> Optional[pd.DataFrame]:
    """
    Score ownership predictions for a single slate.
    
    Args:
        slate_df: DataFrame with salary data for this slate (already normalized)
        draft_group_id: DraftKings draft group ID
        game_date: Game date
        run_id: Run identifier
        data_root: Data root path
        model_run: Ownership model run ID
        model_family: ownership model family ("ownership_v1" or "ownership_v2")
    
    Returns DataFrame with player ownership predictions or None if data unavailable.
    """
    bundle_v1 = None
    bundle_v2 = None
    if model_family == "ownership_v2":
        try:
            bundle_v2 = load_ownership_transformer_bundle(model_run, base_artifacts_root=data_root)
        except FileNotFoundError as e:
            print(f"[ownership] ownership_v2 model not found: {e}")
            return None
        except Exception as e:
            print(f"[ownership] ownership_v2 model failed to load: {e}")
            return None
    else:
        try:
            bundle_v1 = load_ownership_bundle(model_run, base_artifacts_root=data_root)
        except FileNotFoundError as e:
            print(f"[ownership] Model not found: {e}")
            return None
    
    # Use provided slate data
    salaries = slate_df.copy()
    if salaries.empty:
        print(f"[ownership] Empty slate data for {draft_group_id}")
        return None
    
    # Load FPTS predictions from live projections artifacts. For backtests, use
    # the slate cutoff time (usually first tip) to avoid post-lock runs.
    fpts = _load_fpts_predictions(game_date, run_id, data_root, cutoff_ts=injuries_cutoff_ts)
    
    if fpts is None or fpts.empty:
        print(f"[ownership] No FPTS predictions for {game_date}, using salary-based estimate")
        # Use salary as proxy for FPTS if no predictions available
        salaries["proj_fpts"] = salaries["salary"] / 200.0  # Rough conversion
        salaries = _attach_live_ownership_enrichment(
            salaries,
            game_date=game_date,
            data_root=data_root,
            minutes_team_map=None,
            nba_player_ids=None,
            injuries_cutoff_ts=injuries_cutoff_ts,
        )
    else:
        # Load minutes to get player_name -> NBA player_id mapping
        # This bridges DK's display_name to sim's player_id
        import json
        
        minutes_root = data_root / "artifacts" / "minutes_v1" / "daily" / str(game_date)
        latest_pointer = minutes_root / "latest_run.json"

        def _coerce_cutoff_ts() -> datetime | None:
            if injuries_cutoff_ts is None:
                return None
            cutoff_dt = (
                injuries_cutoff_ts.to_pydatetime()
                if hasattr(injuries_cutoff_ts, "to_pydatetime")
                else injuries_cutoff_ts
            )
            if not isinstance(cutoff_dt, datetime):
                return None
            cutoff_dt = cutoff_dt if cutoff_dt.tzinfo is not None else cutoff_dt.replace(tzinfo=UTC)
            return cutoff_dt.astimezone(UTC)

        cutoff_ts_utc = _coerce_cutoff_ts()

        player_id_map = None
        team_id_to_tricode: dict[int, str] | None = None
        status_map = None

        chosen_minutes_run: str | None = None
        # Prefer explicit run_id when minutes artifacts exist for it (production).
        explicit_minutes_path = minutes_root / f"run={run_id}" / "minutes.parquet"
        if explicit_minutes_path.exists():
            chosen_minutes_run = run_id
        elif cutoff_ts_utc is not None and minutes_root.exists():
            # Backtest safety: choose the latest minutes run at or before cutoff.
            best_dt: datetime | None = None
            for p in minutes_root.iterdir():
                if not p.is_dir() or not p.name.startswith("run="):
                    continue
                rid = p.name.split("run=", 1)[1]
                try:
                    dt = datetime.strptime(rid, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
                except ValueError:
                    continue
                if dt <= cutoff_ts_utc and (best_dt is None or dt > best_dt):
                    best_dt = dt
                    chosen_minutes_run = rid

        if chosen_minutes_run is None and latest_pointer.exists():
            try:
                with open(latest_pointer) as f:
                    chosen_minutes_run = json.load(f).get("run_id")
            except Exception as e:
                print(f"[ownership] Failed to load minutes for mapping: {e}")

        if chosen_minutes_run is not None:
            try:
                minutes_path = minutes_root / f"run={chosen_minutes_run}" / "minutes.parquet"
                if minutes_path.exists():
                    # Load player_id, player_name, and status (for OUT filtering)
                    cols_to_load = ["player_id", "player_name"]
                    minutes_df = pd.read_parquet(minutes_path)
                    # Add status column if it exists (for OUT filtering)
                    if "status" in minutes_df.columns:
                        cols_to_load.append("status")
                    if "team_id" in minutes_df.columns:
                        cols_to_load.append("team_id")
                    if "team_tricode" in minutes_df.columns:
                        cols_to_load.append("team_tricode")
                    minutes_df = minutes_df[[c for c in cols_to_load if c in minutes_df.columns]].copy()

                    # Create name -> player_id mapping using normalized names
                    # This handles European characters like Dončić -> doncic
                    minutes_df["_name_norm"] = minutes_df["player_name"].apply(_normalize_name)
                    player_id_map = minutes_df.drop_duplicates("_name_norm").set_index("_name_norm")["player_id"]

                    # Also create player_id -> status map for OUT filtering
                    if "status" in minutes_df.columns:
                        status_map = minutes_df.drop_duplicates("player_id").set_index("player_id")["status"]
                    if {"team_id", "team_tricode"}.issubset(minutes_df.columns):
                        team_id_to_tricode = (
                            minutes_df[["team_id", "team_tricode"]]
                            .dropna()
                            .drop_duplicates("team_id")
                            .set_index("team_id")["team_tricode"]
                            .to_dict()
                        )
                    print(f"[ownership] Loaded {len(player_id_map)} player mappings from minutes")
            except Exception as e:
                print(f"[ownership] Failed to load minutes for mapping: {e}")
        
        if player_id_map is not None:
            # Map DK display_name -> NBA player_id using normalized names
            salaries["_name_norm"] = salaries["player_name"].apply(_normalize_name)
            salaries["nba_player_id"] = salaries["_name_norm"].map(player_id_map)
            
            # Now join with sim FPTS on NBA player_id
            salaries = salaries.merge(
                fpts.rename(columns={"pred_fpts": "proj_fpts"}),
                left_on="nba_player_id",
                right_on="player_id",
                how="left",
                suffixes=("", "_sim")
            )
            salaries = salaries.drop(columns=["player_id_sim"], errors="ignore")
            
            # Add injury status from minutes predictions for OUT filtering
            if status_map is not None:
                # Map player_id back to nba_player_id we got from name matching
                salaries["_temp_pid"] = salaries["player_name"].apply(_normalize_name).map(player_id_map)
                salaries["_injury_status"] = salaries["_temp_pid"].map(status_map)
                salaries = salaries.drop(columns=["_temp_pid"], errors="ignore")
            
            matched = salaries["proj_fpts"].notna().sum()
            print(f"[ownership] Matched {matched}/{len(salaries)} players via name→id mapping")

            # Attach historical ownership + injury enrichment (uses nba_player_id when available).
            salaries = _attach_live_ownership_enrichment(
                salaries,
                game_date=game_date,
                data_root=data_root,
                minutes_team_map=team_id_to_tricode,
                nba_player_ids=salaries.get("nba_player_id"),
                injuries_cutoff_ts=injuries_cutoff_ts,
            )
            salaries = salaries.drop(columns=["_name_norm"], errors="ignore")
        else:
            # Fallback: use salary-based estimate
            print("[ownership] No player_id mapping available, using salary-based estimate")
            salaries["proj_fpts"] = salaries["salary"] / 200.0
            salaries = _attach_live_ownership_enrichment(
                salaries,
                game_date=game_date,
                data_root=data_root,
                minutes_team_map=None,
                nba_player_ids=None,
                injuries_cutoff_ts=injuries_cutoff_ts,
            )
    
    # Fill missing FPTS
    salaries["proj_fpts"] = salaries["proj_fpts"].fillna(salaries["salary"] / 200.0)
    if model_family == "ownership_v2":
        salaries = _ensure_v2_base_feature_defaults(salaries)
    
    # Keep OUT rows in the output contract, but force them to 0% ownership.
    forced_out_mask = pd.Series(False, index=salaries.index)
    if "_injury_status" in salaries.columns:
        forced_out_mask |= salaries["_injury_status"].astype(str).str.upper().eq("OUT")
    if "status" in salaries.columns:
        status_upper = salaries["status"].astype(str).str.upper()
        forced_out_mask |= status_upper.isin({"OUT", "O"})
        if "_injury_status" not in salaries.columns:
            salaries["_injury_status"] = status_upper
    out_count = int(forced_out_mask.sum())
    if out_count > 0:
        out_names = salaries.loc[forced_out_mask, "player_name"].astype(str).tolist()
        print(
            "[ownership] Detected OUT players; forcing 0 ownership for "
            f"{out_count} players: {out_names[:5]}{'...' if out_count > 5 else ''}"
        )
    
    # Validate raw input
    missing = validate_raw_input(salaries)
    if missing:
        print(f"[ownership] Missing required columns: {missing}")
        return None
    
    # Fill optional enrichment columns (injuries, etc.)
    salaries = fill_optional_columns(salaries)
    
    # Compute features
    features = compute_ownership_features(
        salaries,
        proj_fpts_col="proj_fpts",
        salary_col="salary",
        pos_col="pos",
        slate_id_col=None,  # Treat as single slate
    )
    
    pred_pct: pd.Series
    pred_raw: pd.Series

    if model_family == "ownership_v2":
        assert bundle_v2 is not None  # narrowing for type checkers
        required_gtv2_cols = _required_gtv2_feature_columns(bundle_v2.feature_columns)
        features = _attach_live_gtv2_enrichment(
            feature_frame=features,
            salaries=salaries,
            required_gtv2_cols=required_gtv2_cols,
            game_date=game_date,
            run_id=run_id,
            data_root=data_root,
            gtv2_features_path=gtv2_features_path,
        )
        missing_v2 = [c for c in bundle_v2.feature_columns if c not in features.columns]
        if missing_v2:
            print(f"[ownership] ownership_v2 feature mismatch: {missing_v2}")
            return None
        pred_df = predict_ownership_transformer_slate(features, bundle=bundle_v2)
        pred_pct = pd.to_numeric(pred_df["pred_own_pct"], errors="coerce").fillna(0.0)
        pred_raw = pd.to_numeric(pred_df["pred_own_pct_raw"], errors="coerce").fillna(0.0)
    else:
        assert bundle_v1 is not None  # narrowing for type checkers
        # Prepare model input (strict feature selection)
        try:
            _ = prepare_model_input(features, bundle_v1.feature_cols)
        except KeyError as e:
            print(f"[ownership] Feature mismatch: {e}")
            return None
        pred_pct = predict_ownership(features, bundle_v1)
        # Preserve raw model output prior to any filtering/normalization.
        pred_raw = pred_pct.astype(float)

    # Build output DataFrame
    output_cols = ["player_id", "player_name", "salary", "pos", "team"]
    output = salaries[[c for c in output_cols if c in salaries.columns]].copy()
    output["proj_fpts"] = features["proj_fpts"]
    output["pred_own_pct"] = pred_pct.values
    output["pred_own_pct_raw"] = pred_raw.values
    output["game_date"] = game_date
    output["run_id"] = run_id
    output["model_run"] = model_run
    output["model_family"] = model_family

    if forced_out_mask.any():
        output.loc[forced_out_mask, "pred_own_pct"] = 0.0
        output.loc[forced_out_mask, "pred_own_pct_raw"] = 0.0

    config = _load_calibration_config()

    output = _apply_postprocessing(output, salaries, config=config, data_root=data_root)
    
    # Add draft_group_id to output
    output["draft_group_id"] = draft_group_id
    output["is_locked"] = False
    output = output.drop(columns=["nba_player_id"], errors="ignore")

    return output


def score_all_slates(
    game_date: date,
    run_id: str,
    data_root: Path,
    model_run: str = PRODUCTION_MODEL_RUN,
    model_family: OwnershipModelFamily = "ownership_v1",
    *,
    gtv2_features_path: Path | None = None,
    ignore_lock_cache: bool = False,
    write_lock_cache: bool = True,
    current_time: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Score ownership predictions for all slates on a date.
    
    Handles per-slate lock detection: once a slate's first game starts,
    that slate returns cached predictions while other slates continue updating.
    
    Returns dict of {draft_group_id: predictions_df}.
    """
    # Load all slates
    slates = _load_all_slates(game_date, data_root)
    if not slates:
        print(f"[ownership] No slates found for {game_date}")
        return {}
    
    # Load schedule for lock detection
    schedule = _load_schedule_with_times(game_date, data_root)
    if current_time is None:
        current_time = datetime.now(tz=UTC)
    elif current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=UTC)
    else:
        current_time = current_time.astimezone(UTC)
    
    results = {}
    
    for dg_id, slate_df in slates.items():
        slate_teams = set(slate_df["team"].unique()) if "team" in slate_df.columns else set()
        slate_lock_ts = _load_dk_draft_group_lock_ts(draft_group_id=str(dg_id), data_root=data_root)
        if slate_lock_ts is None:
            slate_lock_ts = _get_slate_first_game_time(slate_teams, schedule)
        cutoff_ts = min(current_time, slate_lock_ts) if slate_lock_ts is not None else current_time
        
        # Check if this slate is locked
        is_locked = slate_lock_ts is not None and current_time >= slate_lock_ts
        if is_locked:
            print(f"[ownership] Slate {dg_id} is LOCKED (first game: {slate_lock_ts})")

            if not ignore_lock_cache:
                # Try to load cached predictions
                cached = _load_locked_predictions(
                    game_date,
                    dg_id,
                    data_root,
                    model_family=model_family,
                    model_run=model_run,
                )
                if cached is not None and not cached.empty and slate_lock_ts is not None:
                    cutoff_col = cached.get("injuries_cutoff_ts")
                    cutoff = pd.to_datetime(cutoff_col, utc=True, errors="coerce").max() if cutoff_col is not None else pd.NaT
                    lock_cutoff = pd.Timestamp(slate_lock_ts).tz_convert("UTC")
                    if pd.notna(cutoff) and cutoff == lock_cutoff:
                        cached["is_locked"] = True
                        results[dg_id] = cached
                        print(f"  -> Using cached predictions: {len(cached)} players")
                        continue
                    print("  -> WARNING: Cached predictions stale for lock; rescoring and overwriting cache")
                else:
                    print("  -> WARNING: No cached predictions, scoring anyway")
            else:
                print("  -> Backtest mode: ignoring lock cache, rescoring anyway")
        else:
            print(f"[ownership] Slate {dg_id} is UNLOCKED (first game: {slate_lock_ts})")
        
        # Score this slate
        predictions = score_ownership(
            slate_df=slate_df,
            draft_group_id=dg_id,
            game_date=game_date,
            run_id=run_id,
            data_root=data_root,
            model_run=model_run,
            model_family=model_family,
            injuries_cutoff_ts=cutoff_ts,
            gtv2_features_path=gtv2_features_path,
        )
        
        if predictions is not None:
            if is_locked:
                predictions = predictions.copy()
                predictions["is_locked"] = True
            results[dg_id] = predictions
            
            # Persist lock cache only after lock, to avoid freezing predictions pre-lock.
            if is_locked and write_lock_cache:
                _save_locked_predictions(
                    predictions,
                    game_date,
                    dg_id,
                    data_root,
                    model_family=model_family,
                    model_run=model_run,
                    overwrite=True,
                )
    
    return results


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}.json")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _save_slates_metadata(
    results: dict[str, pd.DataFrame],
    game_date: date,
    schedule: pd.DataFrame,
    data_root: Path,
    *,
    out_dir: Path,
) -> None:
    """Save slates.json metadata file."""
    current_time = datetime.now(tz=UTC)
    slates_meta = {}
    
    for dg_id, df in results.items():
        teams = list(df["team"].unique()) if "team" in df.columns else []
        slate_teams = set(teams)
        first_game = _load_dk_draft_group_lock_ts(draft_group_id=str(dg_id), data_root=data_root)
        if first_game is None:
            first_game = _get_slate_first_game_time(slate_teams, schedule)
        is_locked = first_game is not None and current_time >= first_game
        
        slates_meta[dg_id] = {
            "player_count": len(df),
            "teams": teams,
            "first_game_time": first_game.isoformat() if first_game else None,
            "is_locked": is_locked,
        }

    _atomic_write_json(out_dir / "slates.json", slates_meta)


def _compute_gtv2_match_coverage(
    *,
    results: dict[str, pd.DataFrame],
    game_date: date,
    run_id: str,
    data_root: Path,
    gtv2_features_path: Path | None,
) -> tuple[str | None, dict[str, float]]:
    frame, source = _load_live_gtv2_feature_frame(
        game_date=game_date,
        run_id=run_id,
        data_root=data_root,
        gtv2_features_path=gtv2_features_path,
    )
    if frame is None or frame.empty:
        return source, {}

    id_set: set[int] = set()
    if "player_id" in frame.columns:
        ids = pd.to_numeric(frame["player_id"], errors="coerce").dropna().astype(int)
        id_set = set(ids.tolist())

    name_set: set[str] = set()
    if "_name_norm" in frame.columns:
        name_set = set(
            frame["_name_norm"].astype(str).map(str.strip).loc[lambda s: s.ne("")].tolist()
        )

    coverage: dict[str, float] = {}
    for dg_id, df in results.items():
        if df.empty:
            coverage[dg_id] = 0.0
            continue

        matched = pd.Series(False, index=df.index)
        if id_set and "player_id" in df.columns:
            ids = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
            matched |= ids.map(lambda v: int(v) in id_set if pd.notna(v) else False)
        if name_set and "player_name" in df.columns:
            names = df["player_name"].astype(str).map(_normalize_name)
            matched |= names.map(lambda n: n in name_set)
        coverage[dg_id] = float(matched.mean()) if len(matched) else 0.0

    return source, coverage


def _write_ownership_health_summary(
    *,
    results: dict[str, pd.DataFrame],
    game_date: date,
    run_id: str,
    data_root: Path,
    model_family: OwnershipModelFamily,
    model_run: str,
    gtv2_features_path: Path | None,
    write_lock_cache: bool,
    ignore_lock_cache: bool,
    out_dir: Path,
) -> Path:
    norm_cfg = _load_calibration_config().get("normalization", {})
    target_sum_pct = float(norm_cfg.get("target_sum_pct", 800.0))
    tol = max(5.0, target_sum_pct * 0.03)

    gtv2_source: str | None = None
    gtv2_coverage: dict[str, float] = {}
    if model_family == "ownership_v2":
        gtv2_source, gtv2_coverage = _compute_gtv2_match_coverage(
            results=results,
            game_date=game_date,
            run_id=run_id,
            data_root=data_root,
            gtv2_features_path=gtv2_features_path,
        )

    slates: dict[str, dict[str, object]] = {}
    warnings: list[str] = []
    for dg_id, df in sorted(results.items(), key=lambda kv: kv[0]):
        pred_series = pd.to_numeric(df.get("pred_own_pct"), errors="coerce")
        raw_series = pd.to_numeric(df.get("pred_own_pct_raw"), errors="coerce")
        pred_sum = float(pred_series.fillna(0.0).sum())
        raw_sum = float(raw_series.fillna(0.0).sum())
        zero_pred_count = int((pred_series.fillna(0.0) <= 0.0).sum())
        is_locked = bool(df.get("is_locked", pd.Series(False)).astype(bool).any())

        scoped_lock_path, legacy_lock_path = _lock_cache_paths(
            game_date=game_date,
            draft_group_id=str(dg_id),
            data_root=data_root,
            model_family=model_family,
            model_run=model_run,
        )
        scoped_exists = scoped_lock_path.exists()
        legacy_exists = legacy_lock_path.exists()

        if abs(pred_sum - target_sum_pct) > tol:
            warnings.append(
                f"slate {dg_id}: pred_own_pct sum {pred_sum:.2f} outside target {target_sum_pct:.2f} +/- {tol:.2f}"
            )
        if pred_sum <= 0.0:
            warnings.append(f"slate {dg_id}: pred_own_pct sum is non-positive ({pred_sum:.2f})")
        if raw_sum <= 0.0:
            warnings.append(f"slate {dg_id}: pred_own_pct_raw sum is non-positive ({raw_sum:.2f})")
        if is_locked and write_lock_cache and (not ignore_lock_cache) and (not scoped_exists):
            warnings.append(
                f"slate {dg_id}: locked slate missing scoped lock cache file {scoped_lock_path.name}"
            )

        coverage = gtv2_coverage.get(dg_id)
        if model_family == "ownership_v2":
            if coverage is None:
                warnings.append(f"slate {dg_id}: no computed GTV2 row coverage")
            elif coverage < 0.25:
                warnings.append(
                    f"slate {dg_id}: low GTV2 row coverage ({coverage:.3f})"
                )

        slates[str(dg_id)] = {
            "player_count": int(len(df)),
            "is_locked": bool(is_locked),
            "pred_own_pct_sum": pred_sum,
            "pred_own_pct_raw_sum": raw_sum,
            "zero_pred_count": zero_pred_count,
            "scoped_lock_cache_path": str(scoped_lock_path),
            "scoped_lock_cache_exists": bool(scoped_exists),
            "legacy_lock_cache_path": str(legacy_lock_path),
            "legacy_lock_cache_exists": bool(legacy_exists),
            "gtv2_row_coverage": coverage,
        }

    if model_family == "ownership_v2" and not gtv2_coverage:
        warnings.append("ownership_v2 selected but no GTV2 feature frame was available for coverage checks")

    payload: dict[str, object] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "game_date": str(game_date),
        "run_id": str(run_id),
        "model_family": str(model_family),
        "model_run": str(model_run),
        "target_sum_pct": float(target_sum_pct),
        "sum_tolerance_pct": float(tol),
        "write_lock_cache": bool(write_lock_cache),
        "ignore_lock_cache": bool(ignore_lock_cache),
        "gtv2_features_path": str(gtv2_features_path) if gtv2_features_path else None,
        "gtv2_source": gtv2_source,
        "slates": slates,
        "warning_count": int(len(warnings)),
        "warnings": warnings,
    }
    out_path = out_dir / "ownership_health_summary.json"
    _atomic_write_json(out_path, payload)
    if warnings:
        print(f"[ownership] Health warnings: {len(warnings)} (see {out_path})")
    else:
        print(f"[ownership] Health summary OK -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Score ownership predictions")
    parser.add_argument("--date", required=True, help="Game date (YYYY-MM-DD)")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--data-root", default=None, help="Data root path")
    parser.add_argument(
        "--model-family",
        choices=["ownership_v1", "ownership_v2"],
        default="ownership_v1",
        help="Ownership model family.",
    )
    parser.add_argument(
        "--model-run",
        default=None,
        help="Model run ID (defaults per model family).",
    )
    parser.add_argument(
        "--gtv2-features-path",
        default=None,
        help=(
            "Optional parquet/csv path or directory with live GTV2 features for ownership_v2 enrichment "
            "(zero-filled when unavailable)."
        ),
    )
    parser.add_argument(
        "--ignore-lock-cache",
        action="store_true",
        help="Force scoring even when slates are locked (useful for backtests/rescoring).",
    )
    parser.add_argument(
        "--no-write-lock-cache",
        action="store_true",
        help="Do not write *_locked.parquet cache files (useful for backtests).",
    )
    args = parser.parse_args()
    
    game_date = date.fromisoformat(args.date)
    root = Path(args.data_root) if args.data_root else data_path()
    
    model_family: OwnershipModelFamily = str(args.model_family)
    if args.model_run:
        model_run = str(args.model_run)
    else:
        model_run = PRODUCTION_MODEL_RUN_V2 if model_family == "ownership_v2" else PRODUCTION_MODEL_RUN

    gtv2_features_path = (
        Path(str(args.gtv2_features_path)).expanduser().resolve()
        if args.gtv2_features_path
        else None
    )

    # Score all slates
    results = score_all_slates(
        game_date,
        args.run_id,
        root,
        model_run,
        model_family,
        gtv2_features_path=gtv2_features_path,
        ignore_lock_cache=bool(args.ignore_lock_cache),
        write_lock_cache=not bool(args.no_write_lock_cache),
    )
    
    if not results:
        print("[ownership] No predictions generated")
        return 1
    
    # Save per-slate predictions
    out_dir = root / "silver" / "ownership_predictions" / str(game_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / f"run={args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    allow_legacy_flat = os.environ.get("PROJECTIONS_ALLOW_LEGACY_OWNERSHIP_FLAT_WRITES", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    for dg_id, df in results.items():
        out_path = run_dir / f"{dg_id}.parquet"
        df.to_parquet(out_path)
        if allow_legacy_flat:
            legacy_path = out_dir / f"{dg_id}.parquet"
            df.to_parquet(legacy_path)
        print(f"[ownership] Saved slate {dg_id}: {len(df)} predictions -> {out_path}")

    if os.environ.get("PROJECTIONS_SKIP_POINTER_WRITES", "").strip().lower() not in {"1", "true", "yes"}:
        from projections.pipeline import writer_guard

        writer_guard.assert_can_write_pointers(purpose=f"score_ownership_live promote {out_dir}")
        latest_payload = {
            "run_id": args.run_id,
            "generated_at": datetime.now(tz=UTC).isoformat(),
        }
        (out_dir / "latest_run.json").write_text(json.dumps(latest_payload, indent=2), encoding="utf-8")
    
    # Save slates metadata
    schedule = _load_schedule_with_times(game_date, root)
    _save_slates_metadata(results, game_date, schedule, root, out_dir=run_dir)
    _write_ownership_health_summary(
        results=results,
        game_date=game_date,
        run_id=str(args.run_id),
        data_root=root,
        model_family=model_family,
        model_run=model_run,
        gtv2_features_path=gtv2_features_path,
        write_lock_cache=not bool(args.no_write_lock_cache),
        ignore_lock_cache=bool(args.ignore_lock_cache),
        out_dir=run_dir,
    )
    
    # Print summary for largest slate
    main_dg = max(results.keys(), key=lambda k: len(results[k]))
    main_df = results[main_dg]
    print(f"\n[ownership] Main slate ({main_dg}) top 5 by ownership:")
    print(main_df.nlargest(5, "pred_own_pct")[["player_name", "salary", "pred_own_pct"]].to_string())
    
    return 0


if __name__ == "__main__":
    exit(main())
