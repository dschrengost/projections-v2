"""QuickBuild optimizer service layer.

This module provides the service functions that:
1. Load and merge player pools from projections + DK salaries
2. Execute QuickBuild jobs with progress tracking
3. Manage job lifecycle (create, poll, retrieve results)
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import numpy as np
import yaml

from projections.dk.salaries_schema import dk_salaries_gold_path, normalize_positions
from projections.dk.slates import list_draft_groups_for_date
from projections.fpts_v2.scoring import compute_dk_fpts
from projections.names import normalize_player_name
from projections.pipeline import control_plane
from projections.pipeline.effective_inputs import EFFECTIVE_MINUTES_FILENAME
from projections.optimizer.quick_build import (
    QuickBuildConfig,
    QuickBuildResult,
    WorldSampleConfig,
    quick_build_pool,
)
from projections.optimizer.lineup_sim_stats import (
    compute_lineup_distribution_stats,
    load_world_fpts_matrix,
)
from projections.projections_bundle import add_canonical_projection_fields, resolve_unified_projections_run
from projections.optimizer.objective import (
    set_active_late_swap_bonus,
    LateSwapBonusConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "optimizer.yaml"
_config_cache: Optional[Dict[str, Any]] = None


def load_optimizer_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load optimizer configuration from YAML."""
    global _config_cache
    if _config_cache is not None and path is None:
        return _config_cache

    config_path = path or _CONFIG_PATH
    if not config_path.exists():
        logger.warning("Optimizer config not found at %s; using defaults", config_path)
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    if path is None:
        _config_cache = config
    return config


def get_data_root() -> Path:
    """Get the projections data root directory."""
    return Path(os.environ.get("PROJECTIONS_DATA_ROOT", "/home/daniel/projections-data"))


def get_minutes_daily_root() -> Path:
    """Resolve the minutes daily artifacts root (used to pick the freshest run_id)."""
    raw = os.environ.get("MINUTES_DAILY_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return get_data_root() / "artifacts" / "minutes_v1" / "daily"


def _latest_minutes_run_id(game_date: str, minutes_root: Path) -> str | None:
    """Read artifacts/minutes_v1/daily/<date> promoted run_id when available."""
    return control_plane.read_promoted_run_id(minutes_root / game_date)


def load_ownership_for_date(
    game_date: str,
    *,
    draft_group_id: int | str | None = None,
    run_id: str | None = None,
    data_root: Path | None = None,
) -> pd.DataFrame | None:
    """Load ownership predictions for a slate, preferring the resolved projections run."""
    root = data_root or get_data_root()
    base_dir = root / "silver" / "ownership_predictions" / str(game_date)
    draft_group_str = str(draft_group_id) if draft_group_id is not None else None

    resolved_run = resolve_unified_projections_run(game_date, run_id=run_id, data_root=root)
    candidate_dirs: list[Path] = []
    if resolved_run.run_id:
        candidate_dirs.append(base_dir / f"run={resolved_run.run_id}")
    candidate_dirs.append(base_dir)

    seen: set[Path] = set()
    for slate_dir in candidate_dirs:
        if slate_dir in seen or not slate_dir.exists():
            continue
        seen.add(slate_dir)

        if draft_group_str:
            own_path = slate_dir / f"{draft_group_str}.parquet"
            if own_path.exists():
                return pd.read_parquet(own_path)
            continue

        slate_files = [p for p in slate_dir.glob("*.parquet") if not p.name.endswith("_locked.parquet")]
        if slate_files:
            own_path = max(slate_files, key=lambda p: p.stat().st_size)
            return pd.read_parquet(own_path)

    if draft_group_str is None and run_id is None and control_plane.allow_unpromoted_run_reads():
        legacy_path = root / "silver" / "ownership_predictions" / f"{game_date}.parquet"
        if legacy_path.exists():
            return pd.read_parquet(legacy_path)

    return None


def _load_rates_v1_live(
    game_date: str,
    root: Path,
    *,
    run_id: str | None,
) -> tuple[pd.DataFrame, str | None] | tuple[None, None]:
    """
    Load live rates predictions for a given date and (optionally) run_id.

    Returns (df, resolved_run_id) or (None, None) if not found.
    """
    import json

    base = root / "gold" / "rates_v1_live" / game_date
    if run_id:
        candidate = base / f"run={run_id}" / "rates.parquet"
        if candidate.exists():
            return pd.read_parquet(candidate), run_id

    pointer = base / "latest_run.json"
    if pointer.exists():
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            latest = payload.get("run_id")
        except Exception:
            latest = None
        if latest:
            latest = str(latest)
            candidate = base / f"run={latest}" / "rates.parquet"
            if candidate.exists():
                return pd.read_parquet(candidate), latest

    direct = base / "rates.parquet"
    if direct.exists():
        return pd.read_parquet(direct), None

    return None, None


def _attach_rates_mean_fpts(
    minutes_df: pd.DataFrame,
    rates_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute deterministic mean DK FPTS from minutes + rates predictions and attach as `fpts_mean`.

    This is a fast fallback for optimizer projections when sim outputs are unavailable.
    """
    if minutes_df.empty or rates_df.empty:
        return minutes_df

    join_keys = ["game_date", "game_id", "team_id", "player_id"]
    if any(k not in minutes_df.columns for k in join_keys) or any(k not in rates_df.columns for k in join_keys):
        return minutes_df

    def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for col in candidates:
            if col in df.columns:
                return col
        return None

    minutes_col = _first_present(
        minutes_df,
        ["minutes_final", "minutes_p50_cond", "minutes_p50", "minutes_mean", "minutes", "minutes_pred"],
    )
    if minutes_col is None:
        return minutes_df

    # Normalize join keys and minutes
    left = minutes_df.copy()
    right = rates_df.copy()
    left["game_date"] = pd.to_datetime(left["game_date"], errors="coerce").dt.normalize()
    right["game_date"] = pd.to_datetime(right["game_date"], errors="coerce").dt.normalize()
    for key in ("game_id", "team_id", "player_id"):
        left[key] = pd.to_numeric(left[key], errors="coerce")
        right[key] = pd.to_numeric(right[key], errors="coerce")
    left["_minutes_mean"] = pd.to_numeric(left[minutes_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    merged = left.merge(right, on=join_keys, how="left", suffixes=("", "_rates"))
    if merged.empty:
        return minutes_df

    def _rate_col(target: str) -> str | None:
        if target in merged.columns:
            return target
        pred = f"pred_{target}"
        if pred in merged.columns:
            return pred
        return None

    per_min_targets = [
        "fga2_per_min",
        "fga3_per_min",
        "fta_per_min",
        "ast_per_min",
        "tov_per_min",
        "oreb_per_min",
        "dreb_per_min",
        "stl_per_min",
        "blk_per_min",
    ]
    cols = {t: _rate_col(t) for t in per_min_targets}
    if any(v is None for v in cols.values()):
        return minutes_df

    minutes = merged["_minutes_mean"].to_numpy(dtype=float, copy=False)
    totals = {}
    for target, col in cols.items():
        base = target.replace("_per_min", "")
        rate = pd.to_numeric(merged[col], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
        totals[base] = (minutes * rate).clip(min=0.0)

    # Optional efficiencies
    fg2_col = _rate_col("fg2_pct")
    fg3_col = _rate_col("fg3_pct")
    ft_col = _rate_col("ft_pct")
    have_eff = fg2_col is not None and fg3_col is not None and ft_col is not None

    if have_eff:
        fg2_pct = pd.to_numeric(merged[fg2_col], errors="coerce").fillna(0.55).to_numpy(dtype=float, copy=False)
        fg3_pct = pd.to_numeric(merged[fg3_col], errors="coerce").fillna(0.36).to_numpy(dtype=float, copy=False)
        ft_pct = pd.to_numeric(merged[ft_col], errors="coerce").fillna(0.78).to_numpy(dtype=float, copy=False)
        fg2_pct = np.clip(fg2_pct, 0.30, 0.75)
        fg3_pct = np.clip(fg3_pct, 0.20, 0.55)
        ft_pct = np.clip(ft_pct, 0.50, 0.95)
        fgm2 = totals["fga2"] * fg2_pct
        fgm3 = totals["fga3"] * fg3_pct
        ftm = totals["fta"] * ft_pct
    else:
        # Conservative fallback: treat attempts as makes; ft at 0.75x.
        fgm2 = totals["fga2"]
        fgm3 = totals["fga3"]
        ftm = 0.75 * totals["fta"]

    pts = 2.0 * fgm2 + 3.0 * fgm3 + ftm
    reb = totals["oreb"] + totals["dreb"]

    scoring_frame = pd.DataFrame(
        {
            "pts": pts,
            "fgm": fgm2 + fgm3,
            "fga": totals["fga2"] + totals["fga3"],
            "fg3m": fgm3,
            "fg3a": totals["fga3"],
            "ftm": ftm,
            "fta": totals["fta"],
            "reb": reb,
            "oreb": totals["oreb"],
            "dreb": totals["dreb"],
            "ast": totals["ast"],
            "stl": totals["stl"],
            "blk": totals["blk"],
            "tov": totals["tov"],
            "pf": 0.0,
            "plus_minus": 0.0,
        }
    )

    merged["fpts_mean"] = compute_dk_fpts(scoring_frame).astype(float)

    # Keep the original columns; merge back onto the normalized copy to avoid dtype mismatches.
    out_cols = join_keys + ["fpts_mean"]
    out = left.merge(
        merged[out_cols].drop_duplicates(subset=join_keys, keep="last"),
        on=join_keys,
        how="left",
    )
    return out.drop(columns=["_minutes_mean"], errors="ignore")


# ---------------------------------------------------------------------------
# Player Pool Building
# ---------------------------------------------------------------------------


def load_projections_for_date(
    game_date: str,
    run_id: Optional[str] = None,
    data_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Load projections from unified projections artifact or gold layer.

    Also merges sim_v2 FPTS projections if available.

    Returns DataFrame with columns:
        player_id, player_name, team_tricode, sim_dk_fpts_mean, pred_own_pct, etc.
    """
    root = data_root or get_data_root()
    df: pd.DataFrame | None = None

    # Preferred: unified projections artifact. Resolve run_id using the same
    # blessed/pinned/promoted pointer logic as the dashboard so all consumers
    # agree on the default run selection.
    resolved = resolve_unified_projections_run(game_date, run_id=run_id, data_root=root)
    if resolved.projections_path is not None:
        df = pd.read_parquet(resolved.projections_path)
        df = add_canonical_projection_fields(df)
        logger.info(
            "Loaded unified projections for %s from run=%s (source=%s, rows=%d)",
            game_date,
            resolved.run_id,
            resolved.source,
            len(df),
        )

    # When we need to fall back to per-model artifacts, treat run_id as a minutes run id
    # (historically minutes + sim outputs were partitioned by the same run token).
    minutes_root = get_minutes_daily_root()
    resolved_minutes_run_id = run_id or _latest_minutes_run_id(game_date, minutes_root)

    # Fall back to gold projections_minutes_v1
    if df is None:
        gold_dir = root / "gold" / "projections_minutes_v1" / f"game_date={game_date}"
        if gold_dir.exists():
            gold_run_id = control_plane.read_promoted_run_id(gold_dir)
            if gold_run_id is None and control_plane.allow_unpromoted_run_reads():
                run_dirs = sorted([p for p in gold_dir.glob("run=*") if p.is_dir()], reverse=True)
                if run_dirs:
                    gold_run_id = run_dirs[0].name.split("=", 1)[1]

            if gold_run_id:
                run_dir = gold_dir / f"run={gold_run_id}"
                for candidate in (run_dir / EFFECTIVE_MINUTES_FILENAME, run_dir / "minutes.parquet"):
                    if candidate.exists():
                        df = pd.read_parquet(candidate)
                        logger.info(
                            "Loaded gold projections_minutes_v1 for %s from run=%s (%d rows)",
                            game_date,
                            gold_run_id,
                            len(df),
                        )
                        break


    if df is None:
        raise FileNotFoundError(f"No projections found for {game_date}")

    # Check if we have FPTS data, if not try to merge from sim_v2
    fpts_cols = [
        "sim_dk_fpts_mean_uncond",
        "dk_fpts_mean_uncond",
        "sim_dk_fpts_mean",
        "dk_fpts_mean",
        "proj_fpts",
        "fpts_mean",
    ]
    has_fpts = any(c in df.columns and df[c].notna().any() for c in fpts_cols)

    if not has_fpts:
        sim_df = _load_sim_projections(game_date, root, minutes_run_id=resolved_minutes_run_id)
        if sim_df is not None and not sim_df.empty:
            # Merge sim projections
            join_keys = ["player_id"]
            if "game_id" in df.columns and "game_id" in sim_df.columns:
                join_keys.append("game_id")

            # Merge
            df = df.merge(sim_df, on=join_keys, how="left", suffixes=("", "_sim"))
            logger.info(
                "Merged sim_v2 projections for %s (%d players with FPTS)",
                game_date,
                df["dk_fpts_mean"].notna().sum() if "dk_fpts_mean" in df.columns else 0,
            )

    # Final fallback: compute deterministic mean FPTS from minutes + rates_v1_live.
    has_fpts = any(c in df.columns and df[c].notna().any() for c in fpts_cols)
    if not has_fpts:
        rates_df, rates_run_id = _load_rates_v1_live(game_date, root, run_id=resolved_minutes_run_id)
        if rates_df is not None and not rates_df.empty:
            before = set(df.columns)
            df = _attach_rates_mean_fpts(df, rates_df)
            added = sorted(set(df.columns) - before)
            logger.info(
                "Attached rates-derived fpts_mean for %s (rates_run=%s, added=%s)",
                game_date,
                rates_run_id,
                added,
            )

    return add_canonical_projection_fields(df)


def _load_sim_projections(
    game_date: str,
    root: Path,
    *,
    minutes_run_id: str | None = None,
) -> Optional[pd.DataFrame]:
    """Load sim_v2 FPTS projections for a date."""
    import json

    # Source of truth: projections.parquet emitted by scripts.sim_v2.generate_worlds_fpts_v2
    # under artifacts/sim_v2/worlds_fpts_v2/game_date=.../run=...
    sim_root = root / "artifacts" / "sim_v2" / "worlds_fpts_v2"
    base_candidates = [
        sim_root / f"game_date={game_date}",
        sim_root / f"date={game_date}",
        sim_root / game_date,
    ]

    def _resolve_run_dir(base_dir: Path) -> Path | None:
        pointer = base_dir / "latest_run.json"
        if pointer.exists():
            try:
                payload = json.loads(pointer.read_text(encoding="utf-8"))
                latest = payload.get("run_id")
            except Exception:
                latest = None
            if latest:
                candidate = base_dir / f"run={latest}"
                if candidate.exists():
                    return candidate

        run_dirs = sorted(
            [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run=")],
            reverse=True,
        )
        return run_dirs[0] if run_dirs else None

    def _read_frame(path: Path) -> Optional[pd.DataFrame]:
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Failed to load sim projections from %s: %s", path, exc)
            return None

    def _read_run_dir(run_dir: Path) -> Optional[pd.DataFrame]:
        for name in ("projections.parquet", "sim_v2_projections.parquet"):
            candidate = run_dir / name
            if not candidate.exists():
                continue
            df = _read_frame(candidate)
            if df is not None:
                return df
        return None

    for base in base_candidates:
        if not base.exists():
            continue

        if minutes_run_id and base.is_dir():
            run_dir = base / f"run={minutes_run_id}"
            if run_dir.exists():
                df = _read_run_dir(run_dir)
                if df is not None:
                    return df

        if base.is_file() and base.suffix == ".parquet":
            df = _read_frame(base)
        else:
            df = None
            direct = base / "projections.parquet"
            if direct.exists():
                df = _read_frame(direct)
            if df is None and base.is_dir():
                run_dir = _resolve_run_dir(base)
                if run_dir is not None:
                    df = _read_run_dir(run_dir)
            if df is None:
                continue

        if minutes_run_id:
            if "minutes_run_id" not in df.columns:
                continue
            df = df.loc[df["minutes_run_id"].astype(str) == str(minutes_run_id)].copy()
            if df.empty:
                continue
        return df

    return None


def _load_game_info_from_draftables(
    draft_group_id: int,
    data_root: Path,
) -> Dict[int, Dict[str, Any]]:
    """Load game/competition info from bronze draftables.
    
    Returns dict mapping competition_id -> {matchup, start_time_utc}.
    """
    import json
    from datetime import datetime
    
    bronze_path = data_root / "bronze" / "dk" / "draftables" / f"draftables_raw_{draft_group_id}.json"
    if not bronze_path.exists():
        logger.debug("No bronze draftables found at %s", bronze_path)
        return {}
    
    try:
        with open(bronze_path) as f:
            payload = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load draftables JSON: %s", exc)
        return {}
    
    game_info: Dict[int, Dict[str, Any]] = {}
    
    # Parse competitions array
    competitions = payload.get("competitions", [])
    for comp in competitions:
        comp_id = comp.get("competitionId")
        if comp_id is None:
            continue
        
        # Build matchup from team names
        away = comp.get("awayTeam", {}).get("abbreviation", "???")
        home = comp.get("homeTeam", {}).get("abbreviation", "???")
        matchup = f"{away}@{home}"
        
        # Parse start time
        start_str = comp.get("startTime")
        start_utc = None
        if start_str:
            try:
                # Format: "2025-12-01T00:00:00.0000000Z"
                cleaned = start_str.replace("Z", "+00:00")
                if "." in cleaned:
                    # Truncate microseconds to 6 digits
                    base, rest = cleaned.rsplit(".", 1)
                    tz_idx = rest.find("+")
                    if tz_idx == -1:
                        tz_idx = rest.find("-")
                    if tz_idx > 0:
                        micros = rest[:min(6, tz_idx)]
                        tz_part = rest[tz_idx:]
                        cleaned = f"{base}.{micros}{tz_part}"
                start_utc = datetime.fromisoformat(cleaned)
            except Exception:
                logger.debug("Failed to parse start time: %s", start_str)
        
        game_info[comp_id] = {
            "matchup": matchup,
            "start_time_utc": start_utc,
        }
    
    logger.debug("Loaded %d games from draftables for dg=%d", len(game_info), draft_group_id)
    return game_info


def load_salaries_for_date(
    game_date: str,
    draft_group_id: int,
    site: str = "dk",
    data_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Load DK salaries from gold layer.

    Returns DataFrame with columns:
        dk_player_id, display_name, positions, salary, team_abbrev, status,
        game_matchup, game_start_utc
    """
    root = data_root or get_data_root()
    salaries_path = dk_salaries_gold_path(root, site, game_date, draft_group_id)

    if not salaries_path.exists():
        raise FileNotFoundError(f"Salaries not found: {salaries_path}")

    df = pd.read_parquet(salaries_path)
    
    # Load game info from bronze draftables
    game_info = _load_game_info_from_draftables(draft_group_id, root)
    
    # Add game_matchup and game_start_utc columns
    def get_game_matchup(comp_ids):
        if not comp_ids or not game_info:
            return None
        # Take first competition ID
        if isinstance(comp_ids, (list, np.ndarray)) and len(comp_ids) > 0:
            comp_id = int(comp_ids[0])
        else:
            return None
        info = game_info.get(comp_id)
        return info["matchup"] if info else None
    
    def get_game_start(comp_ids):
        if not comp_ids or not game_info:
            return None
        if isinstance(comp_ids, (list, np.ndarray)) and len(comp_ids) > 0:
            comp_id = int(comp_ids[0])
        else:
            return None
        info = game_info.get(comp_id)
        return info["start_time_utc"] if info else None
    
    if "raw_competition_ids" in df.columns:
        df["game_matchup"] = df["raw_competition_ids"].apply(get_game_matchup)
        df["game_start_utc"] = df["raw_competition_ids"].apply(get_game_start)
    else:
        df["game_matchup"] = None
        df["game_start_utc"] = None
    
    logger.info(
        "Loaded salaries for %s draft_group=%d (%d players, %d games)",
        game_date,
        draft_group_id,
        len(df),
        len(game_info),
    )
    return df


def _normalize_name(val: object) -> str:
    """Normalize player name for fuzzy matching."""
    return normalize_player_name(val)


def _normalize_team(val: object) -> str:
    """Normalize team abbreviation."""
    if val is None:
        return ""
    text = str(val).strip().upper()
    aliases = {
        "PHO": "PHX",
        "GS": "GSW",
        "SA": "SAS",
        "NO": "NOP",
        "NY": "NYK",
        "BRK": "BKN",
    }
    return aliases.get(text, text)


def _normalize_status(val: object) -> str:
    """Normalize salary status value."""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass
    text = str(val).strip().upper()
    if text in {"", "NONE", "NAN", "<NA>"}:
        return ""
    return text


def _coerce_bool(val: object, default: bool = False) -> bool:
    """Safely coerce scalar values to bool."""
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except Exception:
        pass
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, np.integer)):
        return bool(val)
    if isinstance(val, (float, np.floating)):
        return bool(val)
    text = str(val).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", ""}:
        return False
    return default


def _is_out_status(status: object) -> bool:
    """Return True when DK status clearly indicates player is out."""
    # Observed values are mostly OUT / Q / null. Keep this conservative.
    return _normalize_status(status) in {"OUT", "O", "INACTIVE"}


def _overlay_ownership_columns(
    pool_df: pd.DataFrame,
    ownership_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Overlay slate-specific ownership onto a merged projections/salaries frame."""
    if ownership_df is None or ownership_df.empty:
        return pool_df

    base = pool_df.copy()
    own = ownership_df.copy()

    if "dk_player_id" in base.columns and "player_id" in own.columns:
        base["_own_join_dk_player_id"] = pd.to_numeric(base["dk_player_id"], errors="coerce").astype("Int64")
        own["_own_join_dk_player_id"] = pd.to_numeric(own["player_id"], errors="coerce").astype("Int64")
        join_cols = ["_own_join_dk_player_id"]
    else:
        base_name_col = next((c for c in ["player_name", "display_name", "name"] if c in base.columns), None)
        own_name_col = next((c for c in ["player_name", "display_name", "name"] if c in own.columns), None)
        base_team_col = next((c for c in ["team_tricode", "team_abbrev", "team"] if c in base.columns), None)
        own_team_col = next((c for c in ["team", "team_abbrev", "team_tricode"] if c in own.columns), None)
        if base_name_col is not None and own_name_col is not None:
            base["_own_join_name"] = base[base_name_col].apply(_normalize_name)
            own["_own_join_name"] = own[own_name_col].apply(_normalize_name)
            join_cols = ["_own_join_name"]
            if base_team_col and own_team_col:
                base["_own_join_team"] = base[base_team_col].apply(_normalize_team)
                own["_own_join_team"] = own[own_team_col].apply(_normalize_team)
                join_cols.append("_own_join_team")
        elif "player_id" in base.columns and "player_id" in own.columns:
            base["player_id"] = base["player_id"].astype(str)
            own["player_id"] = own["player_id"].astype(str)
            join_cols = ["player_id"]
        else:
            return pool_df

    own_cols = join_cols + [col for col in ("pred_own_pct",) if col in own.columns]
    if len(own_cols) == len(join_cols):
        return pool_df

    sort_cols: list[str] = []
    ascending: list[bool] = []
    if "pred_own_pct" in own.columns:
        sort_cols.append("pred_own_pct")
        ascending.append(False)
    if "salary" in own.columns:
        sort_cols.append("salary")
        ascending.append(True)
    if sort_cols:
        own = own.sort_values(sort_cols, ascending=ascending, na_position="last")

    merged = base.merge(
        own[own_cols].drop_duplicates(subset=join_cols, keep="first"),
        on=join_cols,
        how="left",
        suffixes=("", "__own"),
    )
    if "pred_own_pct__own" in merged.columns:
        merged["pred_own_pct"] = merged["pred_own_pct__own"].where(
            pd.notna(merged["pred_own_pct__own"]),
            merged.get("pred_own_pct"),
        )
        merged = merged.drop(columns=["pred_own_pct__own"])

    return merged.drop(
        columns=["_own_join_dk_player_id", "_own_join_name", "_own_join_team"],
        errors="ignore",
    )


def build_player_pool(
    game_date: str,
    draft_group_id: int,
    site: str = "dk",
    run_id: Optional[str] = None,
    data_root: Optional[Path] = None,
    include_games: Optional[List[str]] = None,
    exclude_games: Optional[List[str]] = None,
    use_user_overrides: bool = False,
    ownership_mode: str = "renormalize",
    include_unmatched_salaries: bool = False,
    allow_zero_projections: bool = False,
    exclude_inactive_players: bool = True,
) -> List[Dict[str, Any]]:
    """Build optimizer-ready player pool by merging projections with salaries.

    Args:
        game_date: Date in YYYY-MM-DD format
        draft_group_id: DraftKings draft group ID
        site: DFS site (dk or fd)
        run_id: Optional projections run ID
        data_root: Optional data root override
        include_games: If set, only include players from these games (e.g., ["MIN@DAL", "LAL@GSW"])
        exclude_games: If set, exclude players from these games
        use_user_overrides: If True, apply user overrides from SlateOverrides
        ownership_mode: For overrides - "raw" or "renormalize" (default)
        include_unmatched_salaries: If True, keep salary rows even when projections don't match.
        allow_zero_projections: If True, include players with missing/zero projection (proj=0.0).
        exclude_inactive_players: When using overrides, drop players marked out (default True).

    Returns list of player dicts with required QuickBuild fields:
        player_id, name, team, positions, salary, proj, own_proj, stddev, dk_id,
        game_matchup, game_start_utc
        
        When use_user_overrides=True, also includes:
        model_proj, model_minutes, model_own, effective_proj, effective_minutes,
        effective_own, has_override, used_fppm_fallback, is_active, fppm
    """
    root = data_root or get_data_root()

    # Load data sources
    proj_df = load_projections_for_date(game_date, run_id=run_id, data_root=root)
    sal_df = load_salaries_for_date(game_date, draft_group_id, site=site, data_root=root)

    # Prepare join keys
    proj_df = proj_df.copy()
    sal_df = sal_df.copy()

    # Normalize names for fuzzy join
    proj_name_col = next(
        (c for c in ["player_name", "name", "display_name"] if c in proj_df.columns),
        None,
    )
    sal_name_col = next(
        (c for c in ["display_name", "name", "player_name"] if c in sal_df.columns),
        None,
    )
    proj_team_col = next(
        (c for c in ["team_tricode", "team_abbrev", "team"] if c in proj_df.columns),
        None,
    )
    sal_team_col = next(
        (c for c in ["team_abbrev", "team_tricode", "team"] if c in sal_df.columns),
        None,
    )

    if not all([proj_name_col, sal_name_col, proj_team_col, sal_team_col]):
        raise ValueError(
            f"Missing required columns for join. "
            f"proj cols: {proj_df.columns.tolist()}, sal cols: {sal_df.columns.tolist()}"
        )

    proj_df["__join_name"] = proj_df[proj_name_col].apply(_normalize_name)
    proj_df["__join_team"] = proj_df[proj_team_col].apply(_normalize_team)
    sal_df["__join_name"] = sal_df[sal_name_col].apply(_normalize_name)
    sal_df["__join_team"] = sal_df[sal_team_col].apply(_normalize_team)

    # Merge on name + team
    merge_how = "right" if include_unmatched_salaries else "inner"
    merged = proj_df.merge(
        sal_df,
        on=["__join_name", "__join_team"],
        how=merge_how,
        suffixes=("", "_sal"),
    )

    ownership_df = load_ownership_for_date(
        game_date,
        draft_group_id=draft_group_id,
        run_id=run_id,
        data_root=root,
    )
    if ownership_df is not None and not ownership_df.empty:
        merged = _overlay_ownership_columns(merged, ownership_df)
        own_rows = int(pd.to_numeric(merged.get("pred_own_pct"), errors="coerce").notna().sum())
        logger.info(
            "Overlayed ownership for %s draft_group=%s (%d rows with ownership)",
            game_date,
            draft_group_id,
            own_rows,
        )

    logger.info(
        "Player pool merge: %d projections x %d salaries → %d matched",
        len(proj_df),
        len(sal_df),
        len(merged),
    )

    if len(merged) == 0:
        raise ValueError("No players matched between projections and salaries")

    # Apply game filters
    if include_games or exclude_games:
        include_set = set(g.upper() for g in include_games) if include_games else None
        exclude_set = set(g.upper() for g in exclude_games) if exclude_games else set()
        
        def game_filter(matchup):
            if pd.isna(matchup) or not matchup:
                return True  # Keep players with unknown games
            matchup_upper = str(matchup).upper()
            if include_set is not None and matchup_upper not in include_set:
                return False
            if matchup_upper in exclude_set:
                return False
            return True
        
        # Get matchup column (prefer _sal suffix from merge if present)
        matchup_col = "game_matchup_sal" if "game_matchup_sal" in merged.columns else "game_matchup"
        if matchup_col in merged.columns:
            before_count = len(merged)
            merged = merged[merged[matchup_col].apply(game_filter)]
            logger.info(
                "Game filter applied: %d → %d players (include=%s, exclude=%s)",
                before_count,
                len(merged),
                include_games,
                exclude_games,
            )

    # Apply user overrides if requested
    slate_overrides = None
    if use_user_overrides:
        from .user_overrides import apply_overrides as apply_user_overrides, load_slate_overrides
        
        slate_overrides = load_slate_overrides(game_date, draft_group_id)
        merged = apply_user_overrides(
            merged,
            slate_overrides,
            ownership_mode=ownership_mode,  # type: ignore
        )
        logger.info(
            "Applied %d user overrides for %s/dg=%d",
            len(slate_overrides.overrides),
            game_date,
            draft_group_id,
        )
        if exclude_inactive_players:
            # Filter out players marked as out
            before_count = len(merged)
            merged = merged[merged["is_active"]]
            if len(merged) < before_count:
                logger.info(
                    "Excluded %d players marked as out",
                    before_count - len(merged),
                )

    # Build player pool list
    pool: List[Dict[str, Any]] = []

    # Prefer salary-derived columns when merge created conflicts
    salary_col = "salary_sal" if "salary_sal" in merged.columns else "salary"
    positions_col = "positions_sal" if "positions_sal" in merged.columns else "positions"
    dk_player_id_col = "dk_player_id_sal" if "dk_player_id_sal" in merged.columns else "dk_player_id"

    # Identify projection column
    proj_col = next(
        (
            c
            for c in [
                # Canonical decision metric (matches sim/contest-sim): unconditional (DNP=0).
                "fpts_sim_uncond_mean",
                # Prefer unconditional (DNP=0) world aggregates when available.
                "sim_dk_fpts_mean_uncond",
                "dk_fpts_mean_uncond",
                "fpts_sim_cond_mean",
                "sim_dk_fpts_mean",
                "dk_fpts_mean",
                "proj_fpts",
                "fpts_mean",
                "proj",
            ]
            if c in merged.columns
        ),
        None,
    )
    if not proj_col:
        raise ValueError("No projection column found in merged data")
    
    # Identify minutes column
    minutes_col = next(
        (
            c
            for c in [
                # Canonical decision metric (matches sim/contest-sim): unconditional (DNP=0).
                "minutes_sim_uncond_mean",
                "minutes_sim_mean_uncond",
                "minutes_sim_uncond_p50",
                "minutes_sim_p50_uncond",
                # Conditional (given plays).
                "minutes_sim_cond_mean",
                "minutes_sim_mean",
                "minutes_sim_cond_p50",
                "minutes_sim_p50",
                "minutes_final",
                "minutes_p50_cond",
                "minutes_p50",
                "minutes",
                "minutes_pred",
            ]
            if c in merged.columns
        ),
        None,
    )

    # Identify ownership column
    own_col = next(
        (c for c in ["pred_own_pct", "own_proj", "ownership"] if c in merged.columns),
        None,
    )

    # Identify stddev column
    stddev_col = next(
        (
            c
            for c in [
                "fpts_sim_uncond_std",
                "sim_dk_fpts_std_uncond",
                "dk_fpts_std_uncond",
                "fpts_sim_cond_std",
                "sim_dk_fpts_std",
                "stddev",
                "fpts_std",
            ]
            if c in merged.columns
        ),
        None,
    )
    
    # Identify p90 column for upside projection
    p90_col = next(
        (
            c
            for c in [
                "fpts_sim_uncond_p90",
                "sim_dk_fpts_p90_uncond",
                "dk_fpts_p90_uncond",
                "fpts_sim_cond_p90",
                "sim_dk_fpts_p90",
                "dk_fpts_p90",
                "fpts_p90",
            ]
            if c in merged.columns
        ),
        None,
    )
    
    # Game info columns (prefer _sal suffix from merge)
    matchup_col = "game_matchup_sal" if "game_matchup_sal" in merged.columns else "game_matchup"
    start_col = "game_start_utc_sal" if "game_start_utc_sal" in merged.columns else "game_start_utc"
    status_col = "status_sal" if "status_sal" in merged.columns else "status"
    disabled_col = "is_disabled_sal" if "is_disabled_sal" in merged.columns else "is_disabled"

    for _, row in merged.iterrows():
        # Get player_id (prefer projection's player_id, fall back to dk_player_id)
        player_id = row.get("player_id")
        if player_id is None or pd.isna(player_id):
            player_id = row.get(dk_player_id_col)

        # Get positions (from salaries - prefer _sal suffix if present from merge)
        positions_raw = row.get(positions_col)
        if positions_raw is None:
            positions_raw = []
        
        if isinstance(positions_raw, np.ndarray):
            positions_raw = positions_raw.tolist()
        
        if isinstance(positions_raw, str):
            positions = normalize_positions(positions_raw)
        elif hasattr(positions_raw, "__iter__"):
            positions = normalize_positions(list(positions_raw))
        else:
            positions = []

        if not positions:
            logger.warning("Player %s has no positions, skipping", player_id)
            continue

        # Get salary
        salary = row.get(salary_col)
        if pd.isna(salary) or salary <= 0:
            logger.warning("Player %s has invalid salary %s, skipping", player_id, salary)
            continue

        # When overrides applied, use effective values; otherwise use model values
        if use_user_overrides and "effective_fpts" in merged.columns:
            proj = row.get("effective_fpts")
            model_proj = row.get("model_fpts", row.get(proj_col))
            model_minutes = row.get("model_minutes", row.get(minutes_col, 0) if minutes_col else 0)
            model_own = row.get("model_own", row.get(own_col, 0) if own_col else 0)
            effective_minutes = row.get("effective_minutes", model_minutes)
            effective_own = row.get("effective_own", model_own)
            has_override = bool(row.get("has_override", False))
            used_fppm_fallback = bool(row.get("used_fppm_fallback", False))
            fppm = row.get("fppm", 1.0)
        else:
            proj = row.get(proj_col)
            model_proj = proj
            model_minutes = row.get(minutes_col, 0) if minutes_col else 0
            model_own = row.get(own_col, 0) if own_col else 0
            effective_minutes = model_minutes
            effective_own = model_own
            has_override = False
            used_fppm_fallback = False
            fppm = float(proj / model_minutes) if model_minutes and model_minutes > 0 else 1.0

            # Back-compat guardrail: if we only have a conditional-on-playing mean, convert to an
            # unconditional (DNP=0) decision metric by multiplying by the best available play prob.
            if proj_col in {"fpts_sim_cond_mean", "sim_dk_fpts_mean", "dk_fpts_mean"}:
                try:
                    p_play = row.get("p_play_eff", row.get("minutes_sim_p_active", row.get("play_prob", 1.0)))
                    p_play_f = float(p_play)
                    if 0.0 <= p_play_f <= 1.0 and proj is not None and not pd.isna(proj):
                        proj = float(proj) * p_play_f
                except Exception:
                    pass
        
        if pd.isna(proj) or float(proj) <= 0:
            if not allow_zero_projections:
                logger.debug("Player %s has no projection, skipping", player_id)
                continue
            proj = 0.0

        dk_player_id = row.get(dk_player_id_col)
        dk_id = "" if dk_player_id is None or pd.isna(dk_player_id) else str(int(dk_player_id))

        player = {
            "player_id": str(player_id),
            "name": (row.get(sal_name_col) if pd.notna(row.get(sal_name_col)) else None)
            or (row.get(proj_name_col) if pd.notna(row.get(proj_name_col)) else None)
            or str(player_id),
            "team": (row.get(sal_team_col) if pd.notna(row.get(sal_team_col)) else None)
            or (row.get(proj_team_col) if pd.notna(row.get(proj_team_col)) else None)
            or "UNK",
            "positions": positions,
            "salary": int(salary),
            "proj": float(proj),
            "dk_id": dk_id,
        }

        status_val = row.get(status_col) if status_col in row.index else None
        disabled_val = row.get(disabled_col) if disabled_col in row.index else None
        status_out = _is_out_status(status_val)
        disabled = _coerce_bool(disabled_val, default=False)
        # Legacy overrides may publish explicit activity fields.
        row_is_out = _coerce_bool(row.get("is_out") if "is_out" in row.index else None, default=False)
        row_is_active = _coerce_bool(row.get("is_active") if "is_active" in row.index else True, default=True)
        is_out = bool(row_is_out or status_out)
        is_active = bool(row_is_active and (not disabled) and (not is_out))
        player["is_out"] = is_out
        player["is_active"] = is_active

        # Optional fields
        if own_col and pd.notna(row.get(own_col)):
            player["own_proj"] = float(effective_own if use_user_overrides else row[own_col])
        if stddev_col and pd.notna(row.get(stddev_col)):
            player["stddev"] = float(row[stddev_col])
        if p90_col and pd.notna(row.get(p90_col)):
            player["p90"] = float(row[p90_col])
        
        # Game info
        if matchup_col in row and pd.notna(row.get(matchup_col)):
            player["game_matchup"] = str(row[matchup_col])
        if start_col in row and pd.notna(row.get(start_col)):
            game_start = row[start_col]
            if hasattr(game_start, "isoformat"):
                player["game_start_utc"] = game_start.isoformat()
            else:
                player["game_start_utc"] = str(game_start)
        
        # Always expose minutes and fppm for UI display
        if model_minutes and pd.notna(model_minutes) and float(model_minutes) > 0:
            player["model_minutes"] = float(model_minutes)
        if pd.notna(fppm):
            player["fppm"] = float(fppm)

        # Add override-related fields when using user overrides
        if use_user_overrides:
            player["model_proj"] = float(model_proj) if pd.notna(model_proj) else 0.0
            player["model_own"] = float(model_own) if pd.notna(model_own) else 0.0
            player["effective_proj"] = float(proj)
            player["effective_minutes"] = float(effective_minutes) if pd.notna(effective_minutes) else 0.0
            player["effective_own"] = float(effective_own) if pd.notna(effective_own) else 0.0
            player["has_override"] = has_override
            player["used_fppm_fallback"] = used_fppm_fallback
            player["override_minutes"] = (
                float(row.get("override_minutes")) if pd.notna(row.get("override_minutes")) else None
            )
            player["override_fpts"] = (
                float(row.get("override_fpts")) if pd.notna(row.get("override_fpts")) else None
            )
            player["override_own"] = (
                float(row.get("override_own")) if pd.notna(row.get("override_own")) else None
            )

        pool.append(player)

    logger.info("Built player pool with %d optimizer-ready players", len(pool))
    return pool


# ---------------------------------------------------------------------------
# Job Management
# ---------------------------------------------------------------------------


@dataclass
class OptimizerJob:
    """Tracks state of a QuickBuild job."""

    job_id: str
    status: str  # pending, running, completed, failed
    created_at: datetime
    game_date: str
    draft_group_id: int
    site: str
    config: Dict[str, Any]

    # Progress tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: int = 0
    target: int = 0

    # Results
    lineups: List[tuple] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    lineup_stats: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "game_date": self.game_date,
            "draft_group_id": self.draft_group_id,
            "site": self.site,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "target": self.target,
            "lineups_count": len(self.lineups),
            "lineup_stats_count": len(self.lineup_stats),
            "wall_time_sec": (
                (self.completed_at - self.started_at).total_seconds()
                if self.completed_at and self.started_at
                else None
            ),
            "error": self.error,
        }


class JobStore:
    """Thread-safe in-memory job store."""

    def __init__(self, max_jobs: int = 100):
        self._jobs: Dict[str, OptimizerJob] = {}
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    def create(
        self,
        game_date: str,
        draft_group_id: int,
        site: str,
        config: Dict[str, Any],
        target: int,
    ) -> OptimizerJob:
        job = OptimizerJob(
            job_id=str(uuid.uuid4()),
            status="pending",
            created_at=datetime.utcnow(),
            game_date=game_date,
            draft_group_id=draft_group_id,
            site=site,
            config=config,
            target=target,
        )
        with self._lock:
            # Evict old jobs if at capacity
            if len(self._jobs) >= self._max_jobs:
                oldest = min(self._jobs.values(), key=lambda j: j.created_at)
                del self._jobs[oldest.job_id]
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[OptimizerJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs) -> Optional[OptimizerJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, v in kwargs.items():
                    if hasattr(job, k):
                        setattr(job, k, v)
            return job

    def list_jobs(self, limit: int = 20) -> List[OptimizerJob]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return jobs[:limit]


# Global job store
_job_store = JobStore()


def get_job_store() -> JobStore:
    return _job_store


# ---------------------------------------------------------------------------
# QuickBuild Execution
# ---------------------------------------------------------------------------


def _build_qb_config(config: Dict[str, Any], defaults: Dict[str, Any]) -> QuickBuildConfig:
    """Build QuickBuildConfig from request config merged with defaults."""
    pool_defaults = defaults.get("pool", {})
    solver_defaults = defaults.get("solver", {})

    return QuickBuildConfig(
        builds=config.get("builds", pool_defaults.get("builds", 4)),
        per_build=config.get("per_build", pool_defaults.get("per_build", 6000)),
        max_pool=config.get("max_pool", pool_defaults.get("max_pool", 20000)),
        min_uniq=config.get("min_uniq", pool_defaults.get("min_uniq", 1)),
        max_exposure_pct=config.get("max_exposure_pct", pool_defaults.get("max_exposure_pct")),
        jitter=config.get("jitter", pool_defaults.get("jitter", 5e-4)),
        near_dup_jaccard=config.get("near_dup_jaccard", pool_defaults.get("near_dup_jaccard", 0.0)),
        enum_enable=config.get("enum_enable", pool_defaults.get("enum_enable", True)),
        enum_time=config.get("enum_time", pool_defaults.get("enum_time", 20.0)),
        enum_warm_time=config.get("enum_warm_time", pool_defaults.get("enum_warm_time", 5.0)),
        timeout=config.get("timeout", solver_defaults.get("timeout", 0.6)),
        threads=config.get("threads", solver_defaults.get("threads", 1)),
        nogood_rate=config.get("nogood_rate", solver_defaults.get("nogood_rate", 20)),
        lineup_size=8 if config.get("site", "dk") == "dk" else 9,
    )


def _build_constraints(
    config: Dict[str, Any],
    site: str,
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """Build constraints dict for QuickBuild."""
    constraints_defaults = defaults.get("constraints", {})
    site_defaults = constraints_defaults.get(site, {})
    ownership_defaults = defaults.get("ownership_penalty", {})

    constraints = {
        "min_salary": config.get("min_salary", site_defaults.get("min_salary")),
        "max_salary": config.get("max_salary", site_defaults.get("max_salary")),
        "global_team_limit": config.get("global_team_limit", site_defaults.get("global_team_limit", 4)),
        "team_limits": config.get("team_limits", {}),
        "lock_ids": config.get("lock_ids", []),
        "ban_ids": config.get("ban_ids", []),
        "unique_players": config.get("unique_players", 1),
        "N_lineups": 1,  # QuickBuild handles this differently
    }
    max_offoptimal_default = constraints_defaults.get("max_offoptimal_pct")
    if config.get("max_offoptimal_pct", max_offoptimal_default) is not None:
        constraints["max_offoptimal_pct"] = config.get(
            "max_offoptimal_pct", max_offoptimal_default
        )

    # Ownership penalty
    if config.get("ownership_penalty_enabled", ownership_defaults.get("enabled", False)):
        constraints["ownership_penalty"] = {
            "enabled": True,
            "mode": config.get("ownership_mode", ownership_defaults.get("mode", "by_percent")),
            "weight_lambda": config.get("ownership_lambda", ownership_defaults.get("weight_lambda", 1.0)),
            "curve_type": config.get("ownership_curve", ownership_defaults.get("curve_type", "sigmoid")),
            "pivot_p0": ownership_defaults.get("pivot_p0", 0.20),
            "curve_alpha": ownership_defaults.get("curve_alpha", 2.0),
            "clamp_min": ownership_defaults.get("clamp_min", 0.01),
            "clamp_max": ownership_defaults.get("clamp_max", 0.80),
            "shrink_gamma": ownership_defaults.get("shrink_gamma", 1.0),
        }

    # Randomness
    if config.get("randomness_pct"):
        constraints["randomness_pct"] = config["randomness_pct"]

    return constraints


def _build_late_swap_config(
    config: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Optional[LateSwapBonusConfig]:
    """Build late swap bonus config from request or defaults."""
    late_swap_defaults = defaults.get("late_swap_bonus", {})

    enabled = config.get(
        "late_swap_enabled",
        late_swap_defaults.get("enabled", False),
    )
    if not enabled:
        return None

    return LateSwapBonusConfig(
        enabled=True,
        bonus_per_hour=float(
            config.get("late_swap_bonus_per_hour", late_swap_defaults.get("bonus_per_hour", 0.2))
        ),
        max_bonus=float(
            config.get("late_swap_max_bonus", late_swap_defaults.get("max_bonus", 1.5))
        ),
    )


def run_quick_build(
    job: OptimizerJob,
    player_pool: List[Dict[str, Any]],
    on_progress: Optional[Callable[[int], None]] = None,
) -> QuickBuildResult:
    """Execute QuickBuild and update job state.

    This runs in a background thread.
    """
    store = get_job_store()
    defaults = load_optimizer_config()

    try:
        store.update(job.job_id, status="running", started_at=datetime.utcnow())

        qb_cfg = _build_qb_config(job.config, defaults)
        constraints = _build_constraints(job.config, job.site, defaults)

        # Set late swap bonus config for the main process
        # (Note: multiprocess workers may not inherit this global state)
        late_swap_cfg = _build_late_swap_config(job.config, defaults)
        set_active_late_swap_bonus(late_swap_cfg)

        logger.info(
            "Starting QuickBuild job %s: max_pool=%d, builds=%d, late_swap=%s, world_sample=%s",
            job.job_id,
            qb_cfg.max_pool,
            qb_cfg.worker_count,
            late_swap_cfg.bonus_per_hour if late_swap_cfg else "off",
            "on" if job.config.get("world_sample_enabled") else "off",
        )

        # Build WorldSampleConfig if enabled
        world_sample_cfg = None
        if job.config.get("world_sample_enabled", False):
            try:
                from projections.contest_sim.contest_sim_service import load_worlds_matrix
                worlds_matrix, player_index = load_worlds_matrix(
                    job.game_date,
                    get_data_root(),
                    worlds_source=str(job.config.get("worlds_source") or "gtv2"),
                )
                # Build mean projections fallback from player pool
                mean_projections = {str(p.get("player_id")): float(p.get("proj", 0)) for p in player_pool}
                world_sample_cfg = WorldSampleConfig(
                    enabled=True,
                    seed=None,  # Random seed for diversity
                    with_replacement=True,
                    worlds_matrix=worlds_matrix,
                    player_index=player_index,
                    mean_projections=mean_projections,
                )
                logger.info(
                    "WorldSample configured: n_worlds=%d, n_players=%d",
                    worlds_matrix.shape[0],
                    worlds_matrix.shape[1],
                )
            except Exception as exc:
                logger.warning("Failed to load worlds_matrix for world sampling: %s", exc)
                world_sample_cfg = None

        try:
            result = quick_build_pool(
                slate=player_pool,
                site=job.site,
                constraints=constraints,
                qb_cfg=qb_cfg,
                run_id=job.job_id[:8],
                world_sample=world_sample_cfg,
            )
        finally:
            # Clear global state after build
            set_active_late_swap_bonus(None)

        lineup_stats: List[Dict[str, Any]] = []
        try:
            data_root = get_data_root()
            worlds_root = data_root / "artifacts" / "sim_v2" / "worlds_fpts_v2"
            run_id = None
            if isinstance(job.config, dict):
                run_id = job.config.get("run_id")
            sim_run_id: str | None = None
            if run_id:
                try:
                    from projections.projections_bundle import load_unified_projections_df

                    bundle = load_unified_projections_df(job.game_date, run_id=str(run_id), data_root=data_root)
                    if "sim_run_id" in bundle.df.columns:
                        vals = bundle.df["sim_run_id"].dropna().astype(str).unique().tolist()
                        sim_run_id = vals[0] if len(vals) == 1 else None
                except Exception:
                    sim_run_id = None

            def _resolve_worlds_dir(base_root: Path, game_date: str, run_id: str | None) -> Path | None:
                import json

                candidates = [
                    base_root / f"game_date={game_date}",
                    base_root / f"date={game_date}",
                    base_root / game_date,
                ]
                for base in candidates:
                    if not base.exists():
                        continue
                    if run_id:
                        candidate = base / f"run={run_id}"
                        if candidate.exists():
                            return candidate

                    pointer = base / "latest_run.json"
                    if pointer.exists():
                        try:
                            payload = json.loads(pointer.read_text(encoding="utf-8"))
                            latest = payload.get("run_id")
                        except Exception:
                            latest = None
                        if latest:
                            candidate = base / f"run={latest}"
                            if candidate.exists():
                                return candidate

                    run_dirs = sorted(
                        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("run=")],
                        reverse=True,
                    )
                    if run_dirs:
                        if run_id:
                            logger.warning(
                                "sim_v2 worlds run_id=%s not found for %s; using %s",
                                run_id,
                                game_date,
                                run_dirs[0].name,
                            )
                        return run_dirs[0]

                    if base.is_dir():
                        if run_id:
                            logger.warning(
                                "sim_v2 worlds run_id=%s not found for %s; using base dir %s",
                                run_id,
                                game_date,
                                base,
                            )
                        return base
                return None

            worlds_dir = _resolve_worlds_dir(worlds_root, job.game_date, sim_run_id or run_id)
            if worlds_dir is None:
                raise FileNotFoundError(
                    f"sim_v2 worlds directory not found for {job.game_date} under {worlds_root}"
                )

            player_ids = sorted(
                {
                    int(str(pid))
                    for lu in result.lineups
                    for pid in lu
                    if str(pid).strip() and str(pid).lower() != "nan"
                }
            )
            if player_ids:
                _, world_player_ids, fpts_by_world = load_world_fpts_matrix(
                    worlds_dir=worlds_dir, player_ids=player_ids
                )
                stats_objs = compute_lineup_distribution_stats(
                    lineups=result.lineups,
                    world_player_ids=world_player_ids,
                    fpts_by_world=fpts_by_world,
                )
                lineup_stats = [s.to_dict() for s in stats_objs]
        except Exception as exc:
            logger.warning("Failed to compute lineup sim percentiles: %s", exc)

        store.update(
            job.job_id,
            status="completed",
            completed_at=datetime.utcnow(),
            lineups=result.lineups,
            stats=result.stats.to_dict(),
            lineup_stats=lineup_stats,
            progress=len(result.lineups),
        )

        # Auto-save build to disk
        try:
            save_build(job, result.lineups, result.stats.to_dict(), lineup_stats=lineup_stats)
        except Exception as save_exc:
            logger.warning("Failed to save build %s to disk: %s", job.job_id, save_exc)

        logger.info(
            "QuickBuild job %s completed: %d lineups in %.1fs",
            job.job_id,
            len(result.lineups),
            result.stats.wall_time_s,
        )

        return result

    except Exception as exc:
        logger.exception("QuickBuild job %s failed: %s", job.job_id, exc)
        store.update(
            job.job_id,
            status="failed",
            completed_at=datetime.utcnow(),
            error=str(exc),
        )
        raise


def get_slates_for_date(game_date: str, slate_type: str = "all") -> List[Dict[str, Any]]:
    """Get available draft groups for a date.

    First tries the live DK API, then falls back to disk-based discovery
    from scraped gold salaries if API fails or returns empty (e.g., after games lock).
    """
    api_slates: List[Dict[str, Any]] = []

    # Try live API first
    try:
        df = list_draft_groups_for_date(game_date, slate_type=slate_type)  # type: ignore
        if not df.empty:
            api_slates = df.to_dict(orient="records")
    except Exception as exc:
        logger.warning("Failed to fetch live slates for %s: %s", game_date, exc)

    # Merge API slates with disk slates to include in-progress slates for late swap
    disk_slates = _discover_slates_from_disk(game_date, slate_type)

    # Build set of draft_group_ids from API
    api_dg_ids = {s["draft_group_id"] for s in api_slates}

    # Add disk slates that aren't in API (in-progress slates)
    for ds in disk_slates:
        if ds["draft_group_id"] not in api_dg_ids:
            api_slates.append(ds)

    return sorted(api_slates, key=lambda s: s["draft_group_id"])


def _discover_slates_from_disk(game_date: str, slate_type: str = "all") -> List[Dict[str, Any]]:
    """Discover available slates from gold dk_salaries directory."""
    root = get_data_root()
    salaries_base = root / "gold" / "dk_salaries" / "site=dk" / f"game_date={game_date}"
    
    if not salaries_base.exists():
        logger.debug("No gold salaries directory for %s", game_date)
        return []
    
    slates: List[Dict[str, Any]] = []
    
    for dg_dir in salaries_base.iterdir():
        if not dg_dir.is_dir() or not dg_dir.name.startswith("draft_group_id="):
            continue
        
        try:
            dg_id = int(dg_dir.name.split("=")[1])
        except (ValueError, IndexError):
            continue
        
        salaries_file = dg_dir / "salaries.parquet"
        if not salaries_file.exists():
            continue
        
        # Try to infer slate type and get games from the bronze draftables
        inferred_type = "main"  # default
        games: List[Dict[str, Any]] = []
        earliest_start = None
        latest_start = None
        example_name = f"Draft Group {dg_id}"
        
        game_info = _load_game_info_from_draftables(dg_id, root)
        if game_info:
            for comp_id, info in game_info.items():
                game_entry = {"matchup": info["matchup"]}
                start_time = info.get("start_time_utc")
                if start_time:
                    game_entry["start_time"] = start_time.isoformat()
                    if earliest_start is None or start_time < earliest_start:
                        earliest_start = start_time
                    if latest_start is None or start_time > latest_start:
                        latest_start = start_time
                games.append(game_entry)
            # Sort games by start time
            games.sort(key=lambda g: g.get("start_time", ""))
        
        bronze_path = root / "bronze" / "dk" / "draftables" / f"draftables_raw_{dg_id}.json"
        if bronze_path.exists():
            try:
                import json
                with open(bronze_path) as f:
                    payload = json.load(f)
                # Try to get contest name from draftables
                contests = payload.get("Contests", [])
                if contests:
                    name = contests[0].get("n", contests[0].get("ContestName", ""))
                    if name:
                        inferred_type = _infer_slate_type(name)
                        example_name = name
            except Exception:
                pass
        
        slate = {
            "game_date": game_date,
            "slate_type": inferred_type,
            "draft_group_id": dg_id,
            "n_contests": 0,  # Unknown from disk
            "earliest_start": earliest_start.isoformat() if earliest_start else None,
            "latest_start": latest_start.isoformat() if latest_start else None,
            "example_contest_name": example_name,
            "games": games,
        }
        
        if slate_type == "all" or inferred_type == slate_type:
            slates.append(slate)
    
    logger.info("Discovered %d slates from disk for %s", len(slates), game_date)
    return sorted(slates, key=lambda s: s["draft_group_id"])


def _infer_slate_type(name: str) -> str:
    """Infer slate type from contest name."""
    name_lower = name.lower()
    if "turbo" in name_lower:
        return "turbo"
    if "late" in name_lower or "night" in name_lower:
        return "night"
    if "early" in name_lower:
        return "early"
    if "showdown" in name_lower or "single game" in name_lower:
        return "showdown"
    return "main"


# ─────────────────────────────────────────────────────────────────────────────
# Build Persistence - Save to projections-data/builds/optimizer
# ─────────────────────────────────────────────────────────────────────────────

def _builds_dir() -> Path:
    """Get the builds directory under projections-data."""
    return get_data_root() / "builds" / "optimizer"


def save_build(
    job: OptimizerJob,
    lineups: List[List[str]],
    stats: Dict[str, Any],
    *,
    lineup_stats: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Save a completed build to disk.
    
    Saves to: projections-data/builds/optimizer/{game_date}/{job_id}.json
    """
    import json
    
    builds_root = _builds_dir() / job.game_date
    builds_root.mkdir(parents=True, exist_ok=True)
    
    build_file = builds_root / f"{job.job_id}.json"
    
    # Get config from job.config dict
    cfg = job.config or {}
    
    build_data = {
        "job_id": job.job_id,
        "game_date": job.game_date,
        "draft_group_id": job.draft_group_id,
        "site": job.site,
        "created_at": job.created_at.isoformat(),
        "completed_at": datetime.utcnow().isoformat(),
        "lineups_count": len(lineups),
        "config": cfg,
        "stats": stats,
        "lineups": [
            {
                **({"lineup_id": i, "player_ids": lu}),
                **(lineup_stats[i] if lineup_stats and i < len(lineup_stats) else {}),
            }
            for i, lu in enumerate(lineups)
        ],
    }
    
    with open(build_file, "w") as f:
        json.dump(build_data, f, indent=2)
    
    logger.info("Saved build %s to %s (%d lineups)", job.job_id, build_file, len(lineups))
    return build_file


def list_saved_builds(game_date: str, draft_group_id: int | None = None) -> List[Dict[str, Any]]:
    """List saved builds for a game date.
    
    Returns summary info (no lineups) for each build.
    """
    import json
    
    builds_root = _builds_dir() / game_date
    if not builds_root.exists():
        return []
    
    builds = []
    for build_file in sorted(builds_root.glob("*.json"), reverse=True):
        try:
            with open(build_file, "r") as f:
                data = json.load(f)
            
            # Filter by draft_group_id if specified
            if draft_group_id is not None and data.get("draft_group_id") != draft_group_id:
                continue
            
            # Return summary without full lineups
            builds.append({
                "job_id": data["job_id"],
                "game_date": data["game_date"],
                "draft_group_id": data["draft_group_id"],
                "site": data["site"],
                "created_at": data["created_at"],
                "completed_at": data.get("completed_at"),
                "lineups_count": data["lineups_count"],
                "config": data.get("config", {}),
                "stats": data.get("stats", {}),
            })
        except Exception as e:
            logger.warning("Failed to read build file %s: %s", build_file, e)
            continue
    
    return builds


def load_saved_build(game_date: str, job_id: str) -> Dict[str, Any] | None:
    """Load a saved build including lineups."""
    import json
    
    build_file = _builds_dir() / game_date / f"{job_id}.json"
    if not build_file.exists():
        return None
    
    with open(build_file, "r") as f:
        return json.load(f)


def delete_saved_build(game_date: str, job_id: str) -> bool:
    """Delete a saved build."""
    build_file = _builds_dir() / game_date / f"{job_id}.json"
    if build_file.exists():
        build_file.unlink()
        logger.info("Deleted build %s", build_file)
        return True
    return False


def save_custom_build(
    game_date: str,
    draft_group_id: int,
    site: str,
    lineups: List[Dict[str, Any]],
    name: str | None = None,
) -> Dict[str, Any]:
    """Save a custom/merged build to disk.
    
    Unlike save_build(), this doesn't require an OptimizerJob.
    Used for merged builds created in the UI.
    
    Returns the saved build summary.
    """
    import json
    
    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    
    builds_root = _builds_dir() / game_date
    builds_root.mkdir(parents=True, exist_ok=True)
    
    build_file = builds_root / f"{job_id}.json"
    
    # Build config to indicate this is a merged build
    config = {
        "merged": True,
        "name": name or f"Merged Build ({len(lineups)} lineups)",
        "source_builds": [],  # Could be populated if we tracked source job IDs
    }
    
    build_data = {
        "job_id": job_id,
        "game_date": game_date,
        "draft_group_id": draft_group_id,
        "site": site,
        "created_at": now,
        "completed_at": now,
        "lineups_count": len(lineups),
        "config": config,
        "stats": {},
        "lineups": [
            {
                "lineup_id": lu.get("lineup_id", i),
                "player_ids": lu.get("player_ids", []),
                **(
                    {k: v for k, v in lu.items() if k not in ("lineup_id", "player_ids")}
                ),
            }
            for i, lu in enumerate(lineups)
        ],
    }
    
    with open(build_file, "w") as f:
        json.dump(build_data, f, indent=2)
    
    logger.info(
        "Saved custom build %s to %s (%d lineups, name=%s)",
        job_id,
        build_file,
        len(lineups),
        name,
    )
    
    # Return summary (without full lineups)
    return {
        "job_id": job_id,
        "game_date": game_date,
        "draft_group_id": draft_group_id,
        "site": site,
        "created_at": now,
        "completed_at": now,
        "lineups_count": len(lineups),
        "config": config,
        "stats": {},
    }
