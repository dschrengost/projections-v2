"""QuickBuild optimizer service layer.

This module provides the service functions that:
1. Load and merge player pools from projections + DK salaries
2. Execute QuickBuild jobs with progress tracking
3. Manage job lifecycle (create, poll, retrieve results)
"""

from __future__ import annotations

import logging
import os
import re
import threading
import uuid
import json
from dataclasses import dataclass, field
import datetime as dt
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import numpy as np
import yaml

from projections.dk.normalize import (
    draftables_json_to_df,
    normalize_draftables_to_salaries,
    write_salaries_gold,
)
from projections.dk.salaries_schema import dk_salaries_gold_path, normalize_positions
from projections.dk.slates import list_draft_groups_for_date
from projections.fd.normalize import normalize_fd_players_to_salaries, players_json_to_df
from projections.fd.slates import list_fixture_lists_for_date
from projections.fpts_v2.scoring import compute_dk_fpts, compute_fd_fpts
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
)
from projections.projections_bundle import add_canonical_projection_fields, resolve_unified_projections_run
from projections.optimizer.objective import (
    set_active_late_swap_bonus,
    LateSwapBonusConfig,
)
from projections.api.slate_analytics_service import load_or_compute_slate_player_analytics

logger = logging.getLogger(__name__)
SUPPORTED_SITES = {"dk", "fd"}


def _normalize_site(site: str | None) -> str:
    value = str(site or "dk").strip().lower()
    if value not in SUPPORTED_SITES:
        raise ValueError(f"Unsupported site '{site}'. Expected one of: {sorted(SUPPORTED_SITES)}")
    return value


def _canonicalize_player_id(raw: object) -> str | None:
    """Normalize player identifiers to stable string form (e.g. 1627742.0 -> '1627742')."""
    if raw is None or pd.isna(raw):
        return None

    if isinstance(raw, (int, np.integer)):
        return str(int(raw))

    if isinstance(raw, (float, np.floating)):
        value = float(raw)
        if np.isnan(value):
            return None
        if value.is_integer():
            return str(int(value))
        return format(value, "f").rstrip("0").rstrip(".")

    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    if re.fullmatch(r"[+-]?\d+", text):
        return str(int(text))
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        try:
            return str(int(float(text)))
        except Exception:
            return text

    return text

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
    *,
    site: str = "dk",
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

    site_norm = _normalize_site(site)
    scoring_fn = compute_fd_fpts if site_norm == "fd" else compute_dk_fpts
    merged["fpts_mean"] = scoring_fn(scoring_frame).astype(float)

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


def _fpts_presence_candidates(site: str) -> list[str]:
    site_norm = _normalize_site(site)
    base = [
        "fpts_sim_uncond_mean",
        "fpts_sim_cond_mean",
        "proj_fpts",
        "fpts_mean",
    ]
    if site_norm == "fd":
        return [
            "sim_fd_fpts_mean_uncond",
            "fd_fpts_mean_uncond",
            "sim_fd_fpts_mean",
            "fd_fpts_mean",
            *base,
        ]
    return [
        "sim_dk_fpts_mean_uncond",
        "dk_fpts_mean_uncond",
        "sim_dk_fpts_mean",
        "dk_fpts_mean",
        *base,
    ]


def load_projections_for_date(
    game_date: str,
    run_id: Optional[str] = None,
    data_root: Optional[Path] = None,
    *,
    site: str = "dk",
) -> pd.DataFrame:
    """Load projections from unified projections artifact or gold layer.

    Also merges sim_v2 FPTS projections if available.

    Returns DataFrame with columns:
        player_id, player_name, team_tricode, sim_dk_fpts_mean, pred_own_pct, etc.
    """
    site_norm = _normalize_site(site)
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
    fpts_cols = _fpts_presence_candidates(site_norm)
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
            df = _attach_rates_mean_fpts(df, rates_df, site=site_norm)
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


def _load_or_build_dk_salaries_from_bronze(
    *,
    root: Path,
    game_date: str,
    draft_group_id: int,
) -> Path | None:
    bronze_path = root / "bronze" / "dk" / "draftables" / f"draftables_raw_{draft_group_id}.json"
    if not bronze_path.exists():
        return None

    try:
        payload = json.loads(bronze_path.read_text(encoding="utf-8"))
        raw_df = draftables_json_to_df(payload, draft_group_id=draft_group_id)
        salaries_df = normalize_draftables_to_salaries(
            root=root,
            site="dk",
            game_date=game_date,
            draft_group_id=draft_group_id,
            df=raw_df,
        )
        written = write_salaries_gold(
            root=root,
            site="dk",
            game_date=game_date,
            draft_group_id=draft_group_id,
            salaries_df=salaries_df,
        )
    except Exception as exc:
        logger.warning(
            "Failed to synthesize DK salaries from bronze draftables for %s/dg=%d: %s",
            game_date,
            draft_group_id,
            exc,
        )
        return None

    logger.info(
        "Synthesized DK salaries from bronze draftables for %s/dg=%d -> %s",
        game_date,
        draft_group_id,
        written,
    )
    return written


def _fd_bronze_slate_dir(root: Path, game_date: str, draft_group_id: int | str) -> Path:
    return (
        root
        / "bronze"
        / "fd"
        / "fixture_lists"
        / f"game_date={game_date}"
        / f"draft_group_id={draft_group_id}"
    )


def _load_or_build_fd_salaries_from_bronze(
    *,
    root: Path,
    game_date: str,
    draft_group_id: int | str,
) -> Path | None:
    slate_dir = _fd_bronze_slate_dir(root, game_date, draft_group_id)
    players_path = slate_dir / "players.json"
    if not players_path.exists():
        return None

    detail_path = slate_dir / "detail.json"
    contests_path = slate_dir / "contests.json"

    try:
        players_payload = json.loads(players_path.read_text(encoding="utf-8"))
        detail_payload = (
            json.loads(detail_path.read_text(encoding="utf-8")) if detail_path.exists() else None
        )
        contests_payload = (
            json.loads(contests_path.read_text(encoding="utf-8")) if contests_path.exists() else None
        )

        raw_df = players_json_to_df(
            players_payload,
            fixture_list_id=draft_group_id,
            fixture_detail=detail_payload if isinstance(detail_payload, dict) else None,
            contests_payload=contests_payload if isinstance(contests_payload, dict) else None,
        )
        salaries_df = normalize_fd_players_to_salaries(
            root=root,
            site="fd",
            game_date=game_date,
            draft_group_id=draft_group_id,
            df=raw_df,
        )
        written = write_salaries_gold(
            root=root,
            site="fd",
            game_date=game_date,
            draft_group_id=draft_group_id,
            salaries_df=salaries_df,
        )
    except Exception as exc:
        logger.warning(
            "Failed to synthesize FD salaries from bronze fixture payloads for %s/dg=%s: %s",
            game_date,
            draft_group_id,
            exc,
        )
        return None

    logger.info(
        "Synthesized FD salaries from bronze payloads for %s/dg=%s -> %s",
        game_date,
        draft_group_id,
        written,
    )
    return written


def _safe_parse_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    if not text:
        return None

    # Try direct parse first.
    try:
        ts = pd.to_datetime(text, utc=True, errors="coerce")
        if pd.notna(ts):
            return ts.to_pydatetime()
    except Exception:
        pass

    # Common fallback: "MM/DD/YYYY HH:MMAM ET"
    cleaned = text.replace(" ET", "").replace("ET", "").strip()
    try:
        ts = datetime.strptime(cleaned, "%m/%d/%Y %I:%M%p")
        return ts
    except Exception:
        return None


def _coerce_competition_ids(raw: object) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for value in raw:
        try:
            out.append(int(value))
        except Exception:
            continue
    return sorted(set(out))


def _infer_game_matchup_from_text(raw: object) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    normalized = text.replace(" at ", "@").replace(" vs ", "@").replace("-", "@")
    token = normalized.split(" ", 1)[0].upper()
    if "@" in token and len(token.split("@")) == 2:
        away, home = token.split("@")
        if away and home:
            return f"{away}@{home}"
    return None


def _add_game_columns_from_available_data(
    *,
    df: pd.DataFrame,
    game_info: Dict[int, Dict[str, Any]] | None,
) -> tuple[pd.Series, pd.Series]:
    if game_info is None:
        game_info = {}

    if "raw_competition_ids" in df.columns and game_info:
        comp_ids = df["raw_competition_ids"].apply(_coerce_competition_ids)
        matchup = comp_ids.apply(
            lambda ids: game_info.get(ids[0], {}).get("matchup") if ids else None
        )
        start = comp_ids.apply(
            lambda ids: game_info.get(ids[0], {}).get("start_time_utc") if ids else None
        )
    else:
        matchup = pd.Series([None] * len(df), index=df.index, dtype="object")
        start = pd.Series([None] * len(df), index=df.index, dtype="object")

    for col in ("game_matchup", "matchup", "game"):
        if col in df.columns:
            parsed = df[col].apply(_infer_game_matchup_from_text)
            matchup = matchup.where(matchup.notna(), parsed)

    for col in ("game_start_utc", "start_time_utc", "start_time", "lock_time"):
        if col in df.columns:
            parsed = df[col].apply(_safe_parse_timestamp)
            start = start.where(start.notna(), parsed)

    if "game_info" in df.columns:
        parsed_start = df["game_info"].apply(
            lambda v: _safe_parse_timestamp(str(v).split(" ", 1)[1] if isinstance(v, str) and " " in v else None)
        )
        start = start.where(start.notna(), parsed_start)
        parsed_matchup = df["game_info"].apply(_infer_game_matchup_from_text)
        matchup = matchup.where(matchup.notna(), parsed_matchup)

    return matchup, start


def load_salaries_for_date(
    game_date: str,
    draft_group_id: int | str,
    site: str = "dk",
    data_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Load salaries from gold layer.

    Returns DataFrame with columns:
        dk_player_id, display_name, positions, salary, team_abbrev, status,
        game_matchup, game_start_utc
    """
    site_norm = _normalize_site(site)
    root = data_root or get_data_root()
    salaries_path = dk_salaries_gold_path(root, site_norm, game_date, draft_group_id)

    if not salaries_path.exists():
        if site_norm == "dk":
            try:
                dk_draft_group_id = int(draft_group_id)
            except Exception as exc:
                raise ValueError(f"DK draft_group_id must be an integer, got {draft_group_id!r}") from exc
            synthesized = _load_or_build_dk_salaries_from_bronze(
                root=root,
                game_date=game_date,
                draft_group_id=dk_draft_group_id,
            )
            if synthesized is not None and synthesized.exists():
                salaries_path = synthesized
        elif site_norm == "fd":
            synthesized = _load_or_build_fd_salaries_from_bronze(
                root=root,
                game_date=game_date,
                draft_group_id=draft_group_id,
            )
            if synthesized is not None and synthesized.exists():
                salaries_path = synthesized
        if not salaries_path.exists():
            raise FileNotFoundError(f"Salaries not found: {salaries_path}")

    df = pd.read_parquet(salaries_path)
    if site_norm == "dk":
        try:
            dk_draft_group_id = int(draft_group_id)
        except Exception as exc:
            raise ValueError(f"DK draft_group_id must be an integer, got {draft_group_id!r}") from exc
        game_info = _load_game_info_from_draftables(dk_draft_group_id, root)
    else:
        game_info = {}
    matchup, start = _add_game_columns_from_available_data(df=df, game_info=game_info)
    df["game_matchup"] = matchup
    df["game_start_utc"] = start

    logger.info(
        "Loaded salaries for %s site=%s draft_group=%s (%d players, %d games)",
        game_date,
        site_norm,
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


def _load_live_out_indicators(
    game_date: str,
    *,
    data_root: Path,
) -> tuple[set[int], set[tuple[str, str]]]:
    """Load latest out-like indicators from official injuries + Rotowire lineups."""
    official_out_player_ids: set[int] = set()
    rotowire_out_keys: set[tuple[str, str]] = set()

    injuries_base = data_root / "bronze" / "injuries_raw"
    injury_candidates = sorted(
        injuries_base.glob(f"season=*/date={game_date}/injuries.parquet")
    )
    if injury_candidates:
        injury_path = max(injury_candidates, key=lambda p: p.stat().st_mtime)
        try:
            injuries = pd.read_parquet(injury_path)
            if not injuries.empty and "status" in injuries.columns:
                injuries = injuries.copy()
                injuries["status_u"] = (
                    injuries["status"].astype(str).str.upper().str.strip()
                )
                if "status_raw" in injuries.columns:
                    injuries["status_raw_u"] = (
                        injuries["status_raw"].astype(str).str.upper().str.strip()
                    )
                else:
                    injuries["status_raw_u"] = ""
                if "as_of_ts" in injuries.columns:
                    injuries["_asof"] = pd.to_datetime(
                        injuries["as_of_ts"], errors="coerce", utc=True
                    )
                elif "ingested_ts" in injuries.columns:
                    injuries["_asof"] = pd.to_datetime(
                        injuries["ingested_ts"], errors="coerce", utc=True
                    )
                else:
                    injuries["_asof"] = pd.NaT
                injuries = injuries.sort_values("_asof")
                if "player_id" in injuries.columns:
                    injuries["player_id"] = pd.to_numeric(
                        injuries["player_id"], errors="coerce"
                    ).astype("Int64")
                    latest = injuries.dropna(subset=["player_id"]).drop_duplicates(
                        subset=["player_id"], keep="last"
                    )
                    out_like = latest["status_u"].isin(
                        {"OUT", "O", "D", "DOUBTFUL", "INACTIVE", "SUSPENDED"}
                    ) | latest["status_raw_u"].str.contains("DOUBT", na=False)
                    official_out_player_ids = set(
                        latest.loc[out_like, "player_id"].astype(int).tolist()
                    )
        except Exception as exc:
            logger.warning("Failed loading official injuries for %s: %s", game_date, exc)

    rotowire_path = (
        data_root
        / "silver"
        / "rotowire_lineups"
        / f"date={game_date}"
        / "lineups.parquet"
    )
    if rotowire_path.exists():
        try:
            rw = pd.read_parquet(rotowire_path)
            if not rw.empty and "player_name" in rw.columns:
                rw = rw.copy()
                role_u = (
                    rw["lineup_role"].astype(str).str.upper().str.strip()
                    if "lineup_role" in rw.columns
                    else pd.Series("", index=rw.index)
                )
                injury_u = (
                    rw["injury_status"].astype(str).str.upper().str.strip()
                    if "injury_status" in rw.columns
                    else pd.Series("", index=rw.index)
                )
                out_like = role_u.eq("OUT") | injury_u.isin(
                    {"OUT", "D", "DOUBT", "DOUBTFUL", "INACTIVE", "SUSPENDED"}
                )
                team_col = (
                    "team_abbreviation"
                    if "team_abbreviation" in rw.columns
                    else ("team" if "team" in rw.columns else None)
                )
                if team_col is not None:
                    rotowire_out_keys = set(
                        zip(
                            rw.loc[out_like, "player_name"].map(_normalize_name),
                            rw.loc[out_like, team_col].map(_normalize_team),
                        )
                    )
        except Exception as exc:
            logger.warning("Failed loading Rotowire lineups for %s: %s", game_date, exc)

    return official_out_player_ids, rotowire_out_keys


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
            base["_own_join_player_id"] = base["player_id"].map(_canonicalize_player_id)
            own["_own_join_player_id"] = own["player_id"].map(_canonicalize_player_id)
            join_cols = ["_own_join_player_id"]
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
        columns=[
            "_own_join_dk_player_id",
            "_own_join_name",
            "_own_join_team",
            "_own_join_player_id",
        ],
        errors="ignore",
    )


def _find_projection_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _extract_single_string_value(df: pd.DataFrame, column: str) -> str | None:
    if column not in df.columns:
        return None
    values = df[column].dropna().astype(str).unique().tolist()
    return values[0] if len(values) == 1 else None


def _projection_fpts_col(df: pd.DataFrame, site: str = "dk") -> Optional[str]:
    site_norm = _normalize_site(site)
    if site_norm == "fd":
        candidates = [
            "sim_fd_fpts_mean_uncond",
            "fd_fpts_mean_uncond",
            "fpts_sim_uncond_mean",
            "sim_fd_fpts_mean",
            "fd_fpts_mean",
            "fpts_sim_cond_mean",
            "proj_fpts",
            "fpts_mean",
            "proj",
        ]
    else:
        candidates = [
            "fpts_sim_uncond_mean",
            "sim_dk_fpts_mean_uncond",
            "dk_fpts_mean_uncond",
            "fpts_sim_cond_mean",
            "sim_dk_fpts_mean",
            "dk_fpts_mean",
            "proj_fpts",
            "fpts_mean",
            "proj",
        ]
    return _find_projection_col(df, candidates)


def _projection_minutes_col(df: pd.DataFrame) -> Optional[str]:
    return _find_projection_col(
        df,
        [
            "minutes_sim_uncond_mean",
            "minutes_sim_mean_uncond",
            "minutes_sim_uncond_p50",
            "minutes_sim_p50_uncond",
            "minutes_sim_cond_mean",
            "minutes_sim_mean",
            "minutes_sim_cond_p50",
            "minutes_sim_p50",
            "minutes_final",
            "minutes_p50_cond",
            "minutes_p50",
            "minutes",
            "minutes_pred",
        ],
    )


def _build_model_value_maps(
    projection_df: pd.DataFrame,
    *,
    site: str = "dk",
) -> tuple[Dict[str, float], Dict[str, float]]:
    player_ids = projection_df.get("player_id")
    if player_ids is None:
        return {}, {}

    base = projection_df.copy()
    base = base.loc[player_ids.notna()].copy()
    if base.empty:
        return {}, {}

    base["_canonical_player_id"] = base["player_id"].map(_canonicalize_player_id)
    base = base.loc[base["_canonical_player_id"].notna()].copy()
    if base.empty:
        return {}, {}
    fpts_col = _projection_fpts_col(base, site=site)
    minutes_col = _projection_minutes_col(base)

    model_fpts_by_player: Dict[str, float] = {}
    model_minutes_by_player: Dict[str, float] = {}
    if fpts_col:
        fpts_series = pd.to_numeric(base[fpts_col], errors="coerce").fillna(0.0)
        model_fpts_by_player = dict(zip(base["_canonical_player_id"], fpts_series.astype(float)))
    if minutes_col:
        minutes_series = pd.to_numeric(base[minutes_col], errors="coerce").fillna(0.0)
        model_minutes_by_player = dict(zip(base["_canonical_player_id"], minutes_series.astype(float)))
    return model_fpts_by_player, model_minutes_by_player


def _ensure_fd_projection_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _num(col: str) -> pd.Series | None:
        if col not in out.columns:
            return None
        return pd.to_numeric(out[col], errors="coerce")

    def _fill_if_missing(target: str, source: str) -> None:
        if source not in out.columns:
            return
        src = _num(source)
        if src is None:
            return
        if target in out.columns:
            dst = _num(target)
            out[target] = dst.where(dst.notna(), src)
        else:
            out[target] = src

    pts_un = _num("pts_mean_uncond")
    reb_un = _num("reb_mean_uncond")
    ast_un = _num("ast_mean_uncond")
    stl_un = _num("stl_mean_uncond")
    blk_un = _num("blk_mean_uncond")
    tov_un = _num("tov_mean_uncond")
    if all(v is not None for v in (pts_un, reb_un, ast_un, stl_un, blk_un, tov_un)):
        fd_un = (
            pts_un
            + 1.2 * reb_un
            + 1.5 * ast_un
            + 3.0 * stl_un
            + 3.0 * blk_un
            - 1.0 * tov_un
        )
        if "fd_fpts_mean_uncond" in out.columns:
            existing = _num("fd_fpts_mean_uncond")
            out["fd_fpts_mean_uncond"] = existing.where(existing.notna(), fd_un)
        else:
            out["fd_fpts_mean_uncond"] = fd_un

    pts_c = _num("pts_mean")
    reb_c = _num("reb_mean")
    ast_c = _num("ast_mean")
    stl_c = _num("stl_mean")
    blk_c = _num("blk_mean")
    tov_c = _num("tov_mean")
    if all(v is not None for v in (pts_c, reb_c, ast_c, stl_c, blk_c, tov_c)):
        fd_c = (
            pts_c
            + 1.2 * reb_c
            + 1.5 * ast_c
            + 3.0 * stl_c
            + 3.0 * blk_c
            - 1.0 * tov_c
        )
        if "fd_fpts_mean" in out.columns:
            existing = _num("fd_fpts_mean")
            out["fd_fpts_mean"] = existing.where(existing.notna(), fd_c)
        else:
            out["fd_fpts_mean"] = fd_c

    # Canonical fallback if stat means are unavailable.
    _fill_if_missing("fd_fpts_mean_uncond", "fpts_sim_uncond_mean")
    _fill_if_missing("fd_fpts_mean", "fpts_sim_cond_mean")
    _fill_if_missing("fd_fpts_std_uncond", "fpts_sim_uncond_std")
    _fill_if_missing("fd_fpts_std", "fpts_sim_cond_std")
    _fill_if_missing("fd_fpts_p90_uncond", "fpts_sim_uncond_p90")
    _fill_if_missing("fd_fpts_p90", "fpts_sim_cond_p90")

    _fill_if_missing("sim_fd_fpts_mean_uncond", "fd_fpts_mean_uncond")
    _fill_if_missing("sim_fd_fpts_mean", "fd_fpts_mean")
    _fill_if_missing("sim_fd_fpts_std_uncond", "fd_fpts_std_uncond")
    _fill_if_missing("sim_fd_fpts_std", "fd_fpts_std")
    _fill_if_missing("sim_fd_fpts_p90_uncond", "fd_fpts_p90_uncond")
    _fill_if_missing("sim_fd_fpts_p90", "fd_fpts_p90")

    return out


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
    include_slate_analytics: bool = False,
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
        use_user_overrides: Legacy flag name for strategy overrides
        ownership_mode: "raw" keeps model ownership; "renormalize" rebalances ownership
            when strategy overrides are enabled.
        include_unmatched_salaries: If True, keep salary rows even when projections don't match.
        allow_zero_projections: If True, include players with missing/zero projection (proj=0.0).
        exclude_inactive_players: Legacy flag; strategy overrides do not mark players out.

    Returns list of player dicts with required QuickBuild fields:
        player_id, name, team, positions, salary, proj, own_proj, stddev, dk_id,
        game_matchup, game_start_utc
        
        When use_user_overrides=True, also includes:
        model_proj, model_minutes, model_own, effective_proj, effective_minutes,
        effective_own, effective_stddev, effective_p90, has_override,
        used_fppm_fallback, is_active, fppm
    """
    site_norm = _normalize_site(site)
    root = data_root or get_data_root()

    # Load data sources
    proj_df = load_projections_for_date(game_date, run_id=run_id, data_root=root, site=site_norm)
    if site_norm == "fd":
        proj_df = _ensure_fd_projection_aliases(proj_df)
    sal_df = load_salaries_for_date(game_date, draft_group_id, site=site_norm, data_root=root)

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

    official_out_player_ids, rotowire_out_keys = _load_live_out_indicators(
        game_date,
        data_root=root,
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

    # Apply downstream strategy overrides if requested.
    slate_overrides = None
    if use_user_overrides:
        from projections.contest_sim.contest_sim_service import load_player_worlds

        from .strategy_overrides import (
            apply_strategy_overrides,
            apply_strategy_overrides_to_worlds,
            load_slate_strategy_overrides,
            summarize_worlds,
        )

        slate_overrides = load_slate_strategy_overrides(game_date, draft_group_id)
        adjusted_summaries = None
        if slate_overrides.overrides:
            try:
                from projections.ops.manual_availability import list_manual_overrides

                model_fpts_by_player, model_minutes_by_player = _build_model_value_maps(
                    proj_df,
                    site=site_norm,
                )
                sim_run_id = _extract_single_string_value(proj_df, "sim_run_id")
                player_worlds = load_player_worlds(
                    game_date,
                    root,
                    run_id=sim_run_id or run_id,
                    worlds_source="gtv2",
                )
                try:
                    _force_in_overrides = list_manual_overrides(
                        pd.Timestamp(game_date).date(),
                        data_root=root,
                        active_only=True,
                    )
                    force_active_ids: set[str] = {
                        str(r["player_id"])
                        for r in _force_in_overrides
                        if str(r.get("override_type", "")).lower() == "force_in"
                    }
                except Exception:
                    force_active_ids = set()
                adjusted_fpts, adjusted_minutes, world_diagnostics = apply_strategy_overrides_to_worlds(
                    fpts_matrix=player_worlds.fpts_matrix,
                    player_index=player_worlds.player_index,
                    overrides=slate_overrides,
                    minutes_matrix=player_worlds.minutes_matrix,
                    model_minutes_by_player=model_minutes_by_player,
                    model_fpts_by_player=model_fpts_by_player,
                    force_active_player_ids=force_active_ids or None,
                )
                adjusted_summaries = summarize_worlds(
                    fpts_matrix=adjusted_fpts,
                    player_index=player_worlds.player_index,
                    minutes_matrix=adjusted_minutes,
                )
                logger.info(
                    "Applied strategy overrides to worlds for %s/dg=%d: matched=%d minutes=%d fpts=%d",
                    game_date,
                    draft_group_id,
                    int(world_diagnostics.get("matched_override_count", 0)),
                    int(world_diagnostics.get("applied_minutes_delta_count", 0)),
                    int(world_diagnostics.get("applied_fpts_delta_count", 0)),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to apply strategy overrides to worlds for %s/dg=%d: %s",
                    game_date,
                    draft_group_id,
                    exc,
                )

        merged = apply_strategy_overrides(
            merged,
            slate_overrides,
            ownership_mode=ownership_mode,  # type: ignore[arg-type]
            adjusted_summaries=adjusted_summaries,
        )
        logger.info(
            "Applied %d strategy overrides for %s/dg=%d",
            len(slate_overrides.overrides),
            game_date,
            draft_group_id,
        )

    # Build player pool list
    pool: List[Dict[str, Any]] = []

    # Prefer salary-derived columns when merge created conflicts
    salary_col = "salary_sal" if "salary_sal" in merged.columns else "salary"
    positions_col = "positions_sal" if "positions_sal" in merged.columns else "positions"
    site_player_id_col = next(
        (
            col
            for col in [
                f"{site_norm}_player_id_sal",
                "site_player_id_sal",
                "dk_player_id_sal",
                "fd_player_id_sal",
                f"{site_norm}_player_id",
                "site_player_id",
                "dk_player_id",
                "fd_player_id",
            ]
            if col in merged.columns
        ),
        None,
    )
    dk_player_id_col = "dk_player_id_sal" if "dk_player_id_sal" in merged.columns else "dk_player_id"
    fd_player_id_col = "fd_player_id_sal" if "fd_player_id_sal" in merged.columns else "fd_player_id"

    # Identify projection column
    proj_col = _projection_fpts_col(merged, site=site_norm)
    if not proj_col:
        raise ValueError("No projection column found in merged data")
    
    # Identify minutes column
    minutes_col = _projection_minutes_col(merged)

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
                "sim_fd_fpts_std_uncond",
                "fd_fpts_std_uncond",
                "sim_dk_fpts_std_uncond",
                "dk_fpts_std_uncond",
                "fpts_sim_cond_std",
                "sim_fd_fpts_std",
                "fd_fpts_std",
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
                "sim_fd_fpts_p90_uncond",
                "fd_fpts_p90_uncond",
                "sim_dk_fpts_p90_uncond",
                "dk_fpts_p90_uncond",
                "fpts_sim_cond_p90",
                "sim_fd_fpts_p90",
                "fd_fpts_p90",
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
    # DK salary status is known to lag and can incorrectly mark playable users as OUT.
    # For DK, only trust projection-side availability fields (plus official/Rotowire signals below).
    if site_norm == "dk":
        status_cols = ["status"] if ("status" in proj_df.columns and "status" in merged.columns) else []
        is_out_cols = ["is_out"] if ("is_out" in proj_df.columns and "is_out" in merged.columns) else []
    else:
        status_cols = [c for c in ("status", "status_sal") if c in merged.columns]
        is_out_cols = [c for c in ("is_out", "is_out_sal") if c in merged.columns]
    disabled_col = "is_disabled_sal" if "is_disabled_sal" in merged.columns else "is_disabled"

    for _, row in merged.iterrows():
        # Get player_id (prefer projection's player_id, fall back to site player id)
        player_id = _canonicalize_player_id(row.get("player_id"))
        if player_id is None and site_player_id_col:
            player_id = _canonicalize_player_id(row.get(site_player_id_col))
        if player_id is None:
            logger.debug("Skipping row with no player_id or site id (site=%s)", site_norm)
            continue

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
            if proj_col in {"fpts_sim_cond_mean", "sim_dk_fpts_mean", "dk_fpts_mean", "sim_fd_fpts_mean", "fd_fpts_mean"}:
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

        raw_site_player_id = row.get(site_player_id_col) if site_player_id_col else None
        site_id = ""
        if raw_site_player_id is not None and not pd.isna(raw_site_player_id):
            try:
                site_id = str(int(raw_site_player_id))
            except Exception:
                site_id = str(raw_site_player_id)

        dk_player_id = row.get(dk_player_id_col) if dk_player_id_col in row.index else None
        dk_id = ""
        if dk_player_id is not None and not pd.isna(dk_player_id):
            try:
                dk_id = str(int(dk_player_id))
            except Exception:
                dk_id = str(dk_player_id)

        fd_player_id = row.get(fd_player_id_col) if fd_player_id_col in row.index else None
        fd_id = ""
        if fd_player_id is not None and not pd.isna(fd_player_id):
            try:
                fd_id = str(int(fd_player_id))
            except Exception:
                fd_id = str(fd_player_id)

        player = {
            "player_id": player_id,
            "name": (row.get(sal_name_col) if pd.notna(row.get(sal_name_col)) else None)
            or (row.get(proj_name_col) if pd.notna(row.get(proj_name_col)) else None)
            or str(player_id),
            "team": (row.get(sal_team_col) if pd.notna(row.get(sal_team_col)) else None)
            or (row.get(proj_team_col) if pd.notna(row.get(proj_team_col)) else None)
            or "UNK",
            "positions": positions,
            "salary": int(salary),
            "proj": float(proj),
            "site_id": site_id,
        }
        if dk_id:
            player["dk_id"] = dk_id
        if fd_id:
            player["fd_id"] = fd_id

        source_out = False
        try:
            player_id_int = int(float(player_id))
        except Exception:
            player_id_int = None
        if player_id_int is not None and player_id_int in official_out_player_ids:
            source_out = True
        if not source_out:
            source_out = (
                _normalize_name(player["name"]),
                _normalize_team(player["team"]),
            ) in rotowire_out_keys

        status_vals = [row.get(col) for col in status_cols if col in row.index]
        disabled_val = row.get(disabled_col) if disabled_col in row.index else None
        status_out = any(_is_out_status(value) for value in status_vals)
        disabled = _coerce_bool(disabled_val, default=False)
        # Legacy overrides may publish explicit activity fields.
        row_is_out = any(_coerce_bool(row.get(col), default=False) for col in is_out_cols if col in row.index)
        row_is_active = _coerce_bool(row.get("is_active") if "is_active" in row.index else True, default=True)
        is_out = bool(row_is_out or status_out or source_out)
        is_active = bool(row_is_active and (not disabled) and (not is_out))
        player["is_out"] = is_out
        player["is_active"] = is_active

        if exclude_inactive_players and not is_active:
            continue

        # Optional fields
        if own_col and pd.notna(row.get(own_col)):
            player["own_proj"] = float(effective_own if use_user_overrides else row[own_col])
        effective_stddev = row.get("effective_fpts_std") if use_user_overrides else None
        effective_p90 = row.get("effective_fpts_p90") if use_user_overrides else None
        if use_user_overrides and pd.notna(effective_stddev):
            player["stddev"] = float(effective_stddev)
        elif stddev_col and pd.notna(row.get(stddev_col)):
            player["stddev"] = float(row[stddev_col])
        if use_user_overrides and pd.notna(effective_p90):
            player["p90"] = float(effective_p90)
        elif p90_col and pd.notna(row.get(p90_col)):
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
            player["effective_stddev"] = float(effective_stddev) if pd.notna(effective_stddev) else None
            player["effective_p90"] = float(effective_p90) if pd.notna(effective_p90) else None
            player["has_override"] = has_override
            player["used_fppm_fallback"] = used_fppm_fallback
            player["override_minutes_delta"] = (
                float(row.get("override_minutes_delta")) if pd.notna(row.get("override_minutes_delta")) else None
            )
            player["override_fpts_delta"] = (
                float(row.get("override_fpts_delta")) if pd.notna(row.get("override_fpts_delta")) else None
            )

        pool.append(player)

    logger.info("Built player pool with %d optimizer-ready players", len(pool))

    if include_slate_analytics and pool:
        try:
            analytics_payload = load_or_compute_slate_player_analytics(
                game_date=game_date,
                draft_group_id=draft_group_id,
                pool_rows=pool,
                run_id=run_id,
                data_root=root,
            )
            analytics_by_pid = {
                str(row.get("player_id")): row
                for row in analytics_payload.get("players", [])
                if isinstance(row, dict) and row.get("player_id") is not None
            }
            for player in pool:
                metrics = analytics_by_pid.get(str(player.get("player_id")))
                if not metrics:
                    continue
                player["optimal_pct"] = float(metrics.get("optimal_pct") or 0.0)
                player["ceiling_leverage"] = float(metrics.get("ceiling_leverage") or 0.0)
                player["boom_pct"] = float(metrics.get("boom_pct") or 0.0)
                player["bust_pct"] = float(metrics.get("bust_pct") or 0.0)
        except Exception as exc:
            logger.warning(
                "Failed to attach slate analytics for %s/dg=%d: %s",
                game_date,
                draft_group_id,
                exc,
            )

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
        strategy_overrides_enabled = bool(
            job.config.get("use_strategy_overrides") or job.config.get("use_user_overrides")
        )

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
        lineup_stats_worlds_matrix: np.ndarray | None = None
        lineup_stats_player_index: Dict[str, int] | None = None
        resolved_worlds_run_id: str | None = None
        if job.config.get("world_sample_enabled", False):
            try:
                from projections.contest_sim.contest_sim_service import load_player_worlds

                from .strategy_overrides import (
                    apply_strategy_overrides_to_worlds,
                    load_slate_strategy_overrides,
                )

                run_id = job.config.get("run_id") if isinstance(job.config, dict) else None
                if run_id:
                    try:
                        from projections.projections_bundle import load_unified_projections_df

                        bundle = load_unified_projections_df(
                            job.game_date,
                            run_id=str(run_id),
                            data_root=get_data_root(),
                        )
                        resolved_worlds_run_id = _extract_single_string_value(bundle.df, "sim_run_id")
                    except Exception:
                        resolved_worlds_run_id = None

                player_worlds = load_player_worlds(
                    job.game_date,
                    get_data_root(),
                    run_id=resolved_worlds_run_id or run_id,
                    worlds_source=str(job.config.get("worlds_source") or "gtv2"),
                )
                worlds_matrix = player_worlds.fpts_matrix
                player_index = player_worlds.player_index
                if strategy_overrides_enabled:
                    slate_overrides = load_slate_strategy_overrides(job.game_date, int(job.draft_group_id))
                    if slate_overrides.overrides:
                        model_fpts_by_player = {
                            str(p.get("player_id")): float(p.get("model_proj", p.get("proj", 0.0)) or 0.0)
                            for p in player_pool
                            if p.get("player_id") is not None
                        }
                        model_minutes_by_player = {
                            str(p.get("player_id")): float(p.get("model_minutes", 0.0) or 0.0)
                            for p in player_pool
                            if p.get("player_id") is not None
                        }
                        try:
                            from projections.ops.manual_availability import list_manual_overrides as _lmo
                            _fi = _lmo(pd.Timestamp(job.game_date).date(), data_root=get_data_root(), active_only=True)
                            _force_active_qb: set[str] = {str(r["player_id"]) for r in _fi if str(r.get("override_type", "")).lower() == "force_in"}
                        except Exception:
                            _force_active_qb = set()
                        worlds_matrix, _, world_diagnostics = apply_strategy_overrides_to_worlds(
                            fpts_matrix=player_worlds.fpts_matrix,
                            player_index=player_index,
                            overrides=slate_overrides,
                            minutes_matrix=player_worlds.minutes_matrix,
                            model_minutes_by_player=model_minutes_by_player,
                            model_fpts_by_player=model_fpts_by_player,
                            force_active_player_ids=_force_active_qb or None,
                        )
                        logger.info(
                            "QuickBuild world-sample strategy overrides applied for %s/dg=%d: matched=%d",
                            job.game_date,
                            job.draft_group_id,
                            int(world_diagnostics.get("matched_override_count", 0)),
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
                lineup_stats_worlds_matrix = worlds_matrix
                lineup_stats_player_index = player_index
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
            if lineup_stats_worlds_matrix is None or lineup_stats_player_index is None:
                from projections.contest_sim.contest_sim_service import load_player_worlds

                from .strategy_overrides import (
                    apply_strategy_overrides_to_worlds,
                    load_slate_strategy_overrides,
                )

                run_id = job.config.get("run_id") if isinstance(job.config, dict) else None
                player_worlds = load_player_worlds(
                    job.game_date,
                    get_data_root(),
                    run_id=resolved_worlds_run_id or run_id,
                    worlds_source=str(job.config.get("worlds_source") or "gtv2"),
                )
                lineup_stats_worlds_matrix = player_worlds.fpts_matrix
                lineup_stats_player_index = player_worlds.player_index
                if strategy_overrides_enabled:
                    slate_overrides = load_slate_strategy_overrides(job.game_date, int(job.draft_group_id))
                    if slate_overrides.overrides:
                        model_fpts_by_player = {
                            str(p.get("player_id")): float(p.get("model_proj", p.get("proj", 0.0)) or 0.0)
                            for p in player_pool
                            if p.get("player_id") is not None
                        }
                        model_minutes_by_player = {
                            str(p.get("player_id")): float(p.get("model_minutes", 0.0) or 0.0)
                            for p in player_pool
                            if p.get("player_id") is not None
                        }
                        lineup_stats_worlds_matrix, _, _ = apply_strategy_overrides_to_worlds(
                            fpts_matrix=player_worlds.fpts_matrix,
                            player_index=player_worlds.player_index,
                            overrides=slate_overrides,
                            minutes_matrix=player_worlds.minutes_matrix,
                            model_minutes_by_player=model_minutes_by_player,
                            model_fpts_by_player=model_fpts_by_player,
                        )

            if lineup_stats_worlds_matrix is not None and lineup_stats_player_index:
                world_player_ids = [0] * len(lineup_stats_player_index)
                for pid, idx in lineup_stats_player_index.items():
                    world_player_ids[int(idx)] = int(str(pid))
                stats_objs = compute_lineup_distribution_stats(
                    lineups=result.lineups,
                    world_player_ids=world_player_ids,
                    fpts_by_world=lineup_stats_worlds_matrix,
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


def get_slates_for_date(
    game_date: str,
    slate_type: str = "all",
    *,
    site: str = "dk",
) -> List[Dict[str, Any]]:
    """Get available draft groups for a date and site."""
    site_norm = _normalize_site(site)
    api_slates: List[Dict[str, Any]] = []

    if site_norm == "dk":
        try:
            df = list_draft_groups_for_date(game_date, slate_type=slate_type)  # type: ignore
            if not df.empty:
                api_slates = df.to_dict(orient="records")
        except Exception as exc:
            logger.warning("Failed to fetch live DK slates for %s: %s", game_date, exc)
    elif site_norm == "fd":
        try:
            df = list_fixture_lists_for_date(game_date, slate_type=slate_type)  # type: ignore[arg-type]
            if not df.empty:
                api_slates = df.to_dict(orient="records")
        except Exception as exc:
            logger.warning("Failed to fetch live FD slates for %s: %s", game_date, exc)

    disk_slates = _discover_slates_from_disk(game_date, slate_type=slate_type, site=site_norm)
    bronze_slates = _discover_slates_from_bronze_draftables(
        game_date,
        slate_type=slate_type,
        site=site_norm,
    )

    by_dg: dict[str, Dict[str, Any]] = {}
    for slate in api_slates:
        dg_key = _draft_group_key(slate.get("draft_group_id"))
        if dg_key is None:
            continue
        by_dg[dg_key] = slate

    for source in [disk_slates, bronze_slates]:
        for slate in source:
            dg_key = _draft_group_key(slate.get("draft_group_id"))
            if dg_key is None:
                continue
            if dg_key not in by_dg:
                by_dg[dg_key] = slate
                continue

            existing = by_dg[dg_key]
            if (not existing.get("games")) and slate.get("games"):
                existing["games"] = slate.get("games")
            if (
                str(existing.get("slate_type", "")).lower() != "showdown"
                and str(slate.get("slate_type", "")).lower() == "showdown"
            ):
                existing["slate_type"] = "showdown"
            if not existing.get("earliest_start") and slate.get("earliest_start"):
                existing["earliest_start"] = slate.get("earliest_start")
            if not existing.get("latest_start") and slate.get("latest_start"):
                existing["latest_start"] = slate.get("latest_start")
            if not existing.get("example_contest_name") and slate.get("example_contest_name"):
                existing["example_contest_name"] = slate.get("example_contest_name")

    merged_slates = sorted(by_dg.values(), key=lambda s: _draft_group_sort_key(s.get("draft_group_id")))
    merged_slates = _dedupe_equivalent_slates(merged_slates)

    compatible_slates = [
        slate
        for slate in merged_slates
        if _is_optimizer_compatible_slate(site=site_norm, slate=slate)
    ]
    # Safety fallback: if heuristics filtered out everything, return unfiltered.
    if compatible_slates:
        return compatible_slates
    return merged_slates


def _draft_group_key(raw: object) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return text


def _draft_group_sort_key(raw: object) -> tuple[int, int | str]:
    key = _draft_group_key(raw)
    if key is None:
        return (2, "")
    try:
        return (0, int(key))
    except ValueError:
        return (1, key)


_DK_UNSUPPORTED_SLATE_NAME_RE = re.compile(
    r"\b(snake|tiers|single[\s-]?stat|points?\s+only|in[\s-]?game|2nd\s+half|second\s+half|4th\s+qtr|fourth\s+qtr)\b",
    re.IGNORECASE,
)
_FD_UNSUPPORTED_SLATE_NAME_RE = re.compile(
    r"\b(snake(\s+draft)?|points?\s+only|2nd\s+half|second\s+half|4th\s+qtr|fourth\s+qtr)\b",
    re.IGNORECASE,
)
_FD_SINGLE_GAME_LABEL_RE = re.compile(r"^[A-Z]{2,4}\s*@\s*[A-Z]{2,4}$")


def _is_optimizer_compatible_slate(*, site: str, slate: Dict[str, Any]) -> bool:
    """Return whether a slate is compatible with the classic optimizer/contest-sim roster."""
    slate_type = str(slate.get("slate_type") or "").strip().lower()
    if slate_type == "showdown":
        return False

    name = str(slate.get("example_contest_name") or "").strip()
    if not name:
        return True

    site_norm = _normalize_site(site)
    if site_norm == "dk":
        return _DK_UNSUPPORTED_SLATE_NAME_RE.search(name) is None

    if _FD_UNSUPPORTED_SLATE_NAME_RE.search(name) is not None:
        return False
    if _FD_SINGLE_GAME_LABEL_RE.match(name.upper()) is not None:
        return False
    return True


def _slate_signature(slate: Dict[str, Any]) -> str | None:
    slate_type = str(slate.get("slate_type") or "").strip().lower()
    games = slate.get("games") if isinstance(slate.get("games"), list) else []
    matchups = sorted(
        {
            str(game.get("matchup") or "").strip().upper()
            for game in games
            if isinstance(game, dict) and str(game.get("matchup") or "").strip()
        }
    )
    if not matchups:
        return None
    earliest = str(slate.get("earliest_start") or "").strip()
    return f"{slate_type}|{earliest}|{'|'.join(matchups)}"


def _dedupe_equivalent_slates(slates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse duplicate slate rows that represent the same game set/start window."""
    by_signature: dict[str, Dict[str, Any]] = {}
    passthrough: List[Dict[str, Any]] = []

    for slate in slates:
        signature = _slate_signature(slate)
        if signature is None:
            passthrough.append(slate)
            continue

        current = by_signature.get(signature)
        if current is None:
            by_signature[signature] = slate
            continue

        curr_n_contests = int(current.get("n_contests") or 0)
        next_n_contests = int(slate.get("n_contests") or 0)
        if next_n_contests > curr_n_contests:
            by_signature[signature] = slate
            continue
        if next_n_contests == curr_n_contests:
            curr_dg = _draft_group_sort_key(current.get("draft_group_id"))
            next_dg = _draft_group_sort_key(slate.get("draft_group_id"))
            if next_dg < curr_dg:
                by_signature[signature] = slate

    merged = passthrough + list(by_signature.values())
    return sorted(merged, key=lambda s: _draft_group_sort_key(s.get("draft_group_id")))


def _extract_games_from_salaries_df(df: pd.DataFrame) -> tuple[List[Dict[str, Any]], datetime | None, datetime | None]:
    if df.empty:
        return [], None, None

    matchup_col = next((c for c in ("game_matchup", "matchup", "game", "game_info") if c in df.columns), None)
    start_col = next((c for c in ("game_start_utc", "start_time_utc", "start_time", "lock_time") if c in df.columns), None)
    if matchup_col is None:
        return [], None, None

    work = df.copy()
    work["__matchup"] = work[matchup_col].apply(_infer_game_matchup_from_text)
    if start_col is not None:
        work["__start"] = work[start_col].apply(_safe_parse_timestamp)
    else:
        work["__start"] = None

    if "__start" in work.columns and work["__start"].isna().all() and "game_info" in work.columns:
        work["__start"] = work["game_info"].apply(
            lambda v: _safe_parse_timestamp(str(v).split(" ", 1)[1] if isinstance(v, str) and " " in v else None)
        )

    work = work.dropna(subset=["__matchup"]).drop_duplicates(subset=["__matchup", "__start"], keep="first")
    if work.empty:
        return [], None, None

    games: list[dict[str, Any]] = []
    starts: list[datetime] = []
    for _, row in work.iterrows():
        entry: dict[str, Any] = {"matchup": str(row["__matchup"])}
        start = row.get("__start")
        if isinstance(start, datetime):
            starts.append(start)
            entry["start_time"] = start.isoformat()
        games.append(entry)

    games.sort(key=lambda g: g.get("start_time", ""))
    earliest = min(starts) if starts else None
    latest = max(starts) if starts else None
    return games, earliest, latest


def _discover_slates_from_disk(
    game_date: str,
    slate_type: str = "all",
    *,
    site: str = "dk",
) -> List[Dict[str, Any]]:
    """Discover available slates from gold salaries partitions."""
    site_norm = _normalize_site(site)
    root = get_data_root()
    salaries_base = root / "gold" / "dk_salaries" / f"site={site_norm}" / f"game_date={game_date}"

    if not salaries_base.exists():
        logger.debug("No gold salaries directory for %s site=%s", game_date, site_norm)
        return []

    slates: List[Dict[str, Any]] = []
    for dg_dir in salaries_base.iterdir():
        if not dg_dir.is_dir() or not dg_dir.name.startswith("draft_group_id="):
            continue
        parts = dg_dir.name.split("=", 1)
        if len(parts) != 2:
            continue
        dg_raw = parts[1].strip()
        if not dg_raw:
            continue
        try:
            dg_id: int | str = int(dg_raw)
        except ValueError:
            dg_id = dg_raw

        salaries_file = dg_dir / "salaries.parquet"
        if not salaries_file.exists():
            continue

        inferred_type = "main"
        example_name = f"{site_norm.upper()} Draft Group {dg_raw}"
        games: List[Dict[str, Any]] = []
        earliest_start = None
        latest_start = None

        game_info = (
            _load_game_info_from_draftables(int(dg_id), root)
            if site_norm == "dk" and isinstance(dg_id, int)
            else {}
        )
        if game_info:
            for info in game_info.values():
                game_entry = {"matchup": info["matchup"]}
                start_time = info.get("start_time_utc")
                if isinstance(start_time, datetime):
                    game_entry["start_time"] = start_time.isoformat()
                    earliest_start = start_time if earliest_start is None else min(earliest_start, start_time)
                    latest_start = start_time if latest_start is None else max(latest_start, start_time)
                games.append(game_entry)
            games.sort(key=lambda g: g.get("start_time", ""))
        else:
            try:
                salaries_df = pd.read_parquet(salaries_file)
                games, earliest_start, latest_start = _extract_games_from_salaries_df(salaries_df)
            except Exception:
                games = []

        if site_norm == "dk" and isinstance(dg_id, int):
            bronze_path = root / "bronze" / "dk" / "draftables" / f"draftables_raw_{dg_id}.json"
            if bronze_path.exists():
                try:
                    payload = json.loads(bronze_path.read_text(encoding="utf-8"))
                    contests = payload.get("Contests", [])
                    if contests:
                        name = contests[0].get("n", contests[0].get("ContestName", ""))
                        if name:
                            inferred_type = _infer_slate_type(name)
                            example_name = name
                except Exception:
                    pass

        if inferred_type == "main" and len(games) == 1:
            inferred_type = "showdown"

        slate = {
            "game_date": game_date,
            "slate_type": inferred_type,
            "draft_group_id": dg_id,
            "n_contests": 0,
            "earliest_start": earliest_start.isoformat() if earliest_start else None,
            "latest_start": latest_start.isoformat() if latest_start else None,
            "example_contest_name": example_name,
            "games": games,
        }
        if slate_type == "all" or inferred_type == slate_type:
            slates.append(slate)

    logger.info("Discovered %d slates from disk for %s site=%s", len(slates), game_date, site_norm)
    return sorted(slates, key=lambda s: _draft_group_sort_key(s.get("draft_group_id")))


def _discover_slates_from_bronze_draftables(
    game_date: str,
    slate_type: str = "all",
    *,
    site: str = "dk",
) -> List[Dict[str, Any]]:
    """Fallback discovery from bronze DK draftables when gold salaries are missing."""
    from zoneinfo import ZoneInfo

    site_norm = _normalize_site(site)
    if site_norm != "dk":
        return []

    root = get_data_root()
    bronze_dir = root / "bronze" / "dk" / "draftables"
    if not bronze_dir.exists():
        return []

    try:
        target_date = dt.date.fromisoformat(game_date)
    except ValueError:
        logger.warning("Invalid game_date format for bronze discovery: %s", game_date)
        return []

    eastern = ZoneInfo("America/New_York")
    slates: list[dict[str, Any]] = []
    for path in sorted(bronze_dir.glob("draftables_raw_*.json"), reverse=True):
        m = re.search(r"draftables_raw_(\d+)\.json$", path.name)
        if not m:
            continue
        dg_id = int(m.group(1))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        competitions = payload.get("competitions", [])
        games: list[dict[str, Any]] = []
        starts: list[datetime] = []
        for comp in competitions:
            away = (comp.get("awayTeam") or {}).get("abbreviation")
            home = (comp.get("homeTeam") or {}).get("abbreviation")
            if not away or not home:
                continue
            game = {"matchup": f"{away}@{home}"}
            start_ts = _safe_parse_timestamp(comp.get("startTime"))
            if isinstance(start_ts, datetime):
                game["start_time"] = start_ts.isoformat()
                starts.append(start_ts)
            games.append(game)

        # Filter by game date: check if any game starts on the target date (in Eastern time)
        if not starts:
            continue
        slate_dates = {s.astimezone(eastern).date() for s in starts}
        if target_date not in slate_dates:
            continue

        contests = payload.get("Contests", [])
        name = ""
        if contests:
            name = contests[0].get("n", contests[0].get("ContestName", "")) or ""
        inferred_type = _infer_slate_type(name) if name else ("showdown" if len(games) == 1 else "main")

        if slate_type != "all" and inferred_type != slate_type:
            continue

        slates.append(
            {
                "game_date": game_date,
                "slate_type": inferred_type,
                "draft_group_id": dg_id,
                "n_contests": 0,
                "earliest_start": min(starts).isoformat() if starts else None,
                "latest_start": max(starts).isoformat() if starts else None,
                "example_contest_name": name or f"Draft Group {dg_id}",
                "games": games,
            }
        )

    return sorted(slates, key=lambda s: _draft_group_sort_key(s.get("draft_group_id")))


def _infer_slate_type(name: str) -> str:
    """Infer slate type from contest name."""
    name_lower = name.lower()
    if re.search(r"\bturbo\b", name_lower):
        return "turbo"
    if re.search(r"\b(late|night)\b", name_lower):
        return "night"
    if re.search(r"\bearly\b", name_lower):
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
