"""CLI for finalizing unified projections artifact.

Merges minutes, sim outputs, and ownership predictions into a single
per-run parquet file with complete projection data.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from projections.paths import data_path, get_project_root


def _normalize_name(value: str | None) -> str:
    """Normalize player name for matching: fold Unicode diacritics, lowercase, strip.
    
    Handles European characters like Dončić -> doncic, Jokić -> jokic.
    """
    if not value:
        return ""
    # Fold Unicode (e.g., Dončić -> Doncic) before stripping non-alphanumerics
    normalized = unicodedata.normalize("NFKD", value)
    ascii_folded = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", ascii_folded.lower())

UTC = timezone.utc

LATEST_POINTER = "latest_run.json"
MINUTES_FILENAME = "minutes.parquet"
PROJECTIONS_FILENAME = "projections.parquet"
SUMMARY_FILENAME = "summary.json"

ENV_MINUTES_DAILY_ROOT = "MINUTES_DAILY_ROOT"


# Columns to include from each source
MINUTES_COLUMNS = [
    # Identity
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "team_tricode",
    "opponent_team_id",
    "opponent_team_name",
    "opponent_team_tricode",
    "game_id",
    "game_date",
    "tip_ts",
    # Status
    "starter_flag",
    "is_projected_starter",
    "is_confirmed_starter",
    "status",
    "play_prob",
    "pos_bucket",
    # Minutes projections
    "minutes_p10",
    "minutes_p50",
    "minutes_p90",
    "minutes_p10_cond",
    "minutes_p50_cond",
    "minutes_p90_cond",
    # Game context
    "spread_home",
    "total",
    "team_implied_total",
    "opponent_implied_total",
    "odds_as_of_ts",
]

SIM_COLUMNS = [
    # FPTS quantiles
    "dk_fpts_mean",
    "dk_fpts_std",
    "dk_fpts_p05",
    "dk_fpts_p10",
    "dk_fpts_p25",
    "dk_fpts_p50",
    "dk_fpts_p75",
    "dk_fpts_p90",
    "dk_fpts_p95",
    # Box score stats
    "pts_mean",
    "reb_mean",
    "ast_mean",
    "stl_mean",
    "blk_mean",
    "tov_mean",
    # Minutes simulation stats
    "minutes_mean",       # model p50 (reference)
    "minutes_sim_mean",   # cross-world average
    "minutes_sim_std",    # cross-world std
    "minutes_sim_p10",    # sim p10
    "minutes_sim_p50",    # sim p50
    "minutes_sim_p90",    # sim p90
    # Metadata
    "sim_profile",
    "n_worlds",
    "rates_run_id",
    "is_starter",  # sim's view of starter
]

OWNERSHIP_COLUMNS = [
    "pred_own_pct",
    "salary",
]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _read_latest_run_id(day_dir: Path) -> str | None:
    payload = _read_json(day_dir / LATEST_POINTER)
    if not payload:
        return None
    run_id = payload.get("run_id")
    return str(run_id) if run_id else None


def _resolve_minutes_daily_root(data_root: Path) -> Path:
    raw = os.environ.get(ENV_MINUTES_DAILY_ROOT)
    if raw:
        return Path(raw).expanduser().resolve()
    return data_root / "artifacts" / "minutes_v1" / "daily"


def _resolve_latest_run_id_from_dirs(day_dirs: list[Path]) -> str | None:
    for day_dir in day_dirs:
        run_id = _read_latest_run_id(day_dir)
        if run_id:
            return run_id
    return None


def _load_minutes(
    game_date: date,
    *,
    minutes_run_id: str | None,
    data_root: Path,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Load minutes artifact for a specific run (prefers data-root artifacts, falls back to gold)."""
    project_root = get_project_root()
    day_token = game_date.isoformat()

    minutes_daily_root = _resolve_minutes_daily_root(data_root)
    daily_day_dirs = [minutes_daily_root / day_token]
    if project_root != data_root:
        daily_day_dirs.append(project_root / "artifacts" / "minutes_v1" / "daily" / day_token)

    gold_day_dir = data_root / "gold" / "projections_minutes_v1" / f"game_date={day_token}"
    if project_root != data_root:
        gold_day_dir_project = project_root / "gold" / "projections_minutes_v1" / f"game_date={day_token}"
    else:
        gold_day_dir_project = None

    candidate_run_ids: list[str] = []
    if minutes_run_id:
        candidate_run_ids.append(minutes_run_id)
    else:
        inferred = _resolve_latest_run_id_from_dirs(daily_day_dirs)
        if inferred:
            candidate_run_ids.append(inferred)
        inferred_gold = _read_latest_run_id(gold_day_dir)
        if inferred_gold and inferred_gold not in candidate_run_ids:
            candidate_run_ids.append(inferred_gold)
        if gold_day_dir_project is not None:
            inferred_gold_project = _read_latest_run_id(gold_day_dir_project)
            if inferred_gold_project and inferred_gold_project not in candidate_run_ids:
                candidate_run_ids.append(inferred_gold_project)

    if not candidate_run_ids:
        run_dirs: list[Path] = []
        for day_dir in daily_day_dirs:
            if not day_dir.exists():
                continue
            run_dirs.extend([p for p in day_dir.iterdir() if p.is_dir() and p.name.startswith("run=")])
        run_dirs = sorted(run_dirs, reverse=True)
        if run_dirs:
            fallback_id = run_dirs[0].name.split("=", 1)[1]
            candidate_run_ids.append(fallback_id)
            print(
                f"[finalize] warning: minutes_run_id missing and {LATEST_POINTER} not found; "
                f"falling back to newest run dir run={fallback_id}"
            )

    candidates: list[tuple[Path, str, str | None]] = []
    for run_id in candidate_run_ids:
        for day_dir in daily_day_dirs:
            candidates.append(
                (day_dir / f"run={run_id}" / MINUTES_FILENAME, "minutes_v1_daily", run_id)
            )
        candidates.append(
            (gold_day_dir / f"run={run_id}" / MINUTES_FILENAME, "projections_minutes_v1_gold", run_id)
        )
        if gold_day_dir_project is not None:
            candidates.append(
                (gold_day_dir_project / f"run={run_id}" / MINUTES_FILENAME, "projections_minutes_v1_gold_project", run_id)
            )

    # Allow legacy flat-file gold outputs (single-file minutes.parquet).
    candidates.extend(
        [
            (gold_day_dir / MINUTES_FILENAME, "projections_minutes_v1_gold_flat", None),
        ]
    )
    if gold_day_dir_project is not None:
        candidates.append((gold_day_dir_project / MINUTES_FILENAME, "projections_minutes_v1_gold_flat_project", None))

    for path, label, resolved_run_id in candidates:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        available = [c for c in MINUTES_COLUMNS if c in df.columns]
        meta = {
            "minutes_run_id": resolved_run_id,
            "minutes_path": str(path),
            "minutes_source": label,
        }
        if label.startswith("projections_minutes_v1_gold"):
            print(f"[finalize] warning: loaded minutes from {label} at {path} (prefer minutes_v1 daily artifacts)")
        return df[available].copy(), meta

    meta = {
        "minutes_run_id": minutes_run_id,
        "minutes_path": None,
        "minutes_source": None,
        "searched": [str(p) for p, _, _ in candidates[:10]],
    }
    print(
        f"[finalize] Minutes not found for {day_token} minutes_run_id={minutes_run_id!r} "
        f"(daily_root={minutes_daily_root})"
    )
    return None, meta


def _load_sim(
    game_date: date,
    data_root: Path,
    *,
    sim_run_id: str | None = None,
    minutes_run_id: str | None = None,
    allow_legacy_sim_projections_root: bool = False,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Load sim projections (prefers worlds_fpts_v2 outputs; falls back to legacy sim_v2/projections)."""

    sim_meta: dict[str, Any] = {
        "sim_run_id": sim_run_id,
        "sim_path": None,
        "sim_source": None,
    }

    worlds_root = data_root / "artifacts" / "sim_v2" / "worlds_fpts_v2"
    worlds_bases = [
        worlds_root / f"game_date={game_date.isoformat()}",
        worlds_root / f"date={game_date.isoformat()}",
        worlds_root / game_date.isoformat(),
    ]

    legacy_root = data_root / "artifacts" / "sim_v2" / "projections"
    legacy_bases = [
        legacy_root / f"game_date={game_date.isoformat()}",
        legacy_root / f"date={game_date.isoformat()}",
        legacy_root / game_date.isoformat(),
    ]

    def _resolve_run_dir(base_dir: Path) -> Path | None:
        pointer = base_dir / LATEST_POINTER
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

    def _read_run_dir(run_dir: Path) -> pd.DataFrame | None:
        for name in (PROJECTIONS_FILENAME, "sim_v2_projections.parquet"):
            candidate = run_dir / name
            if candidate.exists():
                return pd.read_parquet(candidate)
        return None

    def _load_from_bases(
        base_dirs: list[Path],
        *,
        preferred_run_id: str | None,
        source_label: str,
    ) -> tuple[pd.DataFrame | None, dict[str, Any]]:
        for base in base_dirs:
            if not base.exists():
                continue

            df: pd.DataFrame | None = None
            resolved_run_id: str | None = None
            resolved_path: Path | None = None

            if base.is_dir():
                desired_run_id = preferred_run_id
                if desired_run_id is None:
                    desired_run_id = _read_latest_run_id(base)
                    if desired_run_id is None:
                        run_dir = _resolve_run_dir(base)
                        if run_dir is not None and run_dir.name.startswith("run="):
                            desired_run_id = run_dir.name.split("=", 1)[1]
                if desired_run_id:
                    run_dir = base / f"run={desired_run_id}"
                    if run_dir.exists():
                        df = _read_run_dir(run_dir)
                        resolved_run_id = desired_run_id
                        resolved_path = run_dir

                # Fall back to a direct projections.parquet under the day dir.
                if df is None:
                    direct = base / PROJECTIONS_FILENAME
                    if direct.exists():
                        df = pd.read_parquet(direct)
                        resolved_run_id = None
                        resolved_path = direct

            elif base.is_file() and base.suffix == ".parquet":
                df = pd.read_parquet(base)
                resolved_path = base
                resolved_run_id = None

            if df is None:
                continue

            if minutes_run_id:
                if "minutes_run_id" not in df.columns:
                    continue
                df = df.loc[df["minutes_run_id"].astype(str) == str(minutes_run_id)].copy()
                if df.empty:
                    continue

            join_cols = ["player_id"]
            for col in ("game_id", "game_date"):
                if col in df.columns and col not in join_cols:
                    join_cols.append(col)
            available = join_cols + [c for c in SIM_COLUMNS if c in df.columns and c not in join_cols]
            meta = {
                "sim_run_id": resolved_run_id or preferred_run_id,
                "sim_source": source_label,
                "sim_path": str(resolved_path) if resolved_path is not None else None,
            }
            return df[available].copy(), meta

        return None, {"sim_run_id": preferred_run_id, "sim_source": source_label, "sim_path": None}

    # Prefer worlds_fpts_v2 outputs (source of truth for dk_fpts_* and minutes_sim_*)
    df, meta = _load_from_bases(worlds_bases, preferred_run_id=sim_run_id or _resolve_latest_run_id_from_dirs(worlds_bases), source_label="sim_v2_worlds_fpts_v2")
    if df is not None:
        sim_meta.update(meta)
        return df, sim_meta

    if not allow_legacy_sim_projections_root:
        print(
            f"[finalize] Sim projections not found under {worlds_root} for {game_date.isoformat()} "
            "(legacy sim_v2/projections disabled)"
        )
        return None, sim_meta

    # Fall back to legacy sim_v2/projections outputs (may be stale / post-processed)
    df, meta = _load_from_bases(legacy_bases, preferred_run_id=sim_run_id, source_label="sim_v2_projections_legacy")
    if df is not None:
        print("[finalize] warning: loaded sim projections from legacy sim_v2/projections (prefer worlds_fpts_v2)")
        sim_meta.update(meta)
        return df, sim_meta

    print(f"[finalize] Sim projections not found for {game_date.isoformat()} (worlds_root={worlds_root})")
    return None, sim_meta


def _load_ownership(
    game_date: date,
    draft_group_id: str,
    data_root: Path,
    *,
    run_id: str | None = None,
) -> Optional[pd.DataFrame]:
    """Load ownership predictions for a specific slate."""
    # Try run-scoped path first (new format).
    base_dir = data_root / "silver" / "ownership_predictions" / str(game_date)
    own_path = None
    if run_id:
        candidate = base_dir / f"run={run_id}" / f"{draft_group_id}.parquet"
        if candidate.exists():
            own_path = candidate

    if own_path is None:
        latest_pointer = base_dir / "latest_run.json"
        if latest_pointer.exists():
            try:
                payload = json.loads(latest_pointer.read_text(encoding="utf-8"))
                latest = payload.get("run_id")
            except Exception:
                latest = None
            if latest:
                candidate = base_dir / f"run={latest}" / f"{draft_group_id}.parquet"
                if candidate.exists():
                    own_path = candidate

    # Fall back to legacy per-slate path.
    if own_path is None:
        own_path = base_dir / f"{draft_group_id}.parquet"
    
    if not own_path.exists():
        # Fall back to legacy format (single file)
        legacy_path = data_root / "silver" / "ownership_predictions" / f"{game_date}.parquet"
        if legacy_path.exists():
            print(f"[finalize] Using legacy ownership path: {legacy_path}")
            own_path = legacy_path
        else:
            print(f"[finalize] Ownership not found at {own_path}")
            return None
    
    df = pd.read_parquet(own_path)
    
    # Use player_name for joining (ownership uses DK player_id, not NBA)
    available = ["player_name"] + [c for c in OWNERSHIP_COLUMNS if c in df.columns]
    if "player_name" not in df.columns:
        return None
    return df[available].copy()


def _load_salaries(
    game_date: date,
    draft_group_id: str,
    data_root: Path,
) -> Optional[pd.DataFrame]:
    """Load DK salaries for a specific slate."""
    # Try specific draft group first
    salaries_path = data_root / "gold" / "dk_salaries" / "site=dk" / f"game_date={game_date}" / f"draft_group_id={draft_group_id}" / "salaries.parquet"
    
    if salaries_path.exists():
        df = pd.read_parquet(salaries_path)
    else:
        # Fall back to finding any slate
        base = data_root / "gold" / "dk_salaries" / "site=dk" / f"game_date={game_date}"
        if not base.exists():
            return None
        
        draft_group_dirs = sorted(base.glob("draft_group_id=*"))
        if not draft_group_dirs:
            return None
        
        # Use largest slate as fallback
        all_salaries = []
        for dg_dir in draft_group_dirs:
            parquet_path = dg_dir / "salaries.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                all_salaries.append(df)
        
        if not all_salaries:
            return None
        
        combined = pd.concat(all_salaries, ignore_index=True)
        main_dg = combined.groupby("draft_group_id").size().idxmax()
        df = combined[combined["draft_group_id"] == main_dg].copy()
        print(f"[finalize] Salary fallback: using slate {main_dg}")
    
    # Normalize columns
    if "display_name" in df.columns:
        df["player_name"] = df["display_name"]
    
    return df[["player_name", "salary"]].drop_duplicates("player_name")


def finalize_projections(
    game_date: date,
    projections_run_id: str,
    draft_group_id: str,
    data_root: Path,
    *,
    minutes_run_id: str | None = None,
    sim_run_id: str | None = None,
    ownership_run_id: str | None = None,
    allow_legacy_sim_projections_root: bool = False,
) -> Optional[Path]:
    """
    Merge minutes, sim, and ownership into unified projections artifact.
    
    Args:
        game_date: Game date
        projections_run_id: Projections (output) run identifier
        draft_group_id: DraftKings draft group ID for ownership lookup
        data_root: Data root path
        minutes_run_id: Explicit minutes run identifier (defaults to latest pointer when available)
        sim_run_id: Explicit sim run identifier (defaults to sim worlds latest pointer when available)
        ownership_run_id: Explicit ownership run identifier (defaults to projections_run_id)
    
    Returns path to unified artifact or None if minutes unavailable.
    """
    # Load minutes (required - base of the join)
    minutes_df, minutes_meta = _load_minutes(game_date, minutes_run_id=minutes_run_id, data_root=data_root)
    if minutes_df is None or minutes_df.empty:
        print(
            f"[finalize] No minutes for {game_date.isoformat()} minutes_run_id={minutes_run_id!r} "
            f"(projections_run_id={projections_run_id})"
        )
        return None

    resolved_minutes_run_id = minutes_meta.get("minutes_run_id") or minutes_run_id
    if resolved_minutes_run_id is None:
        print(
            "[finalize] warning: minutes_run_id could not be resolved from minutes artifacts; "
            "sim filtering by minutes_run_id will be disabled"
        )

    print(f"[finalize] Loaded {len(minutes_df)} players from minutes (run={resolved_minutes_run_id})")
    unified = minutes_df.copy()
    
    # Load and merge sim projections
    sim_df, sim_meta = _load_sim(
        game_date,
        data_root,
        sim_run_id=sim_run_id,
        minutes_run_id=resolved_minutes_run_id if resolved_minutes_run_id is not None else None,
        allow_legacy_sim_projections_root=allow_legacy_sim_projections_root,
    )
    resolved_sim_run_id = sim_meta.get("sim_run_id") or sim_run_id
    if sim_df is not None and not sim_df.empty:
        join_keys = ["player_id"]
        if "game_id" in unified.columns and "game_id" in sim_df.columns:
            join_keys.append("game_id")
        unified = unified.merge(
            sim_df,
            on=join_keys,
            how="left",
            suffixes=("", "_sim")
        )
        matched = unified["dk_fpts_mean"].notna().sum()
        print(
            f"[finalize] Merged {matched}/{len(unified)} sim projections "
            f"(sim_run_id={resolved_sim_run_id}, source={sim_meta.get('sim_source')})"
        )
    else:
        print("[finalize] No sim projections available")
    
    # Load ownership (join on player_name since DK uses different IDs)
    ownership_run_id = ownership_run_id or projections_run_id
    ownership = _load_ownership(game_date, draft_group_id, data_root, run_id=ownership_run_id)
    if ownership is not None and not ownership.empty:
        # Normalize names for matching (handles Unicode like Dončić -> doncic)
        unified["_name_norm"] = unified["player_name"].apply(_normalize_name)
        ownership["_name_norm"] = ownership["player_name"].apply(_normalize_name)
        
        ownership_cols = ["_name_norm"] + [c for c in OWNERSHIP_COLUMNS if c in ownership.columns]
        unified = unified.merge(
            ownership[ownership_cols],
            on="_name_norm",
            how="left",
            suffixes=("", "_own")
        )
        unified = unified.drop(columns=["_name_norm"])
        
        matched = unified["pred_own_pct"].notna().sum() if "pred_own_pct" in unified.columns else 0
        print(f"[finalize] Merged {matched}/{len(unified)} ownership predictions")
    else:
        print("[finalize] No ownership predictions available")
    
    # Load salaries if not already present
    if "salary" not in unified.columns:
        salaries = _load_salaries(game_date, draft_group_id, data_root)
        if salaries is not None:
            # Join on player_name (normalized handles Unicode like Dončić)
            unified["_name_norm"] = unified["player_name"].apply(_normalize_name)
            salaries["_name_norm"] = salaries["player_name"].apply(_normalize_name)
            unified = unified.merge(
                salaries[["_name_norm", "salary"]],
                on="_name_norm",
                how="left"
            )
            unified = unified.drop(columns=["_name_norm"])
            print(f"[finalize] Merged {unified['salary'].notna().sum()} salaries")
    
    # Compute value (FPTS per $1k salary)
    if "dk_fpts_mean" in unified.columns and "salary" in unified.columns:
        unified["value"] = (unified["dk_fpts_mean"] / unified["salary"] * 1000).round(2)
    
    # Compute is_locked based on whether tip_ts has passed
    if "tip_ts" in unified.columns:
        now = pd.Timestamp.now(tz="UTC")
        tip_ts = pd.to_datetime(unified["tip_ts"], utc=True, errors="coerce")
        unified["is_locked"] = tip_ts.notna() & (tip_ts <= now)
        locked_count = unified["is_locked"].sum()
        print(f"[finalize] Marked {locked_count}/{len(unified)} players as locked (tip_ts <= {now.isoformat()})")
    else:
        unified["is_locked"] = False

    # Attach run IDs for end-to-end traceability.
    unified["projections_run_id"] = str(projections_run_id)
    unified["minutes_run_id"] = (
        str(resolved_minutes_run_id) if resolved_minutes_run_id is not None else pd.NA
    )
    unified["sim_run_id"] = str(resolved_sim_run_id) if resolved_sim_run_id is not None else pd.NA
    
    # Write unified artifact
    out_dir = data_root / "artifacts" / "projections" / str(game_date) / f"run={projections_run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "projections.parquet"
    unified.to_parquet(out_path, index=False)

    # Write run-scoped summary.json
    sim_profile = None
    if "sim_profile" in unified.columns:
        values = [v for v in unified["sim_profile"].dropna().unique().tolist() if v]
        sim_profile = values[0] if values else None
    n_worlds = None
    if "n_worlds" in unified.columns:
        worlds = pd.to_numeric(unified["n_worlds"], errors="coerce").dropna().unique().tolist()
        if worlds:
            n_worlds = int(worlds[0])
    rates_run_id = None
    if "rates_run_id" in unified.columns:
        values = [v for v in unified["rates_run_id"].dropna().unique().tolist() if v]
        rates_run_id = str(values[0]) if values else None

    summary_payload = {
        "game_date": str(game_date),
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "projections_run_id": projections_run_id,
        "minutes_run_id": resolved_minutes_run_id,
        "ownership_run_id": ownership_run_id,
        "sim_run_id": resolved_sim_run_id,
        "rates_run_id": rates_run_id,
        "sim_profile": sim_profile,
        "n_worlds": n_worlds,
        "minutes_source": minutes_meta.get("minutes_source"),
        "minutes_path": minutes_meta.get("minutes_path"),
        "sim_source": sim_meta.get("sim_source"),
        "sim_path": sim_meta.get("sim_path"),
        "draft_group_id": draft_group_id,
        "rows": int(len(unified)),
        "sim_rows_matched": int(unified["dk_fpts_mean"].notna().sum()) if "dk_fpts_mean" in unified.columns else 0,
    }
    _atomic_write_json(out_dir / SUMMARY_FILENAME, summary_payload)
    
    # Update latest_run.json
    latest_pointer = out_dir.parent / "latest_run.json"
    _atomic_write_json(latest_pointer, {"run_id": projections_run_id})
    
    print(f"[finalize] Saved unified projections ({len(unified)} players) to {out_path}")
    
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Finalize unified projections artifact")
    parser.add_argument("--date", required=True, help="Game date (YYYY-MM-DD)")
    parser.add_argument("--run-id", required=True, help="Projections (output) run identifier")
    parser.add_argument(
        "--minutes-run-id",
        default=None,
        help="Explicit minutes run identifier (defaults to latest_run.json when available).",
    )
    parser.add_argument(
        "--sim-run-id",
        default=None,
        help="Explicit sim run identifier (defaults to sim worlds latest_run.json when available).",
    )
    parser.add_argument(
        "--ownership-run-id",
        default=None,
        help="Explicit ownership run identifier (defaults to --run-id).",
    )
    parser.add_argument(
        "--allow-legacy-sim-projections-root",
        action="store_true",
        help=(
            "Allow falling back to artifacts/sim_v2/projections if worlds_fpts_v2 projections are missing "
            "(disabled by default to avoid stale/post-processed sim outputs)."
        ),
    )
    parser.add_argument("--draft-group-id", required=True, help="DraftKings draft group ID")
    parser.add_argument("--data-root", default=None, help="Data root path")
    args = parser.parse_args()
    
    game_date = date.fromisoformat(args.date)
    root = Path(args.data_root) if args.data_root else data_path()
    
    result = finalize_projections(
        game_date,
        args.run_id,
        args.draft_group_id,
        root,
        minutes_run_id=args.minutes_run_id,
        sim_run_id=args.sim_run_id,
        ownership_run_id=args.ownership_run_id,
        allow_legacy_sim_projections_root=bool(args.allow_legacy_sim_projections_root),
    )
    
    if result is None:
        print("[finalize] Failed to create unified projections")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
