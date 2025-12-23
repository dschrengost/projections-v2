"""FastAPI app serving minutes projections and the React dashboard."""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from projections import paths
from projections.api.pipeline_status_api import router as pipeline_status_router
from projections.api.evaluation_api import router as evaluation_router
from projections.api.optimizer_api import router as optimizer_router
from projections.api.contest_api import router as contest_router
from projections.api.contest_sim_api import router as contest_sim_router
from projections.api.entry_manager_api import router as entry_manager_router
from projections.api.diagnostics_api import router as diagnostics_router

DEFAULT_DAILY_ROOT = paths.data_path("artifacts", "minutes_v1", "daily")
DEFAULT_DASHBOARD_DIST = Path("web/minutes-dashboard/dist")
DEFAULT_FPTS_ROOT = paths.data_path("gold", "projections_fpts_v1")
DEFAULT_SIM_ROOT = paths.data_path("artifacts", "sim_v2", "worlds_fpts_v2")
LATEST_POINTER = "latest_run.json"
PINNED_POINTER = "pinned_run.json"
PARQUET_FILENAME = "minutes.parquet"
SUMMARY_FILENAME = "summary.json"
FPTS_FILENAME = "fpts.parquet"
SIM_PROJECTIONS_FILENAME = "sim_v2_projections.parquet"

# Only expose conditional minutes fields to the UI.
PLAYER_COLUMNS: tuple[str, ...] = (
    "game_date",
    "tip_ts",
    "game_id",
    "player_id",
    "player_name",
    "status",
    "team_id",
    "team_name",
    "team_tricode",
    "opponent_team_id",
    "opponent_team_name",
    "opponent_team_tricode",
    "starter_flag",
    "is_projected_starter",
    "is_confirmed_starter",
    "pos_bucket",
    "play_prob",
    "minutes_p10",
    "minutes_p50",
    "minutes_p90",
    "minutes_p10_cond",
    "minutes_p50_cond",
    "minutes_p90_cond",
    "fpts_per_min_pred",
    "proj_fpts",
    "scoring_system",
    "spread_home",
    "total",
    "odds_as_of_ts",
    "blowout_index",
    "blowout_risk_score",
    "close_game_score",
    "team_implied_total",
    "opponent_implied_total",
    "sim_dk_fpts_mean",
    "sim_dk_fpts_std",
    "sim_dk_fpts_p05",
    "sim_dk_fpts_p10",
    "sim_dk_fpts_p25",
    "sim_dk_fpts_p50",
    "sim_dk_fpts_p75",
    "sim_dk_fpts_p90",
    "sim_dk_fpts_p95",
    "sim_pts_mean",
    "sim_reb_mean",
    "sim_ast_mean",
    "sim_stl_mean",
    "sim_blk_mean",
    "sim_tov_mean",
    "sim_minutes_sim_mean",
    "sim_minutes_sim_p10",
    "sim_minutes_sim_p50",
    "sim_minutes_sim_p90",
    "sim_minutes_sim_std",
    # Ownership/DFS columns
    "salary",
    "pred_own_pct",
    "value",
    "is_locked",
)


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default.expanduser().resolve()
    return Path(raw).expanduser().resolve()


def _parse_date(value: str | None) -> date:
    if not value:
        return date.today()
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.") from exc


def _resolve_run_dir(day_dir: Path, run_id: str | None) -> Path:
    if not day_dir.exists():
        raise HTTPException(status_code=404, detail="No artifact for selected date.")

    if run_id:
        candidate = day_dir / f"run={run_id}"
        if candidate.exists():
            return candidate
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found for {day_dir.name}.")

    pointer = day_dir / LATEST_POINTER
    if pointer.exists():
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            latest = payload.get("run_id")
            if latest:
                candidate = day_dir / f"run={latest}"
                if candidate.exists():
                    return candidate
        except json.JSONDecodeError:
            pass

    direct_parquet = day_dir / PARQUET_FILENAME
    if direct_parquet.exists():
        return day_dir

    run_dirs = sorted(
        [path for path in day_dir.iterdir() if path.is_dir() and path.name.startswith("run=")],
        reverse=True,
    )
    if run_dirs:
        return run_dirs[0]

    raise HTTPException(status_code=404, detail="No artifact for selected date.")


def _load_minutes(run_dir: Path) -> pd.DataFrame:
    parquet_path = run_dir / PARQUET_FILENAME
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail="No artifact for selected date.")
    return pd.read_parquet(parquet_path)


def _load_summary(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / SUMMARY_FILENAME
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _extract_run_name(run_dir: Path) -> str | None:
    if run_dir.name.startswith("run="):
        return run_dir.name.split("=", 1)[1]
    return None


def _load_unified_projections(
    day: date, run_id: str | None, data_root: Path
) -> tuple[pd.DataFrame | None, str | None, str | None]:
    """
    Load unified projections artifact (contains minutes, sim, ownership in one file).
    
    Returns (dataframe, run_id, updated_at) tuple.
    Returns (None, None, None) if not available, allowing fallback to legacy loading.
    """
    unified_root = data_root / "artifacts" / "projections" / str(day)
    
    resolved_run_id: str | None = None
    updated_at: str | None = None
    
    if run_id:
        run_dir = unified_root / f"run={run_id}"
        resolved_run_id = run_id
    else:
        # Prefer a pinned projections run when present so rescores can be inspected
        # without being overwritten by the live pipeline updating latest_run.json.
        pinned_pointer = unified_root / PINNED_POINTER
        if pinned_pointer.exists():
            try:
                pinned_data = json.loads(pinned_pointer.read_text(encoding="utf-8"))
            except Exception:
                pinned_data = {}
            pinned_run_id = pinned_data.get("run_id") if isinstance(pinned_data, dict) else None
            if pinned_run_id:
                candidate_dir = unified_root / f"run={pinned_run_id}"
                candidate_path = candidate_dir / "projections.parquet"
                if candidate_path.exists():
                    resolved_run_id = pinned_run_id
                    updated_at = pinned_data.get("updated_at") or pinned_data.get("pinned_at")
                    try:
                        df = pd.read_parquet(candidate_path)
                        return df, resolved_run_id, updated_at
                    except Exception:
                        return None, None, None

        # Try latest pointer
        latest_pointer = unified_root / "latest_run.json"
        if latest_pointer.exists():
            try:
                with open(latest_pointer) as f:
                    pointer_data = json.load(f)
                    resolved_run_id = pointer_data.get("run_id")
                    updated_at = pointer_data.get("updated_at") or pointer_data.get("run_as_of_ts")
                if resolved_run_id:
                    run_dir = unified_root / f"run={resolved_run_id}"
                else:
                    return None, None, None
            except Exception:
                return None, None, None
        else:
            # Fall back to most recent run dir
            if not unified_root.exists():
                return None, None, None
            run_dirs = sorted(
                [p for p in unified_root.iterdir() if p.is_dir() and p.name.startswith("run=")],
                reverse=True,
            )
            if run_dirs:
                run_dir = run_dirs[0]
                resolved_run_id = run_dir.name.split("=", 1)[1] if run_dir.name.startswith("run=") else None
            else:
                run_dir = None
    
    if run_dir is None or not run_dir.exists():
        return None, None, None
    
    parquet_path = run_dir / "projections.parquet"
    if not parquet_path.exists():
        return None, None, None
    
    # Try to get updated_at from file mtime if not from pointer
    if not updated_at:
        try:
            mtime = parquet_path.stat().st_mtime
            updated_at = datetime.fromtimestamp(mtime).isoformat() + "Z"
        except Exception:
            pass
    
    try:
        df = pd.read_parquet(parquet_path)
        return df, resolved_run_id, updated_at
    except Exception:
        return None, None, None

def _load_fpts(day: date, run_name: str | None, root: Path) -> pd.DataFrame | None:
    if run_name is None:
        return None
    day_dir = root / day.isoformat()
    run_dir = day_dir / f"run={run_name}"
    parquet_path = run_dir / FPTS_FILENAME
    if not parquet_path.exists():
        return None
    try:
        return pd.read_parquet(parquet_path)
    except Exception:
        return None


def _resolve_sim_run_dir(base_dir: Path) -> Path | None:
    pointer = base_dir / LATEST_POINTER
    if pointer.exists():
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            run_id = payload.get("run_id")
            if run_id:
                candidate = base_dir / f"run={run_id}"
                if candidate.exists():
                    return candidate
        except json.JSONDecodeError:
            pass

    run_dirs = sorted(
        [path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("run=")],
        reverse=True,
    )
    if run_dirs:
        return run_dirs[0]
    return base_dir if (base_dir / SIM_PROJECTIONS_FILENAME).exists() else None


def _resolve_ownership_run_dir(base_dir: Path, run_id: str | None) -> Path | None:
    if run_id:
        candidate = base_dir / f"run={run_id}"
        return candidate if candidate.exists() else None

    pointer = base_dir / LATEST_POINTER
    if pointer.exists():
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        latest = payload.get("run_id")
        if latest:
            candidate = base_dir / f"run={latest}"
            if candidate.exists():
                return candidate
    if base_dir.exists():
        run_dirs = sorted(
            [path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("run=")],
            reverse=True,
        )
        if run_dirs:
            return run_dirs[0]
    return None


def _load_ownership_predictions(
    day: date,
    data_root: Path,
    *,
    run_id: str | None = None,
) -> pd.DataFrame | None:
    """Load ownership predictions for a date, returning the main slate."""
    base_dir = data_root / "silver" / "ownership_predictions" / str(day)
    slate_dir = _resolve_ownership_run_dir(base_dir, run_id) or base_dir

    if slate_dir.exists():
        slate_files = list(slate_dir.glob("*.parquet"))
        slate_files = [f for f in slate_files if not f.name.endswith("_locked.parquet")]
        if slate_files:
            # Use largest slate (main slate)
            own_path = max(slate_files, key=lambda f: f.stat().st_size)
            try:
                return pd.read_parquet(own_path)
            except Exception:
                pass

    if run_id is not None:
        return None

    # Fall back to legacy single-file format
    legacy_path = data_root / "silver" / "ownership_predictions" / f"{day}.parquet"
    if legacy_path.exists():
        try:
            return pd.read_parquet(legacy_path)
        except Exception:
            pass

    return None


def _load_sim_projections(day: date, root: Path, *, minutes_run_id: str | None = None) -> pd.DataFrame | None:
    candidates = [
        root / f"game_date={day.isoformat()}",  # New format (run_sim_live.py)
        root / f"date={day.isoformat()}",       # Legacy format
        root / day.isoformat(),
    ]
    for base in candidates:
        if not base.exists():
            continue
        if base.is_file() and base.suffix == ".parquet":
            try:
                return pd.read_parquet(base)
            except Exception:
                continue
        if minutes_run_id and base.is_dir():
            run_dir = base / f"run={minutes_run_id}"
            if run_dir.exists():
                candidates = [run_dir / "projections.parquet", run_dir / SIM_PROJECTIONS_FILENAME]
                for candidate_path in candidates:
                    if not candidate_path.exists():
                        continue
                    try:
                        df = pd.read_parquet(candidate_path)
                        return df
                    except Exception:
                        continue
        # Check for projections.parquet directly (new format from run_sim_live.py)
        direct_path = base / "projections.parquet"
        if direct_path.exists():
            try:
                df = pd.read_parquet(direct_path)
                if minutes_run_id:
                    if "minutes_run_id" not in df.columns:
                        continue
                    df = df.loc[df["minutes_run_id"].astype(str) == str(minutes_run_id)].copy()
                    if df.empty:
                        continue
                return df
            except Exception:
                pass
        # Check for run dirs with sim_v2_projections.parquet (legacy aggregator)
        run_dir = _resolve_sim_run_dir(base) if base.is_dir() else None
        if run_dir:
            candidates = []
            if run_dir.is_file() and run_dir.suffix == ".parquet":
                candidates.append(run_dir)
            else:
                candidates.extend([run_dir / "projections.parquet", run_dir / SIM_PROJECTIONS_FILENAME])
            for candidate_path in candidates:
                if not candidate_path.exists():
                    continue
                try:
                    df = pd.read_parquet(candidate_path)
                    if minutes_run_id and "minutes_run_id" in df.columns:
                        df = df.loc[df["minutes_run_id"].astype(str) == str(minutes_run_id)].copy()
                        if df.empty:
                            continue
                    return df
                except Exception:
                    continue
    return None


def _sim_projections_available(day: date, root: Path) -> bool:
    candidates = [
        root / f"game_date={day.isoformat()}",  # New format (run_sim_live.py)
        root / f"date={day.isoformat()}",       # Legacy format
        root / day.isoformat(),
    ]
    for base in candidates:
        if not base.exists():
            continue
        if base.is_file() and base.suffix == ".parquet":
            return True
        # Check for projections.parquet directly (new format)
        if (base / "projections.parquet").exists():
            return True
        if base.is_dir():
            run_dir = _resolve_sim_run_dir(base)
            if run_dir is None:
                continue
            candidates = []
            if run_dir.is_file() and run_dir.suffix == ".parquet":
                candidates.append(run_dir)
            else:
                candidates.extend([run_dir / "projections.parquet", run_dir / SIM_PROJECTIONS_FILENAME])
            if any(candidate.exists() for candidate in candidates):
                return True
    return False


def _serialize_players(df: pd.DataFrame) -> list[dict[str, Any]]:
    available_cols = [col for col in PLAYER_COLUMNS if col in df.columns]
    trimmed = df.loc[:, available_cols].copy()
    # Replace NaN/inf so JSON serialization doesn’t explode.
    trimmed = trimmed.replace([float("inf"), float("-inf")], pd.NA)
    trimmed = trimmed.astype(object).where(pd.notna(trimmed), None)
    trimmed = trimmed.replace({pd.NA: None, float("inf"): None, float("-inf"): None})
    # Prefer the conditional minutes; drop any unconditional columns to keep the UI focused.
    return list(jsonable_encoder(trimmed.to_dict(orient="records")))


def _build_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {"rows": 0, "players": 0, "teams": 0}
    return {
        "rows": int(len(df)),
        "players": int(df["player_id"].nunique()) if "player_id" in df.columns else 0,
        "teams": int(df["team_id"].nunique()) if "team_id" in df.columns else 0,
    }


def create_app(
    *,
    daily_root: Path | None = None,
    dashboard_dist: Path | None = None,
    fpts_root: Path | None = None,
    sim_root: Path | None = None,
) -> FastAPI:
    """Construct the FastAPI app."""

    minutes_root = (daily_root or _env_path("MINUTES_DAILY_ROOT", DEFAULT_DAILY_ROOT)).resolve()
    dist_dir = (dashboard_dist or _env_path("MINUTES_DASHBOARD_DIST", DEFAULT_DASHBOARD_DIST)).resolve()
    fpts_root = (fpts_root or _env_path("MINUTES_FPTS_ROOT", DEFAULT_FPTS_ROOT)).resolve()
    sim_root = (sim_root or _env_path("MINUTES_SIM_ROOT", DEFAULT_SIM_ROOT)).resolve()

    app = FastAPI(title="Minutes API", version="0.1.0")
    app.include_router(pipeline_status_router, prefix="/api")
    app.include_router(evaluation_router, prefix="/api")
    app.include_router(optimizer_router, prefix="/api/optimizer", tags=["optimizer"])
    app.include_router(contest_router, prefix="/api/contest", tags=["contest"])
    app.include_router(contest_sim_router, prefix="/api/contest-sim", tags=["contest-sim"])
    app.include_router(entry_manager_router, prefix="/api/entry-manager", tags=["entry-manager"])
    app.include_router(diagnostics_router)

    @app.get("/api/minutes")
    def get_minutes(date: str | None = None, run_id: str | None = None) -> JSONResponse:
        slate_day = _parse_date(date)

        # Source of truth: unified projections artifact (minutes + sim + ownership).
        # `run_id` is interpreted as projections_run_id when loading unified projections.
        data_root = paths.data_path()
        unified_root = data_root / "artifacts" / "projections" / str(slate_day)
        pinned_run_id = None
        latest_run_id = None
        if unified_root.exists():
            pinned_pointer = unified_root / PINNED_POINTER
            if pinned_pointer.exists():
                try:
                    pinned_payload = json.loads(pinned_pointer.read_text(encoding="utf-8"))
                    if isinstance(pinned_payload, dict):
                        pinned_run_id = pinned_payload.get("run_id")
                except Exception:
                    pinned_run_id = None
            latest_pointer = unified_root / LATEST_POINTER
            if latest_pointer.exists():
                try:
                    latest_payload = json.loads(latest_pointer.read_text(encoding="utf-8"))
                    if isinstance(latest_payload, dict):
                        latest_run_id = latest_payload.get("run_id")
                except Exception:
                    latest_run_id = None

        unified_df, resolved_run_id, updated_at = _load_unified_projections(slate_day, run_id, data_root)

        if unified_df is not None and not unified_df.empty:
            # Rename sim columns to match expected dashboard format
            has_sim_minutes = any(
                col in unified_df.columns
                for col in (
                    "minutes_sim_mean",
                    "minutes_sim_p50",
                    "minutes_sim_p10",
                    "minutes_sim_p90",
                    "minutes_sim_std",
                )
            )
            rename_map = {}
            for col in unified_df.columns:
                if col == "minutes_mean":
                    if has_sim_minutes:
                        continue
                    rename_map[col] = "sim_minutes_sim_mean"
                elif col.startswith("minutes_sim_"):
                    rename_map[col] = f"sim_{col}"
                elif col.startswith("dk_fpts_") or col in ("pts_mean", "reb_mean", "ast_mean", "stl_mean", "blk_mean", "tov_mean"):
                    rename_map[col] = f"sim_{col}"
            if rename_map:
                unified_df = unified_df.rename(columns=rename_map)
            
            players = _serialize_players(unified_df)
            run_summary = None
            if resolved_run_id:
                run_dir = unified_root / f"run={resolved_run_id}"
                run_summary = _load_summary(run_dir)
            payload = {
                "date": slate_day.isoformat(),
                "count": len(players),
                "players": players,
                "run_id": resolved_run_id,
                "last_updated": updated_at,
                "latest_run_id": latest_run_id,
                "pinned_run_id": pinned_run_id,
                "run_summary": run_summary,
            }
            return JSONResponse(payload)

        # Fall back to legacy loading (minutes + fpts + sim joins).
        day_dir = minutes_root / slate_day.isoformat()
        run_dir = _resolve_run_dir(day_dir, run_id)
        df = _load_minutes(run_dir)
        fpts_df = _load_fpts(slate_day, _extract_run_name(run_dir), fpts_root)
        if fpts_df is not None and {"game_id", "player_id"}.issubset(fpts_df.columns):
            join_candidates = (
                "game_id",
                "player_id",
                "fpts_per_min_pred",
                "proj_fpts",
                "scoring_system",
                "team_implied_total",
                "opponent_implied_total",
            )
            join_cols = [col for col in join_candidates if col in fpts_df.columns]
            if len(join_cols) > 2:
                df = df.merge(
                    fpts_df.loc[:, join_cols],
                    on=["game_id", "player_id"],
                    how="left",
                )
        sim_df = _load_sim_projections(
            slate_day,
            sim_root,
            minutes_run_id=_extract_run_name(run_dir),
        )
        if sim_df is not None:
            join_keys = ["game_date", "game_id", "team_id", "player_id"]
            missing_keys = [key for key in join_keys if key not in df.columns or key not in sim_df.columns]
            if not missing_keys:
                df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
                sim_df["game_date"] = pd.to_datetime(sim_df["game_date"]).dt.normalize()
                rename_map = {}
                has_sim_minutes = any(
                    col in sim_df.columns
                    for col in (
                        "minutes_sim_mean",
                        "minutes_sim_p50",
                        "minutes_sim_p10",
                        "minutes_sim_p90",
                        "minutes_sim_std",
                    )
                )
                for col in sim_df.columns:
                    if col in join_keys:
                        continue
                    if col == "minutes_mean":
                        if has_sim_minutes:
                            continue
                        rename_map[col] = "sim_minutes_sim_mean"
                    elif col.startswith("sim_"):
                        rename_map[col] = col
                    else:
                        rename_map[col] = f"sim_{col}"
                df = df.merge(sim_df.rename(columns=rename_map), on=join_keys, how="left")

        # Merge ownership predictions (includes salary)
        # Note: ownership uses DK player_ids, minutes uses NBA player_ids, so join on normalized name+team
        own_df = _load_ownership_predictions(slate_day, data_root, run_id=_extract_run_name(run_dir))
        if own_df is not None and "player_name" in own_df.columns and "player_name" in df.columns:
            import unicodedata
            def _norm_name(s: str) -> str:
                if not s:
                    return ""
                normalized = unicodedata.normalize("NFKD", str(s))
                return normalized.encode("ascii", "ignore").decode("ascii").lower().strip()

            # Create normalized join keys
            own_df = own_df.copy()
            own_df["_join_name"] = own_df["player_name"].apply(_norm_name)
            if "team" in own_df.columns:
                own_df["_join_team"] = own_df["team"].str.upper()
            df["_join_name"] = df["player_name"].apply(_norm_name)
            if "team_tricode" in df.columns:
                df["_join_team"] = df["team_tricode"].str.upper()

            # Select columns for merge
            own_merge_cols = ["_join_name", "_join_team", "salary", "pred_own_pct"]
            if "is_locked" in own_df.columns:
                own_merge_cols.append("is_locked")
            own_merge_cols = [c for c in own_merge_cols if c in own_df.columns]

            join_cols = ["_join_name", "_join_team"] if "_join_team" in df.columns else ["_join_name"]
            join_cols = [c for c in join_cols if c in own_df.columns]

            if join_cols:
                df = df.merge(own_df[own_merge_cols].drop_duplicates(subset=join_cols), on=join_cols, how="left")
                # Clean up temp columns
                df = df.drop(columns=["_join_name", "_join_team"], errors="ignore")

        players = _serialize_players(df)
        payload = {"date": slate_day.isoformat(), "count": len(players), "players": players}
        return JSONResponse(payload)

    @app.get("/api/minutes/meta")
    def get_minutes_meta(date: str | None = None, run_id: str | None = None) -> JSONResponse:
        slate_day = _parse_date(date)
        day_dir = minutes_root / slate_day.isoformat()
        run_dir = _resolve_run_dir(day_dir, run_id)
        summary = _load_summary(run_dir)
        if summary is None:
            df = _load_minutes(run_dir)
            summary = {
                "date": slate_day.isoformat(),
                "counts": _build_counts(df),
                "run_id": run_dir.name.split("=", 1)[1] if run_dir.name.startswith("run=") else None,
                "bundle_dir": None,
                "model_run_id": None,
                "run_as_of_ts": None,
            }
            # Ensure generated_at always present for operators.
            summary.setdefault("generated_at", datetime.utcnow().isoformat() + "Z")
            summary.setdefault("date", slate_day.isoformat())
            summary.setdefault("counts", _build_counts(pd.DataFrame()))
        fpts_run_name = _extract_run_name(run_dir)
        fpts_summary = None
        if fpts_run_name:
            fpts_run_dir = fpts_root / slate_day.isoformat() / f"run={fpts_run_name}"
            if fpts_run_dir.exists():
                fpts_summary = _load_summary(fpts_run_dir)
        summary["fpts_available"] = fpts_summary is not None
        if fpts_summary:
            summary["fpts_meta"] = fpts_summary
        summary["sim_available"] = _sim_projections_available(slate_day, sim_root)
        return JSONResponse(summary)

    @app.get("/api/minutes/runs")
    def list_runs(date: str | None = None) -> JSONResponse:
        """List available run_ids for a given date."""

        slate_day = _parse_date(date)
        data_root = paths.data_path()
        
        # Check multiple path formats in order of preference
        candidate_paths = [
            data_root / "artifacts" / "projections" / str(slate_day),  # Unified projections
            minutes_root / f"game_date={slate_day.isoformat()}",  # New minutes format
            minutes_root / slate_day.isoformat(),  # Legacy format
        ]
        
        day_dir = None
        for candidate in candidate_paths:
            if candidate.exists():
                day_dir = candidate
                break
        
        if day_dir is None:
            raise HTTPException(status_code=404, detail="No artifact for selected date.")
        
        runs: list[dict[str, Any]] = []
        for entry in sorted(day_dir.iterdir()):
            if not (entry.is_dir() and entry.name.startswith("run=")):
                continue
            run_name = entry.name.split("=", 1)[1]
            meta = _load_summary(entry) or {}
            runs.append(
                {
                    "run_id": run_name,
                    "run_as_of_ts": meta.get("run_as_of_ts"),
                    "generated_at": meta.get("generated_at"),
                }
            )
        latest = None
        pointer = day_dir / LATEST_POINTER
        if pointer.exists():
            try:
                payload = json.loads(pointer.read_text(encoding="utf-8"))
                latest = payload.get("run_id")
            except json.JSONDecodeError:
                latest = None

        pinned = None
        pinned_pointer = day_dir / PINNED_POINTER
        if pinned_pointer.exists():
            try:
                payload = json.loads(pinned_pointer.read_text(encoding="utf-8"))
                pinned = payload.get("run_id") if isinstance(payload, dict) else None
            except json.JSONDecodeError:
                pinned = None

        payload = {"date": slate_day.isoformat(), "latest": latest, "pinned": pinned, "runs": runs}
        return JSONResponse(payload)

    @app.get("/api/ownership")
    def get_ownership(
        date: str | None = None,
        draft_group_id: str | None = None,
        run_id: str | None = None,
    ) -> JSONResponse:
        """Return ownership predictions for a date.
        
        If draft_group_id specified, returns that slate's predictions.
        Otherwise, returns the main slate (largest by player count).
        """
        slate_day = _parse_date(date)
        data_root = paths.data_path()
        
        # Check for per-slate structure first (new format)
        base_dir = data_root / "silver" / "ownership_predictions" / str(slate_day)
        run_dir = _resolve_ownership_run_dir(base_dir, run_id) if base_dir.exists() else None
        if run_id is not None and run_dir is None:
            raise HTTPException(status_code=404, detail=f"No ownership predictions for run {run_id}")
        slate_dir = run_dir or base_dir
        
        if slate_dir.exists():
            # New per-slate format
            slate_files = list(slate_dir.glob("*.parquet"))
            slate_files = [f for f in slate_files if not f.name.endswith("_locked.parquet")]
            
            if not slate_files:
                if run_dir is not None:
                    raise HTTPException(status_code=404, detail=f"No ownership predictions for {slate_day}")
                slate_files = []
            
            if slate_files:
                if draft_group_id:
                    # Use specified slate
                    own_path = slate_dir / f"{draft_group_id}.parquet"
                    if not own_path.exists():
                        raise HTTPException(status_code=404, detail=f"No predictions for slate {draft_group_id}")
                else:
                    # Find main slate (largest by file size as proxy for player count)
                    own_path = max(slate_files, key=lambda f: f.stat().st_size)
            else:
                own_path = None
        else:
            own_path = None

        if own_path is None:
            # Fall back to legacy single-file format
            own_path = data_root / "silver" / "ownership_predictions" / f"{slate_day}.parquet"
            if not own_path.exists():
                raise HTTPException(status_code=404, detail=f"No ownership predictions for {slate_day}")
        
        df = pd.read_parquet(own_path)
        
        # Build response
        output_cols = ["player_id", "player_name", "salary", "pos", "team", "proj_fpts", "pred_own_pct", "is_locked", "draft_group_id"]
        available_cols = [c for c in output_cols if c in df.columns]
        
        if "is_locked" not in df.columns:
            df["is_locked"] = False
            available_cols.append("is_locked")
        
        players = df[available_cols].to_dict(orient="records")
        
        payload = {
            "date": slate_day.isoformat(),
            "draft_group_id": df["draft_group_id"].iloc[0] if "draft_group_id" in df.columns else None,
            "count": len(players),
            "players": players,
            "run_id": _extract_run_name(run_dir) if run_dir else None,
        }
        return JSONResponse(jsonable_encoder(payload))

    @app.get("/api/ownership/slates")
    def list_ownership_slates(date: str | None = None, run_id: str | None = None) -> JSONResponse:
        """List available slates for a date with lock status."""
        import json
        
        slate_day = _parse_date(date)
        data_root = paths.data_path()
        base_dir = data_root / "silver" / "ownership_predictions" / str(slate_day)
        
        if not base_dir.exists():
            # Check for legacy format
            legacy_path = data_root / "silver" / "ownership_predictions" / f"{slate_day}.parquet"
            if legacy_path.exists():
                # Single legacy slate
                df = pd.read_parquet(legacy_path)
                return JSONResponse({
                    "date": slate_day.isoformat(),
                    "slates": [{
                        "draft_group_id": "legacy",
                        "player_count": len(df),
                        "is_locked": df["is_locked"].any() if "is_locked" in df.columns else False,
                    }]
                })
            raise HTTPException(status_code=404, detail=f"No ownership predictions for {slate_day}")

        run_dir = _resolve_ownership_run_dir(base_dir, run_id)
        if run_id is not None and run_dir is None:
            raise HTTPException(status_code=404, detail=f"No ownership predictions for run {run_id}")
        slate_dir = run_dir or base_dir
        
        # Try to load slates.json metadata
        meta_path = slate_dir / "slates.json"
        if meta_path.exists():
            with open(meta_path) as f:
                slates_meta = json.load(f)
            slates = [
                {"draft_group_id": dg_id, **info}
                for dg_id, info in slates_meta.items()
            ]
        elif run_dir is not None and (base_dir / "slates.json").exists():
            with open(base_dir / "slates.json") as f:
                slates_meta = json.load(f)
            slates = [
                {"draft_group_id": dg_id, **info}
                for dg_id, info in slates_meta.items()
            ]
        else:
            # Build from parquet files
            slates = []
            for f in slate_dir.glob("*.parquet"):
                if f.name.endswith("_locked.parquet"):
                    continue
                dg_id = f.stem
                df = pd.read_parquet(f)
                slates.append({
                    "draft_group_id": dg_id,
                    "player_count": len(df),
                    "is_locked": df["is_locked"].any() if "is_locked" in df.columns else False,
                })
        
        # Sort by player count descending (main slate first)
        slates.sort(key=lambda x: x.get("player_count", 0), reverse=True)
        
        return JSONResponse({
            "date": slate_day.isoformat(),
            "slates": slates,
            "run_id": _extract_run_name(run_dir) if run_dir else None,
        })

    @app.post("/api/trigger")
    def trigger_pipeline(background_tasks: BackgroundTasks) -> JSONResponse:
        """Manually trigger the live pipeline (scrape -> score)."""
        background_tasks.add_task(_run_pipeline_background)
        return JSONResponse({"status": "triggered", "message": "Pipeline started in background."})

    if dist_dir.exists():
        app.mount("/", StaticFiles(directory=dist_dir, html=True), name="static")

    return app


def _run_pipeline_background() -> None:
    """Execute scrape and score scripts sequentially."""
    import subprocess
    
    # 1. Scrape
    print("[api] Triggering scrape...")
    scrape_res = subprocess.run(
        ["/bin/bash", "scripts/run_live_scrape.sh"],
        cwd=os.getcwd(),
        capture_output=True,
        text=True
    )
    if scrape_res.returncode != 0:
        print(f"[api] Scrape failed: {scrape_res.stderr}")
        return

    # 2. Score
    print("[api] Triggering score...")
    score_res = subprocess.run(
        ["/bin/bash", "scripts/run_live_score.sh"],
        cwd=os.getcwd(),
        capture_output=True,
        text=True
    )
    if score_res.returncode != 0:
        print(f"[api] Score failed: {score_res.stderr}")
        return
        
    print("[api] Manual pipeline run complete.")



app = create_app()
