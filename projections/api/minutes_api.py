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

DEFAULT_DAILY_ROOT = Path("artifacts/minutes_v1/daily")
DEFAULT_DASHBOARD_DIST = Path("web/minutes-dashboard/dist")
DEFAULT_FPTS_ROOT = paths.data_path("gold", "projections_fpts_v1")
DEFAULT_SIM_ROOT = paths.data_path("artifacts", "sim_v2", "projections")
LATEST_POINTER = "latest_run.json"
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
    # Ownership/DFS columns
    "salary",
    "pred_own_pct",
    "value",
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


def _load_unified_projections(day: date, run_id: str | None, data_root: Path) -> pd.DataFrame | None:
    """
    Load unified projections artifact (contains minutes, sim, ownership in one file).
    
    Returns None if not available, allowing fallback to legacy loading.
    """
    unified_root = data_root / "artifacts" / "projections" / str(day)
    
    if run_id:
        run_dir = unified_root / f"run={run_id}"
    else:
        # Try latest pointer
        latest_pointer = unified_root / "latest_run.json"
        if latest_pointer.exists():
            try:
                with open(latest_pointer) as f:
                    latest = json.load(f).get("run_id")
                if latest:
                    run_dir = unified_root / f"run={latest}"
                else:
                    return None
            except Exception:
                return None
        else:
            # Fall back to most recent run dir
            run_dirs = sorted(
                [p for p in unified_root.iterdir() if p.is_dir() and p.name.startswith("run=")],
                reverse=True,
            )
            run_dir = run_dirs[0] if run_dirs else None
    
    if run_dir is None or not run_dir.exists():
        return None
    
    parquet_path = run_dir / "projections.parquet"
    if not parquet_path.exists():
        return None
    
    try:
        return pd.read_parquet(parquet_path)
    except Exception:
        return None

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


def _load_sim_projections(day: date, root: Path) -> pd.DataFrame | None:
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
        # Check for projections.parquet directly (new format from run_sim_live.py)
        direct_path = base / "projections.parquet"
        if direct_path.exists():
            try:
                return pd.read_parquet(direct_path)
            except Exception:
                pass
        # Check for run dirs with sim_v2_projections.parquet (legacy aggregator)
        run_dir = _resolve_sim_run_dir(base) if base.is_dir() else None
        if run_dir:
            candidate_path = (
                run_dir if run_dir.is_file() and run_dir.suffix == ".parquet" else run_dir / SIM_PROJECTIONS_FILENAME
            )
            if candidate_path.exists():
                try:
                    return pd.read_parquet(candidate_path)
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
            candidate = run_dir if run_dir.is_file() else run_dir / SIM_PROJECTIONS_FILENAME
            if candidate.exists():
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

    @app.get("/api/minutes")
    def get_minutes(date: str | None = None, run_id: str | None = None) -> JSONResponse:
        slate_day = _parse_date(date)
        
        # Try unified projections artifact first (contains minutes, sim, ownership)
        data_root = paths.data_path()
        unified_df = _load_unified_projections(slate_day, run_id, data_root)
        
        if unified_df is not None and not unified_df.empty:
            # Rename sim columns to match expected dashboard format
            rename_map = {}
            for col in unified_df.columns:
                if col in ("minutes_mean",):
                    rename_map[col] = "sim_minutes_sim_mean"
                elif col.startswith("dk_fpts_") or col in ("pts_mean", "reb_mean", "ast_mean", "stl_mean", "blk_mean", "tov_mean"):
                    rename_map[col] = f"sim_{col}"
            if rename_map:
                unified_df = unified_df.rename(columns=rename_map)
            
            players = _serialize_players(unified_df)
            payload = {"date": slate_day.isoformat(), "count": len(players), "players": players}
            return JSONResponse(payload)
        
        # Fall back to legacy loading (minutes + fpts + sim joins)
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
        sim_df = _load_sim_projections(slate_day, sim_root)
        if sim_df is not None:
            join_keys = ["game_date", "game_id", "team_id", "player_id"]
            missing_keys = [key for key in join_keys if key not in df.columns or key not in sim_df.columns]
            if not missing_keys:
                df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
                sim_df["game_date"] = pd.to_datetime(sim_df["game_date"]).dt.normalize()
                rename_map = {}
                for col in sim_df.columns:
                    if col in join_keys:
                        continue
                    if col in ("minutes_sim_mean", "minutes_mean"):
                        # Map both old and new column names to the expected dashboard field
                        rename_map[col] = "sim_minutes_sim_mean"
                    elif col.startswith("sim_"):
                        rename_map[col] = col
                    else:
                        rename_map[col] = f"sim_{col}"
                df = df.merge(sim_df.rename(columns=rename_map), on=join_keys, how="left")
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
        day_dir = minutes_root / slate_day.isoformat()
        if not day_dir.exists():
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
        pointer = day_dir / LATEST_POINTER
        latest = None
        if pointer.exists():
            try:
                payload = json.loads(pointer.read_text(encoding="utf-8"))
                latest = payload.get("run_id")
            except json.JSONDecodeError:
                latest = None
        payload = {"date": slate_day.isoformat(), "latest": latest, "runs": runs}
        return JSONResponse(payload)

    @app.get("/api/ownership")
    def get_ownership(date: str | None = None) -> JSONResponse:
        """Return ownership predictions for a date."""
        slate_day = _parse_date(date)
        
        # Load from silver/ownership_predictions
        data_root = paths.data_path()
        own_path = data_root / "silver" / "ownership_predictions" / f"{slate_day}.parquet"
        
        if not own_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"No ownership predictions for {slate_day}"
            )
        
        df = pd.read_parquet(own_path)
        
        # Build response with relevant columns
        output_cols = ["player_id", "player_name", "salary", "pos", "team", "proj_fpts", "pred_own_pct"]
        available_cols = [c for c in output_cols if c in df.columns]
        
        players = df[available_cols].to_dict(orient="records")
        
        payload = {
            "date": slate_day.isoformat(),
            "count": len(players),
            "players": players,
        }
        return JSONResponse(jsonable_encoder(payload))

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
