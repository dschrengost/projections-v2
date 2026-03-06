"""FastAPI router for post-contest flashback endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from projections import paths
from projections.api.contest_service import parse_contest_csv
from projections.post_contest import (
    build_post_contest_replay_analytics,
    build_replay_calibration_artifacts,
    find_latest_export_manifest,
)

router = APIRouter()


def _preview_parquet(path: Path, limit: int = 12) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_parquet(path).head(limit)
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _user_entries_path() -> Path:
    return paths.data_path("analytics", "contest_results", "user_entries.parquet")


def _raw_contest_date_dir(date: str) -> Path:
    return paths.data_path("bronze", "dk_contests", "nba_gpp_data", date)


def _normalize_identifier(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace("$", "").replace(",", "").replace("%", "")
    if not text or text.lower() == "nan":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _load_raw_contest_meta_map(date: str) -> Dict[str, Dict[str, Any]]:
    date_dir = _raw_contest_date_dir(date)
    csv_path = date_dir / f"nba_gpp_{date}.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if df.empty or "contest_id" not in df.columns:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        contest_id = _normalize_identifier(row.get("contest_id"))
        if not contest_id:
            continue
        out[contest_id] = {
            "contest_name": row.get("contest_name"),
            "draft_group_id": int(row["draft_group_id"]) if pd.notna(row.get("draft_group_id")) else None,
            "entry_fee": _safe_float(row.get("entry_fee")),
            "current_entries": int(row["current_entries"]) if pd.notna(row.get("current_entries")) else None,
        }
    return out


def _list_flashback_contests_from_raw(date: str, user_pattern: str) -> List[FlashbackContestSummaryResponse]:
    results_dir = _raw_contest_date_dir(date) / "results"
    if not results_dir.exists():
        return []
    meta_map = _load_raw_contest_meta_map(date)
    entry_name_cols = ("EntryName", "Entry Name", "entry_name")
    entry_id_cols = ("EntryId", "Entry ID", "entry_id")
    rank_cols = ("Rank", "rank")
    prize_cols = ("Prize", "Winnings", "prize", "Payout")

    out: List[FlashbackContestSummaryResponse] = []
    pattern = str(user_pattern)
    result_paths = {
        *results_dir.glob("contest_*_results.csv"),
        *results_dir.glob("contest_*_standings.csv"),
    }
    for results_path in sorted(result_paths):
        contest_id = (
            results_path.stem.replace("contest_", "").replace("_results", "").replace("_standings", "")
        )
        contest_id = _normalize_identifier(contest_id)
        if not contest_id:
            continue
        try:
            df = parse_contest_csv(results_path)
        except Exception:
            continue
        if df.empty:
            continue
        entry_name_col = next((col for col in entry_name_cols if col in df.columns), None)
        if entry_name_col is None:
            continue
        matches = df[df[entry_name_col].astype(str).str.contains(pattern, case=False, na=False)].copy()
        if matches.empty:
            continue
        entry_id_col = next((col for col in entry_id_cols if col in matches.columns), None)
        if entry_id_col is not None:
            matches = matches.drop_duplicates(subset=[entry_id_col])
        else:
            matches = matches.drop_duplicates(subset=[entry_name_col])
        rank_col = next((col for col in rank_cols if col in matches.columns), None)
        prize_col = next((col for col in prize_cols if col in matches.columns), None)
        meta = meta_map.get(contest_id, {})
        draft_group_id = meta.get("draft_group_id")
        manifest_available = False
        if draft_group_id:
            manifest_available = (
                find_latest_export_manifest(
                    game_date=date,
                    draft_group_id=int(draft_group_id),
                    contest_id=contest_id,
                    data_root=paths.data_path(),
                )
                is not None
            )
        out.append(
            FlashbackContestSummaryResponse(
                game_date=date,
                contest_id=contest_id,
                contest_name=str(meta.get("contest_name") or f"Contest {contest_id}"),
                draft_group_id=draft_group_id,
                entry_fee=meta.get("entry_fee"),
                entry_count=int(len(matches)),
                best_rank=int(matches[rank_col].min()) if rank_col and matches[rank_col].notna().any() else None,
                best_prize=max(
                    (_safe_float(value) for value in matches[prize_col].tolist()),
                    default=None,
                )
                if prize_col
                else None,
                candidate_manifest_available=manifest_available,
            )
        )
    out.sort(key=lambda item: ((item.best_rank if item.best_rank is not None else 10**9), -item.entry_count, item.contest_name))
    return out


class FlashbackContestSummaryResponse(BaseModel):
    game_date: str
    contest_id: str
    contest_name: str
    draft_group_id: Optional[int] = None
    entry_fee: Optional[float] = None
    entry_count: int
    best_rank: Optional[int] = None
    best_prize: Optional[float] = None
    candidate_manifest_available: bool = False


class FlashbackRunRequest(BaseModel):
    game_date: str
    contest_id: str
    user_pattern: str
    draft_group_id: Optional[int] = None
    run_id: Optional[str] = None
    entry_fee: Optional[float] = None
    archetype: str = Field(default="medium")
    worlds_source: str = Field(default="gtv2")
    ownership_mode: str = Field(default="field_only")
    modeled_field_version: str = Field(default="v1_calibrated")
    include_modeled_field: bool = True


class FlashbackRunResponse(BaseModel):
    summary: Dict[str, Any] = Field(default_factory=dict)
    previews: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)


class FlashbackCalibrationResponse(BaseModel):
    summary: Dict[str, Any] = Field(default_factory=dict)
    previews: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)


@router.get("/contests", response_model=List[FlashbackContestSummaryResponse])
async def list_flashback_contests(date: str, user_pattern: str) -> List[FlashbackContestSummaryResponse]:
    path = _user_entries_path()
    if path.exists():
        try:
            df = pd.read_parquet(
                path,
                columns=[
                    "date",
                    "contest_id",
                    "draft_group_id",
                    "contest_name",
                    "entry_fee",
                    "entry_name",
                    "rank",
                    "prize_pool",
                    "first_place_prize",
                ],
            )
            frame = df[
                (df["date"].astype(str) == str(date))
                & (df["entry_name"].astype(str).str.contains(user_pattern, case=False, na=False))
            ].copy()
            if not frame.empty:
                grouped = (
                    frame.groupby(["date", "contest_id", "contest_name", "draft_group_id", "entry_fee"], dropna=False)
                    .agg(
                        entry_count=("entry_name", "size"),
                        best_rank=("rank", "min"),
                    )
                    .reset_index()
                    .sort_values(["best_rank", "entry_count"], ascending=[True, False])
                )
                out: List[FlashbackContestSummaryResponse] = []
                for _, row in grouped.iterrows():
                    draft_group_id = int(row["draft_group_id"]) if pd.notna(row["draft_group_id"]) else None
                    manifest_available = False
                    if draft_group_id:
                        manifest_available = (
                            find_latest_export_manifest(
                                game_date=str(row["date"]),
                                draft_group_id=draft_group_id,
                                contest_id=str(row["contest_id"]),
                                data_root=paths.data_path(),
                            )
                            is not None
                        )
                    out.append(
                        FlashbackContestSummaryResponse(
                            game_date=str(row["date"]),
                            contest_id=str(row["contest_id"]),
                            contest_name=str(row["contest_name"]),
                            draft_group_id=draft_group_id,
                            entry_fee=float(row["entry_fee"]) if pd.notna(row["entry_fee"]) else None,
                            entry_count=int(row["entry_count"]),
                            best_rank=int(row["best_rank"]) if pd.notna(row["best_rank"]) else None,
                            best_prize=None,
                            candidate_manifest_available=manifest_available,
                        )
                    )
                return out
        except Exception:
            pass
    return _list_flashback_contests_from_raw(date, user_pattern)


@router.post("/run", response_model=FlashbackRunResponse)
async def run_flashback(request: FlashbackRunRequest) -> FlashbackRunResponse:
    try:
        bundle = build_post_contest_replay_analytics(
            contest_id=request.contest_id,
            game_date=request.game_date,
            user_pattern=request.user_pattern,
            draft_group_id=request.draft_group_id,
            run_id=request.run_id,
            entry_fee=request.entry_fee,
            archetype=request.archetype,
            worlds_source=request.worlds_source,
            ownership_mode=request.ownership_mode,
            data_root=paths.data_path(),
            modeled_field_version=request.modeled_field_version,
            include_modeled_field=request.include_modeled_field,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    summary = _load_json(bundle.summary_path)
    previews = {
        "player_calibration": _preview_parquet(bundle.player_calibration_path, limit=15),
        "lineup_calibration": _preview_parquet(bundle.lineup_calibration_path, limit=15),
        "field_calibration": _preview_parquet(bundle.field_calibration_path, limit=5),
        "regret_summary": _preview_parquet(bundle.regret_summary_path, limit=5),
    }
    return FlashbackRunResponse(summary=summary, previews=previews)


@router.post("/calibration/run", response_model=FlashbackCalibrationResponse)
async def run_flashback_calibration() -> FlashbackCalibrationResponse:
    try:
        bundle = build_replay_calibration_artifacts(data_root=paths.data_path())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    summary = _load_json(bundle.summary_path)
    previews = {
        "player_fpts_calibration": _preview_parquet(bundle.player_fpts_calibration_path, limit=12),
        "player_minutes_calibration": _preview_parquet(bundle.player_minutes_calibration_path, limit=12),
        "ownership_recalibration": _preview_parquet(bundle.ownership_recalibration_path, limit=12),
        "field_model_calibration": _preview_parquet(bundle.field_model_calibration_path, limit=12),
        "optimizer_regret_by_bucket": _preview_parquet(bundle.optimizer_regret_by_bucket_path, limit=12),
    }
    return FlashbackCalibrationResponse(summary=summary, previews=previews)
