"""
Build an in-house ownership training base from DK actual ownership plus our own pre-lock artifacts.

Inputs:
    bronze/dk_contests/ownership_by_slate/*.parquet
    gold/dk_salaries/site=dk/game_date=YYYY-MM-DD/draft_group_id=*/salaries.parquet
    live/features_minutes_v1/YYYY-MM-DD/run=*/features.parquet
    artifacts/projections/YYYY-MM-DD/run=*/projections.parquet
    artifacts/sim_v2/worlds_fpts_v2/game_date=YYYY-MM-DD/run=*/projections.parquet

Output:
    gold/ownership_inhouse_base/ownership_inhouse_base.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
import pandas as pd

from projections.paths import data_path


def normalize_name(val: object) -> str:
    """Normalize player name for cross-source matching."""
    if val is None or pd.isna(val):
        return ""
    normalized = unicodedata.normalize("NFKD", str(val))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", ascii_only).strip().lower()
    return re.sub(r"\s+", " ", cleaned)


@dataclass(frozen=True)
class SlateMatch:
    """Best-effort mapping between a DK ownership slate and a DK salary slate."""

    dk_slate_id: str
    dk_game_date: str
    salary_slate_id: str | None
    salary_game_date: str | None
    dk_players: int
    salary_players: int
    intersection: int
    recall_dk: float
    recall_salary: float
    overlap_coeff: float
    jaccard: float
    date_offset_days: int | None


def load_dk_ownership(dk_ownership_path: Path) -> pd.DataFrame:
    """Load all DK ownership parquet files except the combined aggregate file."""
    all_files = sorted(f for f in dk_ownership_path.glob("*.parquet") if not f.name.startswith("all_"))
    if not all_files:
        raise FileNotFoundError(f"No ownership files found in {dk_ownership_path}")

    dfs: list[pd.DataFrame] = []
    for file_path in all_files:
        try:
            dfs.append(pd.read_parquet(file_path))
        except Exception as exc:
            print(f"[ownership_inhouse] Warning: failed to read {file_path.name}: {exc}")

    if not dfs:
        raise RuntimeError(f"Failed to load any ownership parquet files from {dk_ownership_path}")
    result = pd.concat(dfs, ignore_index=True)
    result["player_name_norm"] = result["Player"].astype(str).map(normalize_name)
    result["game_date"] = result["game_date"].astype(str)
    result["slate_id"] = result["slate_id"].astype(str)
    return result


def _candidate_game_dates(game_date_str: str, *, max_day_offset: int = 1) -> list[str]:
    base = date.fromisoformat(game_date_str)
    return [(base + timedelta(days=offset)).isoformat() for offset in range(-max_day_offset, max_day_offset + 1)]


def load_salary_slates_for_dates(
    salaries_root: Path,
    game_dates: list[str],
    *,
    contests_root: Path | None = None,
    main_only: bool = False,
) -> pd.DataFrame:
    """Load DK salary slates for one or more candidate dates."""
    frames: list[pd.DataFrame] = []
    for game_date_str in game_dates:
        allowed_draft_groups = None
        if main_only and contests_root is not None:
            allowed_draft_groups = _load_main_draft_group_ids_for_date(contests_root, game_date_str)
            if not allowed_draft_groups:
                continue
        day_root = salaries_root / f"game_date={game_date_str}"
        if not day_root.exists():
            continue
        for slate_dir in sorted(day_root.glob("draft_group_id=*")):
            draft_group_id = slate_dir.name.split("=", 1)[1]
            if allowed_draft_groups is not None and draft_group_id not in allowed_draft_groups:
                continue
            salaries_path = slate_dir / "salaries.parquet"
            if not salaries_path.exists():
                continue
            df = pd.read_parquet(salaries_path)
            if df.empty:
                continue
            out = df.copy()
            out["salary_slate_id"] = draft_group_id
            out["salary_game_date"] = game_date_str
            if "display_name" in out.columns:
                out["player_name"] = out["display_name"].astype(str)
            else:
                out["player_name"] = ""
            out["player_name_norm"] = out["player_name"].map(normalize_name)
            if "team_abbrev" in out.columns:
                out["team"] = out["team_abbrev"].astype(str)
            else:
                out["team"] = ""
            if "positions" in out.columns:
                out["pos"] = out["positions"].map(_coerce_positions)
            else:
                out["pos"] = ""
            out["salary"] = pd.to_numeric(out["salary"] if "salary" in out.columns else None, errors="coerce")
            out["dk_player_id"] = pd.to_numeric(
                out["dk_player_id"] if "dk_player_id" in out.columns else None,
                errors="coerce",
            ).astype("Int64")
            out = out[out["player_name_norm"].ne("") & out["salary"].notna()].copy()
            if "is_disabled" in out.columns:
                out = out[~out["is_disabled"].fillna(False)].copy()
            frames.append(
                out[
                    [
                        "salary_slate_id",
                        "salary_game_date",
                        "dk_player_id",
                        "player_name",
                        "player_name_norm",
                        "team",
                        "pos",
                        "salary",
                    ]
                ].drop_duplicates(["salary_slate_id", "player_name_norm"], keep="last")
            )

    if not frames:
        return pd.DataFrame(
            columns=[
                "salary_slate_id",
                "salary_game_date",
                "dk_player_id",
                "player_name",
                "player_name_norm",
                "team",
                "pos",
                "salary",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _select_main_draft_group_ids_from_meta(meta: pd.DataFrame) -> set[str]:
    """Pick the main classic draft group for a date using contest metadata."""
    if meta.empty or "draft_group_id" not in meta.columns:
        return set()

    working = meta.copy()
    if "game_type" in working.columns:
        working = working[working["game_type"].astype(str).str.lower() == "classic"].copy()
    if working.empty:
        return set()

    if "contest_name" in working.columns:
        contest_name = working["contest_name"].astype(str)
        excluded = contest_name.str.contains(
            r"showdown|single[- ]?game|captain|tiers|pick ?6|head[- ]?to[- ]?head|double up|50/50",
            case=False,
            regex=True,
            na=False,
        )
        working = working[~excluded].copy()
    if working.empty:
        return set()

    for col in ["prize_pool", "current_entries", "max_entries"]:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce").fillna(0.0)

    grouped = (
        working.groupby("draft_group_id", as_index=False)
        .agg(
            prize_pool_sum=("prize_pool", "sum"),
            current_entries_sum=("current_entries", "sum"),
            max_entries_sum=("max_entries", "sum"),
            contest_count=("draft_group_id", "size"),
        )
        .sort_values(
            ["prize_pool_sum", "current_entries_sum", "max_entries_sum", "contest_count", "draft_group_id"],
            ascending=[False, False, False, False, True],
        )
    )
    if grouped.empty:
        return set()
    draft_group_id = grouped.iloc[0]["draft_group_id"]
    try:
        return {str(int(draft_group_id))}
    except Exception:
        return {str(draft_group_id)}


def _load_main_draft_group_ids_for_date(contests_root: Path, game_date_str: str) -> set[str]:
    meta_path = contests_root / game_date_str / f"nba_gpp_{game_date_str}.csv"
    if not meta_path.exists():
        return set()
    try:
        meta = pd.read_csv(meta_path)
    except Exception:
        return set()
    return _select_main_draft_group_ids_from_meta(meta)


def _coerce_positions(val: object) -> str:
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, (list, tuple)):
        return "/".join(str(item) for item in val)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val)


def _build_slate_index(df: pd.DataFrame, *, slate_col: str, game_date_col: str) -> pd.DataFrame:
    """Build per-slate player sets for overlap-based slate matching."""
    working = df[[slate_col, game_date_col, "player_name_norm"]].copy()
    working[slate_col] = working[slate_col].astype(str)
    working[game_date_col] = working[game_date_col].astype(str)
    working["player_name_norm"] = working["player_name_norm"].astype(str)
    working = working[working["player_name_norm"].ne("")].copy()
    idx = (
        working.groupby(slate_col, sort=False)
        .agg(
            game_date=(game_date_col, "first"),
            player_set=("player_name_norm", lambda s: set(s.tolist())),
            n_players=("player_name_norm", "size"),
        )
        .reset_index()
    )
    return idx


def match_dk_slates_to_salary_slates(
    dk_own: pd.DataFrame,
    salary_slates: pd.DataFrame,
    *,
    max_day_offset: int = 1,
    min_overlap_coeff: float = 0.85,
    min_intersection: int = 20,
    min_recall_dk: float = 0.75,
) -> tuple[pd.DataFrame, list[SlateMatch]]:
    """Map each DK ownership slate to the best matching DK salary slate by player overlap."""
    salary_idx = _build_slate_index(
        salary_slates,
        slate_col="salary_slate_id",
        game_date_col="salary_game_date",
    )
    by_date: dict[str, list[tuple[str, set[str], int]]] = {}
    for _, row in salary_idx.iterrows():
        by_date.setdefault(str(row["game_date"]), []).append(
            (str(row["salary_slate_id"]), set(row["player_set"]), int(row["n_players"]))
        )

    mapping: dict[str, str] = {}
    matches: list[SlateMatch] = []
    dk = dk_own[["slate_id", "game_date", "player_name_norm"]].copy()
    dk["slate_id"] = dk["slate_id"].astype(str)
    dk["game_date"] = dk["game_date"].astype(str)

    for dk_slate_id, group in dk.groupby("slate_id", sort=False):
        dk_date = str(group["game_date"].iloc[0])
        dk_players = set(group["player_name_norm"].tolist())
        dk_n = len(dk_players)

        best: tuple[float, float, int, int, float, str, str, int, float, float, int] | None = None
        for candidate_date in _candidate_game_dates(dk_date, max_day_offset=max_day_offset):
            for salary_slate_id, salary_players, salary_n in by_date.get(candidate_date, []):
                intersection = len(dk_players & salary_players)
                if intersection == 0:
                    continue
                overlap = intersection / min(dk_n, salary_n)
                jaccard = intersection / len(dk_players | salary_players)
                recall_dk = intersection / dk_n if dk_n else 0.0
                recall_salary = intersection / salary_n if salary_n else 0.0
                offset_days = (date.fromisoformat(candidate_date) - date.fromisoformat(dk_date)).days
                candidate = (
                    recall_dk,
                    overlap,
                    intersection,
                    -abs(offset_days),
                    jaccard,
                    salary_slate_id,
                    candidate_date,
                    salary_n,
                    recall_dk,
                    recall_salary,
                    offset_days,
                )
                if best is None or candidate[:4] > best[:4]:
                    best = candidate

        if best is None:
            matches.append(
                SlateMatch(
                    dk_slate_id=dk_slate_id,
                    dk_game_date=dk_date,
                    salary_slate_id=None,
                    salary_game_date=None,
                    dk_players=dk_n,
                    salary_players=0,
                    intersection=0,
                    recall_dk=0.0,
                    recall_salary=0.0,
                    overlap_coeff=0.0,
                    jaccard=0.0,
                    date_offset_days=None,
                )
            )
            continue

        recall_dk, overlap, intersection, _, jaccard, salary_slate_id, salary_date, salary_n, _, recall_salary, offset = best
        if recall_dk >= min_recall_dk and overlap >= min_overlap_coeff and intersection >= min_intersection:
            mapping[dk_slate_id] = salary_slate_id
            chosen_slate_id: str | None = salary_slate_id
            chosen_game_date: str | None = salary_date
        else:
            chosen_slate_id = None
            chosen_game_date = None

        matches.append(
            SlateMatch(
                dk_slate_id=dk_slate_id,
                dk_game_date=dk_date,
                salary_slate_id=chosen_slate_id,
                salary_game_date=chosen_game_date,
                dk_players=dk_n,
                salary_players=salary_n,
                intersection=intersection,
                recall_dk=float(recall_dk),
                recall_salary=float(recall_salary),
                overlap_coeff=float(overlap),
                jaccard=float(jaccard),
                date_offset_days=int(offset),
            )
        )

    out = dk_own.copy()
    out["salary_slate_id"] = out["slate_id"].map(mapping)
    return out, matches


def _load_dk_draft_group_lock_ts(*, draft_group_id: str, root: Path) -> datetime | None:
    """Load first lock timestamp for a DK draft group from bronze draftables."""
    path = root / "bronze" / "dk" / "draftables" / f"draftables_raw_{draft_group_id}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    competitions = payload.get("competitions") if isinstance(payload, dict) else None
    if not isinstance(competitions, list):
        return None
    starts = [row.get("startTime") for row in competitions if isinstance(row, dict) and row.get("startTime")]
    parsed = pd.to_datetime(pd.Series(starts), utc=True, errors="coerce").dropna()
    if parsed.empty:
        return None
    lock_ts = parsed.min()
    return lock_ts.to_pydatetime() if hasattr(lock_ts, "to_pydatetime") else lock_ts


def _resolve_run_dir(base_dir: Path, *, cutoff_ts: datetime | None) -> Path | None:
    if not base_dir.exists() or not base_dir.is_dir():
        return None
    best_dt: datetime | None = None
    best_dir: Path | None = None
    for path in base_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("run="):
            continue
        run_id = path.name.split("=", 1)[1]
        try:
            run_ts = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except ValueError:
            continue
        if cutoff_ts is not None and run_ts > cutoff_ts:
            continue
        if best_dt is None or run_ts > best_dt:
            best_dt = run_ts
            best_dir = path
    if best_dir is not None:
        return best_dir

    latest_pointer = base_dir / "latest_run.json"
    if latest_pointer.exists():
        try:
            payload = json.loads(latest_pointer.read_text(encoding="utf-8"))
            run_id = payload.get("run_id")
            if run_id:
                candidate = base_dir / f"run={run_id}"
                if candidate.exists():
                    return candidate
        except Exception:
            pass
    return None


def load_minutes_context(
    *,
    minutes_root: Path,
    game_date_str: str,
    cutoff_ts: datetime | None,
) -> pd.DataFrame:
    """Load pre-lock minutes features and derive injury plus betting context."""
    run_dir = _resolve_run_dir(minutes_root / game_date_str, cutoff_ts=cutoff_ts)
    if run_dir is None:
        return pd.DataFrame(
            columns=[
                "player_name_norm",
                "team",
                "nba_player_id",
                "player_is_out",
                "player_is_questionable",
                "team_outs_count",
                "total_close",
                "spread_close",
                "team_implied_total",
            ]
        )

    features_path = run_dir / "features.parquet"
    if not features_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(features_path)
    if df.empty:
        return df

    out = df.copy()
    out["player_name_norm"] = out["player_name"].astype(str).map(normalize_name)
    out["team"] = out.get("team_tricode", "").astype(str)
    out["nba_player_id"] = pd.to_numeric(out.get("player_id"), errors="coerce").astype("Int64")
    out["player_is_out"] = out.get("is_out", 0).fillna(0).astype(int)
    out["player_is_questionable"] = (
        out.get("is_q", 0).fillna(0).astype(int) | out.get("is_prob", 0).fillna(0).astype(int)
    ).astype(int)
    out["team_outs_count"] = out.groupby("team")["player_is_out"].transform("sum").astype(int)
    out["total_close"] = pd.to_numeric(out.get("total"), errors="coerce")
    spread_home = pd.to_numeric(out.get("spread_home"), errors="coerce")
    home_flag = out.get("home_flag", 0).fillna(0).astype(int)
    out["spread_close"] = spread_home.where(home_flag.eq(1), -spread_home)
    out["team_implied_total"] = (out["total_close"] / 2.0) - (out["spread_close"] / 2.0)
    cols = [
        "player_name_norm",
        "team",
        "nba_player_id",
        "player_is_out",
        "player_is_questionable",
        "team_outs_count",
        "total_close",
        "spread_close",
        "team_implied_total",
    ]
    return out[cols].drop_duplicates(["player_name_norm", "team"], keep="last")


def load_projection_context(
    *,
    projections_root: Path,
    sim_root: Path,
    game_date_str: str,
    cutoff_ts: datetime | None,
) -> pd.DataFrame:
    """Load pre-lock projection context, preferring unified projections over raw sim outputs."""
    projections_day = projections_root / game_date_str
    run_dir = _resolve_run_dir(projections_day, cutoff_ts=cutoff_ts)
    frame: pd.DataFrame | None = None
    if run_dir is not None and (run_dir / "projections.parquet").exists():
        frame = pd.read_parquet(run_dir / "projections.parquet")
    else:
        sim_day = sim_root / f"game_date={game_date_str}"
        sim_run_dir = _resolve_run_dir(sim_day, cutoff_ts=cutoff_ts)
        if sim_run_dir is not None:
            for candidate in (sim_run_dir / "projections.parquet", sim_run_dir / "sim_v2_projections.parquet"):
                if candidate.exists():
                    frame = pd.read_parquet(candidate)
                    break

    if frame is None or frame.empty:
        return pd.DataFrame(columns=["nba_player_id", "proj_fpts"])

    out = frame.copy()
    out["nba_player_id"] = pd.to_numeric(out.get("player_id"), errors="coerce").astype("Int64")
    out["proj_fpts"] = pd.to_numeric(out.get("dk_fpts_mean"), errors="coerce")
    extra_cols = [
        "play_prob_eff",
        "minutes_mean",
        "minutes_sim_mean",
        "dk_fpts_p90",
        "dk_fpts_p50",
        "sim_p_active",
    ]
    keep = ["nba_player_id", "proj_fpts"] + [col for col in extra_cols if col in out.columns]
    return out[keep].dropna(subset=["nba_player_id"]).drop_duplicates("nba_player_id", keep="last")


def build_training_rows_for_slate(
    *,
    salary_slate: pd.DataFrame,
    dk_labels: pd.DataFrame,
    minutes_context: pd.DataFrame,
    projection_context: pd.DataFrame,
    dk_slate_id: str,
    game_date_str: str,
) -> pd.DataFrame:
    """Assemble one training slate using the full salary universe and zero-filled ownership labels."""
    slate = salary_slate.copy()
    labels = dk_labels[["player_name_norm", "own_pct"]].drop_duplicates("player_name_norm", keep="last").copy()
    labels = labels.rename(columns={"own_pct": "actual_own_pct"})
    slate = slate.merge(labels, on="player_name_norm", how="left")
    slate["actual_own_pct"] = pd.to_numeric(slate["actual_own_pct"], errors="coerce").fillna(0.0)

    if not minutes_context.empty:
        slate = slate.merge(minutes_context, on=["player_name_norm", "team"], how="left")
    else:
        slate["nba_player_id"] = pd.Series(pd.array([pd.NA] * len(slate), dtype="Int64"))

    if not projection_context.empty:
        slate = slate.merge(projection_context, on="nba_player_id", how="left")

    slate["proj_fpts"] = pd.to_numeric(slate.get("proj_fpts"), errors="coerce")
    slate["proj_fpts"] = slate["proj_fpts"].fillna(slate["salary"] / 200.0)

    for col in ["player_is_out", "player_is_questionable", "team_outs_count"]:
        if col not in slate.columns:
            slate[col] = 0
        slate[col] = pd.to_numeric(slate[col], errors="coerce").fillna(0).astype(int)

    slate["player_id"] = slate["nba_player_id"].map(_stable_player_id)
    missing_player_id = slate["player_id"].eq("")
    if missing_player_id.any():
        slate.loc[missing_player_id, "player_id"] = (
            "dk:"
            + slate.loc[missing_player_id, "dk_player_id"].astype("string").fillna("")
        )
    missing_player_id = slate["player_id"].eq("dk:")
    if missing_player_id.any():
        slate.loc[missing_player_id, "player_id"] = (
            "name:"
            + slate.loc[missing_player_id, "player_name_norm"]
            + ":"
            + slate.loc[missing_player_id, "team"]
        )

    slate["season"] = slate["salary_game_date"].map(_season_from_game_date)
    slate["slate_id"] = dk_slate_id
    slate["game_date"] = game_date_str
    slate["data_source"] = "dk_inhouse"
    return slate[
        [
            "season",
            "slate_id",
            "game_date",
            "player_id",
            "player_name",
            "player_name_norm",
            "team",
            "pos",
            "salary",
            "proj_fpts",
            "player_is_out",
            "player_is_questionable",
            "team_outs_count",
            "total_close",
            "spread_close",
            "team_implied_total",
            "actual_own_pct",
            "data_source",
            "salary_slate_id",
            "salary_game_date",
        ]
        + [col for col in ["play_prob_eff", "minutes_mean", "minutes_sim_mean", "dk_fpts_p90", "dk_fpts_p50", "sim_p_active"] if col in slate.columns]
    ].copy()


def _stable_player_id(val: object) -> str:
    if val is None or pd.isna(val):
        return ""
    try:
        return str(int(val))
    except Exception:
        return str(val)


def _season_from_game_date(game_date_str: str) -> int:
    day = date.fromisoformat(game_date_str)
    return day.year if day.month >= 10 else day.year - 1


def build_inhouse_base(
    *,
    dk_ownership_path: Path,
    salaries_root: Path,
    minutes_root: Path,
    projections_root: Path,
    sim_root: Path,
    output_path: Path,
    root: Path,
    contests_root: Path,
    min_overlap_coeff: float = 0.85,
    min_intersection: int = 20,
    min_recall_dk: float = 0.75,
    min_label_sum: float = 650.0,
    max_label_sum: float = 850.0,
    start_date: str | None = None,
    end_date: str | None = None,
    main_only: bool = False,
) -> pd.DataFrame:
    """Build the in-house ownership base from our own pre-lock artifacts."""
    dk_own = load_dk_ownership(dk_ownership_path)
    if start_date is not None:
        dk_own = dk_own[dk_own["game_date"] >= start_date].copy()
    if end_date is not None:
        dk_own = dk_own[dk_own["game_date"] <= end_date].copy()
    all_rows: list[pd.DataFrame] = []
    minutes_cache: dict[tuple[str, str | None], pd.DataFrame] = {}
    projections_cache: dict[tuple[str, str | None], pd.DataFrame] = {}

    matched_slates = 0
    skipped_slates = 0
    skipped_label_sum = 0

    for game_date_str, dk_date_df in dk_own.groupby("game_date", sort=True):
        candidate_dates = _candidate_game_dates(game_date_str, max_day_offset=1)
        salary_slates = load_salary_slates_for_dates(
            salaries_root,
            candidate_dates,
            contests_root=contests_root,
            main_only=main_only,
        )
        if salary_slates.empty:
            print(f"[ownership_inhouse] No salary slates for {game_date_str}")
            continue

        mapped, matches = match_dk_slates_to_salary_slates(
            dk_date_df,
            salary_slates,
            max_day_offset=1,
            min_overlap_coeff=min_overlap_coeff,
            min_intersection=min_intersection,
            min_recall_dk=min_recall_dk,
        )

        match_map = {match.dk_slate_id: match for match in matches}
        for dk_slate_id, slate_labels in mapped.groupby("slate_id", sort=False):
            raw_own_sum = float(pd.to_numeric(slate_labels["own_pct"], errors="coerce").fillna(0.0).sum())
            if raw_own_sum < min_label_sum or raw_own_sum > max_label_sum:
                skipped_label_sum += 1
                continue
            match = match_map.get(str(dk_slate_id))
            if match is None or match.salary_slate_id is None or match.salary_game_date is None:
                skipped_slates += 1
                continue

            salary_slate = salary_slates[salary_slates["salary_slate_id"] == match.salary_slate_id].copy()
            if salary_slate.empty:
                skipped_slates += 1
                continue

            lock_ts = _load_dk_draft_group_lock_ts(draft_group_id=match.salary_slate_id, root=root)
            cutoff_key = lock_ts.isoformat() if lock_ts is not None else None
            minutes_key = (match.salary_game_date, cutoff_key)
            if minutes_key not in minutes_cache:
                minutes_cache[minutes_key] = load_minutes_context(
                    minutes_root=minutes_root,
                    game_date_str=match.salary_game_date,
                    cutoff_ts=lock_ts,
                )
            projection_key = (match.salary_game_date, cutoff_key)
            if projection_key not in projections_cache:
                projections_cache[projection_key] = load_projection_context(
                    projections_root=projections_root,
                    sim_root=sim_root,
                    game_date_str=match.salary_game_date,
                    cutoff_ts=lock_ts,
                )

            rows = build_training_rows_for_slate(
                salary_slate=salary_slate,
                dk_labels=slate_labels,
                minutes_context=minutes_cache[minutes_key],
                projection_context=projections_cache[projection_key],
                dk_slate_id=str(dk_slate_id),
                game_date_str=game_date_str,
            )
            all_rows.append(rows)
            matched_slates += 1

    if not all_rows:
        raise RuntimeError("No matched slates were assembled for the in-house ownership base")

    result = pd.concat(all_rows, ignore_index=True)
    before = len(result)
    result = result[result["actual_own_pct"] <= 98.0].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    print(f"[ownership_inhouse] Matched slates: {matched_slates}")
    print(f"[ownership_inhouse] Skipped slates: {skipped_slates}")
    print(f"[ownership_inhouse] Skipped on raw label sum: {skipped_label_sum}")
    print(f"[ownership_inhouse] Rows written: {len(result):,} (filtered {before - len(result):,} rows > 98% own)")
    print(f"[ownership_inhouse] Output: {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build in-house ownership training base")
    parser.add_argument(
        "--dk-ownership-path",
        type=Path,
        default=data_path() / "bronze" / "dk_contests" / "ownership_by_slate",
    )
    parser.add_argument(
        "--salaries-root",
        type=Path,
        default=data_path() / "gold" / "dk_salaries" / "site=dk",
    )
    parser.add_argument(
        "--contests-root",
        type=Path,
        default=data_path() / "bronze" / "dk_contests" / "nba_gpp_data",
    )
    parser.add_argument(
        "--minutes-root",
        type=Path,
        default=data_path() / "live" / "features_minutes_v1",
    )
    parser.add_argument(
        "--projections-root",
        type=Path,
        default=data_path() / "artifacts" / "projections",
    )
    parser.add_argument(
        "--sim-root",
        type=Path,
        default=data_path() / "artifacts" / "sim_v2" / "worlds_fpts_v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=data_path() / "gold" / "ownership_inhouse_base" / "ownership_inhouse_base.parquet",
    )
    parser.add_argument("--min-overlap-coeff", type=float, default=0.85)
    parser.add_argument("--min-intersection", type=int, default=20)
    parser.add_argument("--min-recall-dk", type=float, default=0.75)
    parser.add_argument("--min-label-sum", type=float, default=650.0)
    parser.add_argument("--max-label-sum", type=float, default=850.0)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--main-only", action="store_true")
    args = parser.parse_args()

    build_inhouse_base(
        dk_ownership_path=args.dk_ownership_path,
        salaries_root=args.salaries_root,
        minutes_root=args.minutes_root,
        projections_root=args.projections_root,
        sim_root=args.sim_root,
        output_path=args.output,
        root=data_path(),
        contests_root=args.contests_root,
        min_overlap_coeff=float(args.min_overlap_coeff),
        min_intersection=int(args.min_intersection),
        min_recall_dk=float(args.min_recall_dk),
        min_label_sum=float(args.min_label_sum),
        max_label_sum=float(args.max_label_sum),
        start_date=args.start_date,
        end_date=args.end_date,
        main_only=bool(args.main_only),
    )


if __name__ == "__main__":
    main()
