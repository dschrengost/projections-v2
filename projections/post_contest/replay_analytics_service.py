from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from projections.api.contest_service import parse_contest_csv
from projections.api.optimizer_service import build_player_pool
from projections.contest_sim.contest_sim_service import (
    load_player_worlds,
    run_contest_simulation,
    score_lineups,
)
from projections.contest_sim.field_library import FieldLibrary
from projections.contest_sim.field_library_manager import load_or_build_field_library
from projections.contest_sim.scoring_models import ContestSimResult, LineupEVResult
from projections.paths import get_data_root
from projections.post_contest.replay_models import ContestReplayRun, PreparedReplayContext
from projections.post_contest.replay_service import (
    _build_name_to_internal_map,
    _resolve_name_to_player_id,
    build_actual_field_library,
    replay_output_dir,
    run_post_contest_replay,
)

SALARY_CAP = 50000.0


@dataclass(frozen=True)
class ReplayAnalyticsBundle:
    player_calibration_path: Path
    lineup_calibration_path: Path
    field_calibration_path: Path
    regret_summary_path: Path
    summary_path: Path

    def to_dict(self) -> Dict[str, str]:
        return {
            "player_calibration_path": str(self.player_calibration_path),
            "lineup_calibration_path": str(self.lineup_calibration_path),
            "field_calibration_path": str(self.field_calibration_path),
            "regret_summary_path": str(self.regret_summary_path),
            "summary_path": str(self.summary_path),
        }


def _lineup_key(player_ids: Sequence[str]) -> str:
    return "|".join(sorted(str(pid) for pid in player_ids if str(pid)))


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    w = np.asarray(list(weights), dtype=np.float64)
    if arr.size == 0 or w.sum() <= 0:
        return float("nan")
    return float(np.average(arr, weights=w))


def _weighted_hist_l1(
    left_values: Sequence[float],
    left_weights: Sequence[float],
    right_values: Sequence[float],
    right_weights: Sequence[float],
    bins: Sequence[float],
) -> float:
    left_hist, _ = np.histogram(
        np.asarray(list(left_values), dtype=np.float64),
        bins=np.asarray(list(bins), dtype=np.float64),
        weights=np.asarray(list(left_weights), dtype=np.float64),
    )
    right_hist, _ = np.histogram(
        np.asarray(list(right_values), dtype=np.float64),
        bins=np.asarray(list(bins), dtype=np.float64),
        weights=np.asarray(list(right_weights), dtype=np.float64),
    )
    left_total = left_hist.sum()
    right_total = right_hist.sum()
    if left_total > 0:
        left_hist = left_hist / left_total
    if right_total > 0:
        right_hist = right_hist / right_total
    return float(np.abs(left_hist - right_hist).sum())


def _actual_percentile(actual_value: Optional[float], worlds: np.ndarray) -> Optional[float]:
    if actual_value is None or worlds.size == 0:
        return None
    return float(np.mean(worlds <= float(actual_value)))


def _load_actual_minutes_lookup(*, game_date: str, data_root: Path) -> Dict[str, float]:
    year = int(str(game_date).split("-")[0])
    path = data_root / "labels" / f"season={year}" / "boxscore_labels.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["game_date", "player_id", "minutes"])
    df = df[df["game_date"].astype(str) == str(game_date)].copy()
    if df.empty:
        return {}
    df["player_id"] = df["player_id"].astype(str)
    return {str(row["player_id"]): float(row["minutes"]) for _, row in df.iterrows() if pd.notna(row["minutes"])}


def _load_actual_player_fpts_lookup(
    *,
    prepared: PreparedReplayContext,
    data_root: Path,
) -> Dict[str, float]:
    if not prepared.meta.results_path:
        return {}
    results_path = Path(prepared.meta.results_path)
    if not results_path.exists():
        return {}
    results_df = parse_contest_csv(results_path)
    if "Player" not in results_df.columns or "FPTS" not in results_df.columns:
        return {}
    resolved_name_map, ambiguous_name_map, _, resolved_signatures = _build_name_to_internal_map(
        game_date=prepared.meta.game_date,
        draft_group_id=int(prepared.meta.draft_group_id or 0),
        data_root=data_root,
        run_id=None,
    )
    lookup: Dict[str, float] = {}
    player_rows = results_df[["Player", "FPTS"]].dropna().drop_duplicates(subset=["Player"])
    for _, row in player_rows.iterrows():
        player_id, diag = _resolve_name_to_player_id(
            raw_name=str(row["Player"]),
            resolved_name_map=resolved_name_map,
            ambiguous_name_map=ambiguous_name_map,
            resolved_signatures=resolved_signatures,
        )
        fpts = _coerce_float(row["FPTS"])
        if player_id is None or fpts is None or (diag and str(diag.get("method", "")).startswith("ambiguous")):
            continue
        lookup[str(player_id)] = float(fpts)
    return lookup


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace("$", "").replace(",", "").replace("%", "")
    if not text or text.lower() == "nan":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def replay_normalize_name(name: str) -> str:
    from projections.post_contest.replay_service import _normalize_name as _inner_normalize_name

    return _inner_normalize_name(name)


def _player_pool_maps(
    *,
    game_date: str,
    draft_group_id: int,
    data_root: Path,
    run_id: Optional[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    pool = build_player_pool(
        game_date=game_date,
        draft_group_id=draft_group_id,
        site="dk",
        run_id=run_id,
        data_root=data_root,
        include_unmatched_salaries=True,
        allow_zero_projections=True,
        exclude_inactive_players=False,
    )
    meta_by_player: Dict[str, Dict[str, Any]] = {}
    name_by_player: Dict[str, str] = {}
    for row in pool:
        player_id = str(row.get("player_id"))
        meta_by_player[player_id] = dict(row)
        name_by_player[player_id] = str(row.get("name") or player_id)
    return meta_by_player, name_by_player


def _count_player_ownership_from_entries(
    entries: Iterable[Sequence[str]],
    *,
    denominator: int,
) -> Dict[str, float]:
    counts: Counter[str] = Counter()
    for lineup in entries:
        for pid in set(str(player_id) for player_id in lineup):
            counts[pid] += 1
    if denominator <= 0:
        return {}
    return {pid: 100.0 * float(count) / float(denominator) for pid, count in counts.items()}


def _count_player_ownership_from_library(library: FieldLibrary) -> Dict[str, float]:
    counts: Counter[str] = Counter()
    total = int(sum(int(w) for w in library.weights))
    if total <= 0:
        return {}
    for lineup, weight in zip(library.lineups, library.weights):
        for pid in set(str(player_id) for player_id in lineup):
            counts[pid] += int(weight)
    return {pid: 100.0 * float(count) / float(total) for pid, count in counts.items()}


def _lineup_features(lineup: Sequence[str], player_meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rows = [player_meta.get(str(pid), {}) for pid in lineup]
    salary_values = [float(row.get("salary") or 0.0) for row in rows]
    own_values = [float(row.get("own_proj") or 0.0) for row in rows]
    teams = [str(row.get("team") or "") for row in rows if row.get("team")]
    games = [str(row.get("game_matchup") or "") for row in rows if row.get("game_matchup")]
    team_counts = Counter(teams)
    game_counts = Counter(games)
    salary_total = float(sum(salary_values))
    return {
        "salary_total": salary_total,
        "salary_left": float(SALARY_CAP - salary_total) if salary_total > 0 else float("nan"),
        "projected_own_sum": float(sum(own_values)),
        "num_teams": int(len(team_counts)),
        "max_from_team": int(max(team_counts.values())) if team_counts else 0,
        "num_games": int(len(game_counts)),
        "max_from_game": int(max(game_counts.values())) if game_counts else 0,
    }


def _simulation_result_map(results: Sequence[LineupEVResult]) -> Dict[str, LineupEVResult]:
    return {_lineup_key(result.player_ids): result for result in results}


def _world_percentile_map(
    *,
    lineups: Sequence[Sequence[str]],
    actual_scores: Dict[str, Optional[float]],
    worlds_matrix: np.ndarray,
    player_index: Dict[str, int],
) -> Dict[str, Optional[float]]:
    if not lineups:
        return {}
    lineup_scores = score_lineups([list(lineup) for lineup in lineups], worlds_matrix, player_index)
    out: Dict[str, Optional[float]] = {}
    for idx, lineup in enumerate(lineups):
        key = _lineup_key(lineup)
        out[key] = _actual_percentile(actual_scores.get(key), lineup_scores[idx, :])
    return out


def _lineup_actual_counts(entries: Sequence[Sequence[str]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for lineup in entries:
        counts[_lineup_key(lineup)] += 1
    return dict(counts)


def _library_features_frame(
    *,
    library: FieldLibrary,
    player_meta: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for lineup, weight in zip(library.lineups, library.weights):
        features = _lineup_features(lineup, player_meta)
        features["lineup_key"] = _lineup_key(lineup)
        features["weight"] = int(weight)
        rows.append(features)
    return pd.DataFrame(rows)


def _field_summary_row(
    *,
    prepared: PreparedReplayContext,
    actual_field: FieldLibrary,
    modeled_field: Optional[FieldLibrary],
    player_meta: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    actual_df = _library_features_frame(library=actual_field, player_meta=player_meta)
    actual_weights = actual_df["weight"].tolist() if not actual_df.empty else []
    actual_own = _count_player_ownership_from_library(actual_field)

    row: Dict[str, Any] = {
        "game_date": prepared.meta.game_date,
        "contest_id": prepared.meta.contest_id,
        "draft_group_id": prepared.meta.draft_group_id,
        "contest_name": prepared.meta.contest_name,
        "actual_field_size": int(sum(actual_field.weights)),
        "actual_unique_lineups": int(len(actual_field.lineups)),
        "actual_dupe_rate": 1.0 - (
            float(len(actual_field.lineups)) / float(sum(actual_field.weights)) if sum(actual_field.weights) > 0 else 0.0
        ),
        "actual_salary_total_mean": _weighted_mean(actual_df["salary_total"].tolist(), actual_weights) if not actual_df.empty else float("nan"),
        "actual_salary_left_mean": _weighted_mean(actual_df["salary_left"].tolist(), actual_weights) if not actual_df.empty else float("nan"),
        "actual_projected_own_sum_mean": _weighted_mean(actual_df["projected_own_sum"].tolist(), actual_weights) if not actual_df.empty else float("nan"),
        "actual_num_teams_mean": _weighted_mean(actual_df["num_teams"].tolist(), actual_weights) if not actual_df.empty else float("nan"),
        "actual_max_from_team_mean": _weighted_mean(actual_df["max_from_team"].tolist(), actual_weights) if not actual_df.empty else float("nan"),
        "modeled_field_version": None,
        "modeled_field_size_weighted": None,
        "modeled_unique_lineups": None,
        "modeled_dupe_rate": None,
        "modeled_salary_total_mean": None,
        "modeled_salary_left_mean": None,
        "modeled_projected_own_sum_mean": None,
        "modeled_num_teams_mean": None,
        "modeled_max_from_team_mean": None,
        "player_ownership_mae_pct": None,
        "player_ownership_rmse_pct": None,
        "top20_player_ownership_mae_pct": None,
        "salary_left_hist_l1": None,
        "projected_own_sum_hist_l1": None,
        "dupe_hist_l1": None,
    }

    if modeled_field is None:
        return row

    modeled_df = _library_features_frame(library=modeled_field, player_meta=player_meta)
    modeled_weights = modeled_df["weight"].tolist() if not modeled_df.empty else []
    modeled_own = _count_player_ownership_from_library(modeled_field)
    ownership_union = sorted(set(actual_own) | set(modeled_own))
    actual_arr = np.asarray([actual_own.get(pid, 0.0) for pid in ownership_union], dtype=np.float64)
    modeled_arr = np.asarray([modeled_own.get(pid, 0.0) for pid in ownership_union], dtype=np.float64)
    abs_diff = np.abs(actual_arr - modeled_arr)
    top20_idx = np.argsort(-np.maximum(actual_arr, modeled_arr))[:20]

    row.update(
        {
            "modeled_field_version": modeled_field.meta.get("version") or modeled_field.meta.get("build_method"),
            "modeled_field_size_weighted": int(sum(modeled_field.weights)),
            "modeled_unique_lineups": int(len(modeled_field.lineups)),
            "modeled_dupe_rate": 1.0 - (
                float(len(modeled_field.lineups)) / float(sum(modeled_field.weights)) if sum(modeled_field.weights) > 0 else 0.0
            ),
            "modeled_salary_total_mean": _weighted_mean(modeled_df["salary_total"].tolist(), modeled_weights) if not modeled_df.empty else float("nan"),
            "modeled_salary_left_mean": _weighted_mean(modeled_df["salary_left"].tolist(), modeled_weights) if not modeled_df.empty else float("nan"),
            "modeled_projected_own_sum_mean": _weighted_mean(modeled_df["projected_own_sum"].tolist(), modeled_weights) if not modeled_df.empty else float("nan"),
            "modeled_num_teams_mean": _weighted_mean(modeled_df["num_teams"].tolist(), modeled_weights) if not modeled_df.empty else float("nan"),
            "modeled_max_from_team_mean": _weighted_mean(modeled_df["max_from_team"].tolist(), modeled_weights) if not modeled_df.empty else float("nan"),
            "player_ownership_mae_pct": float(abs_diff.mean()) if abs_diff.size else None,
            "player_ownership_rmse_pct": float(np.sqrt(np.mean(np.square(actual_arr - modeled_arr)))) if actual_arr.size else None,
            "top20_player_ownership_mae_pct": float(abs_diff[top20_idx].mean()) if top20_idx.size else None,
            "salary_left_hist_l1": _weighted_hist_l1(
                actual_df["salary_left"].fillna(-1).tolist() if not actual_df.empty else [],
                actual_weights,
                modeled_df["salary_left"].fillna(-1).tolist() if not modeled_df.empty else [],
                modeled_weights,
                bins=[-1, 0, 200, 500, 1000, 1500, 2500, 5000, 10000],
            ),
            "projected_own_sum_hist_l1": _weighted_hist_l1(
                actual_df["projected_own_sum"].fillna(0).tolist() if not actual_df.empty else [],
                actual_weights,
                modeled_df["projected_own_sum"].fillna(0).tolist() if not modeled_df.empty else [],
                modeled_weights,
                bins=[0, 50, 100, 150, 200, 250, 300, 400, 800],
            ),
            "dupe_hist_l1": _weighted_hist_l1(
                actual_field.weights,
                [1.0] * len(actual_field.weights),
                modeled_field.weights,
                [1.0] * len(modeled_field.weights),
                bins=[1, 2, 3, 5, 10, 20, 50, 100, 1000000],
            ),
        }
    )
    return row


def _player_calibration_frame(
    *,
    prepared: PreparedReplayContext,
    actual_field: FieldLibrary,
    modeled_field: Optional[FieldLibrary],
    player_meta: Dict[str, Dict[str, Any]],
    player_name_lookup: Dict[str, str],
    worlds_fpts: np.ndarray,
    worlds_minutes: Optional[np.ndarray],
    player_index: Dict[str, int],
    actual_fpts_lookup: Dict[str, float],
    actual_minutes_lookup: Dict[str, float],
) -> pd.DataFrame:
    actual_contest_own = _count_player_ownership_from_entries(
        [entry.player_ids for entry in prepared.resolved_entries if not entry.unresolved_names],
        denominator=int(prepared.meta.field_size),
    )
    actual_opponent_own = _count_player_ownership_from_library(actual_field)
    modeled_own = _count_player_ownership_from_library(modeled_field) if modeled_field is not None else {}
    player_ids = sorted(set(player_index.keys()) | set(actual_contest_own) | set(actual_opponent_own) | set(modeled_own))
    rows: List[Dict[str, Any]] = []
    for player_id in player_ids:
        meta = player_meta.get(player_id, {})
        world_col = player_index.get(player_id)
        fpts_worlds = worlds_fpts[:, world_col] if world_col is not None else np.asarray([], dtype=np.float64)
        minutes_worlds = worlds_minutes[:, world_col] if (worlds_minutes is not None and world_col is not None) else np.asarray([], dtype=np.float64)
        actual_fpts = actual_fpts_lookup.get(player_id)
        actual_minutes = actual_minutes_lookup.get(player_id)
        projected_own = float(meta.get("own_proj") or 0.0)
        rows.append(
            {
                "game_date": prepared.meta.game_date,
                "contest_id": prepared.meta.contest_id,
                "draft_group_id": prepared.meta.draft_group_id,
                "player_id": player_id,
                "player_name": player_name_lookup.get(player_id) or str(meta.get("name") or player_id),
                "team": meta.get("team"),
                "positions": json.dumps(meta.get("positions") or []),
                "salary": float(meta.get("salary") or 0.0),
                "proj_fpts": float(meta.get("proj") or 0.0),
                "proj_ownership_pct": projected_own,
                "actual_contest_own_pct": actual_contest_own.get(player_id, 0.0),
                "actual_opponent_own_pct": actual_opponent_own.get(player_id, 0.0),
                "modeled_field_own_pct": modeled_own.get(player_id, 0.0),
                "actual_player_fpts": actual_fpts,
                "actual_minutes": actual_minutes,
                "sim_mean_fpts": float(np.mean(fpts_worlds)) if fpts_worlds.size else float("nan"),
                "sim_p10_fpts": float(np.percentile(fpts_worlds, 10)) if fpts_worlds.size else float("nan"),
                "sim_p50_fpts": float(np.percentile(fpts_worlds, 50)) if fpts_worlds.size else float("nan"),
                "sim_p90_fpts": float(np.percentile(fpts_worlds, 90)) if fpts_worlds.size else float("nan"),
                "actual_fpts_sim_percentile": _actual_percentile(actual_fpts, fpts_worlds),
                "sim_mean_minutes": float(np.mean(minutes_worlds)) if minutes_worlds.size else float("nan"),
                "sim_p10_minutes": float(np.percentile(minutes_worlds, 10)) if minutes_worlds.size else float("nan"),
                "sim_p50_minutes": float(np.percentile(minutes_worlds, 50)) if minutes_worlds.size else float("nan"),
                "sim_p90_minutes": float(np.percentile(minutes_worlds, 90)) if minutes_worlds.size else float("nan"),
                "actual_minutes_sim_percentile": _actual_percentile(actual_minutes, minutes_worlds),
                "actual_vs_modeled_own_diff_pct": actual_opponent_own.get(player_id, 0.0) - modeled_own.get(player_id, 0.0),
                "actual_vs_proj_own_diff_pct": actual_contest_own.get(player_id, 0.0) - projected_own,
            }
        )
    return pd.DataFrame(rows)


def _entered_lineup_rows(
    *,
    prepared: PreparedReplayContext,
    simulation: ContestSimResult,
    player_meta: Dict[str, Dict[str, Any]],
    actual_field_counts: Dict[str, int],
    opponent_field_counts: Dict[str, int],
    realized_percentile_map: Dict[str, Optional[float]],
) -> List[Dict[str, Any]]:
    result_map = _simulation_result_map(simulation.results)
    rows: List[Dict[str, Any]] = []
    for entry in prepared.user_entries:
        key = _lineup_key(entry.player_ids)
        sim = result_map.get(key)
        features = _lineup_features(entry.player_ids, player_meta)
        rows.append(
            {
                "game_date": prepared.meta.game_date,
                "contest_id": prepared.meta.contest_id,
                "draft_group_id": prepared.meta.draft_group_id,
                "lineup_key": key,
                "lineup_source": "entered",
                "is_entered": True,
                "player_ids_json": json.dumps(entry.player_ids),
                "entry_id": entry.entry_id,
                "entry_name": entry.entry_name,
                "realized_points": entry.points,
                "realized_rank": entry.rank,
                "realized_prize": entry.prize,
                "realized_score_sim_percentile": realized_percentile_map.get(key),
                "sim_mean": sim.mean if sim else None,
                "sim_std": sim.std if sim else None,
                "sim_p90": sim.p90 if sim else None,
                "sim_p95": sim.p95 if sim else None,
                "sim_roi": sim.roi if sim else None,
                "sim_cash_rate": sim.cash_rate if sim else None,
                "sim_top1pct_rate": sim.top_1pct_rate if sim else None,
                "sim_win_rate": sim.win_rate if sim else None,
                "actual_total_dupe_count": actual_field_counts.get(key, 0),
                "opponent_dupe_count": opponent_field_counts.get(key, 0),
                **features,
            }
        )
    return rows


def find_latest_export_manifest(
    *,
    game_date: str,
    draft_group_id: int,
    contest_id: str,
    data_root: Optional[Path] = None,
) -> Optional[Path]:
    root = data_root or get_data_root()
    exports_dir = root / "contests" / "dk" / f"game_date={game_date}" / f"dg={draft_group_id}" / "exports"
    if not exports_dir.exists():
        return None
    candidates: List[Tuple[datetime, Path]] = []
    for manifest_path in exports_dir.glob("*_manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text())
        except Exception:
            continue
        contest_ids = [str(value) for value in payload.get("contest_ids", [])]
        if str(contest_id) not in contest_ids:
            continue
        created_at = str(payload.get("created_at_utc") or "")
        try:
            ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.fromtimestamp(manifest_path.stat().st_mtime)
        candidates.append((ts, manifest_path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _load_candidate_lineups_from_manifest(
    manifest_path: Path,
    *,
    contest_id: str,
) -> Tuple[List[List[str]], Dict[str, Any]]:
    payload = json.loads(manifest_path.read_text())
    source_run_build_id = str(payload.get("source_run_build_id") or "").strip()
    game_date = str(payload.get("game_date") or "").strip()
    if source_run_build_id and game_date:
        build_path = get_data_root() / "builds" / "contest_sim" / game_date / f"{source_run_build_id}.json"
        if build_path.exists():
            try:
                build = json.loads(build_path.read_text())
                raw_lineups = list(build.get("lineups") or [])
                lineups = [
                    [str(pid).strip() for pid in lineup if str(pid).strip()]
                    for lineup in raw_lineups
                ]
                lineups = [lineup for lineup in lineups if lineup]
                return lineups, {
                    "candidate_manifest_path": str(manifest_path),
                    "candidate_eval_path": None,
                    "candidate_source": "contest_sim_run_build",
                    "candidate_run_build_id": source_run_build_id,
                    "candidate_run_build_path": str(build_path),
                    "candidate_portfolio_build_id": payload.get("source_portfolio_build_id"),
                    "candidate_lineup_count_raw": int(len(lineups)),
                    "candidate_export_id": payload.get("export_id"),
                }
            except Exception:
                pass
    eval_path = payload.get("eval_lineups_path")
    if not isinstance(eval_path, str) or not eval_path:
        return [], {"candidate_manifest_path": str(manifest_path), "candidate_eval_path": None}
    eval_file = Path(eval_path)
    if not eval_file.exists():
        return [], {"candidate_manifest_path": str(manifest_path), "candidate_eval_path": str(eval_file)}
    df = pd.read_csv(eval_file)
    if "contest_id" in df.columns:
        df = df[df["contest_id"].astype(str) == str(contest_id)].copy()
    id_cols = [f"p{i}_id" for i in range(1, 9) if f"p{i}_id" in df.columns]
    lineups: List[List[str]] = []
    for _, row in df.iterrows():
        lineup = [str(int(row[col])) for col in id_cols if pd.notna(row[col])]
        if lineup:
            lineups.append(lineup)
    meta = {
        "candidate_manifest_path": str(manifest_path),
        "candidate_eval_path": str(eval_file),
        "candidate_lineup_count_raw": int(len(lineups)),
        "candidate_export_id": payload.get("export_id"),
    }
    return lineups, meta


def _aggregate_lineups(lineups: Iterable[Sequence[str]]) -> Tuple[List[List[str]], List[int]]:
    counts: Counter[Tuple[str, ...]] = Counter()
    for lineup in lineups:
        counts[tuple(sorted(str(pid) for pid in lineup))] += 1
    unique_lineups = [list(key) for key in counts.keys()]
    weights = [int(count) for count in counts.values()]
    return unique_lineups, weights


def _candidate_lineup_rows(
    *,
    prepared: PreparedReplayContext,
    player_meta: Dict[str, Dict[str, Any]],
    candidate_lineups: List[List[str]],
    candidate_weights: List[int],
    candidate_simulation: ContestSimResult,
    actual_field_counts: Dict[str, int],
    opponent_field_counts: Dict[str, int],
) -> List[Dict[str, Any]]:
    sim_map = _simulation_result_map(candidate_simulation.results)
    entered_keys = {_lineup_key(lineup) for lineup in prepared.user_lineups}
    rows: List[Dict[str, Any]] = []
    for lineup, weight in zip(candidate_lineups, candidate_weights):
        key = _lineup_key(lineup)
        sim = sim_map.get(key)
        rows.append(
            {
                "game_date": prepared.meta.game_date,
                "contest_id": prepared.meta.contest_id,
                "draft_group_id": prepared.meta.draft_group_id,
                "lineup_key": key,
                "lineup_source": "candidate",
                "is_entered": key in entered_keys,
                "player_ids_json": json.dumps(list(lineup)),
                "entry_id": None,
                "entry_name": None,
                "realized_points": None,
                "realized_rank": None,
                "realized_prize": None,
                "realized_score_sim_percentile": None,
                "sim_mean": sim.mean if sim else None,
                "sim_std": sim.std if sim else None,
                "sim_p90": sim.p90 if sim else None,
                "sim_p95": sim.p95 if sim else None,
                "sim_roi": sim.roi if sim else None,
                "sim_cash_rate": sim.cash_rate if sim else None,
                "sim_top1pct_rate": sim.top_1pct_rate if sim else None,
                "sim_win_rate": sim.win_rate if sim else None,
                "actual_total_dupe_count": actual_field_counts.get(key, 0),
                "opponent_dupe_count": opponent_field_counts.get(key, 0),
                "candidate_weight": int(weight),
                **_lineup_features(lineup, player_meta),
            }
        )
    return rows


def _best_lineup_row(df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    if df.empty or metric not in df.columns or df[metric].dropna().empty:
        return {}
    idx = df[metric].astype(float).idxmax()
    return df.loc[idx].to_dict()


def _actual_best_entered_row(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    if "realized_prize" in df.columns and df["realized_prize"].fillna(0).max() > 0:
        idx = df["realized_prize"].fillna(0).astype(float).idxmax()
    elif "realized_points" in df.columns and df["realized_points"].dropna().any():
        idx = df["realized_points"].astype(float).idxmax()
    else:
        idx = df["realized_rank"].fillna(10**9).astype(float).idxmin()
    return df.loc[idx].to_dict()


def _lineup_label(row: Dict[str, Any]) -> Optional[str]:
    if not row:
        return None
    entry_name = row.get("entry_name")
    if entry_name:
        return str(entry_name)
    player_ids_json = row.get("player_ids_json")
    if player_ids_json:
        return str(player_ids_json)
    lineup_key = row.get("lineup_key")
    return str(lineup_key) if lineup_key is not None else None


def _regret_frame(
    *,
    prepared: PreparedReplayContext,
    entered_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    candidate_meta: Dict[str, Any],
) -> pd.DataFrame:
    best_entered = _best_lineup_row(entered_df, "sim_roi")
    actual_best_entered = _actual_best_entered_row(entered_df)
    candidate_only = candidate_df.copy()
    best_candidate = _best_lineup_row(candidate_only, "sim_roi")
    best_finalset = best_entered
    row = {
        "game_date": prepared.meta.game_date,
        "contest_id": prepared.meta.contest_id,
        "draft_group_id": prepared.meta.draft_group_id,
        "entered_unique_count": int(entered_df["lineup_key"].nunique()) if not entered_df.empty else 0,
        "best_entered_lineup_key": best_entered.get("lineup_key"),
        "best_entered_lineup_label": _lineup_label(best_entered),
        "best_entered_sim_roi": best_entered.get("sim_roi"),
        "best_entered_sim_cash_rate": best_entered.get("sim_cash_rate"),
        "best_entered_realized_rank": best_entered.get("realized_rank"),
        "best_entered_realized_prize": best_entered.get("realized_prize"),
        "actual_best_entered_lineup_key": actual_best_entered.get("lineup_key"),
        "actual_best_entered_lineup_label": _lineup_label(actual_best_entered),
        "actual_best_entered_rank": actual_best_entered.get("realized_rank"),
        "actual_best_entered_prize": actual_best_entered.get("realized_prize"),
        "candidate_pool_available": bool(not candidate_df.empty),
        "candidate_manifest_path": candidate_meta.get("candidate_manifest_path"),
        "candidate_unique_count": int(candidate_df["lineup_key"].nunique()) if not candidate_df.empty else 0,
        "best_candidate_lineup_key": best_candidate.get("lineup_key"),
        "best_candidate_lineup_label": _lineup_label(best_candidate),
        "best_candidate_sim_roi": best_candidate.get("sim_roi"),
        "best_candidate_sim_cash_rate": best_candidate.get("sim_cash_rate"),
        "best_candidate_is_entered": bool(best_candidate.get("is_entered")) if best_candidate else False,
        "best_finalset_lineup_key": best_finalset.get("lineup_key"),
        "best_finalset_lineup_label": _lineup_label(best_finalset),
        "best_finalset_sim_roi": best_finalset.get("sim_roi"),
        "selection_regret_roi": (
            float(best_candidate.get("sim_roi")) - float(best_finalset.get("sim_roi"))
            if best_candidate.get("sim_roi") is not None and best_finalset.get("sim_roi") is not None
            else None
        ),
        "selection_regret_cash_rate": (
            float(best_candidate.get("sim_cash_rate")) - float(best_finalset.get("sim_cash_rate"))
            if best_candidate.get("sim_cash_rate") is not None and best_finalset.get("sim_cash_rate") is not None
            else None
        ),
    }
    return pd.DataFrame([row])


def build_post_contest_replay_analytics(
    *,
    contest_id: str,
    game_date: str,
    user_pattern: str,
    draft_group_id: Optional[int] = None,
    run_id: Optional[str] = None,
    entry_fee: Optional[float] = None,
    archetype: str = "medium",
    worlds_source: str = "gtv2",
    ownership_mode: str = "field_only",
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    modeled_field_version: str = "v1_calibrated",
    include_modeled_field: bool = True,
    candidate_manifest_path: Optional[Path] = None,
) -> ReplayAnalyticsBundle:
    data_root = data_root or get_data_root()
    replay_run: ContestReplayRun = run_post_contest_replay(
        contest_id=contest_id,
        game_date=game_date,
        user_pattern=user_pattern,
        draft_group_id=draft_group_id,
        run_id=run_id,
        entry_fee=entry_fee,
        archetype=archetype,
        worlds_source=worlds_source,
        ownership_mode=ownership_mode,
        data_root=data_root,
    )
    prepared = replay_run.prepared
    resolved_draft_group_id = int(prepared.meta.draft_group_id or 0)
    player_meta, player_name_lookup = _player_pool_maps(
        game_date=prepared.meta.game_date,
        draft_group_id=resolved_draft_group_id,
        data_root=data_root,
        run_id=run_id,
    )
    player_worlds = load_player_worlds(
        game_date=prepared.meta.game_date,
        data_root=data_root,
        run_id=run_id,
        worlds_source=worlds_source,
    )
    actual_minutes_lookup = _load_actual_minutes_lookup(game_date=prepared.meta.game_date, data_root=data_root)
    actual_fpts_lookup = _load_actual_player_fpts_lookup(prepared=prepared, data_root=data_root)
    actual_field = build_actual_field_library(
        [entry for entry in prepared.resolved_entries if not entry.unresolved_names],
        meta=prepared.meta,
    )

    modeled_field: Optional[FieldLibrary] = None
    modeled_field_meta: Dict[str, Any] = {}
    if include_modeled_field:
        modeled_field, modeled_path, built_now = load_or_build_field_library(
            game_date=prepared.meta.game_date,
            draft_group_id=resolved_draft_group_id,
            version=modeled_field_version,
            data_root=data_root,
        )
        modeled_field.meta.setdefault("version", modeled_field_version)
        modeled_field_meta = {
            "modeled_field_path": str(modeled_path),
            "modeled_field_built_now": bool(built_now),
            "modeled_field_version": modeled_field_version,
        }

    player_df = _player_calibration_frame(
        prepared=prepared,
        actual_field=prepared.opponent_field_library,
        modeled_field=modeled_field,
        player_meta=player_meta,
        player_name_lookup=player_name_lookup,
        worlds_fpts=player_worlds.fpts_matrix,
        worlds_minutes=player_worlds.minutes_matrix,
        player_index=player_worlds.player_index,
        actual_fpts_lookup=actual_fpts_lookup,
        actual_minutes_lookup=actual_minutes_lookup,
    )

    actual_entry_lineups = [entry.player_ids for entry in prepared.resolved_entries if not entry.unresolved_names]
    actual_field_counts = _lineup_actual_counts(actual_entry_lineups)
    opponent_field_counts = { _lineup_key(lineup): int(weight) for lineup, weight in zip(prepared.opponent_field_library.lineups, prepared.opponent_field_library.weights) }
    user_actual_score_lookup = {
        _lineup_key(entry.player_ids): entry.points for entry in prepared.user_entries
    }
    user_realized_percentiles = _world_percentile_map(
        lineups=prepared.user_lineups,
        actual_scores=user_actual_score_lookup,
        worlds_matrix=player_worlds.fpts_matrix,
        player_index=player_worlds.player_index,
    )

    entered_rows = _entered_lineup_rows(
        prepared=prepared,
        simulation=replay_run.simulation,
        player_meta=player_meta,
        actual_field_counts=actual_field_counts,
        opponent_field_counts=opponent_field_counts,
        realized_percentile_map=user_realized_percentiles,
    )
    entered_df = pd.DataFrame(entered_rows)

    candidate_rows: List[Dict[str, Any]] = []
    candidate_meta: Dict[str, Any] = {}
    if candidate_manifest_path is None:
        candidate_manifest_path = find_latest_export_manifest(
            game_date=prepared.meta.game_date,
            draft_group_id=resolved_draft_group_id,
            contest_id=prepared.meta.contest_id,
            data_root=data_root,
        )
    if candidate_manifest_path is not None:
        candidate_lineups_raw, candidate_meta = _load_candidate_lineups_from_manifest(
            candidate_manifest_path,
            contest_id=prepared.meta.contest_id,
        )
        candidate_lineups, candidate_weights = _aggregate_lineups(candidate_lineups_raw)
        if candidate_lineups:
            candidate_simulation = run_contest_simulation(
                user_lineups=candidate_lineups,
                user_weights=candidate_weights,
                game_date=prepared.meta.game_date,
                draft_group_id=prepared.meta.draft_group_id,
                run_id=run_id,
                archetype=archetype,
                entry_fee=float(entry_fee if entry_fee is not None else prepared.meta.entry_fee),
                field_lineups=prepared.opponent_field_library.lineups,
                field_weights=prepared.opponent_field_library.weights,
                field_size_override=prepared.meta.field_size,
                data_root=data_root,
                ownership_mode=ownership_mode,
                worlds_source=worlds_source,
            )
            candidate_rows = _candidate_lineup_rows(
                prepared=prepared,
                player_meta=player_meta,
                candidate_lineups=candidate_lineups,
                candidate_weights=candidate_weights,
                candidate_simulation=candidate_simulation,
                actual_field_counts=actual_field_counts,
                opponent_field_counts=opponent_field_counts,
            )
            candidate_meta["candidate_unique_count"] = int(len(candidate_lineups))
            candidate_meta["candidate_weight_sum"] = int(sum(candidate_weights))

    candidate_df = pd.DataFrame(candidate_rows)
    lineup_df = pd.concat([entered_df, candidate_df], ignore_index=True, sort=False)

    field_df = pd.DataFrame(
        [
            _field_summary_row(
                prepared=prepared,
                actual_field=prepared.opponent_field_library,
                modeled_field=modeled_field,
                player_meta=player_meta,
            )
        ]
    )
    regret_df = _regret_frame(
        prepared=prepared,
        entered_df=entered_df,
        candidate_df=candidate_df,
        candidate_meta=candidate_meta,
    )

    out_dir = output_dir or replay_output_dir(
        game_date=prepared.meta.game_date,
        contest_id=prepared.meta.contest_id,
        user_pattern=user_pattern,
        data_root=data_root,
    ) / "analytics"
    out_dir.mkdir(parents=True, exist_ok=True)
    player_path = out_dir / "player_calibration.parquet"
    lineup_path = out_dir / "lineup_calibration.parquet"
    field_path = out_dir / "field_calibration.parquet"
    regret_path = out_dir / "regret_summary.parquet"
    summary_path = out_dir / "summary.json"

    player_df.to_parquet(player_path, index=False)
    lineup_df.to_parquet(lineup_path, index=False)
    field_df.to_parquet(field_path, index=False)
    regret_df.to_parquet(regret_path, index=False)

    summary_payload = {
        "contest_id": prepared.meta.contest_id,
        "game_date": prepared.meta.game_date,
        "draft_group_id": prepared.meta.draft_group_id,
        "user_pattern": user_pattern,
        "run_id": run_id,
        "worlds_source": worlds_source,
        "ownership_mode": ownership_mode,
        "modeled_field": modeled_field_meta,
        "candidate_meta": candidate_meta,
        "resolution": prepared.resolution_stats,
        "user_replay_summary": {
            "entered_lineup_count": int(len(entered_df)),
            "best_sim_roi": _best_lineup_row(entered_df, "sim_roi").get("sim_roi"),
            "avg_sim_roi": float(entered_df["sim_roi"].dropna().mean()) if not entered_df.empty and "sim_roi" in entered_df.columns and entered_df["sim_roi"].dropna().any() else None,
            "best_sim_cash_rate": _best_lineup_row(entered_df, "sim_cash_rate").get("sim_cash_rate"),
            "best_realized_rank": _actual_best_entered_row(entered_df).get("realized_rank"),
            "best_realized_prize": _actual_best_entered_row(entered_df).get("realized_prize"),
            "avg_realized_rank": float(entered_df["realized_rank"].dropna().mean()) if not entered_df.empty and "realized_rank" in entered_df.columns and entered_df["realized_rank"].dropna().any() else None,
        },
        "field_summary": field_df.iloc[0].where(pd.notna(field_df.iloc[0]), None).to_dict() if not field_df.empty else {},
        "regret_summary": regret_df.iloc[0].where(pd.notna(regret_df.iloc[0]), None).to_dict() if not regret_df.empty else {},
        "artifacts": {
            "player_calibration_path": str(player_path),
            "lineup_calibration_path": str(lineup_path),
            "field_calibration_path": str(field_path),
            "regret_summary_path": str(regret_path),
        },
        "counts": {
            "player_rows": int(len(player_df)),
            "lineup_rows": int(len(lineup_df)),
            "entered_lineup_rows": int(len(entered_df)),
            "candidate_lineup_rows": int(len(candidate_df)),
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True))

    return ReplayAnalyticsBundle(
        player_calibration_path=player_path,
        lineup_calibration_path=lineup_path,
        field_calibration_path=field_path,
        regret_summary_path=regret_path,
        summary_path=summary_path,
    )
