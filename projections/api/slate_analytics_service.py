from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from projections.contest_sim.contest_sim_service import load_player_worlds
from projections.optimizer.cpsat_solver import build_cpsat_counts
from projections.optimizer.model_spec import Spec, SpecPlayer
from projections.projections_bundle import resolve_unified_projections_run

logger = logging.getLogger(__name__)

_ANALYTICS_VERSION = "v3"
_DEFAULT_SAMPLE_WORLDS = 96
_DEFAULT_SAMPLE_SEED = 42
_DEFAULT_SOLVER_TIME_LIMIT_S = 0.18


def _cache_path(*, data_root: Path, game_date: str, run_id: str, draft_group_id: int) -> Path:
    return (
        data_root
        / "artifacts"
        / "projections"
        / game_date
        / f"run={run_id}"
        / f"slate_analytics_dg={draft_group_id}.json"
    )


def _safe_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _as_player_id(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(int(float(value)))
    except Exception:
        return str(value)


def _to_projection_fallback(row: dict[str, Any]) -> float:
    for key in (
        "proj",
        "effective_proj",
        "model_proj",
        "p90",
    ):
        val = _safe_float(row.get(key))
        if val is not None:
            return max(0.0, val)
    return 0.0


def _position_set(raw: Any) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, str):
        tokens = [part.strip().upper() for part in raw.replace("/", ",").split(",")]
        return {tok for tok in tokens if tok}
    if isinstance(raw, (list, tuple, set)):
        return {str(x).strip().upper() for x in raw if str(x).strip()}
    return set()


def _resolve_dk_spec_players(pool_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_slots = {"PG", "SG", "SF", "PF", "C"}
    out: list[dict[str, Any]] = []
    for row in pool_rows:
        pid = _as_player_id(row.get("player_id"))
        if not pid:
            continue
        positions = sorted(_position_set(row.get("positions")))
        if not (set(positions) & base_slots):
            continue
        salary = _safe_float(row.get("salary"))
        if salary is None or salary <= 0:
            continue
        is_out = bool(row.get("is_out", False))
        is_active = bool(row.get("is_active", not is_out))
        if is_out or not is_active:
            continue
        out.append(
            {
                "player_id": pid,
                "name": str(row.get("name") or pid),
                "team": str(row.get("team") or "UNK"),
                "positions": positions,
                "salary": int(round(salary)),
                "own_proj": _safe_float(row.get("own_proj")),
                "stddev": _safe_float(row.get("stddev")),
                "game_start_utc": row.get("game_start_utc"),
            }
        )
    return out


def _solve_optimal_pct_sampled(
    *,
    dk_players: list[dict[str, Any]],
    worlds_matrix: np.ndarray,
    worlds_player_index: dict[str, int],
    fallback_proj_by_pid: dict[str, float],
    sample_worlds: int,
    sample_seed: int,
    solver_time_limit_s: float,
) -> tuple[dict[str, float], int]:
    if not dk_players:
        return {}, 0

    player_ids = [str(p["player_id"]) for p in dk_players]
    world_cols = np.asarray([int(worlds_player_index.get(pid, -1)) for pid in player_ids], dtype=np.int64)
    fallback = np.asarray([float(max(0.0, fallback_proj_by_pid.get(pid, 0.0))) for pid in player_ids], dtype=np.float64)

    # Need a decent candidate set for DK roster feasibility.
    pos_sets = [set(p["positions"]) for p in dk_players]
    if sum("PG" in s for s in pos_sets) < 2 or sum("SG" in s for s in pos_sets) < 2:
        return {}, 0
    if sum("SF" in s for s in pos_sets) < 2 or sum("PF" in s for s in pos_sets) < 2:
        return {}, 0
    if sum("C" in s for s in pos_sets) < 1:
        return {}, 0

    n_worlds = int(worlds_matrix.shape[0])
    if n_worlds <= 0:
        return {}, 0
    sample_n = int(max(1, min(sample_worlds, n_worlds)))
    rng = np.random.default_rng(int(sample_seed))
    sampled_world_idx = (
        rng.choice(n_worlds, size=sample_n, replace=False)
        if sample_n < n_worlds
        else np.arange(n_worlds, dtype=np.int64)
    )

    # Build template once; player projections are swapped per world via dataclass replace.
    template_players = [
        SpecPlayer(
            player_id=str(p["player_id"]),
            name=str(p["name"]),
            team=str(p["team"]),
            positions=list(p["positions"]),
            salary=int(p["salary"]),
            proj=float(fallback_proj_by_pid.get(str(p["player_id"]), 0.0)),
            dk_id=None,
            own_proj=p.get("own_proj"),
            stddev=p.get("stddev"),
            game_start_utc=p.get("game_start_utc"),
        )
        for p in dk_players
    ]
    spec_template = Spec(
        site="dk",
        roster_slots=["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"],
        salary_cap=50000,
        min_salary=49000,
        players=template_players,
        team_max=4,
        lineup_size=8,
        N_lineups=1,
        unique_players=1,
        cp_sat_params={},
    )

    counts = np.zeros(len(player_ids), dtype=np.int64)
    solved_worlds = 0
    for seq, widx in enumerate(sampled_world_idx.tolist()):
        world_proj = fallback.copy()
        valid = world_cols >= 0
        if bool(valid.any()):
            world_proj[valid] = np.take(worlds_matrix[int(widx)], world_cols[valid], axis=0)
        world_proj = np.maximum(world_proj, 0.0)

        world_players = [
            replace(template_players[i], proj=float(world_proj[i]))
            for i in range(len(template_players))
        ]
        world_spec = replace(spec_template, players=world_players)

        try:
            model, y = build_cpsat_counts(world_spec)
        except Exception:
            continue

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(max(0.05, solver_time_limit_s))
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = int(sample_seed + seq)
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            continue
        solved_worlds += 1
        for i, pid in enumerate(player_ids):
            if solver.BooleanValue(y[pid]):
                counts[i] += 1

    if solved_worlds <= 0:
        return {}, 0
    out = {
        pid: float((counts[i] / solved_worlds) * 100.0)
        for i, pid in enumerate(player_ids)
    }
    return out, solved_worlds


def _build_player_rows(
    *,
    pool_rows: list[dict[str, Any]],
    worlds_matrix: np.ndarray,
    worlds_player_index: dict[str, int],
    optimal_pct_by_pid: dict[str, float],
) -> list[dict[str, Any]]:
    player_ids = [_as_player_id(row.get("player_id")) for row in pool_rows]
    world_cols = np.asarray([int(worlds_player_index.get(pid, -1)) for pid in player_ids], dtype=np.int64)

    n_worlds = int(worlds_matrix.shape[0])
    aligned = np.full((n_worlds, len(pool_rows)), np.nan, dtype=np.float64)
    valid = world_cols >= 0
    if bool(valid.any()):
        aligned[:, valid] = np.take(worlds_matrix, world_cols[valid], axis=1)

    salary = np.asarray(
        [float(max(0.0, _safe_float(row.get("salary")) or 0.0)) for row in pool_rows],
        dtype=np.float64,
    )
    own = np.asarray(
        [float(_safe_float(row.get("own_proj")) or 0.0) for row in pool_rows],
        dtype=np.float64,
    )

    p90 = np.nanquantile(aligned, 0.90, axis=0)
    p90 = np.where(np.isfinite(p90), p90, 0.0)

    boom5x_line = (salary / 1000.0) * 5.0
    boom6x_line = (salary / 1000.0) * 6.0
    bust_line = (salary / 1000.0) * 3.0
    with np.errstate(invalid="ignore"):
        boom5x_pct = np.nanmean(aligned >= boom5x_line, axis=0) * 100.0
        boom6x_pct = np.nanmean(aligned >= boom6x_line, axis=0) * 100.0
        bust_pct = np.nanmean(aligned < bust_line, axis=0) * 100.0
    boom5x_pct = np.where(np.isfinite(boom5x_pct), boom5x_pct, 0.0)
    boom6x_pct = np.where(np.isfinite(boom6x_pct), boom6x_pct, 0.0)
    bust_pct = np.where(np.isfinite(bust_pct), bust_pct, 0.0)

    p90_rank = pd.Series(p90).rank(method="average", pct=True).to_numpy(dtype=np.float64) * 100.0
    own_rank = pd.Series(own).rank(method="average", pct=True).to_numpy(dtype=np.float64) * 100.0
    ceiling_leverage = p90_rank - own_rank

    rows: list[dict[str, Any]] = []
    for i, row in enumerate(pool_rows):
        pid = player_ids[i]
        rows.append(
            {
                "player_id": pid,
                "name": str(row.get("name") or pid),
                "team": str(row.get("team") or "UNK"),
                "salary": int(round(salary[i])) if np.isfinite(salary[i]) else None,
                "own_proj": float(own[i]),
                "optimal_pct": float(optimal_pct_by_pid.get(pid, 0.0)),
                "ceiling_leverage": float(ceiling_leverage[i]),
                "boom_pct": float(boom5x_pct[i]),  # kept for backwards compatibility
                "boom5x_pct": float(boom5x_pct[i]),
                "boom6x_pct": float(boom6x_pct[i]),
                "bust_pct": float(bust_pct[i]),
                "p90": float(p90[i]),
            }
        )
    return rows


def load_or_compute_slate_player_analytics(
    *,
    game_date: str,
    draft_group_id: int,
    pool_rows: list[dict[str, Any]],
    run_id: str | None = None,
    data_root: Path,
    sample_worlds: int = _DEFAULT_SAMPLE_WORLDS,
    sample_seed: int = _DEFAULT_SAMPLE_SEED,
    solver_time_limit_s: float = _DEFAULT_SOLVER_TIME_LIMIT_S,
    force_recompute: bool = False,
) -> dict[str, Any]:
    resolved = resolve_unified_projections_run(game_date, run_id=run_id, data_root=data_root)
    resolved_run_id = resolved.run_id
    if not resolved_run_id:
        raise FileNotFoundError(f"No unified projections run found for {game_date}")

    sample_worlds = int(max(1, sample_worlds))
    sample_seed = int(sample_seed)
    solver_time_limit_s = float(max(0.05, solver_time_limit_s))
    cache_file = _cache_path(
        data_root=data_root,
        game_date=game_date,
        run_id=resolved_run_id,
        draft_group_id=int(draft_group_id),
    )

    if not force_recompute and cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            if (
                isinstance(payload, dict)
                and payload.get("version") == _ANALYTICS_VERSION
                and int(payload.get("sample_worlds") or 0) == sample_worlds
                and int(payload.get("sample_seed") or 0) == sample_seed
                and int(payload.get("draft_group_id") or 0) == int(draft_group_id)
                and str(payload.get("run_id") or "") == str(resolved_run_id)
            ):
                return payload
        except Exception:
            pass

    worlds = load_player_worlds(
        game_date=game_date,
        data_root=data_root,
        run_id=resolved_run_id,
        worlds_source="gtv2",
    )
    worlds_matrix = np.asarray(worlds.fpts_matrix, dtype=np.float64)
    worlds_player_index = {str(k): int(v) for k, v in worlds.player_index.items()}

    fallback_proj_by_pid = {
        _as_player_id(row.get("player_id")): _to_projection_fallback(row)
        for row in pool_rows
    }

    dk_players = _resolve_dk_spec_players(pool_rows)
    optimal_pct_by_pid, solved_worlds = _solve_optimal_pct_sampled(
        dk_players=dk_players,
        worlds_matrix=worlds_matrix,
        worlds_player_index=worlds_player_index,
        fallback_proj_by_pid=fallback_proj_by_pid,
        sample_worlds=sample_worlds,
        sample_seed=sample_seed,
        solver_time_limit_s=solver_time_limit_s,
    )

    player_rows = _build_player_rows(
        pool_rows=pool_rows,
        worlds_matrix=worlds_matrix,
        worlds_player_index=worlds_player_index,
        optimal_pct_by_pid=optimal_pct_by_pid,
    )

    payload: dict[str, Any] = {
        "version": _ANALYTICS_VERSION,
        "game_date": str(game_date),
        "run_id": str(resolved_run_id),
        "draft_group_id": int(draft_group_id),
        "sample_worlds": sample_worlds,
        "sample_seed": sample_seed,
        "solver_time_limit_s": solver_time_limit_s,
        "optimal_worlds_solved": int(solved_worlds),
        "generated_at": datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "players": player_rows,
    }

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(f".tmp.{datetime.now(tz=UTC).timestamp():.0f}.json")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(cache_file)
    except Exception as exc:
        logger.warning("Failed to write slate analytics cache %s: %s", cache_file, exc)

    return payload
