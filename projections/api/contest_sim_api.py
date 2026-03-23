"""FastAPI router for contest simulation endpoints."""

from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple
from uuid import uuid4

import numpy as np

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from projections import paths
from projections.api.optimizer_service import (
    build_player_pool,
    get_slates_for_date,
    load_projections_for_date,
)
from projections.contest_sim.contest_sim_service import load_worlds_matrix, run_contest_simulation
from projections.contest_sim.field_library import load_field_library, list_field_library_paths
from projections.contest_sim.field_library_manager import load_or_build_field_library
from projections.contest_sim.portfolio_optimizer import (
    DecorrelatedPortfolioConfig,
    ExposureBoundsPct,
    PortfolioCandidate,
    build_decorrelated_portfolio,
    build_portfolio,
    compute_total_own,
    get_candidate_metric_value,
    summarize_portfolio,
)
from projections.contest_sim.payout_generator import load_config

logger = logging.getLogger(__name__)
router = APIRouter()

SUPPORTED_SITES = {"dk", "fd"}
DK_ROSTER_SLOTS: Tuple[str, ...] = ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL")
FD_ROSTER_SLOTS: Tuple[str, ...] = ("PG", "PG", "SG", "SG", "SF", "SF", "PF", "PF", "C")


def _normalize_site(site: str | None) -> str:
    site_norm = str(site or "dk").strip().lower()
    if site_norm not in SUPPORTED_SITES:
        raise ValueError(f"Unsupported site '{site}'. Expected one of {sorted(SUPPORTED_SITES)}")
    return site_norm


def _canonicalize_player_id(raw: object) -> str | None:
    """Normalize player identifiers to stable string form (e.g. 1627742.0 -> '1627742')."""
    if raw is None:
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
    if "." in text:
        try:
            value = float(text)
            if np.isfinite(value) and value.is_integer():
                return str(int(value))
        except Exception:
            pass
    return text


def _parse_position_tokens(raw: object) -> Set[str]:
    tokens: List[str] = []
    if isinstance(raw, (list, tuple, set)):
        for item in raw:
            text = str(item or "").strip()
            if text:
                tokens.append(text)
    else:
        text = str(raw or "").strip()
        if text:
            tokens.append(text)

    out: Set[str] = set()
    for token in tokens:
        for piece in token.replace("|", "/").replace(",", "/").split("/"):
            value = piece.strip().upper()
            if value:
                out.add(value)
    return out


def _is_position_eligible(site: str, slot: str, positions: Set[str]) -> bool:
    slot_norm = str(slot or "").upper()
    if site == "fd":
        return slot_norm in positions

    if slot_norm in {"PG", "SG", "SF", "PF", "C"}:
        return slot_norm in positions
    if slot_norm == "G":
        return bool({"PG", "SG", "G"} & positions)
    if slot_norm == "F":
        return bool({"SF", "PF", "F"} & positions)
    if slot_norm == "UTIL":
        return bool({"PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"} & positions)
    return False


def _slots_for_site(site: str) -> Tuple[str, ...]:
    return DK_ROSTER_SLOTS if site == "dk" else FD_ROSTER_SLOTS


def _assign_lineup_slots(
    lineup: Sequence[str],
    *,
    site: str,
    positions_by_player: Dict[str, Set[str]],
    lineup_idx: int,
    context: str,
) -> List[str]:
    slots = _slots_for_site(site)
    expected_size = len(slots)
    players: List[str] = []
    for pid in lineup:
        canonical = _canonicalize_player_id(pid)
        if canonical:
            players.append(canonical)

    if len(players) != expected_size:
        raise ValueError(
            f"{context}[{lineup_idx}] must contain {expected_size} players for site={site}; got {len(players)}"
        )
    if len(set(players)) != expected_size:
        raise ValueError(f"{context}[{lineup_idx}] contains duplicate players")

    missing_positions = [pid for pid in players if pid not in positions_by_player]
    if missing_positions:
        raise ValueError(
            f"{context}[{lineup_idx}] includes players missing slate positions for site={site}: {missing_positions[:6]}"
        )

    slot_candidates: List[List[str]] = []
    for slot in slots:
        elig_slot = ("C" if slot == "C" else slot[:2]) if site == "fd" else slot
        cands = [
            pid
            for pid in players
            if _is_position_eligible(site, elig_slot, positions_by_player[pid])
        ]
        if not cands:
            raise ValueError(
                f"{context}[{lineup_idx}] cannot satisfy slot '{slot}' for site={site}"
            )
        slot_candidates.append(cands)

    assignment: List[str | None] = [None] * expected_size
    used: Set[str] = set()
    slot_order = sorted(range(expected_size), key=lambda idx: len(slot_candidates[idx]))

    def backtrack(order_idx: int) -> bool:
        if order_idx >= len(slot_order):
            return True
        slot_idx = slot_order[order_idx]
        candidates = sorted(
            (pid for pid in slot_candidates[slot_idx] if pid not in used),
            key=lambda pid: (len(positions_by_player[pid]), pid),
        )
        for pid in candidates:
            used.add(pid)
            assignment[slot_idx] = pid
            if backtrack(order_idx + 1):
                return True
            assignment[slot_idx] = None
            used.remove(pid)
        return False

    if not backtrack(0):
        raise ValueError(
            f"{context}[{lineup_idx}] cannot be assigned to a valid {site.upper()} roster"
        )

    return [str(pid) for pid in assignment if pid is not None]


def _build_positions_lookup(
    *,
    game_date: str,
    draft_group_id: int,
    site: str,
    run_id: str | None,
) -> Dict[str, Set[str]]:
    player_pool = build_player_pool(
        game_date=game_date,
        draft_group_id=int(draft_group_id),
        site=site,
        run_id=run_id,
        data_root=paths.data_path(),
        include_unmatched_salaries=True,
        allow_zero_projections=True,
        exclude_inactive_players=False,
        use_user_overrides=False,
    )
    out: Dict[str, Set[str]] = {}
    for player in player_pool:
        pid = _canonicalize_player_id(player.get("player_id"))
        if not pid:
            continue
        positions = _parse_position_tokens(player.get("positions"))
        if positions:
            out[pid] = positions
            # Accept site identifiers as aliases when lineups are site-id keyed.
            for alias_field in ("site_id", "dk_id", "fd_id"):
                alias = _canonicalize_player_id(player.get(alias_field))
                if alias:
                    out.setdefault(alias, positions)
    return out


def _normalize_lineups_for_site(
    lineups: Sequence[Sequence[str]],
    *,
    game_date: str,
    draft_group_id: int | None,
    site: str,
    run_id: str | None,
    context: str,
) -> List[List[str]]:
    if not lineups:
        return []

    slots = _slots_for_site(site)
    expected_size = len(slots)
    for idx, lineup in enumerate(lineups):
        n_players = len([pid for pid in (_canonicalize_player_id(raw) for raw in lineup) if pid])
        if n_players != expected_size:
            raise ValueError(
                f"{context}[{idx}] must contain {expected_size} players for site={site}; got {n_players}"
            )

    if draft_group_id is None:
        if site == "fd":
            raise ValueError("draft_group_id is required for site=fd to validate FanDuel slot compliance")
        # For DK, preserve existing behavior when slate metadata isn't provided.
        normalized_lineups: List[List[str]] = []
        for lineup in lineups:
            normalized_lineups.append(
                [pid for pid in (_canonicalize_player_id(raw) for raw in lineup) if pid]
            )
        return normalized_lineups

    positions_by_player = _build_positions_lookup(
        game_date=game_date,
        draft_group_id=int(draft_group_id),
        site=site,
        run_id=run_id,
    )
    if not positions_by_player:
        raise ValueError(
            f"Unable to load slate positions for {game_date} draft_group_id={draft_group_id} site={site}"
        )

    normalized: List[List[str]] = []
    for idx, lineup in enumerate(lineups):
        normalized.append(
            _assign_lineup_slots(
                lineup,
                site=site,
                positions_by_player=positions_by_player,
                lineup_idx=idx,
                context=context,
            )
        )
    return normalized


def _sample_lineup_player_ids(
    lineups: Sequence[Sequence[str]],
    *,
    max_lineups: int | None = None,
    max_players: int | None = None,
) -> List[str]:
    sampled: List[str] = []
    seen: Set[str] = set()
    for lineup_idx, lineup in enumerate(lineups):
        if max_lineups is not None and lineup_idx >= max_lineups:
            break
        for raw_pid in lineup:
            pid = _canonicalize_player_id(raw_pid)
            if not pid or pid in seen:
                continue
            sampled.append(pid)
            seen.add(pid)
            if max_players is not None and len(sampled) >= max_players:
                return sampled
    return sampled


def _candidate_draft_group_ids_for_resolution(game_date: str, site: str) -> List[int]:
    site_norm = _normalize_site(site)
    try:
        slates = get_slates_for_date(game_date, slate_type="all", site=site_norm)
    except Exception as exc:
        logger.warning(
            "Failed to enumerate slates for draft_group resolution (%s %s): %s",
            game_date,
            site_norm,
            exc,
        )
        slates = []

    def _rank(slate: Dict[str, object]) -> tuple[int, int, int, str]:
        slate_type = str(slate.get("slate_type") or "").strip().lower()
        games = slate.get("games") if isinstance(slate.get("games"), list) else []
        n_games = len(games)
        n_contests = int(slate.get("n_contests") or 0)
        earliest = str(slate.get("earliest_start") or "")
        return (0 if slate_type == "main" else 1, -n_games, -n_contests, earliest)

    candidate_dgs: List[int] = []
    for slate in sorted(slates, key=_rank):
        raw_dg = slate.get("draft_group_id")
        if raw_dg is None:
            continue
        dg = int(raw_dg)
        if dg not in candidate_dgs:
            candidate_dgs.append(dg)
    return candidate_dgs


def _resolve_draft_group_id_for_lineups(
    *,
    game_date: str,
    lineups: Sequence[Sequence[str]],
    site: str,
    run_id: str | None,
    requested_draft_group_id: int | None,
) -> tuple[int | None, Dict[str, object]]:
    """Resolve slate draft group from lineups when request metadata is stale.

    This is primarily for DK where UI state can drift from the lineup source slate.
    """
    site_norm = _normalize_site(site)
    if site_norm != "dk":
        return requested_draft_group_id, {}

    sample_player_ids = _sample_lineup_player_ids(lineups)
    if not sample_player_ids:
        return requested_draft_group_id, {}

    coverage_cache: Dict[int, int] = {}

    def coverage_for_dg(dg: int) -> int:
        if dg in coverage_cache:
            return coverage_cache[dg]
        lookup = _build_positions_lookup(
            game_date=game_date,
            draft_group_id=int(dg),
            site=site_norm,
            run_id=run_id,
        )
        matched = sum(1 for pid in sample_player_ids if pid in lookup)
        coverage_cache[int(dg)] = matched
        return matched

    requested_match = None
    if requested_draft_group_id is not None:
        requested_match = coverage_for_dg(int(requested_draft_group_id))
        if requested_match == len(sample_player_ids):
            return int(requested_draft_group_id), {
                "requested_draft_group_id": int(requested_draft_group_id),
                "effective_draft_group_id": int(requested_draft_group_id),
                "sample_player_count": len(sample_player_ids),
                "requested_match_count": int(requested_match),
                "inferred_from_lineups": False,
            }

    candidate_dgs = _candidate_draft_group_ids_for_resolution(game_date, site_norm)

    best_dg: int | None = None
    best_match = -1
    for dg in candidate_dgs:
        matched = coverage_for_dg(dg)
        if matched > best_match:
            best_match = matched
            best_dg = dg
        if matched == len(sample_player_ids):
            break

    sample_player_count = len(sample_player_ids)
    if best_dg is not None and best_match == sample_player_count:
        effective = int(best_dg)
        requested = int(requested_draft_group_id) if requested_draft_group_id is not None else None
        return effective, {
            "requested_draft_group_id": requested,
            "effective_draft_group_id": effective,
            "sample_player_count": sample_player_count,
            "requested_match_count": int(requested_match) if requested_match is not None else None,
            "best_match_count": int(best_match),
            "best_match_draft_group_id": int(best_dg),
            "inference_reason": "perfect_coverage",
            "inferred_from_lineups": requested != effective,
        }

    # Non-perfect fallback for stale DK draft groups:
    # if another slate has much stronger lineup-player coverage than the requested slate,
    # trust that slate even when one or two sample players are missing from pool metadata.
    if requested_draft_group_id is not None and requested_match is not None and best_dg is not None:
        requested = int(requested_draft_group_id)
        effective = int(best_dg)
        min_support = max(2, int(math.ceil(0.60 * sample_player_count)))
        min_improvement = max(2, int(math.ceil(0.15 * sample_player_count)))
        if (
            effective != requested
            and best_match >= min_support
            and (best_match - int(requested_match)) >= min_improvement
        ):
            return effective, {
                "requested_draft_group_id": requested,
                "effective_draft_group_id": effective,
                "sample_player_count": sample_player_count,
                "requested_match_count": int(requested_match),
                "best_match_count": int(best_match),
                "best_match_draft_group_id": effective,
                "min_support_required": int(min_support),
                "min_improvement_required": int(min_improvement),
                "inference_reason": "coverage_improvement",
                "inferred_from_lineups": True,
            }

    return requested_draft_group_id, {
        "requested_draft_group_id": int(requested_draft_group_id) if requested_draft_group_id is not None else None,
        "effective_draft_group_id": int(requested_draft_group_id) if requested_draft_group_id is not None else None,
        "sample_player_count": sample_player_count,
        "requested_match_count": int(requested_match) if requested_match is not None else None,
        "best_match_count": int(best_match) if best_match >= 0 else None,
        "best_match_draft_group_id": int(best_dg) if best_dg is not None else None,
        "inference_reason": "no_high_confidence_match",
        "inferred_from_lineups": False,
    }


def _normalize_ownership_mode(mode: str | None) -> str:
    m = str(mode or "full").strip().lower()
    aliases = {
        "on": "full",
        "none": "off",
        "no": "off",
        "false": "off",
        "true": "full",
    }
    m = aliases.get(m, m)
    allowed = {"full", "off", "dupe_only", "field_only"}
    if m not in allowed:
        raise ValueError(f"Invalid ownership_mode: {mode!r} (allowed: {sorted(allowed)})")
    return m


def _normalize_rank_mode(mode: str | None) -> str:
    m = str(mode or "current").strip().lower()
    allowed = {"current", "tail_only", "tail_times_dupe"}
    if m not in allowed:
        raise ValueError(f"Invalid rank_mode: {mode!r} (allowed: {sorted(allowed)})")
    return m


def _sim_builds_dir() -> Path:
    """Get the contest sim builds directory under projections-data."""
    return paths.data_path() / "builds" / "contest_sim"


def _save_sim_build(game_date: str, build_data: Dict[str, object]) -> str:
    builds_root = _sim_builds_dir() / game_date
    builds_root.mkdir(parents=True, exist_ok=True)
    build_id = build_data.get("build_id") or str(uuid4())
    build_path = builds_root / f"{build_id}.json"
    build_data["build_id"] = build_id
    with open(build_path, "w") as f:
        import json
        json.dump(build_data, f, indent=2)
    return str(build_id)


def _load_sim_build(game_date: str, build_id: str) -> Optional[Dict[str, object]]:
    build_path = _sim_builds_dir() / game_date / f"{build_id}.json"
    if not build_path.exists():
        return None
    with open(build_path) as f:
        import json
        return json.load(f)


def _delete_sim_build(game_date: str, build_id: str) -> bool:
    build_path = _sim_builds_dir() / game_date / f"{build_id}.json"
    if build_path.exists():
        build_path.unlink()
        return True
    return False


def _list_sim_builds(game_date: str) -> List[Dict[str, object]]:
    builds_root = _sim_builds_dir() / game_date
    if not builds_root.exists():
        return []
    builds = []
    for build_file in sorted(builds_root.glob("*.json"), reverse=True):
        try:
            import json
            with open(build_file) as f:
                data = json.load(f)
            builds.append({
                "build_id": data.get("build_id", build_file.stem),
                "game_date": data.get("game_date"),
                "site": _normalize_site(str(data.get("site") or "dk")),
                "draft_group_id": data.get("draft_group_id"),
                "created_at": data.get("created_at"),
                "lineups_count": data.get("lineups_count"),
                "name": data.get("name"),
                "kind": data.get("kind", "run"),
                "stats": data.get("stats", {}),
            })
        except Exception as exc:
            logger.warning("Failed to read sim build %s: %s", build_file, exc)
            continue
    return builds


def _load_player_ownership(
    game_date: str,
    *,
    site: str = "dk",
    run_id: str | None = None,
    draft_group_id: int | None = None,
    use_strategy_overrides: bool = False,
) -> Dict[str, float]:
    """Load player_id -> ownership % mapping for contest-sim dupe penalties."""
    if draft_group_id is not None:
        try:
            site_norm = _normalize_site(site)
            pool = build_player_pool(
                game_date=game_date,
                draft_group_id=int(draft_group_id),
                site=site_norm,
                run_id=run_id,
                data_root=paths.data_path(),
                use_user_overrides=bool(use_strategy_overrides),
                ownership_mode="renormalize",
            )
            result = {
                _canonicalize_player_id(player.get("player_id")): float(player["own_proj"])
                for player in pool
                if _canonicalize_player_id(player.get("player_id")) and player.get("own_proj") is not None
            }
            if result:
                logger.info(
                    "Loaded ownership for %d players from optimizer pool (%s dg=%s)",
                    len(result),
                    game_date,
                    draft_group_id,
                )
                return result
        except Exception as exc:
            logger.warning(
                "Failed to load slate-specific ownership from optimizer pool for %s dg=%s: %s",
                game_date,
                draft_group_id,
                exc,
            )

    try:
        df = load_projections_for_date(game_date, run_id=run_id, data_root=paths.data_path())
        if "player_id" in df.columns and "pred_own_pct" in df.columns:
            ownership = df.dropna(subset=["pred_own_pct"])
            result: Dict[str, float] = {}
            for _, row in ownership.iterrows():
                pid = _canonicalize_player_id(row.get("player_id"))
                if not pid:
                    continue
                result[pid] = float(row.get("pred_own_pct"))
            if result:
                logger.info("Loaded ownership for %d players from projections bundle", len(result))
                return result
    except Exception as exc:
        logger.warning("Failed to load ownership from projections bundle for %s: %s", game_date, exc)

    logger.info("No ownership data available, dupe penalties disabled")
    return {}


def _lineup_result_to_dict(result: Dict[str, object]) -> Dict[str, object]:
    return dict(result)


def _summary_stats_from_results(
    results: List[Dict[str, object]],
    *,
    worlds_count: int,
    debug: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    if not results:
        return {
            "lineup_count": 0,
            "worlds_count": int(worlds_count),
            "avg_ev": 0.0,
            "avg_roi": 0.0,
            "positive_ev_count": 0,
            "best_ev_lineup_id": -1,
            "best_win_rate_lineup_id": -1,
            "best_top1pct_lineup_id": -1,
            "debug": debug or {},
        }

    avg_ev = sum(float(r.get("expected_value", 0.0) or 0.0) for r in results) / len(results)
    avg_roi = sum(float(r.get("roi", 0.0) or 0.0) for r in results) / len(results)
    positive_ev_count = sum(1 for r in results if float(r.get("expected_value", 0.0) or 0.0) >= 0.0)
    best_ev = max(results, key=lambda r: float(r.get("expected_value", float("-inf")) or float("-inf")))
    best_win = max(results, key=lambda r: float(r.get("win_rate", float("-inf")) or float("-inf")))
    best_top = max(results, key=lambda r: float(r.get("top_1pct_rate", float("-inf")) or float("-inf")))
    return {
        "lineup_count": len(results),
        "worlds_count": int(worlds_count),
        "avg_ev": float(avg_ev),
        "avg_roi": float(avg_roi),
        "positive_ev_count": int(positive_ev_count),
        "best_ev_lineup_id": int(best_ev.get("lineup_id", -1)),
        "best_win_rate_lineup_id": int(best_win.get("lineup_id", -1)),
        "best_top1pct_lineup_id": int(best_top.get("lineup_id", -1)),
        "debug": debug or {},
    }


def _result_to_candidate(result: Dict[str, object], *, total_own: float) -> PortfolioCandidate:
    return PortfolioCandidate(
        lineup_id=int(result["lineup_id"]),
        player_ids=tuple(str(pid) for pid in result.get("player_ids", [])),
        mean=result.get("mean"),
        p90=result.get("p90"),
        p95=result.get("p95"),
        expected_value=result.get("expected_value"),
        roi=result.get("roi"),
        win_rate=result.get("win_rate"),
        top_1pct_rate=result.get("top_1pct_rate"),
        top_5pct_rate=result.get("top_5pct_rate"),
        top_10pct_rate=result.get("top_10pct_rate"),
        cash_rate=result.get("cash_rate"),
        total_own=total_own,
        ucv90=result.get("ucv90"),
        tail_score=result.get("tail_score"),
        select_score=result.get("select_score"),
        score_lcb95=result.get("score_lcb95"),
        score_cvar10=result.get("score_cvar10"),
        robust_floor=result.get("robust_floor"),
    )


def _build_exposure_bounds(
    raw_bounds: Dict[str, PortfolioExposureBoundsRequest],
) -> Dict[str, ExposureBoundsPct]:
    return {
        str(pid): ExposureBoundsPct(min=bounds.min, max=bounds.max)
        for pid, bounds in raw_bounds.items()
    }


def _resolve_train_holdout_indices(
    world_count: int,
    *,
    world_indices: Optional[List[int]],
    worlds_train_frac: Optional[float],
    seed: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    if world_indices is not None:
        train_idx = np.asarray(world_indices, dtype=np.int64)
        if train_idx.ndim != 1 or train_idx.size < 2:
            raise ValueError("world_indices must be a 1D array with at least 2 entries")
        if np.min(train_idx) < 0 or np.max(train_idx) >= world_count:
            raise ValueError("world_indices contain out-of-range values")
        all_idx = np.arange(world_count, dtype=np.int64)
        holdout_mask = np.ones((world_count,), dtype=bool)
        holdout_mask[train_idx] = False
        holdout_idx = all_idx[holdout_mask]
        return train_idx, holdout_idx, "explicit_indices"

    if worlds_train_frac is None:
        return None, None, "all_worlds"

    if not (0.0 < float(worlds_train_frac) < 1.0):
        raise ValueError("worlds_train_frac must be in (0, 1)")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(world_count))
    n_train = int(np.floor(float(worlds_train_frac) * float(world_count)))
    n_train = max(1, min(n_train, int(world_count) - 1))
    train_idx = np.sort(perm[:n_train].astype(np.int64))
    holdout_idx = np.sort(perm[n_train:].astype(np.int64))
    return train_idx, holdout_idx, f"train_frac:{float(worlds_train_frac):.3f}"


def _compute_portfolio_risk(
    selected: List[PortfolioCandidate],
    *,
    worlds_matrix,
    player_index: Dict[str, int],
    world_indices: Optional[np.ndarray] = None,
) -> Optional[float]:
    if not selected:
        return None

    player_ids: List[str] = []
    seen: set[str] = set()
    for candidate in selected:
        for pid in candidate.player_ids:
            pid_s = str(pid)
            if pid_s in seen or pid_s not in player_index:
                continue
            seen.add(pid_s)
            player_ids.append(pid_s)

    if not player_ids:
        return None

    worlds = np.asarray(worlds_matrix, dtype=np.float64)
    if world_indices is not None:
        if world_indices.size < 2:
            return None
        worlds = worlds[np.asarray(world_indices, dtype=np.int64)]
    if worlds.shape[0] < 2:
        return None

    cols = np.asarray([player_index[pid] for pid in player_ids], dtype=np.int64)
    worlds_sub = np.take(worlds, cols, axis=1)
    mu = worlds_sub.mean(axis=0)
    centered = worlds_sub - mu
    sigma = (centered.T @ centered) / float(worlds_sub.shape[0] - 1)
    sigma = np.asarray(sigma, dtype=np.float64)

    pid_to_local = {pid: idx for idx, pid in enumerate(player_ids)}
    counts = np.zeros((len(player_ids),), dtype=np.float64)
    for candidate in selected:
        for pid in candidate.player_ids:
            loc = pid_to_local.get(str(pid))
            if loc is not None:
                counts[loc] += 1.0
    return float(counts @ (sigma @ counts))


def _sort_portfolio_candidates(
    candidates: List[PortfolioCandidate],
    *,
    sort_key: str,
    sort_dir: str,
) -> List[PortfolioCandidate]:
    multiplier = -1.0 if sort_dir == "desc" else 1.0
    return sorted(
        candidates,
        key=lambda candidate: (
            get_candidate_metric_value(candidate, sort_key=sort_key) is None,
            0.0
            if get_candidate_metric_value(candidate, sort_key=sort_key) is None
            else multiplier * float(get_candidate_metric_value(candidate, sort_key=sort_key)),
            candidate.lineup_id,
        ),
    )


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class ContestSimRequest(BaseModel):
    """Request to run a contest simulation."""

    game_date: str = Field(..., description="Game date in YYYY-MM-DD format")
    site: str = Field(default="dk", description="DFS site: dk or fd")
    run_id: Optional[str] = Field(default=None, description="Optional projections run_id (defaults to blessed/pinned/latest)")
    draft_group_id: Optional[int] = Field(default=None, description="Draft group ID")
    lineups: List[List[str]] = Field(..., description="List of lineups (each a list of player_ids)")
    field_mode: str = Field(
        default="self_play",
        description="Field mode: self_play | generated_field",
    )
    field_library_version: str = Field(
        default="v0",
        description="Field library version to load/build (or 'latest')",
    )
    field_library_k: int = Field(
        default=2500,
        description="Target unique field lineups K for generated_field",
    )
    field_candidate_pool_size: int = Field(
        default=40000,
        description="QuickBuild candidate pool size for generated_field",
    )
    field_library_rebuild: bool = Field(
        default=False,
        description="Force rebuild field library even if cached",
    )
    field_library_rebuild_candidates: bool = Field(
        default=False,
        description="Force rebuild raw candidate pool for generated_field",
    )
    archetype: str = Field(
        default="GPP Standard (20% paid)",
        description="Payout archetype name from contest_sim.yaml",
    )
    field_size_bucket: str = Field(
        default="5000",
        description="Field size bucket (numeric string like '5000' or label from contest_sim.yaml)",
    )
    field_size_override: Optional[int] = Field(default=None, description="Exact field size (overrides bucket)")
    entry_fee: float = Field(default=3.0, description="Entry fee per lineup")
    weights: Optional[List[int]] = Field(default=None, description="Entry counts per lineup")
    entry_max: int = Field(default=150, description="Max entries per user (for dupe penalty)")
    ownership_mode: str = Field(
        default="field_only",
        description="Ownership usage: field_only | full | dupe_only | off",
    )
    rank_mode: str = Field(
        default="current",
        description="Ranking mode for select_score: current | tail_only | tail_times_dupe",
    )
    worlds_source: Literal["gtv2", "sim_v2"] = Field(
        default="gtv2",
        description="Worlds family for contest sim scoring: gtv2 (live default) or sim_v2 (explicit backtest fallback)",
    )
    use_strategy_overrides: bool = Field(
        default=False,
        description="Apply persistent slate strategy overrides to downstream worlds before scoring",
    )


class LineupEVResultResponse(BaseModel):
    """Per-lineup simulation results."""

    lineup_id: int
    player_ids: List[str]
    mean: float
    std: float
    p90: float
    p95: float
    expected_payout: float
    expected_value: float
    roi: float
    win_rate: float
    top_1pct_rate: float
    top_5pct_rate: float
    top_10pct_rate: float
    cash_rate: float
    dupe_penalty: float = 1.0  # E[1/K], 1.0 = no penalty
    unadjusted_expected_payout: Optional[float] = None  # expected_payout before dupe penalty
    adjusted_expected_payout: Optional[float] = None  # expected_payout * dupe_penalty
    # Tail / upside selection metrics
    ucv90: Optional[float] = None  # Upper CVaR at 90th pctile (mean of top 10% scores)
    tail_score: Optional[float] = None  # Weighted combo: 0.6*p90 + 0.4*ucv90
    select_score: Optional[float] = None  # tail_score - dupe penalty impact
    score_lcb95: Optional[float] = None  # mean - 1.96*std (score-space lower confidence bound)
    score_cvar10: Optional[float] = None  # mean score in worst 10% worlds
    robust_floor: Optional[float] = None  # min(score_lcb95, score_cvar10)


class ContestConfigResponse(BaseModel):
    """Contest configuration."""

    field_size: int
    entry_fee: float
    archetype: str
    rake: float
    prize_pool: float


class SummaryStatsResponse(BaseModel):
    """Summary statistics."""

    lineup_count: int
    worlds_count: int
    avg_ev: float
    avg_roi: float
    positive_ev_count: int
    best_ev_lineup_id: int
    best_win_rate_lineup_id: int
    best_top1pct_lineup_id: int
    debug: Dict[str, object] = Field(default_factory=dict)


class ContestSimResponse(BaseModel):
    """Complete contest simulation response."""

    results: List[LineupEVResultResponse]
    config: ContestConfigResponse
    stats: SummaryStatsResponse
    build_id: Optional[str] = None


class FieldSizeOption(BaseModel):
    """Field size bucket option."""

    key: str
    label: str
    default: int
    range: List[int]


class PayoutArchetypeOption(BaseModel):
    """Payout archetype option."""

    key: str
    label: str
    first_place_pct: float
    itm_pct: float


class ConfigResponse(BaseModel):
    """Available configuration options."""

    field_sizes: List[FieldSizeOption]
    payout_archetypes: List[PayoutArchetypeOption]
    default_entry_fee: float
    default_archetype: str
    default_field_size_bucket: str


class SavedSimBuildSummary(BaseModel):
    build_id: str
    game_date: str
    site: str = "dk"
    draft_group_id: Optional[int] = None
    created_at: str
    lineups_count: int
    name: Optional[str] = None
    kind: str = "run"  # run | lineups | portfolio
    stats: Dict[str, object] = Field(default_factory=dict)


class SavedSimBuildDetail(BaseModel):
    build_id: str
    game_date: str
    site: str = "dk"
    draft_group_id: Optional[int] = None
    created_at: str
    lineups_count: int
    name: Optional[str] = None
    kind: str = "run"
    config: Optional[Dict[str, object]] = None
    stats: Dict[str, object]
    results: Optional[List[LineupEVResultResponse]] = None
    lineups: List[List[str]]
    request: Optional[Dict[str, object]] = None


class SaveSimLineupsRequest(BaseModel):
    game_date: str
    site: str = "dk"
    draft_group_id: Optional[int] = None
    name: str
    lineups: List[List[str]]
    kind: Literal["lineups", "portfolio"] = "lineups"
    results: Optional[List[LineupEVResultResponse]] = None
    config: Optional[ContestConfigResponse] = None
    stats: Optional[SummaryStatsResponse] = None
    source_build_id: Optional[str] = None
    selection_mode: Optional[str] = None
    selection_config: Optional[Dict[str, object]] = None
    selection_diagnostics: Optional[Dict[str, object]] = None
    warnings: List[str] = Field(default_factory=list)


class PortfolioExposureBoundsRequest(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None


class PortfolioSelectionRequest(BaseModel):
    game_date: str
    site: Optional[str] = None
    draft_group_id: Optional[int] = None
    source_build_id: str
    mode: Literal["greedy_constraints", "decorrelated_ev", "weighted_allocations"] = "decorrelated_ev"
    worlds_source: Literal["gtv2", "sim_v2"] = "gtv2"
    sort_key: Literal[
        "lineup_id",
        "mean",
        "p90",
        "p95",
        "expected_value",
        "roi",
        "win_rate",
        "top_1pct_rate",
        "top_5pct_rate",
        "top_10pct_rate",
        "cash_rate",
        "total_own",
        "ucv90",
        "tail_score",
        "select_score",
        "score_lcb95",
        "score_cvar10",
        "robust_floor",
    ] = "expected_value"
    sort_dir: Literal["asc", "desc"] = "desc"
    portfolio_size: int = Field(..., ge=1)
    ev_retention: float = Field(default=0.99, gt=0.0, le=1.0)
    worlds_sample: int = Field(default=5000, ge=1)
    worlds_train_frac: Optional[float] = Field(default=0.8, gt=0.0, lt=1.0)
    world_indices: Optional[List[int]] = None
    min_uniques: int = Field(default=0, ge=0)
    max_total_own: Optional[float] = None
    filter_positive_ev: bool = False
    top_n: Optional[int] = Field(default=None, ge=1)
    candidate_lineup_ids: Optional[List[int]] = None
    seed_lineup_ids: Optional[List[int]] = None
    exposure_bounds: Dict[str, PortfolioExposureBoundsRequest] = Field(default_factory=dict)
    seed: int = 42


class PortfolioSelectionResponse(BaseModel):
    site: str = "dk"
    mode: str
    source_build_id: str
    candidate_count: int
    filtered_candidate_count: int
    selected_lineup_ids: List[int]
    selected_results: List[LineupEVResultResponse]
    selected_lineups: List[List[str]]
    weights: Optional[List[int]] = None
    diagnostics: Dict[str, object] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class FieldLibrarySummaryResponse(BaseModel):
    """Summary of a cached field library."""

    version: str
    site: str = "dk"
    path: str
    game_date: str
    draft_group_id: int
    method: Optional[str] = None
    generated_at: Optional[str] = None
    selected_k: int
    weights_sum: int
    meta: Dict[str, object] = Field(default_factory=dict)


class BuildFieldLibraryRequest(BaseModel):
    game_date: str
    site: str = "dk"
    draft_group_id: int
    version: str = "v0"
    k: int = 2500
    candidate_pool_size: int = 40000
    rebuild: bool = False
    rebuild_candidates: bool = False
    ownership_mode: str = Field(
        default="field_only",
        description="Ownership usage: field_only | full | dupe_only | off",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/run", response_model=ContestSimResponse)
async def run_simulation(request: ContestSimRequest):
    """Run contest simulation for the given lineups.

    User lineups compete against a modeled opponent field.
    
    Dupe penalties and field generation can be toggled via `ownership_mode`.
    """
    try:
        try:
            site_norm = _normalize_site(request.site)
            ownership_mode = _normalize_ownership_mode(request.ownership_mode)
            rank_mode = _normalize_rank_mode(request.rank_mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        if ownership_mode == "off":
            rank_mode = "tail_only"

        effective_draft_group_id, draft_group_resolution = _resolve_draft_group_id_for_lineups(
            game_date=request.game_date,
            lineups=request.lineups,
            site=site_norm,
            run_id=request.run_id,
            requested_draft_group_id=request.draft_group_id,
        )
        if draft_group_resolution.get("inferred_from_lineups"):
            logger.warning(
                "Contest sim inferred draft_group_id %s from lineups (requested=%s, date=%s, site=%s)",
                draft_group_resolution.get("effective_draft_group_id"),
                draft_group_resolution.get("requested_draft_group_id"),
                request.game_date,
                site_norm,
            )

        if request.use_strategy_overrides and effective_draft_group_id is None:
            raise HTTPException(
                status_code=400,
                detail="draft_group_id is required when use_strategy_overrides=true",
            )

        use_dupe_ownership = ownership_mode in {"full", "dupe_only"}
        use_field_ownership = ownership_mode in {"full", "field_only"}

        normalized_user_lineups: List[List[str]]
        try:
            normalized_user_lineups = _normalize_lineups_for_site(
                request.lineups,
                game_date=request.game_date,
                draft_group_id=effective_draft_group_id,
                site=site_norm,
                run_id=request.run_id,
                context="lineups",
            )
        except ValueError as exc:
            recovered = False
            if site_norm == "dk":
                candidate_dgs = _candidate_draft_group_ids_for_resolution(request.game_date, site_norm)
                for dg in candidate_dgs:
                    if effective_draft_group_id is not None and int(dg) == int(effective_draft_group_id):
                        continue
                    try:
                        normalized_user_lineups = _normalize_lineups_for_site(
                            request.lineups,
                            game_date=request.game_date,
                            draft_group_id=int(dg),
                            site=site_norm,
                            run_id=request.run_id,
                            context="lineups",
                        )
                        previous_dg = (
                            int(effective_draft_group_id)
                            if effective_draft_group_id is not None
                            else None
                        )
                        effective_draft_group_id = int(dg)
                        draft_group_resolution = dict(draft_group_resolution)
                        draft_group_resolution.update(
                            {
                                "requested_draft_group_id": (
                                    int(request.draft_group_id)
                                    if request.draft_group_id is not None
                                    else None
                                ),
                                "effective_draft_group_id": int(dg),
                                "fallback_previous_draft_group_id": previous_dg,
                                "fallback_trigger": str(exc),
                                "inference_reason": "validation_retry_success",
                                "inferred_from_lineups": True,
                            }
                        )
                        logger.warning(
                            "Contest sim recovered from stale DK draft_group_id via validation retry: %s -> %s (%s)",
                            previous_dg,
                            dg,
                            exc,
                        )
                        recovered = True
                        break
                    except ValueError:
                        continue
            if not recovered:
                raise

        # Load ownership data for dupe penalty calculation (only when enabled)
        player_ownership = (
            _load_player_ownership(
                request.game_date,
                site=site_norm,
                run_id=request.run_id,
                draft_group_id=effective_draft_group_id,
                use_strategy_overrides=bool(request.use_strategy_overrides),
            )
            if use_dupe_ownership
            else {}
        )

        field_lineups = None
        field_weights = None
        field_library_info: Dict[str, object] = {}
        if request.field_mode not in {"self_play", "generated_field"}:
            raise HTTPException(status_code=400, detail=f"Invalid field_mode: {request.field_mode}")

        if request.field_mode == "generated_field":
            if effective_draft_group_id is None:
                raise HTTPException(
                    status_code=400,
                    detail="draft_group_id is required when field_mode=generated_field",
                )
            # Cache separation: avoid overwriting ownership-aware field libraries with ownership-free builds.
            version = request.field_library_version
            if not use_field_ownership:
                version = "v0" if version == "latest" else version
                if not str(version).endswith("_noown"):
                    version = f"{version}_noown"

            library, lib_path, built_now = load_or_build_field_library(
                game_date=request.game_date,
                draft_group_id=int(effective_draft_group_id),
                site=site_norm,
                version=version,
                k=int(request.field_library_k),
                candidate_pool_size=int(request.field_candidate_pool_size),
                rebuild=bool(request.field_library_rebuild),
                rebuild_candidates=bool(request.field_library_rebuild_candidates),
                use_ownership_features=use_field_ownership,
            )
            field_lineups = _normalize_lineups_for_site(
                library.lineups,
                game_date=request.game_date,
                draft_group_id=effective_draft_group_id,
                site=site_norm,
                run_id=request.run_id,
                context="field_lineups",
            )
            field_weights = library.weights
            field_library_info = {
                "site": site_norm,
                "field_mode": request.field_mode,
                "field_library_path": str(lib_path),
                "field_library_built_now": built_now,
                "field_library_version": library.meta.get("version", request.field_library_version),
                "field_library_method": library.meta.get("method"),
                "field_library_selected_k": len(library.lineups),
                "field_library_weights_sum": int(sum(library.weights)),
                "ownership_mode": ownership_mode,
                "rank_mode": rank_mode,
                "field_library_use_ownership": bool(use_field_ownership),
            }
        
        result = run_contest_simulation(
            user_lineups=normalized_user_lineups,
            game_date=request.game_date,
            site=site_norm,
            draft_group_id=effective_draft_group_id,
            run_id=request.run_id,
            archetype=request.archetype,
            field_size_bucket=request.field_size_bucket,
            field_size_override=request.field_size_override,
            entry_fee=request.entry_fee,
            user_weights=request.weights,
            field_lineups=field_lineups,
            field_weights=field_weights,
            player_ownership=player_ownership if (use_dupe_ownership and player_ownership) else None,
            entry_max=request.entry_max,
            ownership_mode=ownership_mode,
            rank_mode=rank_mode,
            worlds_source=request.worlds_source,
            use_strategy_overrides=request.use_strategy_overrides,
        )
        if field_library_info:
            result.stats.debug.update(field_library_info)
        else:
            result.stats.debug.update(
                {
                    "site": site_norm,
                    "ownership_mode": ownership_mode,
                    "rank_mode": rank_mode,
                    "worlds_source": request.worlds_source,
                }
            )
        if draft_group_resolution:
            result.stats.debug["draft_group_resolution"] = draft_group_resolution

        build_data = {
            "build_id": str(uuid4()),
            "game_date": request.game_date,
            "site": site_norm,
            "draft_group_id": effective_draft_group_id,
            "created_at": datetime.utcnow().isoformat(),
            "lineups_count": len(normalized_user_lineups),
            "kind": "run",
            "name": None,
            "config": {
                "field_size": result.config.field_size,
                "entry_fee": result.config.entry_fee,
                "archetype": result.config.archetype,
                "rake": result.config.rake,
                "prize_pool": result.config.prize_pool,
            },
            "stats": result.stats.to_dict(),
            "results": [r.to_dict() for r in result.results],
            "lineups": normalized_user_lineups,
            "request": request.model_dump(),
        }
        build_id = _save_sim_build(request.game_date, build_data)

        return ContestSimResponse(
            results=[
                LineupEVResultResponse(**r.to_dict())
                for r in result.results
            ],
            config=ContestConfigResponse(
                field_size=result.config.field_size,
                entry_fee=result.config.entry_fee,
                archetype=result.config.archetype,
                rake=result.config.rake,
                prize_pool=result.config.prize_pool,
            ),
            stats=SummaryStatsResponse(**result.stats.to_dict()),
            build_id=build_id,
        )

    except ValueError as e:
        logger.error("Contest simulation validation failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Worlds data not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Contest simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get available configuration options for contest simulation."""
    try:
        config = load_config()

        # field_sizes is a list of dicts with 'label' and 'value' keys
        field_sizes = []
        for i, fs in enumerate(config.get("field_sizes", [])):
            field_sizes.append(FieldSizeOption(
                key=str(fs.get("value", i)),
                label=fs.get("label", f"Size {i}"),
                default=fs.get("value", 25000),
                range=[0, fs.get("value", 25000) * 2],
            ))

        # payout_archetypes is a list of dicts with 'name', 'field_paid_pct', 'payout_table'
        archetypes = []
        for i, arch in enumerate(config.get("payout_archetypes", [])):
            archetypes.append(PayoutArchetypeOption(
                key=arch.get("name", f"archetype_{i}"),
                label=arch.get("name", f"Archetype {i}"),
                first_place_pct=arch.get("first_place_pct", 0.2),
                itm_pct=arch.get("field_paid_pct", 20.0) / 100.0,
            ))

        defaults = config.get("defaults", {})

        return ConfigResponse(
            field_sizes=field_sizes,
            payout_archetypes=archetypes,
            default_entry_fee=defaults.get("entry_fee", 3.0),
            default_archetype=defaults.get("archetype", "GPP Standard (20% paid)"),
            default_field_size_bucket=defaults.get("field_size_bucket", "5000"),
        )

    except Exception as e:
        logger.exception(f"Failed to load config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/saved-builds", response_model=List[SavedSimBuildSummary])
async def list_saved_sim_builds(
    date: str,
    kind: Optional[str] = None,
    site: Optional[str] = Query(default=None, description="Optional site filter: dk or fd"),
):
    """List saved contest sim builds for a date."""
    builds = _list_sim_builds(date)
    if site is not None:
        site_norm = _normalize_site(site)
        filtered: List[Dict[str, object]] = []
        for b in builds:
            try:
                build_site = _normalize_site(str(b.get("site") or "dk"))
            except ValueError:
                continue
            if build_site == site_norm:
                filtered.append(b)
        builds = filtered
    if kind:
        builds = [b for b in builds if b.get("kind") == kind]
    return builds


def _backfill_tail_metrics(results: List[Dict]) -> List[Dict]:
    """Backfill ucv90/tail_score/select_score for legacy saved builds."""
    tail_weight_p90 = 0.6
    tail_weight_ucv = 0.4
    # Standard-normal CVaR at alpha=10%: mu - sigma * (phi(z_alpha)/alpha)
    cvar10_sigma_mult = 1.755
    for r in results:
        # Skip if already has tail metrics
        if r.get("ucv90") is not None:
            continue
        # Compute from p90 - for legacy builds we estimate UCVaR as ~p90 * 1.05
        # (a rough heuristic since we don't have the full score distribution)
        p90 = r.get("p90")
        mean = r.get("mean")
        dupe_penalty = r.get("dupe_penalty", 1.0)
        if p90 is not None and mean is not None:
            # Estimate UCVaR90 as p90 + 0.5 * (p90 - mean) based on typical distributions
            ucv90_est = p90 + 0.5 * max(0, p90 - mean)
            r["ucv90"] = round(ucv90_est, 2)
            tail_score = tail_weight_p90 * p90 + tail_weight_ucv * ucv90_est
            r["tail_score"] = round(tail_score, 2)
            penalty_impact = (1.0 - dupe_penalty) * mean
            r["select_score"] = round(tail_score - penalty_impact, 2)
        if r.get("score_lcb95") is None and mean is not None:
            std = r.get("std")
            if std is not None:
                lcb95 = float(mean) - 1.96 * float(std)
                cvar10 = float(mean) - cvar10_sigma_mult * float(std)
                r["score_lcb95"] = round(lcb95, 2)
                r["score_cvar10"] = round(cvar10, 2)
                r["robust_floor"] = round(min(lcb95, cvar10), 2)
    return results


@router.get("/saved-builds/{build_id}", response_model=SavedSimBuildDetail)
async def load_saved_sim_build(build_id: str, date: str):
    """Load a saved contest sim build with lineups/results."""
    data = _load_sim_build(date, build_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Sim build {build_id} not found for date {date}")
    results = data.get("results")
    # Backfill tail metrics for legacy builds
    if results:
        results = _backfill_tail_metrics(results)
    return SavedSimBuildDetail(
        build_id=data.get("build_id", build_id),
        game_date=data.get("game_date", date),
        site=_normalize_site(str(data.get("site") or "dk")),
        draft_group_id=data.get("draft_group_id"),
        created_at=data.get("created_at", datetime.utcnow().isoformat()),
        lineups_count=data.get("lineups_count", 0),
        name=data.get("name"),
        kind=data.get("kind", "run"),
        config=data.get("config"),
        stats=data.get("stats", {}),
        results=[LineupEVResultResponse(**r) for r in results] if results else None,
        lineups=data.get("lineups", []),
        request=data.get("request"),
    )


@router.post("/saved-lineups", response_model=SavedSimBuildSummary)
async def save_sim_lineups(request: SaveSimLineupsRequest):
    """Save a named lineup set derived from contest sim results."""
    try:
        site_norm = _normalize_site(request.site)
        if not request.lineups:
            raise HTTPException(status_code=400, detail="No lineups provided")
        if not request.results or not request.config or not request.stats:
            raise HTTPException(status_code=400, detail="Snapshot results/config/stats are required to save sim lineups")

        if site_norm == "fd":
            normalized_lineups = _normalize_lineups_for_site(
                request.lineups,
                game_date=request.game_date,
                draft_group_id=request.draft_group_id,
                site=site_norm,
                run_id=None,
                context="lineups",
            )
        else:
            normalized_lineups = [
                [str(pid).strip() for pid in lineup if str(pid).strip()]
                for lineup in request.lineups
            ]

        results_payload = [r.model_dump() for r in request.results] if request.results else None
        if results_payload:
            base_debug = (
                request.stats.model_dump().get("debug", {})
                if request.stats
                else {}
            )
            stats_payload = _summary_stats_from_results(
                results_payload,
                worlds_count=request.stats.worlds_count if request.stats else 0,
                debug=dict(base_debug),
            )
        else:
            stats_payload = request.stats.model_dump() if request.stats else {}
        if stats_payload:
            debug_payload = dict(stats_payload.get("debug") or {})
            debug_payload["site"] = site_norm
            if request.selection_mode:
                debug_payload["selection_mode"] = request.selection_mode
            if request.source_build_id:
                debug_payload["source_build_id"] = request.source_build_id
            if request.selection_config:
                debug_payload["selection_config"] = request.selection_config
            if request.selection_diagnostics:
                debug_payload["selection"] = request.selection_diagnostics
            if request.warnings:
                debug_payload["selection_warnings"] = list(request.warnings)
            stats_payload["debug"] = debug_payload

        build_data = {
            "build_id": str(uuid4()),
            "game_date": request.game_date,
            "site": site_norm,
            "draft_group_id": request.draft_group_id,
            "created_at": datetime.utcnow().isoformat(),
            "lineups_count": len(normalized_lineups),
            "kind": request.kind,
            "name": request.name,
            "stats": stats_payload,
            "config": request.config.model_dump() if request.config else None,
            "results": results_payload,
            "lineups": normalized_lineups,
            "request": {
                "site": site_norm,
                "source_build_id": request.source_build_id,
                "selection_mode": request.selection_mode,
                "selection_config": request.selection_config,
                "selection_diagnostics": request.selection_diagnostics,
                "warnings": request.warnings,
            },
        }
        build_id = _save_sim_build(request.game_date, build_data)
        return SavedSimBuildSummary(
            build_id=build_id,
            game_date=request.game_date,
            site=site_norm,
            draft_group_id=request.draft_group_id,
            created_at=build_data["created_at"],
            lineups_count=len(normalized_lineups),
            name=request.name,
            kind=request.kind,
            stats=build_data["stats"] or {},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/portfolio", response_model=PortfolioSelectionResponse)
async def select_portfolio(request: PortfolioSelectionRequest):
    """Build an authoritative portfolio from a saved contest-sim build."""
    if request.mode == "weighted_allocations":
        raise HTTPException(
            status_code=400,
            detail="weighted_allocations is not implemented yet",
        )

    source_build = _load_sim_build(request.game_date, request.source_build_id)
    if not source_build:
        raise HTTPException(
            status_code=404,
            detail=f"Sim build {request.source_build_id} not found for date {request.game_date}",
        )

    raw_results = source_build.get("results")
    if not raw_results:
        raise HTTPException(
            status_code=400,
            detail="Source build does not contain contest-sim results",
        )

    try:
        source_site = _normalize_site(str(source_build.get("site") or "dk"))
    except ValueError:
        source_site = "dk"
    if request.site is not None and _normalize_site(request.site) != source_site:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Portfolio request site={request.site} does not match source build site={source_site}"
            ),
        )

    run_request = source_build.get("request") or {}
    source_run_id = run_request.get("run_id")
    draft_group_id = (
        request.draft_group_id
        if request.draft_group_id is not None
        else source_build.get("draft_group_id")
    )
    ownership = _load_player_ownership(
        request.game_date,
        site=source_site,
        run_id=str(source_run_id) if source_run_id else None,
        draft_group_id=int(draft_group_id) if draft_group_id is not None else None,
        use_strategy_overrides=bool(run_request.get("use_strategy_overrides", False)),
    )

    results = _backfill_tail_metrics([dict(r) for r in raw_results])
    if request.candidate_lineup_ids is not None:
        allowed_ids = {int(lineup_id) for lineup_id in request.candidate_lineup_ids}
        results = [r for r in results if int(r.get("lineup_id", -1)) in allowed_ids]

    source_candidate_count = len(results)
    if source_candidate_count == 0:
        raise HTTPException(status_code=400, detail="No source candidates remain after lineup filtering")

    warnings: List[str] = []
    result_by_id: Dict[int, Dict[str, object]] = {}
    candidates: List[PortfolioCandidate] = []
    dropped_missing_metric = 0
    for result in results:
        player_ids = [str(pid) for pid in result.get("player_ids", [])]
        total_own = compute_total_own(player_ids, ownership)
        result["player_ids"] = player_ids
        result["total_own"] = total_own
        candidate = _result_to_candidate(result, total_own=total_own)
        metric_value = get_candidate_metric_value(candidate, sort_key=request.sort_key)
        if metric_value is None:
            dropped_missing_metric += 1
            continue
        if request.filter_positive_ev:
            ev_value = float(result.get("expected_value", 0.0) or 0.0)
            if ev_value < 0.0:
                continue
        if request.max_total_own is not None and total_own > float(request.max_total_own):
            continue
        candidates.append(candidate)
        result_by_id[candidate.lineup_id] = result

    if dropped_missing_metric:
        warnings.append(
            f"Dropped {dropped_missing_metric} candidates with missing/non-finite {request.sort_key}"
        )
    if not candidates:
        raise HTTPException(status_code=400, detail="No candidates remain after portfolio filtering")

    ordered_candidates = _sort_portfolio_candidates(
        candidates,
        sort_key=request.sort_key,
        sort_dir=request.sort_dir,
    )
    if request.top_n is not None:
        ordered_candidates = ordered_candidates[: int(request.top_n)]
    filtered_candidate_count = len(ordered_candidates)

    if request.portfolio_size > filtered_candidate_count:
        raise HTTPException(
            status_code=400,
            detail=(
                f"portfolio_size={request.portfolio_size} exceeds filtered candidate count="
                f"{filtered_candidate_count}"
            ),
        )

    exposure_bounds = _build_exposure_bounds(request.exposure_bounds)
    candidate_ids_after_shortlist = {c.lineup_id for c in ordered_candidates}
    seed_lineup_ids: List[int] = []
    if request.seed_lineup_ids:
        seen_seed_ids: set[int] = set()
        for lineup_id_raw in request.seed_lineup_ids:
            lineup_id = int(lineup_id_raw)
            if lineup_id in seen_seed_ids:
                continue
            if lineup_id not in candidate_ids_after_shortlist:
                continue
            seen_seed_ids.add(lineup_id)
            seed_lineup_ids.append(lineup_id)

    diagnostics: Dict[str, object] = {
        "site": source_site,
        "mode": request.mode,
        "sort_key": request.sort_key,
        "sort_dir": request.sort_dir,
        "worlds_source": request.worlds_source,
        "candidate_count": int(source_candidate_count),
        "candidate_count_after_filters": int(len(candidates)),
        "candidate_count_after_shortlist": int(filtered_candidate_count),
        "portfolio_size": int(request.portfolio_size),
        "world_selection_policy": "not_used",
        "worlds_used": 0,
        "warnings": list(warnings),
        "source_build_id": request.source_build_id,
        "run_id": source_run_id,
        "seed_lineup_count_requested": int(len(request.seed_lineup_ids or [])),
        "seed_lineup_count_after_shortlist": int(len(seed_lineup_ids)),
    }

    try:
        if request.mode == "greedy_constraints":
            selection = build_portfolio(
                ordered_candidates,
                portfolio_size=int(request.portfolio_size),
                sort_key=request.sort_key,
                sort_dir=request.sort_dir,
                filter_positive_ev=False,
                max_total_own=None,
                min_uniques=int(request.min_uniques),
                exposure_bounds=exposure_bounds,
                seed_lineup_ids=seed_lineup_ids,
            )
            diagnostics.update(
                {
                    "ev_best": float(
                        sum(float(c.expected_value or 0.0) for c in selection.selected)
                    ),
                    "ev_target": float(
                        sum(float(c.expected_value or 0.0) for c in selection.selected)
                    ),
                    "ev_selected": float(
                        sum(float(c.expected_value or 0.0) for c in selection.selected)
                    ),
                    "passes": 0,
                    "swaps_made": 0,
                    "risk_var_total_baseline": None,
                    "risk_var_total_selected": None,
                    "risk_var_total_reduction_pct": None,
                }
            )
        else:
            if request.world_indices is None and request.worlds_train_frac is None:
                raise HTTPException(
                    status_code=400,
                    detail="decorrelated_ev requires world_indices or worlds_train_frac",
                )
            worlds_matrix, player_index = load_worlds_matrix(
                request.game_date,
                data_root=paths.data_path(),
                run_id=str(source_run_id) if source_run_id else None,
                worlds_source=request.worlds_source,
            )
            worlds_matrix = np.asarray(worlds_matrix, dtype=np.float64)
            train_idx, holdout_idx, world_policy = _resolve_train_holdout_indices(
                int(worlds_matrix.shape[0]),
                world_indices=request.world_indices,
                worlds_train_frac=request.worlds_train_frac,
                seed=int(request.seed),
            )

            baseline = build_portfolio(
                ordered_candidates,
                portfolio_size=int(request.portfolio_size),
                sort_key="expected_value",
                sort_dir="desc",
                filter_positive_ev=False,
                max_total_own=None,
                min_uniques=int(request.min_uniques),
                exposure_bounds=exposure_bounds,
                seed_lineup_ids=seed_lineup_ids,
            )
            selection, decor_diag = build_decorrelated_portfolio(
                ordered_candidates,
                portfolio_size=int(request.portfolio_size),
                worlds_matrix=worlds_matrix,
                player_index=player_index,
                world_indices=train_idx.tolist() if train_idx is not None else None,
                config=DecorrelatedPortfolioConfig(
                    ev_retention=float(request.ev_retention),
                    worlds_sample=int(request.worlds_sample),
                    seed=int(request.seed),
                ),
                exposure_bounds=exposure_bounds,
                min_uniques=int(request.min_uniques),
                seed_lineup_ids=seed_lineup_ids,
            )
            diagnostics.update(decor_diag.to_dict())
            diagnostics.update(
                {
                    "world_selection_policy": world_policy,
                    "worlds_run_id": source_run_id,
                    "train_worlds_count": (
                        int(train_idx.size)
                        if train_idx is not None
                        else int(worlds_matrix.shape[0])
                    ),
                    "holdout_worlds_count": int(holdout_idx.size) if holdout_idx is not None else 0,
                }
            )
            holdout_risk_baseline = _compute_portfolio_risk(
                baseline.selected,
                worlds_matrix=worlds_matrix,
                player_index=player_index,
                world_indices=holdout_idx,
            )
            holdout_risk_selected = _compute_portfolio_risk(
                selection.selected,
                worlds_matrix=worlds_matrix,
                player_index=player_index,
                world_indices=holdout_idx,
            )
            if holdout_risk_baseline is not None and holdout_risk_selected is not None:
                diagnostics.update(
                    {
                        "holdout_risk_var_total_baseline": float(holdout_risk_baseline),
                        "holdout_risk_var_total_selected": float(holdout_risk_selected),
                        "holdout_risk_var_total_reduction_pct": (
                            0.0
                            if holdout_risk_baseline <= 0.0
                            else float(
                                100.0
                                * (holdout_risk_baseline - holdout_risk_selected)
                                / holdout_risk_baseline
                            )
                        ),
                    }
                )
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Portfolio selection failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    portfolio_summary = summarize_portfolio(selection.selected).to_dict()
    diagnostics.update(portfolio_summary)
    if seed_lineup_ids:
        seed_lineup_set = set(seed_lineup_ids)
        selected_lineup_set = {c.lineup_id for c in selection.selected}
        retained_seed_count = len(seed_lineup_set.intersection(selected_lineup_set))
        diagnostics.update(
            {
                "seed_lineup_count_retained": int(retained_seed_count),
                "seed_lineup_count_replaced": int(len(seed_lineup_set) - retained_seed_count),
            }
        )

    selected_results = [result_by_id[c.lineup_id] for c in selection.selected]
    selected_lineups_raw = [list(r.get("player_ids", [])) for r in selected_results]
    if source_site == "fd":
        try:
            selected_lineups = _normalize_lineups_for_site(
                selected_lineups_raw,
                game_date=request.game_date,
                draft_group_id=int(draft_group_id) if draft_group_id is not None else None,
                site=source_site,
                run_id=str(source_run_id) if source_run_id else None,
                context="selected_lineups",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    else:
        selected_lineups = selected_lineups_raw

    return PortfolioSelectionResponse(
        site=source_site,
        mode=request.mode,
        source_build_id=request.source_build_id,
        candidate_count=source_candidate_count,
        filtered_candidate_count=filtered_candidate_count,
        selected_lineup_ids=[c.lineup_id for c in selection.selected],
        selected_results=[LineupEVResultResponse(**r) for r in selected_results],
        selected_lineups=selected_lineups,
        diagnostics=diagnostics,
        warnings=warnings,
    )


@router.delete("/saved-builds/{build_id}")
async def delete_saved_sim_build(build_id: str, date: str):
    """Delete a saved contest sim build."""
    deleted = _delete_sim_build(date, build_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Sim build {build_id} not found for date {date}")
    return {"status": "deleted", "build_id": build_id}


@router.get("/field-libraries", response_model=List[FieldLibrarySummaryResponse])
async def list_field_libraries(
    date: str,
    draft_group_id: int,
    site: str = Query(default="dk", description="Site: dk or fd"),
):
    """List cached field libraries for a slate."""
    site_norm = _normalize_site(site)
    paths = list_field_library_paths(date, int(draft_group_id))
    summaries: List[FieldLibrarySummaryResponse] = []
    for path in paths:
        try:
            library = load_field_library(path)
            lib_site = _normalize_site(str(library.meta.get("site") or "dk"))
            if lib_site != site_norm:
                continue
            version = path.stem.replace("field_library_", "")
            summaries.append(
                FieldLibrarySummaryResponse(
                    version=version,
                    site=lib_site,
                    path=str(path),
                    game_date=str(library.meta.get("game_date", date)),
                    draft_group_id=int(library.meta.get("draft_group_id", draft_group_id)),
                    method=str(library.meta.get("method", "")) or None,
                    generated_at=str(library.meta.get("generated_at", "")) or None,
                    selected_k=len(library.lineups),
                    weights_sum=int(sum(library.weights)),
                    meta={k: v for k, v in library.meta.items() if k not in {"lineups", "weights"}},
                )
            )
        except Exception as exc:
            logger.warning("Failed to load field library %s: %s", path, exc)
            continue
    return summaries


@router.post("/field-libraries/build", response_model=FieldLibrarySummaryResponse)
async def build_field_library(request: BuildFieldLibraryRequest):
    """Build (or rebuild) a cached field library for a slate."""
    try:
        site_norm = _normalize_site(request.site)
        ownership_mode = _normalize_ownership_mode(request.ownership_mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    use_field_ownership = ownership_mode in {"full", "field_only"}
    version = request.version
    if not use_field_ownership:
        version = "v0" if version == "latest" else version
        if not str(version).endswith("_noown"):
            version = f"{version}_noown"
    library, path, _built_now = load_or_build_field_library(
        game_date=request.game_date,
        draft_group_id=int(request.draft_group_id),
        site=site_norm,
        version=version,
        k=int(request.k),
        candidate_pool_size=int(request.candidate_pool_size),
        rebuild=bool(request.rebuild),
        rebuild_candidates=bool(request.rebuild_candidates),
        use_ownership_features=use_field_ownership,
    )
    version = Path(path).stem.replace("field_library_", "")
    return FieldLibrarySummaryResponse(
        version=version,
        site=site_norm,
        path=str(path),
        game_date=str(library.meta.get("game_date", request.game_date)),
        draft_group_id=int(library.meta.get("draft_group_id", request.draft_group_id)),
        method=str(library.meta.get("method", "")) or None,
        generated_at=str(library.meta.get("generated_at", "")) or None,
        selected_k=len(library.lineups),
        weights_sum=int(sum(library.weights)),
        meta=dict(library.meta),
    )
