"""QuickBuild-backed generator for representative opponent field libraries."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

from projections.api.optimizer_service import build_player_pool
from projections.paths import data_path
from projections.optimizer.quick_build import QuickBuildConfig, quick_build_pool

from .field_library import FieldLibrary
from .field_library_builder import (
    build_field_library_from_candidates,
    build_field_library_from_lineup_counts,
    build_player_info_map,
    renormalize_ownership,
)
from .field_candidate_pool import (
    CandidatePool,
    candidate_pool_path,
    load_candidate_pool,
    merge_candidates,
    save_candidate_pool,
)
from .field_weight_model import default_field_weight_model_path, load_field_weight_model
from .weights import scale_float_weights_to_target

__all__ = ["build_quickbuild_field_library"]


def build_quickbuild_field_library(
    *,
    game_date: str,
    draft_group_id: int,
    version: str = "v0",
    site: str = "dk",
    k: int = 2500,
    candidate_pool_size: int = 40000,
    weight_sum_target: int = 100000,
    weight_model_path: Optional[Path] = None,
    weight_model_bucket: str = "5k-20k",
    use_candidate_cache: bool = True,
    rebuild_candidates: bool = False,
    # Match optimizer page defaults: 22 workers, 1 thread per worker.
    qb_builds: int = 22,
    qb_per_build: int = 3000,
    qb_timeout: float = 0.6,
    qb_threads: int = 1,
    qb_seed: Optional[int] = None,
    min_salary: int = 49000,
    max_salary: int = 50000,
    global_team_limit: int = 4,
    randomness_pct: Optional[float] = None,
) -> FieldLibrary:
    """Generate a compressed opponent field library by sampling many QuickBuild lineups."""
    player_pool = build_player_pool(
        game_date=game_date,
        draft_group_id=int(draft_group_id),
        site=site,
        data_root=data_path(),
        use_user_overrides=False,
        ownership_mode="renormalize",
    )

    qb_cfg = QuickBuildConfig(
        builds=int(qb_builds),
        per_build=int(qb_per_build),
        max_pool=max(1, int(candidate_pool_size)),
        timeout=float(qb_timeout),
        threads=int(qb_threads),
        seed=qb_seed,
        # Match optimizer page default (disabled); helps throughput for large pools.
        near_dup_jaccard=0.0,
        lineup_size=8 if site == "dk" else 9,
    )

    constraints: Dict[str, Any] = {
        "min_salary": int(min_salary),
        "max_salary": int(max_salary),
        "global_team_limit": int(global_team_limit),
        "team_limits": {},
        "lock_ids": [],
        "ban_ids": [],
        "unique_players": 1,
        "N_lineups": 1,
    }
    if randomness_pct is not None:
        constraints["randomness_pct"] = float(randomness_pct)

    candidates = []
    candidate_cache_info: Dict[str, Any] = {}
    qb_stats: Optional[dict] = None
    if use_candidate_cache:
        cache_path = candidate_pool_path(game_date, draft_group_id, version=version)
        cached: Optional[CandidatePool] = None
        if cache_path.exists() and not rebuild_candidates:
            cached = load_candidate_pool(cache_path)
        cached_n = int(len(cached.lineups)) if cached else 0

        built_now = False
        topped_up = False
        if rebuild_candidates or cached_n < int(candidate_pool_size):
            qb_runs = []
            candidates = list(cached.lineups) if (cached and not rebuild_candidates) else []
            topped_up = not rebuild_candidates and cached_n > 0
            for attempt in range(3):
                remaining = int(candidate_pool_size) - len(candidates)
                if remaining <= 0:
                    break
                needed = remaining if attempt == 0 else max(1, remaining)

                run_id = f"fieldcands_{game_date.replace('-', '')}_dg{draft_group_id}_{int(time.time())}_{attempt}"
                qb_cfg_needed = QuickBuildConfig(
                    builds=int(qb_builds),
                    per_build=int(qb_per_build),
                    max_pool=int(needed),
                    timeout=float(qb_timeout),
                    threads=int(qb_threads),
                    seed=qb_seed,
                    near_dup_jaccard=0.0,
                    lineup_size=8 if site == "dk" else 9,
                )
                qb_result = quick_build_pool(
                    slate=player_pool,
                    site=site,
                    constraints=constraints,
                    qb_cfg=qb_cfg_needed,
                    run_id=run_id,
                )
                built_now = True
                qb_stats = qb_result.stats.to_dict()
                qb_runs.append(
                    {
                        "attempt": attempt,
                        "requested": int(needed),
                        "accepted": int(qb_stats.get("accepted") or 0),
                        "wall_time_s": float(qb_stats.get("wall_time_s") or 0.0),
                    }
                )
                candidates = merge_candidates(candidates, qb_result.lineups, max_size=int(candidate_pool_size))

            save_candidate_pool(
                CandidatePool(
                    lineups=candidates,
                    meta={
                        "method": "quickbuild_candidates",
                        "game_date": game_date,
                        "draft_group_id": int(draft_group_id),
                        "version": version,
                        "site": site,
                        "target_candidates": int(candidate_pool_size),
                        "cached_before": cached_n if not rebuild_candidates else 0,
                        "cached_used": not rebuild_candidates and cached_n > 0,
                        "qb_cfg": qb_cfg.to_dict(),
                        "qb_stats": qb_stats,
                        "qb_runs": qb_runs,
                    },
                ),
                cache_path,
            )
        else:
            candidates = list(cached.lineups) if cached else []
            if cached and qb_stats is None:
                qb_stats = cached.meta.get("qb_stats") if isinstance(cached.meta, dict) else None

        candidate_cache_info = {
            "candidate_pool_path": str(cache_path),
            "candidate_pool_cached_n": int(len(candidates)),
            "candidate_pool_built_now": bool(built_now),
            "candidate_pool_topped_up": bool(topped_up),
        }
    else:
        run_id = f"fieldlib_{game_date.replace('-', '')}_dg{draft_group_id}_{int(time.time())}"
        qb_result = quick_build_pool(
            slate=player_pool,
            site=site,
            constraints=constraints,
            qb_cfg=qb_cfg,
            run_id=run_id,
        )
        qb_stats = qb_result.stats.to_dict()
        candidates = qb_result.lineups

    # Optional calibrated weights (trained from bronze contest results).
    lineup_counts: Optional[Counter[tuple[str, ...]]] = None
    try:
        model_path = weight_model_path or default_field_weight_model_path(version="v1")
        if model_path.exists():
            player_info = renormalize_ownership(build_player_info_map(player_pool))
            total_own = sum(p.own_proj for p in player_info.values())
            if total_own > 1.0:
                model = load_field_weight_model(model_path, version="v1")
                bucket = model.get_bucket(weight_model_bucket)
                features = []
                canon_lineups = []
                for lu in candidates:
                    canon = tuple(sorted(str(p) for p in lu))
                    canon_lineups.append(canon)
                    owns = [player_info.get(pid).own_proj if pid in player_info else 0.0 for pid in canon]
                    features.append(
                        {
                            "sum_own": float(sum(owns)),
                            "num_under_5": int(sum(1 for o in owns if o < 5)),
                            "num_under_10": int(sum(1 for o in owns if o < 10)),
                            "num_over_50": int(sum(1 for o in owns if o > 50)),
                        }
                    )

                raw_weights = [bucket.predict_weight(f) for f in features]
                target = max(int(weight_sum_target), len(raw_weights))
                scaled = scale_float_weights_to_target(raw_weights, target, min_weight=1)
                lineup_counts = Counter({lu: int(w) for lu, w in zip(canon_lineups, scaled, strict=True)})
    except Exception:
        lineup_counts = None

    params = {
        "site": site,
        "k": int(k),
        "candidate_pool_size": int(candidate_pool_size),
        "qb_cfg": qb_cfg.to_dict(),
        "constraints": constraints,
        "qb_stats": qb_stats,
        **candidate_cache_info,
    }
    if lineup_counts is not None and sum(lineup_counts.values()) > 0:
        params.update(
            {
                "weight_model_path": str(weight_model_path or default_field_weight_model_path(version="v1")),
                "weight_model_bucket": weight_model_bucket,
                "weight_sum_target": max(int(weight_sum_target), len(candidates)),
            }
        )
        return build_field_library_from_lineup_counts(
            lineup_counts,
            k=k,
            player_pool=player_pool,
            method="quickbuild_v1_calibrated",
            params=params,
        )

    return build_field_library_from_candidates(
        candidates,
        k=k,
        player_pool=player_pool,
        method="quickbuild_v0",
        params=params,
    )
