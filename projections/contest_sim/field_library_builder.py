"""Build compressed weighted opponent fields for contest simulation."""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .field_library import FieldLibrary
from .weights import scale_integer_weights_to_target


@dataclass(frozen=True)
class PlayerInfo:
    salary: int
    own_proj: float
    team: str


def canonicalize_lineup(lineup: Sequence[object]) -> Tuple[str, ...]:
    """Canonicalize a lineup as a sorted tuple of player_id strings."""
    return tuple(sorted(str(p).strip() for p in lineup if str(p).strip()))


def build_player_info_map(player_pool: Sequence[Mapping[str, Any]]) -> Dict[str, PlayerInfo]:
    """Build player_id -> PlayerInfo mapping from optimizer player pool rows."""
    info: Dict[str, PlayerInfo] = {}
    for row in player_pool:
        pid = str(row.get("player_id", "")).strip()
        if not pid:
            continue
        salary = int(row.get("salary") or 0)
        own = float(row.get("own_proj") or 0.0)
        team = str(row.get("team") or "UNK")
        info[pid] = PlayerInfo(salary=salary, own_proj=own, team=team)
    return info


def renormalize_ownership(player_info: Dict[str, PlayerInfo], target_total: float = 800.0) -> Dict[str, PlayerInfo]:
    """Renormalize ownership projections to sum to ~target_total across the slate."""
    total = sum(p.own_proj for p in player_info.values() if math.isfinite(p.own_proj))
    if total <= 0 or not math.isfinite(total):
        return player_info
    scale = target_total / total
    if abs(scale - 1.0) < 1e-6:
        return player_info
    return {
        pid: PlayerInfo(salary=p.salary, own_proj=p.own_proj * scale, team=p.team)
        for pid, p in player_info.items()
    }


@dataclass(frozen=True)
class LineupFeatures:
    salary: int
    total_own: float
    num_under_5: int
    num_under_10: int
    max_team_stack: int

    def bucket_key(self) -> Tuple[int, int, int, int]:
        # Coarse bins to keep bucket counts manageable.
        own_bin = int(self.total_own / 40) * 40  # 0..800 in ~20 bins
        salary_bin = int(self.salary / 1000) * 1000  # coarse salary usage bins
        under_5_bin = min(self.num_under_5, 3)
        under_10_bin = min(self.num_under_10, 3)
        return own_bin, salary_bin, under_5_bin, under_10_bin


def compute_lineup_features(lineup: Tuple[str, ...], player_info: Dict[str, PlayerInfo]) -> LineupFeatures:
    salary = 0
    total_own = 0.0
    under_5 = 0
    under_10 = 0
    team_counts: Dict[str, int] = defaultdict(int)

    for pid in lineup:
        info = player_info.get(pid)
        if info is None:
            continue
        salary += int(info.salary)
        own = float(info.own_proj)
        total_own += own
        if own < 5:
            under_5 += 1
        if own < 10:
            under_10 += 1
        team_counts[info.team] += 1

    max_stack = max(team_counts.values()) if team_counts else 0
    return LineupFeatures(
        salary=salary,
        total_own=total_own,
        num_under_5=under_5,
        num_under_10=under_10,
        max_team_stack=max_stack,
    )


class SevenOfEightGuard:
    """Prevents selecting two lineups that share 7 of 8 players."""

    def __init__(self) -> None:
        self._seen_subsets: set[Tuple[str, ...]] = set()

    def allows(self, lineup: Tuple[str, ...]) -> bool:
        if len(lineup) < 8:
            return True
        for subset in itertools.combinations(lineup, 7):
            if subset in self._seen_subsets:
                return False
        return True

    def add(self, lineup: Tuple[str, ...]) -> None:
        if len(lineup) < 8:
            return
        for subset in itertools.combinations(lineup, 7):
            self._seen_subsets.add(subset)


def compress_lineup_counts(
    lineup_counts: Counter[Tuple[str, ...]],
    *,
    player_info: Optional[Dict[str, PlayerInfo]] = None,
    k: int = 2500,
) -> Tuple[List[Tuple[str, ...]], List[int], Dict[str, Any]]:
    """Compress a large lineup pool to K representatives and weights.

    Returns selected lineups, weights (summing to total candidate count), and debug info.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not lineup_counts:
        raise ValueError("lineup_counts must be non-empty")

    total_candidates = int(sum(lineup_counts.values()))
    unique_candidates = int(len(lineup_counts))

    if k >= unique_candidates:
        selected = list(lineup_counts.keys())
        weights = [int(lineup_counts[lu]) for lu in selected]
        return selected, weights, {
            "total_candidates": total_candidates,
            "unique_candidates": unique_candidates,
            "selected_k": unique_candidates,
            "compression": "none",
        }

    # If we don't have player metadata, fall back to top-K by observed frequency.
    if not player_info:
        selected = [lu for lu, _ in lineup_counts.most_common(k)]
        weights = [int(lineup_counts[lu]) for lu in selected]
        weights = scale_integer_weights_to_target(weights, total_candidates)
        return selected, weights, {
            "total_candidates": total_candidates,
            "unique_candidates": unique_candidates,
            "selected_k": k,
            "compression": "topk_by_frequency",
        }

    # Compute features + stratified buckets
    features: Dict[Tuple[str, ...], LineupFeatures] = {}
    buckets: Dict[Tuple[int, int, int, int], List[Tuple[Tuple[str, ...], int]]] = defaultdict(list)

    for lineup, count in lineup_counts.items():
        feat = compute_lineup_features(lineup, player_info)
        features[lineup] = feat
        buckets[feat.bucket_key()].append((lineup, int(count)))

    # Bucket weights used to allocate quotas; sqrt dampens dominance.
    bucket_keys = list(buckets.keys())
    bucket_scores = {bk: math.sqrt(sum(c for _, c in buckets[bk])) for bk in bucket_keys}
    total_score = sum(bucket_scores.values()) or 1.0

    # Initial quotas
    quotas: Dict[Tuple[int, int, int, int], int] = {}
    if len(bucket_keys) <= k:
        for bk in bucket_keys:
            quotas[bk] = max(1, int(round(k * (bucket_scores[bk] / total_score))))
    else:
        # Too many buckets: guarantee coverage for the top buckets by score.
        top_buckets = sorted(bucket_keys, key=lambda bk: bucket_scores[bk], reverse=True)[:k]
        quotas = {bk: 1 for bk in top_buckets}

    # Fix quota sum to exactly k
    current = sum(quotas.values())
    if current > k:
        # Reduce from smallest buckets first (keeping >= 1)
        for bk in sorted(quotas.keys(), key=lambda b: bucket_scores.get(b, 0.0)):
            if current <= k:
                break
            if quotas[bk] > 1:
                quotas[bk] -= 1
                current -= 1
    elif current < k:
        extra = k - current
        for bk in sorted(quotas.keys(), key=lambda b: bucket_scores.get(b, 0.0), reverse=True):
            if extra <= 0:
                break
            quotas[bk] += 1
            extra -= 1

    guard = SevenOfEightGuard()
    selected: List[Tuple[str, ...]] = []

    # Select within each bucket by frequency, with near-dup guard.
    for bk, quota in sorted(quotas.items(), key=lambda kv: bucket_scores.get(kv[0], 0.0), reverse=True):
        if quota <= 0:
            continue
        candidates = sorted(buckets[bk], key=lambda kv: kv[1], reverse=True)
        for lineup, _count in candidates:
            if len(selected) >= k:
                break
            if quota <= 0:
                break
            if not guard.allows(lineup):
                continue
            selected.append(lineup)
            guard.add(lineup)
            quota -= 1
            if quota <= 0:
                break

    # Fill any remaining slots from global frequency order.
    if len(selected) < k:
        selected_set = set(selected)
        for lineup, _count in lineup_counts.most_common():
            if len(selected) >= k:
                break
            if lineup in selected_set:
                continue
            if not guard.allows(lineup):
                continue
            selected.append(lineup)
            selected_set.add(lineup)
            guard.add(lineup)

    # Assign weights by bucket-mass redistribution where possible.
    # For each represented bucket, split bucket mass among selected lineups in that bucket.
    selected_by_bucket: Dict[Tuple[int, int, int, int], List[Tuple[str, ...]]] = defaultdict(list)
    for lineup in selected:
        selected_by_bucket[features[lineup].bucket_key()].append(lineup)

    assigned_weights: Dict[Tuple[str, ...], int] = {}
    assigned_total = 0
    for bk, members in selected_by_bucket.items():
        bucket_mass = int(sum(c for _, c in buckets[bk]))
        if bucket_mass <= 0:
            continue
        base = [int(lineup_counts[lu]) for lu in members]
        scaled = scale_integer_weights_to_target(base, bucket_mass)
        for lu, w in zip(members, scaled):
            assigned_weights[lu] = int(w)
        assigned_total += bucket_mass

    # Redistribute any unassigned mass (from unrepresented buckets) proportionally.
    remaining = total_candidates - assigned_total
    if remaining != 0:
        base = [assigned_weights.get(lu, int(lineup_counts[lu])) for lu in selected]
        scaled = scale_integer_weights_to_target(base, total_candidates)
        assigned_weights = {lu: int(w) for lu, w in zip(selected, scaled)}

    weights = [assigned_weights.get(lu, int(lineup_counts[lu])) for lu in selected]
    weights = scale_integer_weights_to_target(weights, total_candidates)

    return selected, weights, {
        "total_candidates": total_candidates,
        "unique_candidates": unique_candidates,
        "selected_k": int(len(selected)),
        "compression": "stratified_frequency_with_diversity_guard",
        "bucket_count": int(len(bucket_keys)),
    }


def build_field_library_from_candidates(
    candidates: Iterable[Sequence[object]],
    *,
    k: int = 2500,
    player_pool: Optional[Sequence[Mapping[str, Any]]] = None,
    method: str = "quickbuild_v0",
    params: Optional[Dict[str, Any]] = None,
) -> FieldLibrary:
    """Build a weighted, compressed opponent field library from candidate lineups."""
    lineup_counts: Counter[Tuple[str, ...]] = Counter()
    for lineup in candidates:
        canon = canonicalize_lineup(lineup)
        if not canon:
            continue
        lineup_counts[canon] += 1

    return build_field_library_from_lineup_counts(
        lineup_counts,
        k=k,
        player_pool=player_pool,
        method=method,
        params=params,
    )


def build_field_library_from_lineup_counts(
    lineup_counts: Counter[Tuple[str, ...]],
    *,
    k: int = 2500,
    player_pool: Optional[Sequence[Mapping[str, Any]]] = None,
    method: str = "quickbuild_v0",
    params: Optional[Dict[str, Any]] = None,
) -> FieldLibrary:
    """Build a weighted, compressed opponent field library from weighted lineup counts."""
    player_info = None
    if player_pool is not None:
        player_info = renormalize_ownership(build_player_info_map(player_pool))

    selected, weights, debug = compress_lineup_counts(lineup_counts, player_info=player_info, k=k)
    meta: Dict[str, Any] = {
        "method": method,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": params or {},
        "stats": debug,
    }
    return FieldLibrary(
        lineups=[list(lu) for lu in selected],
        weights=[int(w) for w in weights],
        meta=meta,
    )
