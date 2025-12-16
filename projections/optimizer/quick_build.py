"""QuickBuild pool generation scaffolding.

This module introduces the configuration objects, hashing helpers, and light
bookkeeping required by the QuickBuild spec (see ``lineup-selector.md``).
Actual multiprocessing workers and solver callbacks will land in a follow-up
patch, but the contract defined here is stable enough for the CLI and tests to
target today.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict, replace
from multiprocessing import Event, Process, Queue
import multiprocessing as mp
import multiprocessing as mp
from pathlib import Path
from queue import Empty, Full
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

from ortools.sat.python import cp_model

try:  # Local import path
    from .model_spec import Spec, SpecPlayer
except Exception:  # pragma: no cover - fallback for tests
    from model_spec import Spec, SpecPlayer  # type: ignore

try:
    from .cpsat_solver import build_cpsat_model, build_objective_weights
except Exception:  # pragma: no cover
    from cpsat_solver import build_cpsat_model, build_objective_weights  # type: ignore

FNV_OFFSET_BASIS = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3

DEFAULT_ENUM_TARGETS: Tuple[float, ...] = (
    0.99,
    0.985,
    0.98,
    0.975,
    0.97,
    0.965,
    0.96,
    0.955,
    0.95,
    0.94,
    0.93,
    0.92,
    0.91,
    0.90,
    0.89,
    0.88,
    0.87,
    0.86,
    0.85,
    0.83,
    0.80,
    0.78,
    0.75,
)


def timestamp_run_id(prefix: str = "qb") -> str:
    """Return a filesystem-friendly run identifier."""

    now = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{now}"


def fnv1a_64(parts: Sequence[str]) -> int:
    """FNV-1a hash over an ordered set of strings."""

    h = FNV_OFFSET_BASIS
    for part in parts:
        encoded = part.encode("utf-8")
        for byte in encoded:
            h ^= byte
            h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


class CrossWorkerBloom:
    """Process-safe Bloom filter to reduce cross-worker duplicates.

    Uses a shared QWORD array for bits and a Lock for atomic test-and-set.
    """

    def __init__(self, bits: mp.Array, m_bits: int, k_hashes: int, lock: mp.Lock):
        self._bits = bits  # type: ignore[assignment]
        self._m = int(m_bits)
        self._k = int(max(1, k_hashes))
        self._lock = lock

    @staticmethod
    def create(capacity: int, fpp: float = 0.01) -> "CrossWorkerBloom":
        n = max(1, int(capacity))
        p = min(max(fpp, 1e-6), 0.5)
        # m = - (n * ln p) / (ln 2)^2
        m_float = - (n * math.log(p)) / (math.log(2) ** 2)
        m_bits = int(max(1024, int(m_float)))
        # k = (m / n) * ln 2
        k_hashes = max(1, int(round((m_bits / n) * math.log(2))))
        # Allocate as 64-bit word array
        n_words = (m_bits + 63) // 64
        bits = mp.Array('Q', n_words, lock=False)
        lock = mp.Lock()
        return CrossWorkerBloom(bits, m_bits, k_hashes, lock)

    def _indices(self, h: int) -> List[int]:
        # Double hashing for Bloom indices
        mask64 = (1 << 64) - 1
        h1 = (h ^ 0x9E3779B97F4A7C15) & mask64
        h2 = ((h ^ 0xD2B74407B1CE6E93) * 0xFF51AFD7ED558CCD) & mask64
        if h2 == 0:
            h2 = 0x9E3779B97F4A7C15
        idxs: List[int] = []
        m = self._m
        a = h1 % m
        b = h2 % m
        if b == 0:
            b = 1
        for i in range(self._k):
            idxs.append((a + i * b) % m)
        return idxs

    def test_and_set(self, h: int) -> bool:
        """Return True if this was a new insertion; False if already present.

        Fast path: lockless read to check if all bits are set.
        Slow path: take lock, recheck, then set missing bits.
        """
        idxs = self._indices(h)
        word_masks: List[Tuple[int, int]] = []
        all_set = True
        for bit in idxs:
            wi = bit >> 6
            mask = 1 << (bit & 63)
            word_masks.append((wi, mask))
            # Lockless read: if any bit is 0, we will need to set under lock
            if (self._bits[wi] & mask) == 0:
                all_set = False
        if all_set:
            return False
        with self._lock:
            # Recheck under lock in case of concurrent updates
            all_set2 = True
            for wi, mask in word_masks:
                if (self._bits[wi] & mask) == 0:
                    all_set2 = False
                    break
            if all_set2:
                return False
            for wi, mask in word_masks:
                self._bits[wi] = self._bits[wi] | mask
            return True


def _make_emit_fn(
    queue: Queue,
    stop_event: Event,
    seen_hashes: set[int],
    cross_bloom: "CrossWorkerBloom | None" = None,
    *,
    use_central_dedup: bool = False,
) -> Callable[[Tuple[str, ...], Sequence[int]], bool]:
    def emit(lineup: Tuple[str, ...], _idx: Sequence[int]) -> bool:
        if stop_event.is_set():
            return False
        # Normalize to set identity to match pool-level dedup
        key = tuple(sorted(lineup))
        h = fnv1a_64(key)
        # Per-worker seen set first (cheap, no inter-process contention)
        if h in seen_hashes:
            return False
        seen_hashes.add(h)
        # Cross-worker Bloom prefilter unless centralized collector is enabled
        if (not use_central_dedup) and cross_bloom is not None:
            inserted = cross_bloom.test_and_set(h)
            if not inserted:
                return False
        while not stop_event.is_set():
            try:
                queue.put(key, timeout=0.1)
                return True
            except Full:
                continue
        return False

    return emit


class ShardedBloom:
    """A Bloom filter composed of multiple shards, each with its own bitset and lock.

    Sharding reduces lock contention across many workers; a hash selects the shard.
    """

    def __init__(self, shards: List[CrossWorkerBloom]):
        if not shards:
            raise ValueError("ShardedBloom requires at least one shard")
        self._shards = shards
        self._n = len(shards)

    @staticmethod
    def create(capacity: int, fpp: float, shards: int) -> "ShardedBloom":
        s = max(1, int(shards))
        cap_per = max(1, int(math.ceil(capacity / s)))
        shard_list = [CrossWorkerBloom.create(cap_per, fpp) for _ in range(s)]
        return ShardedBloom(shard_list)

    def _pick(self, h: int) -> CrossWorkerBloom:
        # Use high bits to spread across shards; fall back to mod
        idx = ((h >> 32) ^ h) % self._n
        return self._shards[int(idx)]

    def test_and_set(self, h: int) -> bool:
        return self._pick(h).test_and_set(h)


@dataclass(frozen=True)
class QuickBuildPaths:
    """Resolved filesystem paths for a QuickBuild run."""

    run_id: str
    run_dir: Path
    pool_csv: Path
    stats_json: Path
    spill_dir: Path

    @staticmethod
    def resolve(run_id: Optional[str], base_dir: Path | str = ".") -> "QuickBuildPaths":
        rid = run_id or timestamp_run_id()
        base = Path(base_dir).expanduser().resolve()
        run_dir = base / "runs" / rid
        pool_csv = run_dir / "pool.csv"
        stats_json = run_dir / "qb_stats.json"
        spill_dir = run_dir / "spill"
        return QuickBuildPaths(rid, run_dir, pool_csv, stats_json, spill_dir)

    def ensure_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.spill_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class QuickBuildStats:
    accepted: int = 0
    duplicates: int = 0
    near_duplicates: int = 0
    dropped_low_quality: int = 0
    emitted: int = 0
    wall_time_s: float = 0.0
    # Centralized dedup telemetry (optional)
    bloom_rejects: int = 0
    collector_seen_dups: int = 0
    collector_forwarded: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QuickBuildConfig:
    """CLI-friendly configuration for QuickBuild workers and storage."""

    builds: int = 0
    per_build: int = 1200
    threads: int = 1
    # 0.0 means "unset" and defers to cp_sat_params.max_time_seconds
    timeout: float = 0.0
    min_uniq: int = 1
    nogood_rate: int = 20
    jitter: float = 5e-4
    seed: Optional[int] = None
    max_pool: int = 200_000
    max_pool_ram: int = 0
    max_pool_disk: int = 0
    chunk_size: int = 25_000
    store: str = "parquet"
    approx_uniq: bool = True
    bloom_fpp: float = 0.01
    bloom_shards: int = 8
    near_dup_jaccard: float = 0.75
    quality_quantile: float = 0.50
    archetype_key: Tuple[str, ...] = ("stack", "salbin")
    merge_topk_per_arch: int = 2000
    queue_size: int = 50_000
    lineup_size: int = 8
    enum_enable: bool = True
    enum_targets: Tuple[float, ...] = DEFAULT_ENUM_TARGETS
    enum_k: int = 0
    enum_time: Optional[float] = 20.0
    enum_warm_time: Optional[float] = 5.0
    output_path: Optional[Path] = None
    stats_path: Optional[Path] = None
    spill_dir: Optional[Path] = None
    run_id: Optional[str] = None
    _nogood_cut_fn: Optional[Callable[[Sequence[int]], None]] = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def from_namespace(
        cls,
        args,
        *,
        lineup_size: int,
    ) -> "QuickBuildConfig":
        """Build from argparse namespace without importing argparse here."""

        archetype_key = tuple(
            part.strip() for part in str(getattr(args, "qb_archetype_key", "stack,salbin") or "").split(",") if part.strip()
        ) or ("stack", "salbin")
        enum_targets_raw = str(getattr(args, "qb_enum_targets", "") or "").strip()
        enum_targets = tuple(
            float(part)
            for part in enum_targets_raw.split(",")
            if part.strip()
        )
        if not enum_targets:
            enum_targets = DEFAULT_ENUM_TARGETS

        return cls(
            builds=int(getattr(args, "qb_builds", 0) or 0),
            per_build=int(getattr(args, "qb_per_build", 1200) or 1200),
            threads=int(getattr(args, "qb_threads", 1) or 1),
            timeout=float(getattr(args, "qb_timeout", 0) or 0),
            min_uniq=int(getattr(args, "qb_min_uniq", 1) or 1),
            nogood_rate=int(getattr(args, "qb_nogood_rate", 20) or 20),
            jitter=float(getattr(args, "qb_jitter", 5e-4) or 0.0),
            seed=getattr(args, "qb_seed", None),
            max_pool=int(getattr(args, "qb_max_pool", 20000) or 20000),
            max_pool_ram=int(getattr(args, "qb_max_pool_ram", 250000) or 0),
            max_pool_disk=int(getattr(args, "qb_max_pool_disk", 2000000) or 0),
            chunk_size=int(getattr(args, "qb_chunk_size", 25000) or 25000),
            store=str(getattr(args, "qb_store", "parquet") or "parquet"),
            approx_uniq=bool(getattr(args, "qb_approx_uniq", True)),
            bloom_fpp=float(getattr(args, "qb_bloom_fpp", 0.01) or 0.01),
            bloom_shards=int(getattr(args, "qb_bloom_shards", 8) or 8),
            near_dup_jaccard=float(getattr(args, "qb_near_dup_jaccard", 0.75) or 0.75),
            quality_quantile=float(getattr(args, "qb_quality_quantile", 0.5) or 0.5),
            archetype_key=archetype_key,
            merge_topk_per_arch=int(getattr(args, "qb_merge_topk_per_arch", 2000) or 2000),
            queue_size=int(getattr(args, "qb_queue_size", 10000) or 10000),
            lineup_size=lineup_size,
            enum_enable=bool(getattr(args, "qb_enum", False)),
            enum_targets=enum_targets,
            enum_k=int(getattr(args, "qb_enum_k", 1500) or 1500),
            enum_time=(
                float(getattr(args, "qb_enum_time", 10.0))
                if getattr(args, "qb_enum_time", None) not in (None, "")
                else 10.0
            ),
            enum_warm_time=(
                float(getattr(args, "qb_enum_warm_time", 5.0))
                if getattr(args, "qb_enum_warm_time", None) not in (None, "")
                else 5.0
            ),
            output_path=Path(getattr(args, "qb_out", "")).expanduser().resolve() if getattr(args, "qb_out", None) else None,
            stats_path=Path(getattr(args, "qb_stats", "")).expanduser().resolve() if getattr(args, "qb_stats", None) else None,
            spill_dir=Path(getattr(args, "qb_spill_dir", "")).expanduser().resolve() if getattr(args, "qb_spill_dir", None) else None,
            run_id=getattr(args, "qb_run_id", None),
        )

    def resolved(self, base_dir: Path | str = ".") -> "QuickBuildConfig":
        """Return a config with concrete filesystem paths set."""

        paths = QuickBuildPaths.resolve(self.run_id, base_dir)
        paths.ensure_dirs()
        return replace(
            self,
            run_id=paths.run_id,
            output_path=self.output_path or paths.pool_csv,
            stats_path=self.stats_path or paths.stats_json,
            spill_dir=self.spill_dir or paths.spill_dir,
        )

    def with_cut_fn(self, fn: Optional[Callable[[Sequence[int]], None]]) -> "QuickBuildConfig":
        return replace(self, _nogood_cut_fn=fn)

    def for_enumeration(self) -> "QuickBuildConfig":
        return replace(self, per_build=0, nogood_rate=0, _nogood_cut_fn=None)

    @property
    def nogood_cut_fn(self) -> Optional[Callable[[Sequence[int]], None]]:
        return self._nogood_cut_fn

    @property
    def effective_max_pool(self) -> int:
        limits = [v for v in (self.max_pool, self.max_pool_ram, self.max_pool_disk) if v]
        return min(limits) if limits else self.max_pool

    @property
    def worker_count(self) -> int:
        if self.builds > 0:
            return self.builds
        cpu = os.cpu_count() or 1
        return min(16, cpu)

    def worker_seed(self, offset: int) -> int:
        base = self.seed if self.seed is not None else int(time.time())
        return base + offset

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("_nogood_cut_fn", None)
        if self.output_path is not None:
            data["output_path"] = str(self.output_path)
        if self.stats_path is not None:
            data["stats_path"] = str(self.stats_path)
        if self.spill_dir is not None:
            data["spill_dir"] = str(self.spill_dir)
        return data


class InMemoryPool:
    """Simple deduping pool used before spill-to-disk wiring lands."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._rows: List[Tuple[Tuple[str, ...], int]] = []
        self._seen: set[int] = set()

    def __len__(self) -> int:
        return len(self._rows)

    def try_add(self, lineup: Sequence[str]) -> str:
        key = tuple(sorted(lineup))
        h = fnv1a_64(key)
        if h in self._seen:
            return "dup"
        if len(self._rows) >= self.max_size:
            return "full"
        self._seen.add(h)
        self._rows.append((key, h))
        return "ok"

    def try_add_with_jacc(self, lineup: Sequence[str], threshold: float, window: int = 2000) -> str:
        """Add lineup if it's not an exact dup and not a near-dup by Jaccard.

        threshold: Jaccard similarity in [0,1]. If <= 0, behaves like try_add.
        window: compare against at most this many most-recent accepted lineups.
        """
        if threshold is None or threshold <= 0.0:
            return self.try_add(lineup)
        key = tuple(sorted(lineup))
        h = fnv1a_64(key)
        if h in self._seen:
            return "dup"
        # Near-dup check on a bounded recent window to keep cost small
        a = set(key)
        limit = max(0, len(self._rows) - int(window))
        for idx in range(len(self._rows) - 1, limit - 1, -1):
            b_key, _ = self._rows[idx]
            b = set(b_key)
            inter = len(a & b)
            if inter == 0:
                continue
            union_sz = len(a | b)
            if union_sz == 0:
                continue
            j = inter / union_sz
            if j >= threshold:
                return "near_dup"
        if len(self._rows) >= self.max_size:
            return "full"
        self._seen.add(h)
        self._rows.append((key, h))
        return "ok"

    def rows(self) -> List[Tuple[Tuple[str, ...], int]]:
        return list(self._rows)


def dump_stats(stats: QuickBuildStats, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats.to_dict(), indent=2))


@dataclass
class QuickBuildResult:
    lineups: List[Tuple[str, ...]]
    stats: QuickBuildStats
    config: QuickBuildConfig


def _get_value(src: Any, key: str, default: Any = None) -> Any:
    if hasattr(src, key):
        return getattr(src, key)
    if isinstance(src, dict):
        return src.get(key, default)
    return default


def _ownership_penalty_to_dict(settings: Any) -> Optional[dict]:
    if settings is None:
        return None
    if isinstance(settings, dict):
        return settings
    try:
        return {
            "enabled": bool(settings.enabled),
            "mode": getattr(settings, "mode", "by_points"),
            "weight_lambda": getattr(settings, "weight_lambda", 0.0),
            "curve_type": getattr(settings, "curve_type", "sigmoid"),
            "power_k": getattr(settings, "power_k", 1.5),
            "pivot_p0": getattr(settings, "pivot_p0", 0.20),
            "curve_alpha": getattr(settings, "curve_alpha", 2.0),
            "clamp_min": getattr(settings, "clamp_min", 0.01),
            "clamp_max": getattr(settings, "clamp_max", 0.80),
            "shrink_gamma": getattr(settings, "shrink_gamma", 1.0),
        }
    except Exception:
        return None


def _build_spec_from_payload(
    slate: Any,
    site: str,
    constraints: Any,
) -> Spec:
    if isinstance(slate, Spec):
        return slate

    site = (site or "dk").lower()
    roster = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"] if site == "dk" else [
        "PG",
        "SG",
        "SF",
        "PF",
        "C",
    ]

    def _player_field(player: Any, key: str, default: Any = None) -> Any:
        if isinstance(player, dict):
            return player.get(key, default)
        return getattr(player, key, default)

    players: List[SpecPlayer] = []
    for raw in slate:
        pid = _player_field(raw, "player_id")
        if pid is None:
            raise ValueError("Each player must provide a player_id for QuickBuild.")
        positions = list(_player_field(raw, "positions", []))
        if not positions:
            raise ValueError(f"Player {pid} missing positions for site {site}.")
        players.append(
            SpecPlayer(
                player_id=str(pid),
                name=_player_field(raw, "name", str(pid)),
                team=_player_field(raw, "team", "UNK"),
                positions=positions,
                salary=int(_player_field(raw, "salary", 0)),
                proj=float(_player_field(raw, "proj", 0.0)),
                dk_id=_player_field(raw, "dk_id", None),
                own_proj=_player_field(raw, "own_proj", None),
                stddev=_player_field(raw, "stddev", None),
            )
        )

    max_salary_default = 50000 if site == "dk" else 60000
    max_salary = _get_value(constraints, "max_salary", max_salary_default) or max_salary_default
    min_salary = _get_value(constraints, "min_salary", None)
    team_max = _get_value(constraints, "global_team_limit", None)
    if team_max in (None, 0) and site == "dk":
        team_max = 4

    ownership_penalty = _ownership_penalty_to_dict(_get_value(constraints, "ownership_penalty", None))

    return Spec(
        site=site,  # type: ignore[arg-type]
        roster_slots=roster,
        salary_cap=int(max_salary),
        min_salary=int(min_salary) if min_salary is not None else None,
        players=players,
        team_max=int(team_max) if team_max is not None else None,
        team_limits=_get_value(constraints, "team_limits", {}) or {},
        lock_ids=list(_get_value(constraints, "lock_ids", []) or []),
        ban_ids=list(_get_value(constraints, "ban_ids", []) or []),
        lineup_size=8 if site == "dk" else 9,
        N_lineups=int(_get_value(constraints, "N_lineups", 1) or 1),
        unique_players=int(_get_value(constraints, "unique_players", 1) or 1),
        cp_sat_params=_get_value(constraints, "cp_sat_params", {}) or {},
        engine="cp_sat",
        ownership_penalty=ownership_penalty,
        min_proj_sum=_get_value(constraints, "min_proj_sum", None),
        max_proj_sum=_get_value(constraints, "max_proj_sum", None),
        randomness_pct=_get_value(constraints, "randomness_pct", None),
    )


class PoolingCallback(cp_model.CpSolverSolutionCallback):
    def __init__(
        self,
        player_vars: Sequence[Any],
        player_ids: Sequence[str],
        cfg: QuickBuildConfig,
        emit_fn: Callable[[Tuple[str, ...], Sequence[int]], bool],
        stop_event: Event,
        target_emissions: Optional[int],
    ) -> None:
        super().__init__()
        self.player_vars = player_vars
        self.player_ids = player_ids
        self.cfg = cfg
        self.emit = emit_fn
        self.stop_event = stop_event
        self.target = target_emissions if target_emissions and target_emissions > 0 else None
        self.last_set: Optional[set[str]] = None
        self.emitted = 0
        self.restart_requested = False

    def on_solution_callback(self) -> None:  # pragma: no cover - exercised in integration tests
        if self.stop_event.is_set():
            self.StopSearch()
            return

        idx_sel = [i for i, var in enumerate(self.player_vars) if self.BooleanValue(var)]
        lineup = tuple(sorted(self.player_ids[i] for i in idx_sel))
        lineup_set = set(lineup)

        if self.cfg.min_uniq > 0 and self.last_set is not None:
            overlap = len(self.last_set & lineup_set)
            changes = len(lineup_set) - overlap
            if changes < self.cfg.min_uniq:
                return

        if not self.emit(lineup, idx_sel):
            # Prefilter rejection: defer a no-good cut and continue searching.
            if self.cfg.nogood_cut_fn is not None and idx_sel:
                if not hasattr(self, "_deferred_ng"):
                    self._deferred_ng = []  # type: ignore[attr-defined]
                    self._since_last_apply = 0  # type: ignore[attr-defined]
                self._deferred_ng.append(idx_sel)  # type: ignore[attr-defined]
                self._since_last_apply = getattr(self, "_since_last_apply", 0) + 1  # type: ignore[attr-defined]
                # Apply after a small batch of rejections to pivot the search
                if self._since_last_apply >= 3:
                    for ng in self._deferred_ng[:32]:
                        self.cfg.nogood_cut_fn(ng)
                    self._deferred_ng.clear()
                    self._since_last_apply = 0
                    self.restart_requested = True
                    self.StopSearch()
            return

        self.emitted += 1
        self.last_set = lineup_set

        # Sparse no-good cuts based on accepted count; also apply deferred rejects
        if self.cfg.nogood_rate > 0 and self.cfg.nogood_cut_fn is not None and idx_sel:
            if (self.emitted % self.cfg.nogood_rate) == 0:
                self.cfg.nogood_cut_fn(idx_sel)
                if hasattr(self, "_deferred_ng") and self._deferred_ng:
                    for ng in self._deferred_ng[:32]:
                        self.cfg.nogood_cut_fn(ng)
                    self._deferred_ng.clear()
                    self._since_last_apply = 0
                self.restart_requested = True
                self.StopSearch()
                return

        if self.target is not None and self.emitted >= self.target:
            self.StopSearch()


def _configure_solver(solver: cp_model.CpSolver, spec: Spec, cfg: QuickBuildConfig, seed: int) -> None:
    params = dict(spec.cp_sat_params or {})
    time_limit = float(cfg.timeout or params.get("max_time_seconds", 0) or 0)
    # Prefer short default solves for streaming if unset in both cfg and params
    if time_limit <= 0:
        time_limit = 0.6
    solver.parameters.max_time_in_seconds = time_limit

    # Keep modest parallelism; honor cfg.threads if provided
    try:
        threads = int(cfg.threads)
    except Exception:
        threads = 0
    solver.parameters.num_search_workers = max(0, threads)

    # Diversity and restart behavior for better sampling without excessive restarts
    solver.parameters.random_seed = int(seed)
    solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
    try:
        # randomize_search may not exist on very old versions; guard access
        solver.parameters.randomize_search = True
    except Exception:
        pass
    try:
        solver.parameters.use_phase_saving = False
    except Exception:
        pass
    try:
        solver.parameters.restart_algorithm = cp_model.LUBY_RESTART
    except Exception:
        pass

    solver.parameters.log_search_progress = bool(params.get("log_search_progress", False))

    # Optional tolerances/limits
    if "relative_gap_limit" in params:
        try:
            solver.parameters.relative_gap_limit = float(params["relative_gap_limit"])
        except Exception:
            pass
    # Set deterministic time; default to ~10x wall time if not provided
    if "max_deterministic_time" in params:
        try:
            solver.parameters.max_deterministic_time = float(params["max_deterministic_time"])
        except Exception:
            pass
    else:
        try:
            solver.parameters.max_deterministic_time = float(time_limit) * 10.0
        except Exception:
            pass


def _solve_best_objective(spec: Spec, cfg: QuickBuildConfig, seed: int) -> Optional[int]:
    try:
        # Use raw points (no jitter) for estimating OPT so enum floors are stable
        artifacts = build_cpsat_model(spec, jitter=0.0, seed=seed)
    except Exception:
        return None

    solver = cp_model.CpSolver()
    _configure_solver(solver, spec, cfg, seed)
    warm = cfg.enum_warm_time if cfg.enum_warm_time is not None else min(cfg.timeout or 60.0, 5.0)
    if warm and warm > 0:
        solver.parameters.max_time_in_seconds = float(warm)
    status = solver.Solve(artifacts.model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    try:
        return int(round(solver.ObjectiveValue()))
    except Exception:
        return None


def _run_enum_phase(
    spec: Spec,
    cfg: QuickBuildConfig,
    queue: Queue,
    stop_event: Event,
    seed: int,
    seen_hashes: set[int],
    logger: logging.Logger,
    cross_bloom: "CrossWorkerBloom | None" = None,
) -> None:
    if not cfg.enum_enable or not cfg.enum_targets:
        return
    best = _solve_best_objective(spec, cfg, seed)
    if best is None or best <= 0:
        logger.info("[quick-build] worker seed=%s: enum skipped (best=%s)", seed, best)
        try:
            print(f"[quick-build] worker seed={seed}: enum skipped (best={best})")
        except Exception:
            pass
        return

    targets = sorted(set(cfg.enum_targets), reverse=True)
    try:
        print(
            f"[quick-build] worker seed={seed}: enum start warm_best={best} targets={len(targets)}"
        )
    except Exception:
        pass
    emit_fn = _make_emit_fn(queue, stop_event, seen_hashes, cross_bloom)
    enum_cfg = cfg.for_enumeration()
    for tau in targets:
        if stop_event.is_set():
            break
        floor_val = int(math.floor(best * tau))
        if floor_val <= 0:
            continue
        try:
            artifacts = build_cpsat_model(
                spec,
                jitter=cfg.jitter,
                seed=seed,
                optimize=False,
                randomness_pct=float(getattr(spec, "randomness_pct", 0.0) or 0.0),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("[quick-build] worker seed=%s: enum build failed (%s)", seed, exc)
            break
        # Use raw points (no jitter) for the floor expression to enumerate by pure FPts
        weights, scale = build_objective_weights(spec, jitter=0.0, seed=seed)
        if not weights:
            logger.info("[quick-build] worker seed=%s: enum missing weights", seed)
            break
        obj_expr = cp_model.LinearExpr.Sum([
            int(weights.get(pid, 0)) * var
            for pid, var in zip(artifacts.player_ids, artifacts.player_vars)
        ])
        # Apply optional projection sum bounds in enumeration as well
        min_ps = getattr(spec, "min_proj_sum", None)
        max_ps = getattr(spec, "max_proj_sum", None)
        if min_ps is not None:
            try:
                floor_pts = int(math.floor(float(min_ps) * float(scale)))
                artifacts.model.Add(obj_expr >= floor_pts)
            except Exception:
                pass
        if max_ps is not None:
            try:
                ceil_pts = int(math.ceil(float(max_ps) * float(scale)))
                artifacts.model.Add(obj_expr <= ceil_pts)
            except Exception:
                pass
        artifacts.model.Add(obj_expr >= floor_val)
        enum_solver = cp_model.CpSolver()
        _configure_solver(enum_solver, spec, cfg, seed)
        enum_solver.parameters.enumerate_all_solutions = True
        if cfg.enum_time and cfg.enum_time > 0:
            enum_solver.parameters.max_time_in_seconds = float(cfg.enum_time)
        target_emissions = cfg.enum_k if cfg.enum_k and cfg.enum_k > 0 else None
        cb = PoolingCallback(
            player_vars=artifacts.player_vars,
            player_ids=artifacts.player_ids,
            cfg=enum_cfg,
            emit_fn=emit_fn,
            stop_event=stop_event,
            target_emissions=target_emissions,
        )
        enum_solver.SolveWithSolutionCallback(artifacts.model, cb)
        logger.info(
            "[quick-build] worker seed=%s: enum tau=%.3f floor=%s emitted=%d",
            seed,
            tau,
            floor_val,
            cb.emitted,
        )
        try:
            print(
                f"[quick-build] worker seed={seed}: enum tau={tau:.3f} floor={floor_val} emitted={cb.emitted}"
            )
        except Exception:
            pass
        if cb.emitted == 0:
            continue


def _worker_main(
    spec: Spec,
    cfg: QuickBuildConfig,
    queue: Queue,
    stop_event: Event,
    seed: int,
    cross_bloom: "CrossWorkerBloom | None" = None,
    *,
    use_central_dedup: bool = False,
) -> None:  # pragma: no cover - exercised via integration
    logger = logging.getLogger("optimizer.quick_build.worker")
    try:
        artifacts = build_cpsat_model(
            spec,
            jitter=cfg.jitter,
            seed=seed,
            randomness_pct=float(getattr(spec, "randomness_pct", 0.0) or 0.0),
        )
    except Exception as exc:  # Fail-fast on model issues
        logger.exception("QuickBuild worker failed to build model: %s", exc)
        try:
            print(f"[quick-build] worker seed={seed}: build failed: {exc}")
        except Exception:
            pass
        return

    solver = cp_model.CpSolver()
    _configure_solver(solver, spec, cfg, seed)

    player_vars = artifacts.player_vars
    model = artifacts.model
    # Optional: apply lineup projection sum bounds using raw (non-jittered) weights
    try:
        if getattr(spec, "min_proj_sum", None) is not None or getattr(spec, "max_proj_sum", None) is not None:
            weights, scale = build_objective_weights(spec, jitter=0.0, seed=seed)
            if weights:
                obj_expr = cp_model.LinearExpr.Sum([
                    int(weights.get(pid, 0)) * var
                    for pid, var in zip(artifacts.player_ids, artifacts.player_vars)
                ])
                if getattr(spec, "min_proj_sum", None) is not None:
                    try:
                        floor_val = int(math.floor(float(spec.min_proj_sum) * float(scale)))  # type: ignore[arg-type]
                        model.Add(obj_expr >= floor_val)
                    except Exception:
                        pass
                if getattr(spec, "max_proj_sum", None) is not None:
                    try:
                        ceil_val = int(math.ceil(float(spec.max_proj_sum) * float(scale)))  # type: ignore[arg-type]
                        model.Add(obj_expr <= ceil_val)
                    except Exception:
                        pass
    except Exception:
        # Do not fail; bounds are optional
        pass
    seen_hashes: set[int] = set()
    emit_fn = _make_emit_fn(queue, stop_event, seen_hashes, cross_bloom, use_central_dedup=use_central_dedup)

    if cfg.nogood_rate > 0:
        def add_nogood_cut(idx_sel: Sequence[int]) -> None:
            if not idx_sel:
                return
            model.Add(sum(player_vars[i] for i in idx_sel) <= cfg.lineup_size - 1)

        cfg = cfg.with_cut_fn(add_nogood_cut)

    total_emitted = 0
    empty_cycles = 0
    while not stop_event.is_set():
        remaining_target = None
        if cfg.per_build > 0:
            remaining_target = max(cfg.per_build - total_emitted, 0)
            if remaining_target == 0:
                break

        cb = PoolingCallback(
            player_vars=player_vars,
            player_ids=artifacts.player_ids,
            cfg=cfg,
            emit_fn=emit_fn,
            stop_event=stop_event,
            target_emissions=remaining_target,
        )

        solver.SolveWithSolutionCallback(model, cb)
        total_emitted += cb.emitted
        # Reset empty cycle counter on progress or planned restart
        if cb.emitted > 0 or cb.restart_requested:
            empty_cycles = 0

        # Diagnostic print for streaming phase progress
        try:
            print(
                f"[quick-build] worker seed={seed}: streaming batch_emitted={cb.emitted} total_emitted={total_emitted} restart={cb.restart_requested}"
            )
        except Exception:
            pass

        if stop_event.is_set():
            logger.info(
                "[quick-build] worker seed=%s stop: stop_event set after %d emissions",
                seed,
                total_emitted,
            )
            break
        if cb.emitted == 0:
            # Allow a few consecutive empty cycles before stopping to avoid
            # premature termination when centralized dedup filters heavily.
            if cb.restart_requested:
                empty_cycles = 0
                continue
            empty_cycles += 1
            if empty_cycles < 3:
                continue
            logger.info(
                "[quick-build] worker seed=%s stop: no new solutions after %d empty cycles (total=%d)",
                seed,
                empty_cycles,
                total_emitted,
            )
            break
        if cfg.per_build > 0 and total_emitted >= cfg.per_build:
            logger.info(
                "[quick-build] worker seed=%s stop: per_build reached (%d)",
                seed,
                total_emitted,
            )
            break
        if not cb.restart_requested:
            # Continue next cycle to search further even if no restart was requested
            continue

    if cfg.enum_enable and cfg.enum_targets:
        try:
            print(f"[quick-build] worker seed={seed}: starting enumeration phase")
        except Exception:
            pass
        _run_enum_phase(spec, cfg, queue, stop_event, seed, seen_hashes, logger, cross_bloom)


def quick_build_pool(
    slate: Any,
    site: str,
    constraints: Any,
    qb_cfg: QuickBuildConfig,
    run_id: Optional[str] = None,
) -> QuickBuildResult:
    """Generate a deduped pool of lineup player-id tuples via QuickBuild workers."""

    if not isinstance(qb_cfg, QuickBuildConfig):
        raise TypeError("qb_cfg must be a QuickBuildConfig instance")

    cfg = qb_cfg.resolved()
    if run_id:
        cfg = replace(cfg, run_id=run_id).resolved()

    spec = _build_spec_from_payload(slate, site, constraints)

    # Use a centralized dedup collector to minimize cross-worker contention
    raw_queue: Queue = Queue(max(1, cfg.queue_size * 2))
    dedup_queue: Queue = Queue(max(1, cfg.queue_size))
    stop = Event()
    pool = InMemoryPool(max(1, cfg.effective_max_pool))
    stats = QuickBuildStats()

    # Optional cross-worker Bloom prefilter
    cross_bloom = None
    if cfg.approx_uniq:
        try:
            # Build a sharded Bloom used by the centralized collector
            cross_bloom = ShardedBloom.create(
                capacity=cfg.effective_max_pool,
                fpp=max(1e-6, float(cfg.bloom_fpp)),
                shards=max(1, int(getattr(cfg, "bloom_shards", 8) or 8)),
            )
            try:
                print(
                    f"[quick-build] central dedup: shards={getattr(cross_bloom, '_n', 1)} fpp~{cfg.bloom_fpp}"
                )
            except Exception:
                pass
        except Exception as exc:
            try:
                print(f"[quick-build] cross-bloom init failed: {exc}")
            except Exception:
                pass
            cross_bloom = None

    # Centralized dedup telemetry counters
    bloom_rejects = mp.Value('Q', 0)
    seen_dups = mp.Value('Q', 0)
    forwarded = mp.Value('Q', 0)

    def _collector():
        seen: set[int] = set()
        while not stop.is_set():
            try:
                key = raw_queue.get(timeout=0.1)
            except (Empty, EOFError, OSError):
                continue
            if key is None:
                continue
            h = fnv1a_64(tuple(key))
            if h in seen:
                with seen_dups.get_lock():
                    seen_dups.value += 1
                continue
            if cross_bloom is not None and not cross_bloom.test_and_set(h):
                with bloom_rejects.get_lock():
                    bloom_rejects.value += 1
                continue
            seen.add(h)
            try:
                dedup_queue.put(key, timeout=0.1)
                with forwarded.get_lock():
                    forwarded.value += 1
            except Full:
                # drop under backpressure
                pass

    collector = Process(target=_collector, daemon=True)
    collector.start()

    processes: List[Process] = []
    for idx in range(cfg.worker_count):
        seed_i = cfg.worker_seed(idx)
        proc = Process(
            target=_worker_main,
            kwargs={
                "spec": spec,
                "cfg": cfg,
                "queue": raw_queue,
                "stop_event": stop,
                "seed": seed_i,
                "cross_bloom": None,  # handled in collector
                "use_central_dedup": True,
            },
            daemon=True,
        )
        proc.start()
        processes.append(proc)

    start_time = time.time()
    # Track last accepted lineup to enforce consecutive min-uniques globally
    last_accepted_set: Optional[set[str]] = None
    try:
        while len(pool) < pool.max_size:
            try:
                lineup = dedup_queue.get(timeout=0.5)
            except (Empty, EOFError, OSError):
                if not any(p.is_alive() for p in processes):
                    break
                continue

            if lineup is None:
                continue

            # Enforce consecutive min-uniques in the central collector (global order)
            if cfg.min_uniq > 0 and last_accepted_set is not None:
                try:
                    lineup_set = set(lineup)
                    overlap = len(last_accepted_set & lineup_set)
                    changes = len(lineup_set) - overlap
                    if changes < cfg.min_uniq:
                        stats.dropped_low_quality += 1
                        continue
                except Exception:
                    # If anything goes wrong in the check, fall back to accepting logic
                    pass

            # Apply near-dup filtering if configured
            nd_thresh = float(getattr(cfg, "near_dup_jaccard", 0.0) or 0.0)
            if nd_thresh > 0.0:
                add_status = pool.try_add_with_jacc(lineup, nd_thresh)
            else:
                add_status = pool.try_add(lineup)
            if add_status == "ok":
                stats.accepted += 1
                try:
                    last_accepted_set = set(lineup)
                except Exception:
                    pass
            elif add_status == "dup":
                stats.duplicates += 1
            elif add_status == "near_dup":
                stats.near_duplicates += 1
            else:  # full
                stats.emitted = len(pool.rows())
                break

            if len(pool) >= pool.max_size:
                break
    finally:
        stop.set()
        for proc in processes:
            proc.join(timeout=1.0)
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        for proc in processes:
            if proc.is_alive():
                proc.kill()
        try:
            collector.join(timeout=1.0)
            if collector.is_alive():
                collector.terminate()
        except Exception:
            pass

    pool_rows = pool.rows()
    lineups = [row[0] for row in pool_rows]
    stats.emitted = len(lineups)
    stats.wall_time_s = time.time() - start_time
    # Copy centralized dedup telemetry
    try:
        stats.bloom_rejects = int(bloom_rejects.value)
        stats.collector_seen_dups = int(seen_dups.value)
        stats.collector_forwarded = int(forwarded.value)
    except Exception:
        pass

    if cfg.stats_path:
        dump_stats(stats, cfg.stats_path)

    return QuickBuildResult(lineups=lineups, stats=stats, config=cfg)
