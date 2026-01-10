"""Portfolio evaluation on base + runtime-generated holdout worlds.

This module is intentionally evaluation-only and does not participate in any
selection/optimization pipeline.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from projections.contest_sim.contest_sim_service import score_lineups

_PID_COL_RE = re.compile(r"^p(?P<idx>\d+)_id$", re.IGNORECASE)


def stable_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return stable_hash_bytes(encoded)


@dataclass(frozen=True)
class ParsedLineups:
    site: str
    lineup_ids: list[str]
    lineups: list[list[str]]
    player_exposure_counts: dict[str, int]
    draft_group_id: int | None = None


def _detect_player_id_columns(columns: Iterable[str]) -> list[str]:
    found: list[tuple[int, str]] = []
    for col in columns:
        match = _PID_COL_RE.match(str(col).strip())
        if not match:
            continue
        try:
            idx = int(match.group("idx"))
        except Exception:
            continue
        found.append((idx, str(col)))
    found.sort(key=lambda t: t[0])
    return [col for _, col in found]


def parse_lineups_csv(*, lineups_path: Path, site: str) -> ParsedLineups:
    df = pd.read_csv(lineups_path)
    pid_cols = _detect_player_id_columns(df.columns)
    if not pid_cols:
        raise ValueError(
            f"Unable to find player id columns like p1_id..pN_id in {lineups_path}; found columns={list(df.columns)}"
        )

    lineup_ids: list[str] = []
    if "lineup_id" in df.columns:
        lineup_ids = df["lineup_id"].fillna("").astype(str).tolist()
    else:
        lineup_ids = [str(i) for i in range(len(df))]

    lineups: list[list[str]] = []
    exposure: dict[str, int] = {}
    for _, row in df.iterrows():
        players: list[str] = []
        for col in pid_cols:
            raw = row.get(col)
            if pd.isna(raw):
                continue
            pid: str
            if isinstance(raw, (int, np.integer)):
                pid = str(int(raw))
            elif isinstance(raw, (float, np.floating)) and np.isfinite(raw):
                pid = str(int(raw))
            else:
                pid_raw = str(raw).strip()
                try:
                    num = float(pid_raw)
                    pid = str(int(num)) if np.isfinite(num) and float(num).is_integer() else pid_raw
                except Exception:
                    pid = pid_raw
            if pid.lower() == "nan" or pid == "":
                continue
            players.append(pid)
            exposure[pid] = exposure.get(pid, 0) + 1
        lineups.append(players)

    draft_group_id: int | None = None
    for col in ("draft_group_id", "draftGroupId", "draft_group", "slate_id"):
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna().astype(int).unique()
        if values.size == 1:
            draft_group_id = int(values[0])
            break

    return ParsedLineups(
        site=str(site),
        lineup_ids=lineup_ids,
        lineups=lineups,
        player_exposure_counts=exposure,
        draft_group_id=draft_group_id,
    )


def load_worlds_matrix_parquet(*, worlds_matrix_path: Path) -> tuple[np.ndarray, dict[str, int], list[str]]:
    df = pd.read_parquet(worlds_matrix_path)
    player_ids = [str(c) for c in df.columns]
    player_index = {pid: idx for idx, pid in enumerate(player_ids)}
    matrix = df.to_numpy(dtype=np.float64, copy=False)
    if matrix.ndim != 2 or matrix.shape[1] != len(player_ids):
        raise ValueError(f"Invalid worlds matrix shape={matrix.shape} for {worlds_matrix_path}")
    return matrix, player_index, player_ids


def compute_train_holdout_indices(
    *,
    n_worlds: int,
    train_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if n_worlds <= 0:
        raise ValueError("n_worlds must be positive")
    if not (0.0 < float(train_frac) < 1.0):
        raise ValueError("train_frac must be in (0, 1)")

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n_worlds))
    n_train = int(np.floor(float(train_frac) * float(n_worlds)))
    n_train = max(1, min(n_train, int(n_worlds) - 1))
    train_idx = np.sort(perm[:n_train].astype(np.int64))
    holdout_idx = np.sort(perm[n_train:].astype(np.int64))
    meta = {
        "method": "permute",
        "seed": int(seed),
        "train_frac": float(train_frac),
        "n_worlds": int(n_worlds),
        "n_train": int(train_idx.size),
        "n_holdout": int(holdout_idx.size),
        "train_idx_sha256": stable_hash_bytes(train_idx.tobytes()),
        "holdout_idx_sha256": stable_hash_bytes(holdout_idx.tobytes()),
    }
    return train_idx, holdout_idx, meta


def _safe_percentiles(values: np.ndarray, qs: list[float]) -> dict[str, float | None]:
    if values.size == 0:
        return {f"p{int(q)}": None for q in qs}
    out: dict[str, float | None] = {}
    pct = np.percentile(values, qs).astype(float)
    for q, v in zip(qs, pct, strict=True):
        out[f"p{int(q)}"] = float(v)
    return out


def summarize_lineup_scores(scores: np.ndarray) -> dict[str, Any]:
    """Return per-lineup and portfolio-max summaries for lineup_scores (L, W)."""
    if scores.ndim != 2:
        raise ValueError(f"Expected scores to be 2D, got shape={scores.shape}")
    n_lineups, n_worlds = int(scores.shape[0]), int(scores.shape[1])
    if n_lineups <= 0 or n_worlds <= 0:
        raise ValueError("scores must have non-empty lineup/world dimensions")

    # Vectorized per-lineup summaries.
    means = scores.mean(axis=1)
    pcts = np.percentile(scores, [90, 95, 99], axis=1).astype(float)  # (3, L)

    per_lineup = []
    for i in range(n_lineups):
        per_lineup.append(
            {
                "lineup_idx": int(i),
                "mean": float(means[i]),
                "p90": float(pcts[0, i]),
                "p95": float(pcts[1, i]),
                "p99": float(pcts[2, i]),
            }
        )

    max_scores = scores.max(axis=0)
    max_summary = {
        "mean": float(max_scores.mean()),
        **_safe_percentiles(max_scores, [90.0, 95.0, 99.0]),
    }

    return {
        "n_lineups": n_lineups,
        "n_worlds": n_worlds,
        "per_lineup": per_lineup,
        "portfolio_max": max_summary,
    }


def compute_threshold_from_train_max(
    *,
    train_scores: np.ndarray,
    quantile: float,
) -> dict[str, Any]:
    if train_scores.ndim != 2:
        raise ValueError("train_scores must be 2D")
    if not (0.0 < float(quantile) < 1.0):
        raise ValueError("quantile must be in (0, 1)")
    max_scores = train_scores.max(axis=0)
    threshold = float(np.quantile(max_scores, float(quantile)))
    return {
        "threshold": threshold,
        "source": "base_train_quantile",
        "quantile": float(quantile),
    }


def add_threshold_metric(
    *,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    if scores.ndim != 2:
        raise ValueError("scores must be 2D")
    max_scores = scores.max(axis=0)
    prob = float(np.mean(max_scores > float(threshold)))
    return {
        "threshold": float(threshold),
        "p_max_gt_threshold": prob,
    }


def compute_diversity_diagnostics(lineups: list[list[str]]) -> dict[str, Any]:
    n = len(lineups)
    if n <= 0:
        return {
            "n_lineups": 0,
            "unique_players": 0,
            "exposure_hhi": None,
            "pairwise_overlap_mean": None,
            "pairwise_overlap_p95": None,
            "pairwise_overlap_max": None,
            "duplicate_lineups": 0,
        }

    counts: dict[str, int] = {}
    lineup_sets: list[frozenset[str]] = []
    for lu in lineups:
        s = frozenset(str(pid) for pid in lu if str(pid).strip() and str(pid).lower() != "nan")
        lineup_sets.append(s)
        for pid in s:
            counts[pid] = counts.get(pid, 0) + 1

    shares = np.array([c / float(n) for c in counts.values()], dtype=np.float64)
    exposure_hhi = float(np.sum(shares**2)) if shares.size else None

    # Pairwise overlap summary (O(n^2), n is small ~150).
    overlaps: list[int] = []
    for i in range(n):
        a = lineup_sets[i]
        for j in range(i + 1, n):
            overlaps.append(len(a & lineup_sets[j]))
    overlap_arr = np.asarray(overlaps, dtype=np.int64)
    overlap_mean = float(overlap_arr.mean()) if overlap_arr.size else None
    overlap_p95 = float(np.percentile(overlap_arr, 95)) if overlap_arr.size else None
    overlap_max = int(overlap_arr.max()) if overlap_arr.size else None

    duplicate_lineups = int(n - len(set(lineup_sets)))

    return {
        "n_lineups": int(n),
        "unique_players": int(len(counts)),
        "exposure_hhi": exposure_hhi,
        "pairwise_overlap_mean": overlap_mean,
        "pairwise_overlap_p95": overlap_p95,
        "pairwise_overlap_max": overlap_max,
        "duplicate_lineups": duplicate_lineups,
    }


def score_portfolio_on_worlds(
    *,
    lineups: list[list[str]],
    worlds_matrix: np.ndarray,
    player_index: dict[str, int],
) -> np.ndarray:
    return score_lineups(lineups=lineups, worlds_matrix=worlds_matrix, player_index=player_index)
