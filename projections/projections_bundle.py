"""Canonical projection bundle assembly + normalization.

This module is the single source of truth for:
- selecting a projections run_id for a slate (blessed/pinned/latest)
- loading the unified projections artifact
- adding *canonical* projection contract fields with explicit conditioning

The goal is that UI (Dash/GameView), optimizer, and contest sim all consume the
same canonical fields derived from the same underlying sources.

Notes
-----
We preserve legacy columns for back-compat. Canonical fields are added as
additional columns with explicit names (e.g. `fpts_sim_uncond_mean`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date as date_cls
from pathlib import Path
from typing import Iterable

import pandas as pd

from projections import paths
from projections.pipeline import control_plane


LATEST_POINTER = "latest_run.json"
PINNED_POINTER = "pinned_run.json"
BLESSED_POINTER = "blessed_run.json"


@dataclass(frozen=True, slots=True)
class ResolvedProjectionsRun:
    """Resolved unified projections run selection."""

    run_id: str | None
    source: str | None  # explicit|blessed|pinned|promoted|latest_dir|direct
    updated_at: str | None
    day_dir: Path
    run_dir: Path | None
    projections_path: Path | None
    blessed_run_id: str | None
    pinned_run_id: str | None
    latest_run_id: str | None


@dataclass(frozen=True, slots=True)
class ProjectionBundle:
    game_date: str
    projections_run: ResolvedProjectionsRun
    df: pd.DataFrame


def _to_date_str(day: str | date_cls) -> str:
    if isinstance(day, date_cls):
        return day.isoformat()
    return str(day)


def _read_pointer_run_id(day_dir: Path, filename: str) -> tuple[str | None, str | None]:
    """Return (run_id, updated_at) from a pointer json file."""
    path = day_dir / filename
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    run_id = payload.get("run_id")
    updated_at = payload.get("updated_at") or payload.get("run_as_of_ts")
    return (str(run_id) if run_id else None), (str(updated_at) if updated_at else None)


def resolve_unified_projections_run(
    game_date: str | date_cls,
    *,
    run_id: str | None = None,
    data_root: Path | None = None,
) -> ResolvedProjectionsRun:
    """Resolve which unified projections run to read for a given slate.

    Priority when `run_id` is not provided:
      1) blessed_run.json (validated)
      2) pinned_run.json  (manual override)
      3) promoted pointer (LATEST/current.json or latest_run.json)
      4) newest run=... dir (only when PROJECTIONS_ALLOW_UNPROMOTED_RUN_READS=1)

    This mirrors the dashboard behavior so all consumers agree on a single
    default run_id.
    """

    root = Path(data_root) if data_root is not None else paths.get_data_root()
    day = _to_date_str(game_date)
    day_dir = root / "artifacts" / "projections" / day

    blessed_run_id, blessed_updated = _read_pointer_run_id(day_dir, BLESSED_POINTER)
    pinned_run_id, pinned_updated = _read_pointer_run_id(day_dir, PINNED_POINTER)
    latest_run_id, latest_updated = _read_pointer_run_id(day_dir, LATEST_POINTER)

    # Explicit run_id always wins.
    if run_id:
        candidate = day_dir / f"run={run_id}" / "projections.parquet"
        if candidate.exists():
            return ResolvedProjectionsRun(
                run_id=str(run_id),
                source="explicit",
                updated_at=None,
                day_dir=day_dir,
                run_dir=candidate.parent,
                projections_path=candidate,
                blessed_run_id=blessed_run_id,
                pinned_run_id=pinned_run_id,
                latest_run_id=latest_run_id,
            )

    # Blessed pointer.
    if blessed_run_id:
        candidate = day_dir / f"run={blessed_run_id}" / "projections.parquet"
        if candidate.exists():
            return ResolvedProjectionsRun(
                run_id=str(blessed_run_id),
                source="blessed",
                updated_at=blessed_updated,
                day_dir=day_dir,
                run_dir=candidate.parent,
                projections_path=candidate,
                blessed_run_id=blessed_run_id,
                pinned_run_id=pinned_run_id,
                latest_run_id=latest_run_id,
            )

    # Pinned pointer.
    if pinned_run_id:
        candidate = day_dir / f"run={pinned_run_id}" / "projections.parquet"
        if candidate.exists():
            return ResolvedProjectionsRun(
                run_id=str(pinned_run_id),
                source="pinned",
                updated_at=pinned_updated,
                day_dir=day_dir,
                run_dir=candidate.parent,
                projections_path=candidate,
                blessed_run_id=blessed_run_id,
                pinned_run_id=pinned_run_id,
                latest_run_id=latest_run_id,
            )

    # Promoted pointer (LATEST/current.json or legacy latest_run.json).
    promoted = control_plane.read_promoted_run_id(day_dir) or latest_run_id
    if promoted:
        candidate = day_dir / f"run={promoted}" / "projections.parquet"
        if candidate.exists():
            return ResolvedProjectionsRun(
                run_id=str(promoted),
                source="promoted",
                updated_at=latest_updated,
                day_dir=day_dir,
                run_dir=candidate.parent,
                projections_path=candidate,
                blessed_run_id=blessed_run_id,
                pinned_run_id=pinned_run_id,
                latest_run_id=latest_run_id,
            )

    # Fallback to newest run dir only if explicitly allowed.
    if day_dir.exists() and control_plane.allow_unpromoted_run_reads():
        run_dirs = sorted(
            [p for p in day_dir.iterdir() if p.is_dir() and p.name.startswith("run=")],
            reverse=True,
        )
        for rd in run_dirs:
            candidate = rd / "projections.parquet"
            if candidate.exists():
                inferred = rd.name.split("=", 1)[1]
                return ResolvedProjectionsRun(
                    run_id=str(inferred),
                    source="latest_dir",
                    updated_at=None,
                    day_dir=day_dir,
                    run_dir=rd,
                    projections_path=candidate,
                    blessed_run_id=blessed_run_id,
                    pinned_run_id=pinned_run_id,
                    latest_run_id=latest_run_id,
                )

    # Nothing found.
    return ResolvedProjectionsRun(
        run_id=None,
        source=None,
        updated_at=None,
        day_dir=day_dir,
        run_dir=None,
        projections_path=None,
        blessed_run_id=blessed_run_id,
        pinned_run_id=pinned_run_id,
        latest_run_id=latest_run_id,
    )


def load_unified_projections_df(
    game_date: str | date_cls,
    *,
    run_id: str | None = None,
    data_root: Path | None = None,
) -> ProjectionBundle:
    """Load unified projections parquet for a slate and attach canonical fields."""

    resolved = resolve_unified_projections_run(game_date, run_id=run_id, data_root=data_root)
    if resolved.projections_path is None or resolved.run_id is None:
        raise FileNotFoundError(
            f"Unified projections not found for date={_to_date_str(game_date)} under {resolved.day_dir}"
        )

    df = pd.read_parquet(resolved.projections_path)
    df = add_canonical_projection_fields(df)
    return ProjectionBundle(game_date=_to_date_str(game_date), projections_run=resolved, df=df)


def _first_present(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    cols = set(df.columns)
    return next((c for c in candidates if c in cols), None)


def add_canonical_projection_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical contract columns added.

    Canonical naming rules:
    - `*_cond_*` == conditional on player being active/playing (worlds where inactive excluded)
    - `*_uncond_*` == unconditional (inactive/DNP worlds contribute 0)
    - `*_sim_*` == derived from sim_v2 worlds summaries

    This function is intentionally additive (does not rename/remove legacy columns).
    """

    if df.empty:
        return df.copy()

    out = df.copy()

    # ------------------------------------------------------------------
    # Play probability
    # ------------------------------------------------------------------

    p_raw_col = _first_present(out, ["p_play_raw", "play_prob_raw", "play_prob", "p_play", "play_probability"])
    p_eff_col = _first_present(out, ["p_play_eff", "play_prob_eff"])

    if p_raw_col is None:
        out["p_play_raw"] = 1.0
    else:
        out["p_play_raw"] = pd.to_numeric(out[p_raw_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    # Candidate effective play probability from inputs (before sim realization).
    if p_eff_col is None:
        p_eff_base = out["p_play_raw"].astype(float)
    else:
        p_eff_base = pd.to_numeric(out[p_eff_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    # Realized active rate across simulated worlds (sim_v2 output).
    p_active_col = _first_present(out, ["minutes_sim_p_active", "sim_p_active"])
    if p_active_col is None:
        out["minutes_sim_p_active"] = p_eff_base
    else:
        out["minutes_sim_p_active"] = pd.to_numeric(out[p_active_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    # Effective play probability used for downstream decision metrics.
    #
    # If the sim produced `sim_p_active`, treat that as the effective probability
    # (it is what the worlds actually realized). Otherwise fall back to the best
    # available pre-sim estimate.
    out["p_play_eff"] = out["minutes_sim_p_active"].astype(float) if p_active_col else p_eff_base

    # ------------------------------------------------------------------
    # Minutes: model-conditional and sim summaries
    # ------------------------------------------------------------------

    # Model conditional minutes (given plays).
    for q in (10, 50, 90):
        legacy = f"minutes_p{q}_cond"
        canon = f"minutes_cond_p{q}"
        if legacy in out.columns and canon not in out.columns:
            out[canon] = pd.to_numeric(out[legacy], errors="coerce")

    # Sim conditional (given plays) minutes summaries.
    sim_minutes_cond_map = {
        "minutes_sim_mean": "minutes_sim_cond_mean",
        "minutes_sim_p10": "minutes_sim_cond_p10",
        "minutes_sim_p50": "minutes_sim_cond_p50",
        "minutes_sim_p90": "minutes_sim_cond_p90",
        "minutes_sim_std": "minutes_sim_cond_std",
    }
    for legacy, canon in sim_minutes_cond_map.items():
        if legacy in out.columns and canon not in out.columns:
            out[canon] = pd.to_numeric(out[legacy], errors="coerce")

    # Some legacy/serialized payloads prefix sim columns with `sim_` (e.g. minutes API legacy path).
    sim_minutes_cond_map_prefixed = {
        "sim_minutes_sim_mean": "minutes_sim_cond_mean",
        "sim_minutes_sim_p10": "minutes_sim_cond_p10",
        "sim_minutes_sim_p50": "minutes_sim_cond_p50",
        "sim_minutes_sim_p90": "minutes_sim_cond_p90",
        "sim_minutes_sim_std": "minutes_sim_cond_std",
    }
    for legacy, canon in sim_minutes_cond_map_prefixed.items():
        if legacy in out.columns and canon not in out.columns:
            out[canon] = pd.to_numeric(out[legacy], errors="coerce")

    # Sim unconditional (DNP=0) minutes summaries.
    sim_minutes_uncond_map = {
        "minutes_sim_mean_uncond": "minutes_sim_uncond_mean",
        "minutes_sim_p10_uncond": "minutes_sim_uncond_p10",
        "minutes_sim_p50_uncond": "minutes_sim_uncond_p50",
        "minutes_sim_p90_uncond": "minutes_sim_uncond_p90",
        "minutes_sim_std_uncond": "minutes_sim_uncond_std",
    }
    for legacy, canon in sim_minutes_uncond_map.items():
        if legacy in out.columns and canon not in out.columns:
            out[canon] = pd.to_numeric(out[legacy], errors="coerce")

    sim_minutes_uncond_map_prefixed = {
        "sim_minutes_sim_mean_uncond": "minutes_sim_uncond_mean",
        "sim_minutes_sim_p10_uncond": "minutes_sim_uncond_p10",
        "sim_minutes_sim_p50_uncond": "minutes_sim_uncond_p50",
        "sim_minutes_sim_p90_uncond": "minutes_sim_uncond_p90",
        "sim_minutes_sim_std_uncond": "minutes_sim_uncond_std",
    }
    for legacy, canon in sim_minutes_uncond_map_prefixed.items():
        if legacy in out.columns and canon not in out.columns:
            out[canon] = pd.to_numeric(out[legacy], errors="coerce")

    # If sim unconditional mean is missing but conditional mean is present, fall back
    # to p_active * mean_cond (decision-metric approximation).
    if "minutes_sim_uncond_mean" not in out.columns and "minutes_sim_cond_mean" in out.columns:
        out["minutes_sim_uncond_mean"] = (
            pd.to_numeric(out["minutes_sim_cond_mean"], errors="coerce").fillna(0.0).astype(float)
            * out["minutes_sim_p_active"].astype(float)
        )

    # ------------------------------------------------------------------
    # FPTS (DraftKings): sim summaries
    # ------------------------------------------------------------------

    # Conditional-on-playing FPTS summaries from sim_v2.
    fpts_cond_map = {
        "dk_fpts_mean": "fpts_sim_cond_mean",
        "dk_fpts_std": "fpts_sim_cond_std",
        "dk_fpts_p05": "fpts_sim_cond_p05",
        "dk_fpts_p10": "fpts_sim_cond_p10",
        "dk_fpts_p25": "fpts_sim_cond_p25",
        "dk_fpts_p50": "fpts_sim_cond_p50",
        "dk_fpts_p75": "fpts_sim_cond_p75",
        "dk_fpts_p90": "fpts_sim_cond_p90",
        "dk_fpts_p95": "fpts_sim_cond_p95",
    }
    for legacy, canon in fpts_cond_map.items():
        if legacy in out.columns and canon not in out.columns:
            out[canon] = pd.to_numeric(out[legacy], errors="coerce")

    fpts_cond_map_prefixed = {
        "sim_dk_fpts_mean": "fpts_sim_cond_mean",
        "sim_dk_fpts_std": "fpts_sim_cond_std",
        "sim_dk_fpts_p05": "fpts_sim_cond_p05",
        "sim_dk_fpts_p10": "fpts_sim_cond_p10",
        "sim_dk_fpts_p25": "fpts_sim_cond_p25",
        "sim_dk_fpts_p50": "fpts_sim_cond_p50",
        "sim_dk_fpts_p75": "fpts_sim_cond_p75",
        "sim_dk_fpts_p90": "fpts_sim_cond_p90",
        "sim_dk_fpts_p95": "fpts_sim_cond_p95",
    }
    for legacy, canon in fpts_cond_map_prefixed.items():
        if legacy in out.columns and canon not in out.columns:
            out[canon] = pd.to_numeric(out[legacy], errors="coerce")

    # Unconditional (DNP=0) FPTS summaries.
    fpts_uncond_map = {
        "dk_fpts_mean_uncond": "fpts_sim_uncond_mean",
        "dk_fpts_std_uncond": "fpts_sim_uncond_std",
        "dk_fpts_p05_uncond": "fpts_sim_uncond_p05",
        "dk_fpts_p10_uncond": "fpts_sim_uncond_p10",
        "dk_fpts_p25_uncond": "fpts_sim_uncond_p25",
        "dk_fpts_p50_uncond": "fpts_sim_uncond_p50",
        "dk_fpts_p75_uncond": "fpts_sim_uncond_p75",
        "dk_fpts_p90_uncond": "fpts_sim_uncond_p90",
        "dk_fpts_p95_uncond": "fpts_sim_uncond_p95",
    }
    for legacy, canon in fpts_uncond_map.items():
        if legacy in out.columns and canon not in out.columns:
            out[canon] = pd.to_numeric(out[legacy], errors="coerce")

    fpts_uncond_map_prefixed = {
        "sim_dk_fpts_mean_uncond": "fpts_sim_uncond_mean",
        "sim_dk_fpts_std_uncond": "fpts_sim_uncond_std",
        "sim_dk_fpts_p05_uncond": "fpts_sim_uncond_p05",
        "sim_dk_fpts_p10_uncond": "fpts_sim_uncond_p10",
        "sim_dk_fpts_p25_uncond": "fpts_sim_uncond_p25",
        "sim_dk_fpts_p50_uncond": "fpts_sim_uncond_p50",
        "sim_dk_fpts_p75_uncond": "fpts_sim_uncond_p75",
        "sim_dk_fpts_p90_uncond": "fpts_sim_uncond_p90",
        "sim_dk_fpts_p95_uncond": "fpts_sim_uncond_p95",
    }
    for legacy, canon in fpts_uncond_map_prefixed.items():
        if legacy in out.columns and canon not in out.columns:
            out[canon] = pd.to_numeric(out[legacy], errors="coerce")

    # Fallback: if unconditional mean missing but conditional mean is present, approximate
    # using p_play_eff (closest available to sim masking probability).
    if "fpts_sim_uncond_mean" not in out.columns and "fpts_sim_cond_mean" in out.columns:
        out["fpts_sim_uncond_mean"] = (
            pd.to_numeric(out["fpts_sim_cond_mean"], errors="coerce").fillna(0.0).astype(float)
            * out["p_play_eff"].astype(float)
        )

    # ------------------------------------------------------------------
    # Contract invariants (best-effort).
    # ------------------------------------------------------------------

    # Probabilities must be in [0, 1].
    for col in ("p_play_raw", "p_play_eff", "minutes_sim_p_active"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    # Unconditional minutes/FPTS should be non-negative.
    for col in (
        "minutes_sim_uncond_mean",
        "minutes_sim_uncond_p10",
        "minutes_sim_uncond_p50",
        "minutes_sim_uncond_p90",
        "fpts_sim_uncond_mean",
        "fpts_sim_uncond_p50",
        "fpts_sim_uncond_p90",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    return out
