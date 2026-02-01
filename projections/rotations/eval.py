from __future__ import annotations

import hashlib
import json
import shutil
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import numpy as np
import pandas as pd

from projections.rotations.eval_manifest import (
    build_rot_eval_manifest,
    resolve_rot_bundle_dir,
    write_rot_eval_input_hashes,
)
from projections.rotations.candidate_pool import (
    build_candidate_pool_predictor_threshold,
    build_candidate_pool_prior,
    build_candidate_pool_prior_topn_team_game,
    build_candidate_pool_roster,
    build_candidate_pool_truth,
)
from projections.rotations.generator import TeamContext
from projections.rotations.manifest import write_json, write_latest_published_run_id
from projections.rotations.priors_humility import HumilityConfig, humility_config_as_dict
from projections.rotations.rotation_gate import GateConfig, gate_config_as_dict
from projections.rotations.schemas import LINEUP_COLS
from projections.rotations.template_generator import TemplateRotationGenerator
from projections.rotations.rotation_predictor import (
    canonicalize_game_id,
    load_cached_all_predictions,
    load_cached_predictions,
    load_cached_train_predictions,
    load_rotation_predictor_bundle,
    season_start_year_from_game_id,
)
from projections.rotations.player_map import build_person_id_to_internal_id_map


SampleMode = Literal["random", "first"]
CandidatePoolMode = Literal["truth", "prior_topn", "prior_threshold", "predictor_threshold", "roster"]
REQUIRED_MINUTES_PRIOR_COLS: tuple[str, ...] = ("game_id", "team_id", "player_id", "minutes_prior", "play_prob")
OPTIONAL_MINUTES_PRIOR_COLS: tuple[str, ...] = ("minutes_p10", "minutes_p90")


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)


def _stable_team_game_seed(*, base_seed: int, season_id: str, game_id: str, team_id: int) -> int:
    key = f"{int(base_seed)}|{season_id}|{game_id}|{int(team_id)}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    # Keep within 32-bit to play nice with downstream RNG usage.
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _unique_ints_sorted(values: Iterable[Any]) -> list[int]:
    out: set[int] = set()
    for v in values:
        if v is None:
            continue
        try:
            out.add(int(v))
        except Exception:
            continue
    return sorted(out)


def _coerce_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if str(s.dtype).startswith("boolean"):
        return s.fillna(False).astype(bool)
    # Robust to 0/1, strings, and NaN.
    return (
        s.fillna(False)
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(["1", "true", "t", "yes", "y"])
    )


def _infer_starters_from_events(events_team_game: pd.DataFrame) -> list[int]:
    if events_team_game.empty:
        return []
    segs = events_team_game.sort_values(["segment_idx"], kind="mergesort")
    if "duration_sec" in segs.columns and (segs["duration_sec"].fillna(0).to_numpy() > 0).any():
        first_idx = int(np.argmax(segs["duration_sec"].fillna(0).to_numpy() > 0))
        row = segs.iloc[first_idx]
    else:
        row = segs.iloc[0]
    starters = [row.get(c) for c in LINEUP_COLS]
    starters = [int(v) for v in starters if v is not None]
    return starters


def _compute_player_summary(minutes: np.ndarray) -> dict[str, float]:
    m = np.asarray(minutes, dtype=np.float64)
    if m.size == 0:
        return {
            "minutes_mean": 0.0,
            "minutes_p10": 0.0,
            "minutes_p50": 0.0,
            "minutes_p90": 0.0,
            "p_played_ge_1_pred": 0.0,
            "p_played_ge_5_pred": 0.0,
            "p_minutes_lt5_pred": 0.0,
            "p_minutes_eq0_pred": 0.0,
        }
    q10, q50, q90 = np.quantile(m, [0.1, 0.5, 0.9]).tolist()
    return {
        "minutes_mean": float(m.mean()),
        "minutes_p10": float(q10),
        "minutes_p50": float(q50),
        "minutes_p90": float(q90),
        "p_played_ge_1_pred": float((m >= 1.0).mean()),
        "p_played_ge_5_pred": float((m >= 5.0).mean()),
        "p_minutes_lt5_pred": float((m < 5.0).mean()),
        "p_minutes_eq0_pred": float((m == 0.0).mean()),
    }


@dataclass(frozen=True)
class EvalMetrics:
    brier_ge1: float
    brier_ge5: float
    minutes_mae: float
    n_team_games: int
    n_players: int


def _compute_calibration(
    *,
    player_eval: pd.DataFrame,
    p_col: str,
    y_col: str,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    if player_eval.empty:
        calib = pd.DataFrame(
            {
                "bin_idx": np.arange(n_bins, dtype=np.int64),
                "count": np.zeros(n_bins, dtype=np.int64),
                "p_pred_mean": np.full(n_bins, np.nan, dtype=np.float64),
                "y_true_mean": np.full(n_bins, np.nan, dtype=np.float64),
                "brier_bin_mean": np.full(n_bins, np.nan, dtype=np.float64),
                "brier_contribution": np.zeros(n_bins, dtype=np.float64),
            }
        )
        return calib, float("nan")

    p = pd.to_numeric(player_eval[p_col], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=np.float64)
    y = _coerce_bool(player_eval[y_col]).to_numpy(dtype=np.bool_)
    y_f = y.astype(np.float64)
    err2 = (p - y_f) ** 2
    brier = float(err2.mean()) if err2.size else float("nan")

    # Bin into [0.0,0.1),...,[0.9,1.0]; clamp p=1.0 into last bin.
    bin_idx = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    df = pd.DataFrame({"bin_idx": bin_idx, "p": p, "y": y_f, "err2": err2})
    grouped = df.groupby("bin_idx", sort=True)

    calib = grouped.agg(
        count=("p", "size"),
        p_pred_mean=("p", "mean"),
        y_true_mean=("y", "mean"),
        brier_bin_mean=("err2", "mean"),
        brier_contribution=("err2", "sum"),
    ).reset_index()
    # Normalize brier contribution to global mean terms, so it sums to brier.
    denom = float(len(df))
    calib["brier_contribution"] = calib["brier_contribution"].astype(np.float64) / (denom if denom > 0 else 1.0)

    # Ensure all bins present.
    all_bins = pd.DataFrame({"bin_idx": np.arange(n_bins, dtype=np.int64)})
    calib = all_bins.merge(calib, on="bin_idx", how="left")
    calib["count"] = calib["count"].fillna(0).astype(np.int64)
    for c in ["p_pred_mean", "y_true_mean", "brier_bin_mean", "brier_contribution"]:
        calib[c] = calib[c].astype(np.float64)
    return calib, brier


def run_rotation_generator_eval(
    rot_bundle_path: Path,
    run_id: str,
    n_worlds: int,
    seed: int,
    limit_team_games: int,
    sample_mode: SampleMode,
    out_dir: Path,
    overwrite: bool,
    *,
    use_truth_minutes_prior: bool = True,
    minutes_prior_parquet: Path | None = None,
    restrict_to_prior_games: bool = True,
    candidate_pool: CandidatePoolMode = "truth",
    candidate_top_n: int = 12,
    candidate_min_minutes_prior: float = 0.0,
    candidate_min_play_prob: float = 0.8,
    candidate_min_candidates: int = 8,
    pool_max_size: int = 11,
    pool_t_ge15: float = 0.35,
    pool_t_ge5: float = 0.35,
    pool_always_include_top_n: int = 8,
    humility_config: HumilityConfig | None = None,
    gate_config: GateConfig | None = None,
    rotation_predictor_bundle: Path | None = None,
    gate_feature_source: str = "cached_preds",
    gate_max_train_rows: int | None = None,
    baseline_out_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate TemplateRotationGenerator realism under different candidate-pool modes.

    Legacy/default (`candidate_pool=truth`) evaluates mapping + template sampling realism:
    - `candidate_player_ids` fixed to truth participants (minutes_actual>0 OR played_ge_1==True)
    - `starter_candidates` fixed to truth starters (fallback: first segment lineup)
    - `minutes_prior` can optionally use truth minutes (default True) as a mapping stabilizer

    Non-truth pools must not use minutes_actual/played flags for membership; they are intended
    to mimic live availability errors (false promotions/missed promotions) under injury chaos.
    """
    rot_bundle_path = Path(rot_bundle_path)
    out_dir = Path(out_dir)
    sample_mode = str(sample_mode)  # runtime validation
    if sample_mode not in {"random", "first"}:
        raise ValueError(f"Unknown sample_mode: {sample_mode}")

    candidate_pool = str(candidate_pool).strip().lower()
    if candidate_pool not in {"truth", "prior_topn", "prior_threshold", "predictor_threshold", "roster"}:
        raise ValueError(
            "Unknown candidate_pool: "
            f"{candidate_pool} (expected truth|prior_topn|prior_threshold|predictor_threshold|roster)"
        )

    gate_cfg = gate_config or GateConfig()
    gate_feature_source = str(gate_feature_source).strip().lower()
    if gate_feature_source not in {"cached_all", "cached_preds", "cached_train", "none"}:
        raise ValueError(
            f"Unknown gate_feature_source: {gate_feature_source} (expected cached_all|cached_preds|cached_train|none)"
        )
    if bool(gate_cfg.enabled) and gate_feature_source != "none" and rotation_predictor_bundle is None:
        raise ValueError("--rotation-predictor-bundle is required when --gate is enabled (unless --gate-feature-source none)")

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output dir exists (use overwrite): {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rot_bundle_dir = resolve_rot_bundle_dir(rot_bundle_path)
    events_path = rot_bundle_dir / "rotation_events.parquet"
    labels_path = rot_bundle_dir / "rotation_labels.parquet"

    events_cols = [
        "season_id",
        "game_id",
        "team_id",
        "opponent_team_id",
        "is_home",
        "segment_idx",
        "duration_sec",
        *LINEUP_COLS,
    ]
    labels_cols = [
        "game_id",
        "team_id",
        "player_id",
        "minutes_actual",
        "played_ge_1",
        "played_ge_5",
        "starter_actual",
        "regime_label",
    ]

    events = pd.read_parquet(events_path, columns=events_cols)
    labels = pd.read_parquet(labels_path, columns=labels_cols)

    priors = None
    priors_gb = None
    rot_games_total = 0
    prior_games_total = 0
    overlap_games_total = 0
    overlap_rate = float("nan")
    prior_players_total = 0
    overlap_players_total = 0
    prior_coverage_rate = float("nan")
    use_truth_minutes_prior_for_mapping = bool(use_truth_minutes_prior) and minutes_prior_parquet is None
    if candidate_pool != "truth" and use_truth_minutes_prior_for_mapping:
        raise ValueError(
            "Non-truth candidate pools must not use truth minutes as a prior. "
            "Pass --minutes-prior-parquet (recommended) and/or disable --use-truth-minutes-prior."
        )

    if minutes_prior_parquet is not None:
        p = Path(minutes_prior_parquet)
        if not p.exists():
            raise FileNotFoundError(f"minutes_prior_parquet not found: {p}")
        priors = pd.read_parquet(p)
        missing = [c for c in REQUIRED_MINUTES_PRIOR_COLS if c not in priors.columns]
        if missing:
            raise ValueError(
                f"minutes_prior_parquet missing required columns: {missing}. "
                f"Need {list(REQUIRED_MINUTES_PRIOR_COLS)}. Got columns={list(priors.columns)} from {p}"
            )
        keep_cols = list(REQUIRED_MINUTES_PRIOR_COLS) + [c for c in OPTIONAL_MINUTES_PRIOR_COLS if c in priors.columns]
        priors = priors[keep_cols].copy()
        priors["game_id"] = priors["game_id"].astype("string").map(canonicalize_game_id)
        priors["team_id"] = pd.to_numeric(priors["team_id"], errors="coerce").astype("Int64")
        priors["player_id"] = pd.to_numeric(priors["player_id"], errors="coerce").astype("Int64")
        priors["minutes_prior"] = pd.to_numeric(priors["minutes_prior"], errors="coerce").astype(np.float64).fillna(0.0)
        if "minutes_p10" in priors.columns:
            priors["minutes_p10"] = pd.to_numeric(priors["minutes_p10"], errors="coerce").astype(np.float64).fillna(0.0)
        if "minutes_p90" in priors.columns:
            priors["minutes_p90"] = pd.to_numeric(priors["minutes_p90"], errors="coerce").astype(np.float64).fillna(0.0)
        priors["play_prob"] = pd.to_numeric(priors["play_prob"], errors="coerce").astype(np.float64).fillna(0.0).clip(0.0, 1.0)
        priors = priors.dropna(subset=["game_id", "team_id", "player_id"]).copy()
        priors["team_id"] = priors["team_id"].astype(int)
        priors["player_id"] = priors["player_id"].astype(int)
        priors = priors[priors["game_id"].astype("string") != ""].copy()
        if not priors.empty:
            max_pid = int(priors["player_id"].max())
            # Guardrail: rot_v1 internal IDs are small contiguous ints (historically ~1-600).
            # If this looks like NBA personId space (e.g. 201143), fail fast.
            if max_pid > 2000:
                raise ValueError(
                    f"minutes_prior_parquet appears to use non-internal player_id domain: max(player_id)={max_pid}. "
                    "Expected rot_v1 internal IDs (small contiguous ints)."
                )
        priors = priors.sort_values(["game_id", "team_id", "player_id"], kind="mergesort").reset_index(drop=True)
        priors_gb = priors.groupby(["game_id", "team_id"], sort=False)

    # Normalize core dtypes up-front for stable sampling + joins.
    events["season_id"] = events["season_id"].astype("string")
    events["game_id"] = events["game_id"].astype("string").map(canonicalize_game_id)
    events["team_id"] = pd.to_numeric(events["team_id"], errors="coerce").astype("Int64")
    events["opponent_team_id"] = pd.to_numeric(events["opponent_team_id"], errors="coerce").astype("Int64")
    events["is_home"] = _coerce_bool(events["is_home"])
    events["segment_idx"] = pd.to_numeric(events["segment_idx"], errors="coerce").astype("Int64")
    events["duration_sec"] = pd.to_numeric(events["duration_sec"], errors="coerce").astype("Int64")
    for c in LINEUP_COLS:
        events[c] = pd.to_numeric(events[c], errors="coerce").astype("Int64")

    labels["game_id"] = labels["game_id"].astype("string").map(canonicalize_game_id)
    labels["team_id"] = pd.to_numeric(labels["team_id"], errors="coerce").astype("Int64")
    labels["player_id"] = pd.to_numeric(labels["player_id"], errors="coerce").astype("Int64")
    labels["minutes_actual"] = pd.to_numeric(labels["minutes_actual"], errors="coerce").astype(np.float64).fillna(0.0)
    labels["played_ge_1"] = _coerce_bool(labels["played_ge_1"])
    labels["played_ge_5"] = _coerce_bool(labels["played_ge_5"])
    labels["starter_actual"] = _coerce_bool(labels["starter_actual"])
    labels["regime_label"] = labels["regime_label"].astype("string")

    events = events.dropna(subset=["game_id", "team_id"]).copy()
    labels = labels.dropna(subset=["game_id", "team_id", "player_id"]).copy()
    events = events[events["game_id"].astype("string") != ""].copy()
    labels = labels[labels["game_id"].astype("string") != ""].copy()
    events["team_id"] = events["team_id"].astype(int)
    events["opponent_team_id"] = events["opponent_team_id"].fillna(-1).astype(int)
    events["segment_idx"] = events["segment_idx"].fillna(0).astype(int)
    events["duration_sec"] = events["duration_sec"].fillna(0).astype(int)
    for c in LINEUP_COLS:
        events[c] = events[c].fillna(0).astype(int)
    labels["team_id"] = labels["team_id"].astype(int)
    labels["player_id"] = labels["player_id"].astype(int)

    if priors is not None:
        # Coverage diagnostics in internal-id space (based on rot_v1 labels).
        labels_key = labels[["game_id", "team_id", "player_id"]].drop_duplicates().copy()
        priors_key = priors[["game_id", "team_id", "player_id"]].drop_duplicates().copy()
        prior_players_total = int(priors_key["player_id"].nunique())
        if len(labels_key):
            overlap_key = labels_key.merge(priors_key, on=["game_id", "team_id", "player_id"], how="inner")
            overlap_players_total = int(overlap_key["player_id"].nunique())
            prior_coverage_rate = float(len(overlap_key) / len(labels_key)) if len(labels_key) else float("nan")

    # Team-game metadata comes from events (labels don't include season/opponent/home).
    team_games = (
        labels[["game_id", "team_id"]]
        .drop_duplicates()
        .merge(
            events[["season_id", "game_id", "team_id", "opponent_team_id", "is_home"]]
            .drop_duplicates(subset=["game_id", "team_id"], keep="first"),
            on=["game_id", "team_id"],
            how="left",
        )
    )
    team_games["season_id"] = team_games["season_id"].fillna("unknown").astype("string")
    team_games["opponent_team_id"] = pd.to_numeric(team_games["opponent_team_id"], errors="coerce").fillna(-1).astype(int)
    team_games["is_home"] = _coerce_bool(team_games["is_home"].fillna(False))

    # Deterministic ordering before sampling.
    team_games = team_games.sort_values(["season_id", "game_id", "team_id"], kind="mergesort").reset_index(drop=True)

    rot_games_total = int(team_games["game_id"].nunique())
    if priors is not None:
        prior_games = sorted({str(v) for v in priors["game_id"].dropna().tolist() if str(v) and str(v) != "<NA>"})
        prior_games_set = set(prior_games)
        prior_games_total = int(len(prior_games_set))
        rot_games_set = set(team_games["game_id"].astype("string").tolist())
        overlap_games_total = int(len(rot_games_set & prior_games_set))
        overlap_rate = float(overlap_games_total / rot_games_total) if rot_games_total else float("nan")

        if bool(restrict_to_prior_games):
            team_games = team_games[team_games["game_id"].isin(prior_games_set)].copy()
            team_games = team_games.sort_values(["season_id", "game_id", "team_id"], kind="mergesort").reset_index(drop=True)

    if limit_team_games <= 0:
        selected = team_games
    else:
        limit = min(int(limit_team_games), int(len(team_games)))
        if sample_mode == "first":
            selected = team_games.head(limit).copy()
        else:
            rng = np.random.default_rng(int(seed))
            perm = rng.permutation(len(team_games))[:limit]
            selected = team_games.iloc[perm].copy()
            selected = selected.sort_values(["season_id", "game_id", "team_id"], kind="mergesort").reset_index(drop=True)

    # Group views for fast lookup.
    labels_gb = labels.groupby(["game_id", "team_id"], sort=False)
    events_gb = events.groupby(["game_id", "team_id"], sort=False)

    if candidate_pool in {"prior_topn", "prior_threshold", "predictor_threshold"} and priors is None:
        raise ValueError(
            "--minutes-prior-parquet is required for candidate_pool prior_topn/prior_threshold/predictor_threshold"
        )

    if candidate_pool == "predictor_threshold":
        if rotation_predictor_bundle is None:
            raise ValueError("--rotation-predictor-bundle is required for candidate_pool predictor_threshold")
        if str(gate_feature_source).strip().lower() == "none":
            raise ValueError("--gate-feature-source cannot be 'none' for candidate_pool predictor_threshold")

    candidate_pool_params: dict[str, Any] = {}
    if candidate_pool in {"prior_topn", "prior_threshold"}:
        candidate_pool_params = {
            "candidate_top_n": int(candidate_top_n),
            "candidate_min_minutes_prior": float(candidate_min_minutes_prior),
            "candidate_min_play_prob": float(candidate_min_play_prob),
            "candidate_min_candidates": int(candidate_min_candidates),
        }
    elif candidate_pool == "predictor_threshold":
        candidate_pool_params = {
            "pool_max_size": int(pool_max_size),
            "t_ge15": float(pool_t_ge15),
            "t_ge5": float(pool_t_ge5),
            "always_include_starters": True,
            "always_include_top_n": int(pool_always_include_top_n),
            "rank_key": ["p_ge15_desc", "p_ge5_desc", "minutes_prior_desc", "player_id_asc"],
            "fail_open_fallback": "prior_topn_by_minutes_prior",
        }

    candidate_pool_by_team_game: dict[tuple[str, int], list[int]] = {}
    if candidate_pool in {"prior_topn", "prior_threshold"} and priors is not None and not priors.empty:
        top_n = int(candidate_top_n) if candidate_pool == "prior_topn" else 0
        pool_df = build_candidate_pool_prior(
            priors,
            top_n=top_n,
            min_minutes_prior=float(candidate_min_minutes_prior),
            min_play_prob=float(candidate_min_play_prob),
            min_candidates=int(candidate_min_candidates),
        )
        if not pool_df.empty:
            for (gid, tid), grp in pool_df.groupby(["game_id", "team_id"], sort=False):
                candidate_pool_by_team_game[(str(gid), int(tid))] = _unique_ints_sorted(grp["player_id"].tolist())

    gate_preds = None
    gate_bundle_dir: str | None = None
    gate_pred_source_counts: dict[str, int] | None = None
    gate_person_id_to_internal_id: dict[int, int] | None = None
    if bool(gate_cfg.enabled) and gate_feature_source != "none":
        bundle = load_rotation_predictor_bundle(Path(rotation_predictor_bundle))
        gate_bundle_dir = str(bundle.bundle_dir)

        game_allow = {canonicalize_game_id(v) for v in selected["game_id"].astype("string").tolist() if canonicalize_game_id(v)}
        team_allow = {int(v) for v in selected["team_id"].dropna().tolist()}

        years_raw = sorted({y for y in (season_start_year_from_game_id(g) for g in game_allow) if y is not None})
        # Only build personId->internal_id mapping for plausible modern seasons. For synthetic/unit-test
        # game_ids (e.g. "0000000001" -> year 2000), skip mapping and allow internal-id preds to pass through.
        years = [int(y) for y in years_raw if 2010 <= int(y) <= 2035]

        # Build a deterministic personId->internal_id mapping for the seasons present in this eval slice.
        gate_person_id_to_internal_id = {}
        for y in years:
            diag_dir = out_dir / "_gate_id_map" / f"season_start_year={int(y)}"
            res = build_person_id_to_internal_id_map(
                season_start_year=int(y),
                diagnostics_dir=diag_dir,
            )
            for pid, internal in res.person_id_to_internal_id.items():
                if pid in gate_person_id_to_internal_id and int(gate_person_id_to_internal_id[pid]) != int(internal):
                    raise ValueError(
                        f"personId mapping collision across seasons: nba_person_id={pid} -> {gate_person_id_to_internal_id[pid]} vs {internal}"
                    )
                gate_person_id_to_internal_id[int(pid)] = int(internal)

        if gate_feature_source == "cached_preds":
            gate_preds = load_cached_predictions(
                bundle,
                person_id_to_internal_id=gate_person_id_to_internal_id,
                game_id_allow=game_allow,
                team_id_allow=team_allow,
            )
        elif gate_feature_source == "cached_all":
            gate_preds = load_cached_all_predictions(
                bundle,
                person_id_to_internal_id=gate_person_id_to_internal_id,
                game_id_allow=game_allow,
                team_id_allow=team_allow,
            )
        elif gate_feature_source == "cached_train":
            gate_preds = load_cached_train_predictions(
                bundle,
                person_id_to_internal_id=gate_person_id_to_internal_id,
                game_id_allow=game_allow,
                team_id_allow=team_allow,
                max_rows=gate_max_train_rows,
            )

        if gate_preds is not None and not gate_preds.empty and "pred_source" in gate_preds.columns:
            gate_pred_source_counts = gate_preds["pred_source"].fillna("unknown").value_counts(dropna=False).to_dict()

    pool_preds = None
    pool_bundle_dir: str | None = None
    pool_person_id_to_internal_id: dict[int, int] | None = None
    pool_preds_team_games: set[tuple[str, int]] = set()
    pool_preds_gb = None
    missing_pred_team_games = 0
    missing_pred_player_rows = 0
    if candidate_pool == "predictor_threshold":
        # Reuse gate-loaded predictions when available; otherwise load once for pool selection.
        if gate_preds is not None and not gate_preds.empty and gate_bundle_dir is not None:
            pool_preds = gate_preds.copy()
            pool_bundle_dir = str(gate_bundle_dir)
            pool_person_id_to_internal_id = gate_person_id_to_internal_id
        else:
            bundle = load_rotation_predictor_bundle(Path(rotation_predictor_bundle))
            pool_bundle_dir = str(bundle.bundle_dir)

            game_allow = {canonicalize_game_id(v) for v in selected["game_id"].astype("string").tolist() if canonicalize_game_id(v)}
            team_allow = {int(v) for v in selected["team_id"].dropna().tolist()}

            years_raw = sorted({y for y in (season_start_year_from_game_id(g) for g in game_allow) if y is not None})
            years = [int(y) for y in years_raw if 2010 <= int(y) <= 2035]

            pool_person_id_to_internal_id = {}
            for y in years:
                diag_dir = out_dir / "_pool_id_map" / f"season_start_year={int(y)}"
                res = build_person_id_to_internal_id_map(
                    season_start_year=int(y),
                    diagnostics_dir=diag_dir,
                )
                for pid, internal in res.person_id_to_internal_id.items():
                    if (
                        pid in pool_person_id_to_internal_id
                        and int(pool_person_id_to_internal_id[pid]) != int(internal)
                    ):
                        raise ValueError(
                            f"personId mapping collision across seasons: nba_person_id={pid} -> {pool_person_id_to_internal_id[pid]} vs {internal}"
                        )
                    pool_person_id_to_internal_id[int(pid)] = int(internal)

            if gate_feature_source == "cached_preds":
                pool_preds = load_cached_predictions(
                    bundle,
                    person_id_to_internal_id=pool_person_id_to_internal_id,
                    game_id_allow=game_allow,
                    team_id_allow=team_allow,
                )
            elif gate_feature_source == "cached_all":
                pool_preds = load_cached_all_predictions(
                    bundle,
                    person_id_to_internal_id=pool_person_id_to_internal_id,
                    game_id_allow=game_allow,
                    team_id_allow=team_allow,
                )
            elif gate_feature_source == "cached_train":
                pool_preds = load_cached_train_predictions(
                    bundle,
                    person_id_to_internal_id=pool_person_id_to_internal_id,
                    game_id_allow=game_allow,
                    team_id_allow=team_allow,
                    max_rows=gate_max_train_rows,
                )

        if pool_preds is not None and not pool_preds.empty:
            pool_preds_team_games = set(
                zip(
                    pool_preds["game_id"].astype("string").tolist(),
                    pool_preds["team_id"].astype(int).tolist(),
                )
            )
            pool_preds_gb = pool_preds.groupby(["game_id", "team_id"], sort=False)

    gen = TemplateRotationGenerator(
        rot_bundle=rot_bundle_path,
        humility_config=humility_config,
        gate_config=gate_cfg,
        gate_preds=gate_preds,
    )
    generator_name = type(gen).__name__

    player_rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []
    candidate_pool_rows: list[dict[str, Any]] = []
    candidate_pool_team_game_rows: list[dict[str, Any]] = []

    roster_person_id_to_internal_id: dict[int, int] | None = None
    if candidate_pool == "roster":
        # Build a deterministic personId->internal_id mapping for the seasons present in this eval slice.
        # For synthetic/unit-test game_ids (e.g. "0000000001"), we skip mapping and rely on priors/events fallback.
        years_raw = sorted({y for y in (season_start_year_from_game_id(g) for g in selected["game_id"].tolist()) if y is not None})
        years = [int(y) for y in years_raw if 2010 <= int(y) <= 2035]
        if years:
            roster_person_id_to_internal_id = {}
            for y in years:
                diag_dir = out_dir / "_roster_id_map" / f"season_start_year={int(y)}"
                res = build_person_id_to_internal_id_map(
                    season_start_year=int(y),
                    diagnostics_dir=diag_dir,
                )
                for pid, internal in res.person_id_to_internal_id.items():
                    roster_person_id_to_internal_id[int(pid)] = int(internal)

    for r in selected.itertuples(index=False):
        game_id = str(r.game_id)
        team_id = int(r.team_id)
        season_id = str(r.season_id)
        opponent_team_id = int(r.opponent_team_id) if r.opponent_team_id is not None else -1
        is_home = bool(r.is_home)

        g_labels = labels_gb.get_group((game_id, team_id)) if (game_id, team_id) in labels_gb.groups else pd.DataFrame()
        g_events = events_gb.get_group((game_id, team_id)) if (game_id, team_id) in events_gb.groups else pd.DataFrame()
        if g_labels.empty:
            continue

        g_prior = (
            priors_gb.get_group((game_id, team_id)).copy()
            if priors_gb is not None and (game_id, team_id) in priors_gb.groups
            else None
        )

        if candidate_pool == "truth":
            cand_df = build_candidate_pool_truth(g_labels)
            candidate_ids = _unique_ints_sorted(cand_df["player_id"].tolist())
        elif candidate_pool in {"prior_topn", "prior_threshold"}:
            candidate_ids = candidate_pool_by_team_game.get((game_id, team_id), [])
        elif candidate_pool == "predictor_threshold":
            # Truth starters are allowed here (rot_eval already uses truth starters for starter selection).
            starters = []
            if "starter_actual" in g_labels.columns:
                starter_mask = _coerce_bool(g_labels["starter_actual"].fillna(False))
                starters = g_labels.loc[starter_mask, "player_id"].astype(int).tolist()

            g_prior_use = g_prior if g_prior is not None else pd.DataFrame()
            prior_sorted = pd.DataFrame()
            if g_prior_use is not None and not g_prior_use.empty:
                prior_sorted = g_prior_use.sort_values(["minutes_prior", "player_id"], ascending=[False, True], kind="mergesort")

            team_missing_pred_team_game = (game_id, team_id) not in pool_preds_team_games
            team_missing_pred_rows = 0
            if team_missing_pred_team_game:
                missing_pred_team_games += 1
                if not prior_sorted.empty:
                    team_missing_pred_rows = int(len(prior_sorted))
                    missing_pred_player_rows += int(team_missing_pred_rows)
                # Fail open to prior_topn-by-minutes (deterministic).
                cand_df = build_candidate_pool_prior_topn_team_game(g_prior_use, top_n=int(pool_max_size))
                candidate_ids = _unique_ints_sorted(cand_df["player_id"].tolist()) if not cand_df.empty else []
            else:
                g_pred = pool_preds_gb.get_group((game_id, team_id)).copy() if pool_preds_gb is not None else pd.DataFrame()
                if g_prior_use is not None and not g_prior_use.empty and not g_pred.empty:
                    prior_ids = set(int(x) for x in pd.to_numeric(g_prior_use["player_id"], errors="coerce").dropna().astype(int).tolist())
                    pred_ids = set(int(x) for x in pd.to_numeric(g_pred["player_id"], errors="coerce").dropna().astype(int).tolist())
                    team_missing_pred_rows = int(len(prior_ids - pred_ids))
                    missing_pred_player_rows += int(team_missing_pred_rows)

                cand_df = build_candidate_pool_predictor_threshold(
                    g_prior_use,
                    g_pred,
                    starters=starters,
                    pool_max_size=int(pool_max_size),
                    t_ge15=float(pool_t_ge15),
                    t_ge5=float(pool_t_ge5),
                    always_include_starters=True,
                    always_include_top_n=int(pool_always_include_top_n),
                )
                candidate_ids = _unique_ints_sorted(cand_df["player_id"].tolist()) if not cand_df.empty else []
        else:
            season_start_year = season_start_year_from_game_id(game_id) or 0
            cand_df = build_candidate_pool_roster(
                game_id,
                team_id,
                season_start_year,
                person_id_to_internal_id=roster_person_id_to_internal_id,
                priors_team_game=g_prior,
                events_team_game=g_events,
            )
            candidate_ids = _unique_ints_sorted(cand_df["player_id"].tolist())

        if not candidate_ids:
            continue

        if candidate_pool == "truth":
            starter_pool = g_labels.loc[g_labels["starter_actual"], ["player_id", "minutes_actual"]].copy()
            if not starter_pool.empty:
                starter_pool = starter_pool.sort_values(
                    ["minutes_actual", "player_id"],
                    ascending=[False, True],
                    kind="mergesort",
                )
                starter_candidates = [int(v) for v in starter_pool["player_id"].tolist()]
            else:
                starter_candidates = _infer_starters_from_events(g_events)
        else:
            inferred = _infer_starters_from_events(g_events)
            inferred_in_pool = [int(pid) for pid in inferred if int(pid) in set(candidate_ids)]
            starter_candidates = inferred_in_pool or [int(x) for x in candidate_ids[:5]]
            if len(starter_candidates) < 5 and g_prior is not None and not g_prior.empty:
                ranked = (
                    g_prior[g_prior["player_id"].isin(candidate_ids)]
                    .sort_values(["minutes_prior", "player_id"], ascending=[False, True], kind="mergesort")
                    .get("player_id", pd.Series([], dtype="int64"))
                    .tolist()
                )
                for pid in ranked:
                    if int(pid) in starter_candidates:
                        continue
                    starter_candidates.append(int(pid))
                    if len(starter_candidates) >= 5:
                        break

        minutes_prior: Optional[dict[int, float]] = None
        minutes_p10_prior: Optional[dict[int, float]] = None
        minutes_p90_prior: Optional[dict[int, float]] = None
        play_prob_prior: Optional[dict[int, float]] = None
        g_prior_cand = None
        if g_prior is not None and not g_prior.empty:
            g_prior_cand = g_prior[g_prior["player_id"].isin(candidate_ids)].copy() if candidate_ids else g_prior.copy()
            minutes_prior = {
                int(pid): float(v)
                for pid, v in zip(g_prior_cand["player_id"].tolist(), g_prior_cand["minutes_prior"].tolist())
            }
            if "minutes_p10" in g_prior_cand.columns:
                minutes_p10_prior = {
                    int(pid): float(v)
                    for pid, v in zip(g_prior_cand["player_id"].tolist(), g_prior_cand["minutes_p10"].tolist())
                }
            if "minutes_p90" in g_prior_cand.columns:
                minutes_p90_prior = {
                    int(pid): float(v)
                    for pid, v in zip(g_prior_cand["player_id"].tolist(), g_prior_cand["minutes_p90"].tolist())
                }
            play_prob_prior = {
                int(pid): float(v)
                for pid, v in zip(g_prior_cand["player_id"].tolist(), g_prior_cand["play_prob"].tolist())
            }
        elif use_truth_minutes_prior_for_mapping:
            minutes_prior = {
                int(pid): float(mins)
                for pid, mins in zip(g_labels["player_id"].tolist(), g_labels["minutes_actual"].tolist())
            }

        truth_played = set(
            g_labels.loc[g_labels["played_ge_1"].fillna(False).astype(bool), "player_id"].astype(int).tolist()
        )
        pool_set = set(int(x) for x in candidate_ids)
        overlap = pool_set & truth_played
        recall_played_ge1 = float(len(overlap) / len(truth_played)) if truth_played else float("nan")
        precision_played_ge1 = float(len(overlap) / len(pool_set)) if pool_set else float("nan")

        chaos_index = 0
        missing_prior_players = 0
        if g_prior_cand is not None and not g_prior_cand.empty:
            present = set(int(x) for x in g_prior_cand["player_id"].astype(int).tolist())
            missing_prior_players = int(len(pool_set - present))
            mp = pd.to_numeric(g_prior_cand["minutes_prior"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            pp = (
                pd.to_numeric(g_prior_cand["play_prob"], errors="coerce")
                .fillna(0.0)
                .clip(0.0, 1.0)
                .to_numpy(dtype=np.float64)
            )
            chaos_index = int(((mp == 0.0) & (pp >= 0.8)).sum())

        candidate_pool_team_game_rows.append(
            {
                "season_id": season_id,
                "game_id": game_id,
                "team_id": int(team_id),
                "pool_mode": str(candidate_pool),
                "pool_size": int(len(candidate_ids)),
                "recall_played_ge1": recall_played_ge1,
                "precision_played_ge1": precision_played_ge1,
                "chaos_index": int(chaos_index),
                "missing_prior_players": int(missing_prior_players),
                "missing_pred_team_game": bool(team_missing_pred_team_game) if candidate_pool == "predictor_threshold" else False,
                "missing_pred_player_rows": int(team_missing_pred_rows) if candidate_pool == "predictor_threshold" else 0,
            }
        )
        for pid in candidate_ids:
            candidate_pool_rows.append({"game_id": game_id, "team_id": int(team_id), "player_id": int(pid)})

        regime_label = None
        if "regime_label" in g_labels.columns:
            vals = [v for v in g_labels["regime_label"].dropna().tolist() if str(v) and str(v) != "<NA>"]
            if vals:
                regime_label = str(vals[0])

        ctx = TeamContext(
            season_id=season_id,
            game_id=game_id,
            team_id=team_id,
            opponent_team_id=opponent_team_id,
            is_home=is_home,
            candidate_player_ids=candidate_ids,
            starter_candidates=starter_candidates,
            minutes_prior=minutes_prior,
            minutes_p10_prior=minutes_p10_prior,
            minutes_p90_prior=minutes_p90_prior,
            play_prob_prior=play_prob_prior,
            regime_label=regime_label,
            n_worlds=int(n_worlds),
            rng_seed=_stable_team_game_seed(base_seed=int(seed), season_id=season_id, game_id=game_id, team_id=team_id),
        )

        worlds = gen.generate(ctx)
        diag = worlds.diagnostics or {}

        mapping_success = diag.get("mapping_success_rate", None)
        template_source = diag.get("template_source", None)
        fallback_to_prior_worlds = diag.get("fallback_to_prior_worlds", None)
        template_resamples_total = diag.get("template_resamples_total", None)
        humility_tier_counts = diag.get("humility_tier_counts", None)
        heur_applied_n = diag.get("rotation_prior_heuristics_applied_n", None)
        heur_applied_by_tier = diag.get("rotation_prior_heuristics_applied_by_tier", None)
        heur_stats = diag.get("rotation_prior_heuristics_stats", None)
        gate_tier_counts = diag.get("gate_tier_counts", None)
        gate_missing_preds_n = diag.get("gate_missing_preds_n", None)
        gate_excluded_n = diag.get("gate_excluded_n", None)
        gate_player_p_ge5 = diag.get("gate_player_p_ge5_pred", None) or {}
        gate_player_p_ge15 = diag.get("gate_player_p_ge15_pred", None) or {}
        gate_player_p_ge5_used = diag.get("gate_player_p_ge5_used", None) or {}
        gate_player_p_ge15_used = diag.get("gate_player_p_ge15_used", None) or {}
        gate_player_tier = diag.get("gate_player_tier", None) or {}
        gate_player_reason = diag.get("gate_player_reason", None) or {}
        gate_player_missing_pred = diag.get("gate_player_missing_pred", None) or {}
        gate_player_excluded = diag.get("gate_player_excluded", None) or {}
        gate_player_minutes_cap = diag.get("gate_player_minutes_cap", None) or {}
        gate_player_play_prob_cap = diag.get("gate_player_play_prob_cap", None) or {}
        gate_player_minutes_prior_adj = diag.get("gate_player_minutes_prior_adj", None) or {}
        gate_player_play_prob_adj = diag.get("gate_player_play_prob_adj", None) or {}

        # Build per-player rows (stable order).
        g_labels_idx = g_labels.set_index("player_id", drop=False)
        per_player_pred: dict[int, dict[str, float]] = {}
        for pid in candidate_ids:
            minutes = worlds.minutes_by_player.get(int(pid), np.zeros(int(n_worlds), dtype=np.float64))
            pred = _compute_player_summary(minutes)
            per_player_pred[int(pid)] = pred

            truth_row: Optional[pd.Series]
            if int(pid) not in g_labels_idx.index:
                truth_row = None
            else:
                row_obj = g_labels_idx.loc[int(pid)]
                truth_row = row_obj.iloc[0] if isinstance(row_obj, pd.DataFrame) else row_obj

            player_rows.append(
                {
                    "season_id": season_id,
                    "game_id": game_id,
                    "team_id": team_id,
                    "opponent_team_id": opponent_team_id,
                    "is_home": is_home,
                    "player_id": int(pid),
                    "minutes_actual": float(truth_row["minutes_actual"]) if truth_row is not None else 0.0,
                    "played_ge_1": bool(truth_row["played_ge_1"]) if truth_row is not None else False,
                    "played_ge_5": bool(truth_row["played_ge_5"]) if truth_row is not None else False,
                    "starter_actual": bool(truth_row["starter_actual"]) if truth_row is not None else False,
                    "regime_label": str(truth_row["regime_label"]) if truth_row is not None else (regime_label or "unknown"),
                    **pred,
                    "p_ge5_pred": float(gate_player_p_ge5.get(int(pid), np.nan)),
                    "p_ge15_pred": float(gate_player_p_ge15.get(int(pid), np.nan)),
                    "p_ge5_used": float(gate_player_p_ge5_used.get(int(pid), np.nan)),
                    "p_ge15_used": float(gate_player_p_ge15_used.get(int(pid), np.nan)),
                    "gate_tier": str(gate_player_tier.get(int(pid), "")) if int(pid) in gate_player_tier else "",
                    "gate_reason": str(gate_player_reason.get(int(pid), "")) if int(pid) in gate_player_reason else "",
                    "gate_missing_pred": bool(gate_player_missing_pred.get(int(pid), False)),
                    "gate_excluded": bool(gate_player_excluded.get(int(pid), False)),
                    "gate_minutes_cap": float(gate_player_minutes_cap.get(int(pid), np.nan)),
                    "gate_play_prob_cap": float(gate_player_play_prob_cap.get(int(pid), np.nan)),
                    "minutes_prior_adj": float(gate_player_minutes_prior_adj.get(int(pid), np.nan)),
                    "play_prob_adj": float(gate_player_play_prob_adj.get(int(pid), np.nan)),
                    "n_worlds": int(n_worlds),
                    "seed": int(seed),
                    "generator_name": generator_name,
                    "mapping_success": mapping_success,
                    "template_source": template_source,
                }
            )

        # Team-level aggregates.
        truth_minutes = (
            g_labels_idx.reindex(candidate_ids)["minutes_actual"].fillna(0.0).to_numpy(dtype=np.float64)
            if candidate_ids
            else np.array([], dtype=np.float64)
        )
        truth_rotation_count_ge5 = int((truth_minutes >= 5.0).sum()) if truth_minutes.size else 0
        truth_zero_to_five_count = int(((truth_minutes > 0.0) & (truth_minutes <= 5.0)).sum()) if truth_minutes.size else 0
        truth_total = float(truth_minutes.sum()) if truth_minutes.size else 0.0
        truth_top5_share = float(np.sort(truth_minutes)[-5:].sum() / truth_total) if truth_total > 0 else float("nan")

        # Expected counts derived from per-player probabilities.
        p_ge5 = [float(per_player_pred[int(pid)]["p_played_ge_5_pred"]) for pid in candidate_ids] if candidate_ids else []
        p_lt5 = [float(per_player_pred[int(pid)]["p_minutes_lt5_pred"]) for pid in candidate_ids] if candidate_ids else []
        minutes_mean = [float(per_player_pred[int(pid)]["minutes_mean"]) for pid in candidate_ids] if candidate_ids else []
        minutes_mean_arr = np.asarray(minutes_mean, dtype=np.float64)
        pred_total = float(minutes_mean_arr.sum()) if minutes_mean_arr.size else 0.0
        pred_top5_share = float(np.sort(minutes_mean_arr)[-5:].sum() / pred_total) if pred_total > 0 else float("nan")

        team_rows.append(
            {
                "season_id": season_id,
                "game_id": game_id,
                "team_id": team_id,
                "truth_rotation_count_ge5": truth_rotation_count_ge5,
                "pred_rotation_count_ge5_mean": float(np.sum(p_ge5)) if p_ge5 else 0.0,
                "truth_zero_to_five_count": truth_zero_to_five_count,
                "pred_p_minutes_lt5_sum": float(np.sum(p_lt5)) if p_lt5 else 0.0,
                "top5_minutes_share_truth": truth_top5_share,
                "top5_minutes_share_pred_mean": pred_top5_share,
                "mapping_success_rate": mapping_success,
                "template_fallback_rate": (float(fallback_to_prior_worlds) / float(n_worlds))
                if fallback_to_prior_worlds is not None and int(n_worlds) > 0
                else None,
                "template_source": template_source,
                "template_resamples_total": template_resamples_total,
                "fallback_to_prior_worlds": fallback_to_prior_worlds,
                "humility_tier_counts": humility_tier_counts,
                "rotation_prior_heuristics_applied_n": heur_applied_n,
                "rotation_prior_heuristics_applied_by_tier": heur_applied_by_tier,
                "rotation_prior_heuristics_stats": heur_stats,
                "gate_tier_counts": gate_tier_counts,
                "gate_missing_preds_n": gate_missing_preds_n,
                "gate_excluded_n": gate_excluded_n,
                "n_worlds": int(n_worlds),
                "seed": int(seed),
                "generator_name": generator_name,
            }
        )

    player_eval = pd.DataFrame(player_rows)
    team_eval = pd.DataFrame(team_rows)

    # Stable output ordering.
    if not player_eval.empty:
        player_eval = player_eval.sort_values(["season_id", "game_id", "team_id", "player_id"], kind="mergesort").reset_index(drop=True)
    if not team_eval.empty:
        team_eval = team_eval.sort_values(["season_id", "game_id", "team_id"], kind="mergesort").reset_index(drop=True)

    player_eval_path = out_dir / "player_eval.parquet"
    team_eval_path = out_dir / "team_eval.parquet"
    _atomic_write_parquet(player_eval, player_eval_path)
    _atomic_write_parquet(team_eval, team_eval_path)

    calib_ge1, brier_ge1 = _compute_calibration(player_eval=player_eval, p_col="p_played_ge_1_pred", y_col="played_ge_1")
    calib_ge5, brier_ge5 = _compute_calibration(player_eval=player_eval, p_col="p_played_ge_5_pred", y_col="played_ge_5")
    calib_ge1_path = out_dir / "calibration_played_ge_1.parquet"
    calib_ge5_path = out_dir / "calibration_played_ge_5.parquet"
    _atomic_write_parquet(calib_ge1, calib_ge1_path)
    _atomic_write_parquet(calib_ge5, calib_ge5_path)

    minutes_mae = float(
        (player_eval["minutes_mean"] - player_eval["minutes_actual"]).abs().mean()
    ) if not player_eval.empty else float("nan")

    metrics = EvalMetrics(
        brier_ge1=float(brier_ge1),
        brier_ge5=float(brier_ge5),
        minutes_mae=minutes_mae,
        n_team_games=int(len(team_eval)),
        n_players=int(len(player_eval)),
    )

    # Artifacts: hashes + manifest + report.
    input_hashes_path = out_dir / "input_hashes.json"
    write_rot_eval_input_hashes(
        rot_bundle_dir=rot_bundle_dir,
        minutes_prior_parquet=minutes_prior_parquet,
        out_path=input_hashes_path,
    )

    repo_root = Path(__file__).resolve().parents[2]

    candidate_pool_team_games_df = pd.DataFrame(candidate_pool_team_game_rows)
    if not candidate_pool_team_games_df.empty:
        candidate_pool_team_games_df = candidate_pool_team_games_df.sort_values(
            ["season_id", "game_id", "team_id"], kind="mergesort"
        ).reset_index(drop=True)
    candidate_pool_team_games_path = out_dir / "candidate_pool_team_games.parquet"
    _atomic_write_parquet(candidate_pool_team_games_df, candidate_pool_team_games_path)

    candidate_pool_summary: dict[str, Any] = {
        "pool_mode": str(candidate_pool),
        "pool_params": candidate_pool_params,
        "team_games": int(len(candidate_pool_team_games_df)),
        "pool_size_stats": {},
        "overlap_stats": {},
        "missing_pred_team_games": int(missing_pred_team_games),
        "missing_pred_player_rows": int(missing_pred_player_rows),
    }
    if not candidate_pool_team_games_df.empty:
        sizes = pd.to_numeric(candidate_pool_team_games_df["pool_size"], errors="coerce").dropna().astype(float)
        recall = pd.to_numeric(candidate_pool_team_games_df["recall_played_ge1"], errors="coerce").dropna().astype(float)
        precision = pd.to_numeric(
            candidate_pool_team_games_df["precision_played_ge1"], errors="coerce"
        ).dropna().astype(float)
        candidate_pool_summary["pool_size_stats"] = {
            "mean": float(sizes.mean()) if len(sizes) else float("nan"),
            "p10": float(sizes.quantile(0.1)) if len(sizes) else float("nan"),
            "p50": float(sizes.quantile(0.5)) if len(sizes) else float("nan"),
            "p90": float(sizes.quantile(0.9)) if len(sizes) else float("nan"),
        }
        candidate_pool_summary["overlap_stats"] = {
            "recall_played_ge1_mean": float(recall.mean()) if len(recall) else float("nan"),
            "precision_mean": float(precision.mean()) if len(precision) else float("nan"),
        }

    candidate_pool_summary_path = out_dir / "candidate_pool_summary.json"
    write_json(candidate_pool_summary_path, candidate_pool_summary)

    manifest = build_rot_eval_manifest(
        repo_root=repo_root,
        rot_bundle_path=rot_bundle_path,
        rot_bundle_dir=rot_bundle_dir,
        run_id=run_id,
        n_worlds=n_worlds,
        seed=seed,
        limit_team_games=limit_team_games,
        sample_mode=sample_mode,
        use_truth_minutes_prior=use_truth_minutes_prior_for_mapping,
        minutes_prior_parquet=minutes_prior_parquet,
        restrict_to_prior_games=bool(restrict_to_prior_games),
        candidate_pool=str(candidate_pool),
        candidate_pool_params=candidate_pool_params,
        humility_enabled=bool((humility_config or HumilityConfig()).enabled),
        humility_config=humility_config_as_dict(humility_config or HumilityConfig()),
        gate_enabled=bool(gate_cfg.enabled),
        gate_config=gate_config_as_dict(gate_cfg),
        rotation_predictor_bundle=str(gate_bundle_dir) if gate_bundle_dir is not None else (pool_bundle_dir if pool_bundle_dir is not None else (str(rotation_predictor_bundle) if rotation_predictor_bundle is not None else None)),
        gate_feature_source=str(gate_feature_source),
        gate_max_train_rows=int(gate_max_train_rows) if gate_max_train_rows is not None else None,
        input_hashes_path=input_hashes_path,
    )
    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, manifest)

    if bool(gate_cfg.enabled):
        gate_tier_counts_total = (
            player_eval["gate_tier"].fillna("unknown").value_counts(dropna=False).to_dict()
            if not player_eval.empty and "gate_tier" in player_eval.columns
            else {}
        )
        gate_missing_pred_total = (
            int(_coerce_bool(player_eval["gate_missing_pred"]).sum())
            if not player_eval.empty and "gate_missing_pred" in player_eval.columns
            else 0
        )
        gate_excluded_total = (
            int(_coerce_bool(player_eval["gate_excluded"]).sum())
            if not player_eval.empty and "gate_excluded" in player_eval.columns
            else 0
        )
        gate_summary = {
            "gate_enabled": bool(gate_cfg.enabled),
            "gate_config": gate_config_as_dict(gate_cfg),
            "rotation_predictor_bundle": str(rotation_predictor_bundle) if rotation_predictor_bundle is not None else None,
            "rotation_predictor_bundle_dir": gate_bundle_dir,
            "gate_feature_source": str(gate_feature_source),
            "gate_max_train_rows": int(gate_max_train_rows) if gate_max_train_rows is not None else None,
            "person_id_map_size": int(len(gate_person_id_to_internal_id or {})),
            "pred_source_counts": gate_pred_source_counts,
            "pred_rows": int(len(gate_preds)) if gate_preds is not None else 0,
            "pred_team_games": int(gate_preds[["game_id", "team_id"]].drop_duplicates().shape[0])
            if gate_preds is not None and not gate_preds.empty
            else 0,
            "gate_tier_counts_total": gate_tier_counts_total,
            "gate_missing_pred_total": gate_missing_pred_total,
            "gate_excluded_total": gate_excluded_total,
        }
        write_json(out_dir / "gate_summary.json", gate_summary)

    # Build report.md (human-readable).
    report_path = out_dir / "report.md"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    avg_p_zero = float(player_eval["p_minutes_eq0_pred"].mean()) if not player_eval.empty else float("nan")
    avg_p_lt5 = float(player_eval["p_minutes_lt5_pred"].mean()) if not player_eval.empty else float("nan")

    template_sources = (
        team_eval["template_source"].fillna("unknown").value_counts(dropna=False).to_dict()
        if not team_eval.empty and "template_source" in team_eval.columns
        else {}
    )
    gate_tier_counts_total: dict[str, int] = {}
    gate_missing_pred_total = 0
    gate_excluded_total = 0
    if not player_eval.empty and "gate_tier" in player_eval.columns:
        gate_tier_counts_total = (
            player_eval["gate_tier"].fillna("unknown").value_counts(dropna=False).to_dict()
        )
        if "gate_missing_pred" in player_eval.columns:
            gate_missing_pred_total = int(_coerce_bool(player_eval["gate_missing_pred"]).sum())
        if "gate_excluded" in player_eval.columns:
            gate_excluded_total = int(_coerce_bool(player_eval["gate_excluded"]).sum())

    humility_tier_counts_total: dict[str, int] = {}
    heur_applied_by_tier_total: dict[str, int] = {}
    heur_applied_players_total = 0
    heur_applied_team_games = 0
    if not team_eval.empty and "humility_tier_counts" in team_eval.columns:
        for v in team_eval["humility_tier_counts"].dropna().tolist():
            if not isinstance(v, dict):
                continue
            for k, cnt in v.items():
                try:
                    humility_tier_counts_total[str(k)] = int(humility_tier_counts_total.get(str(k), 0)) + int(cnt)
                except Exception:
                    continue
    if not team_eval.empty and "rotation_prior_heuristics_applied_by_tier" in team_eval.columns:
        for v in team_eval["rotation_prior_heuristics_applied_by_tier"].dropna().tolist():
            if not isinstance(v, dict):
                continue
            for k, cnt in v.items():
                try:
                    heur_applied_by_tier_total[str(k)] = int(heur_applied_by_tier_total.get(str(k), 0)) + int(cnt)
                except Exception:
                    continue
    if not team_eval.empty and "rotation_prior_heuristics_applied_n" in team_eval.columns:
        nums = pd.to_numeric(team_eval["rotation_prior_heuristics_applied_n"], errors="coerce").fillna(0).astype(int)
        heur_applied_players_total = int(nums.sum())
        heur_applied_team_games = int((nums > 0).sum())
    mapping_success_avg = (
        float(pd.to_numeric(team_eval["mapping_success_rate"], errors="coerce").mean())
        if not team_eval.empty and "mapping_success_rate" in team_eval.columns
        else float("nan")
    )
    fallback_rate_avg = (
        float(pd.to_numeric(team_eval["template_fallback_rate"], errors="coerce").mean())
        if not team_eval.empty and "template_fallback_rate" in team_eval.columns
        else float("nan")
    )

    # Catastrophic promotion diagnostics (generator output, not gate predictions):
    # - truth<=5 but pred_mean>=15 (or >=20)
    # - truth<=1 but pred_mean>=10
    catastrophic_truth_le5_pred_ge15 = 0
    catastrophic_truth_le5_pred_ge20 = 0
    catastrophic_truth_le1_pred_ge10 = 0
    brier_ge5_starters = float("nan")
    if not player_eval.empty:
        m_truth = pd.to_numeric(player_eval["minutes_actual"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        m_pred = pd.to_numeric(player_eval["minutes_mean"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        catastrophic_truth_le5_pred_ge15 = int(((m_truth <= 5.0) & (m_pred >= 15.0)).sum())
        catastrophic_truth_le5_pred_ge20 = int(((m_truth <= 5.0) & (m_pred >= 20.0)).sum())
        catastrophic_truth_le1_pred_ge10 = int(((m_truth <= 1.0) & (m_pred >= 10.0)).sum())
        starters_slice = player_eval[player_eval["starter_actual"].fillna(False).astype(bool)].copy()
        _, brier_ge5_starters = _compute_calibration(
            player_eval=starters_slice,
            p_col="p_played_ge_5_pred",
            y_col="played_ge_5",
        )

    baseline_summary: str = ""
    prevented_lines: list[str] = []
    base_player_eval = pd.DataFrame()
    base_candidate_pool_team_games_df = pd.DataFrame()
    base_manifest: dict[str, Any] = {}
    if baseline_out_dir is not None:
        base_dir = Path(baseline_out_dir)
        base_player_path = base_dir / "player_eval.parquet"
        if not base_player_path.exists():
            raise FileNotFoundError(f"--baseline-out-dir missing player_eval.parquet: {base_player_path}")

        base_player_eval = pd.read_parquet(base_player_path)
        if not base_player_eval.empty:
            base_player_eval = base_player_eval.copy()
            base_player_eval["season_id"] = base_player_eval["season_id"].astype("string")
            base_player_eval["game_id"] = base_player_eval["game_id"].astype("string")
            base_player_eval["team_id"] = (
                pd.to_numeric(base_player_eval["team_id"], errors="coerce").astype("Int64").fillna(-1).astype(int)
            )
            base_player_eval["player_id"] = (
                pd.to_numeric(base_player_eval["player_id"], errors="coerce").astype("Int64").fillna(-1).astype(int)
            )

        base_manifest_path = base_dir / "manifest.json"
        if base_manifest_path.exists():
            try:
                base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
            except Exception:
                base_manifest = {}

        base_cand_team_games_path = base_dir / "candidate_pool_team_games.parquet"
        if base_cand_team_games_path.exists():
            try:
                base_candidate_pool_team_games_df = pd.read_parquet(base_cand_team_games_path)
            except Exception:
                base_candidate_pool_team_games_df = pd.DataFrame()

        _, base_brier_ge1 = _compute_calibration(player_eval=base_player_eval, p_col="p_played_ge_1_pred", y_col="played_ge_1")
        _, base_brier_ge5 = _compute_calibration(player_eval=base_player_eval, p_col="p_played_ge_5_pred", y_col="played_ge_5")
        base_minutes_mae = float(
            (base_player_eval["minutes_mean"] - base_player_eval["minutes_actual"]).abs().mean()
        ) if not base_player_eval.empty else float("nan")

        base_brier_ge5_starters = float("nan")
        if not base_player_eval.empty and "starter_actual" in base_player_eval.columns:
            base_starters_slice = base_player_eval[base_player_eval["starter_actual"].fillna(False).astype(bool)].copy()
            _, base_brier_ge5_starters = _compute_calibration(
                player_eval=base_starters_slice,
                p_col="p_played_ge_5_pred",
                y_col="played_ge_5",
            )

        base_cat_truth_le5_pred_ge15 = 0
        base_cat_truth_le5_pred_ge20 = 0
        base_cat_truth_le1_pred_ge10 = 0
        if not base_player_eval.empty:
            m_truth_b = pd.to_numeric(base_player_eval["minutes_actual"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            m_pred_b = pd.to_numeric(base_player_eval["minutes_mean"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            base_cat_truth_le5_pred_ge15 = int(((m_truth_b <= 5.0) & (m_pred_b >= 15.0)).sum())
            base_cat_truth_le5_pred_ge20 = int(((m_truth_b <= 5.0) & (m_pred_b >= 20.0)).sum())
            base_cat_truth_le1_pred_ge10 = int(((m_truth_b <= 1.0) & (m_pred_b >= 10.0)).sum())

        baseline_summary = textwrap.dedent(
            f"""\
            ## Baseline comparison

            baseline_out_dir: {base_dir}

            - brier_played_ge_1: baseline={float(base_brier_ge1):.6f} current={metrics.brier_ge1:.6f} delta={(metrics.brier_ge1 - float(base_brier_ge1)):.6f}
            - brier_played_ge_5: baseline={float(base_brier_ge5):.6f} current={metrics.brier_ge5:.6f} delta={(metrics.brier_ge5 - float(base_brier_ge5)):.6f}
            - minutes_mae: baseline={base_minutes_mae:.3f} current={metrics.minutes_mae:.3f} delta={(metrics.minutes_mae - base_minutes_mae):.3f}
            - brier_played_ge_5 (starters): baseline={float(base_brier_ge5_starters):.6f} current={float(brier_ge5_starters):.6f} delta={(float(brier_ge5_starters) - float(base_brier_ge5_starters)):.6f}

            - catastrophic (truth<=5 & pred_mean>=15): baseline={base_cat_truth_le5_pred_ge15} current={catastrophic_truth_le5_pred_ge15} delta={catastrophic_truth_le5_pred_ge15 - base_cat_truth_le5_pred_ge15}
            - catastrophic (truth<=5 & pred_mean>=20): baseline={base_cat_truth_le5_pred_ge20} current={catastrophic_truth_le5_pred_ge20} delta={catastrophic_truth_le5_pred_ge20 - base_cat_truth_le5_pred_ge20}
            - catastrophic (truth<=1 & pred_mean>=10): baseline={base_cat_truth_le1_pred_ge10} current={catastrophic_truth_le1_pred_ge10} delta={catastrophic_truth_le1_pred_ge10 - base_cat_truth_le1_pred_ge10}
            """
        )

        if not base_player_eval.empty and not player_eval.empty:
            keys = ["season_id", "game_id", "team_id", "player_id"]
            merged = base_player_eval[keys + ["minutes_actual", "minutes_mean"]].merge(
                player_eval[keys + ["minutes_mean"]],
                on=keys,
                how="inner",
                suffixes=("_baseline", "_current"),
            )
            prevented = merged[
                (pd.to_numeric(merged["minutes_actual"], errors="coerce").fillna(0.0) <= 5.0)
                & (pd.to_numeric(merged["minutes_mean_baseline"], errors="coerce").fillna(0.0) >= 15.0)
                & (pd.to_numeric(merged["minutes_mean_current"], errors="coerce").fillna(0.0) < 15.0)
            ].copy()
            if not prevented.empty:
                prevented = prevented.sort_values(["minutes_mean_baseline", "team_id", "player_id"], ascending=[False, True, True], kind="mergesort").head(20)
                for row in prevented.itertuples(index=False):
                    prevented_lines.append(
                        f"- {row.season_id} {row.game_id} team={int(row.team_id)} player={int(row.player_id)} "
                        f"truth={float(row.minutes_actual):.1f} baseline={float(row.minutes_mean_baseline):.1f} current={float(row.minutes_mean_current):.1f}"
                    )

        baseline_summary += "\n## Top prevented catastrophes (baseline -> current, top 20)\n\n"
        if prevented_lines:
            baseline_summary += "\n".join(prevented_lines) + "\n"
        else:
            baseline_summary += "- (no rows)\n"

    def _pool_size_stats(team_games_df: pd.DataFrame) -> dict[str, float]:
        if team_games_df.empty or "pool_size" not in team_games_df.columns:
            return {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
        s = pd.to_numeric(team_games_df["pool_size"], errors="coerce").dropna().astype(float)
        if s.empty:
            return {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
        return {"mean": float(s.mean()), "p10": float(s.quantile(0.1)), "p50": float(s.quantile(0.5)), "p90": float(s.quantile(0.9))}

    def _overlap_stats(team_games_df: pd.DataFrame) -> dict[str, float]:
        if team_games_df.empty:
            return {"recall_played_ge1_mean": float("nan"), "precision_mean": float("nan")}
        r = pd.to_numeric(team_games_df.get("recall_played_ge1"), errors="coerce").dropna().astype(float)
        p = pd.to_numeric(team_games_df.get("precision_played_ge1"), errors="coerce").dropna().astype(float)
        return {
            "recall_played_ge1_mean": float(r.mean()) if not r.empty else float("nan"),
            "precision_mean": float(p.mean()) if not p.empty else float("nan"),
        }

    def _cat_miss_counts(player_df: pd.DataFrame) -> dict[str, int]:
        if player_df.empty:
            return {
                "cat_10": 0,
                "cat_15": 0,
                "cat_20": 0,
                "cat_25": 0,
                "miss_5": 0,
                "miss_8": 0,
                "miss_10": 0,
            }
        m_truth = pd.to_numeric(player_df["minutes_actual"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        m_pred = pd.to_numeric(player_df["minutes_mean"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        cats = {
            "cat_10": int(((m_truth <= 5.0) & (m_pred >= 10.0)).sum()),
            "cat_15": int(((m_truth <= 5.0) & (m_pred >= 15.0)).sum()),
            "cat_20": int(((m_truth <= 5.0) & (m_pred >= 20.0)).sum()),
            "cat_25": int(((m_truth <= 5.0) & (m_pred >= 25.0)).sum()),
        }
        misses = {
            "miss_5": int(((m_truth >= 15.0) & (m_pred <= 5.0)).sum()),
            "miss_8": int(((m_truth >= 15.0) & (m_pred <= 8.0)).sum()),
            "miss_10": int(((m_truth >= 15.0) & (m_pred <= 10.0)).sum()),
        }
        return {**cats, **misses}

    def _chaos_terciles(team_games_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
        if team_games_df.empty or "chaos_index" not in team_games_df.columns:
            return team_games_df.copy(), {"q33": float("nan"), "q66": float("nan")}
        out = team_games_df.copy()
        chaos = pd.to_numeric(out["chaos_index"], errors="coerce").fillna(0).astype(int)
        q33 = float(chaos.quantile(1 / 3))
        q66 = float(chaos.quantile(2 / 3))

        def _bin(v: int) -> str:
            if v <= q33:
                return "low"
            if v <= q66:
                return "med"
            return "high"

        out["chaos_bin"] = chaos.map(_bin).astype("string")
        return out, {"q33": q33, "q66": q66}

    def _counts_by_chaos(player_df: pd.DataFrame, team_games_df: pd.DataFrame) -> tuple[dict[str, dict[str, int]], dict[str, int], dict[str, float]]:
        tg, cutoffs = _chaos_terciles(team_games_df)
        if tg.empty or "chaos_bin" not in tg.columns:
            return {}, {}, cutoffs
        keys = tg[["game_id", "team_id", "chaos_bin"]].copy()
        keys["game_id"] = keys["game_id"].astype("string")
        keys["team_id"] = pd.to_numeric(keys["team_id"], errors="coerce").astype("Int64").fillna(-1).astype(int)
        merged = player_df.merge(keys, on=["game_id", "team_id"], how="left")
        out: dict[str, dict[str, int]] = {}
        team_games_counts = tg["chaos_bin"].value_counts(dropna=False).to_dict()
        for bin_name in ["low", "med", "high"]:
            df_bin = merged[merged["chaos_bin"] == bin_name]
            out[bin_name] = _cat_miss_counts(df_bin)
        return out, {str(k): int(v) for k, v in team_games_counts.items()}, cutoffs

    def _pool_report_block(
        *,
        label: str,
        pool_mode: str,
        gate_enabled_flag: bool,
        player_df: pd.DataFrame,
        team_games_df: pd.DataFrame,
    ) -> str:
        ps = _pool_size_stats(team_games_df)
        ov = _overlap_stats(team_games_df)
        counts = _cat_miss_counts(player_df)
        by_chaos, chaos_team_games_counts, cutoffs = _counts_by_chaos(player_df, team_games_df)
        lines = []
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- pool_mode: {pool_mode}")
        lines.append(f"- gate_enabled: {bool(gate_enabled_flag)}")
        lines.append(f"- pool_size mean/p10/p50/p90: {ps['mean']:.2f} / {ps['p10']:.0f} / {ps['p50']:.0f} / {ps['p90']:.0f}")
        if "missing_pred_team_game" in team_games_df.columns:
            mp_tg = int(_coerce_bool(team_games_df["missing_pred_team_game"]).sum())
            mp_rows = int(pd.to_numeric(team_games_df.get("missing_pred_player_rows"), errors="coerce").fillna(0).sum())
            if mp_tg or mp_rows:
                lines.append(f"- missing predictor coverage: team_games={mp_tg} player_rows={mp_rows} (fail-open to prior_topn)")
        lines.append(
            f"- overlap (post-hoc): recall_played_ge1_mean={ov['recall_played_ge1_mean']:.3f} precision_mean={ov['precision_mean']:.3f}"
        )
        lines.append(
            "- catastrophic promotions (truth<=5 & pred_mean>=X): "
            f"cat_10={counts['cat_10']} cat_15={counts['cat_15']} cat_20={counts['cat_20']} cat_25={counts['cat_25']}"
        )
        lines.append(
            "- missed promotions (truth>=15 & pred_mean<=X): "
            f"miss_5={counts['miss_5']} miss_8={counts['miss_8']} miss_10={counts['miss_10']}"
        )
        lines.append("")
        lines.append(
            f"Chaos proxy: chaos_index := count(candidates with minutes_prior==0 & play_prob>=0.8). "
            f"Terciles cutoffs: q33={cutoffs['q33']:.1f} q66={cutoffs['q66']:.1f}"
        )
        lines.append("")
        lines.append("| chaos_bin | team_games | cat_15 | cat_20 | cat_25 | miss_5 | miss_8 | miss_10 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for bin_name in ["low", "med", "high"]:
            c = by_chaos.get(bin_name, _cat_miss_counts(pd.DataFrame()))
            n_tg = int(chaos_team_games_counts.get(bin_name, 0))
            lines.append(
                f"| {bin_name} | {n_tg} | {c['cat_15']} | {c['cat_20']} | {c['cat_25']} | {c['miss_5']} | {c['miss_8']} | {c['miss_10']} |"
            )
        lines.append("")
        return "\n".join(lines)

    candidate_pool_report_section = "## Candidate pool realism + gate impact\n\n"
    candidate_pool_report_section += _pool_report_block(
        label="current",
        pool_mode=str(candidate_pool),
        gate_enabled_flag=bool(gate_cfg.enabled),
        player_df=player_eval,
        team_games_df=candidate_pool_team_games_df,
    )
    candidate_pool_report_section += "\n"
    if baseline_out_dir is None:
        candidate_pool_report_section += (
            "_Tip: pass `--baseline-out-dir` to compare pool modes and/or compute a clean gate delta table "
            "(e.g. baseline=no-gate, current=gate; or baseline=truth, current=prior_topn)._"
        )
        candidate_pool_report_section += "\n\n"

    if baseline_out_dir is not None and not base_player_eval.empty:
        base_pool_mode = str(base_manifest.get("candidate_pool", "unknown"))
        base_gate_enabled = bool(base_manifest.get("gate_enabled", False))
        candidate_pool_report_section += _pool_report_block(
            label="baseline",
            pool_mode=base_pool_mode,
            gate_enabled_flag=base_gate_enabled,
            player_df=base_player_eval,
            team_games_df=base_candidate_pool_team_games_df,
        )
        candidate_pool_report_section += "\n"

        # Gate delta: interpret baseline as "no gate" and current as "with gate" when applicable.
        cur_counts = _cat_miss_counts(player_eval)
        base_counts = _cat_miss_counts(base_player_eval)
        delta = {k: int(cur_counts.get(k, 0) - base_counts.get(k, 0)) for k in sorted(cur_counts.keys())}
        verdict = "n/a"
        if (not base_gate_enabled) and bool(gate_cfg.enabled) and base_pool_mode == str(candidate_pool):
            cats_delta = delta["cat_15"] + delta["cat_20"] + delta["cat_25"]
            verdict = "YES" if (cats_delta < 0 and delta["miss_5"] <= 0) else "NO"

        candidate_pool_report_section += "### Gate delta (current - baseline)\n\n"
        candidate_pool_report_section += f"- baseline_gate_enabled: {base_gate_enabled}\n"
        candidate_pool_report_section += f"- current_gate_enabled: {bool(gate_cfg.enabled)}\n"
        candidate_pool_report_section += f"- same_pool_mode: {base_pool_mode == str(candidate_pool)}\n"
        candidate_pool_report_section += (
            "- verdict_rule: gate helps iff (delta_cat_15+delta_cat_20+delta_cat_25) < 0 AND delta_miss_5 <= 0 (same pool_mode)\n"
        )
        candidate_pool_report_section += f"- verdict_gate_helps: {verdict}\n\n"
        candidate_pool_report_section += "| metric | baseline | current | delta |\n"
        candidate_pool_report_section += "|---|---:|---:|---:|\n"
        for k in ["cat_10", "cat_15", "cat_20", "cat_25", "miss_5", "miss_8", "miss_10"]:
            candidate_pool_report_section += (
                f"| {k} | {base_counts.get(k, 0)} | {cur_counts.get(k, 0)} | {delta.get(k, 0)} |\n"
            )
        candidate_pool_report_section += "\n"

    worst_minutes = pd.DataFrame()
    if not player_eval.empty:
        worst_minutes = player_eval.assign(abs_err=(player_eval["minutes_mean"] - player_eval["minutes_actual"]).abs())
        worst_minutes = worst_minutes.sort_values("abs_err", ascending=False, kind="mergesort").head(25)

    worst_calib_ge5 = pd.DataFrame()
    if not calib_ge5.empty:
        worst_calib_ge5 = calib_ge5.assign(gap=(calib_ge5["p_pred_mean"] - calib_ge5["y_true_mean"]).abs())
        worst_calib_ge5 = worst_calib_ge5.sort_values("gap", ascending=False, kind="mergesort").head(10)

    report = textwrap.dedent(
        f"""\
        # rot_eval_v1 report

        Generated: {now}

        ## What this evaluates

        This is a “generator realism” backtest for `TemplateRotationGenerator`:
        - candidate_pool: {candidate_pool}
        - candidate_pool_params: {candidate_pool_params}
        - candidate_pool_summary: {candidate_pool_summary}
        - starters source: truth starters (truth mode) else first-segment lineup proxy
        - minutes_prior mapping stabilizer: truth minutes only when candidate_pool=truth and minutes_prior_parquet is None (`use_truth_minutes_prior={use_truth_minutes_prior_for_mapping}`)
        - minutes_prior_parquet: {str(minutes_prior_parquet) if minutes_prior_parquet is not None else None}

        Notes:
        - Non-truth pools do **not** use minutes_actual/played flags for membership (no leakage).
        - Roster pool uses roster_nightly + personId->internal_id mapping; when roster is missing it falls back to priors and first-segment lineup ids.

        ## Prior humility layer

        - humility_enabled: {bool((humility_config or HumilityConfig()).enabled)}
        - humility_config: {humility_config_as_dict(humility_config or HumilityConfig())}
        - heuristics_applied_team_games: {heur_applied_team_games}
        - heuristics_applied_players_total: {heur_applied_players_total}
        - heuristics_applied_by_tier_total: {heur_applied_by_tier_total}
        - humility_tier_counts_total: {humility_tier_counts_total}

        ## Rotation gate layer

        - gate_enabled: {bool(gate_cfg.enabled)}
        - gate_config: {gate_config_as_dict(gate_cfg)}
        - gate_feature_source: {gate_feature_source}
        - gate_missing_pred_behavior: fail_open_noop (no caps/exclusions when p_ge5/p_ge15 is missing)
        - rotation_predictor_bundle_dir: {gate_bundle_dir}
        - gate_pred_source_counts: {gate_pred_source_counts}
        - gate_tier_counts_total: {gate_tier_counts_total}
        - gate_missing_pred_total: {gate_missing_pred_total}
        - gate_excluded_total: {gate_excluded_total}

        ## Headline metrics

        - rot_games_total: {rot_games_total}
        - prior_games_total: {prior_games_total}
        - overlap_games_total: {overlap_games_total}
        - overlap_rate: {overlap_rate:.3f}
        - prior_players_total: {prior_players_total}
        - overlap_players_total: {overlap_players_total}
        - prior_coverage_rate: {prior_coverage_rate:.3f}
        - evaluated_team_games: {metrics.n_team_games}
        - team_games: {metrics.n_team_games}
        - players: {metrics.n_players}
        - brier_played_ge_1: {metrics.brier_ge1:.6f}
        - brier_played_ge_5: {metrics.brier_ge5:.6f}
        - brier_played_ge_5 (starters): {float(brier_ge5_starters):.6f}
        - minutes_mae: {metrics.minutes_mae:.3f}

        ## Tail / mass metrics

        - avg P(minutes==0): {avg_p_zero:.4f}
        - avg P(minutes<5): {avg_p_lt5:.4f}

        ## Catastrophic promotions

        - catastrophic (truth<=5 & pred_mean>=15): {catastrophic_truth_le5_pred_ge15}
        - catastrophic (truth<=5 & pred_mean>=20): {catastrophic_truth_le5_pred_ge20}
        - catastrophic (truth<=1 & pred_mean>=10): {catastrophic_truth_le1_pred_ge10}

        {candidate_pool_report_section}

        ## Mapping diagnostics

        - avg mapping_success_rate: {mapping_success_avg:.4f}
        - avg template_fallback_rate: {fallback_rate_avg:.4f}
        - template_source counts: {template_sources}

        {baseline_summary}

        ## Worst minutes errors (top 25)

        Columns: season_id game_id team_id player_id minutes_actual minutes_mean abs_err
        """
    )

    if not worst_minutes.empty:
        lines = []
        for row in worst_minutes.itertuples(index=False):
            lines.append(
                f"- {row.season_id} {row.game_id} team={row.team_id} player={row.player_id} "
                f"truth={row.minutes_actual:.1f} pred={row.minutes_mean:.1f} abs_err={row.abs_err:.1f}"
            )
        report += "\n" + "\n".join(lines) + "\n"
    else:
        report += "\n- (no rows)\n"

    report += "\n## Worst calibration bins (played_ge_5, top 10 by |gap|)\n\n"
    if not worst_calib_ge5.empty:
        for row in worst_calib_ge5.itertuples(index=False):
            report += (
                f"- bin={int(row.bin_idx)} count={int(row.count)} "
                f"p_mean={row.p_pred_mean:.3f} y_mean={row.y_true_mean:.3f} "
                f"gap={row.gap:.3f} brier_bin_mean={row.brier_bin_mean:.4f}\n"
            )
    else:
        report += "- (no rows)\n"

    report_path.write_text(report, encoding="utf-8")

    # Publishing pointers (same pattern as rot_v1/pbp_v1 bundles).
    (out_dir / "PUBLISHED").write_text("published\n", encoding="utf-8")
    write_latest_published_run_id(out_dir.parent, run_id)

    return {
        "out_dir": str(out_dir),
        "player_eval_path": str(player_eval_path),
        "team_eval_path": str(team_eval_path),
        "calibration_played_ge_1_path": str(calib_ge1_path),
        "calibration_played_ge_5_path": str(calib_ge5_path),
        "candidate_pool_summary_path": str(candidate_pool_summary_path),
        "report_path": str(report_path),
        "manifest_path": str(manifest_path),
        "input_hashes_path": str(input_hashes_path),
        "metrics": metrics.__dict__,
    }
