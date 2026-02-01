from __future__ import annotations

import hashlib
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
from projections.rotations.generator import TeamContext
from projections.rotations.manifest import write_json, write_latest_published_run_id
from projections.rotations.priors_humility import HumilityConfig, humility_config_as_dict
from projections.rotations.schemas import LINEUP_COLS
from projections.rotations.template_generator import TemplateRotationGenerator


SampleMode = Literal["random", "first"]
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
    humility_config: HumilityConfig | None = None,
) -> dict[str, Any]:
    """Evaluate TemplateRotationGenerator realism using truth candidate sets from rot_v1 labels.

    This evaluation intentionally *does not* test candidate-set prediction. It fixes:
    - `candidate_player_ids` to truth participants (minutes_actual>0 or played_ge_1==True)
    - `starter_candidates` to truth starters (fallback: first segment lineup)
    - `minutes_prior` optionally to truth minutes (default True) to stabilize role mapping
    """
    rot_bundle_path = Path(rot_bundle_path)
    out_dir = Path(out_dir)
    sample_mode = str(sample_mode)  # runtime validation
    if sample_mode not in {"random", "first"}:
        raise ValueError(f"Unknown sample_mode: {sample_mode}")

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
        priors["game_id"] = priors["game_id"].astype("string")
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
    events["game_id"] = events["game_id"].astype("string")
    events["team_id"] = pd.to_numeric(events["team_id"], errors="coerce").astype("Int64")
    events["opponent_team_id"] = pd.to_numeric(events["opponent_team_id"], errors="coerce").astype("Int64")
    events["is_home"] = _coerce_bool(events["is_home"])
    events["segment_idx"] = pd.to_numeric(events["segment_idx"], errors="coerce").astype("Int64")
    events["duration_sec"] = pd.to_numeric(events["duration_sec"], errors="coerce").astype("Int64")
    for c in LINEUP_COLS:
        events[c] = pd.to_numeric(events[c], errors="coerce").astype("Int64")

    labels["game_id"] = labels["game_id"].astype("string")
    labels["team_id"] = pd.to_numeric(labels["team_id"], errors="coerce").astype("Int64")
    labels["player_id"] = pd.to_numeric(labels["player_id"], errors="coerce").astype("Int64")
    labels["minutes_actual"] = pd.to_numeric(labels["minutes_actual"], errors="coerce").astype(np.float64).fillna(0.0)
    labels["played_ge_1"] = _coerce_bool(labels["played_ge_1"])
    labels["played_ge_5"] = _coerce_bool(labels["played_ge_5"])
    labels["starter_actual"] = _coerce_bool(labels["starter_actual"])
    labels["regime_label"] = labels["regime_label"].astype("string")

    events = events.dropna(subset=["game_id", "team_id"]).copy()
    labels = labels.dropna(subset=["game_id", "team_id", "player_id"]).copy()
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

    gen = TemplateRotationGenerator(rot_bundle=rot_bundle_path, humility_config=humility_config)
    generator_name = type(gen).__name__

    player_rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []

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

        # Truth candidate set: evaluate generator realism (mapping + template sampling), not candidate prediction.
        cand_mask = (g_labels["minutes_actual"] > 0.0) | (g_labels["played_ge_1"])
        candidate_ids = _unique_ints_sorted(g_labels.loc[cand_mask, "player_id"].tolist())

        starter_pool = g_labels.loc[g_labels["starter_actual"], ["player_id", "minutes_actual"]].copy()
        if not starter_pool.empty:
            starter_pool = starter_pool.sort_values(["minutes_actual", "player_id"], ascending=[False, True], kind="mergesort")
            starter_candidates = [int(v) for v in starter_pool["player_id"].tolist()]
        else:
            starter_candidates = _infer_starters_from_events(g_events)

        minutes_prior: Optional[dict[int, float]] = None
        minutes_p10_prior: Optional[dict[int, float]] = None
        minutes_p90_prior: Optional[dict[int, float]] = None
        play_prob_prior: Optional[dict[int, float]] = None
        if priors_gb is not None and (game_id, team_id) in priors_gb.groups:
            g_prior = priors_gb.get_group((game_id, team_id)).copy()
            if candidate_ids:
                g_prior = g_prior[g_prior["player_id"].isin(candidate_ids)].copy()
            minutes_prior = {int(pid): float(v) for pid, v in zip(g_prior["player_id"].tolist(), g_prior["minutes_prior"].tolist())}
            if "minutes_p10" in g_prior.columns:
                minutes_p10_prior = {int(pid): float(v) for pid, v in zip(g_prior["player_id"].tolist(), g_prior["minutes_p10"].tolist())}
            if "minutes_p90" in g_prior.columns:
                minutes_p90_prior = {int(pid): float(v) for pid, v in zip(g_prior["player_id"].tolist(), g_prior["minutes_p90"].tolist())}
            play_prob_prior = {int(pid): float(v) for pid, v in zip(g_prior["player_id"].tolist(), g_prior["play_prob"].tolist())}
        elif use_truth_minutes_prior_for_mapping:
            minutes_prior = {
                int(pid): float(mins)
                for pid, mins in zip(g_labels["player_id"].tolist(), g_labels["minutes_actual"].tolist())
            }

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
        humility_enabled=bool((humility_config or HumilityConfig()).enabled),
        humility_config=humility_config_as_dict(humility_config or HumilityConfig()),
        input_hashes_path=input_hashes_path,
    )
    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, manifest)

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
        - Uses *truth candidate sets* from `rotation_labels` (minutes_actual>0 OR played_ge_1==True)
        - Uses truth starters when available (fallback: first segment lineup)
        - Uses truth minutes as `minutes_prior` **only** as a mapping stabilizer (`use_truth_minutes_prior={use_truth_minutes_prior_for_mapping}`)
        - If `minutes_prior_parquet` is provided, uses that prior instead (and can restrict game universe)

        It does **not** attempt to predict availability / candidate sets; it evaluates mapping + template sampling realism.

        ## Prior humility layer

        - humility_enabled: {bool((humility_config or HumilityConfig()).enabled)}
        - humility_config: {humility_config_as_dict(humility_config or HumilityConfig())}

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
        - minutes_mae: {metrics.minutes_mae:.3f}

        ## Tail / mass metrics

        - avg P(minutes==0): {avg_p_zero:.4f}
        - avg P(minutes<5): {avg_p_lt5:.4f}

        ## Mapping diagnostics

        - avg mapping_success_rate: {mapping_success_avg:.4f}
        - avg template_fallback_rate: {fallback_rate_avg:.4f}
        - template_source counts: {template_sources}

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
        "report_path": str(report_path),
        "manifest_path": str(manifest_path),
        "input_hashes_path": str(input_hashes_path),
        "metrics": metrics.__dict__,
    }
