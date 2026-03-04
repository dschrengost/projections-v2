"""Score minutes using rotation_set_minutes_v1 (overlay or primary mode).

Modes:
- overlay (legacy): compute baseline minutes_v1 first, then apply a guarded rotation_set overlay.
- primary: skip baseline, produce canonical minutes directly from the transformer.
"""

from __future__ import annotations

# ruff: noqa: E402

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

from projections.runtime_safety import configure_runtime_safety

configure_runtime_safety()

import numpy as np
import pandas as pd
import typer

from projections import paths
from projections.minutes import (
    PLAY_THRESHOLD_MINUTES,
    ROTATION_THRESHOLD_MINUTES,
    build_provenance,
    compute_parity_report,
    inject_parity_into_summary,
    inject_provenance_into_summary,
    write_parity_report,
)
from projections.minutes_v1.reconcile import load_reconcile_config
from projections.rotation.guardrails import apply_rotation_minutes_guardrails
from projections.rotation.live_features_v1 import (
    RotationLiveFeaturesError,
    RotationSetMinutesV1FeatureSpec,
    build_rotation_set_minutes_v1_features,
    load_rotation_priors_for_live_inference,
)
from projections.rotation.projected_starter_boost import apply_projected_starter_boost
from projections.rotation.set_model import (
    predict_minutes as predict_rotation_minutes,
    RotationSetAuxOutputs,
)

# ESPN OUT override - import from baseline scorer to ensure consistency
from projections.cli.score_minutes_v1 import (
    _espn_override_allowed,
    _load_espn_out_players,
    _normalize_name_for_matching,
    _read_live_run_as_of_ts,
    _select_minutes_columns,
)


UTC = timezone.utc

app = typer.Typer(help=__doc__)

DEFAULT_DATA_ROOT = paths.get_data_root()
DEFAULT_MINUTES_FEATURES_ROOT = paths.data_path("live", "features_minutes_v1")
DEFAULT_ROTATION_FEATURES_ROOT = paths.data_path(
    "live", "features_rotation_set_minutes_v1"
)
DEFAULT_MINUTES_ARTIFACT_ROOT = paths.data_path("artifacts", "minutes_v1", "daily")
DEFAULT_ROTATION_CONFIG = Path("config/rotation_set_minutes_live.json")
DEFAULT_RECONCILE_CONFIG = Path("config/minutes_l2_reconcile.yaml")
PRIMARY_PASSTHROUGH_COLS: tuple[str, ...] = (
    "player_name",
    "status",
    "injury_row_present",
    "pos_bucket",
    "starter_flag",
    "is_projected_starter",
    "is_confirmed_starter",
    "tip_ts",
    "team_name",
    "team_tricode",
    "opponent_team_id",
    "opponent_team_name",
    "opponent_team_tricode",
    "spread_home",
    "total",
    "odds_as_of_ts",
    "blowout_index",
    "blowout_risk_score",
    "close_game_score",
    "team_abbr",
    "opp_team_abbr",
    "home_team_id",
    "away_team_id",
    "home_flag",
    "lineup_role",
    "lineup_status",
    "lineup_roster_status",
    "lineup_timestamp",
    "restriction_flag",
    "ramp_flag",
    "games_since_return",
    "days_since_return",
    "injury_as_of_ts",
    "prior_play_prob",
    "is_out",
    "is_q",
    "is_prob",
    # Optional volatility signals for tail reconstruction in primary mode.
    "roll_iqr_5",
    "rotation_minutes_std_5g",
)


@dataclass(frozen=True)
class RotationLiveConfig:
    enabled: bool
    model_dir: str | None
    blend_weight: float
    injury_coverage_threshold: float
    dnp_tail_minutes_threshold: float
    allow_priors_fallback: bool
    enable_blend_to_baseline: bool  # Default False (PR3: disable suppression)
    fallback_mode: str  # "fail_closed" or "degrade_loudly"
    mode: str  # "overlay" or "primary"
    play_prob_source: str  # "minutes_threshold" or "gate_prob"
    rotation_prob_source: str  # "minutes_threshold" or "gate_prob"
    dnp_override_enabled: bool
    # Sparsify parameters (opt-in, default disabled)
    sparsify_enable: bool
    sparsify_topk: int
    sparsify_tau: float
    sparsify_kmax: int
    sparsify_min_keep: int
    sparsify_use_col: str
    alloc_mask_mode: str
    alloc_min_eligible: int
    alloc_prior_play_prob_threshold: float
    alloc_baseline_minutes_threshold: float
    projected_starter_boost_enabled: bool
    projected_starter_gate_prob_boost_threshold: float
    projected_starter_gate_logit_boost: float

    @classmethod
    def load(cls, path: Path) -> "RotationLiveConfig":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            payload = {}
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid rotation minutes config at {path}: {exc}"
            ) from exc
        enabled = bool(payload.get("enabled", False))
        model_dir = payload.get("model_dir")
        blend_weight = float(payload.get("blend_weight", 0.2))
        injury_threshold = float(payload.get("injury_coverage_threshold", 0.5))
        dnp_tail_threshold = float(payload.get("dnp_tail_minutes_threshold", 8.0))
        allow_priors_fallback = bool(payload.get("allow_priors_fallback", False))
        # PR3: blend-to-baseline is now disabled by default
        enable_blend = bool(payload.get("enable_blend_to_baseline", False))
        # PR3: explicit fallback mode (default: degrade_loudly for safer rollout)
        fallback_mode = str(payload.get("fallback_mode", "degrade_loudly"))
        if fallback_mode not in ("fail_closed", "degrade_loudly"):
            fallback_mode = "degrade_loudly"
        mode = _normalize_rotation_mode(str(payload.get("mode", "overlay")))
        play_prob_source = str(payload.get("play_prob_source", "minutes_threshold")).strip().lower()
        if play_prob_source not in {"minutes_threshold", "gate_prob"}:
            play_prob_source = "minutes_threshold"
        rotation_prob_source = str(payload.get("rotation_prob_source", "minutes_threshold")).strip().lower()
        if rotation_prob_source not in {"minutes_threshold", "gate_prob"}:
            rotation_prob_source = "minutes_threshold"
        dnp_override_enabled = bool(payload.get("dnp_override_enabled", True))
        # Sparsify parameters (safe defaults: disabled)
        sparsify_enable = bool(payload.get("sparsify_enable", False))
        sparsify_topk = int(payload.get("sparsify_topk", 9))
        sparsify_tau = float(payload.get("sparsify_tau", 0.55))
        sparsify_kmax = int(payload.get("sparsify_kmax", 11))
        sparsify_min_keep = int(payload.get("sparsify_min_keep", 8))
        sparsify_use_col = str(payload.get("sparsify_use_col", "gate_prob"))
        alloc_mask_mode = str(payload.get("alloc_mask_mode", "strict"))
        if alloc_mask_mode not in ("strict", "not_out"):
            alloc_mask_mode = "strict"
        alloc_min_eligible = int(payload.get("alloc_min_eligible", 9))
        alloc_prior_play_prob_threshold = float(payload.get("alloc_prior_play_prob_threshold", 0.20))
        alloc_baseline_minutes_threshold = float(payload.get("alloc_baseline_minutes_threshold", 4.0))
        projected_starter_boost_enabled = bool(payload.get("projected_starter_boost_enabled", False))
        projected_gate_prob_thr = float(payload.get("projected_starter_gate_prob_boost_threshold", 0.90))
        if not (0.0 < projected_gate_prob_thr < 1.0):
            projected_gate_prob_thr = 0.90
        projected_gate_logit_boost = float(payload.get("projected_starter_gate_logit_boost", 20.0))
        return cls(
            enabled=enabled,
            model_dir=str(model_dir) if model_dir else None,
            blend_weight=blend_weight,
            injury_coverage_threshold=injury_threshold,
            dnp_tail_minutes_threshold=dnp_tail_threshold,
            allow_priors_fallback=allow_priors_fallback,
            enable_blend_to_baseline=enable_blend,
            fallback_mode=fallback_mode,
            mode=mode,
            play_prob_source=play_prob_source,
            rotation_prob_source=rotation_prob_source,
            dnp_override_enabled=dnp_override_enabled,
            sparsify_enable=sparsify_enable,
            sparsify_topk=sparsify_topk,
            sparsify_tau=sparsify_tau,
            sparsify_kmax=sparsify_kmax,
            sparsify_min_keep=sparsify_min_keep,
            sparsify_use_col=sparsify_use_col,
            alloc_mask_mode=alloc_mask_mode,
            alloc_min_eligible=alloc_min_eligible,
            alloc_prior_play_prob_threshold=alloc_prior_play_prob_threshold,
            alloc_baseline_minutes_threshold=alloc_baseline_minutes_threshold,
            projected_starter_boost_enabled=projected_starter_boost_enabled,
            projected_starter_gate_prob_boost_threshold=projected_gate_prob_thr,
            projected_starter_gate_logit_boost=projected_gate_logit_boost,
        )


def _normalize_day(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _season_for_day(day: pd.Timestamp) -> int:
    return int(day.year) if int(day.month) >= 8 else int(day.year) - 1


def _normalize_rotation_mode(value: str | None) -> str:
    if not value:
        return "overlay"
    raw = str(value).strip().lower()
    if raw in {"primary", "transformer", "transformer_only", "transformer-only", "canonical"}:
        return "primary"
    if raw in {"overlay", "legacy", "baseline"}:
        return "overlay"
    return "overlay"


@dataclass
class BaselineMinutesResult:
    """Result of extracting baseline minutes from features."""

    values: np.ndarray
    source_col: str
    gt0_count: int
    min_val: float
    median_val: float
    max_val: float


def _extract_baseline_minutes(
    minutes_feat: pd.DataFrame,
    candidate_cols: tuple[str, ...] = (
        "minutes_p50_model",
        "minutes_final",
        "minutes_p50",
        "roll_mean_5",
        "roll_mean_3",
    ),
) -> BaselineMinutesResult:
    """Extract baseline minutes from the first valid candidate column.

    Priority order: minutes_p50_model > minutes_final > minutes_p50 >
    roll_mean_5 > roll_mean_3 (historical averages as last-resort fallback).
    Uses the first column that has any non-zero values.
    """
    for candidate_col in candidate_cols:
        if candidate_col in minutes_feat.columns:
            candidate_vals = pd.to_numeric(
                minutes_feat[candidate_col], errors="coerce"
            ).fillna(0.0)
            if candidate_vals.gt(0).any():
                vals = candidate_vals.astype(float).to_numpy()
                return BaselineMinutesResult(
                    values=vals,
                    source_col=candidate_col,
                    gt0_count=int((vals > 0).sum()),
                    min_val=float(vals.min()),
                    median_val=float(np.median(vals)),
                    max_val=float(vals.max()),
                )
    # No valid column found
    n_rows = len(minutes_feat)
    return BaselineMinutesResult(
        values=np.zeros(n_rows, dtype=float),
        source_col="none",
        gt0_count=0,
        min_val=0.0,
        median_val=0.0,
        max_val=0.0,
    )


def _build_primary_minutes_frame(
    minutes_feat: pd.DataFrame,
    *,
    day: pd.Timestamp,
) -> pd.DataFrame:
    key_cols = [c for c in ("game_id", "player_id", "team_id") if c in minutes_feat.columns]
    out = minutes_feat.loc[:, key_cols].copy()
    out["game_date"] = day.date()
    for col in PRIMARY_PASSTHROUGH_COLS:
        if col in minutes_feat.columns:
            out[col] = minutes_feat[col]
    return out


def _derive_out_mask(df: pd.DataFrame) -> pd.Series:
    out_mask = pd.Series(False, index=df.index, dtype=bool)
    if "is_out" in df.columns:
        out_flag = pd.to_numeric(df["is_out"], errors="coerce").fillna(0).astype(int)
        out_mask = out_mask | (out_flag.to_numpy(dtype=int) == 1)
    if "status" in df.columns:
        status = df["status"].astype("string").str.upper()
        out_mask = out_mask | status.eq("OUT").fillna(False).to_numpy(dtype=bool)
    if "lineup_role" in df.columns:
        lineup_role = df["lineup_role"].astype("string").str.strip().str.lower()
        out_mask = out_mask | lineup_role.eq("out").fillna(False).to_numpy(dtype=bool)
    return out_mask


def _apply_out_row_overrides(df: pd.DataFrame, out_mask: pd.Series) -> pd.DataFrame:
    """Normalize OUT semantics on scorer outputs.

    Rotation-set scoring can identify OUT rows via lineup_role even when
    injuries snapshot status remains "Ava". Ensure downstream consumers see
    explicit out flags/status for these rows.
    """
    if df.empty:
        return df

    out = df.copy()
    mask = out_mask.reindex(out.index).fillna(False).astype(bool)
    if not mask.any():
        return out

    if "is_out" not in out.columns:
        out["is_out"] = 0
    out.loc[mask, "is_out"] = 1

    if "status" not in out.columns:
        out["status"] = pd.NA
    out.loc[mask, "status"] = "OUT"
    return out


def _scale_minutes_to_team_target(
    df: pd.DataFrame,
    minutes: pd.Series,
    *,
    team_target: float = 240.0,
    group_cols: tuple[str, str] = ("game_id", "team_id"),
) -> tuple[pd.Series, dict[str, float | int]]:
    if minutes.empty:
        return minutes, {"teams": 0, "zero_sum_teams": 0, "max_abs_delta": 0.0}
    team_sum = minutes.groupby([df[c] for c in group_cols], sort=False).transform("sum")
    scale = (float(team_target) / team_sum.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaled = minutes.to_numpy(dtype=float) * scale.to_numpy(dtype=float)
    scaled = np.clip(scaled, 0.0, None)
    scaled_series = pd.Series(scaled, index=minutes.index, dtype=float)
    grouped_before = minutes.groupby([df[c] for c in group_cols], sort=False).sum()
    grouped_after = scaled_series.groupby([df[c] for c in group_cols], sort=False).sum()
    max_abs_delta = float((grouped_after - float(team_target)).abs().max()) if not grouped_after.empty else 0.0
    zero_sum = int((grouped_before <= 0.0).sum()) if not grouped_before.empty else 0
    return scaled_series, {
        "teams": int(grouped_before.shape[0]) if not grouped_before.empty else 0,
        "zero_sum_teams": zero_sum,
        "max_abs_delta": max_abs_delta,
    }


def _derive_gate_probs(
    minutes: pd.Series,
    *,
    out_mask: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    mins = pd.to_numeric(minutes, errors="coerce").fillna(0.0).astype(float)
    play_prob = (mins >= float(PLAY_THRESHOLD_MINUTES)).astype(float)
    rotation_prob = (mins >= float(ROTATION_THRESHOLD_MINUTES)).astype(float)
    if out_mask is not None:
        mask = out_mask.to_numpy(dtype=bool)
        play_prob = play_prob.where(~mask, 0.0)
        rotation_prob = rotation_prob.where(~mask, 0.0)
    return play_prob, rotation_prob


def _run_baseline_minutes(
    *,
    day: pd.Timestamp,
    run_id: str,
    reconcile_team_minutes: str,
    minutes_output: str,
    bundle_dir: Path | None,
    rotshare_quantiles_mode: str | None,
    rotshare_n_worlds: int | None,
    rotshare_concentration: float | None,
    rotshare_seed: int | None,
    rotshare_min_active_players: int | None,
    rotshare_mc_center: str | None,
    data_root: Path,
) -> None:
    typer.echo(
        f"[rotation_minutes] baseline minutes_v1 date={day.date()} run_id={run_id}",
        err=True,
    )
    baseline_args = [
        "--date",
        day.strftime("%Y-%m-%d"),
        "--mode",
        "live",
        "--run-id",
        run_id,
        "--reconcile-team-minutes",
        str(reconcile_team_minutes),
        "--minutes-output",
        str(minutes_output),
    ]
    if bundle_dir is not None:
        baseline_args.extend(["--bundle-dir", str(bundle_dir)])
    if rotshare_quantiles_mode:
        baseline_args.extend(["--rotshare-quantiles-mode", str(rotshare_quantiles_mode)])
    if rotshare_n_worlds is not None:
        baseline_args.extend(["--rotshare-n-worlds", str(int(rotshare_n_worlds))])
    if rotshare_concentration is not None:
        baseline_args.extend(["--rotshare-concentration", str(float(rotshare_concentration))])
    if rotshare_seed is not None:
        baseline_args.extend(["--rotshare-seed", str(int(rotshare_seed))])
    if rotshare_min_active_players is not None:
        baseline_args.extend(
            ["--rotshare-min-active-players", str(int(rotshare_min_active_players))]
        )
    if rotshare_mc_center:
        baseline_args.extend(["--rotshare-mc-center", str(rotshare_mc_center)])

    _run_python_module(
        "projections.cli.score_minutes_v1",
        baseline_args,
        data_root=data_root,
        timeout_s=1200,
    )


def _fallback_primary_to_baseline(
    *,
    reason: str,
    day: pd.Timestamp,
    run_id: str,
    reconcile_team_minutes: str,
    minutes_output: str,
    bundle_dir: Path | None,
    rotshare_quantiles_mode: str | None,
    rotshare_n_worlds: int | None,
    rotshare_concentration: float | None,
    rotshare_seed: int | None,
    rotshare_min_active_players: int | None,
    rotshare_mc_center: str | None,
    data_root: Path,
    summary_path: Path,
    mode: str,
) -> None:
    typer.echo(
        f"[rotation_minutes] WARNING: primary mode fallback to baseline ({reason})",
        err=True,
    )
    _run_baseline_minutes(
        day=day,
        run_id=run_id,
        reconcile_team_minutes=reconcile_team_minutes,
        minutes_output=minutes_output,
        bundle_dir=bundle_dir,
        rotshare_quantiles_mode=rotshare_quantiles_mode,
        rotshare_n_worlds=rotshare_n_worlds,
        rotshare_concentration=rotshare_concentration,
        rotshare_seed=rotshare_seed,
        rotshare_min_active_players=rotshare_min_active_players,
        rotshare_mc_center=rotshare_mc_center,
        data_root=data_root,
    )
    _update_summary_json(
        summary_path,
        {
            "rotation_set_minutes": {
                "enabled": False,
                "mode": mode,
                "error": reason,
                "fallback_to_baseline": True,
            }
        },
    )


def _load_rotation_historical_features_for_dnp(
    data_root: Path,
    *,
    season: int,
    target_day: pd.Timestamp,
    team_ids: set[int],
    player_ids: set[int],
    lookback_days: int | None = 120,
) -> pd.DataFrame:
    """Load realized minutes + pre-tip availability for DNP history features.

    To match training semantics, we build the historical frame from:
      - `labels/season={season}/boxscore_labels.parquet` for realized minutes, and
      - `bronze/injuries_raw/season={season}/date=*/injuries.parquet` for
        pre-tip injury report status to derive `is_out`.

    This approximates the training signal where `is_out` comes from a pre-tip
    injury snapshot. We consider a player OUT on a given game_date if they appear
    in that date partition with `status == OUT`.
    """
    if not team_ids or not player_ids:
        return pd.DataFrame()

    end_day = pd.Timestamp(target_day).normalize()
    if lookback_days is not None and int(lookback_days) <= 0:
        raise ValueError("lookback_days must be > 0 when provided")

    if lookback_days is None:
        label_paths = sorted((data_root / "labels").glob("season=*/boxscore_labels.parquet"))
    else:
        label_paths = [data_root / "labels" / f"season={int(season)}" / "boxscore_labels.parquet"]
    label_paths = [p for p in label_paths if p.exists()]
    if not label_paths:
        return pd.DataFrame()

    label_frames: list[pd.DataFrame] = []
    for labels_path in label_paths:
        try:
            frame = pd.read_parquet(
                labels_path,
                columns=["game_date", "team_id", "player_id", "minutes"],
            )
        except Exception:  # noqa: BLE001
            frame = pd.read_parquet(labels_path)
            if not {"game_date", "team_id", "player_id", "minutes"}.issubset(frame.columns):
                continue
            frame = frame.loc[:, ["game_date", "team_id", "player_id", "minutes"]]
        label_frames.append(frame)
    if not label_frames:
        return pd.DataFrame()
    labels = pd.concat(label_frames, ignore_index=True)

    labels["game_date"] = pd.to_datetime(
        labels["game_date"], errors="coerce"
    ).dt.normalize()
    labels["team_id"] = pd.to_numeric(labels["team_id"], errors="coerce").astype(
        "Int64"
    )
    labels["player_id"] = pd.to_numeric(labels["player_id"], errors="coerce").astype(
        "Int64"
    )
    labels = labels.dropna(subset=["game_date", "team_id", "player_id"]).copy()
    if labels.empty:
        return pd.DataFrame()

    labels = labels.loc[
        (labels["game_date"] < end_day)
        & labels["team_id"].astype(int).isin(team_ids)
        & labels["player_id"].astype(int).isin(player_ids)
    ].copy()
    if lookback_days is not None:
        cutoff = (end_day - timedelta(days=int(lookback_days))).normalize()
        labels = labels.loc[labels["game_date"] >= cutoff].copy()
    if labels.empty:
        return pd.DataFrame()

    labels["minutes"] = (
        pd.to_numeric(labels["minutes"], errors="coerce").fillna(0.0).astype(float)
    )

    # Derive is_out from injuries_raw partitions (keyed by date folder).
    # IMPORTANT: Use the LATEST snapshot per player per date, not "any OUT".
    # The source_row_id format is "{unix_ts}_{row_id}" - we parse the timestamp to sort.
    out_frames: list[pd.DataFrame] = []
    for day in sorted(pd.to_datetime(labels["game_date"], errors="coerce").dropna().dt.normalize().unique().tolist()):
        season_for_day = _season_for_day(pd.Timestamp(day))
        injuries_root = data_root / "bronze" / "injuries_raw" / f"season={int(season_for_day)}"
        if injuries_root.exists():
            day_dir = injuries_root / f"date={pd.Timestamp(day).strftime('%Y-%m-%d')}"
            pq_path = day_dir / "injuries.parquet"
            if not pq_path.exists():
                continue
            try:
                inj = pd.read_parquet(
                    pq_path, columns=["team_id", "player_id", "status", "source_row_id"]
                )
            except Exception:  # noqa: BLE001
                try:
                    inj = pd.read_parquet(pq_path)
                except Exception:  # noqa: BLE001
                    continue
                if not {"team_id", "player_id", "status"}.issubset(inj.columns):
                    continue
                cols = ["team_id", "player_id", "status"]
                if "source_row_id" in inj.columns:
                    cols.append("source_row_id")
                inj = inj.loc[:, cols]

            if inj.empty:
                continue
            inj["team_id"] = pd.to_numeric(inj["team_id"], errors="coerce").astype(
                "Int64"
            )
            inj["player_id"] = pd.to_numeric(inj["player_id"], errors="coerce").astype(
                "Int64"
            )
            inj = inj.dropna(subset=["team_id", "player_id"]).copy()
            if inj.empty:
                continue
            inj = inj.loc[
                inj["team_id"].astype(int).isin(team_ids)
                & inj["player_id"].astype(int).isin(player_ids)
            ].copy()
            if inj.empty:
                continue

            # Extract timestamp from source_row_id to get the LATEST snapshot per player.
            # Format: "{unix_ts}_{row_id}" e.g., "1767501000_2066"
            if "source_row_id" in inj.columns:
                inj["_sort_ts"] = (
                    inj["source_row_id"]
                    .astype(str)
                    .str.split("_")
                    .str[0]
                    .pipe(pd.to_numeric, errors="coerce")
                    .fillna(0)
                )
                # Keep only the LATEST snapshot per player
                inj = inj.sort_values("_sort_ts", ascending=False)
                inj = inj.drop_duplicates(subset=["team_id", "player_id"], keep="first")
                inj = inj.drop(columns=["_sort_ts"])

            # Now check if the FINAL status is OUT
            status = inj["status"].astype("string").str.upper().str.strip()
            out_like = status.eq("OUT").fillna(False)
            inj = inj.loc[out_like, ["team_id", "player_id"]].drop_duplicates()
            if inj.empty:
                continue
            inj["game_date"] = pd.Timestamp(day).normalize()
            inj["is_out"] = 1
            out_frames.append(
                inj.loc[:, ["game_date", "team_id", "player_id", "is_out"]]
            )

    if out_frames:
        out_hist = pd.concat(out_frames, ignore_index=True).drop_duplicates(
            subset=["game_date", "team_id", "player_id"], keep="last"
        )
    else:
        out_hist = pd.DataFrame(columns=["game_date", "team_id", "player_id", "is_out"])

    hist = labels.merge(out_hist, on=["game_date", "team_id", "player_id"], how="left")
    hist["is_out"] = (
        pd.to_numeric(hist["is_out"], errors="coerce").fillna(0).astype(int)
    )
    return hist.reset_index(drop=True)


def _run_python_module(
    module: str, args: list[str], *, data_root: Path, timeout_s: int
) -> None:
    env = os.environ.copy()
    env["PROJECTIONS_DATA_ROOT"] = str(data_root)
    cmd = [sys.executable, "-m", module, *args]
    result = subprocess.run(
        cmd,
        cwd=str(paths.get_project_root()),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{module} failed with exit_code={result.returncode}")


def _derive_minutes_tails(
    *,
    base_p10: pd.Series,
    base_p50: pd.Series,
    base_p90: pd.Series,
    new_p50: pd.Series,
    cap_max: float = 48.0,
    eps: float = 1e-9,
) -> tuple[pd.Series, pd.Series]:
    p10_base = pd.to_numeric(base_p10, errors="coerce").fillna(0.0).astype(float)
    p50_base = pd.to_numeric(base_p50, errors="coerce").fillna(0.0).astype(float)
    p90_base = pd.to_numeric(base_p90, errors="coerce").fillna(0.0).astype(float)
    p50_new = pd.to_numeric(new_p50, errors="coerce").fillna(0.0).astype(float)

    delta_low = (p50_base - p10_base).clip(lower=0.0)
    delta_high = (p90_base - p50_base).clip(lower=0.0)

    p10_new = (p50_new - delta_low).clip(lower=0.0)
    p90_new = p50_new + delta_high

    p10_new = np.minimum(p10_new, p50_new)
    p90_new = np.maximum(p90_new, p50_new)

    zero_mask = p50_new <= eps
    if zero_mask.any():
        p10_new = p10_new.mask(zero_mask, 0.0)
        p90_new = p90_new.mask(zero_mask, 0.0)

    cap = float(cap_max)
    p10_new = np.minimum(p10_new, cap)
    p90_new = np.minimum(p90_new, cap)

    p10_new = np.minimum(p10_new, p50_new)
    p90_new = np.maximum(p90_new, p50_new)
    return p10_new.astype(float), p90_new.astype(float)


def _compute_delta_diagnostics(
    df: pd.DataFrame,
    *,
    baseline_p50_col: str,
    final_p50_col: str,
    raw_rotation_p50_col: str | None = None,
    blend_applied_map: dict[tuple[str, int], bool] | None = None,
    actual_minutes_col: str | None = None,
) -> list[dict[str, Any]]:
    """Compute per-team-game delta diagnostics comparing final vs baseline p50.

    Metrics:
      - mean_abs_delta_p50: mean(|final - baseline|) per team-game
      - p95_abs_delta_p50: 95th percentile of |final - baseline|
      - effective_blend_mass: same as mean_abs_delta_p50 (logged separately as
        a "did the overlay move anything?" signal)
      - directional_win_rate_ge4: For rows where |final - base| >= 4,
        fraction where sign(final - base) == sign(actual - base).
        Only computed if actual_minutes_col is present; otherwise null.

    Pre-blend (raw) metrics (if raw_rotation_p50_col provided):
      - mean_abs_raw_delta_p50: mean(|raw_rot - baseline|) before blend/guardrails
      - max_abs_raw_delta_p50: max(|raw_rot - baseline|)
      - material_raw_delta_share: fraction of rows where |raw_rot - baseline| >= 10
      - blend_applied: whether this team-game had the conservative blend applied

    Returns a list of dicts, one per (game_id, team_id).
    """
    work = df.copy()
    required = ["game_id", "team_id", baseline_p50_col, final_p50_col]
    missing = [c for c in required if c not in work.columns]
    if missing:
        return []

    baseline = pd.to_numeric(work[baseline_p50_col], errors="coerce").fillna(0.0)
    final = pd.to_numeric(work[final_p50_col], errors="coerce").fillna(0.0)
    work["_abs_delta"] = (final - baseline).abs()
    work["_delta"] = final - baseline

    # Pre-blend raw deltas (rotation p50 before guardrails/blend)
    has_raw = raw_rotation_p50_col and raw_rotation_p50_col in work.columns
    if has_raw:
        raw_rot = pd.to_numeric(work[raw_rotation_p50_col], errors="coerce").fillna(0.0)
        work["_abs_raw_delta"] = (raw_rot - baseline).abs()

    # Check for realized minutes (historical/replay mode only)
    has_actual = (
        actual_minutes_col is not None
        and actual_minutes_col in work.columns
        and work[actual_minutes_col].notna().any()
    )
    if has_actual:
        actual = pd.to_numeric(work[actual_minutes_col], errors="coerce").fillna(0.0)
        work["_actual_delta"] = actual - baseline
    else:
        work["_actual_delta"] = np.nan

    results: list[dict[str, Any]] = []
    for (game_id, team_id), grp in work.groupby(["game_id", "team_id"], sort=False):
        abs_delta = grp["_abs_delta"].to_numpy(dtype=float)
        delta = grp["_delta"].to_numpy(dtype=float)

        mean_abs = float(np.mean(abs_delta)) if len(abs_delta) else 0.0
        p95_abs = float(np.percentile(abs_delta, 95)) if len(abs_delta) else 0.0

        # Directional win rate for material changes (|delta| >= 4)
        dir_win_rate: float | None = None
        if has_actual:
            actual_delta = grp["_actual_delta"].to_numpy(dtype=float)
            mask_material = np.abs(delta) >= 4.0
            if mask_material.sum() > 0:
                # sign match: both positive or both negative or both zero
                sign_match = np.sign(delta[mask_material]) == np.sign(
                    actual_delta[mask_material]
                )
                dir_win_rate = float(np.mean(sign_match))

        # Pre-blend raw delta metrics
        raw_mean: float | None = None
        raw_max: float | None = None
        material_share: float | None = None
        if has_raw:
            abs_raw = grp["_abs_raw_delta"].to_numpy(dtype=float)
            raw_mean = float(np.mean(abs_raw)) if len(abs_raw) else 0.0
            raw_max = float(np.max(abs_raw)) if len(abs_raw) else 0.0
            material_share = float((abs_raw >= 10.0).mean()) if len(abs_raw) else 0.0

        # Blend applied flag
        blend_applied: bool | None = None
        if blend_applied_map is not None:
            blend_applied = blend_applied_map.get((str(game_id), int(team_id)))

        results.append(
            {
                "game_id": str(game_id),
                "team_id": int(team_id),
                "mean_abs_delta_p50": round(mean_abs, 3),
                "p95_abs_delta_p50": round(p95_abs, 3),
                "effective_blend_mass": round(mean_abs, 3),
                "directional_win_rate_ge4": (
                    round(dir_win_rate, 3) if dir_win_rate is not None else None
                ),
                # Pre-blend metrics
                "mean_abs_raw_delta_p50": (
                    round(raw_mean, 3) if raw_mean is not None else None
                ),
                "max_abs_raw_delta_p50": (
                    round(raw_max, 3) if raw_max is not None else None
                ),
                "material_raw_delta_share_ge10": (
                    round(material_share, 3) if material_share is not None else None
                ),
                "blend_applied": blend_applied,
            }
        )
    return results



def _log_delta_diagnostics(diagnostics: list[dict[str, Any]]) -> None:
    """Emit structured log lines for delta diagnostics per team-game."""
    for rec in diagnostics:
        dir_rate = rec.get("directional_win_rate_ge4")
        dir_str = "null" if dir_rate is None else f"{dir_rate:.3f}"
        raw_mean = rec.get("mean_abs_raw_delta_p50")
        raw_mean_str = "null" if raw_mean is None else f"{raw_mean:.2f}"
        raw_max = rec.get("max_abs_raw_delta_p50")
        raw_max_str = "null" if raw_max is None else f"{raw_max:.2f}"
        mat_share = rec.get("material_raw_delta_share_ge10")
        mat_share_str = "null" if mat_share is None else f"{mat_share:.3f}"
        blend = rec.get("blend_applied")
        blend_str = "null" if blend is None else str(blend).lower()
        typer.echo(
            f"[rotation_minutes][delta_diag] "
            f"game_id={rec['game_id']} team_id={rec['team_id']} "
            f"mean_abs_delta_p50={rec['mean_abs_delta_p50']:.2f} "
            f"p95_abs_delta_p50={rec['p95_abs_delta_p50']:.2f} "
            f"mean_abs_raw_delta_p50={raw_mean_str} "
            f"max_abs_raw_delta_p50={raw_max_str} "
            f"material_raw_delta_share_ge10={mat_share_str} "
            f"blend_applied={blend_str} "
            f"directional_win_rate_ge4={dir_str}",
            err=True,
        )



def _update_summary_json(summary_path: Path, updates: dict[str, Any]) -> None:

    if not summary_path.exists():
        summary = {}
    else:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    merged = dict(summary)
    merged.update(updates)
    summary_path.write_text(
        json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8"
    )


@app.command()
def main(
    *,
    date: str = typer.Option(..., "--date", help="Slate date (YYYY-MM-DD)."),
    run_id: str = typer.Option(..., "--run-id", help="Run id for run-scoped outputs."),
    reconcile_team_minutes: str = typer.Option(
        "p50", "--reconcile-team-minutes", help="Baseline reconcile mode."
    ),
    reconcile_config: Path = typer.Option(
        DEFAULT_RECONCILE_CONFIG,
        "--reconcile-config",
        help="Reconcile config used to cap/redistribute the rotation overlay (matches baseline minutes_v1).",
    ),
    minutes_output: str = typer.Option(
        "conditional", "--minutes-output", help="Minutes output mode (baseline)."
    ),
    skip_baseline: bool = typer.Option(
        False,
        "--skip-baseline",
        help="Skip baseline minutes_v1 scoring (assumes baseline outputs already exist for this run_id).",
    ),
    bundle_dir: Path | None = typer.Option(
        None, "--bundle-dir", help="Override baseline minutes bundle directory."
    ),
    rotshare_quantiles_mode: str | None = typer.Option(
        None, "--rotshare-quantiles-mode", help="Baseline rotshare mode."
    ),
    rotshare_n_worlds: int | None = typer.Option(
        None, "--rotshare-n-worlds", help="Baseline rotshare n worlds."
    ),
    rotshare_concentration: float | None = typer.Option(
        None, "--rotshare-concentration", help="Baseline rotshare concentration."
    ),
    rotshare_seed: int | None = typer.Option(
        None, "--rotshare-seed", help="Baseline rotshare seed."
    ),
    rotshare_min_active_players: int | None = typer.Option(
        None,
        "--rotshare-min-active-players",
        help="Baseline rotshare min active players.",
    ),
    rotshare_mc_center: str | None = typer.Option(
        None, "--rotshare-mc-center", help="Baseline rotshare mc center."
    ),
    scoring_mode: str | None = typer.Option(
        None,
        "--scoring-mode",
        help="Scoring mode: overlay (legacy) or primary (transformer-only).",
    ),
    data_root: Path = typer.Option(
        DEFAULT_DATA_ROOT, "--data-root", help="PROJECTIONS_DATA_ROOT override."
    ),
    minutes_features_root: Path = typer.Option(
        DEFAULT_MINUTES_FEATURES_ROOT,
        "--minutes-features-root",
        help="Root for live minutes features.",
    ),
    rotation_features_root: Path = typer.Option(
        DEFAULT_ROTATION_FEATURES_ROOT,
        "--rotation-features-root",
        help="Where to write rotation feature slices.",
    ),
    artifact_root: Path = typer.Option(
        DEFAULT_MINUTES_ARTIFACT_ROOT,
        "--artifact-root",
        help="Minutes artifacts root (minutes_v1/daily).",
    ),
    rotation_config: Path = typer.Option(
        DEFAULT_ROTATION_CONFIG,
        "--rotation-config",
        help="JSON config for rotation minutes enable/weights.",
    ),
    model_dir: Path | None = typer.Option(
        None,
        "--model-dir",
        help="Override rotation model dir (defaults to config or ROTATION_SET_MODEL_DIR).",
    ),
    blend_weight: float | None = typer.Option(
        None, "--blend-weight", help="Override conservative blend weight."
    ),
    injury_coverage_threshold: float | None = typer.Option(
        None, "--injury-coverage-threshold", help="Override injury coverage threshold."
    ),
    dnp_tail_minutes_threshold: float | None = typer.Option(
        None,
        "--dnp-tail-minutes-threshold",
        help="Max minutes allowed on OUT/DNP-proxy rows per team-game before falling back to baseline.",
    ),
    device: str = typer.Option(
        "cpu", "--device", help="Torch device for rotation model."
    ),
    batch_size: int = typer.Option(
        64, "--batch-size", help="Batch size (team-games) for rotation model."
    ),
    alloc_mask_mode: str | None = typer.Option(
        None,
        "--alloc-mask-mode",
        help="Inference allocation mask: strict (heuristic eligibility) or not_out (all non-OUT players).",
    ),
    alloc_min_eligible: int | None = typer.Option(
        None,
        "--alloc-min-eligible",
        help="Minimum eligible players per team-game for strict allocation mask mode.",
    ),
    alloc_prior_play_prob_threshold: float | None = typer.Option(
        None,
        "--alloc-prior-play-prob-threshold",
        help="Strict alloc-mask threshold on prior_play_prob eligibility.",
    ),
    alloc_baseline_minutes_threshold: float | None = typer.Option(
        None,
        "--alloc-baseline-minutes-threshold",
        help="Strict alloc-mask threshold on baseline minutes eligibility.",
    ),
) -> None:
    day = _normalize_day(date)
    season = _season_for_day(day)
    data_root = Path(data_root).expanduser().resolve()

    minutes_out_dir = Path(artifact_root) / day.strftime("%Y-%m-%d") / f"run={run_id}"
    minutes_path = minutes_out_dir / "minutes.parquet"
    summary_path = minutes_out_dir / "summary.json"

    config = RotationLiveConfig.load(rotation_config)
    enabled = bool(config.enabled)
    resolved_mode = _normalize_rotation_mode(
        scoring_mode or os.environ.get("ROTATION_SET_SCORING_MODE") or config.mode
    )
    primary_mode = resolved_mode == "primary"
    resolved_model_dir = (
        model_dir
        or (
            Path(os.environ.get("ROTATION_SET_MODEL_DIR", ""))
            if os.environ.get("ROTATION_SET_MODEL_DIR")
            else None
        )
        or (Path(config.model_dir) if config.model_dir else None)
    )
    if not primary_mode:
        # 1) Always run baseline first (safe fallback), unless explicitly skipped.
        if not skip_baseline:
            _run_baseline_minutes(
                day=day,
                run_id=run_id,
                reconcile_team_minutes=reconcile_team_minutes,
                minutes_output=minutes_output,
                bundle_dir=bundle_dir,
                rotshare_quantiles_mode=rotshare_quantiles_mode,
                rotshare_n_worlds=rotshare_n_worlds,
                rotshare_concentration=rotshare_concentration,
                rotshare_seed=rotshare_seed,
                rotshare_min_active_players=rotshare_min_active_players,
                rotshare_mc_center=rotshare_mc_center,
                data_root=data_root,
            )

        if not minutes_path.exists():
            raise RuntimeError(f"Baseline minutes.parquet missing at {minutes_path}")

        base_df = pd.read_parquet(minutes_path)
        if base_df.empty:
            typer.echo("[rotation_minutes] baseline output empty; nothing to do.", err=True)
            return
    else:
        minutes_out_dir.mkdir(parents=True, exist_ok=True)
        base_df = pd.DataFrame()

    if not enabled:
        if primary_mode:
            reason = f"rotation_set_minutes disabled (config={rotation_config})"
            if config.fallback_mode == "degrade_loudly":
                _fallback_primary_to_baseline(
                    reason=reason,
                    day=day,
                    run_id=run_id,
                    reconcile_team_minutes=reconcile_team_minutes,
                    minutes_output=minutes_output,
                    bundle_dir=bundle_dir,
                    rotshare_quantiles_mode=rotshare_quantiles_mode,
                    rotshare_n_worlds=rotshare_n_worlds,
                    rotshare_concentration=rotshare_concentration,
                    rotshare_seed=rotshare_seed,
                    rotshare_min_active_players=rotshare_min_active_players,
                    rotshare_mc_center=rotshare_mc_center,
                    data_root=data_root,
                    summary_path=summary_path,
                    mode=resolved_mode,
                )
                return
            raise RuntimeError(f"Primary mode requires rotation_set enabled. {reason}")
        typer.echo(
            f"[rotation_minutes] rotation_set_minutes disabled (config={rotation_config}); keeping baseline.",
            err=True,
        )
        _update_summary_json(
            summary_path,
            {"rotation_set_minutes": {"enabled": False, "mode": resolved_mode}},
        )
        return
    if resolved_model_dir is None:
        if primary_mode:
            reason = "rotation enabled but model_dir missing"
            if config.fallback_mode == "degrade_loudly":
                _fallback_primary_to_baseline(
                    reason=reason,
                    day=day,
                    run_id=run_id,
                    reconcile_team_minutes=reconcile_team_minutes,
                    minutes_output=minutes_output,
                    bundle_dir=bundle_dir,
                    rotshare_quantiles_mode=rotshare_quantiles_mode,
                    rotshare_n_worlds=rotshare_n_worlds,
                    rotshare_concentration=rotshare_concentration,
                    rotshare_seed=rotshare_seed,
                    rotshare_min_active_players=rotshare_min_active_players,
                    rotshare_mc_center=rotshare_mc_center,
                    data_root=data_root,
                    summary_path=summary_path,
                    mode=resolved_mode,
                )
                return
            raise RuntimeError(f"Primary mode requires model_dir. {reason}")
        typer.echo(
            "[rotation_minutes] WARNING: rotation enabled but model_dir missing; keeping baseline.",
            err=True,
        )
        _update_summary_json(
            summary_path,
            {
                "rotation_set_minutes": {
                    "enabled": True,
                    "mode": resolved_mode,
                    "error": "model_dir_missing",
                }
            },
        )
        return

    w = float(blend_weight) if blend_weight is not None else float(config.blend_weight)
    inj_thr = (
        float(injury_coverage_threshold)
        if injury_coverage_threshold is not None
        else float(config.injury_coverage_threshold)
    )
    dnp_thr = (
        float(dnp_tail_minutes_threshold)
        if dnp_tail_minutes_threshold is not None
        else float(config.dnp_tail_minutes_threshold)
    )
    resolved_alloc_mask_mode = str(
        alloc_mask_mode if alloc_mask_mode is not None else config.alloc_mask_mode
    ).strip().lower()
    if resolved_alloc_mask_mode not in ("strict", "not_out"):
        resolved_alloc_mask_mode = "strict"
    resolved_alloc_min_eligible = int(
        alloc_min_eligible if alloc_min_eligible is not None else config.alloc_min_eligible
    )
    resolved_alloc_prior_play_prob_threshold = float(
        alloc_prior_play_prob_threshold
        if alloc_prior_play_prob_threshold is not None
        else config.alloc_prior_play_prob_threshold
    )
    resolved_alloc_baseline_minutes_threshold = float(
        alloc_baseline_minutes_threshold
        if alloc_baseline_minutes_threshold is not None
        else config.alloc_baseline_minutes_threshold
    )
    allow_priors_fallback = bool(config.allow_priors_fallback)
    # PR3: blend-to-baseline is now disabled by default (let transformer signal through)
    enable_blend = bool(config.enable_blend_to_baseline)
    fallback_mode = str(config.fallback_mode)

    typer.echo(
        f"[rotation_minutes] rotation_set_minutes enabled mode={resolved_mode} model_dir={resolved_model_dir} blend_weight={w} "
        f"injury_thr={inj_thr} dnp_tail_thr={dnp_thr} enable_blend={enable_blend} fallback_mode={fallback_mode}",
        err=True,
    )
    typer.echo(
        "[rotation_minutes] alloc_mask "
        f"mode={resolved_alloc_mask_mode} min_eligible={resolved_alloc_min_eligible} "
        f"prior_thr={resolved_alloc_prior_play_prob_threshold:.2f} "
        f"baseline_thr={resolved_alloc_baseline_minutes_threshold:.1f}",
        err=True,
    )

    minutes_features_path = (
        Path(minutes_features_root)
        / day.strftime("%Y-%m-%d")
        / f"run={run_id}"
        / "features.parquet"
    )
    if not minutes_features_path.exists():
        if primary_mode:
            reason = f"minutes features missing at {minutes_features_path}"
            if config.fallback_mode == "degrade_loudly":
                _fallback_primary_to_baseline(
                    reason=reason,
                    day=day,
                    run_id=run_id,
                    reconcile_team_minutes=reconcile_team_minutes,
                    minutes_output=minutes_output,
                    bundle_dir=bundle_dir,
                    rotshare_quantiles_mode=rotshare_quantiles_mode,
                    rotshare_n_worlds=rotshare_n_worlds,
                    rotshare_concentration=rotshare_concentration,
                    rotshare_seed=rotshare_seed,
                    rotshare_min_active_players=rotshare_min_active_players,
                    rotshare_mc_center=rotshare_mc_center,
                    data_root=data_root,
                    summary_path=summary_path,
                    mode=resolved_mode,
                )
                return
            raise RuntimeError(f"Primary mode requires minutes features. {reason}")
        typer.echo(
            f"[rotation_minutes] WARNING: minutes features missing at {minutes_features_path}; keeping baseline.",
            err=True,
        )
        _update_summary_json(
            summary_path,
            {
                "rotation_set_minutes": {
                    "enabled": True,
                    "mode": resolved_mode,
                    "error": "minutes_features_missing",
                }
            },
        )
        return
    minutes_feat = pd.read_parquet(minutes_features_path)
    run_as_of_ts_value = _read_live_run_as_of_ts(minutes_features_path.parent)
    run_as_of_datetime = run_as_of_ts_value.to_pydatetime() if run_as_of_ts_value is not None else None

    # 3) Build rotation model features + write them to the live features root.
    out_day_dir = Path(rotation_features_root) / day.strftime("%Y-%m-%d")
    out_run_dir = out_day_dir / f"run={run_id}"
    rot_features_path = out_run_dir / "features.parquet"

    # Load feature spec (needed for parity check regardless of feature source)
    spec = RotationSetMinutesV1FeatureSpec.load(Path(resolved_model_dir))

    if rot_features_path.exists():
        typer.echo(
            f"[rotation_minutes] using prebuilt rotation features at {rot_features_path}",
            err=True,
        )
        rot_features = pd.read_parquet(rot_features_path)
        if rot_features.empty:
            if primary_mode:
                reason = "rotation_features_empty"
                if config.fallback_mode == "degrade_loudly":
                    _fallback_primary_to_baseline(
                        reason=reason,
                        day=day,
                        run_id=run_id,
                        reconcile_team_minutes=reconcile_team_minutes,
                        minutes_output=minutes_output,
                        bundle_dir=bundle_dir,
                        rotshare_quantiles_mode=rotshare_quantiles_mode,
                        rotshare_n_worlds=rotshare_n_worlds,
                        rotshare_concentration=rotshare_concentration,
                        rotshare_seed=rotshare_seed,
                        rotshare_min_active_players=rotshare_min_active_players,
                        rotshare_mc_center=rotshare_mc_center,
                        data_root=data_root,
                        summary_path=summary_path,
                        mode=resolved_mode,
                    )
                    return
                raise RuntimeError(f"Primary mode requires rotation features. {reason}")
            typer.echo(
                "[rotation_minutes] WARNING: rotation features empty; keeping baseline.",
                err=True,
            )
            _update_summary_json(
                summary_path,
                {
                    "rotation_set_minutes": {
                        "enabled": True,
                        "mode": resolved_mode,
                        "error": "rotation_features_empty",
                    }
                },
            )
            return
    else:
        try:
            game_ids = (
                pd.to_numeric(minutes_feat["game_id"], errors="coerce")
                .dropna()
                .astype(int)
                .astype(str)
                .str.zfill(10)
                .unique()
                .tolist()
            )
            team_ids = set(
                pd.to_numeric(minutes_feat["team_id"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
            player_ids = set(
                pd.to_numeric(minutes_feat["player_id"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
            priors_result = load_rotation_priors_for_live_inference(
                data_root,
                season=season,
                game_date=day.strftime("%Y-%m-%d"),
                game_ids=game_ids,
                team_ids=team_ids,
                player_ids=player_ids,
                allow_priors_fallback=allow_priors_fallback,
            )
            team_priors = priors_result.team_priors
            player_priors = priors_result.player_priors
            if priors_result.used_latest_fallback:
                typer.echo(
                    f"[rotation_minutes] WARNING: {priors_result.warning_message}",
                    err=True,
                )

            historical_features = _load_rotation_historical_features_for_dnp(
                data_root,
                season=season,
                target_day=day,
                team_ids=team_ids,
                player_ids=player_ids,
            )
            feature_result = build_rotation_set_minutes_v1_features(
                minutes_feat,
                team_priors=team_priors,
                player_priors=player_priors,
                feature_columns=spec.feature_columns,
                historical_features=historical_features,
            )
        except RotationLiveFeaturesError as exc:
            if primary_mode:
                reason = f"feature_build_failed:{exc}"
                if config.fallback_mode == "degrade_loudly":
                    _fallback_primary_to_baseline(
                        reason=reason,
                        day=day,
                        run_id=run_id,
                        reconcile_team_minutes=reconcile_team_minutes,
                        minutes_output=minutes_output,
                        bundle_dir=bundle_dir,
                        rotshare_quantiles_mode=rotshare_quantiles_mode,
                        rotshare_n_worlds=rotshare_n_worlds,
                        rotshare_concentration=rotshare_concentration,
                        rotshare_seed=rotshare_seed,
                        rotshare_min_active_players=rotshare_min_active_players,
                        rotshare_mc_center=rotshare_mc_center,
                        data_root=data_root,
                        summary_path=summary_path,
                        mode=resolved_mode,
                    )
                    return
                raise RuntimeError(f"Primary mode feature build failed: {exc}") from exc
            typer.echo(
                f"[rotation_minutes] WARNING: feature build failed ({exc}); keeping baseline.",
                err=True,
            )
            _update_summary_json(
                summary_path,
                {
                    "rotation_set_minutes": {
                        "enabled": True,
                        "mode": resolved_mode,
                        "error": f"feature_build_failed:{exc}",
                    }
                },
            )
            return

        rot_features = feature_result.features
        out_run_dir.mkdir(parents=True, exist_ok=True)
        rot_features.to_parquet(rot_features_path, index=False)
        (out_run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "generated_at": pd.Timestamp.now(tz=UTC).isoformat(),
                    "run_id": run_id,
                    "model_dir": str(Path(resolved_model_dir).expanduser()),
                    "counts": {"rows": int(len(rot_features))},
                    "dropped_extra_columns": feature_result.dropped_extra_columns,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    # 3b) Compute parity report (detect-only, does not block scoring)
    parity_report = compute_parity_report(
        rot_features,
        model_dir=Path(resolved_model_dir),
        game_date=day.strftime("%Y-%m-%d"),
        run_id=run_id,
        expected_features=spec.feature_columns,
    )
    typer.echo(f"[rotation_minutes] {parity_report.summary_line()}", err=True)

    # Write parity report artifact
    parity_report_path = write_parity_report(parity_report, minutes_out_dir)
    typer.echo(f"[rotation_minutes] wrote parity report -> {parity_report_path}", err=True)

    if parity_report.n_warnings > 0:
        typer.echo(
            f"[rotation_minutes] parity warnings ({parity_report.n_warnings}): "
            f"{[w.column for w in parity_report.warnings[:5]]}",
            err=True,
        )

    # 4) Predict rotation minutes (p50) and align to baseline output rows.
    try:
        rot_result = predict_rotation_minutes(
            rot_features,
            model_dir=Path(resolved_model_dir),
            device=str(device),
            batch_size=int(batch_size),
            return_aux=True,
            alloc_mask_mode=resolved_alloc_mask_mode,
            alloc_min_eligible=resolved_alloc_min_eligible,
            alloc_prior_play_prob_threshold=resolved_alloc_prior_play_prob_threshold,
            alloc_baseline_minutes_threshold=resolved_alloc_baseline_minutes_threshold,
        )
        # Handle both DataFrame and RotationSetAuxOutputs return types
        if isinstance(rot_result, RotationSetAuxOutputs):
            rot_scored = rot_result.player_df
        else:
            rot_scored = rot_result
    except Exception as exc:  # noqa: BLE001
        if primary_mode:
            reason = f"predict_failed:{exc}"
            if config.fallback_mode == "degrade_loudly":
                _fallback_primary_to_baseline(
                    reason=reason,
                    day=day,
                    run_id=run_id,
                    reconcile_team_minutes=reconcile_team_minutes,
                    minutes_output=minutes_output,
                    bundle_dir=bundle_dir,
                    rotshare_quantiles_mode=rotshare_quantiles_mode,
                    rotshare_n_worlds=rotshare_n_worlds,
                    rotshare_concentration=rotshare_concentration,
                    rotshare_seed=rotshare_seed,
                    rotshare_min_active_players=rotshare_min_active_players,
                    rotshare_mc_center=rotshare_mc_center,
                    data_root=data_root,
                    summary_path=summary_path,
                    mode=resolved_mode,
                )
                return
            raise RuntimeError(f"Primary mode predict failed: {exc}") from exc
        typer.echo(
            f"[rotation_minutes] WARNING: rotation model predict failed ({exc}); keeping baseline.",
            err=True,
        )
        _update_summary_json(
            summary_path,
            {
                "rotation_set_minutes": {
                    "enabled": True,
                    "mode": resolved_mode,
                    "error": f"predict_failed:{exc}",
                }
            },
        )
        return

    # Log gate_prob presence for diagnostics
    gate_prob_present = "gate_prob" in rot_scored.columns
    typer.echo(f"[rotation_minutes] gate_prob present: {gate_prob_present}", err=True)

    # Build rot_pred with minutes and optionally gate_prob for sparsify guardrail
    rot_pred_cols = ["game_id", "team_id", "player_id", "pred_minutes"]
    if gate_prob_present:
        rot_pred_cols.append("gate_prob")
    for aux_col in ("gate_logit", "share_logit"):
        if aux_col in rot_scored.columns:
            rot_pred_cols.append(aux_col)
    rot_pred = rot_scored.loc[:, rot_pred_cols].rename(
        columns={"pred_minutes": "rotation_minutes_p50"}
    )
    rot_pred["rotation_minutes_p50"] = (
        pd.to_numeric(rot_pred["rotation_minutes_p50"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    if gate_prob_present:
        rot_pred["gate_prob"] = (
            pd.to_numeric(rot_pred["gate_prob"], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )

        if config.projected_starter_boost_enabled:
            rot_pred, boost_stats = apply_projected_starter_boost(
                rot_pred,
                rot_features=rot_features,
                model_dir=Path(resolved_model_dir),
                alloc_mask_mode=resolved_alloc_mask_mode,
                alloc_min_eligible=int(resolved_alloc_min_eligible),
                alloc_prior_play_prob_threshold=float(resolved_alloc_prior_play_prob_threshold),
                alloc_baseline_minutes_threshold=float(resolved_alloc_baseline_minutes_threshold),
                gate_prob_threshold=float(config.projected_starter_gate_prob_boost_threshold),
                gate_logit_boost=float(config.projected_starter_gate_logit_boost),
            )
            if boost_stats.skipped_reason:
                typer.echo(
                    f"[rotation_minutes] projected starter boost skipped: {boost_stats.skipped_reason}",
                    err=True,
                )
            elif boost_stats.boosted_players > 0:
                typer.echo(
                    f"[rotation_minutes] projected starter boost: boosted={boost_stats.boosted_players} "
                    f"(gate_prob<{boost_stats.gate_prob_threshold:g}, gate_logit={boost_stats.gate_logit_boost:g}) "
                    f"recomputed_team_games={boost_stats.recomputed_team_games}",
                    err=True,
                )

    for aux_col in ("gate_logit", "share_logit"):
        if aux_col in rot_pred.columns:
            rot_pred[aux_col] = (
                pd.to_numeric(rot_pred[aux_col], errors="coerce")
                .fillna(0.0)
                .astype(float)
            )
    if gate_prob_present:
        rot_pred["rotation_size_hat_team"] = (
            rot_pred.groupby(["game_id", "team_id"], sort=False)["gate_prob"]
            .transform("sum")
            .astype(float)
        )
        rot_pred["rotation_size_hat_source"] = "gate_prob_sum"
    else:
        rot_pred["rotation_size_hat_team"] = (
            (rot_pred["rotation_minutes_p50"] >= float(ROTATION_THRESHOLD_MINUTES))
            .groupby([rot_pred["game_id"], rot_pred["team_id"]], sort=False)
            .transform("sum")
            .astype(float)
        )
        rot_pred["rotation_size_hat_source"] = f"minutes_ge_{float(ROTATION_THRESHOLD_MINUTES):g}"

    rotation_size_hat_mean = (
        float(
            rot_pred.drop_duplicates(subset=["game_id", "team_id"])["rotation_size_hat_team"].mean()
        )
        if not rot_pred.empty
        else None
    )

    if primary_mode:
        out_df = _build_primary_minutes_frame(minutes_feat, day=day)

        # Prepare explicit real baseline column for guardrails if available.
        # We prefer 'minutes_p50_model' (raw transformer output) over derived baseline.
        guardrail_baseline_col = "baseline_minutes_p50"
        if "minutes_p50_model" in minutes_feat.columns:
            out_df["minutes_p50_model"] = (
                pd.to_numeric(minutes_feat["minutes_p50_model"], errors="coerce")
                .fillna(0.0)
                .astype(float)
            )
            guardrail_baseline_col = "minutes_p50_model"
        else:
            typer.echo(
                "[rotation_minutes] WARNING: minutes_p50_model missing from features; falling back to baseline_minutes_p50 for guardrails",
                err=True,
            )

        # Extract baseline minutes from the pre-rotation minutes model for pathology fallback.
        # This ensures guardrails can fall back to a real baseline, not the rotation output.
        # Priority: minutes_p50_model > minutes_final > minutes_p50 (if present pre-rotation)
        baseline_result = _extract_baseline_minutes(minutes_feat)
        baseline_source_col = baseline_result.source_col
        baseline_gt0_count = baseline_result.gt0_count
        baseline_min_val = baseline_result.min_val
        baseline_median_val = baseline_result.median_val
        baseline_max_val = baseline_result.max_val

        if baseline_source_col == "none":
            available_cols = [c for c in minutes_feat.columns if "minutes" in c.lower()]
            typer.echo(
                f"[rotation_minutes] WARNING: no baseline minutes column found with >0 values; "
                f"baseline_minutes_p50 set to 0.0 (pathology fallback may not be effective). "
                f"Available minutes columns: {available_cols}",
                err=True,
            )
        else:
            typer.echo(
                f"[rotation_minutes] baseline_minutes_p50: source={baseline_source_col} "
                f"gt0={baseline_gt0_count} min={baseline_min_val:.2f} "
                f"median={baseline_median_val:.2f} max={baseline_max_val:.2f}",
                err=True,
            )
        out_df["baseline_minutes_p50"] = baseline_result.values

        out_df = out_df.merge(rot_pred, on=["game_id", "team_id", "player_id"], how="left")
        out_df["minutes_p50"] = pd.to_numeric(
            out_df["rotation_minutes_p50"], errors="coerce"
        ).fillna(0.0)

        # ESPN OUT override: apply before hard-zero enforcement.
        espn_out_count = 0
        espn_matched_count = 0
        try:
            espn_out_players = _load_espn_out_players(
                day.date() if hasattr(day, "date") else day,
                data_root=data_root,
                run_as_of_ts=run_as_of_datetime,
            )
            espn_out_count = len(espn_out_players) if espn_out_players else 0
            if espn_out_players and "player_name" in out_df.columns:
                normalized_names = out_df["player_name"].astype(str).map(_normalize_name_for_matching)
                espn_raw_mask = normalized_names.isin(espn_out_players)
                espn_matched_count = int(espn_raw_mask.sum())
                allow_override = _espn_override_allowed(out_df)
                espn_mask = espn_raw_mask & allow_override
                espn_applied_count = int(espn_mask.sum())
                if espn_applied_count > 0:
                    out_df.loc[espn_mask, "is_out"] = 1
                    out_df.loc[espn_mask, "status"] = "OUT"
                    typer.echo(
                        f"[rotation_minutes] ESPN OUT override: loaded={espn_out_count} matched={espn_matched_count} applied={espn_applied_count}",
                        err=True,
                    )
                else:
                    if espn_matched_count > 0:
                        typer.echo(
                            f"[rotation_minutes] ESPN OUT: loaded={espn_out_count} matched={espn_matched_count} applied=0 (override not allowed)",
                            err=True,
                        )
                    else:
                        typer.echo(
                            f"[rotation_minutes] ESPN OUT: loaded={espn_out_count} matched=0 (no overlap with slate)",
                            err=True,
                        )
            elif espn_out_count > 0:
                typer.echo(
                    f"[rotation_minutes] ESPN OUT: loaded={espn_out_count} but player_name column missing",
                    err=True,
                )
        except Exception as exc:  # noqa: BLE001
            typer.echo(
                f"[rotation_minutes] WARNING: ESPN OUT load failed ({exc}); using feature-time is_out",
                err=True,
            )

        out_mask = _derive_out_mask(out_df)
        out_df = _apply_out_row_overrides(out_df, out_mask)
        out_df.loc[out_mask, "minutes_p50"] = 0.0

        if config.dnp_override_enabled:
            dnp_cols_needed = ["consecutive_active_dnp", "active_but_dnp_rate_last10"]
            dnp_cols_present = [c for c in dnp_cols_needed if c in rot_features.columns]
            if len(dnp_cols_present) == len(dnp_cols_needed):
                dnp_check = out_df.merge(
                    rot_features[["game_id", "team_id", "player_id"] + dnp_cols_needed],
                    on=["game_id", "team_id", "player_id"],
                    how="left",
                )
                consec_dnp = pd.to_numeric(
                    dnp_check["consecutive_active_dnp"], errors="coerce"
                ).fillna(0)
                dnp_rate = pd.to_numeric(
                    dnp_check["active_but_dnp_rate_last10"], errors="coerce"
                ).fillna(0)
                dnp_override_mask = (consec_dnp >= 3) & (dnp_rate >= 0.8)
                n_overridden = int(dnp_override_mask.sum())
                if n_overridden > 0:
                    out_df.loc[dnp_override_mask, "minutes_p50"] = 0.0
                    typer.echo(
                        f"[rotation_minutes] DNP override: zeroed {n_overridden} players with "
                        f"consecutive_active_dnp>=3 AND dnp_rate>=0.8",
                        err=True,
                    )

        scaled_minutes, scale_summary = _scale_minutes_to_team_target(
            out_df, out_df["minutes_p50"], team_target=240.0
        )
        out_df["minutes_p50"] = scaled_minutes

        # Apply rotation minutes guardrails (including sparsify) UNCONDITIONALLY in primary mode.
        # This ensures guardrails always run and stats are always visible in logs + summary.json.
        # Use baseline_minutes_p50 (from pre-rotation minutes model) for pathology fallback.
        guardrail_result = apply_rotation_minutes_guardrails(
            out_df,
            rotation_p50_col="minutes_p50",
            baseline_p50_col=guardrail_baseline_col,
            minutes_features_row_missing_col="minutes_features_row_missing",
            injury_snapshot_missing_col="injury_snapshot_missing",
            out_flag_col="is_out",
            status_col="status",
            blend_weight=0.0,  # No blend in primary mode
            injury_coverage_threshold=inj_thr,
            dnp_tail_minutes_threshold=dnp_thr,
            cap_max_minutes=None,  # Already scaled to 240
            enable_blend_to_baseline=False,  # No blend in primary mode
            sparsify_enable=config.sparsify_enable,
            sparsify_topk=config.sparsify_topk,
            sparsify_tau=config.sparsify_tau,
            sparsify_kmax=config.sparsify_kmax,
            sparsify_min_keep=config.sparsify_min_keep,
            sparsify_use_col=config.sparsify_use_col,
        )
        out_df["minutes_p50"] = guardrail_result.minutes_p50
        guardrail_stats = guardrail_result.summary
        typer.echo(
            f"[rotation_minutes] guardrails: {json.dumps(guardrail_stats, default=str)}",
            err=True,
        )

        out_mask = _derive_out_mask(out_df)
        out_df = _apply_out_row_overrides(out_df, out_mask)
        if str(config.play_prob_source).strip().lower() == "gate_prob" and "gate_prob" in out_df.columns:
            play_prob = pd.to_numeric(out_df["gate_prob"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            play_prob = play_prob.where(~out_mask.to_numpy(dtype=bool), 0.0)
        else:
            play_prob, _ = _derive_gate_probs(out_df["minutes_p50"], out_mask=out_mask)

        if str(config.rotation_prob_source).strip().lower() == "gate_prob" and "gate_prob" in out_df.columns:
            rotation_prob = pd.to_numeric(out_df["gate_prob"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            rotation_prob = rotation_prob.where(~out_mask.to_numpy(dtype=bool), 0.0)
        else:
            _, rotation_prob = _derive_gate_probs(out_df["minutes_p50"], out_mask=out_mask)

        out_df["play_prob"] = play_prob.astype(float)
        out_df["rotation_prob"] = rotation_prob.astype(float)
        out_df["minutes_alloc_mode"] = "rotation_set"
        out_df["model_run_id"] = str(Path(resolved_model_dir).name)
        out_df = out_df.drop(columns=["rotation_minutes_p50"], errors="ignore")

        # Preserve baseline tail shape in primary mode: map new p50 onto baseline
        # p10/p90 deltas instead of collapsing all quantiles to p50.
        tail_base_p50_col = (
            guardrail_baseline_col
            if guardrail_baseline_col in out_df.columns
            else (
                "baseline_minutes_p50"
                if "baseline_minutes_p50" in out_df.columns
                else "minutes_p50"
            )
        )
        base_p50_tail = pd.to_numeric(out_df[tail_base_p50_col], errors="coerce").fillna(0.0)
        base_p10 = (
            pd.to_numeric(out_df["minutes_p10"], errors="coerce").fillna(base_p50_tail)
            if "minutes_p10" in out_df.columns
            else base_p50_tail.copy()
        )
        base_p90 = (
            pd.to_numeric(out_df["minutes_p90"], errors="coerce").fillna(base_p50_tail)
            if "minutes_p90" in out_df.columns
            else base_p50_tail.copy()
        )
        # Primary mode often has no baseline quantiles in live features. When tails
        # are degenerate, synthesize baseline deltas from volatility proxies.
        degenerate_tail = (
            float((base_p50_tail - base_p10).abs().max()) < 1e-9
            and float((base_p90 - base_p50_tail).abs().max()) < 1e-9
        )
        if degenerate_tail:
            iqr5 = pd.to_numeric(out_df.get("roll_iqr_5"), errors="coerce").fillna(0.0)
            std5 = pd.to_numeric(out_df.get("rotation_minutes_std_5g"), errors="coerce").fillna(0.0)
            sigma_from_iqr = iqr5 / 1.349
            sigma_est = sigma_from_iqr.where(sigma_from_iqr > 0.0, std5)
            sigma_fallback = (0.12 * base_p50_tail + 0.8).clip(lower=0.75, upper=4.5)
            sigma = sigma_est.where(sigma_est > 0.0, sigma_fallback)
            delta = (1.2815515655446004 * sigma).clip(lower=0.75, upper=8.0)
            base_p10 = (base_p50_tail - delta).clip(lower=0.0)
            base_p90 = (base_p50_tail + delta).clip(lower=0.0, upper=48.0)
        p10_new, p90_new = _derive_minutes_tails(
            base_p10=base_p10,
            base_p50=base_p50_tail,
            base_p90=base_p90,
            new_p50=out_df["minutes_p50"],
        )
        out_df["minutes_p10"] = p10_new
        out_df["minutes_p90"] = p90_new
        out_df["minutes_p10_cond"] = p10_new
        out_df["minutes_p50_cond"] = out_df["minutes_p50"]
        out_df["minutes_p90_cond"] = p90_new
        out_df["minutes_p10_uncond"] = p10_new
        out_df["minutes_p50_uncond"] = out_df["minutes_p50"]
        out_df["minutes_p90_uncond"] = p90_new

        out_df = _select_minutes_columns(out_df, minutes_output)
        out_df.to_parquet(minutes_path, index=False)

        _update_summary_json(
            summary_path,
            {
                "date": day.strftime("%Y-%m-%d"),
                "run_id": run_id,
                "generated_at": pd.Timestamp.now(tz=UTC).isoformat(),
                "rotation_set_minutes": {
                    "enabled": True,
                    "mode": resolved_mode,
                    "model_dir": str(Path(resolved_model_dir).expanduser()),
                    "alloc_mask_mode": resolved_alloc_mask_mode,
                    "alloc_min_eligible": int(resolved_alloc_min_eligible),
                    "alloc_prior_play_prob_threshold": float(
                        resolved_alloc_prior_play_prob_threshold
                    ),
                    "alloc_baseline_minutes_threshold": float(
                        resolved_alloc_baseline_minutes_threshold
                    ),
                    "projected_starter_boost_enabled": bool(config.projected_starter_boost_enabled),
                    "projected_starter_gate_prob_boost_threshold": float(
                        config.projected_starter_gate_prob_boost_threshold
                    ),
                    "projected_starter_gate_logit_boost": float(config.projected_starter_gate_logit_boost),
                    "play_prob_source": str(config.play_prob_source),
                    "rotation_prob_source": str(config.rotation_prob_source),
                    "dnp_override_enabled": bool(config.dnp_override_enabled),
                    "team_scale": scale_summary,
                    "espn_out_count": espn_out_count,
                    "espn_matched_count": espn_matched_count,
                    "guardrails": guardrail_stats,
                    "rotation_size_hat_source": str(rot_pred["rotation_size_hat_source"].iloc[0]) if not rot_pred.empty else None,
                    "rotation_size_hat_mean": rotation_size_hat_mean,
                    "baseline_stats": {
                        "source_col": baseline_source_col,
                        "gt0_count": baseline_gt0_count,
                        "min": baseline_min_val,
                        "median": baseline_median_val,
                        "max": baseline_max_val,
                    },
                },
                "counts": {
                    "rows": int(len(out_df)),
                    "games": int(out_df["game_id"].nunique()) if "game_id" in out_df.columns else 0,
                    "teams": int(out_df["team_id"].nunique()) if "team_id" in out_df.columns else 0,
                },
            },
        )

        # Track degradation from guardrails as well as scale issues
        guardrail_degraded = guardrail_result.degraded
        degraded = scale_summary.get("zero_sum_teams", 0) > 0 or guardrail_degraded
        degraded_reasons = []
        if scale_summary.get("zero_sum_teams", 0) > 0:
            degraded_reasons.append(f"zero_sum_teams:{scale_summary.get('zero_sum_teams', 0)}")
        if guardrail_degraded:
            degraded_reasons.extend(guardrail_result.degraded_reasons)

        provenance = build_provenance(
            alloc_mode="rotation_set",
            model_dir=resolved_model_dir,
            run_id=run_id,
            game_date=day.strftime("%Y-%m-%d"),
            degraded=degraded,
            degraded_reason="; ".join(degraded_reasons) if degraded_reasons else "",
            extras={
                "mode": resolved_mode,
                "alloc_mask_mode": resolved_alloc_mask_mode,
                "alloc_min_eligible": int(resolved_alloc_min_eligible),
                "alloc_prior_play_prob_threshold": float(
                    resolved_alloc_prior_play_prob_threshold
                ),
                "alloc_baseline_minutes_threshold": float(
                    resolved_alloc_baseline_minutes_threshold
                ),
                "team_scale": scale_summary,
                "espn_out_count": espn_out_count,
                "espn_matched_count": espn_matched_count,
                "guardrails": guardrail_stats,
            },
        )
        inject_provenance_into_summary(summary_path, provenance)
        inject_parity_into_summary(summary_path, parity_report)

        typer.echo(f"[rotation_minutes] wrote primary minutes -> {minutes_path}", err=True)
        return

    # Guardrail inputs from minutes features (lineup fields) + rotation features (missing flags).
    guard = base_df.merge(rot_pred, on=["game_id", "team_id", "player_id"], how="left")
    guard = guard.merge(
        rot_features.loc[
            :,
            [
                "game_id",
                "team_id",
                "player_id",
                "minutes_features_row_missing",
                "injury_snapshot_missing",
            ],
        ],
        on=["game_id", "team_id", "player_id"],
        how="left",
    )
    guard = guard.merge(
        minutes_feat.loc[
            :,
            [
                "game_id",
                "team_id",
                "player_id",
                "lineup_timestamp",
                "lineup_status",
                "lineup_roster_status",
            ],
        ],
        on=["game_id", "team_id", "player_id"],
        how="left",
    )
    injury_cols = [c for c in ["is_out", "status", "injury_row_present"] if c in minutes_feat.columns]
    if injury_cols:
        guard = guard.merge(
            minutes_feat.loc[:, ["game_id", "team_id", "player_id", *injury_cols]],
            on=["game_id", "team_id", "player_id"],
            how="left",
        )
    else:
        guard["is_out"] = 0
        guard["status"] = pd.NA

    # 4b) ESPN OUT override: Load ESPN injuries and apply same override as baseline scorer.
    # ESPN updates faster than NBA injury reports; this catches late scratches.
    espn_out_count = 0
    espn_matched_count = 0
    try:
        espn_out_players = _load_espn_out_players(
            day.date() if hasattr(day, "date") else day,
            data_root=data_root,
            run_as_of_ts=run_as_of_datetime,
        )
        espn_out_count = len(espn_out_players) if espn_out_players else 0
        if espn_out_players and "player_name" in guard.columns:
            normalized_names = guard["player_name"].astype(str).map(_normalize_name_for_matching)
            espn_raw_mask = normalized_names.isin(espn_out_players)
            espn_matched_count = int(espn_raw_mask.sum())
            allow_override = _espn_override_allowed(guard)
            espn_mask = espn_raw_mask & allow_override
            espn_applied_count = int(espn_mask.sum())
            if espn_applied_count > 0:
                guard.loc[espn_mask, "is_out"] = 1
                guard.loc[espn_mask, "status"] = "OUT"
                espn_names = guard.loc[espn_mask, "player_name"].tolist()
                typer.echo(
                    f"[rotation_minutes] ESPN OUT override: loaded={espn_out_count} matched={espn_matched_count} applied={espn_applied_count} "
                    f"marked OUT: {espn_names[:10]}",
                    err=True,
                )
            else:
                if espn_matched_count > 0:
                    typer.echo(
                        f"[rotation_minutes] ESPN OUT: loaded={espn_out_count} matched={espn_matched_count} applied=0 (override not allowed)",
                        err=True,
                    )
                else:
                    typer.echo(
                        f"[rotation_minutes] ESPN OUT: loaded={espn_out_count} matched=0 (no overlap with slate)",
                        err=True,
                    )
        elif espn_out_count > 0:
            typer.echo(
                f"[rotation_minutes] ESPN OUT: loaded={espn_out_count} but player_name column missing",
                err=True,
            )
    except Exception as exc:  # noqa: BLE001
        typer.echo(
            f"[rotation_minutes] WARNING: ESPN OUT load failed ({exc}); using feature-time is_out",
            err=True,
        )


    # Default any missing rotation preds to baseline p50 (keeps totals sane).

    baseline_p50_col = (
        "minutes_p50_cond" if "minutes_p50_cond" in guard.columns else "minutes_p50"
    )
    guard["rotation_minutes_p50"] = pd.to_numeric(
        guard["rotation_minutes_p50"], errors="coerce"
    )
    guard["rotation_minutes_p50"] = guard["rotation_minutes_p50"].fillna(
        pd.to_numeric(guard[baseline_p50_col], errors="coerce").fillna(0.0)
    )

    cap_max_minutes = None
    if str(reconcile_team_minutes).strip().lower() != "none":
        try:
            reconcile_cfg = load_reconcile_config(reconcile_config)
            cap_max_minutes = float(reconcile_cfg.bounds.hard_cap)
        except Exception as exc:  # noqa: BLE001
            typer.echo(
                f"[rotation_minutes] WARNING: failed to load reconcile_config={reconcile_config}; "
                f"using cap_max_minutes=48 ({exc})",
                err=True,
            )
            cap_max_minutes = 48.0

    guardrail = apply_rotation_minutes_guardrails(
        guard,
        rotation_p50_col="rotation_minutes_p50",
        baseline_p50_col=baseline_p50_col,
        minutes_features_row_missing_col="minutes_features_row_missing",
        injury_snapshot_missing_col="injury_snapshot_missing",
        blend_weight=w,
        injury_coverage_threshold=inj_thr,
        dnp_tail_minutes_threshold=dnp_thr,
        cap_max_minutes=cap_max_minutes,
        enable_blend_to_baseline=enable_blend,  # PR3: disabled by default
        # Sparsify parameters from config (opt-in, default disabled)
        sparsify_enable=config.sparsify_enable,
        sparsify_topk=config.sparsify_topk,
        sparsify_tau=config.sparsify_tau,
        sparsify_kmax=config.sparsify_kmax,
        sparsify_min_keep=config.sparsify_min_keep,
        sparsify_use_col=config.sparsify_use_col,
    )
    new_p50 = guardrail.minutes_p50

    # PR3: Handle degradation according to fallback_mode
    if guardrail.degraded:
        degraded_msg = (
            f"[rotation_minutes] DEGRADED: {len(guardrail.degraded_reasons)} fallback(s) triggered: "
            f"{guardrail.degraded_reasons}"
        )
        if fallback_mode == "fail_closed":
            # In fail_closed mode, raise an error instead of continuing with degraded output
            raise RuntimeError(
                f"Guardrail triggered in fail_closed mode. {degraded_msg}\n"
                "To continue with degraded output, set fallback_mode='degrade_loudly' in config."
            )
        else:
            # degrade_loudly: log warning but continue
            typer.echo(degraded_msg, err=True)

    # 4b) Post-guardrail DNP override: for players with strong DNP-CD signals,
    # override their minutes to 0 regardless of baseline. This allows us to use
    # baseline for normal players while still catching players like Kuminga who
    # are healthy scratches with a consistent DNP pattern.
    if config.dnp_override_enabled:
        dnp_cols_needed = ["consecutive_active_dnp", "active_but_dnp_rate_last10"]
        dnp_cols_present = [c for c in dnp_cols_needed if c in rot_features.columns]
        if len(dnp_cols_present) == len(dnp_cols_needed):
            # Merge DNP features for override check
            dnp_check = guard.merge(
                rot_features[["game_id", "team_id", "player_id"] + dnp_cols_needed],
                on=["game_id", "team_id", "player_id"],
                how="left",
            )
            consec_dnp = pd.to_numeric(
                dnp_check["consecutive_active_dnp"], errors="coerce"
            ).fillna(0)
            dnp_rate = pd.to_numeric(
                dnp_check["active_but_dnp_rate_last10"], errors="coerce"
            ).fillna(0)
            # Override to 0 if: consecutive DNP >= 3 AND DNP rate >= 80%
            dnp_override_mask = (consec_dnp >= 3) & (dnp_rate >= 0.8)
            n_overridden = int(dnp_override_mask.sum())
            if n_overridden > 0:
                new_p50 = new_p50.where(~dnp_override_mask, 0.0)
                typer.echo(
                    f"[rotation_minutes] DNP override: zeroed {n_overridden} players with "
                    f"consecutive_active_dnp>=3 AND dnp_rate>=0.8",
                    err=True,
                )

    # 5) Map rotation p50 to minutes quantiles using baseline tail deltas.
    out_df = base_df.copy()
    if "rotation_size_hat_team" in rot_pred.columns:
        team_hat = (
            rot_pred[["game_id", "team_id", "rotation_size_hat_team", "rotation_size_hat_source"]]
            .drop_duplicates(subset=["game_id", "team_id"], keep="last")
        )
        out_df = out_df.merge(team_hat, on=["game_id", "team_id"], how="left")
    out_df[baseline_p50_col] = new_p50.to_numpy(dtype=float)
    if "minutes_p50" in out_df.columns:
        out_df["minutes_p50"] = out_df[baseline_p50_col]

    if {"minutes_p10", "minutes_p50", "minutes_p90"}.issubset(out_df.columns):
        p10_new, p90_new = _derive_minutes_tails(
            base_p10=out_df["minutes_p10"],
            base_p50=out_df["minutes_p50"],
            base_p90=out_df["minutes_p90"],
            new_p50=out_df["minutes_p50"],
        )
        out_df["minutes_p10"] = p10_new
        out_df["minutes_p90"] = p90_new
        if "minutes_p10_cond" in out_df.columns:
            out_df["minutes_p10_cond"] = p10_new
        if "minutes_p90_cond" in out_df.columns:
            out_df["minutes_p90_cond"] = p90_new

    out_df = _apply_out_row_overrides(out_df, _derive_out_mask(out_df))
    out_df.to_parquet(minutes_path, index=False)

    # 5b) Compute delta diagnostics: compare final p50 to original baseline.
    # This quantifies how much the rotation overlay actually moved minutes.
    # Also tracks pre-blend raw deltas to distinguish blend masking from agreement.
    delta_diagnostics: list[dict[str, Any]] = []
    try:
        # Build a frame with both baseline and final p50 for comparison.
        diag_df = base_df[["game_id", "team_id", "player_id"]].copy()
        diag_df["_baseline_p50"] = pd.to_numeric(
            base_df[baseline_p50_col], errors="coerce"
        ).fillna(0.0)
        diag_df["_final_p50"] = new_p50.to_numpy(dtype=float)
        # Raw rotation p50 before guardrails/blend (from guard df)
        if "rotation_minutes_p50" in guard.columns:
            diag_df["_raw_rotation_p50"] = pd.to_numeric(
                guard["rotation_minutes_p50"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=float)
        # For directional accuracy, check if realized minutes exist (replay/historical).
        # In live mode, "minutes" (actual) won't be populated.
        actual_col = "minutes" if "minutes" in base_df.columns else None
        if actual_col and base_df[actual_col].notna().sum() > 0:
            diag_df["_actual_minutes"] = pd.to_numeric(
                base_df[actual_col], errors="coerce"
            ).fillna(0.0)
        else:
            actual_col = None

        # Build blend_applied map from guardrails team_game_summary
        blend_applied_map: dict[tuple[str, int], bool] | None = None
        if (
            hasattr(guardrail, "team_game_summary")
            and "blend_applied" in guardrail.team_game_summary.columns
        ):
            tg = guardrail.team_game_summary
            blend_applied_map = {
                (str(row["game_id"]), int(row["team_id"])): bool(row["blend_applied"])
                for _, row in tg.iterrows()
            }

        delta_diagnostics = _compute_delta_diagnostics(
            diag_df,
            baseline_p50_col="_baseline_p50",
            final_p50_col="_final_p50",
            raw_rotation_p50_col="_raw_rotation_p50" if "_raw_rotation_p50" in diag_df.columns else None,
            blend_applied_map=blend_applied_map,
            actual_minutes_col="_actual_minutes" if actual_col else None,
        )
        if delta_diagnostics:
            _log_delta_diagnostics(delta_diagnostics)
    except Exception as exc:  # noqa: BLE001
        typer.echo(
            f"[rotation_minutes] WARNING: delta diagnostics failed ({exc}); continuing.",
            err=True,
        )


    # 6) Logging + summary.json augmentation.

    typer.echo(f"[rotation_minutes] guardrails: {guardrail.summary}", err=True)
    if "pathology_fallback" in guardrail.team_game_summary.columns:
        pathological = guardrail.team_game_summary.loc[
            guardrail.team_game_summary["pathology_fallback"]
        ].copy()
        if not pathological.empty:
            cols = [
                "game_id",
                "team_id",
                "pathology_reason",
                "max_minutes",
                "n_gt_40",
                "top5_sum",
            ]
            cols = [c for c in cols if c in pathological.columns]
            sample = (
                pathological.sort_values(["max_minutes", "top5_sum"], ascending=False)
                .head(20)
                .loc[:, cols]
                .to_dict(orient="records")
            )
            typer.echo(
                f"[rotation_minutes] pathology_fallback_team_games (n={len(pathological)}): {sample}",
                err=True,
            )
    blended = guardrail.team_game_summary.loc[
        guardrail.team_game_summary["blend_applied"]
    ].copy()
    if not blended.empty:
        sample = (
            blended.sort_values(
                ["injury_coverage_rate", "row_fallback_count"], ascending=[True, False]
            )
            .head(20)
            .to_dict(orient="records")
        )
        typer.echo(
            f"[rotation_minutes] blended_team_games (n={len(blended)}): {sample}",
            err=True,
        )
    if not guardrail.tail_minutes_top.empty:
        rows = guardrail.tail_minutes_top.head(20).to_dict(orient="records")
        typer.echo(
            f"[rotation_minutes] top_tail_minutes_team_games (n={len(rows)}): {rows}",
            err=True,
        )

    _update_summary_json(
        summary_path,
        {
            "rotation_set_minutes": {
                "enabled": True,
                "mode": resolved_mode,
                "model_dir": str(Path(resolved_model_dir).expanduser()),
                "alloc_mask_mode": resolved_alloc_mask_mode,
                "alloc_min_eligible": int(resolved_alloc_min_eligible),
                "alloc_prior_play_prob_threshold": float(
                    resolved_alloc_prior_play_prob_threshold
                ),
                "alloc_baseline_minutes_threshold": float(
                    resolved_alloc_baseline_minutes_threshold
                ),
                "blend_weight": w,
                "injury_coverage_threshold": inj_thr,
                "dnp_tail_minutes_threshold": dnp_thr,
                "rotation_size_hat_source": str(rot_pred["rotation_size_hat_source"].iloc[0]) if not rot_pred.empty else None,
                "rotation_size_hat_mean": rotation_size_hat_mean,
                "guardrails": guardrail.summary,
                "tail_minutes_team_games_top": (
                    guardrail.tail_minutes_top.head(20).to_dict(orient="records")
                    if not guardrail.tail_minutes_top.empty
                    else []
                ),
                # Delta diagnostics: per-team-game metrics quantifying rotation overlay impact.
                "delta_diagnostics": {
                    "per_team_game": delta_diagnostics,
                },
            }
        },
    )

    # Inject provenance stamp for artifact traceability
    # PR3: Use guardrail's degradation tracking for accurate fallback reporting
    provenance = build_provenance(
        alloc_mode="rotation_set",
        model_dir=resolved_model_dir,
        run_id=run_id,
        game_date=day.strftime("%Y-%m-%d"),
        degraded=guardrail.degraded,
        degraded_reason="; ".join(guardrail.degraded_reasons) if guardrail.degraded_reasons else "",
        extras={
            "blend_weight": w,
            "injury_coverage_threshold": inj_thr,
            "dnp_tail_minutes_threshold": dnp_thr,
            "alloc_mask_mode": resolved_alloc_mask_mode,
            "alloc_min_eligible": int(resolved_alloc_min_eligible),
            "alloc_prior_play_prob_threshold": float(
                resolved_alloc_prior_play_prob_threshold
            ),
            "alloc_baseline_minutes_threshold": float(
                resolved_alloc_baseline_minutes_threshold
            ),
            "enable_blend_to_baseline": enable_blend,
            "fallback_mode": fallback_mode,
            "mode": resolved_mode,
            "espn_out_count": espn_out_count,
            "espn_matched_count": espn_matched_count,
            "degraded_reasons": guardrail.degraded_reasons,
        },
    )
    inject_provenance_into_summary(summary_path, provenance)

    # Inject parity report summary
    inject_parity_into_summary(summary_path, parity_report)

    typer.echo(f"[rotation_minutes] wrote guarded minutes -> {minutes_path}", err=True)


if __name__ == "__main__":
    app()
