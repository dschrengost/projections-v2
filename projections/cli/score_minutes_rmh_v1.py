"""Score minutes using RMH v1 (Rotation Minutes Hurdle) model.

This CLI:
1. Loads the RMH bundle from config/rmh_current_run.json
2. Loads base minutes features and joins rotation priors (critical for training/inference parity)
3. Runs RMH inference via predict_frame()
4. Applies 240-minute team reconciliation with configurable threshold
5. Outputs to standard minutes_v1/daily path for downstream consumption
"""

from __future__ import annotations

import json
import os
from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from projections.minutes import build_provenance, inject_provenance_into_summary

UTC = timezone.utc

DEFAULT_DATA_ROOT = Path(os.environ.get("PROJECTIONS_DATA_ROOT", "data"))
DEFAULT_CONFIG_PATH = Path("config/rmh_current_run.json")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

app = typer.Typer(add_completion=False)

def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean").fillna(False).astype(bool)
    return series.fillna(False).astype(bool)


def _reconcile_team_minutes_with_rotation_cap(
    df_team: pd.DataFrame,
    *,
    target_minutes: float,
    minutes_col: str,
    in_rotation_threshold_min: float,
    max_rotation_players: int | None,
) -> pd.Series:
    """Reconcile team minutes while enforcing a hard cap on rotation size.

    RMH can produce non-trivial minutes for many \"active\" players, which then gets
    flattened by scaling to 240. This helper optionally caps the number of players
    eligible to receive non-zero effective minutes (per game/team) by selecting the
    top-N by the model minutes column, always keeping starters.
    """

    from projections.minutes.reconcile import reconcile_team_minutes

    if df_team.empty:
        return pd.Series([], index=df_team.index, dtype=float)

    if max_rotation_players is None:
        reconciled_minutes, _ = reconcile_team_minutes(
            df_team,
            target_minutes=float(target_minutes),
            minutes_col=minutes_col,
            in_rotation_threshold_min=float(in_rotation_threshold_min),
        )
        return reconciled_minutes

    max_n = int(max_rotation_players)
    if max_n < 1:
        raise ValueError("max_rotation_players must be >= 1")

    eff = pd.Series(0.0, index=df_team.index, dtype=float)

    # Avoid allocating minutes to OUT rows.
    can_play = pd.Series(True, index=df_team.index, dtype=bool)
    if "status" in df_team.columns:
        status = df_team["status"].astype("string").str.upper()
        can_play &= ~status.eq("OUT").fillna(False)
    if "play_prob" in df_team.columns:
        play_prob = pd.to_numeric(df_team["play_prob"], errors="coerce").fillna(0.0)
        can_play &= play_prob > 0.0

    starter_flag = pd.Series(False, index=df_team.index, dtype=bool)
    if "starter_flag" in df_team.columns:
        starter_flag = pd.to_numeric(df_team["starter_flag"], errors="coerce").fillna(0).astype(int) > 0
    if "is_confirmed_starter" in df_team.columns:
        starter_flag = starter_flag | _coerce_bool(df_team["is_confirmed_starter"])
    if "is_projected_starter" in df_team.columns:
        starter_flag = starter_flag | _coerce_bool(df_team["is_projected_starter"])
    always_keep = can_play & starter_flag

    minutes = pd.to_numeric(df_team[minutes_col], errors="coerce").fillna(0.0)
    pool_mask = can_play & ~always_keep & (minutes > 0.0)

    keep = always_keep.copy()
    slots = max_n - int(keep.sum())
    if slots > 0 and pool_mask.any():
        pool = df_team.loc[pool_mask].copy()
        pool_minutes = pd.to_numeric(pool[minutes_col], errors="coerce").fillna(0.0)
        if "player_id" in pool.columns:
            pool_player_id = pd.to_numeric(pool["player_id"], errors="coerce").fillna(0).astype(int)
        else:
            pool_player_id = pd.Series(range(len(pool)), index=pool.index, dtype=int)

        order = (
            pd.DataFrame(
                {
                    "_minutes": pool_minutes,
                    "_player_id": pool_player_id,
                }
            )
            .sort_values(["_minutes", "_player_id"], ascending=[False, True])
            .head(slots)
        )
        keep.loc[order.index] = True

    kept = df_team.loc[keep].copy()
    if kept.empty:
        return eff

    reconciled_kept, _ = reconcile_team_minutes(
        kept,
        target_minutes=float(target_minutes),
        minutes_col=minutes_col,
        in_rotation_threshold_min=float(in_rotation_threshold_min),
    )
    eff.loc[reconciled_kept.index] = reconciled_kept
    return eff


def _normalize_day(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(date_str).normalize()


def _season_for_day(day: pd.Timestamp) -> int:
    """Return the season year (Aug–Jul boundary)."""
    return int(day.year) if int(day.month) >= 8 else int(day.year) - 1


def _load_rmh_config(config_path: Path) -> dict[str, Any]:
    """Load RMH configuration from JSON file."""
    if not config_path.exists():
        raise FileNotFoundError(f"RMH config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


@app.command()
def main(
    *,
    date: str = typer.Option(..., "--date", help="Slate date (YYYY-MM-DD)."),
    run_id: str = typer.Option(..., "--run-id", help="Run id for run-scoped outputs."),
    data_root: Path = typer.Option(
        DEFAULT_DATA_ROOT, "--data-root", help="PROJECTIONS_DATA_ROOT override."
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH, "--config", help="Path to RMH config JSON."
    ),
    minutes_features_root: Path | None = typer.Option(
        None,
        "--minutes-features-root",
        help="Override root for live minutes features.",
    ),
) -> None:
    """Score minutes using RMH v1 model with 240-minute team reconciliation."""
    day = _normalize_day(date)
    season = _season_for_day(day)
    data_root = Path(data_root).expanduser().resolve()

    # Resolve config path relative to project root if not absolute
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = _load_rmh_config(config_path)
    if not config.get("enabled", False):
        typer.echo("[rmh-scorer] RMH disabled in config; exiting.", err=True)
        raise typer.Exit(code=0)

    bundle_dir_raw = config.get("bundle_dir")
    if not bundle_dir_raw:
        raise ValueError("bundle_dir not specified in RMH config")

    # Resolve bundle dir relative to project root if not absolute
    bundle_dir = Path(bundle_dir_raw)
    if not bundle_dir.is_absolute():
        bundle_dir = PROJECT_ROOT / bundle_dir
    bundle_dir = bundle_dir.expanduser().resolve()

    if not bundle_dir.exists():
        raise FileNotFoundError(f"RMH bundle not found: {bundle_dir}")

    reconcile_mode = config.get("reconcile_team_minutes", "p50")
    in_rotation_threshold_min = float(config.get("in_rotation_threshold_min", 5.0))
    max_rotation_players_raw = config.get("max_rotation_players")
    max_rotation_players = int(max_rotation_players_raw) if max_rotation_players_raw is not None else None

    typer.echo(
        f"[rmh-scorer] date={day.date()} run_id={run_id} bundle={bundle_dir.name} "
        f"reconcile={reconcile_mode} threshold={in_rotation_threshold_min} "
        f"max_rotation_players={max_rotation_players}",
        err=True,
    )

    # Import RMH modules lazily to avoid torch import cost when not needed
    from projections.models.rotation_minutes_hurdle_v1 import load_bundle, predict_frame
    from projections.rotation.live_features_v1 import load_rotation_priors_for_live_inference
    from projections.rotation.rotation_set_minutes_features_v1 import (
        apply_odds_missing_flags,
        fill_numeric_missing_with_zero,
        join_rotation_priors,
    )
    from projections.models.rotation_minutes_hurdle_v1.live_features import prepare_live_features_for_rmh

    # 1) Load RMH bundle
    bundle = load_bundle(bundle_dir)
    typer.echo(
        f"[rmh-scorer] loaded bundle delta_out={bundle.model.delta_out} "
        f"n_continuous={len(bundle.feature_spec.continuous)} "
        f"n_categorical={len(bundle.feature_spec.categorical)}",
        err=True,
    )

    # 2) Load minutes features
    if minutes_features_root is None:
        minutes_features_root = data_root / "live" / "features_minutes_v1"
    features_path = (
        Path(minutes_features_root) / day.strftime("%Y-%m-%d") / f"run={run_id}" / "features.parquet"
    )
    if not features_path.exists():
        raise FileNotFoundError(f"Minutes features not found: {features_path}")

    features = pd.read_parquet(features_path)
    typer.echo(f"[rmh-scorer] loaded {len(features)} rows from {features_path}", err=True)

    # 3) Check feature coverage before priors join
    required_cont = set(bundle.feature_spec.continuous)
    required_cat = set(bundle.feature_spec.categorical)
    required = required_cont | required_cat

    missing_cont_before = sorted(required_cont.difference(features.columns))
    missing_frac_before = len(missing_cont_before) / max(len(required_cont), 1)
    if missing_cont_before:
        typer.echo(
            f"[rmh-scorer] feature_coverage before_priors "
            f"missing={len(missing_cont_before)}/{len(required_cont)} "
            f"frac={missing_frac_before:.3f} sample={missing_cont_before[:5]}",
            err=True,
        )

    # 4) Join rotation priors (CRITICAL for training/inference parity)
    if any("_prior_" in col for col in required.difference(features.columns)):
        typer.echo("[rmh-scorer] joining rotation priors...", err=True)

        # Load priors config
        allow_priors_fallback = True
        try:
            rot_cfg_path = PROJECT_ROOT / "config" / "rotation_set_minutes_live.json"
            if rot_cfg_path.exists():
                rot_cfg = json.loads(rot_cfg_path.read_text(encoding="utf-8"))
                allow_priors_fallback = bool(rot_cfg.get("allow_priors_fallback", True))
        except Exception:
            pass

        game_ids = (
            features["game_id"].astype(str).unique().tolist()
            if "game_id" in features.columns
            else []
        )
        team_ids = (
            pd.to_numeric(features["team_id"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
            if "team_id" in features.columns
            else []
        )
        player_ids = (
            pd.to_numeric(features["player_id"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
            if "player_id" in features.columns
            else []
        )

        priors = load_rotation_priors_for_live_inference(
            data_root=data_root,
            season=season,
            game_date=day.strftime("%Y-%m-%d"),
            game_ids=game_ids,
            team_ids=team_ids,
            player_ids=player_ids,
            allow_priors_fallback=allow_priors_fallback,
        )
        if priors.warning_message:
            typer.echo(f"[rmh-scorer] priors warning: {priors.warning_message}", err=True)
        typer.echo(
            f"[rmh-scorer] priors loaded: teams_found={priors.teams_found} "
            f"players_found={priors.players_found} fallback={priors.used_latest_fallback}",
            err=True,
        )

        work = features.copy()
        if {"spread_home", "total"}.issubset(work.columns):
            work = apply_odds_missing_flags(work)
        work = fill_numeric_missing_with_zero(work)
        work = join_rotation_priors(
            work,
            team_priors=priors.team_priors,
            player_priors=priors.player_priors,
        )
        features = prepare_live_features_for_rmh(work)

    # 5) Check feature coverage after priors join
    missing_cont_after = sorted(required_cont.difference(features.columns))
    missing_cat_after = sorted(required_cat.difference(features.columns))
    if missing_cont_after or missing_cat_after:
        typer.echo(
            f"[rmh-scorer] feature_coverage after_priors "
            f"missing_cont={len(missing_cont_after)}/{len(required_cont)} "
            f"missing_cat={len(missing_cat_after)}/{len(required_cat)} "
            f"cont_sample={missing_cont_after[:5]} cat_sample={missing_cat_after[:5]}",
            err=True,
        )
        raise RuntimeError(
            "Insufficient feature coverage for RMH inference; refusing to zero-fill missing inputs."
        )

    # 6) Run RMH inference
    typer.echo("[rmh-scorer] running RMH inference...", err=True)
    preds = predict_frame(features, bundle=bundle)

    # 7) Build output DataFrame with required columns
    key_cols = ["game_id", "player_id", "team_id"]
    out = features.loc[:, [c for c in key_cols if c in features.columns]].copy()
    out["game_date"] = day.date()

    # Copy through useful metadata columns if present
    passthrough_cols = [
        # Identity / roster context
        "player_name",
        "status",
        "pos_bucket",
        "starter_flag",
        "is_projected_starter",
        "is_confirmed_starter",
        # Game context
        "tip_ts",
        "team_name",
        "team_tricode",
        "opponent_team_id",
        "opponent_team_name",
        "opponent_team_tricode",
        # Vegas / game script signals
        "spread_home",
        "total",
        "odds_as_of_ts",
        "blowout_index",
        "blowout_risk_score",
        "close_game_score",
        # Legacy cols (some downstream tooling still uses these)
        "team_abbr",
        "opp_team_abbr",
    ]
    for col in passthrough_cols:
        if col in features.columns:
            out[col] = features[col]

    # Map RMH outputs to standard schema
    out["play_prob"] = pd.to_numeric(preds["p_play"], errors="coerce").fillna(0.0)

    # Unconditional quantiles (these are the "real" predictions)
    for q in ("10", "50", "90"):
        col_src = f"minutes_q{q}_uncond"
        col_dst = f"minutes_p{q}"
        if col_src in preds.columns:
            out[col_dst] = pd.to_numeric(preds[col_src], errors="coerce").fillna(0.0)
        else:
            # Fallback for v1.0 bundles
            out[col_dst] = 0.0

    # Conditional quantiles (for downstream that wants them)
    for q in ("10", "50", "90"):
        col_src = f"minutes_q{q}_cond"
        col_dst = f"minutes_p{q}_cond"
        if col_src in preds.columns:
            out[col_dst] = pd.to_numeric(preds[col_src], errors="coerce").fillna(0.0)
        elif f"minutes_p{q}" in out.columns:
            # Derive from unconditional if conditional not available
            out[col_dst] = out[f"minutes_p{q}"]

    # 8) Apply 240-minute team reconciliation
    if reconcile_mode != "none":
        typer.echo(
            f"[rmh-scorer] applying 240-min reconciliation (threshold={in_rotation_threshold_min})...",
            err=True,
        )

        reconciled_list = []
        for (game_id, team_id), team_df in out.groupby(["game_id", "team_id"]):
            eff = _reconcile_team_minutes_with_rotation_cap(
                team_df,
                target_minutes=240.0,
                minutes_col="minutes_p50",
                in_rotation_threshold_min=in_rotation_threshold_min,
                max_rotation_players=max_rotation_players,
            )
            team_result = team_df.copy()
            team_result["effective_minutes"] = eff.reindex(team_result.index).fillna(0.0).to_numpy(dtype=float)
            reconciled_list.append(team_result)

        out = pd.concat(reconciled_list, ignore_index=True)
        typer.echo(
            f"[rmh-scorer] reconciled {len(out)} players across "
            f"{out['team_id'].nunique()} teams",
            err=True,
        )
        if max_rotation_players is not None:
            typer.echo(
                f"[rmh-scorer] rotation cap applied: max_rotation_players={max_rotation_players}",
                err=True,
            )
    else:
        # No reconciliation - use p50 as effective
        out["effective_minutes"] = out["minutes_p50"]

    # 9) Write output
    out_dir = data_root / "artifacts" / "minutes_v1" / "daily" / day.strftime("%Y-%m-%d") / f"run={run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "minutes.parquet"
    out.to_parquet(out_path, index=False)

    # Write summary
    summary = {
        "date": day.strftime("%Y-%m-%d"),
        "run_id": run_id,
        "generated_at": pd.Timestamp.now(tz=UTC).isoformat(),
        "model": "rmh_v1",
        "bundle_dir": str(bundle_dir),
        "schema_hash": bundle.schema_hash,
        "reconcile_mode": reconcile_mode,
        "in_rotation_threshold_min": in_rotation_threshold_min,
        "max_rotation_players": max_rotation_players,
        "counts": {
            "rows": int(len(out)),
            "games": int(out["game_id"].nunique()) if "game_id" in out.columns else 0,
            "teams": int(out["team_id"].nunique()) if "team_id" in out.columns else 0,
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Inject provenance stamp for artifact traceability
    provenance = build_provenance(
        alloc_mode="rmh",
        model_dir=bundle_dir,
        run_id=run_id,
        game_date=day.strftime("%Y-%m-%d"),
        degraded=False,
        degraded_reason="",
        extras={
            "reconcile_mode": reconcile_mode,
            "in_rotation_threshold_min": in_rotation_threshold_min,
            "max_rotation_players": max_rotation_players,
            "schema_hash": bundle.schema_hash,
        },
    )
    inject_provenance_into_summary(summary_path, provenance)

    typer.echo(f"[rmh-scorer] wrote {len(out)} rows to {out_path}", err=True)


if __name__ == "__main__":
    app()
