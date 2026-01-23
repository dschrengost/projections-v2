"""RMH single-player (and small-sample) inference probe.

This script is designed to answer: "where do impossible RMH minutes come from?"

It prints, for a chosen player:
  1) Raw model forward outputs (logits + conditional quantiles) BEFORE postprocessing
  2) Postprocessed outputs from `predict_frame` (unconditional quantiles)
  3) The row written by the live RMH shadow branch (minutes_models parquet), if present

It can also compare:
  - running RMH on raw minutes features (historically the buggy path), vs
  - running RMH on minutes features augmented with rotation_priors_v1.

Example:
  uv run python scripts/diagnostics/rmh_probe_single_player.py \\
    --artifact-dir /home/daniel/prod/projections-v2/artifacts/rotation_minutes_hurdle_v1/rmh_v1_20260123_noleak_q90x5 \\
    --game-date 2026-01-23 \\
    --run-id 20260123T180000Z \\
    --player-name-contains Durant \\
    --compare-priors
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import typer

from projections import paths
from projections.models.minutes_nn import transform_frame
from projections.models.rotation_minutes_hurdle_v1 import load_bundle, predict_frame
from projections.models.rotation_minutes_hurdle_v1.data import add_has_injury_row
from projections.models.rotation_minutes_hurdle_v1.infer import _ensure_feature_columns
from projections.rotation.live_features_v1 import (
    load_rotation_priors_for_live_inference,
)
from projections.rotation.rotation_set_minutes_features_v1 import (
    apply_odds_missing_flags,
    fill_numeric_missing_with_zero,
    join_rotation_priors,
)


app = typer.Typer(add_completion=False)


def _season_for_day(day: pd.Timestamp) -> int:
    # Match the convention used elsewhere in the repo (Aug–Jul season boundary).
    return int(day.year) if int(day.month) >= 8 else int(day.year) - 1


def _read_rotation_set_live_cfg(project_root: Path) -> dict[str, Any]:
    try:
        return json.loads(
            (project_root / "config/rotation_set_minutes_live.json").read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {}


def _augment_with_rotation_priors(
    features: pd.DataFrame,
    *,
    data_root: Path,
    game_date: str,
    allow_priors_fallback: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    day = pd.Timestamp(game_date).normalize()
    season = _season_for_day(day)

    priors = load_rotation_priors_for_live_inference(
        data_root=data_root,
        season=season,
        game_date=game_date,
        game_ids=features["game_id"].astype(str).unique().tolist()
        if "game_id" in features.columns
        else [],
        team_ids=(
            pd.to_numeric(features["team_id"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
            if "team_id" in features.columns
            else []
        ),
        player_ids=(
            pd.to_numeric(features["player_id"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
            if "player_id" in features.columns
            else []
        ),
        allow_priors_fallback=allow_priors_fallback,
    )

    work = features.copy()
    if {"spread_home", "total"}.issubset(work.columns):
        work = apply_odds_missing_flags(work)
    work = fill_numeric_missing_with_zero(work)
    work = join_rotation_priors(
        work, team_priors=priors.team_priors, player_priors=priors.player_priors
    )

    meta = {
        "used_latest_fallback": priors.used_latest_fallback,
        "teams_found": priors.teams_found,
        "teams_missing": priors.teams_missing,
        "players_found": priors.players_found,
        "players_missing": priors.players_missing,
        "warning_message": priors.warning_message,
    }
    return work, meta


def _missing_required_features(
    df: pd.DataFrame, *, bundle_required: set[str]
) -> list[str]:
    return sorted(bundle_required.difference(df.columns))


@torch.no_grad()
def _raw_forward_one_row(row: pd.DataFrame, *, bundle) -> dict[str, float]:
    work = add_has_injury_row(row)
    work = _ensure_feature_columns(work, bundle)
    x_cont, x_cat = transform_frame(work, bundle.feature_spec, bundle.preprocessor)
    out = bundle.model(torch.from_numpy(x_cont), torch.from_numpy(x_cat))
    logits = float(out.logits_play.detach().cpu().numpy()[0])
    p_play = float(torch.sigmoid(out.logits_play).detach().cpu().numpy()[0])
    return {
        "logits_play": logits,
        "p_play": p_play,
        "q05_cond": float(out.q05_cond.detach().cpu().numpy()[0]),
        "q10_cond": float(out.q10_cond.detach().cpu().numpy()[0]),
        "q25_cond": float(out.q25_cond.detach().cpu().numpy()[0]),
        "q50_cond": float(out.q50_cond.detach().cpu().numpy()[0]),
        "q75_cond": float(out.q75_cond.detach().cpu().numpy()[0]),
        "q90_cond": float(out.q90_cond.detach().cpu().numpy()[0]),
        "q95_cond": float(out.q95_cond.detach().cpu().numpy()[0]),
    }


def _print_player_block(
    title: str, *, player_row: pd.DataFrame, bundle, out_path: Path | None = None
) -> None:
    typer.echo(f"\n== {title} ==")
    raw = _raw_forward_one_row(player_row, bundle=bundle)
    typer.echo("[raw-forward] " + json.dumps(raw, sort_keys=True))

    preds = predict_frame(player_row, bundle=bundle)
    cols = [
        "player_id",
        "player_name",
        "team_id",
        "game_id",
        "p_play",
        "minutes_q10_cond",
        "minutes_q50_cond",
        "minutes_q90_cond",
        "minutes_q10_uncond",
        "minutes_q50_uncond",
        "minutes_q90_uncond",
    ]
    cols = [c for c in cols if c in preds.columns]
    typer.echo("[predict_frame] " + preds.loc[:, cols].to_string(index=False))

    if out_path and out_path.exists():
        df_out = pd.read_parquet(out_path)
        pid = int(pd.to_numeric(player_row["player_id"], errors="coerce").iloc[0])
        hit = df_out[df_out["player_id"] == pid]
        if hit.empty:
            typer.echo(
                f"[written-parquet] no row found in {out_path} for player_id={pid}"
            )
        else:
            typer.echo(f"[written-parquet] {out_path}")
            typer.echo(hit.to_string(index=False))


@app.command()
def main(
    *,
    artifact_dir: Path = typer.Option(
        ..., "--artifact-dir", exists=True, file_okay=False, dir_okay=True
    ),
    game_date: str = typer.Option(..., "--game-date", help="YYYY-MM-DD"),
    run_id: str = typer.Option(..., "--run-id", help="Run id like 20260123T180000Z"),
    player_id: int | None = typer.Option(None, "--player-id"),
    player_name_contains: str | None = typer.Option(None, "--player-name-contains"),
    compare_priors: bool = typer.Option(True, "--compare-priors/--no-compare-priors"),
    data_root: Path = typer.Option(paths.get_data_root(), "--data-root"),
) -> None:
    if player_id is None and not player_name_contains:
        raise typer.BadParameter("Provide either --player-id or --player-name-contains")

    bundle = load_bundle(Path(artifact_dir).expanduser().resolve())
    required = set(bundle.feature_spec.continuous + bundle.feature_spec.categorical)
    typer.echo(
        "[bundle] "
        + json.dumps(
            {
                "artifact_dir": str(Path(artifact_dir).expanduser().resolve()),
                "delta_out": int(bundle.model.delta_out),
                "play_threshold": bundle.config.get("play_threshold"),
                "quantiles": bundle.config.get("quantiles"),
                "n_required_features": len(required),
            },
            sort_keys=True,
        )
    )

    minutes_path = (
        Path(data_root)
        / "live"
        / "features_minutes_v1"
        / str(game_date)
        / f"run={run_id}"
        / "features.parquet"
    )
    if not minutes_path.exists():
        raise FileNotFoundError(f"Minutes features not found: {minutes_path}")
    features = pd.read_parquet(minutes_path)

    if player_id is not None:
        player_row = (
            features[
                pd.to_numeric(features["player_id"], errors="coerce") == int(player_id)
            ]
            .head(1)
            .copy()
        )
    else:
        player_row = (
            features[
                features["player_name"]
                .astype(str)
                .str.contains(str(player_name_contains), case=False, na=False)
            ]
            .head(1)
            .copy()
        )

    if player_row.empty:
        raise RuntimeError("Player not found in minutes features slice.")

    pid = int(pd.to_numeric(player_row["player_id"], errors="coerce").iloc[0])
    pname = str(player_row.get("player_name", pd.Series(["?"])).iloc[0])
    typer.echo(
        f"[player] player_id={pid} player_name={pname} game_date={game_date} run_id={run_id}"
    )

    out_path = (
        Path(data_root)
        / "artifacts"
        / "minutes_models"
        / "daily"
        / "model_id=rmh_v1_1"
        / str(game_date)
        / f"run={run_id}"
        / "minutes.parquet"
    )

    missing_base = _missing_required_features(features, bundle_required=required)
    typer.echo(
        f"[feature_coverage base] missing={len(missing_base)}/{len(required)} sample={missing_base[:12]}"
    )

    _print_player_block(
        "RMH on minutes features (raw)",
        player_row=player_row,
        bundle=bundle,
        out_path=out_path,
    )

    if not compare_priors:
        return

    cfg = _read_rotation_set_live_cfg(paths.get_project_root())
    allow_fallback = bool(cfg.get("allow_priors_fallback", True))
    augmented, priors_meta = _augment_with_rotation_priors(
        features,
        data_root=Path(data_root),
        game_date=game_date,
        allow_priors_fallback=allow_fallback,
    )
    typer.echo("[priors] " + json.dumps(priors_meta, sort_keys=True))

    missing_aug = _missing_required_features(augmented, bundle_required=required)
    typer.echo(
        f"[feature_coverage +priors] missing={len(missing_aug)}/{len(required)} sample={missing_aug[:12]}"
    )

    player_row_aug = (
        augmented[pd.to_numeric(augmented["player_id"], errors="coerce") == pid]
        .head(1)
        .copy()
    )
    if player_row_aug.empty:
        raise RuntimeError("Player missing after priors join (unexpected).")
    _print_player_block(
        "RMH on minutes + rotation priors (fixed)",
        player_row=player_row_aug,
        bundle=bundle,
        out_path=None,
    )


if __name__ == "__main__":
    app()
