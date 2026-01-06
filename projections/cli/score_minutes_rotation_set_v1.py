"""Score minutes using baseline minutes_v1 with a guarded rotation_set_minutes_v1 overlay.

This is intended as a production-safe switch:
- Always computes baseline minutes_v1 first.
- If rotation_set_minutes is enabled, builds live features + predicts rotation minutes.
- Applies guardrails + tail mapping, then overwrites minutes_v1 outputs for downstream consumers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

from projections import paths
from projections.minutes_v1.logs import prediction_logs_base
from projections.minutes_v1.reconcile import load_reconcile_config
from projections.rotation.guardrails import apply_rotation_minutes_guardrails
from projections.rotation.live_features_v1 import (
    RotationLiveFeaturesError,
    RotationSetMinutesV1FeatureSpec,
    build_rotation_set_minutes_v1_features,
    load_rotation_priors_for_game_ids,
)
from projections.rotation.set_model import predict_minutes as predict_rotation_minutes

UTC = timezone.utc

app = typer.Typer(help=__doc__)

DEFAULT_DATA_ROOT = paths.get_data_root()
DEFAULT_MINUTES_FEATURES_ROOT = paths.data_path("live", "features_minutes_v1")
DEFAULT_ROTATION_FEATURES_ROOT = paths.data_path("live", "features_rotation_set_minutes_v1")
DEFAULT_MINUTES_ARTIFACT_ROOT = paths.data_path("artifacts", "minutes_v1", "daily")
DEFAULT_ROTATION_CONFIG = Path("config/rotation_set_minutes_live.json")
DEFAULT_RECONCILE_CONFIG = Path("config/minutes_l2_reconcile.yaml")


@dataclass(frozen=True)
class RotationLiveConfig:
    enabled: bool
    model_dir: str | None
    blend_weight: float
    injury_coverage_threshold: float
    dnp_tail_minutes_threshold: float

    @classmethod
    def load(cls, path: Path) -> "RotationLiveConfig":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            payload = {}
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid rotation minutes config at {path}: {exc}") from exc
        enabled = bool(payload.get("enabled", False))
        model_dir = payload.get("model_dir")
        blend_weight = float(payload.get("blend_weight", 0.2))
        injury_threshold = float(payload.get("injury_coverage_threshold", 0.5))
        dnp_tail_threshold = float(payload.get("dnp_tail_minutes_threshold", 8.0))
        return cls(
            enabled=enabled,
            model_dir=str(model_dir) if model_dir else None,
            blend_weight=blend_weight,
            injury_coverage_threshold=injury_threshold,
            dnp_tail_minutes_threshold=dnp_tail_threshold,
        )


def _normalize_day(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _season_for_day(day: pd.Timestamp) -> int:
    return int(day.year) if int(day.month) >= 8 else int(day.year) - 1


def _load_rotation_historical_features_for_dnp(
    data_root: Path,
    *,
    season: int,
    target_day: pd.Timestamp,
    team_ids: set[int],
    player_ids: set[int],
    lookback_days: int = 120,
) -> pd.DataFrame:
    """Load realized minutes + pre-tip availability for DNP history features.

    Live rotation features need DNP-history columns to match training. We source
    the historical frame from minutes prediction logs (one parquet per slate/run),
    selecting the latest run per day and filtering to the requested entities.
    """
    if not team_ids or not player_ids:
        return pd.DataFrame()

    logs_root = prediction_logs_base(data_root)
    season_dir = logs_root / f"season={int(season)}"
    if not season_dir.exists():
        return pd.DataFrame()

    end_day = pd.Timestamp(target_day).normalize()
    cutoff = (end_day - timedelta(days=int(lookback_days))).normalize()
    days = pd.date_range(cutoff, end_day - timedelta(days=1), freq="D")

    frames: list[pd.DataFrame] = []
    for day in days:
        month_dir = season_dir / f"month={int(day.month):02d}"
        if not month_dir.exists():
            continue
        pattern = f"{day.strftime('%Y-%m-%d')}_*.parquet"
        candidates = sorted(month_dir.glob(pattern))
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda path: path.stem.split("_", 1)[1] if "_" in path.stem else "",
        )
        try:
            df = pd.read_parquet(
                best,
                columns=["game_date", "team_id", "player_id", "minutes", "is_out", "status"],
            )
        except Exception:  # noqa: BLE001
            try:
                df = pd.read_parquet(best)
            except Exception:  # noqa: BLE001
                continue

        if df.empty:
            continue

        df["game_date"] = pd.to_datetime(df.get("game_date"), errors="coerce").dt.normalize()
        df["team_id"] = pd.to_numeric(df.get("team_id"), errors="coerce").astype("Int64")
        df["player_id"] = pd.to_numeric(df.get("player_id"), errors="coerce").astype("Int64")
        df = df.dropna(subset=["game_date", "team_id", "player_id"]).copy()
        if df.empty:
            continue

        # Filter early for speed.
        df = df.loc[
            df["team_id"].astype(int).isin(team_ids) & df["player_id"].astype(int).isin(player_ids)
        ].copy()
        if df.empty:
            continue

        if "minutes" not in df.columns:
            continue
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0).astype(float)

        if "is_out" in df.columns:
            df["is_out"] = pd.to_numeric(df["is_out"], errors="coerce").fillna(0).astype(int)
        elif "status" in df.columns:
            status = df["status"].astype("string").str.upper().str.strip()
            df["is_out"] = status.eq("OUT").fillna(False).astype(int)
        else:
            df["is_out"] = 0

        df = df.loc[df["game_date"] < end_day].copy()
        if df.empty:
            continue

        frames.append(df.loc[:, ["game_date", "team_id", "player_id", "is_out", "minutes"]])

    if not frames:
        return pd.DataFrame()

    hist = pd.concat(frames, ignore_index=True)
    hist = hist.drop_duplicates(subset=["game_date", "team_id", "player_id"], keep="last")
    return hist.reset_index(drop=True)


def _run_python_module(module: str, args: list[str], *, data_root: Path, timeout_s: int) -> None:
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
    summary_path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")


@app.command()
def main(
    *,
    date: str = typer.Option(..., "--date", help="Slate date (YYYY-MM-DD)."),
    run_id: str = typer.Option(..., "--run-id", help="Run id for run-scoped outputs."),
    reconcile_team_minutes: str = typer.Option("p50", "--reconcile-team-minutes", help="Baseline reconcile mode."),
    reconcile_config: Path = typer.Option(
        DEFAULT_RECONCILE_CONFIG,
        "--reconcile-config",
        help="Reconcile config used to cap/redistribute the rotation overlay (matches baseline minutes_v1).",
    ),
    minutes_output: str = typer.Option("conditional", "--minutes-output", help="Minutes output mode (baseline)."),
    skip_baseline: bool = typer.Option(
        False,
        "--skip-baseline",
        help="Skip baseline minutes_v1 scoring (assumes baseline outputs already exist for this run_id).",
    ),
    bundle_dir: Path | None = typer.Option(None, "--bundle-dir", help="Override baseline minutes bundle directory."),
    rotshare_quantiles_mode: str | None = typer.Option(None, "--rotshare-quantiles-mode", help="Baseline rotshare mode."),
    rotshare_n_worlds: int | None = typer.Option(None, "--rotshare-n-worlds", help="Baseline rotshare n worlds."),
    rotshare_concentration: float | None = typer.Option(None, "--rotshare-concentration", help="Baseline rotshare concentration."),
    rotshare_seed: int | None = typer.Option(None, "--rotshare-seed", help="Baseline rotshare seed."),
    rotshare_min_active_players: int | None = typer.Option(
        None, "--rotshare-min-active-players", help="Baseline rotshare min active players."
    ),
    rotshare_mc_center: str | None = typer.Option(None, "--rotshare-mc-center", help="Baseline rotshare mc center."),
    data_root: Path = typer.Option(DEFAULT_DATA_ROOT, "--data-root", help="PROJECTIONS_DATA_ROOT override."),
    minutes_features_root: Path = typer.Option(
        DEFAULT_MINUTES_FEATURES_ROOT, "--minutes-features-root", help="Root for live minutes features."
    ),
    rotation_features_root: Path = typer.Option(
        DEFAULT_ROTATION_FEATURES_ROOT, "--rotation-features-root", help="Where to write rotation feature slices."
    ),
    artifact_root: Path = typer.Option(
        DEFAULT_MINUTES_ARTIFACT_ROOT, "--artifact-root", help="Minutes artifacts root (minutes_v1/daily)."
    ),
    rotation_config: Path = typer.Option(
        DEFAULT_ROTATION_CONFIG, "--rotation-config", help="JSON config for rotation minutes enable/weights."
    ),
    model_dir: Path | None = typer.Option(
        None,
        "--model-dir",
        help="Override rotation model dir (defaults to config or ROTATION_SET_MODEL_DIR).",
    ),
    blend_weight: float | None = typer.Option(None, "--blend-weight", help="Override conservative blend weight."),
    injury_coverage_threshold: float | None = typer.Option(
        None, "--injury-coverage-threshold", help="Override injury coverage threshold."
    ),
    dnp_tail_minutes_threshold: float | None = typer.Option(
        None,
        "--dnp-tail-minutes-threshold",
        help="Max minutes allowed on OUT/DNP-proxy rows per team-game before falling back to baseline.",
    ),
    device: str = typer.Option("cpu", "--device", help="Torch device for rotation model."),
    batch_size: int = typer.Option(64, "--batch-size", help="Batch size (team-games) for rotation model."),
) -> None:
    day = _normalize_day(date)
    season = _season_for_day(day)
    data_root = Path(data_root).expanduser().resolve()

    minutes_out_dir = Path(artifact_root) / day.strftime("%Y-%m-%d") / f"run={run_id}"
    minutes_path = minutes_out_dir / "minutes.parquet"
    summary_path = minutes_out_dir / "summary.json"

    # 1) Always run baseline first (safe fallback), unless explicitly skipped.
    if not skip_baseline:
        typer.echo(f"[rotation_minutes] baseline minutes_v1 date={day.date()} run_id={run_id}", err=True)
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
            baseline_args.extend(["--rotshare-min-active-players", str(int(rotshare_min_active_players))])
        if rotshare_mc_center:
            baseline_args.extend(["--rotshare-mc-center", str(rotshare_mc_center)])

        _run_python_module(
            "projections.cli.score_minutes_v1",
            baseline_args,
            data_root=data_root,
            timeout_s=1200,
        )

    if not minutes_path.exists():
        raise RuntimeError(f"Baseline minutes.parquet missing at {minutes_path}")

    config = RotationLiveConfig.load(rotation_config)
    enabled = bool(config.enabled)
    resolved_model_dir = (
        model_dir
        or (Path(os.environ.get("ROTATION_SET_MODEL_DIR", "")) if os.environ.get("ROTATION_SET_MODEL_DIR") else None)
        or (Path(config.model_dir) if config.model_dir else None)
    )
    if not enabled:
        typer.echo(f"[rotation_minutes] rotation_set_minutes disabled (config={rotation_config}); keeping baseline.", err=True)
        _update_summary_json(summary_path, {"rotation_set_minutes": {"enabled": False}})
        return
    if resolved_model_dir is None:
        typer.echo(
            "[rotation_minutes] WARNING: rotation enabled but model_dir missing; keeping baseline.",
            err=True,
        )
        _update_summary_json(summary_path, {"rotation_set_minutes": {"enabled": True, "error": "model_dir_missing"}})
        return

    w = float(blend_weight) if blend_weight is not None else float(config.blend_weight)
    inj_thr = float(injury_coverage_threshold) if injury_coverage_threshold is not None else float(config.injury_coverage_threshold)
    dnp_thr = (
        float(dnp_tail_minutes_threshold)
        if dnp_tail_minutes_threshold is not None
        else float(config.dnp_tail_minutes_threshold)
    )

    typer.echo(
        f"[rotation_minutes] rotation_set_minutes enabled model_dir={resolved_model_dir} blend_weight={w} "
        f"injury_thr={inj_thr} dnp_tail_thr={dnp_thr}",
        err=True,
    )

    # 2) Load baseline minutes output and required live inputs.
    base_df = pd.read_parquet(minutes_path)
    if base_df.empty:
        typer.echo("[rotation_minutes] baseline output empty; nothing to do.", err=True)
        return

    minutes_features_path = Path(minutes_features_root) / day.strftime("%Y-%m-%d") / f"run={run_id}" / "features.parquet"
    if not minutes_features_path.exists():
        typer.echo(f"[rotation_minutes] WARNING: minutes features missing at {minutes_features_path}; keeping baseline.", err=True)
        _update_summary_json(summary_path, {"rotation_set_minutes": {"enabled": True, "error": "minutes_features_missing"}})
        return
    minutes_feat = pd.read_parquet(minutes_features_path)

    # 3) Build rotation model features + write them to the live features root.
    out_day_dir = Path(rotation_features_root) / day.strftime("%Y-%m-%d")
    out_run_dir = out_day_dir / f"run={run_id}"
    rot_features_path = out_run_dir / "features.parquet"
    if rot_features_path.exists():
        typer.echo(f"[rotation_minutes] using prebuilt rotation features at {rot_features_path}", err=True)
        rot_features = pd.read_parquet(rot_features_path)
        if rot_features.empty:
            typer.echo("[rotation_minutes] WARNING: rotation features empty; keeping baseline.", err=True)
            _update_summary_json(
                summary_path, {"rotation_set_minutes": {"enabled": True, "error": "rotation_features_empty"}}
            )
            return
    else:
        try:
            spec = RotationSetMinutesV1FeatureSpec.load(Path(resolved_model_dir))
            game_ids = (
                pd.to_numeric(minutes_feat["game_id"], errors="coerce")
                .dropna()
                .astype(int)
                .astype(str)
                .str.zfill(10)
                .unique()
                .tolist()
            )
            team_priors, player_priors = load_rotation_priors_for_game_ids(
                data_root, season=season, game_date=day.strftime("%Y-%m-%d"), game_ids=game_ids
            )
            team_ids = set(pd.to_numeric(minutes_feat["team_id"], errors="coerce").dropna().astype(int).tolist())
            player_ids = set(pd.to_numeric(minutes_feat["player_id"], errors="coerce").dropna().astype(int).tolist())
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
            typer.echo(f"[rotation_minutes] WARNING: feature build failed ({exc}); keeping baseline.", err=True)
            _update_summary_json(
                summary_path, {"rotation_set_minutes": {"enabled": True, "error": f"feature_build_failed:{exc}"}}
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

    # 4) Predict rotation minutes (p50) and align to baseline output rows.
    try:
        rot_scored = predict_rotation_minutes(
            rot_features,
            model_dir=Path(resolved_model_dir),
            device=str(device),
            batch_size=int(batch_size),
        )
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[rotation_minutes] WARNING: rotation model predict failed ({exc}); keeping baseline.", err=True)
        _update_summary_json(summary_path, {"rotation_set_minutes": {"enabled": True, "error": f"predict_failed:{exc}"}})
        return

    rot_pred = rot_scored.loc[:, ["game_id", "team_id", "player_id", "pred_minutes"]].rename(
        columns={"pred_minutes": "rotation_minutes_p50"}
    )
    rot_pred["rotation_minutes_p50"] = pd.to_numeric(rot_pred["rotation_minutes_p50"], errors="coerce").fillna(0.0).astype(float)

    # Guardrail inputs from minutes features (lineup fields) + rotation features (missing flags).
    guard = base_df.merge(rot_pred, on=["game_id", "team_id", "player_id"], how="left")
    guard = guard.merge(
        rot_features.loc[:, ["game_id", "team_id", "player_id", "minutes_features_row_missing", "injury_snapshot_missing"]],
        on=["game_id", "team_id", "player_id"],
        how="left",
    )
    guard = guard.merge(
        minutes_feat.loc[:, ["game_id", "team_id", "player_id", "lineup_timestamp", "lineup_status", "lineup_roster_status"]],
        on=["game_id", "team_id", "player_id"],
        how="left",
    )
    injury_cols = [c for c in ["is_out", "status"] if c in minutes_feat.columns]
    if injury_cols:
        guard = guard.merge(
            minutes_feat.loc[:, ["game_id", "team_id", "player_id", *injury_cols]],
            on=["game_id", "team_id", "player_id"],
            how="left",
        )
    else:
        guard["is_out"] = 0
        guard["status"] = pd.NA

    # Default any missing rotation preds to baseline p50 (keeps totals sane).
    baseline_p50_col = "minutes_p50_cond" if "minutes_p50_cond" in guard.columns else "minutes_p50"
    guard["rotation_minutes_p50"] = pd.to_numeric(guard["rotation_minutes_p50"], errors="coerce")
    guard["rotation_minutes_p50"] = guard["rotation_minutes_p50"].fillna(pd.to_numeric(guard[baseline_p50_col], errors="coerce").fillna(0.0))

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
    )
    new_p50 = guardrail.minutes_p50

    # 5) Map rotation p50 to minutes quantiles using baseline tail deltas.
    out_df = base_df.copy()
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

    out_df.to_parquet(minutes_path, index=False)

    # 6) Logging + summary.json augmentation.
    typer.echo(f"[rotation_minutes] guardrails: {guardrail.summary}", err=True)
    if "pathology_fallback" in guardrail.team_game_summary.columns:
        pathological = guardrail.team_game_summary.loc[guardrail.team_game_summary["pathology_fallback"]].copy()
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
            typer.echo(f"[rotation_minutes] pathology_fallback_team_games (n={len(pathological)}): {sample}", err=True)
    blended = guardrail.team_game_summary.loc[guardrail.team_game_summary["blend_applied"]].copy()
    if not blended.empty:
        sample = (
            blended.sort_values(["injury_coverage_rate", "row_fallback_count"], ascending=[True, False])
            .head(20)
            .to_dict(orient="records")
        )
        typer.echo(f"[rotation_minutes] blended_team_games (n={len(blended)}): {sample}", err=True)
    if not guardrail.tail_minutes_top.empty:
        rows = guardrail.tail_minutes_top.head(20).to_dict(orient="records")
        typer.echo(f"[rotation_minutes] top_tail_minutes_team_games (n={len(rows)}): {rows}", err=True)

    _update_summary_json(
        summary_path,
        {
            "rotation_set_minutes": {
                "enabled": True,
                "model_dir": str(Path(resolved_model_dir).expanduser()),
                "blend_weight": w,
                "injury_coverage_threshold": inj_thr,
                "dnp_tail_minutes_threshold": dnp_thr,
                "guardrails": guardrail.summary,
                "tail_minutes_team_games_top": (
                    guardrail.tail_minutes_top.head(20).to_dict(orient="records")
                    if not guardrail.tail_minutes_top.empty
                    else []
                ),
            }
        },
    )
    typer.echo(f"[rotation_minutes] wrote guarded minutes -> {minutes_path}", err=True)


if __name__ == "__main__":
    app()
