"""Score minutes using baseline minutes_v1 with a guarded rotation_set_minutes_v1 overlay.

This is intended as a production-safe switch:
- Always computes baseline minutes_v1 first.
- If rotation_set_minutes is enabled, builds live features + predicts rotation minutes.
- Applies guardrails + tail mapping, then overwrites minutes_v1 outputs for downstream consumers.
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
from projections.minutes_v1.reconcile import load_reconcile_config
from projections.rotation.guardrails import apply_rotation_minutes_guardrails
from projections.rotation.live_features_v1 import (
    RotationLiveFeaturesError,
    RotationSetMinutesV1FeatureSpec,
    build_rotation_set_minutes_v1_features,
    load_rotation_priors_for_live_inference,
)
from projections.rotation.set_model import predict_minutes as predict_rotation_minutes

# ESPN OUT override - import from baseline scorer to ensure consistency
from projections.cli.score_minutes_v1 import (
    _load_espn_out_players,
    _normalize_name_for_matching,
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


@dataclass(frozen=True)
class RotationLiveConfig:
    enabled: bool
    model_dir: str | None
    blend_weight: float
    injury_coverage_threshold: float
    dnp_tail_minutes_threshold: float
    allow_priors_fallback: bool

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
        return cls(
            enabled=enabled,
            model_dir=str(model_dir) if model_dir else None,
            blend_weight=blend_weight,
            injury_coverage_threshold=injury_threshold,
            dnp_tail_minutes_threshold=dnp_tail_threshold,
            allow_priors_fallback=allow_priors_fallback,
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
    cutoff = (end_day - timedelta(days=int(lookback_days))).normalize()
    days = pd.date_range(cutoff, end_day - timedelta(days=1), freq="D")

    labels_path = (
        data_root / "labels" / f"season={int(season)}" / "boxscore_labels.parquet"
    )
    if not labels_path.exists():
        return pd.DataFrame()
    try:
        labels = pd.read_parquet(
            labels_path, columns=["game_date", "team_id", "player_id", "minutes"]
        )
    except Exception:  # noqa: BLE001
        labels = pd.read_parquet(labels_path)
        if not {"game_date", "team_id", "player_id", "minutes"}.issubset(
            labels.columns
        ):
            return pd.DataFrame()
        labels = labels.loc[:, ["game_date", "team_id", "player_id", "minutes"]]

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
        (labels["game_date"] >= cutoff)
        & (labels["game_date"] < end_day)
        & labels["team_id"].astype(int).isin(team_ids)
        & labels["player_id"].astype(int).isin(player_ids)
    ].copy()
    if labels.empty:
        return pd.DataFrame()

    labels["minutes"] = (
        pd.to_numeric(labels["minutes"], errors="coerce").fillna(0.0).astype(float)
    )

    # Derive is_out from injuries_raw partitions (keyed by date folder).
    # IMPORTANT: Use the LATEST snapshot per player per date, not "any OUT".
    # The source_row_id format is "{unix_ts}_{row_id}" - we parse the timestamp to sort.
    injuries_root = data_root / "bronze" / "injuries_raw" / f"season={int(season)}"
    out_frames: list[pd.DataFrame] = []
    if injuries_root.exists():
        for day in days:
            day_dir = injuries_root / f"date={day.strftime('%Y-%m-%d')}"
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
) -> None:
    day = _normalize_day(date)
    season = _season_for_day(day)
    data_root = Path(data_root).expanduser().resolve()

    minutes_out_dir = Path(artifact_root) / day.strftime("%Y-%m-%d") / f"run={run_id}"
    minutes_path = minutes_out_dir / "minutes.parquet"
    summary_path = minutes_out_dir / "summary.json"

    # 1) Always run baseline first (safe fallback), unless explicitly skipped.
    if not skip_baseline:
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
            baseline_args.extend(
                ["--rotshare-quantiles-mode", str(rotshare_quantiles_mode)]
            )
        if rotshare_n_worlds is not None:
            baseline_args.extend(["--rotshare-n-worlds", str(int(rotshare_n_worlds))])
        if rotshare_concentration is not None:
            baseline_args.extend(
                ["--rotshare-concentration", str(float(rotshare_concentration))]
            )
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

    if not minutes_path.exists():
        raise RuntimeError(f"Baseline minutes.parquet missing at {minutes_path}")

    config = RotationLiveConfig.load(rotation_config)
    enabled = bool(config.enabled)
    resolved_model_dir = (
        model_dir
        or (
            Path(os.environ.get("ROTATION_SET_MODEL_DIR", ""))
            if os.environ.get("ROTATION_SET_MODEL_DIR")
            else None
        )
        or (Path(config.model_dir) if config.model_dir else None)
    )
    if not enabled:
        typer.echo(
            f"[rotation_minutes] rotation_set_minutes disabled (config={rotation_config}); keeping baseline.",
            err=True,
        )
        _update_summary_json(summary_path, {"rotation_set_minutes": {"enabled": False}})
        return
    if resolved_model_dir is None:
        typer.echo(
            "[rotation_minutes] WARNING: rotation enabled but model_dir missing; keeping baseline.",
            err=True,
        )
        _update_summary_json(
            summary_path,
            {"rotation_set_minutes": {"enabled": True, "error": "model_dir_missing"}},
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
    allow_priors_fallback = bool(config.allow_priors_fallback)

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

    minutes_features_path = (
        Path(minutes_features_root)
        / day.strftime("%Y-%m-%d")
        / f"run={run_id}"
        / "features.parquet"
    )
    if not minutes_features_path.exists():
        typer.echo(
            f"[rotation_minutes] WARNING: minutes features missing at {minutes_features_path}; keeping baseline.",
            err=True,
        )
        _update_summary_json(
            summary_path,
            {
                "rotation_set_minutes": {
                    "enabled": True,
                    "error": "minutes_features_missing",
                }
            },
        )
        return
    minutes_feat = pd.read_parquet(minutes_features_path)

    # 3) Build rotation model features + write them to the live features root.
    out_day_dir = Path(rotation_features_root) / day.strftime("%Y-%m-%d")
    out_run_dir = out_day_dir / f"run={run_id}"
    rot_features_path = out_run_dir / "features.parquet"
    if rot_features_path.exists():
        typer.echo(
            f"[rotation_minutes] using prebuilt rotation features at {rot_features_path}",
            err=True,
        )
        rot_features = pd.read_parquet(rot_features_path)
        if rot_features.empty:
            typer.echo(
                "[rotation_minutes] WARNING: rotation features empty; keeping baseline.",
                err=True,
            )
            _update_summary_json(
                summary_path,
                {
                    "rotation_set_minutes": {
                        "enabled": True,
                        "error": "rotation_features_empty",
                    }
                },
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
            typer.echo(
                f"[rotation_minutes] WARNING: feature build failed ({exc}); keeping baseline.",
                err=True,
            )
            _update_summary_json(
                summary_path,
                {
                    "rotation_set_minutes": {
                        "enabled": True,
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

    # 4) Predict rotation minutes (p50) and align to baseline output rows.
    try:
        rot_scored = predict_rotation_minutes(
            rot_features,
            model_dir=Path(resolved_model_dir),
            device=str(device),
            batch_size=int(batch_size),
        )
    except Exception as exc:  # noqa: BLE001
        typer.echo(
            f"[rotation_minutes] WARNING: rotation model predict failed ({exc}); keeping baseline.",
            err=True,
        )
        _update_summary_json(
            summary_path,
            {
                "rotation_set_minutes": {
                    "enabled": True,
                    "error": f"predict_failed:{exc}",
                }
            },
        )
        return

    rot_pred = rot_scored.loc[
        :, ["game_id", "team_id", "player_id", "pred_minutes"]
    ].rename(columns={"pred_minutes": "rotation_minutes_p50"})
    rot_pred["rotation_minutes_p50"] = (
        pd.to_numeric(rot_pred["rotation_minutes_p50"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )

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

    # 4b) ESPN OUT override: Load ESPN injuries and apply same override as baseline scorer.
    # ESPN updates faster than NBA injury reports; this catches late scratches.
    espn_out_count = 0
    try:
        espn_out_players = _load_espn_out_players(
            day.date() if hasattr(day, "date") else day,
            data_root=data_root,
            run_as_of_ts=run_as_of_datetime,
        )
        if espn_out_players and "player_name" in guard.columns:
            normalized_names = guard["player_name"].astype(str).map(_normalize_name_for_matching)
            espn_mask = normalized_names.isin(espn_out_players)
            espn_out_count = int(espn_mask.sum())
            if espn_out_count > 0:
                guard.loc[espn_mask, "is_out"] = 1
                guard.loc[espn_mask, "status"] = "OUT"
                espn_names = guard.loc[espn_mask, "player_name"].tolist()
                typer.echo(
                    f"[rotation_minutes] ESPN OUT override: {espn_out_count} players marked OUT: {espn_names[:10]}",
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
    )
    new_p50 = guardrail.minutes_p50

    # 4b) Post-guardrail DNP override: for players with strong DNP-CD signals,
    # override their minutes to 0 regardless of baseline. This allows us to use
    # baseline for normal players while still catching players like Kuminga who
    # are healthy scratches with a consistent DNP pattern.
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
                # Delta diagnostics: per-team-game metrics quantifying rotation overlay impact.
                "delta_diagnostics": {
                    "per_team_game": delta_diagnostics,
                },
            }
        },
    )

    typer.echo(f"[rotation_minutes] wrote guarded minutes -> {minutes_path}", err=True)


if __name__ == "__main__":
    app()
