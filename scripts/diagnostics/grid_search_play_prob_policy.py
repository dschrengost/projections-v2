"""Grid-search guarded-v2 play-prob policy knobs on historical labeled slates.

This tuner replays `play_prob_policy` on `effective_minutes.parquet` snapshots and
scores candidates against realized "played" labels (`minutes > 0`) from
`labels/season=*/boxscore_labels.parquet`.

Example:
  uv run python -m scripts.diagnostics.grid_search_play_prob_policy \
    --start 2026-01-15 --end 2026-02-09 --holdout-days 5
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
import json
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from projections.paths import data_path
from projections.pipeline import control_plane
from projections.sim_v2.config import DEFAULT_PROFILES_PATH, PlayProbPolicyConfig, load_sim_v2_profile
from projections.sim_v2.play_prob_policy_tuning import (
    PolicyObjectiveWeights,
    build_policy_grid_overrides,
    evaluate_policy_candidate,
)

app = typer.Typer(add_completion=False)

EFFECTIVE_MINUTES_FILENAME = "effective_minutes.parquet"
DEFAULT_OUTPUT_ROOT = Path("artifacts") / "tuning" / "play_prob_policy"

REQUIRED_POLICY_COLUMNS: tuple[str, ...] = (
    "player_id",
    "player_name",
    "game_id",
    "team_id",
    "play_prob",
    "rotation_prob",
    "minutes_p50",
    "minutes_p50_cond",
    "status_bucket",
    "status",
    "injury_status",
    "is_starter",
    "starter_flag",
    "dc_role",
    "dc_ahead_global",
    "consecutive_active_dnp",
    "active_but_dnp_rate_last10",
    "inactive_streak_len",
)

TUNED_KNOBS: tuple[str, ...] = (
    "starter_floor",
    "core_floor",
    "core_lock_min_cond_p50",
    "core_lock_topk",
    "max_floor_delta",
    "min_raw_play_prob_for_floor",
    "min_rotation_prob_for_floor",
    "depth_block_min_ahead_global",
    "dnp_block_streak_threshold",
    "dnp_block_rate_threshold",
    "dnp_block_inactive_streak_threshold",
)


def _utc_run_id() -> str:
    return datetime.now(tz=UTC).strftime("play_prob_grid_%Y%m%dT%H%M%SZ")


def _parse_float_grid(raw: str) -> list[float]:
    values: list[float] = []
    for token in (raw or "").split(","):
        text = token.strip()
        if not text:
            continue
        values.append(float(text))
    return values


def _parse_int_grid(raw: str) -> list[int]:
    values: list[int] = []
    for token in (raw or "").split(","):
        text = token.strip()
        if not text:
            continue
        values.append(int(text))
    return values


def _season_from_day(day: date) -> int:
    return int(day.year) if int(day.month) >= 8 else int(day.year) - 1


def _iter_days(start_day: date, end_day: date) -> list[date]:
    if end_day < start_day:
        return []
    cur = start_day
    out: list[date] = []
    while cur <= end_day:
        out.append(cur)
        cur = cur + timedelta(days=1)
    return out


def _resolve_range_from_artifacts(
    *,
    root: Path,
    lookback_days: int,
) -> tuple[date, date]:
    base = root / "artifacts" / "minutes_v1" / "daily"
    if not base.exists():
        raise typer.BadParameter(f"Missing minutes daily root: {base}")
    days = sorted([p.name for p in base.iterdir() if p.is_dir()])
    if not days:
        raise typer.BadParameter(f"No game_date directories found under {base}")

    latest: date | None = None
    for day_str in reversed(days):
        try:
            day = pd.Timestamp(day_str).date()
        except Exception:  # noqa: BLE001
            continue
        if (base / day_str / "latest_run.json").exists():
            latest = day
            break
    if latest is None:
        raise typer.BadParameter(f"No latest_run.json pointers found under {base}")
    start = latest - timedelta(days=max(int(lookback_days) - 1, 0))
    return start, latest


def _resolve_minutes_run(day_dir: Path) -> str | None:
    run_id = control_plane.read_promoted_run_id(day_dir)
    if run_id:
        return run_id
    if control_plane.allow_unpromoted_run_reads():
        run_dirs = sorted([p for p in day_dir.glob("run=*") if p.is_dir()], reverse=True)
        if run_dirs:
            return run_dirs[0].name.split("=", 1)[1]
    return None


def _load_effective_minutes_for_day(root: Path, day: date) -> tuple[pd.DataFrame | None, str | None]:
    day_str = day.isoformat()
    day_dir = root / "artifacts" / "minutes_v1" / "daily" / day_str
    if not day_dir.exists():
        return None, None
    run_id = _resolve_minutes_run(day_dir)
    if not run_id:
        return None, None
    path = day_dir / f"run={run_id}" / EFFECTIVE_MINUTES_FILENAME
    if not path.exists():
        return None, run_id

    df = pd.read_parquet(path)
    keep = [c for c in REQUIRED_POLICY_COLUMNS if c in df.columns]
    out = df[keep].copy()
    out["game_date"] = day_str

    for col in ("player_id", "game_id", "team_id"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["player_id"]).copy()
    out["player_id"] = out["player_id"].astype(int)
    if "game_id" in out.columns:
        out = out.dropna(subset=["game_id"]).copy()
        out["game_id"] = out["game_id"].astype(int)
    if "team_id" in out.columns:
        out = out.dropna(subset=["team_id"]).copy()
        out["team_id"] = out["team_id"].astype(int)
    return out, run_id


def _load_labels_for_days(root: Path, days: list[date]) -> pd.DataFrame:
    if not days:
        return pd.DataFrame(columns=["game_date", "player_id", "plays_target", "minutes_actual"])

    by_season: dict[int, list[date]] = {}
    for day in days:
        by_season.setdefault(_season_from_day(day), []).append(day)

    frames: list[pd.DataFrame] = []
    for season, season_days in sorted(by_season.items()):
        path = root / "labels" / f"season={season}" / "boxscore_labels.parquet"
        if not path.exists():
            continue
        labels = pd.read_parquet(path)
        required = {"game_date", "player_id", "minutes"}
        if not required.issubset(labels.columns):
            continue
        keep_cols = [c for c in ("game_date", "game_id", "team_id", "player_id", "minutes") if c in labels.columns]
        labels = labels[keep_cols].copy()
        labels["game_date"] = pd.to_datetime(labels["game_date"], errors="coerce").dt.date
        labels = labels[labels["game_date"].isin(set(season_days))].copy()
        if labels.empty:
            continue
        labels["player_id"] = pd.to_numeric(labels["player_id"], errors="coerce")
        labels = labels.dropna(subset=["player_id"]).copy()
        labels["player_id"] = labels["player_id"].astype(int)
        if "game_id" in labels.columns:
            labels["game_id"] = pd.to_numeric(labels["game_id"], errors="coerce")
            labels = labels.dropna(subset=["game_id"]).copy()
            labels["game_id"] = labels["game_id"].astype(int)
        if "team_id" in labels.columns:
            labels["team_id"] = pd.to_numeric(labels["team_id"], errors="coerce")
            labels = labels.dropna(subset=["team_id"]).copy()
            labels["team_id"] = labels["team_id"].astype(int)
        labels["minutes_actual"] = pd.to_numeric(labels["minutes"], errors="coerce").fillna(0.0)
        labels["plays_target"] = (labels["minutes_actual"] > 0.0).astype(int)
        labels["game_date"] = labels["game_date"].astype(str)
        labels = labels.drop(columns=["minutes"], errors="ignore")
        join_keys = [k for k in ("game_date", "game_id", "team_id", "player_id") if k in labels.columns]
        labels = labels.drop_duplicates(subset=join_keys, keep="last")
        frames.append(labels)

    if not frames:
        return pd.DataFrame(columns=["game_date", "player_id", "plays_target", "minutes_actual"])
    return pd.concat(frames, ignore_index=True)


def _join_eval_frame(pred_df: pd.DataFrame, labels_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if pred_df.empty or labels_df.empty:
        return pd.DataFrame(), []
    join_keys = [k for k in ("game_date", "game_id", "team_id", "player_id") if k in pred_df.columns and k in labels_df.columns]
    if "game_date" not in join_keys or "player_id" not in join_keys:
        join_keys = [k for k in ("game_date", "player_id") if k in pred_df.columns and k in labels_df.columns]
    pred = pred_df.drop_duplicates(subset=join_keys, keep="last").copy()
    lab = labels_df.drop_duplicates(subset=join_keys, keep="last").copy()
    merged = pred.merge(lab, on=join_keys, how="inner")
    if merged.empty:
        return merged, join_keys
    merged["plays_target"] = pd.to_numeric(merged["plays_target"], errors="coerce").fillna(0).astype(int)
    merged["minutes_actual"] = pd.to_numeric(merged["minutes_actual"], errors="coerce").fillna(0.0)
    return merged, join_keys


def _split_train_holdout(eval_df: pd.DataFrame, *, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if eval_df.empty or int(holdout_days) <= 0:
        return eval_df, eval_df.iloc[0:0].copy(), []
    unique_days = sorted({str(d) for d in eval_df["game_date"].tolist()})
    if len(unique_days) <= int(holdout_days):
        return eval_df, eval_df.iloc[0:0].copy(), []
    holdout_set = set(unique_days[-int(holdout_days):])
    holdout = eval_df[eval_df["game_date"].isin(holdout_set)].copy()
    train = eval_df[~eval_df["game_date"].isin(holdout_set)].copy()
    return train, holdout, sorted(holdout_set)


def _prefix_metrics(metrics: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in metrics.items()}


def _count_violations(
    row: dict[str, Any],
    *,
    rank_prefix: str,
    max_brier_regression: float | None,
    max_logloss_regression: float | None,
    max_fringe_false_active_p90: float | None,
    min_starter_mean_p_eff: float | None,
) -> tuple[int, str]:
    violations: list[str] = []

    if max_brier_regression is not None:
        v = float(row.get(f"{rank_prefix}brier_delta_eff_minus_raw", 0.0))
        if v > float(max_brier_regression):
            violations.append("brier_regression")
    if max_logloss_regression is not None:
        v = float(row.get(f"{rank_prefix}logloss_delta_eff_minus_raw", 0.0))
        if v > float(max_logloss_regression):
            violations.append("logloss_regression")
    if max_fringe_false_active_p90 is not None:
        v = float(row.get(f"{rank_prefix}fringe_false_active_p90_eff", 0.0))
        if v > float(max_fringe_false_active_p90):
            violations.append("fringe_false_active_p90")
    if min_starter_mean_p_eff is not None:
        v = float(row.get(f"{rank_prefix}starter_mean_p_eff", float("nan")))
        if pd.isna(v) or v < float(min_starter_mean_p_eff):
            violations.append("starter_mean_p_eff")
    return len(violations), ",".join(violations)


@app.command()
def main(
    start: str | None = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)."),
    end: str | None = typer.Option(None, "--end", help="End date (YYYY-MM-DD)."),
    lookback_days: int = typer.Option(28, "--lookback-days", min=1, help="Used when --start/--end are omitted."),
    holdout_days: int = typer.Option(5, "--holdout-days", min=0, help="Most recent slates reserved for ranking."),
    profile: str = typer.Option("sim_v3", "--profile", help="Base sim_v2 profile for defaults."),
    profiles_path: Path | None = typer.Option(None, "--profiles-path", help="Override sim profiles JSON path."),
    mode: str = typer.Option("guarded_v2", "--mode", help="Policy mode to evaluate."),
    bins: int = typer.Option(10, "--bins", min=2, help="ECE bins."),
    data_root: Path | None = typer.Option(None, "--data-root", help="Defaults to PROJECTIONS_DATA_ROOT."),
    output_root: Path = typer.Option(DEFAULT_OUTPUT_ROOT, "--output-root", help="Output root for tuning runs."),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional output run id."),
    max_candidates: int = typer.Option(4000, "--max-candidates", min=1),
    top_n: int = typer.Option(20, "--top-n", min=1),
    objective_brier_weight: float = typer.Option(1.0, "--objective-brier-weight"),
    objective_false_active_p90_weight: float = typer.Option(0.25, "--objective-false-active-p90-weight"),
    objective_fringe_false_active_p90_weight: float = typer.Option(0.50, "--objective-fringe-false-active-p90-weight"),
    objective_starter_under95_weight: float = typer.Option(0.20, "--objective-starter-under95-weight"),
    max_brier_regression: float | None = typer.Option(0.0, "--max-brier-regression", help="Constraint on brier delta."),
    max_logloss_regression: float | None = typer.Option(0.0, "--max-logloss-regression", help="Constraint on logloss delta."),
    max_fringe_false_active_p90: float | None = typer.Option(None, "--max-fringe-false-active-p90"),
    min_starter_mean_p_eff: float | None = typer.Option(None, "--min-starter-mean-p-eff"),
    starter_floor_grid: str = typer.Option("", "--starter-floor-grid", help="Comma list; empty uses profile value."),
    core_floor_grid: str = typer.Option("0.88,0.90,0.92", "--core-floor-grid"),
    core_lock_min_cond_p50_grid: str = typer.Option("22,24,26", "--core-lock-min-cond-p50-grid"),
    core_lock_topk_grid: str = typer.Option("3,4", "--core-lock-topk-grid"),
    max_floor_delta_grid: str = typer.Option("0.15,0.20,0.25", "--max-floor-delta-grid"),
    min_raw_play_prob_for_floor_grid: str = typer.Option("0.30,0.35", "--min-raw-play-prob-for-floor-grid"),
    min_rotation_prob_for_floor_grid: str = typer.Option("0.60,0.65,0.70", "--min-rotation-prob-for-floor-grid"),
    depth_block_min_ahead_global_grid: str = typer.Option("", "--depth-block-min-ahead-global-grid"),
    dnp_block_streak_threshold_grid: str = typer.Option("", "--dnp-block-streak-threshold-grid"),
    dnp_block_rate_threshold_grid: str = typer.Option("", "--dnp-block-rate-threshold-grid"),
    dnp_block_inactive_streak_threshold_grid: str = typer.Option("", "--dnp-block-inactive-streak-threshold-grid"),
) -> None:
    root = data_root or data_path()

    if start and end:
        start_day = pd.Timestamp(start).date()
        end_day = pd.Timestamp(end).date()
    elif not start and not end:
        start_day, end_day = _resolve_range_from_artifacts(root=root, lookback_days=int(lookback_days))
    else:
        raise typer.BadParameter("Provide both --start and --end, or neither.")
    if end_day < start_day:
        raise typer.BadParameter("--end must be >= --start")

    profiles_path_eff = profiles_path or DEFAULT_PROFILES_PATH
    profile_cfg = load_sim_v2_profile(profile=profile, profiles_path=profiles_path_eff)
    base_policy: PlayProbPolicyConfig = profile_cfg.play_prob_policy

    grid_map: dict[str, list[Any]] = {
        "starter_floor": _parse_float_grid(starter_floor_grid),
        "core_floor": _parse_float_grid(core_floor_grid),
        "core_lock_min_cond_p50": _parse_float_grid(core_lock_min_cond_p50_grid),
        "core_lock_topk": _parse_int_grid(core_lock_topk_grid),
        "max_floor_delta": _parse_float_grid(max_floor_delta_grid),
        "min_raw_play_prob_for_floor": _parse_float_grid(min_raw_play_prob_for_floor_grid),
        "min_rotation_prob_for_floor": _parse_float_grid(min_rotation_prob_for_floor_grid),
        "depth_block_min_ahead_global": _parse_int_grid(depth_block_min_ahead_global_grid),
        "dnp_block_streak_threshold": _parse_float_grid(dnp_block_streak_threshold_grid),
        "dnp_block_rate_threshold": _parse_float_grid(dnp_block_rate_threshold_grid),
        "dnp_block_inactive_streak_threshold": _parse_float_grid(dnp_block_inactive_streak_threshold_grid),
    }

    candidates = build_policy_grid_overrides(base_policy, grids=grid_map, include_baseline=True)
    if len(candidates) > int(max_candidates):
        raise typer.BadParameter(
            f"Candidate grid too large: {len(candidates)} > max_candidates={int(max_candidates)}. Narrow grid options."
        )

    days = _iter_days(start_day, end_day)
    pred_frames: list[pd.DataFrame] = []
    run_map: dict[str, str] = {}
    for day in days:
        frame, run_id_day = _load_effective_minutes_for_day(root, day)
        if frame is None or frame.empty:
            continue
        pred_frames.append(frame)
        if run_id_day:
            run_map[day.isoformat()] = run_id_day

    if not pred_frames:
        raise typer.BadParameter("No effective_minutes snapshots found for selected date range.")

    pred_df = pd.concat(pred_frames, ignore_index=True)
    labels_df = _load_labels_for_days(root, days)
    eval_df, join_keys = _join_eval_frame(pred_df, labels_df)
    if eval_df.empty:
        raise typer.BadParameter("No joined rows between effective_minutes and labels for selected range.")

    train_df, holdout_df, holdout_dates = _split_train_holdout(eval_df, holdout_days=int(holdout_days))
    rank_prefix = "holdout_" if not holdout_df.empty else "train_"

    weights = PolicyObjectiveWeights(
        brier=float(objective_brier_weight),
        false_active_p90=float(objective_false_active_p90_weight),
        fringe_false_active_p90=float(objective_fringe_false_active_p90_weight),
        starter_under95=float(objective_starter_under95_weight),
    )

    rows: list[dict[str, Any]] = []
    total = len(candidates)
    for idx, overrides in enumerate(candidates, start=1):
        cfg = replace(
            base_policy,
            enabled=True,
            mode=str(mode).strip().lower(),
            **overrides,
        )
        train_metrics = evaluate_policy_candidate(train_df, cfg, bins=int(bins), weights=weights)
        row: dict[str, Any] = {f"cfg_{k}": overrides[k] for k in TUNED_KNOBS}
        row.update(_prefix_metrics(train_metrics, prefix="train_"))
        row["cfg_mode"] = str(mode).strip().lower()

        if not holdout_df.empty:
            hold_metrics = evaluate_policy_candidate(holdout_df, cfg, bins=int(bins), weights=weights)
            row.update(_prefix_metrics(hold_metrics, prefix="holdout_"))
        else:
            for key in train_metrics:
                row[f"holdout_{key}"] = float("nan")

        all_metrics = evaluate_policy_candidate(eval_df, cfg, bins=int(bins), weights=weights)
        row.update(_prefix_metrics(all_metrics, prefix="all_"))

        rank_obj = float(row.get(f"{rank_prefix}objective", float("inf")))
        row["rank_objective"] = rank_obj
        violation_count, violation_tags = _count_violations(
            row,
            rank_prefix=rank_prefix,
            max_brier_regression=max_brier_regression,
            max_logloss_regression=max_logloss_regression,
            max_fringe_false_active_p90=max_fringe_false_active_p90,
            min_starter_mean_p_eff=min_starter_mean_p_eff,
        )
        row["constraint_violations"] = int(violation_count)
        row["constraint_tags"] = violation_tags
        rows.append(row)

        if idx % 50 == 0 or idx == total:
            typer.echo(f"[play-prob-grid] evaluated {idx}/{total} candidates")

    results = pd.DataFrame(rows)
    if results.empty:
        raise typer.Exit(code=2)

    results = results.sort_values(
        by=["constraint_violations", "rank_objective", f"{rank_prefix}brier_eff"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    resolved_run_id = run_id or _utc_run_id()
    out_dir = output_root / resolved_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path_parquet = out_dir / "results.parquet"
    results_path_csv = out_dir / "results.csv"
    top_path_csv = out_dir / "top_candidates.csv"
    summary_path = out_dir / "summary.json"
    best_config_path = out_dir / "best_play_prob_policy.json"

    results.to_parquet(results_path_parquet, index=False)
    results.to_csv(results_path_csv, index=False)

    view_cols = [
        "constraint_violations",
        "rank_objective",
        f"{rank_prefix}brier_eff",
        f"{rank_prefix}brier_delta_eff_minus_raw",
        f"{rank_prefix}fringe_false_active_p90_eff",
        f"{rank_prefix}starter_mean_p_eff",
        "cfg_core_floor",
        "cfg_core_lock_min_cond_p50",
        "cfg_core_lock_topk",
        "cfg_max_floor_delta",
        "cfg_min_raw_play_prob_for_floor",
        "cfg_min_rotation_prob_for_floor",
        "cfg_depth_block_min_ahead_global",
        "cfg_dnp_block_streak_threshold",
        "cfg_dnp_block_rate_threshold",
        "cfg_dnp_block_inactive_streak_threshold",
    ]
    existing_view_cols = [c for c in view_cols if c in results.columns]
    top_df = results.loc[:, existing_view_cols].head(int(top_n)).copy()
    top_df.to_csv(top_path_csv, index=False)

    best = results.iloc[0].to_dict()
    best_config = {
        "enabled": True,
        "mode": str(best.get("cfg_mode", str(mode).strip().lower())),
        "rotation_lock_floor": float(base_policy.rotation_lock_floor),
        "rotation_lock_min_cond_p50": float(base_policy.rotation_lock_min_cond_p50),
        "rotation_lock_topk": int(base_policy.rotation_lock_topk),
        "probable_floor": float(base_policy.probable_floor),
        "starter_floor": float(best["cfg_starter_floor"]),
        "core_floor": float(best["cfg_core_floor"]),
        "core_lock_min_cond_p50": float(best["cfg_core_lock_min_cond_p50"]),
        "core_lock_topk": int(best["cfg_core_lock_topk"]),
        "max_floor_delta": float(best["cfg_max_floor_delta"]),
        "min_raw_play_prob_for_floor": float(best["cfg_min_raw_play_prob_for_floor"]),
        "min_rotation_prob_for_floor": float(best["cfg_min_rotation_prob_for_floor"]),
        "depth_block_roles": list(base_policy.depth_block_roles),
        "depth_block_min_ahead_global": int(best["cfg_depth_block_min_ahead_global"]),
        "dnp_block_streak_threshold": float(best["cfg_dnp_block_streak_threshold"]),
        "dnp_block_rate_threshold": float(best["cfg_dnp_block_rate_threshold"]),
        "dnp_block_inactive_streak_threshold": float(best["cfg_dnp_block_inactive_streak_threshold"]),
        "require_fresh_injury_snapshot": bool(base_policy.require_fresh_injury_snapshot),
        "freshness_minutes": float(base_policy.freshness_minutes),
    }
    best_config_path.write_text(json.dumps(best_config, indent=2), encoding="utf-8")

    summary_payload = {
        "run_id": resolved_run_id,
        "profile": profile,
        "profiles_path": str(Path(profiles_path_eff).resolve()),
        "date_range": {"start": start_day.isoformat(), "end": end_day.isoformat()},
        "n_days_requested": len(days),
        "n_days_with_effective_minutes": len({str(v) for v in pred_df["game_date"].tolist()}),
        "n_eval_rows": int(len(eval_df)),
        "n_train_rows": int(len(train_df)),
        "n_holdout_rows": int(len(holdout_df)),
        "holdout_dates": holdout_dates,
        "rank_split": rank_prefix.rstrip("_"),
        "join_keys": join_keys,
        "n_candidates": int(len(results)),
        "objective_weights": {
            "brier": float(weights.brier),
            "false_active_p90": float(weights.false_active_p90),
            "fringe_false_active_p90": float(weights.fringe_false_active_p90),
            "starter_under95": float(weights.starter_under95),
        },
        "constraints": {
            "max_brier_regression": max_brier_regression,
            "max_logloss_regression": max_logloss_regression,
            "max_fringe_false_active_p90": max_fringe_false_active_p90,
            "min_starter_mean_p_eff": min_starter_mean_p_eff,
        },
        "artifacts": {
            "results_parquet": str(results_path_parquet),
            "results_csv": str(results_path_csv),
            "top_candidates_csv": str(top_path_csv),
            "best_play_prob_policy_json": str(best_config_path),
        },
        "effective_minutes_runs": run_map,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")

    typer.echo(f"[play-prob-grid] wrote {results_path_parquet}")
    typer.echo(f"[play-prob-grid] wrote {results_path_csv}")
    typer.echo(f"[play-prob-grid] wrote {top_path_csv}")
    typer.echo(f"[play-prob-grid] wrote {best_config_path}")
    typer.echo(f"[play-prob-grid] wrote {summary_path}")
    typer.echo("")
    typer.echo(top_df.to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    app()
