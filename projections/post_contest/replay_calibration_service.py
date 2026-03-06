from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from projections.paths import get_data_root


@dataclass(frozen=True)
class ReplayCalibrationBundle:
    player_fpts_calibration_path: Path
    player_minutes_calibration_path: Path
    ownership_recalibration_path: Path
    field_model_calibration_path: Path
    optimizer_regret_by_contest_path: Path
    optimizer_regret_by_bucket_path: Path
    optimizer_regret_examples_path: Path
    summary_path: Path

    def to_dict(self) -> Dict[str, str]:
        return {
            "player_fpts_calibration_path": str(self.player_fpts_calibration_path),
            "player_minutes_calibration_path": str(self.player_minutes_calibration_path),
            "ownership_recalibration_path": str(self.ownership_recalibration_path),
            "field_model_calibration_path": str(self.field_model_calibration_path),
            "optimizer_regret_by_contest_path": str(self.optimizer_regret_by_contest_path),
            "optimizer_regret_by_bucket_path": str(self.optimizer_regret_by_bucket_path),
            "optimizer_regret_examples_path": str(self.optimizer_regret_examples_path),
            "summary_path": str(self.summary_path),
        }


def calibration_output_dir(data_root: Optional[Path] = None) -> Path:
    root = data_root or get_data_root()
    return root / "gold" / "replay_calibration"


def discover_replay_analytics_dirs(data_root: Optional[Path] = None) -> List[Path]:
    root = data_root or get_data_root()
    analytics_root = root / "analytics" / "contest_flashback"
    if not analytics_root.exists():
        return []
    out: List[Path] = []
    for summary_path in analytics_root.rglob("analytics/summary.json"):
        out.append(summary_path.parent)
    return sorted(set(out))


def _load_concat(paths: Iterable[Path], file_name: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for analytics_dir in paths:
        path = analytics_dir / file_name
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _field_size_bucket(series: pd.Series) -> pd.Series:
    return pd.cut(
        series.astype(float),
        bins=[0, 500, 2000, 5000, 20000, np.inf],
        labels=["tiny", "small", "medium", "large", "massive"],
        right=False,
    ).astype("string")


def _player_bucket_frame(
    player_df: pd.DataFrame,
    *,
    actual_col: str,
    sim_mean_col: str,
    p10_col: str,
    p90_col: str,
    percentile_col: str,
) -> pd.DataFrame:
    required = [actual_col, sim_mean_col, p10_col, p90_col, percentile_col, "proj_fpts", "proj_ownership_pct"]
    frame = player_df.dropna(subset=[col for col in required if col in player_df.columns]).copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "proj_fpts_bucket",
                "proj_ownership_bucket",
                "sample_count",
                "actual_mean",
                "sim_mean",
                "mean_bias",
                "percentile_mean",
                "below_p10_rate",
                "above_p90_rate",
                "outside_p10_p90_rate",
                "recommended_mean_shift",
                "recommended_variance_scale",
            ]
        )

    frame["proj_fpts_bucket"] = pd.cut(
        frame["proj_fpts"].astype(float),
        bins=[-np.inf, 10, 20, 30, 40, 50, 70, np.inf],
        labels=["<=10", "10_20", "20_30", "30_40", "40_50", "50_70", "70_plus"],
    ).astype("string")
    frame["proj_ownership_bucket"] = pd.cut(
        frame["proj_ownership_pct"].astype(float),
        bins=[-np.inf, 1, 5, 10, 20, 35, 50, 70, np.inf],
        labels=["<=1", "1_5", "5_10", "10_20", "20_35", "35_50", "50_70", "70_plus"],
    ).astype("string")
    frame["actual_percentile"] = frame[percentile_col].astype(float)
    frame["below_p10"] = (frame[actual_col].astype(float) < frame[p10_col].astype(float)).astype(float)
    frame["above_p90"] = (frame[actual_col].astype(float) > frame[p90_col].astype(float)).astype(float)

    grouped = (
        frame.groupby(["proj_fpts_bucket", "proj_ownership_bucket"], dropna=False)
        .agg(
            sample_count=(actual_col, "size"),
            actual_mean=(actual_col, "mean"),
            sim_mean=(sim_mean_col, "mean"),
            percentile_mean=("actual_percentile", "mean"),
            below_p10_rate=("below_p10", "mean"),
            above_p90_rate=("above_p90", "mean"),
        )
        .reset_index()
    )
    grouped["mean_bias"] = grouped["actual_mean"] - grouped["sim_mean"]
    grouped["outside_p10_p90_rate"] = grouped["below_p10_rate"] + grouped["above_p90_rate"]
    grouped["recommended_mean_shift"] = grouped["mean_bias"]
    grouped["recommended_variance_scale"] = np.clip(grouped["outside_p10_p90_rate"] / 0.20, 0.5, 2.0)
    return grouped


def _ownership_recalibration_frame(player_df: pd.DataFrame) -> pd.DataFrame:
    required = ["proj_ownership_pct", "actual_contest_own_pct", "actual_opponent_own_pct", "modeled_field_own_pct"]
    frame = player_df.dropna(subset=[col for col in required if col in player_df.columns]).copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "projected_ownership_bucket",
                "sample_count",
                "mean_projected_own",
                "mean_actual_contest_own",
                "mean_actual_opponent_own",
                "mean_modeled_field_own",
                "recommended_delta",
                "recommended_multiplier",
                "monotone_target_own",
            ]
        )
    frame["projected_ownership_bucket"] = pd.cut(
        frame["proj_ownership_pct"].astype(float),
        bins=[-np.inf, 1, 2, 5, 10, 20, 35, 50, 70, np.inf],
        labels=["<=1", "1_2", "2_5", "5_10", "10_20", "20_35", "35_50", "50_70", "70_plus"],
    ).astype("string")
    grouped = (
        frame.groupby("projected_ownership_bucket", dropna=False)
        .agg(
            sample_count=("proj_ownership_pct", "size"),
            mean_projected_own=("proj_ownership_pct", "mean"),
            mean_actual_contest_own=("actual_contest_own_pct", "mean"),
            mean_actual_opponent_own=("actual_opponent_own_pct", "mean"),
            mean_modeled_field_own=("modeled_field_own_pct", "mean"),
        )
        .reset_index()
        .sort_values("mean_projected_own")
        .reset_index(drop=True)
    )
    grouped["recommended_delta"] = grouped["mean_actual_contest_own"] - grouped["mean_projected_own"]
    grouped["recommended_multiplier"] = np.where(
        grouped["mean_projected_own"].abs() > 1e-9,
        grouped["mean_actual_contest_own"] / grouped["mean_projected_own"],
        np.nan,
    )
    grouped["monotone_target_own"] = grouped["mean_actual_contest_own"].cummax()
    return grouped


def _field_model_calibration_frame(field_df: pd.DataFrame) -> pd.DataFrame:
    if field_df.empty:
        return pd.DataFrame()
    frame = field_df.copy()
    frame["field_size_bucket"] = _field_size_bucket(frame["actual_field_size"])
    frame["dupe_rate_gap"] = frame["modeled_dupe_rate"].astype(float) - frame["actual_dupe_rate"].astype(float)
    frame["salary_left_gap"] = frame["modeled_salary_left_mean"].astype(float) - frame["actual_salary_left_mean"].astype(float)
    frame["projected_own_sum_gap"] = (
        frame["modeled_projected_own_sum_mean"].astype(float) - frame["actual_projected_own_sum_mean"].astype(float)
    )
    grouped = (
        frame.groupby("field_size_bucket", dropna=False)
        .agg(
            contest_count=("contest_id", "size"),
            mean_actual_field_size=("actual_field_size", "mean"),
            mean_player_ownership_mae_pct=("player_ownership_mae_pct", "mean"),
            mean_top20_player_ownership_mae_pct=("top20_player_ownership_mae_pct", "mean"),
            mean_dupe_rate_gap=("dupe_rate_gap", "mean"),
            mean_salary_left_gap=("salary_left_gap", "mean"),
            mean_projected_own_sum_gap=("projected_own_sum_gap", "mean"),
            mean_salary_left_hist_l1=("salary_left_hist_l1", "mean"),
            mean_projected_own_sum_hist_l1=("projected_own_sum_hist_l1", "mean"),
            mean_dupe_hist_l1=("dupe_hist_l1", "mean"),
        )
        .reset_index()
    )
    return grouped


def _optimizer_regret_frames(regret_df: pd.DataFrame, lineup_df: pd.DataFrame, field_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if regret_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    by_contest = regret_df.copy()
    if not field_df.empty and {"game_date", "contest_id", "draft_group_id", "actual_field_size"}.issubset(field_df.columns):
        join_cols = ["game_date", "contest_id", "draft_group_id"]
        # Field calibration can contain multiple rows per contest key (e.g. multiple user runs).
        # Collapse before joining to avoid cartesian multiplication of regret rows.
        field_join = (
            field_df[join_cols + ["actual_field_size"]]
            .assign(actual_field_size=lambda df: pd.to_numeric(df["actual_field_size"], errors="coerce"))
            .groupby(join_cols, dropna=False, as_index=False)
            .agg(actual_field_size=("actual_field_size", "mean"))
        )
        by_contest = by_contest.merge(
            field_join,
            on=join_cols,
            how="left",
        )
        by_contest["field_size_bucket"] = _field_size_bucket(by_contest["actual_field_size"])
    else:
        by_contest["field_size_bucket"] = pd.Series(["unknown"] * len(by_contest), dtype="string")
    by_contest["positive_regret"] = by_contest["selection_regret_roi"].fillna(0).astype(float) > 0

    by_bucket = (
        by_contest.groupby("field_size_bucket", dropna=False)
        .agg(
            contest_count=("contest_id", "size"),
            candidate_pool_available_rate=("candidate_pool_available", "mean"),
            positive_regret_rate=("positive_regret", "mean"),
            mean_selection_regret_roi=("selection_regret_roi", "mean"),
            median_selection_regret_roi=("selection_regret_roi", "median"),
            mean_selection_regret_cash_rate=("selection_regret_cash_rate", "mean"),
        )
        .reset_index()
    )

    examples = pd.DataFrame()
    if not lineup_df.empty:
        candidate = lineup_df[lineup_df["lineup_source"] == "candidate"].copy()
        entered = lineup_df[lineup_df["lineup_source"] == "entered"].copy()
        if not candidate.empty and not entered.empty:
            group_cols = ["game_date", "contest_id", "draft_group_id"]
            best_candidate = candidate.sort_values(["sim_roi", "sim_cash_rate"], ascending=False).groupby(group_cols, as_index=False).head(1)
            best_entered = entered.sort_values(["sim_roi", "sim_cash_rate"], ascending=False).groupby(group_cols, as_index=False).head(1)
            examples = best_candidate.merge(
                best_entered[group_cols + ["lineup_key", "sim_roi", "sim_cash_rate", "realized_rank", "realized_prize"]],
                on=group_cols,
                how="left",
                suffixes=("_candidate", "_entered"),
            )
            examples["selection_regret_roi"] = examples["sim_roi_candidate"].astype(float) - examples["sim_roi_entered"].astype(float)
            examples["selection_regret_cash_rate"] = examples["sim_cash_rate_candidate"].astype(float) - examples["sim_cash_rate_entered"].astype(float)
    return by_contest, by_bucket, examples


def build_replay_calibration_artifacts(
    *,
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> ReplayCalibrationBundle:
    data_root = data_root or get_data_root()
    analytics_dirs = discover_replay_analytics_dirs(data_root)
    player_df = _load_concat(analytics_dirs, "player_calibration.parquet")
    lineup_df = _load_concat(analytics_dirs, "lineup_calibration.parquet")
    field_df = _load_concat(analytics_dirs, "field_calibration.parquet")
    regret_df = _load_concat(analytics_dirs, "regret_summary.parquet")

    player_fpts = _player_bucket_frame(
        player_df,
        actual_col="actual_player_fpts",
        sim_mean_col="sim_mean_fpts",
        p10_col="sim_p10_fpts",
        p90_col="sim_p90_fpts",
        percentile_col="actual_fpts_sim_percentile",
    )
    player_minutes = _player_bucket_frame(
        player_df,
        actual_col="actual_minutes",
        sim_mean_col="sim_mean_minutes",
        p10_col="sim_p10_minutes",
        p90_col="sim_p90_minutes",
        percentile_col="actual_minutes_sim_percentile",
    )
    ownership = _ownership_recalibration_frame(player_df)
    field_model = _field_model_calibration_frame(field_df)
    regret_by_contest, regret_by_bucket, regret_examples = _optimizer_regret_frames(regret_df, lineup_df, field_df)

    out_dir = output_dir or calibration_output_dir(data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    player_fpts_path = out_dir / "player_fpts_calibration.parquet"
    player_minutes_path = out_dir / "player_minutes_calibration.parquet"
    ownership_path = out_dir / "ownership_recalibration.parquet"
    field_model_path = out_dir / "field_model_calibration.parquet"
    regret_contest_path = out_dir / "optimizer_regret_by_contest.parquet"
    regret_bucket_path = out_dir / "optimizer_regret_by_bucket.parquet"
    regret_examples_path = out_dir / "optimizer_regret_examples.parquet"
    summary_path = out_dir / "summary.json"

    player_fpts.to_parquet(player_fpts_path, index=False)
    player_minutes.to_parquet(player_minutes_path, index=False)
    ownership.to_parquet(ownership_path, index=False)
    field_model.to_parquet(field_model_path, index=False)
    regret_by_contest.to_parquet(regret_contest_path, index=False)
    regret_by_bucket.to_parquet(regret_bucket_path, index=False)
    regret_examples.to_parquet(regret_examples_path, index=False)

    summary = {
        "analytics_dir_count": len(analytics_dirs),
        "source_counts": {
            "player_rows": int(len(player_df)),
            "lineup_rows": int(len(lineup_df)),
            "field_rows": int(len(field_df)),
            "regret_rows": int(len(regret_df)),
        },
        "artifact_counts": {
            "player_fpts_rows": int(len(player_fpts)),
            "player_minutes_rows": int(len(player_minutes)),
            "ownership_rows": int(len(ownership)),
            "field_model_rows": int(len(field_model)),
            "regret_by_contest_rows": int(len(regret_by_contest)),
            "regret_by_bucket_rows": int(len(regret_by_bucket)),
            "regret_examples_rows": int(len(regret_examples)),
        },
        "artifacts": {
            "player_fpts_calibration_path": str(player_fpts_path),
            "player_minutes_calibration_path": str(player_minutes_path),
            "ownership_recalibration_path": str(ownership_path),
            "field_model_calibration_path": str(field_model_path),
            "optimizer_regret_by_contest_path": str(regret_contest_path),
            "optimizer_regret_by_bucket_path": str(regret_bucket_path),
            "optimizer_regret_examples_path": str(regret_examples_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    return ReplayCalibrationBundle(
        player_fpts_calibration_path=player_fpts_path,
        player_minutes_calibration_path=player_minutes_path,
        ownership_recalibration_path=ownership_path,
        field_model_calibration_path=field_model_path,
        optimizer_regret_by_contest_path=regret_contest_path,
        optimizer_regret_by_bucket_path=regret_bucket_path,
        optimizer_regret_examples_path=regret_examples_path,
        summary_path=summary_path,
    )
