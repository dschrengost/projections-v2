#!/usr/bin/env python3
"""Build rotation_priors_v1 from rotation_v1 + schedule.

Outputs:
  <DATA_ROOT>/silver/rotation_priors_v1/
    - team_game_priors/season=YYYY/game_id=XXXXXXXXXX.parquet
    - player_game_priors/season=YYYY/game_id=XXXXXXXXXX.parquet
    - manifest.json

This is incremental by default: existing game partitions are skipped unless
--overwrite is provided.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from projections import paths
from projections.rotation.rotation_set_minutes_features_v1 import (
    SAME_POS_CONTEXT_BUCKETS,
    bucket_same_pos_depth,
    compute_same_pos_depth,
)


@dataclass(frozen=True)
class BuildInputs:
    data_root: Path
    windows: list[int]
    overwrite: bool
    clean: bool


PLAYER_PRIOR_BASE_COLS: tuple[str, ...] = (
    "minutes_from_stints",
    "num_stints",
    "started_proxy",
    # Timing/shape priors (converted to minutes to avoid unit drift).
    "first_in_minute",
    "last_out_minute",
    "max_stint_minutes",
    # Shooting / shot-mix priors.
    "fg2_pct",
    "fg3_pct",
    "ft_pct",
    "efg_pct",
    "fg2a_per_min",
    "fg3a_per_min",
    "fta_per_min",
    "three_pa_share",
)

TEAM_PRIOR_BASE_COLS: tuple[str, ...] = (
    "depth_6",
    "depth_10",
    "depth_14",
    "effective_n",
    "bench_conc_top1",
    "bench_conc_top2",
    "starter_pool_minutes",
    "bench_pool_minutes",
    "team_total_minutes_from_stints",
    "team_ot_flag",
    # Cadence/stability proxies.
    "bench_share",
    "starter_share",
    "depth_gap_10_6",
    # Defensive shot-quality allowance priors.
    "fg2_pct_allowed",
    "fg3_pct_allowed",
    "fta_rate_allowed",
    "efg_pct_allowed",
    "three_pa_share_allowed",
)

# Columns for rolling volatility priors (std across trailing games, shifted).
PLAYER_VOLATILITY_COLS: tuple[str, ...] = (
    "minutes_from_stints",
    "num_stints",
    "first_in_minute",
    "last_out_minute",
    "max_stint_minutes",
)

TEAM_VOLATILITY_COLS: tuple[str, ...] = (
    "depth_6",
    "depth_10",
    "depth_14",
    "effective_n",
    "bench_conc_top1",
    "bench_conc_top2",
    "starter_pool_minutes",
    "bench_pool_minutes",
    "team_total_minutes_from_stints",
    "bench_share",
    "starter_share",
    "depth_gap_10_6",
)

PLAYER_CONTEXT_BASE_COLS: tuple[str, ...] = (
    "minutes_from_stints",
    "started_proxy",
)

PLAYER_SHOOTING_BASE_COLS: tuple[str, ...] = (
    "fg2_pct",
    "fg3_pct",
    "ft_pct",
    "efg_pct",
    "fg2a_per_min",
    "fg3a_per_min",
    "fta_per_min",
    "three_pa_share",
)

TEAM_DEFENSE_ALLOWED_BASE_COLS: tuple[str, ...] = (
    "fg2_pct_allowed",
    "fg3_pct_allowed",
    "fta_rate_allowed",
    "efg_pct_allowed",
    "three_pa_share_allowed",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _season_for_date(day: pd.Timestamp) -> int:
    return int(day.year) if int(day.month) >= 8 else int(day.year) - 1


def _zfill_game_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    # Schedule game_id is 8 digits; rotation data uses 10-digit zfilled string.
    try:
        return str(int(float(text))).zfill(10)
    except Exception:
        return text.zfill(10)


def _load_schedule_map(data_root: Path) -> pd.DataFrame:
    schedule_root = data_root / "silver" / "schedule"
    frames: list[pd.DataFrame] = []
    for path in sorted(schedule_root.glob("season=*/month=*/schedule.parquet")):
        try:
            df = pd.read_parquet(path, columns=["game_id", "game_date"])
        except Exception:
            continue
        if df.empty:
            continue
        df = df.copy()
        df["game_id_norm"] = df["game_id"].map(_zfill_game_id)
        df["game_date"] = pd.to_datetime(
            df["game_date"], errors="coerce"
        ).dt.normalize()
        frames.append(df.loc[:, ["game_id_norm", "game_date"]])
    if not frames:
        return pd.DataFrame(columns=["game_id_norm", "game_date"])
    out = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["game_id_norm"], keep="last"
    )
    return out


def _load_rotation_player_labels(data_root: Path) -> pd.DataFrame:
    root = data_root / "silver" / "rotation_v1" / "player_game_labels"
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("season=*/game_id=*.parquet")):
        try:
            df = pd.read_parquet(
                path,
                columns=[
                    "season",
                    "game_id",
                    "team_id",
                    "person_id",
                    "minutes_from_stints",
                    "num_stints",
                    "started_proxy",
                    "first_in_time_real",
                    "last_out_time_real",
                    "max_stint_len_real",
                    "seconds_per_unit",
                ],
            )
        except Exception:
            continue
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["game_id_norm"] = out["game_id"].map(_zfill_game_id)
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").astype("Int64")
    out["person_id"] = pd.to_numeric(out["person_id"], errors="coerce").astype("Int64")
    out["minutes_from_stints"] = (
        pd.to_numeric(out["minutes_from_stints"], errors="coerce")
        .fillna(0.0)
        .astype("float64")
    )
    out["num_stints"] = (
        pd.to_numeric(out["num_stints"], errors="coerce").fillna(0).astype("float64")
    )
    out["started_proxy"] = out["started_proxy"].fillna(False).astype("int8")
    out["first_in_time_real"] = pd.to_numeric(
        out.get("first_in_time_real"), errors="coerce"
    ).fillna(0.0).astype("float64")
    out["last_out_time_real"] = pd.to_numeric(
        out.get("last_out_time_real"), errors="coerce"
    ).fillna(0.0).astype("float64")
    out["max_stint_len_real"] = pd.to_numeric(
        out.get("max_stint_len_real"), errors="coerce"
    ).fillna(0.0).astype("float64")
    out["seconds_per_unit"] = pd.to_numeric(
        out.get("seconds_per_unit"), errors="coerce"
    ).replace(0.0, pd.NA).fillna(1.0).astype("float64")
    out["first_in_minute"] = (
        (out["first_in_time_real"] * out["seconds_per_unit"]) / 60.0
    ).clip(lower=0.0)
    out["last_out_minute"] = (
        (out["last_out_time_real"] * out["seconds_per_unit"]) / 60.0
    ).clip(lower=0.0)
    out["max_stint_minutes"] = (
        (out["max_stint_len_real"] * out["seconds_per_unit"]) / 60.0
    ).clip(lower=0.0)
    return out


def _load_rotation_team_shape(data_root: Path) -> pd.DataFrame:
    root = data_root / "silver" / "rotation_v1" / "team_game_shape"
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("season=*/game_id=*.parquet")):
        try:
            df = pd.read_parquet(
                path,
                columns=[
                    "season",
                    "game_id",
                    "team_id",
                    "game_duration_minutes",
                    "depth_6",
                    "depth_10",
                    "depth_14",
                    "effective_n",
                    "bench_conc_top1",
                    "bench_conc_top2",
                    "starter_pool_minutes",
                    "bench_pool_minutes",
                    "team_total_minutes_from_stints",
                ],
            )
        except Exception:
            continue
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["game_id_norm"] = out["game_id"].map(_zfill_game_id)
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").astype("Int64")
    out["game_duration_minutes"] = pd.to_numeric(
        out["game_duration_minutes"], errors="coerce"
    ).astype("float64")
    for col in [
        "depth_6",
        "depth_10",
        "depth_14",
        "effective_n",
        "bench_conc_top1",
        "bench_conc_top2",
        "starter_pool_minutes",
        "bench_pool_minutes",
        "team_total_minutes_from_stints",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    denom = out["team_total_minutes_from_stints"].replace(0.0, pd.NA)
    out["bench_share"] = (out["bench_pool_minutes"] / denom).fillna(0.0).astype("float64")
    out["starter_share"] = (out["starter_pool_minutes"] / denom).fillna(0.0).astype("float64")
    out["depth_gap_10_6"] = (out["depth_10"] - out["depth_6"]).fillna(0.0).astype("float64")
    return out


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = pd.to_numeric(denominator, errors="coerce").astype("float64")
    denom = denom.where(denom.ne(0.0), np.nan)
    num = pd.to_numeric(numerator, errors="coerce")
    return (num / denom).fillna(0.0).astype("float64")


def _load_boxscore_counts(data_root: Path) -> pd.DataFrame:
    counts_path = data_root / "gold" / "labels_boxscore_counts" / "labels_boxscore_counts.parquet"
    if not counts_path.exists():
        return pd.DataFrame()

    try:
        counts = pd.read_parquet(
            counts_path,
            columns=[
                "game_id",
                "team_id",
                "player_id",
                "game_date",
                "season",
                "fga2",
                "fg2m",
                "fga3",
                "fg3m",
                "fta",
                "ftm",
                "minutes",
            ],
        )
    except Exception:
        return pd.DataFrame()

    if counts.empty:
        return counts

    counts = counts.copy()
    counts["game_id_norm"] = counts["game_id"].map(_zfill_game_id)
    counts["team_id"] = pd.to_numeric(counts["team_id"], errors="coerce").astype("Int64")
    counts["person_id"] = pd.to_numeric(counts["player_id"], errors="coerce").astype("Int64")
    counts["game_date"] = pd.to_datetime(counts["game_date"], errors="coerce").dt.normalize()
    for col in ("fga2", "fg2m", "fga3", "fg3m", "fta", "ftm", "minutes"):
        counts[col] = pd.to_numeric(counts[col], errors="coerce").fillna(0.0).astype("float64")

    fga = counts["fga2"] + counts["fga3"]
    fgm = counts["fg2m"] + counts["fg3m"]
    counts["fg2_pct"] = _safe_rate(counts["fg2m"], counts["fga2"])
    counts["fg3_pct"] = _safe_rate(counts["fg3m"], counts["fga3"])
    counts["ft_pct"] = _safe_rate(counts["ftm"], counts["fta"])
    counts["efg_pct"] = _safe_rate(counts["fg2m"] + 1.5 * counts["fg3m"], fga)
    counts["fg2a_per_min"] = _safe_rate(counts["fga2"], counts["minutes"])
    counts["fg3a_per_min"] = _safe_rate(counts["fga3"], counts["minutes"])
    counts["fta_per_min"] = _safe_rate(counts["fta"], counts["minutes"])
    counts["three_pa_share"] = _safe_rate(counts["fga3"], fga)
    counts["fgm_total"] = fgm.astype("float64")
    counts["fga_total"] = fga.astype("float64")
    return counts


def _build_team_allowed_boxscore_features(counts: pd.DataFrame) -> pd.DataFrame:
    if counts.empty:
        return pd.DataFrame()

    team_totals = (
        counts.groupby(["game_id_norm", "team_id"], sort=False, as_index=False)
        .agg(
            {
                "game_date": "last",
                "season": "last",
                "fga2": "sum",
                "fg2m": "sum",
                "fga3": "sum",
                "fg3m": "sum",
                "fta": "sum",
                "ftm": "sum",
                "fgm_total": "sum",
                "fga_total": "sum",
            }
        )
        .copy()
    )
    if team_totals.empty:
        return team_totals

    opp = team_totals.rename(
        columns={
            "team_id": "opponent_team_id",
            "fga2": "opp_fga2",
            "fg2m": "opp_fg2m",
            "fga3": "opp_fga3",
            "fg3m": "opp_fg3m",
            "fta": "opp_fta",
            "ftm": "opp_ftm",
            "fgm_total": "opp_fgm_total",
            "fga_total": "opp_fga_total",
        }
    )
    paired = team_totals.merge(opp, on="game_id_norm", how="inner", sort=False)
    paired = paired.loc[paired["team_id"] != paired["opponent_team_id"]].copy()
    if paired.empty:
        return pd.DataFrame(columns=["game_id_norm", "team_id", *TEAM_DEFENSE_ALLOWED_BASE_COLS])

    allowed = paired.loc[:, ["game_id_norm", "team_id"]].copy()
    allowed["fg2_pct_allowed"] = _safe_rate(paired["opp_fg2m"], paired["opp_fga2"])
    allowed["fg3_pct_allowed"] = _safe_rate(paired["opp_fg3m"], paired["opp_fga3"])
    allowed["fta_rate_allowed"] = _safe_rate(paired["opp_fta"], paired["opp_fga_total"])
    allowed["efg_pct_allowed"] = _safe_rate(
        paired["opp_fg2m"] + 1.5 * paired["opp_fg3m"],
        paired["opp_fga_total"],
    )
    allowed["three_pa_share_allowed"] = _safe_rate(paired["opp_fga3"], paired["opp_fga_total"])
    allowed = allowed.drop_duplicates(subset=["game_id_norm", "team_id"], keep="last")
    return allowed


def _load_player_context_features(data_root: Path) -> pd.DataFrame:
    """Load historical pre-tip context snapshots used for context priors."""

    root = data_root / "gold" / "features_minutes_v1"
    frames: list[pd.DataFrame] = []
    desired_cols = [
        "game_id",
        "game_date",
        "team_id",
        "player_id",
        "pos_bucket",
        "available_G",
        "available_W",
        "available_B",
        "depth_same_pos_active",
    ]

    for path in sorted(root.glob("season=*/month=*/features.parquet")):
        try:
            df = pd.read_parquet(path, columns=desired_cols)
        except Exception:
            continue
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["game_id_norm"] = out["game_id"].map(_zfill_game_id)
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").astype("Int64")
    out["person_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()
    out["ctx_same_pos_bucket"] = bucket_same_pos_depth(compute_same_pos_depth(out))
    out = out.dropna(subset=["game_id_norm", "team_id", "person_id", "game_date"]).copy()
    out = out.drop_duplicates(
        subset=["game_id_norm", "team_id", "person_id"],
        keep="last",
    )
    return out.loc[:, ["game_id_norm", "team_id", "person_id", "game_date", "ctx_same_pos_bucket"]]


def _prior_columns_player() -> list[str]:
    return list(PLAYER_PRIOR_BASE_COLS)


def _prior_columns_team() -> list[str]:
    return list(TEAM_PRIOR_BASE_COLS)


def _compute_group_priors(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    date_col: str,
    windows: list[int],
    value_cols: list[str],
    prefix: str,
    std_value_cols: list[str] | None = None,
) -> pd.DataFrame:
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
    work = work.sort_values(
        [*group_cols, date_col, "game_id_norm"], kind="mergesort"
    ).reset_index(drop=True)
    std_cols = list(std_value_cols or [])

    for window in windows:
        n_col = f"{prefix}_prior_n_games_{window}"
        max_col = f"{prefix}_prior_source_max_game_date_{window}"

        def _roll_count(s: pd.Series) -> pd.Series:
            shifted = s.shift(1)
            return shifted.rolling(window, min_periods=1).count()

        work[n_col] = (
            work.groupby(group_cols, sort=False)[date_col]
            .apply(_roll_count)
            .reset_index(level=group_cols, drop=True)
            .fillna(0)
            .astype("int16")
        )

        # pandas doesn't implement rolling max/min for datetime64 directly.
        # Convert to int64 nanoseconds, roll, then convert back to datetime64.
        def _roll_max_ns(s: pd.Series) -> pd.Series:
            shifted = s.shift(1)
            shifted_ns = shifted.astype("int64", copy=False)
            return shifted_ns.rolling(window, min_periods=1).max()

        max_ns = (
            work.groupby(group_cols, sort=False)[date_col]
            .apply(_roll_max_ns)
            .reset_index(level=group_cols, drop=True)
        )
        work[max_col] = pd.to_datetime(max_ns, errors="coerce")
        work.loc[work[n_col] == 0, max_col] = pd.NaT

        for base_col in value_cols:
            out_col = f"{base_col}_prior_{window}"
            miss_col = f"{out_col}_missing"

            def _roll_mean(s: pd.Series) -> pd.Series:
                shifted = s.shift(1)
                return shifted.rolling(window, min_periods=1).mean()

            work[out_col] = (
                work.groupby(group_cols, sort=False)[base_col]
                .apply(_roll_mean)
                .reset_index(level=group_cols, drop=True)
                .fillna(0.0)
                .astype("float64")
            )
            work[miss_col] = (work[n_col] == 0).astype("int8")

        for base_col in std_cols:
            out_col = f"{base_col}_std_prior_{window}"
            miss_col = f"{out_col}_missing"

            def _roll_std(s: pd.Series) -> pd.Series:
                shifted = s.shift(1)
                return shifted.rolling(window, min_periods=1).std(ddof=0)

            work[out_col] = (
                work.groupby(group_cols, sort=False)[base_col]
                .apply(_roll_std)
                .reset_index(level=group_cols, drop=True)
                .fillna(0.0)
                .astype("float64")
            )
            work[miss_col] = (work[n_col] == 0).astype("int8")

    return work


def _compute_context_bucket_priors(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    date_col: str,
    windows: list[int],
    value_cols: list[str],
    bucket_col: str,
    bucket_values: tuple[str, ...],
) -> pd.DataFrame:
    """Compute rolling priors within coarse context buckets over trailing games."""

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
    work = work.sort_values(
        [*group_cols, date_col, "game_id_norm"], kind="mergesort"
    ).reset_index(drop=True)
    bucket_series = work[bucket_col].astype("string").fillna("unknown").str.lower().str.strip()
    group_keys = [work[col] for col in group_cols]

    for window in windows:
        for bucket in bucket_values:
            bucket_mask = bucket_series.eq(bucket)
            count_col = f"ctx_same_pos_{bucket}_prior_n_games_{window}"
            max_col = f"ctx_same_pos_{bucket}_prior_source_max_game_date_{window}"

            def _roll_count(s: pd.Series) -> pd.Series:
                shifted = s.shift(1)
                return shifted.rolling(window, min_periods=1).count()

            masked_dates = work[date_col].where(bucket_mask)
            work[count_col] = (
                masked_dates.groupby(group_keys, sort=False)
                .apply(_roll_count)
                .reset_index(level=group_cols, drop=True)
                .fillna(0)
                .astype("int16")
            )

            def _roll_max_ns(s: pd.Series) -> pd.Series:
                shifted = s.shift(1)
                shifted_ns = shifted.astype("int64", copy=False)
                return shifted_ns.rolling(window, min_periods=1).max()

            max_ns = (
                masked_dates.groupby(group_keys, sort=False)
                .apply(_roll_max_ns)
                .reset_index(level=group_cols, drop=True)
            )
            work[max_col] = pd.to_datetime(max_ns, errors="coerce")
            work.loc[work[count_col] == 0, max_col] = pd.NaT

            for base_col in value_cols:
                out_col = f"{base_col}_ctx_same_pos_{bucket}_prior_{window}"

                def _roll_mean(s: pd.Series) -> pd.Series:
                    shifted = s.shift(1)
                    return shifted.rolling(window, min_periods=1).mean()

                masked_vals = work[base_col].where(bucket_mask)
                work[out_col] = (
                    masked_vals.groupby(group_keys, sort=False)
                    .apply(_roll_mean)
                    .reset_index(level=group_cols, drop=True)
                    .fillna(0.0)
                    .astype("float64")
                )

    return work


def _write_partitioned(
    df: pd.DataFrame,
    *,
    root: Path,
    season_col: str = "season",
    game_id_col: str = "game_id_norm",
    overwrite: bool,
) -> tuple[int, int]:
    written = 0
    skipped = 0
    for (season, game_id), part in df.groupby([season_col, game_id_col], sort=False):
        season_int = int(season)
        game_id_str = str(game_id)
        out_path = root / f"season={season_int}" / f"game_id={game_id_str}.parquet"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(out_path, index=False)
        written += 1
    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build rotation_priors_v1 from rotation_v1 + schedule.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root", type=str, default=None, help="Override PROJECTIONS_DATA_ROOT."
    )
    parser.add_argument(
        "--window",
        type=int,
        action="append",
        default=None,
        help="Rolling window size (repeatable).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing priors partitions."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing rotation_priors_v1 outputs first.",
    )

    args = parser.parse_args()
    data_root = (
        Path(args.data_root).expanduser().resolve()
        if args.data_root
        else paths.get_data_root()
    )
    windows = sorted({int(w) for w in (args.window or [5, 10, 20])})
    inputs = BuildInputs(
        data_root=data_root,
        windows=windows,
        overwrite=bool(args.overwrite),
        clean=bool(args.clean),
    )

    root = inputs.data_root / "silver" / "rotation_priors_v1"
    team_root = root / "team_game_priors"
    player_root = root / "player_game_priors"

    if inputs.clean and root.exists():
        shutil.rmtree(root)

    schedule_map = _load_schedule_map(inputs.data_root)
    if schedule_map.empty:
        raise RuntimeError(
            f"[rotation_priors_v1] schedule map empty at {inputs.data_root}/silver/schedule"
        )

    player = _load_rotation_player_labels(inputs.data_root)
    team = _load_rotation_team_shape(inputs.data_root)
    player_context = _load_player_context_features(inputs.data_root)
    boxscore_counts = _load_boxscore_counts(inputs.data_root)
    if player.empty or team.empty:
        raise RuntimeError(
            "[rotation_priors_v1] rotation_v1 inputs missing; run build_rotation_dataset_v1 first"
        )

    player = player.merge(schedule_map, on="game_id_norm", how="left", sort=False)
    team = team.merge(schedule_map, on="game_id_norm", how="left", sort=False)

    player["game_date"] = pd.to_datetime(
        player["game_date"], errors="coerce"
    ).dt.normalize()
    team["game_date"] = pd.to_datetime(
        team["game_date"], errors="coerce"
    ).dt.normalize()

    player_missing_game_date_rate = (
        float(player["game_date"].isna().mean()) if len(player) else 0.0
    )
    team_missing_game_date_rate = (
        float(team["game_date"].isna().mean()) if len(team) else 0.0
    )

    # Derive season label from date for output partitioning (Aug–Jul boundary).
    player["season"] = player["game_date"].map(
        lambda d: _season_for_date(pd.Timestamp(d)) if pd.notna(d) else pd.NA
    )
    team["season"] = team["game_date"].map(
        lambda d: _season_for_date(pd.Timestamp(d)) if pd.notna(d) else pd.NA
    )

    if not boxscore_counts.empty:
        player_boxscore_cols = ["game_id_norm", "team_id", "person_id", *PLAYER_SHOOTING_BASE_COLS]
        player = player.merge(
            boxscore_counts.loc[:, [c for c in player_boxscore_cols if c in boxscore_counts.columns]].drop_duplicates(
                subset=["game_id_norm", "team_id", "person_id"],
                keep="last",
            ),
            on=["game_id_norm", "team_id", "person_id"],
            how="left",
            sort=False,
        )
        team_allowed = _build_team_allowed_boxscore_features(boxscore_counts)
        if not team_allowed.empty:
            team = team.merge(
                team_allowed,
                on=["game_id_norm", "team_id"],
                how="left",
                sort=False,
            )

    for col in PLAYER_SHOOTING_BASE_COLS:
        if col not in player.columns:
            player[col] = 0.0
        player[col] = pd.to_numeric(player[col], errors="coerce").fillna(0.0).astype("float64")

    for col in TEAM_DEFENSE_ALLOWED_BASE_COLS:
        if col not in team.columns:
            team[col] = 0.0
        team[col] = pd.to_numeric(team[col], errors="coerce").fillna(0.0).astype("float64")

    if not player_context.empty:
        player = player.merge(
            player_context.drop(columns=["game_date"], errors="ignore"),
            on=["game_id_norm", "team_id", "person_id"],
            how="left",
            sort=False,
        )
    player["ctx_same_pos_bucket"] = (
        player.get("ctx_same_pos_bucket", pd.Series(pd.NA, index=player.index))
        .astype("string")
        .fillna("unknown")
        .str.lower()
        .str.strip()
    )
    player_context_match_rate = float(player["ctx_same_pos_bucket"].ne("unknown").mean()) if len(player) else 0.0

    # Player priors (group by person_id).
    player_work = _compute_group_priors(
        player,
        group_cols=["person_id"],
        date_col="game_date",
        windows=inputs.windows,
        value_cols=_prior_columns_player(),
        prefix="player",
        std_value_cols=list(PLAYER_VOLATILITY_COLS),
    )

    player_ctx_work = _compute_context_bucket_priors(
        player,
        group_cols=["person_id"],
        date_col="game_date",
        windows=inputs.windows,
        value_cols=list(PLAYER_CONTEXT_BASE_COLS),
        bucket_col="ctx_same_pos_bucket",
        bucket_values=SAME_POS_CONTEXT_BUCKETS,
    )
    player_ctx_cols = [
        "game_id_norm",
        "team_id",
        "person_id",
        *[
            c
            for c in player_ctx_work.columns
            if "_ctx_same_pos_" in c and "_prior_" in c
        ],
        *[
            c
            for c in player_ctx_work.columns
            if c.startswith("ctx_same_pos_") and "_prior_n_games_" in c
        ],
        *[
            c
            for c in player_ctx_work.columns
            if c.startswith("ctx_same_pos_") and "_prior_source_max_game_date_" in c
        ],
    ]
    player_ctx_cols = [c for i, c in enumerate(player_ctx_cols) if c in player_ctx_work.columns and c not in player_ctx_cols[:i]]
    player_work = player_work.merge(
        player_ctx_work.loc[:, player_ctx_cols],
        on=["game_id_norm", "team_id", "person_id"],
        how="left",
        sort=False,
    )

    # Convert started_proxy rolling mean to a "rate" name to match downstream expectations.
    for window in inputs.windows:
        src = f"started_proxy_prior_{window}"
        dst = f"started_proxy_rate_prior_{window}"
        miss = f"{dst}_missing"
        if src in player_work.columns:
            player_work[dst] = player_work[src]
            player_work = player_work.drop(columns=[src])
        if miss not in player_work.columns and f"{src}_missing" in player_work.columns:
            player_work[miss] = player_work[f"{src}_missing"]
            player_work = player_work.drop(columns=[f"{src}_missing"])
        for bucket in SAME_POS_CONTEXT_BUCKETS:
            ctx_src = f"started_proxy_ctx_same_pos_{bucket}_prior_{window}"
            ctx_dst = f"started_proxy_rate_ctx_same_pos_{bucket}_prior_{window}"
            if ctx_src in player_work.columns:
                player_work[ctx_dst] = player_work[ctx_src]
                player_work = player_work.drop(columns=[ctx_src])

    player_work["game_id"] = player_work["game_id_norm"]
    player_work = player_work[
        [
            "season",
            "game_id",
            "game_id_norm",
            "team_id",
            "person_id",
            "game_date",
            "ctx_same_pos_bucket",
            "minutes_from_stints",
            "started_proxy",
            *[c for c in PLAYER_SHOOTING_BASE_COLS if c in player_work.columns],
            *[c for c in player_work.columns if c.startswith("player_prior_")],
            *[
                c
                for c in player_work.columns
                if "_prior_" in c and not c.startswith("player_prior_")
            ],
        ]
    ]

    # Team priors (group by team_id).
    team_work = team.copy()
    team_work["team_ot_flag"] = (team_work["game_duration_minutes"] > 48.0).astype(
        "int8"
    )
    team_work = _compute_group_priors(
        team_work,
        group_cols=["team_id"],
        date_col="game_date",
        windows=inputs.windows,
        value_cols=_prior_columns_team(),
        prefix="team",
        std_value_cols=list(TEAM_VOLATILITY_COLS),
    )

    for window in inputs.windows:
        src = f"team_ot_flag_prior_{window}"
        dst = f"team_ot_rate_prior_{window}"
        miss = f"{dst}_missing"
        if src in team_work.columns:
            team_work[dst] = team_work[src]
            team_work = team_work.drop(columns=[src])
        if miss not in team_work.columns and f"{src}_missing" in team_work.columns:
            team_work[miss] = team_work[f"{src}_missing"]
            team_work = team_work.drop(columns=[f"{src}_missing"])

    team_work["game_id"] = team_work["game_id_norm"]
    team_work = team_work[
        [
            "season",
            "game_id",
            "game_id_norm",
            "team_id",
            "game_date",
            *[c for c in TEAM_PRIOR_BASE_COLS if c in team_work.columns],
            *[c for c in team_work.columns if c.startswith("team_prior_")],
            *[
                c
                for c in team_work.columns
                if "_prior_" in c and not c.startswith("team_prior_")
            ],
        ]
    ]

    # Drop any rows without a resolved season/game_date.
    player_work = player_work.dropna(subset=["season", "game_date"])
    team_work = team_work.dropna(subset=["season", "game_date"])
    player_work["season"] = pd.to_numeric(
        player_work["season"], errors="coerce"
    ).astype("int64")
    team_work["season"] = pd.to_numeric(team_work["season"], errors="coerce").astype(
        "int64"
    )

    player_written, player_skipped = _write_partitioned(
        player_work,
        root=player_root,
        overwrite=inputs.overwrite,
    )
    team_written, team_skipped = _write_partitioned(
        team_work,
        root=team_root,
        overwrite=inputs.overwrite,
    )

    # Missing summary for debugging/health.
    missing_summary: dict[str, float] = {}
    for window in inputs.windows:
        for col in sorted(
            c
            for c in player_work.columns
            if c.endswith(f"_prior_{window}_missing")
        ):
            if col in player_work.columns:
                missing_summary[col] = float(
                    pd.to_numeric(player_work[col], errors="coerce").fillna(1).mean()
                )
        for col in sorted(
            c
            for c in team_work.columns
            if c.endswith(f"_prior_{window}_missing")
        ):
            if col in team_work.columns:
                missing_summary[col] = float(
                    pd.to_numeric(team_work[col], errors="coerce").fillna(1).mean()
                )

    manifest = {
        "created_at": _utc_now_iso(),
        "counts": {
            "player_games": int(len(player_work)),
            "team_games": int(len(team_work)),
            "unique_games_player": int(player_work["game_id_norm"].nunique()),
            "unique_games_team": int(team_work["game_id_norm"].nunique()),
        },
        "inputs": {
            "rotation_v1_player_game_labels_root": str(
                inputs.data_root / "silver" / "rotation_v1" / "player_game_labels"
            ),
            "rotation_v1_team_game_shape_root": str(
                inputs.data_root / "silver" / "rotation_v1" / "team_game_shape"
            ),
            "schedule_root": str(inputs.data_root / "silver" / "schedule"),
            "player_context_root": str(inputs.data_root / "gold" / "features_minutes_v1"),
            "boxscore_counts_path": str(
                inputs.data_root / "gold" / "labels_boxscore_counts" / "labels_boxscore_counts.parquet"
            ),
        },
        "outputs": {
            "root": str(root),
            "player_game_priors_root": str(player_root),
            "team_game_priors_root": str(team_root),
        },
        "input_coverage": {
            "player_labels_missing_game_date_rate": player_missing_game_date_rate,
            "team_shape_missing_game_date_rate": team_missing_game_date_rate,
            "player_context_match_rate": player_context_match_rate,
            "boxscore_counts_available": bool(not boxscore_counts.empty),
        },
        "write_summary": {
            "player_partitions_written": player_written,
            "player_partitions_skipped": player_skipped,
            "team_partitions_written": team_written,
            "team_partitions_skipped": team_skipped,
            "overwrite": inputs.overwrite,
            "clean": inputs.clean,
            "windows": inputs.windows,
        },
        "missing_rate_summary": missing_summary,
    }

    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("[rotation_priors_v1] wrote manifest:", root / "manifest.json")


if __name__ == "__main__":
    main()
