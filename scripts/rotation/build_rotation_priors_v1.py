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

import pandas as pd

from projections import paths


@dataclass(frozen=True)
class BuildInputs:
    data_root: Path
    windows: list[int]
    overwrite: bool
    clean: bool


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
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def _prior_columns_player() -> list[str]:
    return [
        "minutes_from_stints",
        "num_stints",
        "started_proxy",
    ]


def _prior_columns_team() -> list[str]:
    return [
        "depth_6",
        "depth_10",
        "depth_14",
        "effective_n",
        "bench_conc_top1",
        "bench_conc_top2",
        "starter_pool_minutes",
        "bench_pool_minutes",
        "team_ot_flag",
    ]


def _compute_group_priors(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    date_col: str,
    windows: list[int],
    value_cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
    work = work.sort_values(
        [*group_cols, date_col, "game_id_norm"], kind="mergesort"
    ).reset_index(drop=True)

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

    # Player priors (group by person_id).
    player_work = _compute_group_priors(
        player,
        group_cols=["person_id"],
        date_col="game_date",
        windows=inputs.windows,
        value_cols=_prior_columns_player(),
        prefix="player",
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

    player_work["game_id"] = player_work["game_id_norm"]
    player_work = player_work[
        [
            "season",
            "game_id",
            "game_id_norm",
            "team_id",
            "person_id",
            "game_date",
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
        for col in [
            f"minutes_from_stints_prior_{window}_missing",
            f"num_stints_prior_{window}_missing",
            f"started_proxy_rate_prior_{window}_missing",
        ]:
            if col in player_work.columns:
                missing_summary[col] = float(
                    pd.to_numeric(player_work[col], errors="coerce").fillna(1).mean()
                )
        for col in [
            f"depth_6_prior_{window}_missing",
            f"depth_10_prior_{window}_missing",
            f"depth_14_prior_{window}_missing",
            f"effective_n_prior_{window}_missing",
            f"bench_conc_top1_prior_{window}_missing",
            f"bench_conc_top2_prior_{window}_missing",
            f"starter_pool_minutes_prior_{window}_missing",
            f"bench_pool_minutes_prior_{window}_missing",
            f"team_ot_rate_prior_{window}_missing",
        ]:
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
        },
        "outputs": {
            "root": str(root),
            "player_game_priors_root": str(player_root),
            "team_game_priors_root": str(team_root),
        },
        "input_coverage": {
            "player_labels_missing_game_date_rate": player_missing_game_date_rate,
            "team_shape_missing_game_date_rate": team_missing_game_date_rate,
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
