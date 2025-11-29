"""
Utilities for building tracking-based role features.

Flow:
- Read player tracking tables from bronze/nba/tracking per game_date.
- Derive per-game tracking rates.
- Compute season-to-date pre-game aggregates.
- Cluster players into coarse roles using season-to-date tracking rates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

MIN_ROLE_MINUTES = 200.0


def _season_from_date(day: pd.Timestamp) -> int:
    return day.year if day.month >= 8 else day.year - 1


def _season_label(season: int) -> str:
    return f"{season}-{str(season + 1)[-2:]}"


def _iter_days(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    cur = start
    while cur <= end:
        yield cur
        cur += pd.Timedelta(days=1)


def _glob_measure_paths(root: Path, season_label: str, day: pd.Timestamp, measure_type: str) -> list[Path]:
    token = day.date().isoformat()
    base = root / "bronze" / "nba" / "tracking" / f"season={season_label}"
    pattern = f"season_type=*/game_date={token}/pt_measure_type={measure_type}/*.parquet"
    return sorted(base.glob(pattern))


def _read_measure(root: Path, day: pd.Timestamp, measure_type: str, columns: list[str]) -> pd.DataFrame | None:
    season_label = _season_label(_season_from_date(day))
    files = _glob_measure_paths(root, season_label, day, measure_type)
    if not files:
        return None
    frames = []
    for path in files:
        try:
            df = pd.read_parquet(path)
            desired_cols = []
            for col in columns + ["game_date", "team_id", "player_id", "season"]:
                if col in df.columns and col not in desired_cols:
                    desired_cols.append(col)
            df = df[desired_cols]
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["season"] = df["game_date"].apply(_season_from_date)
    keep = [c for c in df.columns if c in set(columns + ["game_date", "team_id", "player_id", "season"])]
    return df[keep]


def load_tracking_window(data_root: Path, start: pd.Timestamp, end: pd.Timestamp, game_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Read tracking tables (Possessions + Passing + Drives) for the date window and return
    per-game raw counts merged with game ids where available.
    """
    records: list[pd.DataFrame] = []
    for day in _iter_days(start, end):
        poss = _read_measure(
            data_root,
            day,
            "Possessions",
            ["game_date", "team_id", "player_id", "min", "touches", "time_of_poss"],
        )
        if poss is None or poss.empty:
            continue
        passing = _read_measure(
            data_root,
            day,
            "Passing",
            ["game_date", "team_id", "player_id", "potential_ast", "passes_made"],
        )
        drives = _read_measure(
            data_root,
            day,
            "Drives",
            ["game_date", "team_id", "player_id", "drives"],
        )
        base = poss.rename(columns={"min": "minutes_tracking"}).copy()
        for col in ("touches", "time_of_poss", "minutes_tracking"):
            if col in base.columns:
                base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0.0)
        if passing is not None:
            passing = passing.rename(columns={"potential_ast": "potential_ast_raw"})
            base = base.merge(
                passing[["season", "game_date", "team_id", "player_id"] + [c for c in ["potential_ast_raw", "passes_made"] if c in passing.columns]],
                on=["season", "game_date", "team_id", "player_id"],
                how="left",
            )
        if drives is not None:
            base = base.merge(
                drives[["season", "game_date", "team_id", "player_id", "drives"]],
                on=["season", "game_date", "team_id", "player_id"],
                how="left",
            )
        base["potential_ast_raw"] = pd.to_numeric(base.get("potential_ast_raw"), errors="coerce").fillna(0.0)
        base["passes_made"] = pd.to_numeric(base.get("passes_made"), errors="coerce").fillna(0.0)
        base["drives"] = pd.to_numeric(base.get("drives"), errors="coerce").fillna(0.0)
        records.append(base)
    if not records:
        return pd.DataFrame()
    df = pd.concat(records, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    game_lookup = game_lookup.copy()
    game_lookup["game_date"] = pd.to_datetime(game_lookup["game_date"]).dt.normalize()
    merged = df.merge(
        game_lookup[["season", "game_date", "team_id", "player_id", "game_id"]],
        on=["season", "game_date", "team_id", "player_id"],
        how="left",
    )
    return merged


def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    denom_safe = denom.replace({0: np.nan})
    return num / denom_safe


def compute_cumulative_tracking(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["minutes_tracking"] = pd.to_numeric(df["minutes_tracking"], errors="coerce").fillna(0.0)
    df["touches"] = pd.to_numeric(df["touches"], errors="coerce").fillna(0.0)
    df["time_of_poss"] = pd.to_numeric(df["time_of_poss"], errors="coerce").fillna(0.0)
    df["potential_ast_raw"] = pd.to_numeric(df.get("potential_ast_raw"), errors="coerce").fillna(0.0)
    df["passes_made"] = pd.to_numeric(df.get("passes_made"), errors="coerce").fillna(0.0)
    df["drives"] = pd.to_numeric(df.get("drives"), errors="coerce").fillna(0.0)
    df.sort_values(["season", "player_id", "game_date", "game_id"], inplace=True)
    group = df.groupby(["season", "player_id"], sort=False)
    df["cumu_minutes"] = group["minutes_tracking"].cumsum().shift(1).fillna(0.0)
    df["cumu_touches"] = group["touches"].cumsum().shift(1).fillna(0.0)
    df["cumu_time_of_poss"] = group["time_of_poss"].cumsum().shift(1).fillna(0.0)
    df["cumu_potential_ast"] = group["potential_ast_raw"].cumsum().shift(1).fillna(0.0)
    df["cumu_passes_made"] = group["passes_made"].cumsum().shift(1).fillna(0.0)
    df["cumu_drives"] = group["drives"].cumsum().shift(1).fillna(0.0)

    df["track_touches_per_min_szn"] = _safe_div(df["cumu_touches"], df["cumu_minutes"])
    df["track_sec_per_touch_szn"] = _safe_div(df["cumu_time_of_poss"] * 60.0, df["cumu_touches"])
    pot_source = df["cumu_potential_ast"]
    fallback = df["cumu_passes_made"]
    pot_total = np.where(pot_source.notna(), pot_source, fallback)
    df["track_pot_ast_per_min_szn"] = _safe_div(pot_total, df["cumu_minutes"])
    df["track_drives_per_min_szn"] = _safe_div(df["cumu_drives"], df["cumu_minutes"])
    df["track_role_is_low_minutes"] = df["cumu_minutes"] < MIN_ROLE_MINUTES
    return df


def assign_tracking_roles(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    if df.empty:
        return df.assign(track_role_cluster=pd.Series(dtype=int))
    df = df.copy()
    feature_cols = [
        "track_touches_per_min_szn",
        "track_sec_per_touch_szn",
        "track_pot_ast_per_min_szn",
        "track_drives_per_min_szn",
    ]
    latest = (
        df.sort_values(["season", "player_id", "game_date", "game_id"])
        .groupby(["season", "player_id"], sort=False)
        .tail(1)
    )
    role_map = []
    for season, frame in latest.groupby("season"):
        eligible = frame[
            (~frame["track_role_is_low_minutes"])
            & frame[feature_cols].notna().all(axis=1)
        ].copy()
        if eligible.empty:
            continue
        scaler = StandardScaler()
        X = scaler.fit_transform(eligible[feature_cols])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        eligible["track_role_cluster"] = labels
        role_map.append(eligible[["season", "player_id", "track_role_cluster"]])
    if role_map:
        role_df = pd.concat(role_map, ignore_index=True)
    else:
        role_df = pd.DataFrame(columns=["season", "player_id", "track_role_cluster"])
    df = df.merge(role_df, on=["season", "player_id"], how="left")
    df["track_role_cluster"] = df["track_role_cluster"].fillna(-1).astype(int)
    return df


def load_game_lookup(data_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in _iter_days(start, end):
        season = _season_from_date(day)
        path = (
            data_root
            / "gold"
            / "labels_minutes_v1"
            / f"season={season}"
            / f"game_date={day.date().isoformat()}"
            / "labels.parquet"
        )
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=["game_id", "team_id", "player_id", "game_date", "season"])
        except Exception:
            continue
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["game_id", "team_id", "player_id", "game_date", "season"])
    combined = pd.concat(frames, ignore_index=True)
    combined["team_id"] = pd.to_numeric(combined["team_id"], errors="coerce").astype("Int64")
    combined["player_id"] = pd.to_numeric(combined["player_id"], errors="coerce").astype("Int64")
    combined["game_id"] = pd.to_numeric(combined["game_id"], errors="coerce").astype("Int64")
    combined["season"] = combined["season"].astype(int)
    return combined


def write_tracking_partitions(df: pd.DataFrame, output_root: Path, overwrite_existing: bool = True) -> tuple[int, int]:
    if df.empty:
        return (0, 0)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    output_root.mkdir(parents=True, exist_ok=True)
    written_dates = 0
    skipped_dates = 0
    for game_date, frame in df.groupby("game_date"):
        season = int(frame["season"].iloc[0])
        partition_dir = output_root / f"season={season}" / f"game_date={game_date.date().isoformat()}"
        out_path = partition_dir / "tracking_roles.parquet"
        if out_path.exists() and not overwrite_existing:
            skipped_dates += 1
            continue
        partition_dir.mkdir(parents=True, exist_ok=True)
        cols = [
            "season",
            "game_date",
            "game_id",
            "team_id",
            "player_id",
            "track_touches_per_min_szn",
            "track_sec_per_touch_szn",
            "track_pot_ast_per_min_szn",
            "track_drives_per_min_szn",
            "track_role_cluster",
            "track_role_is_low_minutes",
        ]
        present = [c for c in cols if c in frame.columns]
        frame[present].to_parquet(out_path, index=False)
        written_dates += 1
    return written_dates, skipped_dates
