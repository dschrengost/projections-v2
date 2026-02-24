#!/usr/bin/env python3
"""Extract player-level raw count stats from bronze/boxscores_raw.

Reads the JSON payloads stored in bronze/boxscores_raw and extracts
per-player count statistics for all games. Output is written to
$DATA_ROOT/gold/labels_boxscore_counts/ partitioned by season.

Schema:
    game_id, team_id, player_id, game_date, season,
    fga2, fg2m, fga3, fg3m, fta, ftm,
    oreb, dreb, ast, stl, blk, tov, pf,
    minutes, starter_flag, played

Usage:
    uv run python3 scripts/rotation/build_boxscore_count_labels.py
    uv run python3 scripts/rotation/build_boxscore_count_labels.py --seasons 2022 2023
    uv run python3 scripts/rotation/build_boxscore_count_labels.py --out-dir /path/to/out
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from projections import paths

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mapping from NBA.com JSON payload field names to canonical output column names.
COUNT_STAT_MAP: dict[str, str] = {
    "twoPointersAttempted": "fga2",
    "twoPointersMade": "fg2m",
    "threePointersAttempted": "fga3",
    "threePointersMade": "fg3m",
    "freeThrowsAttempted": "fta",
    "freeThrowsMade": "ftm",
    "reboundsOffensive": "oreb",
    "reboundsDefensive": "dreb",
    "assists": "ast",
    "steals": "stl",
    "blocks": "blk",
    "turnovers": "tov",
    "foulsPersonal": "pf",
}

OUTPUT_COLS = [
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
    "oreb",
    "dreb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "minutes",
    "starter_flag",
    "played",
]

_MINUTES_RE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?")

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_minutes_iso(value: str | None) -> float | None:
    """Parse an ISO-8601 duration string (e.g. 'PT32M09.90S') into minutes."""
    if not value:
        return None
    m = _MINUTES_RE.fullmatch(str(value))
    if not m:
        return None
    hours = float(m.group(1) or 0)
    minutes = float(m.group(2) or 0)
    seconds = float(m.group(3) or 0)
    return hours * 60 + minutes + seconds / 60.0


def _season_label_from_date(game_date: pd.Timestamp) -> str:
    """Derive e.g. '2024-25' from game date."""
    year = game_date.year if game_date.month >= 8 else game_date.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


def _parse_payload(
    payload_str: str,
    game_date: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Parse one JSON payload string and return a list of player-stat records."""
    try:
        payload = json.loads(payload_str)
    except (json.JSONDecodeError, TypeError, ValueError):
        return []

    raw_game_id = payload.get("game_id")
    try:
        game_id = int(raw_game_id)
    except (TypeError, ValueError):
        return []

    season = _season_label_from_date(game_date)
    records: list[dict[str, Any]] = []

    for side in ("home", "away"):
        team = payload.get(side)
        if not team:
            continue
        try:
            team_id = int(team.get("team_id"))
        except (TypeError, ValueError):
            continue

        for player in team.get("players") or []:
            try:
                player_id = int(player.get("person_id"))
            except (TypeError, ValueError):
                continue

            stats = player.get("statistics") or {}
            row: dict[str, Any] = {
                "game_id": game_id,
                "team_id": team_id,
                "player_id": player_id,
                "game_date": game_date,
                "season": season,
                "minutes": _parse_minutes_iso(stats.get("minutes")),
                "starter_flag": bool(player.get("starter")),
                "played": bool(player.get("played")),
            }
            for payload_col, output_col in COUNT_STAT_MAP.items():
                v = stats.get(payload_col)
                if isinstance(v, (int, float)) and not (isinstance(v, float) and v != v):
                    row[output_col] = int(v)
                else:
                    row[output_col] = None
            records.append(row)

    return records


# ---------------------------------------------------------------------------
# File-level loading
# ---------------------------------------------------------------------------


def _load_bronze_partition(parquet_path: Path) -> list[dict[str, Any]]:
    """Load one bronze parquet partition and parse all game payloads."""
    # Derive game_date from directory path: .../date=2025-11-01/boxscores_raw.parquet
    date_part = parquet_path.parent.name  # "date=2025-11-01"
    if not date_part.startswith("date="):
        print(f"[count_labels] warning: cannot parse date from path {parquet_path}; skipping.")
        return []
    try:
        game_date = pd.Timestamp(date_part.split("=", 1)[1])
    except Exception:
        print(f"[count_labels] warning: bad date token in {parquet_path}; skipping.")
        return []

    try:
        df = pd.read_parquet(parquet_path, columns=["payload"])
    except Exception as exc:
        print(f"[count_labels] warning: failed to read {parquet_path}: {exc}; skipping.")
        return []

    records: list[dict[str, Any]] = []
    for payload_str in df["payload"]:
        records.extend(_parse_payload(payload_str, game_date))
    return records


# ---------------------------------------------------------------------------
# Main build logic
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=paths.get_project_root())  # noqa: S603,S607
            .decode()
            .strip()
        )
    except Exception:
        return None


def build(
    data_root: Path,
    seasons: list[int] | None,
    out_dir: Path,
    *,
    report_only: bool = False,
) -> pd.DataFrame:
    bronze_root = data_root / "bronze" / "boxscores_raw"
    if not bronze_root.exists():
        raise FileNotFoundError(f"bronze/boxscores_raw not found: {bronze_root}")

    # Resolve which season directories to process.
    all_season_dirs = sorted(p for p in bronze_root.iterdir() if p.is_dir() and p.name.startswith("season="))
    if seasons:
        season_set = {f"season={s}" for s in seasons}
        season_dirs = [p for p in all_season_dirs if p.name in season_set]
        missing = season_set - {p.name for p in season_dirs}
        if missing:
            raise FileNotFoundError(f"Requested seasons not found in {bronze_root}: {sorted(missing)}")
    else:
        season_dirs = all_season_dirs

    print(f"[count_labels] processing {len(season_dirs)} season(s): {[p.name for p in season_dirs]}")

    all_records: list[dict[str, Any]] = []
    for season_dir in season_dirs:
        partition_files = sorted(season_dir.glob("date=*/boxscores_raw.parquet"))
        print(f"[count_labels]   {season_dir.name}: {len(partition_files)} date partitions")
        for pf in partition_files:
            all_records.extend(_load_bronze_partition(pf))

    if not all_records:
        raise RuntimeError("No records parsed from bronze/boxscores_raw.")

    df = pd.DataFrame(all_records)

    # Type coercions
    for col in ("game_id", "team_id", "player_id"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    df["season"] = df["season"].astype("string")
    df["starter_flag"] = df["starter_flag"].astype(bool)
    df["played"] = df["played"].astype(bool)
    for col in list(COUNT_STAT_MAP.values()):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").astype("Float64")

    # Drop rows with unparseable keys
    key_null = df[["game_id", "team_id", "player_id", "game_date"]].isna().any(axis=1)
    if key_null.any():
        print(f"[count_labels] warning: dropping {key_null.sum()} rows with null key columns")
        df = df.loc[~key_null].reset_index(drop=True)

    # Deduplicate on natural key
    key_cols = ["game_id", "team_id", "player_id", "game_date"]
    pre_dupe = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="last")
    if len(df) < pre_dupe:
        print(f"[count_labels] deduplication: {pre_dupe} -> {len(df)} rows")

    df = df.sort_values(["game_date", "game_id", "team_id", "player_id"]).reset_index(drop=True)

    # Ensure all output columns exist (some may be absent if stats dict was empty)
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = None

    df = df[OUTPUT_COLS]

    print(f"[count_labels] rows: {len(df)}")
    print(f"[count_labels] games: {df['game_id'].nunique()}")
    print(f"[count_labels] date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
    print(f"[count_labels] seasons: {sorted(df['season'].dropna().unique())}")
    print(f"[count_labels] played coverage: {df['played'].mean():.1%}")
    print(f"[count_labels] fga2 null rate: {df['fga2'].isna().mean():.1%}")
    print(f"[count_labels] minutes null rate: {df['minutes'].isna().mean():.1%}")

    if report_only:
        print("[count_labels] report-only; no files written.")
        return df

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "labels_boxscore_counts.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[count_labels] wrote {len(df):,} rows -> {out_path}")

    manifest = {
        "created_at": _utc_now_iso(),
        "git_sha": _git_sha(),
        "bronze_root": str(bronze_root),
        "seasons_processed": [p.name for p in season_dirs],
        "rows": int(len(df)),
        "games": int(df["game_id"].nunique()),
        "date_range": {
            "min": str(df["game_date"].min().date()),
            "max": str(df["game_date"].max().date()),
        },
        "output": str(out_path),
    }
    import json as _json
    (out_dir / "manifest.json").write_text(_json.dumps(manifest, indent=2), encoding="utf-8")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=None,
        metavar="YEAR",
        help=(
            "Season start years to process (e.g. 2022 2023 2024 2025). "
            "If omitted, all available seasons in bronze/boxscores_raw are processed."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to $DATA_ROOT/gold/labels_boxscore_counts/.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Data root (default: PROJECTIONS_DATA_ROOT or ./data).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Parse and print diagnostics without writing files.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else paths.get_data_root()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (data_root / "gold" / "labels_boxscore_counts")
    )

    build(
        data_root=data_root,
        seasons=args.seasons,
        out_dir=out_dir,
        report_only=args.report_only,
    )


if __name__ == "__main__":
    main()
