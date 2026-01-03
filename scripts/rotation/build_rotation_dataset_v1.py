#!/usr/bin/env python3
"""Build a first-pass silver rotation dataset from NBA Stats GameRotation bronze JSON.

This script scans the bronze GameRotation responses we have already scraped and
normalizes them into per-game Parquet outputs under:

  ${PROJECTIONS_DATA_ROOT}/silver/rotation_v1/

It is incremental/idempotent: existing outputs are skipped unless --overwrite.
Games that fail parsing or basic sanity checks are appended to:

  silver/rotation_v1/_quarantine.jsonl

Usage examples:
  # Report what would be processed
  uv run python scripts/rotation/build_rotation_dataset_v1.py --report-only

  # Smoke build a handful of games
  uv run python scripts/rotation/build_rotation_dataset_v1.py --max-games 5 --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from projections import paths

BRONZE_REL_ROOT = Path("bronze/nba_stats/gamerotation_v1")
SILVER_REL_ROOT = Path("silver/rotation_v1")


class RotationBuildError(Exception):
    """Base error for rotation_v1 build failures."""


class RotationParseError(RotationBuildError):
    """Raised when a bronze file cannot be parsed into stints."""


class RotationSchemaError(RotationBuildError):
    """Raised when the expected resultSets/columns are missing."""


class RotationSanityError(RotationBuildError):
    """Raised when a parsed game fails sanity checks."""


@dataclass(frozen=True)
class BronzeGameFile:
    season: int
    game_id_int: int
    game_id: str
    path: Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _snake_case(value: str) -> str:
    value = value.strip()
    if not value:
        return value
    out: list[str] = []
    prev_was_underscore = False
    for ch in value:
        if ch.isalnum():
            out.append(ch.lower())
            prev_was_underscore = False
        else:
            if not prev_was_underscore:
                out.append("_")
                prev_was_underscore = True
    snake = "".join(out).strip("_")
    while "__" in snake:
        snake = snake.replace("__", "_")
    return snake


def _extract_partition_value(name: str, prefix: str) -> str | None:
    if not name.startswith(prefix):
        return None
    value = name[len(prefix) :]
    return value or None


def _scan_bronze(bronze_root: Path) -> list[BronzeGameFile]:
    if not bronze_root.exists():
        return []

    games: list[BronzeGameFile] = []
    for path in sorted(bronze_root.glob("season=*/game_id=*.json")):
        season_str = _extract_partition_value(path.parent.name, "season=")
        game_id_str = _extract_partition_value(path.stem, "game_id=")
        if season_str is None or game_id_str is None:
            continue
        try:
            season = int(season_str)
            game_id_int = int(game_id_str)
        except ValueError:
            continue
        games.append(
            BronzeGameFile(
                season=season,
                game_id_int=game_id_int,
                game_id=str(game_id_int).zfill(10),
                path=path,
            )
        )
    return games


def _load_quarantined_games(quarantine_path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    if not quarantine_path.exists():
        return {}

    quarantined: dict[tuple[int, str], dict[str, Any]] = {}
    with quarantine_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            season = record.get("season")
            game_id = record.get("game_id")
            try:
                season_int = int(season)
            except Exception:
                continue
            if game_id is None:
                continue
            quarantined[(season_int, str(game_id))] = {
                "source_mtime": record.get("source_mtime"),
                "source_size": record.get("source_size"),
            }
    return quarantined


def _append_quarantine(quarantine_path: Path, record: dict[str, Any]) -> None:
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    with quarantine_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _extract_result_sets(response_json: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(response_json.get("resultSets"), list):
        return [rs for rs in response_json["resultSets"] if isinstance(rs, dict)]
    result_set = response_json.get("resultSet")
    if isinstance(result_set, dict):
        return [result_set]
    if isinstance(result_set, list):
        return [rs for rs in result_set if isinstance(rs, dict)]
    return []


def _result_set_to_frame(result_set: dict[str, Any]) -> pd.DataFrame | None:
    headers = result_set.get("headers") or result_set.get("Headers")
    row_set = result_set.get("rowSet") or result_set.get("RowSet")
    if not isinstance(headers, list) or not isinstance(row_set, list):
        return None

    cols = [_snake_case(str(h)) for h in headers]
    rows: list[dict[str, Any]] = []
    for row in row_set:
        if not isinstance(row, list):
            continue
        record: dict[str, Any] = {}
        for idx, col in enumerate(cols):
            record[col] = row[idx] if idx < len(row) else None
        rows.append(record)

    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame.from_records(rows)
    name = result_set.get("name")
    if name is not None:
        df["result_set_name"] = str(name)
    return df


def _infer_time_unit(max_out_time_real: float) -> tuple[str, float]:
    """Infer unit for IN_TIME_REAL/OUT_TIME_REAL.

    Heuristic based on observed ranges:
      - ~2880 (or ~3180 with OT): seconds
      - ~28800 (or ~31800 with OT): tenths-of-seconds
    Returns (time_unit, seconds_per_unit).
    """
    if not math.isfinite(max_out_time_real) or max_out_time_real <= 0:
        return "unknown", float("nan")
    if max_out_time_real <= 4000:
        return "seconds", 1.0
    if max_out_time_real <= 40000:
        return "tenth_seconds", 0.1
    return "unknown", float("nan")


def _safe_to_numeric(series: pd.Series, *, name: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.isna().any():
        bad = out.isna().sum()
        raise RotationParseError(f"Column {name} has {bad} non-numeric values")
    return out


def _atomic_write_parquet(df: pd.DataFrame, output_path: Path, *, overwrite: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)


def _output_paths(silver_root: Path, season: int, game_id: str) -> dict[str, Path]:
    return {
        "player_stints": silver_root / "player_stints" / f"season={season}" / f"game_id={game_id}.parquet",
        "player_game_labels": silver_root
        / "player_game_labels"
        / f"season={season}"
        / f"game_id={game_id}.parquet",
        "team_game_shape": silver_root / "team_game_shape" / f"season={season}" / f"game_id={game_id}.parquet",
    }


def _all_outputs_exist(paths_by_table: dict[str, Path]) -> bool:
    return all(p.exists() for p in paths_by_table.values())


def _validate_parquet_output(output_path: Path) -> bool:
    if not output_path.exists():
        return False
    try:
        df = pd.read_parquet(output_path)
    except Exception:
        return False
    return len(df) > 0


def _outputs_are_valid(paths_by_table: dict[str, Path]) -> bool:
    return all(_validate_parquet_output(p) for p in paths_by_table.values())


def _source_signature(path: Path) -> tuple[float, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime, stat.st_size)


def _parse_bronze_game(
    bronze: BronzeGameFile,
    *,
    data_root: Path,
) -> pd.DataFrame:
    try:
        wrapper = json.loads(bronze.path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RotationParseError(f"Invalid JSON in bronze wrapper: {exc}") from exc
    response_json = wrapper.get("response_json")
    if response_json is None:
        error = wrapper.get("error") or "missing response_json"
        raise RotationParseError(f"Bronze wrapper has no response_json (error={error})")
    if not isinstance(response_json, dict):
        raise RotationParseError(f"response_json is not an object: {type(response_json).__name__}")

    frames: list[pd.DataFrame] = []
    for rs in _extract_result_sets(response_json):
        df = _result_set_to_frame(rs)
        if df is None:
            continue
        required = {"team_id", "person_id", "in_time_real", "out_time_real"}
        if not required.issubset(set(df.columns)):
            continue
        frames.append(df)

    if not frames:
        raise RotationSchemaError("No resultSets with required rotation columns found")

    stints = pd.concat(frames, ignore_index=True)

    # Canonicalize identifiers.
    stints["season"] = int(bronze.season)
    stints["game_id"] = bronze.game_id

    source_file: str
    try:
        source_file = str(bronze.path.relative_to(data_root))
    except Exception:
        source_file = str(bronze.path)
    stints["source_file"] = source_file
    stints["parsed_at"] = _utc_now_iso()

    stints["team_id"] = _safe_to_numeric(stints["team_id"], name="team_id").astype("int64")
    stints["person_id"] = _safe_to_numeric(stints["person_id"], name="person_id").astype("int64")
    stints["in_time_real"] = _safe_to_numeric(stints["in_time_real"], name="in_time_real").astype("float64")
    stints["out_time_real"] = _safe_to_numeric(stints["out_time_real"], name="out_time_real").astype("float64")

    max_out = float(stints["out_time_real"].max())
    time_unit, seconds_per_unit = _infer_time_unit(max_out)
    if time_unit == "unknown":
        raise RotationParseError(f"Unable to infer time unit (max_out_time_real={max_out})")

    stints["time_unit"] = time_unit
    stints["time_unit_detected"] = time_unit
    stints["max_out_time_real_raw"] = max_out
    stints["seconds_per_unit"] = float(seconds_per_unit)
    stints["stint_len_real"] = stints["out_time_real"] - stints["in_time_real"]
    stints["stint_minutes"] = stints["stint_len_real"] * seconds_per_unit / 60.0
    stints["game_duration_minutes"] = max_out * seconds_per_unit / 60.0
    team_totals = (
        stints.groupby("team_id", as_index=False)["stint_minutes"]
        .sum()
        .rename(columns={"stint_minutes": "team_total_minutes_from_stints"})
    )
    stints = stints.merge(team_totals, on="team_id", how="left")

    return stints


def _build_player_game_labels(stints: pd.DataFrame) -> pd.DataFrame:
    grouped = stints.groupby(["season", "game_id", "team_id", "person_id"], as_index=False)
    agg = grouped.agg(
        minutes_from_stints=("stint_minutes", "sum"),
        num_stints=("stint_minutes", "size"),
        first_in_time_real=("in_time_real", "min"),
        last_out_time_real=("out_time_real", "max"),
        max_stint_len_real=("stint_len_real", "max"),
        time_unit=("time_unit", "first"),
        time_unit_detected=("time_unit_detected", "first"),
        max_out_time_real_raw=("max_out_time_real_raw", "first"),
        seconds_per_unit=("seconds_per_unit", "first"),
        game_duration_minutes=("game_duration_minutes", "first"),
        source_file=("source_file", "first"),
        parsed_at=("parsed_at", "first"),
    )
    agg["started_proxy"] = agg["first_in_time_real"].astype("float64").abs() < 1e-9
    agg["played_flag"] = agg["minutes_from_stints"] > 0.0

    team_totals = agg.groupby(["season", "game_id", "team_id"], as_index=False).agg(
        team_total_minutes_from_stints=("minutes_from_stints", "sum")
    )
    merged = agg.merge(team_totals, on=["season", "game_id", "team_id"], how="left")
    return merged


def _build_team_game_shape(player_labels: pd.DataFrame) -> pd.DataFrame:
    team_base = player_labels.groupby(["season", "game_id", "team_id"], as_index=False).agg(
        team_total_minutes_from_stints=("minutes_from_stints", "sum"),
        game_duration_minutes=("game_duration_minutes", "first"),
        time_unit=("time_unit", "first"),
        time_unit_detected=("time_unit_detected", "first"),
        max_out_time_real_raw=("max_out_time_real_raw", "first"),
        seconds_per_unit=("seconds_per_unit", "first"),
        source_file=("source_file", "first"),
        parsed_at=("parsed_at", "first"),
    )

    def _shape_for_team(group: pd.DataFrame) -> dict[str, Any]:
        team_total = float(group["minutes_from_stints"].sum())
        minutes = group["minutes_from_stints"].astype("float64")
        depth_6 = int((minutes >= 6.0).sum())
        depth_10 = int((minutes >= 10.0).sum())
        depth_14 = int((minutes >= 14.0).sum())

        if team_total > 0:
            shares = (minutes / team_total).to_numpy()
            effective_n = float(1.0 / float((shares**2).sum())) if shares.size else 0.0
        else:
            effective_n = 0.0

        starters_mask = group["started_proxy"].astype(bool)
        starter_pool = float(group.loc[starters_mask, "minutes_from_stints"].sum())
        bench_pool = float(team_total - starter_pool)

        bench = group.loc[~starters_mask, ["minutes_from_stints"]].copy()
        bench_sorted = bench["minutes_from_stints"].sort_values(ascending=False).to_list()
        top1 = bench_sorted[0] if len(bench_sorted) >= 1 else 0.0
        top2 = bench_sorted[1] if len(bench_sorted) >= 2 else 0.0

        bench_conc_top1 = float(top1 / bench_pool) if bench_pool > 0 else 0.0
        bench_conc_top2 = float((top1 + top2) / bench_pool) if bench_pool > 0 else 0.0

        return {
            "depth_6": depth_6,
            "depth_10": depth_10,
            "depth_14": depth_14,
            "effective_n": effective_n,
            "starter_pool_minutes": starter_pool,
            "bench_pool_minutes": bench_pool,
            "bench_conc_top1": bench_conc_top1,
            "bench_conc_top2": bench_conc_top2,
        }

    shapes: list[dict[str, Any]] = []
    for (season, game_id, team_id), group in player_labels.groupby(["season", "game_id", "team_id"]):
        shape = _shape_for_team(group)
        shape.update({"season": season, "game_id": game_id, "team_id": int(team_id)})
        shapes.append(shape)

    shapes_df = pd.DataFrame(shapes)
    out = team_base.merge(shapes_df, on=["season", "game_id", "team_id"], how="left")
    return out


def _overlap_sanity(team_stints: pd.DataFrame, *, n_points: int = 25) -> dict[str, float] | None:
    if team_stints.empty:
        return None
    max_out = float(team_stints["out_time_real"].max())
    if not math.isfinite(max_out) or max_out <= 0:
        return None
    # Sample points in the interior to avoid boundary effects.
    points = [max_out * (i + 0.5) / n_points for i in range(n_points)]
    counts: list[int] = []
    in_times = team_stints["in_time_real"].to_numpy()
    out_times = team_stints["out_time_real"].to_numpy()
    for t in points:
        active = int(((in_times <= t) & (out_times > t)).sum())
        counts.append(active)
    if not counts:
        return None
    mean_active = float(sum(counts) / len(counts))
    return {
        "mean_active": mean_active,
        "min_active": float(min(counts)),
        "max_active": float(max(counts)),
    }


def _validate_game(stints: pd.DataFrame, player_labels: pd.DataFrame) -> list[str]:
    issues: list[str] = []

    if stints.empty:
        return ["stints table is empty"]

    if (stints["stint_len_real"] < 0).any():
        bad = int((stints["stint_len_real"] < 0).sum())
        issues.append(f"{bad} stints have out_time_real < in_time_real")

    if (stints["stint_minutes"] < -1e-9).any():
        bad = int((stints["stint_minutes"] < -1e-9).sum())
        issues.append(f"{bad} stints have negative minutes")

    max_game_minutes = float(stints["game_duration_minutes"].max())
    if not math.isfinite(max_game_minutes) or max_game_minutes <= 0:
        issues.append("invalid game_duration_minutes")
        return issues

    expected_team_total = 5.0 * max_game_minutes
    tolerance = max(2.0, 0.01 * expected_team_total)

    team_totals = (
        player_labels.groupby(["team_id"], as_index=False)["minutes_from_stints"].sum().rename(
            columns={"minutes_from_stints": "team_total_minutes_from_stints"}
        )
    )
    for _, row in team_totals.iterrows():
        team_id = int(row["team_id"])
        total = float(row["team_total_minutes_from_stints"])
        if not math.isfinite(total) or total < 0:
            issues.append(f"team_id={team_id} has invalid total minutes: {total}")
            continue
        if abs(total - expected_team_total) > tolerance:
            issues.append(
                f"team_id={team_id} total minutes {total:.2f} != expected {expected_team_total:.2f} (tol={tolerance:.2f})"
            )

        overlap = _overlap_sanity(stints.loc[stints["team_id"] == team_id])
        if overlap is not None:
            mean_active = overlap["mean_active"]
            min_active = overlap["min_active"]
            max_active = overlap["max_active"]
            if mean_active < 4.0 or mean_active > 6.0 or min_active < 3.0 or max_active > 7.0:
                issues.append(
                    "overlap_sanity failed for team_id="
                    f"{team_id} (mean={mean_active:.2f}, min={min_active:.0f}, max={max_active:.0f})"
                )

    return issues


def _run_report(bronze_games: list[BronzeGameFile], *, silver_root: Path, quarantined: dict[tuple[int, str], dict[str, Any]]) -> None:
    built = 0
    partial = 0
    time_unit_counts: dict[str, int] = {}
    for game in bronze_games:
        outputs = _output_paths(silver_root, game.season, game.game_id)
        if _outputs_are_valid(outputs):
            built += 1
            try:
                df = pd.read_parquet(outputs["team_game_shape"])
            except Exception:
                time_unit = "unreadable"
            else:
                if df.empty:
                    time_unit = "empty"
                elif "time_unit_detected" in df.columns:
                    time_unit = str(df["time_unit_detected"].iloc[0])
                elif "time_unit" in df.columns:
                    time_unit = str(df["time_unit"].iloc[0])
                else:
                    time_unit = "missing"
            time_unit_counts[time_unit] = time_unit_counts.get(time_unit, 0) + 1
        elif any(p.exists() for p in outputs.values()):
            partial += 1

    quarantined_in_bronze = sum((g.season, g.game_id) in quarantined for g in bronze_games)

    print("[rotation_v1] Report")
    print(f"  Bronze games found:         {len(bronze_games)}")
    print(f"  Processed (valid outputs):  {built}")
    print(f"  Partial/corrupt outputs:    {partial}")
    print(f"  Games quarantined (unique): {len(quarantined)}")
    print(f"  Quarantined present in bronze scan: {quarantined_in_bronze}")
    if time_unit_counts:
        print("  time_unit_detected distribution:")
        for key in sorted(time_unit_counts):
            print(f"    {key}: {time_unit_counts[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build silver rotation_v1 datasets from bronze nba_stats gamerotation_v1 JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=str, default=None, help="Override PROJECTIONS_DATA_ROOT.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Parquet outputs.")
    parser.add_argument(
        "--max-games", type=int, default=None, help="Maximum number of games to process (smoke test)."
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only scan bronze/silver and print counts; do not build.",
    )
    parser.add_argument(
        "--retry-quarantined",
        action="store_true",
        help="Attempt to process games already in _quarantine.jsonl.",
    )

    args = parser.parse_args()
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else paths.get_data_root()

    bronze_root = (data_root / BRONZE_REL_ROOT).resolve()
    silver_root = (data_root / SILVER_REL_ROOT).resolve()
    quarantine_path = silver_root / "_quarantine.jsonl"

    bronze_games = _scan_bronze(bronze_root)
    quarantined = _load_quarantined_games(quarantine_path)

    if args.report_only:
        _run_report(bronze_games, silver_root=silver_root, quarantined=quarantined)
        return

    if args.overwrite:
        retry_quarantined = True
    else:
        retry_quarantined = bool(args.retry_quarantined)

    to_process: list[BronzeGameFile] = []
    for game in bronze_games:
        outputs = _output_paths(silver_root, game.season, game.game_id)
        outputs_valid = _outputs_are_valid(outputs)
        if outputs_valid and not args.overwrite:
            continue
        quarantine_record = quarantined.get((game.season, game.game_id))
        if quarantine_record and not retry_quarantined:
            signature = _source_signature(game.path)
            if signature is not None:
                mtime, size = signature
                if (
                    quarantine_record.get("source_mtime") == mtime
                    and quarantine_record.get("source_size") == size
                ):
                    continue
        to_process.append(game)

    if args.max_games is not None:
        to_process = to_process[: args.max_games]

    print("[rotation_v1] Starting build")
    print(f"  Data root:   {data_root}")
    print(f"  Bronze root: {bronze_root}")
    print(f"  Silver root: {silver_root}")
    print(f"  Games found: {len(bronze_games)}")
    print(f"  To process:  {len(to_process)}")

    processed = 0
    skipped = 0
    quarantined_count = 0

    for idx, game in enumerate(to_process, start=1):
        outputs = _output_paths(silver_root, game.season, game.game_id)
        already = {k: p.exists() for k, p in outputs.items()}
        if not args.overwrite and _outputs_are_valid(outputs):
            skipped += 1
            continue

        print(
            f"[rotation_v1] ({idx}/{len(to_process)}) game_id={game.game_id} season={game.season}",
            flush=True,
        )
        try:
            stints = _parse_bronze_game(game, data_root=data_root)
            player_labels = _build_player_game_labels(stints)
            issues = _validate_game(stints, player_labels)
            if issues:
                raise RotationSanityError("; ".join(issues))
            team_shape = _build_team_game_shape(player_labels)

            # Write what is missing (unless overwriting).
            _atomic_write_parquet(stints, outputs["player_stints"], overwrite=args.overwrite)
            _atomic_write_parquet(player_labels, outputs["player_game_labels"], overwrite=args.overwrite)
            _atomic_write_parquet(team_shape, outputs["team_game_shape"], overwrite=args.overwrite)

            processed += 1
        except Exception as exc:
            quarantined_count += 1
            if isinstance(exc, RotationParseError):
                error_type = "parse_error"
            elif isinstance(exc, RotationSchemaError):
                error_type = "schema_error"
            elif isinstance(exc, RotationSanityError):
                error_type = "sanity_error"
            elif isinstance(exc, OSError):
                error_type = "io_error"
            else:
                error_type = "build_error"

            try:
                quarantine_source = str(game.path.relative_to(data_root))
            except Exception:
                quarantine_source = str(game.path)
            signature = _source_signature(game.path)
            source_mtime = signature[0] if signature is not None else None
            source_size = signature[1] if signature is not None else None
            record = {
                "timestamp": _utc_now_iso(),
                "season": game.season,
                "game_id": game.game_id,
                "error_type": error_type,
                "exception_type": exc.__class__.__name__,
                "message": str(exc),
                "source_file": quarantine_source,
                "source_mtime": source_mtime,
                "source_size": source_size,
                "outputs_present": already,
            }
            _append_quarantine(quarantine_path, record)
            print(f"[rotation_v1] QUARANTINED game_id={game.game_id}: {exc}", flush=True)

    print("[rotation_v1] Build complete")
    print(f"  Processed:   {processed}")
    print(f"  Skipped:     {skipped}")
    print(f"  Quarantined: {quarantined_count}")
    print(f"  Quarantine log: {quarantine_path}")


if __name__ == "__main__":
    main()
