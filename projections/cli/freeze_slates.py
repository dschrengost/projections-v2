"""Freeze immutable per-game slate snapshots at lock time and pre-tip."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import typer

from projections import paths
from projections.etl import storage as bronze_storage
from projections.minutes_v1.features import MinutesFeatureBuilder
from projections.minutes_v1.schemas import (
    BOX_SCORE_LABELS_SCHEMA,
    FEATURES_MINUTES_V1_SCHEMA,
    SLATE_FEATURES_MINUTES_V1_SCHEMA,
    enforce_schema,
    validate_with_pandera,
)
from projections.minutes_v1.snapshots import select_latest_before
from projections.minutes_v1.starter_flags import derive_starter_flag_label, normalize_starter_signals

UTC = timezone.utc
ET_TZ = ZoneInfo("America/New_York")

app = typer.Typer(help=__doc__)


def _read_parquet_tree(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.is_file():
        return pd.read_parquet(path)
    files = sorted(path.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(file) for file in files]
    return pd.concat(frames, ignore_index=True)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _season_label(season_start: int) -> str:
    return f"{season_start}-{(season_start + 1) % 100:02d}"


def _git_rev_parse_head() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S603, S607
            or None
        )
    except Exception:  # noqa: BLE001
        return None


def _max_ts(df: pd.DataFrame, col: str) -> str | None:
    if df.empty or col not in df.columns:
        return None
    ts = pd.to_datetime(df[col], utc=True, errors="coerce").dropna()
    if ts.empty:
        return None
    return ts.max().isoformat()


@dataclass(frozen=True)
class ScheduleRow:
    season: int
    game_id: int
    game_date: date
    tip_ts: pd.Timestamp
    schedule_row: dict[str, Any]


def _load_schedule_row(data_root: Path, game_id: int, *, season: int | None = None) -> ScheduleRow:
    schedule_root = data_root / "silver" / "schedule"
    season_dirs: Iterable[Path]
    if season is None:
        season_dirs = sorted(schedule_root.glob("season=*"))
    else:
        season_dirs = [schedule_root / f"season={season}"]

    target = int(game_id)
    for season_dir in season_dirs:
        if not season_dir.exists():
            continue
        df = _read_parquet_tree(season_dir)
        if df.empty or "game_id" not in df.columns:
            continue
        ids = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
        matches = df.loc[ids == target]
        if matches.empty:
            continue
        row = matches.iloc[0].to_dict()
        season_value = int(season_dir.name.split("=", 1)[1])
        tip_ts = pd.to_datetime(row.get("tip_ts"), utc=True, errors="coerce")
        if pd.isna(tip_ts):
            raise RuntimeError(f"Schedule row for game_id={game_id} missing tip_ts")
        game_date = pd.to_datetime(row.get("game_date"), errors="coerce").normalize()
        if pd.isna(game_date):
            game_date = tip_ts.tz_convert(ET_TZ).tz_localize(None).normalize()
        return ScheduleRow(
            season=season_value,
            game_id=target,
            game_date=game_date.date(),
            tip_ts=pd.Timestamp(tip_ts),
            schedule_row=row,
        )

    raise FileNotFoundError(f"Unable to locate schedule row for game_id={game_id} under {schedule_root}")


def _load_labels_history(data_root: Path, season: int) -> pd.DataFrame:
    labels_path = data_root / "labels" / f"season={season}" / "boxscore_labels.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing boxscore labels at {labels_path}")
    labels = pd.read_parquet(labels_path)
    if "label_frozen_ts" in labels.columns:
        labels["label_frozen_ts"] = pd.to_datetime(labels["label_frozen_ts"], utc=True, errors="coerce")
    else:
        labels["label_frozen_ts"] = pd.NaT

    minutes_col = labels.get("minutes")
    if minutes_col is not None and minutes_col.dtype == object:
        parsed = pd.to_timedelta(minutes_col, errors="coerce")
        labels["minutes"] = (parsed.dt.total_seconds() / 60.0).astype("Float64")
    if "minutes" in labels.columns:
        labels["minutes"] = pd.to_numeric(labels["minutes"], errors="coerce")

    if "starter_flag_label" not in labels.columns:
        starter_series = labels.get("starter_flag")
        starter_bool = (
            starter_series.astype("boolean", copy=False).fillna(False)
            if starter_series is not None
            else pd.Series(False, index=labels.index, dtype="boolean")
        )
        labels["starter_flag_label"] = starter_bool.astype("Int64")

    labels = enforce_schema(labels, BOX_SCORE_LABELS_SCHEMA, allow_missing_optional=True)
    labels["game_date"] = pd.to_datetime(labels["game_date"]).dt.normalize()
    labels.sort_values(
        ["game_id", "team_id", "player_id", "label_frozen_ts"],
        inplace=True,
        kind="mergesort",
    )
    labels = labels.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last")
    return labels


def _load_schedule_for_builder(data_root: Path, season: int, game_ids: list[int]) -> pd.DataFrame:
    schedule_root = data_root / "silver" / "schedule" / f"season={season}"
    schedule_df = _read_parquet_tree(schedule_root)
    if schedule_df.empty:
        raise FileNotFoundError(f"Missing schedule partitions under {schedule_root}")
    schedule_df["game_id"] = pd.to_numeric(schedule_df["game_id"], errors="coerce").astype("Int64")
    schedule_df["tip_ts"] = pd.to_datetime(schedule_df["tip_ts"], utc=True, errors="coerce")
    if "game_date" in schedule_df.columns:
        schedule_df["game_date"] = pd.to_datetime(schedule_df["game_date"], errors="coerce").dt.normalize()
    needed = set(int(gid) for gid in game_ids)
    return schedule_df.loc[schedule_df["game_id"].isin(needed)].copy()


def _load_roster_history(data_root: Path, season: int, game_date: datetime.date) -> pd.DataFrame:
    roster_root = data_root / "silver" / "roster_nightly" / f"season={season}"
    month_path = roster_root / f"month={game_date.month:02d}" / "roster.parquet"
    if month_path.exists():
        df = pd.read_parquet(month_path)
    else:
        df = _read_parquet_tree(roster_root)
    if df.empty:
        return df
    df["game_id"] = pd.to_numeric(df.get("game_id"), errors="coerce").astype("Int64")
    df["player_id"] = pd.to_numeric(df.get("player_id"), errors="coerce").astype("Int64")
    df["team_id"] = pd.to_numeric(df.get("team_id"), errors="coerce").astype("Int64")
    df["as_of_ts"] = pd.to_datetime(df.get("as_of_ts"), utc=True, errors="coerce")
    df["game_date"] = pd.to_datetime(df.get("game_date"), errors="coerce").dt.normalize()
    return df


def _build_live_labels_for_game(
    roster_snapshot: pd.DataFrame,
    *,
    game_id: int,
    game_date: pd.Timestamp,
    season_label: str,
    frozen_at: pd.Timestamp,
) -> pd.DataFrame:
    if roster_snapshot.empty:
        raise RuntimeError("Roster snapshot is empty; cannot build live labels.")

    working = roster_snapshot.copy()
    working = working.dropna(subset=["game_id", "player_id", "team_id"])
    working["game_id"] = pd.to_numeric(working["game_id"], errors="coerce").astype("Int64")
    working["player_id"] = pd.to_numeric(working["player_id"], errors="coerce").astype("Int64")
    working["team_id"] = pd.to_numeric(working["team_id"], errors="coerce").astype("Int64")
    working["game_date"] = pd.to_datetime(working.get("game_date"), errors="coerce").dt.normalize()
    working = working[(working["game_id"] == int(game_id)) & (working["game_date"] == game_date)]
    if working.empty:
        raise RuntimeError(f"Roster snapshot has no rows for game_id={game_id} date={game_date.date()}")

    if "as_of_ts" in working.columns:
        working["as_of_ts"] = pd.to_datetime(working["as_of_ts"], utc=True, errors="coerce")
        working = working.sort_values(["game_id", "team_id", "player_id", "as_of_ts"])
    else:
        working = working.sort_values(["game_id", "team_id", "player_id"])
    working = working.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last")
    working = normalize_starter_signals(working)

    starter_result = derive_starter_flag_label(working, group_cols=("game_id", "team_id"))
    starter_series = starter_result.values.reindex(working.index).fillna(0).astype("Int64")

    live_df = pd.DataFrame(
        {
            "game_id": working["game_id"].astype("Int64"),
            "player_id": working["player_id"].astype("Int64"),
            "team_id": working["team_id"].astype("Int64"),
            "player_name": working.get("player_name"),
            "season": season_label,
            "game_date": working["game_date"],
            "minutes": pd.Series(pd.NA, index=working.index, dtype="Float64"),
            "starter_flag": starter_series,
            "starter_flag_label": starter_series,
            "source": "slate_freeze_roster",
            "label_frozen_ts": frozen_at,
        }
    )
    return enforce_schema(live_df, BOX_SCORE_LABELS_SCHEMA, allow_missing_optional=True)


def _build_live_labels_from_boxscore_labels(
    labels_for_game: pd.DataFrame,
    *,
    game_id: int,
    game_date: pd.Timestamp,
    season_label: str,
    frozen_at: pd.Timestamp,
) -> pd.DataFrame:
    """Fallback live-label scaffold when roster snapshots are unavailable.

    Uses the boxscore label roster for the game to enumerate players/teams, but
    explicitly clears any outcome columns (minutes/starter flags) to avoid
    leakage in lock/pretip snapshots.
    """

    if labels_for_game.empty:
        raise RuntimeError("Boxscore labels are empty; cannot build live labels.")

    working = labels_for_game.copy()
    working = working.dropna(subset=["game_id", "player_id", "team_id"])
    working["game_id"] = pd.to_numeric(working["game_id"], errors="coerce").astype("Int64")
    working["player_id"] = pd.to_numeric(working["player_id"], errors="coerce").astype("Int64")
    working["team_id"] = pd.to_numeric(working["team_id"], errors="coerce").astype("Int64")
    working["game_date"] = pd.to_datetime(working.get("game_date"), errors="coerce").dt.normalize()
    working = working[(working["game_id"] == int(game_id)) & (working["game_date"] == game_date)]
    if working.empty:
        raise RuntimeError(f"Boxscore labels have no rows for game_id={game_id} date={game_date.date()}")

    working = working.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last")
    live_df = pd.DataFrame(
        {
            "game_id": working["game_id"].astype("Int64"),
            "player_id": working["player_id"].astype("Int64"),
            "team_id": working["team_id"].astype("Int64"),
            "player_name": working.get("player_name"),
            "season": season_label,
            "game_date": working["game_date"],
            "minutes": pd.Series(pd.NA, index=working.index, dtype="Float64"),
            "starter_flag": pd.Series(0, index=working.index, dtype="Int64"),
            "starter_flag_label": pd.Series(0, index=working.index, dtype="Int64"),
            "source": "slate_freeze_boxscore_roster",
            "label_frozen_ts": frozen_at,
        }
    )
    return enforce_schema(live_df, BOX_SCORE_LABELS_SCHEMA, allow_missing_optional=True)


def _freeze_game_snapshot(
    schedule: ScheduleRow,
    *,
    snapshot_type: str,
    data_root: Path,
    out_root: Path,
    force: bool,
    history_days: int | None,
    require_history: bool = True,
) -> tuple[Path, Path]:
    snapshot_type_norm = snapshot_type.strip().lower()
    if snapshot_type_norm not in {"lock", "pretip"}:
        raise typer.BadParameter("--snapshot-type must be 'lock' or 'pretip'.")

    tip_ts = pd.to_datetime(schedule.tip_ts, utc=True)
    snapshot_ts = tip_ts - pd.Timedelta(minutes=15) if snapshot_type_norm == "lock" else tip_ts
    frozen_at = pd.Timestamp.now(tz="UTC")

    out_path = (
        out_root
        / f"season={schedule.season}"
        / f"game_date={schedule.game_date.isoformat()}"
        / f"game_id={schedule.game_id}"
    )
    out_path.mkdir(parents=True, exist_ok=True)
    parquet_path = out_path / f"{snapshot_type_norm}.parquet"
    manifest_path = out_path / f"manifest.{snapshot_type_norm}.json"
    if parquet_path.exists() and not force:
        raise RuntimeError(f"Gold snapshot already exists at {parquet_path}. Use --force to overwrite.")

    # Bronze odds history is partitioned by game_date (ET).
    odds_raw = bronze_storage.read_bronze_day(
        "odds_raw",
        data_root,
        schedule.season,
        schedule.game_date,
        include_runs=True,
        prefer_history=True,
    )
    odds_raw = odds_raw[odds_raw.get("game_id") == schedule.game_id].copy() if not odds_raw.empty else odds_raw

    # Bronze injuries are partitioned by report date (ET); include previous day for safety.
    injuries_frames: list[pd.DataFrame] = []
    for day in (schedule.game_date - timedelta(days=1), schedule.game_date):
        frame = bronze_storage.read_bronze_day(
            "injuries_raw",
            data_root,
            schedule.season,
            day,
            include_runs=False,
            prefer_history=True,
        )
        if not frame.empty:
            injuries_frames.append(frame)
    injuries_raw = (
        pd.concat(injuries_frames, ignore_index=True) if injuries_frames else pd.DataFrame()
    )
    injuries_raw = (
        injuries_raw[injuries_raw.get("game_id") == schedule.game_id].copy()
        if not injuries_raw.empty
        else injuries_raw
    )

    roster_history = _load_roster_history(data_root, schedule.season, schedule.game_date)
    roster_history = (
        roster_history[roster_history.get("game_id") == schedule.game_id].copy()
        if not roster_history.empty
        else roster_history
    )

    injuries_at_ts = (
        select_latest_before(
            injuries_raw,
            snapshot_ts,
            group_cols=["game_id", "player_id"],
            as_of_col="as_of_ts",
            ingested_col="ingested_ts",
        )
        if not injuries_raw.empty
        else injuries_raw
    )
    odds_at_ts = (
        select_latest_before(
            odds_raw,
            snapshot_ts,
            group_cols=["game_id"],
            as_of_col="as_of_ts",
            ingested_col="ingested_ts",
        )
        if not odds_raw.empty
        else odds_raw
    )
    roster_at_ts = (
        select_latest_before(
            roster_history,
            snapshot_ts,
            group_cols=["game_id", "player_id"],
            as_of_col="as_of_ts",
            ingested_col="ingested_ts",
        )
        if not roster_history.empty
        else roster_history
    )
    roster_cutoff_fallback_used = False
    roster_effective = roster_at_ts
    if roster_effective.empty and not roster_history.empty:
        roster_at_tip = select_latest_before(
            roster_history,
            tip_ts,
            group_cols=["game_id", "player_id"],
            as_of_col="as_of_ts",
            ingested_col="ingested_ts",
        )
        if not roster_at_tip.empty:
            roster_effective = roster_at_tip
            roster_cutoff_fallback_used = True

    labels = _load_labels_history(data_root, schedule.season)
    target_day = pd.Timestamp(schedule.game_date).normalize()
    history = labels[labels["game_date"] < target_day].copy()
    if history_days is not None and history_days > 0:
        cutoff = target_day - pd.Timedelta(days=history_days)
        history = history[history["game_date"] >= cutoff].copy()
    if "minutes" in history.columns:
        history = history.dropna(subset=["minutes"])
    if history.empty and require_history:
        raise RuntimeError(f"No historical label rows found before {target_day.date()}; cannot build features.")

    season_label = _season_label(schedule.season)
    live_labels_source = "roster_nightly"
    if roster_at_ts.empty and roster_effective is not roster_at_ts:
        live_labels = _build_live_labels_for_game(
            roster_effective,
            game_id=schedule.game_id,
            game_date=target_day,
            season_label=season_label,
            frozen_at=frozen_at,
        )
        live_labels_source = "roster_nightly_tip_fallback"
    elif roster_at_ts.empty:
        labels_for_game = labels[
            (labels["game_id"] == schedule.game_id) & (labels["game_date"] == target_day)
        ].copy()
        live_labels = _build_live_labels_from_boxscore_labels(
            labels_for_game,
            game_id=schedule.game_id,
            game_date=target_day,
            season_label=season_label,
            frozen_at=frozen_at,
        )
        live_labels_source = "boxscore_labels"
    else:
        live_labels = _build_live_labels_for_game(
            roster_at_ts,
            game_id=schedule.game_id,
            game_date=target_day,
            season_label=season_label,
            frozen_at=frozen_at,
        )
    combined_labels = (
        pd.concat([history, live_labels], ignore_index=True, sort=False) if not history.empty else live_labels
    )
    needed_ids = (
        pd.to_numeric(combined_labels["game_id"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    schedule_for_builder = _load_schedule_for_builder(data_root, schedule.season, needed_ids)
    if schedule_for_builder.empty:
        raise RuntimeError("Schedule slice for feature builder is empty after filtering by game_ids.")

    roles_path = data_root / "gold" / "minutes_roles" / f"season={schedule.season}" / "roles.parquet"
    archetype_path = (
        data_root
        / "gold"
        / "features_minutes_v1"
        / f"season={schedule.season}"
        / "archetype_deltas.parquet"
    )
    roles_df = pd.read_parquet(roles_path) if roles_path.exists() else None
    archetype_deltas_df = pd.read_parquet(archetype_path) if archetype_path.exists() else None

    builder = MinutesFeatureBuilder(
        schedule=schedule_for_builder,
        injuries_snapshot=injuries_at_ts,
        odds_snapshot=odds_at_ts,
        roster_nightly=roster_effective,
        archetype_roles=roles_df,
        archetype_deltas=archetype_deltas_df,
    )
    features = builder.build(combined_labels)
    frozen_features = features[features.get("game_id") == schedule.game_id].copy()
    if frozen_features.empty:
        raise RuntimeError(f"Feature builder produced zero rows for game_id={schedule.game_id}")

    frozen_features = enforce_schema(frozen_features, FEATURES_MINUTES_V1_SCHEMA, allow_missing_optional=True)
    validate_with_pandera(frozen_features, FEATURES_MINUTES_V1_SCHEMA)

    frozen_features["snapshot_type"] = snapshot_type_norm
    frozen_features["snapshot_ts"] = pd.to_datetime(snapshot_ts, utc=True)
    frozen_features["frozen_at"] = frozen_at
    frozen_features = enforce_schema(
        frozen_features, SLATE_FEATURES_MINUTES_V1_SCHEMA, allow_missing_optional=True
    )
    validate_with_pandera(frozen_features, SLATE_FEATURES_MINUTES_V1_SCHEMA)

    _atomic_write_parquet(frozen_features, parquet_path)
    manifest = {
        "game_id": schedule.game_id,
        "season": schedule.season,
        "game_date": schedule.game_date.isoformat(),
        "tip_ts": tip_ts.isoformat(),
        "snapshot_type": snapshot_type_norm,
        "snapshot_ts": pd.Timestamp(snapshot_ts).isoformat(),
        "frozen_at": frozen_at.isoformat(),
        "git_sha": _git_rev_parse_head(),
        "row_count": int(len(frozen_features)),
        "inputs": {
            "history_required": bool(require_history),
            "history_label_rows": int(len(history)),
            "live_labels_source": live_labels_source,
            "live_label_rows": int(len(live_labels)),
            "injuries_raw_days_read": [
                (schedule.game_date - timedelta(days=1)).isoformat(),
                schedule.game_date.isoformat(),
            ],
            "injuries_raw_max_as_of_ts": _max_ts(injuries_at_ts, "as_of_ts"),
            "injuries_raw_max_ingested_ts": _max_ts(injuries_at_ts, "ingested_ts"),
            "odds_raw_max_as_of_ts": _max_ts(odds_at_ts, "as_of_ts"),
            "odds_raw_max_ingested_ts": _max_ts(odds_at_ts, "ingested_ts"),
            "roster_cutoff_fallback_used": bool(roster_cutoff_fallback_used),
            "roster_rows": int(len(roster_effective)),
            "roster_max_as_of_ts": _max_ts(roster_effective, "as_of_ts"),
            "roster_max_ingested_ts": _max_ts(roster_effective, "ingested_ts"),
        },
    }
    _atomic_write_json(manifest_path, manifest)
    return parquet_path, manifest_path


@app.command()
def freeze(
    game_id: int = typer.Option(..., help="Game ID to freeze."),
    snapshot_type: str = typer.Option(..., help="'lock' or 'pretip'."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Base data directory (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    out_root: Path | None = typer.Option(
        None,
        "--out-root",
        help="Optional override for gold/slates root (defaults to <data_root>/gold/slates).",
    ),
    season: int | None = typer.Option(None, "--season", help="Optional schedule season partition override."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing gold snapshot."),
    history_days: int | None = typer.Option(
        None, "--history-days", min=1, help="Optional rolling history window (in days) for label context."
    ),
) -> None:
    """Create an immutable gold snapshot for a single game."""
    data_root = data_root.resolve()
    out_root = (out_root or (data_root / "gold" / "slates")).resolve()
    schedule_row = _load_schedule_row(data_root, game_id, season=season)
    parquet_path, manifest_path = _freeze_game_snapshot(
        schedule_row,
        snapshot_type=snapshot_type,
        data_root=data_root,
        out_root=out_root,
        force=force,
        history_days=history_days,
    )
    typer.echo(f"[slates] wrote {parquet_path}")
    typer.echo(f"[slates] wrote {manifest_path}")


@app.command("freeze-pending")
def freeze_pending(
    snapshot_type: str = typer.Option(
        "both",
        "--snapshot-type",
        help="Which snapshot(s) to freeze: 'lock', 'pretip', or 'both'.",
    ),
    lock_lookahead_minutes: int = typer.Option(20, "--lock-lookahead-minutes", min=1),
    pretip_lookahead_minutes: int = typer.Option(5, "--pretip-lookahead-minutes", min=1),
    season: int | None = typer.Option(None, "--season", help="Optional season partition override."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Base data directory (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    out_root: Path | None = typer.Option(
        None,
        "--out-root",
        help="Optional override for gold/slates root (defaults to <data_root>/gold/slates).",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing gold snapshots."),
    history_days: int | None = typer.Option(
        None, "--history-days", min=1, help="Optional rolling history window (in days) for label context."
    ),
) -> None:
    """Freeze lock/pretip slates that are due within the configured lookahead windows."""
    data_root = data_root.resolve()
    out_root = (out_root or (data_root / "gold" / "slates")).resolve()

    now = pd.Timestamp.now(tz="UTC")
    now_et = now.tz_convert(ET_TZ)
    season_value = season or (now_et.year if now_et.month >= 8 else now_et.year - 1)

    schedule_root = data_root / "silver" / "schedule" / f"season={season_value}"
    schedule_df = _read_parquet_tree(schedule_root)
    if schedule_df.empty:
        raise FileNotFoundError(f"Missing schedule partitions under {schedule_root}")
    schedule_df["game_id"] = pd.to_numeric(schedule_df["game_id"], errors="coerce").astype("Int64")
    schedule_df["tip_ts"] = pd.to_datetime(schedule_df["tip_ts"], utc=True, errors="coerce")
    if "game_date" in schedule_df.columns:
        schedule_df["game_date"] = pd.to_datetime(schedule_df["game_date"], errors="coerce").dt.normalize()

    requested = snapshot_type.strip().lower()
    requested_types = {"lock", "pretip"} if requested == "both" else {requested}
    if not requested_types.issubset({"lock", "pretip"}):
        raise typer.BadParameter("--snapshot-type must be 'lock', 'pretip', or 'both'.")

    frozen = 0
    skipped = 0
    for snap in ("lock", "pretip"):
        if snap not in requested_types:
            continue
        lookahead = lock_lookahead_minutes if snap == "lock" else pretip_lookahead_minutes
        window_start = now - pd.Timedelta(minutes=lookahead)
        window_end = now

        tip_ts = schedule_df["tip_ts"]
        snapshot_ts = tip_ts - pd.Timedelta(minutes=15) if snap == "lock" else tip_ts
        mask = snapshot_ts.notna() & (snapshot_ts >= window_start) & (snapshot_ts <= window_end)
        candidates = schedule_df.loc[mask].copy()
        candidates = candidates.dropna(subset=["game_id", "tip_ts"])
        if candidates.empty:
            continue

        for row in candidates.to_dict("records"):
            game_id_value = int(row["game_id"])
            try:
                tip_ts_value = pd.to_datetime(row.get("tip_ts"), utc=True, errors="coerce")
                if pd.isna(tip_ts_value):
                    raise RuntimeError(f"Schedule row for game_id={game_id_value} missing tip_ts")
                game_date_value = pd.to_datetime(row.get("game_date"), errors="coerce")
                if pd.isna(game_date_value):
                    game_date_value = (
                        tip_ts_value.tz_convert(ET_TZ).tz_localize(None).normalize()
                    )
                else:
                    if getattr(game_date_value, "tzinfo", None) is not None:
                        game_date_value = game_date_value.tz_convert(ET_TZ).tz_localize(None)
                    game_date_value = game_date_value.normalize()
                schedule_row = ScheduleRow(
                    season=season_value,
                    game_id=game_id_value,
                    game_date=game_date_value.date(),
                    tip_ts=pd.Timestamp(tip_ts_value),
                    schedule_row=row,
                )
                _freeze_game_snapshot(
                    schedule_row,
                    snapshot_type=snap,
                    data_root=data_root,
                    out_root=out_root,
                    force=force,
                    history_days=history_days,
                )
            except RuntimeError as exc:
                msg = str(exc)
                if "already exists" in msg and not force:
                    skipped += 1
                    continue
                raise
            else:
                frozen += 1

    typer.echo(f"[slates] frozen={frozen} skipped_existing={skipped}")


if __name__ == "__main__":  # pragma: no cover
    app()
