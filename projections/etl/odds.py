"""Ingest Oddstrader odds snapshots into bronze/silver parquet outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import typer
from zoneinfo import ZoneInfo

from projections import paths
from projections.etl import storage
from projections.etl.common import load_schedule_data
from projections.minutes_v1.schemas import (
    ODDS_RAW_SCHEMA,
    ODDS_SNAPSHOT_SCHEMA,
    enforce_schema,
    validate_with_pandera,
)
from projections.minutes_v1.snapshots import latest_pre_tip_snapshot
from projections.minutes_v1.season_dataset import TeamResolver
from projections.pipeline.status import JobStatus, write_status
from scrapers.oddstrader import EventOdds, OddstraderScraper

app = typer.Typer(help=__doc__)
ODDSTRADER_TZ = ZoneInfo("America/New_York")
ODDS_SNAPSHOT_COLS: tuple[str, ...] = ODDS_SNAPSHOT_SCHEMA.columns


def _status_target(start_day: pd.Timestamp, end_day: pd.Timestamp) -> str:
    if start_day == end_day:
        return start_day.date().isoformat()
    return f"{start_day.date()}_{end_day.date()}"


def _nan_rate(df: pd.DataFrame, cols: list[str]) -> float | None:
    present = [col for col in cols if col in df.columns]
    if not present or df.empty:
        return 0.0
    return float(df[present].isna().mean().mean())


def _iter_dates(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    current = start
    while current <= end:
        yield current
        current += pd.Timedelta(days=1)


def _build_odds_raw(
    events: List[EventOdds],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    resolver: TeamResolver,
) -> pd.DataFrame:
    ingested_ts = pd.Timestamp.now(tz="UTC")
    records: list[dict] = []
    for event in events:
        scheduled = pd.Timestamp(event.scheduled)
        if scheduled.tzinfo is None:
            scheduled = scheduled.tz_localize("UTC")
        else:
            scheduled = scheduled.tz_convert("UTC")
        event_day = scheduled.tz_convert(ODDSTRADER_TZ).tz_localize(None).normalize()
        if not (start <= event_day <= end):
            continue
        home_tri = resolver.resolve_tricode(event.home_team)
        away_tri = resolver.resolve_tricode(event.away_team)
        if not (home_tri and away_tri):
            continue
        game_id = resolver.lookup_game_id(
            event_day.strftime("%Y-%m-%d"), away_tri, home_tri, tip_ts=scheduled
        )
        if not game_id:
            continue
        markets = event.markets or {}
        spread = markets.get("spread", {})
        total = markets.get("total", {})
        home_spread = spread.get("home")
        over_total = total.get("over")
        timestamps: list[pd.Timestamp] = []
        for line in (home_spread, over_total):
            if line is None or line.updated_at is None:
                continue
            ts = pd.Timestamp(line.updated_at)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            timestamps.append(ts)
        if not timestamps:
            continue
        as_of_ts = min(max(timestamps), scheduled)
        records.append(
            {
                "game_id": int(game_id),
                "home_team_id": resolver.resolve_team_id(event.home_team),
                "away_team_id": resolver.resolve_team_id(event.away_team),
                "spread_home": home_spread.point if home_spread else None,
                "total": over_total.point if over_total else None,
                "book": (home_spread or over_total).book if (home_spread or over_total) else None,
                "market": "spread_total",
                "as_of_ts": as_of_ts,
                "ingested_ts": ingested_ts,
                "source": "oddstrader",
            }
        )
    columns = list(ODDS_RAW_SCHEMA.columns)
    return pd.DataFrame(records, columns=columns)


def _build_odds_snapshot(odds_raw: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    if odds_raw.empty:
        return pd.DataFrame(columns=ODDS_SNAPSHOT_SCHEMA.columns)
    merged = odds_raw.merge(schedule_df[["game_id", "tip_ts"]], on="game_id", how="left")
    merged.loc[merged["as_of_ts"] > merged["tip_ts"], "as_of_ts"] = merged["tip_ts"]
    snapshot = latest_pre_tip_snapshot(
        merged,
        group_cols=["game_id"],
        tip_ts_col="tip_ts",
        as_of_col="as_of_ts",
    )
    snapshot["book_pref"] = snapshot["book"]
    return snapshot[
        ["game_id", "as_of_ts", "spread_home", "total", "book", "book_pref", "ingested_ts", "source"]
    ]


def _dedupe_snapshot_by_game(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    working = df.copy()
    working["game_id"] = pd.to_numeric(working["game_id"], errors="coerce")
    working["as_of_ts"] = pd.to_datetime(working.get("as_of_ts"), utc=True, errors="coerce")
    working["ingested_ts"] = pd.to_datetime(working.get("ingested_ts"), utc=True, errors="coerce")
    working = working.dropna(subset=["game_id"]).copy()
    if working.empty:
        return working
    working.sort_values(["as_of_ts", "ingested_ts"], inplace=True, na_position="first")
    return working.drop_duplicates(subset=["game_id"], keep="last")


def _merge_odds_snapshots(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    existing = _dedupe_snapshot_by_game(existing)
    incoming = _dedupe_snapshot_by_game(incoming)
    if existing.empty:
        return incoming
    if incoming.empty:
        return existing

    cols = [col for col in ODDS_SNAPSHOT_COLS if col != "game_id"]
    existing = existing.rename(columns={col: f"{col}_old" for col in cols})
    incoming = incoming.rename(columns={col: f"{col}_new" for col in cols})
    merged = existing.merge(incoming, on="game_id", how="outer")

    as_of_old = pd.to_datetime(merged.get("as_of_ts_old"), utc=True, errors="coerce")
    as_of_new = pd.to_datetime(merged.get("as_of_ts_new"), utc=True, errors="coerce")
    ing_old = pd.to_datetime(merged.get("ingested_ts_old"), utc=True, errors="coerce")
    ing_new = pd.to_datetime(merged.get("ingested_ts_new"), utc=True, errors="coerce")

    use_new = as_of_new.notna() & (
        as_of_old.isna()
        | (as_of_new > as_of_old)
        | ((as_of_new == as_of_old) & (ing_old.isna() | (ing_new >= ing_old)))
    )

    out: dict[str, pd.Series] = {"game_id": pd.to_numeric(merged["game_id"], errors="coerce")}
    for col in cols:
        new_col = merged.get(f"{col}_new")
        old_col = merged.get(f"{col}_old")
        chosen = new_col.where(use_new, old_col)
        fallback = old_col.where(use_new, new_col)
        out[col] = chosen.combine_first(fallback)

    frame = pd.DataFrame(out)
    frame = frame.dropna(subset=["game_id", "as_of_ts"])
    return frame


@app.command()
def main(
    start: datetime = typer.Option(..., help="Start date inclusive (YYYY-MM-DD)."),
    end: datetime = typer.Option(..., help="End date inclusive (YYYY-MM-DD)."),
    season: int = typer.Option(..., help="Season partition label (e.g., 2025)."),
    month: int = typer.Option(..., help="Month partition (1-12)."),
    schedule: List[str] = typer.Option(
        [],
        "--schedule",
        help="Optional schedule parquet glob(s). Falls back to live NBA API when omitted.",
    ),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        help="Base data directory (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    bronze_root: Path | None = typer.Option(
        None,
        "--bronze-root",
        help="Optional override for odds_raw bronze root (defaults to the standard contract).",
    ),
    bronze_out: Path | None = typer.Option(
        None,
        "--bronze-out",
        help="[deprecated] Write a single parquet instead of partitioned bronze outputs.",
    ),
    silver_out: Path | None = typer.Option(
        None,
        "--silver-out",
        help="Optional explicit output path for odds_snapshot parquet.",
    ),
    scraper_timeout: float = typer.Option(10.0, "--timeout", help="Oddstrader HTTP timeout (seconds)."),
    schedule_timeout: float = typer.Option(
        10.0,
        "--schedule-timeout",
        help="Timeout (seconds) for NBA schedule API fallback.",
    ),
) -> None:
    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end).normalize()
    if end_day < start_day:
        raise typer.BadParameter("--end must be on/after --start.")

    target_date = _status_target(start_day, end_day)
    run_dt = datetime.now(timezone.utc)
    run_ts = run_dt.isoformat()
    rows_written = 0
    try:
        schedule_df = load_schedule_data(schedule, start_day, end_day, schedule_timeout)
        resolver = TeamResolver(schedule_df)

        scraper = OddstraderScraper(timeout=scraper_timeout)
        events: list[EventOdds] = []
        for cursor in _iter_dates(start_day, end_day):
            events.extend(scraper.fetch_daily_odds(cursor.date()))

        odds_raw = _build_odds_raw(events, start=start_day, end=end_day, resolver=resolver)
        odds_snapshot = _build_odds_snapshot(odds_raw.copy(), schedule_df)

        odds_raw = enforce_schema(odds_raw, ODDS_RAW_SCHEMA)
        validate_with_pandera(odds_raw, ODDS_RAW_SCHEMA)
        odds_snapshot = enforce_schema(odds_snapshot, ODDS_SNAPSHOT_SCHEMA, allow_missing_optional=True)
        validate_with_pandera(odds_snapshot, ODDS_SNAPSHOT_SCHEMA)

        data_root = data_root.resolve()
        silver_default = (
            data_root
            / "silver"
            / "odds_snapshot"
            / f"season={season}"
            / f"month={month:02d}"
            / "odds_snapshot.parquet"
        )

        bronze_root_path = (bronze_root or storage.default_bronze_root("odds_raw", data_root)).resolve()
        if bronze_out:
            bronze_out.parent.mkdir(parents=True, exist_ok=True)
            odds_raw.to_parquet(bronze_out, index=False)
            typer.echo(
                f"[odds] wrote {len(odds_raw):,} raw rows -> {bronze_out} (legacy bronze_out path)."
            )
        else:
            if odds_raw.empty:
                typer.echo("[odds] no raw rows to persist for the requested window.")
            else:
                working = odds_raw.merge(
                    schedule_df.loc[:, ["game_id", "game_date"]],
                    on="game_id",
                    how="left",
                )
                working["game_date"] = pd.to_datetime(working["game_date"], errors="coerce").dt.date
                missing_dates = working["game_date"].isna().sum()
                if missing_dates:
                    typer.echo(
                        f"[odds] warning: {missing_dates} row(s) missing game_date after schedule merge; dropping.",
                        err=True,
                    )
                    working = working.dropna(subset=["game_date"]).copy()

                for game_date, day_frame in working.groupby("game_date"):
                    payload = day_frame.drop(columns=["game_date"])
                    history_result = storage.write_bronze_partition_run(
                        payload,
                        dataset="odds_raw",
                        data_root=data_root,
                        season=season,
                        target_date=game_date,
                        run_ts=run_dt,
                        bronze_root=bronze_root_path,
                    )
                    typer.echo(
                        f"[odds] bronze history date={game_date} run_ts={run_dt:%Y%m%dT%H%M%SZ}: "
                        f"{history_result.rows} rows -> {history_result.path}"
                    )

                    # Transitional: keep the legacy flat daily file updated as a "latest view".
                    latest_result = storage.write_bronze_partition(
                        payload,
                        dataset="odds_raw",
                        data_root=data_root,
                        season=season,
                        target_date=game_date,
                        bronze_root=bronze_root_path,
                    )
                    typer.echo(
                        f"[odds] bronze latest date={game_date}: "
                        f"{latest_result.rows} rows -> {latest_result.path}"
                    )

        silver_path = silver_out or silver_default
        silver_path.parent.mkdir(parents=True, exist_ok=True)
        merged_snapshot = odds_snapshot
        if silver_path.exists():
            try:
                existing_snapshot = pd.read_parquet(silver_path)
            except Exception as exc:  # noqa: BLE001
                typer.echo(f"[odds] warning: failed reading existing snapshot ({exc}); overwriting.", err=True)
            else:
                try:
                    existing_snapshot = enforce_schema(existing_snapshot, ODDS_SNAPSHOT_SCHEMA, allow_missing_optional=True)
                except Exception as exc:  # noqa: BLE001
                    typer.echo(f"[odds] warning: existing snapshot schema mismatch ({exc}); overwriting.", err=True)
                else:
                    merged_snapshot = _merge_odds_snapshots(existing_snapshot, odds_snapshot)
                    merged_snapshot = enforce_schema(
                        merged_snapshot, ODDS_SNAPSHOT_SCHEMA, allow_missing_optional=True
                    )
                    typer.echo(
                        f"[odds] merged with existing: {len(existing_snapshot)} existing + "
                        f"{len(merged_snapshot) - len(existing_snapshot)} new = {len(merged_snapshot)} total"
                    )

        merged_snapshot.to_parquet(silver_path, index=False)
        rows_written = len(merged_snapshot)

        typer.echo(f"[odds] wrote {len(merged_snapshot):,} snapshot rows -> {silver_path}")
        write_status(
            JobStatus(
                job_name="odds_live",
                stage="silver",
                target_date=target_date,
                run_ts=run_ts,
                status="success",
                rows_written=rows_written,
                expected_rows=None,
                nan_rate_key_cols=_nan_rate(merged_snapshot, ["game_id", "spread_home", "total"]),
            )
        )
    except Exception as exc:  # noqa: BLE001
        write_status(
            JobStatus(
                job_name="odds_live",
                stage="silver",
                target_date=target_date,
                run_ts=run_ts,
                status="error",
                rows_written=rows_written,
                expected_rows=None,
                message=str(exc),
            )
        )
        raise


if __name__ == "__main__":  # pragma: no cover
    app()
