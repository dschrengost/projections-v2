"""Fetch NBA schedule slices (daily or monthly) and write silver parquets."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

from projections import paths
from projections.etl.common import month_slug, normalize_schedule_frame, schedule_from_api
from projections.pipeline.status import JobStatus, write_status

app = typer.Typer(help=__doc__)


def _status_target(start_day: pd.Timestamp, end_day: pd.Timestamp) -> str:
    if start_day == end_day:
        return start_day.date().isoformat()
    return f"{start_day.date()}_{end_day.date()}"


def _nan_rate(df: pd.DataFrame, cols: list[str]) -> float | None:
    present = [col for col in cols if col in df.columns]
    if not present or df.empty:
        return 0.0
    return float(df[present].isna().mean().mean())


def _normalize_day(value: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _write_month_partition(
    df: pd.DataFrame,
    *,
    season: int,
    month: int,
    data_root: Path,
) -> Path:
    dest = (
        data_root
        / "silver"
        / "schedule"
        / f"season={season}"
        / f"month={month:02d}"
    )
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "schedule.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


@app.command()
def main(
    start: datetime = typer.Option(..., "--start", help="Inclusive start date (YYYY-MM-DD)."),
    end: datetime = typer.Option(..., "--end", help="Inclusive end date (YYYY-MM-DD)."),
    season: int = typer.Option(..., "--season", help="Season partition label (e.g., 2025)."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Base data directory (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help=(
            "Optional explicit parquet path. When omitted, writes month partitions under "
            "<data_root>/silver/schedule/season=YYYY/month=MM/."
        ),
    ),
    timeout: float = typer.Option(10.0, "--timeout", help="NBA schedule API timeout in seconds."),
) -> None:
    """Download a date range from the NBA schedule API and persist it to silver."""

    start_day = _normalize_day(start)
    end_day = _normalize_day(end)
    if end_day < start_day:
        raise typer.BadParameter("--end must be on/after --start.")

    target_date = _status_target(start_day, end_day)
    run_ts = datetime.now(timezone.utc).isoformat()

    rows_written = 0
    try:
        typer.echo(f"[schedule] Fetching games between {start_day.date()} and {end_day.date()}...")
        raw = schedule_from_api(start_day, end_day, timeout)
        normalized = normalize_schedule_frame(raw)
        mask = (normalized["game_date"] >= start_day) & (normalized["game_date"] <= end_day)
        filtered = normalized.loc[mask].copy()
        if filtered.empty:
            raise RuntimeError("NBA schedule API returned zero games for requested window.")

        filtered = filtered.drop_duplicates(subset="game_id", keep="last")
        filtered.sort_values("tip_ts", inplace=True)
        filtered.reset_index(drop=True, inplace=True)
        rows_written = len(filtered)

        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            filtered.to_parquet(out, index=False)
            typer.echo(f"[schedule] wrote {len(filtered):,} rows -> {out}")
        else:
            grouped = filtered.groupby(filtered["game_date"].dt.to_period("M"))
            total = 0
            for period, frame in grouped:
                month_value = period.month
                out_path = _write_month_partition(frame, season=season, month=month_value, data_root=data_root)
                total += len(frame)
                typer.echo(
                    f"[schedule] month={period.strftime('%Y-%m')} wrote {len(frame):,} rows -> {out_path}"
                )
            typer.echo(f"[schedule] completed {len(grouped)} partition(s), {total:,} total rows.")

        write_status(
            JobStatus(
                job_name="schedule_live",
                stage="silver",
                target_date=target_date,
                run_ts=run_ts,
                status="success",
                rows_written=rows_written,
                expected_rows=rows_written,
                nan_rate_key_cols=_nan_rate(filtered, ["game_id", "tip_ts"]),
            )
        )
    except Exception as exc:  # noqa: BLE001
        write_status(
            JobStatus(
                job_name="schedule_live",
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
