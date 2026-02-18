"""Build normalized Action Network player-props snapshots from bronze JSON."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import typer

from projections import paths
from projections.features.action_props import (
    build_action_props_feature_snapshots,
    load_action_props_long_from_bronze,
)

app = typer.Typer(help=__doc__)


def _normalize_day(value: datetime) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


@app.command()
def main(
    start_date: datetime = typer.Option(..., "--start-date", help="Start date (YYYY-MM-DD)."),
    end_date: datetime = typer.Option(..., "--end-date", help="End date (YYYY-MM-DD)."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Data root (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    bronze_dir: Path | None = typer.Option(
        None,
        "--bronze-dir",
        help="Optional override for bronze Action props directory.",
    ),
    out_root: Path | None = typer.Option(
        None,
        "--out-root",
        help="Optional override for silver output root.",
    ),
) -> None:
    """Build day-partitioned silver snapshots from Action Network bronze files."""

    start_day = _normalize_day(start_date)
    end_day = _normalize_day(end_date)
    if start_day > end_day:
        raise typer.BadParameter("--start-date must be <= --end-date")

    props_dir = bronze_dir or (data_root / "bronze" / "action_network" / "props")
    output_root = out_root or (data_root / "silver" / "action_network_props")
    output_root.mkdir(parents=True, exist_ok=True)

    days = pd.date_range(start_day, end_day, freq="D")
    total_long = 0
    total_snapshots = 0
    written_days = 0

    for day in days:
        long_df = load_action_props_long_from_bronze(props_dir=props_dir, game_date=day)
        if long_df.empty:
            continue
        snapshot_df = build_action_props_feature_snapshots(long_df)
        if snapshot_df.empty:
            continue

        day_dir = output_root / f"date={day.date().isoformat()}"
        day_dir.mkdir(parents=True, exist_ok=True)

        long_path = day_dir / "action_props_long.parquet"
        snap_path = day_dir / "action_props_features.parquet"
        summary_path = day_dir / "summary.json"

        long_df.to_parquet(long_path, index=False)
        snapshot_df.to_parquet(snap_path, index=False)
        summary = {
            "date": day.date().isoformat(),
            "long_rows": int(len(long_df)),
            "feature_rows": int(len(snapshot_df)),
            "players_with_props": int(snapshot_df["an_has_any_props"].sum()),
            "generated_at": datetime.utcnow().isoformat(),
            "source_dir": str(props_dir),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        total_long += len(long_df)
        total_snapshots += len(snapshot_df)
        written_days += 1
        typer.echo(
            f"[action-props] {day.date().isoformat()} -> {len(long_df):,} long rows, "
            f"{len(snapshot_df):,} feature rows"
        )

    typer.echo(
        f"[action-props] wrote {written_days} day(s); "
        f"{total_long:,} long rows; {total_snapshots:,} feature rows"
    )


if __name__ == "__main__":  # pragma: no cover
    app()

