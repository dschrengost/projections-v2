"""Backfill common production datasets for a given date window.

This is intended for incident recovery after the pipeline was down.

Defaults:
- backfill boxscores bronze for each day
- backfill gamerotation bronze for each day
- refresh gold labels + rates training base for the whole window

This script runs synchronously (no Prefect server required). It calls the same
Prefect flow functions used by deployments so we reuse the existing ops logic.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd


def _parse_date(value: str) -> date:
    out = pd.Timestamp(value).date()
    return out


def _date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError(f"end_date {end} < start_date {start}")
    cur = start
    out: list[date] = []
    while cur <= end:
        out.append(cur)
        cur = cur + timedelta(days=1)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill projections-v2 datasets for a date window.")
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--skip-boxscores", action="store_true", help="Skip boxscores ETL.")
    p.add_argument("--skip-gamerotation", action="store_true", help="Skip gamerotation scrape.")
    p.add_argument("--skip-labels", action="store_true", help="Skip minutes labels refresh.")
    p.add_argument("--skip-rates-training-base", action="store_true", help="Skip rates training base refresh.")
    p.add_argument("--dry-run", action="store_true", help="Print what would run without executing.")
    args = p.parse_args()

    start = _parse_date(args.start_date)
    end = _parse_date(args.end_date)
    days = _date_range(start, end)

    print(f"[backfill] window start={start.isoformat()} end={end.isoformat()} days={len(days)}")

    if not args.skip_boxscores:
        from prefect_flows.boxscores_etl import boxscores_etl_flow

        for d in days:
            iso = d.isoformat()
            print(f"[backfill] boxscores-etl game_date={iso}")
            if not args.dry_run:
                boxscores_etl_flow(game_date=iso)

    if not args.skip_gamerotation:
        from prefect_flows.gamerotation_scrape import gamerotation_scrape_flow

        for d in days:
            iso = d.isoformat()
            print(f"[backfill] gamerotation-scrape game_date={iso}")
            if not args.dry_run:
                gamerotation_scrape_flow(game_date=iso, overwrite=False)

    if not args.skip_labels:
        from prefect_flows.minutes_labels_refresh import minutes_labels_refresh_flow

        print(f"[backfill] minutes-labels-refresh start_date={start.isoformat()} end_date={end.isoformat()}")
        if not args.dry_run:
            minutes_labels_refresh_flow(start_date=start.isoformat(), end_date=end.isoformat())

    if not args.skip_rates_training_base:
        from prefect_flows.rates_training_base_refresh import rates_training_base_refresh_flow

        print(
            f"[backfill] rates-training-base-refresh start_date={start.isoformat()} end_date={end.isoformat()}"
        )
        if not args.dry_run:
            rates_training_base_refresh_flow(start_date=start.isoformat(), end_date=end.isoformat())

    print("[backfill] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

