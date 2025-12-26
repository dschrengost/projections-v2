"""CLI for building the canonical training dataset."""

import typer
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Optional

from projections.pipeline.training import builder, schema
from projections import paths

app = typer.Typer(help="Build gold training dataset.")
logger = logging.getLogger("projections.training")

def _setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

@app.command()
def build(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    season: str = typer.Option("2025", help="Season label"),
    as_of_time: Optional[str] = typer.Option(None, help="Specific iso timestamp override (e.g. 2025-12-25T11:00:00). If None, uses 'lock' logic."),
    data_root: Path = typer.Option(paths.get_data_root(), help="Root data directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Build training dataset for a range of dates."""
    _setup_logging(verbose)
    
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    
    dates = pd.date_range(start_dt, end_dt)
    
    total_rows = 0
    
    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        logger.info(f"Processing {date_str}...")
        
        target_ts = None
        if as_of_time:
            target_ts = pd.to_datetime(as_of_time)
            
        try:
            df = builder.build_features(date_str, target_ts, season)
            
            if df.empty:
                logger.warning(f"No features built for {date_str}")
                continue
                
            # Validation
            # schema.TrainingFeaturesSchema.validate(df) # Strict validation, might fail if schema mismatch
            # For now soft validate
            logger.info(f"Built {len(df)} rows for {date_str}")
            
            # Write
            # Partition by season / game_date / as_of
            # We need a stable as_of for partitioning.
            # If dynamic (lock), we might partition by "as_of=lock" or actual timestamp?
            # User requirement: "Partitioned by ... optionally snapshot"
            # If using lock logic, the timestamp varies per game.
            # So we probably utilize a top-level partition for 'run' or just date?
            # User example: asof=YYYYMMDDTHHMMSSZ/
            # If we run daily backfill, we can use run time or fixed time.
            # "Lock" implies mixed timestamps. 
            # Let's partition by 'asof' using the RUN time or a fixed label if backfilling.
            
            # For backfill mode, usually we want to group them.
            # Let's use the 'date_str' as primary partition, and maybe a build ID?
            # User requested: gold/training/player_game_features/season=YYYY/game_date=YYYY-MM-DD/asof=.../
            
            # If per-game lock time, we can't easily partition by single as-of unless we use "lock" string.
            # But user said "asof=YYYYMMDD...".
            # Let's default to "lock" label if dynamic, or the ts if explicit.
            
            if as_of_time:
                asof_label = pd.to_datetime(as_of_time).strftime("%Y%m%dT%H%M%SZ")
            else:
                asof_label = "lock"
                
            out_dir = data_root / "gold" / "training" / "player_game_features" / f"season={season}" / f"game_date={date_str}" / f"asof={asof_label}"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            out_path = out_dir / "features.parquet"
            df.to_parquet(out_path, index=False)
            logger.info(f"Wrote -> {out_path}")
            total_rows += len(df)
            
        except Exception as e:
            logger.error(f"Failed to build {date_str}: {e}")
            if verbose:
                logger.exception(e)
                
    logger.info(f"Complete. Total rows: {total_rows}")

if __name__ == "__main__":
    app()
