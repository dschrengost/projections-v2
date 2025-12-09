#!/usr/bin/env python3
"""
Backfill injury data from NBA PDF reports using the existing NBAInjuryScraper.

This script:
1. For each date, gets game tip times from box scores
2. For each tip time, fetches the injury PDF from ~1 hour before
3. Writes the injury data to bronze layer with correct as_of_ts

Usage:
    python -m scripts.backfill_injuries_v2 --date 2025-11-12
    python -m scripts.backfill_injuries_v2 --start 2025-11-01 --end 2025-11-30 --no-dry-run
"""
import json
import typer
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd
from rich.console import Console
from rich.progress import track

# Import the existing scraper
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from scrapers.nba_injuries import NBAInjuryScraper, InjuryRecord, ET_TZ

console = Console()
app = typer.Typer()

# Player name to ID mapping cache
_player_name_map: dict[str, int] = {}


def load_player_name_map(data_root: Path) -> dict[str, int]:
    """Load player name to ID mapping from roster data."""
    global _player_name_map
    if _player_name_map:
        return _player_name_map
    
    # Load from roster snapshots
    roster_root = data_root / "silver" / "roster_nightly"
    for month_dir in roster_root.glob("season=2025/month=*/roster.parquet"):
        try:
            df = pd.read_parquet(month_dir)
            for _, row in df.iterrows():
                name = str(row.get('player_name', '')).lower().strip()
                pid = row.get('player_id')
                if name and pid:
                    _player_name_map[name] = int(pid)
        except Exception:
            continue
    
    console.print(f"  [dim]Loaded {len(_player_name_map)} player name mappings[/dim]")
    return _player_name_map


def normalize_name(name: str) -> str:
    """Normalize player name for matching."""
    # Handle "Last, First" format
    if ',' in name:
        parts = name.split(',', 1)
        name = f"{parts[1].strip()} {parts[0].strip()}"
    return name.lower().strip()


def get_tip_times_from_boxscores(date_str: str, data_root: Path) -> list[datetime]:
    """Get tip times for all games on a date from box scores."""
    box_path = data_root / "bronze" / "boxscores_raw" / "season=2025" / f"date={date_str}" / "boxscores_raw.parquet"
    if not box_path.exists():
        return []
    
    tip_times = []
    df = pd.read_parquet(box_path)
    for _, row in df.iterrows():
        payload = json.loads(row['payload']) if isinstance(row['payload'], str) else row['payload']
        tip_str = payload.get('game_time_utc')
        if tip_str:
            tip = pd.to_datetime(tip_str, utc=True)
            tip_times.append(tip.to_pydatetime())
    
    return sorted(set(tip_times))


def records_to_bronze_df(records: list[InjuryRecord], as_of_ts: datetime, player_map: dict[str, int]) -> pd.DataFrame:
    """Convert InjuryRecords to bronze DataFrame format."""
    rows = []
    for rec in records:
        # Normalize name and look up player ID
        name = normalize_name(rec.player_name)
        player_id = player_map.get(name)
        
        if player_id is None:
            # Try partial matches
            for stored_name, pid in player_map.items():
                if name in stored_name or stored_name in name:
                    player_id = pid
                    break
        
        if player_id is None:
            continue  # Skip if we can't find the player
        
        # Map status
        status = rec.current_status.upper() if rec.current_status else ''
        if 'OUT' in status:
            status = 'OUT'
        elif 'QUESTIONABLE' in status:
            status = 'GTD'  # Game time decision
        elif 'DOUBTFUL' in status:
            status = 'GTD'
        elif 'PROBABLE' in status:
            status = 'AVAIL'
        elif 'AVAILABLE' in status:
            status = 'AVAIL'
        else:
            status = 'AVAIL'
        
        rows.append({
            'player_id': player_id,
            'status': status,
            'reason': rec.reason,
            'game_date': rec.game_date,
            'as_of_ts': as_of_ts,
        })
    
    return pd.DataFrame(rows)


def update_bronze_injuries(
    date_str: str,
    new_data: pd.DataFrame,
    data_root: Path,
    dry_run: bool,
) -> int:
    """Merge new injury data into bronze layer."""
    bronze_path = data_root / "bronze" / "injuries_raw" / "season=2025" / f"date={date_str}" / "injuries.parquet"
    
    if bronze_path.exists():
        existing = pd.read_parquet(bronze_path)
        existing["as_of_ts"] = pd.to_datetime(existing["as_of_ts"], utc=True)
    else:
        existing = pd.DataFrame()
    
    if len(new_data) == 0:
        return 0
    
    # Combine existing and new data
    if len(existing) > 0:
        combined = pd.concat([existing, new_data], ignore_index=True)
        # Keep unique player/as_of_ts combinations, preferring new data
        combined = combined.drop_duplicates(subset=['player_id', 'as_of_ts'], keep='last')
    else:
        combined = new_data
    
    if not dry_run:
        bronze_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(bronze_path, index=False)
    
    return len(new_data)


@app.command()
def backfill(
    date_str: str = typer.Option(None, "--date", "-d", help="Single date (YYYY-MM-DD)"),
    start: str = typer.Option(None, "--start", "-s", help="Start date"),
    end: str = typer.Option(None, "--end", "-e", help="End date"),
    data_root: Path = typer.Option(Path("/home/daniel/projections-data"), "--data-root"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Don't modify data"),
):
    """Backfill injury data from official NBA PDFs."""
    
    if date_str:
        dates = [date_str]
    elif start and end:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current.isoformat())
            current += timedelta(days=1)
    else:
        console.print("[red]Must specify --date or --start/--end[/red]")
        return
    
    console.print(f"[bold]Backfilling {len(dates)} days{'(DRY RUN)' if dry_run else ''}[/bold]\n")
    
    # Load player name mapping
    player_map = load_player_name_map(data_root)
    
    total_records = 0
    
    with NBAInjuryScraper() as scraper:
        for d in track(dates, description="Processing dates..."):
            # Get tip times
            tip_times = get_tip_times_from_boxscores(d, data_root)
            if not tip_times:
                continue
            
            # Get latest tip time (use report from 1 hour before)
            latest_tip = max(tip_times)
            report_time = latest_tip - timedelta(hours=1)
            
            try:
                records = scraper.fetch_report(report_time)
                if isinstance(records, pd.DataFrame):
                    continue
                
                # Convert to bronze format
                new_df = records_to_bronze_df(records, latest_tip, player_map)
                
                # Update bronze
                count = update_bronze_injuries(d, new_df, data_root, dry_run)
                total_records += count
                
            except Exception as e:
                console.print(f"  [yellow]{d}: Error - {e}[/yellow]")
                continue
    
    console.print(f"\n[bold green]Done! Added {total_records} injury records[/bold green]")
    if dry_run:
        console.print("[yellow]This was a dry run - no data was modified[/yellow]")
        console.print("Run with --no-dry-run to actually update the data")


if __name__ == "__main__":
    app()
