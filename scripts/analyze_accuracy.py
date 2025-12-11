#!/usr/bin/env python3
"""
Analyze prediction accuracy by comparing sim projections vs actual box scores.

Usage:
    python -m scripts.analyze_accuracy --date 2025-12-03
    python -m scripts.analyze_accuracy --start 2025-12-01 --end 2025-12-05
"""
import json
import typer
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer()

DK_FPTS_WEIGHTS = {
    'points': 1.0,
    'reboundsTotal': 1.25,
    'assists': 1.5,
    'steals': 2.0,
    'blocks': 2.0,
    'turnovers': -0.5,
    # Bonus: +1.5 for double-double, +3 for triple-double (computed separately)
}


def parse_boxscores(boxscore_path: Path, game_date: str) -> pd.DataFrame:
    """Parse bronze boxscore parquet into flat player-level DataFrame."""
    if not boxscore_path.exists():
        return pd.DataFrame()
    
    df = pd.read_parquet(boxscore_path)
    rows = []
    
    for _, row in df.iterrows():
        try:
            payload = json.loads(row['payload']) if isinstance(row['payload'], str) else row['payload']
        except (json.JSONDecodeError, TypeError):
            continue
            
        game_id = payload.get('game_id')
        
        for side in ['home', 'away']:
            team_data = payload.get(side, {})
            team_id = team_data.get('team_id')
            
            for player in team_data.get('players', []):
                stats = player.get('statistics', {})
                minutes_str = stats.get('minutes', 'PT0M0S')
                
                # Parse minutes from ISO duration format (PT12M30S)
                try:
                    if isinstance(minutes_str, str) and minutes_str.startswith('PT'):
                        mins = 0
                        if 'M' in minutes_str:
                            mins_part = minutes_str.split('PT')[1].split('M')[0]
                            mins = int(mins_part)
                        if 'S' in minutes_str:
                            secs_part = minutes_str.split('M')[-1].replace('S', '')
                            if secs_part:
                                mins += int(float(secs_part)) / 60
                        minutes = mins
                    else:
                        minutes = float(minutes_str) if minutes_str else 0
                except (ValueError, AttributeError):
                    minutes = 0
                
                pts = stats.get('points', 0) or 0
                reb = stats.get('reboundsTotal', 0) or 0
                ast = stats.get('assists', 0) or 0
                stl = stats.get('steals', 0) or 0
                blk = stats.get('blocks', 0) or 0
                tov = stats.get('turnovers', 0) or 0
                
                # Calculate DK FPTS
                dk_fpts = (
                    pts * 1.0 +
                    reb * 1.25 +
                    ast * 1.5 +
                    stl * 2.0 +
                    blk * 2.0 +
                    tov * -0.5
                )
                
                # Double-double / triple-double bonuses
                stats_10 = sum(1 for x in [pts, reb, ast, stl, blk] if x >= 10)
                if stats_10 >= 3:
                    dk_fpts += 3.0  # Triple-double
                elif stats_10 >= 2:
                    dk_fpts += 1.5  # Double-double
                
                rows.append({
                    'game_date': game_date,
                    'game_id': game_id,
                    'player_id': player.get('person_id'),
                    'player_name': f"{player.get('first_name', '')} {player.get('family_name', '')}".strip(),
                    'team_id': team_id,
                    'actual_minutes': minutes,
                    'actual_pts': pts,
                    'actual_reb': reb,
                    'actual_ast': ast,
                    'actual_stl': stl,
                    'actual_blk': blk,
                    'actual_tov': tov,
                    'actual_dk_fpts': dk_fpts,
                })
    
    return pd.DataFrame(rows)


def load_predictions(date_str: str, data_root: Path) -> pd.DataFrame:
    """Load sim projections for a date, merging from best pre-tip runs for each game.
    
    The live pipeline filters out games that have already tipped, so the "latest" run
    may only have the late game(s). This function finds the best (latest) run that
    includes each game and merges them together.
    """
    # First, try the unified projections artifact (includes minutes + sim)
    # Check both possible locations
    proj_dir = data_root / "artifacts" / "projections" / date_str
    if not proj_dir.exists():
        # Try the sim_v2 path as fallback
        proj_path = data_root / "artifacts" / "sim_v2" / "projections" / f"game_date={date_str}" / "projections.parquet"
        if proj_path.exists():
            return pd.read_parquet(proj_path)
        return pd.DataFrame()
    
    # Get all runs for this date
    runs = sorted(proj_dir.glob("run=*/projections.parquet"))
    if not runs:
        return pd.DataFrame()
    
    # Build a map of game_id -> best (latest) run that has it
    game_to_best_run: dict[int, Path] = {}
    
    for run_path in runs:
        try:
            df = pd.read_parquet(run_path)
            if 'game_id' not in df.columns:
                continue
            for gid in df['game_id'].unique():
                # Later run is better (sorted ascending, so last wins)
                game_to_best_run[int(gid)] = run_path
        except Exception:
            continue
    
    if not game_to_best_run:
        return pd.DataFrame()
    
    # Deduplicate: group by run path to minimize reads
    runs_to_games: dict[Path, list[int]] = {}
    for gid, run_path in game_to_best_run.items():
        if run_path not in runs_to_games:
            runs_to_games[run_path] = []
        runs_to_games[run_path].append(gid)
    
    # Load and merge
    all_dfs = []
    for run_path, game_ids in runs_to_games.items():
        try:
            df = pd.read_parquet(run_path)
            # Filter to only the games we want from this run
            df = df[df['game_id'].isin(game_ids)]
            all_dfs.append(df)
        except Exception:
            continue
    
    if not all_dfs:
        return pd.DataFrame()
    
    merged = pd.concat(all_dfs, ignore_index=True)
    # Drop duplicates (shouldn't happen, but just in case)
    if 'player_id' in merged.columns and 'game_id' in merged.columns:
        merged = merged.drop_duplicates(subset=['player_id', 'game_id'], keep='last')
    
    return merged


def load_actual_ownership(date_str: str, data_root: Path) -> pd.DataFrame:
    """Load actual contest ownership for a date."""
    own_path = data_root / "bronze" / "dk_contests" / "ownership_by_date" / f"{date_str}.parquet"
    if not own_path.exists():
        return pd.DataFrame()
    
    df = pd.read_parquet(own_path)
    # Normalize name for joining
    df['player_name_lower'] = df['Player'].str.lower().str.strip()
    return df


def load_ownership_predictions(date_str: str, data_root: Path) -> pd.DataFrame:
    """Load ownership predictions for a date."""
    pred_path = data_root / "silver" / "ownership_predictions" / f"{date_str}.parquet"
    if not pred_path.exists():
        return pd.DataFrame()
    
    df = pd.read_parquet(pred_path)
    # Normalize name for joining
    df['player_name_lower'] = df['player_name'].str.lower().str.strip()
    return df


def compute_ownership_metrics(actual: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Compute ownership prediction accuracy metrics."""
    if actual.empty or predictions.empty:
        return {}
    
    # Merge on normalized player name
    merged = actual.merge(
        predictions[['player_name_lower', 'pred_own_pct', 'salary']],
        on='player_name_lower',
        how='inner'
    )
    
    if len(merged) == 0:
        return {'own_players_matched': 0}
    
    # MAE
    own_mae = np.abs(merged['own_pct'] - merged['pred_own_pct']).mean()
    
    # Chalk accuracy (identify top 5 owned correctly)
    actual_top5 = set(merged.nlargest(5, 'own_pct')['player_name_lower'])
    pred_top5 = set(merged.nlargest(5, 'pred_own_pct')['player_name_lower'])
    chalk_accuracy = len(actual_top5 & pred_top5) / 5
    
    # Correlation
    corr = merged['own_pct'].corr(merged['pred_own_pct'])
    
    # Bias (predicted - actual)
    own_bias = merged['pred_own_pct'].mean() - merged['own_pct'].mean()
    
    # High ownership accuracy (>20% actual)
    high_own = merged[merged['own_pct'] >= 20]
    high_own_mae = np.abs(high_own['own_pct'] - high_own['pred_own_pct']).mean() if len(high_own) > 0 else np.nan
    
    return {
        'own_players_matched': len(merged),
        'own_mae': round(own_mae, 2),
        'own_corr': round(corr, 3) if not np.isnan(corr) else None,
        'chalk_top5_acc': round(chalk_accuracy, 2),
        'own_bias': round(own_bias, 2),
        'high_own_mae': round(high_own_mae, 2) if not np.isnan(high_own_mae) else None,
    }


def compute_metrics(actuals: pd.DataFrame, predictions: pd.DataFrame, min_minutes: int = 5) -> dict:
    """Compute accuracy metrics for predictions vs actuals."""
    # Ensure player_id is same type for merge
    actuals = actuals.copy()
    predictions = predictions.copy()
    actuals['player_id'] = actuals['player_id'].astype(str)
    predictions['player_id'] = predictions['player_id'].astype(str)
    
    # === Merge for accuracy metrics ===
    merged = actuals.merge(predictions, on='player_id', how='inner', suffixes=('_actual', '_pred'))
    
    if len(merged) == 0:
        return {'players_matched': 0}
    
    # === Roster Accuracy (based on merged predictions only) ===
    # Missed: players who played 5+ min but we predicted < threshold
    min_pred_threshold = min_minutes * 0.5  # More lenient threshold (2.5 min for 5 min cutoff)
    played_5plus = merged[merged['actual_minutes'] >= min_minutes]
    missed = played_5plus[played_5plus['minutes_mean'] < min_pred_threshold] if 'minutes_mean' in merged.columns else pd.DataFrame()
    
    # False predictions: we predicted 5+ min but they played < min threshold
    high_pred = merged[merged['minutes_mean'] >= min_minutes] if 'minutes_mean' in merged.columns else pd.DataFrame()
    false_preds = high_pred[high_pred['actual_minutes'] < min_minutes]
    
    # Filter to players who played meaningful minutes
    played = merged[merged['actual_minutes'] >= min_minutes].copy()
    
    if len(played) == 0:
        return {'players_matched': 0, 'missed': len(missed), 'false_preds': len(false_preds)}
    
    # Minutes metrics
    mins_mae = np.abs(played['actual_minutes'] - played['minutes_mean']).mean() if 'minutes_mean' in played.columns else np.nan
    
    # FPTS metrics
    fpts_mae = np.abs(played['actual_dk_fpts'] - played['dk_fpts_mean']).mean() if 'dk_fpts_mean' in played.columns else np.nan
    
    # Coverage (% of actuals within p10-p90)
    if 'dk_fpts_p10' in played.columns and 'dk_fpts_p90' in played.columns:
        in_range = (played['actual_dk_fpts'] >= played['dk_fpts_p10']) & (played['actual_dk_fpts'] <= played['dk_fpts_p90'])
        coverage_80 = in_range.mean()
    else:
        coverage_80 = np.nan
    
    # p05-p95 coverage
    if 'dk_fpts_p05' in played.columns and 'dk_fpts_p95' in played.columns:
        in_range_90 = (played['actual_dk_fpts'] >= played['dk_fpts_p05']) & (played['actual_dk_fpts'] <= played['dk_fpts_p95'])
        coverage_90 = in_range_90.mean()
    else:
        coverage_90 = np.nan
    
    # Bias (mean prediction - mean actual)
    fpts_bias = (played['dk_fpts_mean'].mean() - played['actual_dk_fpts'].mean()) if 'dk_fpts_mean' in played.columns else np.nan
    
    return {
        'players_matched': len(played),
        'minutes_mae': round(mins_mae, 2) if not np.isnan(mins_mae) else None,
        'fpts_mae': round(fpts_mae, 2) if not np.isnan(fpts_mae) else None,
        'coverage_80': round(coverage_80, 3) if not np.isnan(coverage_80) else None,
        'coverage_90': round(coverage_90, 3) if not np.isnan(coverage_90) else None,
        'bias': round(fpts_bias, 2) if not np.isnan(fpts_bias) else None,
        'missed': len(missed),
        'false_preds': len(false_preds),
    }


@app.command()
def analyze(
    date_str: str = typer.Option(None, "--date", "-d", help="Single date to analyze (YYYY-MM-DD)"),
    start: str = typer.Option(None, "--start", "-s", help="Start date for range"),
    end: str = typer.Option(None, "--end", "-e", help="End date for range"),
    data_root: Path = typer.Option(Path("/home/daniel/projections-data"), "--data-root"),
    output: Path = typer.Option(None, "--output", "-o", help="Output JSON path"),
):
    """Analyze prediction accuracy vs actuals."""
    
    # Determine dates to analyze
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
        # Default to yesterday
        dates = [(date.today() - timedelta(days=1)).isoformat()]
    
    all_results = []
    
    for d in dates:
        console.print(f"[blue]Analyzing {d}...[/blue]")
        
        # Load actuals
        boxscore_path = data_root / "bronze" / "boxscores_raw" / "season=2025" / f"date={d}" / "boxscores_raw.parquet"
        actuals = parse_boxscores(boxscore_path, d)
        
        if len(actuals) == 0:
            console.print(f"  [yellow]No box scores found for {d}[/yellow]")
            continue
        
        # Load predictions
        predictions = load_predictions(d, data_root)
        
        if len(predictions) == 0:
            console.print(f"  [yellow]No predictions found for {d}[/yellow]")
            continue
        
        # Compute FPTS metrics
        metrics = compute_metrics(actuals, predictions)
        metrics['date'] = d
        metrics['total_players_actual'] = len(actuals)
        metrics['total_players_pred'] = len(predictions)
        
        # Compute ownership metrics
        actual_own = load_actual_ownership(d, data_root)
        pred_own = load_ownership_predictions(d, data_root)
        if not actual_own.empty and not pred_own.empty:
            own_metrics = compute_ownership_metrics(actual_own, pred_own)
            metrics.update(own_metrics)
            if own_metrics.get('own_mae'):
                console.print(f"  Ownership MAE: {own_metrics['own_mae']}, Corr: {own_metrics.get('own_corr', '-')}")
        
        all_results.append(metrics)
        
        console.print(f"  Players: {metrics['players_matched']}/{len(actuals)} matched")
        if metrics.get('fpts_mae'):
            console.print(f"  FPTS MAE: {metrics['fpts_mae']}")
        if metrics.get('coverage_80'):
            console.print(f"  Coverage (p10-p90): {metrics['coverage_80']:.1%}")
    
    # Summary table
    if all_results:
        console.print("\n[bold]Summary[/bold]")
        table = Table()
        table.add_column("Date")
        table.add_column("Players")
        table.add_column("Min MAE")
        table.add_column("FPTS MAE")
        table.add_column("Cov 80%")
        table.add_column("Cov 90%")
        table.add_column("Bias")
        table.add_column("Missed")
        table.add_column("F.Pred")
        
        for r in all_results:
            table.add_row(
                r['date'],
                str(r['players_matched']),
                str(r.get('minutes_mae', '-')),
                str(r.get('fpts_mae', '-')),
                f"{r['coverage_80']:.1%}" if r.get('coverage_80') else '-',
                f"{r['coverage_90']:.1%}" if r.get('coverage_90') else '-',
                str(r.get('bias', '-')),
                str(r.get('missed', '-')),
                str(r.get('false_preds', '-')),
            )
        
        console.print(table)
        
        # Overall averages
        avg_fpts_mae = np.mean([r['fpts_mae'] for r in all_results if r.get('fpts_mae')])
        avg_coverage = np.mean([r['coverage_80'] for r in all_results if r.get('coverage_80')])
        total_missed = sum(r.get('missed', 0) for r in all_results)
        total_false = sum(r.get('false_preds', 0) for r in all_results)
        console.print(f"\n[bold]Overall: FPTS MAE = {avg_fpts_mae:.2f}, Coverage = {avg_coverage:.1%}, Missed = {total_missed}, False Preds = {total_false}[/bold]")
        
        # === Ownership Summary ===
        own_results = [r for r in all_results if r.get('own_mae') is not None]
        if own_results:
            console.print("\n[bold]Ownership Summary[/bold]")
            own_table = Table()
            own_table.add_column("Date")
            own_table.add_column("Players")
            own_table.add_column("Own MAE")
            own_table.add_column("Corr")
            own_table.add_column("Top5 Acc")
            own_table.add_column("Bias")
            own_table.add_column("Hi Own MAE")
            
            for r in own_results:
                own_table.add_row(
                    r['date'],
                    str(r.get('own_players_matched', '-')),
                    str(r.get('own_mae', '-')),
                    str(r.get('own_corr', '-')),
                    f"{r['chalk_top5_acc']:.0%}" if r.get('chalk_top5_acc') is not None else '-',
                    str(r.get('own_bias', '-')),
                    str(r.get('high_own_mae', '-')),
                )
            console.print(own_table)
            
            # Overall ownership averages
            avg_own_mae = np.mean([r['own_mae'] for r in own_results if r.get('own_mae') is not None])
            avg_own_corr = np.mean([r['own_corr'] for r in own_results if r.get('own_corr') is not None])
            avg_chalk = np.mean([r['chalk_top5_acc'] for r in own_results if r.get('chalk_top5_acc') is not None])
            console.print(f"\n[bold]Ownership Overall: MAE = {avg_own_mae:.2f}, Corr = {avg_own_corr:.3f}, Top5 Acc = {avg_chalk:.0%}[/bold]")
        
        if output:
            # Merge with existing data to persist historical results
            existing_data = []
            if output.exists():
                try:
                    existing_data = json.loads(output.read_text())
                    if not isinstance(existing_data, list):
                        existing_data = []
                except (json.JSONDecodeError, OSError):
                    existing_data = []
            
            # Build map of date -> metrics for deduplication
            data_by_date = {r['date']: r for r in existing_data}
            
            # Update with new results (overwrites existing for same date)
            for r in all_results:
                data_by_date[r['date']] = r
            
            # Sort by date and keep last 30 days
            final_data = sorted(data_by_date.values(), key=lambda x: x['date'])[-30:]
            
            with open(output, 'w') as f:
                json.dump(final_data, f, indent=2)
            console.print(f"\n[green]Wrote {len(final_data)} day(s) of results to {output}[/green]")


if __name__ == "__main__":
    app()
