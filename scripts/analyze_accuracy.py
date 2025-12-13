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
    """Load actual contest ownership for a date.
    
    Reads from ownership_by_slate/ and picks the main slate (largest player pool).
    """
    # Try new path first: ownership_by_slate/{date}_{slate_id}.parquet
    slate_dir = data_root / "bronze" / "dk_contests" / "ownership_by_slate"
    if slate_dir.exists():
        # Find all slates for this date
        slate_files = list(slate_dir.glob(f"{date_str}_*.parquet"))
        if slate_files:
            # Pick the largest slate (main slate has most players)
            best_file = None
            best_size = 0
            for f in slate_files:
                try:
                    df = pd.read_parquet(f)
                    if len(df) > best_size:
                        best_size = len(df)
                        best_file = f
                except Exception:
                    continue
            
            if best_file:
                df = pd.read_parquet(best_file)
                df['player_name_lower'] = df['Player'].str.lower().str.strip()
                return df
    
    # Fallback to legacy path: ownership_by_date/{date}.parquet
    own_path = data_root / "bronze" / "dk_contests" / "ownership_by_date" / f"{date_str}.parquet"
    if not own_path.exists():
        return pd.DataFrame()
    
    df = pd.read_parquet(own_path)
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
    
    # Compute errors
    merged['own_error'] = merged['own_pct'] - merged['pred_own_pct']
    merged['own_abs_error'] = np.abs(merged['own_error'])
    
    # MAE
    own_mae = merged['own_abs_error'].mean()
    
    # SMAPE (Symmetric Mean Absolute Percentage Error)
    # SMAPE = |actual - pred| / ((|actual| + |pred|) / 2) * 100
    # Handles zero values better than MAPE
    denominator = (np.abs(merged['own_pct']) + np.abs(merged['pred_own_pct'])) / 2
    # Avoid division by zero (when both are 0, which shouldn't happen much for ownership)
    valid_mask = denominator > 0.1  # Minimum threshold to avoid blow-ups
    if valid_mask.sum() > 0:
        smape = (merged.loc[valid_mask, 'own_abs_error'] / denominator[valid_mask]).mean() * 100
    else:
        smape = np.nan
    
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
    high_own_mae = high_own['own_abs_error'].mean() if len(high_own) > 0 else np.nan
    
    # Top 5 ownership misses (largest absolute errors)
    top_misses = merged.nlargest(5, 'own_abs_error')[['Player', 'own_pct', 'pred_own_pct', 'own_error']].copy()
    top_misses_list = []
    for _, row in top_misses.iterrows():
        top_misses_list.append({
            'player': row['Player'],
            'actual': round(row['own_pct'], 1),
            'pred': round(row['pred_own_pct'], 1),
            'error': round(row['own_error'], 1),
        })
    
    return {
        'own_players_matched': len(merged),
        'own_mae': round(own_mae, 2),
        'own_smape': round(smape, 1) if not np.isnan(smape) else None,
        'own_corr': round(corr, 3) if not np.isnan(corr) else None,
        'chalk_top5_acc': round(chalk_accuracy, 2),
        'own_bias': round(own_bias, 2),
        'high_own_mae': round(high_own_mae, 2) if not np.isnan(high_own_mae) else None,
        'own_top_misses': top_misses_list,
    }


def compute_metrics(actuals: pd.DataFrame, predictions: pd.DataFrame, min_minutes: int = 5) -> dict:
    """Compute accuracy metrics for predictions vs actuals.
    
    Includes:
    - Core metrics: FPTS MAE, minutes MAE, coverage, bias
    - Salary tier accuracy: elite/mid/value/punt
    - Starter vs bench accuracy
    - Calibration: p25-p75, p10-p90, p05-p95 vs expected
    - Edge cases: DNP false positives, starter misses
    """
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
    min_pred_threshold = min_minutes * 0.5
    played_5plus = merged[merged['actual_minutes'] >= min_minutes]
    missed = played_5plus[played_5plus['minutes_mean'] < min_pred_threshold] if 'minutes_mean' in merged.columns else pd.DataFrame()
    
    high_pred = merged[merged['minutes_mean'] >= min_minutes] if 'minutes_mean' in merged.columns else pd.DataFrame()
    false_preds = high_pred[high_pred['actual_minutes'] < min_minutes]
    
    # Filter to players who played meaningful minutes
    played = merged[merged['actual_minutes'] >= min_minutes].copy()
    
    if len(played) == 0:
        return {'players_matched': 0, 'missed': len(missed), 'false_preds': len(false_preds)}
    
    # Compute error columns for reuse
    if 'dk_fpts_mean' in played.columns:
        played['fpts_error'] = played['actual_dk_fpts'] - played['dk_fpts_mean']
    if 'minutes_mean' in played.columns:
        played['mins_error'] = played['actual_minutes'] - played['minutes_mean']
    
    # === Core Metrics ===
    mins_mae = np.abs(played['mins_error']).mean() if 'mins_error' in played.columns else np.nan
    fpts_mae = np.abs(played['fpts_error']).mean() if 'fpts_error' in played.columns else np.nan
    fpts_bias = played['fpts_error'].mean() * -1 if 'fpts_error' in played.columns else np.nan  # Negative because error = actual - pred
    
    # === Calibration Metrics ===
    calibration = {}
    intervals = [
        ('50', 'dk_fpts_p25', 'dk_fpts_p75', 0.50),
        ('80', 'dk_fpts_p10', 'dk_fpts_p90', 0.80),
        ('90', 'dk_fpts_p05', 'dk_fpts_p95', 0.90),
    ]
    for name, lo_col, hi_col, expected in intervals:
        if lo_col in played.columns and hi_col in played.columns:
            in_range = (played['actual_dk_fpts'] >= played[lo_col]) & (played['actual_dk_fpts'] <= played[hi_col])
            observed = in_range.mean()
            calibration[f'coverage_{name}'] = round(observed, 3)
            calibration[f'cal_gap_{name}'] = round(observed - expected, 3)
        else:
            calibration[f'coverage_{name}'] = None
            calibration[f'cal_gap_{name}'] = None
    
    # === Salary Tier Accuracy ===
    salary_tiers = {}
    if 'salary' in played.columns and 'fpts_error' in played.columns:
        tiers = {
            'elite': (8000, 15000),
            'mid': (5500, 7999),
            'value': (3500, 5499),
            'punt': (3000, 3499),
        }
        for tier_name, (low, high) in tiers.items():
            tier_df = played[(played['salary'] >= low) & (played['salary'] <= high)]
            if len(tier_df) > 0:
                salary_tiers[f'fpts_mae_{tier_name}'] = round(np.abs(tier_df['fpts_error']).mean(), 2)
                salary_tiers[f'n_{tier_name}'] = len(tier_df)
            else:
                salary_tiers[f'fpts_mae_{tier_name}'] = None
                salary_tiers[f'n_{tier_name}'] = 0
    
    # === Starter vs Bench Accuracy ===
    starter_metrics = {}
    starter_col = 'is_starter' if 'is_starter' in played.columns else None
    if starter_col is None and 'starter_flag' in played.columns:
        starter_col = 'starter_flag'
    
    if starter_col and 'fpts_error' in played.columns:
        starters = played[played[starter_col] == True]
        bench = played[played[starter_col] == False]
        if len(starters) > 0:
            starter_metrics['fpts_mae_starters'] = round(np.abs(starters['fpts_error']).mean(), 2)
            starter_metrics['n_starters'] = len(starters)
        if len(bench) > 0:
            starter_metrics['fpts_mae_bench'] = round(np.abs(bench['fpts_error']).mean(), 2)
            starter_metrics['n_bench'] = len(bench)
    
    # === Edge Case Detection ===
    edge_cases = {}
    
    # DNP false positives: predicted > 10 min but played 0
    if 'minutes_mean' in merged.columns:
        dnp_fp = merged[(merged['actual_minutes'] == 0) & (merged['minutes_mean'] >= 10)]
        edge_cases['dnp_false_positives'] = len(dnp_fp)
        if len(dnp_fp) > 0:
            name_col = 'player_name_actual' if 'player_name_actual' in dnp_fp.columns else 'player_name'
            edge_cases['dnp_fp_names'] = dnp_fp[name_col].tolist()[:5]  # Top 5 for debugging
    
    # Starter misses: played > 30 min but predicted < 20
    if 'minutes_mean' in merged.columns:
        starter_misses = merged[(merged['actual_minutes'] >= 30) & (merged['minutes_mean'] < 20)]
        edge_cases['starter_misses'] = len(starter_misses)
        if len(starter_misses) > 0:
            name_col = 'player_name_actual' if 'player_name_actual' in starter_misses.columns else 'player_name'
            edge_cases['starter_miss_names'] = starter_misses[name_col].tolist()[:5]
    
    # Blowup misses: actual FPTS > 50 but predicted < 30
    if 'dk_fpts_mean' in merged.columns:
        blowup_misses = merged[(merged['actual_dk_fpts'] >= 50) & (merged['dk_fpts_mean'] < 30)]
        edge_cases['blowup_misses'] = len(blowup_misses)
    
    # === Positional Accuracy ===
    pos_metrics = {}
    if 'pos_bucket' in played.columns and 'fpts_error' in played.columns:
        for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
            pos_df = played[played['pos_bucket'] == pos]
            if len(pos_df) > 0:
                pos_metrics[f'fpts_mae_{pos}'] = round(np.abs(pos_df['fpts_error']).mean(), 2)
                pos_metrics[f'n_{pos}'] = len(pos_df)
    
    # === Combine all metrics ===
    result = {
        # Core
        'players_matched': len(played),
        'minutes_mae': round(mins_mae, 2) if not np.isnan(mins_mae) else None,
        'fpts_mae': round(fpts_mae, 2) if not np.isnan(fpts_mae) else None,
        'bias': round(fpts_bias, 2) if not np.isnan(fpts_bias) else None,
        'missed': len(missed),
        'false_preds': len(false_preds),
    }
    
    # Add calibration (backward compatible: coverage_80, coverage_90)
    result['coverage_80'] = calibration.get('coverage_80')
    result['coverage_90'] = calibration.get('coverage_90')
    result['coverage_50'] = calibration.get('coverage_50')
    result['cal_gap_50'] = calibration.get('cal_gap_50')
    result['cal_gap_80'] = calibration.get('cal_gap_80')
    result['cal_gap_90'] = calibration.get('cal_gap_90')
    
    # Add salary tiers
    result.update(salary_tiers)
    
    # Add starter/bench
    result.update(starter_metrics)
    
    # Add edge cases
    result.update(edge_cases)
    
    # Add positional (these can be large, so only include counts)
    result.update(pos_metrics)
    
    return result


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
