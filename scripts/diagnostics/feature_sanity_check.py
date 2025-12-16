"""
Feature Sanity Check - Daily diagnostic report for live features.

Generates a comprehensive report showing:
1. Feature summary statistics (min, max, mean, outliers)
2. Player-level feature inspection (top/bottom players by each key metric)
3. Comparison to historical baselines
4. Flagged anomalies (values outside expected ranges)

Usage:
    python -m scripts.diagnostics.feature_sanity_check --date 2025-12-15
    python -m scripts.diagnostics.feature_sanity_check --date 2025-12-15 --player-id 203999
"""

from __future__ import annotations

import json
from datetime import date, datetime, UTC
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from projections import paths

app = typer.Typer(help=__doc__)
console = Console()

# Expected ranges for key features (based on per-minute rates)
EXPECTED_RANGES = {
    # Season per-minute rates (should be small decimals, not game totals!)
    "season_fga_per_min": (0.1, 1.2),      # ~4-40 FGA per 36 min
    "season_3pa_per_min": (0.0, 0.6),      # 0-20 3PA per 36 min
    "season_fta_per_min": (0.0, 0.4),      # 0-14 FTA per 36 min
    "season_ast_per_min": (0.0, 0.5),      # 0-18 AST per 36 min
    "season_tov_per_min": (0.0, 0.2),      # 0-7 TOV per 36 min
    "season_reb_per_min": (0.0, 0.6),      # 0-22 REB per 36 min
    "season_stl_per_min": (0.0, 0.1),      # 0-3.5 STL per 36 min
    "season_blk_per_min": (0.0, 0.15),     # 0-5 BLK per 36 min
    # Minutes predictions
    "minutes_pred_p50": (0, 42),
    "minutes_pred_p10": (0, 35),
    "minutes_pred_p90": (0, 48),
    # Other
    "is_starter": (0, 1),
    "days_rest": (0, 3),
    "spread_close": (-25, 25),
    "total_close": (180, 260),
}

# Known elite players for reference
ELITE_PLAYERS = {
    203999: "Nikola Jokic",
    201142: "Kevin Durant",
    203507: "Giannis Antetokounmpo",
    1629029: "Luka Doncic",
    203954: "Joel Embiid",
    201566: "Russell Westbrook",
    201935: "James Harden",
    2544: "LeBron James",
    1628369: "Jayson Tatum",
    201950: "Jrue Holiday",
}


def load_latest_features(data_root: Path, feature_type: str, game_date: date) -> pd.DataFrame:
    """Load latest features for a given date."""
    feature_dir = data_root / "live" / f"features_{feature_type}" / game_date.isoformat()
    if not feature_dir.exists():
        raise FileNotFoundError(f"No features found at {feature_dir}")
    
    latest_pointer = feature_dir / "latest_run.json"
    if latest_pointer.exists():
        with open(latest_pointer) as f:
            run_id = json.load(f).get("run_id")
        run_dir = feature_dir / f"run={run_id}"
    else:
        runs = sorted(feature_dir.glob("run=*"))
        if not runs:
            raise FileNotFoundError(f"No run directories in {feature_dir}")
        run_dir = runs[-1]
    
    parquet_path = run_dir / "features.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"No features.parquet in {run_dir}")
    
    return pd.read_parquet(parquet_path)


def load_latest_rates(data_root: Path, game_date: date) -> pd.DataFrame:
    """Load latest rates predictions for a given date."""
    rates_dir = data_root / "gold" / "rates_v1_live" / game_date.isoformat()
    if not rates_dir.exists():
        return pd.DataFrame()
    
    runs = sorted(rates_dir.glob("run=*/rates.parquet"))
    if not runs:
        return pd.DataFrame()
    
    return pd.read_parquet(runs[-1])


def check_feature_ranges(df: pd.DataFrame) -> list[dict]:
    """Check features against expected ranges and flag anomalies."""
    anomalies = []
    for col, (min_val, max_val) in EXPECTED_RANGES.items():
        if col not in df.columns:
            continue
        
        below = df[df[col] < min_val]
        above = df[df[col] > max_val]
        
        if len(below) > 0:
            anomalies.append({
                "feature": col,
                "issue": "below_min",
                "expected_min": min_val,
                "count": len(below),
                "examples": below.nsmallest(3, col)[["player_id", col]].to_dict("records"),
            })
        
        if len(above) > 0:
            anomalies.append({
                "feature": col,
                "issue": "above_max", 
                "expected_max": max_val,
                "count": len(above),
                "examples": above.nlargest(3, col)[["player_id", col]].to_dict("records"),
            })
    
    return anomalies


def compute_fpts_estimate(rates: pd.DataFrame) -> pd.Series:
    """Compute estimated FPTS/min from rates predictions."""
    if rates.empty:
        return pd.Series(dtype=float)
    
    # DraftKings scoring
    pts_per_min = (
        2 * rates["pred_fga2_per_min"] * rates["pred_fg2_pct"] +
        3 * rates["pred_fga3_per_min"] * rates["pred_fg3_pct"] +
        rates["pred_fta_per_min"] * rates["pred_ft_pct"]
    )
    reb_per_min = rates["pred_oreb_per_min"] + rates["pred_dreb_per_min"]
    
    fpts_per_min = (
        pts_per_min * 1.0 +
        reb_per_min * 1.25 +
        rates["pred_ast_per_min"] * 1.5 +
        rates["pred_stl_per_min"] * 2.0 +
        rates["pred_blk_per_min"] * 2.0 +
        rates["pred_tov_per_min"] * -0.5
    )
    
    return fpts_per_min


def print_feature_summary(df: pd.DataFrame, title: str):
    """Print summary statistics for key features."""
    table = Table(title=title, show_header=True)
    table.add_column("Feature", style="cyan")
    table.add_column("Min", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Missing", justify="right")
    table.add_column("Status", justify="center")
    
    key_features = [
        "minutes_pred_p50", "is_starter",
        "season_fga_per_min", "season_ast_per_min", "season_reb_per_min",
        "season_stl_per_min", "season_blk_per_min", "season_tov_per_min",
    ]
    
    for feat in key_features:
        if feat not in df.columns:
            continue
        
        col = df[feat]
        min_val = col.min()
        mean_val = col.mean()
        max_val = col.max()
        missing = col.isna().sum()
        
        # Check if in expected range
        status = "✓"
        if feat in EXPECTED_RANGES:
            exp_min, exp_max = EXPECTED_RANGES[feat]
            if max_val > exp_max * 1.5 or min_val < exp_min * 0.5:
                status = "⚠"
        
        table.add_row(
            feat,
            f"{min_val:.4f}" if pd.notna(min_val) else "N/A",
            f"{mean_val:.4f}" if pd.notna(mean_val) else "N/A",
            f"{max_val:.4f}" if pd.notna(max_val) else "N/A",
            str(missing),
            status,
        )
    
    console.print(table)


def print_player_details(df: pd.DataFrame, rates: pd.DataFrame, player_id: Optional[int] = None):
    """Print detailed features for a specific player or top players."""
    
    if not rates.empty:
        rates["fpts_per_min"] = compute_fpts_estimate(rates)
        rates["fpts_36"] = rates["fpts_per_min"] * 36
        df = df.merge(rates[["player_id", "fpts_per_min", "fpts_36"]], on="player_id", how="left")
    
    if player_id:
        players = df[df["player_id"] == player_id]
        title = f"Player {player_id} Details"
    else:
        # Show elite players if available
        elite_ids = [pid for pid in ELITE_PLAYERS.keys() if pid in df["player_id"].values]
        if elite_ids:
            players = df[df["player_id"].isin(elite_ids)]
            title = "Elite Players"
        else:
            players = df.nlargest(10, "minutes_pred_p50")
            title = "Top 10 by Predicted Minutes"
    
    table = Table(title=title, show_header=True)
    table.add_column("Player ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Min P50", justify="right")
    table.add_column("Starter", justify="center")
    table.add_column("FGA/m", justify="right")
    table.add_column("AST/m", justify="right")
    table.add_column("REB/m", justify="right")
    table.add_column("FPTS/36", justify="right", style="green")
    
    for _, row in players.iterrows():
        pid = int(row["player_id"])
        name = ELITE_PLAYERS.get(pid, "-")
        
        table.add_row(
            str(pid),
            name,
            f"{row.get('minutes_pred_p50', 0):.1f}",
            "✓" if row.get("is_starter", 0) == 1 else "-",
            f"{row.get('season_fga_per_min', 0):.3f}",
            f"{row.get('season_ast_per_min', 0):.3f}",
            f"{row.get('season_reb_per_min', 0):.3f}",
            f"{row.get('fpts_36', 0):.1f}" if "fpts_36" in row else "-",
        )
    
    console.print(table)


def print_anomalies(anomalies: list[dict]):
    """Print detected anomalies."""
    if not anomalies:
        console.print("[green]✓ No anomalies detected - all features within expected ranges[/green]")
        return
    
    console.print(f"[red]⚠ {len(anomalies)} potential issues detected:[/red]")
    for anomaly in anomalies:
        feat = anomaly["feature"]
        issue = anomaly["issue"]
        count = anomaly["count"]
        
        if issue == "below_min":
            console.print(f"  • {feat}: {count} rows below expected min ({anomaly['expected_min']})")
        else:
            console.print(f"  • {feat}: {count} rows above expected max ({anomaly['expected_max']})")
        
        # Show examples
        for ex in anomaly["examples"][:2]:
            console.print(f"    - player_id={int(ex['player_id'])}: {feat}={ex[feat]:.4f}")


@app.command()
def main(
    date_value: datetime = typer.Option(..., "--date", help="Date for features (YYYY-MM-DD)"),
    player_id: Optional[int] = typer.Option(None, "--player-id", help="Specific player to inspect"),
    data_root: Optional[Path] = typer.Option(None, "--data-root", help="Data root path"),
    output_json: Optional[Path] = typer.Option(None, "--output-json", help="Save report as JSON"),
):
    """Run feature sanity checks for a given date."""
    
    game_date = date_value.date()
    root = data_root or paths.data_path()
    
    console.print(Panel(f"[bold]Feature Sanity Check[/bold]\nDate: {game_date}", style="blue"))
    
    # Load features
    try:
        rates_features = load_latest_features(root, "rates_v1", game_date)
        console.print(f"[green]✓ Loaded {len(rates_features)} rows from rates features[/green]")
    except FileNotFoundError as e:
        console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(1)
    
    # Load rates predictions
    rates = load_latest_rates(root, game_date)
    if not rates.empty:
        console.print(f"[green]✓ Loaded {len(rates)} rows from rates predictions[/green]")
    else:
        console.print("[yellow]⚠ No rates predictions found[/yellow]")
    
    console.print()
    
    # Feature summary
    print_feature_summary(rates_features, "Feature Summary (Rates Features)")
    console.print()
    
    # Check for anomalies
    anomalies = check_feature_ranges(rates_features)
    print_anomalies(anomalies)
    console.print()
    
    # Player details
    print_player_details(rates_features, rates, player_id)
    
    # Save JSON if requested
    if output_json:
        report = {
            "date": game_date.isoformat(),
            "generated_at": datetime.now(UTC).isoformat(),
            "row_count": len(rates_features),
            "anomalies": anomalies,
            "feature_stats": {
                col: {
                    "min": float(rates_features[col].min()) if pd.notna(rates_features[col].min()) else None,
                    "mean": float(rates_features[col].mean()) if pd.notna(rates_features[col].mean()) else None,
                    "max": float(rates_features[col].max()) if pd.notna(rates_features[col].max()) else None,
                }
                for col in EXPECTED_RANGES.keys() if col in rates_features.columns
            },
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2))
        console.print(f"\n[green]Report saved to {output_json}[/green]")


if __name__ == "__main__":
    app()
