"""Command line entry-points for running training pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console

from projections import data, features, paths, utils
from .models import classical, deep

console = Console()
app = typer.Typer(help=__doc__)


@app.command()
def classical_minutes(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a YAML config file (defaults to config/settings.yaml).",
    ),
    raw_filename: str = typer.Option(
        "nba_minutes.csv", help="Raw CSV filename stored inside data/raw."
    ),
) -> None:
    """Train a classical (XGBoost) model using the configured pipeline."""

    cfg = utils.load_yaml_config(config)
    utils.set_seeds(int(cfg.get("seed", 42)))
    data_paths = utils.resolve_data_paths(cfg)

    # New logic: Load Gold Features (Parquet)
    features_root = paths.data_path("gold", "features_minutes_v1")
    if not features_root.exists():
        console.print(f"[red]Features root not found at {features_root}. Run backfill_features first![/red]")
        raise typer.Exit(code=1)

    # Load all available seasons
    files = sorted(features_root.rglob("*.parquet"))
    if not files:
        console.print(f"[red]No feature parquet files found in {features_root}.[/red]")
        raise typer.Exit(code=1)
    
    console.print(f"Loading features from {len(files)} parquet files...")
    feat_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    console.print(f"Loaded {len(feat_df)} rows.")

    # Define metadata columns to drop (non-features)
    # We keep 'minutes' as target, but drop it from X later
    metadata_cols = [
        "game_id", "player_id", "team_id", "player_name", "team_name", "team_tricode",
        "season", "game_date", "tip_ts", "home_team_id", "away_team_id", 
        "opponent_team_id", "opponent_team_name", "opponent_team_tricode",
        "injury_as_of_ts", "odds_as_of_ts", "roster_as_of_ts", "lineup_timestamp", 
        "feature_as_of_ts", "archetype", "pos_bucket", "lineup_role", 
        "lineup_status", "lineup_roster_status", "status", "home_flag",
        "injury_snapshot_missing", "is_confirmed_starter", # maybe keep is_projected_starter?
        "available_B", "available_G", "available_W" # keep these? they are features.
    ]
    # Refine drop list: keep numeric/boolean features. 
    # The schema has many columns. Let's drop the obvious metadata.
    drop_cols = [
        "game_id", "player_id", "team_id", "player_name", "team_name", "team_tricode",
        "season", "game_date", "tip_ts", "home_team_id", "away_team_id",
        "opponent_team_id", "opponent_team_name", "opponent_team_tricode",
        "injury_as_of_ts", "odds_as_of_ts", "roster_as_of_ts", "lineup_timestamp",
        "feature_as_of_ts", "archetype", "lineup_role", "lineup_status", 
        "lineup_roster_status", "status"
    ]
    
    # Handle categorical columns if needed (e.g. pos_bucket). 
    # For now, let's drop pos_bucket if we aren't encoding it, or keep it if XGBoost handles it.
    # XGBoost can handle categoricals but needs config. Let's drop for simplicity or assume numeric.
    # pos_bucket is string. Let's drop it for this first pass or one-hot it?
    # The original pipeline didn't use it. Let's drop it to be safe.
    drop_cols.append("pos_bucket")

    # Drop rows where target is missing (e.g. future games)
    target_col = cfg.get("targets", {}).get("minutes_col", "minutes")
    initial_len = len(feat_df)
    feat_df = feat_df.dropna(subset=[target_col])
    dropped_len = initial_len - len(feat_df)
    if dropped_len > 0:
        console.print(f"Dropped {dropped_len} rows with missing target '{target_col}'.")

    X, y = features.build_feature_target_split(
        feat_df,
        target_col=target_col,
        drop_cols=drop_cols,
    )
    
    # Ensure X only has numeric columns (basic safeguard)
    X = X.select_dtypes(include=["number", "bool"])

    X_train, X_valid, y_train, y_valid = features.stratified_split(
        X,
        y,
        test_size=float(cfg.get("training", {}).get("validation_size", 0.2)),
        random_state=int(cfg.get("seed", 42)),
    )
    result = classical.train_lightgbm_model(
        X_train,
        y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        params=cfg.get("training", {}).get("lightgbm", {}),
    )

    if result.metrics:
        console.print(f"Validation metrics: {result.metrics}")
    console.print("Model training complete.")


@app.command()
def deep_minutes(
    input_size: int = typer.Option(..., help="Number of input features per timestep."),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Optional config override."
    ),
) -> None:
    """Example CLI for kicking off deep learning experiments."""

    cfg = utils.load_yaml_config(config)
    utils.set_seeds(int(cfg.get("seed", 42)))
    trainer_cfg = cfg.get("training", {}).get("deep", {})

    model = deep.LSTMMinutesPredictor(
        input_size=input_size,
        hidden_size=int(trainer_cfg.get("hidden_size", 64)),
        num_layers=int(trainer_cfg.get("num_layers", 1)),
        dropout=float(trainer_cfg.get("dropout", 0.0)),
    )

    console.print(
        "Deep learning placeholder invoked. "
        "Hook up actual dataloaders to train_lstm_model when ready."
    )


def main() -> None:
    """Entry-point for `python -m projections.train`."""

    app()


if __name__ == "__main__":
    main()
