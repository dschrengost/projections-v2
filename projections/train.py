"""Command line entry-points for running training pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from . import data, features, utils
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

    raw_df = data.load_raw_minutes(data_paths, raw_filename)
    cleaned = data.clean_minutes(
        raw_df, columns=cfg.get("pipeline", {}).get("columns_to_keep")
    )
    interim_name = cfg.get("pipeline", {}).get("interim_filename", "clean_minutes.csv")
    data.write_interim(cleaned, data_paths, interim_name)

    rolling_windows = cfg.get("features", {}).get("rolling_windows", [3, 5, 10])
    feat_df = features.add_rolling_features(
        cleaned,
        group_cols=cfg.get("features", {}).get("group_by", ["player_id"]),
        target_col=cfg.get("targets", {}).get("minutes_col", "minutes"),
        windows=rolling_windows,
    )

    drop_cols = cfg.get("features", {}).get(
        "drop_columns",
        ["player_name", "team", "opponent", "game_date"],
    )
    X, y = features.build_feature_target_split(
        feat_df,
        target_col=cfg.get("targets", {}).get("minutes_col", "minutes"),
        drop_cols=drop_cols,
    )
    X_train, X_valid, y_train, y_valid = features.stratified_split(
        X,
        y,
        test_size=float(cfg.get("training", {}).get("validation_size", 0.2)),
        random_state=int(cfg.get("seed", 42)),
    )
    result = classical.train_xgboost_model(
        X_train,
        y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        params=cfg.get("training", {}).get("xgboost", {}),
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
