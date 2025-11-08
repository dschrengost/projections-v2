# projections-v2

Maintainable scaffolding for an NBA player minutes prediction project. The layout follows data-science best practices (Cookiecutter-style) with clear separation between data storage, reusable Python modules, notebooks, configuration, and tests.

## Directory layout

```
.
├── config/          # YAML/TOML configs that drive experiments
├── data/
│   ├── raw/         # Immutable source data dumps
│   ├── external/    # Third-party data
│   ├── interim/     # Reusable cleaned datasets
│   └── processed/   # Feature matrices ready for modeling
├── models/          # Serialized model artifacts, predictions, logs
├── notebooks/       # Exploratory notebooks (EDA, prototyping)
├── projections/     # Python package containing reusable code
│   ├── data.py      # Raw/interim data handling utilities
│   ├── features.py  # Feature engineering helpers
│   ├── models/      # Classical + deep learning training code
│   ├── train.py     # Typer CLI entry-points for end-to-end runs
│   ├── evaluate.py  # Metric utilities
│   └── utils.py     # Shared helpers (config loading, seeds, paths)
├── tests/           # Pytest-based unit tests mirroring package layout
├── pyproject.toml   # Dependency + tool configuration (managed by uv)
└── uv.lock          # Generated lockfile (create via `uv lock`)
```

## Getting started

1. [Install uv](https://github.com/astral-sh/uv) if it is not already available.
2. Create the virtual environment and install dependencies:
   ```bash
   uv sync
   ```
3. Activate the environment (uv will print the correct command, typically `source .venv/bin/activate`).

## Running experiments

1. Place your raw CSV (e.g., `nba_minutes.csv`) in `data/raw/`.
2. Adjust `config/settings.yaml` as needed (columns, rolling windows, model hyperparameters).
3. Execute the classical training pipeline:
   ```bash
   uv run python -m projections.train classical-minutes --raw-filename nba_minutes.csv
   ```
   Behind the scenes this will:
   - load raw data and create a cleaned interim artifact,
   - build rolling-window features,
   - split the data into train/validation partitions, and
   - train an XGBoost regressor while reporting validation metrics.

For deep-learning experiments, wire up PyTorch dataloaders and call:
```bash
uv run python -m projections.train deep-minutes --input-size <feature_count>
```

## Testing

Run the lightweight test suite before committing changes:
```bash
uv run pytest
```
Tests focus on deterministic, fast feedback (no large datasets required).

## Next steps

- Populate `data/raw` with historical NBA minutes and flesh out the feature engineering logic.
- Expand `config/` with experiment-specific overrides (e.g., configs/classical.yaml).
- Add CI (e.g., GitHub Actions) to automatically run formatter + tests.
