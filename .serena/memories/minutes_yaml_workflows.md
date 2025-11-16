## YAML-driven minutes workflows (Nov 17 2025)
- Added `projections/minutes_v1/config.py` with Pydantic models + loaders for training (`MinutesTrainingConfig`) and scoring (`MinutesScoringConfig`) YAMLs.
- Both minutes CLIs now accept `--config`. YAML values hydrate defaults while explicit CLI flags still win (checked via Click parameter source).
- Example configs live under `config/minutes_training_example.yaml` and `config/minutes_scoring_example.yaml`; README and ROADMAP mention how to run them.
- Tests: `tests/test_minutes_v1_config.py` exercises the override behavior with dummy `typer.Context` objects.