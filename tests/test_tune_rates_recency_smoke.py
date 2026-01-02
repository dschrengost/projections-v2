from __future__ import annotations

import json
import subprocess
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.rates.train_rates_v1 import FEATURES_STAGE0


def _write_rates_partition(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_tune_rates_recency_smoke(tmp_path: Path) -> None:
    # Build a tiny synthetic rates_training_base tree under a temp data root.
    data_root = tmp_path / "data_root"
    season = 2024
    season_root = data_root / "gold" / "rates_training_base" / f"season={season}"

    start_day = date(2024, 1, 1)
    n_days = 60
    n_rows_per_day = 10
    rng = np.random.default_rng(seed=0)

    stage0_features = list(FEATURES_STAGE0)

    # Targets + efficiency label columns
    target_cols = [
        "fga2_per_min",
        "fga3_per_min",
        "fta_per_min",
        "ast_per_min",
        "tov_per_min",
        "oreb_per_min",
        "dreb_per_min",
        "stl_per_min",
        "blk_per_min",
        "fg2_pct_label",
        "fg3_pct_label",
        "ft_pct_label",
    ]

    for i in range(n_days):
        day = start_day + timedelta(days=i)
        game_date = pd.Timestamp(day).strftime("%Y-%m-%d")
        game_id = 100000 + i

        rows = []
        for j in range(n_rows_per_day):
            player_id = 2000 + j
            team_id = 10 if j < n_rows_per_day // 2 else 20
            opponent_id = 20 if team_id == 10 else 10
            home_flag = 1 if team_id == 10 else 0
            is_starter = 1 if j < 5 else 0
            minutes_actual = float(12 + 2 * is_starter + rng.normal(0, 1))
            tip_ts = pd.Timestamp(f"{game_date}T23:30:00Z")
            feature_as_of_ts = tip_ts - pd.Timedelta(minutes=30)

            row = {
                "season": season,
                "game_id": game_id,
                "game_date": game_date,
                "tip_ts": tip_ts,
                "feature_as_of_ts": feature_as_of_ts,
                "team_id": team_id,
                "opponent_id": opponent_id,
                "home_flag": home_flag,
                "player_id": player_id,
                "has_odds": 1,
                "spread_close": float(rng.normal(0, 5)),
                "total_close": float(220 + rng.normal(0, 5)),
                "team_itt": float(110 + rng.normal(0, 3)),
                "opp_itt": float(110 + rng.normal(0, 3)),
                "is_starter": is_starter,
                "minutes_actual": minutes_actual,
            }

            # Fill remaining stage0 features and targets with simple synthetic signals.
            for col in stage0_features:
                row.setdefault(col, float(rng.uniform(0.0, 0.5)))
            for col in target_cols:
                if col.endswith("_pct_label"):
                    row[col] = float(np.clip(rng.normal(0.5, 0.1), 0.0, 1.0))
                else:
                    row[col] = float(np.clip(rng.normal(0.10, 0.05), 0.0, 1.0))

            rows.append(row)

        df = pd.DataFrame(rows)
        out_path = season_root / f"game_date={game_date}" / "rates_training_base.parquet"
        _write_rates_partition(out_path, df)

    output_root = tmp_path / "out"
    run_id = "pytest_smoke"

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/tune_rates_recency.py",
        "--season",
        str(season),
        "--start",
        str(start_day),
        "--end",
        str(start_day + timedelta(days=n_days - 1)),
        "--feature-set",
        "stage0",
        "--n-trials",
        "2",
        "--max-folds",
        "2",
        "--train-months",
        "1",
        "--cal-weeks",
        "1",
        "--val-weeks",
        "1",
        "--step-weeks",
        "1",
        "--min-train-rows",
        "10",
        "--min-val-rows",
        "10",
        "--seed",
        "0",
        "--run-id",
        run_id,
        "--data-root",
        str(data_root),
        "--output-root",
        str(output_root),
    ]
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    study_path = output_root / run_id / "study.json"
    assert study_path.exists()
    payload = json.loads(study_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == run_id
    assert payload["best"]["params"]
    assert "baseline_unweighted_score" in payload
    assert "baseline_default_score" in payload
    assert "best_score" in payload
    assert "baselines" in payload
    assert "unweighted" in payload["baselines"]
    assert "default" in payload["baselines"]
    assert len(payload["trials"]) == 2


def test_tune_rates_recency_fails_without_allow_game_date_weighting(tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    season = 2024
    season_root = data_root / "gold" / "rates_training_base" / f"season={season}"

    start_day = date(2024, 1, 1)
    n_days = 40
    n_rows_per_day = 5
    rng = np.random.default_rng(seed=0)

    stage0_features = list(FEATURES_STAGE0)
    target_cols = [
        "fga2_per_min",
        "fga3_per_min",
        "fta_per_min",
        "ast_per_min",
        "tov_per_min",
        "oreb_per_min",
        "dreb_per_min",
        "stl_per_min",
        "blk_per_min",
        "fg2_pct_label",
        "fg3_pct_label",
        "ft_pct_label",
    ]

    for i in range(n_days):
        day = start_day + timedelta(days=i)
        game_date = pd.Timestamp(day).strftime("%Y-%m-%d")
        game_id = 100000 + i
        rows = []
        for j in range(n_rows_per_day):
            row = {
                "season": season,
                "game_id": game_id,
                "game_date": game_date,
                "team_id": 10,
                "opponent_id": 20,
                "home_flag": 1,
                "player_id": 2000 + j,
                "has_odds": 1,
                "spread_close": float(rng.normal(0, 5)),
                "total_close": float(220 + rng.normal(0, 5)),
                "team_itt": float(110 + rng.normal(0, 3)),
                "opp_itt": float(110 + rng.normal(0, 3)),
                "is_starter": 1 if j < 3 else 0,
                "minutes_actual": float(12 + rng.normal(0, 1)),
            }
            for col in stage0_features:
                row.setdefault(col, float(rng.uniform(0.0, 0.5)))
            for col in target_cols:
                row[col] = float(np.clip(rng.normal(0.10, 0.05), 0.0, 1.0))
            rows.append(row)
        df = pd.DataFrame(rows)
        out_path = season_root / f"game_date={game_date}" / "rates_training_base.parquet"
        _write_rates_partition(out_path, df)

    output_root = tmp_path / "out"
    run_id = "pytest_failfast"
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/tune_rates_recency.py",
        "--season",
        str(season),
        "--start",
        str(start_day),
        "--end",
        str(start_day + timedelta(days=n_days - 1)),
        "--feature-set",
        "stage0",
        "--n-trials",
        "1",
        "--max-folds",
        "1",
        "--train-months",
        "1",
        "--cal-weeks",
        "1",
        "--val-weeks",
        "1",
        "--step-weeks",
        "1",
        "--min-train-rows",
        "5",
        "--min-val-rows",
        "5",
        "--seed",
        "0",
        "--run-id",
        run_id,
        "--data-root",
        str(data_root),
        "--output-root",
        str(output_root),
    ]
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode != 0
    assert "allow-game-date-weighting" in result.stderr.lower() or "allow-game-date-weighting" in result.stdout.lower()
