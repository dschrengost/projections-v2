from __future__ import annotations

from pathlib import Path

import pytest
import pandas as pd

from projections.features.prop_implied_minutes import (
    load_rates_training_base_pra_history_multi_season,
)


def _write_rates_partition(
    *,
    data_root: Path,
    season: int,
    game_date: str,
    rows: list[dict[str, object]],
) -> None:
    out_dir = (
        data_root
        / "gold"
        / "rates_training_base"
        / f"season={int(season)}"
        / f"game_date={game_date}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_dir / "rates_training_base.parquet", index=False)


def test_load_rates_training_base_pra_history_derives_totals(tmp_path: Path) -> None:
    _write_rates_partition(
        data_root=tmp_path,
        season=2025,
        game_date="2026-03-20",
        rows=[
            {
                "game_id": 22501000,
                "team_id": 1610612737,
                "player_id": 1643016,
                "game_date": "2026-03-20",
                "minutes_actual": 30.0,
                "ast_per_min": 0.25,
                "oreb_per_min": 0.10,
                "dreb_per_min": 0.20,
                "fga2_per_min": 0.20,
                "fga3_per_min": 0.10,
                "fta_per_min": 0.15,
                "fg2_pct_label": 0.50,
                "fg3_pct_label": 0.40,
                "ft_pct_label": 0.80,
            }
        ],
    )

    history = load_rates_training_base_pra_history_multi_season(
        data_root=tmp_path,
        start=pd.Timestamp("2026-03-19"),
        end=pd.Timestamp("2026-03-21"),
        player_ids=[1643016],
    )

    assert len(history) == 1
    row = history.iloc[0]
    assert int(row["player_id"]) == 1643016
    assert float(row["minutes_actual"]) == pytest.approx(30.0)
    assert float(row["pts"]) == pytest.approx(13.2)
    assert float(row["reb"]) == pytest.approx(9.0)
    assert float(row["ast"]) == pytest.approx(7.5)


def test_load_rates_training_base_pra_history_filters_players(tmp_path: Path) -> None:
    _write_rates_partition(
        data_root=tmp_path,
        season=2025,
        game_date="2026-03-20",
        rows=[
            {
                "game_id": 22501000,
                "team_id": 1610612737,
                "player_id": 1643016,
                "game_date": "2026-03-20",
                "minutes_actual": 25.0,
                "ast_per_min": 0.20,
                "oreb_per_min": 0.05,
                "dreb_per_min": 0.10,
                "fga2_per_min": 0.15,
                "fga3_per_min": 0.08,
                "fta_per_min": 0.12,
                "fg2_pct_label": 0.48,
                "fg3_pct_label": 0.36,
                "ft_pct_label": 0.78,
            },
            {
                "game_id": 22501000,
                "team_id": 1610612737,
                "player_id": 9999999,
                "game_date": "2026-03-20",
                "minutes_actual": 20.0,
                "ast_per_min": 0.10,
                "oreb_per_min": 0.08,
                "dreb_per_min": 0.18,
                "fga2_per_min": 0.10,
                "fga3_per_min": 0.06,
                "fta_per_min": 0.09,
                "fg2_pct_label": 0.50,
                "fg3_pct_label": 0.33,
                "ft_pct_label": 0.75,
            },
        ],
    )

    history = load_rates_training_base_pra_history_multi_season(
        data_root=tmp_path,
        start=pd.Timestamp("2026-03-19"),
        end=pd.Timestamp("2026-03-21"),
        player_ids=[1643016],
    )

    assert len(history) == 1
    assert set(history["player_id"].astype(int).tolist()) == {1643016}
