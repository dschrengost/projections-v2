from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from projections.cli import check_minutes_live


def _write_projections(root: Path, game_date: str) -> None:
    day_dir = root / game_date
    day_dir.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "game_date": game_date,
            "game_id": 1000,
            "team_id": 1,
            "player_id": player_id,
            "status": "ACTIVE",
            "starter_flag": 1,
            "minutes_p50": 45.0,
        }
        for player_id in range(5)
    ]
    pd.DataFrame(data).to_parquet(day_dir / "minutes.parquet", index=False)


def test_check_minutes_live_happy_path(tmp_path: Path) -> None:
    game_date = "2025-01-02"
    projections_root = tmp_path / "artifacts" / "minutes_v1" / "daily"
    qc_root = tmp_path / "qc"
    _write_projections(projections_root, game_date)

    salaries_path = tmp_path / "salaries.csv"
    salaries_path.write_text("player_id\n0\n1\n2\n3\n4\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        check_minutes_live.app,
        [
            "--game-date",
            game_date,
            "--salaries-path",
            str(salaries_path),
            "--projections-root",
            str(projections_root),
            "--qc-root",
            str(qc_root),
        ],
    )

    assert result.exit_code == 0
    summary_path = qc_root / f"game_date={game_date}" / "summary.json"
    assert summary_path.exists()


def test_check_minutes_live_missing_player(tmp_path: Path) -> None:
    game_date = "2025-01-03"
    projections_root = tmp_path / "artifacts" / "minutes_v1" / "daily"
    qc_root = tmp_path / "qc"
    _write_projections(projections_root, game_date)

    salaries_path = tmp_path / "salaries.csv"
    salaries_path.write_text("player_id\n0\n999\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        check_minutes_live.app,
        [
            "--game-date",
            game_date,
            "--salaries-path",
            str(salaries_path),
            "--projections-root",
            str(projections_root),
            "--qc-root",
            str(qc_root),
        ],
    )

    assert result.exit_code != 0
    summary_path = qc_root / f"game_date={game_date}" / "summary.json"
    assert summary_path.exists()
