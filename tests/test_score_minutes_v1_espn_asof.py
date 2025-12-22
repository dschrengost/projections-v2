from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from projections.cli.score_minutes_v1 import _load_espn_out_players  # noqa: SLF001


def test_espn_out_players_respects_run_as_of_ts(tmp_path) -> None:
    data_root = tmp_path
    game_date = datetime(2025, 1, 1).date()
    injuries_dir = data_root / "silver" / "espn_injuries" / f"date={game_date}"
    injuries_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {
                "player_name": "Early Player",
                "status": "OUT",
                "as_of_ts": "2025-01-01T10:00:00Z",
            },
            {
                "player_name": "Late Player",
                "status": "OUT",
                "as_of_ts": "2025-01-01T12:30:00Z",
            },
        ]
    )
    df.to_parquet(injuries_dir / "injuries.parquet", index=False)

    cutoff = datetime(2025, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
    out_players = _load_espn_out_players(game_date, data_root=data_root, run_as_of_ts=cutoff)

    assert "early player" in out_players
    assert "late player" not in out_players
