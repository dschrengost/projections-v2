from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.cli.build_minutes_live import _read_parquet_tree


def test_read_parquet_tree_unions_mixed_schema_shards(tmp_path: Path) -> None:
    root = tmp_path / "shards"
    root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "game_id": [1001],
            "player_id": [1],
            "lineup_status": ["Confirmed"],
        }
    ).to_parquet(root / "part-000.parquet", index=False)

    pd.DataFrame(
        {
            "game_id": [1002],
            "player_id": [2],
            "lineup_role": ["bench"],
        }
    ).to_parquet(root / "part-001.parquet", index=False)

    out = _read_parquet_tree(root)

    assert len(out) == 2
    assert set(out.columns) == {"game_id", "player_id", "lineup_status", "lineup_role"}
    by_player = out.set_index("player_id")
    assert by_player.loc[1, "lineup_status"] == "Confirmed"
    assert by_player.loc[2, "lineup_role"] == "bench"
    assert pd.isna(by_player.loc[1, "lineup_role"])
    assert pd.isna(by_player.loc[2, "lineup_status"])
