from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from scrapers.realgm_depth_charts import save_realgm_depth_charts_bronze


def test_save_realgm_depth_charts_bronze_writes_history_and_latest(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "team_name": "New York Knicks",
                "player_name": "Jalen Brunson",
                "realgm_player_id": 1001,
                "position": "PG",
                "depth_role": "starter",
                "depth_order": 0,
                "recent_stats": "",
                "movement": None,
                "scraped_at": "2026-01-18T18:15:00Z",
            }
        ]
    )
    out = save_realgm_depth_charts_bronze(
        df,
        game_date=date(2026, 1, 18),
        data_root=tmp_path,
    )

    for key in ("history_path", "day_latest_path", "global_latest_path", "global_alias_path"):
        assert key in out
        assert out[key].exists()

    history_text = str(out["history_path"])
    assert "/bronze/realgm/depth_charts/season=2025/date=2026-01-18/run_ts=" in history_text
    loaded = pd.read_parquet(out["global_latest_path"])
    assert len(loaded) == 1
    assert int(pd.to_numeric(loaded["realgm_player_id"], errors="coerce").iloc[0]) == 1001
