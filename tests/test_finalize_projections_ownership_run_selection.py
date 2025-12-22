from __future__ import annotations

from datetime import date

import pandas as pd

from projections.cli.finalize_projections import _load_ownership  # noqa: SLF001


def test_load_ownership_prefers_run_dir(tmp_path) -> None:
    data_root = tmp_path
    game_date = date(2025, 1, 1)
    draft_group_id = "12345"
    run_id = "RUN123"

    run_dir = (
        data_root
        / "silver"
        / "ownership_predictions"
        / str(game_date)
        / f"run={run_id}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    run_df = pd.DataFrame(
        [
            {
                "player_name": "Run Player",
                "pred_own_pct": 12.0,
            }
        ]
    )
    run_df.to_parquet(run_dir / f"{draft_group_id}.parquet", index=False)

    legacy_dir = data_root / "silver" / "ownership_predictions" / str(game_date)
    legacy_df = pd.DataFrame(
        [
            {
                "player_name": "Legacy Player",
                "pred_own_pct": 5.0,
            }
        ]
    )
    legacy_df.to_parquet(legacy_dir / f"{draft_group_id}.parquet", index=False)

    loaded = _load_ownership(game_date, draft_group_id, data_root, run_id=run_id)

    assert loaded is not None
    assert loaded["pred_own_pct"].iloc[0] == 12.0
