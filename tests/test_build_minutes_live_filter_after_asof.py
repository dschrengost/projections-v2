from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from projections.cli.build_minutes_live import _filter_after_run_asof


def test_filter_after_run_asof_drops_future_rows() -> None:
    run_ts = pd.Timestamp(datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc))
    df = pd.DataFrame(
        {
            "as_of_ts": [
                "2025-01-01T11:59:00Z",
                "2025-01-01T12:01:00Z",  # future vs run_ts
                None,  # kept
            ],
            "value": [1, 2, 3],
        }
    )
    filtered, dropped = _filter_after_run_asof(df, time_col="as_of_ts", run_as_of_ts=run_ts)
    assert dropped == 1
    assert len(filtered) == 2
    assert set(filtered["value"].tolist()) == {1, 3}

