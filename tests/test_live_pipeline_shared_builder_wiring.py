from __future__ import annotations

import inspect

from prefect_flows import live_nba_pipeline


def test_live_minutes_features_task_uses_shared_builder() -> None:
    source = inspect.getsource(live_nba_pipeline.build_minutes_features_task)
    assert "build_shared_features" in source
