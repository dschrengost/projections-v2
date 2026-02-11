from __future__ import annotations

import re

import pandas as pd

from prefect_flows.live_nba_pipeline import _utc_now_iso


def test_utc_now_iso_includes_microseconds() -> None:
    value = _utc_now_iso()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}", value)

    parsed = pd.to_datetime(value, errors="coerce")
    assert pd.notna(parsed)
