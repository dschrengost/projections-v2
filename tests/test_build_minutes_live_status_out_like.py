from __future__ import annotations

import pandas as pd

from projections.cli.build_minutes_live import _status_series_is_out_like


def test_status_series_is_out_like_treats_doubtful_as_out() -> None:
    status = pd.Series(
        [
            "OUT",
            "Out For Season",
            "DOUBTFUL",
            "D",
            "inactive",
            "QUESTIONABLE",
            "Q",
            "PROBABLE",
            None,
        ]
    )
    mask = _status_series_is_out_like(status)
    assert mask.tolist() == [True, True, True, True, True, False, False, False, False]
