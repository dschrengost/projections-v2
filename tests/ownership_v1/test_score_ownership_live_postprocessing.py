from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.cli.score_ownership_live import _apply_postprocessing


def test_postprocessing_normalizes_when_calibrator_missing(tmp_path: Path) -> None:
    output = pd.DataFrame(
        {
            "proj_fpts": [30.0, 25.0, 20.0, 15.0],
            "pred_own_pct_raw": [10.0, 20.0, 0.0, 5.0],
            "pred_own_pct": [10.0, 20.0, 0.0, 5.0],
        }
    )
    salaries = pd.DataFrame(index=output.index)

    config = {
        "playable_filter": {"enabled": False},
        "calibration": {
            "enabled": True,
            "method": "power",
            "calibrator_path": "does_not_exist.json",
            "R": 8.0,
            "structural_zeros": {"exclude_zero_prediction": True},
        },
        "normalization": {"enabled": True, "target_sum_pct": 80.0, "cap_pct": 100.0},
        "logging": {"log_metrics": False},
    }

    out = _apply_postprocessing(output, salaries, config=config, data_root=tmp_path)

    assert float(out["pred_own_pct"].sum()) == pytest.approx(80.0)
    assert float(out.loc[2, "pred_own_pct"]) == 0.0
    assert float(out.loc[0, "pred_own_pct"] / out.loc[1, "pred_own_pct"]) == pytest.approx(0.5)

