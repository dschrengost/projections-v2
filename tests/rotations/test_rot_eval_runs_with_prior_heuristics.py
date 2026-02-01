from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.rotations.priors_humility import HumilityConfig, apply_prior_humility


def test_rot_eval_priors_slice_runs_through_humility_with_heuristics() -> None:
    p = Path(
        "/home/daniel/projections-data/artifacts/rot_eval_v1/_priors/minutes_prior_internal_season=2024.parquet"
    )
    if not p.exists():
        pytest.skip(f"real priors parquet not available: {p}")

    df = pd.read_parquet(p).head(200)
    out = apply_prior_humility(df, HumilityConfig())

    for col in [
        "minutes_prior_adj",
        "minutes_p10_adj",
        "minutes_p90_adj",
        "play_prob_adj",
        "p_played_ge_5_pred_adj",
        "p_minutes_eq0_pred_adj",
        "humility_tier",
        "humility_reason",
        "p_ge5_prior_heur",
        "p_eq0_prior_heur",
    ]:
        assert col in out.columns

