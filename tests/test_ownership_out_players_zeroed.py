from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from projections.cli import score_ownership_live
from projections.ownership_v1.loader import OwnershipBundle


def test_score_ownership_sets_dk_out_players_to_zero(monkeypatch, tmp_path: Path) -> None:
    game_date = date(2025, 12, 27)
    draft_group_id = "139438"
    run_id = "20251228T001000Z"

    slate = pd.DataFrame(
        [
            {
                "player_id": 1,
                "player_name": "Playable One",
                "salary": 9000,
                "pos": "C",
                "team": "AAA",
                "status": "ACTIVE",
                "is_disabled": False,
            },
            {
                "player_id": 2,
                "player_name": "Out Guy",
                "salary": 3000,
                "pos": "C",
                "team": "BBB",
                "status": "OUT",
                "is_disabled": False,
            },
        ]
    )

    # Avoid filesystem reads: no sim fpts, no enrichment.
    monkeypatch.setattr(score_ownership_live, "_load_fpts_predictions", lambda *args, **kwargs: None)
    monkeypatch.setattr(score_ownership_live, "_attach_live_ownership_enrichment", lambda df, **kwargs: df)

    # Minimal bundle + feature path; model execution is patched.
    monkeypatch.setattr(
        score_ownership_live,
        "load_ownership_bundle",
        lambda *args, **kwargs: OwnershipBundle(model=object(), feature_cols=["proj_fpts", "salary"], meta={}),
    )
    monkeypatch.setattr(score_ownership_live, "validate_raw_input", lambda df: [])
    monkeypatch.setattr(score_ownership_live, "fill_optional_columns", lambda df: df)
    monkeypatch.setattr(
        score_ownership_live,
        "compute_ownership_features",
        lambda df, **kwargs: df.assign(proj_fpts=df["proj_fpts"].astype(float), salary=df["salary"].astype(float)),
    )
    monkeypatch.setattr(
        score_ownership_live,
        "predict_ownership",
        lambda features, bundle: pd.Series([10.0] * len(features), index=features.index),
    )
    monkeypatch.setattr(
        score_ownership_live,
        "_load_calibration_config",
        lambda: {"playable_filter": {"enabled": False}, "normalization": {"enabled": False}},
    )

    out = score_ownership_live.score_ownership(
        game_date=game_date,
        draft_group_id=draft_group_id,
        run_id=run_id,
        slate_df=slate,
        data_root=tmp_path,
        model_run="test",
        injuries_cutoff_ts=None,
    )
    assert out is not None
    out = out.copy()
    out["pred_own_pct"] = pd.to_numeric(out["pred_own_pct"], errors="coerce").fillna(0.0)

    out_row = out.loc[out["player_name"] == "Out Guy"].iloc[0]
    assert float(out_row["pred_own_pct"]) == 0.0

