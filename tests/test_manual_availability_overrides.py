from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from projections.ops.manual_availability import (
    apply_manual_overrides_to_frame,
    clear_manual_override,
    list_manual_overrides,
    load_manual_overrides_df,
    manual_override_report,
    upsert_manual_override,
)


def test_manual_override_upsert_and_clear_roundtrip(tmp_path: Path) -> None:
    slate_day = date(2026, 3, 1)

    first = upsert_manual_override(
        slate_day,
        game_id="1001",
        player_id="42",
        player_name="Test Player",
        team_id=7,
        team_tricode="AAA",
        override_type="force_out",
        entered_by="daniel",
        reason_code="operator_report",
        data_root=tmp_path,
    )
    assert first["override_type"] == "force_out"

    second = upsert_manual_override(
        slate_day,
        game_id="1001",
        player_id="42",
        player_name="Test Player",
        team_id=7,
        team_tricode="AAA",
        override_type="force_in",
        entered_by="daniel",
        reason_code="source_correction",
        data_root=tmp_path,
    )
    assert second["override_type"] == "force_in"

    active = list_manual_overrides(slate_day, data_root=tmp_path, active_only=True)
    assert len(active) == 1
    assert active[0]["override_type"] == "force_in"

    all_rows = list_manual_overrides(slate_day, data_root=tmp_path, active_only=False)
    assert len(all_rows) == 2
    cleared_prior = [row for row in all_rows if row["override_type"] == "force_out"][0]
    assert cleared_prior["active"] is False
    assert cleared_prior["cleared_by"] == "daniel"

    cleared = clear_manual_override(
        slate_day,
        override_id=str(second["override_id"]),
        cleared_by="daniel",
        data_root=tmp_path,
    )
    assert cleared is not None
    assert cleared["active"] is False

    assert list_manual_overrides(slate_day, data_root=tmp_path, active_only=True) == []
    report = manual_override_report(slate_day, data_root=tmp_path)
    assert report["active_override_count"] == 0


def test_apply_manual_overrides_to_frame_enforces_force_out_and_force_in(tmp_path: Path) -> None:
    slate_day = date(2026, 3, 1)
    upsert_manual_override(
        slate_day,
        game_id="1001",
        player_id="42",
        player_name="Out Player",
        team_id=7,
        team_tricode="AAA",
        override_type="force_out",
        entered_by="daniel",
        data_root=tmp_path,
    )
    upsert_manual_override(
        slate_day,
        game_id="1001",
        player_id="43",
        player_name="In Player",
        team_id=7,
        team_tricode="AAA",
        override_type="force_in",
        entered_by="daniel",
        data_root=tmp_path,
    )
    overrides_df = load_manual_overrides_df(slate_day, data_root=tmp_path)

    base = pd.DataFrame(
        [
            {
                "game_id": 1001,
                "player_id": 42,
                "status": "ACTIVE",
                "is_out": 0,
                "prior_play_prob": 0.8,
                "play_prob": 0.75,
                "lineup_role": "projected_starter",
                "is_projected_starter": True,
                "is_confirmed_starter": True,
                "minutes_p50": 31.0,
                "minutes_final": 31.0,
            },
            {
                "game_id": 1001,
                "player_id": 43,
                "status": "OUT",
                "is_out": 1,
                "prior_play_prob": 0.0,
                "play_prob": 0.0,
                "lineup_role": "out",
                "is_projected_starter": False,
                "is_confirmed_starter": False,
                "minutes_p50": 0.0,
                "minutes_final": 0.0,
                "is_q": 1,
                "is_prob": 1,
            },
        ]
    )

    # as_of_ts=None defaults to now; using a past date would filter out overrides
    # whose effective_ts was set to the upsert time.
    out, diag = apply_manual_overrides_to_frame(
        base,
        overrides_df=overrides_df,
    )
    assert diag["matched_override_count"] == 2

    forced_out = out.loc[out["player_id"] == 42].iloc[0]
    assert forced_out["status"] == "OUT"
    assert int(forced_out["is_out"]) == 1
    assert float(forced_out["play_prob"]) == 0.0
    assert float(forced_out["prior_play_prob"]) == 0.0
    assert float(forced_out["minutes_p50"]) == 0.0
    assert float(forced_out["minutes_final"]) == 0.0
    assert forced_out["manual_override_type"] == "force_out"
    assert bool(forced_out["manual_override_used"]) is True

    forced_in = out.loc[out["player_id"] == 43].iloc[0]
    assert forced_in["status"] == "ACTIVE"
    assert int(forced_in["is_out"]) == 0
    assert pd.isna(forced_in["lineup_role"])
    assert int(forced_in["is_q"]) == 0
    assert int(forced_in["is_prob"]) == 0
    assert float(forced_in["play_prob"]) == 1.0
    assert float(forced_in["prior_play_prob"]) == 1.0
    assert forced_in["manual_override_type"] == "force_in"
