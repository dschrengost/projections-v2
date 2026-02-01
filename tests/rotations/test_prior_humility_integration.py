from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.rotations.generator import TeamContext
from projections.rotations.priors_humility import HumilityConfig
from projections.rotations.template_generator import TemplateRotationGenerator


def _write_rot_bundle(tmp_path: Path) -> None:
    events = pd.DataFrame(
        [
            {
                "team_id": 10,
                "game_id": "0000000001",
                "segment_idx": 0,
                "duration_sec": 600,
                "lineup_p1": 1,
                "lineup_p2": 2,
                "lineup_p3": 3,
                "lineup_p4": 4,
                "lineup_p5": 5,
            },
            {
                "team_id": 10,
                "game_id": "0000000001",
                "segment_idx": 1,
                "duration_sec": 600,
                "lineup_p1": 1,
                "lineup_p2": 2,
                "lineup_p3": 3,
                "lineup_p4": 4,
                "lineup_p5": 6,
            },
        ]
    )
    labels = pd.DataFrame([{"team_id": 10, "game_id": "0000000001", "regime_label": "tight"}])
    events.to_parquet(tmp_path / "rotation_events.parquet", index=False)
    labels.to_parquet(tmp_path / "rotation_labels.parquet", index=False)


def test_template_generator_calls_humility_when_enabled(tmp_path: Path) -> None:
    _write_rot_bundle(tmp_path)
    cfg = HumilityConfig(enabled=True, protect_starters=True, protect_top_n=True, top_n_lock=2)
    gen = TemplateRotationGenerator(rot_bundle=tmp_path, humility_config=cfg)

    candidate_ids = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
    minutes_prior = {
        1001: 34.0,
        1002: 32.0,
        1003: 30.0,
        1004: 28.0,
        1005: 26.0,
        1006: 12.0,
        1007: 9.0,
        1008: 7.0,
        1009: 4.0,
        1010: 2.0,
    }
    ctx = TeamContext(
        season_id="2024-25",
        game_id="live",
        team_id=10,
        opponent_team_id=20,
        is_home=True,
        candidate_player_ids=candidate_ids,
        starter_candidates=[1001, 1002, 1003, 1004, 1005],
        minutes_prior=minutes_prior,
        n_worlds=10,
        rng_seed=123,
        regime_label="tight",
    )

    worlds = gen.generate(ctx)
    diag = worlds.diagnostics or {}
    assert diag.get("humility_enabled") is True
    assert isinstance(diag.get("humility_config"), dict)
    assert isinstance(diag.get("humility_tier_counts"), dict)
    assert "starter" in diag["humility_tier_counts"]

