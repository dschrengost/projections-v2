from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from projections.rotations.generator import TeamContext
from projections.rotations.template_generator import TemplateRotationGenerator


def _write_rot_bundle(tmp_path: Path) -> None:
    events = pd.DataFrame(
        [
            # game A
            {
                "team_id": 10,
                "game_id": "0000000001",
                "segment_idx": 0,
                "duration_sec": 300,
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
                "duration_sec": 420,
                "lineup_p1": 1,
                "lineup_p2": 2,
                "lineup_p3": 3,
                "lineup_p4": 4,
                "lineup_p5": 6,
            },
            # game B
            {
                "team_id": 10,
                "game_id": "0000000002",
                "segment_idx": 0,
                "duration_sec": 360,
                "lineup_p1": 11,
                "lineup_p2": 12,
                "lineup_p3": 13,
                "lineup_p4": 14,
                "lineup_p5": 15,
            },
            {
                "team_id": 10,
                "game_id": "0000000002",
                "segment_idx": 1,
                "duration_sec": 360,
                "lineup_p1": 11,
                "lineup_p2": 12,
                "lineup_p3": 13,
                "lineup_p4": 14,
                "lineup_p5": 16,
            },
        ]
    )
    labels = pd.DataFrame(
        [
            {"team_id": 10, "game_id": "0000000001", "regime_label": "tight"},
            {"team_id": 10, "game_id": "0000000002", "regime_label": "tight"},
        ]
    )
    events.to_parquet(tmp_path / "rotation_events.parquet", index=False)
    labels.to_parquet(tmp_path / "rotation_labels.parquet", index=False)


def test_template_generator_determinism(tmp_path: Path) -> None:
    _write_rot_bundle(tmp_path)

    gen = TemplateRotationGenerator(rot_bundle=tmp_path)
    ctx = TeamContext(
        season_id="2024-25",
        game_id="live",
        team_id=10,
        opponent_team_id=20,
        is_home=True,
        candidate_player_ids=[1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
        starter_candidates=[1001, 1002, 1003, 1004, 1005],
        minutes_prior={
            1001: 32.0,
            1002: 30.0,
            1003: 28.0,
            1004: 26.0,
            1005: 24.0,
            1006: 12.0,
            1007: 10.0,
            1008: 8.0,
        },
        n_worlds=200,
        rng_seed=123,
        regime_label="tight",
    )

    out1 = gen.generate(ctx)
    out2 = gen.generate(ctx)

    for pid in ctx.candidate_player_ids or []:
        a1 = out1.minutes_by_player[int(pid)]
        a2 = out2.minutes_by_player[int(pid)]
        assert np.array_equal(a1, a2)

