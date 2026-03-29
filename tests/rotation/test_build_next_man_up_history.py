from __future__ import annotations

import pandas as pd

from scripts.rotation.build_next_man_up_history import build_next_man_up_labels


def test_build_next_man_up_labels_assigns_expected_archetypes() -> None:
    features = pd.DataFrame(
        [
            {
                "game_id": 1,
                "game_date": "2026-01-01",
                "season": "2025-26",
                "team_id": 10,
                "player_id": 100,
                "player_name": "A",
                "team_tricode": "AAA",
                "archetype": "big",
                "pos_bucket": "B",
                "prior_play_prob": 0.20,
                "minutes_from_stints_prior_20": 6.0,
                "an_has_any_props": 0.0,
                "an_implied_minutes": 0.0,
                "recent_start_pct_10": 0.0,
                "started_proxy_rate_prior_10": 0.0,
                "started_proxy_rate_prior_20": 0.0,
                "lineup_starter_announced": 0.0,
            },
            {
                "game_id": 1,
                "game_date": "2026-01-01",
                "season": "2025-26",
                "team_id": 10,
                "player_id": 101,
                "player_name": "B",
                "team_tricode": "AAA",
                "archetype": "wing",
                "pos_bucket": "W",
                "prior_play_prob": 0.95,
                "minutes_from_stints_prior_20": 8.0,
                "an_has_any_props": 0.0,
                "an_implied_minutes": 0.0,
                "recent_start_pct_10": 1.0,
                "started_proxy_rate_prior_10": 0.0,
                "started_proxy_rate_prior_20": 0.0,
                "lineup_starter_announced": 0.0,
            },
            {
                "game_id": 1,
                "game_date": "2026-01-01",
                "season": "2025-26",
                "team_id": 10,
                "player_id": 102,
                "player_name": "C",
                "team_tricode": "AAA",
                "archetype": "guard",
                "pos_bucket": "G",
                "prior_play_prob": 0.95,
                "minutes_from_stints_prior_20": 10.0,
                "an_has_any_props": 1.0,
                "an_implied_minutes": 0.0,
                "recent_start_pct_10": 0.0,
                "started_proxy_rate_prior_10": 0.0,
                "started_proxy_rate_prior_20": 0.0,
                "lineup_starter_announced": 0.0,
            },
            {
                "game_id": 1,
                "game_date": "2026-01-01",
                "season": "2025-26",
                "team_id": 10,
                "player_id": 103,
                "player_name": "D",
                "team_tricode": "AAA",
                "archetype": "guard",
                "pos_bucket": "G",
                "prior_play_prob": 0.95,
                "minutes_from_stints_prior_20": 11.0,
                "an_has_any_props": 0.0,
                "an_implied_minutes": 0.0,
                "recent_start_pct_10": 0.0,
                "started_proxy_rate_prior_10": 0.0,
                "started_proxy_rate_prior_20": 0.0,
                "lineup_starter_announced": 0.0,
            },
        ]
    )
    labels_minutes = pd.DataFrame(
        [
            {"game_id": 1, "game_date": "2026-01-01", "team_id": 10, "player_id": 100, "minutes_label": 28.0, "starter_flag_label": 1},
            {"game_id": 1, "game_date": "2026-01-01", "team_id": 10, "player_id": 101, "minutes_label": 18.0, "starter_flag_label": 0},
            {"game_id": 1, "game_date": "2026-01-01", "team_id": 10, "player_id": 102, "minutes_label": 26.0, "starter_flag_label": 0},
            {"game_id": 1, "game_date": "2026-01-01", "team_id": 10, "player_id": 103, "minutes_label": 12.0, "starter_flag_label": 0},
        ]
    )

    out = build_next_man_up_labels(
        features,
        labels_minutes,
        sparse_prior_play_prob_max=0.5,
        sparse_prior_minutes_max=12.0,
        surprise_actual_min=8.0,
        entrant_actual_min=16.0,
        core_actual_min=24.0,
        starter_actual_min=20.0,
        starter_hist_start_rate_max=0.2,
    )
    labels = dict(zip(out["player_id"], out["primary_archetype"], strict=True))
    assert labels[100] == "emergency_starter"
    assert labels[101] == "bench_rotation_entrant"
    assert labels[102] == "bench_core_riser"
    assert labels[103] == "sparse_active_surprise"
