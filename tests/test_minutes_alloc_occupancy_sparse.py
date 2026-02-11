from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.minutes_alloc.occupancy_sparse import (
    OccupancySparseConfig,
    apply_occupancy_sparse_allocation,
)


def _base_team_frame(*, all_out: bool = False) -> pd.DataFrame:
    n = 8
    status = ["OK"] * n
    is_out = [0] * n
    if all_out:
        status = ["OUT"] * n
        is_out = [1] * n
    else:
        status[6] = "OUT"
        is_out[6] = 1

    frame = pd.DataFrame(
        {
            "game_id": [101] * n,
            "team_id": [200] * n,
            "player_id": list(range(1, n + 1)),
            "play_prob": np.linspace(0.2, 0.95, n),
            "minutes_p10": np.linspace(6.0, 24.0, n),
            "minutes_p50": np.linspace(10.0, 32.0, n),
            "minutes_p90": np.linspace(14.0, 38.0, n),
            "status": status,
            "is_out": is_out,
            "lineup_role": ["PROJECTED_STARTER"] * 5 + ["BENCH"] * 3,
            "starter_flag": [1] * 5 + [0] * 3,
            "is_projected_starter": [1] * 5 + [0] * 3,
            "is_confirmed_starter": [0] * n,
            "is_starter": [1] * 5 + [0] * 3,
            "spread_home": [-4.5] * n,
            "total": [228.5] * n,
            "home_flag": [1] * n,
        }
    )
    return frame


def _deep_rotation_frame() -> pd.DataFrame:
    n = 13
    pos_bucket = [
        "BIG",
        "G",
        "W",
        "W",
        "BIG",
        "G",
        "W",
        "BIG",
        "G",
        "W",
        "BIG",
        "BIG",
        "W",
    ]
    return pd.DataFrame(
        {
            "game_id": [102] * n,
            "team_id": [201] * n,
            "player_id": list(range(1, n + 1)),
            "play_prob": [0.98, 0.96, 0.95, 0.93, 0.91, 0.89, 0.87, 0.84, 0.82, 0.80, 0.79, 0.77, 0.75],
            "minutes_p10": np.linspace(20.0, 5.0, n),
            "minutes_p50": np.linspace(30.0, 10.0, n),
            "minutes_p90": np.linspace(36.0, 14.0, n),
            "status": ["OK"] * n,
            "is_out": [0] * n,
            "lineup_role": ["PROJECTED_STARTER"] * 5 + ["BENCH"] * (n - 5),
            "starter_flag": [1] * 5 + [0] * (n - 5),
            "is_projected_starter": [1] * 5 + [0] * (n - 5),
            "is_confirmed_starter": [0] * n,
            "is_starter": [1] * 5 + [0] * (n - 5),
            "spread_home": [-10.5] * n,
            "total": [239.5] * n,
            "home_flag": [1] * n,
            "pos_bucket": pos_bucket,
        }
    )


def test_apply_occupancy_sparse_allocation_invariants() -> None:
    frame = _base_team_frame(all_out=False)
    cfg = OccupancySparseConfig(starter_floor=0.8)
    out, diag = apply_occupancy_sparse_allocation(frame, config=cfg)

    assert not out.empty
    assert set(
        [
            "minutes_occ",
            "play_prob_occ",
            "minutes_p10_occ",
            "minutes_p90_occ",
            "eligible_flag_occ",
            "out_flag_occ",
            "starter_flag_occ",
        ]
    ).issubset(out.columns)

    out_rows = out[out["out_flag_occ"] == 1]
    assert not out_rows.empty
    assert (pd.to_numeric(out_rows["minutes_occ"], errors="coerce").fillna(0.0) == 0.0).all()
    assert (pd.to_numeric(out_rows["play_prob_occ"], errors="coerce").fillna(0.0) == 0.0).all()

    active = out["out_flag_occ"] == 0
    assert float(pd.to_numeric(out.loc[active, "minutes_occ"], errors="coerce").fillna(0.0).sum()) == pytest.approx(240.0)

    p10 = pd.to_numeric(out["minutes_p10_occ"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    p50 = pd.to_numeric(out["minutes_occ"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    p90 = pd.to_numeric(out["minutes_p90_occ"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    assert np.all(p10 <= p50 + 1e-9)
    assert np.all(p50 <= p90 + 1e-9)

    starter_active = (out["starter_flag_occ"] == 1) & (out["out_flag_occ"] == 0) & (p50 > 0.0)
    starter_probs = pd.to_numeric(out.loc[starter_active, "play_prob_occ"], errors="coerce").fillna(0.0)
    assert not starter_probs.empty
    assert float(starter_probs.min()) >= 0.8

    assert not diag.empty
    assert float(pd.to_numeric(diag["team_minutes_sum_dev"], errors="coerce").fillna(0.0).max()) < 1e-6


def test_apply_occupancy_sparse_allocation_handles_all_out_team() -> None:
    frame = _base_team_frame(all_out=True)
    out, diag = apply_occupancy_sparse_allocation(frame, config=OccupancySparseConfig())

    assert float(pd.to_numeric(out["minutes_occ"], errors="coerce").fillna(0.0).sum()) == 0.0
    assert float(pd.to_numeric(out["play_prob_occ"], errors="coerce").fillna(0.0).sum()) == 0.0
    assert not diag.empty
    assert int(pd.to_numeric(diag["active_count"], errors="coerce").fillna(0).iloc[0]) == 0
    assert float(pd.to_numeric(diag["team_minutes_sum"], errors="coerce").fillna(0.0).iloc[0]) == 0.0


def test_occupancy_sparse_dynamic_k_max_expands_for_deep_team() -> None:
    frame = _deep_rotation_frame()

    static_cfg = OccupancySparseConfig(
        k_min=8,
        k_max=11,
        dynamic_k_bounds_enabled=False,
    )
    _, static_diag = apply_occupancy_sparse_allocation(frame, config=static_cfg)
    static_n_eligible = int(pd.to_numeric(static_diag["n_eligible"], errors="coerce").fillna(0).iloc[0])

    dynamic_cfg = OccupancySparseConfig(
        k_min=8,
        k_max=11,
        dynamic_k_bounds_enabled=True,
        dynamic_k_max_cap=13,
        dynamic_k_min_floor=7,
        dynamic_k_window=3,
        dynamic_depth_prob_floor=0.06,
        dynamic_depth_minutes_floor=4.0,
        dynamic_bench_share_midpoint=0.18,
        dynamic_bench_share_scale=25.0,
    )
    _, dynamic_diag = apply_occupancy_sparse_allocation(frame, config=dynamic_cfg)
    dynamic_n_eligible = int(pd.to_numeric(dynamic_diag["n_eligible"], errors="coerce").fillna(0).iloc[0])
    dynamic_k_max_eff = int(pd.to_numeric(dynamic_diag["k_max_eff"], errors="coerce").fillna(0).iloc[0])

    assert static_n_eligible <= 11
    assert dynamic_n_eligible > static_n_eligible
    assert dynamic_k_max_eff > 11


def test_occupancy_sparse_config_parses_dynamic_payload() -> None:
    cfg = OccupancySparseConfig.from_payload(
        {
            "dynamic_k_bounds_enabled": "true",
            "dynamic_k_max_cap": 14,
            "dynamic_k_min_floor": 6,
            "dynamic_k_window": 4,
            "dynamic_depth_prob_floor": 0.08,
            "dynamic_depth_minutes_floor": 5.0,
            "dynamic_bench_share_midpoint": 0.2,
            "dynamic_bench_share_scale": 30.0,
            "dnp_suppression_enabled": "true",
            "dnp_rate_threshold": 0.4,
            "dnp_prior_play_prob_max": 0.45,
            "dnp_inactive_streak_threshold": 4,
            "dnp_consecutive_active_dnp_threshold": 3,
            "dnp_suppression_relax_in_injury_regime": "false",
            "dnp_injury_regime_out_count_threshold": 3,
            "dnp_injury_regime_out_starters_threshold": 2,
            "dnp_injury_regime_min_bench_share_pred": 0.25,
            "archetype_shortage_enabled": "true",
            "archetype_source_col": "pos_bucket",
            "archetype_out_count_threshold": 2,
            "archetype_out_inactive_streak_max": 12,
            "archetype_dnp_rate_relax_add": 0.12,
            "archetype_dnp_inactive_relax_add": 2,
            "archetype_dnp_consecutive_relax_add": 1,
            "archetype_play_prob_floor": 0.09,
            "archetype_seed_p90_min": 9.0,
            "archetype_seed_minutes_min": 5.0,
            "archetype_seed_minutes_max": 11.0,
            "archetype_seed_minutes_p90_mult": 0.6,
            "archetype_seed_max_players": 3,
        }
    )
    assert cfg.dynamic_k_bounds_enabled is True
    assert cfg.dynamic_k_max_cap == 14
    assert cfg.dynamic_k_min_floor == 6
    assert cfg.dynamic_k_window == 4
    assert cfg.dynamic_depth_prob_floor == pytest.approx(0.08)
    assert cfg.dynamic_depth_minutes_floor == pytest.approx(5.0)
    assert cfg.dynamic_bench_share_midpoint == pytest.approx(0.2)
    assert cfg.dynamic_bench_share_scale == pytest.approx(30.0)
    assert cfg.dnp_suppression_enabled is True
    assert cfg.dnp_rate_threshold == pytest.approx(0.4)
    assert cfg.dnp_prior_play_prob_max == pytest.approx(0.45)
    assert cfg.dnp_inactive_streak_threshold == 4
    assert cfg.dnp_consecutive_active_dnp_threshold == 3
    assert cfg.dnp_suppression_relax_in_injury_regime is False
    assert cfg.dnp_injury_regime_out_count_threshold == 3
    assert cfg.dnp_injury_regime_out_starters_threshold == 2
    assert cfg.dnp_injury_regime_min_bench_share_pred == pytest.approx(0.25)
    assert cfg.archetype_shortage_enabled is True
    assert cfg.archetype_source_col == "pos_bucket"
    assert cfg.archetype_out_count_threshold == 2
    assert cfg.archetype_out_inactive_streak_max == 12
    assert cfg.archetype_dnp_rate_relax_add == pytest.approx(0.12)
    assert cfg.archetype_dnp_inactive_relax_add == 2
    assert cfg.archetype_dnp_consecutive_relax_add == 1
    assert cfg.archetype_play_prob_floor == pytest.approx(0.09)
    assert cfg.archetype_seed_p90_min == pytest.approx(9.0)
    assert cfg.archetype_seed_minutes_min == pytest.approx(5.0)
    assert cfg.archetype_seed_minutes_max == pytest.approx(11.0)
    assert cfg.archetype_seed_minutes_p90_mult == pytest.approx(0.6)
    assert cfg.archetype_seed_max_players == 3


def test_occupancy_sparse_suppresses_high_dnp_risk_bench_players() -> None:
    frame = _deep_rotation_frame()
    frame["active_but_dnp_rate_last10"] = 0.0
    frame["consecutive_active_dnp"] = 0
    frame["inactive_streak_len"] = 0
    frame["prior_play_prob"] = 0.8

    # Mark two fringe bench players as high DNP risk with weak/unknown prior.
    risky = frame["player_id"].isin([12, 13])
    frame.loc[risky, "active_but_dnp_rate_last10"] = 0.6
    frame.loc[risky, "prior_play_prob"] = np.nan

    cfg_no_suppress = OccupancySparseConfig(
        dynamic_k_bounds_enabled=True,
        dnp_suppression_enabled=False,
    )
    out_no, diag_no = apply_occupancy_sparse_allocation(frame, config=cfg_no_suppress)
    risky_eligible_no = (
        out_no.loc[out_no["player_id"].isin([12, 13]), "eligible_flag_occ"]
        .astype(int)
        .sum()
    )
    assert risky_eligible_no >= 1

    cfg_suppress = OccupancySparseConfig(
        dynamic_k_bounds_enabled=True,
        dnp_suppression_enabled=True,
        dnp_rate_threshold=0.35,
        dnp_prior_play_prob_max=0.5,
        dnp_inactive_streak_threshold=3,
        dnp_consecutive_active_dnp_threshold=2,
    )
    out_yes, diag_yes = apply_occupancy_sparse_allocation(frame, config=cfg_suppress)
    risky_eligible_yes = (
        out_yes.loc[out_yes["player_id"].isin([12, 13]), "eligible_flag_occ"]
        .astype(int)
        .sum()
    )

    assert risky_eligible_yes == 0
    assert "n_dnp_suppressed" in diag_yes.columns
    assert int(pd.to_numeric(diag_yes["n_dnp_suppressed"], errors="coerce").fillna(0).iloc[0]) >= 2


def test_occupancy_sparse_relaxes_suppression_in_injury_regime() -> None:
    frame = _deep_rotation_frame()
    frame["active_but_dnp_rate_last10"] = 0.0
    frame["consecutive_active_dnp"] = 0
    frame["inactive_streak_len"] = 0
    frame["prior_play_prob"] = 0.8

    # Create injury regime: two outs, including one starter.
    frame.loc[frame["player_id"] == 1, ["status", "is_out"]] = ["OUT", 1]
    frame.loc[frame["player_id"] == 11, ["status", "is_out"]] = ["OUT", 1]

    # Fringe players that should be suppressed in normal regime.
    risky = frame["player_id"].isin([12, 13])
    frame.loc[risky, "active_but_dnp_rate_last10"] = 0.45
    frame.loc[risky, "prior_play_prob"] = np.nan
    frame.loc[risky, "minutes_p50"] = [20.0, 18.0]
    frame.loc[risky, "play_prob"] = [0.90, 0.88]

    cfg_no_relax = OccupancySparseConfig(
        dynamic_k_bounds_enabled=True,
        dnp_suppression_enabled=True,
        dnp_suppression_relax_in_injury_regime=False,
        dnp_rate_threshold=0.35,
        archetype_shortage_enabled=False,
    )
    out_no, _ = apply_occupancy_sparse_allocation(frame, config=cfg_no_relax)
    risky_eligible_no = (
        out_no.loc[out_no["player_id"].isin([12, 13]), "eligible_flag_occ"]
        .astype(int)
        .sum()
    )
    assert risky_eligible_no == 0

    cfg_relax = OccupancySparseConfig(
        dynamic_k_bounds_enabled=True,
        dnp_suppression_enabled=True,
        dnp_suppression_relax_in_injury_regime=True,
        dnp_injury_regime_out_count_threshold=2,
        dnp_injury_regime_out_starters_threshold=1,
        dnp_injury_regime_min_bench_share_pred=0.10,
        dnp_rate_threshold=0.35,
        archetype_shortage_enabled=False,
    )
    out_yes, diag_yes = apply_occupancy_sparse_allocation(frame, config=cfg_relax)
    risky_eligible_yes = (
        out_yes.loc[out_yes["player_id"].isin([12, 13]), "eligible_flag_occ"]
        .astype(int)
        .sum()
    )

    assert risky_eligible_yes > risky_eligible_no
    assert "injury_regime_active" in diag_yes.columns
    assert int(pd.to_numeric(diag_yes["injury_regime_active"], errors="coerce").fillna(0).iloc[0]) == 1


def test_occupancy_sparse_archetype_shortage_seeds_same_archetype_replacement() -> None:
    frame = _deep_rotation_frame()
    frame["inactive_streak_len"] = 0
    frame["active_but_dnp_rate_last10"] = 0.0
    frame["consecutive_active_dnp"] = 0

    # Starting big out; bench big replacement has zero baseline center.
    frame.loc[frame["player_id"] == 1, ["status", "is_out"]] = ["OUT", 1]
    frame.loc[frame["player_id"] == 1, "inactive_streak_len"] = 1
    frame.loc[frame["player_id"] == 12, ["minutes_p50", "minutes_p10", "minutes_p90", "play_prob"]] = [0.0, 0.0, 16.0, 0.01]

    cfg_disabled = OccupancySparseConfig(
        dynamic_k_bounds_enabled=True,
        dnp_suppression_enabled=False,
        archetype_shortage_enabled=False,
    )
    out_disabled, diag_disabled = apply_occupancy_sparse_allocation(frame, config=cfg_disabled)
    repl_disabled = out_disabled.loc[out_disabled["player_id"] == 12].iloc[0]
    assert float(pd.to_numeric(repl_disabled["minutes_occ"], errors="coerce") or 0.0) == pytest.approx(0.0, abs=1e-9)

    cfg_enabled = OccupancySparseConfig(
        dynamic_k_bounds_enabled=True,
        dnp_suppression_enabled=False,
        archetype_shortage_enabled=True,
        archetype_out_count_threshold=1,
        archetype_out_inactive_streak_max=10,
        archetype_play_prob_floor=0.08,
        archetype_seed_p90_min=8.0,
        archetype_seed_minutes_min=4.0,
        archetype_seed_minutes_max=10.0,
        archetype_seed_minutes_p90_mult=0.5,
        archetype_seed_max_players=2,
    )
    out_enabled, diag_enabled = apply_occupancy_sparse_allocation(frame, config=cfg_enabled)
    repl_enabled = out_enabled.loc[out_enabled["player_id"] == 12].iloc[0]
    assert float(pd.to_numeric(repl_enabled["minutes_occ"], errors="coerce") or 0.0) > 0.0
    assert int(pd.to_numeric(diag_enabled["archetype_shortage_active"], errors="coerce").fillna(0).iloc[0]) == 1
    assert int(pd.to_numeric(diag_enabled["n_archetype_seeded"], errors="coerce").fillna(0).iloc[0]) >= 1
    assert int(pd.to_numeric(diag_enabled["archetype_out_big"], errors="coerce").fillna(0).iloc[0]) >= 1
    assert int(pd.to_numeric(diag_disabled["n_archetype_seeded"], errors="coerce").fillna(0).iloc[0]) == 0


def test_occupancy_sparse_archetype_shortage_ignores_stale_long_term_out() -> None:
    frame = _deep_rotation_frame()
    frame["inactive_streak_len"] = 0
    frame["active_but_dnp_rate_last10"] = 0.0
    frame["consecutive_active_dnp"] = 0

    # Out big has been out a long time and is not a starter signal.
    frame.loc[frame["player_id"] == 11, ["status", "is_out", "starter_flag", "is_projected_starter", "is_starter"]] = ["OUT", 1, 0, 0, 0]
    frame.loc[frame["player_id"] == 11, "inactive_streak_len"] = 35
    frame.loc[frame["player_id"] == 12, ["minutes_p50", "minutes_p10", "minutes_p90", "play_prob"]] = [0.0, 0.0, 16.0, 0.01]

    cfg = OccupancySparseConfig(
        dynamic_k_bounds_enabled=True,
        dnp_suppression_enabled=False,
        archetype_shortage_enabled=True,
        archetype_out_count_threshold=1,
        archetype_out_inactive_streak_max=10,
        archetype_play_prob_floor=0.08,
        archetype_seed_p90_min=8.0,
    )
    out, diag = apply_occupancy_sparse_allocation(frame, config=cfg)
    repl = out.loc[out["player_id"] == 12].iloc[0]

    assert float(pd.to_numeric(repl["minutes_occ"], errors="coerce") or 0.0) == pytest.approx(0.0, abs=1e-9)
    assert int(pd.to_numeric(diag["archetype_out_big"], errors="coerce").fillna(0).iloc[0]) == 0
    assert int(pd.to_numeric(diag["archetype_shortage_active"], errors="coerce").fillna(0).iloc[0]) == 0
    assert int(pd.to_numeric(diag["n_archetype_seeded"], errors="coerce").fillna(0).iloc[0]) == 0


def test_occupancy_sparse_archetype_shortage_rescues_dnp_suppressed_replacement() -> None:
    frame = _deep_rotation_frame()
    frame["inactive_streak_len"] = 0
    frame["active_but_dnp_rate_last10"] = 0.0
    frame["consecutive_active_dnp"] = 0

    # Starter big out; replacement big has strong baseline minutes but high DNP signal.
    frame.loc[frame["player_id"] == 1, ["status", "is_out"]] = ["OUT", 1]
    frame.loc[frame["player_id"] == 1, "inactive_streak_len"] = 1
    frame.loc[frame["player_id"] == 12, ["minutes_p50", "minutes_p10", "minutes_p90", "play_prob"]] = [14.0, 8.0, 20.0, 0.20]
    frame.loc[frame["player_id"] == 12, "active_but_dnp_rate_last10"] = 0.80

    cfg_disabled = OccupancySparseConfig(
        dynamic_k_bounds_enabled=True,
        dnp_suppression_enabled=True,
        dnp_suppression_relax_in_injury_regime=False,
        dnp_rate_threshold=0.35,
        archetype_shortage_enabled=False,
    )
    out_disabled, _ = apply_occupancy_sparse_allocation(frame, config=cfg_disabled)
    repl_disabled = out_disabled.loc[out_disabled["player_id"] == 12].iloc[0]
    assert float(pd.to_numeric(repl_disabled["minutes_occ"], errors="coerce") or 0.0) == pytest.approx(0.0, abs=1e-9)

    cfg_enabled = OccupancySparseConfig(
        dynamic_k_bounds_enabled=True,
        dnp_suppression_enabled=True,
        dnp_suppression_relax_in_injury_regime=False,
        dnp_rate_threshold=0.35,
        archetype_shortage_enabled=True,
        archetype_out_count_threshold=1,
        archetype_out_inactive_streak_max=10,
        archetype_play_prob_floor=0.08,
        archetype_seed_p90_min=8.0,
        archetype_seed_max_players=2,
    )
    out_enabled, diag_enabled = apply_occupancy_sparse_allocation(frame, config=cfg_enabled)
    repl_enabled = out_enabled.loc[out_enabled["player_id"] == 12].iloc[0]

    assert float(pd.to_numeric(repl_enabled["minutes_occ"], errors="coerce") or 0.0) > 0.0
    assert int(pd.to_numeric(repl_enabled["eligible_flag_occ"], errors="coerce")) == 1
    assert int(pd.to_numeric(diag_enabled["archetype_shortage_active"], errors="coerce").fillna(0).iloc[0]) == 1
    assert int(pd.to_numeric(diag_enabled["n_archetype_dnp_rescued"], errors="coerce").fillna(0).iloc[0]) >= 1
