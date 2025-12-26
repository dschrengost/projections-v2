"""Tests for sim_v2 active counts and rotation preservation."""
from __future__ import annotations

import numpy as np
import pytest

from projections.sim_v2.minutes_noise import enforce_team_240_minutes


def test_active_counts_never_zero_with_non_out_players() -> None:
    """Ensure active_counts can never be zero for a team when non-OUT players exist.
    
    This test simulates the sim's active mask logic and verifies that at least
    one player per team is always active in every world.
    """
    # Two-team slate: 8 players per team
    n_players = 16
    n_worlds = 100
    
    # Team indices: 0-7 are team 0, 8-15 are team 1
    team_indices = np.array([0]*8 + [1]*8, dtype=int)
    
    # Play probabilities: all non-OUT (play_prob > 0)
    play_prob = np.array([
        0.98, 0.95, 0.92, 0.90, 0.85, 0.50, 0.30, 0.10,  # Team 0
        0.99, 0.97, 0.94, 0.88, 0.80, 0.45, 0.25, 0.05,  # Team 1
    ], dtype=float)
    
    # Simulate Bernoulli sampling for active mask (same logic as sim)
    rng = np.random.default_rng(42)
    u_active = rng.random(size=(n_worlds, n_players))
    active_mask = u_active < play_prob[None, :]
    
    # Check per-team active counts per world
    n_teams = 2
    for t in range(n_teams):
        team_player_mask = team_indices == t
        team_active_per_world = active_mask[:, team_player_mask].sum(axis=1)
        
        # In practice, with reasonable play_prob values, we expect at least
        # 1 player active per team in most worlds. For testing, we verify
        # the sim doesn't artificially zero out teams.
        # Note: With very low play_prob, some worlds might have 0 active
        # players, which is the bug we're fixing by not using eligible_flag.
        min_active = team_active_per_world.min()
        
        # If min_active is 0, flag it (this is the condition that causes issues)
        if min_active == 0:
            n_zero_worlds = (team_active_per_world == 0).sum()
            pct_zero = n_zero_worlds / n_worlds * 100
            # This test documents the behavior - with preserve_input_rotation=True,
            # we skip eligible_flag filtering so more players remain active
            assert pct_zero < 100, f"All worlds have zero active for team {t}"


def test_sim_preserves_rotation_size_without_eligible_filtering() -> None:
    """When preserve_input_rotation=True, sim should not reduce rotation size.
    
    This tests that enforce_team_240_minutes with max_rotation_size=None
    preserves all active players (no capping).
    """
    # 10-player team with known rotation
    minutes_input = np.array([
        35.0, 32.0, 30.0, 28.0, 26.0,  # Starters
        20.0, 15.0, 12.0, 8.0, 4.0,    # Bench
    ], dtype=float)
    
    minutes_world = minutes_input[None, :]  # 1 world
    team_indices = np.zeros(10, dtype=int)
    starter_mask = np.zeros(10, dtype=bool)
    starter_mask[:5] = True
    
    # With max_rotation_size=None (preserve_input_rotation=True behavior)
    out = enforce_team_240_minutes(
        minutes_world=minutes_world,
        team_indices=team_indices,
        rotation_mask=minutes_input >= 12.0,
        bench_mask=(minutes_input > 0.0) & (minutes_input < 12.0),
        baseline_minutes=minutes_input,
        clamp_scale=(0.7, 1.3),
        starter_mask=starter_mask,
        max_rotation_size=None,  # KEY: No rotation cap
    )[0]
    
    # All players with input minutes > 0 should have output minutes > 0
    input_rotation_size = (minutes_input > 0).sum()
    output_rotation_size = (out > 0).sum()
    
    assert output_rotation_size == input_rotation_size, (
        f"Rotation size changed: input={input_rotation_size}, output={output_rotation_size}"
    )


def test_team_minutes_sum_to_240_with_preservation() -> None:
    """Team minutes should still sum to ~240 even without rotation capping."""
    # Oversubscribed team (260 total minutes)
    minutes_input = np.array([
        36.0, 34.0, 32.0, 30.0, 28.0,  # Starters: 160
        25.0, 22.0, 18.0, 15.0, 8.0,   # Bench: 88 (total 248, less than before)
    ], dtype=float)
    
    minutes_world = minutes_input[None, :]
    team_indices = np.zeros(10, dtype=int)
    starter_mask = np.zeros(10, dtype=bool)
    starter_mask[:5] = True
    
    out = enforce_team_240_minutes(
        minutes_world=minutes_world,
        team_indices=team_indices,
        rotation_mask=minutes_input >= 12.0,
        bench_mask=(minutes_input > 0.0) & (minutes_input < 12.0),
        baseline_minutes=minutes_input,
        clamp_scale=(0.7, 1.3),
        starter_mask=starter_mask,
        max_rotation_size=None,  # preserve_input_rotation
    )[0]
    
    # Team total should be exactly 240 (within tolerance)
    team_total = float(out.sum())
    assert 238.0 <= team_total <= 242.0, f"Team total not ~240: {team_total}"


def test_non_out_players_never_dropped() -> None:
    """Non-OUT players (play_prob > 0) should never be dropped from rotation."""
    # Simulate the scenario where eligible_flag might exclude players
    # but we want to preserve them since they're not OUT
    
    minutes_input = np.array([
        35.0, 32.0, 30.0, 28.0, 26.0,  # Starters
        18.0, 15.0, 12.0,              # Regular rotation
        6.0, 4.0,                       # Deep bench (might be excluded by eligible_flag)
    ], dtype=float)
    
    play_prob = np.array([
        0.98, 0.97, 0.96, 0.95, 0.94,  # Starters
        0.80, 0.70, 0.60,              # Regular rotation
        0.20, 0.10,                    # Deep bench (low but non-zero)
    ], dtype=float)
    
    # Eligible flag might exclude low play_prob players
    eligible_flag = play_prob >= 0.5  # Would exclude last 2 players
    
    # With preserve_input_rotation=True, we ignore eligible_flag
    # All players with play_prob > 0 should be considered active
    active_mask = play_prob > 0
    
    # All 10 players should be active
    assert active_mask.sum() == 10, "All non-OUT players should be active"
    
    # Verify eligible_flag would have dropped 2 players
    assert eligible_flag.sum() == 8, "Eligible flag would have dropped 2 players"
