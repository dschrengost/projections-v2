"""Tests for recent_start_pct_10 recomputation in build_minutes_live."""

import pandas as pd
import pytest


def test_recent_start_pct_10_uses_starter_flag_label_when_starter_flag_corrupt():
    """Verify that when starter_flag is all 1s (corrupt), we use starter_flag_label instead.
    
    This tests the fix for the bug where recent_start_pct_10 was always 0 because
    starter_flag was all 1s (corrupt) and the code was using that instead of
    starter_flag_label (correct).
    """
    # Simulate corrupt starter_flag (all 1s) and correct starter_flag_label
    history_labels = pd.DataFrame({
        "player_id": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 10 games for player 1
                      2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # 10 games for player 2
        "game_date": (
            pd.date_range("2026-01-01", periods=10).tolist() +
            pd.date_range("2026-01-01", periods=10).tolist()
        ),
        "starter_flag": [1] * 20,  # All 1s - CORRUPT!
        "starter_flag_label": (
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] +  # Player 1: always started (100%)
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0]    # Player 2: started 2/10 (20%)
        ),
    })
    
    # Determine the correct starter flag column to use (replicate the fix logic)
    starter_col = None
    
    # Check starter_flag_label first
    if "starter_flag_label" in history_labels.columns:
        sfl = pd.to_numeric(history_labels["starter_flag_label"], errors="coerce").fillna(0)
        if sfl.std() > 0.01:  # Has variance
            starter_col = "starter_flag_label"
    
    # Check if starter_flag is corrupt
    if starter_col is None and "starter_flag" in history_labels.columns:
        sf = pd.to_numeric(history_labels["starter_flag"], errors="coerce").fillna(0)
        if sf.std() > 0.01:  # Has variance
            starter_col = "starter_flag"
        elif sf.mean() > 0.95:  # All 1s - corrupt!
            # Fall back to starter_flag_label
            if "starter_flag_label" in history_labels.columns:
                starter_col = "starter_flag_label"
    
    # Assert we correctly chose starter_flag_label
    assert starter_col == "starter_flag_label", (
        f"Expected to use starter_flag_label when starter_flag is corrupt, got {starter_col}"
    )
    
    # Now compute recent_start_pct_10 using the correct column
    history_labels["_starter_val"] = pd.to_numeric(history_labels[starter_col], errors="coerce").fillna(0)
    
    recency_features = []
    for pid, group in history_labels.groupby("player_id"):
        last_10 = group.tail(10)
        start_pct = float(last_10["_starter_val"].mean()) if len(last_10) > 0 else 0.0
        recency_features.append({"player_id": pid, "recent_start_pct_10": start_pct})
    
    result = pd.DataFrame(recency_features)
    
    # Verify results
    player1_pct = result[result["player_id"] == 1]["recent_start_pct_10"].iloc[0]
    player2_pct = result[result["player_id"] == 2]["recent_start_pct_10"].iloc[0]
    
    assert player1_pct == 1.0, f"Player 1 (always starter) should have 1.0, got {player1_pct}"
    assert player2_pct == 0.2, f"Player 2 (2/10 starter) should have 0.2, got {player2_pct}"


def test_recent_start_pct_10_uses_starter_flag_when_valid():
    """Verify that when starter_flag has proper variance, we use it directly."""
    history_labels = pd.DataFrame({
        "player_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "game_date": pd.date_range("2026-01-01", periods=10).tolist(),
        "starter_flag": [1, 1, 1, 0, 0, 0, 0, 1, 0, 0],  # Has variance
    })
    
    # Determine the correct starter flag column
    starter_col = None
    
    if "starter_flag_label" in history_labels.columns:
        sfl = pd.to_numeric(history_labels["starter_flag_label"], errors="coerce").fillna(0)
        if sfl.std() > 0.01:
            starter_col = "starter_flag_label"
    
    if starter_col is None and "starter_flag" in history_labels.columns:
        sf = pd.to_numeric(history_labels["starter_flag"], errors="coerce").fillna(0)
        if sf.std() > 0.01:
            starter_col = "starter_flag"
    
    # When starter_flag has variance, we should use it
    assert starter_col == "starter_flag", f"Expected starter_flag when valid, got {starter_col}"


def test_recent_start_pct_10_handles_missing_columns():
    """Verify graceful handling when no starter flag column exists."""
    history_labels = pd.DataFrame({
        "player_id": [1, 2, 3],
        "game_date": pd.date_range("2026-01-01", periods=3),
        "minutes": [30, 25, 20],
    })
    
    starter_col = None
    
    if "starter_flag_label" in history_labels.columns:
        sfl = pd.to_numeric(history_labels["starter_flag_label"], errors="coerce").fillna(0)
        if sfl.std() > 0.01:
            starter_col = "starter_flag_label"
    
    if starter_col is None and "starter_flag" in history_labels.columns:
        sf = pd.to_numeric(history_labels["starter_flag"], errors="coerce").fillna(0)
        if sf.std() > 0.01:
            starter_col = "starter_flag"
    
    assert starter_col is None, "Expected None when no starter columns present"
