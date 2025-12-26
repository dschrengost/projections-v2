"""Schema definition for the training dataset pipeline.

DATASET CONTRACT:
-----------------
1. Primary Key: (game_id, team_id, player_id)
   - Guaranteed unique per partition (season/date/asof).
   - Enforced by builder.assert_unique_primary_key() with fail-fast.
   - Duplicates cause ValueError unless --allow-dedup is passed.

2. Label Definition:
   - share_label = actual_minutes / divisor
   - label_mode controls the divisor:
     * REG240 (default): divisor = 240.0 (shares may sum >1.0 for OT games)
     * TEAM_TOTAL_ACTUAL: divisor = sum(actual_minutes) per team-game (OT-aware)
   
3. OT Semantics:
   - REG240 mode: OT minutes still divided by 240, so shares can sum >1.0.
     This is standard for DFS since platforms use 240-minute projections.
   - TEAM_TOTAL_ACTUAL mode: shares always sum to exactly 1.0.
     Useful for diagnostics and understanding OT impact.
   - OT games are ~5% of regular season games.

4. Normalization Keys:
   - All model outputs normalized by Group[game_id, team_id].
   - Predicted shares sum to 1.0 within this group after normalization.
   - Do NOT normalize by team_id alone (would span multiple games).

5. Integrity Constraints:
   - No row duplication (enforced by builder fail-fast assertion).
   - No salary features (DK/FD) included (removed to prevent duplication).
   - Builder raises ValueError on duplicate primary keys (not silent dedup).
"""

import pandera as pa
from pandera.typing import DateTime, Series

class TrainingFeaturesSchema(pa.DataFrameModel):
    """Schema for the canonical training features dataset."""

    # Keys
    season: Series[int] = pa.Field(coerce=True)
    game_date: Series[DateTime] = pa.Field(coerce=True)
    game_id: Series[int] = pa.Field(coerce=True)
    team_id: Series[int] = pa.Field(coerce=True)
    player_id: Series[int] = pa.Field(coerce=True)
    
    # Timestamps
    feature_as_of_ts: Series[DateTime] = pa.Field(coerce=True)  # The snapshot time for features
    tip_ts: Series[DateTime] = pa.Field(coerce=True)  # Game tip time
    
    # Roster / Context
    player_name: Series[str] = pa.Field()
    team_tricode: Series[str] = pa.Field(nullable=True)
    opponent_team_id: Series[int] = pa.Field(coerce=True)
    is_home: Series[bool] = pa.Field(coerce=True)
    
    # Features (Point-in-Time)

    
    # Odds
    spread_home: Series[float] = pa.Field(nullable=True)
    total: Series[float] = pa.Field(nullable=True)
    implied_team_score: Series[float] = pa.Field(nullable=True)
    
    # Injuries
    injury_status: Series[str] = pa.Field(nullable=True)
    
    # Derived / Computed
    is_confirmed_starter: Series[bool] = pa.Field(coerce=True)
    
    # Labels (Post-Facto)
    minutes: Series[float] = pa.Field(nullable=True, ge=0)
    played_flag: Series[int] = pa.Field(isin=[0, 1], nullable=True)
    
    # Detailed Stats Labels
    pts: Series[float] = pa.Field(nullable=True)
    reb: Series[float] = pa.Field(nullable=True)
    ast: Series[float] = pa.Field(nullable=True)
    stl: Series[float] = pa.Field(nullable=True)
    blk: Series[float] = pa.Field(nullable=True)
    tov: Series[float] = pa.Field(nullable=True)
    fg3m: Series[float] = pa.Field(nullable=True)
    dk_fpts_actual: Series[float] = pa.Field(nullable=True)
    
    class Config:
        strict = True
        coerce = True
