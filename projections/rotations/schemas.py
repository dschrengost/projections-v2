from __future__ import annotations

ROT_V1_SCHEMA_VERSION = "rot_v1.0"
ROT_EVAL_V1_SCHEMA_VERSION = "rot_eval_v1.0"

LINEUP_COLS: tuple[str, ...] = tuple(f"lineup_p{i}" for i in range(1, 6))

ROTATION_EVENTS_COLS: tuple[str, ...] = (
    "season_id",
    "game_id",
    "team_id",
    "opponent_team_id",
    "is_home",
    "segment_idx",
    "period",
    "start_clock_sec",
    "end_clock_sec",
    "duration_sec",
    *LINEUP_COLS,
    "raw_ref",
)

ROTATION_LABELS_COLS: tuple[str, ...] = (
    "game_id",
    "team_id",
    "player_id",
    "minutes_actual",
    "played_ge_1",
    "played_ge_5",
    "starter_actual",
    "regime_label",
)
