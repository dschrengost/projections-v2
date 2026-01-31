from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from projections.rotations.schemas import LINEUP_COLS


def infer_game_seconds(max_period: int) -> int:
    if int(max_period) <= 4:
        return 720 * int(max_period)
    return 4 * 720 + 300 * (int(max_period) - 4)


@dataclass(frozen=True)
class QaOutputs:
    report: dict[str, Any]
    failures: pd.DataFrame


def run_qa_gates(
    rotation_events: pd.DataFrame,
    *,
    season_id: str,
    run_id: str,
    schema_version: str,
    tolerance_sec: int = 0,
) -> QaOutputs:
    required = [
        "season_id",
        "game_id",
        "team_id",
        "segment_idx",
        "period",
        "start_clock_sec",
        "duration_sec",
        *LINEUP_COLS,
    ]
    missing = [c for c in required if c not in rotation_events.columns]
    if missing:
        raise ValueError(f"rotation_events missing required columns for QA: {missing}")

    failures: list[dict[str, Any]] = []
    zero_duration_count = 0
    max_zero_duration_segments_in_team_game = 0
    total_segments = int(len(rotation_events))

    for (team_id, game_id), g in rotation_events.groupby(["team_id", "game_id"], sort=True):
        # Deterministic ordering: period asc, clock desc, segment_idx asc.
        expected = g.sort_values(
            ["period", "start_clock_sec", "segment_idx"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        if expected["segment_idx"].tolist() != g["segment_idx"].tolist():
            failures.append(
                {
                    "season_id": season_id,
                    "game_id": str(game_id),
                    "team_id": int(team_id),
                    "reason": "non_deterministic_ordering",
                    "details": json.dumps({"expected_sort": ["period", "start_clock_sec(desc)", "segment_idx"]}),
                }
            )

        # Segment index invariant: unique and contiguous from 0.
        seg = pd.to_numeric(g["segment_idx"], errors="coerce").astype("Int64")
        if seg.isna().any() or int(seg.nunique()) != int(len(seg)) or int(seg.min()) != 0 or int(seg.max()) != int(len(seg)) - 1:
            failures.append(
                {
                    "season_id": season_id,
                    "game_id": str(game_id),
                    "team_id": int(team_id),
                    "reason": "segment_idx_invalid",
                    "details": json.dumps(
                        {
                            "n_segments": int(len(seg)),
                            "nunique": int(seg.nunique()),
                            "min": None if seg.isna().all() else int(seg.min()),
                            "max": None if seg.isna().all() else int(seg.max()),
                        }
                    ),
                }
            )

        lineup = g.loc[:, list(LINEUP_COLS)]
        if lineup.isna().any().any():
            bad = g.loc[g[list(LINEUP_COLS)].isna().any(axis=1), ["segment_idx", *LINEUP_COLS]].head(5).to_dict(
                orient="records"
            )
            failures.append(
                {
                    "season_id": season_id,
                    "game_id": str(game_id),
                    "team_id": int(team_id),
                    "reason": "missing_lineup_player_id",
                    "details": json.dumps({"examples": bad}),
                }
            )

        # Uniqueness per segment.
        if not lineup.empty and not lineup.isna().any().any():
            unique_counts = lineup.astype("int64").nunique(axis=1)
            if (unique_counts != 5).any():
                bad_row = int(unique_counts[unique_counts != 5].index[0])
                example = g.loc[bad_row, ["segment_idx", *LINEUP_COLS]].to_dict()
                failures.append(
                    {
                        "season_id": season_id,
                        "game_id": str(game_id),
                        "team_id": int(team_id),
                        "reason": "non_unique_lineup",
                        "details": json.dumps({"example": example}),
                    }
                )

        # Duration checks.
        if (g["duration_sec"] < 0).any():
            bad = g[g["duration_sec"] < 0][["segment_idx", "duration_sec"]].head(20).to_dict(orient="records")
            failures.append(
                {
                    "season_id": season_id,
                    "game_id": str(game_id),
                    "team_id": int(team_id),
                    "reason": "negative_duration",
                    "details": json.dumps({"examples": bad}),
                }
            )

        zero_segments = int((g["duration_sec"] == 0).sum())
        zero_duration_count += zero_segments
        max_zero_duration_segments_in_team_game = max(max_zero_duration_segments_in_team_game, zero_segments)

        max_period = int(g["period"].max())
        expected = infer_game_seconds(max_period)
        actual = int(pd.to_numeric(g["duration_sec"], errors="coerce").fillna(0).sum())
        if abs(actual - expected) > int(tolerance_sec):
            failures.append(
                {
                    "season_id": season_id,
                    "game_id": str(game_id),
                    "team_id": int(team_id),
                    "reason": "duration_sum_mismatch",
                    "details": json.dumps(
                        {
                            "expected_game_seconds": expected,
                            "actual_sum_duration_sec": actual,
                            "max_period": max_period,
                            "tolerance_sec": int(tolerance_sec),
                        }
                    ),
                }
            )

    failures_df = pd.DataFrame(failures, columns=["season_id", "game_id", "team_id", "reason", "details"])

    games_total = int(rotation_events[["team_id", "game_id"]].drop_duplicates().shape[0])
    failed_team_games = (
        failures_df.dropna(subset=["team_id", "game_id"])[["team_id", "game_id"]].drop_duplicates()
        if len(failures_df)
        else pd.DataFrame(columns=["team_id", "game_id"])
    )
    team_games_failed = int(len(failed_team_games))
    team_games_passed = games_total - team_games_failed
    pass_rate = (team_games_passed / games_total) if games_total else 0.0

    reason_counts = Counter(failures_df["reason"].tolist()) if len(failures_df) else Counter()
    top_reasons = dict(reason_counts.most_common(20))

    durations = pd.to_numeric(rotation_events["duration_sec"], errors="coerce").dropna().astype(int)
    if len(durations):
        duration_summary = {
            "min": int(durations.min()),
            "p50": int(durations.quantile(0.50, interpolation="nearest")),
            "p95": int(durations.quantile(0.95, interpolation="nearest")),
            "max": int(durations.max()),
        }
    else:
        duration_summary = {"min": None, "p50": None, "p95": None, "max": None}

    report: dict[str, Any] = {
        "schema_version": schema_version,
        "season_id": season_id,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "team_games_total": games_total,
            "team_games_passed": team_games_passed,
            "team_games_failed": team_games_failed,
            "pass_rate": pass_rate,
            "segments_total": total_segments,
            "zero_duration_segments_total": int(zero_duration_count),
        },
        "trust_metrics": {
            "zero_duration_segments_total": int(zero_duration_count),
            "max_zero_duration_segments_in_team_game": int(max_zero_duration_segments_in_team_game),
            "duration_summary": duration_summary,
        },
        "top_failure_reasons": top_reasons,
    }

    return QaOutputs(report=report, failures=failures_df)
