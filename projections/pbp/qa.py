from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd


def infer_game_seconds(max_period: int) -> int:
    if max_period <= 4:
        return 720 * max_period
    return 4 * 720 + 300 * (max_period - 4)


@dataclass
class QaOutputs:
    report: dict[str, Any]
    failures: pd.DataFrame


def run_qa_gates(
    stints: pd.DataFrame,
    *,
    season_id: str,
    run_id: str,
    schema_version: str,
    tolerance_sec: int = 1,
) -> QaOutputs:
    required = [
        "game_id",
        "stint_id",
        "period",
        "duration_sec",
    ] + [f"home_p{i}" for i in range(1, 6)] + [f"away_p{i}" for i in range(1, 6)]
    missing = [c for c in required if c not in stints.columns]
    if missing:
        raise ValueError(f"stints missing required columns for QA: {missing}")

    home_cols = [f"home_p{i}" for i in range(1, 6)]
    away_cols = [f"away_p{i}" for i in range(1, 6)]

    failures: list[dict[str, Any]] = []
    zero_duration_count = 0
    max_zero_duration_stints_in_game = 0
    total_stints = len(stints)

    for game_id, g in stints.groupby("game_id", sort=True):
        game_fail_reasons: set[str] = set()

        # Lineup size and uniqueness.
        for _, row in g.iterrows():
            if any(pd.isna(row[c]) for c in home_cols + away_cols):
                game_fail_reasons.add("missing_lineup_player_id")
                failures.append(
                    {
                        "season_id": season_id,
                        "game_id": game_id,
                        "reason": "missing_lineup_player_id",
                        "details": json.dumps({"stint_id": int(row["stint_id"])}),
                    }
                )
                break
            home = [int(row[c]) for c in home_cols]
            away = [int(row[c]) for c in away_cols]

            if len(set(home)) != 5 or len(set(away)) != 5:
                game_fail_reasons.add("non_unique_lineup")
                failures.append(
                    {
                        "season_id": season_id,
                        "game_id": game_id,
                        "reason": "non_unique_lineup",
                        "details": json.dumps(
                            {
                                "stint_id": int(row["stint_id"]),
                                "home": home,
                                "away": away,
                            }
                        ),
                    }
                )
                break

            if set(home) & set(away):
                game_fail_reasons.add("player_on_both_teams")
                failures.append(
                    {
                        "season_id": season_id,
                        "game_id": game_id,
                        "reason": "player_on_both_teams",
                        "details": json.dumps(
                            {
                                "stint_id": int(row["stint_id"]),
                                "overlap": sorted(list(set(home) & set(away))),
                            }
                        ),
                    }
                )
                break

        # Duration checks.
        if (g["duration_sec"] < 0).any():
            game_fail_reasons.add("negative_duration")
            bad = g[g["duration_sec"] < 0][["stint_id", "duration_sec"]].head(20).to_dict(orient="records")
            failures.append(
                {
                    "season_id": season_id,
                    "game_id": game_id,
                    "reason": "negative_duration",
                    "details": json.dumps({"examples": bad}),
                }
            )

        zero_stints_in_game = int((g["duration_sec"] == 0).sum())
        zero_duration_count += zero_stints_in_game
        max_zero_duration_stints_in_game = max(max_zero_duration_stints_in_game, zero_stints_in_game)

        # Coverage check (per-team is the same for these stints, but validate against expected game_seconds).
        max_period = int(g["period"].max())
        expected = infer_game_seconds(max_period)
        actual = int(g["duration_sec"].sum())
        if abs(actual - expected) > tolerance_sec:
            game_fail_reasons.add("duration_sum_mismatch")
            failures.append(
                {
                    "season_id": season_id,
                    "game_id": game_id,
                    "reason": "duration_sum_mismatch",
                    "details": json.dumps(
                        {
                            "expected_game_seconds": expected,
                            "actual_sum_duration_sec": actual,
                            "max_period": max_period,
                            "tolerance_sec": tolerance_sec,
                        }
                    ),
                }
            )

    failures_df = pd.DataFrame(failures, columns=["season_id", "game_id", "reason", "details"])
    failed_games = sorted(failures_df["game_id"].unique().tolist()) if len(failures_df) else []

    games_total = int(stints["game_id"].nunique())
    games_failed = len(failed_games)
    games_passed = games_total - games_failed
    pass_rate = (games_passed / games_total) if games_total else 0.0

    reason_counts = Counter(failures_df["reason"].tolist()) if len(failures_df) else Counter()
    top_reasons = dict(reason_counts.most_common(20))

    durations = stints["duration_sec"].astype(int) if len(stints) else pd.Series([], dtype="int64")
    if len(durations):
        stint_duration_summary = {
            "min": int(durations.min()),
            "p50": int(durations.quantile(0.50, interpolation="nearest")),
            "p95": int(durations.quantile(0.95, interpolation="nearest")),
            "max": int(durations.max()),
        }
    else:
        stint_duration_summary = {
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
        }

    report: dict[str, Any] = {
        "schema_version": schema_version,
        "season_id": season_id,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "games_total": games_total,
            "games_passed": games_passed,
            "games_failed": games_failed,
            "pass_rate": pass_rate,
            "stints_total": total_stints,
            "zero_duration_stints_total": int(zero_duration_count),
        },
        "trust_metrics": {
            "zero_duration_stints_total": int(zero_duration_count),
            "max_zero_duration_stints_in_game": int(max_zero_duration_stints_in_game),
            "stint_duration_summary": stint_duration_summary,
        },
        "top_failure_reasons": top_reasons,
        "failed_games": failed_games[:200],
    }

    return QaOutputs(report=report, failures=failures_df)
