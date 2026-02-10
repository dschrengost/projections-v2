"""Schedule-aware coverage checks for ETL snapshots."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GameCoverageStats:
    scheduled_games: int
    observed_games: int
    overlap_games: int

    @property
    def coverage_rate(self) -> float | None:
        if self.scheduled_games <= 0:
            return None
        return float(self.overlap_games) / float(self.scheduled_games)

    @property
    def off_schedule_games(self) -> int:
        return max(self.observed_games - self.overlap_games, 0)


def _extract_game_ids(df: pd.DataFrame, col: str) -> set[int]:
    if df.empty or col not in df.columns:
        return set()
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return set()
    return set(values.astype("int64").tolist())


def compute_game_coverage(
    *,
    schedule_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    schedule_game_col: str = "game_id",
    observed_game_col: str = "game_id",
) -> GameCoverageStats:
    scheduled = _extract_game_ids(schedule_df, schedule_game_col)
    observed = _extract_game_ids(observed_df, observed_game_col)
    overlap = scheduled & observed
    return GameCoverageStats(
        scheduled_games=len(scheduled),
        observed_games=len(observed),
        overlap_games=len(overlap),
    )


def format_game_coverage(dataset_name: str, stats: GameCoverageStats) -> str:
    if stats.scheduled_games <= 0:
        return f"[{dataset_name}] no scheduled games in window; empty coverage is expected."
    coverage = (stats.coverage_rate or 0.0) * 100.0
    return (
        f"[{dataset_name}] schedule coverage: {stats.overlap_games}/{stats.scheduled_games} "
        f"({coverage:.1f}%), observed_games={stats.observed_games}, "
        f"off_schedule_games={stats.off_schedule_games}"
    )


def enforce_game_coverage(
    *,
    dataset_name: str,
    stats: GameCoverageStats,
    strict: bool,
    min_overlap_games: int = 1,
) -> None:
    if not strict or stats.scheduled_games <= 0:
        return
    if stats.overlap_games >= min_overlap_games:
        return
    raise RuntimeError(
        f"[{dataset_name}] no overlapping games with schedule "
        f"(scheduled={stats.scheduled_games}, observed={stats.observed_games}, "
        f"overlap={stats.overlap_games})."
    )
