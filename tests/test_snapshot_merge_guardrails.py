from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.etl.injuries import _merge_with_existing_snapshot as merge_injuries_snapshot
from projections.etl.odds import _merge_with_existing_snapshot as merge_odds_snapshot
from projections.minutes_v1.schemas import (
    INJURIES_SNAPSHOT_SCHEMA,
    ODDS_SNAPSHOT_SCHEMA,
    enforce_schema,
)


def _odds_snapshot_frame(game_id: int) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "game_id": [game_id],
            "as_of_ts": ["2026-02-10T12:00:00Z"],
            "spread_home": [-2.5],
            "total": [228.5],
            "book": ["consensus"],
            "book_pref": [pd.NA],
            "ingested_ts": ["2026-02-10T12:00:05Z"],
            "source": ["oddstrader"],
        }
    )
    return enforce_schema(frame, ODDS_SNAPSHOT_SCHEMA, allow_missing_optional=True)


def _injuries_snapshot_frame(
    *,
    game_id: int,
    player_id: int,
    status: str,
    as_of_ts: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "game_id": [game_id],
            "player_id": [player_id],
            "as_of_ts": [as_of_ts],
            "status": [status],
            "restriction_flag": [False],
            "ramp_flag": [False],
            "games_since_return": [pd.NA],
            "days_since_return": [pd.NA],
            "ingested_ts": [as_of_ts],
            "source": ["nba-injury-report"],
            "selection_rule": ["latest"],
            "snapshot_missing": [0],
        }
    )
    return enforce_schema(frame, INJURIES_SNAPSHOT_SCHEMA)


def test_odds_merge_refuses_unreadable_existing_without_override(tmp_path: Path) -> None:
    silver_path = tmp_path / "odds_snapshot.parquet"
    silver_path.write_text("not-a-parquet-file", encoding="utf-8")
    incoming = _odds_snapshot_frame(22500001)

    with pytest.raises(RuntimeError, match="refusing overwrite because existing snapshot cannot be read"):
        merge_odds_snapshot(
            incoming,
            silver_path=silver_path,
            allow_snapshot_regression=False,
        )


def test_odds_merge_allows_unreadable_existing_with_override(tmp_path: Path) -> None:
    silver_path = tmp_path / "odds_snapshot.parquet"
    silver_path.write_text("not-a-parquet-file", encoding="utf-8")
    incoming = _odds_snapshot_frame(22500001)

    merged = merge_odds_snapshot(
        incoming,
        silver_path=silver_path,
        allow_snapshot_regression=True,
    )
    assert len(merged) == 1
    assert int(merged.iloc[0]["game_id"]) == 22500001


def test_injuries_merge_prefers_latest_for_duplicate_keys(tmp_path: Path) -> None:
    silver_path = tmp_path / "injuries_snapshot.parquet"

    existing = _injuries_snapshot_frame(
        game_id=22500001,
        player_id=123,
        status="OUT",
        as_of_ts="2026-02-10T11:00:00Z",
    )
    existing.to_parquet(silver_path, index=False)

    incoming = pd.concat(
        [
            _injuries_snapshot_frame(
                game_id=22500001,
                player_id=123,
                status="AVAILABLE",
                as_of_ts="2026-02-10T12:00:00Z",
            ),
            _injuries_snapshot_frame(
                game_id=22500002,
                player_id=456,
                status="QUESTIONABLE",
                as_of_ts="2026-02-10T12:00:00Z",
            ),
        ],
        ignore_index=True,
    )
    incoming = enforce_schema(incoming, INJURIES_SNAPSHOT_SCHEMA)

    merged = merge_injuries_snapshot(
        incoming,
        silver_path=silver_path,
        allow_snapshot_regression=False,
    )
    merged = merged.sort_values(["game_id", "player_id"]).reset_index(drop=True)
    assert len(merged) == 2
    assert merged.loc[0, "status"] == "AVAILABLE"
    assert merged.loc[1, "status"] == "QUESTIONABLE"
