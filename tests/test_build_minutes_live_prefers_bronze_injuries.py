"""Regression test: live minutes features should not lose injuries due to silver-only snapshots.

We observed a production failure mode where `silver/injuries_snapshot` only contained a
post-tip refresh for an early game, so as-of filtering dropped *all* injury rows and
the slate behaved as if no one was injured.

The live builder now prefers bronze `injuries_raw` (which contains multiple snapshots
throughout the day) and selects the latest snapshot <= tip/run_as_of_ts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import projections.cli.build_minutes_live as build_minutes_live
from typer.testing import CliRunner


def test_build_minutes_live_uses_bronze_injuries_when_available(tmp_path: Path, monkeypatch) -> None:
    run_id = "pytest_bronze_injuries"

    # Avoid pointer publishing in tests.
    monkeypatch.setenv("PROJECTIONS_SKIP_POINTER_WRITES", "1")

    # Minimal history labels (one prior game) so history features can be computed.
    history = pd.DataFrame(
        {
            "game_id": [999],
            "player_id": [101],
            "team_id": [1610612751],
            "player_name": ["Michael Porter Jr."],
            "season": ["2025-26"],
            "game_date": ["2025-12-30"],
            "minutes": [30.0],
            "starter_flag": [1],
            "starter_flag_label": [1],
            "source": ["test"],
            "label_frozen_ts": [pd.Timestamp("2025-12-31T00:00:00Z")],
        }
    )

    # Schedule must include both the historical game and the live slate game.
    schedule = pd.DataFrame(
        {
            "game_id": [999, 22500471],
            "season": ["2025-26", "2025-26"],
            "game_date": ["2025-12-30", "2026-01-01"],
            "tip_ts": ["2025-12-30T23:00:00Z", "2026-01-01T23:00:00Z"],
            "home_team_id": [1610612751, 1610612751],
            "away_team_id": [1610612745, 1610612745],
        }
    )

    odds = pd.DataFrame(
        {
            "game_id": [22500471],
            "home_line": [-1.0],
            "total": [225.0],
            "as_of_ts": ["2026-01-01T22:50:00Z"],
        }
    )

    # Roster snapshot for the live slate game.
    roster = pd.DataFrame(
        {
            "game_id": [22500471, 22500471],
            "team_id": [1610612751, 1610612751],
            "game_date": ["2026-01-01", "2026-01-01"],
            "player_id": [101, 102],
            "player_name": ["Michael Porter Jr.", "Nic Claxton"],
            "active_flag": [True, True],
            "listed_pos": ["SF", "C"],
            "as_of_ts": ["2026-01-01T22:00:00Z", "2026-01-01T22:00:00Z"],
            # Include lineup metadata columns expected by the live depth join.
            "lineup_status": [pd.NA, pd.NA],
            "is_projected_starter": [pd.NA, pd.NA],
            "is_confirmed_starter": [pd.NA, pd.NA],
        }
    )

    # Bronze injuries contains a pre-tip snapshot marking player_id=101 OUT.
    bronze_injuries = pd.DataFrame(
        {
            "game_id": [22500471],
            "player_id": [101],
            "team_id": [1610612751],
            "player_name": ["Porter Jr., Michael"],
            "status": ["OUT"],
            "restriction_flag": [False],
            "ramp_flag": [False],
            "games_since_return": [0],
            "days_since_return": [0],
            "as_of_ts": ["2026-01-01T20:00:00Z"],
            "ingested_ts": ["2026-01-01T20:00:00Z"],
            "source": ["nba.com"],
            "source_row_id": ["x"],
            "status_raw": ["Out"],
            "notes_raw": [""],
            "report_date": ["2026-01-01"],
        }
    )

    def fake_read_bronze_day(*args, **kwargs):  # noqa: ANN001
        return bronze_injuries.copy()

    def fake_load_table(default_dir: Path, override: Path | None):  # noqa: ANN001
        target = override or default_dir
        s = str(target)
        if "schedule" in s:
            return schedule.copy()
        if "odds_snapshot" in s:
            return odds.copy()
        if "roster_nightly" in s:
            return roster.copy()
        # Silver injuries_snapshot should not be required when bronze has rows.
        if "injuries_snapshot" in s:
            return pd.DataFrame()
        return pd.DataFrame()

    monkeypatch.setattr(build_minutes_live.bronze_storage, "read_bronze_day", fake_read_bronze_day)
    monkeypatch.setattr(build_minutes_live, "_load_table", fake_load_table)
    monkeypatch.setattr(build_minutes_live, "_load_label_sources", lambda **_: (pd.DataFrame(), "mock"))
    monkeypatch.setattr(
        build_minutes_live,
        "_load_label_history",
        lambda *_, **__: history.copy(),
    )

    out_root = tmp_path / "live_features"
    runner = CliRunner()
    result = runner.invoke(
        build_minutes_live.app,
        [
            "--date",
            "2026-01-01",
            "--run-as-of-ts",
            "2026-01-01T22:59:59",
            "--run-id",
            run_id,
            "--season-start",
            "2025",
            "--data-root",
            str(tmp_path),
            "--out-root",
            str(out_root),
        ],
    )
    assert result.exit_code == 0, result.output

    features_path = out_root / "2026-01-01" / f"run={run_id}" / "features.parquet"
    assert features_path.exists()
    features = pd.read_parquet(features_path)
    mpj = features[(features["game_id"] == 22500471) & (features["player_id"] == 101)].iloc[0]
    assert int(mpj["injury_snapshot_missing"]) == 0
    assert str(mpj["status"]).upper() == "OUT"
    assert int(mpj["is_out"]) == 1
