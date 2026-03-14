from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

import projections.cli.build_minutes_live as build_minutes_live


def test_build_minutes_live_matches_rotowire_names_with_suffix_variants(tmp_path: Path, monkeypatch) -> None:
    run_id = "pytest_rotowire_out_roles"
    monkeypatch.setenv("PROJECTIONS_SKIP_POINTER_WRITES", "1")

    history = pd.DataFrame(
        {
            "game_id": [999],
            "player_id": [101],
            "team_id": [1610612751],
            "player_name": ["Player Out Jr."],
            "season": ["2025-26"],
            "game_date": ["2025-12-30"],
            "minutes": [28.0],
            "starter_flag": [1],
            "starter_flag_label": [1],
            "source": ["test"],
            "label_frozen_ts": [pd.Timestamp("2025-12-31T00:00:00Z")],
        }
    )

    schedule = pd.DataFrame(
        {
            "game_id": [999, 22500471],
            "season": ["2025-26", "2025-26"],
            "game_date": ["2025-12-30", "2026-01-01"],
            "tip_ts": ["2025-12-30T23:00:00Z", "2026-01-02T00:00:00Z"],
            "home_team_id": [1610612751, 1610612751],
            "away_team_id": [1610612745, 1610612745],
        }
    )

    odds = pd.DataFrame(
        {
            "game_id": [22500471],
            "home_line": [-2.0],
            "total": [226.5],
            "as_of_ts": ["2026-01-01T22:50:00Z"],
        }
    )

    roster = pd.DataFrame(
        {
            "game_id": [22500471, 22500471],
            "team_id": [1610612751, 1610612751],
            "game_date": ["2026-01-01", "2026-01-01"],
            "tip_ts": ["2026-01-02T00:00:00Z", "2026-01-02T00:00:00Z"],
            "player_id": [101, 102],
            "player_name": ["Player Out Jr.", "Player Starter III"],
            "active_flag": [True, True],
            "listed_pos": ["SF", "PG"],
            "as_of_ts": ["2026-01-01T22:00:00Z", "2026-01-01T22:00:00Z"],
            "lineup_role": [pd.NA, pd.NA],
            "lineup_status": [pd.NA, pd.NA],
            "lineup_roster_status": [pd.NA, pd.NA],
            "is_projected_starter": [False, False],
            "is_confirmed_starter": [False, False],
        }
    )

    rotowire_path = (
        tmp_path / "silver" / "rotowire_lineups" / "date=2026-01-01" / "lineups.parquet"
    )
    rotowire_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            # Suffixes intentionally omitted to verify name normalization.
            "player_name": ["Player Out", "Player Starter"],
            "lineup_role": ["out", "projected_starter"],
            "ingested_ts": [
                pd.Timestamp("2026-01-01T22:30:00Z"),
                pd.Timestamp("2026-01-01T22:30:00Z"),
            ],
        }
    ).to_parquet(rotowire_path, index=False)

    empty_injuries = pd.DataFrame({"as_of_ts": pd.Series(dtype="datetime64[ns, UTC]")})

    def fake_load_table(default_dir: Path, override: Path | None):  # noqa: ANN001
        target = override or default_dir
        target_str = str(target)
        if "schedule" in target_str:
            return schedule.copy()
        if "odds_snapshot" in target_str:
            return odds.copy()
        if "roster_nightly" in target_str:
            return roster.copy()
        if "injuries_snapshot" in target_str:
            return empty_injuries.copy()
        return pd.DataFrame()

    monkeypatch.setattr(
        build_minutes_live.bronze_storage,
        "read_bronze_day",
        lambda *_, **__: empty_injuries.copy(),
    )
    monkeypatch.setattr(build_minutes_live, "_load_table", fake_load_table)
    monkeypatch.setattr(build_minutes_live, "_load_label_sources", lambda **_: (pd.DataFrame(), "mock"))
    monkeypatch.setattr(build_minutes_live, "_load_label_history", lambda *_, **__: history.copy())

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

    by_pid = features.set_index("player_id")
    assert str(by_pid.loc[101, "lineup_role"]).lower() == "out"
    assert int(by_pid.loc[101, "is_out"]) == 1
    assert str(by_pid.loc[101, "status"]).upper() == "OUT"
    assert str(by_pid.loc[102, "lineup_role"]).lower() == "projected_starter"
    assert int(by_pid.loc[102, "is_out"]) == 0


def test_build_minutes_live_uses_schedule_tip_when_roster_tip_missing(tmp_path: Path, monkeypatch) -> None:
    run_id = "pytest_rotowire_tip_fallback"
    monkeypatch.setenv("PROJECTIONS_SKIP_POINTER_WRITES", "1")

    history = pd.DataFrame(
        {
            "game_id": [999],
            "player_id": [201],
            "team_id": [1610612746],
            "player_name": ["Starter One"],
            "season": ["2025-26"],
            "game_date": ["2025-12-30"],
            "minutes": [30.0],
            "starter_flag": [1],
            "starter_flag_label": [1],
            "source": ["test"],
            "label_frozen_ts": [pd.Timestamp("2025-12-31T00:00:00Z")],
        }
    )

    schedule = pd.DataFrame(
        {
            "game_id": [22500471],
            "season": ["2025-26"],
            "game_date": ["2026-01-01"],
            "tip_ts": ["2026-01-02T00:30:00Z"],
            "home_team_id": [1610612746],
            "away_team_id": [1610612741],
        }
    )

    odds = pd.DataFrame(
        {
            "game_id": [22500471],
            "home_line": [-2.0],
            "total": [226.5],
            "as_of_ts": ["2026-01-01T22:50:00Z"],
        }
    )

    roster = pd.DataFrame(
        {
            "game_id": [22500471, 22500471],
            "team_id": [1610612746, 1610612746],
            "game_date": ["2026-01-01", "2026-01-01"],
            "player_id": [201, 202],
            "player_name": ["Starter One", "Starter Two"],
            "active_flag": [True, True],
            "listed_pos": ["SF", "PG"],
            "as_of_ts": ["2026-01-01T22:00:00Z", "2026-01-01T22:00:00Z"],
            "lineup_role": [pd.NA, pd.NA],
            "lineup_status": [pd.NA, pd.NA],
            "lineup_roster_status": [pd.NA, pd.NA],
            "is_projected_starter": [False, False],
            "is_confirmed_starter": [False, False],
        }
    )

    rotowire_path = (
        tmp_path / "silver" / "rotowire_lineups" / "date=2026-01-01" / "lineups.parquet"
    )
    rotowire_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "team_abbreviation": ["LAC", "LAC"],
            "opponent_abbreviation": ["CHI", "CHI"],
            "player_name": ["Starter One", "Starter Two"],
            "lineup_role": ["confirmed_starter", "confirmed_starter"],
            "is_confirmed": [True, True],
            "injury_status": [None, None],
            # After run_as_of_ts, but before actual tip_ts.
            "ingested_ts": [
                pd.Timestamp("2026-01-02T00:25:14Z"),
                pd.Timestamp("2026-01-02T00:25:14Z"),
            ],
        }
    ).to_parquet(rotowire_path, index=False)

    empty_injuries = pd.DataFrame({"as_of_ts": pd.Series(dtype="datetime64[ns, UTC]")})

    def fake_load_table(default_dir: Path, override: Path | None):  # noqa: ANN001
        target = override or default_dir
        target_str = str(target)
        if "schedule" in target_str:
            return schedule.copy()
        if "odds_snapshot" in target_str:
            return odds.copy()
        if "roster_nightly" in target_str:
            return roster.copy()
        if "injuries_snapshot" in target_str:
            return empty_injuries.copy()
        return pd.DataFrame()

    monkeypatch.setattr(
        build_minutes_live.bronze_storage,
        "read_bronze_day",
        lambda *_, **__: empty_injuries.copy(),
    )
    monkeypatch.setattr(build_minutes_live, "_load_table", fake_load_table)
    monkeypatch.setattr(build_minutes_live, "_load_label_sources", lambda **_: (pd.DataFrame(), "mock"))
    monkeypatch.setattr(build_minutes_live, "_load_label_history", lambda *_, **__: history.copy())

    out_root = tmp_path / "live_features"
    runner = CliRunner()
    result = runner.invoke(
        build_minutes_live.app,
        [
            "--date",
            "2026-01-01",
            "--run-as-of-ts",
            "2026-01-02T00:25:00",
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
    features = pd.read_parquet(features_path).set_index("player_id")

    assert bool(features.loc[201, "is_projected_starter"]) is True
    assert bool(features.loc[201, "is_confirmed_starter"]) is True
    assert str(features.loc[201, "lineup_role"]).lower() == "confirmed_starter"
