from __future__ import annotations

from pathlib import Path

import pandas as pd

from prefect_flows.live_nba_pipeline_v3 import _build_feature_input_checklist, _resolve_season_month


def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_feature_input_checklist_passes_with_required_inputs(tmp_path: Path) -> None:
    game_date = "2026-02-24"
    season, month = _resolve_season_month(game_date)
    game_id = 22500831

    _write(
        tmp_path / "silver" / "schedule" / f"season={season}" / f"month={month:02d}" / "schedule.parquet",
        pd.DataFrame({"game_id": [game_id], "game_date": [game_date]}),
    )
    _write(
        tmp_path / "silver" / "roster_nightly" / f"season={season}" / f"month={month:02d}" / "roster.parquet",
        pd.DataFrame(
            {
                "game_id": [game_id],
                "player_id": [1],
                "team_id": [10],
                "game_date": [game_date],
                "as_of_ts": ["2026-02-24T16:00:00Z"],
            }
        ),
    )
    _write(
        tmp_path / "silver" / "odds_snapshot" / f"season={season}" / f"month={month:02d}" / "odds_snapshot.parquet",
        pd.DataFrame({"game_id": [game_id], "as_of_ts": ["2026-02-24T16:00:00Z"]}),
    )
    _write(
        tmp_path
        / "silver"
        / "injuries_snapshot"
        / f"season={season}"
        / f"month={month:02d}"
        / "injuries_snapshot.parquet",
        pd.DataFrame({"game_id": [game_id], "player_id": [1], "as_of_ts": ["2026-02-24T16:00:00Z"]}),
    )
    _write(
        tmp_path / "labels" / f"season={season}" / "boxscore_labels.parquet",
        pd.DataFrame({"game_id": [123], "player_id": [1], "team_id": [10], "game_date": ["2026-02-23"], "minutes": [20.0]}),
    )
    _write(
        tmp_path
        / "silver"
        / "rotation_priors_v1"
        / "team_game_priors"
        / f"season={season}"
        / f"game_id={str(game_id).zfill(10)}.parquet",
        pd.DataFrame({"game_id": [str(game_id).zfill(10)], "team_id": [10]}),
    )
    _write(
        tmp_path
        / "silver"
        / "rotation_priors_v1"
        / "player_game_priors"
        / f"season={season}"
        / f"game_id={str(game_id).zfill(10)}.parquet",
        pd.DataFrame({"game_id": [str(game_id).zfill(10)], "person_id": [1]}),
    )

    report = _build_feature_input_checklist(
        game_date=game_date,
        run_as_of_ts="2026-02-24T16:30:00Z",
        data_root=tmp_path,
        allow_priors_fallback=False,
        require_action_props=False,
    )
    assert report["failed_required_checks"] == []


def test_feature_input_checklist_fails_when_required_snapshot_missing(tmp_path: Path) -> None:
    game_date = "2026-02-24"
    season, month = _resolve_season_month(game_date)
    game_id = 22500831

    _write(
        tmp_path / "silver" / "schedule" / f"season={season}" / f"month={month:02d}" / "schedule.parquet",
        pd.DataFrame({"game_id": [game_id], "game_date": [game_date]}),
    )
    _write(
        tmp_path / "silver" / "roster_nightly" / f"season={season}" / f"month={month:02d}" / "roster.parquet",
        pd.DataFrame(
            {
                "game_id": [game_id],
                "player_id": [1],
                "team_id": [10],
                "game_date": [game_date],
                "as_of_ts": ["2026-02-24T16:00:00Z"],
            }
        ),
    )
    _write(
        tmp_path
        / "silver"
        / "injuries_snapshot"
        / f"season={season}"
        / f"month={month:02d}"
        / "injuries_snapshot.parquet",
        pd.DataFrame({"game_id": [game_id], "player_id": [1], "as_of_ts": ["2026-02-24T16:00:00Z"]}),
    )
    _write(
        tmp_path / "labels" / f"season={season}" / "boxscore_labels.parquet",
        pd.DataFrame({"game_id": [123], "player_id": [1], "team_id": [10], "game_date": ["2026-02-23"], "minutes": [20.0]}),
    )
    _write(
        tmp_path
        / "silver"
        / "rotation_priors_v1"
        / "team_game_priors"
        / f"season={season}"
        / f"game_id={str(game_id).zfill(10)}.parquet",
        pd.DataFrame({"game_id": [str(game_id).zfill(10)], "team_id": [10]}),
    )
    _write(
        tmp_path
        / "silver"
        / "rotation_priors_v1"
        / "player_game_priors"
        / f"season={season}"
        / f"game_id={str(game_id).zfill(10)}.parquet",
        pd.DataFrame({"game_id": [str(game_id).zfill(10)], "person_id": [1]}),
    )

    report = _build_feature_input_checklist(
        game_date=game_date,
        run_as_of_ts="2026-02-24T16:30:00Z",
        data_root=tmp_path,
        allow_priors_fallback=False,
        require_action_props=False,
    )
    assert "odds_snapshot_slate_rows" in report["failed_required_checks"]


def test_feature_input_checklist_fails_when_action_props_required_and_missing(tmp_path: Path) -> None:
    game_date = "2026-02-24"
    season, month = _resolve_season_month(game_date)
    game_id = 22500831

    _write(
        tmp_path / "silver" / "schedule" / f"season={season}" / f"month={month:02d}" / "schedule.parquet",
        pd.DataFrame({"game_id": [game_id], "game_date": [game_date]}),
    )
    _write(
        tmp_path / "silver" / "roster_nightly" / f"season={season}" / f"month={month:02d}" / "roster.parquet",
        pd.DataFrame(
            {
                "game_id": [game_id],
                "player_id": [1],
                "team_id": [10],
                "game_date": [game_date],
                "as_of_ts": ["2026-02-24T16:00:00Z"],
            }
        ),
    )
    _write(
        tmp_path / "silver" / "odds_snapshot" / f"season={season}" / f"month={month:02d}" / "odds_snapshot.parquet",
        pd.DataFrame({"game_id": [game_id], "as_of_ts": ["2026-02-24T16:00:00Z"]}),
    )
    _write(
        tmp_path
        / "silver"
        / "injuries_snapshot"
        / f"season={season}"
        / f"month={month:02d}"
        / "injuries_snapshot.parquet",
        pd.DataFrame({"game_id": [game_id], "player_id": [1], "as_of_ts": ["2026-02-24T16:00:00Z"]}),
    )
    _write(
        tmp_path / "labels" / f"season={season}" / "boxscore_labels.parquet",
        pd.DataFrame({"game_id": [123], "player_id": [1], "team_id": [10], "game_date": ["2026-02-23"], "minutes": [20.0]}),
    )
    _write(
        tmp_path
        / "silver"
        / "rotation_priors_v1"
        / "team_game_priors"
        / f"season={season}"
        / f"game_id={str(game_id).zfill(10)}.parquet",
        pd.DataFrame({"game_id": [str(game_id).zfill(10)], "team_id": [10]}),
    )
    _write(
        tmp_path
        / "silver"
        / "rotation_priors_v1"
        / "player_game_priors"
        / f"season={season}"
        / f"game_id={str(game_id).zfill(10)}.parquet",
        pd.DataFrame({"game_id": [str(game_id).zfill(10)], "person_id": [1]}),
    )

    report = _build_feature_input_checklist(
        game_date=game_date,
        run_as_of_ts="2026-02-24T16:30:00Z",
        data_root=tmp_path,
        allow_priors_fallback=False,
        require_action_props=True,
    )
    assert "props_source_policy_satisfied" in report["failed_required_checks"]


def test_feature_input_checklist_passes_with_rotowire_fallback(tmp_path: Path) -> None:
    game_date = "2026-02-24"
    season, month = _resolve_season_month(game_date)
    game_id = 22500831

    _write(
        tmp_path / "silver" / "schedule" / f"season={season}" / f"month={month:02d}" / "schedule.parquet",
        pd.DataFrame({"game_id": [game_id], "game_date": [game_date]}),
    )
    _write(
        tmp_path / "silver" / "roster_nightly" / f"season={season}" / f"month={month:02d}" / "roster.parquet",
        pd.DataFrame(
            {
                "game_id": [game_id],
                "player_id": [1],
                "team_id": [10],
                "game_date": [game_date],
                "as_of_ts": ["2026-02-24T16:00:00Z"],
            }
        ),
    )
    _write(
        tmp_path / "silver" / "odds_snapshot" / f"season={season}" / f"month={month:02d}" / "odds_snapshot.parquet",
        pd.DataFrame({"game_id": [game_id], "as_of_ts": ["2026-02-24T16:00:00Z"]}),
    )
    _write(
        tmp_path
        / "silver"
        / "injuries_snapshot"
        / f"season={season}"
        / f"month={month:02d}"
        / "injuries_snapshot.parquet",
        pd.DataFrame({"game_id": [game_id], "player_id": [1], "as_of_ts": ["2026-02-24T16:00:00Z"]}),
    )
    _write(
        tmp_path / "labels" / f"season={season}" / "boxscore_labels.parquet",
        pd.DataFrame({"game_id": [123], "player_id": [1], "team_id": [10], "game_date": ["2026-02-23"], "minutes": [20.0]}),
    )
    _write(
        tmp_path
        / "silver"
        / "rotation_priors_v1"
        / "team_game_priors"
        / f"season={season}"
        / f"game_id={str(game_id).zfill(10)}.parquet",
        pd.DataFrame({"game_id": [str(game_id).zfill(10)], "team_id": [10]}),
    )
    _write(
        tmp_path
        / "silver"
        / "rotation_priors_v1"
        / "player_game_priors"
        / f"season={season}"
        / f"game_id={str(game_id).zfill(10)}.parquet",
        pd.DataFrame({"game_id": [str(game_id).zfill(10)], "person_id": [1]}),
    )
    _write(
        tmp_path / "bronze" / "props" / f"game_date={game_date}" / "props_1.parquet",
        pd.DataFrame(
            {
                "player_id": ["1"],
                "player_name": ["Player One"],
                "team": ["NYK"],
                "opponent": ["BOS"],
                "game_id": ["999"],
                "book": ["draftkings"],
                "prop_type": ["pts"],
                "line": [22.5],
                "over_odds": [-110],
                "under_odds": [-110],
                "implied_over_prob": [0.5],
                "implied_under_prob": [0.5],
                "scraped_at": ["2026-02-24T16:00:00Z"],
            }
        ),
    )

    report = _build_feature_input_checklist(
        game_date=game_date,
        run_as_of_ts="2026-02-24T16:30:00Z",
        data_root=tmp_path,
        allow_priors_fallback=False,
        allow_rotowire_props_fallback=True,
        require_action_props=True,
    )
    assert report["failed_required_checks"] == []
