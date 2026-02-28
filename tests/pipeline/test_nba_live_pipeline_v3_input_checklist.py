from __future__ import annotations

from pathlib import Path

import pandas as pd

from prefect_flows.live_nba_pipeline_v3 import (
    _build_feature_input_checklist,
    _build_input_change_set,
    _build_publish_superseded_report,
    _build_rerun_plan,
    _compute_per_game_input_digests,
    _detect_stale_authoritative_inputs,
    _merge_parquet_for_target_games,
    _report_window_status,
    _summarize_world_contracts_from_frame,
    _resolve_season_month,
)


def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_feature_input_checklist_passes_with_required_inputs(tmp_path: Path) -> None:
    game_date = "2026-02-24"
    season, month = _resolve_season_month(game_date)
    game_id = 22500831

    _write(
        tmp_path
        / "silver"
        / "schedule"
        / f"season={season}"
        / f"month={month:02d}"
        / "schedule.parquet",
        pd.DataFrame({"game_id": [game_id], "game_date": [game_date]}),
    )
    _write(
        tmp_path
        / "silver"
        / "roster_nightly"
        / f"season={season}"
        / f"month={month:02d}"
        / "roster.parquet",
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
        / "odds_snapshot"
        / f"season={season}"
        / f"month={month:02d}"
        / "odds_snapshot.parquet",
        pd.DataFrame({"game_id": [game_id], "as_of_ts": ["2026-02-24T16:00:00Z"]}),
    )
    _write(
        tmp_path
        / "silver"
        / "injuries_snapshot"
        / f"season={season}"
        / f"month={month:02d}"
        / "injuries_snapshot.parquet",
        pd.DataFrame(
            {
                "game_id": [game_id],
                "player_id": [1],
                "as_of_ts": ["2026-02-24T16:00:00Z"],
            }
        ),
    )
    _write(
        tmp_path / "labels" / f"season={season}" / "boxscore_labels.parquet",
        pd.DataFrame(
            {
                "game_id": [123],
                "player_id": [1],
                "team_id": [10],
                "game_date": ["2026-02-23"],
                "minutes": [20.0],
            }
        ),
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
    assert report["source_freshness"]["summary"]["slate_game_count"] == 1
    assert report["freshness_gates"]["lock_window"]["ok"] is True


def test_feature_input_checklist_emits_lock_window_failures(tmp_path: Path) -> None:
    game_date = "2026-02-24"
    season, month = _resolve_season_month(game_date)
    game_id = 22500831

    _write(
        tmp_path
        / "silver"
        / "schedule"
        / f"season={season}"
        / f"month={month:02d}"
        / "schedule.parquet",
        pd.DataFrame(
            {
                "game_id": [game_id],
                "game_date": [game_date],
                "tip_ts": ["2026-02-24T17:00:00Z"],
            }
        ),
    )
    _write(
        tmp_path
        / "silver"
        / "roster_nightly"
        / f"season={season}"
        / f"month={month:02d}"
        / "roster.parquet",
        pd.DataFrame(
            {
                "game_id": [game_id],
                "player_id": [1],
                "team_id": [10],
                "game_date": [game_date],
                "as_of_ts": ["2026-02-24T16:20:00Z"],
            }
        ),
    )
    _write(
        tmp_path
        / "silver"
        / "odds_snapshot"
        / f"season={season}"
        / f"month={month:02d}"
        / "odds_snapshot.parquet",
        pd.DataFrame({"game_id": [game_id], "as_of_ts": ["2026-02-24T16:20:00Z"]}),
    )
    _write(
        tmp_path
        / "silver"
        / "injuries_snapshot"
        / f"season={season}"
        / f"month={month:02d}"
        / "injuries_snapshot.parquet",
        pd.DataFrame(
            {
                "game_id": [game_id],
                "player_id": [1],
                "as_of_ts": ["2026-02-24T15:20:00Z"],
            }
        ),
    )
    _write(
        tmp_path
        / "silver"
        / "rotowire_lineups"
        / f"date={game_date}"
        / "lineups.parquet",
        pd.DataFrame(
            {
                "game_id": [game_id],
                "player_id": [1],
                "player_name": ["Player One"],
                "lineup_role": ["confirmed_starter"],
                "ingested_ts": ["2026-02-24T15:10:00Z"],
            }
        ),
    )
    _write(
        tmp_path / "labels" / f"season={season}" / "boxscore_labels.parquet",
        pd.DataFrame(
            {
                "game_id": [123],
                "player_id": [1],
                "team_id": [10],
                "game_date": ["2026-02-23"],
                "minutes": [20.0],
            }
        ),
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
        run_as_of_ts="2026-02-24T16:40:00Z",
        data_root=tmp_path,
        allow_priors_fallback=False,
        require_action_props=False,
    )
    assert report["failed_required_checks"] == []
    lock_window = report["freshness_gates"]["lock_window"]
    assert lock_window["ok"] is False
    assert lock_window["failures"][0]["game_id"] == game_id


def test_detect_stale_authoritative_inputs_flags_newer_injuries_and_lineups() -> None:
    frozen = {
        "per_game": {
            "22500831": {
                "game_id": 22500831,
                "is_live_game": True,
                "sources": {
                    "injuries": {
                        "latest_as_of_ts": "2026-02-24T16:00:00Z",
                        "source_used": "silver",
                    },
                    "lineups": {
                        "latest_as_of_ts": "2026-02-24T16:05:00Z",
                        "source_used": "rotowire",
                    },
                },
            }
        }
    }
    current = {
        "per_game": {
            "22500831": {
                "game_id": 22500831,
                "is_live_game": True,
                "tip_ts": "2026-02-24T17:00:00Z",
                "minutes_to_tip": 20.0,
                "sources": {
                    "injuries": {
                        "latest_as_of_ts": "2026-02-24T16:15:00Z",
                        "source_used": "silver",
                    },
                    "lineups": {
                        "latest_as_of_ts": "2026-02-24T16:10:00Z",
                        "source_used": "rotowire",
                    },
                },
            }
        }
    }

    report = _detect_stale_authoritative_inputs(
        frozen_source_freshness=frozen,
        current_source_freshness=current,
        as_of_ts="2026-02-24T16:40:00Z",
    )
    assert report["stale"] is True
    assert report["stale_games"][0]["game_id"] == 22500831


def test_report_window_status_activates_for_230pm_et_boundary() -> None:
    report = _report_window_status(
        run_ts=pd.Timestamp("2026-02-28T19:33:00Z"),
        per_game_freshness={
            "22500863": {
                "game_id": 22500863,
                "tip_ts": "2026-02-28T23:00:00Z",
                "is_live_game": True,
                "sources": {
                    "injuries": {
                        "latest_as_of_ts": "2026-02-28T19:10:00Z",
                        "source_used": "bronze",
                    }
                },
            }
        },
    )
    assert report["active"] is True
    assert report["label"] == "nba_injury_report_230pm_et"
    assert report["needs_wait"] is True
    assert report["blocking_games"][0]["game_id"] == 22500863


def test_input_change_set_detects_changed_and_new_games() -> None:
    current = {
        "per_game": {
            "1": {
                "game_id": 1,
                "tip_ts": "2026-02-24T17:00:00Z",
                "is_live_game": True,
                "sources": {
                    "injuries": {
                        "latest_as_of_ts": "2026-02-24T16:00:00Z",
                        "source_used": "bronze",
                    },
                    "lineups": {
                        "latest_as_of_ts": "2026-02-24T16:05:00Z",
                        "source_used": "rotowire",
                    },
                    "odds": {
                        "latest_as_of_ts": "2026-02-24T16:00:00Z",
                        "source_used": "silver",
                    },
                    "props": {
                        "latest_as_of_ts": "2026-02-24T16:02:00Z",
                        "source_used": "action_network",
                    },
                    "roster": {
                        "latest_as_of_ts": "2026-02-24T16:00:00Z",
                        "source_used": "silver",
                    },
                },
            },
            "2": {
                "game_id": 2,
                "tip_ts": "2026-02-24T19:00:00Z",
                "is_live_game": True,
                "sources": {
                    "injuries": {
                        "latest_as_of_ts": "2026-02-24T16:00:00Z",
                        "source_used": "bronze",
                    },
                    "lineups": {
                        "latest_as_of_ts": "2026-02-24T16:00:00Z",
                        "source_used": "rotowire",
                    },
                    "odds": {
                        "latest_as_of_ts": "2026-02-24T16:00:00Z",
                        "source_used": "silver",
                    },
                    "props": {
                        "latest_as_of_ts": "2026-02-24T16:00:00Z",
                        "source_used": "action_network",
                    },
                    "roster": {
                        "latest_as_of_ts": "2026-02-24T16:00:00Z",
                        "source_used": "silver",
                    },
                },
            },
        }
    }
    previous_source_freshness = {
        "per_game": {
            "1": {
                "game_id": 1,
                "tip_ts": "2026-02-24T17:00:00Z",
                "is_live_game": True,
                "sources": {
                    "injuries": {
                        "latest_as_of_ts": "2026-02-24T15:30:00Z",
                        "source_used": "bronze",
                    },
                    "lineups": {
                        "latest_as_of_ts": "2026-02-24T16:05:00Z",
                        "source_used": "rotowire",
                    },
                    "odds": {
                        "latest_as_of_ts": "2026-02-24T16:00:00Z",
                        "source_used": "silver",
                    },
                    "props": {
                        "latest_as_of_ts": "2026-02-24T16:02:00Z",
                        "source_used": "action_network",
                    },
                    "roster": {
                        "latest_as_of_ts": "2026-02-24T16:00:00Z",
                        "source_used": "silver",
                    },
                },
            }
        }
    }
    previous_manifest = {
        "run_id": "prev_run",
        "source_freshness": previous_source_freshness,
        "input_change_set": {
            "per_game_digests": _compute_per_game_input_digests(
                previous_source_freshness
            ),
        },
    }

    report = _build_input_change_set(
        game_date="2026-02-24",
        current_source_freshness=current,
        previous_manifest_payload=previous_manifest,
    )
    assert report["previous_run_id"] == "prev_run"
    assert report["changed_game_ids"] == [1]
    assert report["new_game_ids"] == [2]
    assert report["changed_games"][0]["changed_sources"] == ["injuries"]


def test_input_change_set_ignores_timestamp_only_refresh_when_content_same() -> None:
    previous_source_freshness = {
        "per_game": {
            "1": {
                "game_id": 1,
                "tip_ts": "2026-02-24T17:00:00Z",
                "is_live_game": True,
                "sources": {
                    "injuries": {
                        "latest_as_of_ts": "2026-02-24T15:30:00Z",
                        "source_used": "bronze",
                        "content_digest": "abc",
                    }
                },
            }
        }
    }
    current = {
        "per_game": {
            "1": {
                "game_id": 1,
                "tip_ts": "2026-02-24T17:00:00Z",
                "is_live_game": True,
                "sources": {
                    "injuries": {
                        "latest_as_of_ts": "2026-02-24T16:00:00Z",
                        "source_used": "bronze",
                        "content_digest": "abc",
                    }
                },
            }
        }
    }
    previous_manifest = {
        "run_id": "prev_run",
        "source_freshness": previous_source_freshness,
        "input_change_set": {
            "per_game_digests": _compute_per_game_input_digests(
                previous_source_freshness
            ),
        },
    }
    report = _build_input_change_set(
        game_date="2026-02-24",
        current_source_freshness=current,
        previous_manifest_payload=previous_manifest,
    )
    assert report["changed_game_ids"] == []


def test_rerun_plan_uses_game_scoped_policy_for_material_pre_tip_changes(
    tmp_path: Path,
) -> None:
    current = {
        "summary": {"slate_game_count": 2},
        "per_game": {
            "1": {
                "game_id": 1,
                "minutes_to_tip": 25.0,
                "is_live_game": True,
                "sources": {},
            },
            "2": {
                "game_id": 2,
                "minutes_to_tip": 240.0,
                "is_live_game": True,
                "sources": {},
            },
        },
    }
    input_change_set = {
        "changed_games": [
            {"game_id": 1, "changed_sources": ["injuries"]},
            {"game_id": 2, "changed_sources": ["odds"]},
        ],
        "new_game_ids": [],
        "removed_game_ids": [],
    }
    selector = tmp_path / "selector.json"
    selector.write_text("{}", encoding="utf-8")
    previous_manifest = {
        "run_id": "prev_run",
        "minutes_current_run_path": str(selector),
        "rates_current_run_path": str(selector),
        "v3": {"bundle_hash": "bundle123"},
    }
    plan = _build_rerun_plan(
        game_date="2026-02-24",
        input_change_set=input_change_set,
        current_source_freshness=current,
        previous_manifest_payload=previous_manifest,
        current_bundle_hash="bundle123",
        current_minutes_selector_path=selector,
        current_rates_selector_path=selector,
    )
    assert plan["mode"] == "game_scoped"
    assert plan["target_game_ids"] == [1]


def test_publish_superseded_report_flags_newer_current_pointer(tmp_path: Path) -> None:
    manifest_path = tmp_path / "artifacts" / "runs" / "nba_live" / "game_date=2026-02-24" / "run=candidate" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        '{"run_id":"candidate","as_of_ts":"2026-02-24T16:00:00Z"}',
        encoding="utf-8",
    )
    dataset_dir = tmp_path / "artifacts" / "projections" / "2026-02-24"
    pointer_path = dataset_dir / "LATEST" / "current.json"
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(
        '{"run_id":"published","as_of_ts":"2026-02-24T16:05:00Z","manifest_path":"x"}',
        encoding="utf-8",
    )

    report = _build_publish_superseded_report(
        run_id="candidate",
        manifest_path=manifest_path,
        dataset_dir=dataset_dir,
    )
    assert report["superseded"] is True
    assert report["reason"] == "newer_pointer_as_of_ts"


def test_merge_parquet_for_target_games_replaces_only_changed_games(
    tmp_path: Path,
) -> None:
    previous = tmp_path / "prev.parquet"
    current = tmp_path / "current.parquet"
    _write(
        previous,
        pd.DataFrame({"game_id": [1, 2], "player_id": [10, 20], "value": [100, 200]}),
    )
    _write(current, pd.DataFrame({"game_id": [2], "player_id": [20], "value": [250]}))
    merged = _merge_parquet_for_target_games(
        current_path=current,
        previous_path=previous,
        target_game_ids=[2],
    )
    assert merged.sort_values(["game_id", "player_id"])["value"].tolist() == [100, 250]


def test_summarize_world_contracts_from_frame_handles_clean_worlds() -> None:
    worlds = pd.DataFrame(
        {
            "world_idx": [0, 0],
            "game_id": [1, 1],
            "team_id": [10, 20],
            "player_id": [100, 200],
            "active": [1, 1],
            "minutes": [240.0, 240.0],
            "fga2": [1.0, 1.0],
            "fg2m": [1.0, 1.0],
            "fga3": [0.0, 0.0],
            "fg3m": [0.0, 0.0],
            "fta": [0.0, 0.0],
            "ftm": [0.0, 0.0],
            "pts": [2.0, 2.0],
            "reb": [0.0, 0.0],
            "ast": [0.0, 0.0],
            "stl": [0.0, 0.0],
            "blk": [0.0, 0.0],
            "tov": [0.0, 0.0],
            "dk_fpts": [2.0, 2.0],
        }
    )
    checks = _summarize_world_contracts_from_frame(worlds)
    assert checks["minutes_negative"] == 0


def test_feature_input_checklist_fails_when_required_snapshot_missing(
    tmp_path: Path,
) -> None:
    game_date = "2026-02-24"
    season, month = _resolve_season_month(game_date)
    game_id = 22500831

    _write(
        tmp_path
        / "silver"
        / "schedule"
        / f"season={season}"
        / f"month={month:02d}"
        / "schedule.parquet",
        pd.DataFrame({"game_id": [game_id], "game_date": [game_date]}),
    )
    _write(
        tmp_path
        / "silver"
        / "roster_nightly"
        / f"season={season}"
        / f"month={month:02d}"
        / "roster.parquet",
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
        pd.DataFrame(
            {
                "game_id": [game_id],
                "player_id": [1],
                "as_of_ts": ["2026-02-24T16:00:00Z"],
            }
        ),
    )
    _write(
        tmp_path / "labels" / f"season={season}" / "boxscore_labels.parquet",
        pd.DataFrame(
            {
                "game_id": [123],
                "player_id": [1],
                "team_id": [10],
                "game_date": ["2026-02-23"],
                "minutes": [20.0],
            }
        ),
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


def test_feature_input_checklist_fails_when_action_props_required_and_missing(
    tmp_path: Path,
) -> None:
    game_date = "2026-02-24"
    season, month = _resolve_season_month(game_date)
    game_id = 22500831

    _write(
        tmp_path
        / "silver"
        / "schedule"
        / f"season={season}"
        / f"month={month:02d}"
        / "schedule.parquet",
        pd.DataFrame({"game_id": [game_id], "game_date": [game_date]}),
    )
    _write(
        tmp_path
        / "silver"
        / "roster_nightly"
        / f"season={season}"
        / f"month={month:02d}"
        / "roster.parquet",
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
        / "odds_snapshot"
        / f"season={season}"
        / f"month={month:02d}"
        / "odds_snapshot.parquet",
        pd.DataFrame({"game_id": [game_id], "as_of_ts": ["2026-02-24T16:00:00Z"]}),
    )
    _write(
        tmp_path
        / "silver"
        / "injuries_snapshot"
        / f"season={season}"
        / f"month={month:02d}"
        / "injuries_snapshot.parquet",
        pd.DataFrame(
            {
                "game_id": [game_id],
                "player_id": [1],
                "as_of_ts": ["2026-02-24T16:00:00Z"],
            }
        ),
    )
    _write(
        tmp_path / "labels" / f"season={season}" / "boxscore_labels.parquet",
        pd.DataFrame(
            {
                "game_id": [123],
                "player_id": [1],
                "team_id": [10],
                "game_date": ["2026-02-23"],
                "minutes": [20.0],
            }
        ),
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


def test_feature_input_checklist_passes_with_rotowire_live_props(tmp_path: Path) -> None:
    game_date = "2026-02-24"
    season, month = _resolve_season_month(game_date)
    game_id = 22500831

    _write(
        tmp_path
        / "silver"
        / "schedule"
        / f"season={season}"
        / f"month={month:02d}"
        / "schedule.parquet",
        pd.DataFrame({"game_id": [game_id], "game_date": [game_date]}),
    )
    _write(
        tmp_path
        / "silver"
        / "roster_nightly"
        / f"season={season}"
        / f"month={month:02d}"
        / "roster.parquet",
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
        / "odds_snapshot"
        / f"season={season}"
        / f"month={month:02d}"
        / "odds_snapshot.parquet",
        pd.DataFrame({"game_id": [game_id], "as_of_ts": ["2026-02-24T16:00:00Z"]}),
    )
    _write(
        tmp_path
        / "silver"
        / "injuries_snapshot"
        / f"season={season}"
        / f"month={month:02d}"
        / "injuries_snapshot.parquet",
        pd.DataFrame(
            {
                "game_id": [game_id],
                "player_id": [1],
                "as_of_ts": ["2026-02-24T16:00:00Z"],
            }
        ),
    )
    _write(
        tmp_path / "labels" / f"season={season}" / "boxscore_labels.parquet",
        pd.DataFrame(
            {
                "game_id": [123],
                "player_id": [1],
                "team_id": [10],
                "game_date": ["2026-02-23"],
                "minutes": [20.0],
            }
        ),
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
    props_check = next(
        item
        for item in report["checks"]
        if item["name"] == "props_source_policy_satisfied"
    )
    assert props_check["details"]["selected_source"] == "rotowire"
