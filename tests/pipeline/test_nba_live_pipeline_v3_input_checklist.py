from __future__ import annotations

from datetime import date
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import pytest

from prefect_flows.live_nba_pipeline_v3 import (
    _atomic_write_validated_parquet,
    _build_feature_input_checklist,
    _build_input_change_set,
    _build_publish_superseded_report,
    _coerce_world_game_date,
    _factorize_int_key_arrays_preserve_order,
    _group_mean_by_keys_without_pandas_groupby,
    _build_rerun_plan,
    _compute_per_game_input_digests,
    _detect_stale_authoritative_inputs,
    _merge_parquet_for_target_games,
    _run_python_module,
    _report_window_status,
    _repair_world_frame_contract_fields,
    _sanitize_frame_to_expected_keys,
    _stream_validate_parquet,
    _summarize_world_contracts_from_frame,
    _team_minutes_sums_without_pandas_groupby,
    _resolve_season_month,
    publish_atomic_task,
)
from projections.features.action_props import load_rotowire_props_long_from_bronze
from projections.ops.manual_availability import upsert_manual_override
from projections.pipeline import writer_guard


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


def test_feature_input_checklist_includes_manual_override_freshness(tmp_path: Path) -> None:
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
        pd.DataFrame({"game_id": [game_id], "player_id": [1], "as_of_ts": ["2026-02-24T16:00:00Z"]}),
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
    upsert_manual_override(
        date.fromisoformat(game_date),
        game_id=str(game_id),
        player_id="1",
        player_name="Player One",
        team_id=10,
        team_tricode="AAA",
        override_type="force_out",
        entered_by="daniel",
        effective_ts="2026-02-24T16:10:00Z",
        data_root=tmp_path,
    )

    report = _build_feature_input_checklist(
        game_date=game_date,
        run_as_of_ts="2026-02-24T16:30:00Z",
        data_root=tmp_path,
        allow_priors_fallback=False,
        require_action_props=False,
    )
    assert report["source_freshness"]["summary"]["manual_override_count"] == 1
    info = report["source_freshness"]["per_game"][str(game_id)]["sources"]["manual_overrides"]
    assert info["source_used"] == "manual_override"
    assert info["content_digest"] is not None


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


def test_feature_input_checklist_maps_rotowire_lineups_by_team_without_game_id(
    tmp_path: Path,
) -> None:
    game_date = "2026-02-28"
    season, month = _resolve_season_month(game_date)
    game_id = 22500865

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
                "tip_ts": ["2026-03-01T00:00:00Z"],
                "home_team_tricode": ["HOU"],
                "away_team_tricode": ["MIA"],
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
                "game_id": [game_id, game_id],
                "player_id": [1, 2],
                "team_id": [10, 20],
                "game_date": [game_date, game_date],
                "as_of_ts": ["2026-02-28T20:00:00Z", "2026-02-28T20:00:00Z"],
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
        pd.DataFrame({"game_id": [game_id], "as_of_ts": ["2026-02-28T20:00:00Z"]}),
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
                "as_of_ts": ["2026-02-28T20:00:00Z"],
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
                "team_abbreviation": ["HOU", "MIA"],
                "opponent_abbreviation": ["MIA", "HOU"],
                "player_name": ["Amen Thompson", "Kevin Love"],
                "lineup_role": ["confirmed_starter", "confirmed_starter"],
                "ingested_ts": ["2026-02-28T20:30:05Z", "2026-02-28T20:30:05Z"],
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
                "game_date": ["2026-02-27"],
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
        run_as_of_ts="2026-02-28T20:30:20Z",
        data_root=tmp_path,
        allow_priors_fallback=False,
        require_action_props=False,
    )
    lineup_info = report["source_freshness"]["per_game"][str(game_id)]["sources"][
        "lineups"
    ]
    assert lineup_info["latest_as_of_ts"] == "2026-02-28T20:30:05+00:00"
    assert lineup_info["content_digest"] is not None


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


def test_input_change_set_detects_manual_override_change() -> None:
    previous_source_freshness = {
        "per_game": {
            "1": {
                "game_id": 1,
                "tip_ts": "2026-02-24T17:00:00Z",
                "is_live_game": True,
                "sources": {
                    "manual_overrides": {
                        "latest_as_of_ts": "2026-02-24T16:00:00Z",
                        "source_used": "manual_override",
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
                    "manual_overrides": {
                        "latest_as_of_ts": "2026-02-24T16:05:00Z",
                        "source_used": "manual_override",
                        "content_digest": "def",
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
    assert report["changed_game_ids"] == [1]
    assert report["changed_games"][0]["changed_sources"] == ["manual_overrides"]


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
        "ownership_current_run_path": str(selector),
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
        current_ownership_selector_path=selector,
    )
    assert plan["mode"] == "game_scoped"
    assert plan["target_game_ids"] == [1]


def test_rerun_plan_honors_manual_target_override(tmp_path: Path) -> None:
    current = {
        "summary": {"slate_game_count": 2},
        "per_game": {
            "1": {"game_id": 1, "minutes_to_tip": 25.0, "is_live_game": True, "sources": {}},
            "2": {"game_id": 2, "minutes_to_tip": 240.0, "is_live_game": True, "sources": {}},
        },
    }
    selector = tmp_path / "selector.json"
    selector.write_text("{}", encoding="utf-8")
    previous_manifest = {
        "run_id": "prev_run",
        "minutes_current_run_path": str(selector),
        "rates_current_run_path": str(selector),
        "ownership_current_run_path": str(selector),
        "v3": {"bundle_hash": "bundle123"},
    }
    plan = _build_rerun_plan(
        game_date="2026-02-24",
        input_change_set={"changed_games": [], "new_game_ids": [], "removed_game_ids": []},
        current_source_freshness=current,
        previous_manifest_payload=previous_manifest,
        current_bundle_hash="bundle123",
        current_minutes_selector_path=selector,
        current_rates_selector_path=selector,
        current_ownership_selector_path=selector,
        manual_target_game_ids=[2],
    )
    assert plan["mode"] == "game_scoped"
    assert plan["reason"] == "manual_operator_trigger"
    assert plan["target_game_ids"] == [2]
    assert plan["manual_trigger"]["applied_game_ids"] == [2]


def test_rerun_plan_skips_when_manual_targets_not_on_slate(tmp_path: Path) -> None:
    current = {
        "summary": {"slate_game_count": 1},
        "per_game": {
            "1": {"game_id": 1, "minutes_to_tip": 45.0, "is_live_game": True, "sources": {}},
        },
    }
    selector = tmp_path / "selector.json"
    selector.write_text("{}", encoding="utf-8")
    previous_manifest = {
        "run_id": "prev_run",
        "minutes_current_run_path": str(selector),
        "rates_current_run_path": str(selector),
        "ownership_current_run_path": str(selector),
        "v3": {"bundle_hash": "bundle123"},
    }
    plan = _build_rerun_plan(
        game_date="2026-02-24",
        input_change_set={"changed_games": [], "new_game_ids": [], "removed_game_ids": []},
        current_source_freshness=current,
        previous_manifest_payload=previous_manifest,
        current_bundle_hash="bundle123",
        current_minutes_selector_path=selector,
        current_rates_selector_path=selector,
        current_ownership_selector_path=selector,
        manual_target_game_ids=[999],
    )
    assert plan["mode"] == "skip"
    assert plan["reason"] == "manual_targets_not_on_slate"
    assert plan["target_game_ids"] == []
    assert plan["manual_trigger"]["invalid_game_ids"] == [999]


def test_rerun_plan_forces_full_slate_when_ownership_selector_changes(
    tmp_path: Path,
) -> None:
    current = {
        "summary": {"slate_game_count": 1},
        "per_game": {
            "1": {"game_id": 1, "minutes_to_tip": 45.0, "is_live_game": True, "sources": {}},
        },
    }
    prev_selector = tmp_path / "selector_prev.json"
    curr_selector = tmp_path / "selector_curr.json"
    prev_selector.write_text("{}", encoding="utf-8")
    curr_selector.write_text("{}", encoding="utf-8")
    previous_manifest = {
        "run_id": "prev_run",
        "minutes_current_run_path": str(prev_selector),
        "rates_current_run_path": str(prev_selector),
        "ownership_current_run_path": str(prev_selector),
        "v3": {"bundle_hash": "bundle123"},
    }
    plan = _build_rerun_plan(
        game_date="2026-02-24",
        input_change_set={"changed_games": [], "new_game_ids": [], "removed_game_ids": []},
        current_source_freshness=current,
        previous_manifest_payload=previous_manifest,
        current_bundle_hash="bundle123",
        current_minutes_selector_path=prev_selector,
        current_rates_selector_path=prev_selector,
        current_ownership_selector_path=curr_selector,
    )
    assert plan["mode"] == "full_slate"
    assert plan["reason"] == "ownership_selector_changed"


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


def test_merge_parquet_for_target_games_falls_back_when_promoted_baseline_is_corrupt(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "artifacts" / "gtv2_worlds" / "game_date=2026-02-24"
    older_full = dataset_dir / "run=20260224T210000Z" / "worlds.parquet"
    newer_partial = dataset_dir / "run=20260224T211500Z" / "worlds.parquet"
    corrupt_promoted = dataset_dir / "run=20260224T213000Z" / "worlds.parquet"
    current = dataset_dir / "run=20260224T220000Z" / "worlds.parquet"

    _write(
        older_full,
        pd.DataFrame(
            {
                "game_id": [1, 2, 3],
                "player_id": [10, 20, 30],
                "value": [100, 200, 300],
            }
        ),
    )
    _write(
        newer_partial,
        pd.DataFrame({"game_id": [2], "player_id": [20], "value": [225]}),
    )
    corrupt_promoted.parent.mkdir(parents=True, exist_ok=True)
    corrupt_promoted.write_text("not a parquet file", encoding="utf-8")
    _write(current, pd.DataFrame({"game_id": [2], "player_id": [20], "value": [250]}))

    merged = _merge_parquet_for_target_games(
        current_path=current,
        previous_path=corrupt_promoted,
        target_game_ids=[2],
    )

    merged = merged.sort_values(["game_id", "player_id"]).reset_index(drop=True)
    assert merged["game_id"].tolist() == [1, 2, 3]
    assert merged["value"].tolist() == [100, 250, 300]


def test_merge_parquet_for_target_games_falls_back_when_current_run_is_corrupt_with_explicit_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "artifacts" / "gtv2_worlds" / "game_date=2026-02-24"
    older_full = dataset_dir / "run=20260224T210000Z" / "worlds.parquet"
    promoted_full = dataset_dir / "run=20260224T213000Z" / "worlds.parquet"
    current_corrupt = dataset_dir / "run=20260224T220000Z" / "worlds.parquet"

    _write(
        older_full,
        pd.DataFrame(
            {
                "game_id": [1, 2, 3],
                "player_id": [10, 20, 30],
                "value": [100, 200, 300],
            }
        ),
    )
    _write(
        promoted_full,
        pd.DataFrame(
            {
                "game_id": [1, 2, 3],
                "player_id": [10, 20, 30],
                "value": [101, 201, 301],
            }
        ),
    )
    current_corrupt.parent.mkdir(parents=True, exist_ok=True)
    current_corrupt.write_text("not a parquet file", encoding="utf-8")
    monkeypatch.setenv("PROJECTIONS_ALLOW_STALE_MERGE_FALLBACK", "1")

    merged = _merge_parquet_for_target_games(
        current_path=current_corrupt,
        previous_path=promoted_full,
        target_game_ids=[2],
    )

    merged = merged.sort_values(["game_id", "player_id"]).reset_index(drop=True)
    assert merged["game_id"].tolist() == [1, 2, 3]
    assert merged["value"].tolist() == [101, 201, 301]


def test_merge_parquet_for_target_games_fails_closed_when_current_run_is_corrupt(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "artifacts" / "gtv2_worlds" / "game_date=2026-02-24"
    older_full = dataset_dir / "run=20260224T210000Z" / "worlds.parquet"
    promoted_full = dataset_dir / "run=20260224T213000Z" / "worlds.parquet"
    current_corrupt = dataset_dir / "run=20260224T220000Z" / "worlds.parquet"

    _write(
        older_full,
        pd.DataFrame(
            {
                "game_id": [1, 2, 3],
                "player_id": [10, 20, 30],
                "value": [100, 200, 300],
            }
        ),
    )
    _write(
        promoted_full,
        pd.DataFrame(
            {
                "game_id": [1, 2, 3],
                "player_id": [10, 20, 30],
                "value": [101, 201, 301],
            }
        ),
    )
    current_corrupt.parent.mkdir(parents=True, exist_ok=True)
    current_corrupt.write_text("not a parquet file", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Refusing implicit stale fallback"):
        _merge_parquet_for_target_games(
            current_path=current_corrupt,
            previous_path=promoted_full,
            target_game_ids=[2],
        )


def test_atomic_write_validated_parquet_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "artifact.parquet"
    df = pd.DataFrame({"game_id": [1, 2], "team_id": [10, 20], "player_id": [100, 200]})

    report = _atomic_write_validated_parquet(
        df,
        path,
        required_cols=("game_id", "team_id", "player_id"),
    )

    assert path.exists()
    assert report["rows"] == 2
    assert "game_id" in report["columns"]
    assert not list(tmp_path.glob("*.tmp.*.parquet"))
    reloaded = pd.read_parquet(path).sort_values("game_id").reset_index(drop=True)
    assert reloaded.equals(df.sort_values("game_id").reset_index(drop=True))


def test_sanitize_frame_to_expected_keys_drops_invalid_world_rows_before_contract_summary() -> None:
    expected_keys = pd.DataFrame(
        {
            "game_id": [1, 1],
            "team_id": [10, 20],
            "player_id": [100, 200],
        }
    )
    worlds = pd.DataFrame(
        {
            "world_idx": [0, 0, 0, 1],
            "game_id": [1, 1, 576460752325924364, None],
            "team_id": [10, 20, 10, 20],
            "player_id": [100, 200, 100, 200],
            "active": [1, 1, 1, 1],
            "minutes": [240.0, 240.0, 17.0, 240.0],
            "fga2": [1.0, 1.0, 0.0, 1.0],
            "fg2m": [1.0, 1.0, 0.0, 1.0],
            "fga3": [0.0, 0.0, 0.0, 0.0],
            "fg3m": [0.0, 0.0, 0.0, 0.0],
            "fta": [0.0, 0.0, 0.0, 0.0],
            "ftm": [0.0, 0.0, 0.0, 0.0],
            "pts": [2.0, 2.0, 0.0, 2.0],
            "reb": [0.0, 0.0, 0.0, 0.0],
            "ast": [0.0, 0.0, 0.0, 0.0],
            "stl": [0.0, 0.0, 0.0, 0.0],
            "blk": [0.0, 0.0, 0.0, 0.0],
            "tov": [0.0, 0.0, 0.0, 0.0],
            "dk_fpts": [2.0, 2.0, 0.0, 2.0],
        }
    )

    cleaned, report = _sanitize_frame_to_expected_keys(
        worlds,
        expected_keys_df=expected_keys,
        key_cols=("game_id", "team_id", "player_id"),
        label="unit-test worlds",
    )

    assert report["rows_in"] == 4
    assert report["rows_out"] == 2
    assert report["dropped_null_key_rows"] == 1
    assert report["dropped_unexpected_key_rows"] == 1

    checks = _summarize_world_contracts_from_frame(cleaned)
    assert checks["team_minutes_not_240"] == 0
    assert checks["team_minutes_total_checks"] == 2
    assert cleaned["game_id"].tolist() == [1, 1]


def test_sanitize_frame_to_expected_keys_handles_sparse_large_index_labels() -> None:
    expected_keys = pd.DataFrame(
        {
            "game_id": [1, 1],
            "team_id": [10, 20],
            "player_id": [100, 200],
        }
    )
    worlds = pd.DataFrame(
        {
            "world_idx": [0, 0, 0, 1],
            "game_id": [1, 1, 9, 1],
            "team_id": [10, 20, 10, 20],
            "player_id": [100, 200, 100, 200],
            "minutes": [240.0, 240.0, 17.0, 240.0],
        }
    )
    worlds.index = pd.Index([0, 134224841, 223344556, 334455667], dtype="int64")

    cleaned, report = _sanitize_frame_to_expected_keys(
        worlds,
        expected_keys_df=expected_keys,
        key_cols=("game_id", "team_id", "player_id"),
        label="unit-test sparse-index worlds",
    )

    assert report["rows_in"] == 4
    assert report["rows_out"] == 3
    assert report["dropped_null_key_rows"] == 0
    assert report["dropped_unexpected_key_rows"] == 1
    assert cleaned["game_id"].tolist() == [1, 1, 1]
    assert cleaned["team_id"].tolist() == [10, 20, 20]


def test_coerce_world_game_date_normalizes_noncanonical_values() -> None:
    worlds = pd.DataFrame(
        {
            "world_idx": [0, 1, 2, 3],
            "game_date": ["2026-03%12", "2026-03-12", None, "2026-03-12 "],
            "game_id": [1, 1, 1, 1],
            "team_id": [10, 10, 10, 10],
            "player_id": [100, 100, 100, 100],
        }
    )

    normalized, report = _coerce_world_game_date(worlds, game_date="2026-03-12")

    assert report["applied"] is True
    assert report["normalized_rows"] == 3
    assert report["canonical_game_date"] == "2026-03-12"
    assert normalized["game_date"].eq("2026-03-12").all()


def test_publish_atomic_task_rejects_corrupt_worlds_before_pointer_promotion(
    tmp_path: Path,
) -> None:
    game_date = "2026-02-24"
    run_id = "20260224T220000Z"
    manifest_path = (
        tmp_path
        / "artifacts"
        / "runs"
        / "nba_live"
        / f"game_date={game_date}"
        / f"run={run_id}"
        / "manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        '{"run_id":"20260224T220000Z","as_of_ts":"2026-02-24T22:00:00Z","source_freshness":{"summary":{}}}',
        encoding="utf-8",
    )

    _atomic_write_validated_parquet(
        pd.DataFrame({"game_id": [1], "team_id": [10], "player_id": [100]}),
        tmp_path / "live" / "features_gtv2_v1" / game_date / f"run={run_id}" / "features.parquet",
        required_cols=("game_id", "team_id", "player_id"),
    )
    _atomic_write_validated_parquet(
        pd.DataFrame({"game_id": [1], "team_id": [10], "player_id": [100]}),
        tmp_path / "live" / "features_minutes_v1" / game_date / f"run={run_id}" / "features.parquet",
        required_cols=("game_id", "team_id", "player_id"),
    )
    _atomic_write_validated_parquet(
        pd.DataFrame(
            {"game_date": [game_date], "game_id": [1], "team_id": [10], "player_id": [100]}
        ),
        tmp_path
        / "artifacts"
        / "gtv2_scores"
        / f"game_date={game_date}"
        / f"run={run_id}"
        / "scores.parquet",
        required_cols=("game_date", "game_id", "team_id", "player_id"),
    )
    worlds_dir = (
        tmp_path / "artifacts" / "gtv2_worlds" / f"game_date={game_date}" / f"run={run_id}"
    )
    worlds_dir.mkdir(parents=True, exist_ok=True)
    (worlds_dir / "worlds.parquet").write_text("not a parquet file", encoding="utf-8")
    _atomic_write_validated_parquet(
        pd.DataFrame(
            {"game_date": [game_date], "game_id": [1], "team_id": [10], "player_id": [100]}
        ),
        worlds_dir / "projections.parquet",
        required_cols=("game_date", "game_id", "team_id", "player_id"),
    )
    _atomic_write_validated_parquet(
        pd.DataFrame(
            {"game_date": [game_date], "game_id": [1], "team_id": [10], "player_id": [100]}
        ),
        tmp_path
        / "artifacts"
        / "projections"
        / game_date
        / f"run={run_id}"
        / "projections.parquet",
        required_cols=("game_date", "game_id", "team_id", "player_id"),
    )
    _atomic_write_validated_parquet(
        pd.DataFrame({"player_id": [100], "pred_own_pct": [0.1]}),
        tmp_path
        / "silver"
        / "ownership_predictions"
        / game_date
        / f"run={run_id}"
        / "123.parquet",
        required_cols=("player_id",),
    )

    with writer_guard.PipelineWriterLock(data_root=tmp_path, run_id=run_id):
        with pytest.raises(RuntimeError, match="failed to open parquet|failed to stream-validate parquet contents"):
            publish_atomic_task.fn(
                game_date=game_date,
                run_id=run_id,
                manifest_path=manifest_path,
                data_root=tmp_path,
            )

    assert not (
        tmp_path / "artifacts" / "gtv2_worlds" / f"game_date={game_date}" / "LATEST" / "current.json"
    ).exists()


def test_publish_atomic_task_rejects_world_key_contract_violation_before_pointer_promotion(
    tmp_path: Path,
) -> None:
    game_date = "2026-02-24"
    run_id = "20260224T221500Z"
    manifest_path = (
        tmp_path
        / "artifacts"
        / "runs"
        / "nba_live"
        / f"game_date={game_date}"
        / f"run={run_id}"
        / "manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        '{"run_id":"20260224T221500Z","as_of_ts":"2026-02-24T22:15:00Z","source_freshness":{"summary":{}}}',
        encoding="utf-8",
    )

    _atomic_write_validated_parquet(
        pd.DataFrame({"game_id": [1], "team_id": [10], "player_id": [100]}),
        tmp_path / "live" / "features_gtv2_v1" / game_date / f"run={run_id}" / "features.parquet",
        required_cols=("game_id", "team_id", "player_id"),
    )
    _atomic_write_validated_parquet(
        pd.DataFrame({"game_id": [1], "team_id": [10], "player_id": [100]}),
        tmp_path / "live" / "features_minutes_v1" / game_date / f"run={run_id}" / "features.parquet",
        required_cols=("game_id", "team_id", "player_id"),
    )
    _atomic_write_validated_parquet(
        pd.DataFrame(
            {"game_date": [game_date], "game_id": [1], "team_id": [10], "player_id": [100]}
        ),
        tmp_path
        / "artifacts"
        / "gtv2_scores"
        / f"game_date={game_date}"
        / f"run={run_id}"
        / "scores.parquet",
        required_cols=("game_date", "game_id", "team_id", "player_id"),
    )
    _atomic_write_validated_parquet(
        pd.DataFrame(
            {
                "world_idx": [0, 0],
                "game_id": [1, 576460752325924364],
                "team_id": [10, 10],
                "player_id": [100, 100],
                "minutes": [240.0, 17.0],
            }
        ),
        tmp_path
        / "artifacts"
        / "gtv2_worlds"
        / f"game_date={game_date}"
        / f"run={run_id}"
        / "worlds.parquet",
        required_cols=("world_idx",),
    )
    _atomic_write_validated_parquet(
        pd.DataFrame(
            {"game_date": [game_date], "game_id": [1], "team_id": [10], "player_id": [100]}
        ),
        tmp_path
        / "artifacts"
        / "gtv2_worlds"
        / f"game_date={game_date}"
        / f"run={run_id}"
        / "projections.parquet",
        required_cols=("game_date", "game_id", "team_id", "player_id"),
    )
    _atomic_write_validated_parquet(
        pd.DataFrame(
            {"game_date": [game_date], "game_id": [1], "team_id": [10], "player_id": [100]}
        ),
        tmp_path
        / "artifacts"
        / "projections"
        / game_date
        / f"run={run_id}"
        / "projections.parquet",
        required_cols=("game_date", "game_id", "team_id", "player_id"),
    )
    _atomic_write_validated_parquet(
        pd.DataFrame({"player_id": [100], "pred_own_pct": [0.1]}),
        tmp_path
        / "silver"
        / "ownership_predictions"
        / game_date
        / f"run={run_id}"
        / "123.parquet",
        required_cols=("player_id",),
    )

    with writer_guard.PipelineWriterLock(data_root=tmp_path, run_id=run_id):
        with pytest.raises(RuntimeError, match="key contract failed"):
            publish_atomic_task.fn(
                game_date=game_date,
                run_id=run_id,
                manifest_path=manifest_path,
                data_root=tmp_path,
            )


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
    assert checks["team_minutes_not_240"] == 0
    assert checks["team_minutes_total_checks"] == 2
    assert checks["team_minutes_max_abs_drift"] == pytest.approx(0.0)
    assert checks["minutes_negative"] == 0


def test_summarize_world_contracts_from_frame_tolerates_small_float_drift() -> None:
    worlds = pd.DataFrame(
        {
            "world_idx": [0, 0, 1, 1],
            "game_id": [1, 1, 1, 1],
            "team_id": [10, 20, 10, 20],
            "player_id": [100, 200, 100, 200],
            "active": [1, 1, 1, 1],
            "minutes": [240.00003, 239.99997, 239.99998, 240.00002],
            "fga2": [1.0, 1.0, 1.0, 1.0],
            "fg2m": [1.0, 1.0, 1.0, 1.0],
            "fga3": [0.0, 0.0, 0.0, 0.0],
            "fg3m": [0.0, 0.0, 0.0, 0.0],
            "fta": [0.0, 0.0, 0.0, 0.0],
            "ftm": [0.0, 0.0, 0.0, 0.0],
            "pts": [2.0, 2.0, 2.0, 2.0],
            "reb": [0.0, 0.0, 0.0, 0.0],
            "ast": [0.0, 0.0, 0.0, 0.0],
            "stl": [0.0, 0.0, 0.0, 0.0],
            "blk": [0.0, 0.0, 0.0, 0.0],
            "tov": [0.0, 0.0, 0.0, 0.0],
            "dk_fpts": [2.0, 2.0, 2.0, 2.0],
        }
    )
    checks = _summarize_world_contracts_from_frame(worlds)
    assert checks["team_minutes_not_240"] == 0
    assert checks["team_minutes_total_checks"] == 4
    assert checks["team_minutes_max_abs_drift"] == pytest.approx(0.00003)


def test_summarize_world_contracts_from_frame_counts_inactive_nonzero_stats() -> None:
    worlds = pd.DataFrame(
        {
            "world_idx": [0, 0, 0],
            "game_id": [1, 1, 1],
            "team_id": [10, 10, 20],
            "player_id": [100, 101, 200],
            "active": [0, 0, 1],
            "minutes": [0.0, 2.0, 238.0],
            "fga2": [0.0, 1.0, 4.0],
            "fg2m": [0.0, 1.0, 3.0],
            "fga3": [0.0, 0.0, 1.0],
            "fg3m": [0.0, 0.0, 1.0],
            "fta": [0.0, 0.0, 2.0],
            "ftm": [0.0, 0.0, 2.0],
            "pts": [0.0, 2.0, 9.0],
            "reb": [0.0, 1.0, 4.0],
            "ast": [0.0, 0.0, 5.0],
            "stl": [0.0, 0.0, 1.0],
            "blk": [0.0, 0.0, 1.0],
            "tov": [0.0, 0.0, 2.0],
            "dk_fpts": [0.0, 5.0, 35.0],
        }
    )
    checks = _summarize_world_contracts_from_frame(worlds)
    assert checks["inactive_nonzero_stats"] == 1
    assert checks["inactive_nonzero_fpts_proxy"] == 1


def test_team_minutes_sums_without_pandas_groupby_matches_groupby_reference() -> None:
    world_count = 4096
    game_ids = [22500901, 22500902, 22500903, 22500904]
    team_ids = [1610612737, 1610612738]
    player_slots = [1, 2, 3]

    rows: list[dict[str, Any]] = []
    for world_idx in range(world_count):
        for game_id in game_ids:
            for team_id in team_ids:
                for slot in player_slots:
                    rows.append(
                        {
                            "world_idx": world_idx,
                            "game_id": str(game_id),  # object/string input is common in live merges
                            "team_id": team_id,
                            "player_id": int(team_id * 100 + slot),
                            "minutes": 80.0,
                        }
                    )
    worlds = pd.DataFrame(rows)
    worlds.loc[0, "minutes"] = None
    worlds = pd.concat(
        [
            worlds,
            pd.DataFrame(
                {
                    "world_idx": [float("nan"), 12],
                    "game_id": [22500901, None],
                    "team_id": [1610612737, 1610612738],
                    "player_id": [999001, 999002],
                    "minutes": [31.0, 29.0],
                }
            ),
        ],
        ignore_index=True,
    )

    uniq_world, uniq_game, uniq_team, sums = _team_minutes_sums_without_pandas_groupby(
        world_idx_col=worlds["world_idx"],
        game_id_col=worlds["game_id"],
        team_id_col=worlds["team_id"],
        minutes_col=worlds["minutes"],
    )
    actual = pd.DataFrame(
        {
            "world_idx": uniq_world.astype("int64", copy=False),
            "game_id": uniq_game.astype("int64", copy=False),
            "team_id": uniq_team.astype("int64", copy=False),
            "minutes": sums,
        }
    ).sort_values(["world_idx", "game_id", "team_id"]).reset_index(drop=True)

    ref = worlds.copy()
    ref["world_idx"] = pd.to_numeric(ref["world_idx"], errors="coerce")
    ref["game_id"] = pd.to_numeric(ref["game_id"], errors="coerce")
    ref["team_id"] = pd.to_numeric(ref["team_id"], errors="coerce")
    ref["minutes"] = pd.to_numeric(ref["minutes"], errors="coerce").fillna(0.0)
    ref = ref.loc[
        ref["world_idx"].notna() & ref["game_id"].notna() & ref["team_id"].notna()
    ].copy()
    ref["world_idx"] = ref["world_idx"].astype("int64", copy=False)
    ref["game_id"] = ref["game_id"].astype("int64", copy=False)
    ref["team_id"] = ref["team_id"].astype("int64", copy=False)
    expected = (
        ref.groupby(["world_idx", "game_id", "team_id"], as_index=False, sort=False)[
            "minutes"
        ]
        .sum()
        .sort_values(["world_idx", "game_id", "team_id"])
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(actual, expected)


def test_group_mean_by_keys_without_pandas_groupby_matches_groupby_reference() -> None:
    df = pd.DataFrame(
        {
            "game_id": ["1", "1", "2", "2", "2", None],
            "team_id": [10, 10, 20, 20, 21, 21],
            "player_id": [100, 100, 200, 200, 210, 210],
            "pts": [10.0, 14.0, 20.0, 26.0, 7.0, 99.0],
            "reb": [4.0, 6.0, 8.0, 10.0, 3.0, 99.0],
            "ast": [2.0, 4.0, 5.0, 7.0, 1.0, 99.0],
        }
    )

    actual = _group_mean_by_keys_without_pandas_groupby(
        df,
        key_cols=("game_id", "team_id", "player_id"),
        value_cols=("pts", "reb", "ast"),
        label="unit-test group means",
    ).sort_values(["game_id", "team_id", "player_id"]).reset_index(drop=True)

    ref = df.copy()
    ref["game_id"] = pd.to_numeric(ref["game_id"], errors="coerce")
    ref["team_id"] = pd.to_numeric(ref["team_id"], errors="coerce")
    ref["player_id"] = pd.to_numeric(ref["player_id"], errors="coerce")
    ref = ref.dropna(subset=["game_id", "team_id", "player_id"])
    ref["game_id"] = ref["game_id"].astype("int64", copy=False)
    ref["team_id"] = ref["team_id"].astype("int64", copy=False)
    ref["player_id"] = ref["player_id"].astype("int64", copy=False)
    expected = (
        ref.groupby(["game_id", "team_id", "player_id"], as_index=False, sort=False)[
            ["pts", "reb", "ast"]
        ]
        .mean()
        .sort_values(["game_id", "team_id", "player_id"])
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(actual, expected)


def test_factorize_int_key_arrays_preserve_order_handles_large_sparse_keys() -> None:
    game_ids = np.array(
        [5_000_000_000_001, 5_000_000_000_001, 6_000_000_000_002, 5_000_000_000_001, 5_000_000_000_001],
        dtype=np.int64,
    )
    team_ids = np.array([10, 10, 10, 11, 10], dtype=np.int64)
    player_ids = np.array([100, 100, 100, 100, 100], dtype=np.int64)

    codes, uniques = _factorize_int_key_arrays_preserve_order(game_ids, team_ids, player_ids)

    assert codes.tolist() == [0, 0, 1, 2, 0]
    assert len(uniques) == 3
    assert uniques[0].tolist() == [5_000_000_000_001, 6_000_000_000_002, 5_000_000_000_001]
    assert uniques[1].tolist() == [10, 10, 11]
    assert uniques[2].tolist() == [100, 100, 100]


def test_factorize_int_key_arrays_preserve_order_stress_codes_stay_in_bounds() -> None:
    rng = np.random.default_rng(42)
    row_count = 20_000
    game_ids = rng.choice(
        np.array([22500963, 22500968, 5_000_000_000_001, 7_205_759_405_470_556], dtype=np.int64),
        size=row_count,
        replace=True,
    )
    team_ids = rng.choice(
        np.array([1610612741, 1610612746, 1610612747, 1610612752], dtype=np.int64),
        size=row_count,
        replace=True,
    )
    player_ids = rng.integers(1000, 1600, size=row_count, dtype=np.int64)

    codes, uniques = _factorize_int_key_arrays_preserve_order(game_ids, team_ids, player_ids)

    assert len(codes) == row_count
    assert len(uniques) == 3
    group_count = len(uniques[0])
    assert group_count > 0
    assert int(codes.min()) >= 0
    assert int(codes.max()) < group_count

    first_seen: dict[tuple[int, int, int], int] = {}
    for idx, key in enumerate(zip(game_ids.tolist(), team_ids.tolist(), player_ids.tolist(), strict=False)):
        if key not in first_seen:
            first_seen[key] = len(first_seen)
        assert int(codes[idx]) == first_seen[key]


def test_repair_world_frame_contract_fields_normalizes_game_id_and_makes() -> None:
    worlds = pd.DataFrame(
        {
            "world_idx": [0, 0, 0, 0],
            "game_id": [5723706, 22500922, 22500922, 22500922],
            "game_id_norm": ["0022500922", "0022500922", "0022500922", "0022500922"],
            "team_id": [1610612747, 1610612747, 1610612752, 1610612752],
            "player_id": [1, 2, 3, 4],
            "active": [1, 1, 1, 1],
            "minutes": [120.0, 120.0, 120.0, 120.0],
            "fga2": [4.0, 5.0, 6.0, 7.0],
            "fg2m": [5.0, 4.0, 6.0, 7.0],
            "fga3": [0.001, 3.0, 4.0, 5.0],
            "fg3m": [0.9, 1.0, 2.0, 3.0],
            "fta": [2.0, 4.0, 3.0, 1.0],
            "ftm": [5.0, 2.0, 1.0, 0.5],
            "oreb": [1.0, 1.0, 1.0, 1.0],
            "dreb": [2.0, 3.0, 4.0, 5.0],
            "ast": [1.0, 2.0, 3.0, 4.0],
            "stl": [0.0, 0.0, 0.0, 0.0],
            "blk": [0.0, 0.0, 0.0, 0.0],
            "tov": [0.0, 0.0, 0.0, 0.0],
            "dk_fpts": [0.0, 0.0, 0.0, 0.0],
        }
    )

    repaired, report = _repair_world_frame_contract_fields(worlds)
    checks = _summarize_world_contracts_from_frame(repaired)

    assert bool(report["applied"]) is True
    assert report["game_id_from_norm_rows"] == 1
    assert report["fg2m_clipped_to_fga2_rows"] == 1
    assert report["fg3m_clipped_to_fga3_rows"] == 1
    assert report["ftm_clipped_to_fta_rows"] == 1
    assert int(repaired["game_id"].nunique()) == 1
    assert checks["team_minutes_not_240"] == 0
    assert checks["fg2m_gt_fga2"] == 0
    assert checks["fg3m_gt_fga3"] == 0
    assert checks["ftm_gt_fta"] == 0


def test_repair_world_frame_contract_fields_preserves_uplifted_totals_without_repairs() -> None:
    worlds = pd.DataFrame(
        {
            "world_idx": [0, 1],
            "game_id": [22500931, 22500931],
            "team_id": [1610612739, 1610612739],
            "player_id": [1628378, 1628378],
            "active": [1, 1],
            "minutes": [30.0, 31.0],
            "fga2": [6.0, 7.0],
            "fg2m": [4.0, 5.0],
            "fga3": [8.0, 8.0],
            "fg3m": [4.0, 4.0],
            "fta": [6.0, 5.0],
            "ftm": [5.0, 4.0],
            "oreb": [1.0, 1.0],
            "dreb": [3.0, 4.0],
            "pts": [31.0, 33.0],  # Intentionally uplifted beyond boxscore-derived totals.
            "reb": [7.0, 8.0],  # Intentionally uplifted beyond oreb+dreb.
            "ast": [6.0, 7.0],
            "stl": [1.0, 1.0],
            "blk": [0.0, 0.0],
            "tov": [2.0, 3.0],
            "dk_fpts": [49.0, 51.0],  # Intentionally uplifted and should be preserved.
        }
    )
    original_pts = worlds["pts"].copy()
    original_reb = worlds["reb"].copy()
    original_dk = worlds["dk_fpts"].copy()

    repaired, report = _repair_world_frame_contract_fields(worlds)

    assert bool(report["applied"]) is False
    assert report["fg2m_clipped_to_fga2_rows"] == 0
    assert report["fg3m_clipped_to_fga3_rows"] == 0
    assert report["ftm_clipped_to_fta_rows"] == 0
    assert repaired["pts"].tolist() == pytest.approx(original_pts.tolist())
    assert repaired["reb"].tolist() == pytest.approx(original_reb.tolist())
    assert repaired["dk_fpts"].tolist() == pytest.approx(original_dk.tolist())


def test_repair_world_frame_contract_fields_drops_bad_world_game_minutes_slices() -> None:
    worlds = pd.DataFrame(
        {
            "world_idx": [0, 0, 0, 0, 1, 1, 1, 1],
            "game_id": [22500932] * 8,
            "team_id": [1610612751, 1610612751, 1610612763, 1610612763] * 2,
            "player_id": [1, 2, 3, 4, 1, 2, 3, 4],
            "active": [1] * 8,
            "minutes": [120.0, 120.0, 120.0, 120.0, 100.0, 100.0, 120.0, 120.0],
            "fga2": [0.0] * 8,
            "fg2m": [0.0] * 8,
            "fga3": [0.0] * 8,
            "fg3m": [0.0] * 8,
            "fta": [0.0] * 8,
            "ftm": [0.0] * 8,
            "pts": [0.0] * 8,
            "reb": [0.0] * 8,
            "ast": [0.0] * 8,
            "stl": [0.0] * 8,
            "blk": [0.0] * 8,
            "tov": [0.0] * 8,
            "dk_fpts": [0.0] * 8,
        }
    )

    repaired, report = _repair_world_frame_contract_fields(worlds)
    checks = _summarize_world_contracts_from_frame(repaired)

    assert bool(report["applied"]) is True
    assert report["dropped_bad_world_game_pairs"] == 1
    assert report["dropped_bad_world_rows"] == 4
    assert int(len(repaired)) == 4
    assert int(pd.to_numeric(repaired["world_idx"], errors="coerce").nunique()) == 1
    assert checks["team_minutes_not_240"] == 0


def test_repair_world_frame_contract_fields_handles_sparse_index_world_rows() -> None:
    worlds = pd.DataFrame(
        {
            "world_idx": [0, 0, 0, 0, 1, 1, 1, 1],
            "game_id": [22500932] * 8,
            "team_id": [1610612751, 1610612751, 1610612763, 1610612763] * 2,
            "player_id": [1, 2, 3, 4, 1, 2, 3, 4],
            "active": [1] * 8,
            "minutes": [120.0, 120.0, 120.0, 120.0, 100.0, 100.0, 120.0, 120.0],
            "fga2": [0.0] * 8,
            "fg2m": [0.0] * 8,
            "fga3": [0.0] * 8,
            "fg3m": [0.0] * 8,
            "fta": [0.0] * 8,
            "ftm": [0.0] * 8,
            "pts": [0.0] * 8,
            "reb": [0.0] * 8,
            "ast": [0.0] * 8,
            "stl": [0.0] * 8,
            "blk": [0.0] * 8,
            "tov": [0.0] * 8,
            "dk_fpts": [0.0] * 8,
        }
    )
    worlds.index = pd.Index([100, 200, 300, 400, 1000, 1100, 1200, 1300])

    repaired, report = _repair_world_frame_contract_fields(worlds)
    checks = _summarize_world_contracts_from_frame(repaired)

    assert bool(report["applied"]) is True
    assert report["dropped_bad_world_game_pairs"] == 1
    assert report["dropped_bad_world_rows"] == 4
    assert int(len(repaired)) == 4
    assert int(pd.to_numeric(repaired["world_idx"], errors="coerce").nunique()) == 1
    assert checks["team_minutes_not_240"] == 0


def test_atomic_write_validated_parquet_retries_transient_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_path = tmp_path / "artifact.parquet"
    df = pd.DataFrame({"game_id": [1, 2], "team_id": [10, 20], "player_id": [100, 200]})
    calls = {"count": 0}
    original = _stream_validate_parquet

    def flaky_validate(
        path: Path,
        *,
        expected_rows: int | None = None,
        required_cols: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError(
                f"failed to stream-validate parquet contents: {path}"
            ) from OSError("Corrupt snappy compressed data.")
        return original(path, expected_rows=expected_rows, required_cols=required_cols)

    monkeypatch.setattr(
        "prefect_flows.live_nba_pipeline_v3._stream_validate_parquet",
        flaky_validate,
    )

    report = _atomic_write_validated_parquet(
        df,
        out_path,
        required_cols=("game_id", "team_id", "player_id"),
    )
    assert out_path.exists()
    assert calls["count"] == 2
    assert report["rows"] == 2


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


def test_load_rotowire_props_long_from_bronze_normalizes_without_dataframe_masks(
    tmp_path: Path,
) -> None:
    game_date = pd.Timestamp("2026-03-01")
    _write(
        tmp_path / "game_date=2026-03-01" / "props_1.parquet",
        pd.DataFrame(
            {
                "player_name": ["Player One", "Player One", None, "Player Two"],
                "team": ["NY", "NYK", "BOS", "BOS"],
                "prop_type": ["pts", "pts", "ast", "bogus"],
                "line": [22.5, 23.5, 6.5, 3.5],
                "book": ["draftkings", "fanduel", "betmgm", "draftkings"],
                "scraped_at": [
                    "2026-03-01T17:00:00Z",
                    "2026-03-01T17:00:00Z",
                    "2026-03-01T17:00:00Z",
                    None,
                ],
                "over_odds": [-110, -105, -120, -110],
                "implied_over_prob": [None, 0.54, 0.51, 0.5],
                "game_id": ["123", "123", "123", "123"],
            }
        ),
    )
    _write(
        tmp_path / "game_date=2026-03-01" / "props_missing_cols.parquet",
        pd.DataFrame({"player_name": ["Ignored"], "team": ["NYK"]}),
    )

    actual = load_rotowire_props_long_from_bronze(
        rotowire_props_root=tmp_path,
        game_date=game_date,
    ).sort_values(["player_name", "prop_key"]).reset_index(drop=True)

    assert list(actual["player_name"]) == ["Player One"]
    assert list(actual["team_tricode"]) == ["NYK"]
    assert list(actual["prop_key"]) == ["pts"]
    assert actual.loc[0, "line"] == pytest.approx(23.0)
    assert actual.loc[0, "p_over"] == pytest.approx((0.5238095238 + 0.54) / 2.0)
    assert actual.loc[0, "line_std"] == pytest.approx(0.5)
    assert actual.loc[0, "books"] == pytest.approx(2.0)
    assert actual.loc[0, "action_game_id"] == 123


def test_run_python_module_retries_retryable_exit_codes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempts: list[int] = []

    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        attempts.append(len(attempts) + 1)
        if len(attempts) == 1:
            return subprocess.CompletedProcess(
                args=args[0],
                returncode=139,
                stdout="",
                stderr="segfault",
            )
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="ok",
            stderr="",
        )

    monkeypatch.setattr("prefect_flows.live_nba_pipeline_v3.subprocess.run", fake_run)
    monkeypatch.setattr("prefect_flows.live_nba_pipeline_v3.time.sleep", lambda _: None)

    _run_python_module(
        "projections.cli.fake_module",
        ["--flag"],
        data_root=tmp_path,
        timeout_s=5,
    )

    assert attempts == [1, 2]
