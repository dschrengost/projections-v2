from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.contest_sim.field_library import FieldLibrary
from projections.contest_sim.scoring_models import ContestConfig, ContestSimResult, SummaryStats
from projections.post_contest import replay_service


def _write_results_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_contest_entries_dedupes_entry_rows(tmp_path: Path) -> None:
    results_path = (
        tmp_path
        / "bronze"
        / "dk_contests"
        / "nba_gpp_data"
        / "2099-01-01"
        / "results"
        / "contest_123_results.csv"
    )
    lineup = "PG Alpha SG Beta SF Gamma PF Delta C Epsilon G Zeta F Eta UTIL Theta"
    _write_results_csv(
        results_path,
        [
            {"Rank": 1, "EntryId": 11, "EntryName": "daniel", "Points": 300.5, "Lineup": lineup, "Player": "Alpha"},
            {"Rank": 1, "EntryId": 11, "EntryName": "daniel", "Points": 300.5, "Lineup": lineup, "Player": "Beta"},
            {"Rank": 2, "EntryId": 12, "EntryName": "villain", "Points": 295.0, "Lineup": lineup, "Player": "Alpha"},
        ],
    )

    meta, entries = replay_service.load_contest_entries(
        contest_id="123",
        game_date="2099-01-01",
        data_root=tmp_path,
    )

    assert meta.field_size == 2
    assert len(entries) == 2
    assert entries[0].entry_id == "11"
    assert entries[0].lineup_names == ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]


def test_prepare_post_contest_replay_builds_exact_opponent_field(tmp_path: Path, monkeypatch) -> None:
    results_path = (
        tmp_path
        / "bronze"
        / "dk_contests"
        / "nba_gpp_data"
        / "2099-01-01"
        / "results"
        / "contest_123_results.csv"
    )
    lineup_a = "PG Alpha SG Beta SF Gamma PF Delta C Epsilon G Zeta F Eta UTIL Theta"
    lineup_b = "PG Iota SG Kappa SF Lambda PF Mu C Nu G Xi F Omicron UTIL Pi"
    lineup_c = "PG Rho SG Sigma SF Tau PF Upsilon C Phi G Chi F Psi UTIL Omega"
    _write_results_csv(
        results_path,
        [
            {"Rank": 1, "EntryId": 11, "EntryName": "daniel (1/2)", "Points": 300.5, "Lineup": lineup_a},
            {"Rank": 2, "EntryId": 12, "EntryName": "villain a", "Points": 295.0, "Lineup": lineup_b},
            {"Rank": 3, "EntryId": 13, "EntryName": "villain b", "Points": 294.0, "Lineup": lineup_b},
            {"Rank": 4, "EntryId": 14, "EntryName": "daniel (2/2)", "Points": 292.5, "Lineup": lineup_c},
        ],
    )

    all_names = [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
        "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi",
        "Rho", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega",
    ]
    monkeypatch.setattr(
        replay_service,
        "build_player_pool",
        lambda **kwargs: [
            {"player_id": str(idx), "name": name, "dk_id": idx}
            for idx, name in enumerate(all_names, start=1)
        ],
    )
    monkeypatch.setattr(
        replay_service,
        "_load_dk_nba_draftable_ids_by_player",
        lambda draft_group_id: ({}, {}),
    )

    prepared = replay_service.prepare_post_contest_replay(
        contest_id="123",
        game_date="2099-01-01",
        user_pattern="daniel",
        draft_group_id=999,
        data_root=tmp_path,
    )

    assert prepared.meta.field_size == 4
    assert len(prepared.user_entries) == 2
    assert prepared.user_weights == [1, 1]
    assert prepared.opponent_field_library.weights == [2]
    assert prepared.resolution_stats["resolved_user_entry_count"] == 2


def test_run_post_contest_replay_passes_user_and_field_weights(tmp_path: Path, monkeypatch) -> None:
    results_path = (
        tmp_path
        / "bronze"
        / "dk_contests"
        / "nba_gpp_data"
        / "2099-01-01"
        / "results"
        / "contest_123_results.csv"
    )
    lineup_a = "PG Alpha SG Beta SF Gamma PF Delta C Epsilon G Zeta F Eta UTIL Theta"
    lineup_b = "PG Iota SG Kappa SF Lambda PF Mu C Nu G Xi F Omicron UTIL Pi"
    _write_results_csv(
        results_path,
        [
            {"Rank": 1, "EntryId": 11, "EntryName": "daniel", "Points": 300.5, "Lineup": lineup_a},
            {"Rank": 2, "EntryId": 12, "EntryName": "villain a", "Points": 295.0, "Lineup": lineup_b},
            {"Rank": 3, "EntryId": 13, "EntryName": "villain b", "Points": 294.0, "Lineup": lineup_b},
        ],
    )

    all_names = [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
        "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi",
    ]
    monkeypatch.setattr(
        replay_service,
        "build_player_pool",
        lambda **kwargs: [
            {"player_id": str(idx), "name": name, "dk_id": idx}
            for idx, name in enumerate(all_names, start=1)
        ],
    )
    monkeypatch.setattr(
        replay_service,
        "_load_dk_nba_draftable_ids_by_player",
        lambda draft_group_id: ({}, {}),
    )

    captured: dict[str, object] = {}

    def _fake_run_contest_simulation(**kwargs) -> ContestSimResult:
        captured.update(kwargs)
        return ContestSimResult(
            results=[],
            config=ContestConfig(field_size=3, entry_fee=1.0, archetype="medium"),
            stats=SummaryStats(
                lineup_count=1,
                worlds_count=10,
                avg_ev=0.0,
                avg_roi=0.0,
                positive_ev_count=0,
                best_ev_lineup_id=0,
                best_win_rate_lineup_id=0,
                best_top1pct_lineup_id=0,
                debug={},
            ),
        )

    monkeypatch.setattr(replay_service, "run_contest_simulation", _fake_run_contest_simulation)

    replay_run = replay_service.run_post_contest_replay(
        contest_id="123",
        game_date="2099-01-01",
        user_pattern="daniel",
        draft_group_id=999,
        entry_fee=1.0,
        data_root=tmp_path,
    )

    assert replay_run.run_meta["mode"] == "exact_replay"
    assert captured["field_size_override"] == 3
    assert captured["user_weights"] == [1]
    assert captured["field_weights"] == [2]
    assert captured["field_lineups"] == [sorted([str(i) for i in range(9, 17)])]
