from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from projections.contest_sim.field_library import FieldLibrary
from projections.contest_sim.scoring_models import (
    ContestConfig,
    ContestSimResult,
    LineupEVResult,
    SummaryStats,
)
from projections.post_contest import replay_analytics_service
from projections.post_contest.replay_models import (
    ContestReplayMeta,
    ContestReplayRun,
    PreparedReplayContext,
    ResolvedContestReplayEntry,
)


def _fake_sim_result(lineups: list[list[str]]) -> ContestSimResult:
    results = []
    for idx, lineup in enumerate(lineups):
        results.append(
            LineupEVResult(
                lineup_id=idx,
                player_ids=list(lineup),
                mean=100.0 + idx,
                std=10.0,
                p90=120.0 + idx,
                p95=125.0 + idx,
                expected_payout=2.0 + idx,
                expected_value=1.0 + idx,
                roi=1.0 + idx,
                win_rate=0.01 * (idx + 1),
                top_1pct_rate=0.02 * (idx + 1),
                top_5pct_rate=0.05 * (idx + 1),
                top_10pct_rate=0.10 * (idx + 1),
                cash_rate=0.20 * (idx + 1),
            )
        )
    return ContestSimResult(
        results=results,
        config=ContestConfig(field_size=10, entry_fee=1.0, archetype="medium"),
        stats=SummaryStats(
            lineup_count=len(results),
            worlds_count=3,
            avg_ev=1.0,
            avg_roi=1.0,
            positive_ev_count=len(results),
            best_ev_lineup_id=0,
            best_win_rate_lineup_id=0,
            best_top1pct_lineup_id=0,
            debug={},
        ),
    )


def test_find_latest_export_manifest_filters_by_contest(tmp_path: Path) -> None:
    exports_dir = tmp_path / "contests" / "dk" / "game_date=2099-01-01" / "dg=999" / "exports"
    exports_dir.mkdir(parents=True)
    old_manifest = exports_dir / "export_old_manifest.json"
    new_manifest = exports_dir / "export_new_manifest.json"
    old_manifest.write_text(json.dumps({"contest_ids": ["123"], "created_at_utc": "2099-01-01T20:00:00Z"}))
    new_manifest.write_text(json.dumps({"contest_ids": ["123"], "created_at_utc": "2099-01-01T21:00:00Z"}))
    (exports_dir / "export_irrelevant_manifest.json").write_text(
        json.dumps({"contest_ids": ["999"], "created_at_utc": "2099-01-01T22:00:00Z"})
    )

    found = replay_analytics_service.find_latest_export_manifest(
        game_date="2099-01-01",
        draft_group_id=999,
        contest_id="123",
        data_root=tmp_path,
    )

    assert found == new_manifest


def test_build_post_contest_replay_analytics_writes_all_artifacts(tmp_path: Path, monkeypatch) -> None:
    results_path = tmp_path / "contest_123_results.csv"
    pd.DataFrame(
        [
            {"Player": "Alpha", "FPTS": 30.0},
            {"Player": "Beta", "FPTS": 25.0},
            {"Player": "Gamma", "FPTS": 20.0},
            {"Player": "Delta", "FPTS": 15.0},
            {"Player": "Epsilon", "FPTS": 10.0},
            {"Player": "Zeta", "FPTS": 8.0},
            {"Player": "Eta", "FPTS": 7.0},
            {"Player": "Theta", "FPTS": 6.0},
            {"Player": "Iota", "FPTS": 29.0},
            {"Player": "Kappa", "FPTS": 24.0},
            {"Player": "Lambda", "FPTS": 19.0},
            {"Player": "Mu", "FPTS": 14.0},
            {"Player": "Nu", "FPTS": 9.0},
            {"Player": "Xi", "FPTS": 7.0},
            {"Player": "Omicron", "FPTS": 6.0},
            {"Player": "Pi", "FPTS": 5.0},
        ]
    ).to_csv(results_path, index=False)

    prepared = PreparedReplayContext(
        meta=ContestReplayMeta(
            game_date="2099-01-01",
            contest_id="123",
            contest_name="Test Contest",
            draft_group_id=999,
            entry_fee=1.0,
            field_size=3,
            results_path=str(results_path),
        ),
        entries=[],
        resolved_entries=[
            ResolvedContestReplayEntry(
                entry_id="1",
                entry_name="daniel",
                rank=1,
                points=121.0,
                lineup_names=["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"],
                raw_lineup="",
                lineup_key="",
                prize=10.0,
                player_ids=[str(i) for i in range(1, 9)],
                unresolved_names=[],
            ),
            ResolvedContestReplayEntry(
                entry_id="2",
                entry_name="villain",
                rank=2,
                points=119.0,
                lineup_names=["Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi"],
                raw_lineup="",
                lineup_key="",
                prize=0.0,
                player_ids=[str(i) for i in range(9, 17)],
                unresolved_names=[],
            ),
            ResolvedContestReplayEntry(
                entry_id="3",
                entry_name="villain2",
                rank=3,
                points=119.0,
                lineup_names=["Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi"],
                raw_lineup="",
                lineup_key="",
                prize=0.0,
                player_ids=[str(i) for i in range(9, 17)],
                unresolved_names=[],
            ),
        ],
        user_entries=[
            ResolvedContestReplayEntry(
                entry_id="1",
                entry_name="daniel",
                rank=1,
                points=121.0,
                lineup_names=["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"],
                raw_lineup="",
                lineup_key="",
                prize=10.0,
                player_ids=[str(i) for i in range(1, 9)],
                unresolved_names=[],
            )
        ],
        user_lineups=[[str(i) for i in range(1, 9)]],
        user_weights=[1],
        opponent_field_library=FieldLibrary(
            lineups=[[str(i) for i in range(9, 17)]],
            weights=[2],
            meta={"source": "actual_contest_results"},
        ),
        resolution_stats={},
    )
    replay_run = ContestReplayRun(
        prepared=prepared,
        simulation=_fake_sim_result([[str(i) for i in range(1, 9)]]),
        run_meta={},
    )

    monkeypatch.setattr(replay_analytics_service, "run_post_contest_replay", lambda **kwargs: replay_run)
    monkeypatch.setattr(
        replay_analytics_service,
        "build_player_pool",
        lambda **kwargs: [
            {"player_id": str(i), "name": name, "team": ("A" if i % 2 == 0 else "B"), "positions": ["UTIL"], "salary": 5000, "own_proj": 12.5, "proj": 30.0, "game_matchup": "A@B"}
            for i, name in enumerate(
                ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi"],
                start=1,
            )
        ],
    )
    monkeypatch.setattr(
        replay_analytics_service,
        "load_player_worlds",
        lambda **kwargs: type(
            "PW",
            (),
            {
                "fpts_matrix": np.asarray(
                    [
                        [20.0] * 16,
                        [30.0] * 16,
                        [40.0] * 16,
                    ],
                    dtype=np.float64,
                ),
                "minutes_matrix": np.asarray(
                    [
                        [20.0] * 16,
                        [30.0] * 16,
                        [40.0] * 16,
                    ],
                    dtype=np.float64,
                ),
                "player_index": {str(i): i - 1 for i in range(1, 17)},
            },
        )(),
    )
    monkeypatch.setattr(
        replay_analytics_service,
        "_load_actual_minutes_lookup",
        lambda **kwargs: {str(i): 30.0 for i in range(1, 17)},
    )
    monkeypatch.setattr(
        replay_analytics_service,
        "_build_name_to_internal_map",
        lambda **kwargs: (
            {name.lower(): str(i) for i, name in enumerate(["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi"], start=1)},
            {},
            {},
            {},
        ),
    )
    monkeypatch.setattr(
        replay_analytics_service,
        "load_or_build_field_library",
        lambda **kwargs: (
            FieldLibrary(
                lineups=[[str(i) for i in range(1, 9)]],
                weights=[100],
                meta={"version": "v1_calibrated"},
            ),
            tmp_path / "modeled.json",
            False,
        ),
    )

    manifest_path = tmp_path / "manifest.json"
    eval_path = tmp_path / "eval_lineups.csv"
    pd.DataFrame(
        [
            {"contest_id": 123, **{f"p{i}_id": i for i in range(1, 9)}},
            {"contest_id": 123, **{f"p{i}_id": (999999 if i == 8 else i + 8) for i in range(1, 9)}},
        ]
    ).to_csv(eval_path, index=False)
    manifest_path.write_text(json.dumps({"contest_ids": ["123"], "eval_lineups_path": str(eval_path), "created_at_utc": "2099-01-01T20:00:00Z"}))
    monkeypatch.setattr(
        replay_analytics_service,
        "find_latest_export_manifest",
        lambda **kwargs: manifest_path,
    )
    monkeypatch.setattr(
        replay_analytics_service,
        "run_contest_simulation",
        lambda **kwargs: _fake_sim_result(kwargs["user_lineups"]),
    )

    bundle = replay_analytics_service.build_post_contest_replay_analytics(
        contest_id="123",
        game_date="2099-01-01",
        user_pattern="daniel",
        data_root=tmp_path,
        output_dir=tmp_path / "out",
    )

    assert bundle.player_calibration_path.exists()
    assert bundle.lineup_calibration_path.exists()
    assert bundle.field_calibration_path.exists()
    assert bundle.regret_summary_path.exists()
    assert bundle.summary_path.exists()
    summary = json.loads(bundle.summary_path.read_text())
    assert summary["replay_trust_status"] == "broken"
    assert summary["candidate_universe_source"] == "eval_lineups_csv"
    assert summary["candidate_universe_lineup_count"] == 1
    assert summary["resolution"]["candidate_missing_player_id_count"] == 1
    assert "decision_guidance" in summary
    assert "attribution_summary" in summary

    regret_df = pd.read_parquet(bundle.regret_summary_path)
    assert "best_entered_lineup_players" in regret_df.columns
    assert "best_candidate_lineup_players" in regret_df.columns
    assert "Alpha" in str(regret_df.iloc[0]["best_entered_lineup_players"])
