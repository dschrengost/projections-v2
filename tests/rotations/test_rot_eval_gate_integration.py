from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from projections.rotations.eval import run_rotation_generator_eval
from projections.rotations.rotation_gate import GateConfig


def _write_synth_rot_bundle(rot_bundle_dir: Path) -> None:
    rot_bundle_dir.mkdir(parents=True, exist_ok=True)

    # Four 12-minute segments -> full regulation game.
    events = pd.DataFrame(
        [
            {
                "season_id": "0000-01",
                "game_id": "0000000001",
                "team_id": 10,
                "opponent_team_id": 20,
                "is_home": True,
                "segment_idx": 0,
                "duration_sec": 720,
                "lineup_p1": 100,
                "lineup_p2": 101,
                "lineup_p3": 102,
                "lineup_p4": 103,
                "lineup_p5": 104,
            },
            {
                "season_id": "0000-01",
                "game_id": "0000000001",
                "team_id": 10,
                "opponent_team_id": 20,
                "is_home": True,
                "segment_idx": 1,
                "duration_sec": 720,
                "lineup_p1": 100,
                "lineup_p2": 101,
                "lineup_p3": 102,
                "lineup_p4": 103,
                "lineup_p5": 105,
            },
            {
                "season_id": "0000-01",
                "game_id": "0000000001",
                "team_id": 10,
                "opponent_team_id": 20,
                "is_home": True,
                "segment_idx": 2,
                "duration_sec": 720,
                "lineup_p1": 100,
                "lineup_p2": 101,
                "lineup_p3": 102,
                "lineup_p4": 106,
                "lineup_p5": 104,
            },
            {
                "season_id": "0000-01",
                "game_id": "0000000001",
                "team_id": 10,
                "opponent_team_id": 20,
                "is_home": True,
                "segment_idx": 3,
                "duration_sec": 720,
                "lineup_p1": 100,
                "lineup_p2": 101,
                "lineup_p3": 102,
                "lineup_p4": 103,
                "lineup_p5": 104,
            },
        ]
    )

    # Minutes implied by segments above:
    # 100..102: 48; 103..104: 36; 105..106: 12; 107: 0.5 (truth participant).
    minutes = {
        100: 48.0,
        101: 48.0,
        102: 48.0,
        103: 36.0,
        104: 36.0,
        105: 12.0,
        106: 12.0,
        107: 0.5,
    }
    starters = {100, 101, 102, 103, 104}
    labels_rows = []
    for pid, mins in minutes.items():
        labels_rows.append(
            {
                "game_id": "0000000001",
                "team_id": 10,
                "player_id": pid,
                "minutes_actual": mins,
                "played_ge_1": bool(mins >= 1.0),
                "played_ge_5": bool(mins >= 5.0),
                "starter_actual": bool(pid in starters),
                "regime_label": "tight",
            }
        )
    labels = pd.DataFrame(labels_rows)

    events.to_parquet(rot_bundle_dir / "rotation_events.parquet", index=False)
    labels.to_parquet(rot_bundle_dir / "rotation_labels.parquet", index=False)


def _write_synth_rot_bundle_two_games(rot_bundle_dir: Path) -> None:
    rot_bundle_dir.mkdir(parents=True, exist_ok=True)

    def _events_for_game(game_id: str) -> list[dict]:
        return [
            {
                "season_id": "0000-01",
                "game_id": game_id,
                "team_id": 10,
                "opponent_team_id": 20,
                "is_home": True,
                "segment_idx": 0,
                "duration_sec": 720,
                "lineup_p1": 100,
                "lineup_p2": 101,
                "lineup_p3": 102,
                "lineup_p4": 103,
                "lineup_p5": 104,
            },
            {
                "season_id": "0000-01",
                "game_id": game_id,
                "team_id": 10,
                "opponent_team_id": 20,
                "is_home": True,
                "segment_idx": 1,
                "duration_sec": 720,
                "lineup_p1": 100,
                "lineup_p2": 101,
                "lineup_p3": 102,
                "lineup_p4": 103,
                "lineup_p5": 105,
            },
            {
                "season_id": "0000-01",
                "game_id": game_id,
                "team_id": 10,
                "opponent_team_id": 20,
                "is_home": True,
                "segment_idx": 2,
                "duration_sec": 720,
                "lineup_p1": 100,
                "lineup_p2": 101,
                "lineup_p3": 102,
                "lineup_p4": 106,
                "lineup_p5": 104,
            },
            {
                "season_id": "0000-01",
                "game_id": game_id,
                "team_id": 10,
                "opponent_team_id": 20,
                "is_home": True,
                "segment_idx": 3,
                "duration_sec": 720,
                "lineup_p1": 100,
                "lineup_p2": 101,
                "lineup_p3": 102,
                "lineup_p4": 103,
                "lineup_p5": 104,
            },
        ]

    events = pd.DataFrame(_events_for_game("0000000001") + _events_for_game("0000000002"))

    minutes = {
        100: 48.0,
        101: 48.0,
        102: 48.0,
        103: 36.0,
        104: 36.0,
        105: 12.0,
        106: 12.0,
        107: 0.5,
    }
    starters = {100, 101, 102, 103, 104}
    labels_rows = []
    for game_id in ["0000000001", "0000000002"]:
        for pid, mins in minutes.items():
            labels_rows.append(
                {
                    "game_id": game_id,
                    "team_id": 10,
                    "player_id": pid,
                    "minutes_actual": mins,
                    "played_ge_1": bool(mins >= 1.0),
                    "played_ge_5": bool(mins >= 5.0),
                    "starter_actual": bool(pid in starters),
                    "regime_label": "tight",
                }
            )
    labels = pd.DataFrame(labels_rows)

    events.to_parquet(rot_bundle_dir / "rotation_events.parquet", index=False)
    labels.to_parquet(rot_bundle_dir / "rotation_labels.parquet", index=False)


def _write_synth_predictor_bundle(bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": "synth",
        "created_at": "1970-01-01T00:00:00Z",
        "feature_columns": [],
    }
    (bundle_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    preds = pd.DataFrame(
        {
            "game_id": ["0000000001"] * 8,
            "team_id": [10] * 8,
            # Use internal ids <=2000 so eval can skip personId mapping entirely.
            "player_id": [100, 101, 102, 103, 104, 105, 106, 107],
            "p_ge5": [0.99, 0.99, 0.99, 0.95, 0.95, 0.80, 0.80, 0.01],
            "p_ge15": [0.99, 0.99, 0.99, 0.95, 0.95, 0.10, 0.10, 0.00],
        }
    )
    preds.to_parquet(bundle_dir / "predictions_test.parquet", index=False)


def test_rot_eval_gate_integration_smoke(tmp_path: Path) -> None:
    rot_bundle_dir = tmp_path / "rot_v1_bundle"
    _write_synth_rot_bundle(rot_bundle_dir)

    pred_bundle_dir = tmp_path / "rotation_predictor_v1_bundle"
    _write_synth_predictor_bundle(pred_bundle_dir)

    out_root = tmp_path / "artifacts" / "rot_eval_v1"
    out_dir = out_root / "run_gate_smoke"

    gate_cfg = replace(GateConfig(), enabled=True, protect_top_n=True, top_n_lock=5)
    result = run_rotation_generator_eval(
        rot_bundle_path=rot_bundle_dir,
        run_id="run_gate_smoke",
        n_worlds=40,
        seed=123,
        limit_team_games=1,
        sample_mode="first",
        out_dir=out_dir,
        overwrite=True,
        use_truth_minutes_prior=True,
        gate_config=gate_cfg,
        rotation_predictor_bundle=pred_bundle_dir,
        gate_feature_source="cached_preds",
    )

    player_eval = pd.read_parquet(Path(result["player_eval_path"]))
    for c in [
        "p_ge5_pred",
        "p_ge15_pred",
        "p_ge5_used",
        "p_ge15_used",
        "gate_tier",
        "gate_reason",
        "gate_missing_pred",
        "gate_excluded",
        "gate_minutes_cap",
        "gate_play_prob_cap",
        "minutes_prior_adj",
        "play_prob_adj",
    ]:
        assert c in player_eval.columns

    # Gate is non-structural: never excludes players.
    row_107 = player_eval[player_eval["player_id"] == 107].iloc[0]
    assert row_107["gate_tier"] in {"fringe", "unknown"}
    assert bool(row_107["gate_excluded"]) is False

    assert (out_dir / "gate_summary.json").exists()
    assert (out_dir / "candidate_pool_summary.json").exists()
    report_text = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "## Candidate pool realism + gate impact" in report_text


def test_rot_eval_candidate_pool_prior_topn_smoke(tmp_path: Path) -> None:
    rot_bundle_dir = tmp_path / "rot_v1_bundle"
    _write_synth_rot_bundle(rot_bundle_dir)

    # Prior parquet must use internal ids <=2000 for rot_eval.
    priors = pd.DataFrame(
        {
            "game_id": ["0000000001"] * 8,
            "team_id": [10] * 8,
            "player_id": [100, 101, 102, 103, 104, 105, 106, 107],
            "minutes_prior": [48.0, 48.0, 48.0, 36.0, 36.0, 12.0, 12.0, 0.0],
            "minutes_p10": [0.0] * 8,
            "minutes_p90": [48.0] * 8,
            "play_prob": [1.0] * 8,
        }
    )
    priors_path = tmp_path / "priors.parquet"
    priors.to_parquet(priors_path, index=False)

    out_root = tmp_path / "artifacts" / "rot_eval_v1"
    out_dir = out_root / "run_prior_topn"
    result = run_rotation_generator_eval(
        rot_bundle_path=rot_bundle_dir,
        run_id="run_prior_topn",
        n_worlds=20,
        seed=123,
        limit_team_games=1,
        sample_mode="first",
        out_dir=out_dir,
        overwrite=True,
        use_truth_minutes_prior=False,
        minutes_prior_parquet=priors_path,
        restrict_to_prior_games=True,
        candidate_pool="prior_topn",
        candidate_top_n=5,
        candidate_min_minutes_prior=0.0,
        candidate_min_play_prob=0.8,
        candidate_min_candidates=8,
        gate_config=replace(GateConfig(), enabled=False),
        gate_feature_source="none",
    )

    assert (out_dir / "candidate_pool_summary.json").exists()
    assert (out_dir / "candidate_pool_team_games.parquet").exists()
    assert (out_dir / "manifest.json").exists()
    report_text = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "## Candidate pool realism + gate impact" in report_text

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("candidate_pool") == "prior_topn"


def test_rot_eval_gate_missing_preds_do_not_crater_mapping(tmp_path: Path) -> None:
    rot_bundle_dir = tmp_path / "rot_v1_bundle"
    _write_synth_rot_bundle_two_games(rot_bundle_dir)

    pred_bundle_dir = tmp_path / "rotation_predictor_v1_bundle"
    _write_synth_predictor_bundle(pred_bundle_dir)

    # Overwrite predictions to only cover a subset of (game_id, player_id) so missing preds exist.
    preds = pd.DataFrame(
        {
            "game_id": ["0000000001"] * 7,
            "team_id": [10] * 7,
            "player_id": [100, 101, 102, 103, 104, 105, 106],  # 107 missing; game 0000000002 missing entirely
            "p_ge5": [0.99, 0.99, 0.99, 0.95, 0.95, 0.80, 0.80],
            "p_ge15": [0.99, 0.99, 0.99, 0.95, 0.95, 0.10, 0.10],
        }
    )
    preds.to_parquet(pred_bundle_dir / "predictions_test.parquet", index=False)

    out_root = tmp_path / "artifacts" / "rot_eval_v1"

    base_dir = out_root / "run_base"
    base = run_rotation_generator_eval(
        rot_bundle_path=rot_bundle_dir,
        run_id="run_base",
        n_worlds=50,
        seed=123,
        limit_team_games=2,
        sample_mode="first",
        out_dir=base_dir,
        overwrite=True,
        use_truth_minutes_prior=True,
        gate_config=replace(GateConfig(), enabled=False),
        gate_feature_source="none",
    )
    base_team_eval = pd.read_parquet(Path(base["team_eval_path"]))
    base_mapping = float(base_team_eval["mapping_success_rate"].mean())
    base_fallback = float(base_team_eval["template_fallback_rate"].mean())

    gated_dir = out_root / "run_gate_missing"
    gate_cfg = replace(GateConfig(), enabled=True, protect_top_n=True, top_n_lock=5)
    gated = run_rotation_generator_eval(
        rot_bundle_path=rot_bundle_dir,
        run_id="run_gate_missing",
        n_worlds=50,
        seed=123,
        limit_team_games=2,
        sample_mode="first",
        out_dir=gated_dir,
        overwrite=True,
        use_truth_minutes_prior=True,
        gate_config=gate_cfg,
        rotation_predictor_bundle=pred_bundle_dir,
        gate_feature_source="cached_preds",
    )
    team_eval = pd.read_parquet(Path(gated["team_eval_path"]))
    mapping = float(team_eval["mapping_success_rate"].mean())
    fallback = float(team_eval["template_fallback_rate"].mean())

    # Gate is non-structural: mapping/template selection artifacts must be identical with gate on/off.
    assert mapping == base_mapping
    assert fallback == base_fallback

    player_eval = pd.read_parquet(Path(gated["player_eval_path"]))
    missing = player_eval[player_eval["gate_missing_pred"].fillna(False).astype(bool)]
    assert len(missing) > 0
    assert bool(missing["gate_excluded"].fillna(False).any()) is False
