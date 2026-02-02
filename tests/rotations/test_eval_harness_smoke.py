from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from projections.rotations.eval import run_rotation_generator_eval


def _write_synth_rot_bundle(rot_bundle_dir: Path) -> None:
    rot_bundle_dir.mkdir(parents=True, exist_ok=True)

    def make_events(*, game_id: str, team_id: int, opponent_team_id: int, base_pid: int) -> list[dict]:
        # Four 12-minute segments -> full regulation game.
        return [
            {
                "season_id": "2024-25",
                "game_id": game_id,
                "team_id": team_id,
                "opponent_team_id": opponent_team_id,
                "is_home": True,
                "segment_idx": 0,
                "duration_sec": 720,
                "lineup_p1": base_pid + 0,
                "lineup_p2": base_pid + 1,
                "lineup_p3": base_pid + 2,
                "lineup_p4": base_pid + 3,
                "lineup_p5": base_pid + 4,
            },
            {
                "season_id": "2024-25",
                "game_id": game_id,
                "team_id": team_id,
                "opponent_team_id": opponent_team_id,
                "is_home": True,
                "segment_idx": 1,
                "duration_sec": 720,
                "lineup_p1": base_pid + 0,
                "lineup_p2": base_pid + 1,
                "lineup_p3": base_pid + 2,
                "lineup_p4": base_pid + 3,
                "lineup_p5": base_pid + 5,  # bench
            },
            {
                "season_id": "2024-25",
                "game_id": game_id,
                "team_id": team_id,
                "opponent_team_id": opponent_team_id,
                "is_home": True,
                "segment_idx": 2,
                "duration_sec": 720,
                "lineup_p1": base_pid + 0,
                "lineup_p2": base_pid + 1,
                "lineup_p3": base_pid + 2,
                "lineup_p4": base_pid + 6,  # bench
                "lineup_p5": base_pid + 4,
            },
            {
                "season_id": "2024-25",
                "game_id": game_id,
                "team_id": team_id,
                "opponent_team_id": opponent_team_id,
                "is_home": True,
                "segment_idx": 3,
                "duration_sec": 720,
                "lineup_p1": base_pid + 0,
                "lineup_p2": base_pid + 1,
                "lineup_p3": base_pid + 2,
                "lineup_p4": base_pid + 3,
                "lineup_p5": base_pid + 4,
            },
        ]

    events = pd.DataFrame(
        make_events(game_id="0000000001", team_id=10, opponent_team_id=20, base_pid=100)
        + make_events(game_id="0000000002", team_id=10, opponent_team_id=30, base_pid=200)
    )

    def make_labels(*, game_id: str, team_id: int, base_pid: int, include_truth_starters: bool) -> list[dict]:
        # Truth minutes implied by segments above:
        # base+0..2: 48; base+3..4: 36; base+5..6: 12.
        rows: list[dict] = []
        minutes = {
            base_pid + 0: 48.0,
            base_pid + 1: 48.0,
            base_pid + 2: 48.0,
            base_pid + 3: 36.0,
            base_pid + 4: 36.0,
            base_pid + 5: 12.0,
            base_pid + 6: 12.0,
            base_pid + 7: 0.5,  # minutes_actual>0 but played_ge_1=False
        }
        starters = {base_pid + i for i in range(5)}
        for pid, mins in minutes.items():
            played_ge_1 = bool(mins >= 1.0)
            played_ge_5 = bool(mins >= 5.0)
            rows.append(
                {
                    "game_id": game_id,
                    "team_id": team_id,
                    "player_id": pid,
                    "minutes_actual": mins,
                    "played_ge_1": played_ge_1,
                    "played_ge_5": played_ge_5,
                    "starter_actual": bool(pid in starters) if include_truth_starters else False,
                    "regime_label": "tight",
                }
            )
        return rows

    labels = pd.DataFrame(
        make_labels(game_id="0000000001", team_id=10, base_pid=100, include_truth_starters=True)
        + make_labels(game_id="0000000002", team_id=10, base_pid=200, include_truth_starters=False)
    )

    events.to_parquet(rot_bundle_dir / "rotation_events.parquet", index=False)
    labels.to_parquet(rot_bundle_dir / "rotation_labels.parquet", index=False)


def test_eval_harness_smoke_and_determinism(tmp_path: Path) -> None:
    rot_bundle_dir = tmp_path / "rot_v1_bundle"
    _write_synth_rot_bundle(rot_bundle_dir)

    out_root = tmp_path / "artifacts" / "rot_eval_v1"
    out_dir1 = out_root / "run1"
    out_dir2 = out_root / "run2"

    r1 = run_rotation_generator_eval(
        rot_bundle_path=rot_bundle_dir,
        run_id="run1",
        n_worlds=50,
        seed=123,
        limit_team_games=1,
        sample_mode="random",
        out_dir=out_dir1,
        overwrite=True,
        use_truth_minutes_prior=True,
    )
    r2 = run_rotation_generator_eval(
        rot_bundle_path=rot_bundle_dir,
        run_id="run2",
        n_worlds=50,
        seed=123,
        limit_team_games=1,
        sample_mode="random",
        out_dir=out_dir2,
        overwrite=True,
        use_truth_minutes_prior=True,
    )

    assert r1["metrics"] == r2["metrics"]

    player_eval_1 = pd.read_parquet(Path(r1["player_eval_path"]))
    player_eval_2 = pd.read_parquet(Path(r2["player_eval_path"]))

    required_cols = {
        "season_id",
        "game_id",
        "team_id",
        "opponent_team_id",
        "is_home",
        "player_id",
        "minutes_actual",
        "played_ge_1",
        "played_ge_5",
        "starter_actual",
        "regime_label",
        "minutes_mean",
        "minutes_p10",
        "minutes_p50",
        "minutes_p90",
        "p_played_ge_1_pred",
        "p_played_ge_5_pred",
        "n_worlds",
        "seed",
        "generator_name",
        "mapping_success",
        "template_source",
    }
    assert required_cols.issubset(set(player_eval_1.columns))
    assert required_cols.issubset(set(player_eval_2.columns))

    # Deterministic output for a fixed seed.
    sort_cols = ["season_id", "game_id", "team_id", "player_id"]
    player_eval_1 = player_eval_1.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    player_eval_2 = player_eval_2.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(player_eval_1, player_eval_2, check_dtype=False)


def test_eval_predictor_threshold_uses_cached_all_by_default_and_has_no_missing_team_games(tmp_path: Path) -> None:
    rot_bundle_dir = tmp_path / "rot_v1_bundle"
    _write_synth_rot_bundle(rot_bundle_dir)

    # Minutes priors in internal-id space (required for predictor_threshold).
    priors_rows: list[dict] = []
    for game_id, base_pid in [("0000000001", 100), ("0000000002", 200)]:
        for offset in range(8):  # 0..7 (matches labels)
            priors_rows.append(
                {
                    "game_id": game_id,
                    "team_id": 10,
                    "player_id": base_pid + offset,
                    "minutes_prior": float(30 - offset),
                    "play_prob": 0.9,
                }
            )
    priors = pd.DataFrame(priors_rows)
    priors_path = tmp_path / "minutes_priors.parquet"
    priors.to_parquet(priors_path, index=False)

    # Predictor bundle with:
    # - predictions_all.parquet matching the rot_eval universe
    # - predictions_test.parquet intentionally NOT matching (regression guard)
    pred_bundle_dir = tmp_path / "rotation_predictor_bundle"
    pred_bundle_dir.mkdir(parents=True, exist_ok=True)
    (pred_bundle_dir / "meta.json").write_text('{"feature_columns": []}\n', encoding="utf-8")

    preds_all_rows: list[dict] = []
    for game_id, base_pid in [("0000000001", 100), ("0000000002", 200)]:
        for offset in range(8):
            preds_all_rows.append(
                {
                    "game_id": game_id,
                    "team_id": 10,
                    "player_id": base_pid + offset,
                    "p_ge5_pred": 0.8,
                    "p_ge15_pred": 0.6,
                }
            )
    pd.DataFrame(preds_all_rows).to_parquet(pred_bundle_dir / "predictions_all.parquet", index=False)

    pd.DataFrame(
        [
            {
                "game_id": 999,
                "team_id": 10,
                "player_id": 100,
                "p_ge5": 0.1,
                "p_ge15": 0.1,
            }
        ]
    ).to_parquet(pred_bundle_dir / "predictions_test.parquet", index=False)

    out_dir = tmp_path / "artifacts" / "rot_eval_v1" / "pred_threshold"
    result = run_rotation_generator_eval(
        rot_bundle_path=rot_bundle_dir,
        run_id="pred_threshold",
        n_worlds=10,
        seed=0,
        limit_team_games=2,
        sample_mode="first",
        out_dir=out_dir,
        overwrite=True,
        use_truth_minutes_prior=False,
        minutes_prior_parquet=priors_path,
        candidate_pool="predictor_threshold",
        rotation_predictor_bundle=pred_bundle_dir,
        pool_max_size=11,
        pool_t_ge15=0.3,
        pool_t_ge5=0.2,
        pool_always_include_top_n=5,
    )

    summary_path = Path(result["candidate_pool_summary_path"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert int(summary["missing_pred_team_games"]) == 0
    assert int(summary["missing_pred_player_rows"]) == 0

    debug = summary["predictor_join_debug"]
    assert debug["gate_feature_source"]["resolved"] == "cached_all"
    assert debug["artifact_kind"] == "predictions_all.parquet"
