from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from prefect_flows import live_nba_pipeline_v3


def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_resolve_gtv2_inference_backend_auto() -> None:
    assert (
        live_nba_pipeline_v3._resolve_gtv2_inference_backend(  # noqa: SLF001
            requested="auto",
            config_payload={"enabled": True, "backend": "triton"},
        )
        == "triton"
    )
    assert (
        live_nba_pipeline_v3._resolve_gtv2_inference_backend(  # noqa: SLF001
            requested="auto",
            config_payload={"enabled": False, "backend": "triton"},
        )
        == "local"
    )


def test_score_gtv2_live_task_triton_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game_date = "2026-03-10"
    run_id = "testrun"
    features_path = tmp_path / "features.parquet"
    _write(
        features_path,
        pd.DataFrame(
            {
                "game_date": [game_date],
                "game_id": [22500999],
                "team_id": [1610612747],
                "player_id": [1234],
            }
        ),
    )

    monkeypatch.setattr(
        live_nba_pipeline_v3,
        "check_triton_health",
        lambda endpoint, timeout_seconds: (True, "ok"),
    )

    captured_payload: dict[str, object] = {}

    def _fake_infer(*, cfg, request_payload):
        captured_payload.update(request_payload)
        out_path = Path(str(request_payload["out_path"]))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "game_date": [game_date],
                "game_id": [22500999],
                "team_id": [1610612747],
                "player_id": [1234],
                "minutes_deterministic": [31.2],
                "active_deterministic": [1],
                "active_logit": [2.3],
                "active_prob_proxy": [0.91],
            }
        ).to_parquet(out_path, index=False)
        return {"ok": True, "device": "cuda:0"}

    monkeypatch.setattr(live_nba_pipeline_v3, "infer_json_action", _fake_infer)

    out_path = live_nba_pipeline_v3.score_gtv2_live_task.fn(
        game_date=game_date,
        run_id=run_id,
        features_path=features_path,
        bundle_dir=tmp_path / "bundle",
        data_root=tmp_path,
        placeholder_mode=False,
        inference_backend="triton",
        triton_endpoint="localhost:8000",
        triton_model_name="gtv2_scorer",
        triton_model_version="1",
        triton_timeout_seconds=30.0,
        triton_healthcheck_timeout_seconds=1.0,
        gtv2_device="cuda:0",
        random_seed=42,
    )
    assert out_path.exists()
    summary_path = out_path.parent / "score_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["inference_backend"] == "triton"
    assert summary["device"] == "cuda:0"
    assert "summary_path" not in captured_payload


def test_score_gtv2_live_task_local_backend_full_slate_runs_per_game_sequentially(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game_date = "2026-03-10"
    run_id = "testrun"
    features_path = tmp_path / "features.parquet"
    _write(
        features_path,
        pd.DataFrame(
            {
                "game_date": [game_date, game_date],
                "game_id": [22500991, 22500992],
                "team_id": [1610612747, 1610612748],
                "player_id": [1234, 2234],
            }
        ),
    )

    monkeypatch.setattr(live_nba_pipeline_v3, "_set_inference_seed", lambda seed: None)
    monkeypatch.setattr(
        live_nba_pipeline_v3,
        "_resolve_torch_device",
        lambda device: "cpu",
    )
    monkeypatch.setattr(
        live_nba_pipeline_v3,
        "_load_gtv2_model",
        lambda bundle_dir, device: ({}, object()),
    )

    captured_game_ids: list[int] = []

    def _fake_local_score(*, features_df, game_date, config, model, device, batch_size):
        captured_game_ids.append(int(pd.to_numeric(features_df["game_id"], errors="coerce").iloc[0]))
        out = features_df[["game_date", "game_id", "team_id", "player_id"]].copy()
        out["minutes_deterministic"] = 30.0
        out["active_deterministic"] = 1
        out["active_logit"] = 2.0
        out["active_prob_proxy"] = 0.88
        return out.reset_index(drop=True)

    monkeypatch.setattr(
        live_nba_pipeline_v3,
        "shared_score_gtv2_features_df",
        _fake_local_score,
    )

    out_path = live_nba_pipeline_v3.score_gtv2_live_task.fn(
        game_date=game_date,
        run_id=run_id,
        features_path=features_path,
        bundle_dir=tmp_path / "bundle",
        data_root=tmp_path,
        placeholder_mode=False,
        inference_backend="local",
        random_seed=42,
    )

    summary = json.loads((out_path.parent / "score_summary.json").read_text(encoding="utf-8"))
    assert summary["inference_backend"] == "local"
    assert summary["triton_request_count"] == 0
    assert captured_game_ids == [22500991, 22500992]

    scored = pd.read_parquet(out_path)
    assert scored["game_id"].astype(int).tolist() == [22500991, 22500992]


def test_generate_worlds_gtv2_live_task_triton_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game_date = "2026-03-10"
    run_id = "testrun"
    features_path = tmp_path / "features.parquet"
    _write(
        features_path,
        pd.DataFrame(
            {
                "game_date": [game_date],
                "game_id": [22500999],
                "team_id": [1610612747],
                "player_id": [1234],
                "lineup_starter_announced": [1],
            }
        ),
    )
    scores_path = tmp_path / "scores.parquet"
    _write(
        scores_path,
        pd.DataFrame(
            {
                "game_date": [game_date],
                "game_id": [22500999],
                "team_id": [1610612747],
                "player_id": [1234],
                "minutes_deterministic": [31.2],
            }
        ),
    )

    monkeypatch.setattr(
        live_nba_pipeline_v3,
        "check_triton_health",
        lambda endpoint, timeout_seconds: (True, "ok"),
    )
    monkeypatch.setattr(
        live_nba_pipeline_v3,
        "get_run_logger",
        lambda: logging.getLogger("test-gtv2-triton-worlds"),
    )

    def _fake_infer(*, cfg, request_payload):
        out_path = Path(str(request_payload["out_path"]))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "world_idx": [0],
                "game_date": [game_date],
                "game_id": [22500999],
                "team_id": [1610612747],
                "player_id": [1234],
                "minutes": [30.0],
                "fga2": [5.0],
                "fg2m": [3.0],
                "fga3": [4.0],
                "fg3m": [1.0],
                "fta": [2.0],
                "ftm": [2.0],
                "oreb": [1.0],
                "dreb": [3.0],
                "ast": [4.0],
                "stl": [1.0],
                "blk": [0.0],
                "tov": [2.0],
                "pts": [11.0],
                "reb": [4.0],
                "dk_fpts": [24.0],
                "active": [1],
            }
        ).to_parquet(out_path, index=False)
        return {
            "ok": True,
            "device": "cuda:0",
            "contract_checks": {"team_minutes_not_240": 0},
        }

    monkeypatch.setattr(live_nba_pipeline_v3, "infer_json_action", _fake_infer)
    monkeypatch.setattr(
        live_nba_pipeline_v3,
        "summarize_worlds_to_projections",
        lambda worlds_df, sim_profile: pd.DataFrame(
            {
                "game_date": [game_date],
                "game_id": [22500999],
                "team_id": [1610612747],
                "player_id": [1234],
            }
        ),
    )

    outputs = live_nba_pipeline_v3.generate_worlds_gtv2_live_task.fn(
        game_date=game_date,
        run_id=run_id,
        run_as_of_ts="2026-03-10T18:00:00Z",
        features_path=features_path,
        scores_path=scores_path,
        bundle_dir=tmp_path / "bundle",
        data_root=tmp_path,
        sim_worlds=32,
        placeholder_mode=False,
        inference_backend="triton",
        triton_endpoint="localhost:8000",
        triton_model_name="gtv2_scorer",
        triton_model_version="1",
        triton_timeout_seconds=30.0,
        triton_healthcheck_timeout_seconds=1.0,
        gtv2_device="cuda:0",
        world_chunk_size=8,
        active_temperature=1.0,
        random_seed=42,
        strict_world_contracts=True,
        apply_props_uplift=False,
        apply_world_realism_controls=False,
    )

    summary_path = Path(outputs["world_contract_summary_path"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["inference_backend"] == "triton"
    assert summary["device"] == "cuda:0"


def test_score_gtv2_live_task_triton_backend_full_slate_runs_per_game_sequentially(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game_date = "2026-03-10"
    run_id = "testrun"
    features_path = tmp_path / "features.parquet"
    _write(
        features_path,
        pd.DataFrame(
            {
                "game_date": [game_date, game_date],
                "game_id": [22500991, 22500992],
                "team_id": [1610612747, 1610612748],
                "player_id": [1234, 2234],
            }
        ),
    )

    monkeypatch.setattr(
        live_nba_pipeline_v3,
        "check_triton_health",
        lambda endpoint, timeout_seconds: (True, "ok"),
    )

    captured_payloads: list[dict[str, object]] = []

    def _fake_infer(*, cfg, request_payload):
        captured_payloads.append(dict(request_payload))
        game_features = pd.read_parquet(Path(str(request_payload["features_path"])))
        out_path = Path(str(request_payload["out_path"]))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scored = game_features[
            ["game_date", "game_id", "team_id", "player_id"]
        ].copy()
        scored["minutes_deterministic"] = 30.5
        scored["active_deterministic"] = 1
        scored["active_logit"] = 2.0
        scored["active_prob_proxy"] = 0.88
        scored.to_parquet(out_path, index=False)
        return {"ok": True, "device": "cuda:0"}

    monkeypatch.setattr(live_nba_pipeline_v3, "infer_json_action", _fake_infer)

    out_path = live_nba_pipeline_v3.score_gtv2_live_task.fn(
        game_date=game_date,
        run_id=run_id,
        features_path=features_path,
        bundle_dir=tmp_path / "bundle",
        data_root=tmp_path,
        placeholder_mode=False,
        inference_backend="triton",
        triton_endpoint="localhost:8000",
        triton_model_name="gtv2_scorer",
        triton_model_version="1",
        triton_timeout_seconds=30.0,
        triton_healthcheck_timeout_seconds=1.0,
        gtv2_device="cuda:0",
        random_seed=42,
    )

    summary = json.loads((out_path.parent / "score_summary.json").read_text(encoding="utf-8"))
    assert summary["triton_request_count"] == 2
    assert len(captured_payloads) == 2
    seeds = {int(payload["random_seed"]) for payload in captured_payloads}
    assert len(seeds) == 2

    per_request_game_ids = [
        int(
            pd.read_parquet(Path(str(payload["features_path"])))
            .iloc[0]["game_id"]
        )
        for payload in captured_payloads
    ]
    assert per_request_game_ids == [22500991, 22500992]

    scored = pd.read_parquet(out_path)
    assert scored["game_id"].astype(int).tolist() == [22500991, 22500992]


def test_generate_worlds_gtv2_live_task_triton_backend_full_slate_runs_per_game_sequentially(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game_date = "2026-03-10"
    run_id = "testrun"
    features_path = tmp_path / "features.parquet"
    _write(
        features_path,
        pd.DataFrame(
            {
                "game_date": [game_date, game_date],
                "game_id": [22500991, 22500992],
                "team_id": [1610612747, 1610612748],
                "player_id": [1234, 2234],
                "lineup_starter_announced": [1, 1],
            }
        ),
    )
    scores_path = tmp_path / "scores.parquet"
    _write(
        scores_path,
        pd.DataFrame(
            {
                "game_date": [game_date, game_date],
                "game_id": [22500991, 22500992],
                "team_id": [1610612747, 1610612748],
                "player_id": [1234, 2234],
                "minutes_deterministic": [31.2, 29.1],
            }
        ),
    )

    monkeypatch.setattr(
        live_nba_pipeline_v3,
        "check_triton_health",
        lambda endpoint, timeout_seconds: (True, "ok"),
    )
    monkeypatch.setattr(
        live_nba_pipeline_v3,
        "get_run_logger",
        lambda: logging.getLogger("test-gtv2-triton-worlds-seq"),
    )

    captured_payloads: list[dict[str, object]] = []

    def _fake_infer(*, cfg, request_payload):
        captured_payloads.append(dict(request_payload))
        game_features = pd.read_parquet(Path(str(request_payload["features_path"])))
        out_path = Path(str(request_payload["out_path"]))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        worlds = game_features[["game_date", "game_id", "team_id", "player_id"]].copy()
        worlds.insert(0, "world_idx", 0)
        worlds["minutes"] = 30.0
        worlds["fga2"] = 5.0
        worlds["fg2m"] = 3.0
        worlds["fga3"] = 4.0
        worlds["fg3m"] = 1.0
        worlds["fta"] = 2.0
        worlds["ftm"] = 2.0
        worlds["oreb"] = 1.0
        worlds["dreb"] = 3.0
        worlds["ast"] = 4.0
        worlds["stl"] = 1.0
        worlds["blk"] = 0.0
        worlds["tov"] = 2.0
        worlds["pts"] = 11.0
        worlds["reb"] = 4.0
        worlds["dk_fpts"] = 24.0
        worlds["active"] = 1
        worlds.to_parquet(out_path, index=False)
        return {
            "ok": True,
            "device": "cuda:0",
            "contract_checks": {"team_minutes_not_240": 0},
        }

    monkeypatch.setattr(live_nba_pipeline_v3, "infer_json_action", _fake_infer)
    monkeypatch.setattr(
        live_nba_pipeline_v3,
        "summarize_worlds_to_projections",
        lambda worlds_df, sim_profile: (
            worlds_df[["game_date", "game_id", "team_id", "player_id"]]
            .drop_duplicates()
            .reset_index(drop=True)
        ),
    )

    outputs = live_nba_pipeline_v3.generate_worlds_gtv2_live_task.fn(
        game_date=game_date,
        run_id=run_id,
        run_as_of_ts="2026-03-10T18:00:00Z",
        features_path=features_path,
        scores_path=scores_path,
        bundle_dir=tmp_path / "bundle",
        data_root=tmp_path,
        sim_worlds=16,
        placeholder_mode=False,
        inference_backend="triton",
        triton_endpoint="localhost:8000",
        triton_model_name="gtv2_scorer",
        triton_model_version="1",
        triton_timeout_seconds=30.0,
        triton_healthcheck_timeout_seconds=1.0,
        gtv2_device="cuda:0",
        world_chunk_size=8,
        active_temperature=1.0,
        random_seed=42,
        strict_world_contracts=True,
        apply_props_uplift=False,
        apply_world_realism_controls=False,
    )

    summary = json.loads(
        Path(outputs["world_contract_summary_path"]).read_text(encoding="utf-8")
    )
    assert summary["triton_request_count"] == 2
    assert summary["triton"]["request_count"] == 2
    assert len(captured_payloads) == 2
    seeds = {int(payload["random_seed"]) for payload in captured_payloads}
    assert len(seeds) == 2

    per_request_game_ids = [
        int(
            pd.read_parquet(Path(str(payload["features_path"])))
            .iloc[0]["game_id"]
        )
        for payload in captured_payloads
    ]
    assert per_request_game_ids == [22500991, 22500992]

    worlds = pd.read_parquet(Path(outputs["worlds_path"]))
    assert sorted(worlds["game_id"].astype(int).unique().tolist()) == [22500991, 22500992]
