from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import torch

from projections.cli import score_ownership_live
from projections.ownership_v2 import OwnershipSlateTransformer, OwnershipSlateTransformerConfig


def test_load_fpts_predictions_prefers_gtv2_worlds_over_unified(tmp_path: Path) -> None:
    game_date = date(2026, 1, 2)
    run_id = "20260102T190000Z"

    gtv2_run = (
        tmp_path
        / "artifacts"
        / "gtv2_worlds"
        / f"game_date={game_date.isoformat()}"
        / f"run={run_id}"
    )
    gtv2_run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "player_id": [1, 2],
            "dk_fpts_mean": [41.0, 33.5],
            "minutes_sim_mean": [34.0, 31.0],
        }
    ).to_parquet(gtv2_run / "projections.parquet", index=False)

    unified_run = (
        tmp_path
        / "artifacts"
        / "projections"
        / game_date.isoformat()
        / f"run={run_id}"
    )
    unified_run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "player_id": [1, 2],
            "dk_fpts_mean": [99.0, 98.0],
            "minutes_sim_mean": [10.0, 10.0],
        }
    ).to_parquet(unified_run / "projections.parquet", index=False)

    out = score_ownership_live._load_fpts_predictions(
        game_date=game_date,
        run_id=run_id,
        data_root=tmp_path,
    )

    assert out is not None
    by_player = out.set_index("player_id")
    assert by_player.loc[1, "pred_fpts"] == pytest.approx(41.0)
    assert by_player.loc[2, "pred_fpts"] == pytest.approx(33.5)


def test_load_fpts_predictions_falls_back_when_primary_source_invalid(tmp_path: Path) -> None:
    game_date = date(2026, 1, 2)
    run_id = "20260102T190000Z"

    gtv2_run = (
        tmp_path
        / "artifacts"
        / "gtv2_worlds"
        / f"game_date={game_date.isoformat()}"
        / f"run={run_id}"
    )
    gtv2_run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "player_id": [1, 2],
            "dk_fpts_mean": [1.5544067944846e39, 1.5544067944846e39],
        }
    ).to_parquet(gtv2_run / "projections.parquet", index=False)

    unified_run = (
        tmp_path
        / "artifacts"
        / "projections"
        / game_date.isoformat()
        / f"run={run_id}"
    )
    unified_run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "player_id": [1, 2],
            "dk_fpts_mean": [44.0, 37.0],
            "minutes_sim_mean": [35.0, 33.0],
        }
    ).to_parquet(unified_run / "projections.parquet", index=False)

    out = score_ownership_live._load_fpts_predictions(
        game_date=game_date,
        run_id=run_id,
        data_root=tmp_path,
    )

    assert out is not None
    by_player = out.set_index("player_id")
    assert by_player.loc[1, "pred_fpts"] == pytest.approx(44.0)
    assert by_player.loc[2, "pred_fpts"] == pytest.approx(37.0)


def test_load_fpts_predictions_does_not_fallback_to_sim_v2(tmp_path: Path) -> None:
    game_date = date(2026, 1, 2)
    run_id = "20260102T190000Z"

    sim_run = (
        tmp_path
        / "artifacts"
        / "sim_v2"
        / "worlds_fpts_v2"
        / f"game_date={game_date.isoformat()}"
        / f"run={run_id}"
    )
    sim_run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "player_id": [1, 2],
            "dk_fpts_mean": [48.0, 36.0],
        }
    ).to_parquet(sim_run / "projections.parquet", index=False)

    out = score_ownership_live._load_fpts_predictions(
        game_date=game_date,
        run_id=run_id,
        data_root=tmp_path,
    )

    assert out is None


def _write_v2_artifact(
    *,
    data_root: Path,
    run_id: str,
    feature_columns: list[str],
    target_sum_pct: float = 800.0,
) -> None:
    run_dir = data_root / "artifacts" / "ownership_v2" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = OwnershipSlateTransformerConfig(
        feature_columns=list(feature_columns),
        feature_mean=[0.0] * len(feature_columns),
        feature_std=[1.0] * len(feature_columns),
        d_model=8,
        num_heads=2,
        num_layers=1,
        hidden_dim=16,
        dropout=0.0,
        target_sum_pct=float(target_sum_pct),
        max_players=64,
    )
    model = OwnershipSlateTransformer(config)
    for param in model.parameters():
        torch.nn.init.zeros_(param)
    torch.save(model.state_dict(), run_dir / "model.pt")
    config.save(run_dir / "config.json")


def test_score_ownership_live_v2_scores_with_missing_gtv2_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    game_date = date(2026, 1, 2)
    run_id = "20260102T190000Z"
    model_run = "ownership_xfmr_test_v2"
    feature_columns = [
        "proj_fpts",
        "salary",
        "value_per_k",
        "salary_rank",
        "proj_fpts_rank",
        "proj_fpts_zscore",
        "gtv2_minutes_deterministic",
        "gtv2_state_000",
    ]
    _write_v2_artifact(
        data_root=tmp_path,
        run_id=model_run,
        feature_columns=feature_columns,
        target_sum_pct=800.0,
    )

    slate = pd.DataFrame(
        [
            {"player_id": 1, "player_name": "Alpha Guard", "salary": 9200, "pos": "PG", "team": "AAA"},
            {"player_id": 2, "player_name": "Bravo Wing", "salary": 7100, "pos": "SF", "team": "BBB"},
            {"player_id": 3, "player_name": "Charlie Big", "salary": 4200, "pos": "C", "team": "CCC"},
            {"player_id": 4, "player_name": "Delta Value", "salary": 3500, "pos": "SG", "team": "DDD"},
        ]
    )

    monkeypatch.setattr(score_ownership_live, "_load_fpts_predictions", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        score_ownership_live,
        "_load_calibration_config",
        lambda: {"playable_filter": {"enabled": False}, "normalization": {"enabled": False}},
    )

    out = score_ownership_live.score_ownership(
        slate_df=slate,
        draft_group_id="123",
        game_date=game_date,
        run_id=run_id,
        data_root=tmp_path,
        model_run=model_run,
        model_family="ownership_v2",
        injuries_cutoff_ts=None,
    )

    assert out is not None
    expected = {
        "player_id",
        "player_name",
        "salary",
        "pos",
        "team",
        "proj_fpts",
        "pred_own_pct",
        "pred_own_pct_raw",
        "game_date",
        "run_id",
        "model_run",
        "model_family",
    }
    assert expected.issubset(set(out.columns))
    assert set(out["model_family"].astype(str)) == {"ownership_v2"}
    assert float(pd.to_numeric(out["pred_own_pct"], errors="coerce").sum()) == pytest.approx(800.0, abs=1e-6)
    assert float(pd.to_numeric(out["pred_own_pct_raw"], errors="coerce").sum()) == pytest.approx(100.0, abs=1e-6)


def test_score_ownership_live_v2_uses_live_features_mapping_when_minutes_daily_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    game_date = date(2026, 3, 13)
    run_id = "20260313T191758Z"
    model_run = "ownership_xfmr_test_v2_mapping"
    feature_columns = [
        "proj_fpts",
        "salary",
        "value_per_k",
        "salary_rank",
        "proj_fpts_rank",
        "proj_fpts_zscore",
    ]
    _write_v2_artifact(
        data_root=tmp_path,
        run_id=model_run,
        feature_columns=feature_columns,
        target_sum_pct=800.0,
    )

    live_features_dir = (
        tmp_path
        / "live"
        / "features_minutes_v1"
        / game_date.isoformat()
        / f"run={run_id}"
    )
    live_features_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "player_id": 101,
                "player_name": "Mapped Guard",
                "team_id": 1,
                "team_tricode": "AAA",
                "status": "Q",
            },
            {
                "player_id": 202,
                "player_name": "Mapped Center",
                "team_id": 2,
                "team_tricode": "BBB",
                "status": "P",
            },
        ]
    ).to_parquet(live_features_dir / "features.parquet", index=False)

    slate = pd.DataFrame(
        [
            {"player_id": 11, "player_name": "Mapped Guard", "salary": 9200, "pos": "PG", "team": "AAA"},
            {"player_id": 22, "player_name": "Mapped Center", "salary": 6800, "pos": "C", "team": "BBB"},
        ]
    )
    fpts = pd.DataFrame(
        [
            {"player_id": 101, "pred_fpts": 44.0},
            {"player_id": 202, "pred_fpts": 36.5},
        ]
    )
    monkeypatch.setattr(score_ownership_live, "_load_fpts_predictions", lambda *args, **kwargs: fpts)
    monkeypatch.setattr(
        score_ownership_live,
        "_load_calibration_config",
        lambda: {"playable_filter": {"enabled": False}, "normalization": {"enabled": False}},
    )

    out = score_ownership_live.score_ownership(
        slate_df=slate,
        draft_group_id="123456",
        game_date=game_date,
        run_id=run_id,
        data_root=tmp_path,
        model_run=model_run,
        model_family="ownership_v2",
        injuries_cutoff_ts=None,
    )

    assert out is not None
    by_name = out.set_index("player_name")
    assert by_name.loc["Mapped Guard", "proj_fpts"] == pytest.approx(44.0)
    assert by_name.loc["Mapped Center", "proj_fpts"] == pytest.approx(36.5)
    assert by_name.loc["Mapped Guard", "proj_fpts"] != pytest.approx(9200 / 200.0)
    assert by_name.loc["Mapped Center", "proj_fpts"] != pytest.approx(6800 / 200.0)


def test_write_ownership_health_summary_reports_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    game_date = date(2026, 1, 2)
    run_id = "20260102T190000Z"
    model_run = "ownership_xfmr_test_v2"
    results = {
        "123": pd.DataFrame(
            {
                "player_id": [1, 2],
                "player_name": ["Alpha Guard", "Bravo Wing"],
                "pred_own_pct": [400.0, 400.0],
                "pred_own_pct_raw": [50.0, 50.0],
                "is_locked": [True, True],
            }
        )
    }
    monkeypatch.setattr(
        score_ownership_live,
        "_load_calibration_config",
        lambda: {"normalization": {"target_sum_pct": 800.0}},
    )

    out_path = score_ownership_live._write_ownership_health_summary(
        results=results,
        game_date=game_date,
        run_id=run_id,
        data_root=tmp_path,
        model_family="ownership_v2",
        model_run=model_run,
        gtv2_features_path=None,
        write_lock_cache=True,
        ignore_lock_cache=False,
        out_dir=tmp_path,
    )

    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["warning_count"] >= 1
    assert "123" in payload["slates"]
