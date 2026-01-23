from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd


def test_rmh_shadow_joins_rotation_priors(monkeypatch, tmp_path):
    """Regression test: RMH shadow must join rotation_priors_v1 before predict_frame.

    Without priors, most *_prior_* features are missing and RMH minutes can blow up.
    This test patches the priors loader + RMH inference and asserts the priors
    column is present in the DataFrame passed to predict_frame.
    """

    import prefect_flows.live_nba_pipeline as pipeline

    monkeypatch.setattr(
        pipeline, "get_run_logger", lambda: logging.getLogger("rmh-shadow-test")
    )

    monkeypatch.setenv("RMH_SHADOW_ENABLED", "1")
    artifact_dir = tmp_path / "rmh_artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("RMH_ARTIFACT_DIR", str(artifact_dir))

    features_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "game_id": [1],
            "team_id": [100],
            "player_id": [200],
            "spread_home": [-2.5],
            "total": [220.5],
        }
    ).to_parquet(features_path, index=False)

    class _DummyModel:
        delta_out = 2

    dummy_bundle = SimpleNamespace(
        model=_DummyModel(),
        feature_spec=SimpleNamespace(continuous=["foo_prior_5"], categorical=[]),
        config={},
        schema_hash="test",
    )

    import projections.models.rotation_minutes_hurdle_v1 as rmh

    monkeypatch.setattr(rmh, "load_bundle", lambda _path: dummy_bundle)

    passed_cols: dict[str, set[str]] = {}

    def _fake_predict_frame(
        df: pd.DataFrame, *, bundle, batch_size: int = 8192
    ) -> pd.DataFrame:
        assert "foo_prior_5" in df.columns, (
            "Expected RMH shadow to join priors before predict_frame."
        )
        passed_cols["cols"] = set(df.columns)
        return pd.DataFrame(
            {
                "game_id": [1],
                "player_id": [200],
                "team_id": [100],
                "p_play": [0.5],
                "minutes_mean_uncond": [10.0],
                "minutes_q10_uncond": [5.0],
                "minutes_q50_uncond": [10.0],
                "minutes_q90_uncond": [20.0],
            }
        )

    monkeypatch.setattr(rmh, "predict_frame", _fake_predict_frame)

    import projections.rotation.live_features_v1 as rotation_live

    def _fake_load_priors(*args, **kwargs):  # noqa: ANN002, ANN003
        return SimpleNamespace(
            team_priors=pd.DataFrame(
                {"game_id": [1], "team_id": [100], "foo_prior_5": [123.0]}
            ),
            player_priors=pd.DataFrame(),
            teams_found=1,
            teams_missing=0,
            players_found=1,
            players_missing=0,
            used_latest_fallback=False,
            warning_message=None,
        )

    monkeypatch.setattr(
        rotation_live, "load_rotation_priors_for_live_inference", _fake_load_priors
    )

    out_path = pipeline.rmh_shadow_minutes_task.fn(
        game_date="2026-01-23",
        run_id="run123",
        run_as_of_ts="2026-01-23T00:00:00Z",
        features_path=features_path,
        data_root=tmp_path,
    )

    assert out_path is not None
    assert out_path.exists()
    assert passed_cols.get("cols") is not None
