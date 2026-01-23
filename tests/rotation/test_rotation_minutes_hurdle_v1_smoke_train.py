from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.models.rotation_minutes_hurdle_v1.config import RMHConfig
from projections.models.rotation_minutes_hurdle_v1.train import train_rmh


def test_rmh_train_smoke_writes_artifact_bundle(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    n = 200
    features = pd.DataFrame(
        {
            "game_id": list(range(1, n + 1)),
            "team_id": [1610612737] * n,
            "player_id": list(range(1000, 1000 + n)),
            "game_date": pd.to_datetime(["2026-01-01"] * n),
            "tip_ts": pd.to_datetime(["2026-01-01T00:00:00Z"] * n, utc=True),
            "status": ["Ava"] * (n // 2) + ["OUT"] * (n - n // 2),
            "injury_as_of_ts": ([pd.NaT] * (n // 2))
            + list(pd.to_datetime(["2025-12-31T12:00:00Z"] * (n - n // 2), utc=True)),
            # A couple numeric features that are NOT in the excluded list
            "prior_play_prob": [0.95] * (n // 2) + [0.05] * (n - n // 2),
            "is_out": [0] * (n // 2) + [1] * (n - n // 2),
            "some_numeric_feature": list(range(n)),
        }
    )
    labels = pd.DataFrame(
        {
            "game_id": features["game_id"],
            "team_id": features["team_id"],
            "player_id": features["player_id"],
            "game_date": features["game_date"],
            # First half plays, second half DNP (matches OUT status)
            "minutes": [28.0] * (n // 2) + [0.0] * (n - n // 2),
        }
    )

    features.to_parquet(dataset_dir / "features.parquet", index=False)
    labels.to_parquet(dataset_dir / "labels.parquet", index=False)

    cfg = RMHConfig(
        train_dataset_dir=dataset_dir,
        artifact_dir=tmp_path / "artifacts",
        run_id="smoke",
        device="cpu",
        epochs=1,
        batch_size=64,
        hidden_dims=[32],
        dropout=0.0,
        lr=1e-3,
        seed=123,
        half_life_play_days=30.0,
        half_life_minutes_days=30.0,
    )

    run_dir = train_rmh(cfg)
    assert run_dir.exists()

    for required in [
        "config.json",
        "schema.json",
        "preprocessor.json",
        "metrics.json",
        "meta.json",
        "model.pt",
    ]:
        assert (run_dir / required).exists(), f"missing {required}"
