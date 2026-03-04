from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.rotation.live_features_v1 import load_latest_rotation_priors_by_entity


def test_load_latest_rotation_priors_skips_suspicious_large_partitions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    season = 2025
    team_root = (
        tmp_path
        / "silver"
        / "rotation_priors_v1"
        / "team_game_priors"
        / f"season={season}"
    )
    player_root = (
        tmp_path
        / "silver"
        / "rotation_priors_v1"
        / "player_game_priors"
        / f"season={season}"
    )
    team_root.mkdir(parents=True, exist_ok=True)
    player_root.mkdir(parents=True, exist_ok=True)

    # Baseline valid partitions.
    pd.DataFrame(
        {
            "game_date": ["2026-03-01"],
            "team_id": [1610612760],
            "marker": [1],
        }
    ).to_parquet(team_root / "game_id=0022500001.parquet", index=False)
    pd.DataFrame(
        {
            "game_date": ["2026-03-01"],
            "person_id": [1642349],
            "marker": [1],
            "game_id": ["0022500001"],
        }
    ).to_parquet(player_root / "game_id=0022500001.parquet", index=False)

    # Extra partition that will be monkeypatched to an absurdly large row count.
    pd.DataFrame({"dummy": [1]}).to_parquet(team_root / "game_id=0022500002.parquet", index=False)
    pd.DataFrame({"dummy": [1]}).to_parquet(player_root / "game_id=0022500002.parquet", index=False)

    original_read_parquet = pd.read_parquet

    def _fake_read_parquet(path, *args, **kwargs):  # noqa: ANN001
        text = str(path)
        name = Path(path).name
        if name == "game_id=0022500002.parquet" and "team_game_priors" in text:
            return pd.DataFrame(
                {
                    "game_date": ["2026-03-04"] * 6001,
                    "team_id": [1610612760] * 6001,
                    "marker": [999] * 6001,
                }
            )
        if name == "game_id=0022500002.parquet" and "player_game_priors" in text:
            return pd.DataFrame(
                {
                    "game_date": ["2026-03-04"] * 6001,
                    "person_id": [1642349] * 6001,
                    "marker": [999] * 6001,
                    "game_id": ["0022500002"] * 6001,
                }
            )
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _fake_read_parquet)

    team_priors, player_priors = load_latest_rotation_priors_by_entity(
        tmp_path,
        season=season,
        team_ids=[1610612760],
        player_ids=[1642349],
    )

    assert len(team_priors) == 1
    assert len(player_priors) == 1
    # If suspicious partitions are not skipped, marker would be 999 from newer date.
    assert int(team_priors.iloc[0]["marker"]) == 1
    assert int(player_priors.iloc[0]["marker"]) == 1
