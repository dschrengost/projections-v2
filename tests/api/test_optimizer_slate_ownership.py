from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.api import optimizer_service


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_build_player_pool_uses_run_scoped_slate_ownership(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "projections-data"
    monkeypatch.setenv("PROJECTIONS_DATA_ROOT", str(data_root))

    day = "2026-02-28"
    run_id = "20260228T180002Z"
    draft_group_id = 222222

    projections_dir = data_root / "artifacts" / "projections" / day / f"run={run_id}"
    projections_df = pd.DataFrame(
        [
            {
                "game_date": day,
                "player_id": 2,
                "player_name": "Beta Wing",
                "team_tricode": "CCC",
                "proj_fpts": 36.0,
                "pred_own_pct": 18.0,
            }
        ]
    )
    _write_parquet(projections_dir / "projections.parquet", projections_df)

    ownership_dir = data_root / "silver" / "ownership_predictions" / day / f"run={run_id}"
    ownership_df = pd.DataFrame(
        [
            {
                "player_name": "Beta Wing",
                "team": "CCC",
                "salary": 6400,
                "pred_own_pct": 42.5,
                "draft_group_id": str(draft_group_id),
            }
        ]
    )
    _write_parquet(ownership_dir / f"{draft_group_id}.parquet", ownership_df)

    salaries_dir = (
        data_root
        / "gold"
        / "dk_salaries"
        / "site=dk"
        / f"game_date={day}"
        / f"draft_group_id={draft_group_id}"
    )
    salaries_df = pd.DataFrame(
        [
            {
                "dk_player_id": 902,
                "display_name": "Beta Wing",
                "positions": ["SF"],
                "salary": 6400,
                "team_abbrev": "CCC",
                "status": None,
                "is_disabled": False,
            }
        ]
    )
    _write_parquet(salaries_dir / "salaries.parquet", salaries_df)

    pool = optimizer_service.build_player_pool(
        game_date=day,
        draft_group_id=draft_group_id,
        site="dk",
        run_id=run_id,
        use_user_overrides=False,
    )

    assert len(pool) == 1
    assert pool[0]["player_id"] == "2"
    assert pool[0]["own_proj"] == 42.5


def test_build_player_pool_can_match_ownership_on_dk_player_id(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "projections-data"
    monkeypatch.setenv("PROJECTIONS_DATA_ROOT", str(data_root))

    day = "2026-02-28"
    run_id = "20260228T180002Z"
    draft_group_id = 222222

    projections_dir = data_root / "artifacts" / "projections" / day / f"run={run_id}"
    projections_df = pd.DataFrame(
        [
            {
                "game_date": day,
                "player_id": 2,
                "player_name": "Beta Wing",
                "team_tricode": "CCC",
                "proj_fpts": 36.0,
            }
        ]
    )
    _write_parquet(projections_dir / "projections.parquet", projections_df)

    ownership_dir = data_root / "silver" / "ownership_predictions" / day / f"run={run_id}"
    ownership_df = pd.DataFrame(
        [
            {
                "player_id": 902,
                "pred_own_pct": 42.5,
                "draft_group_id": str(draft_group_id),
            }
        ]
    )
    _write_parquet(ownership_dir / f"{draft_group_id}.parquet", ownership_df)

    salaries_dir = (
        data_root
        / "gold"
        / "dk_salaries"
        / "site=dk"
        / f"game_date={day}"
        / f"draft_group_id={draft_group_id}"
    )
    salaries_df = pd.DataFrame(
        [
            {
                "dk_player_id": 902,
                "display_name": "Beta Wing",
                "positions": ["SF"],
                "salary": 6400,
                "team_abbrev": "CCC",
                "status": None,
                "is_disabled": False,
            }
        ]
    )
    _write_parquet(salaries_dir / "salaries.parquet", salaries_df)

    pool = optimizer_service.build_player_pool(
        game_date=day,
        draft_group_id=draft_group_id,
        site="dk",
        run_id=run_id,
        use_user_overrides=False,
    )

    assert len(pool) == 1
    assert pool[0]["own_proj"] == 42.5
