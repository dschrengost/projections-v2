from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import projections.cli.score_ownership_live as ownership_live


def test_score_ownership_live_lock_cache_written_only_after_lock(tmp_path: Path, monkeypatch) -> None:
    game_date = date(2025, 1, 1)
    dg_id = "999"
    lock_ts = datetime(2025, 1, 1, 23, 0, tzinfo=timezone.utc)

    slate_df = pd.DataFrame(
        [
            {
                "player_id": 1,
                "player_name": "X",
                "salary": 5000,
                "pos": "PG",
                "team": "AAA",
            }
        ]
    )

    monkeypatch.setattr(ownership_live, "_load_all_slates", lambda *_args, **_kwargs: {dg_id: slate_df})
    monkeypatch.setattr(ownership_live, "_load_schedule_with_times", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(ownership_live, "_load_dk_draft_group_lock_ts", lambda *_args, **_kwargs: lock_ts)

    def _fake_score_ownership(
        slate_df: pd.DataFrame,
        draft_group_id: str,
        game_date: date,
        run_id: str,
        data_root: Path,
        model_run: str,
        model_family: str = "ownership_v1",
        injuries_cutoff_ts: datetime | None = None,
        gtv2_features_path: Path | None = None,
    ) -> pd.DataFrame:
        _ = gtv2_features_path
        if injuries_cutoff_ts is None:
            cutoff = pd.NaT
        else:
            cutoff = pd.Timestamp(injuries_cutoff_ts)
            if cutoff.tzinfo is None:
                cutoff = cutoff.tz_localize("UTC")
            else:
                cutoff = cutoff.tz_convert("UTC")
        return pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "player_name": "X",
                    "pred_own_pct": 10.0,
                    "draft_group_id": draft_group_id,
                    "run_id": run_id,
                    "model_run": model_run,
                    "model_family": model_family,
                    "is_locked": False,
                    "injuries_cutoff_ts": cutoff,
                }
            ]
        )

    monkeypatch.setattr(ownership_live, "score_ownership", _fake_score_ownership)

    scoped_locked_path, legacy_locked_path = ownership_live._lock_cache_paths(
        game_date=game_date,
        draft_group_id=dg_id,
        data_root=tmp_path,
        model_family="ownership_v1",
        model_run=ownership_live.PRODUCTION_MODEL_RUN,
    )

    # Before lock: do not create *_locked.parquet.
    ownership_live.score_all_slates(
        game_date,
        "20250101T000000Z",
        tmp_path,
        current_time=lock_ts - timedelta(minutes=10),
    )
    assert not scoped_locked_path.exists()
    assert not legacy_locked_path.exists()

    # If a stale lock cache exists, locked scoring should overwrite it.
    legacy_locked_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "player_id": 1,
                "player_name": "X",
                "pred_own_pct": 10.0,
                "draft_group_id": dg_id,
                "run_id": "stale",
                "model_run": "stale",
                "model_family": "ownership_v1",
                "is_locked": True,
                "injuries_cutoff_ts": pd.Timestamp(lock_ts - timedelta(hours=2)).tz_convert("UTC"),
            }
        ]
    ).to_parquet(legacy_locked_path, index=False)

    ownership_live.score_all_slates(
        game_date,
        "20250101T000500Z",
        tmp_path,
        current_time=lock_ts + timedelta(minutes=1),
    )

    assert scoped_locked_path.exists()
    assert legacy_locked_path.exists()
    locked_df = pd.read_parquet(scoped_locked_path)
    assert bool(locked_df["is_locked"].iloc[0]) is True
    cutoff = pd.to_datetime(locked_df["injuries_cutoff_ts"], utc=True, errors="coerce").max()
    assert cutoff == pd.Timestamp(lock_ts)
