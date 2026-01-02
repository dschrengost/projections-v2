from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.api import user_overrides
from projections.optimizer import player_pool_loader


def test_legacy_user_overrides_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(user_overrides.ENV_ALLOW_LEGACY_USER_OVERRIDES, raising=False)
    with pytest.raises(RuntimeError, match=r"legacy-override"):
        user_overrides.load_slate_overrides("2025-01-01", 123)


def test_readers_do_not_scan_run_dirs_without_pointer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    game_date = "2025-01-01"
    run_id = "TEST_RUN"

    monkeypatch.delenv("PROJECTIONS_ALLOW_UNPROMOTED_RUN_READS", raising=False)
    monkeypatch.delenv("PROJECTIONS_ALLOW_LEGACY_FLAT_GOLD_READS", raising=False)

    run_dir = tmp_path / "gold" / "projections_minutes_v1" / f"game_date={game_date}" / f"run={run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"player_id": [1], "minutes_p50": [30.0]}).to_parquet(run_dir / "minutes.parquet", index=False)

    with pytest.raises(FileNotFoundError):
        player_pool_loader.load_minutes_for_date(game_date, root=str(tmp_path))

    monkeypatch.setenv("PROJECTIONS_ALLOW_UNPROMOTED_RUN_READS", "1")
    loaded = player_pool_loader.load_minutes_for_date(game_date, root=str(tmp_path))
    assert not loaded.empty

