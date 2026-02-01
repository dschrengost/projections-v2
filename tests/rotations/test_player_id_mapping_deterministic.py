from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pytest

from projections.rotations.player_map import build_person_id_to_internal_id_map


def _data_root() -> Path:
    return Path(os.environ.get("PROJECTIONS_DATA_ROOT", "/home/daniel/projections-data")).expanduser().resolve()


def _require_season_2024_inputs(root: Path) -> None:
    if not (root / "silver" / "nba_daily_lineups" / "season=2024").exists():
        pytest.skip("Missing silver/nba_daily_lineups season=2024 (requires local projections-data)")
    if not (root / "gold" / "minutes_for_rates_reconciled" / "season=2024").exists():
        pytest.skip("Missing gold/minutes_for_rates_reconciled season=2024 (requires local projections-data)")
    if not (root / "artifacts" / "pbp_v1" / "LATEST_PUBLISHED").exists():
        pytest.skip("Missing artifacts/pbp_v1/LATEST_PUBLISHED (requires local projections-data)")


def _minutes_person_ids_season_2024(root: Path) -> list[int]:
    base = root / "gold" / "minutes_for_rates_reconciled" / "season=2024"
    dataset = ds.dataset(str(base), format="parquet")
    table = dataset.to_table(columns=["player_id"])
    df = table.to_pandas()
    s = pd.to_numeric(df["player_id"], errors="coerce").dropna().astype(int)
    return sorted(set(s.tolist()))


def test_person_id_mapping_season_2024_is_deterministic_and_high_coverage() -> None:
    root = _data_root()
    _require_season_2024_inputs(root)

    result1 = build_person_id_to_internal_id_map(season_start_year=2024, data_root=root)
    result2 = build_person_id_to_internal_id_map(season_start_year=2024, data_root=root)

    assert result1.person_id_to_internal_id == result2.person_id_to_internal_id

    mapping = result1.person_id_to_internal_id
    assert len(mapping) > 0
    assert len(set(mapping.values())) == len(mapping)  # zero collisions
    assert max(mapping.values()) <= 2000

    person_ids = _minutes_person_ids_season_2024(root)
    assert len(person_ids) > 0
    mapped = sum(1 for pid in person_ids if int(pid) in mapping)
    coverage = mapped / len(person_ids)
    assert coverage >= 0.99, f"mapping coverage too low: {coverage:.4%} ({mapped}/{len(person_ids)})"

