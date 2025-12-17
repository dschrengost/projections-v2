"""Tests for DK contest ownership aggregation (zero-fill across contests)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scrapers" / "dk_contests" / "build_ownership_data.py"
    spec = importlib.util.spec_from_file_location("dk_build_ownership_data", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"Failed to load module spec: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_aggregate_slate_ownership_treats_missing_players_as_zero():
    mod = _load_module()

    contest1 = pd.DataFrame(
        {
            "Player": ["A", "B"],
            "own_pct": [500.0, 300.0],  # sums to 800
            "FPTS": [10.0, 20.0],
            "contest_id": [1, 1],
            "entries": [100.0, 100.0],
        }
    )
    contest2 = pd.DataFrame(
        {
            "Player": ["A"],  # B missing => should be treated as 0
            "own_pct": [800.0],  # sums to 800
            "FPTS": [11.0],
            "contest_id": [2],
            "entries": [100.0],
        }
    )
    contest_data = {"1": contest1, "2": contest2}

    agg = mod.aggregate_slate_ownership(["1", "2"], contest_data)
    got = dict(zip(agg["Player"], agg["own_pct"]))
    got_simple = dict(zip(agg["Player"], agg["own_pct_simple"]))

    assert np.isclose(got["A"], 650.0)
    assert np.isclose(got["B"], 150.0)
    assert np.isclose(got_simple["B"], 150.0)

    # With proper zero-fill, slate sums remain stable at 800.
    assert np.isclose(float(agg["own_pct"].sum()), 800.0)

    assert "slate_entries" in agg.columns
    assert "slate_num_contests" in agg.columns
    assert np.isclose(float(agg["slate_entries"].iloc[0]), 200.0)
    assert int(agg["slate_num_contests"].iloc[0]) == 2


def test_aggregate_slate_ownership_entry_weighting_uses_total_entries_denominator():
    mod = _load_module()

    contest1 = pd.DataFrame(
        {
            "Player": ["A", "B"],
            "own_pct": [500.0, 300.0],  # sums to 800
            "FPTS": [10.0, 20.0],
            "contest_id": [1, 1],
            "entries": [100.0, 100.0],
        }
    )
    contest2 = pd.DataFrame(
        {
            "Player": ["A"],  # B missing
            "own_pct": [800.0],  # sums to 800
            "FPTS": [11.0],
            "contest_id": [2],
            "entries": [300.0],
        }
    )
    contest_data = {"1": contest1, "2": contest2}

    agg = mod.aggregate_slate_ownership(["1", "2"], contest_data)
    got = dict(zip(agg["Player"], agg["own_pct"]))

    # Total entries = 400. A = (500*100 + 800*300)/400 = 725. B = (300*100 + 0*300)/400 = 75.
    assert np.isclose(got["A"], 725.0)
    assert np.isclose(got["B"], 75.0)
    assert np.isclose(float(agg["own_pct"].sum()), 800.0)

