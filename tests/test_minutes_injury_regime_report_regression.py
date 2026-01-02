"""Regression checks for the injury-regime minutes evaluation report.

This guards the core "next man up" promise: when starters are OUT, the model
should stop smearing minutes and correctly call up the bench core.
"""

from __future__ import annotations

import json
from pathlib import Path


def test_rotshare_report_improves_bench_core_without_regressing_overall():
    # Use a longer historical window where the strict non-injury slice has enough samples
    # to make regressions meaningful.
    report_path = Path("reports/minutes_injury_regime/2024-10_2025-04_rotshare_msfix_k10_e1.json")
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    current_injury = float(payload["models"]["current"]["injury_regime"]["bench_core_mae"])
    candidate_injury = float(payload["models"]["candidate"]["injury_regime"]["bench_core_mae"])
    assert current_injury - candidate_injury >= 0.5

    current_non_injury = float(payload["models"]["current"]["non_injury"]["player_mae"])
    candidate_non_injury = float(payload["models"]["candidate"]["non_injury"]["player_mae"])
    assert candidate_non_injury <= current_non_injury + 0.1

    non_injury_rows = payload["slices"]["non_injury"]["player_rows"]
    all_games_rows = payload["slices"]["all_games"]["player_rows"]
    assert non_injury_rows < all_games_rows

    starters_out_buckets = payload["models"]["current"]["non_injury"]["player_error_by_starters_out"]
    for label, row in starters_out_buckets.items():
        if label != "0":
            assert int(row.get("n", 0)) == 0
