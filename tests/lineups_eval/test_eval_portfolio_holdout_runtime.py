from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from projections.eval.portfolio_holdout_runtime import add_threshold_metric, summarize_lineup_scores
from projections.sim_v2.runtime_worlds import generate_synthetic_worlds_matrix


def test_runtime_seed_determinism_checksum() -> None:
    player_ids = ["1", "2", "3"]
    means = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    stdevs = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    res_a = generate_synthetic_worlds_matrix(
        player_ids=player_ids,
        means=means,
        stdevs=stdevs,
        num_worlds=17,
        seed=123,
    )
    res_b = generate_synthetic_worlds_matrix(
        player_ids=player_ids,
        means=means,
        stdevs=stdevs,
        num_worlds=17,
        seed=123,
    )
    assert np.array_equal(res_a.worlds_matrix, res_b.worlds_matrix)
    chk_a = hashlib.sha256(res_a.worlds_matrix.tobytes()).hexdigest()
    chk_b = hashlib.sha256(res_b.worlds_matrix.tobytes()).hexdigest()
    assert chk_a == chk_b

    res_c = generate_synthetic_worlds_matrix(
        player_ids=player_ids,
        means=means,
        stdevs=stdevs,
        num_worlds=17,
        seed=124,
    )
    chk_c = hashlib.sha256(res_c.worlds_matrix.tobytes()).hexdigest()
    assert chk_a != chk_c


def test_metrics_correctness_toy_example() -> None:
    # 2 lineups, 3 worlds.
    scores = np.array(
        [
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
        ],
        dtype=np.float64,
    )
    summary = summarize_lineup_scores(scores)
    assert summary["n_lineups"] == 2
    assert summary["n_worlds"] == 3

    # Per-lineup means.
    per = summary["per_lineup"]
    assert per[0]["mean"] == 2.0
    assert per[1]["mean"] == 2.0

    # Portfolio max is computed per-world, then summarized.
    # max per world: [3, 2, 3]
    max_summary = summary["portfolio_max"]
    assert max_summary["mean"] == (3.0 + 2.0 + 3.0) / 3.0

    thresh = add_threshold_metric(scores=scores, threshold=2.0)
    assert thresh["threshold"] == 2.0
    assert thresh["p_max_gt_threshold"] == 2.0 / 3.0


def test_eval_script_smoke_base_only(tmp_path: Path) -> None:
    # Tiny base worlds matrix fixture.
    base_worlds = pd.DataFrame(
        np.array(
            [
                [10.0, 0.0, 5.0],
                [0.0, 20.0, 5.0],
                [5.0, 5.0, 5.0],
                [8.0, 8.0, 8.0],
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0],
                [7.0, 7.0, 7.0],
                [4.0, 0.0, 4.0],
                [0.0, 4.0, 4.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=np.float64,
        ),
        columns=["1", "2", "3"],
    )
    base_path = tmp_path / "worlds_matrix.parquet"
    base_worlds.to_parquet(base_path, index=False)

    # Tiny portfolio (p1_id..p2_id) + draft_group_id for provenance.
    lineups_df = pd.DataFrame(
        {
            "lineup_id": ["L0", "L1"],
            "draft_group_id": [999, 999],
            "p1_id": [1, 2],
            "p2_id": [3, 3],
        }
    )
    lineups_path = tmp_path / "lineups.csv"
    lineups_df.to_csv(lineups_path, index=False)

    out_dir = tmp_path / "eval_out"
    cmd = [
        sys.executable,
        "scripts/lineups/eval_portfolio_holdout_runtime.py",
        "--date",
        "2026-01-10",
        "--site",
        "dk",
        "--lineups-path",
        str(lineups_path),
        "--base-worlds-path",
        str(base_path),
        "--k-runtime-holdouts",
        "0",
        "--num-worlds-runtime",
        "10",
        "--seed",
        "123",
        "--output-root",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True, cwd=str(Path(__file__).resolve().parents[2]))

    assert (out_dir / "eval_report.md").exists()
    assert (out_dir / "eval_report.json").exists()
    assert (out_dir / "runtime_holdout_manifest.json").exists()

    payload = json.loads((out_dir / "eval_report.json").read_text(encoding="utf-8"))
    assert payload["selection_inputs"]["draft_group_id"] == 999
    assert payload["metrics"]["base_holdout"]["n_worlds"] == 3
    assert payload["metrics"]["runtime_holdouts"] == []

