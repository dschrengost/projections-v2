from __future__ import annotations

from pathlib import Path

from projections.storage_retention.config import GuardPolicy
from projections.storage_retention.guard import evaluate_storage_guard


def test_storage_guard_ok_with_low_thresholds(tmp_path: Path) -> None:
    policy = GuardPolicy(
        hot_warn_free_gb=0.0,
        hot_warn_free_pct=0.0,
        hot_hard_free_gb=0.0,
        hot_hard_free_pct=0.0,
        root_hard_free_gb=0.0,
    )
    result = evaluate_storage_guard(hot_root=tmp_path, guard_policy=policy, root_path=Path("/"))
    assert result.ok
    assert not result.hard_stop
    assert len(result.failures) == 0


def test_storage_guard_hard_stop_with_high_thresholds(tmp_path: Path) -> None:
    policy = GuardPolicy(
        hot_warn_free_gb=10_000.0,
        hot_warn_free_pct=99.9,
        hot_hard_free_gb=10_000.0,
        hot_hard_free_pct=99.9,
        root_hard_free_gb=10_000.0,
    )
    result = evaluate_storage_guard(hot_root=tmp_path, guard_policy=policy, root_path=Path("/"))
    assert not result.ok
    assert result.hard_stop
    assert len(result.failures) >= 1
