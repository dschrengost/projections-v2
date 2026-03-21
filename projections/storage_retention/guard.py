"""Free-space guard checks for storage safety."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from projections.storage_retention.config import GuardPolicy


@dataclass(frozen=True)
class StorageGuardResult:
    ok: bool
    hard_stop: bool
    warnings: tuple[str, ...]
    failures: tuple[str, ...]
    payload: dict[str, Any]


def _disk_snapshot(path: Path) -> dict[str, float | str]:
    usage = shutil.disk_usage(path)
    total = float(usage.total)
    free = float(usage.free)
    used = float(usage.used)
    pct_free = (free / total) * 100.0 if total > 0 else 0.0
    return {
        "path": str(path),
        "total_bytes": int(total),
        "used_bytes": int(used),
        "free_bytes": int(free),
        "free_gb": free / (1024.0 ** 3),
        "free_pct": pct_free,
    }


def evaluate_storage_guard(
    *,
    hot_root: Path,
    guard_policy: GuardPolicy,
    root_path: Path | None = None,
) -> StorageGuardResult:
    root = root_path or Path("/")
    hot_snapshot = _disk_snapshot(hot_root)
    root_snapshot = _disk_snapshot(root)

    warnings: list[str] = []
    failures: list[str] = []

    hot_free_gb = float(hot_snapshot["free_gb"])
    hot_free_pct = float(hot_snapshot["free_pct"])
    root_free_gb = float(root_snapshot["free_gb"])

    if hot_free_gb < float(guard_policy.hot_warn_free_gb) or hot_free_pct < float(
        guard_policy.hot_warn_free_pct
    ):
        warnings.append(
            "hot_root below warning threshold "
            f"(free_gb={hot_free_gb:.1f}, free_pct={hot_free_pct:.1f})"
        )

    if hot_free_gb < float(guard_policy.hot_hard_free_gb) or hot_free_pct < float(
        guard_policy.hot_hard_free_pct
    ):
        failures.append(
            "hot_root below hard threshold "
            f"(free_gb={hot_free_gb:.1f}, free_pct={hot_free_pct:.1f})"
        )

    if root_free_gb < float(guard_policy.root_hard_free_gb):
        failures.append(
            "root filesystem below hard threshold "
            f"(free_gb={root_free_gb:.1f})"
        )

    payload = {
        "checked_at": datetime.now(tz=UTC).isoformat(),
        "hot": hot_snapshot,
        "root": root_snapshot,
        "thresholds": {
            "hot_warn_free_gb": float(guard_policy.hot_warn_free_gb),
            "hot_warn_free_pct": float(guard_policy.hot_warn_free_pct),
            "hot_hard_free_gb": float(guard_policy.hot_hard_free_gb),
            "hot_hard_free_pct": float(guard_policy.hot_hard_free_pct),
            "root_hard_free_gb": float(guard_policy.root_hard_free_gb),
        },
        "warnings": warnings,
        "failures": failures,
    }
    return StorageGuardResult(
        ok=len(failures) == 0,
        hard_stop=len(failures) > 0,
        warnings=tuple(warnings),
        failures=tuple(failures),
        payload=payload,
    )


def ensure_storage_headroom_or_raise(
    *,
    hot_root: Path,
    guard_policy: GuardPolicy,
    root_path: Path | None = None,
) -> dict[str, Any]:
    result = evaluate_storage_guard(
        hot_root=hot_root,
        guard_policy=guard_policy,
        root_path=root_path,
    )
    if result.hard_stop:
        raise RuntimeError(
            "Storage guard hard stop: " + "; ".join(result.failures)
        )
    return result.payload
