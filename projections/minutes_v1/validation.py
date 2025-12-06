"""Validation helpers for Minutes V1 guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .constants import ROLE_MINUTES_CAPS


@dataclass(slots=True)
class ReconciliationConfig:
    """Minimal config for reconciliation sanity check (legacy)."""

    target_total: float = 240.0


def sample_anti_leak_check(
    df: pd.DataFrame,
    *,
    tip_col: str = "tip_ts",
    columns: Iterable[str] = ("feature_as_of_ts", "injury_as_of_ts", "odds_as_of_ts"),
    sample_size: int = 1000,
) -> None:
    """Probe a sample of player-games to ensure as-of timestamps never exceed tip."""

    if tip_col not in df:
        raise ValueError(f"Missing tip timestamp column '{tip_col}'.")
    population = df.dropna(subset=[tip_col])
    if population.empty:
        raise ValueError("Cannot perform anti-leak check on empty dataframe.")
    sample = population.sample(n=min(sample_size, len(population)), random_state=42)
    sample_tip = pd.to_datetime(sample[tip_col], utc=True)
    for column in columns:
        if column not in sample:
            continue
        as_of = pd.to_datetime(sample[column], utc=True, errors="coerce")
        # Skip NaT values (missing timestamps) - only check rows with valid data
        valid_mask = as_of.notna() & sample_tip.notna()
        if valid_mask.any() and not (as_of[valid_mask] <= sample_tip[valid_mask]).all():
            raise AssertionError(f"Anti-leak violation detected: '{column}' exceeds '{tip_col}'.")


def hash_season_labels(root: Path) -> dict[str, str]:
    """Compute SHA256 hashes for each season's boxscore label parquet."""

    import hashlib

    hashes: dict[str, str] = {}
    for label_file in sorted(root.glob("season=*/boxscore_labels.parquet")):
        season = label_file.parent.name.split("=", 1)[-1]
        buffer_size = 1024 * 1024
        digest = hashlib.sha256()
        with label_file.open("rb") as handle:
            while chunk := handle.read(buffer_size):
                digest.update(chunk)
        hashes[season] = digest.hexdigest()
    if not hashes:
        raise FileNotFoundError(f"No label parquet files found under {root}")
    return hashes


def validate_label_hashes(root: Path, expected_hashes: Mapping[str, str]) -> None:
    """Confirm that stored label hashes match the on-disk parquet content."""

    current_hashes = hash_season_labels(root)
    for season, expected in expected_hashes.items():
        actual = current_hashes.get(season)
        if actual is None:
            raise FileNotFoundError(f"Missing frozen labels for season {season}")
        if actual != expected:
            raise AssertionError(f"Label hash mismatch for season {season}: {actual} != {expected}")


@dataclass
class ReconciliationReport:
    """Results for a reconciliation sanity check."""

    teams_checked: int
    cap_violations: int
    total_error_before: float
    total_error_after: float

    @property
    def improved(self) -> bool:
        return self.total_error_after <= self.total_error_before


def reconciliation_sanity_check(
    df: pd.DataFrame,
    *,
    cfg: ReconciliationConfig | None = None,
    minutes_col: str = "p50",
    reconciled_col: str = "minutes_reconciled",
) -> ReconciliationReport:
    """Validate that reconciliation sums ≈ target, respects caps, and improves totals."""

    config = cfg or ReconciliationConfig()
    required_cols = {
        minutes_col,
        reconciled_col,
        "team_id",
        "game_id",
        "starter_prev_game_asof",
        "ramp_flag",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for reconciliation sanity: {', '.join(sorted(missing))}")

    total_error_before = 0.0
    total_error_after = 0.0
    cap_violations = 0
    teams_checked = 0

    for _, group in df.groupby(["game_id", "team_id"]):
        teams_checked += 1
        raw_sum = float(group[minutes_col].sum())
        reconciled_sum = float(group[reconciled_col].sum())
        total_error_before += (raw_sum - config.target_total) ** 2
        total_error_after += (reconciled_sum - config.target_total) ** 2

        caps = np.where(
            group["ramp_flag"].astype(bool),
            ROLE_MINUTES_CAPS["ramp"],
            np.where(
                group["starter_prev_game_asof"].astype(bool),
                ROLE_MINUTES_CAPS["starter"],
                ROLE_MINUTES_CAPS["bench"],
            ),
        )
        exceeded = group[reconciled_col].to_numpy() - caps
        cap_violations += int((exceeded > 1e-6).sum())
        if not np.isclose(reconciled_sum, config.target_total, atol=1e-2):
            raise AssertionError("Reconciled team total deviates from 240 target.")

    report = ReconciliationReport(
        teams_checked=teams_checked,
        cap_violations=cap_violations,
        total_error_before=total_error_before,
        total_error_after=total_error_after,
    )
    if cap_violations:
        raise AssertionError(f"Detected {cap_violations} cap violations after reconciliation.")
    if report.total_error_after > report.total_error_before + 1e-6:
        raise AssertionError("Reconciliation increased total-sum squared error.")
    return report
