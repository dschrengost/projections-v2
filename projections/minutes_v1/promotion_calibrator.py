"""Promotion prior calibrator for injury-promotion starters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass
class PromotionPriorConfig:
    min_starter_games_for_confidence: int = 5
    min_team_starts: int = 20
    promotion_delta_min: float = 4.0
    promotion_floor_default: float = 24.0
    promotion_floor_by_pos: dict[str, float] | None = None
    alpha_p10: float = 0.3
    alpha_p50: float = 0.5
    alpha_p90: float = 0.4

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromotionPriorConfig":
        kwargs = dict(data or {})
        return cls(**kwargs)


@dataclass
class PromotionPriorContext:
    priors: pd.DataFrame
    player_history: pd.DataFrame
    config: PromotionPriorConfig


def load_promotion_config(path: Path | None) -> PromotionPriorConfig:
    if path is None or not path.exists():
        return PromotionPriorConfig()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return PromotionPriorConfig.from_dict(data)


def _prepare_prior_tables(priors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_priors = priors.loc[priors["scope"] == "team"].copy()
    league_priors = priors.loc[priors["scope"] == "league"].copy()
    league_priors = league_priors.drop(columns=["team_id", "scope"]).rename(
        columns={
            "starter_p25_minutes": "league_p25",
            "starter_p50_minutes": "league_p50",
            "starter_p75_minutes": "league_p75",
            "num_starts": "league_num_starts",
        }
    )
    return team_priors, league_priors


def apply_promotion_prior(
    df: pd.DataFrame,
    priors: pd.DataFrame,
    config: PromotionPriorConfig,
) -> pd.DataFrame:
    if df.empty:
        return df
    required_cols = {
        "team_id",
        "pos_bucket",
        "is_projected_starter",
        "starter_history_games",
        "minutes_p10",
        "minutes_p50",
        "minutes_p90",
    }
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Promotion prior missing columns: {', '.join(missing)}")

    work = df.copy()
    team_priors, league_priors = _prepare_prior_tables(priors)
    work = work.merge(
        team_priors[
            [
                "team_id",
                "pos_bucket",
                "starter_p25_minutes",
                "starter_p50_minutes",
                "starter_p75_minutes",
                "num_starts",
            ]
        ],
        on=["team_id", "pos_bucket"],
        how="left",
    )
    work = work.merge(league_priors, on="pos_bucket", how="left")
    for column in ["starter_history_games"]:
        work[column] = work[column].fillna(0).astype(float)
    work["minutes_p10_raw"] = work["minutes_p10"]
    work["minutes_p50_raw"] = work["minutes_p50"]
    work["minutes_p90_raw"] = work["minutes_p90"]

    effective_p25 = work["starter_p25_minutes"]
    effective_p50 = work["starter_p50_minutes"]
    effective_p75 = work["starter_p75_minutes"]
    effective_n = work["num_starts"]

    if config.min_team_starts:
        fallback_mask = effective_n < config.min_team_starts
        effective_p25 = effective_p25.where(~fallback_mask, work["league_p25"])
        effective_p50 = effective_p50.where(~fallback_mask, work["league_p50"])
        effective_p75 = effective_p75.where(~fallback_mask, work["league_p75"])

    floors = pd.Series(
        config.promotion_floor_by_pos or {},
        name="floor",
    )
    work = work.merge(
        floors.rename_axis("pos_bucket").rename("pos_floor"),
        left_on="pos_bucket",
        right_index=True,
        how="left",
    )
    work["pos_floor"] = work["pos_floor"].astype(float, copy=False)
    work["pos_floor"] = work["pos_floor"].fillna(float(config.promotion_floor_default))

    delta = effective_p50 - work["minutes_p50_raw"]
    candidate_mask = (
        (work["is_projected_starter"] == 1)
        & (work["starter_history_games"] < config.min_starter_games_for_confidence)
        & (effective_p50 >= work["pos_floor"])
        & (delta >= config.promotion_delta_min)
        & effective_p50.notna()
    )

    work["promotion_prior_applied"] = candidate_mask.astype(int)

    for alpha, target_col, prior_series in [
        (config.alpha_p10, "minutes_p10", effective_p25),
        (config.alpha_p50, "minutes_p50", effective_p50),
        (config.alpha_p90, "minutes_p90", effective_p75),
    ]:
        if alpha <= 0.0:
            continue
        work.loc[candidate_mask, target_col] = (
            (1 - alpha) * work.loc[candidate_mask, f"{target_col}_raw"]
            + alpha * prior_series[candidate_mask]
        )

    cleanup = [
        "starter_p25_minutes",
        "starter_p50_minutes",
        "starter_p75_minutes",
        "num_starts",
        "league_p25",
        "league_p50",
        "league_p75",
        "league_num_starts",
        "pos_floor",
    ]
    work = work.drop(columns=[col for col in cleanup if col in work.columns])
    return work
