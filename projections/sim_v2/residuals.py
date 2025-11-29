from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from projections.fpts_v2.loader import ResidualBucket, ResidualModel


def select_bucket(row: pd.Series, residual_model: ResidualModel) -> Optional[ResidualBucket]:
    minutes = row.get("minutes_p50")
    if pd.isna(minutes):
        return None
    starter = row.get("is_starter")
    if pd.isna(starter):
        return None
    minutes_val = float(minutes)
    starter_val = int(starter)
    for bucket in residual_model.buckets:
        if bucket.is_starter != starter_val:
            continue
        if minutes_val < bucket.min_minutes:
            continue
        if bucket.max_minutes is not None and minutes_val >= bucket.max_minutes:
            continue
        return bucket
    return None


def get_bucket_params(row: pd.Series, residual_model: ResidualModel) -> tuple[float, int]:
    bucket = select_bucket(row, residual_model)
    if bucket is None:
        return residual_model.sigma_default, residual_model.nu_default
    return bucket.sigma, bucket.nu


def _sample_student_t(rng: np.random.Generator, nu: int, size: int) -> np.ndarray:
    # Use numpy.random.standard_t if available
    return rng.standard_t(df=nu, size=size)


def sample_residuals(df: pd.DataFrame, residual_model: ResidualModel, rng: np.random.Generator) -> np.ndarray:
    sigmas = []
    nus = []
    for _, row in df.iterrows():
        sigma, nu = get_bucket_params(row, residual_model)
        sigmas.append(sigma)
        nus.append(nu)
    sigmas_arr = np.array(sigmas, dtype=float)
    nus_arr = np.array(nus, dtype=int)

    residuals = np.zeros(len(df))
    for nu_val in np.unique(nus_arr):
        idx = np.where(nus_arr == nu_val)[0]
        if idx.size == 0:
            continue
        t_samples = _sample_student_t(rng, nu_val, size=idx.size)
        residuals[idx] = sigmas_arr[idx] * t_samples
    return residuals


def sample_residuals_with_team_factor(
    df: pd.DataFrame,
    residual_model: ResidualModel,
    rng: np.random.Generator,
    *,
    dk_fpts_col: str = "dk_fpts_mean",
    minutes_col: str = "minutes_p50",
    is_starter_col: str = "is_starter",
    game_id_col: str = "game_id",
    team_id_col: str = "team_id",
    team_factor_sigma: float = 0.0,
    alpha_gamma: float = 1.0,
) -> np.ndarray:
    """
    Sample residuals with optional team-level latent factor.

    If team_factor_sigma <= 0, falls back to independent residuals (sample_residuals).
    """

    base_residuals = sample_residuals(df, residual_model, rng)
    n = len(df)
    if n == 0 or team_factor_sigma <= 0:
        return base_residuals

    required_cols = {dk_fpts_col, minutes_col, is_starter_col, game_id_col, team_id_col}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for team factor sampling: {missing}")

    means = df[dk_fpts_col].to_numpy()
    game_ids = df[game_id_col].to_numpy()
    team_ids = df[team_id_col].to_numpy()

    keys = list(zip(game_ids, team_ids))
    group_map: dict[tuple, list[int]] = {}
    for idx, key in enumerate(keys):
        group_map.setdefault(key, []).append(idx)

    residuals = base_residuals.copy()

    for idx_list in group_map.values():
        idx_arr = np.array(idx_list, dtype=int)
        m = means[idx_arr]
        alpha = np.power(np.maximum(m, 0.0), alpha_gamma)
        alpha = np.nan_to_num(alpha, nan=0.0)
        if alpha.sum() <= 0:
            alpha = np.ones_like(alpha, dtype=float)
        alpha = alpha / alpha.sum()
        g_team = rng.normal(loc=0.0, scale=team_factor_sigma)
        residuals[idx_arr] += g_team * alpha

    return residuals


__all__ = ["select_bucket", "get_bucket_params", "sample_residuals", "sample_residuals_with_team_factor"]
