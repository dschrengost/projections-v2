from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from projections.sim_v1.residuals import ResidualBucket, ResidualModel, assign_bucket, from_json

logger = logging.getLogger(__name__)


class FptsResidualSampler:
    def __init__(self, model: ResidualModel):
        self.model = model
        self._bucket_lookup = {bucket.name: bucket for bucket in model.buckets}
        self._default_bucket = ResidualBucket(
            name="__default__",
            min_minutes=0.0,
            max_minutes=None,
            is_starter=None,
            sigma=model.sigma_default,
            nu=model.nu_default,
            n=0,
        )

    @classmethod
    def from_json_file(cls, path: Path) -> "FptsResidualSampler":
        data = json.loads(path.read_text(encoding="utf-8"))
        model = from_json(data)
        return cls(model)

    def _bucket_for_row(self, row: pd.Series) -> ResidualBucket:
        bucket_defs = [
            {
                "name": bucket.name,
                "min_minutes": bucket.min_minutes,
                "max_minutes": bucket.max_minutes,
                "is_starter": bucket.is_starter,
            }
            for bucket in self.model.buckets
        ]
        bucket_name = assign_bucket(row, bucket_defs) if bucket_defs else None
        if bucket_name and bucket_name in self._bucket_lookup:
            return self._bucket_lookup[bucket_name]
        return self._default_bucket

    def sample_worlds(
        self,
        df: pd.DataFrame,
        fpts_mean_col: str = "dk_fpts_pred",
        minutes_col: str = "minutes_pred_p50",
        is_starter_col: str = "is_starter",
        n_worlds: int = 1000,
        seed: int | None = None,
        game_factor_std: float = 0.0,
        game_id_col: str = "game_id",
    ) -> pd.DataFrame:
        """
        Return a long-format DataFrame with columns:
          - identifier columns (season, game_date, game_id, team_id, player_id)
          - dk_fpts_mean (copy of fpts_mean_col)
          - world_id (0..n_worlds-1)
          - dk_fpts_world (simulated FPTS)
        """

        working = df.reset_index(drop=True).copy()

        if fpts_mean_col not in working.columns:
            raise KeyError(f"Missing fpts mean column {fpts_mean_col}")

        if minutes_col != "minutes_pred_p50" and "minutes_pred_p50" not in working.columns:
            working["minutes_pred_p50"] = working.get(minutes_col)
        if is_starter_col != "is_starter" and "is_starter" not in working.columns:
            working["is_starter"] = working.get(is_starter_col, 0)

        if "minutes_pred_p50" in working.columns:
            working["minutes_pred_p50"] = pd.to_numeric(working.get("minutes_pred_p50"), errors="coerce")
        if "minutes_p50" in working.columns:
            working["minutes_p50"] = pd.to_numeric(working.get("minutes_p50"), errors="coerce")
        if "minutes_actual" in working.columns:
            working["minutes_actual"] = pd.to_numeric(working.get("minutes_actual"), errors="coerce")
        working["is_starter"] = working.get("is_starter")
        means = pd.to_numeric(working[fpts_mean_col], errors="coerce").fillna(0.0).to_numpy()

        bucket_names = working.apply(lambda row: self._bucket_for_row(row).name, axis=1)
        working["_bucket_name"] = bucket_names

        n_rows = working.shape[0]
        rng = np.random.default_rng(seed)

        # Draw game-level factors per world.
        samples = np.zeros((n_rows, n_worlds))
        for bucket_name, indices in working.groupby("_bucket_name").groups.items():
            bucket = self._bucket_lookup.get(bucket_name, self._default_bucket)
            sigma = bucket.sigma if bucket.sigma is not None else self.model.sigma_default
            nu = bucket.nu if bucket.nu else self.model.nu_default
            count = len(indices)
            if count == 0:
                continue
            noise = rng.standard_t(df=nu, size=(count, n_worlds)) * sigma
            samples[indices, :] = noise

        if game_factor_std > 0:
            if game_id_col not in working.columns:
                logger.warning("[sim_sampler] game_factor_std>0 but %s missing; skipping game factors", game_id_col)
            else:
                game_ids_clean = working[game_id_col].fillna("__missing__")
                unique_game_ids, inv = np.unique(game_ids_clean.to_numpy(), return_inverse=True)
                factors = rng.normal(loc=0.0, scale=game_factor_std, size=(len(unique_game_ids), n_worlds))
                samples = samples + factors[inv, :]

        samples = np.maximum(0.0, means[:, None] + samples)

        id_cols = [col for col in ("season", "game_date", "game_id", "team_id", "player_id") if col in working.columns]
        id_frame = working[id_cols].reset_index(drop=True) if id_cols else pd.DataFrame(index=working.index)
        base = pd.concat([id_frame, pd.DataFrame({"dk_fpts_mean": means})], axis=1)

        wide = pd.DataFrame(samples, columns=[f"world_{i}" for i in range(n_worlds)])
        wide = pd.concat([base, wide], axis=1)
        long = wide.melt(
            id_vars=list(base.columns),
            value_vars=[c for c in wide.columns if c.startswith("world_")],
            var_name="world_id",
            value_name="dk_fpts_world",
        )
        long["world_id"] = long["world_id"].str.removeprefix("world_").astype(int)
        return long.reset_index(drop=True)


__all__ = ["FptsResidualSampler"]
