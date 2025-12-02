from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from projections.fpts_v2.current import _load_current_run_id as load_current_fpts_run_id
from projections.minutes_v1.production import resolve_production_run_dir
from projections.paths import get_project_root
from projections.rates_v1.current import get_rates_current_run_id


DEFAULT_PROFILES_PATH = get_project_root() / "config" / "sim_v2_profiles.json"


@dataclass
class SimV2Profile:
    name: str
    fpts_run_id: str
    rates_run_id: Optional[str]
    minutes_run_id: Optional[str]
    use_rates_noise: bool
    rates_noise_split: Optional[str]
    use_minutes_noise: bool
    minutes_sigma_min: float
    worlds_per_chunk: int
    seed: Optional[int]
    min_play_prob: float
    team_factor_sigma: float
    team_factor_gamma: float


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON at {path}: {exc}") from exc


def _resolve_minutes_run_id(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    _, run_id = resolve_production_run_dir()
    return str(run_id) if run_id else None


def load_sim_v2_profile(
    *,
    profile: str = "baseline",
    profiles_path: Optional[Path] = None,
) -> SimV2Profile:
    path = (profiles_path or DEFAULT_PROFILES_PATH).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"sim_v2 profiles config missing at {path}")
    payload = _read_json(path)
    profiles = payload.get("profiles") or {}
    config = profiles.get(profile)
    if config is None:
        raise KeyError(f"Profile '{profile}' not found in {path}")

    fpts_run_id = config.get("fpts_run_id") or load_current_fpts_run_id()
    rates_run_id = config.get("rates_run_id") or get_rates_current_run_id()
    minutes_run_id = _resolve_minutes_run_id(config.get("minutes_run_id"))

    use_rates_noise = bool(config.get("rates_noise", {}).get("enabled", True))
    rates_noise_split = config.get("rates_noise", {}).get("split", "val")

    use_minutes_noise = bool(config.get("minutes_noise", {}).get("enabled", True))
    minutes_sigma_min = float(config.get("minutes_noise", {}).get("sigma_min", 1.0))

    worlds_per_chunk = int(config.get("worlds_per_chunk", 2000))
    seed = config.get("seed")
    seed = int(seed) if seed is not None else None
    min_play_prob = float(config.get("min_play_prob", 0.05))
    team_factor_sigma = float(config.get("team_factor_sigma", 0.0))
    team_factor_gamma = float(config.get("team_factor_gamma", 1.0))

    return SimV2Profile(
        name=profile,
        fpts_run_id=str(fpts_run_id),
        rates_run_id=str(rates_run_id) if rates_run_id is not None else None,
        minutes_run_id=minutes_run_id,
        use_rates_noise=use_rates_noise,
        rates_noise_split=str(rates_noise_split) if rates_noise_split is not None else None,
        use_minutes_noise=use_minutes_noise,
        minutes_sigma_min=minutes_sigma_min,
        worlds_per_chunk=worlds_per_chunk,
        seed=seed,
        min_play_prob=min_play_prob,
        team_factor_sigma=team_factor_sigma,
        team_factor_gamma=team_factor_gamma,
    )


__all__ = ["SimV2Profile", "load_sim_v2_profile", "DEFAULT_PROFILES_PATH"]
