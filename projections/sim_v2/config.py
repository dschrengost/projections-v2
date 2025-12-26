from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from projections.paths import get_project_root


DEFAULT_PROFILES_PATH = get_project_root() / "config" / "sim_v2_profiles.json"


@dataclass
class MinutesNoiseConfig:
    """Config for per-world minutes noise with cheap team-240 projection."""

    enabled: bool = True
    # Noise sigma for starters vs bench
    sigma_starter: float = 2.0
    sigma_bench: float = 3.0
    # Minimum baseline minutes to apply noise (skip deep bench)
    min_minutes_for_noise: float = 8.0
    # Hard cap on absolute noise per player
    cap_abs: float = 6.0
    # Whether to use Student-t instead of normal
    use_student_t: bool = False
    t_df: float = 8.0
    # Bounds for clamping: "zero" | "p10" for lo_source; hi_source typically "p90"
    lo_source: str = "zero"
    hi_source: str = "p90"
    # Padding for bounds (additive)
    lo_pad: float = 0.0
    hi_pad: float = 2.0


@dataclass
class PreSimReconcileConfig:
    """Config for pre-sim QP reconciliation (runs once before simulation)."""

    enabled: bool = False
    # Weight multiplier for starters (higher = more rigid, less adjustment)
    starter_weight: float = 2.0
    # Weight multiplier for high-minute players (base: minutes / 20)
    minutes_weight_scale: float = 1.0


@dataclass
class UsageSharesConfig:
    """Config for stochastic usage share allocation within teams."""

    enabled: bool = False
    targets: tuple[str, ...] = ("fga", "fta", "tov")
    backend: str = "rate_weighted"  # "rate_weighted" | "lgbm_residual"
    run_id: Optional[str] = None  # Run ID for learned model (None = use default)
    shrink: Optional[float] = None  # Shrinkage for residual model (None = use bundle default)
    share_temperature: float = 1.0
    share_noise_std: float = 0.15
    min_minutes_active_cutoff: float = 2.0
    fallback: str = "rate_weighted"


@dataclass
class SimV2Profile:
    name: str
    rates_run_id: Optional[str]
    minutes_run_id: Optional[str]
    use_rates_noise: bool
    rates_noise_split: Optional[str]
    rates_noise_run_id: Optional[str]
    use_minutes_noise: bool
    minutes_sigma_min: float
    worlds_per_chunk: int
    seed: Optional[int]
    min_play_prob: float
    team_factor_sigma: float
    team_factor_gamma: float
    enforce_team_240: bool
    use_efficiency_scoring: bool = True
    rates_sigma_scale: float = 1.0
    team_sigma_scale: float = 1.0
    player_sigma_scale: float = 1.0
    mean_source: str = "rates"
    minutes_source: Optional[str] = None
    rates_source: Optional[str] = None
    noise: dict[str, Any] = field(default_factory=dict)
    worlds_n: Optional[int] = None
    worlds_batch_size: Optional[int] = None
    # Game script adjustments
    use_game_scripts: bool = False
    game_script_margin_std: float = 13.4
    game_script_spread_coef: float = -0.726
    game_script_quantile_targets: dict[str, tuple[float, float]] = field(default_factory=dict)
    game_script_quantile_noise_std: float = 0.08
    game_script_rotation_threshold: float = 20.0  # p50 >= this → use starter quantiles
    # Vegas anchoring (team points vs implied totals)
    vegas_points_anchor: bool = False
    vegas_points_drift_pct: float = 0.05
    # Rotation handling
    rotation_minutes_floor: float = 0.0  # Prune players with < floor minutes
    max_rotation_size: int | None = None  # None = legacy (10), 0 = disabled
    protected_rotation_size: int | None = None  # Optional protected core size within rotation cap
    # Usage shares allocation (stochastic within-team opportunity distribution)
    usage_shares: UsageSharesConfig = field(default_factory=UsageSharesConfig)
    # Vacancy feature mode: "none" = set to 0, "game" = compute from play_prob
    vacancy_mode: str = "game"  # "none" | "game"
    # Whether to sample active mask from play_prob (True = Bernoulli sampling, False = all active)
    use_play_prob_masking: bool = True
    # When True, sim does not apply eligible_flag filtering or rotation pruning - RotAlloc already handled it
    preserve_input_rotation: bool = False
    # New structured minutes noise config (per-world noise + cheap team-240 projection)
    minutes_noise_config: MinutesNoiseConfig = field(default_factory=MinutesNoiseConfig)
    # Pre-sim QP reconciliation (runs once before simulation)
    pre_sim_reconcile: PreSimReconcileConfig = field(default_factory=PreSimReconcileConfig)
    # Optional explicit minutes bundle path (overrides minutes_run_id resolution)
    minutes_bundle_path: Optional[str] = None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON at {path}: {exc}") from exc


def _resolve_minutes_run_id(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return str(explicit)
    # Default to latest minutes artifacts for the target date; do not auto-resolve to a
    # model/bundle run id here (that is handled separately for minutes noise params).
    return None


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

    mean_source = str(config.get("mean_source", "rates"))
    if mean_source != "rates":
        raise ValueError(f"Unsupported sim_v2 mean_source={mean_source!r}; rates-only path is supported.")
    # For rates mean_source, don't auto-resolve rates_run_id - it's determined per-date from rates_v1_live
    rates_run_id = config.get("rates_run_id")  # Keep as None if not specified
    minutes_run_id = _resolve_minutes_run_id(config.get("minutes_run_id"))

    use_rates_noise = bool(config.get("rates_noise", {}).get("enabled", True))
    rates_noise_split = config.get("rates_noise", {}).get("split", "val")
    rates_noise_run_id = config.get("rates_noise", {}).get("run_id")
    rates_sigma_scale = float(config.get("rates_sigma_scale", 1.0))
    team_sigma_scale = float(config.get("team_sigma_scale", 1.0))
    player_sigma_scale = float(config.get("player_sigma_scale", 1.0))

    use_minutes_noise = bool(config.get("minutes_noise", {}).get("enabled", True))
    minutes_sigma_min = float(config.get("minutes_noise", {}).get("sigma_min", 1.0))

    worlds_cfg = config.get("worlds", {}) or {}
    worlds_n = worlds_cfg.get("n_worlds")
    worlds_batch_size_raw = worlds_cfg.get("batch_size")
    worlds_batch_size = int(worlds_batch_size_raw) if worlds_batch_size_raw is not None else None
    worlds_per_chunk = int(config.get("worlds_per_chunk", worlds_batch_size or 2000))
    seed = config.get("seed")
    seed = int(seed) if seed is not None else None
    min_play_prob = float(config.get("min_play_prob", 0.05))
    team_factor_sigma = float(config.get("team_factor_sigma", 0.0))
    team_factor_gamma = float(config.get("team_factor_gamma", 1.0))
    enforce_team_240 = bool(config.get("enforce_team_240", False))
    minutes_source = config.get("minutes_source")
    rates_source = config.get("rates_source")
    noise_cfg = config.get("noise", {}) or {}
    use_efficiency_scoring = bool(config.get("efficiency_scoring", True))

    vegas_cfg = config.get("vegas_anchoring", {}) or {}
    vegas_points_anchor = bool(vegas_cfg.get("enabled", False))
    vegas_points_drift_pct = float(vegas_cfg.get("drift_pct", 0.05))

    # Rotation handling config
    rotation_cfg = config.get("rotation", {}) or {}
    rotation_minutes_floor = float(rotation_cfg.get("minutes_floor", 0.0))
    max_rotation_size_raw = rotation_cfg.get("max_size")
    protected_rotation_size_raw = rotation_cfg.get("protected_size")
    if max_rotation_size_raw is not None:
        max_rotation_size = int(max_rotation_size_raw) if max_rotation_size_raw else None
    else:
        max_rotation_size = None  # Will use legacy default (10) in sim code

    if protected_rotation_size_raw is not None:
        protected_rotation_size = int(protected_rotation_size_raw) if protected_rotation_size_raw else None
    else:
        protected_rotation_size = None

    # Usage shares config
    usage_shares_cfg = config.get("usage_shares", {}) or {}
    usage_shares_run_id_raw = usage_shares_cfg.get("run_id")
    usage_shares_shrink_raw = usage_shares_cfg.get("shrink")
    usage_shares = UsageSharesConfig(
        enabled=bool(usage_shares_cfg.get("enabled", False)),
        targets=tuple(usage_shares_cfg.get("targets", ("fga", "fta", "tov"))),
        backend=str(usage_shares_cfg.get("backend", "rate_weighted")),
        run_id=str(usage_shares_run_id_raw) if usage_shares_run_id_raw is not None else None,
        shrink=float(usage_shares_shrink_raw) if usage_shares_shrink_raw is not None else None,
        share_temperature=float(usage_shares_cfg.get("share_temperature", 1.0)),
        share_noise_std=float(usage_shares_cfg.get("share_noise_std", 0.15)),
        min_minutes_active_cutoff=float(usage_shares_cfg.get("min_minutes_active_cutoff", 2.0)),
        fallback=str(usage_shares_cfg.get("fallback", "rate_weighted")),
    )

    # Vacancy mode config
    vacancy_mode_raw = config.get("vacancy_mode", "game")
    vacancy_mode = str(vacancy_mode_raw) if vacancy_mode_raw else "game"
    if vacancy_mode not in ("none", "game"):
        raise ValueError(f"Invalid vacancy_mode: {vacancy_mode}. Must be 'none' or 'game'.")

    # Play prob masking config (defaults to True for backward compat)
    use_play_prob_masking = bool(config.get("use_play_prob_masking", True))

    # Preserve input rotation: when True, sim skips eligible_flag filtering and rotation pruning
    # (RotAlloc already handled rotation selection upstream)
    preserve_input_rotation = bool(config.get("preserve_input_rotation", False))

    # Minutes bundle path (explicit override for bundle location)
    minutes_bundle_path_raw = config.get("minutes_bundle_path")
    minutes_bundle_path = str(minutes_bundle_path_raw) if minutes_bundle_path_raw else None

    # New structured minutes noise config
    minutes_noise_cfg_raw = config.get("minutes_noise_config", {}) or {}
    minutes_noise_config = MinutesNoiseConfig(
        enabled=bool(minutes_noise_cfg_raw.get("enabled", True)),
        sigma_starter=float(minutes_noise_cfg_raw.get("sigma_starter", 2.0)),
        sigma_bench=float(minutes_noise_cfg_raw.get("sigma_bench", 3.0)),
        min_minutes_for_noise=float(minutes_noise_cfg_raw.get("min_minutes_for_noise", 8.0)),
        cap_abs=float(minutes_noise_cfg_raw.get("cap_abs", 6.0)),
        use_student_t=bool(minutes_noise_cfg_raw.get("use_student_t", False)),
        t_df=float(minutes_noise_cfg_raw.get("t_df", 8.0)),
        lo_source=str(minutes_noise_cfg_raw.get("lo_source", "zero")),
        hi_source=str(minutes_noise_cfg_raw.get("hi_source", "p90")),
        lo_pad=float(minutes_noise_cfg_raw.get("lo_pad", 0.0)),
        hi_pad=float(minutes_noise_cfg_raw.get("hi_pad", 2.0)),
    )

    # Pre-sim QP reconciliation config
    pre_sim_reconcile_cfg_raw = config.get("pre_sim_reconcile", {}) or {}
    pre_sim_reconcile = PreSimReconcileConfig(
        enabled=bool(pre_sim_reconcile_cfg_raw.get("enabled", False)),
        starter_weight=float(pre_sim_reconcile_cfg_raw.get("starter_weight", 2.0)),
        minutes_weight_scale=float(pre_sim_reconcile_cfg_raw.get("minutes_weight_scale", 1.0)),
    )

    # Game script config
    game_script_cfg = config.get("game_script", {}) or {}
    use_game_scripts = bool(game_script_cfg.get("enabled", False))
    game_script_margin_std = float(game_script_cfg.get("margin_std", 13.4))
    game_script_spread_coef = float(game_script_cfg.get("spread_coef", -0.726))
    game_script_quantile_noise_std = float(game_script_cfg.get("quantile_noise_std", 0.08))
    game_script_rotation_threshold = float(game_script_cfg.get("rotation_p50_threshold", 20.0))
    quantile_targets: dict[str, tuple[float, float]] = {}
    raw_targets = game_script_cfg.get("quantile_targets") or {}
    if isinstance(raw_targets, dict):
        for script, targets in raw_targets.items():
            if not isinstance(targets, dict):
                continue
            starter = float(targets.get("starter", 0.5))
            bench = float(targets.get("bench", 0.5))
            quantile_targets[str(script)] = (starter, bench)

    return SimV2Profile(
        name=profile,
        rates_run_id=str(rates_run_id) if rates_run_id is not None else None,
        minutes_run_id=minutes_run_id,
        use_rates_noise=use_rates_noise,
        rates_noise_split=str(rates_noise_split) if rates_noise_split is not None else None,
        rates_noise_run_id=str(rates_noise_run_id) if rates_noise_run_id is not None else None,
        rates_sigma_scale=rates_sigma_scale,
        team_sigma_scale=team_sigma_scale,
        player_sigma_scale=player_sigma_scale,
        use_minutes_noise=use_minutes_noise,
        minutes_sigma_min=minutes_sigma_min,
        worlds_per_chunk=worlds_per_chunk,
        seed=seed,
        min_play_prob=min_play_prob,
        team_factor_sigma=team_factor_sigma,
        team_factor_gamma=team_factor_gamma,
        enforce_team_240=enforce_team_240,
        use_efficiency_scoring=use_efficiency_scoring,
        mean_source=mean_source,
        minutes_source=str(minutes_source) if minutes_source is not None else None,
        rates_source=str(rates_source) if rates_source is not None else None,
        noise=noise_cfg,
        worlds_n=int(worlds_n) if worlds_n is not None else None,
        worlds_batch_size=worlds_batch_size,
        use_game_scripts=use_game_scripts,
        game_script_margin_std=game_script_margin_std,
        game_script_spread_coef=game_script_spread_coef,
        game_script_quantile_targets=quantile_targets,
        game_script_quantile_noise_std=game_script_quantile_noise_std,
        game_script_rotation_threshold=game_script_rotation_threshold,
        vegas_points_anchor=vegas_points_anchor,
        vegas_points_drift_pct=vegas_points_drift_pct,
        rotation_minutes_floor=rotation_minutes_floor,
        max_rotation_size=max_rotation_size,
        protected_rotation_size=protected_rotation_size,
        usage_shares=usage_shares,
        vacancy_mode=vacancy_mode,
        use_play_prob_masking=use_play_prob_masking,
        preserve_input_rotation=preserve_input_rotation,
        minutes_noise_config=minutes_noise_config,
        pre_sim_reconcile=pre_sim_reconcile,
        minutes_bundle_path=minutes_bundle_path,
    )


__all__ = [
    "SimV2Profile",
    "UsageSharesConfig",
    "MinutesNoiseConfig",
    "PreSimReconcileConfig",
    "load_sim_v2_profile",
    "DEFAULT_PROFILES_PATH",
]
