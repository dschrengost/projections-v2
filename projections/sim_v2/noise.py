from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from projections.paths import data_path
from projections.rates_v1.current import get_rates_current_run_id

logger = logging.getLogger(__name__)


def load_rates_noise_params(
    data_root: Optional[Path] = None,
    run_id: Optional[str] = None,
    split: str = "val",
    sigma_scale: float = 1.0,
) -> Tuple[Dict[str, Dict[str, float]], Path]:
    """
    Load rates noise parameters for the given run/split.

    Returns a mapping: target -> {sigma_team, sigma_player, ...}
    """

    root = data_root or data_path()
    resolved_run_id = run_id or get_rates_current_run_id()
    path = root / "artifacts" / "sim_v2" / "rates_noise" / f"{resolved_run_id}_{split}_noise.json"
    if not path.exists():
        raise FileNotFoundError(f"Noise params not found at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets = payload.get("targets") or {}

    if sigma_scale != 1.0:
        logger.info(
            "[sim_v2] rates_noise: applying sigma_scale=%.3f to team/player sigmas",
            sigma_scale,
        )
    for params in targets.values():
        if "sigma_team" in params:
            params["sigma_team"] = float(params["sigma_team"]) * sigma_scale
        if "sigma_player" in params:
            params["sigma_player"] = float(params["sigma_player"]) * sigma_scale

    return targets, path


__all__ = ["load_rates_noise_params"]
