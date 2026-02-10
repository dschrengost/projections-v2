"""Helpers for resolving the current production rates_v1 bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from projections.model_selectors import active_rates_selector_path
from projections.rates_v1.loader import RatesBundle, load_rates_bundle


def get_rates_current_run_id(config_path: Optional[Path] = None) -> str:
    """
    Load the current production rates_v1 run_id from the active selector config.

    Raises RuntimeError if the config is missing or malformed.
    """

    config_file = (config_path or active_rates_selector_path()).expanduser().resolve()
    if not config_file.exists():
        raise RuntimeError(f"rates_current_run.json not found at {config_file}")
    try:
        payload = json.loads(config_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {config_file}: {exc}") from exc
    run_id = payload.get("run_id")
    if not run_id:
        raise RuntimeError(f"rates_current_run.json missing 'run_id' at {config_file}")
    return str(run_id)


def load_current_rates_bundle(config_path: Optional[Path] = None) -> RatesBundle:
    """
    Convenience helper: reads the current run_id from the active rates selector
    and returns the corresponding RatesBundle.
    """

    run_id = get_rates_current_run_id(config_path=config_path)
    return load_rates_bundle(run_id)
