"""Production loader for rates_v1 bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from projections.rates_v1.current import get_rates_current_run_id
from projections.rates_v1.loader import load_rates_bundle, RatesBundle


def load_production_rates_bundle(*, config_path: Optional[Path] = None) -> RatesBundle:
    """
    Load the production rates_v1 bundle indicated by config/rates_current_run.json.
    """

    run_id = get_rates_current_run_id(config_path=config_path)
    return load_rates_bundle(run_id)


__all__ = ["load_production_rates_bundle"]
