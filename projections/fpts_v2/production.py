"""Production loader for FPTS v2 bundles.

Thin wrapper around projections.fpts_v2.current so callers can depend on a
stable \"production\" entrypoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from projections.fpts_v2.current import load_current_fpts_bundle


def load_production_fpts_bundle(*, config_path: Optional[Path] = None, data_root: Optional[Path] = None):
    """
    Load the production FPTS bundle indicated by config/fpts_current_run.json.

    This resolves to fpts_v2_stage0_20251129_062655 by default in this repo.
    """

    return load_current_fpts_bundle(config_path=config_path, data_root=data_root)


__all__ = ["load_production_fpts_bundle"]
