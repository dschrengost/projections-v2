"""Rotation Minutes Hurdle Model v1 (RMH_v1).

This package implements a two-head (hurdle) minutes model:
- Head A: play probability p_play (minutes >= T)
- Head B: conditional minutes quantiles (q10/q50/q90) trained only on played rows

See: docs/models/rotation_minutes_hurdle_v1.md
"""

from .bundle import RMHBundle, load_bundle, save_bundle
from .config import RMHConfig, load_config
from .infer import predict_frame

__all__ = [
    "RMHBundle",
    "RMHConfig",
    "load_bundle",
    "load_config",
    "predict_frame",
    "save_bundle",
]

