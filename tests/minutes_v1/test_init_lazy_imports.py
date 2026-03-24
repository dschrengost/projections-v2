from __future__ import annotations

import importlib
import sys


def test_minutes_v1_import_does_not_eager_import_lightgbm() -> None:
    sys.modules.pop("projections.minutes_v1", None)
    sys.modules.pop("lightgbm", None)

    minutes_v1 = importlib.import_module("projections.minutes_v1")

    assert "lightgbm" not in sys.modules

    # Access a non-modeling export to verify lazy exports still work
    # without pulling in LightGBM transitively.
    _ = minutes_v1.ensure_as_of_column

    assert "lightgbm" not in sys.modules
