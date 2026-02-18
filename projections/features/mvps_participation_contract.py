"""MVPS participation live feature contract helpers.

The MVPS participation LightGBM model expects a fixed set of *post-preprocessing*
feature columns (one-hot expanded, numeric-only) captured in
`feature_columns.json` inside the MVPS model artifact directory.

This module provides:
  - loading the required column list
  - identifying one-hot columns by naming convention
  - completing the contract by filling missing one-hot columns with zeros
  - failing loudly if any required non-one-hot (numeric) columns are missing
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

ONEHOT_PREFIXES: tuple[str, ...] = (
    "archetype_",
    "lineup_role_",
    "lineup_status_",
    "lineup_roster_status_",
    "pos_bucket_",
    "snapshot_type_",
    "status_",
    "team_id_",
    "team_tricode_",
)


def load_feature_columns(path: str | Path) -> list[str]:
    """Load MVPS `feature_columns.json`.

    The file is expected to be either:
      - a JSON list[str], or
      - a JSON object with a top-level "columns": list[str]
    """

    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(payload, list):
        cols = payload
    elif isinstance(payload, dict) and isinstance(payload.get("columns"), list):
        cols = payload["columns"]
    else:
        raise ValueError(f"Invalid feature_columns.json at {path}: expected list or {{'columns': list}}")

    out = [str(c) for c in cols if str(c).strip()]
    if not out:
        raise ValueError(f"Invalid feature_columns.json at {path}: empty column list")
    return out


def is_onehot_col(name: str) -> bool:
    return any(str(name).startswith(prefix) for prefix in ONEHOT_PREFIXES)


def required_onehot_groups(required_cols: Iterable[str]) -> dict[str, list[str]]:
    """Return mapping raw_col -> required one-hot columns.

    Example: {"status": ["status_Ava", "status_OUT", ...], "team_tricode": [...]}.
    """

    groups: dict[str, list[str]] = {}
    for col in required_cols:
        text = str(col)
        for prefix in ONEHOT_PREFIXES:
            if not text.startswith(prefix):
                continue
            raw = prefix[:-1]
            groups.setdefault(raw, []).append(text)
            break
    return groups


def add_required_onehots_from_raw(df: pd.DataFrame, *, required_cols: Iterable[str]) -> pd.DataFrame:
    """Add required one-hot columns based on raw categorical columns.

    This is intentionally conservative: it *only* creates one-hot columns that
    are present in `required_cols` and does not attempt to learn new categories.
    """

    out = df.copy()
    groups = required_onehot_groups(required_cols)
    if not groups or out.empty:
        return out

    for raw_col, onehot_cols in groups.items():
        if raw_col not in out.columns:
            continue

        raw = out[raw_col]
        cats = [c[len(raw_col) + 1 :] for c in onehot_cols if c.startswith(f"{raw_col}_")]
        missing_token = "<NA>" if "<NA>" in cats else ("nan" if "nan" in cats else None)

        raw_str = raw.astype(str)
        if missing_token is not None:
            raw_str = raw_str.where(~raw.isna(), other=missing_token)

        prefix = f"{raw_col}_"
        for onehot in onehot_cols:
            if onehot in out.columns:
                continue
            if not onehot.startswith(prefix):
                continue
            cat = onehot[len(prefix) :]
            out[onehot] = (raw_str == cat).astype("int8")

    return out


@dataclass(frozen=True)
class ContractReport:
    required: list[str]
    missing_required: list[str]
    missing_onehot_filled: list[str]
    missing_non_onehot: list[str]
    extra_columns: list[str]


def complete_and_validate_contract(
    df: pd.DataFrame,
    *,
    required_cols: list[str],
    key_cols: tuple[str, ...] = ("game_id", "team_id", "player_id"),
    timestamp_cols: tuple[str, ...] = ("feature_as_of_ts", "tip_ts"),
    coerce_required_to_float32: bool = True,
) -> tuple[pd.DataFrame, ContractReport]:
    """Ensure df contains all required MVPS feature columns.

    Rules:
      - Missing required one-hot columns (by prefix convention) are added as 0.0.
      - Missing required non-one-hot columns raise ValueError.
      - Required feature columns must be numeric/bool (after coercion).
    """

    required_set = set(required_cols)
    present = set(df.columns)
    missing = sorted(required_set - present)

    missing_onehot = [c for c in missing if is_onehot_col(c)]
    missing_non_onehot = [c for c in missing if not is_onehot_col(c)]
    if missing_non_onehot:
        raise ValueError(
            "MVPS participation feature contract missing required non-one-hot columns: "
            + ", ".join(missing_non_onehot)
        )

    out = df.copy()
    for col in missing_onehot:
        out[col] = 0.0

    for col in required_cols:
        series = out[col]
        if pd.api.types.is_bool_dtype(series):
            out[col] = series.fillna(False).astype("int8")
            continue
        if pd.api.types.is_numeric_dtype(series):
            if coerce_required_to_float32:
                out[col] = pd.to_numeric(series, errors="coerce").astype("float32")
            else:
                out[col] = pd.to_numeric(series, errors="coerce")
            continue
        raise TypeError(f"Required feature column {col!r} has non-numeric dtype: {series.dtype}")

    ordered: list[str] = []
    for col in key_cols:
        if col in out.columns:
            ordered.append(col)
    for col in timestamp_cols:
        if col in out.columns:
            ordered.append(col)
    ordered.extend(required_cols)

    keep_set = set(ordered)
    extra = sorted(c for c in out.columns if c not in keep_set)
    out = out.loc[:, ordered].copy()

    report = ContractReport(
        required=list(required_cols),
        missing_required=missing,
        missing_onehot_filled=missing_onehot,
        missing_non_onehot=missing_non_onehot,
        extra_columns=extra,
    )
    return out, report


__all__ = [
    "ContractReport",
    "ONEHOT_PREFIXES",
    "add_required_onehots_from_raw",
    "complete_and_validate_contract",
    "is_onehot_col",
    "load_feature_columns",
    "required_onehot_groups",
]
