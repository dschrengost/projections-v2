"""Utilities for storing multiple minutes model outputs side-by-side.

This is intentionally additive: production ("prod") artifacts and APIs remain
unchanged. Non-prod model outputs are written under a model-partitioned root
and can be selected by the dashboard via `model_id`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from projections import paths

MODEL_ID_PROD = "prod"
MODEL_ID_RMH_V1_1 = "rmh_v1_1"

OUTPUT_FILENAME = "minutes.parquet"
SUMMARY_FILENAME = "summary.json"


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_model_id(model_id: str | None) -> str:
    raw = (model_id or "").strip()
    if not raw:
        return MODEL_ID_PROD
    value = raw.lower()
    aliases = {
        "production": MODEL_ID_PROD,
        "prod": MODEL_ID_PROD,
        "rmh": MODEL_ID_RMH_V1_1,
        "rmh_v1.1": MODEL_ID_RMH_V1_1,
        "rmh-v1.1": MODEL_ID_RMH_V1_1,
        "rmh_v1_1": MODEL_ID_RMH_V1_1,
    }
    return aliases.get(value, value)


def minutes_models_root(data_root: Path | None = None) -> Path:
    root = Path(data_root) if data_root is not None else paths.get_data_root()
    return root / "artifacts" / "minutes_models" / "daily"


def model_root(*, data_root: Path, model_id: str) -> Path:
    model_id = normalize_model_id(model_id)
    if model_id == MODEL_ID_PROD:
        raise ValueError("prod minutes are not stored under minutes_models_root")
    return minutes_models_root(data_root) / f"model_id={model_id}"


def model_day_dir(*, data_root: Path, model_id: str, game_date: str) -> Path:
    return model_root(data_root=data_root, model_id=model_id) / str(game_date)


def model_has_any_runs(*, data_root: Path, model_id: str, game_date: str | None = None) -> bool:
    try:
        base = model_root(data_root=data_root, model_id=model_id)
    except ValueError:
        return True
    if game_date is not None:
        base = base / str(game_date)
    if not base.exists():
        return False
    return any(p.is_dir() and p.name.startswith("run=") for p in base.rglob("run=*"))


@dataclass(frozen=True, slots=True)
class MinutesModelDescriptor:
    model_id: str
    label: str
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"model_id": self.model_id, "label": self.label}
        if self.meta:
            payload["meta"] = self.meta
        return payload


def write_minutes_model_outputs(
    *,
    data_root: Path,
    game_date: str,
    run_id: str,
    model_id: str,
    model_label: str,
    df: pd.DataFrame,
    run_as_of_ts: str | None = None,
    model_meta: dict[str, Any] | None = None,
    extra_summary: dict[str, Any] | None = None,
) -> Path:
    """Write minutes outputs for a non-prod model under a model-partitioned root.

    Notes:
    - Does NOT write latest pointers; Prefect flow promotes atomically at the end.
    - Intended for shadow models (e.g. RMH v1.1) only.
    """
    model_id = normalize_model_id(model_id)
    if model_id == MODEL_ID_PROD:
        raise ValueError("Refusing to write prod minutes under minutes_models_root")

    day_dir = model_day_dir(data_root=data_root, model_id=model_id, game_date=game_date)
    run_dir = day_dir / f"run={run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    out_path = run_dir / OUTPUT_FILENAME
    df.to_parquet(out_path, index=False)

    summary: dict[str, Any] = {
        "date": str(game_date),
        "generated_at": _utc_now_iso(),
        "run_id": str(run_id),
        "run_as_of_ts": str(run_as_of_ts) if run_as_of_ts else None,
        "model_id": model_id,
        "model_label": model_label,
        "counts": {
            "rows": int(len(df)),
            "players": int(df["player_id"].nunique()) if (not df.empty and "player_id" in df.columns) else 0,
            "teams": int(df["team_id"].nunique()) if (not df.empty and "team_id" in df.columns) else 0,
        },
        "model_meta": model_meta or {},
    }
    if extra_summary:
        summary.update(extra_summary)
    (run_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path

