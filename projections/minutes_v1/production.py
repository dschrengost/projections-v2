"""Helpers for loading the production minutes model bundle."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib

DEFAULT_PRODUCTION_RUN_ID = "lgbm_full_v1_no_p_play_20251202"
DEFAULT_PRODUCTION_ROOT = Path("artifacts/minutes_lgbm")
DEFAULT_PRODUCTION_CONFIG = Path("config/minutes_current_run.json")
DEFAULT_LATE_THRESHOLD_MIN = 60.0
DEFAULT_BLEND_BAND_MIN = 30.0

ENV_RUN_ID = "MINUTES_PRODUCTION_RUN_ID"
ENV_RUN_DIR = "MINUTES_PRODUCTION_DIR"
ENV_CONFIG_PATH = "MINUTES_PRODUCTION_CONFIG"


def _expand(path: Path) -> Path:
    return path.expanduser().resolve()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - config errors are fatal
        raise RuntimeError(f"Invalid production config JSON at {path}: {exc}") from exc


def _resolve_dir(raw: str | Path, *, base: Path | None = None) -> Path:
    value = Path(raw)
    if value.is_absolute():
        return _expand(value)
    base_dir = _expand(base) if base is not None else _expand(Path.cwd())
    return _expand(base_dir / value)


def _load_bundle_from_dir(run_dir: Path, run_id: str | None) -> dict[str, Any]:
    bundle_path = run_dir / "lgbm_quantiles.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Production bundle missing at {bundle_path}")
    bundle = joblib.load(bundle_path)
    meta_path = run_dir / "meta.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    bundle.setdefault("run_dir", str(run_dir))
    bundle.setdefault("run_id", run_id or run_dir.name)
    bundle.setdefault("meta", meta)
    return bundle


def resolve_production_run_dir(config_path: Path | None = None) -> tuple[Path, str | None]:
    """Return the filesystem path that houses the production bundle."""

    env_dir = os.environ.get(ENV_RUN_DIR)
    env_run = os.environ.get(ENV_RUN_ID)
    if env_dir:
        run_dir = _expand(Path(env_dir))
        return run_dir, env_run or run_dir.name

    config_candidate = os.environ.get(ENV_CONFIG_PATH)
    config_file = Path(config_candidate).expanduser() if config_candidate else (config_path or DEFAULT_PRODUCTION_CONFIG)
    config_file = _expand(config_file)
    if config_file.exists():
        payload = _load_json(config_file)
        mode = str(payload.get("mode") or "single").strip().lower()
        if mode == "dual":
            late_run_id = payload.get("late_run_id") or payload.get("late_run") or payload.get("run_id")
            if not late_run_id:
                raise RuntimeError(f"Production config {config_file} missing 'late_run_id'.")
            late_bundle_dir = payload.get("late_bundle_dir")
            if late_bundle_dir:
                resolved = _resolve_dir(late_bundle_dir)
            else:
                artifact_root = payload.get("artifact_root") or DEFAULT_PRODUCTION_ROOT
                resolved = _resolve_dir(artifact_root) / str(late_run_id)
            return resolved, str(late_run_id)

        bundle_dir = payload.get("bundle_dir")
        run_id = payload.get("run_id") or env_run
        if bundle_dir:
            resolved = _resolve_dir(bundle_dir)
            return resolved, str(run_id) if run_id is not None else resolved.name

        # Backward compatibility: allow configs that only specify run_id + optional artifact_root.
        if run_id is not None:
            artifact_root = payload.get("artifact_root") or DEFAULT_PRODUCTION_ROOT
            resolved = _resolve_dir(artifact_root) / str(run_id)
            return resolved, str(run_id)
        raise RuntimeError(f"Production config {config_file} must set either 'bundle_dir' or 'run_id'.")

    # Final fallback: assume default root/run-id relative to repo.
    root = _expand(DEFAULT_PRODUCTION_ROOT)
    run_id = env_run or DEFAULT_PRODUCTION_RUN_ID
    return root / run_id, run_id


@lru_cache(maxsize=1)
def load_production_minutes_bundle(
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Load the production minutes model bundle(s).

    Returns either:
      - a legacy single-bundle payload (backward compatible), or
      - a dual-bundle payload when `config/minutes_current_run.json` is in dual mode.
    """

    env_dir = os.environ.get(ENV_RUN_DIR)
    env_run = os.environ.get(ENV_RUN_ID)
    if env_dir:
        run_dir = _expand(Path(env_dir))
        return _load_bundle_from_dir(run_dir, env_run or run_dir.name)

    config_candidate = os.environ.get(ENV_CONFIG_PATH)
    config_file = Path(config_candidate).expanduser() if config_candidate else (config_path or DEFAULT_PRODUCTION_CONFIG)
    config_file = _expand(config_file)
    if config_file.exists():
        payload = _load_json(config_file)
        mode: Literal["single", "dual"] = str(payload.get("mode") or "single").strip().lower()  # type: ignore[assignment]

        if mode == "dual":
            early_run_id = payload.get("early_run_id")
            late_run_id = payload.get("late_run_id")
            if not early_run_id or not late_run_id:
                raise RuntimeError(f"Production config {config_file} missing early_run_id/late_run_id.")

            artifact_root = payload.get("artifact_root") or DEFAULT_PRODUCTION_ROOT
            artifact_root_path = _resolve_dir(artifact_root)

            early_dir_raw = payload.get("early_bundle_dir") or (artifact_root_path / str(early_run_id))
            late_dir_raw = payload.get("late_bundle_dir") or (artifact_root_path / str(late_run_id))
            early_dir = _resolve_dir(early_dir_raw, base=Path.cwd())
            late_dir = _resolve_dir(late_dir_raw, base=Path.cwd())

            early_bundle = _load_bundle_from_dir(early_dir, str(early_run_id))
            late_bundle = _load_bundle_from_dir(late_dir, str(late_run_id))

            return {
                "mode": "dual",
                "early_run_id": str(early_run_id),
                "late_run_id": str(late_run_id),
                "early_bundle_dir": str(early_dir),
                "late_bundle_dir": str(late_dir),
                "late_threshold_min": float(payload.get("late_threshold_min", DEFAULT_LATE_THRESHOLD_MIN)),
                "blend_band_min": float(payload.get("blend_band_min", DEFAULT_BLEND_BAND_MIN)),
                "early_bundle": early_bundle,
                "late_bundle": late_bundle,
            }

        bundle_dir = payload.get("bundle_dir")
        run_id = payload.get("run_id") or env_run
        if bundle_dir:
            run_dir = _resolve_dir(bundle_dir)
            return _load_bundle_from_dir(run_dir, str(run_id) if run_id is not None else run_dir.name)

        # Backward compatibility: allow configs that only specify run_id + optional artifact_root.
        if run_id is not None:
            artifact_root = payload.get("artifact_root") or DEFAULT_PRODUCTION_ROOT
            run_dir = _resolve_dir(artifact_root) / str(run_id)
            return _load_bundle_from_dir(run_dir, str(run_id))
        raise RuntimeError(f"Production config {config_file} must set either 'bundle_dir' or 'run_id'.")

    # Final fallback: assume default root/run-id relative to repo.
    root = _expand(DEFAULT_PRODUCTION_ROOT)
    run_id = env_run or DEFAULT_PRODUCTION_RUN_ID
    return _load_bundle_from_dir(root / run_id, run_id)
