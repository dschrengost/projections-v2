"""Helpers for loading the production minutes model bundle."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

DEFAULT_PRODUCTION_RUN_ID = "lgbm_full_v1_no_p_play_20251202"
DEFAULT_PRODUCTION_ROOT = Path("artifacts/minutes_lgbm")
DEFAULT_PRODUCTION_CONFIG = Path("config/minutes_current_run.json")

ENV_RUN_ID = "MINUTES_PRODUCTION_RUN_ID"
ENV_RUN_DIR = "MINUTES_PRODUCTION_DIR"
ENV_CONFIG_PATH = "MINUTES_PRODUCTION_CONFIG"


def _expand(path: Path) -> Path:
    return path.expanduser().resolve()


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
        try:
            payload = json.loads(config_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - config errors are fatal
            raise RuntimeError(f"Invalid production config JSON at {config_file}: {exc}") from exc
        bundle_dir = payload.get("bundle_dir")
        if not bundle_dir:
            raise RuntimeError(f"Production config {config_file} missing 'bundle_dir'.")
        run_id = payload.get("run_id") or env_run
        resolved = Path(bundle_dir)
        if not resolved.is_absolute():
            resolved = _expand(Path.cwd() / resolved)
        else:
            resolved = _expand(resolved)
        return resolved, run_id

    # Final fallback: assume default root/run-id relative to repo.
    root = _expand(DEFAULT_PRODUCTION_ROOT)
    run_id = env_run or DEFAULT_PRODUCTION_RUN_ID
    return root / run_id, run_id


@lru_cache(maxsize=1)
def load_production_minutes_bundle(
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Load the production LightGBM bundle (quantiles + calibrators)."""

    run_dir, run_id = resolve_production_run_dir(config_path)
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
    bundle.setdefault("run_id", run_id)
    bundle.setdefault("meta", meta)
    return bundle
