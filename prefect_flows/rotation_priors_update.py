"""Prefect flow to build rotation_v1 and rotation_priors_v1.

This runs after the daily gamerotation scrape to refresh silver rotation data
and derived priors used by the rotation overlay minutes model.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from prefect import flow, get_run_logger, task

from projections import paths

PROJECT_ROOT = paths.get_project_root()


def _run_python_script(
    script_rel_path: str,
    args: list[str],
    *,
    data_root: Path,
    timeout_s: int,
) -> None:
    env = os.environ.copy()
    env["PROJECTIONS_DATA_ROOT"] = str(data_root)
    cmd = [sys.executable, str(PROJECT_ROOT / script_rel_path), *args]
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{script_rel_path} failed exit_code={result.returncode}")


@task(name="build-rotation-v1", retries=1, retry_delay_seconds=120)
def build_rotation_v1_task(
    *, data_root: Path, overwrite: bool, retry_quarantined: bool
) -> None:
    args: list[str] = []
    if overwrite:
        args.append("--overwrite")
    if retry_quarantined:
        args.append("--retry-quarantined")
    _run_python_script(
        "scripts/rotation/build_rotation_dataset_v1.py",
        args,
        data_root=data_root,
        timeout_s=3600,
    )


@task(name="build-rotation-priors-v1", retries=1, retry_delay_seconds=120)
def build_rotation_priors_v1_task(
    *,
    data_root: Path,
    priors_windows: list[int] | None,
    clean_priors: bool,
) -> None:
    args: list[str] = []
    if priors_windows:
        for w in priors_windows:
            args.extend(["--window", str(int(w))])
    if clean_priors:
        args.append("--clean")
        args.append("--overwrite")
    _run_python_script(
        "scripts/rotation/build_rotation_priors_v1.py",
        args,
        data_root=data_root,
        timeout_s=3600,
    )


@flow(name="rotation-priors-update", log_prints=True)
def rotation_priors_update_flow(
    *,
    overwrite_rotation_v1: bool = False,
    retry_quarantined: bool = True,
    priors_windows: list[int] | None = None,
    clean_priors: bool = False,
) -> dict[str, str]:
    logger = get_run_logger()
    data_root = paths.get_data_root()
    logger.info(
        "[rotation-priors-update] data_root=%s overwrite_rotation_v1=%s retry_quarantined=%s clean_priors=%s priors_windows=%s",
        data_root,
        overwrite_rotation_v1,
        retry_quarantined,
        clean_priors,
        priors_windows,
    )

    build_rotation_v1_task(
        data_root=data_root,
        overwrite=overwrite_rotation_v1,
        retry_quarantined=retry_quarantined,
    )
    build_rotation_priors_v1_task(
        data_root=data_root, priors_windows=priors_windows, clean_priors=clean_priors
    )
    return {"status": "ok"}
