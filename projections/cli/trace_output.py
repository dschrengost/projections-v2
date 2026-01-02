"""Trace lineage for a pipeline output.

Given a gold (or artifact) output path, print:
- producing run_id
- manifest.json (if present)
- minutes/rates model bundle run_ids from current_run.json selectors
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from projections import paths
from projections.pipeline import control_plane

app = typer.Typer(add_completion=False)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON at {path}")
    return payload


def _find_run_dir(start: Path) -> tuple[Path | None, str | None]:
    """Return (run_dir, run_id) if path is within a run=<id> directory."""
    for parent in [start, *start.parents]:
        if parent.name.startswith("run="):
            run_id = parent.name.split("=", 1)[1]
            return parent, run_id
    return None, None


def _resolve_pointer_run_id(dir_path: Path) -> tuple[str | None, Path | None, dict[str, Any] | None]:
    """Return (run_id, run_dir, pointer_payload) from LATEST/current.json or latest_run.json."""
    for pointer in (dir_path / control_plane.LATEST_DIRNAME / control_plane.CURRENT_POINTER_NAME, dir_path / "latest_run.json"):
        if not pointer.exists():
            continue
        try:
            payload = _read_json(pointer)
        except Exception:
            continue
        run_id = payload.get("run_id")
        if not run_id:
            continue
        run_dir = dir_path / f"run={run_id}"
        return str(run_id), run_dir if run_dir.exists() else None, payload
    return None, None, None


def _read_selector_run_id(selector_path: Path) -> str | None:
    try:
        payload = _read_json(selector_path)
    except Exception:
        return None
    run_id = payload.get("run_id")
    return str(run_id) if run_id else None


@app.command()
def main(
    target: Path = typer.Argument(..., exists=True, help="Output file or directory to trace."),
    data_root: Path | None = typer.Option(None, "--data-root", help="Optional override for data root."),
) -> None:
    root = data_root or paths.get_data_root()
    typer.echo(f"[lineage] data_root={root}")

    target_path = target.expanduser().resolve()
    if target_path.is_file():
        target_path = target_path.parent

    run_dir, run_id = _find_run_dir(target_path)

    pointer_payload: dict[str, Any] | None = None
    if run_id is None:
        run_id, run_dir, pointer_payload = _resolve_pointer_run_id(target_path)
        if run_id is None and target_path.parent != target_path:
            run_id, run_dir, pointer_payload = _resolve_pointer_run_id(target_path.parent)

    if run_id is None:
        raise typer.BadParameter(f"Unable to resolve run_id from {target}")

    typer.echo(f"[lineage] run_id={run_id}")
    if run_dir is not None:
        typer.echo(f"[lineage] run_dir={run_dir}")
    if pointer_payload is not None:
        typer.echo(f"[lineage] pointer={json.dumps(pointer_payload, indent=2, sort_keys=True)}")

    manifest_path: Path | None = None
    if run_dir is not None and (run_dir / control_plane.RUN_MANIFEST_NAME).exists():
        manifest_path = run_dir / control_plane.RUN_MANIFEST_NAME
    elif pointer_payload and pointer_payload.get("manifest_path"):
        candidate = Path(str(pointer_payload["manifest_path"]))
        if candidate.exists():
            manifest_path = candidate

    if manifest_path is None:
        typer.echo("[lineage] manifest=NOT_FOUND")
        raise typer.Exit(code=2)

    manifest = _read_json(manifest_path)
    typer.echo(f"[lineage] manifest_path={manifest_path}")
    typer.echo(f"[lineage] manifest={json.dumps(manifest, indent=2, sort_keys=True)}")

    summary_path = None
    if run_dir is not None and (run_dir / "summary.json").exists():
        summary_path = run_dir / "summary.json"
    if summary_path is not None:
        summary = _read_json(summary_path)
        typer.echo(f"[lineage] summary_path={summary_path}")
        typer.echo(f"[lineage] summary={json.dumps(summary, indent=2, sort_keys=True)}")

    minutes_selector = Path(str(manifest.get("minutes_current_run_path", "config/minutes_current_run.json")))
    rates_selector = Path(str(manifest.get("rates_current_run_path", "config/rates_current_run.json")))
    minutes_model_run = _read_selector_run_id(minutes_selector)
    rates_model_run = _read_selector_run_id(rates_selector)
    typer.echo(f"[inputs] minutes_current_run={minutes_selector} run_id={minutes_model_run}")
    typer.echo(f"[inputs] rates_current_run={rates_selector} run_id={rates_model_run}")

    # Upstream pipeline run manifests (when present in summary.json)
    game_date = str(manifest.get("game_date") or "")
    if summary_path is not None and game_date:
        summary = _read_json(summary_path)
        minutes_run_id = summary.get("minutes_run_id")
        rates_run_id = summary.get("rates_run_id")
        if minutes_run_id:
            minutes_run_dir = root / "artifacts" / "minutes_v1" / "daily" / game_date / f"run={minutes_run_id}"
            minutes_manifest = minutes_run_dir / control_plane.RUN_MANIFEST_NAME
            if minutes_manifest.exists():
                typer.echo(f"[upstream] minutes_manifest_path={minutes_manifest}")
                typer.echo(f"[upstream] minutes_manifest={json.dumps(_read_json(minutes_manifest), indent=2, sort_keys=True)}")
        if rates_run_id:
            rates_run_dir = root / "gold" / "rates_v1_live" / game_date / f"run={rates_run_id}"
            rates_manifest = rates_run_dir / control_plane.RUN_MANIFEST_NAME
            if rates_manifest.exists():
                typer.echo(f"[upstream] rates_manifest_path={rates_manifest}")
                typer.echo(f"[upstream] rates_manifest={json.dumps(_read_json(rates_manifest), indent=2, sort_keys=True)}")


if __name__ == "__main__":
    app()
