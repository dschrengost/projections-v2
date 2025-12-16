"""Convert minutes residual buckets into noise parameters for the simulator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from projections.minutes_v1.production import resolve_production_run_dir
from projections.paths import data_path
from projections.sim_v2.minutes_noise import MINUTES_P50_BIN_EDGES

app = typer.Typer(add_completion=False)


def _resolve_run_id(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    _, run_id = resolve_production_run_dir()
    if not run_id:
        raise RuntimeError("Unable to resolve minutes run_id from config/minutes_current_run.json")
    return str(run_id)


@app.command()
def main(
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Minutes run id (defaults to current minutes head)."),
    data_root: Optional[Path] = typer.Option(None, "--data-root", help="Root for data (defaults PROJECTIONS_DATA_ROOT)."),
    sigma_min: float = typer.Option(1.0, "--sigma-min", help="Minimum sigma floor applied to each bucket."),
) -> None:
    root = data_root or data_path()
    resolved_run = _resolve_run_id(run_id)
    residuals_path = root / "artifacts" / "minutes_v1" / "residuals" / f"{resolved_run}_minutes_residuals.json"
    if not residuals_path.exists():
        raise FileNotFoundError(f"Residuals file not found at {residuals_path}")

    data = json.loads(residuals_path.read_text(encoding="utf-8"))
    buckets_raw = data.get("buckets") or []
    bin_edges = data.get("bin_edges") or list(MINUTES_P50_BIN_EDGES)

    buckets: list[dict[str, object]] = []
    for entry in buckets_raw:
        std_r = entry.get("std_r")
        sigma = float(max(std_r, sigma_min)) if std_r is not None else float(sigma_min)
        buckets.append(
            {
                "starter_flag": int(entry.get("starter_flag", 0)),
                "status_bucket": str(entry.get("status_bucket", "unknown")),
                "p50_bin_idx": int(entry.get("p50_bin_idx", 0)),
                "p50_bin": entry.get("p50_bin"),
                "count": int(entry.get("count", 0)),
                "sigma": sigma,
            }
        )

    payload = {
        "run_id": resolved_run,
        "sigma_min": float(sigma_min),
        "source_residuals": str(residuals_path),
        "bin_edges": bin_edges,
        "buckets": buckets,
    }

    out_root = root / "artifacts" / "sim_v2" / "minutes_noise"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{resolved_run}_minutes_noise.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    typer.echo(
        f"[minutes_noise] run_id={resolved_run} buckets={len(buckets)} sigma_min={sigma_min} "
        f"residuals={residuals_path} written={out_path}"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
