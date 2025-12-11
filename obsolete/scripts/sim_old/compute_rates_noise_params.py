"""
Convert rates residual summaries into simulator noise parameters.

For each target, derive team and player noise scales from residual std and
same-team correlation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import typer

from projections.paths import data_path
from projections.rates_v1.current import get_rates_current_run_id

app = typer.Typer(add_completion=False)


def _clamp_rho(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return float(min(max(value, 0.0), 0.8))


@app.command()
def main(
    split: str = typer.Option("val", "--split", case_sensitive=False, help="Split to use: train|cal|val|all"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Optional rates run id; defaults to current rates head."),
    residuals_root: Optional[Path] = typer.Option(
        None,
        "--residuals-root",
        help="Root containing residual summaries (default: $DATA_ROOT/artifacts/rates_v1/residuals)",
    ),
    output_root: Optional[Path] = typer.Option(
        None,
        "--output-root",
        help="Output root for noise params (default: $DATA_ROOT/artifacts/sim_v2/rates_noise)",
    ),
) -> None:
    split_norm = split.lower()
    data_root = data_path()
    resolved_run_id = run_id or get_rates_current_run_id()
    residuals_base = residuals_root or (data_root / "artifacts" / "rates_v1" / "residuals")
    output_base = output_root or (data_root / "artifacts" / "sim_v2" / "rates_noise")
    output_base.mkdir(parents=True, exist_ok=True)

    residuals_path = residuals_base / f"{resolved_run_id}_{split_norm}_residuals.json"
    if not residuals_path.exists():
        raise FileNotFoundError(f"Residuals file not found at {residuals_path}")

    data = json.loads(residuals_path.read_text(encoding="utf-8"))
    targets = data.get("targets") or {}
    noise_targets: dict[str, dict[str, float | int | None]] = {}
    for name, info in targets.items():
        count = info.get("count", 0)
        std_resid = info.get("std_resid")
        rho_raw = info.get("same_team_corr")
        if std_resid is None or not math.isfinite(std_resid):
            noise_targets[name] = {
                "count": count,
                "std_resid": std_resid,
                "var_resid": None,
                "same_team_corr_raw": rho_raw,
                "same_team_corr_used": None,
                "sigma_team": None,
                "sigma_player": None,
            }
            continue
        var_resid = float(std_resid) ** 2
        rho = _clamp_rho(rho_raw)
        sigma_team2 = rho * var_resid
        sigma_player2 = max(var_resid - sigma_team2, 0.0)
        noise_targets[name] = {
            "count": int(count) if count is not None else 0,
            "std_resid": float(std_resid),
            "var_resid": var_resid,
            "same_team_corr_raw": rho_raw,
            "same_team_corr_used": rho,
            "sigma_team": math.sqrt(sigma_team2),
            "sigma_player": math.sqrt(sigma_player2),
        }

    payload = {
        "run_id": resolved_run_id,
        "split": split_norm,
        "source_residuals": str(residuals_path),
        "targets": noise_targets,
    }
    out_path = output_base / f"{resolved_run_id}_{split_norm}_noise.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(
        f"[noise] run_id={resolved_run_id} split={split_norm} targets={len(noise_targets)} "
        f"residuals={residuals_path} written={out_path}"
    )


if __name__ == "__main__":
    app()
