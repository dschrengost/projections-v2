"""Verify live minutes_v1 feature coverage (NaN rates) for team/opponent context + dispersion.

Example:
    uv run python -m scripts.minutes.check_live_minutes_feature_coverage --game-date 2026-01-02
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from projections.paths import data_path

app = typer.Typer(add_completion=False)

TEAM_CONTEXT = ["team_pace_szn", "team_off_rtg_szn", "team_def_rtg_szn"]
OPP_CONTEXT = ["opp_pace_szn", "opp_def_rtg_szn"]
DISPERSION = ["team_minutes_dispersion_prior"]

THRESHOLDS = {
    "team_context": 0.05,
    "opp_context": 0.05,
    "dispersion": 0.0,
}


def _nan_rate(series: pd.Series) -> float:
    return float(series.isna().mean()) if not series.empty else 0.0


def _read_latest_run_id(day_dir: Path) -> str | None:
    pointer = day_dir / "latest_run.json"
    if not pointer.exists():
        return None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    run_id = payload.get("run_id")
    return str(run_id) if run_id else None


def _resolve_latest_day_dir(root: Path) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir()]
    dates = []
    for path in candidates:
        try:
            dates.append((pd.Timestamp(path.name).date(), path))
        except Exception:  # noqa: BLE001
            continue
    if not dates:
        raise FileNotFoundError(f"No dated subdirectories found under {root}")
    _, best = max(dates, key=lambda pair: pair[0])
    return best


def _resolve_run_dir(day_dir: Path, run_id: str | None) -> Path:
    resolved = run_id or _read_latest_run_id(day_dir)
    if resolved:
        return day_dir / f"run={resolved}"

    run_dirs = sorted([p for p in day_dir.glob("run=*") if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {day_dir}")
    # run_id is timestamp-like, so lexicographic max is typically newest.
    return run_dirs[-1]


@app.command()
def main(
    data_root: Path | None = typer.Option(
        None, help="Root containing data (defaults to PROJECTIONS_DATA_ROOT or ./data)."
    ),
    game_date: str | None = typer.Option(
        None, help="Slate date YYYY-MM-DD (defaults to latest dir under live/features_minutes_v1)."
    ),
    run_id: str | None = typer.Option(None, help="Run ID override (defaults to latest_run.json or newest run=*)."),
    features_path: Path | None = typer.Option(None, help="Direct path to features.parquet (overrides date/run)."),
) -> None:
    root = data_root or data_path()
    if features_path is None:
        live_root = root / "live" / "features_minutes_v1"
        if game_date is None:
            day_dir = _resolve_latest_day_dir(live_root)
        else:
            day_dir = live_root / game_date
        run_dir = _resolve_run_dir(day_dir, run_id)
        features_path = run_dir / "features.parquet"

    features_path = features_path.expanduser()
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features parquet at {features_path}")

    df = pd.read_parquet(features_path)
    if df.empty:
        typer.echo(f"[minutes-feature-coverage] {features_path}: empty parquet (FAIL)")
        raise typer.Exit(code=2)

    col_rates: dict[str, float] = {}
    for col in TEAM_CONTEXT + OPP_CONTEXT + DISPERSION:
        if col not in df.columns:
            col_rates[col] = 1.0
            continue
        col_rates[col] = _nan_rate(df[col])

    team_ok = all(col_rates[c] <= THRESHOLDS["team_context"] for c in TEAM_CONTEXT)
    opp_ok = all(col_rates[c] <= THRESHOLDS["opp_context"] for c in OPP_CONTEXT)
    disp_ok = all(col_rates[c] <= THRESHOLDS["dispersion"] for c in DISPERSION)
    passed = bool(team_ok and opp_ok and disp_ok)

    typer.echo(f"[minutes-feature-coverage] path={features_path}")
    for col in TEAM_CONTEXT + OPP_CONTEXT + DISPERSION:
        rate = col_rates.get(col, 1.0)
        typer.echo(f"  {col}: nan_rate={rate:.3f}")
    typer.echo("PASS" if passed else "FAIL")
    raise typer.Exit(code=0 if passed else 1)


if __name__ == "__main__":
    app()

