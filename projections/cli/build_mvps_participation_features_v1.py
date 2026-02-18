"""Build live MVPS participation features (v1) with strict train-live parity.

Why this exists
---------------
The MVPS participation model (MVPS repo) was trained on a fixed set of
post-preprocessing feature columns (one-hot expanded) captured in:
  MVPS artifacts/participation_model/mvps_part_v1/feature_columns.json

The existing live `rotation_set_minutes_v1` feature product is a different
contract and does not include many of the participation model's required
features, which collapses `p_rotation` and causes threshold selection to fail.

This CLI produces a dedicated live feature product that:
  - starts from the canonical Minutes V1 live features (pregame/leakage-safe)
  - joins rotation_priors_v1 (prior_5/10/20 windows)
  - adds pregame derived rotation/team context features (depth, bench conc, etc.)
  - performs the MVPS model feature contract completion (fill missing one-hots)
  - fails loudly if any required non-one-hot/numeric features are missing
  - enforces anti-leak guard: feature_as_of_ts <= tip_ts per row

Output
------
  <DATA_ROOT>/live/features_mvps_participation_v1/<YYYY-MM-DD>/run=<run_id>/
    - features.parquet
    - run_meta.json
    - (optional) latest_run.json pointer (writer-guarded)
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from projections import paths
from projections.features.mvps_participation_contract import (
    add_required_onehots_from_raw,
    complete_and_validate_contract,
    load_feature_columns,
)
from projections.models.rotation_minutes_hurdle_v1.live_features import (
    prepare_live_features_for_rmh,
)
from projections.rotation.live_features_v1 import load_rotation_priors_for_live_inference
from projections.rotation.rotation_set_minutes_features_v1 import (
    apply_odds_missing_flags,
    join_rotation_priors,
)

UTC = timezone.utc

app = typer.Typer(help=__doc__)

DEFAULT_DATA_ROOT = paths.get_data_root()
DEFAULT_OUTPUT_ROOT = paths.data_path("live", "features_mvps_participation_v1")
DEFAULT_FEATURE_COLUMNS_PATH = Path(
    "/home/daniel/projects/MVPS/artifacts/participation_model/mvps_part_v1/feature_columns.json"
)

FEATURE_FILENAME = "features.parquet"
RUN_META_FILENAME = "run_meta.json"
LATEST_POINTER = "latest_run.json"


def _normalize_day(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _season_for_day(day: pd.Timestamp) -> int:
    return int(day.year) if int(day.month) >= 8 else int(day.year) - 1


def _read_latest_run_id(day_dir: Path) -> str | None:
    pointer = day_dir / LATEST_POINTER
    if not pointer.exists():
        return None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    run_id = payload.get("run_id")
    return str(run_id) if run_id else None


def _resolve_run_id(day_dir: Path, run_id: str | None) -> str:
    if run_id:
        return str(run_id)
    latest = _read_latest_run_id(day_dir)
    if latest:
        return latest
    raise typer.BadParameter(f"--run-id is required (no {LATEST_POINTER} under {day_dir})", param_hint="run_id")


def _ensure_run_output_dir(root: Path, day: pd.Timestamp, run_id: str) -> tuple[Path, Path]:
    day_dir = root / day.strftime("%Y-%m-%d")
    run_dir = day_dir / f"run={run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return day_dir, run_dir


def _write_latest_pointer(day_dir: Path, *, run_id: str, run_as_of_ts: pd.Timestamp) -> None:
    if os.environ.get("PROJECTIONS_SKIP_POINTER_WRITES", "").strip().lower() in {"1", "true", "yes"}:
        return
    from projections.pipeline import writer_guard

    writer_guard.assert_can_write_pointers(purpose=f"build_mvps_participation_features_v1 promote {day_dir}")
    pointer = day_dir / LATEST_POINTER
    payload = {"run_id": run_id, "run_as_of_ts": run_as_of_ts.isoformat()}
    pointer.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_columns(df: pd.DataFrame, cols: list[str], *, label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _enforce_asof_guard(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _require_columns(out, ["feature_as_of_ts", "tip_ts"], label="as-of guard")
    out["feature_as_of_ts"] = pd.to_datetime(out["feature_as_of_ts"], utc=True, errors="coerce")
    out["tip_ts"] = pd.to_datetime(out["tip_ts"], utc=True, errors="coerce")

    if out["feature_as_of_ts"].isna().any():
        raise ValueError("feature_as_of_ts contains null/unparseable values")
    if out["tip_ts"].isna().any():
        raise ValueError("tip_ts contains null/unparseable values")

    bad = out["feature_as_of_ts"] > out["tip_ts"]
    if bool(bad.any()):
        sample = out.loc[bad, ["game_id", "team_id", "player_id", "feature_as_of_ts", "tip_ts"]].head(10).to_dict(
            orient="records"
        )
        raise ValueError(
            f"as-of guard violated: feature_as_of_ts > tip_ts for {int(bad.sum())} rows; sample={sample}"
        )
    return out


@app.command()
def main(
    date: str = typer.Option(..., "--date", help="ET game date, e.g. 2026-01-26"),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id (defaults to latest minutes run for the date)"),
    data_root: Path = typer.Option(DEFAULT_DATA_ROOT, "--data-root", help="Data root (default from PROJECTIONS_DATA_ROOT)"),
    out_root: Path = typer.Option(
        DEFAULT_OUTPUT_ROOT,
        "--out-root",
        help="Output root (default: <data-root>/live/features_mvps_participation_v1)",
    ),
    model_feature_columns: Path | None = typer.Option(
        None,
        "--model-feature-columns",
        help="Path to MVPS mvps_part_v1/feature_columns.json (required unless default exists)",
    ),
    allow_unsafe_pointer_writes: bool = typer.Option(
        False,
        "--allow-unsafe-pointer-writes",
        help="Allow updating latest_run.json pointer outside Prefect writer-guard (dev only).",
    ),
) -> None:
    day = _normalize_day(date)
    data_root = Path(data_root).expanduser().resolve()
    out_root = Path(out_root).expanduser().resolve()

    if allow_unsafe_pointer_writes:
        os.environ["PROJECTIONS_ALLOW_UNSAFE_POINTER_WRITES"] = "1"

    minutes_root = (data_root / "live" / "features_minutes_v1").resolve()
    minutes_day_dir = minutes_root / day.strftime("%Y-%m-%d")
    resolved_run_id = _resolve_run_id(minutes_day_dir, run_id)
    minutes_run_dir = minutes_day_dir / f"run={resolved_run_id}"
    minutes_features_path = minutes_run_dir / FEATURE_FILENAME
    if not minutes_features_path.exists():
        raise typer.BadParameter(f"Missing minutes features at {minutes_features_path} (build Minutes V1 features first).")

    feature_cols_path = (
        model_feature_columns
        if model_feature_columns
        else (DEFAULT_FEATURE_COLUMNS_PATH if DEFAULT_FEATURE_COLUMNS_PATH.exists() else None)
    )
    if feature_cols_path is None:
        raise typer.BadParameter("--model-feature-columns is required (default MVPS path not found).")
    feature_cols_path = Path(feature_cols_path).expanduser().resolve()
    required_features = load_feature_columns(feature_cols_path)

    df = pd.read_parquet(minutes_features_path)
    _require_columns(df, ["game_id", "team_id", "player_id", "feature_as_of_ts", "tip_ts"], label="minutes features")
    df = _enforce_asof_guard(df)

    season = _season_for_day(day)

    game_ids = (
        pd.to_numeric(df["game_id"], errors="coerce")
        .dropna()
        .astype(int)
        .astype(str)
        .str.zfill(10)
        .unique()
        .tolist()
    )
    team_ids = pd.to_numeric(df["team_id"], errors="coerce").dropna().astype(int).unique().tolist()
    player_ids = pd.to_numeric(df["player_id"], errors="coerce").dropna().astype(int).unique().tolist()

    priors = load_rotation_priors_for_live_inference(
        data_root,
        season=season,
        game_date=day.date().isoformat(),
        game_ids=game_ids,
        team_ids=team_ids,
        player_ids=player_ids,
        allow_priors_fallback=True,
    )
    df = join_rotation_priors(df, team_priors=priors.team_priors, player_priors=priors.player_priors)

    df = prepare_live_features_for_rmh(df)

    # Some snapshots may be missing the raw num_stints source but include prior windows.
    if "num_stints" not in df.columns:
        for cand in ("num_stints_prior_20", "num_stints_prior_10", "num_stints_prior_5"):
            if cand in df.columns:
                df["num_stints"] = pd.to_numeric(df[cand], errors="coerce").fillna(0.0).astype("float64")
                break

    df = apply_odds_missing_flags(df)

    if "snapshot_type" not in df.columns:
        df["snapshot_type"] = "pretip"

    df = add_required_onehots_from_raw(df, required_cols=required_features)

    out_df, contract_report = complete_and_validate_contract(
        df,
        required_cols=required_features,
        key_cols=("game_id", "team_id", "player_id"),
        timestamp_cols=("feature_as_of_ts", "tip_ts"),
        coerce_required_to_float32=True,
    )

    missing_required = sorted(set(required_features) - set(out_df.columns))
    extra = sorted(set(out_df.columns) - set(required_features) - {"game_id", "team_id", "player_id", "feature_as_of_ts", "tip_ts"})

    typer.echo(
        "[mvps_participation_features_v1] rows="
        f"{len(out_df)} required={len(required_features)} missing={len(missing_required)} "
        f"extra={len(extra)} onehot_filled={len(contract_report.missing_onehot_filled)}"
    )
    if missing_required:
        raise RuntimeError(f"Contract check failed: missing required columns: {missing_required[:50]}")

    out_day_dir, out_run_dir = _ensure_run_output_dir(out_root, day, resolved_run_id)
    out_features_path = out_run_dir / FEATURE_FILENAME
    out_df.to_parquet(out_features_path, index=False)

    file_sha = _sha256_file(out_features_path)
    run_as_of_ts = pd.to_datetime(out_df["feature_as_of_ts"], utc=True, errors="coerce").min()

    meta: dict[str, Any] = {
        "date": day.date().isoformat(),
        "run_id": resolved_run_id,
        "minutes_features_path": str(minutes_features_path),
        "rotation_priors": {
            "used_latest_fallback": bool(priors.used_latest_fallback),
            "teams_found": int(priors.teams_found),
            "teams_missing": int(priors.teams_missing),
            "players_found": int(priors.players_found),
            "players_missing": int(priors.players_missing),
            "warning_message": priors.warning_message,
        },
        "rows": int(len(out_df)),
        "counts": {
            "games": int(pd.to_numeric(out_df["game_id"], errors="coerce").dropna().nunique()),
            "teams": int(pd.to_numeric(out_df["team_id"], errors="coerce").dropna().nunique()),
            "players": int(pd.to_numeric(out_df["player_id"], errors="coerce").dropna().nunique()),
        },
        "feature_as_of_ts_min": run_as_of_ts.isoformat() if pd.notna(run_as_of_ts) else None,
        "tip_ts_min": pd.to_datetime(out_df["tip_ts"], utc=True, errors="coerce").min().isoformat(),
        "tip_ts_max": pd.to_datetime(out_df["tip_ts"], utc=True, errors="coerce").max().isoformat(),
        "contract": {
            "required_n": int(len(required_features)),
            "missing_onehot_filled_n": int(len(contract_report.missing_onehot_filled)),
            "missing_onehot_filled": contract_report.missing_onehot_filled[:50],
        },
        "output": {
            "features_parquet": str(out_features_path),
            "sha256": file_sha,
        },
    }
    (out_run_dir / RUN_META_FILENAME).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    can_publish_pointer = (
        allow_unsafe_pointer_writes
        or os.environ.get("PROJECTIONS_ALLOW_UNSAFE_POINTER_WRITES", "").strip().lower() in {"y", "true", "1", "yes"}
        or os.environ.get("PROJECTIONS_PIPELINE_ENTRYPOINT") == "prefect"
    )
    if can_publish_pointer and run_as_of_ts is not None and pd.notna(run_as_of_ts):
        _write_latest_pointer(out_day_dir, run_id=resolved_run_id, run_as_of_ts=pd.Timestamp(run_as_of_ts))
        return

    if not can_publish_pointer:
        typer.echo(
            "[mvps_participation_features_v1] Skipping latest_run.json pointer write "
            "(set --allow-unsafe-pointer-writes or PROJECTIONS_ALLOW_UNSAFE_POINTER_WRITES=1 to enable)."
        )


if __name__ == "__main__":
    app()
