"""Audit training/inference parity for the RMH minutes model.

Compares:
1) The RMH training dataset referenced by the bundle's `config.json`
2) Live features for a given (date, run_id), after applying the same
   priors-join + feature prep path used by the RMH scorer.

Writes:
- `reports/parity_audit/rmh/date=YYYY-MM-DD/run=RUN_ID/report.json`
- `reports/parity_audit/rmh/date=YYYY-MM-DD/run=RUN_ID/report.md`
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd
import typer

UTC = timezone.utc

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_ROOT = Path(os.environ.get("PROJECTIONS_DATA_ROOT", "data"))
DEFAULT_RMH_CONFIG_PATH = PROJECT_ROOT / "config" / "rmh_current_run.json"

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class FeatureSpec:
    categorical: tuple[str, ...]
    continuous: tuple[str, ...]


def _normalize_day(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(date_str).normalize()


def _season_for_day(day: pd.Timestamp) -> int:
    return int(day.year) if int(day.month) >= 8 else int(day.year) - 1


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_bundle_dir(*, config_path: Path) -> Path:
    config = _read_json(config_path)
    bundle_dir_raw = config.get("bundle_dir")
    if not bundle_dir_raw:
        raise ValueError(f"bundle_dir not present in {config_path}")
    bundle_dir = Path(bundle_dir_raw)
    if not bundle_dir.is_absolute():
        bundle_dir = PROJECT_ROOT / bundle_dir
    bundle_dir = bundle_dir.expanduser().resolve()
    if not bundle_dir.exists():
        raise FileNotFoundError(f"RMH bundle not found: {bundle_dir}")
    return bundle_dir


def _load_feature_spec(bundle_dir: Path) -> FeatureSpec:
    schema_path = bundle_dir / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"RMH bundle missing schema.json: {schema_path}")
    payload = _read_json(schema_path)
    spec = payload.get("feature_spec") or {}
    categorical = tuple(spec.get("categorical") or [])
    continuous = tuple(spec.get("continuous") or [])
    if not categorical and not continuous:
        raise ValueError(f"Invalid feature_spec in {schema_path}")
    return FeatureSpec(categorical=categorical, continuous=continuous)


def _resolve_training_features_path(bundle_dir: Path) -> Path:
    cfg_path = bundle_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"RMH bundle missing config.json: {cfg_path}")
    cfg = _read_json(cfg_path)
    train_dir_raw = cfg.get("train_dataset_dir")
    if not train_dir_raw:
        raise ValueError(f"train_dataset_dir not present in {cfg_path}")
    train_dir = Path(train_dir_raw).expanduser()
    train_path = train_dir / "features.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Training features not found: {train_path}")
    return train_path


def _load_live_features(*, data_root: Path, day: pd.Timestamp, run_id: str) -> tuple[pd.DataFrame, Path]:
    features_path = (
        data_root / "live" / "features_minutes_v1" / day.strftime("%Y-%m-%d") / f"run={run_id}" / "features.parquet"
    )
    if not features_path.exists():
        raise FileNotFoundError(f"Live features not found: {features_path}")
    return pd.read_parquet(features_path), features_path


def _prepare_live_features_like_scorer(
    *,
    df: pd.DataFrame,
    data_root: Path,
    season: int,
    day: pd.Timestamp,
    feature_spec: FeatureSpec,
) -> pd.DataFrame:
    """Apply the priors-join + prep path used by projections.cli.score_minutes_rmh_v1."""

    required = set(feature_spec.continuous) | set(feature_spec.categorical)
    needs_priors = any("_prior_" in col for col in required) or ("minutes_from_stints_prior_20" in required)
    if not needs_priors:
        return df

    from projections.models.rotation_minutes_hurdle_v1.live_features import prepare_live_features_for_rmh
    from projections.rotation.live_features_v1 import load_rotation_priors_for_live_inference
    from projections.rotation.rotation_set_minutes_features_v1 import (
        apply_odds_missing_flags,
        fill_numeric_missing_with_zero,
        join_rotation_priors,
    )

    if "game_id" not in df.columns:
        raise RuntimeError("Live features missing game_id; cannot join rotation priors.")

    game_ids = df["game_id"].astype(str).unique().tolist()
    team_ids = (
        pd.to_numeric(df["team_id"], errors="coerce").dropna().astype(int).unique().tolist()
        if "team_id" in df.columns
        else []
    )
    player_ids = (
        pd.to_numeric(df["player_id"], errors="coerce").dropna().astype(int).unique().tolist()
        if "player_id" in df.columns
        else []
    )

    priors = load_rotation_priors_for_live_inference(
        data_root=data_root,
        season=season,
        game_date=day.strftime("%Y-%m-%d"),
        game_ids=game_ids,
        team_ids=team_ids,
        player_ids=player_ids,
        allow_priors_fallback=True,
    )

    work = df.copy()
    if {"spread_home", "total"}.issubset(work.columns):
        work = apply_odds_missing_flags(work)
    work = fill_numeric_missing_with_zero(work)
    work = join_rotation_priors(
        work,
        team_priors=priors.team_priors,
        player_priors=priors.player_priors,
    )
    return prepare_live_features_for_rmh(work)


def _series_numeric_summary(series: pd.Series) -> dict[str, Any]:
    # Treat bool as numeric (0/1) to avoid numpy quantile issues on boolean dtype.
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    missing_rate = float(values.isna().mean()) if len(values) else 0.0
    nonnull = values.dropna()
    if nonnull.empty:
        return {"missing_rate": missing_rate, "count": int(len(values)), "count_nonnull": 0}
    qs = nonnull.quantile([0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]).to_dict()
    return {
        "missing_rate": missing_rate,
        "count": int(len(values)),
        "count_nonnull": int(len(nonnull)),
        "mean": float(nonnull.mean()),
        "std": float(nonnull.std(ddof=0)),
        "min": float(nonnull.min()),
        "max": float(nonnull.max()),
        "q01": float(qs.get(0.01)),
        "q05": float(qs.get(0.05)),
        "q10": float(qs.get(0.10)),
        "q50": float(qs.get(0.50)),
        "q90": float(qs.get(0.90)),
        "q95": float(qs.get(0.95)),
        "q99": float(qs.get(0.99)),
    }


def _series_categorical_summary(series: pd.Series, *, top_n: int = 10) -> dict[str, Any]:
    values = series.astype("string")
    missing_rate = float(values.isna().mean()) if len(values) else 0.0
    nonnull = values.dropna()
    counts = nonnull.value_counts(dropna=True).head(top_n)
    return {
        "missing_rate": missing_rate,
        "count": int(len(values)),
        "count_nonnull": int(len(nonnull)),
        "n_unique": int(nonnull.nunique(dropna=True)),
        "top_values": {str(k): int(v) for k, v in counts.to_dict().items()},
    }


def _markdown_section(title: str, lines: list[str]) -> str:
    out = [f"## {title}"]
    out.extend(lines or ["(none)"])
    return "\n".join(out)


@app.command()
def main(
    *,
    date: str = typer.Option(..., "--date", help="Slate date (YYYY-MM-DD)."),
    run_id: str = typer.Option(..., "--run-id", help="Live run_id (e.g., 20260124T195959Z)."),
    data_root: Path = typer.Option(DEFAULT_DATA_ROOT, "--data-root", help="PROJECTIONS_DATA_ROOT override."),
    rmh_config: Path = typer.Option(
        DEFAULT_RMH_CONFIG_PATH, "--rmh-config", help="RMH config JSON (defaults to config/rmh_current_run.json)."
    ),
    output_root: Path = typer.Option(
        Path("reports/parity_audit"),
        "--output-root",
        help="Directory to write report artifacts (json + md).",
    ),
    sample_train_rows: int = typer.Option(
        200_000,
        "--sample-train-rows",
        help="Sample up to N training rows for stats (0 = use all).",
    ),
) -> None:
    day = _normalize_day(date)
    season = _season_for_day(day)
    data_root = Path(data_root).expanduser().resolve()

    if not rmh_config.is_absolute():
        rmh_config = (PROJECT_ROOT / rmh_config).resolve()

    bundle_dir = _resolve_bundle_dir(config_path=rmh_config)
    feature_spec = _load_feature_spec(bundle_dir)
    train_features_path = _resolve_training_features_path(bundle_dir)

    typer.echo(
        f"[rmh-parity] date={day.date()} run_id={run_id} bundle={bundle_dir.name} train={train_features_path}",
        err=True,
    )

    live_df, live_path = _load_live_features(data_root=data_root, day=day, run_id=run_id)
    live_prepared = _prepare_live_features_like_scorer(
        df=live_df, data_root=data_root, season=season, day=day, feature_spec=feature_spec
    )

    required = set(feature_spec.continuous) | set(feature_spec.categorical)
    live_missing_required = sorted(required.difference(live_prepared.columns))
    if live_missing_required:
        raise RuntimeError(f"Prepared live features missing required RMH inputs: {live_missing_required[:10]}")

    # Training slice
    train_cols = sorted(
        set(["status", "injury_snapshot_missing", "prior_play_prob", "inactive_streak_len"]) | required
    )
    try:
        train_df = pd.read_parquet(train_features_path, columns=train_cols)
        train_cols_selected: list[str] | None = train_cols
    except Exception:
        train_df = pd.read_parquet(train_features_path)
        train_cols_selected = None

    if sample_train_rows and sample_train_rows > 0 and len(train_df) > sample_train_rows:
        train_df = train_df.sample(n=sample_train_rows, random_state=0)

    # Build report payload
    report: dict[str, Any] = {
        "meta": {
            "date": day.strftime("%Y-%m-%d"),
            "run_id": run_id,
            "generated_at": pd.Timestamp.now(tz=UTC).isoformat(),
            "bundle_dir": str(bundle_dir),
            "live_features_path": str(live_path),
            "training_features_path": str(train_features_path),
            "training_columns_selected": train_cols_selected,
            "counts": {"live_rows": int(len(live_prepared)), "train_rows": int(len(train_df))},
        },
        "feature_spec": asdict(feature_spec),
        "coverage": {
            "required_total": int(len(required)),
            "required_continuous": int(len(feature_spec.continuous)),
            "required_categorical": int(len(feature_spec.categorical)),
        },
        "columns": {
            "live_extra": sorted(set(live_prepared.columns) - required),
            "train_missing_required": sorted(required - set(train_df.columns)),
        },
        "continuous": {},
        "categorical": {},
        "checks": {},
    }

    for col in feature_spec.continuous:
        report["continuous"][col] = {
            "train": _series_numeric_summary(train_df[col]) if col in train_df.columns else {"missing": True},
            "live": _series_numeric_summary(live_prepared[col]),
        }

    for col in feature_spec.categorical:
        train_series = train_df[col] if col in train_df.columns else pd.Series([pd.NA] * len(train_df))
        live_series = live_prepared[col]
        train_summary = _series_categorical_summary(train_series)
        live_summary = _series_categorical_summary(live_series)
        train_values = set(train_series.astype("string").dropna().unique().tolist())
        live_values = set(live_series.astype("string").dropna().unique().tolist())
        report["categorical"][col] = {
            "train": train_summary,
            "live": live_summary,
            "train_only_values": sorted(train_values - live_values),
            "live_only_values": sorted(live_values - train_values),
        }

    def _safe_mean(df: pd.DataFrame, col: str) -> float | None:
        if col not in df.columns:
            return None
        return float(pd.to_numeric(df[col], errors="coerce").mean())

    report["checks"]["injury_snapshot_missing_mean"] = {
        "train": _safe_mean(train_df, "injury_snapshot_missing"),
        "live": _safe_mean(live_prepared, "injury_snapshot_missing"),
    }
    report["checks"]["prior_play_prob_nonnull_rate"] = {
        "train": float(pd.to_numeric(train_df.get("prior_play_prob"), errors="coerce").notna().mean())
        if "prior_play_prob" in train_df.columns
        else None,
        "live": float(pd.to_numeric(live_prepared.get("prior_play_prob"), errors="coerce").notna().mean())
        if "prior_play_prob" in live_prepared.columns
        else None,
    }
    report["checks"]["inactive_streak_len_max"] = {
        "train": int(pd.to_numeric(train_df.get("inactive_streak_len"), errors="coerce").fillna(0).max())
        if "inactive_streak_len" in train_df.columns
        else None,
        "live": int(pd.to_numeric(live_prepared.get("inactive_streak_len"), errors="coerce").fillna(0).max())
        if "inactive_streak_len" in live_prepared.columns
        else None,
    }

    # Starter coverage by game-team (live only)
    if {"game_id", "team_id"}.issubset(live_prepared.columns):
        starter_flag = pd.to_numeric(live_prepared.get("starter_flag"), errors="coerce").fillna(0).astype(int)
        by_team = (
            pd.concat([live_prepared[["game_id", "team_id"]].copy(), starter_flag.rename("starter_flag")], axis=1)
            .groupby(["game_id", "team_id"], as_index=False)["starter_flag"]
            .sum()
        )
        report["checks"]["starter_flag_sums"] = by_team.sort_values("starter_flag").to_dict(orient="records")

    # Write outputs
    out_dir = output_root / "rmh" / f"date={day.strftime('%Y-%m-%d')}" / f"run={run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    md_sections: list[str] = []
    md_sections.append(
        _markdown_section(
            "Meta",
            [
                f"- date: {report['meta']['date']}",
                f"- run_id: {report['meta']['run_id']}",
                f"- bundle: {bundle_dir.name}",
                f"- live_rows: {report['meta']['counts']['live_rows']}",
                f"- train_rows: {report['meta']['counts']['train_rows']}",
                f"- live_features: {live_path}",
                f"- training_features: {train_features_path}",
            ],
        )
    )
    md_sections.append(
        _markdown_section(
            "Coverage",
            [
                f"- required_total: {report['coverage']['required_total']}",
                f"- train_missing_required: {len(report['columns']['train_missing_required'])}",
                f"- live_extra_columns: {len(report['columns']['live_extra'])}",
            ],
        )
    )
    md_sections.append(
        _markdown_section(
            "Injuries",
            [
                f"- injury_snapshot_missing_mean: train={report['checks']['injury_snapshot_missing_mean']['train']} "
                f"live={report['checks']['injury_snapshot_missing_mean']['live']}",
                f"- prior_play_prob_nonnull_rate: train={report['checks']['prior_play_prob_nonnull_rate']['train']} "
                f"live={report['checks']['prior_play_prob_nonnull_rate']['live']}",
            ],
        )
    )
    md_sections.append(
        _markdown_section(
            "DNP History",
            [
                f"- inactive_streak_len_max: train={report['checks']['inactive_streak_len_max']['train']} "
                f"live={report['checks']['inactive_streak_len_max']['live']}",
            ],
        )
    )

    # Lineup categorical parity (key driver of flat RMH outputs when missing).
    lineup_cols = ("lineup_role", "lineup_status", "lineup_roster_status")
    if all(col in report["categorical"] for col in lineup_cols):
        lines: list[str] = []
        for col in lineup_cols:
            train = report["categorical"][col]["train"]
            live = report["categorical"][col]["live"]
            lines.append(f"- {col}_missing_rate: train={train['missing_rate']} live={live['missing_rate']}")
        md_sections.append(_markdown_section("Lineups", lines))

    # Starter coverage quick check (live only).
    if "starter_flag_sums" in report["checks"]:
        sums = report["checks"]["starter_flag_sums"]
        bad = [row for row in sums if int(row.get("starter_flag") or 0) != 5]
        if bad:
            sample = bad[:5]
            md_sections.append(
                _markdown_section(
                    "Starters",
                    [
                        f"- teams_not_5: {len(bad)}",
                        f"- sample: {sample}",
                    ],
                )
            )

    md_path.write_text("\n\n".join(md_sections) + "\n", encoding="utf-8")

    typer.echo(f"[rmh-parity] wrote {json_path}", err=True)
    typer.echo(f"[rmh-parity] wrote {md_path}", err=True)


if __name__ == "__main__":
    app()
