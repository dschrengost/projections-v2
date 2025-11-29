"""Score the minutes_v1 LightGBM bundle for a daily slate.

Outputs include minutes quantiles, play_prob, and now a canonical ``is_starter`` column derived from the feature slice.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Set
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import typer
from click.core import ParameterSource

from projections import paths
from projections.labels import derive_starter_flag_labels
from projections.minutes_v1 import modeling
from projections.minutes_v1.config import load_scoring_config
from projections.minutes_v1.production import load_production_minutes_bundle
from projections.minutes_v1.datasets import KEY_COLUMNS, deduplicate_latest
from projections.minutes_v1.logs import prediction_logs_base
from projections.minutes_v1.promotion_calibrator import (
    PromotionPriorContext,
    apply_promotion_prior,
    load_promotion_config,
)
from projections.minutes_v1.pos import canonical_pos_bucket_series
from projections.minutes_v1.reconcile import (
    TeamReconcileDebug,
    load_reconcile_config,
    reconcile_minutes_p50_all,
)
from projections.minutes_v1.starter_flags import (
    StarterFlagResult,
    derive_starter_flag_label,
    normalize_starter_signals,
)
from projections.models.minutes_lgbm import (
    _filter_out_players,
    apply_conformal,
    apply_play_probability_mixture,
    predict_play_probability,
)

UTC = timezone.utc
DEFAULT_FEATURES_ROOT = paths.data_path("gold", "features_minutes_v1")
DEFAULT_LIVE_FEATURES_ROOT = paths.data_path("live", "features_minutes_v1")
DEFAULT_DAILY_ROOT = Path("artifacts/minutes_v1/daily")
DEFAULT_BUNDLE_CONFIG = Path("config/minutes_current_run.json")
DEFAULT_INJURIES_ROOT = paths.data_path("bronze", "injuries_raw")
DEFAULT_SCHEDULE_ROOT = paths.data_path("silver", "schedule")
DEFAULT_STARTER_PRIORS = paths.data_path("gold", "minutes_priors", "starter_slot_priors.parquet")
DEFAULT_STARTER_HISTORY = paths.data_path("gold", "minutes_priors", "player_starter_history.parquet")
DEFAULT_PROMOTION_CONFIG = Path("config/minutes_promotion.yaml")
DEFAULT_RECONCILE_CONFIG = Path("config/minutes_l2_reconcile.yaml")
DEFAULT_PREDICTION_LOGS_ROOT = prediction_logs_base()
OUTPUT_FILENAME = "minutes.parquet"
SUMMARY_FILENAME = "summary.json"
LATEST_POINTER = "latest_run.json"
FEATURE_FILENAME = "features.parquet"

Mode = Literal["historical", "live"]
MinutesOutputMode = Literal["conditional", "unconditional", "both"]
ReconcileMode = Literal["none", "p50", "p50_and_tails"]

app = typer.Typer(help=__doc__)


_DEFAULT_PARAMETER_SOURCES = {ParameterSource.DEFAULT, ParameterSource.DEFAULT_MAP}


def _apply_scoring_overrides(
    ctx: typer.Context,
    cli_params: dict[str, Any],
    config_path: Path | None,
) -> dict[str, Any]:
    """Load YAML overrides for the scoring CLI."""

    if config_path is None:
        return cli_params

    config = load_scoring_config(config_path)
    overrides = config.model_dump(exclude_unset=True)
    for name, value in overrides.items():
        if name not in cli_params:
            continue
        source = ctx.get_parameter_source(name)
        if source in _DEFAULT_PARAMETER_SOURCES:
            cli_params[name] = value
    return cli_params


def _season_from_date(day: date) -> int:
    """Season is keyed by start year (Aug–Jul)."""

    return day.year if day.month >= 8 else day.year - 1


def _iter_days(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _make_reconcile_debugger(
    enabled: bool,
) -> Callable[[TeamReconcileDebug], None] | None:
    if not enabled:
        return None

    def _log(payload: TeamReconcileDebug) -> None:
        header = (
            f"[l2-reconcile] game={payload.game_id} team={payload.team_id} "
            f"pre={payload.pre_total:.1f} post={payload.post_total:.1f}"
        )
        typer.echo(header, err=True)
        for entry in payload.top_deltas:
            name = entry.get("player_name") or entry.get("player_id")
            typer.echo(
                f"  - {name}: Δ={entry.get('delta', 0.0):.2f} "
                f"({entry.get('minutes_before', 0.0):.1f}→{entry.get('minutes_after', 0.0):.1f})",
                err=True,
            )

    return _log


def _write_latest_pointer(day_dir: Path, *, run_id: str, run_as_of_ts: datetime | None) -> None:
    pointer = day_dir / LATEST_POINTER
    payload = {
        "run_id": run_id,
        "run_as_of_ts": run_as_of_ts.isoformat() if run_as_of_ts else None,
    }
    pointer.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _ensure_bundle_defaults(bundle: dict) -> dict:
    bundle.setdefault("bucket_mode", "none")
    bundle.setdefault(
        "bucket_offsets",
        {"__global__": {"d10": 0.0, "d90": 0.0, "n": 0}},
    )
    bundle.setdefault("conformal_mode", "tail-deltas")
    return bundle


def _load_bundle(bundle_dir: Path) -> dict:
    model_path = bundle_dir / "lgbm_quantiles.joblib"
    if not model_path.exists():
        raise typer.BadParameter(f"Bundle missing {model_path}", param_name="bundle_dir")
    bundle = joblib.load(model_path)
    return _ensure_bundle_defaults(bundle)


def _load_meta(bundle_dir: Path) -> dict:
    meta_path = bundle_dir / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_bundle_artifacts(
    bundle_dir: Path | None,
    config_path: Path,
) -> tuple[dict, Path, dict, str]:
    if bundle_dir is not None:
        resolved = bundle_dir.expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"Bundle directory not found at {resolved}")
        bundle = _load_bundle(resolved)
        model_meta = _load_meta(resolved)
        run_id = resolved.name
        return bundle, resolved, model_meta, run_id

    production_bundle = load_production_minutes_bundle(config_path=config_path)
    bundle = _ensure_bundle_defaults(dict(production_bundle))
    run_dir_str = bundle.get("run_dir")
    if not run_dir_str:
        raise RuntimeError("Production bundle missing run_dir metadata.")
    resolved = Path(run_dir_str).expanduser()
    model_meta = bundle.get("meta", {})
    run_id = bundle.get("run_id") or resolved.name
    return bundle, resolved, model_meta, run_id


def _read_parquet_source(path: Path) -> pd.DataFrame:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Features path {path} does not exist")
    if path.is_file():
        return pd.read_parquet(path)
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files discovered under {path}")
    frames = [pd.read_parquet(file) for file in files]
    return pd.concat(frames, ignore_index=True)


def _load_feature_slice(
    start: date,
    end: date,
    *,
    features_root: Path,
    features_path: Path | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    if features_path is not None:
        frames = [_read_parquet_source(features_path)]
    else:
        partitions: set[tuple[int, int]] = {
            (_season_from_date(day), day.month) for day in _iter_days(start, end)
        }
        frames = []
        for season, month in sorted(partitions):
            feature_path = (
                features_root / f"season={season}" / f"month={month:02d}" / "features.parquet"
            )
            if not feature_path.exists():
                raise FileNotFoundError(f"Missing feature parquet at {feature_path}")
            frame = pd.read_parquet(feature_path)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    features = pd.concat(frames, ignore_index=True)
    features["game_date"] = pd.to_datetime(features["game_date"]).dt.date
    if "feature_as_of_ts" in features.columns:
        features["feature_as_of_ts"] = pd.to_datetime(features["feature_as_of_ts"])
    mask = (features["game_date"] >= start) & (features["game_date"] <= end)
    filtered = features.loc[mask].copy()
    if run_id is not None:
        filtered["run_id"] = run_id
    return filtered


def _load_promotion_context(
    *,
    enabled: bool,
    priors_path: Path,
    history_path: Path,
    config_path: Path,
) -> PromotionPriorContext | None:
    if not enabled:
        return None
    priors_path = priors_path.expanduser()
    history_path = history_path.expanduser()
    config_path = config_path.expanduser()
    if not priors_path.exists() or not history_path.exists():
        typer.echo(
            f"[promotion-prior] ERROR: missing priors/history ({priors_path}, {history_path}); aborting.",
            err=True,
        )
        raise typer.Exit(code=1)
    priors_df = pd.read_parquet(priors_path)
    history_df = pd.read_parquet(history_path)
    if "starter_history_games" not in history_df.columns:
        raise ValueError(f"Starter history parquet {history_path} missing 'starter_history_games'.")
    config = load_promotion_config(config_path)
    return PromotionPriorContext(priors=priors_df, player_history=history_df, config=config)


def _apply_promotion_prior_if_enabled(
    df: pd.DataFrame,
    ctx: PromotionPriorContext | None,
) -> pd.DataFrame:
    if ctx is None or df.empty:
        return df
    working = df.copy()
    if "pos_bucket" not in working.columns:
        raise ValueError("Promotion prior requires pos_bucket column.")
    if "starter_flag" in working.columns:
        starter_series = working["starter_flag"]
    elif "starter_flag_label" in working.columns:
        starter_series = working["starter_flag_label"]
    else:
        starter_series = pd.Series(0, index=working.index)
    working["is_projected_starter"] = starter_series.astype(int)
    working = working.merge(ctx.player_history, on="player_id", how="left")
    if "starter_history_games" not in working.columns:
        working["starter_history_games"] = 0
    else:
        working["starter_history_games"] = working["starter_history_games"].fillna(0)
    calibrated = apply_promotion_prior(working, ctx.priors, ctx.config)
    return calibrated


@lru_cache(maxsize=8)
def _cached_player_name_map(root: str, seasons_key: tuple[int, ...]) -> dict[int, str]:
    root_path = Path(root)
    seasons = [int(season) for season in seasons_key]
    frames: list[pd.DataFrame] = []
    for season in seasons:
        season_dir = root_path / f"season={season}"
        if not season_dir.exists():
            continue
        for file in sorted(season_dir.glob("*.parquet")):
            df = pd.read_parquet(file, columns=["player_id", "player_name", "as_of_ts"])
            frames.append(df)
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["player_id", "player_name"])
    combined["player_id"] = combined["player_id"].astype(int)
    combined = combined.sort_values("as_of_ts").drop(columns=["as_of_ts"])
    deduped = combined.drop_duplicates(subset=["player_id"], keep="last")
    return {int(row.player_id): str(row.player_name) for row in deduped.itertuples()}


@lru_cache(maxsize=8)
def _cached_team_metadata(root: str, seasons_key: tuple[int, ...]) -> pd.DataFrame:
    root_path = Path(root)
    seasons = [int(season) for season in seasons_key]
    frames: list[pd.DataFrame] = []
    for season in seasons:
        season_dir = root_path / f"season={season}"
        if not season_dir.exists():
            continue
        for month_dir in sorted(season_dir.glob("month=*")):
            sched_path = month_dir / "schedule.parquet"
            if not sched_path.exists():
                continue
            df = pd.read_parquet(
                sched_path,
                columns=[
                    "home_team_id",
                    "home_team_name",
                    "home_team_tricode",
                    "away_team_id",
                    "away_team_name",
                    "away_team_tricode",
                ],
            )
            home = df[
                ["home_team_id", "home_team_name", "home_team_tricode"]
            ].rename(
                columns={
                    "home_team_id": "team_id",
                    "home_team_name": "team_name",
                    "home_team_tricode": "team_tricode",
                }
            )
            away = df[
                ["away_team_id", "away_team_name", "away_team_tricode"]
            ].rename(
                columns={
                    "away_team_id": "team_id",
                    "away_team_name": "team_name",
                    "away_team_tricode": "team_tricode",
                }
            )
            frames.extend([home, away])
    if not frames:
        return pd.DataFrame(columns=["team_id", "team_name", "team_tricode"])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["team_id"]).copy()
    combined["team_id"] = combined["team_id"].astype(int)
    combined = combined.drop_duplicates(subset=["team_id"], keep="last")
    return combined


def _player_name_map(injuries_root: Path, seasons: set[int]) -> dict[int, str]:
    if not seasons:
        return {}
    return _cached_player_name_map(str(injuries_root.expanduser()), tuple(sorted(seasons)))


def _team_metadata(schedule_root: Path, seasons: set[int]) -> pd.DataFrame:
    if not seasons:
        return pd.DataFrame(columns=["team_id", "team_name", "team_tricode"])
    return _cached_team_metadata(str(schedule_root.expanduser()), tuple(sorted(seasons)))


def _prepare_features(df: pd.DataFrame, *, mode: Mode = "historical") -> pd.DataFrame:
    if df.empty:
        return df
    if mode == "live":
        working = df.copy()
        if "starter_flag_label" not in working.columns:
            working = normalize_starter_signals(working)
            starter_result: StarterFlagResult = derive_starter_flag_label(
                working,
                group_cols=("game_id", "team_id"),
            )
            if starter_result.overflow:
                for game_id, team_id, count in starter_result.overflow:
                    typer.echo(
                        f"[live] warning: starter_flag_label derived {count} starters for game {game_id} team {team_id}; expected <=5.",
                        err=True,
                    )
            working["starter_flag_label"] = starter_result.values.fillna(0).astype(int)
    else:
        try:
            working = derive_starter_flag_labels(df)
        except ValueError:
            working = df.copy()
            source = "starter_prev_game_asof" if "starter_prev_game_asof" in working.columns else None
            placeholder = (
                working[source].astype(bool)
                if source is not None
                else pd.Series(False, index=working.index, dtype=bool)
            )
            working["starter_flag_label"] = placeholder.astype(int)
    try:
        working = _filter_out_players(working)
    except ValueError:
        return pd.DataFrame()
    working = deduplicate_latest(working, key_cols=KEY_COLUMNS, order_cols=["feature_as_of_ts"])
    return working.reset_index(drop=True)


def _score_rows(
    df: pd.DataFrame,
    bundle: dict,
    *,
    disable_play_prob: bool = False,
    promotion_ctx: PromotionPriorContext | None = None,
    promotion_debug: bool = False,
) -> pd.DataFrame:
    if df.empty:
        return df
    feature_columns: list[str] = bundle["feature_columns"]
    quantiles = bundle["quantiles"]
    calibrator = bundle.get("calibrator")

    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise RuntimeError(f"Feature frame missing required columns: {', '.join(sorted(missing))}")

    feature_matrix = df[feature_columns]
    preds = modeling.predict_quantiles(quantiles, feature_matrix)
    p10_raw = np.minimum(preds[0.1], preds[0.5])
    p90_raw = np.maximum(preds[0.9], preds[0.5])
    if calibrator is not None:
        p10_cal, p90_cal = calibrator.calibrate(p10_raw, p90_raw)
    else:
        p10_cal, p90_cal = p10_raw, p90_raw

    working = df.copy()
    working["p10_pred"] = p10_cal
    working["p50_pred"] = preds[0.5]
    working["p90_pred"] = p90_cal
    working = apply_conformal(
        working,
        bundle["bucket_offsets"],
        mode=bundle["conformal_mode"],
        bucket_mode=bundle["bucket_mode"],
    )

    # Conditional (if active) quantiles.
    working["minutes_p10"] = working["p10_adj"]
    working["minutes_p50"] = working["p50_adj"]
    working["minutes_p90"] = working["p90_adj"]
    working["minutes_p10_cond"] = working["minutes_p10"]
    working["minutes_p50_cond"] = working["minutes_p50"]
    working["minutes_p90_cond"] = working["minutes_p90"]
    # Legacy aliases for coverage tooling.
    working["p10_cond"] = working["minutes_p10"]
    working["p50_cond"] = working["minutes_p50"]
    working["p90_cond"] = working["minutes_p90"]

    play_prob_artifacts = None if disable_play_prob else bundle.get("play_probability")
    if play_prob_artifacts is not None:
        play_prob = predict_play_probability(play_prob_artifacts, feature_matrix)
    else:
        play_prob = np.ones(len(working), dtype=float)
    working["play_prob"] = play_prob
    if not disable_play_prob:
        working = apply_play_probability_mixture(working, play_prob)

    if "pos_bucket" not in working.columns:
        base_series = working.get("archetype")
        if base_series is None:
            base_series = pd.Series("UNK", index=working.index)
        working["pos_bucket"] = canonical_pos_bucket_series(base_series)
    else:
        working["pos_bucket"] = canonical_pos_bucket_series(working["pos_bucket"])

    if promotion_ctx is not None:
        working = _apply_promotion_prior_if_enabled(working, promotion_ctx)
        if promotion_debug:
            applied_flag = working.get("promotion_prior_applied")
            if applied_flag is not None:
                applied_mask = applied_flag.astype(bool)
                applied_count = int(applied_mask.sum())
                if applied_count:
                    debug_cols = [
                        col
                        for col in (
                            "player_name",
                            "player_id",
                            "team_id",
                            "pos_bucket",
                            "minutes_p50_raw",
                            "minutes_p50",
                        )
                        if col in working.columns
                    ]
                    typer.echo(
                        f"[promotion-prior] applied to {applied_count} players",
                        err=True,
                    )
                    typer.echo(
                        working.loc[applied_mask, debug_cols].head().to_string(index=False),
                        err=True,
                    )
    if "starter_flag" not in working.columns and "starter_flag_label" in working.columns:
        working["starter_flag"] = working["starter_flag_label"].astype(bool)

    # Propagate a canonical is_starter flag into outputs.
    if "starter_flag" in working.columns:
        working["is_starter"] = pd.to_numeric(working["starter_flag"], errors="coerce").fillna(0).astype("int8")
    elif {"is_confirmed_starter", "is_projected_starter"}.issubset(working.columns):
        tmp = (
            pd.to_numeric(working["is_confirmed_starter"], errors="coerce").fillna(0).astype(int)
            | pd.to_numeric(working["is_projected_starter"], errors="coerce").fillna(0).astype(int)
        )
        working["is_starter"] = tmp.astype("int8")
    else:
        working["is_starter"] = 0
        typer.echo("[minutes] warning: is_starter not found in features; defaulting to 0", err=True)
    return working


def _annotate_metadata(
    df: pd.DataFrame,
    *,
    player_lookup: dict[int, str],
    team_meta: pd.DataFrame,
) -> pd.DataFrame:
    if df.empty:
        return df
    working = df.copy()
    player_series = working["player_id"] if "player_id" in working.columns else None
    player_ids = (
        pd.to_numeric(player_series, errors="coerce") if player_series is not None else None
    )
    if player_lookup and player_ids is not None:
        def _player_name(pid: float | int | None) -> str | None:
            if pid is None or pd.isna(pid):
                return None
            pid_int = int(pid)
            return player_lookup.get(pid_int)

        lookup_series = player_ids.map(_player_name)
        if "player_name" in working.columns:
            working["player_name"] = working["player_name"].where(
                working["player_name"].notna(),
                lookup_series,
            )
        else:
            working["player_name"] = lookup_series
    if "player_name" not in working.columns:
        working["player_name"] = None
    if "player_id" in working.columns:
        working["player_name"] = working["player_name"].where(
            working["player_name"].notna(),
            working["player_id"].map(lambda pid: str(pid) if pid is not None else None),
        )

    if not team_meta.empty and "team_id" in working.columns:
        team_meta = team_meta.copy()
        team_meta["team_id"] = team_meta["team_id"].astype(int)
        working["team_id"] = pd.to_numeric(working["team_id"], errors="coerce").astype("Int64")
        team_meta = team_meta.rename(
            columns={
                "team_name": "team_name_meta",
                "team_tricode": "team_tricode_meta",
            }
        )
        working = working.merge(team_meta, on="team_id", how="left")
        if "opponent_team_id" in working.columns:
            opponent_meta = team_meta.rename(
                columns={
                    "team_id": "opponent_team_id",
                    "team_name_meta": "opponent_team_name_meta",
                    "team_tricode_meta": "opponent_team_tricode_meta",
                }
            )
            opponent_meta["opponent_team_id"] = opponent_meta["opponent_team_id"].astype(int)
            working["opponent_team_id"] = pd.to_numeric(
                working["opponent_team_id"], errors="coerce"
            ).astype("Int64")
            working = working.merge(opponent_meta, on="opponent_team_id", how="left")
    else:
        working["team_name_meta"] = None
        working["team_tricode_meta"] = None
        working["opponent_team_name_meta"] = None
        working["opponent_team_tricode_meta"] = None

    if "team_name" not in working.columns:
        working["team_name"] = pd.NA
    if "team_tricode" not in working.columns:
        working["team_tricode"] = pd.NA
    working["team_name"] = working["team_name"].where(
        working["team_name"].notna(), working.get("team_name_meta")
    )
    working["team_tricode"] = working["team_tricode"].where(
        working["team_tricode"].notna(),
        working.get("team_tricode_meta", working["team_name"]),
    )

    if "opponent_team_name" not in working.columns:
        working["opponent_team_name"] = pd.NA
    if "opponent_team_tricode" not in working.columns:
        working["opponent_team_tricode"] = pd.NA
    working["opponent_team_name"] = working["opponent_team_name"].where(
        working["opponent_team_name"].notna(), working.get("opponent_team_name_meta")
    )
    working["opponent_team_tricode"] = working["opponent_team_tricode"].where(
        working["opponent_team_tricode"].notna(),
        working.get("opponent_team_tricode_meta", working["opponent_team_name"]),
    )

    working = working.drop(
        columns=[
            col
            for col in (
                "team_name_meta",
                "team_tricode_meta",
                "opponent_team_name_meta",
                "opponent_team_tricode_meta",
            )
            if col in working.columns
        ]
    )
    return working


def _resolve_run_dir(root: Path, day: date, run_id: str | None, *, write_mode: bool) -> Path:
    day_dir = root / day.isoformat()
    if not day_dir.exists():
        if write_mode:
            day_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"No artifacts found for {day.isoformat()} under {root}")
    if run_id:
        run_dir = day_dir / f"run={run_id}"
        if write_mode:
            run_dir.mkdir(parents=True, exist_ok=True)
        elif not run_dir.exists():
            raise FileNotFoundError(f"Requested run={run_id} missing under {day_dir}")
        return run_dir
    pointer = day_dir / LATEST_POINTER
    if pointer.exists():
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            latest_run = payload.get("run_id")
        except json.JSONDecodeError:
            latest_run = None
        if latest_run:
            run_dir = day_dir / f"run={latest_run}"
            if run_dir.exists():
                if write_mode:
                    run_dir.mkdir(parents=True, exist_ok=True)
                return run_dir
    if write_mode:
        day_dir.mkdir(parents=True, exist_ok=True)
        return day_dir
    # Historical mode default path (no run ids)
    return day_dir


def _select_minutes_columns(df: pd.DataFrame, minutes_output: MinutesOutputMode) -> pd.DataFrame:
    cond_cols = [
        "minutes_p10",
        "minutes_p50",
        "minutes_p90",
        "minutes_p10_cond",
        "minutes_p50_cond",
        "minutes_p90_cond",
    ]
    uncond_cols = [
        "minutes_p10_uncond",
        "minutes_p50_uncond",
        "minutes_p90_uncond",
    ]
    if minutes_output == "conditional":
        return df.drop(columns=[col for col in uncond_cols if col in df.columns], errors="ignore")
    if minutes_output == "unconditional":
        return df.drop(columns=[col for col in cond_cols if col in df.columns], errors="ignore")
    return df


def _write_daily_outputs(
    day: date,
    df: pd.DataFrame,
    *,
    out_root: Path,
    model_run_id: str,
    bundle_dir: Path,
    model_meta: dict,
    run_id: str | None,
    run_as_of_ts: datetime | None,
    minutes_output: MinutesOutputMode,
) -> None:
    run_dir = _resolve_run_dir(out_root, day, run_id, write_mode=True)
    parquet_path = run_dir / OUTPUT_FILENAME
    summary_path = run_dir / SUMMARY_FILENAME

    export_df = df.copy()
    if not export_df.empty:
        export_df["model_run_id"] = model_run_id
        export_df = _select_minutes_columns(export_df, minutes_output)
        export_df.to_parquet(parquet_path, index=False)
    else:
        # Always leave a valid parquet with the expected schema to simplify tooling.
        empty_columns = [
            "game_date",
            "tip_ts",
            "game_id",
            "player_id",
            "status",
            "team_id",
            "opponent_team_id",
            "starter_flag",
            "play_prob",
            "minutes_p10",
            "minutes_p50",
            "minutes_p90",
            "minutes_p10_cond",
            "minutes_p50_cond",
            "minutes_p90_cond",
            "minutes_p10_uncond",
            "minutes_p50_uncond",
            "minutes_p90_uncond",
        ]
        empty_frame = pd.DataFrame(columns=empty_columns)
        empty_frame = _select_minutes_columns(empty_frame, minutes_output)
        empty_frame.to_parquet(parquet_path, index=False)

    summary = {
        "date": day.isoformat(),
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "model_run_id": model_run_id,
        "bundle_dir": str(bundle_dir),
        "model_meta": model_meta,
        "run_id": run_id,
        "run_as_of_ts": run_as_of_ts.isoformat() if run_as_of_ts else None,
        "counts": {
            "rows": int(len(df)),
            "players": int(df["player_id"].nunique()) if not df.empty else 0,
            "teams": int(df["team_id"].nunique()) if not df.empty else 0,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if run_id:
        _write_latest_pointer(run_dir.parent, run_id=run_id, run_as_of_ts=run_as_of_ts)
    typer.echo(f"[minutes] {day.isoformat()}: wrote {len(df)} rows to {parquet_path}")


def _log_predictions(
    df: pd.DataFrame,
    *,
    logs_root: Path,
    run_id: str | None,
    run_as_of_ts: datetime | None,
    model_run_id: str,
) -> None:
    """Append predictions and features to the long-term log."""
    if df.empty:
        return

    # Ensure directory exists (partition by minutes run when available)
    logs_root = logs_root.expanduser()
    if run_id:
        logs_root = logs_root / f"run={run_id}"
    logs_root.mkdir(parents=True, exist_ok=True)

    # Add metadata
    log_df = df.copy()
    log_df["log_timestamp"] = datetime.now(tz=UTC)
    log_df["run_id"] = run_id
    log_df["run_as_of_ts"] = run_as_of_ts
    log_df["model_run_id"] = model_run_id
    
    # Partition by season/month (derived from game_date)
    # We can't easily append to parquet files in a thread-safe way without a proper engine (like Delta/Iceberg).
    # For now, we will write a unique file per run to avoid concurrency issues, 
    # and rely on downstream compaction if needed.
    # Or, since this is a single box, we can just write a file named by run_id.
    
    # Filename: {game_date}_{run_id}.parquet
    # If multiple dates in df, we group.
    
    for game_date, group in log_df.groupby("game_date"):
        season = _season_from_date(game_date)
        month = game_date.month
        partition_dir = logs_root / f"season={season}" / f"month={month:02d}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        
        # Use a unique name to prevent overwrites
        safe_run_id = run_id or "manual"
        filename = f"{game_date}_{safe_run_id}.parquet"
        path = partition_dir / filename
        
        group.to_parquet(path, index=False)
        # typer.echo(f"[logging] Wrote prediction log to {path}")


def score_minutes_range_to_parquet(
    start_day: date,
    end_day: date,
    *,
    features_root: Path = DEFAULT_FEATURES_ROOT,
    features_path: Path | None = None,
    bundle_dir: Path | None = None,
    bundle_config: Path = DEFAULT_BUNDLE_CONFIG,
    artifact_root: Path = DEFAULT_DAILY_ROOT,
    injuries_root: Path = DEFAULT_INJURIES_ROOT,
    schedule_root: Path = DEFAULT_SCHEDULE_ROOT,
    limit_rows: int | None = None,
    mode: Mode = "historical",
    run_id: str | None = None,
    live_features_root: Path = DEFAULT_LIVE_FEATURES_ROOT,
    minutes_output: MinutesOutputMode = "both",
    starter_priors_path: Path = DEFAULT_STARTER_PRIORS,
    starter_history_path: Path = DEFAULT_STARTER_HISTORY,
    promotion_config: Path = DEFAULT_PROMOTION_CONFIG,
    promotion_prior_enabled: bool = True,
    promotion_prior_debug: bool = False,
    reconcile_team_minutes: ReconcileMode = "none",
    reconcile_config: Path = DEFAULT_RECONCILE_CONFIG,
    reconcile_debug: bool = False,
    prediction_logs_root: Path = DEFAULT_PREDICTION_LOGS_ROOT,
    disable_play_prob: bool = False,
    target_dates: Optional[Set[date]] = None,
    debug_describe: bool | None = None,
) -> pd.DataFrame:
    """Programmatic wrapper around the scoring flow used by the CLI.

    This mirrors the historical-mode path in ``main``: load bundle, load features,
    prepare + score rows, annotate metadata, optionally reconcile, then write daily
    outputs to ``artifact_root``. Returns the full scored dataframe (all days in
    the requested range).
    """

    normalized_mode: Mode = mode.lower()  # type: ignore[assignment]
    if normalized_mode != "historical":
        raise ValueError("score_minutes_range_to_parquet currently supports historical mode only.")
    if run_id is not None:
        raise typer.BadParameter("--run-id is only valid when --mode live.", param_name="run_id")

    promotion_ctx = _load_promotion_context(
        enabled=promotion_prior_enabled,
        priors_path=starter_priors_path,
        history_path=starter_history_path,
        config_path=promotion_config,
    )

    bundle, resolved_bundle_dir, model_meta, model_run_id = _resolve_bundle_artifacts(
        bundle_dir, bundle_config
    )

    try:
        raw_features = _load_feature_slice(
            start_day,
            end_day,
            features_root=features_root,
            features_path=features_path,
            run_id=None,
        )
    except FileNotFoundError as exc:
        typer.echo(f"[minutes] ERROR: {exc}", err=True)
        raise
    if raw_features.empty:
        raise ValueError("No feature rows available for requested date range.")

    prepared = _prepare_features(raw_features, mode=normalized_mode)
    if prepared.empty:
        raise ValueError("No eligible rows found for requested date range.")
    if limit_rows is not None:
        prepared = prepared.head(limit_rows).copy()
    scored = _score_rows(
        prepared,
        bundle,
        disable_play_prob=disable_play_prob,
        promotion_ctx=promotion_ctx,
        promotion_debug=promotion_prior_debug,
    )
    scored["game_date"] = pd.to_datetime(scored["game_date"]).dt.date

    should_debug = debug_describe if debug_describe is not None else (start_day == end_day)
    if should_debug and "minutes_p50" in scored.columns:
        for day in sorted(set(scored["game_date"].tolist())):
            day_slice = scored.loc[scored["game_date"] == day]
            typer.echo(f"[minutes_debug] {day} raw minutes_p50 describe():")
            typer.echo(day_slice["minutes_p50"].describe().to_string())

    seasons: set[int] = set()
    if "season" in scored.columns:
        for value in scored["season"].dropna().unique().tolist():
            text = str(value)
            try:
                seasons.add(int(text))
                continue
            except ValueError:
                pass
            if "-" in text:
                prefix = text.split("-", 1)[0]
                try:
                    seasons.add(int(prefix))
                except ValueError:
                    continue
    if not seasons:
        seasons = {_season_from_date(start_day)}
    player_lookup = _player_name_map(injuries_root, seasons)
    team_meta = _team_metadata(schedule_root, seasons)
    if player_lookup or not team_meta.empty:
        scored = _annotate_metadata(scored, player_lookup=player_lookup, team_meta=team_meta)

    normalized_reconcile_mode = reconcile_team_minutes.lower()
    if normalized_reconcile_mode != "none":
        if normalized_reconcile_mode == "p50_and_tails":
            typer.echo(
                "[l2-reconcile] Tail reconciliation is not yet implemented; clamping only.",
                err=True,
            )
        reconcile_cfg = load_reconcile_config(reconcile_config)
        debug_hook = _make_reconcile_debugger(reconcile_debug)
        scored = reconcile_minutes_p50_all(scored, reconcile_cfg, debug_hook=debug_hook)
        if "play_prob" not in scored.columns:
            raise ValueError("play_prob column missing after reconciliation.")
        scored = apply_play_probability_mixture(
            scored,
            scored["play_prob"].to_numpy(dtype=float),
        )

    if should_debug and "minutes_p50" in scored.columns:
        for day in sorted(set(scored["game_date"].tolist())):
            day_slice = scored.loc[scored["game_date"] == day]
            typer.echo(f"[minutes_debug] {day} reconciled minutes_p50 describe():")
            typer.echo(day_slice["minutes_p50"].describe().to_string())

    for day in _iter_days(start_day, end_day):
        if target_dates is not None and day not in target_dates:
            continue
        day_df = scored.loc[scored["game_date"] == day].copy()
        _write_daily_outputs(
            day,
            day_df,
            out_root=artifact_root,
            model_run_id=model_run_id,
            bundle_dir=resolved_bundle_dir,
            model_meta=model_meta,
            run_id=None,
            run_as_of_ts=None,
            minutes_output=minutes_output,
        )

    return scored


@app.command()
def main(
    ctx: typer.Context,
    config_path: Path | None = typer.Option(
        None,
        "--config",
        help="Optional YAML config describing an inference run.",
    ),
    date: datetime | None = typer.Option(None, "--date", help="Slate date (YYYY-MM-DD)"),
    end_date: datetime | None = typer.Option(
        None,
        "--end-date",
        help="Optional inclusive end date (defaults to --date).",
    ),
    features_root: Path = typer.Option(
        DEFAULT_FEATURES_ROOT,
        "--features-root",
        help="Root directory containing season=*/month=* feature partitions (historical mode).",
    ),
    features_path: Path | None = typer.Option(
        None,
        "--features-path",
        help="Explicit parquet file/directory with features (overrides --features-root).",
    ),
    bundle_dir: Path | None = typer.Option(
        None,
        "--bundle-dir",
        help="Explicit path to the trained minutes bundle.",
    ),
    bundle_config: Path = typer.Option(
        DEFAULT_BUNDLE_CONFIG,
        "--bundle-config",
        help="JSON file pointing at the default minutes bundle.",
    ),
    artifact_root: Path = typer.Option(
        DEFAULT_DAILY_ROOT,
        "--artifact-root",
        help="Directory where daily outputs will be written.",
    ),
    injuries_root: Path = typer.Option(
        DEFAULT_INJURIES_ROOT,
        "--injuries-root",
        help="Root directory containing injuries_raw season=*/ parquet files for player names.",
    ),
    schedule_root: Path = typer.Option(
        DEFAULT_SCHEDULE_ROOT,
        "--schedule-root",
        help="Root directory containing schedule season=*/month=*/schedule.parquet for team metadata.",
    ),
    limit_rows: int | None = typer.Option(
        None,
        "--limit-rows",
        min=1,
        help="Optional limit for debugging/tests.",
    ),
    mode: Mode = typer.Option(
        "historical",
        "--mode",
        case_sensitive=False,
        help="historical (default) uses gold features; live skips starter label derivation and expects --features-path.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Optional run identifier when reading/writing run-scoped live artifacts.",
    ),
    live_features_root: Path = typer.Option(
        DEFAULT_LIVE_FEATURES_ROOT,
        "--live-features-root",
        help="Root directory containing live feature runs (used when --mode live and --features-path is not provided).",
    ),
    minutes_output: MinutesOutputMode = typer.Option(
        "both",
        "--minutes-output",
        case_sensitive=False,
        help=(
            "Which minutes columns to emit: conditional (if active), unconditional (p_play-weighted), "
            "or both (default)."
        ),
    ),
    starter_priors_path: Path = typer.Option(
        DEFAULT_STARTER_PRIORS,
        "--starter-priors",
        help="Parquet file containing starter slot priors.",
    ),
    starter_history_path: Path = typer.Option(
        DEFAULT_STARTER_HISTORY,
        "--starter-history",
        help="Parquet file containing per-player starter history counts.",
    ),
    promotion_config: Path = typer.Option(
        DEFAULT_PROMOTION_CONFIG,
        "--promotion-config",
        help="YAML file with promotion prior hyperparameters.",
    ),
    promotion_prior_enabled: bool = typer.Option(
        True,
        "--enable-promotion-prior/--disable-promotion-prior",
        help="Toggle the promotion prior calibrator.",
    ),
    promotion_prior_debug: bool = typer.Option(
        False,
        "--promotion-debug",
        help="Log promotion-prior adjustments (for inspection).",
    ),
    reconcile_team_minutes: ReconcileMode = typer.Option(
        "none",
        "--reconcile-team-minutes",
        case_sensitive=False,
        help="none skips L2, p50 reconciles medians, p50_and_tails reserves hooks for future tail logic.",
    ),
    reconcile_config: Path = typer.Option(
        DEFAULT_RECONCILE_CONFIG,
        "--reconcile-config",
        help="YAML config with L2 reconciliation parameters.",
    ),
    reconcile_debug: bool = typer.Option(
        False,
        "--reconcile-debug",
        help="Log per-team reconciliation summaries and top deltas.",
    ),
    prediction_logs_root: Path = typer.Option(
        DEFAULT_PREDICTION_LOGS_ROOT,
        "--prediction-logs-root",
        help="Root directory for long-term prediction logs.",
    ),
    disable_play_prob: bool = typer.Option(
        False,
        "--disable-play-prob",
        help="Skip play probability usage; emit only conditional minutes (uncond columns unchanged).",
        is_flag=True,
    ),
) -> None:
    cli_params: dict[str, Any] = {
        "date": date,
        "end_date": end_date,
        "features_root": features_root,
        "features_path": features_path,
        "bundle_dir": bundle_dir,
        "bundle_config": bundle_config,
        "artifact_root": artifact_root,
        "injuries_root": injuries_root,
        "schedule_root": schedule_root,
        "limit_rows": limit_rows,
        "mode": mode,
        "run_id": run_id,
        "live_features_root": live_features_root,
        "minutes_output": minutes_output,
        "starter_priors_path": starter_priors_path,
        "starter_history_path": starter_history_path,
        "promotion_config": promotion_config,
        "promotion_prior_enabled": promotion_prior_enabled,
        "promotion_prior_debug": promotion_prior_debug,
        "reconcile_team_minutes": reconcile_team_minutes,
        "reconcile_config": reconcile_config,
        "reconcile_debug": reconcile_debug,
        "prediction_logs_root": prediction_logs_root,
        "disable_play_prob": disable_play_prob,
    }
    resolved_params = _apply_scoring_overrides(ctx, cli_params, config_path)
    date = resolved_params["date"]
    end_date = resolved_params["end_date"]
    features_root = resolved_params["features_root"]
    features_path = resolved_params["features_path"]
    bundle_dir = resolved_params["bundle_dir"]
    bundle_config = resolved_params["bundle_config"]
    artifact_root = resolved_params["artifact_root"]
    injuries_root = resolved_params["injuries_root"]
    schedule_root = resolved_params["schedule_root"]
    limit_rows = resolved_params["limit_rows"]
    mode = resolved_params["mode"]
    run_id = resolved_params["run_id"]
    live_features_root = resolved_params["live_features_root"]
    minutes_output = resolved_params["minutes_output"]
    starter_priors_path = resolved_params["starter_priors_path"]
    starter_history_path = resolved_params["starter_history_path"]
    promotion_config = resolved_params["promotion_config"]
    promotion_prior_enabled = resolved_params["promotion_prior_enabled"]
    promotion_prior_debug = resolved_params["promotion_prior_debug"]
    reconcile_team_minutes = resolved_params["reconcile_team_minutes"]
    reconcile_config = resolved_params["reconcile_config"]
    reconcile_config = resolved_params["reconcile_config"]
    reconcile_debug = resolved_params["reconcile_debug"]
    prediction_logs_root = cli_params.get("prediction_logs_root", DEFAULT_PREDICTION_LOGS_ROOT)
    disable_play_prob = resolved_params.get("disable_play_prob", False)

    if date is None:
        raise typer.BadParameter("--date is required (set via CLI or config file).")

    start_day = date.date()
    final_day = end_date.date() if end_date else start_day
    if final_day < start_day:
        raise typer.BadParameter("--end-date cannot be before --date")

    try:
        bundle, resolved_bundle_dir, model_meta, model_run_id = _resolve_bundle_artifacts(
            bundle_dir, bundle_config
        )
    except FileNotFoundError as exc:
        typer.echo(f"[minutes] ERROR: {exc}", err=True)
        raise typer.Exit(code=1)

    features_root = features_root.expanduser()
    features_path = features_path.expanduser() if features_path else None
    live_features_root = live_features_root.expanduser()
    artifact_root = artifact_root.expanduser()
    injuries_root = injuries_root.expanduser()
    schedule_root = schedule_root.expanduser()
    promotion_ctx = _load_promotion_context(
        enabled=promotion_prior_enabled,
        priors_path=starter_priors_path,
        history_path=starter_history_path,
        config_path=promotion_config,
    )

    run_dir_path: Path | None = None
    run_summary: dict | None = None
    run_as_of_ts_value: pd.Timestamp | None = None

    normalized_mode: Mode = mode.lower()  # type: ignore[assignment]
    if normalized_mode == "live":
        if features_path is None:
            run_dir_path = _resolve_run_dir(live_features_root, start_day, run_id, write_mode=False)
            features_path = run_dir_path / FEATURE_FILENAME
            if not features_path.exists():
                raise FileNotFoundError(f"Live features parquet missing at {features_path}")
        else:
            if features_path.is_dir():
                run_dir_path = features_path
                features_path = run_dir_path / FEATURE_FILENAME
            else:
                run_dir_path = features_path.parent
            if run_dir_path.name.startswith("run=") and run_id is None:
                run_id = run_dir_path.name.split("=", 1)[1]
        summary_path = run_dir_path / SUMMARY_FILENAME if run_dir_path else None
        if summary_path and summary_path.exists():
            try:
                run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                run_as_of_str = run_summary.get("run_as_of_ts")
                if run_as_of_str:
                    run_as_of_ts_value = pd.to_datetime(run_as_of_str, utc=True)
            except (json.JSONDecodeError, TypeError, ValueError):
                run_summary = None
    elif run_id is not None:
        raise typer.BadParameter("--run-id is only valid when --mode live.", param_name="run_id")

    run_as_of_datetime = run_as_of_ts_value.to_pydatetime() if run_as_of_ts_value is not None else None

    try:
        raw_features = _load_feature_slice(
            start_day,
            final_day,
            features_root=features_root,
            features_path=features_path,
            run_id=run_id if normalized_mode == "live" else None,
        )
    except FileNotFoundError as exc:
        typer.echo(f"[minutes] ERROR: {exc}", err=True)
        raise typer.Exit(code=1)
    if raw_features.empty:
        typer.echo(
            "[minutes] ERROR: no feature rows available for requested date range.",
            err=True,
        )
        raise typer.Exit(code=1)
    prepared = _prepare_features(raw_features, mode=normalized_mode)
    if prepared.empty:
        typer.echo("[minutes] ERROR: no eligible rows found for requested date range.", err=True)
        raise typer.Exit(code=1)
    if limit_rows is not None:
        prepared = prepared.head(limit_rows).copy()
    scored = _score_rows(
        prepared,
        bundle,
        disable_play_prob=disable_play_prob,
        promotion_ctx=promotion_ctx,
        promotion_debug=promotion_prior_debug,
    )
    scored["game_date"] = pd.to_datetime(scored["game_date"]).dt.date
    seasons: set[int] = set()
    if "season" in scored.columns:
        for value in scored["season"].dropna().unique().tolist():
            text = str(value)
            try:
                seasons.add(int(text))
                continue
            except ValueError:
                pass
            if "-" in text:
                prefix = text.split("-", 1)[0]
                try:
                    seasons.add(int(prefix))
                except ValueError:
                    continue
    if not seasons:
        seasons = {_season_from_date(start_day)}
    player_lookup = _player_name_map(injuries_root, seasons)
    team_meta = _team_metadata(schedule_root, seasons)
    if player_lookup or not team_meta.empty:
        scored = _annotate_metadata(scored, player_lookup=player_lookup, team_meta=team_meta)

    normalized_reconcile_mode = reconcile_team_minutes.lower()
    if normalized_reconcile_mode != "none":
        if normalized_reconcile_mode == "p50_and_tails":
            typer.echo(
                "[l2-reconcile] Tail reconciliation is not yet implemented; clamping only.",
                err=True,
            )
        reconcile_cfg = load_reconcile_config(reconcile_config)
        debug_hook = _make_reconcile_debugger(reconcile_debug)
        scored = reconcile_minutes_p50_all(scored, reconcile_cfg, debug_hook=debug_hook)
        if "play_prob" not in scored.columns:
            raise ValueError("play_prob column missing after reconciliation.")
        scored = apply_play_probability_mixture(
            scored,
            scored["play_prob"].to_numpy(dtype=float),
        )

    # Log predictions (Phase 3 Hardening)
    if normalized_mode == "live":
        _log_predictions(
            scored,
            logs_root=prediction_logs_root,
            run_id=run_id,
            run_as_of_ts=run_as_of_datetime,
            model_run_id=model_run_id,
        )

    for day in _iter_days(start_day, final_day):
        day_df = scored.loc[scored["game_date"] == day].copy()
        columns_to_keep = [
            "game_date",
            "tip_ts",
            "game_id",
            "player_id",
            "status",
            "player_name",
            "team_id",
            "team_name",
            "team_tricode",
            "opponent_team_id",
            "opponent_team_name",
            "opponent_team_tricode",
            "starter_flag",
            "is_starter",
            "is_projected_starter",
            "is_confirmed_starter",
            "pos_bucket",
            "spread_home",
            "total",
            "odds_as_of_ts",
            "blowout_index",
            "blowout_risk_score",
            "close_game_score",
            "play_prob",
            "minutes_p10",
            "minutes_p50",
            "minutes_p90",
            "minutes_p10_cond",
            "minutes_p50_cond",
            "minutes_p90_cond",
            "minutes_p10_uncond",
            "minutes_p50_uncond",
            "minutes_p90_uncond",
        ]
        available_cols = [col for col in columns_to_keep if col in day_df.columns]
        day_df = day_df.loc[:, available_cols]
        _write_daily_outputs(
            day,
            day_df,
            out_root=artifact_root,
            model_run_id=model_run_id,
            bundle_dir=resolved_bundle_dir,
            model_meta=model_meta,
            run_id=run_id if normalized_mode == "live" else None,
            run_as_of_ts=run_as_of_datetime,
            minutes_output=minutes_output,
        )


if __name__ == "__main__":  # pragma: no cover
    app()
