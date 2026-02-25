"""V3 live pipeline scaffold for GameTransformerV2 integration.

This flow intentionally keeps a minimal critical path and strict gate boundaries:
- scrape core inputs
- freeze run manifest
- build features
- preflight parity validation
- score model
- generate worlds
- finalize projections
- postflight contract validation
- atomic pointer publish

The current implementation supports placeholder mode for end-to-end dev plumbing
while strict preflight/postflight gates and parity checks are enforced.
"""

from __future__ import annotations

# ruff: noqa: E402

import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from prefect import flow, get_run_logger, task
from torch.utils.data import DataLoader
from zoneinfo import ZoneInfo

from projections import model_selectors, paths
from projections.etl import storage as bronze_storage
from projections.features.action_props import (
    build_action_props_feature_snapshots,
    load_action_props_feature_snapshots_for_date,
    load_rotowire_props_long_from_bronze,
)
from projections.pipeline import control_plane, writer_guard
from projections.pipeline.gtv2_live_features import (
    build_gtv2_live_features,
    load_gtv2_feature_spec,
)
from projections.pipeline.parity_manifest import (
    build_parity_manifest,
    hash_paths,
    load_parity_manifest,
    resolve_parity_manifest_path,
    stable_json_sha256,
    write_parity_manifest,
)
from projections.pipeline.v3_postflight import run_postflight_gate
from projections.pipeline.v3_preflight import run_preflight_gate
from projections.rotation.game_transformer_v2 import (
    GameLevelDataset,
    GameTransformerV2Config,
    build_game_level_examples,
    build_game_transformer_v2,
    collate_game_level_examples,
)
from projections.rotation.sample_worlds_v2 import (
    sample_worlds_for_batch,
    summarize_worlds_to_projections,
)
from projections.runtime_stamp import enforce_clean_tree, enforce_prod_sanity, log_runtime_stamp


PROJECT_ROOT = paths.get_project_root()
_DEFAULT_UV_PATH = Path("/home/daniel/.local/bin/uv")

FEATURES_ROOT = "features_gtv2_v1"
SCORES_ROOT = "gtv2_scores"
WORLDS_ROOT = "gtv2_worlds"

PLACEHOLDER_PROJECTION_COLUMNS = [
    "game_date",
    "game_id",
    "team_id",
    "player_id",
    "minutes_sim_mean",
    "minutes_sim_p50",
    "dk_fpts_mean",
    "dk_fpts_p50",
    "sim_p_active",
    "n_worlds",
    "sim_profile",
]


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _cli_compatible_ts(ts_value: str) -> str:
    ts = pd.to_datetime(ts_value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise RuntimeError(f"invalid timestamp: {ts_value}")
    ts_utc = pd.Timestamp(ts).tz_convert("UTC").tz_localize(None)
    return ts_utc.strftime("%Y-%m-%dT%H:%M:%S")


def _uv_bin() -> str:
    env_uv = os.environ.get("UV_BIN")
    if env_uv:
        if Path(env_uv).exists():
            return env_uv
        raise FileNotFoundError(f"UV_BIN={env_uv} specified but file does not exist")
    if _DEFAULT_UV_PATH.exists():
        return str(_DEFAULT_UV_PATH)
    which_uv = shutil.which("uv")
    if which_uv:
        return which_uv
    raise FileNotFoundError("Could not find 'uv' executable")


def _run_python_module(
    module: str,
    args: list[str],
    *,
    data_root: Path,
    timeout_s: int,
) -> None:
    env = os.environ.copy()
    env["PROJECTIONS_DATA_ROOT"] = str(data_root)
    cmd = [_uv_bin(), "run", "python", "-m", module, *args]
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
        raise RuntimeError(f"{module} failed with exit_code={result.returncode}")


def _resolve_game_date(game_date: str | None) -> str:
    if game_date is not None:
        return str(game_date)
    et = ZoneInfo("America/New_York")
    return datetime.now(tz=et).date().isoformat()


def _resolve_season_month(game_date: str) -> tuple[int, int]:
    ts = pd.Timestamp(game_date)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    season = int(ts.year) if int(ts.month) >= 8 else int(ts.year) - 1
    return season, int(ts.month)


def _resolve_bundle_dir(*, data_root: Path, gtv2_bundle_dir: str | None) -> Path:
    if gtv2_bundle_dir:
        return Path(gtv2_bundle_dir).expanduser().resolve()
    env = os.environ.get("PROJECTIONS_GTV2_BUNDLE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (data_root / "artifacts" / "game_transformer_v2" / "bundle_current").resolve()


def _placeholder_feature_frame(*, game_date: str, as_of_ts: str) -> pd.DataFrame:
    game_id = 900001
    team_a = 100
    team_b = 200
    rows: list[dict[str, Any]] = []
    for idx in range(20):
        team_id = team_a if idx < 10 else team_b
        local_idx = idx if idx < 10 else idx - 10
        rows.append(
            {
                "game_date": game_date,
                "game_id": game_id,
                "team_id": team_id,
                "player_id": 1000 + idx,
                "lineup_available": 1,
                "lineup_starter_announced": 1 if local_idx < 5 else 0,
                "vegas_total": 228.5,
                "vegas_spread": -2.5,
                "estimated_possessions": 99.4,
                "minutes_prior": 26.0 - float(local_idx),
                "usage_prior": 0.22 - 0.005 * float(local_idx),
                "as_of_ts": as_of_ts,
            }
        )
    return pd.DataFrame(rows)


def _ensure_placeholder_bundle(
    *,
    bundle_dir: Path,
    features_df: pd.DataFrame,
    transform_manifest: dict[str, Any],
    integrity: dict[str, Any],
) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    stub = bundle_dir / "bundle_stub.txt"
    if not stub.exists():
        stub.write_text("placeholder bundle", encoding="utf-8")

    manifest_path = resolve_parity_manifest_path(bundle_dir)
    manifest_payload = build_parity_manifest(
        model_id="game_transformer_v2_placeholder",
        features_df=features_df,
        feature_columns=list(features_df.columns),
        missing_value_policy={"disallow_null_columns": list(features_df.columns)},
        transform_manifest=transform_manifest,
        output_manifest={
            "projection_columns": list(PLACEHOLDER_PROJECTION_COLUMNS),
            "semantics": {
                "dk_fpts_mean": "conditional_on_active",
                "minutes_sim_mean": "conditional_on_active",
            },
        },
        integrity=integrity,
    )
    write_parity_manifest(manifest_path, manifest_payload)
    return manifest_path


def _bundle_artifact_hash(bundle_dir: Path) -> str:
    if not bundle_dir.exists():
        return stable_json_sha256([])
    files = [p for p in bundle_dir.rglob("*") if p.is_file()]
    return hash_paths(files)


def _set_inference_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _resolve_torch_device(device: str | None) -> torch.device:
    if device:
        return torch.device(str(device))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_gtv2_model(bundle_dir: Path, *, device: torch.device) -> tuple[GameTransformerV2Config, torch.nn.Module]:
    config_path = Path(bundle_dir) / "config.json"
    model_path = Path(bundle_dir) / "model.pt"
    if not config_path.exists():
        raise RuntimeError(f"missing bundle config: {config_path}")
    if not model_path.exists():
        raise RuntimeError(f"missing bundle model: {model_path}")

    config = GameTransformerV2Config.load(config_path)
    model = build_game_transformer_v2(config)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device=device)
    model.eval()
    return config, model


def _coerce_frame_to_manifest_schema(features_df: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    schema = manifest.get("feature_schema")
    if not isinstance(schema, list) or len(schema) <= 0:
        raise RuntimeError("parity manifest missing feature_schema")

    out = features_df.copy()
    ordered_cols: list[str] = []
    for row in schema:
        col = str(row.get("name"))
        dtype = str(row.get("dtype"))
        nullable = bool(row.get("nullable", True))
        if col not in out.columns:
            raise RuntimeError(f"feature frame missing manifest column: {col}")
        series = out[col]
        try:
            if dtype in {"int64", "int32", "int16", "int8", "Int64", "Int32", "Int16", "Int8"}:
                series = pd.to_numeric(series, errors="coerce").astype(dtype)
            elif dtype in {"float64", "float32", "float16"}:
                series = pd.to_numeric(series, errors="coerce").astype(dtype)
            elif dtype in {"bool", "boolean"}:
                if dtype == "bool":
                    series = series.fillna(False).astype(bool)
                else:
                    series = series.astype("boolean")
            elif dtype.startswith("datetime64"):
                utc = "UTC" in dtype
                series = pd.to_datetime(series, errors="coerce", utc=utc)
                if not utc:
                    series = series.dt.tz_localize(None)
            else:
                series = series.astype(dtype)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"failed to coerce feature column '{col}' to dtype '{dtype}': {exc}") from exc

        if not nullable and bool(series.isna().any()):
            raise RuntimeError(f"non-nullable feature column has nulls after coercion: {col}")
        out[col] = series
        ordered_cols.append(col)

    return out.loc[:, ordered_cols].copy()


def _build_gtv2_inference_examples(
    *,
    features_df: pd.DataFrame,
    game_date: str,
    config: GameTransformerV2Config,
) -> list[Any]:
    frame = features_df.copy()
    frame["game_date"] = pd.Timestamp(game_date).normalize()
    frame["minutes"] = 0.0

    examples = build_game_level_examples(
        frame,
        feature_columns=list(config.feature_columns),
        feature_mean=np.asarray(config.feature_mean, dtype=np.float32),
        feature_std=np.asarray(config.feature_std, dtype=np.float32),
        game_feature_columns=list(config.game_feature_columns),
        team_feature_columns=list(config.team_feature_columns),
        flow_label_columns=[],
        minutes_label_col="minutes",
        min_valid_players_per_team=max(1, int(config.min_active_count)),
        overflow_protected_prior_play_prob_floor=float(config.overflow_protected_prior_play_prob_floor),
        overflow_protected_prior_minutes_floor=float(config.overflow_protected_prior_minutes_floor),
        overflow_risk_weight_consecutive_active_dnp=float(config.overflow_risk_weight_consecutive_active_dnp),
        overflow_risk_weight_active_but_dnp_rate_last10=float(config.overflow_risk_weight_active_but_dnp_rate_last10),
        overflow_risk_weight_inactive_streak_len=float(config.overflow_risk_weight_inactive_streak_len),
        overflow_keep_weight_prior_play_prob=float(config.overflow_keep_weight_prior_play_prob),
        overflow_keep_weight_prior_minutes=float(config.overflow_keep_weight_prior_minutes),
    )
    if not examples:
        raise RuntimeError("no valid game examples produced for GTV2 inference")
    return examples


def _selected_props_source_from_checklist(checklist: dict[str, Any]) -> str | None:
    checks = checklist.get("checks")
    if not isinstance(checks, list):
        return None
    for entry in checks:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("name")) != "props_source_policy_satisfied":
            continue
        details = entry.get("details")
        if isinstance(details, dict):
            val = details.get("selected_source")
            return None if val is None else str(val)
    return None


def _read_parquet_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _filter_slate_rows(df: pd.DataFrame, game_ids: list[int]) -> pd.DataFrame:
    if df.empty or not game_ids or "game_id" not in df.columns:
        return df.iloc[0:0].copy()
    gids = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
    return df.loc[gids.isin(game_ids)].copy()


def _latest_ts(df: pd.DataFrame, *, time_col: str = "as_of_ts") -> pd.Timestamp | None:
    if df.empty or time_col not in df.columns:
        return None
    ts = pd.to_datetime(df[time_col], utc=True, errors="coerce").dropna()
    if ts.empty:
        return None
    return pd.Timestamp(ts.max())


def _build_feature_input_checklist(
    *,
    game_date: str,
    run_as_of_ts: str,
    data_root: Path,
    allow_priors_fallback: bool,
    allow_rotowire_props_fallback: bool = False,
    require_action_props: bool = True,
    action_props_loader: Callable[..., pd.DataFrame] = load_action_props_feature_snapshots_for_date,
) -> dict[str, Any]:
    day = pd.Timestamp(game_date).normalize()
    run_ts = pd.to_datetime(run_as_of_ts, utc=True, errors="coerce")
    if pd.isna(run_ts):
        raise RuntimeError(f"invalid run_as_of_ts: {run_as_of_ts}")

    season, month = _resolve_season_month(game_date)
    schedule_path = data_root / "silver" / "schedule" / f"season={season}" / f"month={month:02d}" / "schedule.parquet"
    roster_path = data_root / "silver" / "roster_nightly" / f"season={season}" / f"month={month:02d}" / "roster.parquet"
    odds_path = data_root / "silver" / "odds_snapshot" / f"season={season}" / f"month={month:02d}" / "odds_snapshot.parquet"
    injuries_silver_path = (
        data_root / "silver" / "injuries_snapshot" / f"season={season}" / f"month={month:02d}" / "injuries_snapshot.parquet"
    )
    rotowire_path = data_root / "silver" / "rotowire_lineups" / f"date={day.date()}" / "lineups.parquet"
    labels_gold_root = data_root / "gold" / "labels_minutes_v1" / f"season={season}"
    labels_legacy_path = data_root / "labels" / f"season={season}" / "boxscore_labels.parquet"
    priors_team_root = data_root / "silver" / "rotation_priors_v1" / "team_game_priors" / f"season={season}"
    priors_player_root = data_root / "silver" / "rotation_priors_v1" / "player_game_priors" / f"season={season}"

    checks: list[dict[str, Any]] = []

    schedule_df = _read_parquet_if_exists(schedule_path)
    schedule_days = pd.to_datetime(schedule_df.get("game_date"), errors="coerce").dt.normalize() if not schedule_df.empty else pd.Series(dtype="datetime64[ns]")
    slate_df = schedule_df.loc[schedule_days == day].copy() if not schedule_df.empty else pd.DataFrame()
    slate_game_ids = (
        pd.to_numeric(slate_df.get("game_id"), errors="coerce").dropna().astype(int).unique().tolist()
        if not slate_df.empty
        else []
    )
    checks.append(
        {
            "name": "schedule_slate_rows",
            "required": True,
            "ok": bool(not slate_df.empty and len(slate_game_ids) > 0),
            "details": {
                "path": str(schedule_path),
                "rows_total": int(len(schedule_df)),
                "rows_slate": int(len(slate_df)),
                "games_slate": int(len(slate_game_ids)),
            },
        }
    )

    def _snapshot_check(name: str, path: Path, *, required: bool = True) -> pd.DataFrame:
        df = _read_parquet_if_exists(path)
        slate_rows = _filter_slate_rows(df, slate_game_ids)
        latest = _latest_ts(slate_rows)
        age_minutes = None
        if latest is not None:
            age_minutes = float((run_ts - latest).total_seconds() / 60.0)
        checks.append(
            {
                "name": name,
                "required": required,
                "ok": bool(not slate_rows.empty),
                "details": {
                    "path": str(path),
                    "rows_total": int(len(df)),
                    "rows_slate": int(len(slate_rows)),
                    "latest_as_of_ts": None if latest is None else latest.isoformat(),
                    "age_minutes": age_minutes,
                },
            }
        )
        return slate_rows

    _ = _snapshot_check("roster_snapshot_slate_rows", roster_path, required=True)
    _ = _snapshot_check("odds_snapshot_slate_rows", odds_path, required=True)
    injuries_silver_slate = _snapshot_check("injuries_snapshot_silver_slate_rows", injuries_silver_path, required=False)

    # Bronze injuries are preferred by build_minutes_live; verify at least one injuries source has slate rows.
    bronze_frames: list[pd.DataFrame] = []
    for offset in (-1, 0, 1):
        day_i = (day + pd.Timedelta(days=offset)).date()
        frame = bronze_storage.read_bronze_day(
            "injuries_raw",
            data_root,
            season,
            day_i,
            include_runs=False,
            prefer_history=True,
        )
        if not frame.empty:
            bronze_frames.append(frame)
    injuries_bronze = pd.concat(bronze_frames, ignore_index=True) if bronze_frames else pd.DataFrame()
    injuries_bronze_slate = _filter_slate_rows(injuries_bronze, slate_game_ids)
    injury_rows_ok = bool(not injuries_bronze_slate.empty or not injuries_silver_slate.empty)
    checks.append(
        {
            "name": "injuries_any_source_slate_rows",
            "required": True,
            "ok": injury_rows_ok,
            "details": {
                "bronze_rows_window": int(len(injuries_bronze)),
                "bronze_rows_slate": int(len(injuries_bronze_slate)),
                "silver_rows_slate": int(len(injuries_silver_slate)),
            },
        }
    )

    rotowire_df = _read_parquet_if_exists(rotowire_path)
    checks.append(
        {
            "name": "rotowire_lineups_file",
            "required": False,
            "ok": bool(not rotowire_df.empty),
            "details": {
                "path": str(rotowire_path),
                "rows": int(len(rotowire_df)),
            },
        }
    )

    gold_exists = labels_gold_root.exists() and any(labels_gold_root.rglob("*.parquet"))
    legacy_exists = labels_legacy_path.exists()
    checks.append(
        {
            "name": "labels_source_available",
            "required": True,
            "ok": bool(gold_exists or legacy_exists),
            "details": {
                "gold_root": str(labels_gold_root),
                "gold_exists": bool(gold_exists),
                "legacy_path": str(labels_legacy_path),
                "legacy_exists": bool(legacy_exists),
            },
        }
    )

    action_props_dir = data_root / "bronze" / "action_network" / "props"
    rotowire_props_root = data_root / "bronze" / "props"
    action_props_day = day.date().isoformat()
    action_props_next_day = (day + pd.Timedelta(days=1)).date().isoformat()
    raw_action_props_files = sorted(action_props_dir.glob(f"{action_props_day}_*.json")) + sorted(
        action_props_dir.glob(f"{action_props_next_day}_*.json")
    )
    action_props_snapshots = pd.DataFrame()
    action_props_parse_error: str | None = None
    if action_props_dir.exists():
        try:
            day_snap = action_props_loader(props_dir=action_props_dir, game_date=day)
            next_snap = action_props_loader(props_dir=action_props_dir, game_date=day + pd.Timedelta(days=1))
            frames = [df for df in (day_snap, next_snap) if isinstance(df, pd.DataFrame) and not df.empty]
            action_props_snapshots = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        except Exception as exc:  # noqa: BLE001
            action_props_parse_error = str(exc)

    rotowire_raw_files = sorted((rotowire_props_root / f"game_date={action_props_day}").glob("*.parquet")) + sorted(
        (rotowire_props_root / f"game_date={action_props_next_day}").glob("*.parquet")
    )
    rotowire_snapshots = pd.DataFrame()
    rotowire_parse_error: str | None = None
    if rotowire_props_root.exists():
        try:
            rw_day_long = load_rotowire_props_long_from_bronze(
                rotowire_props_root=rotowire_props_root,
                game_date=day,
            )
            rw_next_long = load_rotowire_props_long_from_bronze(
                rotowire_props_root=rotowire_props_root,
                game_date=day + pd.Timedelta(days=1),
            )
            rw_day_snap = build_action_props_feature_snapshots(rw_day_long)
            rw_next_snap = build_action_props_feature_snapshots(rw_next_long)
            rw_frames = [df for df in (rw_day_snap, rw_next_snap) if isinstance(df, pd.DataFrame) and not df.empty]
            rotowire_snapshots = pd.concat(rw_frames, ignore_index=True) if rw_frames else pd.DataFrame()
        except Exception as exc:  # noqa: BLE001
            rotowire_parse_error = str(exc)

    checks.append(
        {
            "name": "action_network_props_raw_files",
            "required": False,
            "ok": bool(len(raw_action_props_files) > 0),
            "details": {
                "dir": str(action_props_dir),
                "day_file_glob": f"{action_props_day}_*.json",
                "next_day_file_glob": f"{action_props_next_day}_*.json",
                "raw_file_count": int(len(raw_action_props_files)),
            },
        }
    )
    latest_action_props_ts = _latest_ts(action_props_snapshots, time_col="action_props_as_of_ts")
    checks.append(
        {
            "name": "action_network_props_parsed_snapshots",
            "required": False,
            "ok": bool(not action_props_snapshots.empty and action_props_parse_error is None),
            "details": {
                "parsed_rows": int(len(action_props_snapshots)),
                "latest_action_props_as_of_ts": None
                if latest_action_props_ts is None
                else latest_action_props_ts.isoformat(),
                "parse_error": action_props_parse_error,
            },
        }
    )
    checks.append(
        {
            "name": "rotowire_props_raw_files",
            "required": False,
            "ok": bool(len(rotowire_raw_files) > 0),
            "details": {
                "root": str(rotowire_props_root),
                "day_partition": str(rotowire_props_root / f"game_date={action_props_day}"),
                "next_day_partition": str(rotowire_props_root / f"game_date={action_props_next_day}"),
                "raw_file_count": int(len(rotowire_raw_files)),
            },
        }
    )
    latest_rotowire_props_ts = _latest_ts(rotowire_snapshots, time_col="action_props_as_of_ts")
    checks.append(
        {
            "name": "rotowire_props_parsed_snapshots",
            "required": False,
            "ok": bool(not rotowire_snapshots.empty and rotowire_parse_error is None),
            "details": {
                "parsed_rows": int(len(rotowire_snapshots)),
                "latest_action_props_as_of_ts": None
                if latest_rotowire_props_ts is None
                else latest_rotowire_props_ts.isoformat(),
                "parse_error": rotowire_parse_error,
            },
        }
    )
    action_ok = bool(not action_props_snapshots.empty and action_props_parse_error is None)
    rotowire_ok = bool(not rotowire_snapshots.empty and rotowire_parse_error is None)
    policy_ok = (not require_action_props) or action_ok or (allow_rotowire_props_fallback and rotowire_ok)
    checks.append(
        {
            "name": "props_source_policy_satisfied",
            "required": True,
            "ok": bool(policy_ok),
            "details": {
                "require_action_props": bool(require_action_props),
                "allow_rotowire_props_fallback": bool(allow_rotowire_props_fallback),
                "action_network_ok": bool(action_ok),
                "rotowire_ok": bool(rotowire_ok),
                "selected_source": (
                    "action_network"
                    if action_ok
                    else ("rotowire_fallback" if allow_rotowire_props_fallback and rotowire_ok else "none")
                ),
            },
        }
    )

    team_partitions = list(priors_team_root.glob("game_id=*.parquet")) if priors_team_root.exists() else []
    player_partitions = list(priors_player_root.glob("game_id=*.parquet")) if priors_player_root.exists() else []
    checks.append(
        {
            "name": "rotation_priors_roots_nonempty",
            "required": True,
            "ok": bool(team_partitions and player_partitions),
            "details": {
                "team_root": str(priors_team_root),
                "player_root": str(priors_player_root),
                "team_partition_count": int(len(team_partitions)),
                "player_partition_count": int(len(player_partitions)),
            },
        }
    )

    checks.append(
        {
            "name": "rotation_priors_mode_explicit",
            "required": True,
            "ok": True,
            "details": {
                "allow_priors_fallback": bool(allow_priors_fallback),
                "mode": "game_id_partitions_or_latest_by_entity_fallback",
                "explanation": (
                    "Live slates commonly have no same-day game_id priors partitions pre-tip. "
                    "Fallback uses latest completed-game priors by team/player (not future info)."
                ),
            },
        }
    )

    missing_team: list[str] = []
    missing_player: list[str] = []
    for gid in slate_game_ids:
        gid_norm = str(int(gid)).zfill(10)
        if not (priors_team_root / f"game_id={gid_norm}.parquet").exists():
            missing_team.append(gid_norm)
        if not (priors_player_root / f"game_id={gid_norm}.parquet").exists():
            missing_player.append(gid_norm)
    all_gameid_missing = bool(slate_game_ids) and len(missing_team) == len(slate_game_ids) and len(missing_player) == len(slate_game_ids)
    checks.append(
        {
            "name": "rotation_priors_gameid_partition_coverage",
            "required": False,
            "ok": bool(not all_gameid_missing),
            "details": {
                "slate_games": int(len(slate_game_ids)),
                "present_team_partitions": int(len(slate_game_ids) - len(missing_team)),
                "present_player_partitions": int(len(slate_game_ids) - len(missing_player)),
                "missing_team_partitions": int(len(missing_team)),
                "missing_player_partitions": int(len(missing_player)),
            },
        }
    )
    checks.append(
        {
            "name": "rotation_priors_policy_allows_current_coverage",
            "required": True,
            "ok": bool((not all_gameid_missing) or allow_priors_fallback),
            "details": {
                "allow_priors_fallback": bool(allow_priors_fallback),
                "slate_games": int(len(slate_game_ids)),
                "missing_team_partitions": int(len(missing_team)),
                "missing_player_partitions": int(len(missing_player)),
                "all_gameid_partitions_missing": bool(all_gameid_missing),
                "note": (
                    "If all game_id partitions are missing and fallback is disabled, this fails closed "
                    "to avoid ambiguous priors behavior."
                ),
            },
        }
    )

    failed_required = [c["name"] for c in checks if bool(c.get("required")) and not bool(c.get("ok"))]
    return {
        "builder_input_checklist_version": 1,
        "game_date": game_date,
        "season": int(season),
        "month": int(month),
        "run_as_of_ts": str(run_ts.isoformat()),
        "checks": checks,
        "failed_required_checks": failed_required,
    }


@task(name="scrape-core-inputs", retries=1, retry_delay_seconds=30)
def scrape_core_inputs_task(
    *,
    game_date: str,
    data_root: Path,
    placeholder_mode: bool,
    require_action_props: bool,
    allow_rotowire_props_fallback: bool,
    replay_mode: bool = False,
) -> Path:
    marker = data_root / "bronze" / "v3_core_inputs" / f"date={game_date}" / "core_inputs_ready.json"
    marker.parent.mkdir(parents=True, exist_ok=True)

    if placeholder_mode:
        payload = {
            "game_date": game_date,
            "placeholder_mode": True,
            "completed_at": _utc_now_iso(),
        }
        marker.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return marker

    if replay_mode:
        payload = {
            "game_date": game_date,
            "placeholder_mode": False,
            "replay_mode": True,
            "completed_at": _utc_now_iso(),
            "action_props_required": bool(require_action_props),
            "allow_rotowire_props_fallback": bool(allow_rotowire_props_fallback),
            "note": "scrape step skipped in replay_mode; existing snapshots are used",
        }
        marker.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return marker

    season, month = _resolve_season_month(game_date)
    schedule_path = (
        data_root
        / "silver"
        / "schedule"
        / f"season={season}"
        / f"month={month:02d}"
        / "schedule.parquet"
    )
    args = [
        "--start",
        game_date,
        "--end",
        game_date,
        "--season",
        str(season),
        "--month",
        str(month),
        "--data-root",
        str(data_root),
    ]
    if schedule_path.exists():
        args.extend(["--schedule", str(schedule_path)])
    _run_python_module(
        "projections.cli.live_pipeline",
        args,
        data_root=data_root,
        timeout_s=900,
    )

    action_props_status: dict[str, str] = {"scrape_props_cli": "not_run", "action_network_backfill": "not_run"}
    try:
        _run_python_module(
            "projections.cli.scrape_props",
            ["scrape", "--date", game_date],
            data_root=data_root,
            timeout_s=300,
        )
        action_props_status["scrape_props_cli"] = "ok"
    except Exception as exc:  # noqa: BLE001
        action_props_status["scrape_props_cli"] = f"failed: {exc}"
        if require_action_props and not allow_rotowire_props_fallback:
            raise RuntimeError(
                "action props scrape failed and require_action_props=True with no Rotowire fallback: "
                f"{exc}"
            ) from exc

    try:
        action_backfill_timeout_s = 240 if allow_rotowire_props_fallback else 1200
        _run_python_module(
            "scrapers.action_network.props_backfill",
            [
                "--start-date",
                game_date,
                "--end-date",
                game_date,
                "--workers",
                "40",
                "--refresh-older-than-minutes",
                "20",
            ],
            data_root=data_root,
            timeout_s=action_backfill_timeout_s,
        )
        action_props_status["action_network_backfill"] = "ok"
    except Exception as exc:  # noqa: BLE001
        action_props_status["action_network_backfill"] = f"failed: {exc}"
        if require_action_props and not allow_rotowire_props_fallback:
            raise RuntimeError(
                "Action Network props backfill failed and require_action_props=True with no Rotowire fallback: "
                f"{exc}"
            ) from exc

    props_dir = data_root / "bronze" / "action_network" / "props"
    day = pd.Timestamp(game_date).normalize()
    raw_props_files = sorted(props_dir.glob(f"{day.date().isoformat()}_*.json"))
    _run_python_module(
        "scripts.dk.run_daily_salaries",
        ["--game-date", game_date],
        data_root=data_root,
        timeout_s=600,
    )
    payload = {
        "game_date": game_date,
        "placeholder_mode": False,
        "completed_at": _utc_now_iso(),
        "action_props_required": bool(require_action_props),
        "allow_rotowire_props_fallback": bool(allow_rotowire_props_fallback),
        "action_props_status": action_props_status,
        "action_props_raw_file_count": int(len(raw_props_files)),
    }
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return marker


@task(name="freeze-run-inputs", retries=0)
def freeze_run_inputs_task(
    *,
    game_date: str,
    run_id: str,
    as_of_ts: str,
    bundle_dir: Path,
    data_root: Path,
) -> Path:
    minutes_selector_path = model_selectors.active_minutes_selector_path(
        data_root=data_root,
        project_root=PROJECT_ROOT,
    )
    rates_selector_path = model_selectors.active_rates_selector_path(
        data_root=data_root,
        project_root=PROJECT_ROOT,
    )

    manifest_path = control_plane.write_run_manifest_start(
        data_root=data_root,
        game_date=game_date,
        run_id=run_id,
        as_of_ts=as_of_ts,
        sim_profile="game_transformer_v2",
        entrypoint="prefect-v3",
        minutes_current_run_path=minutes_selector_path,
        rates_current_run_path=rates_selector_path,
        slate={},
    )

    bundle_hash = _bundle_artifact_hash(bundle_dir)
    control_plane.atomic_update_json(
        manifest_path,
        {
            "v3": {
                "bundle_dir": str(bundle_dir),
                "bundle_hash": bundle_hash,
                "parity_manifest_path": str(resolve_parity_manifest_path(bundle_dir)),
            }
        },
    )
    return manifest_path


@task(name="build-features-gtv2-live", retries=0)
def build_features_gtv2_live_task(
    *,
    game_date: str,
    run_id: str,
    run_as_of_ts: str,
    data_root: Path,
    bundle_dir: Path,
    manifest_path: Path,
    placeholder_mode: bool,
    require_action_props: bool,
    allow_rotowire_props_fallback: bool,
) -> Path:
    run_dir = data_root / "live" / FEATURES_ROOT / game_date / f"run={run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "features.parquet"
    runtime_manifest_path = run_dir / "feature_runtime_manifest.json"
    input_checklist_path = run_dir / "feature_input_checklist.json"

    manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    v3_meta = dict(manifest_payload.get("v3", {}))
    if placeholder_mode:
        features_df = _placeholder_feature_frame(game_date=game_date, as_of_ts=run_as_of_ts)
        features_df.to_parquet(out_path, index=False)

        transform_manifest = {
            "feature_builder": "placeholder_gtv2_live_v1",
            "scaling": "none",
            "encoding": "none",
        }
        integrity = {
            "git_sha": str(manifest_payload.get("git_sha")),
            "config_hash": str(v3_meta.get("bundle_hash")),
            "artifact_hash": str(v3_meta.get("bundle_hash")),
        }
        parity_path = _ensure_placeholder_bundle(
            bundle_dir=bundle_dir,
            features_df=features_df,
            transform_manifest=transform_manifest,
            integrity=integrity,
        )
        diagnostics: dict[str, Any] = {
            "placeholder_mode": True,
            "rows": int(len(features_df)),
        }
    else:
        parity_path = resolve_parity_manifest_path(bundle_dir)
        if not parity_path.exists():
            raise RuntimeError(
                f"bundle parity manifest missing (fail-closed): {parity_path}. "
                "Create and ship parity_manifest.json with the promoted bundle."
            )
        parity_payload = load_parity_manifest(parity_path)
        expected_transform = dict(parity_payload.get("transform_manifest", {}))
        expected_priors = dict(expected_transform.get("priors", {}))
        expected_dnp = dict(expected_transform.get("dnp_history", {}))

        allow_priors_fallback = bool(expected_priors.get("allow_priors_fallback", True))
        dnp_mode = str(expected_dnp.get("mode", "bounded_lookback")).strip().lower()
        dnp_lookback_days: int | None
        if dnp_mode in {"full_prior_history", "full-history", "full"}:
            dnp_lookback_days = None
        else:
            raw_lookback = expected_dnp.get("lookback_days", 120)
            dnp_lookback_days = int(raw_lookback) if raw_lookback is not None else None

        allow_rotowire_fallback_cfg = bool(allow_rotowire_props_fallback)

        checklist = _build_feature_input_checklist(
            game_date=game_date,
            run_as_of_ts=run_as_of_ts,
            data_root=data_root,
            allow_priors_fallback=allow_priors_fallback,
            allow_rotowire_props_fallback=allow_rotowire_fallback_cfg,
            require_action_props=bool(require_action_props),
        )
        input_checklist_path.write_text(
            json.dumps(checklist, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        failed = checklist.get("failed_required_checks", [])
        if failed:
            raise RuntimeError(
                "live feature input checklist failed required checks: "
                f"{failed}. See {input_checklist_path}"
            )
        selected_props_source = _selected_props_source_from_checklist(checklist)
        if selected_props_source == "rotowire_fallback":
            print(
                "[v3][props] WARNING: Rotowire fallback source selected; "
                "source-distribution parity drift risk is elevated."
            )
        props_source_report_path = run_dir / "props_source_report.json"
        props_source_report_path.write_text(
            json.dumps(
                {
                    "game_date": game_date,
                    "run_id": run_id,
                    "run_as_of_ts": run_as_of_ts,
                    "selected_source": selected_props_source,
                    "require_action_props": bool(require_action_props),
                    "allow_rotowire_props_fallback": bool(allow_rotowire_fallback_cfg),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        # Build canonical live minutes features first, then project to GTV2 model contract.
        run_as_of_ts_cli = _cli_compatible_ts(run_as_of_ts)
        _run_python_module(
            "projections.cli.build_minutes_live",
            [
                "--date",
                game_date,
                "--run-id",
                run_id,
                "--run-as-of-ts",
                run_as_of_ts_cli,
                "--data-root",
                str(data_root),
                *(
                    ["--allow-rotowire-props-fallback"]
                    if allow_rotowire_fallback_cfg
                    else []
                ),
            ],
            data_root=data_root,
            timeout_s=1200,
        )
        base_minutes_path = (
            data_root / "live" / "features_minutes_v1" / game_date / f"run={run_id}" / "features.parquet"
        )
        if not base_minutes_path.exists():
            raise RuntimeError(f"base minutes features not found: {base_minutes_path}")

        spec = load_gtv2_feature_spec(bundle_dir)
        base_df = pd.read_parquet(base_minutes_path)

        built = build_gtv2_live_features(
            minutes_features=base_df,
            spec=spec,
            data_root=data_root,
            game_date=game_date,
            allow_priors_fallback=allow_priors_fallback,
            dnp_lookback_days=dnp_lookback_days,
        )
        transform_manifest = dict(built.transform_manifest)
        if stable_json_sha256(transform_manifest) != stable_json_sha256(expected_transform):
            raise RuntimeError(
                "observed transform manifest does not match bundle parity manifest "
                "(fail-closed transform parity gate)"
            )

        features_df = _coerce_frame_to_manifest_schema(built.features, parity_payload)
        features_df.to_parquet(out_path, index=False)

        integrity_src = dict(parity_payload.get("integrity", {}))
        integrity = {
            "git_sha": integrity_src.get("git_sha"),
            "config_hash": integrity_src.get("config_hash"),
            "artifact_hash": integrity_src.get("artifact_hash"),
        }
        diagnostics = dict(built.diagnostics)
        diagnostics["placeholder_mode"] = False
        diagnostics["base_minutes_features_path"] = str(base_minutes_path)
        diagnostics["feature_input_checklist_path"] = str(input_checklist_path)
        diagnostics["props_source_report_path"] = str(props_source_report_path)
        diagnostics["props_source_selected"] = selected_props_source
        diagnostics["dnp_history_mode"] = (
            "full_prior_history" if dnp_lookback_days is None else "bounded_lookback"
        )
        diagnostics["dnp_lookback_days"] = None if dnp_lookback_days is None else int(dnp_lookback_days)
        diagnostics["allow_rotowire_props_fallback"] = bool(allow_rotowire_fallback_cfg)

    runtime_manifest_path.write_text(
        json.dumps(
            {
                "transform_manifest": transform_manifest,
                "integrity": integrity,
                "parity_manifest_path": str(parity_path),
                "diagnostics": diagnostics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return out_path


@task(name="v3-preflight", retries=0)
def preflight_gate_task(
    *,
    as_of_ts: str,
    required_inputs: dict[str, Path],
    run_dirs: list[Path],
    features_path: Path,
    parity_manifest_path: Path,
    runtime_manifest_path: Path,
    input_max_age_minutes: float,
    bundle_config_path: Path | None = None,
) -> dict[str, Any]:
    runtime_payload = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    return run_preflight_gate(
        as_of_ts=as_of_ts,
        required_inputs=required_inputs,
        run_dirs=run_dirs,
        features_path=features_path,
        parity_manifest_path=parity_manifest_path,
        observed_transform_manifest=dict(runtime_payload.get("transform_manifest", {})),
        observed_integrity=dict(runtime_payload.get("integrity", {})),
        input_max_age_minutes=float(input_max_age_minutes),
        bundle_config_path=bundle_config_path,
    )


@task(name="score-gtv2-live", retries=0)
def score_gtv2_live_task(
    *,
    game_date: str,
    run_id: str,
    features_path: Path,
    bundle_dir: Path,
    data_root: Path,
    placeholder_mode: bool,
    gtv2_device: str | None = None,
    random_seed: int = 42,
) -> Path:
    run_dir = data_root / "artifacts" / SCORES_ROOT / f"game_date={game_date}" / f"run={run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "scores.parquet"
    summary_path = run_dir / "score_summary.json"

    if placeholder_mode:
        features = pd.read_parquet(features_path)
        scores = features[["game_date", "game_id", "team_id", "player_id"]].copy()
        scores["minutes_mean"] = pd.to_numeric(features["minutes_prior"], errors="coerce").fillna(0.0)
        scores["play_prob"] = 0.95
        scores["dk_rate"] = pd.to_numeric(features["usage_prior"], errors="coerce").fillna(0.0) * 100.0
        scores.to_parquet(out_path, index=False)
        summary_path.write_text(
            json.dumps(
                {
                    "placeholder_mode": True,
                    "rows": int(len(scores)),
                    "created_at": _utc_now_iso(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return out_path

    _set_inference_seed(int(random_seed))
    device = _resolve_torch_device(gtv2_device)
    config, model = _load_gtv2_model(bundle_dir, device=device)

    features_df = pd.read_parquet(features_path)
    examples = _build_gtv2_inference_examples(
        features_df=features_df,
        game_date=game_date,
        config=config,
    )
    loader = DataLoader(
        GameLevelDataset(examples),
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_game_level_examples,
    )

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            player_features = batch["player_features"].to(device=device)
            player_valid_mask = batch["player_valid_mask"].to(device=device)
            game_features = batch["game_features"].to(device=device)
            team_features = batch["team_features"].to(device=device)

            out = model(
                player_features,
                player_valid_mask,
                game_features=game_features,
                team_features=team_features,
                sample_active=False,
                run_flow=False,
            )

            valid = batch["player_valid_mask"].cpu().numpy().astype(bool)
            player_ids = batch["player_ids"].cpu().numpy().astype(np.int64)
            team_ids = batch["team_ids"].cpu().numpy().astype(np.int64)
            game_ids = [int(v) for v in batch["game_id_norm"]]
            game_dates = [str(v) for v in batch["game_date"]]

            minutes = out.minutes.minutes.detach().cpu().numpy()
            active_mask = out.active.active_mask.detach().cpu().numpy().astype(bool)
            player_logits = out.active.player_logits.detach().cpu().numpy()
            active_prob = 1.0 / (1.0 + np.exp(-np.clip(player_logits, -40.0, 40.0)))

            for b_idx in range(minutes.shape[0]):
                valid_flat = np.concatenate([valid[b_idx, 0], valid[b_idx, 1]], axis=0)
                player_flat = np.concatenate([player_ids[b_idx, 0], player_ids[b_idx, 1]], axis=0)
                team_flat = np.concatenate(
                    [
                        np.full((15,), int(team_ids[b_idx, 0]), dtype=np.int64),
                        np.full((15,), int(team_ids[b_idx, 1]), dtype=np.int64),
                    ],
                    axis=0,
                )
                for idx in np.where(valid_flat)[0]:
                    rows.append(
                        {
                            "game_date": game_dates[b_idx],
                            "game_id": game_ids[b_idx],
                            "team_id": int(team_flat[idx]),
                            "player_id": int(player_flat[idx]),
                            "minutes_deterministic": float(minutes[b_idx, idx]),
                            "active_deterministic": int(bool(active_mask[b_idx, idx])),
                            "active_logit": float(player_logits[b_idx, idx]),
                            "active_prob_proxy": float(active_prob[b_idx, idx]),
                        }
                    )

    scores = pd.DataFrame(rows)
    if scores.empty:
        raise RuntimeError("GTV2 scoring produced zero rows")
    scores = scores.sort_values(["game_date", "game_id", "team_id", "player_id"]).reset_index(drop=True)
    scores.to_parquet(out_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "placeholder_mode": False,
                "rows": int(len(scores)),
                "games": int(scores["game_id"].nunique()),
                "players": int(scores["player_id"].nunique()),
                "device": str(device),
                "bundle_dir": str(bundle_dir),
                "created_at": _utc_now_iso(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return out_path


@task(name="generate-worlds-gtv2-live", retries=0)
def generate_worlds_gtv2_live_task(
    *,
    game_date: str,
    run_id: str,
    features_path: Path,
    scores_path: Path,
    bundle_dir: Path,
    data_root: Path,
    sim_worlds: int,
    placeholder_mode: bool,
    gtv2_device: str | None = None,
    world_chunk_size: int = 64,
    active_temperature: float = 1.0,
    random_seed: int = 42,
    strict_world_contracts: bool = True,
) -> dict[str, str]:
    run_dir = data_root / "artifacts" / WORLDS_ROOT / f"game_date={game_date}" / f"run={run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    worlds_path = run_dir / "worlds.parquet"
    projections_path = run_dir / "projections.parquet"
    worlds_summary_path = run_dir / "world_contracts_summary.json"

    if placeholder_mode:
        scores = pd.read_parquet(scores_path)
        projections = scores[["game_date", "game_id", "team_id", "player_id"]].copy()
        projections["minutes_sim_mean"] = scores["minutes_mean"].astype(float)
        projections["minutes_sim_p50"] = scores["minutes_mean"].astype(float)
        projections["dk_fpts_mean"] = (scores["minutes_mean"].astype(float) * scores["dk_rate"].astype(float) / 60.0).round(4)
        projections["dk_fpts_p50"] = projections["dk_fpts_mean"]
        projections["sim_p_active"] = scores["play_prob"].astype(float)
        projections["n_worlds"] = int(sim_worlds)
        projections["sim_profile"] = "game_transformer_v2"
        projections = projections[PLACEHOLDER_PROJECTION_COLUMNS]
        projections.to_parquet(projections_path, index=False)
        pd.DataFrame(columns=["world_idx"]).to_parquet(worlds_path, index=False)
        contract_summary = {
            "contract_checks": {
                "team_minutes_not_240": 0,
                "minutes_negative": 0,
                "minutes_over_48": 0,
                "negative_stats": 0,
                "fg2m_gt_fga2": 0,
                "fg3m_gt_fga3": 0,
                "ftm_gt_fta": 0,
                "inactive_nonzero_stats": 0,
                "inactive_nonzero_fpts_proxy": 0,
                "total_violations": 0,
            },
            "placeholder_mode": True,
        }
    else:
        _set_inference_seed(int(random_seed))
        device = _resolve_torch_device(gtv2_device)
        config, model = _load_gtv2_model(bundle_dir, device=device)
        features_df = pd.read_parquet(features_path)
        examples = _build_gtv2_inference_examples(
            features_df=features_df,
            game_date=game_date,
            config=config,
        )
        loader = DataLoader(
            GameLevelDataset(examples),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_game_level_examples,
        )

        world_frames: list[pd.DataFrame] = []
        contract_counter: Counter[str] = Counter()
        for batch in loader:
            df_batch, checks = sample_worlds_for_batch(
                model,
                batch,
                device=device,
                num_worlds=int(sim_worlds),
                chunk_size=max(1, int(world_chunk_size)),
                active_temperature=float(active_temperature),
                strict_contracts=bool(strict_world_contracts),
            )
            world_frames.append(df_batch)
            contract_counter.update(checks)

        worlds_df = pd.concat(world_frames, ignore_index=True) if world_frames else pd.DataFrame()
        if worlds_df.empty:
            raise RuntimeError("GTV2 worlds generation produced zero rows")
        worlds_df.to_parquet(worlds_path, index=False)

        projections = summarize_worlds_to_projections(
            worlds_df,
            sim_profile="game_transformer_v2",
        )
        projections.to_parquet(projections_path, index=False)
        contract_summary = {
            "contract_checks": dict(contract_counter),
            "placeholder_mode": False,
            "world_rows": int(len(worlds_df)),
            "projection_rows": int(len(projections)),
            "bundle_dir": str(bundle_dir),
            "device": str(device),
            "created_at": _utc_now_iso(),
        }

    worlds_summary_path.write_text(
        json.dumps(contract_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "worlds_dir": str(run_dir),
        "worlds_path": str(worlds_path),
        "projections_path": str(projections_path),
        "world_contract_summary_path": str(worlds_summary_path),
    }


@task(name="finalize-projections-live", retries=0)
def finalize_projections_live_task(
    *,
    game_date: str,
    run_id: str,
    worlds_projections_path: Path,
    data_root: Path,
    placeholder_mode: bool,
) -> Path:
    out_dir = data_root / "artifacts" / "projections" / game_date / f"run={run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "projections.parquet"

    df = pd.read_parquet(worlds_projections_path)
    if df.empty:
        raise RuntimeError(f"world projections are empty: {worlds_projections_path}")

    # Enrich run-scoped projections with display + vegas context fields so the
    # dashboard can render a read-only game view without additional joins.
    display_src = data_root / "live" / "features_minutes_v1" / game_date / f"run={run_id}" / "features.parquet"
    if display_src.exists():
        src_df = pd.read_parquet(display_src)
        join_keys = ["game_id", "team_id", "player_id"]
        needed = [
            "player_name",
            "team_name",
            "team_tricode",
            "opponent_team_id",
            "opponent_team_name",
            "opponent_team_tricode",
            "status",
            "tip_ts",
            "is_projected_starter",
            "is_confirmed_starter",
            "team_implied_total",
            "opponent_implied_total",
            "total",
            "spread_home",
        ]
        present = [c for c in join_keys + needed if c in src_df.columns]
        if all(k in present for k in join_keys):
            enrich = src_df.loc[:, present].copy()
            for key in join_keys:
                enrich[key] = pd.to_numeric(enrich[key], errors="coerce").astype("Int64")
                df[key] = pd.to_numeric(df[key], errors="coerce").astype("Int64")
            enrich = enrich.dropna(subset=join_keys).drop_duplicates(subset=join_keys, keep="last")
            df = df.merge(enrich, on=join_keys, how="left", suffixes=("", "__src"))
            for col in needed:
                src_col = f"{col}__src"
                if src_col not in df.columns:
                    continue
                if col in df.columns:
                    df[col] = df[col].where(pd.notna(df[col]), df[src_col])
                else:
                    df[col] = df[src_col]
            df = df.drop(columns=[c for c in df.columns if c.endswith("__src")], errors="ignore")

    df.to_parquet(out_path, index=False)
    return out_dir


@task(name="v3-postflight", retries=0)
def postflight_gate_task(
    *,
    projections_path: Path,
    parity_manifest_path: Path,
    world_contract_summary_path: Path,
) -> dict[str, Any]:
    return run_postflight_gate(
        projections_path=projections_path,
        parity_manifest_path=parity_manifest_path,
        world_contract_summary_path=world_contract_summary_path,
        key_columns=("game_id", "team_id", "player_id"),
        min_rows=20,
    )


@task(name="publish-atomic", retries=0)
def publish_atomic_task(
    *,
    game_date: str,
    run_id: str,
    manifest_path: Path,
    data_root: Path,
) -> dict[str, str]:
    pointers: dict[str, str] = {}
    targets = {
        "features_gtv2_v1": data_root / "live" / FEATURES_ROOT / game_date,
        "scores_gtv2": data_root / "artifacts" / SCORES_ROOT / f"game_date={game_date}",
        "worlds_gtv2": data_root / "artifacts" / WORLDS_ROOT / f"game_date={game_date}",
        "unified_projections": data_root / "artifacts" / "projections" / game_date,
    }
    for stage, dataset_dir in targets.items():
        pointer = control_plane.promote_run_pointer(
            dataset_dir=dataset_dir,
            run_id=run_id,
            manifest_path=manifest_path,
            extra={"entrypoint": "prefect-v3", "stage": stage},
        )
        pointers[stage] = str(pointer)
    return pointers


@flow(name="nba-live-pipeline-v3", log_prints=True)
def nba_live_pipeline_v3_flow(
    *,
    game_date: str | None = None,
    sim_worlds: int = 25000,
    run_id_override: str | None = None,
    promote_pointers: bool = True,
    placeholder_mode: bool = True,
    replay_mode: bool = False,
    as_of_ts_override: str | None = None,
    gtv2_bundle_dir: str | None = None,
    gtv2_device: str | None = None,
    gtv2_world_chunk_size: int = 64,
    gtv2_active_temperature: float = 1.0,
    gtv2_seed: int = 42,
    gtv2_strict_world_contracts: bool = True,
    input_max_age_minutes: float = 360.0,
    require_action_props: bool = True,
    allow_rotowire_props_fallback: bool = True,
) -> dict[str, str]:
    logger = get_run_logger()
    data_root = paths.get_data_root()
    resolved_game_date = _resolve_game_date(game_date)
    run_id = run_id_override or control_plane.canonical_run_id()
    bundle_dir = _resolve_bundle_dir(data_root=data_root, gtv2_bundle_dir=gtv2_bundle_dir)
    resolved_allow_rotowire_props_fallback = bool(allow_rotowire_props_fallback)
    rotation_cfg_path = PROJECT_ROOT / "config" / "rotation_set_minutes_live.json"
    if rotation_cfg_path.exists():
        try:
            rotation_cfg = json.loads(rotation_cfg_path.read_text(encoding="utf-8"))
            if "allow_rotowire_props_fallback" in rotation_cfg:
                resolved_allow_rotowire_props_fallback = bool(rotation_cfg.get("allow_rotowire_props_fallback"))
        except Exception:
            pass

    # Runtime stamp for reproducibility and incident triage.
    enforce_clean_tree()
    enforce_prod_sanity()
    log_runtime_stamp(
        entrypoint="prefect:nba-live-pipeline-v3",
        config_paths={
            "minutes_current_run": model_selectors.active_minutes_selector_path(
                data_root=data_root,
                project_root=PROJECT_ROOT,
            ),
            "rates_current_run": model_selectors.active_rates_selector_path(
                data_root=data_root,
                project_root=PROJECT_ROOT,
            ),
            "gtv2_bundle_dir": bundle_dir,
        },
        project_root=PROJECT_ROOT,
        logger=logger,
    )

    v3_run_dir = (
        data_root
        / "artifacts"
        / "runs"
        / "nba_live_v3"
        / f"game_date={resolved_game_date}"
        / f"run={run_id}"
    )
    v3_run_dir.mkdir(parents=True, exist_ok=True)

    with writer_guard.PipelineWriterLock(data_root=data_root, run_id=run_id):
        os.environ["PROJECTIONS_SKIP_POINTER_WRITES"] = "1"

        scrape_marker = scrape_core_inputs_task(
            game_date=resolved_game_date,
            data_root=data_root,
            placeholder_mode=bool(placeholder_mode),
            require_action_props=bool(require_action_props),
            allow_rotowire_props_fallback=bool(resolved_allow_rotowire_props_fallback),
            replay_mode=bool(replay_mode),
        )

        as_of_ts = str(as_of_ts_override) if as_of_ts_override else _utc_now_iso()
        manifest_path = freeze_run_inputs_task(
            game_date=resolved_game_date,
            run_id=run_id,
            as_of_ts=as_of_ts,
            bundle_dir=bundle_dir,
            data_root=data_root,
        )

        features_path = build_features_gtv2_live_task(
            game_date=resolved_game_date,
            run_id=run_id,
            run_as_of_ts=as_of_ts,
            data_root=data_root,
            bundle_dir=bundle_dir,
            manifest_path=manifest_path,
            placeholder_mode=bool(placeholder_mode),
            require_action_props=bool(require_action_props),
            allow_rotowire_props_fallback=bool(resolved_allow_rotowire_props_fallback),
        )
        parity_manifest_path = resolve_parity_manifest_path(bundle_dir)
        runtime_manifest_path = features_path.parent / "feature_runtime_manifest.json"

        score_run_dir = data_root / "artifacts" / SCORES_ROOT / f"game_date={resolved_game_date}" / f"run={run_id}"
        worlds_run_dir = data_root / "artifacts" / WORLDS_ROOT / f"game_date={resolved_game_date}" / f"run={run_id}"
        projections_run_dir = data_root / "artifacts" / "projections" / resolved_game_date / f"run={run_id}"

        preflight_report = preflight_gate_task(
            as_of_ts=as_of_ts,
            required_inputs={
                "core_inputs_marker": scrape_marker,
                "features": features_path,
            },
            run_dirs=[score_run_dir, worlds_run_dir, projections_run_dir],
            features_path=features_path,
            parity_manifest_path=parity_manifest_path,
            runtime_manifest_path=runtime_manifest_path,
            input_max_age_minutes=float(input_max_age_minutes),
            bundle_config_path=(bundle_dir / "config.json"),
        )
        (v3_run_dir / "preflight_report.json").write_text(
            json.dumps(preflight_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        scores_path = score_gtv2_live_task(
            game_date=resolved_game_date,
            run_id=run_id,
            features_path=features_path,
            bundle_dir=bundle_dir,
            data_root=data_root,
            placeholder_mode=bool(placeholder_mode),
            gtv2_device=gtv2_device,
            random_seed=int(gtv2_seed),
        )

        worlds_outputs = generate_worlds_gtv2_live_task(
            game_date=resolved_game_date,
            run_id=run_id,
            features_path=features_path,
            scores_path=scores_path,
            bundle_dir=bundle_dir,
            data_root=data_root,
            sim_worlds=int(sim_worlds),
            placeholder_mode=bool(placeholder_mode),
            gtv2_device=gtv2_device,
            world_chunk_size=int(gtv2_world_chunk_size),
            active_temperature=float(gtv2_active_temperature),
            random_seed=int(gtv2_seed),
            strict_world_contracts=bool(gtv2_strict_world_contracts),
        )

        projections_dir = finalize_projections_live_task(
            game_date=resolved_game_date,
            run_id=run_id,
            worlds_projections_path=Path(worlds_outputs["projections_path"]),
            data_root=data_root,
            placeholder_mode=bool(placeholder_mode),
        )

        # Validate output schema contract against parity manifest before publish.
        _ = load_parity_manifest(parity_manifest_path)
        postflight_report = postflight_gate_task(
            projections_path=projections_dir / "projections.parquet",
            parity_manifest_path=parity_manifest_path,
            world_contract_summary_path=Path(worlds_outputs["world_contract_summary_path"]),
        )
        (v3_run_dir / "postflight_report.json").write_text(
            json.dumps(postflight_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        pointer_payload: dict[str, str] = {}
        if promote_pointers:
            pointer_payload = publish_atomic_task(
                game_date=resolved_game_date,
                run_id=run_id,
                manifest_path=manifest_path,
                data_root=data_root,
            )

    return {
        "run_id": run_id,
        "game_date": resolved_game_date,
        "manifest_path": str(manifest_path),
        "features_path": str(features_path),
        "projections_path": str(projections_dir / "projections.parquet"),
        "bundle_dir": str(bundle_dir),
        "pointer_count": str(len(pointer_payload)),
    }
