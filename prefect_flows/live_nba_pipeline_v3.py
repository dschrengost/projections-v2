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
import hashlib
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from prefect import flow, get_run_logger, task
from torch.utils.data import DataLoader
from zoneinfo import ZoneInfo

from projections import model_selectors, paths
from projections.etl import storage as bronze_storage
from projections.names import normalize_player_name
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
from projections.ops.manual_availability import list_manual_overrides, manual_override_report
from projections.rotation.game_transformer_v2 import (
    GameLevelDataset,
    GameTransformerV2Config,
    build_game_level_examples,
    build_game_transformer_v2,
    collate_game_level_examples,
)
from projections.rotation.sample_worlds_v2 import (
    MakeModelConfig,
    sample_worlds_for_batch,
    summarize_worlds_to_projections,
)
from projections.runtime_stamp import (
    enforce_clean_tree,
    enforce_prod_sanity,
    log_runtime_stamp,
)


PROJECT_ROOT = paths.get_project_root()

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

_PROPS_TEAM_ABBR_TO_NBA: dict[str, str] = {
    "PHO": "PHX",
    "GS": "GSW",
    "NO": "NOP",
    "SA": "SAS",
    "NY": "NYK",
}

_ET = ZoneInfo("America/New_York")
_LOCK_WINDOW_THRESHOLDS = (
    {
        "window": "last_30",
        "max_minutes_to_tip": 30.0,
        "injuries_max_age_minutes": 30.0,
        "lineups_max_age_minutes": 30.0,
    },
    {
        "window": "last_60",
        "max_minutes_to_tip": 60.0,
        "injuries_max_age_minutes": 60.0,
        "lineups_max_age_minutes": 60.0,
    },
)
_REPORT_WINDOWS = (
    {
        "label": "nba_injury_report_1pm_et",
        "hour": 13,
        "minute": 0,
        "pre_minutes": 10,
        "post_minutes": 15,
    },
    {
        "label": "nba_injury_report_230pm_et",
        "hour": 14,
        "minute": 30,
        "pre_minutes": 10,
        "post_minutes": 15,
    },
    {
        "label": "nba_injury_report_5pm_et",
        "hour": 17,
        "minute": 0,
        "pre_minutes": 10,
        "post_minutes": 15,
    },
)
_REPORT_WINDOW_WAIT_TIMEOUT_SECONDS = 300
_REPORT_WINDOW_WAIT_INTERVAL_SECONDS = 30
_STALE_INPUT_TOLERANCE_SECONDS = 30
_ODDS_MATERIALITY_MAX_MINUTES_TO_TIP = 180.0
_PROPS_PLAYER_SET_EXPANSION_MAX_MINUTES_TO_TIP = 360.0
_WORLD_CONTRACT_TOL = 1e-4
_WORLD_REALISM_SHORT_MINUTES_DK_THRESHOLD = 35.0
_WORLD_REALISM_GAME_PTS_MAX_THRESHOLD = 340.0
_WORLD_REALISM_GAME_PTS_MIN_THRESHOLD = 110.0
_RETRYABLE_SUBPROCESS_EXIT_CODES = frozenset({-11, -6, 134, 139})
_SUBPROCESS_CRASH_RETRY_ATTEMPTS = max(
    1,
    int(os.environ.get("PROJECTIONS_SUBPROCESS_CRASH_RETRY_ATTEMPTS", "5")),
)
_SUBPROCESS_CRASH_RETRY_DELAY_SECONDS = max(
    0,
    int(os.environ.get("PROJECTIONS_SUBPROCESS_CRASH_RETRY_DELAY_SECONDS", "3")),
)
_TORCH_RUNTIME_CONFIGURED = False


def _utc_now_iso() -> str:
    return (
        datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )


def _cli_compatible_ts(ts_value: str) -> str:
    ts = pd.to_datetime(ts_value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise RuntimeError(f"invalid timestamp: {ts_value}")
    ts_utc = pd.Timestamp(ts).tz_convert("UTC").tz_localize(None)
    return ts_utc.strftime("%Y-%m-%dT%H:%M:%S")


def _subprocess_python() -> str:
    # Use the exact interpreter running the flow to avoid `uv run` syncing
    # and mutating site-packages while the worker is active.
    override = os.environ.get("PROJECTIONS_SUBPROCESS_PYTHON")
    if override:
        resolved = (
            shutil.which(override) if Path(override).name == override else override
        )
        if resolved and Path(resolved).exists():
            return str(resolved)
        raise FileNotFoundError(
            f"PROJECTIONS_SUBPROCESS_PYTHON={override} does not exist"
        )
    return sys.executable


def _run_python_module(
    module: str,
    args: list[str],
    *,
    data_root: Path,
    timeout_s: int,
) -> None:
    env = os.environ.copy()
    env["PROJECTIONS_DATA_ROOT"] = str(data_root)
    cmd = [_subprocess_python(), "-m", module, *args]
    last_result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, _SUBPROCESS_CRASH_RETRY_ATTEMPTS + 1):
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
        if result.returncode == 0:
            return
        last_result = result
        if (
            result.returncode not in _RETRYABLE_SUBPROCESS_EXIT_CODES
            or attempt >= _SUBPROCESS_CRASH_RETRY_ATTEMPTS
        ):
            break
        print(
            f"[subprocess-retry] {module} exited with {result.returncode}; "
            f"retrying attempt {attempt + 1}/{_SUBPROCESS_CRASH_RETRY_ATTEMPTS}",
            file=sys.stderr,
        )
        time.sleep(_SUBPROCESS_CRASH_RETRY_DELAY_SECONDS)
    raise RuntimeError(
        f"{module} failed with exit_code={last_result.returncode if last_result else 'unknown'}"
    )


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
    return (
        data_root / "artifacts" / "game_transformer_v2" / "bundle_current"
    ).resolve()


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
    _configure_torch_runtime_for_inference()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _configure_torch_runtime_for_inference() -> None:
    """Apply conservative torch runtime settings for long live inference tasks.

    We intentionally default to single-threaded CPU execution and disabled MKLDNN
    to reduce intermittent native crashes observed in long-running world generation.
    Operators can override defaults via environment variables:
      - PROJECTIONS_TORCH_NUM_THREADS
      - PROJECTIONS_TORCH_NUM_INTEROP_THREADS
      - PROJECTIONS_TORCH_DISABLE_MKLDNN
    """
    global _TORCH_RUNTIME_CONFIGURED
    if _TORCH_RUNTIME_CONFIGURED:
        return

    num_threads = int(os.environ.get("PROJECTIONS_TORCH_NUM_THREADS", "1"))
    interop_threads = int(
        os.environ.get("PROJECTIONS_TORCH_NUM_INTEROP_THREADS", "1")
    )
    disable_mkldnn = (
        str(os.environ.get("PROJECTIONS_TORCH_DISABLE_MKLDNN", "1"))
        .strip()
        .lower()
        in {"1", "true", "yes"}
    )

    try:
        torch.set_num_threads(max(1, int(num_threads)))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, int(interop_threads)))
    except Exception:
        pass
    if disable_mkldnn:
        try:
            torch.backends.mkldnn.enabled = False
        except Exception:
            pass

    _TORCH_RUNTIME_CONFIGURED = True


def _resolve_torch_device(device: str | None) -> torch.device:
    if device:
        return torch.device(str(device))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_gtv2_model(
    bundle_dir: Path,
    *,
    device: torch.device,
    flow_scale_clip_override: float | None = None,
) -> tuple[GameTransformerV2Config, torch.nn.Module]:
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

    # Inference-only scale_clip override (H1 experiment support)
    if flow_scale_clip_override is not None:
        model.flow_head.set_scale_clip(float(flow_scale_clip_override))

    model = model.to(device=device)
    model.eval()
    return config, model


def _coerce_frame_to_manifest_schema(
    features_df: pd.DataFrame, manifest: dict[str, Any]
) -> pd.DataFrame:
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
            if dtype in {
                "int64",
                "int32",
                "int16",
                "int8",
                "Int64",
                "Int32",
                "Int16",
                "Int8",
            }:
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
            raise RuntimeError(
                f"failed to coerce feature column '{col}' to dtype '{dtype}': {exc}"
            ) from exc

        if not nullable and bool(series.isna().any()):
            raise RuntimeError(
                f"non-nullable feature column has nulls after coercion: {col}"
            )
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
        overflow_protected_prior_play_prob_floor=float(
            config.overflow_protected_prior_play_prob_floor
        ),
        overflow_protected_prior_minutes_floor=float(
            config.overflow_protected_prior_minutes_floor
        ),
        overflow_risk_weight_consecutive_active_dnp=float(
            config.overflow_risk_weight_consecutive_active_dnp
        ),
        overflow_risk_weight_active_but_dnp_rate_last10=float(
            config.overflow_risk_weight_active_but_dnp_rate_last10
        ),
        overflow_risk_weight_inactive_streak_len=float(
            config.overflow_risk_weight_inactive_streak_len
        ),
        overflow_keep_weight_prior_play_prob=float(
            config.overflow_keep_weight_prior_play_prob
        ),
        overflow_keep_weight_prior_minutes=float(
            config.overflow_keep_weight_prior_minutes
        ),
    )
    if not examples:
        raise RuntimeError("no valid game examples produced for GTV2 inference")
    return examples


def _attach_gtv2_force_active_worlds(
    features_df: pd.DataFrame,
    *,
    game_date: str,
    data_root: Path,
    as_of_ts: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = features_df.copy()
    if out.empty:
        out["force_active_worlds"] = pd.Series(dtype="int8")
        return out, {"starter_rows": 0, "manual_force_in_rows": 0, "total_force_active_rows": 0}

    starter_mask = np.zeros(len(out), dtype=bool)
    for col in ("lineup_starter_announced", "is_projected_starter", "is_confirmed_starter"):
        if col in out.columns:
            starter_mask |= (
                pd.to_numeric(out[col], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
                >= 0.5
            )

    manual_force_in_mask = np.zeros(len(out), dtype=bool)
    try:
        overrides = list_manual_overrides(
            pd.Timestamp(game_date).date(),
            data_root=data_root,
            active_only=True,
            as_of_ts=as_of_ts,
        )
    except Exception:
        overrides = []
    force_in_keys: set[str] = set()
    for row in overrides:
        if str(row.get("override_type", "")).strip().lower() != "force_in":
            continue
        game_id_raw = pd.to_numeric(pd.Series([row.get("game_id")]), errors="coerce").iloc[0]
        player_id_raw = pd.to_numeric(pd.Series([row.get("player_id")]), errors="coerce").iloc[0]
        if pd.isna(game_id_raw) or pd.isna(player_id_raw):
            continue
        force_in_keys.add(f"{int(game_id_raw)}|{int(player_id_raw)}")
    if force_in_keys and {"game_id", "player_id"}.issubset(out.columns):
        game_ids = pd.to_numeric(out["game_id"], errors="coerce").astype("Int64")
        player_ids = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
        keys = (game_ids.astype("string") + "|" + player_ids.astype("string")).fillna("")
        manual_force_in_mask = keys.isin(force_in_keys).to_numpy(dtype=bool)

    force_active_mask = starter_mask | manual_force_in_mask
    out["force_active_worlds"] = force_active_mask.astype("int8")
    return out, {
        "starter_rows": int(starter_mask.sum()),
        "manual_force_in_rows": int(manual_force_in_mask.sum()),
        "total_force_active_rows": int(force_active_mask.sum()),
    }


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


def _normalize_props_team_abbr(value: object) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    return _PROPS_TEAM_ABBR_TO_NBA.get(raw, raw)


def _ts_to_iso(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _age_minutes(run_ts: pd.Timestamp, source_ts: pd.Timestamp | None) -> float | None:
    if source_ts is None or pd.isna(source_ts):
        return None
    return float((run_ts - source_ts).total_seconds() / 60.0)


def _latest_ts_by_game(
    df: pd.DataFrame, *, time_col: str = "as_of_ts"
) -> dict[int, pd.Timestamp]:
    if df.empty or "game_id" not in df.columns or time_col not in df.columns:
        return {}
    working = df.loc[:, ["game_id", time_col]].copy()
    working["game_id"] = pd.to_numeric(working["game_id"], errors="coerce").astype(
        "Int64"
    )
    working[time_col] = pd.to_datetime(working[time_col], utc=True, errors="coerce")
    working = working.dropna(subset=["game_id", time_col])
    if working.empty:
        return {}
    latest = working.groupby("game_id", sort=False)[time_col].max()
    return {int(game_id): pd.Timestamp(ts) for game_id, ts in latest.items()}


def _latest_ts_by_game_from_teams(
    slate_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    time_col: str,
) -> dict[int, pd.Timestamp]:
    if slate_df.empty or source_df.empty or time_col not in source_df.columns:
        return {}
    team_col = next(
        (
            candidate
            for candidate in ("team_tricode", "team_abbreviation", "team")
            if candidate in source_df.columns
        ),
        None,
    )
    if team_col is None:
        return {}
    working = source_df.loc[:, [team_col, time_col]].copy()
    working["team_tricode"] = working[team_col].map(_normalize_props_team_abbr)
    working[time_col] = pd.to_datetime(working[time_col], utc=True, errors="coerce")
    working = working.dropna(subset=["team_tricode", time_col])
    if working.empty:
        return {}
    per_team = working.groupby("team_tricode", sort=False)[time_col].max().to_dict()
    out: dict[int, pd.Timestamp] = {}
    for row in slate_df.itertuples(index=False):
        game_id = pd.to_numeric(getattr(row, "game_id", None), errors="coerce")
        if pd.isna(game_id):
            continue
        ts_values: list[pd.Timestamp] = []
        for attr in ("home_team_tricode", "away_team_tricode"):
            team = _normalize_props_team_abbr(getattr(row, attr, None))
            if not team:
                continue
            ts = per_team.get(team)
            if ts is not None and not pd.isna(ts):
                ts_values.append(pd.Timestamp(ts))
        if ts_values:
            out[int(game_id)] = max(ts_values)
    return out


def _probe_rotowire_props_snapshot_summary(
    *,
    rotowire_props_root: Path,
    game_date: pd.Timestamp,
    data_root: Path,
    run_as_of_ts: pd.Timestamp | None = None,
    timeout_s: int = 180,
) -> dict[str, Any]:
    probe_code = """
import json
import hashlib
import sys
from pathlib import Path

import pandas as pd

from projections.features.action_props import (
    build_action_props_feature_snapshots,
    load_rotowire_props_long_from_bronze,
)

root = Path(sys.argv[1])
day = pd.Timestamp(sys.argv[2])
run_as_of = pd.to_datetime(sys.argv[3], utc=True, errors="coerce")
frames = []
for offset in (0, 1):
    current_day = day + pd.Timedelta(days=offset)
    long_df = load_rotowire_props_long_from_bronze(
        rotowire_props_root=root,
        game_date=current_day,
    )
    if not pd.isna(run_as_of):
        asof = pd.to_datetime(long_df.get("action_props_as_of_ts"), utc=True, errors="coerce")
        long_df = long_df.loc[asof.notna() & (asof <= run_as_of)].copy()
    snap_df = build_action_props_feature_snapshots(long_df)
    if not snap_df.empty:
        keep_cols = [
            c
            for c in ("team_tricode", "action_props_as_of_ts", "player_name_norm", "an_has_any_props")
            if c in snap_df.columns
        ]
        frames.append(snap_df.loc[:, keep_cols].copy())

if frames:
    combined = pd.concat(frames, ignore_index=True)
    combined["team_tricode"] = combined["team_tricode"].astype(str).str.strip().str.upper()
    combined["action_props_as_of_ts"] = pd.to_datetime(
        combined["action_props_as_of_ts"], utc=True, errors="coerce"
    )
    combined = combined.dropna(subset=["team_tricode", "action_props_as_of_ts", "player_name_norm"])
    if "an_has_any_props" in combined.columns:
        combined = combined.loc[
            pd.to_numeric(combined["an_has_any_props"], errors="coerce").fillna(0.0)
            > 0.0
        ].copy()
else:
    combined = pd.DataFrame(
        columns=["team_tricode", "action_props_as_of_ts", "player_name_norm"]
    )

team_latest = (
    combined.groupby("team_tricode", sort=False)["action_props_as_of_ts"].max().to_dict()
    if not combined.empty
    else {}
)
team_player_digest = {}
team_player_count = {}
if not combined.empty:
    players_by_team = combined.groupby("team_tricode", sort=False)["player_name_norm"]
    for team, players in players_by_team:
        names = sorted({str(v).strip() for v in players if str(v).strip()})
        payload = json.dumps(names, separators=(",", ":"), ensure_ascii=True)
        team_player_digest[str(team)] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        team_player_count[str(team)] = int(len(names))
latest = combined["action_props_as_of_ts"].max() if not combined.empty else None
payload = {
    "parsed_rows": int(len(combined)),
    "latest_action_props_as_of_ts": (
        None if latest is None or pd.isna(latest) else pd.Timestamp(latest).isoformat()
    ),
    "teams": sorted(team_latest.keys()),
    "team_latest_as_of_ts": {
        str(team): pd.Timestamp(ts).isoformat()
        for team, ts in team_latest.items()
        if ts is not None and not pd.isna(ts)
    },
    "team_player_digest": team_player_digest,
    "team_player_count": team_player_count,
}
print(json.dumps(payload))
""".strip()
    env = os.environ.copy()
    env["PROJECTIONS_DATA_ROOT"] = str(data_root)
    cmd = [
        _subprocess_python(),
        "-c",
        probe_code,
        str(rotowire_props_root),
        pd.Timestamp(game_date).normalize().date().isoformat(),
        (
            pd.Timestamp(run_as_of_ts).isoformat()
            if run_as_of_ts is not None and not pd.isna(run_as_of_ts)
            else ""
        ),
    ]
    last_error = "rotowire props probe did not run"
    for attempt in range(1, _SUBPROCESS_CRASH_RETRY_ATTEMPTS + 1):
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        if result.returncode == 0:
            stdout = result.stdout.strip()
            if not stdout:
                return {
                    "parsed_rows": 0,
                    "latest_action_props_as_of_ts": None,
                    "teams": [],
                    "team_latest_as_of_ts": {},
                    "team_player_digest": {},
                    "team_player_count": {},
                    "parse_error": "rotowire props probe returned empty stdout",
                }
            try:
                payload = json.loads(stdout)
            except json.JSONDecodeError as exc:
                last_error = f"rotowire props probe invalid json: {exc}"
            else:
                payload["parse_error"] = None
                return payload
        else:
            last_error = f"rotowire props probe exited with code {result.returncode}"
            if (
                result.returncode in _RETRYABLE_SUBPROCESS_EXIT_CODES
                and attempt < _SUBPROCESS_CRASH_RETRY_ATTEMPTS
            ):
                print(
                    "[subprocess-retry] rotowire props probe exited with "
                    f"{result.returncode}; retrying attempt "
                    f"{attempt + 1}/{_SUBPROCESS_CRASH_RETRY_ATTEMPTS}",
                    file=sys.stderr,
                )
                time.sleep(_SUBPROCESS_CRASH_RETRY_DELAY_SECONDS)
                continue
        break
    return {
        "parsed_rows": 0,
        "latest_action_props_as_of_ts": None,
        "teams": [],
        "team_latest_as_of_ts": {},
        "team_player_digest": {},
        "team_player_count": {},
        "parse_error": last_error,
    }


def _content_digest_by_game_from_teams(
    slate_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    exclude_columns: set[str] | None = None,
) -> dict[int, str | None]:
    if slate_df.empty or source_df.empty:
        return {}
    team_col = next(
        (
            candidate
            for candidate in ("team_tricode", "team_abbreviation", "team")
            if candidate in source_df.columns
        ),
        None,
    )
    if team_col is None:
        return {}
    working = source_df.copy()
    working["_team_tricode"] = working[team_col].map(_normalize_props_team_abbr)
    working = working.loc[working["_team_tricode"].astype(str).str.len() > 0].copy()
    if working.empty:
        return {}
    out: dict[int, str | None] = {}
    for row in slate_df.itertuples(index=False):
        game_id = pd.to_numeric(getattr(row, "game_id", None), errors="coerce")
        if pd.isna(game_id):
            continue
        teams = {
            _normalize_props_team_abbr(getattr(row, attr, None))
            for attr in ("home_team_tricode", "away_team_tricode")
        }
        teams = {team for team in teams if team}
        game_df = working.loc[working["_team_tricode"].isin(teams)].copy()
        out[int(game_id)] = _frame_content_digest(
            game_df, exclude_columns=set(exclude_columns or set()) | {"_team_tricode"}
        )
    return out


def _report_window_status(
    *,
    run_ts: pd.Timestamp,
    per_game_freshness: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    run_ts_et = pd.Timestamp(run_ts).tz_convert(_ET)
    live_games = [
        game for game in per_game_freshness.values() if bool(game.get("is_live_game"))
    ]
    for window in _REPORT_WINDOWS:
        boundary_et = run_ts_et.normalize() + pd.Timedelta(
            hours=int(window["hour"]), minutes=int(window["minute"])
        )
        window_start = boundary_et - pd.Timedelta(minutes=float(window["pre_minutes"]))
        window_end = boundary_et + pd.Timedelta(minutes=float(window["post_minutes"]))
        if not (window_start <= run_ts_et <= window_end):
            continue
        blocking_games: list[dict[str, Any]] = []
        for game in live_games:
            tip_ts = pd.to_datetime(game.get("tip_ts"), utc=True, errors="coerce")
            if pd.isna(tip_ts):
                continue
            if pd.Timestamp(tip_ts).tz_convert(_ET) < boundary_et:
                continue
            injuries = dict(game.get("sources", {}).get("injuries", {}))
            latest = pd.to_datetime(
                injuries.get("latest_as_of_ts"), utc=True, errors="coerce"
            )
            if pd.isna(latest) or pd.Timestamp(latest).tz_convert(_ET) < boundary_et:
                blocking_games.append(
                    {
                        "game_id": int(game["game_id"]),
                        "tip_ts": game.get("tip_ts"),
                        "latest_injuries_ts": injuries.get("latest_as_of_ts"),
                        "injuries_source_used": injuries.get("source_used"),
                    }
                )
        return {
            "active": True,
            "label": str(window["label"]),
            "boundary_ts": boundary_et.tz_convert("UTC").isoformat(),
            "window_start_ts": window_start.tz_convert("UTC").isoformat(),
            "window_end_ts": window_end.tz_convert("UTC").isoformat(),
            "wait_timeout_seconds": int(_REPORT_WINDOW_WAIT_TIMEOUT_SECONDS),
            "wait_interval_seconds": int(_REPORT_WINDOW_WAIT_INTERVAL_SECONDS),
            "needs_wait": bool(blocking_games),
            "blocking_games": blocking_games,
        }
    return {
        "active": False,
        "label": None,
        "boundary_ts": None,
        "window_start_ts": None,
        "window_end_ts": None,
        "wait_timeout_seconds": int(_REPORT_WINDOW_WAIT_TIMEOUT_SECONDS),
        "wait_interval_seconds": int(_REPORT_WINDOW_WAIT_INTERVAL_SECONDS),
        "needs_wait": False,
        "blocking_games": [],
    }


def _lock_window_gate_status(
    *,
    per_game_freshness: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checked_games = 0
    for game in per_game_freshness.values():
        if not bool(game.get("is_live_game")):
            continue
        minutes_to_tip = game.get("minutes_to_tip")
        if not isinstance(minutes_to_tip, (int, float)):
            continue
        threshold = next(
            (
                item
                for item in _LOCK_WINDOW_THRESHOLDS
                if float(minutes_to_tip) <= float(item["max_minutes_to_tip"])
            ),
            None,
        )
        if threshold is None:
            continue
        checked_games += 1
        sources = dict(game.get("sources", {}))
        injuries = dict(sources.get("injuries", {}))
        lineups = dict(sources.get("lineups", {}))
        game_failures: list[str] = []
        injury_age = injuries.get("age_minutes")
        if injury_age is None or float(injury_age) > float(
            threshold["injuries_max_age_minutes"]
        ):
            game_failures.append(
                "injuries age="
                f"{'missing' if injury_age is None else f'{float(injury_age):.1f}m'} "
                f"> {float(threshold['injuries_max_age_minutes']):.1f}m"
            )
        lineup_age = lineups.get("age_minutes")
        if lineup_age is None or float(lineup_age) > float(
            threshold["lineups_max_age_minutes"]
        ):
            game_failures.append(
                "lineups age="
                f"{'missing' if lineup_age is None else f'{float(lineup_age):.1f}m'} "
                f"> {float(threshold['lineups_max_age_minutes']):.1f}m"
            )
        if game_failures:
            failures.append(
                {
                    "game_id": int(game["game_id"]),
                    "window": str(threshold["window"]),
                    "minutes_to_tip": float(minutes_to_tip),
                    "failures": game_failures,
                    "sources": {
                        "injuries": injuries,
                        "lineups": lineups,
                    },
                }
            )
    return {
        "ok": len(failures) == 0,
        "checked_games": int(checked_games),
        "failures": failures,
        "thresholds": list(_LOCK_WINDOW_THRESHOLDS),
    }


def _detect_stale_authoritative_inputs(
    *,
    frozen_source_freshness: dict[str, Any] | None,
    current_source_freshness: dict[str, Any] | None,
    as_of_ts: str,
) -> dict[str, Any]:
    frozen_games = dict((frozen_source_freshness or {}).get("per_game", {}))
    current_games = dict((current_source_freshness or {}).get("per_game", {}))
    stale_games: list[dict[str, Any]] = []
    tolerance = pd.Timedelta(seconds=_STALE_INPUT_TOLERANCE_SECONDS)
    for game_id, frozen in frozen_games.items():
        current = current_games.get(str(game_id)) or current_games.get(game_id)
        if not isinstance(current, dict):
            continue
        if not bool(current.get("is_live_game")):
            continue
        sources_out: dict[str, dict[str, str | None]] = {}
        for source_name in ("injuries", "lineups", "manual_overrides"):
            frozen_source = dict(frozen.get("sources", {}).get(source_name, {}))
            current_source = dict(current.get("sources", {}).get(source_name, {}))
            frozen_ts = pd.to_datetime(
                frozen_source.get("latest_as_of_ts"), utc=True, errors="coerce"
            )
            current_ts = pd.to_datetime(
                current_source.get("latest_as_of_ts"), utc=True, errors="coerce"
            )
            if pd.isna(current_ts):
                continue
            if (
                pd.isna(frozen_ts)
                or pd.Timestamp(current_ts) > pd.Timestamp(frozen_ts) + tolerance
            ):
                sources_out[source_name] = {
                    "frozen_ts": None
                    if pd.isna(frozen_ts)
                    else pd.Timestamp(frozen_ts).isoformat(),
                    "current_ts": pd.Timestamp(current_ts).isoformat(),
                    "frozen_source_used": frozen_source.get("source_used"),
                    "current_source_used": current_source.get("source_used"),
                }
        if sources_out:
            stale_games.append(
                {
                    "game_id": int(current.get("game_id", game_id)),
                    "tip_ts": current.get("tip_ts"),
                    "minutes_to_tip": current.get("minutes_to_tip"),
                    "sources": sources_out,
                }
            )
    return {
        "checked_at": as_of_ts,
        "stale": len(stale_games) > 0,
        "stale_games": stale_games,
        "tolerance_seconds": int(_STALE_INPUT_TOLERANCE_SECONDS),
    }


def _stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _source_digest_payload(source_payload: dict[str, Any]) -> dict[str, Any]:
    source = dict(source_payload)
    content_digest = source.get("content_digest")
    latest_as_of_ts = source.get("latest_as_of_ts")
    if content_digest is not None:
        latest_as_of_ts = None
    return {
        "latest_as_of_ts": latest_as_of_ts,
        "source_used": source.get("source_used"),
        "content_digest": content_digest,
    }


def _compute_per_game_input_digests(
    source_freshness: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    per_game = dict((source_freshness or {}).get("per_game", {}))
    digests: dict[str, dict[str, Any]] = {}
    for game_id, payload in per_game.items():
        game = dict(payload)
        sources = dict(game.get("sources", {}))
        digest_payload = {
            "game_id": int(game.get("game_id", game_id)),
            "tip_ts": game.get("tip_ts"),
            "is_live_game": bool(game.get("is_live_game")),
            "injuries": _source_digest_payload(dict(sources.get("injuries", {}))),
            "lineups": _source_digest_payload(dict(sources.get("lineups", {}))),
            "odds": _source_digest_payload(dict(sources.get("odds", {}))),
            "props": {
                "latest_as_of_ts": dict(sources.get("props", {})).get(
                    "latest_as_of_ts"
                ),
                "source_used": dict(sources.get("props", {})).get("source_used"),
                "player_set_digest": dict(sources.get("props", {})).get(
                    "player_set_digest"
                ),
                "player_set_count": dict(sources.get("props", {})).get(
                    "player_set_count"
                ),
            },
            "roster": _source_digest_payload(dict(sources.get("roster", {}))),
            "manual_overrides": _source_digest_payload(
                dict(sources.get("manual_overrides", {}))
            ),
        }
        digests[str(game_id)] = {
            "digest_sha256": _stable_digest(digest_payload),
            "payload": digest_payload,
        }
    return digests


def _normalize_game_ids(values: list[int] | list[str] | None) -> list[int]:
    if not values:
        return []
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        num = pd.to_numeric(value, errors="coerce")
        if pd.isna(num):
            continue
        game_id = int(num)
        if game_id in seen:
            continue
        seen.add(game_id)
        out.append(game_id)
    return out


def _frame_content_digest(
    df: pd.DataFrame,
    *,
    exclude_columns: set[str] | None = None,
) -> str | None:
    if df.empty:
        return None
    exclude = set(exclude_columns or set())
    cols = sorted(c for c in df.columns if c not in exclude)
    if not cols:
        return None
    work = df.loc[:, cols].copy()
    for col in cols:
        series = work[col]
        if pd.api.types.is_datetime64_any_dtype(
            series
        ) or pd.api.types.is_datetime64tz_dtype(series):
            work[col] = pd.to_datetime(series, utc=True, errors="coerce").astype(
                "string"
            )
        else:
            work[col] = series.astype("string")
    work = work.sort_values(by=cols, kind="stable", na_position="last").reset_index(
        drop=True
    )
    payload = {
        "columns": cols,
        "rows": work.where(pd.notna(work), None).to_dict(orient="records"),
    }
    return _stable_digest(payload)


def _content_digest_by_game(
    df: pd.DataFrame,
    game_ids: list[int],
    *,
    exclude_columns: set[str] | None = None,
) -> dict[int, str | None]:
    out: dict[int, str | None] = {}
    if not game_ids:
        return out
    if df.empty or "game_id" not in df.columns:
        return {int(gid): None for gid in game_ids}
    gids = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
    for game_id in game_ids:
        game_df = df.loc[gids == int(game_id)].copy()
        out[int(game_id)] = _frame_content_digest(
            game_df, exclude_columns=exclude_columns
        )
    return out


def _load_promoted_manifest_payload(
    *, data_root: Path, game_date: str
) -> dict[str, Any] | None:
    pointer_path = (
        data_root
        / "artifacts"
        / "projections"
        / game_date
        / control_plane.LATEST_DIRNAME
        / control_plane.CURRENT_POINTER_NAME
    )
    if not pointer_path.exists():
        return None
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    manifest_path = pointer.get("manifest_path")
    if not manifest_path:
        return None
    path = Path(str(manifest_path))
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_promoted_pointer_payload(*, dataset_dir: Path) -> dict[str, Any] | None:
    for candidate in (
        dataset_dir / control_plane.LATEST_DIRNAME / control_plane.CURRENT_POINTER_NAME,
        dataset_dir / control_plane.LEGACY_POINTER_NAME,
    ):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _build_publish_superseded_report(
    *,
    run_id: str,
    manifest_path: Path,
    dataset_dir: Path,
) -> dict[str, Any]:
    current_pointer = _load_promoted_pointer_payload(dataset_dir=dataset_dir)
    current_run_id = None if current_pointer is None else current_pointer.get("run_id")
    current_as_of_ts = None if current_pointer is None else current_pointer.get("as_of_ts")
    try:
        manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest_payload = {}
    candidate_as_of_ts = manifest_payload.get("as_of_ts")

    superseded = False
    reason: str | None = None
    current_ts = pd.to_datetime(current_as_of_ts, utc=True, errors="coerce")
    candidate_ts = pd.to_datetime(candidate_as_of_ts, utc=True, errors="coerce")
    if current_pointer and str(current_run_id or "") != str(run_id):
        if not pd.isna(current_ts) and not pd.isna(candidate_ts):
            if pd.Timestamp(current_ts) > pd.Timestamp(candidate_ts):
                superseded = True
                reason = "newer_pointer_as_of_ts"
            elif pd.Timestamp(current_ts) == pd.Timestamp(candidate_ts):
                superseded = True
                reason = "equal_as_of_ts_other_run_already_published"
        else:
            superseded = True
            reason = "existing_pointer_present_unknown_order"

    return {
        "checked_at": _utc_now_iso(),
        "superseded": bool(superseded),
        "reason": reason,
        "candidate": {
            "run_id": str(run_id),
            "manifest_path": str(manifest_path),
            "as_of_ts": candidate_as_of_ts,
        },
        "current_pointer": current_pointer,
    }


def _build_input_change_set(
    *,
    game_date: str,
    current_source_freshness: dict[str, Any] | None,
    previous_manifest_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    current_digests = _compute_per_game_input_digests(current_source_freshness)
    previous_source_freshness = {}
    previous_run_id = None
    previous_digests: dict[str, dict[str, Any]] = {}
    if isinstance(previous_manifest_payload, dict):
        previous_run_id = previous_manifest_payload.get("run_id")
        previous_source_freshness = dict(
            previous_manifest_payload.get("source_freshness", {})
        )
        previous_digests = dict(
            previous_manifest_payload.get("input_change_set", {}).get(
                "per_game_digests", {}
            )
        )
        if not previous_digests and previous_source_freshness:
            previous_digests = _compute_per_game_input_digests(
                previous_source_freshness
            )

    current_games = dict((current_source_freshness or {}).get("per_game", {}))
    previous_games = dict(previous_source_freshness.get("per_game", {}))

    changed_games: list[dict[str, Any]] = []
    unchanged_games: list[int] = []
    new_games: list[int] = []
    removed_games: list[int] = []

    for game_id, current in current_digests.items():
        previous = previous_digests.get(str(game_id))
        if previous is None:
            new_games.append(int(game_id))
            continue
        if str(previous.get("digest_sha256")) == str(current.get("digest_sha256")):
            unchanged_games.append(int(game_id))
            continue
        current_payload = dict(current.get("payload", {}))
        previous_payload = dict(previous.get("payload", {}))
        changed_sources: list[str] = []
        source_deltas: dict[str, dict[str, Any]] = {}
        for source_name in (
            "injuries",
            "lineups",
            "odds",
            "props",
            "roster",
            "manual_overrides",
        ):
            current_source = dict(current_payload.get(source_name, {}))
            previous_source = dict(previous_payload.get(source_name, {}))
            current_digest = current_source.get("content_digest")
            previous_digest = previous_source.get("content_digest")
            source_used_changed = current_source.get(
                "source_used"
            ) != previous_source.get("source_used")
            content_changed = False
            if current_digest is not None or previous_digest is not None:
                content_changed = str(current_digest) != str(previous_digest)
            else:
                content_changed = current_source.get(
                    "latest_as_of_ts"
                ) != previous_source.get("latest_as_of_ts")
            if source_used_changed or content_changed:
                changed_sources.append(source_name)
                source_deltas[source_name] = {
                    "previous": previous_source,
                    "current": current_source,
                }
        changed_games.append(
            {
                "game_id": int(game_id),
                "changed_sources": changed_sources,
                "current_digest_sha256": current.get("digest_sha256"),
                "previous_digest_sha256": previous.get("digest_sha256"),
                "tip_ts": dict(current_games.get(str(game_id), {})).get("tip_ts")
                or dict(previous_games.get(str(game_id), {})).get("tip_ts"),
                "source_deltas": source_deltas,
            }
        )

    for game_id in previous_digests:
        if str(game_id) not in current_digests:
            removed_games.append(int(game_id))

    return {
        "version": 1,
        "game_date": game_date,
        "previous_run_id": previous_run_id,
        "current_game_count": int(len(current_digests)),
        "changed_game_ids": sorted(item["game_id"] for item in changed_games),
        "unchanged_game_ids": sorted(unchanged_games),
        "new_game_ids": sorted(new_games),
        "removed_game_ids": sorted(removed_games),
        "changed_games": changed_games,
        "per_game_digests": current_digests,
    }


def _build_rerun_plan(
    *,
    game_date: str,
    input_change_set: dict[str, Any],
    current_source_freshness: dict[str, Any] | None,
    previous_manifest_payload: dict[str, Any] | None,
    current_bundle_hash: str,
    current_minutes_selector_path: Path,
    current_rates_selector_path: Path,
) -> dict[str, Any]:
    current_games = dict((current_source_freshness or {}).get("per_game", {}))
    current_game_ids = sorted(
        int(v)
        for v in pd.to_numeric(list(current_games.keys()), errors="coerce")
        if not pd.isna(v)
    )
    if previous_manifest_payload is None:
        return {
            "policy_version": 1,
            "game_date": game_date,
            "mode": "full_slate",
            "reason": "no_previous_published_run",
            "target_game_ids": current_game_ids,
            "ignored_changes": [],
        }

    previous_v3 = dict(previous_manifest_payload.get("v3", {}))
    if str(previous_v3.get("bundle_hash") or "") != str(current_bundle_hash):
        return {
            "policy_version": 1,
            "game_date": game_date,
            "mode": "full_slate",
            "reason": "bundle_hash_changed",
            "target_game_ids": current_game_ids,
            "ignored_changes": [],
        }
    if (
        Path(
            str(previous_manifest_payload.get("minutes_current_run_path", ""))
        ).resolve()
        != current_minutes_selector_path.resolve()
    ):
        return {
            "policy_version": 1,
            "game_date": game_date,
            "mode": "full_slate",
            "reason": "minutes_selector_changed",
            "target_game_ids": current_game_ids,
            "ignored_changes": [],
        }
    if (
        Path(str(previous_manifest_payload.get("rates_current_run_path", ""))).resolve()
        != current_rates_selector_path.resolve()
    ):
        return {
            "policy_version": 1,
            "game_date": game_date,
            "mode": "full_slate",
            "reason": "rates_selector_changed",
            "target_game_ids": current_game_ids,
            "ignored_changes": [],
        }

    if input_change_set.get("new_game_ids") or input_change_set.get("removed_game_ids"):
        return {
            "policy_version": 1,
            "game_date": game_date,
            "mode": "full_slate",
            "reason": "slate_composition_changed",
            "target_game_ids": current_game_ids,
            "ignored_changes": [],
        }

    changed_games = list(input_change_set.get("changed_games", []))
    if not changed_games:
        return {
            "policy_version": 1,
            "game_date": game_date,
            "mode": "skip",
            "reason": "no_changed_games",
            "target_game_ids": [],
            "ignored_changes": [],
        }

    material_targets: list[int] = []
    ignored_changes: list[dict[str, Any]] = []
    for change in changed_games:
        game_id = int(change.get("game_id"))
        current_game = dict(current_games.get(str(game_id), {}))
        minutes_to_tip = current_game.get("minutes_to_tip")
        changed_sources = [str(v) for v in change.get("changed_sources", [])]
        if not isinstance(minutes_to_tip, (int, float)) or float(minutes_to_tip) <= 0.0:
            ignored_changes.append(
                {
                    "game_id": game_id,
                    "changed_sources": changed_sources,
                    "reason": "game_not_pre_tip",
                }
            )
            continue
        material = False
        material_reason: str | None = None
        if any(
            source in {"injuries", "lineups", "roster"} for source in changed_sources
        ):
            material = True
            material_reason = "always_material_source_changed"
        elif "odds" in changed_sources and float(minutes_to_tip) <= float(
            _ODDS_MATERIALITY_MAX_MINUTES_TO_TIP
        ):
            material = True
            material_reason = "odds_change_within_tip_window"
        elif "props" in changed_sources and float(minutes_to_tip) <= float(
            _PROPS_PLAYER_SET_EXPANSION_MAX_MINUTES_TO_TIP
        ):
            props_delta = dict(change.get("source_deltas", {})).get("props", {})
            props_previous = dict(props_delta.get("previous", {}))
            props_current = dict(props_delta.get("current", {}))
            prev_digest = str(props_previous.get("player_set_digest") or "").strip()
            curr_digest = str(props_current.get("player_set_digest") or "").strip()
            prev_count_num = pd.to_numeric(
                props_previous.get("player_set_count"), errors="coerce"
            )
            curr_count_num = pd.to_numeric(
                props_current.get("player_set_count"), errors="coerce"
            )
            prev_count = int(prev_count_num) if pd.notna(prev_count_num) else 0
            curr_count = int(curr_count_num) if pd.notna(curr_count_num) else 0
            if curr_digest and curr_digest != prev_digest and curr_count > prev_count:
                material = True
                material_reason = "props_player_set_expanded"
        if material:
            if material_reason is not None:
                change["material_reason"] = material_reason
            material_targets.append(game_id)
        else:
            ignored_changes.append(
                {
                    "game_id": game_id,
                    "changed_sources": changed_sources,
                    "reason": "changes_below_materiality_policy",
                    "minutes_to_tip": minutes_to_tip,
                }
            )

    material_targets = sorted(set(material_targets))
    if not material_targets:
        return {
            "policy_version": 1,
            "game_date": game_date,
            "mode": "skip",
            "reason": "no_material_game_changes",
            "target_game_ids": [],
            "ignored_changes": ignored_changes,
            "materiality_policy": {
                "always_material_sources": ["injuries", "lineups", "roster"],
                "odds_materiality_max_minutes_to_tip": float(
                    _ODDS_MATERIALITY_MAX_MINUTES_TO_TIP
                ),
                "props_auto_trigger_enabled": False,
                "props_player_set_expansion_enabled": True,
                "props_player_set_expansion_max_minutes_to_tip": float(
                    _PROPS_PLAYER_SET_EXPANSION_MAX_MINUTES_TO_TIP
                ),
            },
        }
    if len(material_targets) >= len(current_game_ids):
        mode = "full_slate"
        target_game_ids = current_game_ids
        reason = "all_games_material"
    else:
        mode = "game_scoped"
        target_game_ids = material_targets
        reason = "material_game_subset"
    return {
        "policy_version": 1,
        "game_date": game_date,
        "mode": mode,
        "reason": reason,
        "target_game_ids": target_game_ids,
        "ignored_changes": ignored_changes,
        "materiality_policy": {
            "always_material_sources": ["injuries", "lineups", "roster"],
            "odds_materiality_max_minutes_to_tip": float(
                _ODDS_MATERIALITY_MAX_MINUTES_TO_TIP
            ),
            "props_auto_trigger_enabled": False,
            "props_player_set_expansion_enabled": True,
            "props_player_set_expansion_max_minutes_to_tip": float(
                _PROPS_PLAYER_SET_EXPANSION_MAX_MINUTES_TO_TIP
            ),
        },
    }


def _filter_to_target_games(
    df: pd.DataFrame, target_game_ids: list[int] | None
) -> pd.DataFrame:
    if df.empty or not target_game_ids or "game_id" not in df.columns:
        return df.copy()
    gids = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
    return df.loc[gids.isin(target_game_ids)].copy()


def _sort_for_stable_write(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    preferred = [
        c
        for c in ("world_idx", "game_date", "game_id", "team_id", "player_id")
        if c in df.columns
    ]
    if not preferred:
        return df.reset_index(drop=True)
    return df.sort_values(preferred, kind="stable").reset_index(drop=True)


def _stream_validate_parquet(
    path: Path,
    *,
    expected_rows: int | None = None,
    required_cols: tuple[str, ...] = (),
) -> dict[str, Any]:
    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as exc:
        raise RuntimeError(f"failed to open parquet for validation: {path}") from exc

    columns = tuple(str(name) for name in parquet_file.schema_arrow.names)
    missing = [col for col in required_cols if col not in columns]
    if missing:
        raise RuntimeError(
            f"validated parquet missing required columns {missing}: {path}"
        )

    row_count = 0
    try:
        for batch in parquet_file.iter_batches(batch_size=65536):
            row_count += int(batch.num_rows)
    except Exception as exc:
        raise RuntimeError(f"failed to stream-validate parquet contents: {path}") from exc

    if expected_rows is not None and row_count != int(expected_rows):
        raise RuntimeError(
            f"validated parquet row count mismatch for {path}: "
            f"expected={expected_rows} actual={row_count}"
        )

    return {
        "path": str(path),
        "rows": int(row_count),
        "columns": list(columns),
    }


def _atomic_write_validated_parquet(
    df: pd.DataFrame,
    path: Path,
    *,
    required_cols: tuple[str, ...] = (),
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(
        f".tmp.{control_plane.canonical_run_id()}.{os.getpid()}.parquet"
    )
    try:
        df.to_parquet(tmp, index=False)
        validation = _stream_validate_parquet(
            tmp,
            expected_rows=int(len(df)),
            required_cols=required_cols,
        )
        tmp.replace(path)
        return validation
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def _distinct_game_count(df: pd.DataFrame) -> int:
    if "game_id" not in df.columns:
        return 0
    gids = pd.to_numeric(df["game_id"], errors="coerce")
    return int(gids.dropna().nunique())


def _sanitize_frame_to_expected_keys(
    df: pd.DataFrame,
    *,
    expected_keys_df: pd.DataFrame,
    key_cols: Sequence[str],
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    key_cols = tuple(str(col) for col in key_cols)
    if df.empty:
        return (
            df.copy(),
            {
                "label": str(label),
                "rows_in": 0,
                "rows_out": 0,
                "dropped_null_key_rows": 0,
                "dropped_unexpected_key_rows": 0,
                "expected_distinct_keys": 0,
            },
        )

    missing_df = [col for col in key_cols if col not in df.columns]
    if missing_df:
        raise RuntimeError(f"{label} missing required key columns: {missing_df}")
    missing_expected = [col for col in key_cols if col not in expected_keys_df.columns]
    if missing_expected:
        raise RuntimeError(
            f"{label} expected-keys frame missing required columns: {missing_expected}"
        )

    work = df.copy()
    expected = expected_keys_df.loc[:, list(key_cols)].copy()
    # Keep key columns as numeric until null filtering is complete; converting
    # nullable pandas Int64 -> numpy int64 can raise intermittently in large
    # frames when masks are present.
    for col in key_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        expected[col] = pd.to_numeric(expected[col], errors="coerce")

    rows_in = int(len(work))
    null_mask = work.loc[:, list(key_cols)].isna().any(axis=1)
    dropped_null_key_rows = int(null_mask.sum())
    if dropped_null_key_rows:
        work = work.loc[~null_mask].copy()
    for col in key_cols:
        work[col] = work[col].astype("int64", copy=False)

    expected = (
        expected.dropna(subset=list(key_cols))
        .drop_duplicates(ignore_index=True)
        .reset_index(drop=True)
    )
    for col in key_cols:
        expected[col] = expected[col].astype("int64", copy=False)
    expected_distinct_keys = int(len(expected))

    if expected.empty:
        return (
            work.iloc[0:0].copy(),
            {
                "label": str(label),
                "rows_in": rows_in,
                "rows_out": 0,
                "dropped_null_key_rows": dropped_null_key_rows,
                "dropped_unexpected_key_rows": int(len(work)),
                "expected_distinct_keys": 0,
            },
        )

    # NOTE: Avoid dataframe merge here. Large-key merges have intermittently
    # triggered low-level pandas segmentation faults in production workers.
    expected_key_index = pd.MultiIndex.from_frame(
        expected.loc[:, list(key_cols)], names=list(key_cols)
    )
    work_key_index = pd.MultiIndex.from_frame(
        work.loc[:, list(key_cols)], names=list(key_cols)
    )
    keep_mask = work_key_index.isin(expected_key_index)
    dropped_unexpected_key_rows = int((~keep_mask).sum())
    merged = work.loc[keep_mask].copy().reset_index(drop=True)
    for col in key_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    post_filter_null_mask = merged.loc[:, list(key_cols)].isna().any(axis=1)
    post_filter_null_rows = int(post_filter_null_mask.sum())
    if post_filter_null_rows:
        merged = merged.loc[~post_filter_null_mask].copy()
        dropped_null_key_rows += post_filter_null_rows
    for col in key_cols:
        merged[col] = merged[col].astype("int64", copy=False)

    return (
        merged,
        {
            "label": str(label),
            "rows_in": rows_in,
            "rows_out": int(len(merged)),
            "dropped_null_key_rows": dropped_null_key_rows,
            "dropped_unexpected_key_rows": dropped_unexpected_key_rows,
            "expected_distinct_keys": expected_distinct_keys,
        },
    )


def _left_overlay_from_source_by_keys(
    base_df: pd.DataFrame,
    *,
    source_df: pd.DataFrame,
    key_cols: Sequence[str],
    value_cols: Sequence[str],
    label: str,
) -> pd.DataFrame:
    key_cols = tuple(str(col) for col in key_cols)
    value_cols = [str(col) for col in value_cols if str(col) in source_df.columns]
    if base_df.empty or source_df.empty or not value_cols:
        return base_df

    missing_base = [col for col in key_cols if col not in base_df.columns]
    missing_source = [col for col in key_cols if col not in source_df.columns]
    if missing_base or missing_source:
        raise RuntimeError(
            f"{label} missing join columns; "
            f"base_missing={missing_base} source_missing={missing_source}"
        )

    base = base_df.copy()
    source = source_df.loc[:, list(key_cols) + value_cols].copy()
    for col in key_cols:
        base[col] = pd.to_numeric(base[col], errors="coerce")
        source[col] = pd.to_numeric(source[col], errors="coerce")

    source = source.dropna(subset=list(key_cols))
    if source.empty:
        return base
    source = source.drop_duplicates(subset=list(key_cols), keep="last").reset_index(
        drop=True
    )
    for col in key_cols:
        source[col] = source[col].astype("int64", copy=False)

    base_valid_mask = ~base.loc[:, list(key_cols)].isna().any(axis=1)
    if not bool(base_valid_mask.any()):
        return base
    base_valid_positions = np.flatnonzero(base_valid_mask.to_numpy())
    base_keys_valid = base.loc[base_valid_mask, list(key_cols)].copy()
    for col in key_cols:
        base_keys_valid[col] = base_keys_valid[col].astype("int64", copy=False)

    source_key_index = pd.MultiIndex.from_frame(
        source.loc[:, list(key_cols)], names=list(key_cols)
    )
    base_key_index = pd.MultiIndex.from_frame(base_keys_valid, names=list(key_cols))
    key_indexer = source_key_index.get_indexer(base_key_index)
    hit_mask = key_indexer >= 0
    if not bool(hit_mask.any()):
        return base

    hit_base_positions = base_valid_positions[hit_mask]
    hit_source_positions = key_indexer[hit_mask]

    for col in value_cols:
        if col not in base.columns:
            base[col] = pd.NA
        source_values = source[col].to_numpy(copy=False)[hit_source_positions]
        source_notna = pd.notna(source_values)
        if not bool(source_notna.any()):
            continue
        col_idx = base.columns.get_loc(col)
        base.iloc[hit_base_positions[source_notna], col_idx] = source_values[source_notna]

    return base


def _validate_parquet_key_contract(
    path: Path,
    *,
    expected_keys_df: pd.DataFrame,
    key_cols: Sequence[str],
    label: str,
) -> dict[str, Any]:
    key_cols = tuple(str(col) for col in key_cols)
    df = pd.read_parquet(path, columns=list(key_cols))
    _, report = _sanitize_frame_to_expected_keys(
        df,
        expected_keys_df=expected_keys_df,
        key_cols=key_cols,
        label=label,
    )
    if report["dropped_null_key_rows"] > 0 or report["dropped_unexpected_key_rows"] > 0:
        raise RuntimeError(
            f"{label} key contract failed for {path}: "
            f"null_key_rows={report['dropped_null_key_rows']} "
            f"unexpected_key_rows={report['dropped_unexpected_key_rows']}"
        )
    report["path"] = str(path)
    return report


def _load_fallback_merge_baseline(
    *,
    current_path: Path,
    failed_previous_path: Path,
) -> tuple[pd.DataFrame, Path] | None:
    dataset_dir = current_path.parent.parent
    current_run_dir = current_path.parent.name
    filename = current_path.name
    candidates: list[tuple[int, str, Path]] = []

    for run_dir in sorted(dataset_dir.glob("run=*"), reverse=True):
        if not run_dir.is_dir() or run_dir.name >= current_run_dir:
            continue
        candidate_path = run_dir / filename
        if candidate_path == failed_previous_path or not candidate_path.exists():
            continue
        try:
            probe = pd.read_parquet(candidate_path, columns=["game_id"])
        except Exception:
            continue
        candidates.append((_distinct_game_count(probe), run_dir.name, candidate_path))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for game_count, _, candidate_path in candidates:
        try:
            fallback_df = pd.read_parquet(candidate_path)
        except Exception:
            continue
        print(
            "[materialize] promoted baseline unreadable; "
            f"falling back from {failed_previous_path} to {candidate_path} "
            f"(distinct_games={game_count})"
        )
        return fallback_df, candidate_path
    return None


def _merge_parquet_for_target_games(
    *,
    current_path: Path,
    previous_path: Path | None,
    target_game_ids: list[int],
) -> pd.DataFrame:
    current_df = (
        pd.read_parquet(current_path) if current_path.exists() else pd.DataFrame()
    )
    if previous_path is None or not previous_path.exists():
        merged = current_df
    else:
        try:
            previous_df = pd.read_parquet(previous_path)
        except Exception:
            fallback = _load_fallback_merge_baseline(
                current_path=current_path,
                failed_previous_path=previous_path,
            )
            if fallback is None:
                raise
            previous_df, previous_path = fallback
        previous_keep = previous_df
        if "game_id" in previous_df.columns and target_game_ids:
            gids = pd.to_numeric(previous_df["game_id"], errors="coerce").astype(
                "Int64"
            )
            previous_keep = previous_df.loc[~gids.isin(target_game_ids)].copy()
        merged = pd.concat([previous_keep, current_df], ignore_index=True, sort=False)
    merged = _sort_for_stable_write(merged)
    required_cols = (
        ("game_id", "team_id", "player_id")
        if {"game_id", "team_id", "player_id"}.issubset(merged.columns)
        else tuple()
    )
    _atomic_write_validated_parquet(
        merged,
        current_path,
        required_cols=required_cols,
    )
    return merged


def _summarize_world_contracts_from_frame(worlds_df: pd.DataFrame) -> dict[str, Any]:
    if worlds_df.empty:
        return {
            "team_minutes_not_240": 0,
            "team_minutes_total_checks": 0,
            "team_minutes_max_abs_drift": 0.0,
            "minutes_negative": 0,
            "minutes_over_48": 0,
            "negative_stats": 0,
            "fg2m_gt_fga2": 0,
            "fg3m_gt_fga3": 0,
            "ftm_gt_fta": 0,
            "inactive_nonzero_stats": 0,
            "inactive_nonzero_fpts_proxy": 0,
        }
    df = worlds_df.copy()
    numeric_cols = [
        "minutes",
        "fga2",
        "fg2m",
        "fga3",
        "fg3m",
        "fta",
        "ftm",
        "oreb",
        "dreb",
        "ast",
        "stl",
        "blk",
        "tov",
        "pf",
        "pts",
        "reb",
        "dk_fpts",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if {"world_idx", "game_id", "team_id", "minutes"}.issubset(df.columns):
        team_minutes = (
            df.groupby(["world_idx", "game_id", "team_id"], dropna=False)["minutes"]
            .sum()
            .reset_index()
        )
        team_minute_delta = team_minutes["minutes"].sub(240.0).abs()
        team_minutes_not_240 = int(
            (team_minute_delta > _WORLD_CONTRACT_TOL).sum()
        )
        team_minutes_total_checks = int(len(team_minutes))
        team_minutes_max_abs_drift = float(team_minute_delta.max()) if not team_minutes.empty else 0.0
    else:
        team_minutes_not_240 = 0
        team_minutes_total_checks = 0
        team_minutes_max_abs_drift = 0.0
    negative_stats = 0
    for col in ("pts", "reb", "ast", "stl", "blk", "tov"):
        if col in df.columns:
            negative_stats += int((df[col] < -_WORLD_CONTRACT_TOL).sum())
    if "active" in df.columns:
        inactive_mask = (
            pd.to_numeric(df["active"], errors="coerce").fillna(0).astype(int) <= 0
        )
        stat_cols = [
            c
            for c in (
                "pts",
                "reb",
                "ast",
                "stl",
                "blk",
                "tov",
                "fga2",
                "fg2m",
                "fga3",
                "fg3m",
                "fta",
                "ftm",
            )
            if c in df.columns
        ]
        if stat_cols:
            nonzero_stats = (
                df.loc[:, stat_cols].abs().sum(axis=1) > _WORLD_CONTRACT_TOL
            )
            inactive_nonzero_stats = int((inactive_mask & nonzero_stats).sum())
        else:
            inactive_nonzero_stats = 0
        dk_nonzero = (
            pd.to_numeric(df.get("dk_fpts", 0), errors="coerce").fillna(0.0).abs()
            > _WORLD_CONTRACT_TOL
        ) | (
            pd.to_numeric(df.get("minutes", 0), errors="coerce").fillna(0.0).abs()
            > _WORLD_CONTRACT_TOL
        )
        inactive_nonzero_fpts_proxy = int((inactive_mask & dk_nonzero).sum())
    else:
        inactive_nonzero_stats = 0
        inactive_nonzero_fpts_proxy = 0
    return {
        "team_minutes_not_240": team_minutes_not_240,
        "team_minutes_total_checks": team_minutes_total_checks,
        "team_minutes_max_abs_drift": team_minutes_max_abs_drift,
        "minutes_negative": int(
            (df.get("minutes", pd.Series(dtype=float)) < -_WORLD_CONTRACT_TOL).sum()
        ),
        "minutes_over_48": int(
            (
                df.get("minutes", pd.Series(dtype=float))
                > 48.0 + _WORLD_CONTRACT_TOL
            ).sum()
        ),
        "negative_stats": int(negative_stats),
        "fg2m_gt_fga2": int(
            ((df.get("fg2m", 0) - df.get("fga2", 0)) > _WORLD_CONTRACT_TOL).sum()
        ),
        "fg3m_gt_fga3": int(
            ((df.get("fg3m", 0) - df.get("fga3", 0)) > _WORLD_CONTRACT_TOL).sum()
        ),
        "ftm_gt_fta": int(
            ((df.get("ftm", 0) - df.get("fta", 0)) > _WORLD_CONTRACT_TOL).sum()
        ),
        "inactive_nonzero_stats": inactive_nonzero_stats,
        "inactive_nonzero_fpts_proxy": inactive_nonzero_fpts_proxy,
    }


def _repair_world_frame_contract_fields(
    worlds_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Repair known contract-field corruption patterns in sampled worlds."""
    if worlds_df.empty:
        return worlds_df, {"applied": False, "reason": "empty_worlds"}

    # Mutate in place to avoid an extra full-frame copy on large live slates.
    out = worlds_df
    report: dict[str, Any] = {
        "applied": False,
        "game_id_from_norm_rows": 0,
        "fg2m_clipped_to_fga2_rows": 0,
        "fg3m_clipped_to_fga3_rows": 0,
        "ftm_clipped_to_fta_rows": 0,
    }

    if "game_id" in out.columns and "game_id_norm" in out.columns:
        game_id = pd.to_numeric(out["game_id"], errors="coerce").astype("Int64")
        game_id_norm = pd.to_numeric(out["game_id_norm"], errors="coerce").astype("Int64")
        replace_mask = game_id_norm.notna() & game_id.ne(game_id_norm)
        replaced = int(replace_mask.sum())
        if replaced > 0:
            out["game_id"] = game_id.where(~replace_mask, game_id_norm)
            report["game_id_from_norm_rows"] = replaced
            report["applied"] = True

    def _clip_makes_to_attempts(
        attempts_col: str,
        makes_col: str,
        report_key: str,
    ) -> None:
        if attempts_col not in out.columns or makes_col not in out.columns:
            return
        attempts = pd.to_numeric(out[attempts_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        makes = pd.to_numeric(out[makes_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        attempts = np.clip(attempts, a_min=0.0, a_max=None)
        makes = np.clip(makes, a_min=0.0, a_max=None)
        over_mask = makes > (attempts + _WORLD_CONTRACT_TOL)
        clipped = int(np.count_nonzero(over_mask))
        if clipped > 0:
            makes = np.minimum(makes, attempts)
            report["applied"] = True
            report[report_key] = clipped
        out[attempts_col] = attempts
        out[makes_col] = makes

    _clip_makes_to_attempts("fga2", "fg2m", "fg2m_clipped_to_fga2_rows")
    _clip_makes_to_attempts("fga3", "fg3m", "fg3m_clipped_to_fga3_rows")
    _clip_makes_to_attempts("fta", "ftm", "ftm_clipped_to_fta_rows")

    if {"fga2", "fga3"}.issubset(out.columns):
        out["fga"] = (
            pd.to_numeric(out["fga2"], errors="coerce").fillna(0.0)
            + pd.to_numeric(out["fga3"], errors="coerce").fillna(0.0)
        )
    if {"fg2m", "fg3m"}.issubset(out.columns):
        out["fgm"] = (
            pd.to_numeric(out["fg2m"], errors="coerce").fillna(0.0)
            + pd.to_numeric(out["fg3m"], errors="coerce").fillna(0.0)
        )
    if {"fg2m", "fg3m", "ftm"}.issubset(out.columns):
        out["pts"] = (
            2.0 * pd.to_numeric(out["fg2m"], errors="coerce").fillna(0.0)
            + 3.0 * pd.to_numeric(out["fg3m"], errors="coerce").fillna(0.0)
            + pd.to_numeric(out["ftm"], errors="coerce").fillna(0.0)
        )
    if {"oreb", "dreb"}.issubset(out.columns):
        out["reb"] = (
            pd.to_numeric(out["oreb"], errors="coerce").fillna(0.0)
            + pd.to_numeric(out["dreb"], errors="coerce").fillna(0.0)
        )
    if {"pts", "reb", "ast", "stl", "blk", "tov"}.issubset(out.columns):
        out["dk_fpts"] = _recompute_dk_fpts(out)

    return out, report


def _recompute_dk_fpts(worlds_df: pd.DataFrame) -> pd.Series:
    pts = pd.to_numeric(worlds_df.get("pts", 0.0), errors="coerce").fillna(0.0)
    reb = pd.to_numeric(worlds_df.get("reb", 0.0), errors="coerce").fillna(0.0)
    ast = pd.to_numeric(worlds_df.get("ast", 0.0), errors="coerce").fillna(0.0)
    stl = pd.to_numeric(worlds_df.get("stl", 0.0), errors="coerce").fillna(0.0)
    blk = pd.to_numeric(worlds_df.get("blk", 0.0), errors="coerce").fillna(0.0)
    tov = pd.to_numeric(worlds_df.get("tov", 0.0), errors="coerce").fillna(0.0)
    base = pts + 1.25 * reb + 1.5 * ast + 2.0 * stl + 2.0 * blk - 0.5 * tov
    qualifying = pd.concat([pts, reb, ast, stl, blk], axis=1).ge(10.0).sum(axis=1)
    bonus_dd = qualifying.eq(2).astype(float) * 1.5
    bonus_td = qualifying.ge(3).astype(float) * 3.0
    return (base + bonus_dd + bonus_td).clip(lower=0.0)


def _apply_low_minutes_tail_damping_to_worlds(
    worlds_df: pd.DataFrame,
    *,
    minutes_threshold: float = 12.0,
    min_scale: float = 0.55,
    target_game_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Shrink low-minute tail residuals toward each player's world mean."""
    if worlds_df.empty:
        return worlds_df, {"applied": False, "reason": "empty_worlds"}
    required = {"game_id", "team_id", "player_id", "minutes", "pts", "reb", "ast", "dk_fpts"}
    if not required.issubset(worlds_df.columns):
        return worlds_df, {
            "applied": False,
            "reason": "missing_required_columns",
            "missing_columns": sorted(required - set(worlds_df.columns)),
        }
    if minutes_threshold <= 0.0:
        return worlds_df, {
            "applied": False,
            "reason": "invalid_minutes_threshold",
            "minutes_threshold": float(minutes_threshold),
        }

    low = float(minutes_threshold)
    floor_scale = float(np.clip(min_scale, 0.0, 1.0))
    out = worlds_df.copy()
    minutes = pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ramp = np.clip((low - minutes) / low, 0.0, 1.0)
    scale = 1.0 - (1.0 - floor_scale) * ramp
    damp_mask = (minutes > 0.0) & (minutes < low)

    if target_game_ids:
        game_ids = pd.to_numeric(out["game_id"], errors="coerce").astype("Int64")
        damp_mask = damp_mask & game_ids.isin(sorted(target_game_ids)).to_numpy(dtype=bool)

    if not bool(damp_mask.any()):
        return out, {
            "applied": False,
            "reason": "no_low_minutes_rows",
            "minutes_threshold": low,
            "min_scale": floor_scale,
            "target_game_count": int(len(target_game_ids or set())),
        }

    key_cols = ["game_id", "team_id", "player_id"]
    stat_cols = [c for c in ("pts", "reb", "ast", "stl", "blk", "tov") if c in out.columns]
    for col in stat_cols:
        x = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        mu = x.groupby([out[k] for k in key_cols], dropna=False).transform("mean")
        new_vals = (mu + pd.Series(scale, index=out.index) * (x - mu)).clip(lower=0.0)
        out[col] = np.where(damp_mask, new_vals.to_numpy(dtype=float), x.to_numpy(dtype=float))

    if {"oreb", "dreb", "reb"}.issubset(out.columns):
        oreb = pd.to_numeric(out["oreb"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        dreb = pd.to_numeric(out["dreb"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        reb = pd.to_numeric(out["reb"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        split_sum = np.maximum(oreb + dreb, 1e-6)
        oreb_share = np.divide(oreb, split_sum)
        oreb_new = reb * oreb_share
        dreb_new = reb * (1.0 - oreb_share)
        out["oreb"] = np.where(damp_mask, oreb_new, oreb)
        out["dreb"] = np.where(damp_mask, dreb_new, dreb)

    out["dk_fpts"] = _recompute_dk_fpts(out)
    report = {
        "applied": True,
        "minutes_threshold": low,
        "min_scale": floor_scale,
        "target_game_count": int(len(target_game_ids or set())),
        "affected_rows": int(np.count_nonzero(damp_mask)),
        "affected_players": int(
            out.loc[damp_mask, key_cols].drop_duplicates().shape[0]
            if np.count_nonzero(damp_mask) > 0
            else 0
        ),
        "scale_mean": float(np.mean(scale[damp_mask])) if np.count_nonzero(damp_mask) > 0 else 1.0,
        "scale_p10": float(np.quantile(scale[damp_mask], 0.10)) if np.count_nonzero(damp_mask) > 0 else 1.0,
        "scale_p90": float(np.quantile(scale[damp_mask], 0.90)) if np.count_nonzero(damp_mask) > 0 else 1.0,
    }
    return out, report


def _resample_extreme_game_worlds(
    worlds_df: pd.DataFrame,
    *,
    random_seed: int,
    max_passes: int = 1,
    short_minutes_threshold: float = 12.0,
    short_minutes_dk_threshold: float = _WORLD_REALISM_SHORT_MINUTES_DK_THRESHOLD,
    game_pts_max: float = _WORLD_REALISM_GAME_PTS_MAX_THRESHOLD,
    game_pts_min: float = _WORLD_REALISM_GAME_PTS_MIN_THRESHOLD,
    target_game_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace extreme game-world pairs with sampled in-game donor worlds."""
    if worlds_df.empty:
        return worlds_df, {"applied": False, "reason": "empty_worlds"}
    required = {"world_idx", "game_id", "team_id", "player_id", "minutes", "pts", "dk_fpts"}
    if not required.issubset(worlds_df.columns):
        return worlds_df, {
            "applied": False,
            "reason": "missing_required_columns",
            "missing_columns": sorted(required - set(worlds_df.columns)),
        }

    out = worlds_df.copy()
    key_cols = ["game_id", "team_id", "player_id"]
    pair_cols = ["world_idx", "game_id"]
    max_iter = max(0, int(max_passes))
    if max_iter == 0:
        return out, {"applied": False, "reason": "disabled_max_passes"}

    rng = np.random.default_rng(int(random_seed))
    pass_reports: list[dict[str, Any]] = []
    total_replaced = 0

    for pass_idx in range(max_iter):
        minutes = pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0)
        dk = pd.to_numeric(out["dk_fpts"], errors="coerce").fillna(0.0)
        pts = pd.to_numeric(out["pts"], errors="coerce").fillna(0.0)
        game_id = pd.to_numeric(out["game_id"], errors="coerce").astype("Int64")

        row_spike = (minutes < float(short_minutes_threshold)) & (
            dk > float(short_minutes_dk_threshold)
        )
        if target_game_ids:
            row_spike = row_spike & game_id.isin(sorted(target_game_ids))

        pair_short = (
            out.loc[row_spike, pair_cols]
            .drop_duplicates()
            .assign(short_spike=True)
        )

        game_pts = (
            out.assign(_pts=pts)
            .groupby(pair_cols, dropna=False, as_index=False)
            .agg(game_pts=("_pts", "sum"))
        )
        if target_game_ids:
            game_pts = game_pts.loc[
                pd.to_numeric(game_pts["game_id"], errors="coerce")
                .astype("Int64")
                .isin(sorted(target_game_ids))
            ].copy()
        pair_hi = game_pts.loc[
            game_pts["game_pts"] > float(game_pts_max), pair_cols
        ].drop_duplicates().assign(game_hi=True)
        pair_lo = game_pts.loc[
            game_pts["game_pts"] < float(game_pts_min), pair_cols
        ].drop_duplicates().assign(game_lo=True)

        pair_flags = (
            game_pts.loc[:, pair_cols]
            .drop_duplicates()
            .merge(pair_short, on=pair_cols, how="left")
            .merge(pair_hi, on=pair_cols, how="left")
            .merge(pair_lo, on=pair_cols, how="left")
        )
        for flag_col in ("short_spike", "game_hi", "game_lo"):
            if flag_col not in pair_flags.columns:
                pair_flags[flag_col] = False
            else:
                pair_flags[flag_col] = pair_flags[flag_col].eq(True)
        pair_flags["is_bad"] = (
            pair_flags["short_spike"] | pair_flags["game_hi"] | pair_flags["game_lo"]
        )

        bad_pairs = pair_flags.loc[pair_flags["is_bad"], pair_cols].copy()
        if bad_pairs.empty:
            break

        good_by_game: dict[int, list[int]] = {}
        for gid, grp in pair_flags.groupby("game_id", dropna=False):
            gid_i = int(gid)
            goods = grp.loc[~grp["is_bad"], "world_idx"].astype(int).tolist()
            good_by_game[gid_i] = goods

        replaced_this_pass = 0
        skipped_no_donor = 0
        skipped_key_mismatch = 0
        for row in bad_pairs.sort_values(pair_cols).itertuples(index=False):
            target_world = int(row.world_idx)
            gid = int(row.game_id)
            donors = good_by_game.get(gid, [])
            if not donors:
                skipped_no_donor += 1
                continue
            donor_world = int(rng.choice(donors))
            target_rows = out.loc[
                (pd.to_numeric(out["world_idx"], errors="coerce").astype("Int64") == target_world)
                & (pd.to_numeric(out["game_id"], errors="coerce").astype("Int64") == gid)
            ].copy()
            donor_rows = out.loc[
                (pd.to_numeric(out["world_idx"], errors="coerce").astype("Int64") == donor_world)
                & (pd.to_numeric(out["game_id"], errors="coerce").astype("Int64") == gid)
            ].copy()
            if target_rows.empty or donor_rows.empty:
                skipped_no_donor += 1
                continue
            target_rows = target_rows.sort_values(key_cols)
            donor_rows = donor_rows.sort_values(key_cols)
            if (
                len(target_rows) != len(donor_rows)
                or not target_rows[key_cols].reset_index(drop=True).equals(
                    donor_rows[key_cols].reset_index(drop=True)
                )
            ):
                skipped_key_mismatch += 1
                continue
            replace_cols = [c for c in out.columns if c != "world_idx"]
            target_idx = target_rows.index.to_numpy()
            for col in replace_cols:
                out.loc[target_idx, col] = donor_rows[col].to_numpy()
            replaced_this_pass += 1

        pass_reports.append(
            {
                "pass_idx": int(pass_idx + 1),
                "bad_pair_count": int(len(bad_pairs)),
                "bad_short_spike_count": int(pair_flags["short_spike"].sum()),
                "bad_game_hi_count": int(pair_flags["game_hi"].sum()),
                "bad_game_lo_count": int(pair_flags["game_lo"].sum()),
                "replaced_pair_count": int(replaced_this_pass),
                "skipped_no_donor": int(skipped_no_donor),
                "skipped_key_mismatch": int(skipped_key_mismatch),
            }
        )
        total_replaced += int(replaced_this_pass)
        if replaced_this_pass == 0:
            break

    report = {
        "applied": bool(total_replaced > 0),
        "random_seed": int(random_seed),
        "max_passes": int(max_iter),
        "target_game_count": int(len(target_game_ids or set())),
        "short_minutes_threshold": float(short_minutes_threshold),
        "short_minutes_dk_threshold": float(short_minutes_dk_threshold),
        "game_pts_max": float(game_pts_max),
        "game_pts_min": float(game_pts_min),
        "total_replaced_pairs": int(total_replaced),
        "passes": pass_reports,
    }
    if total_replaced == 0 and not pass_reports:
        report["applied"] = False
        report["reason"] = "no_outlier_pairs"
    return out, report


def _apply_world_realism_controls_to_worlds(
    worlds_df: pd.DataFrame,
    *,
    enabled: bool,
    random_seed: int,
    low_minutes_tail_damping_enabled: bool,
    low_minutes_tail_minutes_threshold: float,
    low_minutes_tail_min_scale: float,
    outlier_resample_enabled: bool,
    outlier_resample_max_passes: int,
    target_game_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not enabled:
        return worlds_df, {"applied": False, "reason": "disabled"}

    out = worlds_df.copy()
    report: dict[str, Any] = {"applied": False}
    if low_minutes_tail_damping_enabled:
        out, damp_report = _apply_low_minutes_tail_damping_to_worlds(
            out,
            minutes_threshold=float(low_minutes_tail_minutes_threshold),
            min_scale=float(low_minutes_tail_min_scale),
            target_game_ids=target_game_ids,
        )
    else:
        damp_report = {"applied": False, "reason": "disabled"}
    report["low_minutes_tail_damping"] = damp_report

    if outlier_resample_enabled:
        out, resample_report = _resample_extreme_game_worlds(
            out,
            random_seed=int(random_seed),
            max_passes=int(outlier_resample_max_passes),
            target_game_ids=target_game_ids,
        )
    else:
        resample_report = {"applied": False, "reason": "disabled"}
    report["outlier_resample"] = resample_report
    report["applied"] = bool(
        bool((damp_report or {}).get("applied"))
        or bool((resample_report or {}).get("applied"))
    )
    return out, report


def _apply_props_uplift_calibration_to_worlds(
    worlds_df: pd.DataFrame,
    *,
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply one-sided stat uplifts with tail broadening for undercalled prop-heavy players."""
    if worlds_df.empty:
        return worlds_df, {"applied": False, "reason": "empty_worlds"}

    required_world_cols = {
        "game_id",
        "team_id",
        "player_id",
        "pts",
        "reb",
        "ast",
        "stl",
        "blk",
        "tov",
        "dk_fpts",
    }
    if not required_world_cols.issubset(worlds_df.columns):
        return worlds_df, {
            "applied": False,
            "reason": "missing_world_cols",
            "missing_world_cols": sorted(required_world_cols - set(worlds_df.columns)),
        }

    required_feature_cols = {"game_id", "team_id", "player_id"}
    if not required_feature_cols.issubset(features_df.columns):
        return worlds_df, {
            "applied": False,
            "reason": "missing_feature_keys",
            "missing_feature_cols": sorted(required_feature_cols - set(features_df.columns)),
        }

    stat_cfg: dict[str, dict[str, float | str]] = {
        "pts": {
            "line_col": "an_pts_line",
            "has_col": "an_has_pts",
            "min_line": 20.0,
            "min_gap": 2.5,
            "weight": 0.88,
            "max_scale": 2.0,
            "var_weight": 0.40,
            "max_var_scale": 1.50,
            "line_anchor_min_line": 28.0,
            "line_anchor_frac": 0.93,
            "min_line_down": 12.0,
            "min_gap_down": 2.5,
            "weight_down": 0.45,
            "min_scale_down": 0.70,
            "var_weight_down": 0.15,
            "min_var_scale_down": 0.80,
        },
        "reb": {
            "line_col": "an_reb_line",
            "has_col": "an_has_reb",
            "min_line": 7.0,
            "min_gap": 1.5,
            "weight": 0.92,
            "max_scale": 2.2,
            "var_weight": 0.45,
            "max_var_scale": 1.60,
            "line_anchor_min_line": 10.0,
            "line_anchor_frac": 0.92,
            "min_line_down": 3.0,
            "min_gap_down": 1.3,
            "weight_down": 0.60,
            "min_scale_down": 0.55,
            "var_weight_down": 0.25,
            "min_var_scale_down": 0.75,
        },
        "ast": {
            "line_col": "an_ast_line",
            "has_col": "an_has_ast",
            "min_line": 5.5,
            "min_gap": 1.0,
            "weight": 0.92,
            "max_scale": 2.2,
            "var_weight": 0.50,
            "max_var_scale": 1.65,
            "line_anchor_min_line": 8.0,
            "line_anchor_frac": 0.92,
            "min_line_down": 1.5,
            "min_gap_down": 1.0,
            "weight_down": 0.65,
            "min_scale_down": 0.50,
            "var_weight_down": 0.25,
            "min_var_scale_down": 0.72,
        },
    }
    key_cols = ["game_id", "team_id", "player_id"]

    player_means = (
        worlds_df.groupby(key_cols, dropna=False)[["pts", "reb", "ast"]]
        .mean()
        .reset_index()
        .rename(columns={"pts": "pts_mean", "reb": "reb_mean", "ast": "ast_mean"})
    )

    feat_cols = list(key_cols)
    for cfg in stat_cfg.values():
        line_col = str(cfg["line_col"])
        has_col = str(cfg["has_col"])
        if line_col in features_df.columns:
            feat_cols.append(line_col)
        if has_col in features_df.columns:
            feat_cols.append(has_col)
    feat = features_df.loc[:, sorted(set(feat_cols), key=feat_cols.index)].copy()

    agg_dict: dict[str, str] = {}
    for col in feat.columns:
        if col in key_cols:
            continue
        agg_dict[col] = "max" if col.startswith("an_has_") else "first"
    feat = feat.groupby(key_cols, dropna=False, as_index=False).agg(agg_dict)

    meta = player_means.merge(feat, on=key_cols, how="left")
    if "player_name" in features_df.columns:
        names = (
            features_df.loc[:, key_cols + ["player_name"]]
            .drop_duplicates(subset=key_cols, keep="last")
            .copy()
        )
        meta = meta.merge(names, on=key_cols, how="left")
    for cfg in stat_cfg.values():
        line_col = str(cfg["line_col"])
        has_col = str(cfg["has_col"])
        if line_col in meta.columns:
            meta[line_col] = pd.to_numeric(meta[line_col], errors="coerce")
        if has_col in meta.columns:
            meta[has_col] = pd.to_numeric(meta[has_col], errors="coerce").fillna(0.0)

    out = worlds_df.copy()
    report: dict[str, Any] = {"applied": True, "stats": {}}
    adjusted_key_frames: list[pd.DataFrame] = []

    for stat_name, cfg in stat_cfg.items():
        line_col = str(cfg["line_col"])
        has_col = str(cfg["has_col"])
        mean_col = f"{stat_name}_mean"
        if line_col not in meta.columns or mean_col not in meta.columns:
            report["stats"][stat_name] = {
                "applied_player_count": 0,
                "reason": "missing_line_or_mean_column",
            }
            continue

        line = pd.to_numeric(meta[line_col], errors="coerce")
        mean = pd.to_numeric(meta[mean_col], errors="coerce")
        gap = line - mean
        denom = line.clip(lower=1.0)
        has_market = pd.Series(True, index=meta.index, dtype=bool)
        if has_col in meta.columns:
            has_market = pd.to_numeric(meta[has_col], errors="coerce").fillna(0.0).ge(0.5)
        mask_up = line.ge(float(cfg["min_line"])) & gap.ge(float(cfg["min_gap"])) & mean.gt(0.0) & has_market
        over_gap = mean - line
        mask_down = (
            line.ge(float(cfg["min_line_down"]))
            & over_gap.ge(float(cfg["min_gap_down"]))
            & mean.gt(0.0)
            & has_market
        )

        target_up = mean + float(cfg["weight"]) * gap
        target_up = target_up.where(
            line.lt(float(cfg["line_anchor_min_line"])),
            np.maximum(
                pd.to_numeric(target_up, errors="coerce").to_numpy(dtype=float),
                float(cfg["line_anchor_frac"]) * pd.to_numeric(line, errors="coerce").to_numpy(dtype=float),
            ),
        )
        scale_up = (target_up / mean).clip(lower=1.0, upper=float(cfg["max_scale"]))
        var_scale_up = (
            1.0 + float(cfg["var_weight"]) * (gap / denom).clip(lower=0.0)
        ).clip(lower=1.0, upper=float(cfg["max_var_scale"]))
        target_down = mean - float(cfg["weight_down"]) * over_gap
        scale_down = (target_down / mean).clip(
            lower=float(cfg["min_scale_down"]),
            upper=1.0,
        )
        var_scale_down = (
            1.0 - float(cfg["var_weight_down"]) * (over_gap / denom).clip(lower=0.0)
        ).clip(lower=float(cfg["min_var_scale_down"]), upper=1.0)

        up_df = meta.loc[mask_up, key_cols].copy()
        down_df = meta.loc[mask_down, key_cols].copy()
        if "player_name" in meta.columns:
            up_df["player_name"] = meta.loc[mask_up, "player_name"].astype(str).values
            down_df["player_name"] = meta.loc[mask_down, "player_name"].astype(str).values
        up_df["mu"] = mean.loc[mask_up].astype(float).values
        up_df["sf_mean"] = scale_up.loc[mask_up].astype(float).values
        up_df["sf_var"] = var_scale_up.loc[mask_up].astype(float).values
        up_df["line_gap"] = gap.loc[mask_up].astype(float).values
        up_df["direction"] = "up"

        down_df["mu"] = mean.loc[mask_down].astype(float).values
        down_df["sf_mean"] = scale_down.loc[mask_down].astype(float).values
        down_df["sf_var"] = var_scale_down.loc[mask_down].astype(float).values
        down_df["line_gap"] = gap.loc[mask_down].astype(float).values
        down_df["direction"] = "down"

        scale_df = pd.concat([up_df, down_df], ignore_index=True)

        if scale_df.empty:
            report["stats"][stat_name] = {
                "applied_player_count": 0,
                "applied_player_count_up": 0,
                "applied_player_count_down": 0,
                "mean_gap_pre": float((mean - line).mean()) if (mean - line).notna().any() else float("nan"),
                "mean_gap_post": float((mean - line).mean()) if (mean - line).notna().any() else float("nan"),
            }
            continue

        adjusted_key_frames.append(scale_df.loc[:, key_cols].copy())
        # Keep report-only fields (e.g. direction/line_gap/player_name) out of the
        # simulation frame to avoid suffix collisions across per-stat passes.
        scale_apply = scale_df.loc[:, key_cols + ["mu", "sf_mean", "sf_var"]]
        out = out.merge(scale_apply, on=key_cols, how="left")
        mu = pd.to_numeric(out["mu"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        sf_mean = pd.to_numeric(out["sf_mean"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        sf_var = pd.to_numeric(out["sf_var"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        target_mu = mu * sf_mean
        if "minutes" in out.columns:
            active_mask = pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.0
        else:
            active_mask = pd.to_numeric(out["dk_fpts"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.0
        if stat_name == "pts":
            x = pd.to_numeric(out["pts"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            pts_new = np.clip(target_mu + sf_var * (x - mu), 0.0, None)
            out["pts"] = np.where(active_mask, pts_new, x)
        elif stat_name == "reb":
            x = pd.to_numeric(out["reb"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            reb_new = np.clip(target_mu + sf_var * (x - mu), 0.0, None)
            reb_new = np.where(active_mask, reb_new, x)
            if "oreb" in out.columns and "dreb" in out.columns:
                oreb = pd.to_numeric(out["oreb"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                dreb = pd.to_numeric(out["dreb"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                reb_split_sum = np.maximum(oreb + dreb, 1e-6)
                oreb_share = np.divide(oreb, reb_split_sum)
                out["oreb"] = np.where(active_mask, reb_new * oreb_share, oreb)
                out["dreb"] = np.where(active_mask, reb_new * (1.0 - oreb_share), dreb)
            out["reb"] = reb_new
        elif stat_name == "ast":
            x = pd.to_numeric(out["ast"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            ast_new = np.clip(target_mu + sf_var * (x - mu), 0.0, None)
            out["ast"] = np.where(active_mask, ast_new, x)
        out = out.drop(columns=["mu", "sf_mean", "sf_var", "line_gap", "player_name"], errors="ignore")

        post_means = (
            out.groupby(key_cols, dropna=False)[[stat_name]]
            .mean()
            .reset_index()
            .rename(columns={stat_name: f"{stat_name}_mean_post"})
        )
        merged_gap = meta.merge(post_means, on=key_cols, how="left")
        gap_pre = pd.to_numeric(merged_gap[mean_col], errors="coerce") - pd.to_numeric(merged_gap[line_col], errors="coerce")
        gap_post = pd.to_numeric(merged_gap[f"{stat_name}_mean_post"], errors="coerce") - pd.to_numeric(
            merged_gap[line_col], errors="coerce"
        )
        report["stats"][stat_name] = {
            "applied_player_count": int(len(scale_df)),
            "applied_player_count_up": int(len(up_df)),
            "applied_player_count_down": int(len(down_df)),
            "mean_gap_pre": float(gap_pre.mean()) if gap_pre.notna().any() else float("nan"),
            "mean_gap_post": float(gap_post.mean()) if gap_post.notna().any() else float("nan"),
            "median_gap_pre": float(gap_pre.median()) if gap_pre.notna().any() else float("nan"),
            "median_gap_post": float(gap_post.median()) if gap_post.notna().any() else float("nan"),
            "mean_scale_mean": float(scale_df["sf_mean"].mean()),
            "mean_scale_p90": float(scale_df["sf_mean"].quantile(0.90)),
            "var_scale_mean": float(scale_df["sf_var"].mean()),
        }
        top_cols = [c for c in ["player_name", "player_id", "direction", "line_gap", "sf_mean", "sf_var"] if c in scale_df.columns]
        top_rows = (
            scale_df.loc[:, top_cols]
            .assign(abs_line_gap=lambda d: pd.to_numeric(d["line_gap"], errors="coerce").abs())
            .sort_values("abs_line_gap", ascending=False)
            .head(8)
            .drop(columns=["abs_line_gap"], errors="ignore")
            .replace([np.inf, -np.inf], np.nan)
            .fillna("")
        )
        report["stats"][stat_name]["top_adjustments"] = top_rows.to_dict(orient="records")

    out["dk_fpts"] = _recompute_dk_fpts(out)

    report["total_adjustment_events"] = int(
        sum(int((report["stats"].get(s) or {}).get("applied_player_count", 0)) for s in stat_cfg)
    )
    if adjusted_key_frames:
        report["total_adjusted_players"] = int(
            len(pd.concat(adjusted_key_frames, ignore_index=True).drop_duplicates(subset=key_cols))
        )
    else:
        report["total_adjusted_players"] = 0
    return out, report


def _resolve_previous_run_file(*, dataset_dir: Path, filename: str) -> Path | None:
    run_id = control_plane.read_promoted_run_id(dataset_dir)
    if not run_id:
        return None
    candidate = dataset_dir / f"run={run_id}" / filename
    return candidate if candidate.exists() else None


def _build_feature_input_checklist(
    *,
    game_date: str,
    run_as_of_ts: str,
    data_root: Path,
    allow_priors_fallback: bool,
    allow_rotowire_props_fallback: bool = False,
    require_action_props: bool = True,
) -> dict[str, Any]:
    day = pd.Timestamp(game_date).normalize()
    run_ts = pd.to_datetime(run_as_of_ts, utc=True, errors="coerce")
    if pd.isna(run_ts):
        raise RuntimeError(f"invalid run_as_of_ts: {run_as_of_ts}")

    season, month = _resolve_season_month(game_date)
    schedule_path = (
        data_root
        / "silver"
        / "schedule"
        / f"season={season}"
        / f"month={month:02d}"
        / "schedule.parquet"
    )
    roster_path = (
        data_root
        / "silver"
        / "roster_nightly"
        / f"season={season}"
        / f"month={month:02d}"
        / "roster.parquet"
    )
    odds_path = (
        data_root
        / "silver"
        / "odds_snapshot"
        / f"season={season}"
        / f"month={month:02d}"
        / "odds_snapshot.parquet"
    )
    injuries_silver_path = (
        data_root
        / "silver"
        / "injuries_snapshot"
        / f"season={season}"
        / f"month={month:02d}"
        / "injuries_snapshot.parquet"
    )
    rotowire_path = (
        data_root
        / "silver"
        / "rotowire_lineups"
        / f"date={day.date()}"
        / "lineups.parquet"
    )
    labels_gold_root = data_root / "gold" / "labels_minutes_v1" / f"season={season}"
    labels_legacy_path = (
        data_root / "labels" / f"season={season}" / "boxscore_labels.parquet"
    )
    priors_team_root = (
        data_root
        / "silver"
        / "rotation_priors_v1"
        / "team_game_priors"
        / f"season={season}"
    )
    priors_player_root = (
        data_root
        / "silver"
        / "rotation_priors_v1"
        / "player_game_priors"
        / f"season={season}"
    )

    checks: list[dict[str, Any]] = []

    schedule_df = _read_parquet_if_exists(schedule_path)
    schedule_days = (
        pd.to_datetime(schedule_df.get("game_date"), errors="coerce").dt.normalize()
        if not schedule_df.empty
        else pd.Series(dtype="datetime64[ns]")
    )
    slate_df = (
        schedule_df.loc[schedule_days == day].copy()
        if not schedule_df.empty
        else pd.DataFrame()
    )
    slate_game_ids = (
        pd.to_numeric(slate_df.get("game_id"), errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
        if not slate_df.empty
        else []
    )
    expected_props_teams: set[str] = set()
    if not slate_df.empty:
        for team_col in ("home_team_tricode", "away_team_tricode"):
            if team_col not in slate_df.columns:
                continue
            vals = (
                slate_df[team_col].dropna().astype(str).str.strip().str.upper().tolist()
            )
            expected_props_teams.update(
                _normalize_props_team_abbr(v) for v in vals if str(v).strip()
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

    def _snapshot_check(
        name: str, path: Path, *, required: bool = True
    ) -> pd.DataFrame:
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

    roster_slate = _snapshot_check(
        "roster_snapshot_slate_rows", roster_path, required=True
    )
    odds_slate = _snapshot_check("odds_snapshot_slate_rows", odds_path, required=True)
    injuries_silver_slate = _snapshot_check(
        "injuries_snapshot_silver_slate_rows", injuries_silver_path, required=False
    )

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
    injuries_bronze = (
        pd.concat(bronze_frames, ignore_index=True) if bronze_frames else pd.DataFrame()
    )
    injuries_bronze_slate = _filter_slate_rows(injuries_bronze, slate_game_ids)
    injury_rows_ok = bool(
        not injuries_bronze_slate.empty or not injuries_silver_slate.empty
    )
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
    rotowire_slate = _filter_slate_rows(rotowire_df, slate_game_ids)
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

    rotowire_props_root = data_root / "bronze" / "props"
    action_props_day = day.date().isoformat()
    action_props_next_day = (day + pd.Timedelta(days=1)).date().isoformat()
    rotowire_raw_files = sorted(
        (rotowire_props_root / f"game_date={action_props_day}").glob("*.parquet")
    ) + sorted(
        (rotowire_props_root / f"game_date={action_props_next_day}").glob("*.parquet")
    )
    rotowire_props_summary = {
        "parsed_rows": 0,
        "latest_action_props_as_of_ts": None,
        "teams": [],
        "team_latest_as_of_ts": {},
        "parse_error": None,
    }
    if rotowire_props_root.exists():
        rotowire_props_summary = _probe_rotowire_props_snapshot_summary(
            rotowire_props_root=rotowire_props_root,
            game_date=day,
            data_root=data_root,
            run_as_of_ts=run_ts,
        )
    rotowire_parse_error = rotowire_props_summary.get("parse_error")

    checks.append(
        {
            "name": "rotowire_props_raw_files",
            "required": False,
            "ok": bool(len(rotowire_raw_files) > 0),
            "details": {
                "root": str(rotowire_props_root),
                "day_partition": str(
                    rotowire_props_root / f"game_date={action_props_day}"
                ),
                "next_day_partition": str(
                    rotowire_props_root / f"game_date={action_props_next_day}"
                ),
                "raw_file_count": int(len(rotowire_raw_files)),
            },
        }
    )
    latest_rotowire_props_ts = pd.to_datetime(
        rotowire_props_summary.get("latest_action_props_as_of_ts"),
        utc=True,
        errors="coerce",
    )
    checks.append(
        {
            "name": "rotowire_props_parsed_snapshots",
            "required": False,
            "ok": bool(
                int(rotowire_props_summary.get("parsed_rows", 0)) > 0
                and rotowire_parse_error is None
            ),
            "details": {
                "parsed_rows": int(rotowire_props_summary.get("parsed_rows", 0)),
                "latest_action_props_as_of_ts": None
                if pd.isna(latest_rotowire_props_ts)
                else latest_rotowire_props_ts.isoformat(),
                "parse_error": rotowire_parse_error,
            },
        }
    )
    rotowire_props_teams = {
        _normalize_props_team_abbr(team)
        for team in rotowire_props_summary.get("teams", [])
        if str(team).strip()
    }
    rotowire_props_team_latest = pd.DataFrame(
        {
            "team_tricode": list(
                (rotowire_props_summary.get("team_latest_as_of_ts") or {}).keys()
            ),
            "action_props_as_of_ts": list(
                (rotowire_props_summary.get("team_latest_as_of_ts") or {}).values()
            ),
        }
    )
    rotowire_props_team_overlap = rotowire_props_teams.intersection(
        expected_props_teams
    )
    rotowire_overlap_ok = bool(
        (not expected_props_teams) or rotowire_props_team_overlap
    )
    checks.append(
        {
            "name": "rotowire_props_team_overlap",
            "required": False,
            "ok": bool(rotowire_overlap_ok),
            "details": {
                "expected_slate_team_count": int(len(expected_props_teams)),
                "snapshot_team_count": int(len(rotowire_props_teams)),
                "overlap_team_count": int(len(rotowire_props_team_overlap)),
                "expected_slate_teams": sorted(expected_props_teams),
                "overlap_teams": sorted(rotowire_props_team_overlap),
            },
        }
    )
    rotowire_ok = bool(
        int(rotowire_props_summary.get("parsed_rows", 0)) > 0
        and rotowire_parse_error is None
        and rotowire_overlap_ok
    )
    policy_ok = (not require_action_props) or rotowire_ok
    checks.append(
        {
            "name": "props_source_policy_satisfied",
            "required": True,
            "ok": bool(policy_ok),
            "details": {
                "require_action_props": bool(require_action_props),
                "allow_rotowire_props_fallback": bool(allow_rotowire_props_fallback),
                "live_props_source": "rotowire",
                "rotowire_ok": bool(rotowire_ok),
                "selected_source": "rotowire" if rotowire_ok else "none",
            },
        }
    )

    team_partitions = (
        list(priors_team_root.glob("game_id=*.parquet"))
        if priors_team_root.exists()
        else []
    )
    player_partitions = (
        list(priors_player_root.glob("game_id=*.parquet"))
        if priors_player_root.exists()
        else []
    )
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
    all_gameid_missing = (
        bool(slate_game_ids)
        and len(missing_team) == len(slate_game_ids)
        and len(missing_player) == len(slate_game_ids)
    )
    checks.append(
        {
            "name": "rotation_priors_gameid_partition_coverage",
            "required": False,
            "ok": bool(not all_gameid_missing),
            "details": {
                "slate_games": int(len(slate_game_ids)),
                "present_team_partitions": int(len(slate_game_ids) - len(missing_team)),
                "present_player_partitions": int(
                    len(slate_game_ids) - len(missing_player)
                ),
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

    selected_props_source = "rotowire" if rotowire_ok else "none"
    manual_override_summary = manual_override_report(
        date.fromisoformat(game_date),
        data_root=data_root,
        as_of_ts=run_ts,
    )
    schedule_tip_by_game = _latest_ts_by_game(slate_df, time_col="tip_ts")
    odds_latest_by_game = _latest_ts_by_game(odds_slate, time_col="as_of_ts")
    roster_latest_by_game = _latest_ts_by_game(roster_slate, time_col="as_of_ts")
    roster_digest_by_game = _content_digest_by_game(
        roster_slate,
        slate_game_ids,
        exclude_columns={"as_of_ts", "game_date", "created_at", "updated_at"},
    )
    odds_digest_by_game = _content_digest_by_game(
        odds_slate,
        slate_game_ids,
        exclude_columns={"as_of_ts", "created_at", "updated_at", "snapshot_ts"},
    )
    injuries_bronze_latest_by_game = _latest_ts_by_game(
        injuries_bronze_slate, time_col="as_of_ts"
    )
    injuries_bronze_digest_by_game = _content_digest_by_game(
        injuries_bronze_slate,
        slate_game_ids,
        exclude_columns={"as_of_ts", "ingested_ts", "created_at", "updated_at"},
    )
    injuries_silver_latest_by_game = _latest_ts_by_game(
        injuries_silver_slate, time_col="as_of_ts"
    )
    injuries_silver_digest_by_game = _content_digest_by_game(
        injuries_silver_slate,
        slate_game_ids,
        exclude_columns={"as_of_ts", "ingested_ts", "created_at", "updated_at"},
    )
    rotowire_latest_by_game = (
        _latest_ts_by_game(rotowire_slate, time_col="ingested_ts")
        if not rotowire_slate.empty
        else _latest_ts_by_game_from_teams(
            slate_df, rotowire_df, time_col="ingested_ts"
        )
    )
    rotowire_digest_by_game = (
        _content_digest_by_game(
            rotowire_slate,
            slate_game_ids,
            exclude_columns={
                "ingested_ts",
                "lineup_timestamp",
                "created_at",
                "updated_at",
            },
        )
        if not rotowire_slate.empty
        else _content_digest_by_game_from_teams(
            slate_df,
            rotowire_df,
            exclude_columns={
                "ingested_ts",
                "lineup_timestamp",
                "created_at",
                "updated_at",
            },
        )
    )
    rotowire_props_latest_by_game = _latest_ts_by_game_from_teams(
        slate_df,
        rotowire_props_team_latest,
        time_col="action_props_as_of_ts",
    )
    rotowire_props_team_player_digest = {
        _normalize_props_team_abbr(team): str(digest)
        for team, digest in (
            dict(rotowire_props_summary.get("team_player_digest", {})).items()
        )
        if str(team).strip() and str(digest).strip()
    }
    rotowire_props_team_player_count = {
        _normalize_props_team_abbr(team): int(num)
        for team, count in (
            dict(rotowire_props_summary.get("team_player_count", {})).items()
        )
        if str(team).strip()
        and pd.notna(num := pd.to_numeric(count, errors="coerce"))
    }
    slate_teams_by_game: dict[int, list[str]] = {}
    for row in slate_df.itertuples(index=False):
        gid_num = pd.to_numeric(getattr(row, "game_id", None), errors="coerce")
        if pd.isna(gid_num):
            continue
        teams = []
        for attr in ("home_team_tricode", "away_team_tricode"):
            team = _normalize_props_team_abbr(getattr(row, attr, None))
            if team:
                teams.append(team)
        if teams:
            slate_teams_by_game[int(gid_num)] = sorted(set(teams))
    per_game_freshness: dict[str, dict[str, Any]] = {}
    for gid in slate_game_ids:
        tip_ts = schedule_tip_by_game.get(int(gid))
        minutes_to_tip = None
        is_live_game = False
        if tip_ts is not None:
            minutes_to_tip = float((tip_ts - run_ts).total_seconds() / 60.0)
            is_live_game = bool(minutes_to_tip > 0.0)
        injuries_bronze_ts = injuries_bronze_latest_by_game.get(int(gid))
        injuries_silver_ts = injuries_silver_latest_by_game.get(int(gid))
        if injuries_bronze_ts is not None:
            injuries_source_used = "bronze"
            injuries_latest = injuries_bronze_ts
            injuries_digest = injuries_bronze_digest_by_game.get(int(gid))
        elif injuries_silver_ts is not None:
            injuries_source_used = "silver"
            injuries_latest = injuries_silver_ts
            injuries_digest = injuries_silver_digest_by_game.get(int(gid))
        else:
            injuries_source_used = "none"
            injuries_latest = None
            injuries_digest = None
        rotowire_props_ts = rotowire_props_latest_by_game.get(int(gid))
        props_latest = rotowire_props_ts
        props_player_payload: list[dict[str, Any]] = []
        props_player_count = 0
        for team in slate_teams_by_game.get(int(gid), []):
            team_digest = rotowire_props_team_player_digest.get(team)
            if not team_digest:
                continue
            team_count = int(rotowire_props_team_player_count.get(team, 0))
            props_player_payload.append(
                {
                    "team_tricode": team,
                    "player_set_digest": team_digest,
                    "player_set_count": team_count,
                }
            )
            props_player_count += team_count
        props_player_set_digest = (
            _stable_digest(props_player_payload) if props_player_payload else None
        )
        per_game_freshness[str(int(gid))] = {
            "game_id": int(gid),
            "tip_ts": _ts_to_iso(tip_ts),
            "minutes_to_tip": minutes_to_tip,
            "is_live_game": bool(is_live_game),
            "sources": {
                "roster": {
                    "source_used": "silver",
                    "latest_as_of_ts": _ts_to_iso(roster_latest_by_game.get(int(gid))),
                    "age_minutes": _age_minutes(
                        run_ts, roster_latest_by_game.get(int(gid))
                    ),
                    "content_digest": roster_digest_by_game.get(int(gid)),
                },
                "odds": {
                    "source_used": "silver",
                    "latest_as_of_ts": _ts_to_iso(odds_latest_by_game.get(int(gid))),
                    "age_minutes": _age_minutes(
                        run_ts, odds_latest_by_game.get(int(gid))
                    ),
                    "content_digest": odds_digest_by_game.get(int(gid)),
                },
                "injuries": {
                    "source_used": injuries_source_used,
                    "latest_as_of_ts": _ts_to_iso(injuries_latest),
                    "age_minutes": _age_minutes(run_ts, injuries_latest),
                    "content_digest": injuries_digest,
                    "bronze_latest_as_of_ts": _ts_to_iso(injuries_bronze_ts),
                    "bronze_content_digest": injuries_bronze_digest_by_game.get(
                        int(gid)
                    ),
                    "silver_latest_as_of_ts": _ts_to_iso(injuries_silver_ts),
                    "silver_content_digest": injuries_silver_digest_by_game.get(
                        int(gid)
                    ),
                },
                "lineups": {
                    "source_used": "rotowire",
                    "latest_as_of_ts": _ts_to_iso(
                        rotowire_latest_by_game.get(int(gid))
                    ),
                    "age_minutes": _age_minutes(
                        run_ts, rotowire_latest_by_game.get(int(gid))
                    ),
                    "content_digest": rotowire_digest_by_game.get(int(gid)),
                },
                "props": {
                    "source_used": selected_props_source,
                    "latest_as_of_ts": _ts_to_iso(props_latest),
                    "age_minutes": _age_minutes(run_ts, props_latest),
                    "rotowire_latest_as_of_ts": _ts_to_iso(rotowire_props_ts),
                    "player_set_digest": props_player_set_digest,
                    "player_set_count": int(props_player_count),
                },
                "manual_overrides": dict(
                    manual_override_summary.get("per_game", {}).get(str(int(gid)), {})
                )
                or {
                    "source_used": "none",
                    "latest_as_of_ts": None,
                    "content_digest": None,
                    "active_override_count": 0,
                },
            },
        }
    report_window = _report_window_status(
        run_ts=run_ts, per_game_freshness=per_game_freshness
    )
    lock_window = _lock_window_gate_status(per_game_freshness=per_game_freshness)
    source_freshness = {
        "summary": {
            "run_as_of_ts": str(run_ts.isoformat()),
            "slate_game_count": int(len(slate_game_ids)),
            "live_game_count": int(
                sum(
                    1
                    for game in per_game_freshness.values()
                    if bool(game.get("is_live_game"))
                )
            ),
            "selected_props_source": selected_props_source,
            "manual_override_count": int(
                manual_override_summary.get("active_override_count", 0)
            ),
            "manual_override_games": list(
                manual_override_summary.get("affected_game_ids", [])
            ),
            "manual_override_digest": manual_override_summary.get("override_digest"),
        },
        "per_game": per_game_freshness,
    }

    failed_required = [
        c["name"] for c in checks if bool(c.get("required")) and not bool(c.get("ok"))
    ]
    return {
        "builder_input_checklist_version": 1,
        "game_date": game_date,
        "season": int(season),
        "month": int(month),
        "run_as_of_ts": str(run_ts.isoformat()),
        "checks": checks,
        "source_freshness": source_freshness,
        "freshness_gates": {
            "lock_window": lock_window,
            "report_window": report_window,
        },
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
    marker = (
        data_root
        / "bronze"
        / "v3_core_inputs"
        / f"date={game_date}"
        / "core_inputs_ready.json"
    )
    marker.parent.mkdir(parents=True, exist_ok=True)

    if placeholder_mode:
        payload = {
            "game_date": game_date,
            "placeholder_mode": True,
            "completed_at": _utc_now_iso(),
        }
        marker.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        return marker

    if replay_mode:
        payload = {
            "game_date": game_date,
            "placeholder_mode": False,
            "replay_mode": True,
            "completed_at": _utc_now_iso(),
            "props_required": bool(require_action_props),
            "live_props_source": "rotowire",
            "allow_rotowire_props_fallback": bool(allow_rotowire_props_fallback),
            "note": "scrape step skipped in replay_mode; existing snapshots are used",
        }
        marker.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
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

    props_status: dict[str, str] = {
        "scrape_props_cli": "not_run",
    }
    try:
        _run_python_module(
            "projections.cli.scrape_props",
            ["scrape", "--date", game_date],
            data_root=data_root,
            timeout_s=300,
        )
        props_status["scrape_props_cli"] = "ok"
    except Exception as exc:  # noqa: BLE001
        props_status["scrape_props_cli"] = f"failed: {exc}"
        if require_action_props:
            raise RuntimeError(
                "live props scrape failed while require_action_props=True: " f"{exc}"
            ) from exc
    props_dir = data_root / "bronze" / "props"
    day = pd.Timestamp(game_date).normalize()
    raw_props_files = sorted(
        (props_dir / f"game_date={day.date().isoformat()}").glob("*.parquet")
    )
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
        "props_required": bool(require_action_props),
        "live_props_source": "rotowire",
        "allow_rotowire_props_fallback": bool(allow_rotowire_props_fallback),
        "props_status": props_status,
        "rotowire_props_raw_file_count": int(len(raw_props_files)),
    }
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return marker


@task(name="score-ownership-linestar", retries=2, retry_delay_seconds=120)
def score_ownership_linestar_task(
    *,
    game_date: str,
    run_id: str,
    data_root: Path,
    placeholder_mode: bool,
) -> Path:
    out_dir = data_root / "silver" / "ownership_predictions" / game_date / f"run={run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if placeholder_mode:
        features_path = (
            data_root
            / "live"
            / FEATURES_ROOT
            / game_date
            / f"run={run_id}"
            / "features.parquet"
        )
        if features_path.exists():
            features_df = pd.read_parquet(features_path)
            placeholder_df = features_df.copy()
            if "player_id" not in placeholder_df.columns:
                placeholder_df["player_id"] = np.arange(len(placeholder_df)) + 1
            if "player_name" not in placeholder_df.columns:
                placeholder_df["player_name"] = placeholder_df["player_id"].map(
                    lambda value: f"Player {value}"
                )
            keep_cols = [
                column
                for column in ["player_id", "player_name", "team_id", "game_id"]
                if column in placeholder_df.columns
            ]
            placeholder_df = (
                placeholder_df.loc[:, keep_cols]
                .drop_duplicates(subset=["player_id"], keep="last")
                .reset_index(drop=True)
            )
        else:
            placeholder_df = pd.DataFrame(
                {
                    "player_id": list(range(1, 21)),
                    "player_name": [f"Player {idx}" for idx in range(1, 21)],
                }
            )
        placeholder_df["pred_own_pct"] = 0.05
        placeholder_df["source"] = "linestar"
        placeholder_df["model_run"] = "linestar_placeholder"
        _atomic_write_validated_parquet(
            placeholder_df,
            out_dir / "123.parquet",
            required_cols=("player_id",),
        )
        (out_dir / "slates.json").write_text(
            json.dumps(
                {
                    "123": {
                        "player_count": int(len(placeholder_df)),
                        "teams": [],
                        "first_game_time": None,
                        "is_locked": False,
                        "source": "linestar",
                    }
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return out_dir

    _run_python_module(
        "projections.cli.score_ownership_linestar",
        [
            "--date",
            game_date,
            "--run-id",
            run_id,
            "--data-root",
            str(data_root),
        ],
        data_root=data_root,
        timeout_s=1200,
    )
    return out_dir


@task(name="freeze-run-inputs", retries=0)
def freeze_run_inputs_task(
    *,
    game_date: str,
    run_id: str,
    as_of_ts: str,
    bundle_dir: Path,
    data_root: Path,
    source_freshness: dict[str, Any] | None = None,
    freshness_gates: dict[str, Any] | None = None,
    bounded_wait: dict[str, Any] | None = None,
    input_change_set: dict[str, Any] | None = None,
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
            "source_freshness": source_freshness or {},
            "freshness_gates": freshness_gates or {},
            "bounded_wait": bounded_wait or {},
            "input_change_set": input_change_set or {},
            "v3": {
                "bundle_dir": str(bundle_dir),
                "bundle_hash": bundle_hash,
                "parity_manifest_path": str(resolve_parity_manifest_path(bundle_dir)),
            },
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
    target_game_ids: list[int] | None = None,
) -> Path:
    run_dir = data_root / "live" / FEATURES_ROOT / game_date / f"run={run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "features.parquet"
    runtime_manifest_path = run_dir / "feature_runtime_manifest.json"
    input_checklist_path = run_dir / "feature_input_checklist.json"

    manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    v3_meta = dict(manifest_payload.get("v3", {}))
    if placeholder_mode:
        features_df = _placeholder_feature_frame(
            game_date=game_date, as_of_ts=run_as_of_ts
        )
        features_df = _filter_to_target_games(features_df, target_game_ids)
        _atomic_write_validated_parquet(
            features_df,
            out_path,
            required_cols=("game_id", "team_id", "player_id"),
        )

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
        props_source_report_path = run_dir / "props_source_report.json"
        props_source_report_path.write_text(
            json.dumps(
                {
                    "game_date": game_date,
                    "run_id": run_id,
                    "run_as_of_ts": run_as_of_ts,
                    "selected_source": selected_props_source,
                    "live_props_source": "rotowire",
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
            data_root
            / "live"
            / "features_minutes_v1"
            / game_date
            / f"run={run_id}"
            / "features.parquet"
        )
        if not base_minutes_path.exists():
            raise RuntimeError(f"base minutes features not found: {base_minutes_path}")

        spec = load_gtv2_feature_spec(bundle_dir)
        base_df = pd.read_parquet(base_minutes_path)
        base_df = _filter_to_target_games(base_df, target_game_ids)
        if base_df.empty:
            raise RuntimeError(
                "base minutes features are empty after applying target_game_ids: "
                f"{target_game_ids}"
            )

        built = build_gtv2_live_features(
            minutes_features=base_df,
            spec=spec,
            data_root=data_root,
            game_date=game_date,
            allow_priors_fallback=allow_priors_fallback,
            dnp_lookback_days=dnp_lookback_days,
        )
        transform_manifest = dict(built.transform_manifest)
        if stable_json_sha256(transform_manifest) != stable_json_sha256(
            expected_transform
        ):
            raise RuntimeError(
                "observed transform manifest does not match bundle parity manifest "
                "(fail-closed transform parity gate)"
            )

        features_df = _coerce_frame_to_manifest_schema(built.features, parity_payload)
        _atomic_write_validated_parquet(
            features_df,
            out_path,
            required_cols=("game_id", "team_id", "player_id"),
        )

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
        diagnostics["dnp_lookback_days"] = (
            None if dnp_lookback_days is None else int(dnp_lookback_days)
        )
        diagnostics["allow_rotowire_props_fallback"] = bool(allow_rotowire_fallback_cfg)
        diagnostics["target_game_ids"] = _normalize_game_ids(target_game_ids)

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
    manifest_path: Path,
    required_inputs: dict[str, Path],
    run_dirs: list[Path],
    features_path: Path,
    parity_manifest_path: Path,
    runtime_manifest_path: Path,
    input_max_age_minutes: float,
    bundle_config_path: Path | None = None,
) -> dict[str, Any]:
    runtime_payload = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
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
        frozen_source_freshness=dict(manifest_payload.get("source_freshness", {})),
        frozen_freshness_gates=dict(manifest_payload.get("freshness_gates", {})),
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
    run_dir = (
        data_root
        / "artifacts"
        / SCORES_ROOT
        / f"game_date={game_date}"
        / f"run={run_id}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "scores.parquet"
    summary_path = run_dir / "score_summary.json"

    if placeholder_mode:
        features = pd.read_parquet(features_path)
        scores = features[["game_date", "game_id", "team_id", "player_id"]].copy()
        scores["minutes_mean"] = pd.to_numeric(
            features["minutes_prior"], errors="coerce"
        ).fillna(0.0)
        scores["play_prob"] = 0.95
        scores["dk_rate"] = (
            pd.to_numeric(features["usage_prior"], errors="coerce").fillna(0.0) * 100.0
        )
        _atomic_write_validated_parquet(
            scores,
            out_path,
            required_cols=("game_date", "game_id", "team_id", "player_id"),
        )
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
                player_flat = np.concatenate(
                    [player_ids[b_idx, 0], player_ids[b_idx, 1]], axis=0
                )
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
    scores = scores.sort_values(
        ["game_date", "game_id", "team_id", "player_id"]
    ).reset_index(drop=True)
    _atomic_write_validated_parquet(
        scores,
        out_path,
        required_cols=("game_date", "game_id", "team_id", "player_id"),
    )
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
    run_as_of_ts: str | None = None,
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
    flow_scale_clip_override: float | None = None,
    make_model_mode: str = "beta_binomial_all",
    make_model_use_learned_efficiency: bool = True,
    apply_props_uplift: bool = True,
    apply_world_realism_controls: bool = True,
    world_realism_low_minutes_tail_damping_enabled: bool = True,
    world_realism_low_minutes_threshold: float = 12.0,
    world_realism_low_minutes_min_scale: float = 0.55,
    world_realism_outlier_resample_enabled: bool = True,
    world_realism_outlier_resample_max_passes: int = 1,
) -> dict[str, str]:
    run_dir = (
        data_root
        / "artifacts"
        / WORLDS_ROOT
        / f"game_date={game_date}"
        / f"run={run_id}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    worlds_path = run_dir / "worlds.parquet"
    projections_path = run_dir / "projections.parquet"
    worlds_summary_path = run_dir / "world_contracts_summary.json"

    if placeholder_mode:
        scores = pd.read_parquet(scores_path)
        projections = scores[["game_date", "game_id", "team_id", "player_id"]].copy()
        projections["minutes_sim_mean"] = scores["minutes_mean"].astype(float)
        projections["minutes_sim_p50"] = scores["minutes_mean"].astype(float)
        projections["dk_fpts_mean"] = (
            scores["minutes_mean"].astype(float)
            * scores["dk_rate"].astype(float)
            / 60.0
        ).round(4)
        projections["dk_fpts_p50"] = projections["dk_fpts_mean"]
        projections["sim_p_active"] = scores["play_prob"].astype(float)
        projections["n_worlds"] = int(sim_worlds)
        projections["sim_profile"] = "game_transformer_v2"
        projections = projections[PLACEHOLDER_PROJECTION_COLUMNS]
        _atomic_write_validated_parquet(
            projections,
            projections_path,
            required_cols=("game_date", "game_id", "team_id", "player_id"),
        )
        _atomic_write_validated_parquet(
            pd.DataFrame(columns=["world_idx"]),
            worlds_path,
            required_cols=("world_idx",),
        )
        contract_summary = {
            "contract_checks": {
                "team_minutes_not_240": 0,
                "team_minutes_total_checks": 0,
                "team_minutes_max_abs_drift": 0.0,
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
            "world_realism_controls": {
                "applied": False,
                "reason": "placeholder_mode",
            },
            "placeholder_mode": True,
        }
    else:
        logger = get_run_logger()
        _set_inference_seed(int(random_seed))
        device = _resolve_torch_device(gtv2_device)
        make_model_cfg = MakeModelConfig(
            mode=str(make_model_mode),
            use_learned_efficiency=bool(make_model_use_learned_efficiency),
        )

        # Warn loudly if using scale_clip override (experimental mode)
        if flow_scale_clip_override is not None:
            logger.warning("=" * 80)
            logger.warning(
                "EXPERIMENTAL: flow_scale_clip_override = %.2f",
                flow_scale_clip_override,
            )
            logger.warning("This is a non-default setting for H1 hypothesis testing.")
            logger.warning("Production runs should use the trained default (2.0).")
            logger.warning("=" * 80)

        config, model = _load_gtv2_model(
            bundle_dir,
            device=device,
            flow_scale_clip_override=flow_scale_clip_override,
        )
        features_df_raw = pd.read_parquet(features_path)
        features_df, force_active_diag = _attach_gtv2_force_active_worlds(
            features_df_raw,
            game_date=game_date,
            data_root=data_root,
            as_of_ts=run_as_of_ts,
        )
        logger.info("Applied force-active world guardrails: %s", force_active_diag)
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
                make_model_config=make_model_cfg,
            )
            world_frames.append(df_batch)
            contract_counter.update(checks)

        worlds_df = (
            pd.concat(world_frames, ignore_index=True)
            if world_frames
            else pd.DataFrame()
        )
        if worlds_df.empty:
            raise RuntimeError("GTV2 worlds generation produced zero rows")
        worlds_df, world_key_report = _sanitize_frame_to_expected_keys(
            worlds_df,
            expected_keys_df=features_df,
            key_cols=("game_id", "team_id", "player_id"),
            label="generated worlds",
        )
        if worlds_df.empty:
            raise RuntimeError("GTV2 worlds generation produced zero valid rows after key sanitization")
        if (
            world_key_report["dropped_null_key_rows"] > 0
            or world_key_report["dropped_unexpected_key_rows"] > 0
        ):
            logger.warning(
                "Dropped invalid world rows before publish: %s",
                world_key_report,
            )
        props_uplift_report: dict[str, Any]
        if bool(apply_props_uplift):
            worlds_df, props_uplift_report = _apply_props_uplift_calibration_to_worlds(
                worlds_df,
                features_df=features_df,
            )
            if bool(props_uplift_report.get("applied")):
                logger.info("Applied props uplift calibration: %s", props_uplift_report)
        else:
            props_uplift_report = {"applied": False, "reason": "disabled"}
        worlds_df, world_realism_report = _apply_world_realism_controls_to_worlds(
            worlds_df,
            enabled=bool(apply_world_realism_controls),
            random_seed=int(random_seed),
            low_minutes_tail_damping_enabled=bool(
                world_realism_low_minutes_tail_damping_enabled
            ),
            low_minutes_tail_minutes_threshold=float(
                world_realism_low_minutes_threshold
            ),
            low_minutes_tail_min_scale=float(world_realism_low_minutes_min_scale),
            outlier_resample_enabled=bool(world_realism_outlier_resample_enabled),
            outlier_resample_max_passes=int(
                world_realism_outlier_resample_max_passes
            ),
            target_game_ids=None,
        )
        if bool(world_realism_report.get("applied")):
            logger.info("Applied world realism controls: %s", world_realism_report)
        worlds_df, world_contract_repair_report = _repair_world_frame_contract_fields(
            worlds_df
        )
        if bool(world_contract_repair_report.get("applied")):
            logger.warning(
                "Applied world contract field repair before publish: %s",
                world_contract_repair_report,
            )
        _atomic_write_validated_parquet(
            worlds_df,
            worlds_path,
            required_cols=("world_idx", "game_id", "team_id", "player_id"),
        )

        projections = summarize_worlds_to_projections(
            worlds_df,
            sim_profile="game_transformer_v2",
        )
        projections, projection_key_report = _sanitize_frame_to_expected_keys(
            projections,
            expected_keys_df=features_df,
            key_cols=("game_id", "team_id", "player_id"),
            label="generated world projections",
        )
        _atomic_write_validated_parquet(
            projections,
            projections_path,
            required_cols=("game_date", "game_id", "team_id", "player_id"),
        )
        contract_checks = dict(contract_counter)
        contract_checks.update(_summarize_world_contracts_from_frame(worlds_df))
        contract_summary = {
            "contract_checks": contract_checks,
            "placeholder_mode": False,
            "world_rows": int(len(worlds_df)),
            "projection_rows": int(len(projections)),
            "bundle_dir": str(bundle_dir),
            "device": str(device),
            "key_sanitization": {
                "worlds": world_key_report,
                "projections": projection_key_report,
            },
            "make_model": {
                "mode": str(make_model_cfg.mode),
                "use_learned_efficiency": bool(make_model_cfg.use_learned_efficiency),
            },
            "force_active_guardrails": force_active_diag,
            "props_uplift_calibration": props_uplift_report,
            "world_realism_controls": world_realism_report,
            "world_contract_field_repair": world_contract_repair_report,
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


def _merge_live_ownership_into_projections(
    df: pd.DataFrame,
    *,
    game_date: str,
    run_id: str,
    data_root: Path,
) -> pd.DataFrame:
    own_dir = data_root / "silver" / "ownership_predictions" / game_date / f"run={run_id}"
    if not own_dir.exists():
        return df

    slate_files = [
        path
        for path in own_dir.glob("*.parquet")
        if not path.name.endswith("_locked.parquet")
    ]
    if not slate_files:
        return df

    own_path = max(slate_files, key=lambda path: path.stat().st_size)
    own_df = pd.read_parquet(own_path)
    if own_df.empty or "player_name" not in own_df.columns:
        return df

    merged = df.copy()
    own = own_df.copy()
    if "player_name" in merged.columns and "player_name" in own.columns:
        merged["_join_name"] = merged["player_name"].apply(normalize_player_name)
        own["_join_name"] = own["player_name"].apply(normalize_player_name)

        join_cols = ["_join_name"]
        if "team_tricode" in merged.columns and "team" in own.columns:
            merged["_join_team"] = merged["team_tricode"].astype(str).str.upper()
            own["_join_team"] = own["team"].astype(str).str.upper()
            join_cols.append("_join_team")
    elif "player_id" in merged.columns and "player_id" in own.columns:
        merged["player_id"] = pd.to_numeric(merged["player_id"], errors="coerce").astype(
            "Int64"
        )
        own["player_id"] = pd.to_numeric(own["player_id"], errors="coerce").astype(
            "Int64"
        )
        join_cols = ["player_id"]
    else:
        return df

    own_cols = join_cols + [
        col for col in ("salary", "pred_own_pct", "draft_group_id") if col in own.columns
    ]
    if len(own_cols) == len(join_cols):
        return df

    merged = merged.merge(
        own[own_cols].drop_duplicates(subset=join_cols, keep="last"),
        on=join_cols,
        how="left",
        suffixes=("", "__own"),
    )
    for col in ("salary", "pred_own_pct", "draft_group_id"):
        own_col = f"{col}__own"
        if own_col not in merged.columns:
            continue
        merged[col] = merged[col].where(pd.notna(merged[col]), merged[own_col])
        merged = merged.drop(columns=[own_col])

    return merged.drop(columns=["_join_name", "_join_team"], errors="ignore")


@task(name="finalize-projections-live", retries=0)
def finalize_projections_live_task(
    *,
    game_date: str,
    run_id: str,
    worlds_projections_path: Path,
    data_root: Path,
    placeholder_mode: bool,
    target_game_ids: list[int] | None = None,
) -> Path:
    out_dir = data_root / "artifacts" / "projections" / game_date / f"run={run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "projections.parquet"

    df = pd.read_parquet(worlds_projections_path)
    if df.empty:
        raise RuntimeError(f"world projections are empty: {worlds_projections_path}")
    df = _filter_to_target_games(df, target_game_ids)
    if df.empty:
        raise RuntimeError(
            "world projections are empty after applying target_game_ids: "
            f"{target_game_ids}"
        )

    # Enrich run-scoped projections with display + vegas context fields so the
    # dashboard can render a read-only game view without additional joins.
    display_src = (
        data_root
        / "live"
        / "features_minutes_v1"
        / game_date
        / f"run={run_id}"
        / "features.parquet"
    )
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
            "is_out",
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
                enrich[key] = pd.to_numeric(enrich[key], errors="coerce").astype(
                    "Int64"
                )
                df[key] = pd.to_numeric(df[key], errors="coerce").astype("Int64")
            enrich = enrich.dropna(subset=join_keys).drop_duplicates(
                subset=join_keys, keep="last"
            )
            df = df.merge(enrich, on=join_keys, how="left", suffixes=("", "__src"))
            for col in needed:
                src_col = f"{col}__src"
                if src_col not in df.columns:
                    continue
                if col in df.columns:
                    df[col] = df[col].where(pd.notna(df[col]), df[src_col])
                else:
                    df[col] = df[src_col]
            df = df.drop(
                columns=[c for c in df.columns if c.endswith("__src")], errors="ignore"
            )

    status_series = (
        df["status"].fillna("").astype(str).str.upper().str.strip()
        if "status" in df.columns
        else pd.Series("", index=df.index, dtype="string")
    )
    status_out_mask = (
        status_series.isin({"OUT", "O", "INACTIVE", "D", "DOUBTFUL", "SUSPENDED"})
        | status_series.str.contains("DOUBT", na=False)
    )
    is_out_series = (
        pd.to_numeric(df["is_out"], errors="coerce").fillna(0).astype(int).eq(1)
        if "is_out" in df.columns
        else pd.Series(False, index=df.index)
    )
    out_mask = status_out_mask | is_out_series
    if bool(out_mask.any()):
        df["is_out"] = out_mask.astype(int)
        if "status" in df.columns:
            df.loc[out_mask, "status"] = "OUT"
        else:
            df["status"] = np.where(out_mask, "OUT", "")

        zero_prefixes = (
            "minutes",
            "sim_minutes",
            "dk_fpts",
            "sim_dk_fpts",
            "fpts_sim",
            "pts_",
            "reb_",
            "ast_",
            "stl_",
            "blk_",
            "tov_",
            "sim_pts_",
            "sim_reb_",
            "sim_ast_",
            "sim_stl_",
            "sim_blk_",
            "sim_tov_",
            "p_play",
        )
        zero_exact = {
            "value",
            "play_prob",
            "pred_own_pct",
            "own_proj",
            "minutes_sim_p_active",
        }
        id_like_cols = {
            "game_id",
            "team_id",
            "player_id",
            "opponent_team_id",
            "n_worlds",
            "season",
        }
        zero_cols: list[str] = []
        for col in df.columns:
            if col in id_like_cols:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            if col in zero_exact or col.startswith(zero_prefixes):
                zero_cols.append(col)
        if zero_cols:
            df.loc[out_mask, zero_cols] = 0.0

    df = _merge_live_ownership_into_projections(
        df,
        game_date=game_date,
        run_id=run_id,
        data_root=data_root,
    )
    if "dk_fpts_mean" in df.columns and "salary" in df.columns:
        salary = pd.to_numeric(df["salary"], errors="coerce")
        df["value"] = (
            pd.to_numeric(df["dk_fpts_mean"], errors="coerce")
            .div(salary.where(salary > 0))
            .mul(1000)
            .round(2)
        )

    _atomic_write_validated_parquet(
        df,
        out_path,
        required_cols=("game_date", "game_id", "team_id", "player_id"),
    )
    return out_dir


@task(name="materialize-unified-run-artifacts", retries=0)
def materialize_unified_run_artifacts_task(
    *,
    game_date: str,
    run_id: str,
    data_root: Path,
    target_game_ids: list[int],
    apply_world_realism_controls: bool = True,
    world_realism_low_minutes_tail_damping_enabled: bool = True,
    world_realism_low_minutes_threshold: float = 12.0,
    world_realism_low_minutes_min_scale: float = 0.55,
    world_realism_outlier_resample_enabled: bool = True,
    world_realism_outlier_resample_max_passes: int = 1,
    random_seed: int = 42,
) -> dict[str, Any]:
    target_ids = _normalize_game_ids(target_game_ids)
    if not target_ids:
        return {"mode": "no_target_games"}

    features_dir = data_root / "live" / FEATURES_ROOT / game_date
    scores_dir = data_root / "artifacts" / SCORES_ROOT / f"game_date={game_date}"
    worlds_dir = data_root / "artifacts" / WORLDS_ROOT / f"game_date={game_date}"
    projections_dir = data_root / "artifacts" / "projections" / game_date

    merged_features = _merge_parquet_for_target_games(
        current_path=features_dir / f"run={run_id}" / "features.parquet",
        previous_path=_resolve_previous_run_file(
            dataset_dir=features_dir, filename="features.parquet"
        ),
        target_game_ids=target_ids,
    )
    merged_scores = _merge_parquet_for_target_games(
        current_path=scores_dir / f"run={run_id}" / "scores.parquet",
        previous_path=_resolve_previous_run_file(
            dataset_dir=scores_dir, filename="scores.parquet"
        ),
        target_game_ids=target_ids,
    )
    merged_worlds = _merge_parquet_for_target_games(
        current_path=worlds_dir / f"run={run_id}" / "worlds.parquet",
        previous_path=_resolve_previous_run_file(
            dataset_dir=worlds_dir, filename="worlds.parquet"
        ),
        target_game_ids=target_ids,
    )
    merged_final = _merge_parquet_for_target_games(
        current_path=projections_dir / f"run={run_id}" / "projections.parquet",
        previous_path=_resolve_previous_run_file(
            dataset_dir=projections_dir, filename="projections.parquet"
        ),
        target_game_ids=target_ids,
    )

    expected_feature_keys = merged_features.loc[:, ["game_id", "team_id", "player_id"]]

    merged_scores, score_key_report = _sanitize_frame_to_expected_keys(
        merged_scores,
        expected_keys_df=expected_feature_keys,
        key_cols=("game_id", "team_id", "player_id"),
        label="merged scores",
    )
    _atomic_write_validated_parquet(
        merged_scores,
        scores_dir / f"run={run_id}" / "scores.parquet",
        required_cols=("game_date", "game_id", "team_id", "player_id"),
    )

    merged_worlds, world_key_report = _sanitize_frame_to_expected_keys(
        merged_worlds,
        expected_keys_df=expected_feature_keys,
        key_cols=("game_id", "team_id", "player_id"),
        label="merged worlds",
    )
    merged_worlds, props_uplift_report = _apply_props_uplift_calibration_to_worlds(
        merged_worlds,
        features_df=merged_features,
    )
    merged_worlds, world_realism_report = _apply_world_realism_controls_to_worlds(
        merged_worlds,
        enabled=bool(apply_world_realism_controls),
        random_seed=int(random_seed),
        low_minutes_tail_damping_enabled=bool(
            world_realism_low_minutes_tail_damping_enabled
        ),
        low_minutes_tail_minutes_threshold=float(world_realism_low_minutes_threshold),
        low_minutes_tail_min_scale=float(world_realism_low_minutes_min_scale),
        outlier_resample_enabled=bool(world_realism_outlier_resample_enabled),
        outlier_resample_max_passes=int(world_realism_outlier_resample_max_passes),
        target_game_ids=set(target_ids),
    )
    merged_worlds, world_contract_repair_report = _repair_world_frame_contract_fields(
        merged_worlds
    )
    _atomic_write_validated_parquet(
        merged_worlds,
        worlds_dir / f"run={run_id}" / "worlds.parquet",
        required_cols=("world_idx", "game_id", "team_id", "player_id"),
    )

    merged_world_projections = summarize_worlds_to_projections(
        merged_worlds,
        sim_profile="game_transformer_v2",
    )
    merged_world_projections, world_projection_key_report = _sanitize_frame_to_expected_keys(
        merged_world_projections,
        expected_keys_df=expected_feature_keys,
        key_cols=("game_id", "team_id", "player_id"),
        label="merged world projections",
    )
    _atomic_write_validated_parquet(
        merged_world_projections,
        worlds_dir / f"run={run_id}" / "projections.parquet",
        required_cols=("game_date", "game_id", "team_id", "player_id"),
    )

    projection_join_keys = ["game_id", "team_id", "player_id"]
    projection_value_cols = [
        col
        for col in merged_world_projections.columns
        if col not in {"game_date", "game_id", "team_id", "player_id"}
    ]
    merged_final = _left_overlay_from_source_by_keys(
        merged_final,
        source_df=merged_world_projections.loc[
            :, projection_join_keys + projection_value_cols
        ],
        key_cols=projection_join_keys,
        value_cols=projection_value_cols,
        label="materialize_unified_run_artifacts_task/world_projection_overlay",
    )
    if "dk_fpts_mean" in merged_final.columns and "salary" in merged_final.columns:
        salary = pd.to_numeric(merged_final["salary"], errors="coerce")
        merged_final["value"] = (
            pd.to_numeric(merged_final["dk_fpts_mean"], errors="coerce")
            .div(salary.where(salary > 0))
            .mul(1000)
            .round(2)
        )

    merged_final, final_projection_key_report = _sanitize_frame_to_expected_keys(
        merged_final,
        expected_keys_df=expected_feature_keys,
        key_cols=("game_id", "team_id", "player_id"),
        label="merged unified projections",
    )
    _atomic_write_validated_parquet(
        merged_final,
        projections_dir / f"run={run_id}" / "projections.parquet",
        required_cols=("game_date", "game_id", "team_id", "player_id"),
    )

    world_summary_path = worlds_dir / f"run={run_id}" / "world_contracts_summary.json"
    world_summary_payload = {
        "contract_checks": _summarize_world_contracts_from_frame(merged_worlds),
        "merged_from_previous": True,
        "key_sanitization": {
            "scores": score_key_report,
            "worlds": world_key_report,
            "world_projections": world_projection_key_report,
            "unified_projections": final_projection_key_report,
        },
        "target_game_ids": target_ids,
        "rows": int(len(merged_worlds)),
        "projection_rows": int(len(merged_world_projections)),
        "props_uplift_calibration": props_uplift_report,
        "world_realism_controls": world_realism_report,
        "world_contract_field_repair": world_contract_repair_report,
        "created_at": _utc_now_iso(),
    }
    world_summary_path.write_text(
        json.dumps(world_summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "mode": "merged",
        "target_game_ids": target_ids,
        "features_rows": int(len(merged_features)),
        "scores_rows": int(len(merged_scores)),
        "world_rows": int(len(merged_worlds)),
        "projection_rows": int(len(merged_final)),
        "world_contract_summary_path": str(world_summary_path),
    }


def _validate_publishable_run_artifacts(
    *,
    game_date: str,
    run_id: str,
    data_root: Path,
) -> dict[str, Any]:
    stage_reports: dict[str, Any] = {}
    single_file_targets = {
        "features_minutes_v1": (
            data_root
            / "live"
            / "features_minutes_v1"
            / game_date
            / f"run={run_id}"
            / "features.parquet",
            ("game_id", "team_id", "player_id"),
        ),
        "features_gtv2_v1": (
            data_root / "live" / FEATURES_ROOT / game_date / f"run={run_id}" / "features.parquet",
            ("game_id", "team_id", "player_id"),
        ),
        "scores_gtv2": (
            data_root / "artifacts" / SCORES_ROOT / f"game_date={game_date}" / f"run={run_id}" / "scores.parquet",
            ("game_date", "game_id", "team_id", "player_id"),
        ),
        "worlds_gtv2/worlds": (
            data_root / "artifacts" / WORLDS_ROOT / f"game_date={game_date}" / f"run={run_id}" / "worlds.parquet",
            ("world_idx",),
        ),
        "worlds_gtv2/projections": (
            data_root / "artifacts" / WORLDS_ROOT / f"game_date={game_date}" / f"run={run_id}" / "projections.parquet",
            ("game_date", "game_id", "team_id", "player_id"),
        ),
        "unified_projections": (
            data_root / "artifacts" / "projections" / game_date / f"run={run_id}" / "projections.parquet",
            ("game_date", "game_id", "team_id", "player_id"),
        ),
    }
    for stage, (path, required_cols) in single_file_targets.items():
        if not path.exists():
            raise RuntimeError(f"publish validation missing required parquet: {path}")
        stage_reports[stage] = _stream_validate_parquet(
            path,
            required_cols=required_cols,
        )

    feature_keys = pd.read_parquet(
        single_file_targets["features_gtv2_v1"][0],
        columns=["game_id", "team_id", "player_id"],
    )
    stage_reports["semantic_key_contracts"] = {
        "scores_gtv2": _validate_parquet_key_contract(
            single_file_targets["scores_gtv2"][0],
            expected_keys_df=feature_keys,
            key_cols=("game_id", "team_id", "player_id"),
            label="scores_gtv2",
        ),
        "worlds_gtv2/worlds": _validate_parquet_key_contract(
            single_file_targets["worlds_gtv2/worlds"][0],
            expected_keys_df=feature_keys,
            key_cols=("game_id", "team_id", "player_id"),
            label="worlds_gtv2/worlds",
        ),
        "worlds_gtv2/projections": _validate_parquet_key_contract(
            single_file_targets["worlds_gtv2/projections"][0],
            expected_keys_df=feature_keys,
            key_cols=("game_id", "team_id", "player_id"),
            label="worlds_gtv2/projections",
        ),
        "unified_projections": _validate_parquet_key_contract(
            single_file_targets["unified_projections"][0],
            expected_keys_df=feature_keys,
            key_cols=("game_id", "team_id", "player_id"),
            label="unified_projections",
        ),
    }

    ownership_dir = (
        data_root / "silver" / "ownership_predictions" / game_date / f"run={run_id}"
    )
    ownership_files = sorted(
        path for path in ownership_dir.glob("*.parquet") if path.is_file()
    )
    if not ownership_files:
        raise RuntimeError(
            f"publish validation found no ownership parquet files under {ownership_dir}"
        )
    ownership_reports = []
    for path in ownership_files:
        ownership_reports.append(
            _stream_validate_parquet(path, required_cols=("player_id",))
        )
    stage_reports["ownership_predictions"] = {
        "dir": str(ownership_dir),
        "file_count": int(len(ownership_reports)),
        "files": ownership_reports,
    }
    return stage_reports


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
    manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    freshness_summary = dict(
        manifest_payload.get("source_freshness", {}).get("summary", {})
    )
    validation_report = _validate_publishable_run_artifacts(
        game_date=game_date,
        run_id=run_id,
        data_root=data_root,
    )
    manifest_payload["publish_validation"] = {
        "validated_at": _utc_now_iso(),
        "stages": validation_report,
    }
    control_plane.atomic_write_json(Path(manifest_path), manifest_payload)
    pointers: dict[str, str] = {}
    targets = {
        "features_minutes_v1": data_root / "live" / "features_minutes_v1" / game_date,
        "features_gtv2_v1": data_root / "live" / FEATURES_ROOT / game_date,
        "scores_gtv2": data_root / "artifacts" / SCORES_ROOT / f"game_date={game_date}",
        "worlds_gtv2": data_root / "artifacts" / WORLDS_ROOT / f"game_date={game_date}",
        "ownership_predictions": data_root / "silver" / "ownership_predictions" / game_date,
        "unified_projections": data_root / "artifacts" / "projections" / game_date,
    }
    for stage, dataset_dir in targets.items():
        pointer = control_plane.promote_run_pointer(
            dataset_dir=dataset_dir,
            run_id=run_id,
            manifest_path=manifest_path,
            extra={
                "entrypoint": "prefect-v3",
                "stage": stage,
                "as_of_ts": manifest_payload.get("as_of_ts"),
                "source_freshness_summary": freshness_summary,
            },
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
    gtv2_flow_scale_clip_override: float | None = None,
    gtv2_make_model_mode: str = "beta_binomial_all",
    gtv2_make_model_use_learned_efficiency: bool = True,
    gtv2_apply_world_realism_controls: bool = True,
    gtv2_world_realism_low_minutes_tail_damping_enabled: bool = True,
    gtv2_world_realism_low_minutes_threshold: float = 12.0,
    gtv2_world_realism_low_minutes_min_scale: float = 0.55,
    gtv2_world_realism_outlier_resample_enabled: bool = True,
    gtv2_world_realism_outlier_resample_max_passes: int = 1,
    input_max_age_minutes: float = 360.0,
    require_action_props: bool = True,
    allow_rotowire_props_fallback: bool = True,
) -> dict[str, str]:
    logger = get_run_logger()
    data_root = paths.get_data_root()
    resolved_game_date = _resolve_game_date(game_date)
    run_id = run_id_override or control_plane.canonical_run_id()
    minutes_selector_path = model_selectors.active_minutes_selector_path(
        data_root=data_root,
        project_root=PROJECT_ROOT,
    )
    rates_selector_path = model_selectors.active_rates_selector_path(
        data_root=data_root,
        project_root=PROJECT_ROOT,
    )
    bundle_dir = _resolve_bundle_dir(
        data_root=data_root, gtv2_bundle_dir=gtv2_bundle_dir
    )
    bundle_hash = _bundle_artifact_hash(bundle_dir)
    resolved_allow_rotowire_props_fallback = bool(allow_rotowire_props_fallback)
    rotation_cfg_path = PROJECT_ROOT / "config" / "rotation_set_minutes_live.json"
    if rotation_cfg_path.exists():
        try:
            rotation_cfg = json.loads(rotation_cfg_path.read_text(encoding="utf-8"))
            if "allow_rotowire_props_fallback" in rotation_cfg:
                resolved_allow_rotowire_props_fallback = bool(
                    rotation_cfg.get("allow_rotowire_props_fallback")
                )
        except Exception:
            pass

    # Resolve flow_scale_clip override: CLI param > env var > None
    resolved_flow_scale_clip_override = gtv2_flow_scale_clip_override
    if resolved_flow_scale_clip_override is None:
        env_clip = os.environ.get("GT_FLOW_SCALE_CLIP")
        if env_clip is not None:
            resolved_flow_scale_clip_override = float(env_clip)
            logger.warning(
                "GT_FLOW_SCALE_CLIP env var set to %.2f — using experimental scale_clip override",
                resolved_flow_scale_clip_override,
            )

    # Append suffix to run_id if using experimental scale_clip (avoids overwriting production)
    if resolved_flow_scale_clip_override is not None:
        clip_suffix = f"_clip{resolved_flow_scale_clip_override:.1f}".replace(".", "p")
        run_id = run_id + clip_suffix
        logger.info("Experimental run_id with clip suffix: %s", run_id)

    # Runtime stamp for reproducibility and incident triage.
    enforce_clean_tree()
    enforce_prod_sanity()
    log_runtime_stamp(
        entrypoint="prefect:nba-live-pipeline-v3",
        config_paths={
            "minutes_current_run": minutes_selector_path,
            "rates_current_run": rates_selector_path,
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

    try:
        writer_lock = writer_guard.PipelineWriterLock(data_root=data_root, run_id=run_id)
        writer_lock.__enter__()
    except RuntimeError as exc:
        if "Another writer is active" not in str(exc):
            raise
        duplicate_report = {
            "checked_at": _utc_now_iso(),
            "status": "skipped_due_to_active_writer",
            "reason": str(exc),
            "run_id": run_id,
            "game_date": resolved_game_date,
        }
        (v3_run_dir / "duplicate_run_report.json").write_text(
            json.dumps(duplicate_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "run_id": run_id,
            "game_date": resolved_game_date,
            "manifest_path": "",
            "features_path": "",
            "projections_path": "",
            "bundle_dir": str(bundle_dir),
            "pointer_count": "0",
            "rerun_mode": "",
            "rerun_reason": "",
            "publish_status": "skipped_active_writer",
        }

    try:
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
        frozen_checklist: dict[str, Any] = {}
        previous_manifest_payload = _load_promoted_manifest_payload(
            data_root=data_root,
            game_date=resolved_game_date,
        )
        bounded_wait_report: dict[str, Any] = {
            "performed": False,
            "reason": "placeholder_mode" if placeholder_mode else "not_needed",
        }
        if not placeholder_mode:
            frozen_checklist = _build_feature_input_checklist(
                game_date=resolved_game_date,
                run_as_of_ts=as_of_ts,
                data_root=data_root,
                allow_priors_fallback=True,
                allow_rotowire_props_fallback=bool(
                    resolved_allow_rotowire_props_fallback
                ),
                require_action_props=bool(require_action_props),
            )
            report_window = dict(
                frozen_checklist.get("freshness_gates", {}).get("report_window", {})
            )
            bounded_wait_report = {
                "performed": False,
                "reason": "not_needed",
                "report_window": report_window,
                "attempts": 0,
                "timed_out": False,
            }
            wait_allowed = (as_of_ts_override is None) and (not replay_mode)
            if (
                bool(report_window.get("active"))
                and bool(report_window.get("needs_wait"))
                and wait_allowed
            ):
                bounded_wait_report["performed"] = True
                bounded_wait_report["reason"] = "report_window_wait"
                deadline = time.monotonic() + float(
                    report_window.get(
                        "wait_timeout_seconds", _REPORT_WINDOW_WAIT_TIMEOUT_SECONDS
                    )
                )
                while (
                    bool(report_window.get("needs_wait"))
                    and time.monotonic() < deadline
                ):
                    sleep_s = min(
                        float(
                            report_window.get(
                                "wait_interval_seconds",
                                _REPORT_WINDOW_WAIT_INTERVAL_SECONDS,
                            )
                        ),
                        max(1.0, deadline - time.monotonic()),
                    )
                    logger.warning(
                        "Freshness wait active for %s; sleeping %.0fs before re-scrape. blocking_games=%s",
                        report_window.get("label"),
                        sleep_s,
                        [
                            game.get("game_id")
                            for game in report_window.get("blocking_games", [])
                        ],
                    )
                    time.sleep(sleep_s)
                    bounded_wait_report["attempts"] = (
                        int(bounded_wait_report.get("attempts", 0)) + 1
                    )
                    scrape_marker = scrape_core_inputs_task(
                        game_date=resolved_game_date,
                        data_root=data_root,
                        placeholder_mode=False,
                        require_action_props=bool(require_action_props),
                        allow_rotowire_props_fallback=bool(
                            resolved_allow_rotowire_props_fallback
                        ),
                        replay_mode=False,
                    )
                    as_of_ts = _utc_now_iso()
                    frozen_checklist = _build_feature_input_checklist(
                        game_date=resolved_game_date,
                        run_as_of_ts=as_of_ts,
                        data_root=data_root,
                        allow_priors_fallback=True,
                        allow_rotowire_props_fallback=bool(
                            resolved_allow_rotowire_props_fallback
                        ),
                        require_action_props=bool(require_action_props),
                    )
                    report_window = dict(
                        frozen_checklist.get("freshness_gates", {}).get(
                            "report_window", {}
                        )
                    )
                    bounded_wait_report["report_window"] = report_window
                if bool(report_window.get("needs_wait")):
                    bounded_wait_report["timed_out"] = True
                    bounded_wait_report["reason"] = "report_window_wait_timed_out"
                    logger.warning(
                        "Freshness wait timed out for %s; continuing with explicit diagnostics.",
                        report_window.get("label"),
                    )
            elif bool(report_window.get("needs_wait")) and not wait_allowed:
                bounded_wait_report["reason"] = "wait_skipped_override_or_replay"
                logger.warning(
                    "Skipping freshness wait because as_of_ts_override or replay_mode is active; report_window=%s",
                    report_window.get("label"),
                )

        input_change_set = _build_input_change_set(
            game_date=resolved_game_date,
            current_source_freshness=dict(frozen_checklist.get("source_freshness", {})),
            previous_manifest_payload=previous_manifest_payload,
        )
        rerun_plan = _build_rerun_plan(
            game_date=resolved_game_date,
            input_change_set=input_change_set,
            current_source_freshness=dict(frozen_checklist.get("source_freshness", {})),
            previous_manifest_payload=previous_manifest_payload,
            current_bundle_hash=bundle_hash,
            current_minutes_selector_path=minutes_selector_path,
            current_rates_selector_path=rates_selector_path,
        )
        target_game_ids = _normalize_game_ids(rerun_plan.get("target_game_ids"))

        manifest_path = freeze_run_inputs_task(
            game_date=resolved_game_date,
            run_id=run_id,
            as_of_ts=as_of_ts,
            bundle_dir=bundle_dir,
            data_root=data_root,
            source_freshness=dict(frozen_checklist.get("source_freshness", {})),
            freshness_gates=dict(frozen_checklist.get("freshness_gates", {})),
            bounded_wait=bounded_wait_report,
            input_change_set={**input_change_set, "rerun_plan": rerun_plan},
        )
        (v3_run_dir / "input_change_set.json").write_text(
            json.dumps(
                {**input_change_set, "rerun_plan": rerun_plan}, indent=2, sort_keys=True
            ),
            encoding="utf-8",
        )
        control_plane.atomic_update_json(manifest_path, {"rerun_plan": rerun_plan})

        if rerun_plan.get("mode") == "skip":
            skip_report = {
                "mode": "skip",
                "reason": rerun_plan.get("reason"),
                "target_game_ids": target_game_ids,
                "previous_run_id": input_change_set.get("previous_run_id"),
            }
            (v3_run_dir / "skip_report.json").write_text(
                json.dumps(skip_report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return {
                "run_id": run_id,
                "game_date": resolved_game_date,
                "manifest_path": str(manifest_path),
                "features_path": "",
                "projections_path": "",
                "bundle_dir": str(bundle_dir),
                "pointer_count": "0",
                "rerun_mode": str(rerun_plan.get("mode")),
                "rerun_reason": str(rerun_plan.get("reason")),
                "publish_status": "not_requested",
            }

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
            target_game_ids=(
                None if rerun_plan.get("mode") == "full_slate" else target_game_ids
            ),
        )
        parity_manifest_path = resolve_parity_manifest_path(bundle_dir)
        runtime_manifest_path = features_path.parent / "feature_runtime_manifest.json"

        score_run_dir = (
            data_root
            / "artifacts"
            / SCORES_ROOT
            / f"game_date={resolved_game_date}"
            / f"run={run_id}"
        )
        worlds_run_dir = (
            data_root
            / "artifacts"
            / WORLDS_ROOT
            / f"game_date={resolved_game_date}"
            / f"run={run_id}"
        )
        projections_run_dir = (
            data_root
            / "artifacts"
            / "projections"
            / resolved_game_date
            / f"run={run_id}"
        )

        preflight_report = preflight_gate_task(
            as_of_ts=as_of_ts,
            manifest_path=manifest_path,
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
            run_as_of_ts=as_of_ts,
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
            flow_scale_clip_override=resolved_flow_scale_clip_override,
            make_model_mode=str(gtv2_make_model_mode),
            make_model_use_learned_efficiency=bool(
                gtv2_make_model_use_learned_efficiency
            ),
            apply_props_uplift=bool(rerun_plan.get("mode") == "full_slate"),
            apply_world_realism_controls=bool(gtv2_apply_world_realism_controls),
            world_realism_low_minutes_tail_damping_enabled=bool(
                gtv2_world_realism_low_minutes_tail_damping_enabled
            ),
            world_realism_low_minutes_threshold=float(
                gtv2_world_realism_low_minutes_threshold
            ),
            world_realism_low_minutes_min_scale=float(
                gtv2_world_realism_low_minutes_min_scale
            ),
            world_realism_outlier_resample_enabled=bool(
                gtv2_world_realism_outlier_resample_enabled
            ),
            world_realism_outlier_resample_max_passes=int(
                gtv2_world_realism_outlier_resample_max_passes
            ),
        )

        ownership_dir = score_ownership_linestar_task(
            game_date=resolved_game_date,
            run_id=run_id,
            data_root=data_root,
            placeholder_mode=bool(placeholder_mode),
        )
        if ownership_dir.exists():
            control_plane.copy_manifest_to_dir(manifest_path, ownership_dir)

        projections_dir = finalize_projections_live_task(
            game_date=resolved_game_date,
            run_id=run_id,
            worlds_projections_path=Path(worlds_outputs["projections_path"]),
            data_root=data_root,
            placeholder_mode=bool(placeholder_mode),
            target_game_ids=(
                None if rerun_plan.get("mode") == "full_slate" else target_game_ids
            ),
        )
        if rerun_plan.get("mode") == "game_scoped":
            unified_report = materialize_unified_run_artifacts_task(
                game_date=resolved_game_date,
                run_id=run_id,
                data_root=data_root,
                target_game_ids=target_game_ids,
                apply_world_realism_controls=bool(gtv2_apply_world_realism_controls),
                world_realism_low_minutes_tail_damping_enabled=bool(
                    gtv2_world_realism_low_minutes_tail_damping_enabled
                ),
                world_realism_low_minutes_threshold=float(
                    gtv2_world_realism_low_minutes_threshold
                ),
                world_realism_low_minutes_min_scale=float(
                    gtv2_world_realism_low_minutes_min_scale
                ),
                world_realism_outlier_resample_enabled=bool(
                    gtv2_world_realism_outlier_resample_enabled
                ),
                world_realism_outlier_resample_max_passes=int(
                    gtv2_world_realism_outlier_resample_max_passes
                ),
                random_seed=int(gtv2_seed),
            )
            (v3_run_dir / "unified_artifacts_report.json").write_text(
                json.dumps(unified_report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            worlds_outputs["world_contract_summary_path"] = str(
                unified_report["world_contract_summary_path"]
            )

        # Validate output schema contract against parity manifest before publish.
        _ = load_parity_manifest(parity_manifest_path)
        postflight_report = postflight_gate_task(
            projections_path=projections_dir / "projections.parquet",
            parity_manifest_path=parity_manifest_path,
            world_contract_summary_path=Path(
                worlds_outputs["world_contract_summary_path"]
            ),
        )
        (v3_run_dir / "postflight_report.json").write_text(
            json.dumps(postflight_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        pointer_payload: dict[str, str] = {}
        publish_status = "not_requested" if not promote_pointers else "pending"
        if promote_pointers:
            if placeholder_mode:
                stale_publish_report: dict[str, Any] = {
                    "checked_at": _utc_now_iso(),
                    "stale": False,
                    "stale_games": [],
                    "skipped": "placeholder_mode",
                }
            else:
                publish_checklist = _build_feature_input_checklist(
                    game_date=resolved_game_date,
                    run_as_of_ts=_utc_now_iso(),
                    data_root=data_root,
                    allow_priors_fallback=True,
                    allow_rotowire_props_fallback=bool(
                        resolved_allow_rotowire_props_fallback
                    ),
                    require_action_props=bool(require_action_props),
                )
                stale_publish_report = _detect_stale_authoritative_inputs(
                    frozen_source_freshness=dict(
                        frozen_checklist.get("source_freshness", {})
                    ),
                    current_source_freshness=dict(
                        publish_checklist.get("source_freshness", {})
                    ),
                    as_of_ts=str(
                        publish_checklist.get("run_as_of_ts") or _utc_now_iso()
                    ),
                )
                control_plane.atomic_update_json(
                    manifest_path,
                    {
                        "publish_precheck": stale_publish_report,
                    },
                )
                (v3_run_dir / "stale_publish_report.json").write_text(
                    json.dumps(stale_publish_report, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                if bool(stale_publish_report.get("stale")):
                    raise RuntimeError(
                        "stale publish blocked: newer authoritative injuries/lineups arrived after freeze. "
                        f"See {v3_run_dir / 'stale_publish_report.json'}"
                    )
            superseded_report = _build_publish_superseded_report(
                run_id=run_id,
                manifest_path=manifest_path,
                dataset_dir=data_root / "artifacts" / "projections" / resolved_game_date,
            )
            control_plane.atomic_update_json(
                manifest_path,
                {
                    "publish_superseded": superseded_report,
                },
            )
            (v3_run_dir / "publish_superseded_report.json").write_text(
                json.dumps(superseded_report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            if bool(superseded_report.get("superseded")):
                logger.warning(
                    "Skipping publish for %s because a newer run is already published. reason=%s current_run_id=%s",
                    run_id,
                    superseded_report.get("reason"),
                    dict(superseded_report.get("current_pointer") or {}).get("run_id"),
                )
                publish_status = "superseded"
            else:
                pointer_payload = publish_atomic_task(
                    game_date=resolved_game_date,
                    run_id=run_id,
                    manifest_path=manifest_path,
                    data_root=data_root,
                )
                publish_status = "published"
            if placeholder_mode:
                (v3_run_dir / "stale_publish_report.json").write_text(
                    json.dumps(stale_publish_report, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
        return {
            "run_id": run_id,
            "game_date": resolved_game_date,
            "manifest_path": str(manifest_path),
            "features_path": str(features_path),
            "projections_path": str(projections_dir / "projections.parquet"),
            "bundle_dir": str(bundle_dir),
            "pointer_count": str(len(pointer_payload)),
            "rerun_mode": str(rerun_plan.get("mode")),
            "rerun_reason": str(rerun_plan.get("reason")),
            "publish_status": publish_status,
        }
    finally:
        writer_lock.__exit__(None, None, None)
