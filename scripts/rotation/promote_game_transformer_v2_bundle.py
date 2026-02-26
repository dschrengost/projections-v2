#!/usr/bin/env python3
"""Freeze a Phase-3 candidate and package a promoted GTV2 inference bundle."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from projections.pipeline.control_plane import resolve_git_sha
from projections.pipeline.parity_manifest import (
    build_parity_manifest,
    hash_paths,
    resolve_parity_manifest_path,
    sha256_file,
    write_parity_manifest,
)
from projections.rotation.game_transformer_v2 import GameTransformerV2Config
from projections.rotation.sample_worlds_v2 import summarize_worlds_to_projections


DEFAULT_BUNDLE_ROOT = Path("/home/daniel/projections-data/artifacts/game_transformer_v2")
KEY_COLS = ["game_id", "team_id", "player_id"]
ROTATION_PRIORS_CONTRACT_GAME_ID_ONLY = "game_id_partitions_only"
ROTATION_PRIORS_CONTRACT_GAME_ID_PLUS_PRE_GAME_ENTITY_FALLBACK = (
    "game_id_partitions_plus_pre_game_entity_fallback"
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_transform_manifest(
    *,
    allow_priors_fallback: bool,
    training_rotation_priors_contract: str | None = None,
) -> dict[str, Any]:
    priors_mode = (
        str(training_rotation_priors_contract)
        if training_rotation_priors_contract
        else ROTATION_PRIORS_CONTRACT_GAME_ID_PLUS_PRE_GAME_ENTITY_FALLBACK
    )
    return {
        "builder": "gtv2_live_features_v1",
        "lineup_contract": "joint_rotation_rates_v1_section_4_7",
        "game_context_contract": {
            "league_ppp": 2.27,
            "estimated_possessions_pace_weight": 0.0,
            "estimated_possessions_clip_min": 85.0,
            "estimated_possessions_clip_max": 130.0,
        },
        "priors": {
            "allow_priors_fallback": bool(allow_priors_fallback),
            "mode": priors_mode,
            "explanation": (
                "Live slates often have no same-day game_id priors partitions pre-tip. "
                "Fallback uses latest pre-game priors by team/player; training contract is recorded "
                "via priors.mode for parity auditing."
            ),
        },
        "dnp_history": {
            "mode": "full_prior_history",
            "lookback_days": None,
        },
    }


def _default_projection_columns() -> list[str]:
    worlds = pd.DataFrame(
        {
            "world_idx": [0, 1],
            "game_date": ["2026-01-01", "2026-01-01"],
            "game_id": [1, 1],
            "team_id": [10, 10],
            "player_id": [100, 100],
            "active": [1, 1],
            "minutes": [30.0, 32.0],
            "dk_fpts": [42.0, 45.0],
            "pts": [22.0, 24.0],
            "reb": [6.0, 7.0],
            "ast": [5.0, 6.0],
            "stl": [1.0, 1.0],
            "blk": [0.0, 1.0],
            "tov": [3.0, 2.0],
        }
    )
    out = summarize_worlds_to_projections(worlds, sim_profile="game_transformer_v2")
    return list(out.columns)


def _rotation_priors_contract_from_dataset_manifest(dataset_dir: Path) -> str | None:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = _read_json(manifest_path)
    except Exception:
        return None

    # Joint dataset manifest points to rotation dataset dir.
    rotation_dataset_dir = (
        manifest.get("args", {}).get("rotation_dataset_dir")
        if isinstance(manifest.get("args"), dict)
        else None
    )
    if not rotation_dataset_dir:
        return None
    rotation_manifest_path = Path(str(rotation_dataset_dir)).expanduser().resolve() / "manifest.json"
    if not rotation_manifest_path.exists():
        return None
    try:
        rotation_manifest = _read_json(rotation_manifest_path)
    except Exception:
        return None

    opts = rotation_manifest.get("options", {})
    if isinstance(opts, dict):
        mode = opts.get("rotation_priors_contract")
        if isinstance(mode, str) and mode.strip():
            return str(mode)
    priors = rotation_manifest.get("rotation_priors_v1", {})
    if isinstance(priors, dict):
        mode = priors.get("contract_mode")
        if isinstance(mode, str) and mode.strip():
            return str(mode)
    return None


def _build_distribution_contract(
    *,
    features_df: pd.DataFrame,
    config: GameTransformerV2Config,
) -> dict[str, Any]:
    # Focus on high-impact parity-sensitive priors/vacancy features.
    feature_limits: dict[str, dict[str, float]] = {
        "vacated_minutes_prior_20_total": {"max_abs_mean_z": 10.0, "max_p95_abs_z": 30.0},
        "vacated_minutes_prior_20_same_pos": {"max_abs_mean_z": 10.0, "max_p95_abs_z": 30.0},
        "team_prior_minutes_20_not_out": {"max_abs_mean_z": 6.0, "max_p95_abs_z": 12.0},
    }
    conditional_limits: list[dict[str, Any]] = [
        {
            "name": "out_rows_prior_missing_rate",
            "condition_col": "is_out",
            "condition_eq": 1,
            "metric_col": "minutes_from_stints_prior_20_missing",
            "max_rate": 0.70,
        }
    ]

    feature_index = {c: i for i, c in enumerate(config.feature_columns)}
    mean = np.asarray(config.feature_mean, dtype=np.float64)
    std = np.asarray(config.feature_std, dtype=np.float64)
    std = np.where(std <= 1e-6, 1.0, std)

    training_reference: dict[str, Any] = {}
    for feature_name in list(feature_limits.keys()):
        if feature_name not in features_df.columns or feature_name not in feature_index:
            continue
        idx = int(feature_index[feature_name])
        vals = pd.to_numeric(features_df[feature_name], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        vals = np.nan_to_num(vals, nan=float(mean[idx]), posinf=float(mean[idx]), neginf=float(mean[idx]))
        z = np.abs((vals - float(mean[idx])) / float(std[idx]))
        training_reference[feature_name] = {
            "mean_abs_z": float(np.mean(z)) if z.size > 0 else 0.0,
            "p95_abs_z": float(np.percentile(z, 95)) if z.size > 0 else 0.0,
            "max_abs_z": float(np.max(z)) if z.size > 0 else 0.0,
        }

    if {"is_out", "minutes_from_stints_prior_20_missing"}.issubset(features_df.columns):
        is_out = pd.to_numeric(features_df["is_out"], errors="coerce").fillna(0)
        miss = pd.to_numeric(features_df["minutes_from_stints_prior_20_missing"], errors="coerce").fillna(1)
        mask = is_out.eq(1)
        training_reference["out_rows_prior_missing_rate"] = {
            "n": int(mask.sum()),
            "rate": float(miss.loc[mask].mean()) if bool(mask.any()) else 0.0,
        }

    return {
        "enabled": True,
        "feature_limits": feature_limits,
        "conditional_limits": conditional_limits,
        "training_reference": training_reference,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_metrics(seed_dir: Path) -> dict[str, Any]:
    offline_path = seed_dir / "offline_eval_vs_sim_v2_60d_64w_strict.json"
    payload = _read_json(offline_path) if offline_path.exists() else {}
    checks = payload.get("go_no_go_checks", {}) if isinstance(payload, dict) else {}
    go_no_go_pass = bool(checks) and all(bool(v) for v in checks.values())
    gt_metrics = payload.get("game_transformer_v2", {}) if isinstance(payload, dict) else {}
    return {
        "seed_dir": str(seed_dir),
        "seed": str(seed_dir.name.replace("seed_", "")),
        "go_no_go_checks": checks,
        "go_no_go_pass": go_no_go_pass,
        "offline_eval_path": str(offline_path),
        "crps_mean": gt_metrics.get("crps_mean"),
        "p90_err": gt_metrics.get("p90_calibration_error_abs"),
        "p95_err": gt_metrics.get("p95_calibration_error_abs"),
        "team_total_mae": gt_metrics.get("team_total_mae"),
    }


def _pick_seed(candidate_root: Path, seed_override: str | None) -> dict[str, Any]:
    if seed_override:
        seed_dir = candidate_root / f"seed_{seed_override}"
        if not seed_dir.exists():
            raise FileNotFoundError(f"seed dir not found: {seed_dir}")
        return _seed_metrics(seed_dir)

    seed_dirs = sorted([p for p in candidate_root.glob("seed_*") if p.is_dir()], key=lambda p: p.name)
    if not seed_dirs:
        raise FileNotFoundError(f"no seed_* dirs found under {candidate_root}")
    rows = [_seed_metrics(p) for p in seed_dirs]
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no seed metrics discovered")
    if "go_no_go_pass" in df.columns and bool(df["go_no_go_pass"].any()):
        df = df.loc[df["go_no_go_pass"]].copy()
    df["crps_mean"] = pd.to_numeric(df["crps_mean"], errors="coerce")
    df["team_total_mae"] = pd.to_numeric(df["team_total_mae"], errors="coerce")
    df["p95_err"] = pd.to_numeric(df["p95_err"], errors="coerce")
    df = df.sort_values(["crps_mean", "team_total_mae", "p95_err"], ascending=[True, True, True], kind="mergesort")
    return dict(df.iloc[0].to_dict())


def _prepare_bundle_dir(bundle_dir: Path, *, overwrite: bool) -> None:
    if bundle_dir.exists():
        if not overwrite:
            raise FileExistsError(f"bundle already exists: {bundle_dir} (use --overwrite)")
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)


def _update_bundle_current(bundle_root: Path, bundle_dir: Path) -> Path:
    bundle_current = bundle_root / "bundle_current"
    if bundle_current.exists() or bundle_current.is_symlink():
        if bundle_current.is_symlink() or bundle_current.is_file():
            bundle_current.unlink()
        else:
            raise RuntimeError(
                f"bundle_current exists as a directory: {bundle_current}. "
                "Remove/move it first, then rerun promotion."
            )
    bundle_current.symlink_to(bundle_dir)
    return bundle_current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True, help="Phase-3 multiseed root directory")
    parser.add_argument("--seed", default=None, help="Optional explicit seed id (e.g. 123)")
    parser.add_argument("--bundle-root", default=str(DEFAULT_BUNDLE_ROOT))
    parser.add_argument("--bundle-name", default=None)
    parser.add_argument("--waiver-rationale", default=None)
    parser.add_argument("--dataset-dir", default=None, help="Optional override for dataset directory")
    parser.add_argument(
        "--disallow-priors-fallback",
        action="store_true",
        help="Force transform manifest priors.allow_priors_fallback=false (default true).",
    )
    parser.add_argument(
        "--drift-report-json",
        default=None,
        help="Optional props drift report JSON path to link in promotion metadata.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing bundle dir if present")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_root = Path(args.candidate_root).expanduser().resolve()
    if not candidate_root.exists():
        raise FileNotFoundError(f"candidate_root not found: {candidate_root}")

    selected = _pick_seed(candidate_root, args.seed)
    seed_dir = Path(str(selected["seed_dir"])).resolve()
    summary_path = seed_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json in selected seed dir: {seed_dir}")
    summary = _read_json(summary_path)
    dataset_dir = (
        Path(args.dataset_dir).expanduser().resolve()
        if args.dataset_dir
        else Path(str(summary.get("dataset_dir", ""))).expanduser().resolve()
    )
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset dir not found: {dataset_dir}")

    model_path_src = seed_dir / "model.pt"
    config_path_src = seed_dir / "config.json"
    if not model_path_src.exists() or not config_path_src.exists():
        raise FileNotFoundError(f"seed dir missing model/config: {seed_dir}")

    bundle_root = Path(args.bundle_root).expanduser().resolve()
    bundle_name = args.bundle_name or f"phase3_{candidate_root.name}_{seed_dir.name}_{_utc_now_compact()}"
    bundle_dir = bundle_root / "bundles" / bundle_name
    _prepare_bundle_dir(bundle_dir, overwrite=bool(args.overwrite))

    model_path = bundle_dir / "model.pt"
    config_path = bundle_dir / "config.json"
    shutil.copy2(model_path_src, model_path)
    shutil.copy2(config_path_src, config_path)

    config = GameTransformerV2Config.load(config_path)
    feature_cols = [*KEY_COLS, *list(config.feature_columns)]
    features_path = dataset_dir / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"features parquet not found: {features_path}")
    features_df = pd.read_parquet(features_path, columns=feature_cols)

    projection_columns = _default_projection_columns()
    integrity = {
        "git_sha": resolve_git_sha(),
        "config_hash": sha256_file(config_path),
        "artifact_hash": hash_paths([config_path, model_path]),
        "source_run_dir": str(seed_dir),
    }
    training_priors_contract = _rotation_priors_contract_from_dataset_manifest(dataset_dir)
    distribution_contract = _build_distribution_contract(features_df=features_df, config=config)
    parity_manifest = build_parity_manifest(
        model_id=f"game_transformer_v2:{bundle_name}",
        features_df=features_df,
        feature_columns=feature_cols,
        missing_value_policy={
            "disallow_null_columns": list(KEY_COLS),
            "distribution_contract": distribution_contract,
        },
        transform_manifest=_default_transform_manifest(
            allow_priors_fallback=not bool(args.disallow_priors_fallback),
            training_rotation_priors_contract=training_priors_contract,
        ),
        output_manifest={
            "projection_columns": projection_columns,
            "evaluation_default": {
                "fpts_mean": "dk_fpts_mean_uncond",
                "minutes_mean": "minutes_sim_mean_uncond",
                "semantics": "unconditional_preferred",
            },
            "semantics": {
                "dk_fpts_mean_uncond": "unconditional_dnp_zero",
                "dk_fpts_mean": "conditional_on_active",
                "minutes_sim_mean_uncond": "unconditional_dnp_zero",
                "minutes_sim_mean": "conditional_on_active",
            },
        },
        integrity=integrity,
    )
    parity_path = resolve_parity_manifest_path(bundle_dir)
    write_parity_manifest(parity_path, parity_manifest)

    waiver_rationale = args.waiver_rationale or (
        "all go/no-go checks pass" if bool(selected.get("go_no_go_pass", False)) else "selected best available seed with waiver"
    )
    promotion_record = {
        "created_at": _utc_now_iso(),
        "candidate_root": str(candidate_root),
        "selected_seed_dir": str(seed_dir),
        "seed": str(selected.get("seed")),
        "metrics": {
            "crps_mean": selected.get("crps_mean"),
            "p90_err": selected.get("p90_err"),
            "p95_err": selected.get("p95_err"),
            "team_total_mae": selected.get("team_total_mae"),
        },
        "go_no_go_checks": selected.get("go_no_go_checks", {}),
        "go_no_go_pass": bool(selected.get("go_no_go_pass", False)),
        "waiver_rationale": waiver_rationale,
        "git_sha": resolve_git_sha(),
        "dataset_dir": str(dataset_dir),
        "training_rotation_priors_contract": training_priors_contract,
        "bundle_dir": str(bundle_dir),
        "parity_manifest_path": str(parity_path),
        "parity_manifest_hash": parity_manifest.get("integrity", {}).get("parity_manifest_hash"),
    }
    promoted_path = candidate_root / "promoted_phase3.json"
    promoted_path.write_text(json.dumps(promotion_record, indent=2, sort_keys=True), encoding="utf-8")

    meta = {
        **promotion_record,
        "model_source_path": str(model_path_src),
        "config_source_path": str(config_path_src),
        "drift_report_json": str(Path(args.drift_report_json).expanduser().resolve())
        if args.drift_report_json
        else None,
    }
    (bundle_dir / "promotion_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    bundle_root.mkdir(parents=True, exist_ok=True)
    bundle_current = _update_bundle_current(bundle_root, bundle_dir)

    print(
        json.dumps(
            {
                "promoted_phase3_json": str(promoted_path),
                "bundle_dir": str(bundle_dir),
                "bundle_current": str(bundle_current),
                "parity_manifest_path": str(parity_path),
                "selected_seed": str(selected.get("seed")),
                "go_no_go_pass": bool(selected.get("go_no_go_pass", False)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
