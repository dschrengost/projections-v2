"""Basic end-to-end smoke test for Triton-backed GTV2 scoring/world generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from projections import paths
from projections.pipeline.triton_inference_client import (
    TritonEndpointConfig,
    check_triton_health,
    infer_json_action,
)


def _resolve_features_path(*, data_root: Path, game_date: str) -> Path:
    base = data_root / "live" / "features_gtv2_v1" / str(game_date)
    pointer_path = base / "LATEST" / "current.json"
    if pointer_path.exists():
        try:
            payload = json.loads(pointer_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            for key in ("path", "run_dir", "dataset_dir", "target_dir"):
                raw = payload.get(key)
                if not raw:
                    continue
                candidate = Path(str(raw))
                if candidate.is_dir():
                    candidate = candidate / "features.parquet"
                if candidate.exists():
                    return candidate
            run_id = payload.get("run_id")
            if run_id:
                candidate = base / f"run={run_id}" / "features.parquet"
                if candidate.exists():
                    return candidate
    runs = sorted(base.glob("run=*/features.parquet"))
    if not runs:
        raise FileNotFoundError(f"no features parquet found under {base}")
    return runs[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton-endpoint", type=str, default="localhost:18000")
    parser.add_argument("--model-name", type=str, default="gtv2_scorer")
    parser.add_argument("--model-version", type=str, default="1")
    parser.add_argument("--game-date", type=str, required=True)
    parser.add_argument("--features-path", type=Path, default=None)
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--num-worlds", type=int, default=256)
    parser.add_argument("--world-chunk-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = paths.get_data_root()

    features_path = (
        args.features_path.expanduser().resolve()
        if args.features_path is not None
        else _resolve_features_path(data_root=data_root, game_date=str(args.game_date))
    )
    if not features_path.exists():
        raise FileNotFoundError(f"features path not found: {features_path}")

    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else (data_root / "tmp" / "gtv2_triton_smoke").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    score_out = out_dir / "scores.parquet"
    score_summary = out_dir / "score_summary.json"
    worlds_out = out_dir / "worlds.parquet"
    worlds_summary = out_dir / "worlds_summary.json"

    ready, detail = check_triton_health(
        args.triton_endpoint,
        timeout_seconds=min(5.0, float(args.timeout_seconds)),
    )
    if not ready:
        raise RuntimeError(f"triton readiness failed: endpoint={args.triton_endpoint} detail={detail}")

    endpoint_cfg = TritonEndpointConfig(
        endpoint=str(args.triton_endpoint),
        model_name=str(args.model_name),
        model_version=str(args.model_version) if args.model_version else None,
        timeout_seconds=float(args.timeout_seconds),
    )

    common: dict[str, Any] = {
        "game_date": str(args.game_date),
        "features_path": str(features_path),
        "bundle_dir": str(args.bundle_dir.expanduser().resolve()) if args.bundle_dir else None,
        "device": str(args.device),
        "random_seed": 42,
    }

    score_resp = infer_json_action(
        cfg=endpoint_cfg,
        request_payload={
            **common,
            "action": "score",
            "out_path": str(score_out),
            "summary_path": str(score_summary),
            "batch_size": 4,
        },
    )
    if not bool(score_resp.get("ok")):
        raise RuntimeError(f"score action failed: {score_resp}")
    if not score_out.exists():
        raise RuntimeError(f"score output missing: {score_out}")

    worlds_resp = infer_json_action(
        cfg=endpoint_cfg,
        request_payload={
            **common,
            "action": "worlds",
            "out_path": str(worlds_out),
            "summary_path": str(worlds_summary),
            "num_worlds": int(args.num_worlds),
            "world_chunk_size": int(args.world_chunk_size),
            "active_temperature": 1.0,
            "strict_world_contracts": True,
            "make_model_mode": "beta_binomial_all",
            "make_model_use_learned_efficiency": True,
        },
    )
    if not bool(worlds_resp.get("ok")):
        raise RuntimeError(f"worlds action failed: {worlds_resp}")
    if not worlds_out.exists():
        raise RuntimeError(f"worlds output missing: {worlds_out}")

    scores = pd.read_parquet(score_out)
    worlds = pd.read_parquet(worlds_out)
    print("Smoke test passed.")
    print(f"- features: {features_path}")
    print(f"- score rows: {len(scores)}")
    print(f"- worlds rows: {len(worlds)}")
    print(f"- score response: {json.dumps(score_resp, sort_keys=True)}")
    print(f"- worlds response: {json.dumps(worlds_resp, sort_keys=True)}")


if __name__ == "__main__":
    main()
