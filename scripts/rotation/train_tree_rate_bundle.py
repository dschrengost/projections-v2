#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from projections.rotation.tree_rate_bundle import train_tree_rate_bundle


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and persist a tree-rate bundle for live GTv2 overrides.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--bundle-dir", default=None)
    parser.add_argument("--model-type", choices=["lgbm", "xgb"], default="lgbm")
    parser.add_argument("--target-set", choices=["astreb", "full"], default="astreb")
    parser.add_argument("--cal-days", type=int, default=30)
    parser.add_argument("--num-boost-round", type=int, default=5000)
    parser.add_argument("--ast-line-threshold", type=float, default=7.0)
    parser.add_argument("--ast-weight-mult", type=float, default=1.0)
    parser.add_argument("--reb-line-threshold", type=float, default=10.0)
    parser.add_argument("--reb-weight-mult", type=float, default=1.0)
    parser.add_argument("--live-features-path", default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    bundle_dir = (
        Path(args.bundle_dir).expanduser().resolve()
        if args.bundle_dir
        else Path("/home/daniel/projections-data/artifacts/tree_rate_bundles")
        / f"tree_rate_{args.target_set}_{args.model_type}_{_utc_now_compact()}"
    )

    live_feature_columns: set[str] | None = None
    if args.live_features_path:
        live_features = pd.read_parquet(Path(args.live_features_path).expanduser().resolve())
        live_feature_columns = set(str(c) for c in live_features.columns)

    meta = train_tree_rate_bundle(
        dataset_dir=dataset_dir,
        bundle_dir=bundle_dir,
        model_type=str(args.model_type),
        target_set=str(args.target_set),
        cal_days=int(args.cal_days),
        num_boost_round=int(args.num_boost_round),
        ast_line_threshold=float(args.ast_line_threshold),
        ast_weight_mult=float(args.ast_weight_mult),
        reb_line_threshold=float(args.reb_line_threshold),
        reb_weight_mult=float(args.reb_weight_mult),
        live_feature_columns=live_feature_columns,
    )
    print(json.dumps({"bundle_dir": str(bundle_dir), **meta}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
