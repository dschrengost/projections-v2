#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from projections.rotation.tree_rate_bundle import score_tree_rate_bundle_features_to_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a persisted tree-rate bundle on live GTv2 features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--game-date", default=None)
    args = parser.parse_args()

    features_df = pd.read_parquet(Path(args.features_path).expanduser().resolve())
    report = score_tree_rate_bundle_features_to_csv(
        features_df=features_df,
        bundle_dir=Path(args.bundle_dir).expanduser().resolve(),
        output_csv=Path(args.output_csv).expanduser().resolve(),
        include_extra_cols=["player_name"],
        game_date=str(args.game_date) if args.game_date else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
