from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from projections import paths


ENRICHMENT_BASE_COLUMNS = {
    "vac_min_szn",
    "vac_min_guard_szn",
    "vac_min_wing_szn",
    "vac_min_big_szn",
    "team_pace_szn",
    "opp_pace_szn",
    "team_off_rtg_szn",
    "team_def_rtg_szn",
    "opp_def_rtg_szn",
}


def _resolve_features_path(*, version: str | None, features_path: Path | None, data_root: Path) -> Path:
    if features_path is not None:
        return features_path.expanduser().resolve()
    if version is None:
        raise ValueError("Provide --version or --features-path.")
    return (data_root / "training" / "datasets" / version / "features.parquet").expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick post-build check that training dataset enrichment columns are populated.",
    )
    parser.add_argument("--version", help="Dataset version under <data_root>/training/datasets/<version>.")
    parser.add_argument("--features-path", type=Path, help="Explicit path to features.parquet.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=paths.get_data_root(),
        help="Data root (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    )
    args = parser.parse_args()

    features_path = _resolve_features_path(
        version=args.version,
        features_path=args.features_path,
        data_root=args.data_root,
    )
    if not features_path.exists():
        raise SystemExit(f"features.parquet not found: {features_path}")

    parquet = pq.ParquetFile(features_path)
    cols = parquet.schema.names
    suffixed = [c for c in cols if c.endswith("_x") or c.endswith("_y")]
    suffixed_enrich = [c for c in suffixed if c.rsplit("_", 1)[0] in ENRICHMENT_BASE_COLUMNS]
    if suffixed_enrich:
        raise SystemExit(f"FAIL: found suffixed enrichment columns: {suffixed_enrich}")

    required = ["opp_pace_szn", "vac_min_szn", "opp_ctx_missing", "vac_missing"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise SystemExit(f"FAIL: features.parquet missing required columns: {missing}")

    df = pd.read_parquet(features_path, columns=required)
    opp_pace_rate = float(df["opp_pace_szn"].notna().mean())
    vac_rate = float(df["vac_min_szn"].notna().mean())
    opp_missing_rate = float(pd.to_numeric(df["opp_ctx_missing"], errors="coerce").fillna(1).mean())
    vac_missing_rate = float(pd.to_numeric(df["vac_missing"], errors="coerce").fillna(1).mean())

    print("features_path", str(features_path))
    print("rows", len(df))
    print("opp_pace_szn_non_null_rate", opp_pace_rate)
    print("vac_min_szn_non_null_rate", vac_rate)
    print("opp_ctx_missing_mean", opp_missing_rate)
    print("vac_missing_mean", vac_missing_rate)

    if opp_pace_rate <= 0:
        raise SystemExit("FAIL: opp_pace_szn non-null rate is 0.")
    if vac_rate <= 0:
        raise SystemExit("FAIL: vac_min_szn non-null rate is 0.")
    if opp_missing_rate >= 1.0:
        raise SystemExit("FAIL: opp_ctx_missing is 100%.")
    if vac_missing_rate >= 1.0:
        raise SystemExit("FAIL: vac_missing is 100%.")


if __name__ == "__main__":
    main()

