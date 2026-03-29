from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_bins(raw: str) -> list[float]:
    vals = [float(part.strip()) for part in str(raw).split(",") if str(part).strip()]
    if len(vals) < 2:
        raise ValueError("--bins must provide at least 2 comma-separated values")
    if any(later <= prev for later, prev in zip(vals[1:], vals[:-1])):
        raise ValueError("--bins must be strictly increasing")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Build empirical minutes uncertainty lookup from GTv2 eval rows.")
    parser.add_argument("--player-summary-eval", type=str, required=True, help="Path to eval CSV or parquet with pred/actual minutes.")
    parser.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Optional explicit output path. Defaults under artifact dir next to the input run.",
    )
    parser.add_argument(
        "--bins",
        type=str,
        default="0,8,12,16,20,24,28,32,36,40,60",
        help="Comma-separated minute bin edges.",
    )
    parser.add_argument(
        "--active-prob-min",
        type=float,
        default=0.5,
        help="Filter eval rows to predicted-active players at or above this threshold.",
    )
    parser.add_argument(
        "--sigma-source",
        type=str,
        choices=["p80", "iqr"],
        default="iqr",
        help="Residual width estimator to convert into Gaussian sigma.",
    )
    args = parser.parse_args()

    src = Path(str(args.player_summary_eval)).expanduser().resolve()
    if src.suffix == ".parquet":
        df = pd.read_parquet(src)
    else:
        df = pd.read_csv(src)
    # Accept both naming conventions
    col_renames = {"minutes_pred": "pred_minutes", "minutes_actual": "actual_minutes", "active_pred": "pred_active_prob"}
    df = df.rename(columns={k: v for k, v in col_renames.items() if k in df.columns})
    required = {"pred_minutes", "actual_minutes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns in {src}: {sorted(missing)}")

    if "pred_active_prob" in df.columns:
        df = df.loc[pd.to_numeric(df["pred_active_prob"], errors="coerce").fillna(0.0) >= float(args.active_prob_min)].copy()
    else:
        df = df.copy()
    if df.empty:
        raise ValueError("no rows left after active-prob filter")

    bins = _parse_bins(args.bins)
    df["pred_minutes"] = pd.to_numeric(df["pred_minutes"], errors="coerce")
    df["actual_minutes"] = pd.to_numeric(df["actual_minutes"], errors="coerce")
    df = df.loc[df["pred_minutes"].notna() & df["actual_minutes"].notna()].copy()
    df["resid"] = df["actual_minutes"] - df["pred_minutes"]
    df["bucket"] = pd.cut(df["pred_minutes"], bins=bins, right=False, include_lowest=True)

    rows: list[dict[str, float | int | str]] = []
    sigma_values: list[float] = []
    for bucket, g in df.groupby("bucket", observed=False):
        if len(g) == 0:
            sigma = sigma_values[-1] if sigma_values else 1.5
            rows.append(
                {
                    "bucket": str(bucket),
                    "n": 0,
                    "pred_minutes_mean": float("nan"),
                    "actual_minutes_mean": float("nan"),
                    "bias": float("nan"),
                    "mae": float("nan"),
                    "q10_resid": float("nan"),
                    "q90_resid": float("nan"),
                    "q25_resid": float("nan"),
                    "q75_resid": float("nan"),
                    "sigma_p80": float("nan"),
                    "sigma_iqr": float("nan"),
                    "selected_sigma": float(sigma),
                }
            )
            sigma_values.append(float(sigma))
            continue
        q10 = float(g["resid"].quantile(0.10))
        q90 = float(g["resid"].quantile(0.90))
        q25 = float(g["resid"].quantile(0.25))
        q75 = float(g["resid"].quantile(0.75))
        sigma_p80 = float((q90 - q10) / 2.5631031311)
        sigma_iqr = float((q75 - q25) / 1.3489795)
        sigma = sigma_p80 if str(args.sigma_source) == "p80" else sigma_iqr
        sigma = max(float(sigma), 0.25)
        rows.append(
            {
                "bucket": str(bucket),
                "n": int(len(g)),
                "pred_minutes_mean": float(g["pred_minutes"].mean()),
                "actual_minutes_mean": float(g["actual_minutes"].mean()),
                "bias": float(g["resid"].mean()),
                "mae": float(g["resid"].abs().mean()),
                "q10_resid": q10,
                "q90_resid": q90,
                "q25_resid": q25,
                "q75_resid": q75,
                "sigma_p80": sigma_p80,
                "sigma_iqr": sigma_iqr,
                "selected_sigma": float(sigma),
            }
        )
        sigma_values.append(float(sigma))

    if args.out_json:
        out_json = Path(str(args.out_json)).expanduser().resolve()
    else:
        run_root = src.parent
        out_dir = run_root / "minutes_uncertainty_lookup"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / f"lookup_{args.sigma_source}_{_utc_now_compact()}.json"

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(src),
        "active_prob_min": float(args.active_prob_min),
        "sigma_source": str(args.sigma_source),
        "bin_edges": [float(x) for x in bins],
        "sigma_by_bin": [float(x) for x in sigma_values],
        "bucket_summary": rows,
        "row_count": int(len(df)),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "row_count": int(len(df)), "sigma_by_bin": sigma_values}, indent=2))


if __name__ == "__main__":
    main()
