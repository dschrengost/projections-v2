#!/usr/bin/env python3
"""Sweep RotAlloc pruning knobs to reduce leak_dnp on TeamAlloc key universe.

Example:
  uv run python -m scripts.experiments.sweep_rotalloc_leak \
    --val-parquet artifacts/experiments/team_alloc_v3_time/val_predictions.parquet \
    --rotalloc-dir artifacts/experiments/lgbm_rotalloc_v1 \
    --projections-data-root /home/daniel/projections-data \
    --out-dir artifacts/experiments/rotalloc_sweep
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from projections.minutes_v1.datasets import default_features_path, ensure_columns
from projections.models.rotalloc import allocate_team_minutes, build_eligible_mask, topk_overlap


KEY_COLS = ("game_id", "team_id", "player_id")
GROUP_COLS = ("game_id", "team_id")
MAX_MISSING_KEYS = 5


def _season_guess(game_date: pd.Timestamp) -> int:
    if int(game_date.month) >= 7:
        return int(game_date.year)
    return int(game_date.year) - 1


def _load_features_for_keys(keys: pd.DataFrame, *, data_root: Path) -> pd.DataFrame:
    keys = keys.copy()
    keys["game_date"] = pd.to_datetime(keys["game_date"], errors="raise")
    keys["month"] = keys["game_date"].dt.month.astype(int)
    keys["season_guess"] = keys["game_date"].apply(_season_guess).astype(int)

    frames: list[pd.DataFrame] = []
    for (season_guess, month), part_keys in keys.groupby(["season_guess", "month"], sort=False):
        season_candidates = [int(season_guess), int(season_guess) - 1, int(season_guess) + 1]
        filtered: pd.DataFrame | None = None
        for season in season_candidates:
            path = default_features_path(data_root, season=season, month=int(month))
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            ensure_columns(df, KEY_COLS)
            candidate = df.merge(part_keys.loc[:, list(KEY_COLS)], on=list(KEY_COLS), how="inner")
            if not candidate.empty:
                filtered = candidate
                break
        if filtered is None:
            raise FileNotFoundError(
                "No matching gold feature rows found for "
                f"season_candidates={season_candidates} month={month:02d} "
                f"(searched under {data_root / 'gold' / 'features_minutes_v1'})"
            )
        frames.append(filtered)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    dup = int(out.duplicated(subset=list(KEY_COLS)).sum())
    if dup:
        out = out.drop_duplicates(subset=list(KEY_COLS), keep="last").reset_index(drop=True)
    return out


def ensure_infer_feature_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    indicator_suffix: str = "_is_nan",
) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if col.endswith(indicator_suffix):
            base = col[: -len(indicator_suffix)]
            if base not in df.columns:
                df[base] = 0.0
        elif col not in df.columns:
            df[col] = 0.0
    for col in feature_cols:
        if not col.endswith(indicator_suffix):
            continue
        base = col[: -len(indicator_suffix)]
        if base in df.columns:
            df[col] = df[base].isna().astype(np.float32)
        else:
            df[col] = 1.0
    return df


def fill_missing_values(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def _leak_series(df: pd.DataFrame, *, pred_col: str) -> pd.Series:
    dnp = df["minutes_actual"] == 0
    totals = df.groupby(list(GROUP_COLS))[pred_col].sum()
    leak = df.loc[dnp].groupby(list(GROUP_COLS))[pred_col].sum()
    return leak.reindex(totals.index, fill_value=0.0)


def _mae(df: pd.DataFrame, *, pred_col: str) -> float:
    return float((df[pred_col] - df["minutes_actual"]).abs().mean()) if len(df) else float("nan")


def _mae_ge_10(df: pd.DataFrame, *, pred_col: str) -> float:
    mask = df["minutes_actual"] >= 10
    return float((df.loc[mask, pred_col] - df.loc[mask, "minutes_actual"]).abs().mean()) if int(mask.sum()) else float("nan")


def _leak_concentration(leak: pd.Series, *, frac: float = 0.5) -> int:
    total = float(leak.sum())
    if total <= 0:
        return 0
    ordered = leak.sort_values(ascending=False)
    csum = ordered.cumsum()
    return int((csum <= (frac * total)).sum() + 1)


def _trimmed_metrics(
    df: pd.DataFrame,
    *,
    pred_col: str,
    trim_frac: float,
) -> dict[str, float]:
    leak = _leak_series(df, pred_col=pred_col)
    if leak.empty:
        return {"mae": float("nan"), "leak_p90": float("nan"), "leak_p99": float("nan")}
    n = len(leak)
    k = max(1, int(math.ceil(n * float(trim_frac))))
    keep_team_games = leak.sort_values(ascending=False).iloc[k:].index
    kept = df.set_index(list(GROUP_COLS)).loc[keep_team_games].reset_index()
    kept_leak = _leak_series(kept, pred_col=pred_col)
    leak_p90 = float(kept_leak.quantile(0.9)) if len(kept_leak) else float("nan")
    leak_p99 = float(kept_leak.quantile(0.99)) if len(kept_leak) else float("nan")
    return {"mae": _mae(kept, pred_col=pred_col), "leak_p90": leak_p90, "leak_p99": leak_p99}


def _parse_float_list(value: str) -> list[float | None]:
    items: list[float | None] = []
    for raw in value.split(","):
        token = raw.strip().lower()
        if token in {"none", "null", ""}:
            items.append(None)
        else:
            items.append(float(token))
    return items


def _parse_numeric_list(value: str, *, dtype: type) -> list:
    items: list = []
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        items.append(dtype(token))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-parquet", type=Path, required=True)
    parser.add_argument("--rotalloc-dir", type=Path, required=True)
    parser.add_argument("--projections-data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/experiments/rotalloc_sweep"))
    parser.add_argument("--a-values", type=str, default="1.5,2.5,3.5")
    parser.add_argument("--p-cutoffs", type=str, default="none,0.2,0.4,0.65")
    parser.add_argument("--k-min", type=int, default=8)
    parser.add_argument("--k-max-values", type=str, default="9,10,11")
    parser.add_argument("--cap-max", type=float, default=48.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--mae-ge-10-guardrail", type=float, default=0.5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    val_df = pd.read_parquet(args.val_parquet).copy()
    ensure_columns(val_df, {"minutes_actual", *KEY_COLS})
    if "game_date" not in val_df.columns:
        raise ValueError("--val-parquet must include game_date to load gold features")

    dup = int(val_df.duplicated(subset=list(KEY_COLS)).sum())
    if dup:
        val_df = val_df.drop_duplicates(subset=list(KEY_COLS), keep="last").reset_index(drop=True)

    rotalloc_dir = args.rotalloc_dir
    feature_payload = json.loads((rotalloc_dir / "feature_columns.json").read_text(encoding="utf-8"))
    feature_cols = list(feature_payload.get("columns", [])) if isinstance(feature_payload, dict) else list(feature_payload)
    if not feature_cols:
        raise ValueError(f"Empty feature list in {rotalloc_dir / 'feature_columns.json'}")

    clf = joblib.load(rotalloc_dir / "rot8_classifier.joblib")
    reg = joblib.load(rotalloc_dir / "minutes_regressor.joblib")
    calibrator = None
    for name in ("rot8_calibrator_sigmoid.joblib", "rot8_calibrator_isotonic.joblib"):
        calibrator_path = rotalloc_dir / name
        if calibrator_path.exists():
            calibrator = joblib.load(calibrator_path)
            break

    keys = val_df.loc[:, ["game_date", *KEY_COLS]].drop_duplicates().copy()
    features = _load_features_for_keys(keys, data_root=args.projections_data_root)
    if features.empty:
        raise ValueError("Loaded zero feature rows for provided validation keys")

    eval_df = val_df.merge(features, on=list(KEY_COLS), how="inner")
    if "game_date" not in eval_df.columns:
        if "game_date_x" in eval_df.columns:
            eval_df = eval_df.rename(columns={"game_date_x": "game_date"})
            if "game_date_y" in eval_df.columns:
                eval_df = eval_df.drop(columns=["game_date_y"])
        elif "game_date_y" in eval_df.columns:
            eval_df = eval_df.rename(columns={"game_date_y": "game_date"})

    missing = len(val_df) - len(eval_df)
    if missing:
        if missing > MAX_MISSING_KEYS:
            sample_missing = (
                val_df.merge(eval_df[list(KEY_COLS)], on=list(KEY_COLS), how="left", indicator=True)
                .query("_merge == 'left_only'")
                .loc[:, list(KEY_COLS)]
                .head(10)
            )
            raise ValueError(
                f"Missing {missing} key rows in gold features (limit={MAX_MISSING_KEYS}). "
                f"Sample missing keys:\n{sample_missing.to_string(index=False)}"
            )
        print(f"[warn] Dropped {missing} val rows missing from gold features (limit={MAX_MISSING_KEYS}).")

    missing_feature_cols = [
        c for c in feature_cols if c not in eval_df.columns and not c.endswith("_is_nan")
    ]
    if missing_feature_cols:
        raise ValueError(
            "Gold features frame missing required columns: "
            f"{missing_feature_cols[:20]}{'...' if len(missing_feature_cols) > 20 else ''}"
        )

    eval_df = eval_df.sort_values(list(KEY_COLS)).reset_index(drop=True)
    eval_df = ensure_infer_feature_columns(eval_df, feature_cols)
    eval_df = fill_missing_values(eval_df, feature_cols)

    X = eval_df[feature_cols]
    p_raw = clf.predict_proba(X)[:, 1]
    p_rot = calibrator.transform(p_raw) if calibrator is not None else p_raw
    mu = reg.predict(X)

    eval_df["p_rot"] = p_rot.astype(float)
    eval_df["mu"] = mu.astype(float)

    groups = []
    for _, g in eval_df.groupby(list(GROUP_COLS), sort=False):
        groups.append(
            {
                "idx": g.index.to_numpy(),
                "p_rot": g["p_rot"].to_numpy(dtype=float),
                "mu": g["mu"].to_numpy(dtype=float),
            }
        )

    a_values = _parse_numeric_list(args.a_values, dtype=float)
    p_cutoffs = _parse_float_list(args.p_cutoffs)
    k_max_values = _parse_numeric_list(args.k_max_values, dtype=int)
    use_expected_k = True

    configs = []
    for a in a_values:
        for p_cut in p_cutoffs:
            for k_max in k_max_values:
                configs.append(
                    {
                        "a": float(a),
                        "p_cutoff": p_cut,
                        "use_expected_k": bool(use_expected_k),
                        "k_max": int(k_max),
                    }
                )

    summary_rows = []
    baseline_mae_ge_10 = float("nan")
    baseline_key = {
        "a": float(a_values[0]) if a_values else 1.5,
        "p_cutoff": None,
        "use_expected_k": True,
        "k_max": int(k_max_values[0]) if k_max_values else 11,
    }

    for cfg in configs:
        pred = np.zeros(len(eval_df), dtype=np.float64)
        eligible_sizes = []
        for g in groups:
            eligible = build_eligible_mask(
                g["p_rot"],
                g["mu"],
                np.ones_like(g["p_rot"], dtype=bool),
                a=cfg["a"],
                p_cutoff=cfg["p_cutoff"],
                k_min=int(args.k_min),
                k_max=int(cfg["k_max"]),
                use_expected_k=cfg["use_expected_k"],
            )
            eligible_sizes.append(int(np.sum(eligible)))
            minutes = allocate_team_minutes(
                g["p_rot"],
                g["mu"],
                np.ones_like(g["p_rot"], dtype=bool),
                a=cfg["a"],
                cap_max=float(args.cap_max),
                p_cutoff=cfg["p_cutoff"],
                k_min=int(args.k_min),
                k_max=int(cfg["k_max"]),
                use_expected_k=cfg["use_expected_k"],
            )
            pred[g["idx"]] = minutes

        col = "rotalloc_pred_240"
        eval_df[col] = pred
        leak = _leak_series(eval_df, pred_col=col)
        leak_p50, leak_p90, leak_p99 = leak.quantile([0.5, 0.9, 0.99]).tolist() if len(leak) else (
            float("nan"),
            float("nan"),
            float("nan"),
        )
        eligible_series = pd.Series(eligible_sizes)
        elig_p50, elig_p90 = (
            eligible_series.quantile([0.5, 0.9]).tolist() if len(eligible_series) else (float("nan"),) * 2
        )

        row = {
            "run_name": f"a{cfg['a']}_pc{cfg['p_cutoff']}_expk{int(cfg['use_expected_k'])}_kmax{cfg['k_max']}",
            "a": cfg["a"],
            "p_cutoff": cfg["p_cutoff"],
            "use_expected_k": cfg["use_expected_k"],
            "k_min": int(args.k_min),
            "k_max": int(cfg["k_max"]),
            "mae": _mae(eval_df, pred_col=col),
            "mae_ge_10": _mae_ge_10(eval_df, pred_col=col),
            "top8_overlap": topk_overlap(eval_df, pred_col=col, top_k=int(args.top_k)),
            "leak_p50": float(leak_p50),
            "leak_p90": float(leak_p90),
            "leak_p99": float(leak_p99),
            "leak50_concentration": _leak_concentration(leak),
            "eligible_size_p50": float(elig_p50),
            "eligible_size_p90": float(elig_p90),
        }
        summary_rows.append(row)

        if (
            math.isfinite(row["mae_ge_10"])
            and cfg["a"] == baseline_key["a"]
            and cfg["p_cutoff"] is None
            and cfg["use_expected_k"] is True
            and cfg["k_max"] == baseline_key["k_max"]
        ):
            baseline_mae_ge_10 = float(row["mae_ge_10"])

    summary = pd.DataFrame(summary_rows).sort_values(
        ["leak_p50", "leak_p90", "leak_p99", "mae_ge_10", "mae"]
    )
    summary_path = args.out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    if not math.isfinite(baseline_mae_ge_10) and not summary.empty:
        baseline_mae_ge_10 = float(summary.iloc[0]["mae_ge_10"])

    if math.isfinite(baseline_mae_ge_10):
        eligible = summary[summary["mae_ge_10"] <= baseline_mae_ge_10 + float(args.mae_ge_10_guardrail)]
    else:
        eligible = summary

    print(f"Wrote: {summary_path}")
    print(f"Baseline MAE>=10: {baseline_mae_ge_10:.3f}")
    if eligible.empty:
        print("No configs met MAE>=10 constraint; showing best overall by leak.")
        best = summary.head(1)
    else:
        best = eligible.sort_values(["leak_p50", "leak_p90", "leak_p99", "mae_ge_10", "mae"]).head(1)

    if not best.empty:
        print("Best config:")
        print(best.to_string(index=False))
        best_cfg = best.iloc[0].to_dict()
        pred = np.zeros(len(eval_df), dtype=np.float64)
        for g in groups:
            minutes = allocate_team_minutes(
                g["p_rot"],
                g["mu"],
                np.ones_like(g["p_rot"], dtype=bool),
                a=float(best_cfg["a"]),
                cap_max=float(args.cap_max),
                p_cutoff=best_cfg["p_cutoff"],
                k_min=int(best_cfg["k_min"]),
                k_max=int(best_cfg["k_max"]),
                use_expected_k=bool(best_cfg["use_expected_k"]),
            )
            pred[g["idx"]] = minutes
        eval_df["rotalloc_pred_240"] = pred
        for frac in (0.05, 0.10):
            tm = _trimmed_metrics(eval_df, pred_col="rotalloc_pred_240", trim_frac=frac)
            print(
                f"  best trim top {int(frac*100)}% by leak: "
                f"MAE={tm['mae']:.3f} leak_p90={tm['leak_p90']:.1f} leak_p99={tm['leak_p99']:.1f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
