#!/usr/bin/env python3
"""Evaluate a rotation+minutes+allocation (RotAlloc) baseline on TeamAlloc key universe.

Example:
  uv run python -m scripts.experiments.eval_rotalloc_vs_teamalloc \\
    --val-parquet artifacts/experiments/team_alloc_v3_time/val_predictions.parquet \\
    --rotalloc-dir artifacts/experiments/lgbm_rotalloc_v1 \\
    --projections-data-root /home/daniel/projections-data \\
    --out-dir artifacts/experiments/rotalloc_eval \\
    --a 1.5 --cap-max 48 --inspect-top-leak 25

Outputs:
  - <out-dir>/rows_scored.parquet
  - <out-dir>/teamgame_leak.parquet
  - <out-dir>/worst_teamgames_players.csv (optional)
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
    # gold/features_minutes_v1 uses season=<start_year> (e.g. 2024 for 2024-25)
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


def _print_method_summary(
    df: pd.DataFrame,
    *,
    name: str,
    pred_col: str,
    top_k: int,
    eligible_sizes: pd.Series | None = None,
) -> None:
    leak = _leak_series(df, pred_col=pred_col)
    p50, p90, p99 = leak.quantile([0.5, 0.9, 0.99]).tolist() if len(leak) else (float("nan"),) * 3
    if eligible_sizes is not None and len(eligible_sizes):
        elig_p50, elig_p90 = eligible_sizes.quantile([0.5, 0.9]).tolist()
        elig_text = f" | eligible_size p50/p90={elig_p50:.1f}/{elig_p90:.1f}"
    else:
        elig_text = ""
    print(
        f"[{name}] MAE={_mae(df, pred_col=pred_col):.3f} "
        f"MAE>=10={_mae_ge_10(df, pred_col=pred_col):.3f} "
        f"Top-{top_k} overlap={topk_overlap(df, pred_col=pred_col, top_k=top_k):.3f} | "
        f"leak_dnp p50/p90/p99={p50:.1f}/{p90:.1f}/{p99:.1f} | "
        f"leak50% concentration={_leak_concentration(leak)} team-games"
        f"{elig_text}"
    )
    for frac in (0.05, 0.10):
        tm = _trimmed_metrics(df, pred_col=pred_col, trim_frac=frac)
        print(
            f"  trim top {int(frac*100)}% by leak: "
            f"MAE={tm['mae']:.3f} leak_p90={tm['leak_p90']:.1f} leak_p99={tm['leak_p99']:.1f}"
        )

    worst = leak.sort_values(ascending=False).reset_index().rename(columns={0: "leak"}).head(20)
    print(f"  worst 20 by leak ({pred_col}):")
    print(worst.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-parquet", type=Path, required=True)
    parser.add_argument("--rotalloc-dir", type=Path, required=True)
    parser.add_argument("--projections-data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/experiments/rotalloc_eval"))
    parser.add_argument("--a", type=float, default=1.5)
    parser.add_argument("--cap-max", type=float, default=48.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--p-cutoff", type=float, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--use-expected-k", action="store_true")
    parser.add_argument("--k-min", type=int, default=8)
    parser.add_argument("--k-max", type=int, default=11)
    parser.add_argument("--inspect-top-leak", type=int, default=0)

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    val_df = pd.read_parquet(args.val_parquet).copy()
    ensure_columns(val_df, {"minutes_actual", *KEY_COLS})
    if "game_date" not in val_df.columns:
        raise ValueError("--val-parquet must include game_date to load gold features")

    # Normalize optional comparison columns to a consistent naming scheme.
    if "minutes_pred" in val_df.columns and "teamalloc_pred" not in val_df.columns:
        val_df["teamalloc_pred"] = pd.to_numeric(val_df["minutes_pred"], errors="coerce")
    if "minutes_pred_hybrid" in val_df.columns and "teamalloc_pred_hybrid" not in val_df.columns:
        val_df["teamalloc_pred_hybrid"] = pd.to_numeric(val_df["minutes_pred_hybrid"], errors="coerce")

    # Ensure unique keys.
    dup = int(val_df.duplicated(subset=list(KEY_COLS)).sum())
    if dup:
        val_df = val_df.drop_duplicates(subset=list(KEY_COLS), keep="last").reset_index(drop=True)

    # Load trained models + feature list.
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

    # Join features from gold on the exact key universe.
    keys = val_df.loc[:, ["game_date", *KEY_COLS]].drop_duplicates().copy()
    features = _load_features_for_keys(keys, data_root=args.projections_data_root)
    if features.empty:
        raise ValueError("Loaded zero feature rows for provided validation keys")

    eval_df = val_df.merge(features, on=list(KEY_COLS), how="inner")
    # Avoid pandas merge suffixing when gold also contains game_date.
    # Prefer the val parquet's game_date (x) for determinism.
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

    # `_is_nan` indicator columns are derived at eval time from base feature NaNs, so it's fine
    # if they are absent from gold. Hard-fail only if non-indicator feature columns are missing.
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

    # Allocate to 240 per team-game.
    pred = np.zeros(len(eval_df), dtype=np.float64)
    eligible_sizes: list[int] = []
    for _, g in eval_df.groupby(list(GROUP_COLS), sort=False):
        idx = g.index.to_numpy()
        eligible = build_eligible_mask(
            g["p_rot"].to_numpy(dtype=float),
            g["mu"].to_numpy(dtype=float),
            np.ones(len(g), dtype=bool),
            a=float(args.a),
            p_cutoff=args.p_cutoff,
            topk=args.topk,
            k_min=int(args.k_min),
            k_max=int(args.k_max),
            use_expected_k=bool(args.use_expected_k),
        )
        eligible_sizes.append(int(np.sum(eligible)))
        minutes = allocate_team_minutes(
            g["p_rot"].to_numpy(dtype=float),
            g["mu"].to_numpy(dtype=float),
            np.ones(len(g), dtype=bool),
            a=float(args.a),
            cap_max=float(args.cap_max),
            p_cutoff=args.p_cutoff,
            topk=args.topk,
            k_min=int(args.k_min),
            k_max=int(args.k_max),
            use_expected_k=bool(args.use_expected_k),
        )
        pred[idx] = minutes
    eval_df["rotalloc_pred_240"] = pred
    eligible_sizes_series = pd.Series(eligible_sizes)

    # Persist per-row scored output.
    base_cols = [
        "game_id",
        "team_id",
        "player_id",
        "game_date",
        "minutes_actual",
        "p_rot",
        "mu",
        "rotalloc_pred_240",
    ]
    for col in ["teamalloc_pred", "teamalloc_pred_hybrid", "lgbm_pred_rescaled_240", "lgbm_pred_raw"]:
        if col in eval_df.columns:
            base_cols.append(col)
    for col in ["player_name", "status"]:
        if col in eval_df.columns:
            base_cols.append(col)

    rows_scored = eval_df.loc[:, base_cols].copy()
    rows_scored_path = args.out_dir / "rows_scored.parquet"
    rows_scored.to_parquet(rows_scored_path, index=False)

    # Per-team-game leak + totals.
    teamgame = pd.DataFrame(index=eval_df.groupby(list(GROUP_COLS)).size().index)
    teamgame.index.names = list(GROUP_COLS)

    def _add_method(prefix: str, col: str) -> None:
        teamgame[f"total_{prefix}"] = eval_df.groupby(list(GROUP_COLS))[col].sum()
        teamgame[f"leak_{prefix}"] = _leak_series(eval_df, pred_col=col)

    _add_method("rotalloc", "rotalloc_pred_240")
    if "teamalloc_pred" in eval_df.columns:
        _add_method("teamalloc", "teamalloc_pred")
    if "teamalloc_pred_hybrid" in eval_df.columns:
        _add_method("teamalloc_hybrid", "teamalloc_pred_hybrid")
    if "lgbm_pred_rescaled_240" in eval_df.columns:
        _add_method("lgbm_rescaled", "lgbm_pred_rescaled_240")
    if "lgbm_pred_raw" in eval_df.columns:
        _add_method("lgbm_raw", "lgbm_pred_raw")

    teamgame = teamgame.reset_index().sort_values(list(GROUP_COLS)).reset_index(drop=True)
    teamgame_path = args.out_dir / "teamgame_leak.parquet"
    teamgame.to_parquet(teamgame_path, index=False)

    # Print metrics (leak-focused).
    _print_method_summary(
        eval_df,
        name="rotalloc",
        pred_col="rotalloc_pred_240",
        top_k=int(args.top_k),
        eligible_sizes=eligible_sizes_series,
    )
    if "teamalloc_pred" in eval_df.columns:
        _print_method_summary(eval_df, name="teamalloc", pred_col="teamalloc_pred", top_k=int(args.top_k))
    if "teamalloc_pred_hybrid" in eval_df.columns:
        _print_method_summary(eval_df, name="teamalloc_hybrid", pred_col="teamalloc_pred_hybrid", top_k=int(args.top_k))

    # Optional: offender report for worst rotalloc leak team-games.
    if args.inspect_top_leak > 0:
        top_n = int(args.inspect_top_leak)
        leak = _leak_series(eval_df, pred_col="rotalloc_pred_240").sort_values(ascending=False)
        worst_keys = leak.head(top_n).index
        worst_mask = pd.MultiIndex.from_frame(rows_scored.loc[:, list(GROUP_COLS)]).isin(worst_keys)
        offenders = (
            rows_scored.loc[
                (rows_scored["minutes_actual"] == 0)
                & (rows_scored["rotalloc_pred_240"] >= 5)
                & worst_mask,
                [
                    "game_id",
                    "team_id",
                    "player_id",
                    "minutes_actual",
                    "rotalloc_pred_240",
                    "p_rot",
                    "mu",
                ],
            ]
            .sort_values(["game_id", "team_id", "rotalloc_pred_240"], ascending=[True, True, False])
            .reset_index(drop=True)
        )
        offenders_path = args.out_dir / "worst_teamgames_players.csv"
        offenders.to_csv(offenders_path, index=False)
        print(f"Wrote: {offenders_path}")

    print(f"Wrote: {rows_scored_path}")
    print(f"Wrote: {teamgame_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
