"""Apples-to-apples evaluator: LGBM minutes vs TeamAlloc on identical val keys.

Example:
    uv run python -m scripts.experiments.eval_minutes_lgbm_vs_teamalloc \\
      --val-parquet artifacts/experiments/teamalloc_run/val_predictions.parquet \\
      --lgbm-artifact minutes_v1_safe_starter_20251214 \\
      --projections-data-root /home/daniel/projections-data

Outputs:
    - artifacts/experiments/team_alloc_eval/lgbm_vs_teamalloc.csv

Notes:
    - Grouping keys are always (game_id, team_id).
    - Top-8 overlap is set overlap of top-8 by actual vs predicted minutes per team-game.
    - "Smear" is 0 < pred < 3 minutes; "smear-on-DNP" is smear rows with minutes_actual == 0.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from projections.cli.score_minutes_v1 import _ensure_bundle_defaults, _score_rows
from projections.minutes_v1.datasets import default_features_path, ensure_columns


KEY_COLS = ("game_id", "team_id", "player_id")
GROUP_COLS = ("game_id", "team_id")
MAX_MISSING_KEYS = 5


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _top_k_overlap(df: pd.DataFrame, *, pred_col: str, top_k: int) -> float:
    overlaps: list[float] = []
    for _, g in df.groupby(list(GROUP_COLS), sort=False):
        if g.empty:
            continue
        n = len(g)
        k = min(int(top_k), n)
        if k <= 0:
            continue
        top_actual = set(g.nlargest(k, "minutes_actual")["player_id"].tolist())
        top_pred = set(g.nlargest(k, pred_col)["player_id"].tolist())
        overlaps.append(len(top_actual & top_pred) / k)
    return _safe_mean(overlaps)


def _compute_metrics(df: pd.DataFrame, *, pred_col: str, top_k: int) -> dict[str, float]:
    abs_err = (df[pred_col] - df["minutes_actual"]).abs()
    mae = float(abs_err.mean()) if len(df) else float("nan")

    ge10 = df["minutes_actual"] >= 10
    mae_ge_10 = float(abs_err[ge10].mean()) if int(ge10.sum()) else float("nan")

    top8 = _top_k_overlap(df, pred_col=pred_col, top_k=top_k)

    smear = (df[pred_col] > 0) & (df[pred_col] < 3)
    smear_per_team = df.assign(_smear=smear.astype(int)).groupby(list(GROUP_COLS))["_smear"].sum()
    smear_avg_per_team = float(smear_per_team.mean()) if len(smear_per_team) else float("nan")

    smear_on_dnp = smear & (df["minutes_actual"] == 0)
    smear_on_dnp_frac = float(smear_on_dnp.mean()) if len(df) else float("nan")

    return {
        "mae": mae,
        "mae_ge_10": mae_ge_10,
        "top8_overlap": top8,
        "smear_avg_per_team": smear_avg_per_team,
        "smear_on_dnp_frac": smear_on_dnp_frac,
    }


def _teamgame_leak(df: pd.DataFrame, *, pred_col: str) -> pd.Series:
    dnp = df["minutes_actual"] == 0
    totals = df.groupby(list(GROUP_COLS))[pred_col].sum()
    leak = df.loc[dnp].groupby(list(GROUP_COLS))[pred_col].sum()
    leak = leak.reindex(totals.index, fill_value=0.0)
    return leak


def _print_leak_percentiles(method: str, leak: pd.Series) -> None:
    if leak.empty:
        print(f"[leak] {method}: no team-games")
        return
    p50, p90, p99 = leak.quantile([0.5, 0.9, 0.99]).tolist()
    print(f"[leak] {method}: p50={p50:.1f} p90={p90:.1f} p99={p99:.1f}")


def _top_overlap(
    a: pd.Series,
    b: pd.Series,
    *,
    frac: float,
    label: str,
) -> None:
    if a.empty or b.empty:
        print(f"[overlap] {label}: empty")
        return
    n = len(a)
    k = max(1, int(math.ceil(n * frac)))
    a_top = set(a.sort_values(ascending=False).head(k).index.tolist())
    b_top = set(b.sort_values(ascending=False).head(k).index.tolist())
    overlap = len(a_top & b_top) / k
    print(f"[overlap] {label}: top {int(frac * 100)}% overlap={overlap:.3f} (k={k})")


def _print_worst(method: str, leak: pd.Series, *, limit: int = 20) -> None:
    if leak.empty:
        print(f"[worst] {method}: no team-games")
        return
    worst = (
        leak.sort_values(ascending=False)
        .reset_index()
        .rename(columns={0: "leak"})
        .head(limit)
    )
    print(f"[worst] {method} top {limit}")
    print(worst.to_string(index=False))


def _season_guess(game_date: pd.Timestamp) -> int:
    """Best-effort mapping from game_date -> season label used in gold paths.

    Minutes V1 gold paths use `season=<start_year>` (e.g. 2024 for 2024-25).
    """
    if int(game_date.month) >= 7:
        return int(game_date.year)
    return int(game_date.year) - 1


def _resolve_lgbm_dir(value: str) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate
    return Path("artifacts") / "minutes_lgbm" / value


def _resolve_val_parquet_path(value: Path) -> Path:
    candidate = value.expanduser()
    if candidate.is_dir():
        candidate = candidate / "val_predictions.parquet"
    if candidate.suffix != ".parquet" and candidate.exists() is False:
        # Allow passing a run directory path without the parquet suffix.
        alt = candidate / "val_predictions.parquet"
        if alt.exists():
            candidate = alt
    if candidate.exists():
        return candidate

    hints: list[str] = []
    root = Path("artifacts") / "experiments"
    if root.exists():
        matches = list(root.glob("**/val_predictions.parquet"))
        for match in sorted(matches, key=lambda p: str(p))[:10]:
            hints.append(str(match))
    hint_text = " Examples:\n  - " + "\n  - ".join(hints) if hints else ""
    raise FileNotFoundError(f"--val-parquet not found: {candidate}{hint_text}")


def _load_feature_columns(feature_columns_path: Path) -> list[str]:
    payload = json.loads(feature_columns_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(x) for x in payload]
    if isinstance(payload, dict):
        cols = payload.get("columns")
        if isinstance(cols, list):
            return [str(x) for x in cols]
    raise ValueError(f"Unrecognized feature_columns.json format at {feature_columns_path}")


def _load_features_for_keys(
    keys: pd.DataFrame,
    *,
    data_root: Path,
) -> pd.DataFrame:
    keys = keys.copy()
    keys["game_date"] = pd.to_datetime(keys["game_date"], errors="raise")
    keys["month"] = keys["game_date"].dt.month.astype(int)
    keys["season_guess"] = keys["game_date"].apply(_season_guess).astype(int)

    frames: list[pd.DataFrame] = []
    for (season_guess, month), part_keys in keys.groupby(["season_guess", "month"], sort=False):
        # Try the guessed season first, then fall back to the adjacent season if needed.
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
    # Guard against duplicate keys (should be unique in gold).
    dup = int(out.duplicated(subset=list(KEY_COLS)).sum())
    if dup:
        out = out.drop_duplicates(subset=list(KEY_COLS), keep="last").reset_index(drop=True)
    return out


def _rescale_to_team_total(df: pd.DataFrame, *, pred_col: str, out_col: str) -> pd.DataFrame:
    working = df.copy()
    pred = working[pred_col].to_numpy(dtype=float)
    pred = np.maximum(pred, 0.0)
    working[pred_col] = pred

    sums = working.groupby(list(GROUP_COLS))[pred_col].transform("sum").to_numpy(dtype=float)
    scale = np.where(sums > 0, 240.0 / sums, 0.0)
    working[out_col] = working[pred_col].to_numpy(dtype=float) * scale
    return working


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-parquet", type=Path, required=True)
    parser.add_argument(
        "--lgbm-artifact",
        type=str,
        required=True,
        help="Run id under artifacts/minutes_lgbm/ or an explicit bundle directory path",
    )
    parser.add_argument(
        "--projections-data-root",
        type=Path,
        required=True,
        help="Root containing gold/features_minutes_v1/season=*/month=*/features.parquet",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("artifacts/experiments/team_alloc_eval/lgbm_vs_teamalloc.csv"),
    )
    parser.add_argument(
        "--inspect-top-leak",
        type=int,
        default=0,
        help="If >0, write a worst_teamgames_players.csv report for top N leak team-games",
    )

    args = parser.parse_args()

    val_path = _resolve_val_parquet_path(args.val_parquet)
    val_df = pd.read_parquet(val_path)
    ensure_columns(val_df, {"game_date", "minutes_actual", "minutes_pred", *KEY_COLS})
    val_df = val_df.copy()
    hybrid_available = "minutes_pred_hybrid" in val_df.columns

    # Ensure unique key rows.
    dup = int(val_df.duplicated(subset=list(KEY_COLS)).sum())
    if dup:
        val_df = val_df.drop_duplicates(subset=list(KEY_COLS), keep="last").reset_index(drop=True)

    # Load LGBM bundle + feature columns.
    bundle_dir = _resolve_lgbm_dir(args.lgbm_artifact)
    if not bundle_dir.exists():
        raise FileNotFoundError(f"LGBM artifact not found at {bundle_dir}")

    bundle = _ensure_bundle_defaults(joblib.load(bundle_dir / "lgbm_quantiles.joblib"))
    feature_columns = _load_feature_columns(bundle_dir / "feature_columns.json")
    bundle["feature_columns"] = feature_columns

    # Load feature rows for the exact validation keys.
    keys = val_df.loc[:, ["game_date", *KEY_COLS]].drop_duplicates().copy()
    features = _load_features_for_keys(keys, data_root=args.projections_data_root)
    if features.empty:
        raise ValueError("Loaded zero feature rows for provided validation keys")

    # Restrict evaluation to the intersection of keys present in gold features.
    eval_df = val_df.merge(features, on=list(KEY_COLS), how="inner")
    if eval_df.empty:
        raise ValueError("No overlap between val_parquet keys and gold feature rows")

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
        print(
            f"[warn] Dropped {missing} val rows missing from gold features (limit={MAX_MISSING_KEYS})."
        )

    missing_feature_cols = [col for col in feature_columns if col not in eval_df.columns]
    if missing_feature_cols:
        raise ValueError(
            "Gold features frame missing required LGBM feature columns: "
            f"{missing_feature_cols[:20]}{'...' if len(missing_feature_cols) > 20 else ''}"
        )

    # LGBM predictions via Minutes V1 scorer (includes conformal offsets + rotation caps; no 240 constraint).
    scored = _score_rows(eval_df.copy(), bundle)
    if "minutes_p50" not in scored.columns:
        raise RuntimeError("LGBM scoring did not produce minutes_p50")
    eval_df["lgbm_raw"] = np.maximum(scored["minutes_p50"].to_numpy(dtype=float), 0.0)
    eval_df = _rescale_to_team_total(eval_df, pred_col="lgbm_raw", out_col="lgbm_rescaled")

    # Persist per-row scored output (exact key universe).
    out_dir = args.out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cols = ["game_id", "team_id", "player_id", "minutes_actual", "minutes_pred"]
    if "game_date" in eval_df.columns:
        base_cols.insert(3, "game_date")
    if hybrid_available and "minutes_pred_hybrid" in eval_df.columns:
        base_cols.append("minutes_pred_hybrid")
    base_cols.extend(["lgbm_raw", "lgbm_rescaled"])
    for optional in ["player_name", "status"]:
        if optional in eval_df.columns:
            base_cols.append(optional)
    rows_scored = eval_df.loc[:, base_cols].copy()
    rows_scored = rows_scored.rename(
        columns={
            "minutes_pred": "teamalloc_pred",
            "minutes_pred_hybrid": "teamalloc_pred_hybrid",
            "lgbm_raw": "lgbm_pred_raw",
            "lgbm_rescaled": "lgbm_pred_rescaled_240",
        }
    )
    rows_scored = rows_scored.sort_values(list(KEY_COLS)).reset_index(drop=True)
    rows_scored_path = out_dir / "rows_scored.parquet"
    rows_scored.to_parquet(rows_scored_path, index=False)

    # Metrics for each method.
    results: list[dict[str, Any]] = []
    methods: list[tuple[str, str]] = [("teamalloc", "minutes_pred")]
    if hybrid_available:
        methods.append(("teamalloc_hybrid", "minutes_pred_hybrid"))
    methods.extend([("lgbm_raw", "lgbm_raw"), ("lgbm_rescaled", "lgbm_rescaled")])

    for method, pred_col in methods:
        metrics = _compute_metrics(eval_df, pred_col=pred_col, top_k=args.top_k)
        results.append(
            {
                "method": method,
                "n_rows": int(len(eval_df)),
                "n_team_games": int(eval_df.groupby(list(GROUP_COLS)).ngroups),
                **metrics,
            }
        )

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out_path, index=False)

    # Team-game leak report.
    leak_frame = pd.DataFrame(index=eval_df.groupby(list(GROUP_COLS)).size().index)
    leak_frame.index.names = list(GROUP_COLS)
    leak_frame["total_teamalloc"] = eval_df.groupby(list(GROUP_COLS))["minutes_pred"].sum()
    leak_frame["leak_teamalloc"] = _teamgame_leak(eval_df, pred_col="minutes_pred")
    if hybrid_available:
        leak_frame["total_teamalloc_hybrid"] = eval_df.groupby(list(GROUP_COLS))["minutes_pred_hybrid"].sum()
        leak_frame["leak_teamalloc_hybrid"] = _teamgame_leak(eval_df, pred_col="minutes_pred_hybrid")
    leak_frame["total_lgbm_raw"] = eval_df.groupby(list(GROUP_COLS))["lgbm_raw"].sum()
    leak_frame["leak_lgbm_raw"] = _teamgame_leak(eval_df, pred_col="lgbm_raw")
    leak_frame["total_lgbm_rescaled"] = eval_df.groupby(list(GROUP_COLS))["lgbm_rescaled"].sum()
    leak_frame["leak_lgbm_rescaled"] = _teamgame_leak(eval_df, pred_col="lgbm_rescaled")
    leak_frame = leak_frame.reset_index().sort_values(list(GROUP_COLS)).reset_index(drop=True)
    teamgame_leak_path = out_dir / "teamgame_leak.parquet"
    leak_frame.to_parquet(teamgame_leak_path, index=False)

    # Leak percentiles summary.
    _print_leak_percentiles("teamalloc", leak_frame["leak_teamalloc"])
    if hybrid_available:
        _print_leak_percentiles("teamalloc_hybrid", leak_frame["leak_teamalloc_hybrid"])
    _print_leak_percentiles("lgbm_raw", leak_frame["leak_lgbm_raw"])
    _print_leak_percentiles("lgbm_rescaled", leak_frame["leak_lgbm_rescaled"])

    # Overlap of worst-leak team-games between TeamAlloc and LGBM.
    _top_overlap(
        leak_frame.set_index(list(GROUP_COLS))["leak_teamalloc"],
        leak_frame.set_index(list(GROUP_COLS))["leak_lgbm_rescaled"],
        frac=0.05,
        label="teamalloc vs lgbm_rescaled",
    )
    _top_overlap(
        leak_frame.set_index(list(GROUP_COLS))["leak_teamalloc"],
        leak_frame.set_index(list(GROUP_COLS))["leak_lgbm_rescaled"],
        frac=0.10,
        label="teamalloc vs lgbm_rescaled",
    )

    # Worst team-games per method.
    _print_worst("teamalloc", leak_frame.set_index(list(GROUP_COLS))["leak_teamalloc"])
    if hybrid_available:
        _print_worst(
            "teamalloc_hybrid",
            leak_frame.set_index(list(GROUP_COLS))["leak_teamalloc_hybrid"],
        )
    _print_worst("lgbm_raw", leak_frame.set_index(list(GROUP_COLS))["leak_lgbm_raw"])
    _print_worst("lgbm_rescaled", leak_frame.set_index(list(GROUP_COLS))["leak_lgbm_rescaled"])

    # Optional: inspect top leak team-games for TeamAlloc.
    if args.inspect_top_leak > 0:
        top_n = int(args.inspect_top_leak)
        worst = (
            leak_frame.set_index(list(GROUP_COLS))["leak_teamalloc"]
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )
        worst_keys = set(worst)
        offender_mask = (
            (rows_scored["minutes_actual"] == 0)
            & (rows_scored["teamalloc_pred"] >= 5)
            & rows_scored.apply(lambda r: (r["game_id"], r["team_id"]) in worst_keys, axis=1)
        )
        offenders = rows_scored.loc[
            offender_mask,
            [
                "game_id",
                "team_id",
                "player_id",
                "minutes_actual",
                "teamalloc_pred",
                "lgbm_pred_raw",
                "lgbm_pred_rescaled_240",
            ],
        ].sort_values(["game_id", "team_id", "teamalloc_pred"], ascending=[True, True, False])
        offenders_path = out_dir / "worst_teamgames_players.csv"
        offenders.to_csv(offenders_path, index=False)

    # Concise summary
    print(out_df.to_string(index=False))
    print(f"Wrote: {args.out_path}")
    print(f"Wrote: {rows_scored_path}")
    print(f"Wrote: {teamgame_leak_path}")
    if args.inspect_top_leak > 0:
        print(f"Wrote: {out_dir / 'worst_teamgames_players.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
