"""
Decision Report for Usage Shares v1 (FGA Residual-on-Baseline).

Produces a comprehensive wire/don't-wire decision including:
- Bootstrap significance tests
- High-vacancy behavior analysis
- Comparison across variants (baseline, residual, starterless residual)

Usage:
    uv run python -m scripts.usage_shares_v1.decision_report \
        --data-root /home/daniel/projections-data \
        --start-date 2024-10-22 \
        --end-date 2025-11-28 \
        --seed 1337
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import typer

from projections.paths import data_path
from projections.usage_shares_v1.features import (
    CATEGORICAL_COLS,
    GROUP_COLS,
    add_derived_features,
)
from projections.usage_shares_v1.metrics import compute_baseline_log_weights

app = typer.Typer(add_completion=False, help=__doc__)


# Starterless feature set (removes is_starter and its interactions)
NUMERIC_COLS_NO_STARTER = [
    "minutes_pred_p50",
    "minutes_pred_play_prob",
    "minutes_pred_p50_team_scaled",
    "minutes_pred_team_sum_invalid",
    "minutes_pred_team_rank",
    "position_flags_PG",
    "position_flags_SG",
    "position_flags_SF",
    "position_flags_PF",
    "position_flags_C",
    "spread_close",
    "total_close",
    "team_itt",
    "opp_itt",
    "has_odds",
    "odds_lead_time_minutes",
    "vac_min_szn",
    "vac_fga_szn",
    "vac_min_guard_szn",
    "vac_min_wing_szn",
    "vac_min_big_szn",
    "vac_min_szn_x_minutes_rank",
    "season_fga_per_min",
    "season_fta_per_min",
    "season_tov_per_min",
]


def load_training_data(
    data_root: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load usage_shares_training_base partitions."""
    root = data_root / "gold" / "usage_shares_training_base"
    frames = []
    for season_dir in root.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day_str = day_dir.name.split("=", 1)[1]
                day = pd.Timestamp(day_str).normalize()
            except (ValueError, IndexError):
                continue
            if day < start_date or day > end_date:
                continue
            path = day_dir / "usage_shares_training_base.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


def train_residual_model(
    train_df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    alpha: float,
    seed: int,
) -> tuple[lgb.LGBMRegressor, float]:
    """Train residual model with shrink grid search, return (model, best_shrink)."""
    share_col = f"share_{target}"
    valid_col = f"share_{target}_valid"
    
    # Filter valid
    valid_mask = train_df[valid_col].fillna(True) & train_df[share_col].notna()
    train_df = train_df[valid_mask].copy()
    train_df = train_df[np.isfinite(train_df[target])].copy()
    
    y_true = np.log(train_df[target].values + alpha)
    y_baseline = compute_baseline_log_weights(train_df, target, alpha)
    y_residual = y_true - y_baseline
    
    X = train_df[feature_cols].copy()
    for col in feature_cols:
        if col not in CATEGORICAL_COLS:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
        else:
            X[col] = X[col].fillna(-1).astype(int)
    
    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 0.5,
        "reg_alpha": 0.1,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y_residual)
    
    # Shrink grid search is done at eval time; return model and default shrink
    return model, 0.5


def predict_shares(
    df: pd.DataFrame,
    model: lgb.LGBMRegressor | None,
    target: str,
    feature_cols: list[str],
    alpha: float,
    shrink: float,
) -> np.ndarray:
    """Predict shares for a dataframe."""
    baseline_logw = compute_baseline_log_weights(df, target, alpha)
    
    if model is None:
        # Baseline only
        logw = baseline_logw
    else:
        X = df[feature_cols].copy()
        for col in feature_cols:
            if col not in CATEGORICAL_COLS:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
            else:
                X[col] = X[col].fillna(-1).astype(int)
        delta_pred = model.predict(X)
        logw = baseline_logw + shrink * delta_pred
    
    # Convert to shares per team-game
    df = df.copy()
    df["logw"] = logw
    df["weight"] = np.exp(logw)
    group_sum = df.groupby(GROUP_COLS)["weight"].transform("sum")
    shares = df["weight"] / group_sum
    return shares.values


def compute_team_game_metrics(
    df: pd.DataFrame,
    shares_pred: np.ndarray,
    target: str,
) -> pd.DataFrame:
    """Compute per team-game metrics (MAE, KL)."""
    share_col = f"share_{target}"
    eps = 1e-9
    
    working = df[GROUP_COLS + [share_col, "vac_min_szn"]].copy()
    working["pred"] = shares_pred
    working["true"] = working[share_col]
    working["mae"] = np.abs(working["pred"] - working["true"])
    working["kl_contrib"] = working["true"] * np.log((working["true"] + eps) / (working["pred"] + eps))
    
    group_metrics = working.groupby(GROUP_COLS).agg(
        mae=("mae", "mean"),
        kl=("kl_contrib", "sum"),
        vac_min_szn=("vac_min_szn", "first"),
    ).reset_index()
    
    return group_metrics


def bootstrap_significance(
    metrics_baseline: pd.DataFrame,
    metrics_model: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 1337,
) -> dict[str, Any]:
    """Bootstrap significance test for MAE and KL improvement."""
    rng = np.random.default_rng(seed)
    
    n_games = len(metrics_baseline)
    
    mae_delta = metrics_baseline["mae"].values - metrics_model["mae"].values
    kl_delta = metrics_baseline["kl"].values - metrics_model["kl"].values
    
    mae_boot_means = []
    kl_boot_means = []
    
    for _ in range(n_bootstrap):
        idx = rng.choice(n_games, size=n_games, replace=True)
        mae_boot_means.append(mae_delta[idx].mean())
        kl_boot_means.append(kl_delta[idx].mean())
    
    mae_boot = np.array(mae_boot_means)
    kl_boot = np.array(kl_boot_means)
    
    return {
        "mae_mean_delta": float(mae_delta.mean()),
        "mae_ci_95": (float(np.percentile(mae_boot, 2.5)), float(np.percentile(mae_boot, 97.5))),
        "mae_pct_positive": float((mae_boot > 0).mean() * 100),
        "kl_mean_delta": float(kl_delta.mean()),
        "kl_ci_95": (float(np.percentile(kl_boot, 2.5)), float(np.percentile(kl_boot, 97.5))),
        "kl_pct_positive": float((kl_boot > 0).mean() * 100),
        "n_games": n_games,
        "n_bootstrap": n_bootstrap,
    }


def sanity_metrics(
    df: pd.DataFrame,
    shares_pred: np.ndarray,
    shares_baseline: np.ndarray,
    target: str,
) -> dict[str, Any]:
    """Compute reallocation sanity metrics for high-vacancy games."""
    share_col = f"share_{target}"
    
    working = df[GROUP_COLS + [share_col, "vac_min_szn", "minutes_pred_p50"]].copy()
    working["pred"] = shares_pred
    working["baseline"] = shares_baseline
    working["true"] = working[share_col]
    
    # Compute team rank
    working["rank"] = working.groupby(GROUP_COLS)["minutes_pred_p50"].rank(ascending=False, method="min")
    
    # High-vacancy games (top decile)
    group_vac = working.groupby(GROUP_COLS)["vac_min_szn"].first()
    threshold_q90 = group_vac.quantile(0.9)
    threshold_q75 = group_vac.quantile(0.75)
    
    high_vac_groups_q90 = set(group_vac[group_vac >= threshold_q90].index)
    high_vac_groups_q75 = set(group_vac[group_vac >= threshold_q75].index)
    
    results = {}
    
    for name, groups in [("top_decile", high_vac_groups_q90), ("top_quartile", high_vac_groups_q75)]:
        working["group_key"] = list(zip(working["game_id"], working["team_id"]))
        subset = working[working["group_key"].isin(groups)].copy()
        
        if len(subset) == 0:
            continue
        
        abs_is_top2 = []
        abs_ranks = []
        delta_is_top2 = []
        delta_ranks = []
        
        for _, group in subset.groupby(GROUP_COLS):
            abs_top_idx = group["pred"].idxmax()
            abs_rank = group.loc[abs_top_idx, "rank"]
            abs_is_top2.append(abs_rank <= 2)
            abs_ranks.append(abs_rank)
            
            group["delta"] = group["pred"] - group["baseline"]
            delta_top_idx = group["delta"].idxmax()
            delta_rank = group.loc[delta_top_idx, "rank"]
            delta_is_top2.append(delta_rank <= 2)
            delta_ranks.append(delta_rank)
        
        results[name] = {
            "n_games": len(abs_is_top2),
            "abs_pct_top2_min": float(np.mean(abs_is_top2) * 100),
            "abs_avg_rank": float(np.mean(abs_ranks)),
            "delta_pct_top2_min": float(np.mean(delta_is_top2) * 100),
            "delta_avg_rank": float(np.mean(delta_ranks)),
        }
    
    return results


def worst_examples(
    df: pd.DataFrame,
    shares_pred: np.ndarray,
    shares_baseline: np.ndarray,
    target: str,
    n_examples: int = 10,
) -> list[dict[str, Any]]:
    """Find worst team-games by KL in high-vacancy subset."""
    share_col = f"share_{target}"
    eps = 1e-9
    
    working = df.copy()
    working["pred"] = shares_pred
    working["baseline"] = shares_baseline
    working["true"] = working[share_col]
    
    # Compute KL per team-game
    working["kl_contrib"] = working["true"] * np.log((working["true"] + eps) / (working["pred"] + eps))
    group_kl = working.groupby(GROUP_COLS)["kl_contrib"].sum()
    
    # Get high-vacancy games (top quartile)
    group_vac = working.groupby(GROUP_COLS)["vac_min_szn"].first()
    threshold = group_vac.quantile(0.75)
    high_vac_groups = set(group_vac[group_vac >= threshold].index)
    
    high_vac_kl = group_kl[group_kl.index.isin(high_vac_groups)]
    worst = high_vac_kl.nlargest(n_examples)
    
    examples = []
    for (game_id, team_id), kl in worst.items():
        group = working[(working["game_id"] == game_id) & (working["team_id"] == team_id)].copy()
        
        ex = {
            "game_id": int(game_id),
            "team_id": int(team_id),
            "game_date": str(group["game_date"].iloc[0].date()),
            "kl": round(float(kl), 4),
            "vac_min_szn": round(float(group["vac_min_szn"].iloc[0]), 1),
            "vac_fga_szn": round(float(group["vac_fga_szn"].iloc[0]), 1) if "vac_fga_szn" in group.columns else None,
        }
        
        # Top 6 players by true share
        top_players = group.nlargest(6, "true")
        players = []
        for _, row in top_players.iterrows():
            players.append({
                "minutes_pred": round(float(row.get("minutes_pred_p50", 0)), 1),
                "season_fga_per_min": round(float(row.get("season_fga_per_min", 0)), 3),
                "baseline": round(float(row["baseline"]), 4),
                "pred": round(float(row["pred"]), 4),
                "true": round(float(row["true"]), 4),
            })
        ex["players"] = players
        examples.append(ex)
    
    return examples


@app.command()
def main(
    data_root: Path = typer.Option(None),
    start_date: str = typer.Option("2024-10-22"),
    end_date: str = typer.Option("2025-11-28"),
    val_days: int = typer.Option(30),
    seed: int = typer.Option(1337),
    n_bootstrap: int = typer.Option(1000),
) -> None:
    """Generate decision report for usage shares FGA residual model."""
    
    np.random.seed(seed)
    root = data_root or data_path()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    
    run_id = f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target = "fga"
    alpha = 0.5
    
    typer.echo(f"[decision] Loading data from {start.date()} to {end.date()}...")
    df = load_training_data(root, start, end)
    typer.echo(f"[decision] Loaded {len(df):,} rows")
    
    df = add_derived_features(df)
    
    # Filter valid
    share_col = f"share_{target}"
    valid_col = f"share_{target}_valid"
    valid_mask = df[valid_col].fillna(True) & df[share_col].notna()
    df = df[valid_mask].copy()
    df = df[np.isfinite(df[target])].copy()
    
    # Split
    unique_dates = sorted(df["game_date"].unique())
    val_start = unique_dates[-val_days]
    train_df = df[df["game_date"] < val_start].copy()
    val_df = df[df["game_date"] >= val_start].copy()
    
    typer.echo(f"[decision] Train: {len(train_df):,} rows, Val: {len(val_df):,} rows")
    
    # Get available starterless features
    feature_cols = [c for c in NUMERIC_COLS_NO_STARTER if c in train_df.columns]
    typer.echo(f"[decision] Using {len(feature_cols)} starterless features")
    
    # Train starterless residual model
    typer.echo("[decision] Training starterless residual model...")
    model, _ = train_residual_model(train_df, target, feature_cols, alpha, seed)
    
    # Evaluate at different shrink values
    best_shrink = 0.5
    best_mae = float("inf")
    shrink_grid = [0.25, 0.5, 0.75, 1.0]
    
    for s in shrink_grid:
        shares_pred = predict_shares(val_df, model, target, feature_cols, alpha, s)
        metrics = compute_team_game_metrics(val_df, shares_pred, target)
        mae = metrics["mae"].mean()
        if mae < best_mae:
            best_mae = mae
            best_shrink = s
        typer.echo(f"  shrink={s}: MAE={mae:.5f}")
    
    typer.echo(f"[decision] Best shrink: {best_shrink}")
    
    # === COMPUTE ALL METRICS ===
    
    # Baseline
    shares_baseline = predict_shares(val_df, None, target, feature_cols, alpha, 0)
    metrics_baseline = compute_team_game_metrics(val_df, shares_baseline, target)
    
    # Starterless residual with best shrink
    shares_residual = predict_shares(val_df, model, target, feature_cols, alpha, best_shrink)
    metrics_residual = compute_team_game_metrics(val_df, shares_residual, target)
    
    # Global metrics
    baseline_global = {
        "mae": float(metrics_baseline["mae"].mean()),
        "kl": float(metrics_baseline["kl"].mean()),
    }
    residual_global = {
        "mae": float(metrics_residual["mae"].mean()),
        "kl": float(metrics_residual["kl"].mean()),
    }
    
    typer.echo("\n=== GLOBAL METRICS (Val) ===")
    typer.echo(f"Baseline: MAE={baseline_global['mae']:.5f} KL={baseline_global['kl']:.5f}")
    typer.echo(f"Residual: MAE={residual_global['mae']:.5f} KL={residual_global['kl']:.5f}")
    improvement = (1 - residual_global["mae"] / baseline_global["mae"]) * 100
    typer.echo(f"MAE Improvement: {improvement:+.2f}%")
    
    # Bootstrap significance
    typer.echo(f"\n[decision] Running bootstrap significance test ({n_bootstrap} samples)...")
    bootstrap_overall = bootstrap_significance(metrics_baseline, metrics_residual, n_bootstrap, seed)
    
    typer.echo("\n=== BOOTSTRAP SIGNIFICANCE (Overall) ===")
    typer.echo(f"MAE Delta: {bootstrap_overall['mae_mean_delta']*1000:.3f}e-3 (baseline - model, +ve = model better)")
    typer.echo(f"MAE 95% CI: [{bootstrap_overall['mae_ci_95'][0]*1000:.3f}e-3, {bootstrap_overall['mae_ci_95'][1]*1000:.3f}e-3]")
    typer.echo(f"MAE % positive (model wins): {bootstrap_overall['mae_pct_positive']:.1f}%")
    typer.echo(f"KL Delta: {bootstrap_overall['kl_mean_delta']:.4f}")
    typer.echo(f"KL 95% CI: [{bootstrap_overall['kl_ci_95'][0]:.4f}, {bootstrap_overall['kl_ci_95'][1]:.4f}]")
    typer.echo(f"KL % positive: {bootstrap_overall['kl_pct_positive']:.1f}%")
    
    # High-vacancy bootstrap
    vac_threshold = metrics_baseline["vac_min_szn"].quantile(0.9)
    high_vac_mask = metrics_baseline["vac_min_szn"] >= vac_threshold
    
    if high_vac_mask.sum() >= 20:
        bootstrap_high_vac = bootstrap_significance(
            metrics_baseline[high_vac_mask].reset_index(drop=True),
            metrics_residual[high_vac_mask].reset_index(drop=True),
            n_bootstrap, seed
        )
        typer.echo("\n=== BOOTSTRAP SIGNIFICANCE (High-Vacancy Top Decile) ===")
        typer.echo(f"MAE Delta: {bootstrap_high_vac['mae_mean_delta']*1000:.3f}e-3")
        typer.echo(f"MAE 95% CI: [{bootstrap_high_vac['mae_ci_95'][0]*1000:.3f}e-3, {bootstrap_high_vac['mae_ci_95'][1]*1000:.3f}e-3]")
        typer.echo(f"MAE % positive: {bootstrap_high_vac['mae_pct_positive']:.1f}%")
    else:
        bootstrap_high_vac = None
        typer.echo("\n[decision] Insufficient high-vacancy games for bootstrap")
    
    # Sanity metrics
    sanity = sanity_metrics(val_df, shares_residual, shares_baseline, target)
    typer.echo("\n=== REALLOCATION SANITY (High-Vacancy) ===")
    for name, m in sanity.items():
        typer.echo(f"\n{name} ({m['n_games']} games):")
        typer.echo(f"  ABS (top share): {m['abs_pct_top2_min']:.1f}% in top2 min, avg rank={m['abs_avg_rank']:.1f}")
        typer.echo(f"  DELTA (biggest gain): {m['delta_pct_top2_min']:.1f}% in top2 min, avg rank={m['delta_avg_rank']:.1f}")
    
    # Worst examples
    typer.echo("\n=== WORST EXAMPLES (Top 10 High-Vacancy by KL) ===")
    worst = worst_examples(val_df, shares_residual, shares_baseline, target, 10)
    for ex in worst[:5]:
        typer.echo(f"\ngame={ex['game_id']} team={ex['team_id']} date={ex['game_date']} KL={ex['kl']:.4f} vac={ex['vac_min_szn']:.0f}")
        typer.echo("  Top players: mins | season_fga_pm | baseline | pred | true")
        for p in ex["players"][:4]:
            typer.echo(f"    {p['minutes_pred']:5.1f} | {p['season_fga_per_min']:.3f} | {p['baseline']:.4f} | {p['pred']:.4f} | {p['true']:.4f}")
    
    # Save results
    out_dir = root / "artifacts" / "usage_shares_v1" / "decision" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "run_id": run_id,
        "target": target,
        "alpha": alpha,
        "best_shrink": best_shrink,
        "feature_cols": feature_cols,
        "val_dates": [str(unique_dates[-val_days].date()), str(unique_dates[-1].date())],
        "global_metrics": {
            "baseline": baseline_global,
            "residual": residual_global,
            "improvement_pct": improvement,
        },
        "bootstrap_overall": bootstrap_overall,
        "bootstrap_high_vac": bootstrap_high_vac,
        "sanity": sanity,
        "worst_examples": worst,
        "created_at": datetime.now().isoformat(),
    }
    
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    
    # Save model
    model.booster_.save_model(str(out_dir / "model_fga_starterless.txt"))
    
    typer.echo(f"\n[decision] Results saved to {out_dir}")
    
    # DECISION
    typer.echo("\n" + "=" * 80)
    typer.echo("DECISION SUMMARY")
    typer.echo("=" * 80)
    
    gates_passed = 0
    gates_total = 4
    
    # Gate 1: Overall improvement
    gate1 = improvement > 0
    typer.echo(f"\n1. Overall MAE improvement: {improvement:+.2f}% {'✓' if gate1 else '✗'}")
    if gate1:
        gates_passed += 1
    
    # Gate 2: Bootstrap significance (>80% positive)
    gate2 = bootstrap_overall["mae_pct_positive"] > 80
    typer.echo(f"2. Bootstrap significance: {bootstrap_overall['mae_pct_positive']:.1f}% positive {'✓' if gate2 else '✗'}")
    if gate2:
        gates_passed += 1
    
    # Gate 3: High-vacancy improvement (if available)
    if bootstrap_high_vac:
        gate3 = bootstrap_high_vac["mae_pct_positive"] > 50
        typer.echo(f"3. High-vacancy bootstrap: {bootstrap_high_vac['mae_pct_positive']:.1f}% positive {'✓' if gate3 else '✗'}")
        if gate3:
            gates_passed += 1
    else:
        gate3 = False
        typer.echo("3. High-vacancy bootstrap: N/A (insufficient data)")
    
    # Gate 4: Sanity check (ABS top share goes to high-min players)
    top_dec_sanity = sanity.get("top_decile", {})
    gate4 = top_dec_sanity.get("abs_pct_top2_min", 0) > 70
    typer.echo(f"4. Sanity (ABS top2 min): {top_dec_sanity.get('abs_pct_top2_min', 0):.1f}% {'✓' if gate4 else '✗'}")
    if gate4:
        gates_passed += 1
    
    typer.echo(f"\nGates passed: {gates_passed}/{gates_total}")
    
    if gates_passed >= 3:
        typer.echo("\n🟢 RECOMMENDATION: WIRE FGA behind flag")
        typer.echo(f"   Best shrink: {best_shrink}")
        typer.echo("   Model: starterless (until is_starter fix lands)")
    else:
        typer.echo("\n🔴 RECOMMENDATION: DO NOT WIRE")
        typer.echo("   Improvements are not statistically significant")


if __name__ == "__main__":
    app()
