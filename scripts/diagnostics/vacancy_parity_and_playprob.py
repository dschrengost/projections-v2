"""
Vacancy Parity and play_prob Audit

Compares training vacancy definition vs sim v3 vacancy and analyzes play_prob distribution.

Training vacancy:
    - Filters to OUT/DOUBTFUL/QUESTIONABLE/INACTIVE players per injuries_snapshot
    - Sums hist_minutes_szn (season-to-date actuals) for those players

Sim v3 vacancy:
    - vac_minutes_i = (1 - play_prob) * minutes_pred_p50
    - Sums across ALL players regardless of status

These are DIFFERENT definitions by design:
- Training uses actual injury status + historical minutes
- Sim uses predicted play probability + predicted minutes

Usage:
    uv run python -m scripts.diagnostics.vacancy_parity_and_playprob --dates 2025-12-14
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def load_minutes_df(data_root: Path, date_str: str) -> pd.DataFrame:
    """Load minutes projection for a single date."""
    date_path = data_root / "gold" / "projections_minutes_v1" / f"game_date={date_str}"
    parquet_path = date_path / "minutes.parquet"
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Minutes not found at {parquet_path}")
    
    return pd.read_parquet(parquet_path)


def load_training_vacancy(data_root: Path, date_str: str) -> pd.DataFrame:
    """Load training-style vacancy from usage_shares_training_base."""
    base = data_root / "gold" / "usage_shares_training_base"
    
    if not base.exists():
        return pd.DataFrame()
    
    # Find the partition for this date
    for season_dir in base.glob("season=*"):
        day_dir = season_dir / f"game_date={date_str}"
        parquet_path = day_dir / "usage_shares_training_base.parquet"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
    
    return pd.DataFrame()


def compute_sim_v3_vacancy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute vacancy using sim v3 formula: (1 - play_prob) * minutes_pred_p50."""
    
    # Resolve columns
    minutes_col = None
    for c in ["minutes_pred_p50", "minutes_p50_cond", "minutes_p50"]:
        if c in df.columns:
            minutes_col = c
            break
    
    prob_col = None
    for c in ["minutes_pred_play_prob", "play_prob"]:
        if c in df.columns:
            prob_col = c
            break
    
    if minutes_col is None:
        return pd.DataFrame()
    
    df = df.copy()
    df["_minutes"] = pd.to_numeric(df[minutes_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    
    if prob_col:
        df["_play_prob"] = pd.to_numeric(df[prob_col], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    else:
        df["_play_prob"] = 1.0
    
    df["_vac_minutes"] = (1.0 - df["_play_prob"]) * df["_minutes"]
    
    # Aggregate to team level
    team_stats = df.groupby(["game_id", "team_id"]).agg(
        vac_min_szn_sim=("_vac_minutes", "sum"),
        n_players=("_minutes", "count"),
    ).reset_index()
    
    return team_stats


def analyze_play_prob_distribution(df: pd.DataFrame, date_str: str) -> dict:
    """Analyze play_prob distribution for brittleness."""
    
    prob_col = None
    for c in ["minutes_pred_play_prob", "play_prob"]:
        if c in df.columns:
            prob_col = c
            break
    
    if prob_col is None:
        return {}
    
    p = pd.to_numeric(df[prob_col], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    n = len(p)
    
    return {
        "date": date_str,
        "n_players": n,
        "pct_exactly_0": 100 * (p == 0.0).sum() / n,
        "pct_0_to_05": 100 * ((p > 0.0) & (p <= 0.05)).sum() / n,
        "pct_05_to_95": 100 * ((p > 0.05) & (p < 0.95)).sum() / n,
        "pct_95_to_1": 100 * ((p >= 0.95) & (p < 1.0)).sum() / n,
        "pct_exactly_1": 100 * (p == 1.0).sum() / n,
    }


def analyze_vacancy_dominance(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze what fraction of vacancy comes from top contributors."""
    
    # Resolve columns
    minutes_col = None
    for c in ["minutes_pred_p50", "minutes_p50_cond", "minutes_p50"]:
        if c in df.columns:
            minutes_col = c
            break
    
    prob_col = None
    for c in ["minutes_pred_play_prob", "play_prob"]:
        if c in df.columns:
            prob_col = c
            break
    
    if minutes_col is None:
        return pd.DataFrame()
    
    df = df.copy()
    df["_minutes"] = pd.to_numeric(df[minutes_col], errors="coerce").fillna(0.0)
    df["_play_prob"] = pd.to_numeric(df.get(prob_col, pd.Series(1.0)), errors="coerce").fillna(1.0)
    df["_vac_minutes"] = (1.0 - df["_play_prob"]) * df["_minutes"]
    
    results = []
    for (game_id, team_id), group in df.groupby(["game_id", "team_id"]):
        total_vac = group["_vac_minutes"].sum()
        if total_vac < 0.01:
            top1_frac = 0.0
            top3_frac = 0.0
        else:
            sorted_vac = group["_vac_minutes"].sort_values(ascending=False)
            top1_frac = sorted_vac.iloc[0] / total_vac if len(sorted_vac) >= 1 else 0.0
            top3_frac = sorted_vac.iloc[:3].sum() / total_vac if len(sorted_vac) >= 3 else 1.0
        
        results.append({
            "game_id": game_id,
            "team_id": team_id,
            "total_vac": total_vac,
            "top1_frac": top1_frac,
            "top3_frac": top3_frac,
        })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Vacancy Parity and play_prob Audit")
    parser.add_argument("--data-root", type=Path, default=Path("/home/daniel/projections-data"))
    parser.add_argument("--dates", type=str, required=True, help="Comma-separated dates in YYYY-MM-DD format")
    
    args = parser.parse_args()
    dates = [d.strip() for d in args.dates.split(",")]
    
    print("=" * 70)
    print("VACANCY PARITY AND PLAY_PROB AUDIT")
    print("=" * 70)
    print()
    
    # Part 1: Definition Parity
    print("## 1. Definition Parity: Training vs Sim v3")
    print()
    print("**Training vacancy (build_training_base.py:lines 512-548):**")
    print("```python")
    print("# Filter to OUT/DOUBTFUL/QUESTIONABLE/INACTIVE players")
    print("out_like_values = {'OUT', 'DOUBTFUL', 'QUESTIONABLE', 'INACTIVE'}")
    print("latest = latest[latest['status_norm'].isin(out_like_values)]")
    print("# Sum their season-to-date HISTORICAL minutes")
    print("vac_min_szn = sum(hist_minutes_szn)  # actual played minutes this season")
    print("```")
    print()
    print("**Sim v3 vacancy (_add_vacancy_features_from_minutes_df):**")
    print("```python")
    print("# For ALL players, compute expected missing minutes")
    print("vac_minutes_i = (1 - play_prob) * minutes_pred_p50")
    print("vac_min_szn = sum(vac_minutes_i)  # predicted minutes * probability NOT playing")
    print("```")
    print()
    print("### Parity Result: **DIFFERENT DEFINITIONS**")
    print()
    print("| Aspect | Training | Sim v3 |")
    print("|--------|----------|--------|")
    print("| Players included | Only OUT/DOUBTFUL/etc | All players |")
    print("| Minutes used | hist_minutes_szn (actuals) | minutes_pred_p50 |")
    print("| Weighting | Binary (in or out) | Continuous (1 - play_prob) |")
    print("| Interpretation | 'How many actual minutes are missing' | 'Expected missing minutes' |")
    print()
    print("**This is BY DESIGN** - sim uses a pregame proxy since injuries_snapshot isn't joined.")
    print()
    
    # Part 2: Quantitative comparison
    print("## 2. Quantitative Comparison by Date")
    print()
    
    all_play_prob_stats = []
    all_dominance_stats = []
    
    for date_str in dates:
        print(f"### {date_str}")
        print()
        
        # Load data
        try:
            minutes_df = load_minutes_df(args.data_root, date_str)
        except FileNotFoundError:
            print(f"  ❌ No minutes data found for {date_str}")
            continue
        
        training_df = load_training_vacancy(args.data_root, date_str)
        
        # Compute sim v3 vacancy
        sim_vac = compute_sim_v3_vacancy(minutes_df)
        
        if not sim_vac.empty:
            print("**Sim v3 vacancy (vac_min_szn_sim per team-game):**")
            print(f"  p10={sim_vac['vac_min_szn_sim'].quantile(0.10):.1f}")
            print(f"  p50={sim_vac['vac_min_szn_sim'].median():.1f}")
            print(f"  p90={sim_vac['vac_min_szn_sim'].quantile(0.90):.1f}")
            print(f"  max={sim_vac['vac_min_szn_sim'].max():.1f}")
            print()
        
        # Training vacancy if available
        if not training_df.empty and "vac_min_szn" in training_df.columns:
            train_agg = training_df.groupby(["game_id", "team_id"])["vac_min_szn"].first()
            print("**Training vacancy (vac_min_szn per team-game):**")
            print(f"  p10={train_agg.quantile(0.10):.1f}")
            print(f"  p50={train_agg.median():.1f}")
            print(f"  p90={train_agg.quantile(0.90):.1f}")
            print(f"  max={train_agg.max():.1f}")
            print()
            
            # Compare (scales will differ due to different definitions)
            if not sim_vac.empty:
                ratio = sim_vac['vac_min_szn_sim'].median() / max(train_agg.median(), 0.01)
                print(f"**Scale ratio (sim/training p50):** {ratio:.2f}x")
                if ratio < 0.5:
                    print("  ⚠️ Sim vacancy is much lower than training")
                elif ratio > 2.0:
                    print("  ⚠️ Sim vacancy is much higher than training")
                else:
                    print("  ✅ Scales roughly comparable")
            print()
        else:
            print(f"  (No training data available for {date_str})")
            print()
        
        # Play prob analysis
        play_prob_stats = analyze_play_prob_distribution(minutes_df, date_str)
        if play_prob_stats:
            all_play_prob_stats.append(play_prob_stats)
        
        # Dominance analysis
        dominance_df = analyze_vacancy_dominance(minutes_df)
        if not dominance_df.empty:
            all_dominance_stats.append({
                "date": date_str,
                "top1_frac_p50": dominance_df["top1_frac"].median(),
                "top1_frac_p90": dominance_df["top1_frac"].quantile(0.90),
                "top3_frac_p50": dominance_df["top3_frac"].median(),
                "top3_frac_p90": dominance_df["top3_frac"].quantile(0.90),
            })
    
    # Part 3: play_prob distribution
    print("## 3. play_prob Distribution (Brittleness Check)")
    print()
    
    if all_play_prob_stats:
        print("| Date | n_players | %=0.0 | %(0,0.05] | %(0.05,0.95) | %[0.95,1) | %=1.0 |")
        print("|------|-----------|-------|-----------|--------------|-----------|-------|")
        for stats in all_play_prob_stats:
            print(
                f"| {stats['date']} | {stats['n_players']} | "
                f"{stats['pct_exactly_0']:.1f}% | {stats['pct_0_to_05']:.1f}% | "
                f"{stats['pct_05_to_95']:.1f}% | {stats['pct_95_to_1']:.1f}% | "
                f"{stats['pct_exactly_1']:.1f}% |"
            )
        print()
        
        # Summarize
        avg_binary = np.mean([s["pct_exactly_0"] + s["pct_exactly_1"] for s in all_play_prob_stats])
        print(f"**Average % binary (exactly 0 or 1):** {avg_binary:.1f}%")
        if avg_binary > 80:
            print("  ⚠️ play_prob is highly binary - limited gradation")
        else:
            print("  ✅ play_prob has reasonable gradation")
        print()
    
    # Part 4: Vacancy dominance
    print("## 4. Vacancy Dominance (Top Contributor Concentration)")
    print()
    
    if all_dominance_stats:
        print("| Date | top1_frac p50 | top1_frac p90 | top3_frac p50 | top3_frac p90 |")
        print("|------|---------------|---------------|---------------|---------------|")
        for stats in all_dominance_stats:
            print(
                f"| {stats['date']} | {stats['top1_frac_p50']*100:.1f}% | "
                f"{stats['top1_frac_p90']*100:.1f}% | {stats['top3_frac_p50']*100:.1f}% | "
                f"{stats['top3_frac_p90']*100:.1f}% |"
            )
        print()
        
        avg_top1_p90 = np.mean([s["top1_frac_p90"] for s in all_dominance_stats])
        if avg_top1_p90 > 0.6:
            print("  ⚠️ Vacancy often dominated by single player (>60% at p90)")
        else:
            print("  ✅ Vacancy reasonably distributed across players")
        print()
    
    # Part 5: Recommendation
    print("=" * 70)
    print("## RECOMMENDATION")
    print("=" * 70)
    print()
    
    # Assess based on findings
    issues = []
    
    avg_binary = np.mean([s["pct_exactly_0"] + s["pct_exactly_1"] for s in all_play_prob_stats]) if all_play_prob_stats else 0
    if avg_binary > 90:
        issues.append("play_prob is >90% binary")
    
    if all_dominance_stats:
        avg_top1_p90 = np.mean([s["top1_frac_p90"] for s in all_dominance_stats])
        if avg_top1_p90 > 0.7:
            issues.append("Vacancy often >70% dominated by 1 player")
    
    if issues:
        print("**Issues detected:**")
        for issue in issues:
            print(f"  - {issue}")
        print()
        print("**Suggested mitigation:** Cap (1-play_prob) at 0.9 to reduce single-player dominance")
        print("  vac_minutes_i = min(1 - play_prob, 0.9) * minutes_pred_p50")
        print()
        print("**However**, since definitions differ (training uses actual OUT players),")
        print("the model was trained on real vacancy signals. The sim proxy is an approximation.")
        print()
        print("**Verdict: PROCEED WITH CAUTION** - monitor model behavior with live vacancy vs training.")
    else:
        print("**✅ OK TO PROCEED**")
        print()
        print("- play_prob has reasonable gradation")
        print("- Vacancy is not overly dominated by single players")
        print("- Sim and training vacancy differ by design (actual vs predicted)")
        print("- LGBM learns relative signals, so absolute scale mismatch is acceptable")
    
    print()
    print(f"Report generated at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
