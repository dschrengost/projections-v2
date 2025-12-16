"""
Vacancy Sanity Diagnostic

Analyzes whether vacancy v1 computed in sim is inflated due to bench scrubs
with low play_prob contributing disproportionately.

Usage:
    uv run python -m scripts.diagnostics.vacancy_sanity --date 2025-12-14

Output:
    - Per-team vacancy breakdown
    - Comparison of vac_all vs vac_rot (rotation-aware)
    - Top contributors to vacancy
    - Training vs live distribution comparison
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from projections.paths import data_path


def load_minutes_df(data_root: Path, date_str: str) -> pd.DataFrame:
    """Load minutes projection for a single date."""
    date_path = data_root / "gold" / "projections_minutes_v1" / f"game_date={date_str}"
    parquet_path = date_path / "minutes.parquet"
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Minutes not found at {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    return df


def load_rates_df(data_root: Path, date_str: str) -> pd.DataFrame:
    """Load rates for a single date (for season_fga_per_min proxy)."""
    base = data_root / "gold" / "rates_v1_live" / date_str
    
    # Try to find latest run
    if not base.exists():
        return pd.DataFrame()
    
    runs = sorted(base.glob("run=*"))
    if runs:
        parquet_path = runs[-1] / "rates.parquet"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
    
    # Fallback to direct path
    parquet_path = base / "rates.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    
    return pd.DataFrame()


def load_training_vacancy_distribution(data_root: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """Load vacancy distribution from training data for comparison."""
    base = data_root / "gold" / "usage_shares_training_base"
    
    if not base.exists():
        return pd.DataFrame()
    
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    
    frames = []
    for season_dir in base.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day_str = day_dir.name.split("=", 1)[1]
                day_ts = pd.Timestamp(day_str)
            except (ValueError, IndexError):
                continue
            
            if day_ts < start_ts or day_ts > end_ts:
                continue
            
            parquet_path = day_dir / "usage_shares_training_base.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                frames.append(df)
    
    if not frames:
        return pd.DataFrame()
    
    return pd.concat(frames, ignore_index=True)


def compute_vacancy_stats(
    df: pd.DataFrame,
    rotation_floor: float = 6.0,
    rotation_topk: int = 10,
) -> pd.DataFrame:
    """Compute per-team vacancy stats for both all_players and rotation_candidates modes."""
    
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
        raise ValueError("No minutes column found")
    
    # Prepare data
    df = df.copy()
    df["_minutes"] = pd.to_numeric(df[minutes_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    
    if prob_col:
        df["_play_prob"] = pd.to_numeric(df[prob_col], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    else:
        df["_play_prob"] = 1.0
    
    df["_vac_minutes"] = (1.0 - df["_play_prob"]) * df["_minutes"]
    df["_played_minutes"] = df["_play_prob"] * df["_minutes"]
    
    # Compute rank within team
    df["_minutes_rank"] = df.groupby(["game_id", "team_id"])["_minutes"].rank(ascending=False, method="min")
    
    # Rotation candidate mask
    df["_is_rotation"] = (df["_minutes"] >= rotation_floor) | (df["_minutes_rank"] <= rotation_topk)
    
    # Per-team aggregations
    team_stats = df.groupby(["game_id", "team_id"]).agg(
        sum_m=("_minutes", "sum"),
        sum_pm=("_played_minutes", "sum"),
        vac_all=("_vac_minutes", "sum"),
        n_total=("_minutes", "count"),
        n_playprob_lt_02=("_play_prob", lambda x: (x < 0.2).sum()),
        n_m_ge_floor=("_minutes", lambda x: (x >= rotation_floor).sum()),
        n_rotation=("_is_rotation", "sum"),
    ).reset_index()
    
    # Compute rotation-only vacancy
    rotation_agg = df[df["_is_rotation"]].groupby(["game_id", "team_id"]).agg(
        sum_m_rot=("_minutes", "sum"),
        sum_pm_rot=("_played_minutes", "sum"),
        vac_rot=("_vac_minutes", "sum"),
    ).reset_index()
    
    team_stats = team_stats.merge(rotation_agg, on=["game_id", "team_id"], how="left")
    team_stats[["sum_m_rot", "sum_pm_rot", "vac_rot"]] = team_stats[["sum_m_rot", "sum_pm_rot", "vac_rot"]].fillna(0.0)
    
    # Identity check: sum_m should equal sum_pm + vac_all
    team_stats["identity_check"] = team_stats["sum_m"] - (team_stats["sum_pm"] + team_stats["vac_all"])
    
    return team_stats


def get_top_vacancy_contributors(
    df: pd.DataFrame,
    team_id: int,
    game_id: int,
    top_n: int = 8,
) -> pd.DataFrame:
    """Get top contributors to vacancy for a team."""
    
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
    
    team_df = df[(df["team_id"] == team_id) & (df["game_id"] == game_id)].copy()
    
    if team_df.empty:
        return pd.DataFrame()
    
    team_df["_minutes"] = pd.to_numeric(team_df[minutes_col], errors="coerce").fillna(0.0)
    team_df["_play_prob"] = pd.to_numeric(team_df.get(prob_col, pd.Series(1.0)), errors="coerce").fillna(1.0)
    team_df["_vac_minutes"] = (1.0 - team_df["_play_prob"]) * team_df["_minutes"]
    team_df["_minutes_rank"] = team_df["_minutes"].rank(ascending=False, method="min")
    
    # Get player name if available
    name_col = "player_name" if "player_name" in team_df.columns else None
    
    cols = ["player_id", "_minutes", "_play_prob", "_vac_minutes", "_minutes_rank"]
    if name_col:
        cols.insert(1, name_col)
    
    return team_df.nlargest(top_n, "_vac_minutes")[cols].rename(columns={
        "_minutes": "minutes_pred_p50",
        "_play_prob": "play_prob",
        "_vac_minutes": "vac_contribution",
        "_minutes_rank": "minutes_rank",
    })


def print_quantiles(series: pd.Series, label: str) -> str:
    """Format quantiles as a string."""
    if series.empty:
        return f"{label}: (no data)"
    return f"{label}: p05={series.quantile(0.05):.1f} p50={series.median():.1f} p90={series.quantile(0.90):.1f} max={series.max():.1f}"


def generate_report(
    date_str: str,
    team_stats: pd.DataFrame,
    minutes_df: pd.DataFrame,
    training_vac: pd.DataFrame | None,
    output_dir: Path,
    rotation_floor: float,
    rotation_topk: int,
) -> str:
    """Generate markdown report."""
    
    lines = []
    lines.append(f"# Vacancy Sanity Report: {date_str}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Rotation floor: {rotation_floor} min, top-k: {rotation_topk}")
    lines.append("")
    
    # Summary quantiles
    lines.append("## Summary Quantiles")
    lines.append("")
    lines.append("| Metric | p05 | p50 | p90 | max |")
    lines.append("|--------|-----|-----|-----|-----|")
    
    for col, label in [
        ("vac_all", "vac_all (all players)"),
        ("vac_rot", "vac_rot (rotation only)"),
        ("sum_m", "sum_m (total minutes)"),
        ("sum_pm", "sum_pm (expected played)"),
    ]:
        q05 = team_stats[col].quantile(0.05)
        q50 = team_stats[col].median()
        q90 = team_stats[col].quantile(0.90)
        mx = team_stats[col].max()
        lines.append(f"| {label} | {q05:.1f} | {q50:.1f} | {q90:.1f} | {mx:.1f} |")
    
    lines.append("")
    
    # Vacancy comparison
    lines.append("## Vacancy Comparison: All vs Rotation")
    lines.append("")
    vac_diff = team_stats["vac_all"] - team_stats["vac_rot"]
    lines.append(f"- **Mean vac_all**: {team_stats['vac_all'].mean():.1f}")
    lines.append(f"- **Mean vac_rot**: {team_stats['vac_rot'].mean():.1f}")
    lines.append(f"- **Mean difference (all - rot)**: {vac_diff.mean():.1f}")
    lines.append(f"- **Median difference**: {vac_diff.median():.1f}")
    lines.append(f"- **Max difference**: {vac_diff.max():.1f}")
    lines.append("")
    
    if vac_diff.mean() > 20:
        lines.append("> ⚠️ **INFLATION DETECTED**: vac_all significantly exceeds vac_rot. Bench scrubs are inflating vacancy.")
    else:
        lines.append("> ✅ Vacancy appears reasonable. Bench contribution is modest.")
    lines.append("")
    
    # Top 10 teams by vac_all
    lines.append("## Top 10 Teams by vac_all")
    lines.append("")
    lines.append("| game_id | team_id | vac_all | vac_rot | diff | n_total | n_rot | n_pp<0.2 |")
    lines.append("|---------|---------|---------|---------|------|---------|-------|----------|")
    
    top_teams = team_stats.nlargest(10, "vac_all")
    for _, row in top_teams.iterrows():
        diff = row["vac_all"] - row["vac_rot"]
        lines.append(
            f"| {int(row['game_id'])} | {int(row['team_id'])} | {row['vac_all']:.1f} | "
            f"{row['vac_rot']:.1f} | {diff:.1f} | {int(row['n_total'])} | "
            f"{int(row['n_rotation'])} | {int(row['n_playprob_lt_02'])} |"
        )
    
    lines.append("")
    
    # Top contributors for highest vac_all team
    lines.append("## Top Vacancy Contributors (Highest vac_all Team)")
    lines.append("")
    
    top_team = top_teams.iloc[0]
    contributors = get_top_vacancy_contributors(
        minutes_df, int(top_team["team_id"]), int(top_team["game_id"])
    )
    
    if not contributors.empty:
        lines.append("| player_id | minutes_pred | play_prob | vac_contrib | rank |")
        lines.append("|-----------|--------------|-----------|-------------|------|")
        for _, row in contributors.iterrows():
            lines.append(
                f"| {int(row['player_id'])} | {row['minutes_pred_p50']:.1f} | "
                f"{row['play_prob']:.2f} | {row['vac_contribution']:.1f} | "
                f"{int(row['minutes_rank'])} |"
            )
    lines.append("")
    
    # Training comparison
    if training_vac is not None and not training_vac.empty and "vac_min_szn" in training_vac.columns:
        lines.append("## Training vs Live Comparison")
        lines.append("")
        lines.append("| Feature | Live p50 | Live p90 | Train p50 | Train p90 |")
        lines.append("|---------|----------|----------|-----------|-----------|")
        
        # Aggregate training to team-game level
        train_agg = training_vac.groupby(["game_id", "team_id"])["vac_min_szn"].first().reset_index()
        
        live_p50 = team_stats["vac_all"].median()
        live_p90 = team_stats["vac_all"].quantile(0.90)
        train_p50 = train_agg["vac_min_szn"].median()
        train_p90 = train_agg["vac_min_szn"].quantile(0.90)
        
        lines.append(f"| vac_min_szn | {live_p50:.1f} | {live_p90:.1f} | {train_p50:.1f} | {train_p90:.1f} |")
        lines.append("")
        
        ratio = live_p50 / max(train_p50, 0.01)
        if ratio > 2.0:
            lines.append(f"> ⚠️ **SCALE MISMATCH**: Live vacancy is {ratio:.1f}x training. Feature scaling issue likely.")
        elif ratio < 0.5:
            lines.append(f"> ⚠️ **SCALE MISMATCH**: Live vacancy is {ratio:.1f}x training. May be under-computing.")
        else:
            lines.append(f"> ✅ Scale roughly matches training ({ratio:.1f}x)")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    
    vac_ratio = team_stats["vac_rot"].mean() / max(team_stats["vac_all"].mean(), 0.01)
    if vac_ratio < 0.3:
        lines.append("**Implement vacancy v1.1 (rotation-aware)**: vac_rot is significantly smaller than vac_all.")
        lines.append("Bench scrubs with low play_prob are dominating vacancy computation.")
    else:
        lines.append("**Keep vacancy v1 (all players)**: Rotation players contribute most of the vacancy.")
    
    report = "\n".join(lines)
    
    # Write to file
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"
    report_path.write_text(report)
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Vacancy Sanity Diagnostic")
    parser.add_argument("--data-root", type=Path, default=Path("/home/daniel/projections-data"))
    parser.add_argument("--date", type=str, required=True, help="Date in YYYY-MM-DD format")
    parser.add_argument("--profile", type=str, default="baseline")
    parser.add_argument("--min-play-prob", type=float, default=0.05)
    parser.add_argument("--rotation-floor-candidates", type=float, default=6.0)
    parser.add_argument("--rotation-topk", type=int, default=10)
    parser.add_argument("--training-start", type=str, default="2025-10-29")
    parser.add_argument("--training-end", type=str, default="2025-11-28")
    parser.add_argument("--seed", type=int, default=123)
    
    args = parser.parse_args()
    
    print(f"[vacancy_sanity] Loading data for {args.date}...")
    
    # Load minutes
    minutes_df = load_minutes_df(args.data_root, args.date)
    print(f"[vacancy_sanity] Loaded {len(minutes_df)} rows from minutes_df")
    
    # Compute vacancy stats
    team_stats = compute_vacancy_stats(
        minutes_df,
        rotation_floor=args.rotation_floor_candidates,
        rotation_topk=args.rotation_topk,
    )
    print(f"[vacancy_sanity] Computed stats for {len(team_stats)} teams")
    
    # Print summary
    print()
    print("=" * 60)
    print("VACANCY SUMMARY")
    print("=" * 60)
    print()
    print(print_quantiles(team_stats["vac_all"], "vac_all (all players)"))
    print(print_quantiles(team_stats["vac_rot"], "vac_rot (rotation)"))
    print(print_quantiles(team_stats["sum_m"], "sum_m (total minutes)"))
    print(print_quantiles(team_stats["sum_pm"], "sum_pm (expected played)"))
    print()
    
    vac_diff = team_stats["vac_all"] - team_stats["vac_rot"]
    print(f"Vacancy inflation (all - rot): mean={vac_diff.mean():.1f} median={vac_diff.median():.1f} max={vac_diff.max():.1f}")
    print()
    
    # Identity check
    max_identity_err = team_stats["identity_check"].abs().max()
    print(f"Identity check (sum_m - sum_pm - vac_all): max error = {max_identity_err:.6f}")
    if max_identity_err > 0.01:
        print("  ⚠️ Identity check failed!")
    else:
        print("  ✅ Identity check passed")
    print()
    
    # Load training data
    training_vac = None
    try:
        training_vac = load_training_vacancy_distribution(
            args.data_root, args.training_start, args.training_end
        )
        if not training_vac.empty:
            print(f"[vacancy_sanity] Loaded {len(training_vac)} rows from training data")
    except Exception as e:
        print(f"[vacancy_sanity] Could not load training data: {e}")
    
    # Generate report
    output_dir = args.data_root / "artifacts" / "diagnostics" / "vacancy_sanity" / f"date={args.date}"
    report = generate_report(
        args.date,
        team_stats,
        minutes_df,
        training_vac,
        output_dir,
        args.rotation_floor_candidates,
        args.rotation_topk,
    )
    
    print()
    print(f"Report written to: {output_dir / 'report.md'}")
    print()
    print("=" * 60)
    print("REPORT PREVIEW")
    print("=" * 60)
    print(report[:2000])
    if len(report) > 2000:
        print("...")
        print(f"(truncated, full report at {output_dir / 'report.md'})")


if __name__ == "__main__":
    main()
