"""
Sim V3 A/B Backtest Harness

Compares sim_v3 (learned FGA + vacancy), sim_v3_novacancy, and baseline.
Quantifies vacancy incremental effect on FGA share allocation.

Usage:
    uv run python -m scripts.diagnostics.sim_v3_ab_backtest \
        --data-root /home/daniel/projections-data \
        --start-date 2025-12-10 --end-date 2025-12-14 \
        --n-worlds 300 --seed 123 \
        --profiles baseline,sim_v3,sim_v3_novacancy
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def run_sim_for_date(
    data_root: Path,
    date_str: str,
    profile: str,
    n_worlds: int,
    seed: int,
    output_dir: Path,
) -> Path | None:
    """Run sim for a single date/profile. Copies result to unique output_dir."""
    cmd = [
        sys.executable, "-m", "scripts.sim_v2.generate_worlds_fpts_v2",
        "--start-date", date_str,
        "--end-date", date_str,
        "--profile", profile,
        "--n-worlds", str(n_worlds),
        "--seed", str(seed),
        "--export-attempt-means",  # Enable FGA share diagnostics
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(data_root.parent / "projects" / "projections-v2"),
    )

    if result.returncode != 0:
        print(f"  ⚠️ {profile}/{date_str} failed: {result.stderr[:200]}")
        return None

    # Copy to unique profile-specific path
    src_path = data_root / "artifacts" / "sim_v2" / "worlds_fpts_v2" / f"game_date={date_str}" / "projections.parquet"
    if src_path.exists():
        dst_path = output_dir / f"{profile}_{date_str}.parquet"
        shutil.copy(src_path, dst_path)
        return dst_path
    return None


# Valid NBA team_id range
NBA_TEAM_ID_MIN = 1610612737  # Atlanta Hawks
NBA_TEAM_ID_MAX = 1610612766  # Charlotte Hornets


def load_projections(path: Path) -> pd.DataFrame:
    """Load projections parquet with team_id sanitization."""
    if not path or not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    
    # Filter to valid NBA team_ids only
    if "team_id" in df.columns:
        n_before = len(df)
        df = df[df["team_id"].between(NBA_TEAM_ID_MIN, NBA_TEAM_ID_MAX)]
        n_filtered = n_before - len(df)
        if n_filtered > 0:
            print(f"    ⚠️ Filtered {n_filtered} rows with invalid team_id")
    
    return df


def check_key_integrity(df: pd.DataFrame, date_str: str) -> dict:
    """Check team-game key integrity and return metrics."""
    if df.empty or "team_id" not in df.columns or "game_id" not in df.columns:
        return {}
    
    team_ids = df["team_id"].unique()
    game_ids = df["game_id"].unique()
    
    # Teams per game should be exactly 2
    teams_per_game = df.groupby("game_id")["team_id"].nunique()
    bad_games = teams_per_game[teams_per_game != 2]
    
    return {
        "date": date_str,
        "n_distinct_teams": len(team_ids),
        "team_id_min": int(team_ids.min()),
        "team_id_max": int(team_ids.max()),
        "n_games": len(game_ids),
        "teams_valid_range": all(NBA_TEAM_ID_MIN <= t <= NBA_TEAM_ID_MAX for t in team_ids),
        "all_games_have_2_teams": len(bad_games) == 0,
        "bad_game_count": len(bad_games),
    }


def compute_fpts_deltas(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str = "a", label_b: str = "b") -> dict:
    """Compute player-level FPTS deltas between two profiles."""
    if df_a.empty or df_b.empty:
        return {}

    cols_a = ["player_id", "game_id", "team_id", "dk_fpts_mean"]
    cols_b = ["player_id", "dk_fpts_mean"]
    if "vac_min_szn" in df_a.columns:
        cols_a.append("vac_min_szn")
    # FGA columns for allocation analysis
    for c in ["fga2_mean", "fga3_mean"]:
        if c in df_a.columns:
            cols_a.append(c)
        if c in df_b.columns:
            cols_b.append(c)

    merged = df_a[[c for c in cols_a if c in df_a.columns]].merge(
        df_b[[c for c in cols_b if c in df_b.columns]],
        on="player_id",
        suffixes=(f"_{label_a}", f"_{label_b}"),
        how="inner",
    )

    if merged.empty:
        return {}

    fpts_a = f"dk_fpts_mean_{label_a}"
    fpts_b = f"dk_fpts_mean_{label_b}"
    merged["delta"] = merged[fpts_b] - merged[fpts_a]
    merged["abs_delta"] = merged["delta"].abs()

    return {
        "mean_abs_delta": merged["abs_delta"].mean(),
        "max_abs_delta": merged["abs_delta"].max(),
        "n_players": len(merged),
        "merged_df": merged,  # Keep for further analysis
    }


def compute_team_game_diagnostics(df_sim_v3: pd.DataFrame, df_novac: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-team-game diagnostics comparing sim_v3 vs sim_v3_novacancy.
    
    Returns DataFrame with:
    - date, game_id, team_id (keys)
    - vac_min_szn (from sim_v3)
    - mean_abs_delta_fpts, max_abs_delta_fpts (player FPTS deltas within team)
    - l1_fga_share_delta (sum of |share_a - share_b| per team-game)
    - max_player_delta (single biggest player delta)
    - max_player_id (player with biggest delta)
    """
    if df_sim_v3.empty or df_novac.empty:
        return pd.DataFrame()
    
    # Merge player-level data
    cols_v3 = ["game_id", "team_id", "player_id", "dk_fpts_mean"]
    cols_novac = ["player_id", "dk_fpts_mean"]
    
    # Include vac_min_szn from sim_v3
    if "vac_min_szn" in df_sim_v3.columns:
        cols_v3.append("vac_min_szn")
    
    # Include FGA for share computation
    has_fga = "fga2_mean" in df_sim_v3.columns and "fga3_mean" in df_sim_v3.columns
    has_fta = "fta_mean" in df_sim_v3.columns
    if has_fga:
        cols_v3.extend(["fga2_mean", "fga3_mean"])
        cols_novac.extend(["fga2_mean", "fga3_mean"])
    if has_fta:
        cols_v3.append("fta_mean")
        cols_novac.append("fta_mean")
    
    merged = df_sim_v3[[c for c in cols_v3 if c in df_sim_v3.columns]].merge(
        df_novac[[c for c in cols_novac if c in df_novac.columns]],
        on="player_id",
        suffixes=("_v3", "_novac"),
        how="inner",
    )
    
    if merged.empty:
        return pd.DataFrame()
    
    # Compute player-level FPTS delta
    merged["fpts_delta"] = merged["dk_fpts_mean_novac"] - merged["dk_fpts_mean_v3"]
    merged["abs_fpts_delta"] = merged["fpts_delta"].abs()
    
    # Compute FGA shares if available
    if has_fga:
        merged["fga_v3"] = merged["fga2_mean_v3"] + merged["fga3_mean_v3"]
        merged["fga_novac"] = merged["fga2_mean_novac"] + merged["fga3_mean_novac"]
        
        # Team FGA totals
        team_fga = merged.groupby(["game_id", "team_id"]).agg(
            team_fga_v3=("fga_v3", "sum"),
            team_fga_novac=("fga_novac", "sum"),
        ).reset_index()
        merged = merged.merge(team_fga, on=["game_id", "team_id"])
        
        # Compute shares (within team-game)
        merged["share_fga_v3"] = merged["fga_v3"] / merged["team_fga_v3"].clip(lower=1)
        merged["share_fga_novac"] = merged["fga_novac"] / merged["team_fga_novac"].clip(lower=1)
        merged["share_fga_delta"] = (merged["share_fga_novac"] - merged["share_fga_v3"]).abs()
    
    # Compute FTA shares if available
    if has_fta:
        team_fta = merged.groupby(["game_id", "team_id"]).agg(
            team_fta_v3=("fta_mean_v3", "sum"),
            team_fta_novac=("fta_mean_novac", "sum"),
        ).reset_index()
        merged = merged.merge(team_fta, on=["game_id", "team_id"])
        
        merged["share_fta_v3"] = merged["fta_mean_v3"] / merged["team_fta_v3"].clip(lower=1)
        merged["share_fta_novac"] = merged["fta_mean_novac"] / merged["team_fta_novac"].clip(lower=1)
        merged["share_fta_delta"] = (merged["share_fta_novac"] - merged["share_fta_v3"]).abs()
    
    # Aggregate to team-game level
    agg_dict = {
        "abs_fpts_delta": ["mean", "max"],
        "player_id": "count",
    }
    
    if "vac_min_szn" in merged.columns:
        agg_dict["vac_min_szn"] = "first"
    
    if has_fga:
        agg_dict["share_fga_delta"] = "sum"  # L1 distance for FGA
    if has_fta:
        agg_dict["share_fta_delta"] = "sum"  # L1 distance for FTA
    
    team_agg = merged.groupby(["game_id", "team_id"]).agg(agg_dict).reset_index()
    
    # Flatten column names
    flat_cols = ["game_id", "team_id", "mean_abs_delta_fpts", "max_abs_delta_fpts", "n_players"]
    if "vac_min_szn" in merged.columns:
        flat_cols.append("vac_min_szn")
    if has_fga:
        flat_cols.append("l1_fga_share")
    if has_fta:
        flat_cols.append("l1_fta_share")
    team_agg.columns = flat_cols
    
    # Get max player delta info per team-game
    idx_max = merged.groupby(["game_id", "team_id"])["abs_fpts_delta"].idxmax()
    max_players = merged.loc[idx_max, ["game_id", "team_id", "player_id", "abs_fpts_delta"]].rename(
        columns={"player_id": "max_player_id", "abs_fpts_delta": "max_player_delta"}
    )
    team_agg = team_agg.merge(max_players[["game_id", "team_id", "max_player_id"]], on=["game_id", "team_id"], how="left")
    
    return team_agg


def compute_fga_share_l1(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """Compute FGA share L1 distances per team-game (legacy wrapper)."""
    if df_a.empty or df_b.empty:
        return {}

    # Check for FGA columns
    fga_cols = ["fga2_mean", "fga3_mean"]
    has_fga_a = all(c in df_a.columns for c in fga_cols)
    has_fga_b = all(c in df_b.columns for c in fga_cols)
    if not has_fga_a or not has_fga_b:
        return {}

    # Compute total FGA per player
    df_a = df_a.copy()
    df_b = df_b.copy()
    df_a["fga_total"] = df_a["fga2_mean"] + df_a["fga3_mean"]
    df_b["fga_total"] = df_b["fga2_mean"] + df_b["fga3_mean"]

    # Merge
    merged = df_a[["game_id", "team_id", "player_id", "fga_total"]].merge(
        df_b[["player_id", "fga_total"]],
        on="player_id",
        suffixes=("_a", "_b"),
        how="inner",
    )
    if merged.empty:
        return {}

    # Add vacancy if available
    if "vac_min_szn" in df_a.columns:
        vac_map = df_a.groupby(["game_id", "team_id"])["vac_min_szn"].first().reset_index()
        merged = merged.merge(vac_map, on=["game_id", "team_id"], how="left")

    # Compute team totals
    team_sums = merged.groupby(["game_id", "team_id"]).agg(
        team_fga_a=("fga_total_a", "sum"),
        team_fga_b=("fga_total_b", "sum"),
    ).reset_index()
    merged = merged.merge(team_sums, on=["game_id", "team_id"])

    # Compute shares
    merged["share_a"] = merged["fga_total_a"] / merged["team_fga_a"].clip(lower=1)
    merged["share_b"] = merged["fga_total_b"] / merged["team_fga_b"].clip(lower=1)
    merged["share_delta"] = (merged["share_b"] - merged["share_a"]).abs()

    # L1 distance per team-game
    team_l1 = merged.groupby(["game_id", "team_id"]).agg(
        l1_distance=("share_delta", "sum"),
        n_players=("player_id", "count"),
    ).reset_index()

    if "vac_min_szn" in merged.columns:
        vac_agg = merged.groupby(["game_id", "team_id"])["vac_min_szn"].first().reset_index()
        team_l1 = team_l1.merge(vac_agg, on=["game_id", "team_id"], how="left")

    return {
        "mean_l1": team_l1["l1_distance"].mean(),
        "max_l1": team_l1["l1_distance"].max(),
        "team_l1_df": team_l1,
        "player_df": merged,
    }


def compute_vacancy_bucket_stats(l1_result: dict) -> pd.DataFrame:
    """Bucket team-games by vac_min_szn and compute L1 stats per bucket."""
    if not l1_result or "team_l1_df" not in l1_result:
        return pd.DataFrame()

    df = l1_result["team_l1_df"].copy()
    if "vac_min_szn" not in df.columns or df["vac_min_szn"].isna().all():
        return pd.DataFrame()

    # Create quartile buckets
    df["vac_bucket"] = pd.qcut(df["vac_min_szn"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"], duplicates="drop")

    bucket_stats = df.groupby("vac_bucket", observed=True).agg(
        n_teams=("game_id", "count"),
        mean_l1=("l1_distance", "mean"),
        max_l1=("l1_distance", "max"),
        mean_vac=("vac_min_szn", "mean"),
    ).reset_index()

    return bucket_stats


def get_top_movers(l1_result: dict, n: int = 10) -> pd.DataFrame:
    """Get top team-games by share L1 distance."""
    if not l1_result or "team_l1_df" not in l1_result:
        return pd.DataFrame()

    df = l1_result["team_l1_df"].copy()
    top = df.nlargest(n, "l1_distance")
    return top


def main():
    parser = argparse.ArgumentParser(description="Sim V3 A/B Backtest")
    parser.add_argument("--data-root", type=Path, default=Path("/home/daniel/projections-data"))
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--n-worlds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--profiles", type=str, default="baseline,sim_v3,sim_v3_novacancy")
    parser.add_argument("--skip-runs", action="store_true", help="Skip running sims")
    parser.add_argument("--export-attempt-means", action="store_true", help="Export fga2_mean/fga3_mean/fta_mean for share L1 analysis")

    args = parser.parse_args()
    profiles = [p.strip() for p in args.profiles.split(",")]

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir = args.data_root / "artifacts" / "diagnostics" / "sim_v3_ab_backtest" / f"run={timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SIM V3 A/B BACKTEST")
    print("=" * 70)
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Profiles: {profiles}")
    print(f"Worlds: {args.n_worlds}, Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print()

    dates = pd.date_range(args.start_date, args.end_date, freq="D")

    baseline_results = []
    vacancy_results = []
    all_l1_stats = []
    key_integrity_results = []  # Track team-game key integrity

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        print(f"## {date_str}")

        profile_data = {}
        profile_paths = {}

        for profile in profiles:
            if not args.skip_runs:
                print(f"  Running {profile}...", end=" ", flush=True)
                output_path = run_sim_for_date(
                    args.data_root, date_str, profile, args.n_worlds, args.seed, output_dir
                )
                if output_path:
                    print("✓")
                    profile_paths[profile] = output_path
                else:
                    print("✗")
                    continue
            else:
                output_path = output_dir / f"{profile}_{date_str}.parquet"
                if not output_path.exists():
                    # Try shared path
                    output_path = args.data_root / "artifacts" / "sim_v2" / "worlds_fpts_v2" / f"game_date={date_str}" / "projections.parquet"
                profile_paths[profile] = output_path

            if output_path and output_path.exists():
                df = load_projections(output_path)
                profile_data[profile] = df

        # Compare vs baseline
        if "baseline" in profile_data:
            for variant in ["sim_v3", "sim_v3_novacancy"]:
                if variant not in profile_data:
                    continue
                deltas = compute_fpts_deltas(profile_data["baseline"], profile_data[variant], "baseline", variant)
                if deltas:
                    result = {
                        "date": date_str,
                        "comparison": f"{variant} vs baseline",
                        "mean_abs_delta": deltas["mean_abs_delta"],
                        "max_abs_delta": deltas["max_abs_delta"],
                    }
                    baseline_results.append(result)
                    print(f"  {variant} vs baseline: mean_abs_delta={deltas['mean_abs_delta']:.4f}")

        # sim_v3 vs sim_v3_novacancy (vacancy effect)
        if "sim_v3" in profile_data and "sim_v3_novacancy" in profile_data:
            # Check key integrity for sim_v3
            integrity = check_key_integrity(profile_data["sim_v3"], date_str)
            if integrity:
                key_integrity_results.append(integrity)
            
            fpts_deltas = compute_fpts_deltas(profile_data["sim_v3"], profile_data["sim_v3_novacancy"], "sim_v3", "novac")
            l1_result = compute_fga_share_l1(profile_data["sim_v3"], profile_data["sim_v3_novacancy"])
            
            # Compute team-game level diagnostics
            team_diag = compute_team_game_diagnostics(profile_data["sim_v3"], profile_data["sim_v3_novacancy"])
            if not team_diag.empty:
                team_diag["date"] = date_str
                all_l1_stats.append(team_diag)

            if fpts_deltas:
                print(f"  sim_v3 vs sim_v3_novacancy: mean_abs_delta={fpts_deltas['mean_abs_delta']:.4f}")
                vacancy_results.append({
                    "date": date_str,
                    "mean_abs_delta_fpts": fpts_deltas["mean_abs_delta"],
                    "max_abs_delta_fpts": fpts_deltas["max_abs_delta"],
                    "mean_l1": l1_result.get("mean_l1", 0),
                    "max_l1": l1_result.get("max_l1", 0),
                })

        print()

    # ==================== SUMMARY ====================
    print("=" * 70)
    print("SUMMARY: Baseline Comparisons")
    print("=" * 70)
    print()

    if baseline_results:
        baseline_df = pd.DataFrame(baseline_results)
        for comparison in baseline_df["comparison"].unique():
            subset = baseline_df[baseline_df["comparison"] == comparison]
            print(f"### {comparison}")
            print(f"  Mean abs delta (across dates): {subset['mean_abs_delta'].mean():.4f}")
            print(f"  Max abs delta (any player): {subset['max_abs_delta'].max():.4f}")
            print()

    # ==================== KEY INTEGRITY ====================
    print("=" * 70)
    print("KEY INTEGRITY (team-game)")
    print("=" * 70)
    print()
    
    if key_integrity_results:
        integrity_df = pd.DataFrame(key_integrity_results)
        all_valid = integrity_df["teams_valid_range"].all()
        all_2teams = integrity_df["all_games_have_2_teams"].all()
        
        print(f"  Distinct team_ids (total): {integrity_df['n_distinct_teams'].sum()}")
        print(f"  team_id min: {integrity_df['team_id_min'].min()}")
        print(f"  team_id max: {integrity_df['team_id_max'].max()}")
        print(f"  All team_ids in valid NBA range: {'✅ Yes' if all_valid else '❌ No'}")
        print(f"  All games have exactly 2 teams: {'✅ Yes' if all_2teams else '❌ No'}")
        if not all_2teams:
            bad_count = integrity_df["bad_game_count"].sum()
            print(f"    ⚠️ {bad_count} games with != 2 teams")
        print()
    else:
        print("  No integrity data collected")
        print()

    # ==================== VACANCY INCREMENTAL EFFECT ====================
    print("=" * 70)
    print("VACANCY INCREMENTAL EFFECT (sim_v3 vs sim_v3_novacancy)")
    print("=" * 70)
    print()

    if vacancy_results:
        vac_df = pd.DataFrame(vacancy_results)
        print("### Per-Date Results")
        print("| Date | Mean Δ FPTS | Max Δ FPTS | Mean L1 (shares) | Max L1 |")
        print("|------|-------------|------------|------------------|--------|")
        for _, row in vac_df.iterrows():
            print(f"| {row['date']} | {row['mean_abs_delta_fpts']:.4f} | {row['max_abs_delta_fpts']:.4f} | {row['mean_l1']:.4f} | {row['max_l1']:.4f} |")
        print()

        print("### Overall Summary")
        print(f"  Mean absolute FPTS delta: {vac_df['mean_abs_delta_fpts'].mean():.4f}")
        print(f"  Max absolute FPTS delta: {vac_df['max_abs_delta_fpts'].max():.4f}")
        print(f"  Mean share L1 distance: {vac_df['mean_l1'].mean():.4f}")
        print()

        # Vacancy bucket analysis (strict team-game granularity)
        if all_l1_stats:
            from scipy.stats import spearmanr
            
            combined = pd.concat(all_l1_stats, ignore_index=True)
            
            # Now `combined` is team-game level from compute_team_game_diagnostics
            if "vac_min_szn" in combined.columns and combined["vac_min_szn"].notna().any():
                print("### Spearman Correlations (team-game level)")
                
                # Spearman correlation between vac_min_szn and FPTS delta
                corr_fpts, pval_fpts = spearmanr(combined["vac_min_szn"], combined["mean_abs_delta_fpts"])
                print(f"  vac_min_szn vs mean_abs_delta_fpts: r={corr_fpts:.4f} (p={pval_fpts:.4f})")
                
                # Spearman correlation between vac_min_szn and FGA share L1
                if "l1_fga_share" in combined.columns:
                    corr_l1, pval_l1 = spearmanr(combined["vac_min_szn"], combined["l1_fga_share"])
                    print(f"  vac_min_szn vs l1_fga_share: r={corr_l1:.4f} (p={pval_l1:.4f})")
                
                # Spearman correlation between vac_min_szn and FTA share L1
                if "l1_fta_share" in combined.columns:
                    corr_l1_fta, pval_l1_fta = spearmanr(combined["vac_min_szn"], combined["l1_fta_share"])
                    print(f"  vac_min_szn vs l1_fta_share: r={corr_l1_fta:.4f} (p={pval_l1_fta:.4f})")
                print()
                
                print("### Vacancy Bucket Analysis (team-game level)")
                print("(Higher vacancy should → higher FPTS delta if vacancy matters)")
                print()
                
                combined["vac_bucket"] = pd.qcut(
                    combined["vac_min_szn"].clip(lower=0),
                    q=4, labels=["Q1 (low vac)", "Q2", "Q3", "Q4 (high vac)"],
                    duplicates="drop"
                )
                
                # Bucket stats with median/p90
                bucket_stats = combined.groupby("vac_bucket", observed=True).agg(
                    n_teams=("game_id", "count"),
                    median_delta=("mean_abs_delta_fpts", "median"),
                    p90_delta=("mean_abs_delta_fpts", lambda x: x.quantile(0.9)),
                    mean_vac=("vac_min_szn", "mean"),
                ).reset_index()
                
                # Add FGA L1 if available
                has_fga_l1 = "l1_fga_share" in combined.columns
                if has_fga_l1:
                    l1_bucket = combined.groupby("vac_bucket", observed=True).agg(
                        median_l1_fga=("l1_fga_share", "median"),
                        p90_l1_fga=("l1_fga_share", lambda x: x.quantile(0.9)),
                    ).reset_index()
                    bucket_stats = bucket_stats.merge(l1_bucket, on="vac_bucket")
                
                header = "| Bucket | n | Med Δ FPTS | p90 Δ FPTS |" + (" Med L1 FGA | p90 L1 FGA |" if has_fga_l1 else "") + " Mean Vac |"
                sep = "|--------|---|------------|------------|" + ("------------|------------|" if has_fga_l1 else "") + "----------|"
                print(header)
                print(sep)
                for _, row in bucket_stats.iterrows():
                    line = f"| {row['vac_bucket']} | {row['n_teams']:.0f} | {row['median_delta']:.4f} | {row['p90_delta']:.4f} |"
                    if has_fga_l1:
                        line += f" {row['median_l1_fga']:.4f} | {row['p90_l1_fga']:.4f} |"
                    line += f" {row['mean_vac']:.1f} |"
                    print(line)
                print()

                # Monotonicity check
                delta_values = bucket_stats["median_delta"].tolist()
                is_monotone = all(delta_values[i] <= delta_values[i + 1] for i in range(len(delta_values) - 1))
                q4_gt_q1 = delta_values[-1] > delta_values[0] if len(delta_values) >= 2 else False

                print(f"  Monotone increasing: {'✅ Yes' if is_monotone else '❌ No'}")
                print(f"  Q4 > Q1: {'✅ Yes' if q4_gt_q1 else '❌ No'}")
                print()
                
                # Top 5 Q1 team-games by FPTS delta (outlier investigation)
                q1_teams = combined[combined["vac_bucket"] == "Q1 (low vac)"]
                if not q1_teams.empty:
                    print("### Top 5 Q1 (low vac) Team-Games by FPTS Delta")
                    print("(Investigating why low vacancy teams have high FPTS delta)")
                    print()
                    q1_top = q1_teams.nlargest(5, "mean_abs_delta_fpts")
                    cols = ["date", "game_id", "team_id", "mean_abs_delta_fpts", "max_abs_delta_fpts", "vac_min_szn"]
                    if "max_player_id" in q1_top.columns:
                        cols.append("max_player_id")
                    print(q1_top[[c for c in cols if c in q1_top.columns]].to_string(index=False))
                    print()
            
            # Top 10 team-games by FPTS delta
            print("### Top 10 Team-Games by FPTS Delta")
            top10 = combined.nlargest(10, "mean_abs_delta_fpts")
            cols = ["date", "game_id", "team_id", "mean_abs_delta_fpts", "max_abs_delta_fpts", "vac_min_szn"]
            print(top10[[c for c in cols if c in top10.columns]].to_string(index=False))
            print()

    # ==================== SHIP/NO-SHIP DECISION ====================
    print("=" * 70)
    print("SHIP/NO-SHIP DECISION")
    print("=" * 70)
    print()

    if vacancy_results and all_l1_stats:
        vac_df = pd.DataFrame(vacancy_results)
        mean_delta = vac_df["mean_abs_delta_fpts"].mean()
        mean_l1 = vac_df["mean_l1"].mean()

        # Check vacancy effect size vs baseline effect
        baseline_effect = 0
        if baseline_results:
            sim_v3_vs_baseline = [r for r in baseline_results if "sim_v3 vs baseline" in r["comparison"]]
            if sim_v3_vs_baseline:
                baseline_effect = np.mean([r["mean_abs_delta"] for r in sim_v3_vs_baseline])

        vacancy_effect_ratio = mean_delta / baseline_effect if baseline_effect > 0 else 0

        print(f"  Vacancy effect (sim_v3 vs novacancy): {mean_delta:.4f} FPTS")
        print(f"  Model effect (sim_v3 vs baseline): {baseline_effect:.4f} FPTS")
        print(f"  Vacancy effect ratio: {vacancy_effect_ratio:.2%}")
        print()

        if mean_delta < 0.001:
            print("  ❌ NO-SHIP: Vacancy has negligible effect (<0.001 FPTS)")
        elif mean_l1 < 0.001:
            print("  ❌ NO-SHIP: Vacancy has no share allocation effect")
        elif vacancy_effect_ratio > 0.5:
            print("  ⚠️ REVIEW: Vacancy effect is >50% of model effect (maybe too strong?)")
        else:
            print("  ✅ SHIP: Vacancy has measurable, reasonable effect")
    else:
        print("  ⚠️ Insufficient data for decision")

    print()

    # ==================== WRITE REPORT ====================
    report_lines = [
        "# Sim V3 A/B Backtest Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"Date range: {args.start_date} to {args.end_date}",
        f"Profiles: {args.profiles}",
        f"Worlds: {args.n_worlds}, Seed: {args.seed}",
        "",
    ]

    if baseline_results:
        report_lines.append("## Baseline Comparisons")
        report_lines.append("")
        report_lines.append("| Date | Comparison | Mean Δ FPTS | Max Δ FPTS |")
        report_lines.append("|------|------------|-------------|------------|")
        for r in baseline_results:
            report_lines.append(f"| {r['date']} | {r['comparison']} | {r['mean_abs_delta']:.4f} | {r['max_abs_delta']:.4f} |")
        report_lines.append("")

    if vacancy_results:
        report_lines.append("## Vacancy Incremental Effect")
        report_lines.append("")
        report_lines.append("| Date | Mean Δ FPTS | Max Δ FPTS | Mean L1 | Max L1 |")
        report_lines.append("|------|-------------|------------|---------|--------|")
        for r in vacancy_results:
            report_lines.append(f"| {r['date']} | {r['mean_abs_delta_fpts']:.4f} | {r['max_abs_delta_fpts']:.4f} | {r['mean_l1']:.4f} | {r['max_l1']:.4f} |")
        report_lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines))
    print(f"Report written to: {report_path}")

    # Write parquets
    if baseline_results:
        pd.DataFrame(baseline_results).to_parquet(output_dir / "baseline_results.parquet")
    if vacancy_results:
        pd.DataFrame(vacancy_results).to_parquet(output_dir / "vacancy_results.parquet")
    if all_l1_stats:
        pd.concat(all_l1_stats, ignore_index=True).to_parquet(output_dir / "team_l1_stats.parquet")


if __name__ == "__main__":
    main()
