#!/usr/bin/env python3
"""
Calibrate vegas_env parameters for sim_v2.

This script checks whether the vegas_env sampling produces realistic:
1. Game total distributions (simulated vs actual)
2. Spread/margin distributions (simulated vs actual)
3. Team point distributions
4. Pace multiplier effects

Compares simulated distributions against historical actuals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.paths import data_path

app = typer.Typer()


def load_historical_game_outcomes(
    data_root: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load historical game outcomes from rates_training_base."""

    base_path = data_root / "gold" / "rates_training_base"
    if not base_path.exists():
        raise FileNotFoundError(f"Rates training base not found at {base_path}")

    dfs = []
    for season_dir in sorted(base_path.glob("season=*")):
        for date_dir in sorted(season_dir.glob("game_date=*")):
            try:
                date_str = date_dir.name.split("=")[1]
                game_date = pd.Timestamp(date_str).normalize()
            except Exception:
                continue

            if game_date < start_date or game_date > end_date:
                continue

            parquet_path = date_dir / "rates_training_base.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                df["game_date"] = game_date
                dfs.append(df)

    if not dfs:
        raise ValueError(f"No data found in range {start_date.date()} to {end_date.date()}")

    df = pd.concat(dfs, ignore_index=True)

    # Compute player points - handle NaN shooting percentages (player didn't take that shot type)
    # fillna(0) for pct is safe: 0 FGA * any_pct = 0 pts anyway
    fgm2 = df["fga2_per_min"] * df["fg2_pct_label"].fillna(0)
    fgm3 = df["fga3_per_min"] * df["fg3_pct_label"].fillna(0)
    ftm = df["fta_per_min"] * df["ft_pct_label"].fillna(0)
    df["pts"] = (2 * fgm2 + 3 * fgm3 + ftm) * df["minutes_actual"]

    # Aggregate to team level
    team_df = df.groupby(["game_date", "game_id", "team_id"]).agg({
        "pts": "sum",
        "spread_close": "first",
        "total_close": "first",
        "opponent_id": "first",
    }).reset_index()

    # Get opponent points
    team_df = team_df.merge(
        team_df[["game_date", "game_id", "team_id", "pts"]].rename(
            columns={"team_id": "opponent_id", "pts": "opp_pts"}
        ),
        on=["game_date", "game_id", "opponent_id"],
        how="left"
    )

    team_df["margin"] = team_df["pts"] - team_df["opp_pts"]

    # Game total is pts + opp_pts (full game, not doubled)
    # Since we have each game from both team perspectives, we need to dedupe
    game_df = team_df.groupby(["game_date", "game_id"]).agg({
        "pts": "sum",  # This gives us the full game total (both teams)
        "total_close": "first",
        "spread_close": "first",
    }).reset_index()

    # pts sum is actually doubled (each team counted once), so divide by 2...
    # Actually no - we sum pts from both rows which gives game total
    game_df["game_total"] = game_df["pts"]  # Already the full game total
    game_df["total_vs_vegas"] = game_df["game_total"] - game_df["total_close"]

    # For margin, we need to look at it from one team's perspective
    # Use the home team (positive spread_close means home is underdog)
    team_df["margin_vs_spread"] = team_df["margin"] - (-team_df["spread_close"])

    # Merge game-level back
    team_df = team_df.merge(
        game_df[["game_date", "game_id", "game_total", "total_vs_vegas"]],
        on=["game_date", "game_id"],
        how="left"
    )

    return team_df


def simulate_vegas_env(
    n_games: int,
    n_worlds: int,
    total_mu: float = 230.0,
    spread_mu: float = 0.0,
    total_sigma: float = 12.0,
    spread_sigma: float = 8.0,
    dist: str = "student_t",
    df: float = 5.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Simulate vegas_env sampling to get distributions."""

    rng = np.random.default_rng(seed)

    # Sample total and spread per game per world
    if dist == "normal":
        total_noise = rng.normal(0, total_sigma, size=(n_worlds, n_games))
        spread_noise = rng.normal(0, spread_sigma, size=(n_worlds, n_games))
    else:  # student_t
        total_noise = total_sigma * rng.standard_t(df=df, size=(n_worlds, n_games))
        spread_noise = spread_sigma * rng.standard_t(df=df, size=(n_worlds, n_games))

    total_world = total_mu + total_noise
    spread_world = spread_mu + spread_noise

    home_pts = (total_world / 2.0) - (spread_world / 2.0)
    away_pts = total_world - home_pts
    margin = home_pts - away_pts  # from home perspective

    return {
        "total": total_world,
        "spread": spread_world,
        "home_pts": home_pts,
        "away_pts": away_pts,
        "margin": margin,
        "total_vs_vegas": total_world - total_mu,
        "margin_vs_spread": margin - (-spread_mu),
    }


def print_comparison(
    actual: pd.DataFrame,
    sim: dict[str, np.ndarray],
    label: str,
) -> dict[str, float]:
    """Compare actual vs simulated distributions."""

    results = {}

    print(f"\n{'='*70}")
    print(f"{label}")
    print("="*70)

    # Game totals
    actual_totals = actual["game_total"].dropna()
    sim_totals = sim["total"].flatten()

    print(f"\n{'GAME TOTALS':-^50}")
    print(f"{'Metric':<25} {'Actual':>12} {'Simulated':>12} {'Delta':>10}")
    print("-" * 60)

    metrics = [
        ("Mean", actual_totals.mean(), sim_totals.mean()),
        ("Std", actual_totals.std(), sim_totals.std()),
        ("P10", np.percentile(actual_totals, 10), np.percentile(sim_totals, 10)),
        ("P50", np.percentile(actual_totals, 50), np.percentile(sim_totals, 50)),
        ("P90", np.percentile(actual_totals, 90), np.percentile(sim_totals, 90)),
        ("Min", actual_totals.min(), sim_totals.min()),
        ("Max", actual_totals.max(), sim_totals.max()),
    ]

    for name, act, sim_val in metrics:
        delta = sim_val - act
        print(f"{name:<25} {act:>12.1f} {sim_val:>12.1f} {delta:>+10.1f}")
        results[f"total_{name.lower()}"] = delta

    # Total vs Vegas (deviation from line)
    actual_vs_vegas = actual["total_vs_vegas"].dropna()
    sim_vs_vegas = sim["total_vs_vegas"].flatten()

    print(f"\n{'TOTAL vs VEGAS (deviation from line)':-^50}")
    print(f"{'Metric':<25} {'Actual':>12} {'Simulated':>12} {'Delta':>10}")
    print("-" * 60)

    metrics = [
        ("Mean", actual_vs_vegas.mean(), sim_vs_vegas.mean()),
        ("Std", actual_vs_vegas.std(), sim_vs_vegas.std()),
        ("P10", np.percentile(actual_vs_vegas, 10), np.percentile(sim_vs_vegas, 10)),
        ("P90", np.percentile(actual_vs_vegas, 90), np.percentile(sim_vs_vegas, 90)),
    ]

    for name, act, sim_val in metrics:
        delta = sim_val - act
        flag = " ***" if name == "Std" and abs(delta) > 3 else ""
        print(f"{name:<25} {act:>12.1f} {sim_val:>12.1f} {delta:>+10.1f}{flag}")
        results[f"total_vs_vegas_{name.lower()}"] = delta

    # Margin vs Spread
    actual_margin = actual["margin_vs_spread"].dropna()
    sim_margin = sim["margin_vs_spread"].flatten()

    print(f"\n{'MARGIN vs SPREAD (deviation from line)':-^50}")
    print(f"{'Metric':<25} {'Actual':>12} {'Simulated':>12} {'Delta':>10}")
    print("-" * 60)

    metrics = [
        ("Mean", actual_margin.mean(), sim_margin.mean()),
        ("Std", actual_margin.std(), sim_margin.std()),
        ("P10", np.percentile(actual_margin, 10), np.percentile(sim_margin, 10)),
        ("P90", np.percentile(actual_margin, 90), np.percentile(sim_margin, 90)),
    ]

    for name, act, sim_val in metrics:
        delta = sim_val - act
        flag = " ***" if name == "Std" and abs(delta) > 3 else ""
        print(f"{name:<25} {act:>12.1f} {sim_val:>12.1f} {delta:>+10.1f}{flag}")
        results[f"margin_vs_spread_{name.lower()}"] = delta

    return results


def print_bucketed_analysis(games: pd.DataFrame, team_df: pd.DataFrame) -> dict:
    """Analyze variance by Vegas total and spread buckets."""

    print("\n" + "=" * 70)
    print("BUCKETED VARIANCE ANALYSIS")
    print("=" * 70)

    results = {}

    # --- By Vegas Total buckets ---
    print("\n" + "-" * 70)
    print("TOTAL DEVIATION STD by Vegas Total Bucket")
    print("-" * 70)
    print(f"{'Bucket':<20} {'Games':>8} {'Mean Dev':>12} {'Std Dev':>12} {'Recommended σ':>15}")
    print("-" * 70)

    total_buckets = [
        ("Low (<215)", games["total_close"] < 215),
        ("Medium (215-230)", (games["total_close"] >= 215) & (games["total_close"] < 230)),
        ("High (230-240)", (games["total_close"] >= 230) & (games["total_close"] < 240)),
        ("Very High (≥240)", games["total_close"] >= 240),
    ]

    results["by_total"] = {}
    for label, mask in total_buckets:
        bucket = games[mask]
        if len(bucket) >= 20:
            std = bucket["total_vs_vegas"].std()
            mean = bucket["total_vs_vegas"].mean()
            results["by_total"][label] = {"n": len(bucket), "std": float(std), "mean": float(mean)}
            print(f"{label:<20} {len(bucket):>8} {mean:>+12.1f} {std:>12.1f} {std:>15.1f}")
        else:
            print(f"{label:<20} {len(bucket):>8} {'(insufficient data)':>40}")

    # --- By Spread buckets ---
    print("\n" + "-" * 70)
    print("MARGIN DEVIATION STD by Spread Bucket (absolute spread)")
    print("-" * 70)
    print(f"{'Bucket':<20} {'Games':>8} {'Mean Dev':>12} {'Std Dev':>12} {'Implied spread_σ':>18}")
    print("-" * 70)

    # Use team_df for margin analysis (one row per team per game)
    # Dedupe to one row per game for spread analysis
    spread_games = team_df.drop_duplicates(subset=["game_date", "game_id"]).copy()
    spread_games["abs_spread"] = spread_games["spread_close"].abs()

    spread_buckets = [
        ("Pick'em (0-2)", spread_games["abs_spread"] <= 2),
        ("Close (2.5-5)", (spread_games["abs_spread"] > 2) & (spread_games["abs_spread"] <= 5)),
        ("Medium (5.5-8)", (spread_games["abs_spread"] > 5) & (spread_games["abs_spread"] <= 8)),
        ("Large (8.5-12)", (spread_games["abs_spread"] > 8) & (spread_games["abs_spread"] <= 12)),
        ("Blowout (>12)", spread_games["abs_spread"] > 12),
    ]

    # For margin analysis, use team_df which has margin_vs_spread
    results["by_spread"] = {}
    for label, mask in spread_buckets:
        game_ids = spread_games[mask]["game_id"].unique()
        bucket = team_df[team_df["game_id"].isin(game_ids)]
        n_games = len(game_ids)
        if n_games >= 20:
            std = bucket["margin_vs_spread"].std()
            mean = bucket["margin_vs_spread"].mean()
            implied_spread_sigma = std / np.sqrt(2)
            results["by_spread"][label] = {
                "n": n_games,
                "std": float(std),
                "mean": float(mean),
                "implied_spread_sigma": float(implied_spread_sigma),
            }
            print(f"{label:<20} {n_games:>8} {mean:>+12.1f} {std:>12.1f} {implied_spread_sigma:>18.1f}")
        else:
            print(f"{label:<20} {n_games:>8} {'(insufficient data)':>45}")

    # --- Cross-tabulation summary ---
    print("\n" + "-" * 70)
    print("SUMMARY: Variance patterns")
    print("-" * 70)

    if results["by_total"]:
        total_stds = [v["std"] for v in results["by_total"].values()]
        print(f"Total deviation std range: {min(total_stds):.1f} - {max(total_stds):.1f}")
        if max(total_stds) - min(total_stds) > 3:
            print("  → Significant variation by total bucket - consider conditional sigma")
        else:
            print("  → Relatively uniform across total buckets")

    if results["by_spread"]:
        spread_stds = [v["std"] for v in results["by_spread"].values()]
        print(f"Margin deviation std range: {min(spread_stds):.1f} - {max(spread_stds):.1f}")
        if max(spread_stds) - min(spread_stds) > 3:
            print("  → Significant variation by spread bucket - consider conditional sigma")

            # Compute recommended conditional config
            close_stds = []
            blowout_stds = []
            for label, data in results["by_spread"].items():
                if "Pick'em" in label or "Close" in label or "Medium" in label:
                    close_stds.append(data["implied_spread_sigma"])
                else:
                    blowout_stds.append(data["implied_spread_sigma"])

            if close_stds and blowout_stds:
                base_sigma = np.mean(close_stds)
                blowout_sigma = np.mean(blowout_stds)
                scale = blowout_sigma / base_sigma
                print(f"\n  RECOMMENDED spread_sigma_conditional config:")
                print(f"    spread_sigma: {base_sigma:.1f} (base for close games)")
                print(f"    threshold: 8.0 (|spread| cutoff)")
                print(f"    scale_above: {scale:.2f} (→ {base_sigma * scale:.1f} for blowouts)")
        else:
            print("  → Relatively uniform across spread buckets")

    return results


@app.command()
def main(
    start_date: str = typer.Option("2024-10-01", help="Start date"),
    end_date: str = typer.Option("2025-04-01", help="End date"),
    data_root: Optional[Path] = typer.Option(None, help="Data root"),
    n_worlds: int = typer.Option(25000, help="Number of worlds to simulate"),
    total_sigma: float = typer.Option(12.0, help="vegas_env total_sigma"),
    spread_sigma: float = typer.Option(8.0, help="vegas_env spread_sigma"),
    dist: str = typer.Option("student_t", help="Distribution (normal or student_t)"),
    df: float = typer.Option(5.0, help="Degrees of freedom for student_t"),
    output_json: Optional[Path] = typer.Option(None, help="Output JSON path"),
) -> None:
    """Calibrate vegas_env parameters against historical data."""

    root = data_root or data_path()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    typer.echo(f"[calibrate] Loading historical games from {start.date()} to {end.date()}")
    actual = load_historical_game_outcomes(root, start, end)

    # Deduplicate to game level (each game appears twice, once per team)
    games = actual.drop_duplicates(subset=["game_date", "game_id"])[
        ["game_date", "game_id", "game_total", "total_close", "total_vs_vegas"]
    ].copy()

    n_games = len(games)
    typer.echo(f"[calibrate] Found {n_games} unique games")

    # Get median vegas total as baseline
    median_total = games["total_close"].median()
    typer.echo(f"[calibrate] Median Vegas total: {median_total:.1f}")

    # Simulate with current config
    typer.echo(f"\n[calibrate] Simulating {n_worlds} worlds with:")
    typer.echo(f"  total_sigma={total_sigma}, spread_sigma={spread_sigma}")
    typer.echo(f"  dist={dist}, df={df}")

    sim = simulate_vegas_env(
        n_games=n_games,
        n_worlds=n_worlds,
        total_mu=median_total,
        spread_mu=0.0,  # Use 0 as baseline, actual spread varies
        total_sigma=total_sigma,
        spread_sigma=spread_sigma,
        dist=dist,
        df=df,
    )

    results = print_comparison(actual, sim, "VEGAS_ENV CALIBRATION")

    # Bucketed analysis
    # Deduplicate games for total analysis
    games = actual.drop_duplicates(subset=["game_date", "game_id"])[
        ["game_date", "game_id", "game_total", "total_close", "total_vs_vegas", "spread_close"]
    ].copy()
    bucket_results = print_bucketed_analysis(games, actual)

    # Recommendations
    print("\n" + "="*70)
    print("CALIBRATION SUMMARY")
    print("="*70)

    actual_total_std = actual["total_vs_vegas"].std()
    actual_margin_std = actual["margin_vs_spread"].std()

    print(f"\nActual total deviation std:  {actual_total_std:.1f}")
    print(f"Config total_sigma:          {total_sigma:.1f}")
    print(f"Recommended total_sigma:     {actual_total_std:.1f}")

    print(f"\nActual margin deviation std: {actual_margin_std:.1f}")
    print(f"Config spread_sigma:         {spread_sigma:.1f}")
    # Margin std ≈ spread_sigma * sqrt(2) because margin = home - away
    implied_spread_sigma = actual_margin_std / np.sqrt(2)
    print(f"Recommended spread_sigma:    {implied_spread_sigma:.1f}")

    if abs(total_sigma - actual_total_std) > 2:
        print(f"\n⚠️  total_sigma is OFF by {total_sigma - actual_total_std:+.1f}")
    else:
        print(f"\n✓ total_sigma is well calibrated")

    if abs(spread_sigma - implied_spread_sigma) > 2:
        print(f"⚠️  spread_sigma is OFF by {spread_sigma - implied_spread_sigma:+.1f}")
    else:
        print(f"✓ spread_sigma is well calibrated")

    if output_json:
        output = {
            "actual_total_std": float(actual_total_std),
            "actual_margin_std": float(actual_margin_std),
            "recommended_total_sigma": float(actual_total_std),
            "recommended_spread_sigma": float(implied_spread_sigma),
            "current_total_sigma": float(total_sigma),
            "current_spread_sigma": float(spread_sigma),
            "n_games": n_games,
            "n_worlds": n_worlds,
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(output, indent=2))
        typer.echo(f"\n[calibrate] Wrote results to {output_json}")


if __name__ == "__main__":
    app()
