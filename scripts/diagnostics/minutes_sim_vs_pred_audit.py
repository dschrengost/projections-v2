"""
Diagnostic script to analyze how enforce_team_240 changes minutes distribution.

Compares minutes_pred_p50 (pre-sim) with simulated minutes after applying
the production enforce_team_240 logic. Outputs per-player metrics and a
summary markdown report.

Usage:
    uv run python -m scripts.diagnostics.minutes_sim_vs_pred_audit \
        --data-root /home/daniel/projections-data \
        --start-date 2025-01-20 \
        --end-date 2025-01-26 \
        --profile baseline \
        --num-samples 300 \
        --seed 123
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.paths import data_path
from projections.sim_v2.config import load_sim_v2_profile
from projections.sim_v2.game_script import GameScriptConfig, sample_minutes_with_scripts
from projections.sim_v2.minutes_noise import (
    enforce_team_240_minutes,
    enforce_team_240_with_pruning,
)

app = typer.Typer(add_completion=False, help=__doc__)

DEFAULT_MAX_ROTATION_SIZE = 10


def _season_from_date(d: pd.Timestamp) -> int:
    return d.year if d.month >= 8 else d.year - 1


def _load_minutes_for_date(data_root: Path, game_date: pd.Timestamp) -> pd.DataFrame | None:
    """Load minutes projections for a date."""
    date_token = game_date.date().isoformat()
    path = data_root / "gold" / "projections_minutes_v1" / f"game_date={date_token}" / "minutes.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


def _load_schedule_for_date(data_root: Path, game_date: pd.Timestamp) -> pd.DataFrame:
    """Load schedule for a date."""
    season = _season_from_date(game_date)
    month = game_date.month
    path = data_root / "silver" / "schedule" / f"season={season}" / f"month={month:02d}" / "schedule.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    day = game_date.date()
    return df[df["game_date"] == day].copy()


def _simulate_minutes_for_date(
    minutes_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    *,
    n_samples: int,
    game_script_config: GameScriptConfig | None,
    enforce_240: bool,
    rng: np.random.Generator,
    rotation_minutes_floor: float = 0.0,
    max_rotation_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Simulate minutes for a single date using production logic.
    
    Returns:
        - DataFrame with per-player summary statistics
        - DataFrame with per-team-game stage diagnostics
        - Dict with prune diagnostics (scale_before_readd, players_readded, etc.)
    """
    if minutes_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # Sort for determinism
    minutes_df = minutes_df.sort_values(["game_id", "team_id", "player_id"]).reset_index(drop=True)
    n_players = len(minutes_df)
    
    # Extract arrays
    game_ids = pd.to_numeric(minutes_df["game_id"], errors="coerce").fillna(0).astype(int).to_numpy()
    team_ids = pd.to_numeric(minutes_df["team_id"], errors="coerce").fillna(0).astype(int).to_numpy()
    player_ids = pd.to_numeric(minutes_df["player_id"], errors="coerce").fillna(0).astype(int).to_numpy()
    
    # Minutes quantiles
    minutes_p10 = pd.to_numeric(minutes_df.get("minutes_p10", 0), errors="coerce").fillna(0).to_numpy()
    minutes_p50 = pd.to_numeric(
        minutes_df.get("minutes_p50", minutes_df.get("minutes_p50_cond", 0)), 
        errors="coerce"
    ).fillna(0).to_numpy()
    minutes_p90 = pd.to_numeric(minutes_df.get("minutes_p90", 0), errors="coerce").fillna(0).to_numpy()
    
    # Play probability and starter flag
    play_prob = pd.to_numeric(minutes_df.get("play_prob", 1.0), errors="coerce").fillna(1.0).to_numpy()
    is_starter = (
        minutes_df.get("is_confirmed_starter", pd.Series(False, index=minutes_df.index)).fillna(False).astype(bool) |
        minutes_df.get("is_projected_starter", pd.Series(False, index=minutes_df.index)).fillna(False).astype(bool)
    ).astype(int).to_numpy()
    
    # Build team indices and group map
    unique_teams = np.unique(team_ids)
    team_to_idx = {t: i for i, t in enumerate(unique_teams)}
    team_indices = np.array([team_to_idx[t] for t in team_ids], dtype=int)
    n_teams = len(unique_teams)
    
    # Build (game_id, team_id) -> player indices mapping
    game_team_to_players: dict[tuple[int, int], np.ndarray] = {}
    for gid in np.unique(game_ids):
        for tid in np.unique(team_ids[game_ids == gid]):
            mask = (game_ids == gid) & (team_ids == tid)
            game_team_to_players[(int(gid), int(tid))] = np.flatnonzero(mask)
    
    # Rotation and bench masks (rotation = p50 >= 12)
    rotation_mask = minutes_p50 >= 12.0
    bench_mask = (minutes_p50 > 0) & (minutes_p50 < 12.0)
    
    # Active mask: sample play_prob per world
    active_mask = rng.random(size=(n_samples, n_players)) < play_prob[None, :]
    
    # Build home team lookup
    home_team_ids: dict[int, int] = {}
    if not schedule_df.empty and {"game_id", "home_team_id"}.issubset(schedule_df.columns):
        for _, row in schedule_df.iterrows():
            gid = int(row["game_id"])
            home_team_ids[gid] = int(row["home_team_id"])
    
    # Spreads
    spreads_home = pd.to_numeric(minutes_df.get("spread_home", 0), errors="coerce").fillna(0).to_numpy()
    
    # Step 1: Sample minutes
    if game_script_config is not None:
        minutes_worlds = sample_minutes_with_scripts(
            minutes_p10=minutes_p10,
            minutes_p50=minutes_p50,
            minutes_p90=minutes_p90,
            is_starter=is_starter,
            game_ids=game_ids,
            team_ids=team_ids,
            spreads_home=spreads_home,
            home_team_ids=home_team_ids,
            n_worlds=n_samples,
            config=game_script_config,
            rng=rng,
        )
    else:
        # Fallback: simple sampling from split-normal
        z90 = 1.2815515655446004
        p50 = minutes_p50
        p10 = np.minimum(minutes_p10, p50)
        p90 = np.maximum(minutes_p90, p50)
        sigma_low = np.maximum((p50 - p10) / z90, 0.5)
        sigma_high = np.maximum((p90 - p50) / z90, 0.5)
        z = rng.standard_normal(size=(n_samples, n_players))
        sigma = np.where(z < 0.0, sigma_low[None, :], sigma_high[None, :])
        minutes_worlds = np.maximum(p50[None, :] + z * sigma, 0.0)
    
    # === STAGE TRACKING: pre-enforce (after sampling, before any zeroing) ===
    minutes_pre_zeroing = minutes_worlds.copy()
    
    # Step 2: Zero out inactive players
    minutes_worlds = minutes_worlds * active_mask.astype(float)
    
    # === STAGE TRACKING: post-zeroing (after inactive zeroing, before enforce_240) ===
    minutes_pre_enforce = minutes_worlds.copy()
    
    # Step 3: Enforce 240 per team
    prune_diagnostics: dict = {}
    effective_cap = max_rotation_size if max_rotation_size is not None else DEFAULT_MAX_ROTATION_SIZE
    
    if enforce_240 and n_teams > 0:
        if rotation_minutes_floor > 0:
            # Use new function with floor pruning and iterative re-add
            minutes_worlds, prune_diagnostics = enforce_team_240_with_pruning(
                minutes_world=minutes_worlds,
                team_indices=team_indices,
                active_mask=active_mask,
                baseline_minutes=minutes_p50,
                starter_mask=is_starter > 0,
                rotation_minutes_floor=rotation_minutes_floor,
                max_rotation_size=effective_cap,
                clamp_scale=(0.7, 1.3),
            )
        else:
            # Legacy function for bitwise compatibility when floor=0
            minutes_worlds = enforce_team_240_minutes(
                minutes_world=minutes_worlds,
                team_indices=team_indices,
                rotation_mask=rotation_mask,
                bench_mask=bench_mask,
                baseline_minutes=minutes_p50,
                clamp_scale=(0.7, 1.3),
                active_mask=active_mask,
                starter_mask=is_starter > 0,
                max_rotation_size=effective_cap,
            )
    
    # === STAGE TRACKING: post-enforce (after enforce_240) ===
    minutes_post_enforce = minutes_worlds.copy()
    
    # No additional processing after enforce_240 in this script
    # (In production sim, there may be other steps - this tracks what we have)
    minutes_final = minutes_worlds.copy()
    
    # Compute per-team-game stage diagnostics
    team_game_records = []
    for (gid, tid), player_idxs in game_team_to_players.items():
        # Per-world sums for this team-game
        sum_pre_zeroing = minutes_pre_zeroing[:, player_idxs].sum(axis=1)  # (n_samples,)
        sum_pre_enforce = minutes_pre_enforce[:, player_idxs].sum(axis=1)
        sum_post_enforce = minutes_post_enforce[:, player_idxs].sum(axis=1)
        sum_final = minutes_final[:, player_idxs].sum(axis=1)
        
        # How many worlds have > N active players (rotation cap check)
        active_counts_per_world = active_mask[:, player_idxs].sum(axis=1)
        rotation_capped = (active_counts_per_world > DEFAULT_MAX_ROTATION_SIZE).astype(int)
        
        team_game_records.append({
            "game_date": minutes_df["game_date"].iloc[player_idxs[0]],
            "game_id": gid,
            "team_id": tid,
            "n_players": len(player_idxs),
            # Stage sums - aggregated
            "sum_pre_zeroing_mean": float(sum_pre_zeroing.mean()),
            "sum_pre_zeroing_p05": float(np.percentile(sum_pre_zeroing, 5)),
            "sum_pre_zeroing_p95": float(np.percentile(sum_pre_zeroing, 95)),
            "sum_pre_enforce_mean": float(sum_pre_enforce.mean()),
            "sum_pre_enforce_p05": float(np.percentile(sum_pre_enforce, 5)),
            "sum_pre_enforce_p95": float(np.percentile(sum_pre_enforce, 95)),
            "sum_post_enforce_mean": float(sum_post_enforce.mean()),
            "sum_post_enforce_p05": float(np.percentile(sum_post_enforce, 5)),
            "sum_post_enforce_p95": float(np.percentile(sum_post_enforce, 95)),
            "sum_final_mean": float(sum_final.mean()),
            "sum_final_p05": float(np.percentile(sum_final, 5)),
            "sum_final_p95": float(np.percentile(sum_final, 95)),
            # Quality flags
            "frac_post_enforce_not_240": float((np.abs(sum_post_enforce - 240.0) > 1e-6).mean()),
            "frac_final_lt_240": float((sum_final < 240.0 - 1e-6).mean()),
            "frac_final_not_240": float((np.abs(sum_final - 240.0) > 1e-6).mean()),
            "frac_rotation_capped": float(rotation_capped.mean()),
            # Active player counts
            "active_players_mean": float(active_counts_per_world.mean()),
            "active_players_min": int(active_counts_per_world.min()),
            "active_players_max": int(active_counts_per_world.max()),
            # K_sim: count of players with minutes > 0 in final
            "K_sim_mean": float((minutes_final[:, player_idxs] > 0).sum(axis=1).mean()),
            "K_sim_ge_2_mean": float((minutes_final[:, player_idxs] >= 2.0).sum(axis=1).mean()),
            # Max minutes check
            "max_minutes_in_world": float(minutes_final[:, player_idxs].max()),
        })
    
    team_game_df = pd.DataFrame(team_game_records)
    
    # Compute per-player statistics across worlds
    sim_mean = np.mean(minutes_worlds, axis=0)
    sim_std = np.std(minutes_worlds, axis=0)
    sim_p10 = np.percentile(minutes_worlds, 10, axis=0)
    sim_p50 = np.percentile(minutes_worlds, 50, axis=0)
    sim_p90 = np.percentile(minutes_worlds, 90, axis=0)
    
    # Delta metrics
    delta = sim_mean - minutes_p50
    delta_abs = np.abs(delta)
    delta_pct = np.where(minutes_p50 > 1e-6, 100 * delta / minutes_p50, 0.0)
    
    # Per-team metrics
    team_sum_pred = np.zeros(n_players, dtype=float)
    team_sum_sim_mean = np.zeros(n_players, dtype=float)
    for t_idx, team_id in enumerate(unique_teams):
        mask = team_ids == team_id
        team_sum_pred[mask] = minutes_p50[mask].sum()
        team_sum_sim_mean[mask] = sim_mean[mask].sum()
    
    team_scale = np.where(team_sum_pred > 1e-6, 240.0 / team_sum_pred, 1.0)
    scaled_pred = minutes_p50 * team_scale
    residual = sim_mean - scaled_pred
    residual_abs = np.abs(residual)
    
    # Build result DataFrame
    result = pd.DataFrame({
        "game_date": minutes_df["game_date"].values,
        "game_id": game_ids,
        "team_id": team_ids,
        "player_id": player_ids,
        "player_name": minutes_df.get("player_name", ""),
        "minutes_pred_p50": minutes_p50,
        "minutes_pred_p10": minutes_p10,
        "minutes_pred_p90": minutes_p90,
        "play_prob": play_prob,
        "is_starter": is_starter,
        "minutes_sim_mean": sim_mean,
        "minutes_sim_std": sim_std,
        "minutes_sim_p10": sim_p10,
        "minutes_sim_p50": sim_p50,
        "minutes_sim_p90": sim_p90,
        "delta": delta,
        "delta_abs": delta_abs,
        "delta_pct": delta_pct,
        "team_sum_pred": team_sum_pred,
        "team_sum_sim_mean": team_sum_sim_mean,
        "team_scale": team_scale,
        "residual": residual,
        "residual_abs": residual_abs,
    })
    
    return result, team_game_df, prune_diagnostics


def _generate_report(
    results_df: pd.DataFrame,
    team_game_df: pd.DataFrame,
    *,
    n_samples: int,
    profile_name: str,
    start_date: str,
    end_date: str,
) -> str:
    """Generate markdown report from aggregated results."""
    lines = []
    lines.append("# Minutes Sim vs Pred Audit Report")
    lines.append("")
    lines.append(f"**Date Range**: {start_date} to {end_date}")
    lines.append(f"**Profile**: {profile_name}")
    lines.append(f"**Samples per game**: {n_samples}")
    lines.append(f"**Total rows**: {len(results_df):,}")
    lines.append(f"**Generated**: {datetime.now().isoformat()}")
    lines.append("")
    
    # Filter to rotation players only for main analysis
    rotation_df = results_df[results_df["minutes_pred_p50"] >= 12].copy()
    
    lines.append("---")
    lines.append("")
    lines.append("## 1. Overall Delta Distribution")
    lines.append("")
    lines.append("| Metric | All Players | Rotation (p50≥12) |")
    lines.append("|--------|-------------|-------------------|")
    
    for label, df in [("All", results_df), ("Rotation", rotation_df)]:
        if df.empty:
            continue
        delta_mean = df["delta"].mean()
        delta_p50 = df["delta"].median()
        delta_p90 = np.percentile(df["delta"], 90)
        abs_mean = df["delta_abs"].mean()
        abs_p95 = np.percentile(df["delta_abs"], 95)
        if label == "All":
            lines.append(f"| Delta mean | {delta_mean:.2f} | - |")
            lines.append(f"| Delta p50 | {delta_p50:.2f} | - |")
            lines.append(f"| Delta p90 | {delta_p90:.2f} | - |")
            lines.append(f"| Abs delta mean | {abs_mean:.2f} | - |")
            lines.append(f"| Abs delta p95 | {abs_p95:.2f} | - |")
        else:
            lines[-5] = lines[-5].rstrip(" |") + f" {delta_mean:.2f} |"
            lines[-4] = lines[-4].rstrip(" |") + f" {delta_p50:.2f} |"
            lines[-3] = lines[-3].rstrip(" |") + f" {delta_p90:.2f} |"
            lines[-2] = lines[-2].rstrip(" |") + f" {abs_mean:.2f} |"
            lines[-1] = lines[-1].rstrip(" |") + f" {abs_p95:.2f} |"
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Correlation Analysis")
    lines.append("")
    
    # Filter to players with positive predictions
    valid = results_df[results_df["minutes_pred_p50"] > 1].copy()
    if len(valid) > 10:
        corr_raw = np.corrcoef(valid["minutes_pred_p50"], valid["minutes_sim_mean"])[0, 1]
        corr_scaled = np.corrcoef(valid["minutes_pred_p50"] * valid["team_scale"], valid["minutes_sim_mean"])[0, 1]
        lines.append(f"- **corr(pred_p50, sim_mean)**: {corr_raw:.4f}")
        lines.append(f"- **corr(pred_p50 × team_scale, sim_mean)**: {corr_scaled:.4f}")
    else:
        lines.append("- Insufficient data for correlation analysis")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Scale-Only Fit Quality")
    lines.append("")
    
    if len(valid) > 10:
        residual_mean = valid["residual"].mean()
        residual_std = valid["residual"].std()
        residual_p95 = np.percentile(valid["residual_abs"], 95)
        lines.append(f"- **Residual mean**: {residual_mean:.3f} min")
        lines.append(f"- **Residual std**: {residual_std:.3f} min")
        lines.append(f"- **Residual abs p95**: {residual_p95:.3f} min")
        lines.append("")
        lines.append("> If residuals are near zero, enforce_240 is mostly scaling.")
        lines.append("> Large residuals indicate reshuffling beyond pure scaling.")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Per-Stage Team Sum Diagnostics")
    lines.append("")
    lines.append("> This section tracks where minutes are lost/gained in the pipeline.")
    lines.append("")
    
    if not team_game_df.empty:
        lines.append("### 4.1 Stage-by-Stage Team Sums (mean across worlds)")
        lines.append("")
        lines.append("| Stage | Mean | P05 | P95 |")
        lines.append("|-------|------|-----|-----|")
        
        for stage in ["pre_zeroing", "pre_enforce", "post_enforce", "final"]:
            mean_col = f"sum_{stage}_mean"
            p05_col = f"sum_{stage}_p05"
            p95_col = f"sum_{stage}_p95"
            if mean_col in team_game_df.columns:
                mean_val = team_game_df[mean_col].mean()
                p05_val = team_game_df[p05_col].mean()
                p95_val = team_game_df[p95_col].mean()
                lines.append(f"| {stage} | {mean_val:.1f} | {p05_val:.1f} | {p95_val:.1f} |")
        
        lines.append("")
        lines.append("### 4.2 Enforcement Quality Flags")
        lines.append("")
        
        frac_not_240 = team_game_df["frac_post_enforce_not_240"].mean()
        frac_lt_240 = team_game_df["frac_final_lt_240"].mean()
        frac_capped = team_game_df["frac_rotation_capped"].mean()
        
        lines.append(f"- **Fraction of worlds where post_enforce ≠ 240**: {frac_not_240:.1%}")
        lines.append(f"- **Fraction of worlds where final < 240**: {frac_lt_240:.1%}")
        lines.append(f"- **Fraction of worlds where rotation cap applied (>10 active)**: {frac_capped:.1%}")
        lines.append("")
        
        if frac_not_240 > 0.01:
            lines.append("> ⚠️ enforce_240 is NOT hitting 240 target in some worlds.")
            lines.append("> Check clamp_scale bounds and rotation cap logic.")
        elif frac_lt_240 > frac_not_240 + 0.01:
            lines.append("> ⚠️ Minutes are being lost AFTER enforce_240.")
            lines.append("> This suggests additional zeroing/cutoffs post-enforcement.")
        else:
            lines.append("> ✅ enforce_240 is working as expected for most worlds.")
        
        lines.append("")
        lines.append("### 4.3 Active Player Counts")
        lines.append("")
        active_mean = team_game_df["active_players_mean"].mean()
        active_min = team_game_df["active_players_min"].min()
        active_max = team_game_df["active_players_max"].max()
        lines.append(f"- **Active players per team (mean)**: {active_mean:.1f}")
        lines.append(f"- **Active players range**: [{active_min}, {active_max}]")
        lines.append("")
        lines.append(f"- **Max rotation size**: {DEFAULT_MAX_ROTATION_SIZE}")
        if active_mean > DEFAULT_MAX_ROTATION_SIZE:
            lines.append(f"> ⚠️ Average active players ({active_mean:.1f}) exceeds rotation cap ({DEFAULT_MAX_ROTATION_SIZE}).")
            lines.append("> This explains why team sums < 240 after rotation capping.")
    else:
        lines.append("No team-game data available.")
    
    lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("## 5. Breakdown by Starter/Bench")
    lines.append("")
    lines.append("| Role | Count | Delta Mean | Delta Abs Mean | Residual Abs P95 |")
    lines.append("|------|-------|------------|----------------|------------------|")
    
    for role, role_mask in [("Starter", results_df["is_starter"] == 1), ("Bench", results_df["is_starter"] == 0)]:
        subset = results_df[role_mask & (results_df["minutes_pred_p50"] > 1)]
        if len(subset) > 0:
            delta_mean = subset["delta"].mean()
            delta_abs_mean = subset["delta_abs"].mean()
            res_abs_p95 = np.percentile(subset["residual_abs"], 95)
            lines.append(f"| {role} | {len(subset):,} | {delta_mean:+.2f} | {delta_abs_mean:.2f} | {res_abs_p95:.2f} |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Breakdown by Minutes Bucket")
    lines.append("")
    lines.append("| Bucket | Count | Delta Mean | Residual Mean | Residual P95 |")
    lines.append("|--------|-------|------------|---------------|--------------|")
    
    buckets = [(0, 10), (10, 20), (20, 30), (30, 48)]
    for lo, hi in buckets:
        subset = results_df[(results_df["minutes_pred_p50"] >= lo) & (results_df["minutes_pred_p50"] < hi)]
        if len(subset) > 0:
            delta_mean = subset["delta"].mean()
            res_mean = subset["residual"].mean()
            res_p95 = np.percentile(subset["residual_abs"], 95)
            lines.append(f"| {lo}-{hi} | {len(subset):,} | {delta_mean:+.2f} | {res_mean:+.2f} | {res_p95:.2f} |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 7. Top 20 Largest Absolute Deltas")
    lines.append("")
    lines.append("| Player | Team | Pred P50 | Sim Mean | Delta |")
    lines.append("|--------|------|----------|----------|-------|")
    
    top_delta = results_df.nlargest(20, "delta_abs")
    for _, row in top_delta.iterrows():
        lines.append(f"| {row['player_name'][:20]} | {int(row['team_id'])} | {row['minutes_pred_p50']:.1f} | {row['minutes_sim_mean']:.1f} | {row['delta']:+.1f} |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 8. Top 20 Largest Scale Residuals")
    lines.append("")
    lines.append("| Player | Team | Pred×Scale | Sim Mean | Residual |")
    lines.append("|--------|------|------------|----------|----------|")
    
    top_residual = results_df.nlargest(20, "residual_abs")
    for _, row in top_residual.iterrows():
        scaled = row["minutes_pred_p50"] * row["team_scale"]
        lines.append(f"| {row['player_name'][:20]} | {int(row['team_id'])} | {scaled:.1f} | {row['minutes_sim_mean']:.1f} | {row['residual']:+.1f} |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 9. Conclusions")
    lines.append("")
    
    # Compute key metrics for conclusions
    if len(valid) > 10:
        corr_raw = np.corrcoef(valid["minutes_pred_p50"], valid["minutes_sim_mean"])[0, 1]
        residual_p95 = np.percentile(valid["residual_abs"], 95)
        
        lines.append(f"1. **Correlation**: pred_p50 → sim_mean correlation is **{corr_raw:.3f}**")
        
        if corr_raw > 0.95:
            lines.append("   - Very high correlation suggests enforce_240 preserves relative ordering well.")
        elif corr_raw > 0.85:
            lines.append("   - High correlation, but some reshuffling occurs within teams.")
        else:
            lines.append("   - Lower correlation suggests significant reshuffling.")
        
        lines.append("")
        lines.append(f"2. **Residual P95**: {residual_p95:.2f} min after scaling")
        
        if residual_p95 < 2.0:
            lines.append("   - enforce_240 is **mostly pure scaling** with minor reshuffling.")
            lines.append("   - Training usage on pred_p50 should generalize well to sim.")
        elif residual_p95 < 5.0:
            lines.append("   - enforce_240 involves **moderate reshuffling** beyond scaling.")
            lines.append("   - May see some drift between training and sim minutes distribution.")
        else:
            lines.append("   - enforce_240 involves **significant reshuffling**.")
            lines.append("   - Consider training usage on simulated minutes or adding noise.")
        
        lines.append("")
        
        # Compute sum_mean from team_game_df if available, else from results_df
        if not team_game_df.empty:
            sum_mean = team_game_df["sum_final_mean"].mean()
        else:
            team_sums = results_df.groupby(["game_date", "game_id", "team_id"])["minutes_sim_mean"].sum()
            sum_mean = team_sums.mean() if len(team_sums) > 0 else 0
        
        lines.append(f"3. **Team Sum Accuracy**: mean = {sum_mean:.1f} (target 240)")
        if abs(sum_mean - 240) < 1:
            lines.append("   - ✅ Team totals are correctly enforced to ~240.")
        else:
            lines.append(f"   - ⚠️ Team totals deviate from 240 by {abs(sum_mean - 240):.1f} min on average.")
    
    lines.append("")
    
    return "\n".join(lines)


@app.command()
def main(
    start_date: str = typer.Option(..., help="Start date YYYY-MM-DD (inclusive)"),
    end_date: str = typer.Option(..., help="End date YYYY-MM-DD (inclusive)"),
    data_root: Optional[Path] = typer.Option(None, help="Data root (defaults to PROJECTIONS_DATA_ROOT)"),
    profile: str = typer.Option("baseline", help="Sim v2 profile name"),
    num_samples: int = typer.Option(300, "--num-samples", min=50, max=2000, help="Worlds per game"),
    seed: int = typer.Option(123, help="Random seed for reproducibility"),
    max_dates: Optional[int] = typer.Option(None, "--max-dates", help="Optional cap on dates to process"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Output directory"),
    skip_game_scripts: bool = typer.Option(False, "--skip-game-scripts", help="Use simple sampling instead of game scripts"),
    rotation_minutes_floor: float = typer.Option(0.0, "--rotation-floor", help="Floor prune players with < this many minutes"),
    max_rotation_size: Optional[int] = typer.Option(None, "--max-rotation", help="Max rotation size (None=legacy 10)"),
) -> None:
    root = data_root or data_path()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    
    # Load profile
    try:
        profile_cfg = load_sim_v2_profile(profile=profile)
        enforce_240 = profile_cfg.enforce_team_240
        typer.echo(f"[audit] loaded profile '{profile}' (enforce_240={enforce_240})")
    except Exception as e:
        typer.echo(f"[audit] warning: could not load profile '{profile}': {e}; using defaults", err=True)
        enforce_240 = True
    
    # Game script config
    game_script_config: GameScriptConfig | None = None
    if not skip_game_scripts:
        try:
            if hasattr(profile_cfg, "game_script") and profile_cfg.game_script:
                game_script_config = GameScriptConfig.from_profile_config(profile_cfg.game_script)
                typer.echo("[audit] using game script sampling")
        except Exception:
            pass
    if game_script_config is None:
        game_script_config = GameScriptConfig()  # Use defaults
        typer.echo("[audit] using default game script config")
    
    # Setup RNG
    rng = np.random.default_rng(seed)
    typer.echo(f"[audit] seed={seed}, num_samples={num_samples}")
    typer.echo(f"[audit] rotation_minutes_floor={rotation_minutes_floor}, max_rotation_size={max_rotation_size}")
    
    # Output directory
    run_ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    if out_dir is None:
        out_dir = root / "artifacts" / "diagnostics" / "minutes_sim_vs_pred" / f"run={run_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Iterate dates
    all_results: list[pd.DataFrame] = []
    all_team_game: list[pd.DataFrame] = []
    current = start
    dates_processed = 0
    while current <= end:
        if max_dates is not None and dates_processed >= max_dates:
            typer.echo(f"[audit] reached max_dates={max_dates}, stopping")
            break
        
        minutes_df = _load_minutes_for_date(root, current)
        if minutes_df is None or minutes_df.empty:
            typer.echo(f"[audit] {current.date()}: no minutes projections, skipping")
            current += pd.Timedelta(days=1)
            continue
        
        schedule_df = _load_schedule_for_date(root, current)
        
        typer.echo(f"[audit] {current.date()}: {len(minutes_df)} players...")
        
        result, team_game, prune_diag = _simulate_minutes_for_date(
            minutes_df,
            schedule_df,
            n_samples=num_samples,
            game_script_config=game_script_config,
            enforce_240=enforce_240,
            rng=rng,
            rotation_minutes_floor=rotation_minutes_floor,
            max_rotation_size=max_rotation_size,
        )
        
        if not result.empty:
            all_results.append(result)
            all_team_game.append(team_game)
            dates_processed += 1
        
        current += pd.Timedelta(days=1)
    
    if not all_results:
        typer.echo("[audit] no data processed", err=True)
        raise typer.Exit(1)
    
    # Combine results
    combined = pd.concat(all_results, ignore_index=True)
    combined_team_game = pd.concat(all_team_game, ignore_index=True) if all_team_game else pd.DataFrame()
    
    # Write parquets
    parquet_path = out_dir / "minutes_sim_vs_pred.parquet"
    combined.to_parquet(parquet_path, index=False)
    typer.echo(f"[audit] wrote {len(combined):,} rows to {parquet_path}")
    
    if not combined_team_game.empty:
        team_game_path = out_dir / "team_game_stage_diagnostics.parquet"
        combined_team_game.to_parquet(team_game_path, index=False)
        typer.echo(f"[audit] wrote {len(combined_team_game):,} team-game rows to {team_game_path}")
    
    # Generate report  
    report = _generate_report(
        combined,
        combined_team_game,
        n_samples=num_samples,
        profile_name=profile,
        start_date=start_date,
        end_date=end_date,
    )
    
    # Write report to docs/diagnostics
    docs_dir = Path("docs/diagnostics")
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_path = docs_dir / "minutes_sim_vs_pred_audit.md"
    report_path.write_text(report, encoding="utf-8")
    typer.echo(f"[audit] wrote report to {report_path}")
    
    # Also write to output directory
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    
    # Print summary
    typer.echo("")
    typer.echo("=" * 60)
    typer.echo("SUMMARY")
    typer.echo("=" * 60)
    
    valid = combined[combined["minutes_pred_p50"] > 1]
    if len(valid) > 10:
        corr = np.corrcoef(valid["minutes_pred_p50"], valid["minutes_sim_mean"])[0, 1]
        residual_p95 = np.percentile(valid["residual_abs"], 95)
        delta_abs_mean = valid["delta_abs"].mean()
        
        team_sums = combined.groupby(["game_date", "game_id", "team_id"])["minutes_sim_mean"].sum()
        sum_mean = team_sums.mean()
        
        typer.echo(f"  corr(pred_p50, sim_mean) = {corr:.4f}")
        typer.echo(f"  delta_abs_mean = {delta_abs_mean:.2f} min")
        typer.echo(f"  residual_abs_p95 = {residual_p95:.2f} min")
        typer.echo(f"  team_sum_mean = {sum_mean:.1f} (target: 240)")
    
    typer.echo(f"  dates_processed = {dates_processed}")
    typer.echo(f"  parquet: {parquet_path}")
    typer.echo(f"  report: {report_path}")


if __name__ == "__main__":
    app()
