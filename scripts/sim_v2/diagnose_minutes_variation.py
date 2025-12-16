#!/usr/bin/env python
"""
Deeper analysis: Why do game script means hover near p50?
"""

import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter

from projections.sim_v2.game_script import (
    GameScriptConfig,
    classify_script,
    sample_minutes_with_scripts,
)


def analyze_script_distribution():
    """Analyze what scripts are being sampled and their effect."""
    
    config = GameScriptConfig()
    
    # Simulate 1000 worlds for a team favored by 5 points
    spread_home = -5.0  # Home team favored
    team_spread = spread_home  # This is the home team
    mean_margin = config.spread_coef * team_spread  # = -0.726 * -5 = +3.63
    
    print("=" * 70)
    print("ANALYSIS: Script Distribution for Home Team Favored by 5")
    print("=" * 70)
    print(f"\nSpread (home): {spread_home}")
    print(f"Expected margin: {mean_margin:.2f} (team expected to win by ~3.6 pts)")
    print(f"Margin std: {config.margin_std} (huge uncertainty!)")
    
    # Sample margins
    n_worlds = 10000
    rng = np.random.default_rng(42)
    margins = rng.normal(mean_margin, config.margin_std, size=n_worlds)
    
    # Classify each margin
    scripts = [classify_script(m, config) for m in margins]
    script_counts = Counter(scripts)
    
    print(f"\n{'Script Distribution across {n_worlds} worlds:'}")
    print("-" * 50)
    for script, count in sorted(script_counts.items(), key=lambda x: -x[1]):
        pct = count / n_worlds * 100
        starter_q, bench_q = config.quantile_targets[script]
        print(f"  {script:20s}: {count:5d} ({pct:5.1f}%) -> starter_q={starter_q:.2f}, bench_q={bench_q:.2f}")
    
    # Calculate expected quantile for starters
    print("\n" + "=" * 70)
    print("EXPECTED QUANTILE ANALYSIS")
    print("=" * 70)
    
    expected_starter_q = 0.0
    expected_bench_q = 0.0
    for script, count in script_counts.items():
        weight = count / n_worlds
        starter_q, bench_q = config.quantile_targets[script]
        expected_starter_q += weight * starter_q
        expected_bench_q += weight * bench_q
    
    print(f"\nWeighted average target quantile:")
    print(f"  Starters: {expected_starter_q:.3f} (vs 0.50 baseline)")
    print(f"  Bench:    {expected_bench_q:.3f} (vs 0.50 baseline)")
    
    # Convert quantile shift to z-score and minute shift
    z_starter = stats.norm.ppf(expected_starter_q)
    z_bench = stats.norm.ppf(expected_bench_q)
    z_baseline = stats.norm.ppf(0.5)  # = 0
    
    print(f"\nZ-score shift from baseline:")
    print(f"  Starters: {z_starter:.3f} (shift of {z_starter - z_baseline:.3f})")  
    print(f"  Bench:    {z_bench:.3f} (shift of {z_bench - z_baseline:.3f})")
    
    # For a player with p10=19.2, p50=32, p90=44.8
    p50 = 32.0
    p10 = p50 * 0.6
    p90 = p50 * 1.4
    sigma = (p90 - p10) / 2.56
    
    expected_mins_starter = p50 + sigma * z_starter
    expected_mins_bench = p50 + sigma * z_bench
    
    print(f"\nFor a player with p50={p50:.0f}min (sigma={sigma:.1f}min):")
    print(f"  Expected with game scripts: {expected_mins_starter:.1f}min")
    print(f"  Shift from p50: {expected_mins_starter - p50:+.1f}min")
    
    print("\n" + "=" * 70)
    print("THE PROBLEM: Scripts are roughly symmetric around 'close'")
    print("=" * 70)
    print("""
For a 5-point favorite:
- ~40% of worlds are 'close' (starter_q=0.65, +minutes)
- ~20% of worlds are 'comfortable_win' (starter_q=0.45, -minutes)
- ~10% of worlds are 'comfortable_loss' (starter_q=0.50, neutral)
- ~15% of worlds are 'blowout_win' (starter_q=0.35, -minutes)
- These mostly CANCEL OUT!

The current quantile targets balance positive and negative shifts.
""")
    
    print("=" * 70)
    print("SOLUTION OPTIONS")
    print("=" * 70)
    print("""
1. INTENDED BEHAVIOR? 
   - Game scripts add VARIANCE but shouldn't shift the mean much
   - Each world varies, but average is near p50 by design
   - FPTS distributions get wider, not shifted

2. TO SHIFT THE MEAN MORE:
   - Increase quantile targets asymmetrically
   - e.g., close: starter_q=0.75 (instead of 0.65)
   - e.g., blowout_win: starter_q=0.25 (instead of 0.35)

3. ALTERNATIVE APPROACH:
   - Add a "tilt" based on spread
   - Favorites get consistent small boost (e.g., +2 mins for starters)
   - Underdogs get consistent small reduction
""")
    
    return expected_starter_q, expected_bench_q


def show_within_script_behavior():
    """Show that within each script, minutes DO shift significantly."""
    
    print("\n" + "=" * 70)
    print("WITHIN-SCRIPT ANALYSIS: Minutes DO shift within each game type")
    print("=" * 70)
    
    config = GameScriptConfig()
    
    # For a starter with p50=32
    p50 = 32.0
    p10 = p50 * 0.6  # 19.2
    p90 = p50 * 1.4  # 44.8
    sigma = (p90 - p10) / 2.56  # ~10 min
    
    print(f"\nFor starter with p50={p50:.0f}min, p10={p10:.1f}min, p90={p90:.1f}min:")
    print(f"(Estimated sigma={sigma:.1f}min)")
    print()
    print(f"{'Script':<20} {'Target Q':>10} {'Z-score':>10} {'Minutes':>10} {'vs p50':>10}")
    print("-" * 65)
    
    for script, (starter_q, bench_q) in config.quantile_targets.items():
        z = stats.norm.ppf(starter_q)
        mins = p50 + sigma * z
        shift = mins - p50
        print(f"{script:<20} {starter_q:>10.2f} {z:>10.2f} {mins:>10.1f} {shift:>+10.1f}")
    
    print(f"\n➜ Close games: starters get +3.8 mins")
    print(f"➜ Blowout wins: starters get -3.8 mins")
    print(f"➜ But these CANCEL OUT when averaging across all worlds!")


if __name__ == "__main__":
    analyze_script_distribution()
    show_within_script_behavior()
    
    print("\n" + "=" * 70)
    print("BOTTOM LINE")
    print("=" * 70)
    print("""
The current game script implementation:
✓ DOES create per-world variation (some worlds favor starters, some don't)
✓ DOES widen the FPTS distribution (reflecting game outcome uncertainty)  
✗ Does NOT shift the MEAN significantly (positive/negative scripts cancel)

Is this the intended behavior? If you want the mean to shift:
- For favorites: starters should expect FEWER minutes (blowouts more likely)
- For underdogs: starters should expect MORE minutes (close games more likely)

Current config seems designed for variance, not mean shift.
""")
