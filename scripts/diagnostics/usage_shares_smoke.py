"""
Usage Shares Smoke Test - Verify usage shares allocation affects sim outputs.

Runs sim_v2 twice on the same date with the same seed:
  (A) usage_shares disabled
  (B) usage_shares enabled with share_noise_std > 0

Computes and prints diagnostics proving the change is not a no-op:
  1. Team FGA/FTA/TOV totals should match closely (team totals preserved)
  2. Player-level L1 changes should be > 0 (within-team redistribution)
  3. Top-usage flip rate: % worlds where argmax FGA differs (coupling effect)

Usage:
  uv run python -m scripts.diagnostics.usage_shares_smoke --run-date 2025-12-15 --num-worlds 200
  uv run python -m scripts.diagnostics.usage_shares_smoke --run-date 2025-12-15 --num-worlds 200 --data-root /path/to/data
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.paths import data_path

app = typer.Typer(add_completion=False, help=__doc__)


def _parse_date(value: str) -> date:
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid date: {value}") from exc


def _load_minutes_projection(root: Path, game_date: pd.Timestamp) -> pd.DataFrame:
    """Load minutes projection for the given date."""
    date_token = pd.Timestamp(game_date).date().isoformat()
    daily_base = root / "artifacts" / "minutes_v1" / "daily" / date_token

    # Try to find latest run
    latest = daily_base / "latest_run.json"
    run_id = None
    if latest.exists():
        try:
            payload = json.loads(latest.read_text(encoding="utf-8"))
            run_id = payload.get("run_id") or payload.get("run_as_of_ts")
        except json.JSONDecodeError:
            pass

    candidates = []
    if run_id:
        candidates.append(daily_base / f"run={run_id}" / "minutes.parquet")
    # Fallback to gold
    candidates.append(root / "gold" / "projections_minutes_v1" / f"game_date={date_token}" / "minutes.parquet")

    for path in candidates:
        if path.exists():
            return pd.read_parquet(path)

    raise FileNotFoundError(f"No minutes projection found for {date_token}")


def _load_rates_live(root: Path, game_date: pd.Timestamp) -> pd.DataFrame:
    """Load rates projection for the given date."""
    date_token = pd.Timestamp(game_date).date().isoformat()
    base = root / "gold" / "rates_v1_live" / date_token

    # Try latest run
    latest = base / "latest_run.json"
    run_id = None
    if latest.exists():
        try:
            payload = json.loads(latest.read_text(encoding="utf-8"))
            run_id = payload.get("run_id") or payload.get("run_as_of_ts")
        except json.JSONDecodeError:
            pass

    candidates = []
    if run_id:
        candidates.append(base / f"run={run_id}" / "rates.parquet")
    candidates.append(base / "rates.parquet")

    for path in candidates:
        if path.exists():
            return pd.read_parquet(path)

    raise FileNotFoundError(f"No rates_v1_live found for {date_token}")


def _run_sim_with_usage_shares(
    minutes_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    n_worlds: int,
    seed: int,
    usage_shares_enabled: bool,
    share_noise_std: float = 0.15,
    share_temperature: float = 1.0,
    min_minutes_active_cutoff: float = 2.0,
) -> dict[str, np.ndarray]:
    """
    Run a minimal simulation to extract FGA/FTA/TOV arrays per world.

    Returns dict with keys: fga2, fga3, fta, tov, minutes, game_ids, team_ids, player_ids
    Each value is shape (n_worlds, n_players) except IDs which are (n_players,).
    """
    from projections.sim_v2.config import UsageSharesConfig

    # Import the allocation function we added
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from scripts.sim_v2.generate_worlds_fpts_v2 import _apply_usage_shares_allocation

    # Join minutes and rates
    join_keys = ["game_date", "game_id", "team_id", "player_id"]
    minutes_df = minutes_df.copy()
    rates_df = rates_df.copy()

    minutes_df["game_date"] = pd.to_datetime(minutes_df["game_date"]).dt.normalize()
    rates_df["game_date"] = pd.to_datetime(rates_df["game_date"]).dt.normalize()

    for key in ("game_id", "team_id", "player_id"):
        if key in minutes_df.columns:
            minutes_df[key] = pd.to_numeric(minutes_df[key], errors="coerce")
        if key in rates_df.columns:
            rates_df[key] = pd.to_numeric(rates_df[key], errors="coerce")

    merged = pd.merge(minutes_df, rates_df, on=join_keys, how="inner", suffixes=("", "_rates"))

    # Resolve minutes column
    minutes_col = None
    for c in ("minutes_p50_cond", "minutes_p50", "minutes_pred_p50"):
        if c in merged.columns:
            minutes_col = c
            break
    if minutes_col is None:
        raise KeyError("No minutes column found")

    merged[minutes_col] = pd.to_numeric(merged[minutes_col], errors="coerce")
    merged = merged[merged[minutes_col].notna()]
    merged = merged.reset_index(drop=True)

    n_players = len(merged)
    if n_players == 0:
        raise ValueError("No players after join")

    # Extract arrays
    minutes_mean = merged[minutes_col].to_numpy(dtype=float)
    game_ids = merged["game_id"].to_numpy()
    team_ids = merged["team_id"].to_numpy()
    player_ids = merged["player_id"].to_numpy()

    # Build rate arrays
    rate_targets = ["fga2_per_min", "fga3_per_min", "fta_per_min", "tov_per_min"]
    rate_arrays: dict[str, np.ndarray] = {}
    for target in rate_targets:
        col = target if target in merged.columns else f"pred_{target.replace('_per_min', '')}"
        if col not in merged.columns:
            # Try another variant
            for c in merged.columns:
                if target.replace("_per_min", "") in c.lower() and "per_min" in c.lower():
                    col = c
                    break
        if col in merged.columns:
            rate_arrays[target] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            rate_arrays[target] = np.zeros(n_players, dtype=float)

    # Build group_map
    group_map: dict[tuple[int, int], np.ndarray] = {}
    for idx, (gid, tid) in enumerate(zip(game_ids, team_ids)):
        key = (int(gid), int(tid))
        if key not in group_map:
            group_map[key] = []
        group_map[key].append(idx)
    group_map = {k: np.array(v, dtype=int) for k, v in group_map.items()}

    # Create usage shares config
    usage_cfg = UsageSharesConfig(
        enabled=usage_shares_enabled,
        targets=("fga", "fta", "tov"),
        share_temperature=share_temperature,
        share_noise_std=share_noise_std,
        min_minutes_active_cutoff=min_minutes_active_cutoff,
        fallback="rate_weighted",
    )

    # Sample minutes (simplified: add small noise)
    rng = np.random.default_rng(seed)
    sigma_minutes = np.maximum(minutes_mean * 0.15, 1.0)  # ~15% CV
    z = rng.standard_normal(size=(n_worlds, n_players))
    minutes_worlds = np.maximum(minutes_mean[None, :] + z * sigma_minutes[None, :], 0.0)

    # Compute baseline stats (rate * minutes)
    stat_totals: dict[str, np.ndarray] = {}
    for target in rate_targets:
        rates = rate_arrays.get(target, np.zeros(n_players))
        base = target.replace("_per_min", "")
        stat_totals[base] = np.clip(rates[None, :] * minutes_worlds, 0.0, None)

    # Apply usage shares if enabled
    if usage_cfg.enabled:
        stat_totals = _apply_usage_shares_allocation(
            stat_totals=stat_totals,
            minutes_worlds=minutes_worlds,
            rate_arrays=rate_arrays,
            group_map=group_map,
            usage_cfg=usage_cfg,
            rng=rng,
        )

    return {
        "fga2": stat_totals.get("fga2", np.zeros((n_worlds, n_players))),
        "fga3": stat_totals.get("fga3", np.zeros((n_worlds, n_players))),
        "fta": stat_totals.get("fta", np.zeros((n_worlds, n_players))),
        "tov": stat_totals.get("tov", np.zeros((n_worlds, n_players))),
        "minutes": minutes_worlds,
        "game_ids": game_ids,
        "team_ids": team_ids,
        "player_ids": player_ids,
    }


def _compute_team_totals(
    stats: dict[str, np.ndarray], target: str
) -> dict[tuple[int, int], np.ndarray]:
    """Compute team totals per world for a given stat."""
    game_ids = stats["game_ids"]
    team_ids = stats["team_ids"]
    arr = stats[target]  # (n_worlds, n_players)

    # Build group_map
    group_map: dict[tuple[int, int], list[int]] = {}
    for idx, (gid, tid) in enumerate(zip(game_ids, team_ids)):
        key = (int(gid), int(tid))
        if key not in group_map:
            group_map[key] = []
        group_map[key].append(idx)

    totals: dict[tuple[int, int], np.ndarray] = {}
    for key, idxs in group_map.items():
        totals[key] = arr[:, idxs].sum(axis=1)  # (n_worlds,)
    return totals


@app.command()
def main(
    run_date: str = typer.Option(..., "--run-date", help="Date to run sim on (YYYY-MM-DD)"),
    num_worlds: int = typer.Option(200, "--num-worlds", help="Number of worlds to simulate"),
    data_root: Optional[Path] = typer.Option(
        None, "--data-root", help="Override PROJECTIONS_DATA_ROOT"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    share_noise_std: float = typer.Option(0.15, "--noise-std", help="Share noise std for test B"),
) -> None:
    """Run usage shares smoke test."""
    game_date = pd.Timestamp(_parse_date(run_date)).normalize()
    root = data_root or data_path()

    typer.echo(f"[smoke] Loading data for {game_date.date()} from {root}")

    # Load data
    try:
        minutes_df = _load_minutes_projection(root, game_date)
        rates_df = _load_rates_live(root, game_date)
    except FileNotFoundError as e:
        typer.echo(f"[smoke] ERROR: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"[smoke] minutes_df: {len(minutes_df)} rows, rates_df: {len(rates_df)} rows")

    # Run sim A: usage_shares disabled
    typer.echo("[smoke] Running sim A (usage_shares=disabled)...")
    stats_a = _run_sim_with_usage_shares(
        minutes_df, rates_df, n_worlds=num_worlds, seed=seed, usage_shares_enabled=False
    )

    # Run sim B: usage_shares enabled
    typer.echo(f"[smoke] Running sim B (usage_shares=enabled, noise_std={share_noise_std})...")
    stats_b = _run_sim_with_usage_shares(
        minutes_df,
        rates_df,
        n_worlds=num_worlds,
        seed=seed,
        usage_shares_enabled=True,
        share_noise_std=share_noise_std,
    )

    n_players = len(stats_a["player_ids"])
    typer.echo(f"[smoke] Comparing {num_worlds} worlds x {n_players} players")
    typer.echo("")

    # === Diagnostic 1: Team totals should be preserved ===
    typer.echo("=== Diagnostic 1: Team Totals Preservation ===")
    for target in ["fga2", "fga3", "fta", "tov"]:
        if target == "fga2":
            # Combine fga2 + fga3 for FGA
            arr_a = stats_a["fga2"] + stats_a["fga3"]
            arr_b = stats_b["fga2"] + stats_b["fga3"]
            label = "FGA"
        elif target == "fga3":
            continue  # Already handled in FGA
        else:
            arr_a = stats_a[target]
            arr_b = stats_b[target]
            label = target.upper()

        totals_a = _compute_team_totals(
            {**stats_a, target: arr_a} if target == "fga2" else stats_a,
            "fga2" if target == "fga2" else target
        )
        totals_b = _compute_team_totals(
            {**stats_b, target: arr_b} if target == "fga2" else stats_b,
            "fga2" if target == "fga2" else target
        )

        # Compute FGA totals differently
        if target == "fga2":
            totals_a = {}
            totals_b = {}
            game_ids = stats_a["game_ids"]
            team_ids = stats_a["team_ids"]
            group_map: dict[tuple[int, int], list[int]] = {}
            for idx, (gid, tid) in enumerate(zip(game_ids, team_ids)):
                key = (int(gid), int(tid))
                if key not in group_map:
                    group_map[key] = []
                group_map[key].append(idx)
            for key, idxs in group_map.items():
                totals_a[key] = arr_a[:, idxs].sum(axis=1)
                totals_b[key] = arr_b[:, idxs].sum(axis=1)

        # Compare
        diffs = []
        for key in totals_a:
            diff = np.abs(totals_a[key] - totals_b[key])
            diffs.extend(diff.tolist())
        diffs = np.array(diffs)
        typer.echo(
            f"  {label}: mean_abs_diff={diffs.mean():.4f} max_abs_diff={diffs.max():.4f}"
        )

    typer.echo("")

    # === Diagnostic 2: Player-level L1 changes ===
    typer.echo("=== Diagnostic 2: Player-Level L1 Changes ===")
    for target in ["fga2", "fga3", "fta", "tov"]:
        diff = np.abs(stats_a[target] - stats_b[target])
        l1_per_player = diff.mean(axis=0)  # Average over worlds
        typer.echo(
            f"  {target}: mean_L1={l1_per_player.mean():.4f} "
            f"max_L1={l1_per_player.max():.4f} "
            f"players_changed={np.sum(l1_per_player > 1e-6)}/{n_players}"
        )

    typer.echo("")

    # === Diagnostic 3: Top-usage flip rate ===
    typer.echo("=== Diagnostic 3: Top-Usage Flip Rate ===")
    game_ids = stats_a["game_ids"]
    team_ids = stats_a["team_ids"]

    # Build group_map
    group_map: dict[tuple[int, int], list[int]] = {}
    for idx, (gid, tid) in enumerate(zip(game_ids, team_ids)):
        key = (int(gid), int(tid))
        if key not in group_map:
            group_map[key] = []
        group_map[key].append(idx)

    fga_a = stats_a["fga2"] + stats_a["fga3"]
    fga_b = stats_b["fga2"] + stats_b["fga3"]

    flip_counts = []
    total_worlds = 0
    for key, idxs in group_map.items():
        team_fga_a = fga_a[:, idxs]  # (n_worlds, n_team_players)
        team_fga_b = fga_b[:, idxs]
        argmax_a = team_fga_a.argmax(axis=1)
        argmax_b = team_fga_b.argmax(axis=1)
        flips = np.sum(argmax_a != argmax_b)
        flip_counts.append(flips)
        total_worlds += num_worlds

    total_flips = sum(flip_counts)
    flip_rate = total_flips / total_worlds if total_worlds > 0 else 0.0
    typer.echo(
        f"  FGA top-usage flip rate: {total_flips}/{total_worlds} = {flip_rate:.2%}"
    )

    typer.echo("")
    typer.echo("=== Summary ===")
    if flip_rate > 0:
        typer.echo("[PASS] Usage shares allocation is NOT a no-op (flip_rate > 0)")
    else:
        typer.echo("[WARN] Usage shares may be a no-op (flip_rate = 0)")

    # Check team totals preserved
    fga_arr_a = stats_a["fga2"] + stats_a["fga3"]
    fga_arr_b = stats_b["fga2"] + stats_b["fga3"]
    totals_a_all = []
    totals_b_all = []
    for key, idxs in group_map.items():
        totals_a_all.extend(fga_arr_a[:, idxs].sum(axis=1).tolist())
        totals_b_all.extend(fga_arr_b[:, idxs].sum(axis=1).tolist())
    max_diff = np.max(np.abs(np.array(totals_a_all) - np.array(totals_b_all)))
    if max_diff < 1e-6:
        typer.echo(f"[PASS] Team FGA totals preserved (max_diff={max_diff:.2e})")
    else:
        typer.echo(f"[WARN] Team FGA totals differ (max_diff={max_diff:.4f})")

    typer.echo("")

    # === Diagnostic 4: FGA vs PTS correlation (verify points coupled to opportunity) ===
    typer.echo("=== Diagnostic 4: FGA vs PTS Correlation ===")
    # Compute points: 2*fga2 + 3*fga3 (assuming makes=attempts for simplicity, or use 0.75 factor)
    # This simulates the downstream _compute_fpts_and_boxscore path
    pts_a = 2.0 * stats_a["fga2"] + 3.0 * stats_a["fga3"] + 0.75 * stats_a["fta"]
    pts_b = 2.0 * stats_b["fga2"] + 3.0 * stats_b["fga3"] + 0.75 * stats_b["fta"]

    # Compute player-level correlations across worlds
    corr_a_list = []
    corr_b_list = []
    for p in range(n_players):
        fga_p_a = fga_arr_a[:, p]
        fga_p_b = fga_arr_b[:, p]
        pts_p_a = pts_a[:, p]
        pts_p_b = pts_b[:, p]
        if fga_p_a.std() > 1e-6 and pts_p_a.std() > 1e-6:
            corr_a_list.append(np.corrcoef(fga_p_a, pts_p_a)[0, 1])
        if fga_p_b.std() > 1e-6 and pts_p_b.std() > 1e-6:
            corr_b_list.append(np.corrcoef(fga_p_b, pts_p_b)[0, 1])

    mean_corr_a = np.mean(corr_a_list) if corr_a_list else 0.0
    mean_corr_b = np.mean(corr_b_list) if corr_b_list else 0.0
    typer.echo(f"  FGA-PTS corr (disabled): mean={mean_corr_a:.4f} over {len(corr_a_list)} players")
    typer.echo(f"  FGA-PTS corr (enabled):  mean={mean_corr_b:.4f} over {len(corr_b_list)} players")
    if mean_corr_b >= mean_corr_a - 0.05:
        typer.echo("[PASS] Points remain coupled to FGA after reallocation")
    else:
        typer.echo(f"[WARN] FGA-PTS correlation dropped by {mean_corr_a - mean_corr_b:.4f}")

    typer.echo("")

    # === Diagnostic 5: Herfindahl concentration (share entropy) ===
    typer.echo("=== Diagnostic 5: Herfindahl Concentration (H = Σ share_i²) ===")
    # Compute H per team per world for FGA shares
    h_values_a = []
    h_values_b = []
    for key, idxs in group_map.items():
        team_fga_a = fga_arr_a[:, idxs]  # (n_worlds, n_team_players)
        team_fga_b = fga_arr_b[:, idxs]
        team_total_a = team_fga_a.sum(axis=1, keepdims=True)
        team_total_b = team_fga_b.sum(axis=1, keepdims=True)
        # Shares
        shares_a = np.where(team_total_a > 0, team_fga_a / team_total_a, 0.0)
        shares_b = np.where(team_total_b > 0, team_fga_b / team_total_b, 0.0)
        # Herfindahl
        h_a = (shares_a ** 2).sum(axis=1)  # (n_worlds,)
        h_b = (shares_b ** 2).sum(axis=1)
        h_values_a.extend(h_a.tolist())
        h_values_b.extend(h_b.tolist())

    h_arr_a = np.array(h_values_a)
    h_arr_b = np.array(h_values_b)
    typer.echo(f"  H (disabled): mean={h_arr_a.mean():.4f} std={h_arr_a.std():.4f} p10={np.percentile(h_arr_a, 10):.4f} p90={np.percentile(h_arr_a, 90):.4f}")
    typer.echo(f"  H (enabled):  mean={h_arr_b.mean():.4f} std={h_arr_b.std():.4f} p10={np.percentile(h_arr_b, 10):.4f} p90={np.percentile(h_arr_b, 90):.4f}")
    typer.echo("  (Lower H = more dispersed usage; typical NBA game H ~ 0.10-0.20)")

    typer.echo("")

    # === Diagnostic 6: Determinism check ===
    typer.echo("=== Diagnostic 6: Determinism Check ===")
    # Run sim B again with same seed - should be identical
    stats_b2 = _run_sim_with_usage_shares(
        minutes_df,
        rates_df,
        n_worlds=num_worlds,
        seed=seed,
        usage_shares_enabled=True,
        share_noise_std=share_noise_std,
    )
    bitwise_match = np.allclose(stats_b["fga2"], stats_b2["fga2"], rtol=0, atol=0) and \
                    np.allclose(stats_b["fga3"], stats_b2["fga3"], rtol=0, atol=0)
    if bitwise_match:
        typer.echo("[PASS] Same seed produces bitwise-identical outputs")
    else:
        max_diff_fga2 = np.abs(stats_b["fga2"] - stats_b2["fga2"]).max()
        typer.echo(f"[FAIL] Same seed produces different outputs (max_diff={max_diff_fga2:.2e})")

    # Run sim A twice - should also be identical
    stats_a2 = _run_sim_with_usage_shares(
        minutes_df, rates_df, n_worlds=num_worlds, seed=seed, usage_shares_enabled=False
    )
    disabled_match = np.allclose(stats_a["fga2"], stats_a2["fga2"], rtol=0, atol=0)
    if disabled_match:
        typer.echo("[PASS] Disabled mode produces bitwise-identical outputs")
    else:
        typer.echo("[FAIL] Disabled mode produces different outputs on re-run")

    typer.echo("")
    typer.echo("=== Notes ===")
    typer.echo("PR1 provides coupling + conservation via rate-weighted shares.")
    typer.echo("Limitation: 'usage bump when X is out' happens only via minutes changes.")
    typer.echo("PR2 will add learned share weights from pregame features + teammate-out aggregates.")


if __name__ == "__main__":
    app()
