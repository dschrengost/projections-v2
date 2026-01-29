"""World sparsity + cap-hit diagnostics for sim minutes availability physics.

This script diagnoses the root cause of "bench sponge" minutes worlds:
availability sampling with a strict team-sum-to-240 allocator can inflate the
surviving actives when a team's active set is too sparse.

By default, it loads the `sim_v3` profile and applies the same feasibility gate
and absorption caps used by production simulation.

Examples:
  uv run python -m scripts.diagnostics.world_sparsity_stats --date 2026-01-28 --n-worlds 1000
  uv run python -m scripts.diagnostics.world_sparsity_stats --date 2026-01-28 --n-worlds 1000 --no-physics
  uv run python -m scripts.diagnostics.world_sparsity_stats --date 2026-01-28 --n-worlds 1000 --assert
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.paths import data_path
from projections.sim_v2.config import DEFAULT_PROFILES_PATH, SimV2Profile, load_sim_v2_profile
from projections.sim_v2.minutes_allocator import allocate_team_minutes_matrix
from projections.sim_v2.minutes_noise import status_bucket_from_raw
from projections.sim_v2.minutes_physics import (
    apply_minutes_availability_policy,
    apply_team_feasibility_gate,
    compute_max_increase_by_depth,
    compute_rotation_lock_mask,
)
from projections.sim_v2.play_prob_policy import apply_play_prob_policy_with_diagnostics

app = typer.Typer(add_completion=False)


def _read_latest_run_id(base_dir: Path) -> Optional[str]:
    latest = base_dir / "latest_run.json"
    if not latest.exists():
        return None
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    run_id = payload.get("run_id") or payload.get("run_as_of_ts")
    return str(run_id) if run_id else None


def _load_minutes_projection(root: Path, game_date: str, *, run_id: str | None) -> pd.DataFrame:
    # Prefer effective minutes if present (matches sim pipeline inputs).
    from projections.pipeline.effective_inputs import EFFECTIVE_MINUTES_FILENAME

    daily_base = root / "artifacts" / "minutes_v1" / "daily" / game_date
    gold_base = root / "gold" / "projections_minutes_v1" / f"game_date={game_date}"

    resolved_daily = _read_latest_run_id(daily_base)
    resolved_gold = _read_latest_run_id(gold_base)
    resolved_run = run_id or resolved_daily or resolved_gold

    candidates: list[Path] = []
    if resolved_run:
        candidates.extend(
            [
                daily_base / f"run={resolved_run}" / EFFECTIVE_MINUTES_FILENAME,
                daily_base / f"run={resolved_run}" / "minutes.parquet",
                gold_base / f"run={resolved_run}" / EFFECTIVE_MINUTES_FILENAME,
                gold_base / f"run={resolved_run}" / "minutes.parquet",
            ]
        )
    # Legacy fallback (flat gold).
    candidates.append(gold_base / "minutes.parquet")

    for path in candidates:
        if path.exists():
            df = pd.read_parquet(path)
            if "game_date" not in df.columns:
                df["game_date"] = pd.to_datetime(game_date)
            return df

    raise FileNotFoundError(f"No minutes_v1 projection found for game_date={game_date} (run_id={run_id}).")


def _resolve_minutes_column(df: pd.DataFrame) -> str:
    for candidate in ("minutes_final", "minutes_p50_cond", "minutes_p50", "minutes_pred_p50", "minutes_mean"):
        if candidate in df.columns:
            return candidate
    raise KeyError("Missing minutes column (expected one of minutes_final/minutes_p50_cond/minutes_p50/minutes_pred_p50).")


def _ensure_status_bucket(df: pd.DataFrame) -> pd.Series:
    if "status_bucket" in df.columns:
        return df["status_bucket"].astype(str).apply(status_bucket_from_raw)
    for col in ("status", "injury_status", "availability_status"):
        if col in df.columns:
            return df[col].astype(str).apply(status_bucket_from_raw)
    return pd.Series(["healthy"] * len(df), index=df.index, dtype=object)


def _dist_int(arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return {"min": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    a = arr.astype(float)
    return {
        "min": float(np.min(a)),
        "p5": float(np.percentile(a, 5)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def _dist_float(arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return {"min": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    a = arr.astype(float)
    return {
        "min": float(np.min(a)),
        "p5": float(np.percentile(a, 5)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def _simulate_team_stats(
    *,
    team_df: pd.DataFrame,
    n_worlds: int,
    rng: np.random.Generator,
    profile: SimV2Profile,
    hard_cap: float,
    physics: bool,
) -> tuple[dict[str, object], pd.DataFrame | None]:
    min_col = _resolve_minutes_column(team_df)

    baseline_minutes = pd.to_numeric(team_df[min_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    play_prob_raw = pd.to_numeric(team_df.get("play_prob", 1.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    play_prob_raw = np.clip(play_prob_raw, 0.0, 1.0)

    is_starter = None
    if "is_starter" in team_df.columns:
        is_starter = pd.to_numeric(team_df["is_starter"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.5

    status_bucket = _ensure_status_bucket(team_df).astype(str).to_numpy(dtype=object)

    group_map = {"team": np.arange(len(team_df), dtype=int)}

    # Absorption caps
    max_increase = None
    cap_upper = np.full_like(baseline_minutes, float(hard_cap), dtype=float)
    if physics and getattr(profile.minutes_absorption_caps, "enabled", False):
        max_increase = compute_max_increase_by_depth(
            baseline_minutes=baseline_minutes,
            is_starter=is_starter,
            group_map=group_map,
            cfg=profile.minutes_absorption_caps,
        )
        cap_upper = np.minimum(float(hard_cap), np.clip(baseline_minutes, 0.0, None) + np.clip(max_increase, 0.0, None))

    # Availability policy (p_eff)
    play_prob_eff = play_prob_raw
    rotation_lock_mask = None
    policy_reason = np.array(["raw"] * len(team_df), dtype=object)
    if physics and getattr(profile.play_prob_policy, "enabled", False):
        policy_input = team_df.copy()
        policy_input["status_bucket"] = status_bucket
        policy_df, _policy_diag = apply_play_prob_policy_with_diagnostics(policy_input, profile.play_prob_policy)
        play_prob_eff = pd.to_numeric(policy_df["play_prob_eff"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        play_prob_eff = np.clip(play_prob_eff, 0.0, 1.0)
        rotation_lock_mask = policy_df["rotation_lock"].astype(bool).to_numpy()
        policy_reason = policy_df["play_prob_policy_reason"].astype(str).to_numpy(dtype=object)
    elif physics and getattr(profile.minutes_availability_policy, "enabled", False):
        rotation_lock_mask, play_prob_eff, _policy_diag = apply_minutes_availability_policy(
            play_prob_raw=play_prob_raw,
            baseline_minutes=baseline_minutes,
            is_starter=is_starter,
            status_bucket=status_bucket,
            group_map=group_map,
            cfg=profile.minutes_availability_policy,
        )
    elif physics and getattr(profile.minutes_feasibility, "min_rotation_locks_active", None) is not None:
        rotation_lock_mask = compute_rotation_lock_mask(
            baseline_minutes=baseline_minutes,
            is_starter=is_starter,
            group_map=group_map,
            top_k=8,
            minutes_threshold=20.0,
        )

    # Sample availability
    active = rng.random(size=(n_worlds, len(team_df))) < play_prob_eff[None, :]

    # Feasibility gate + resampling
    gate_diag = None
    if physics and getattr(profile.minutes_feasibility, "enabled", False):
        active, gate_diag = apply_team_feasibility_gate(
            active,
            play_prob=play_prob_eff,
            baseline_minutes=baseline_minutes,
            cap_upper=cap_upper,
            group_map=group_map,
            cfg=profile.minutes_feasibility,
            rng=rng,
            eligible_mask=None,
            rotation_lock_mask=rotation_lock_mask,
            target_total=240.0,
            eps=1e-6,
        )

    n_active = active.sum(axis=1)
    sum_demand_active = (active.astype(float) * baseline_minutes[None, :]).sum(axis=1)

    # Team-240 allocation (diagnostic approximation: demand = baseline for actives)
    demand = baseline_minutes[None, :] * active.astype(float)
    allocated, alloc_stats = allocate_team_minutes_matrix(
        demand,
        active,
        priority=baseline_minutes,
        cap=float(hard_cap),
        max_increase=max_increase,
        baseline=(baseline_minutes if max_increase is not None else None),
        target_total=240.0,
        k=3.0,
        eps=1e-6,
    )

    cap48_hits = (allocated >= (48.0 - 1e-6)).any(axis=1)
    cap_hits = (allocated >= (float(hard_cap) - 1e-6)).any(axis=1)

    frac_sparse = float((n_active < 8).mean())  # fixed definition per acceptance criteria
    frac_alloc_infeasible = float(int(alloc_stats["n_cap_infeasible_rows"])) / float(n_worlds)

    out: dict[str, object] = {
        "n_players": int(len(team_df)),
        "n_active": _dist_int(n_active),
        "sum_demand_active": _dist_float(sum_demand_active),
        "frac_any_cap48": float(cap48_hits.mean()),
        "frac_any_hard_cap": float(cap_hits.mean()),
        "n_team_worlds": int(n_worlds),
        "n_cap_hits": int(cap_hits.sum()),
        "n_cap48_hits": int(cap48_hits.sum()),
        "frac_sparse_n_active_lt8": float(frac_sparse),
        "frac_allocator_infeasible": float(frac_alloc_infeasible),
    }

    if gate_diag is not None and gate_diag.enabled:
        out["frac_worlds_infeasible_pre_resample"] = float(gate_diag.n_infeasible_pre_resample) / float(n_worlds)
        out["frac_worlds_resampled"] = float(gate_diag.n_resampled_team_worlds) / float(n_worlds)
        out["avg_resample_attempts"] = (
            float(gate_diag.resample_attempts_total) / float(gate_diag.n_resampled_team_worlds)
            if gate_diag.n_resampled_team_worlds
            else 0.0
        )
        out["frac_worlds_promoted"] = float(gate_diag.n_promoted_team_worlds) / float(n_worlds)
        out["n_infeasible_pre_resample"] = int(gate_diag.n_infeasible_pre_resample)
        out["n_resampled_team_worlds"] = int(gate_diag.n_resampled_team_worlds)
        out["n_promoted_team_worlds"] = int(gate_diag.n_promoted_team_worlds)
        out["promoted_players_total"] = int(gate_diag.promoted_players_total)
    else:
        out["frac_worlds_infeasible_pre_resample"] = None
        out["frac_worlds_resampled"] = None
        out["avg_resample_attempts"] = None
        out["frac_worlds_promoted"] = None
        out["n_infeasible_pre_resample"] = 0
        out["n_resampled_team_worlds"] = 0
        out["n_promoted_team_worlds"] = 0
        out["promoted_players_total"] = 0

    # Per-player diagnostics for policy audits.
    rotation_lock = rotation_lock_mask if rotation_lock_mask is not None else np.zeros(len(team_df), dtype=bool)
    player_diag = pd.DataFrame(
        {
            "player_id": pd.to_numeric(team_df.get("player_id"), errors="coerce"),
            "player_name": team_df.get("player_name"),
            "status_bucket": status_bucket.astype(str),
            "baseline_minutes": baseline_minutes.astype(float),
            "rotation_lock": rotation_lock.astype(bool),
            "play_prob_raw": play_prob_raw.astype(float),
            "play_prob_eff": play_prob_eff.astype(float),
            "sim_p_active": active.mean(axis=0).astype(float),
            "policy_reason": policy_reason.astype(str),
        }
    )
    for extra in ("game_id", "team_id"):
        if extra in team_df.columns:
            player_diag[extra] = pd.to_numeric(team_df[extra], errors="coerce")

    return out, player_diag


@app.command()
def main(
    date: str = typer.Option(..., "--date", help="Game date (YYYY-MM-DD)"),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional minutes run_id override"),
    n_worlds: int = typer.Option(1000, "--n-worlds", help="Worlds to simulate per team"),
    seed: int = typer.Option(42, "--seed", help="RNG seed"),
    profile: str = typer.Option("sim_v3", "--profile", help="sim_v2 profile name"),
    profiles_path: Path | None = typer.Option(None, "--profiles-path", help="Override sim_v2 profiles json path"),
    data_root: Path | None = typer.Option(None, "--data-root", help="Data root (default: PROJECTIONS_DATA_ROOT/./data)"),
    hard_cap: float = typer.Option(48.0, "--hard-cap", help="Hard cap minutes for allocator diagnostics"),
    physics: bool = typer.Option(True, "--physics/--no-physics", help="Enable feasibility gate + absorption caps"),
    assert_mode: bool = typer.Option(False, "--assert", help="Fail non-zero if thresholds are violated"),
    max_frac_sparse: float = typer.Option(0.01, "--max-frac-sparse", help="Max allowed P(n_active<8) per team"),
    max_frac_cap: float = typer.Option(0.05, "--max-frac-cap", help="Max allowed P(any hard cap hit) per team"),
    max_frac_alloc_infeasible: float = typer.Option(
        1e-3, "--max-frac-alloc-infeasible", help="Max allowed allocator infeasible rate per team"
    ),
    out: Path | None = typer.Option(None, "--out", help="Optional path to write a CSV report"),
    player_out: Path | None = typer.Option(None, "--player-out", help="Optional path to write per-player audit CSV"),
    offenders_n: int = typer.Option(20, "--offenders-n", help="How many offender rows to print"),
) -> None:
    date_norm = pd.Timestamp(date).date().isoformat()
    root = Path(data_root) if data_root is not None else data_path()

    profiles_path_eff = profiles_path or DEFAULT_PROFILES_PATH
    profile_cfg = load_sim_v2_profile(profile=profile, profiles_path=profiles_path_eff)

    if not physics:
        # Disable physics blocks in profile for this run (keep raw sampling behavior).
        profile_cfg = replace(
            profile_cfg,
            minutes_feasibility=replace(profile_cfg.minutes_feasibility, enabled=False),
            minutes_absorption_caps=replace(profile_cfg.minutes_absorption_caps, enabled=False),
            minutes_availability_policy=replace(profile_cfg.minutes_availability_policy, enabled=False),
            play_prob_policy=replace(profile_cfg.play_prob_policy, enabled=False),
        )

    df = _load_minutes_projection(root, date_norm, run_id=run_id)
    if "team_id" not in df.columns:
        raise RuntimeError("minutes projection missing team_id")
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
        df = df[df["game_date"] == pd.Timestamp(date_norm)]

    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")
    df = df.dropna(subset=["team_id"]).copy()
    df["team_id"] = df["team_id"].astype(int)
    if df.empty:
        raise RuntimeError(f"No minutes rows found for game_date={date_norm}")

    # Ensure play_prob exists (fallback to 1.0 for diagnosis if missing).
    if "play_prob" not in df.columns:
        df["play_prob"] = 1.0

    group_cols = ["team_id"]
    if "game_id" in df.columns:
        df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce")
        if not df["game_id"].isna().all():
            df["game_id"] = df["game_id"].fillna(-1).astype(int)
            group_cols = ["game_id", "team_id"]

    grouped = list(df.groupby(group_cols, sort=True))
    if not grouped:
        raise RuntimeError("No teams found after grouping")

    # Deterministic team RNGs independent of group iteration / resampling loops.
    ss = np.random.SeedSequence(int(seed))
    child_seeds = ss.spawn(len(grouped))

    rows: list[dict[str, object]] = []
    rows_player: list[pd.DataFrame] = []
    failures: list[str] = []

    for (key, team_df), child in zip(grouped, child_seeds):
        rng = np.random.default_rng(child)
        stats, player_diag = _simulate_team_stats(
            team_df=team_df,
            n_worlds=int(n_worlds),
            rng=rng,
            profile=profile_cfg,
            hard_cap=float(hard_cap),
            physics=bool(physics),
        )
        if isinstance(key, tuple):
            game_id, team_id = key
        else:
            game_id, team_id = None, key

        row = {
            "game_id": int(game_id) if game_id is not None else None,
            "team_id": int(team_id),
            **stats,
        }
        rows.append(row)
        if player_diag is not None and not player_diag.empty:
            player_diag = player_diag.copy()
            player_diag["game_id"] = int(game_id) if game_id is not None else player_diag.get("game_id")
            player_diag["team_id"] = int(team_id)
            rows_player.append(player_diag)

        if assert_mode:
            frac_sparse = float(row["frac_sparse_n_active_lt8"])
            frac_cap = float(row["frac_any_hard_cap"])
            frac_inf = float(row["frac_allocator_infeasible"])
            if frac_sparse > float(max_frac_sparse) or frac_cap > float(max_frac_cap) or frac_inf > float(max_frac_alloc_infeasible):
                failures.append(
                    f"team_id={team_id} frac_sparse={frac_sparse:.4f} frac_cap={frac_cap:.4f} frac_alloc_infeasible={frac_inf:.4f}"
                )

    out_df = pd.DataFrame(rows)
    player_df = pd.concat(rows_player, ignore_index=True) if rows_player else pd.DataFrame()
    # Expand nested dict columns for readability when writing CSV.
    for col in ("n_active", "sum_demand_active"):
        if col in out_df.columns:
            expanded = out_df[col].apply(lambda x: x if isinstance(x, dict) else {}).apply(pd.Series)
            expanded = expanded.add_prefix(f"{col}_")
            out_df = pd.concat([out_df.drop(columns=[col]), expanded], axis=1)

    # Console summary (compact; per-team rows sorted by worst sparsity then cap hits).
    view_cols = [
        "game_id",
        "team_id",
        "n_players",
        "n_active_min",
        "n_active_p5",
        "n_active_p50",
        "n_active_p95",
        "sum_demand_active_p50",
        "frac_sparse_n_active_lt8",
        "frac_any_hard_cap",
        "frac_any_cap48",
        "frac_worlds_infeasible_pre_resample",
        "frac_worlds_resampled",
        "avg_resample_attempts",
        "frac_allocator_infeasible",
    ]
    for col in view_cols:
        if col not in out_df.columns:
            out_df[col] = None

    out_df = out_df.sort_values(
        by=["frac_sparse_n_active_lt8", "frac_any_hard_cap", "team_id"],
        ascending=[False, False, True],
        na_position="last",
    )

    pd.set_option("display.max_columns", 200)
    print(out_df[view_cols].to_string(index=False))

    # Aggregate summary (team-world weighted), plus play_prob policy audit.
    try:
        total_team_worlds = int(out_df["n_team_worlds"].fillna(0).sum()) if "n_team_worlds" in out_df.columns else 0
        infeasible = int(out_df["n_infeasible_pre_resample"].fillna(0).sum()) if "n_infeasible_pre_resample" in out_df.columns else 0
        promoted_worlds = int(out_df["n_promoted_team_worlds"].fillna(0).sum()) if "n_promoted_team_worlds" in out_df.columns else 0
        cap_hits = int(out_df["n_cap_hits"].fillna(0).sum()) if "n_cap_hits" in out_df.columns else 0
        promoted_players = int(out_df["promoted_players_total"].fillna(0).sum()) if "promoted_players_total" in out_df.columns else 0

        if total_team_worlds > 0:
            print(
                "\nAGGREGATE (team-world weighted): "
                f"frac_infeasible_pre_resample={infeasible/total_team_worlds:.4f} "
                f"frac_promoted={promoted_worlds/total_team_worlds:.4f} "
                f"cap_hit_rate={cap_hits/total_team_worlds:.4f} "
                f"promoted_players_total={promoted_players}"
            )
    except Exception:
        pass

    if not player_df.empty:
        # Bucket summaries.
        not_out_or_q = ~player_df["status_bucket"].astype(str).str.lower().isin(["out", "questionable"])
        rot = player_df["rotation_lock"].astype(bool) & not_out_or_q
        fringe = (~player_df["rotation_lock"].astype(bool)) & not_out_or_q

        def _dist(series: pd.Series) -> dict[str, float]:
            vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size == 0:
                return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
            return {
                "mean": float(vals.mean()),
                "p10": float(np.percentile(vals, 10)),
                "p50": float(np.percentile(vals, 50)),
                "p90": float(np.percentile(vals, 90)),
            }

        print("\nPLAY_PROB POLICY AUDIT:")
        print("  reasons:", player_df["policy_reason"].astype(str).value_counts().to_dict())
        print("  rotation_lock sim_p_active:", _dist(player_df.loc[rot, "sim_p_active"]))
        print("  fringe sim_p_active:", _dist(player_df.loc[fringe, "sim_p_active"]))
        print("  rotation_lock p_raw:", _dist(player_df.loc[rot, "play_prob_raw"]))
        print("  rotation_lock p_eff:", _dist(player_df.loc[rot, "play_prob_eff"]))
        print("  fringe p_raw:", _dist(player_df.loc[fringe, "play_prob_raw"]))
        print("  fringe p_eff:", _dist(player_df.loc[fringe, "play_prob_eff"]))

        player_df["delta_eff_raw"] = player_df["play_prob_eff"] - player_df["play_prob_raw"]
        player_df["delta_sim_eff"] = (player_df["sim_p_active"] - player_df["play_prob_eff"]).abs()

        top_bumps = player_df.sort_values("delta_eff_raw", ascending=False).head(int(offenders_n))
        top_dev = player_df.sort_values("delta_sim_eff", ascending=False).head(int(offenders_n))
        print("\nTOP p_eff - p_raw:")
        print(
            top_bumps[
                ["game_id", "team_id", "player_name", "player_id", "status_bucket", "baseline_minutes", "rotation_lock", "play_prob_raw", "play_prob_eff", "delta_eff_raw", "policy_reason"]
            ].to_string(index=False)
        )
        print("\nTOP |sim_p_active - p_eff|:")
        print(
            top_dev[
                ["game_id", "team_id", "player_name", "player_id", "status_bucket", "baseline_minutes", "rotation_lock", "play_prob_eff", "sim_p_active", "delta_sim_eff", "policy_reason"]
            ].to_string(index=False)
        )

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out, index=False)
        print(f"\nWrote: {out}")

    if player_out is not None and not player_df.empty:
        player_out.parent.mkdir(parents=True, exist_ok=True)
        player_df.to_csv(player_out, index=False)
        print(f"\nWrote: {player_out}")

    if failures:
        print("\nFAILED thresholds:")
        for msg in failures:
            print("  -", msg)
        raise typer.Exit(code=1)

    return


if __name__ == "__main__":
    app()
