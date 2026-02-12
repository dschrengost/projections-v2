from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.sim_v2.generate_worlds_fpts_v2 import main as generate_worlds_main


def _write_profiles(path: Path) -> None:
    payload = {
        "profiles": {
            "test_override_v2": {
                "mean_source": "rates",
                "minutes_source": "minutes_v1",
                "rates_source": "rates_v1_live",
                "worlds": {"n_worlds": 120, "batch_size": 120},
                "noise": {"epsilon_dist": "normal", "nu": 5, "k_default": 0.0},
                "rates_noise": {"enabled": False},
                "minutes_noise": {"enabled": False, "sigma_min": 0.0},
                "minutes_worlds": {"mode": "model_space_v1", "gate_temperature": 1.0},
                "team_factor_sigma": 0.0,
                "team_factor_gamma": 1.0,
                "enforce_team_240": False,
                "use_play_prob_masking": True,
                "seed": 4242,
            }
        }
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_minutes_projection(root: Path, game_date: str) -> list[int]:
    base_dir = root / "gold" / "projections_minutes_v1" / f"game_date={game_date}"
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = "test"
    (base_dir / "latest_run.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    out_dir = base_dir / f"run={run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    all_player_ids: list[int] = []
    for team_id, start_pid in [(10, 1000), (20, 2000)]:
        for idx in range(10):
            pid = start_pid + idx + 1
            all_player_ids.append(pid)
            minutes = 24.0
            rows.append(
                {
                    "game_date": game_date,
                    "tip_ts": f"{game_date}T00:00:00Z",
                    "game_id": 1,
                    "team_id": team_id,
                    "player_id": pid,
                    "player_name": f"P{pid}",
                    "status": "available",
                    "starter_flag": 1 if idx < 5 else 0,
                    "is_projected_starter": 1 if idx < 5 else 0,
                    "play_prob": 0.95,
                    "minutes_p10": max(minutes - 4.0, 0.0),
                    "minutes_p50": minutes,
                    "minutes_p90": min(minutes + 4.0, 48.0),
                    "minutes_p10_cond": max(minutes - 4.0, 0.0),
                    "minutes_p50_cond": minutes,
                    "minutes_p90_cond": min(minutes + 4.0, 48.0),
                    "minutes_final": minutes,
                }
            )

    pd.DataFrame(rows).to_parquet(out_dir / "minutes.parquet", index=False)
    return all_player_ids


def _write_rates_live(root: Path, game_date: str, player_ids: list[int]) -> None:
    out_dir = root / "gold" / "rates_v1_live" / game_date
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pid in player_ids:
        team_id = 10 if str(pid).startswith("1") else 20
        rows.append(
            {
                "game_date": game_date,
                "game_id": 1,
                "team_id": team_id,
                "player_id": pid,
                "fga2_per_min": 0.55,
                "fga3_per_min": 0.25,
                "fta_per_min": 0.17,
                "ast_per_min": 0.10,
                "tov_per_min": 0.06,
                "oreb_per_min": 0.04,
                "dreb_per_min": 0.12,
                "stl_per_min": 0.03,
                "blk_per_min": 0.02,
                "fg2_pct": 0.53,
                "fg3_pct": 0.36,
                "ft_pct": 0.79,
            }
        )
    pd.DataFrame(rows).to_parquet(out_dir / "rates.parquet", index=False)


def _write_overrides(root: Path, game_date: str) -> tuple[int, int]:
    # Team 10: lock player 1001 at 20, cap all other active players at 34, and hard out 1010.
    lock_pid = 1001
    zero_pid = 1010

    overrides = [
        {
            "game_id": 1,
            "player_id": lock_pid,
            "fields": {"minutes_target": 20.0, "minutes_lock": True},
            "updated_at": f"{game_date}T12:00:00Z",
            "sticky_fields": [],
        },
        {
            "game_id": 1,
            "player_id": zero_pid,
            "fields": {"status": "out"},
            "updated_at": f"{game_date}T12:00:00Z",
            "sticky_fields": [],
        },
    ]
    for pid in range(1002, 1010):
        overrides.append(
            {
                "game_id": 1,
                "player_id": pid,
                "fields": {"minutes_cap": 34.0},
                "updated_at": f"{game_date}T12:00:00Z",
                "sticky_fields": [],
            }
        )

    path = root / "artifacts" / "ops" / "overrides_v1" / f"game_date={game_date}" / "overrides.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "game_date": game_date,
        "updated_at": f"{game_date}T12:00:00Z",
        "overrides": overrides,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return lock_pid, zero_pid


def test_sim_worlds_minutes_override_mode_v2_persists_artifacts_and_enforces_constraints(tmp_path: Path) -> None:
    game_date = "2025-01-08"
    profiles_path = tmp_path / "sim_profiles.json"
    _write_profiles(profiles_path)

    player_ids = _write_minutes_projection(tmp_path, game_date)
    _write_rates_live(tmp_path, game_date, player_ids)
    lock_pid, zero_pid = _write_overrides(tmp_path, game_date)

    output_root = tmp_path / "out"
    generate_worlds_main(
        start_date=game_date,
        end_date=game_date,
        n_worlds=120,
        profile="test_override_v2",
        data_root=tmp_path,
        profiles_path=profiles_path,
        output_root=output_root,
        sim_run_id=None,
        use_rates_noise=False,
        rates_noise_split=None,
        team_sigma_scale=None,
        player_sigma_scale=None,
        rates_run_id=None,
        minutes_run_id=None,
        use_minutes_noise=False,
        minutes_noise_run_id=None,
        minutes_sigma_min=None,
        seed=123,
        min_play_prob=None,
        team_factor_sigma=None,
        team_factor_gamma=None,
        use_efficiency_scoring=True,
        minutes_override_mode="v2",
        override_infeasible="error",
    )

    run_dir = output_root / f"game_date={game_date}"

    assert (run_dir / "overrides_input.json").exists()
    assert (run_dir / "overrides_compiled_v2.json").exists()
    assert (run_dir / "override_resolved_minutes.parquet").exists()
    assert (run_dir / "override_diag.json").exists()

    resolved = pd.read_parquet(run_dir / "override_resolved_minutes.parquet")
    lock_row = resolved.loc[resolved["player_id"] == lock_pid].iloc[0]
    assert abs(float(lock_row["lb_minutes"]) - 20.0) <= 1e-9
    assert abs(float(lock_row["ub_minutes"]) - 20.0) <= 1e-9

    zero_row = resolved.loc[resolved["player_id"] == zero_pid].iloc[0]
    assert abs(float(zero_row["lb_minutes"])) <= 1e-9
    assert abs(float(zero_row["ub_minutes"])) <= 1e-9
    assert bool(zero_row["force_inactive"]) is True

    proj = pd.read_parquet(run_dir / "projections.parquet")

    lock_proj = proj.loc[proj["player_id"] == lock_pid].iloc[0]
    assert abs(float(lock_proj["minutes_sim_mean"]) - 20.0) <= 0.25

    zero_proj = proj.loc[proj["player_id"] == zero_pid].iloc[0]
    assert abs(float(zero_proj["sim_p_active"])) <= 1e-9

    capped = proj.loc[(proj["team_id"] == 10) & (proj["player_id"] != zero_pid)]
    cap_check_col = None
    for candidate in ("minutes_sim_p95", "minutes_sim_p90", "minutes_sim_mean"):
        if candidate in capped.columns:
            cap_check_col = candidate
            break
    assert cap_check_col is not None
    assert float(capped[cap_check_col].max()) <= 34.05

    with open(run_dir / "override_diag.json", encoding="utf-8") as f:
        diag = json.load(f)
    assert diag["team_diagnostics"], "expected per-team override diagnostics"
    assert any(str(d.get("team_id")) == "10" for d in diag["team_diagnostics"])
