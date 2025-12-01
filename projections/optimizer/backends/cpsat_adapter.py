"""
CP-SAT Backend Adapter for NBA DFS Optimizer

Thin shim that converts from CSV-based NBA optimizer format to CP-SAT solver format
and back to the driver's expected contract.
"""

from typing import Dict, List, Tuple, Optional, Any
import logging


def solve(
    players: Dict[Tuple[str, str, str], Any],
    constraints: Dict[str, Any],
    seed: Optional[int] = None,
) -> List[List[Tuple[Tuple[str, str, str], str, str]]]:
    """
    Convert NBA optimizer format to CP-SAT solver and back.

    Args:
        players: Dict with key=(name_norm, pos_str, team_norm), value=Player object
        constraints: Dict with solver parameters
            Recognized optional keys in `constraints` (any of these):
                - time_limit_sec / time_limit_s / opt.time_limit_sec / max_time_seconds (float seconds)
                - threads / opt.threads / num_search_workers (int)
        seed: Random seed for reproducibility

    Returns:
        List of lineups, each lineup is list of (player_key, final_slot, pid) tuples
    """
    # Local imports to avoid hard dependency when using PuLP backend
    try:
        from .. import cpsat_solver
        from ..optimizer_types import Constraints, OwnershipPenaltySettings
    except ImportError:
        import cpsat_solver
        from ..optimizer_types import Constraints, OwnershipPenaltySettings

    # Build player payload (neutral → CP-SAT format)
    players_payload = []
    for player_key, player_obj in sorted(players.items(), key=lambda x: x[1].pid):
        # Expand positions for DK (G/F/UTIL eligibility)
        positions = list(player_obj.positions)  # Already expanded in driver

        player_record = {
            "player_id": player_obj.pid,
            "name": player_obj.name,
            "team": player_obj.team,
            "positions": positions,
            "salary": player_obj.salary,
            "proj": player_obj.fpts,
            "own_proj": player_obj.ownership,  # Pass as-is, no unit conversion
            "minutes": player_obj.minutes,
            "stddev": player_obj.stddev,
            "dk_id": player_obj.pid,  # Use pid as dk_id for now
        }
        players_payload.append(player_record)

    # --- Ownership scale doctor: normalize to fractions [0,1] (silent) ---
    try:
        own_vals = [
            float(p["own_proj"]) for p in players_payload if p.get("own_proj") is not None
        ]
        own_max = max(own_vals) if own_vals else 0.0
        detected_fmt = "percent" if own_max > 1.5 else "fraction"
        if detected_fmt == "percent":
            for p in players_payload:
                if p.get("own_proj") is not None:
                    p["own_proj"] = float(p["own_proj"]) / 100.0
        else:
            pass
        # Make available downstream (optional)
        constraints["own_input_fmt"] = detected_fmt
    except Exception as _e:
        pass

    # Map constraints to CP-SAT expected schema
    site = constraints["site"]

    # Normalize optional runtime/threads parameters from various config sources
    time_limit_candidates = [
        constraints.get("time_limit_sec"),
        constraints.get("time_limit_s"),
        constraints.get("opt.time_limit_sec"),  # allow raw Sheet key pass-through
        constraints.get("max_time_seconds"),
    ]
    time_limit_sec = next((float(v) for v in time_limit_candidates if v is not None and v != ""), None)

    thread_candidates = [
        constraints.get("threads"),
        constraints.get("opt.threads"),  # allow raw Sheet key pass-through
        constraints.get("num_search_workers"),
    ]
    threads_val = next((int(v) for v in thread_candidates if v is not None and v != ""), 0)

    # Provide conservative defaults if not specified
    if time_limit_sec is None:
        time_limit_sec = 2.0  # per-lineup default cap

    # Create OwnershipPenaltySettings object if ownership penalties are configured
    ownership_penalty = None

    # Map Sheets-style key `own_lambda1` to the solver's `lambda_own` if not explicitly provided
    if (
        "lambda_own" not in constraints
        or constraints.get("lambda_own") in (None, "", 0, 0.0)
    ):
        constraints["lambda_own"] = float(constraints.get("own_lambda1", 0.0) or 0.0)

    # Normalize to float
    try:
        lambda_own = float(constraints.get("lambda_own", 0.0) or 0.0)
    except Exception:
        lambda_own = 0.0

    try:
        lambda_var = float(constraints.get("lambda_var", 0.0) or 0.0)
    except Exception:
        lambda_var = 0.0

    # ---- Ensure CP-SAT sees expected ownership keys (legacy/diagnostic path) ----
    # Base/user-input lambda used by some solver prints/paths
    try:
        constraints["lambda_ui"] = float(lambda_own)
    except Exception:
        constraints["lambda_ui"] = lambda_own

    # Pass through pct scale for solver-side diagnostics/objective math
    if "own_pct_scale" not in constraints:
        try:
            constraints["own_pct_scale"] = float(
                constraints.get("opt.own_pct_scale", 100.0)
            )
        except Exception:
            constraints["own_pct_scale"] = 100.0

    # Aggro/ramp knobs (fallbacks if not present)
    constraints["aggro_enable"] = bool(constraints.get("aggro_enable", False))

    # Parse aggro_ramp as "start,end"
    ramp_raw = str(constraints.get("aggro_ramp", "")).replace(" ", "")
    if ramp_raw and "," in ramp_raw:
        try:
            rstart, rend = [int(v) for v in ramp_raw.split(",")]
        except Exception:
            rstart, rend = 25, 300
    else:
        rstart, rend = 25, 300

    try:
        constraints["aggro_ramp_start"] = int(
            constraints.get("aggro_ramp_start", rstart)
        )
    except Exception:
        constraints["aggro_ramp_start"] = rstart
    try:
        constraints["aggro_ramp_end"] = int(
            constraints.get("aggro_ramp_end", rend)
        )
    except Exception:
        constraints["aggro_ramp_end"] = rend
    try:
        constraints["aggro_start_lam"] = float(
            constraints.get("aggro_start_lam", lambda_own)
        )
    except Exception:
        constraints["aggro_start_lam"] = float(lambda_own)
    try:
        constraints["aggro_max_lam"] = float(
            constraints.get("aggro_max_lam", max(lambda_own, 0.0))
        )
    except Exception:
        constraints["aggro_max_lam"] = float(max(lambda_own, 0.0))

    # Remove legacy ownership logging; remain silent unless explicitly enabled elsewhere

    if lambda_own > 0.0:
        ownership_penalty = OwnershipPenaltySettings(
            enabled=True,
            mode="by_points",  # Use by_points mode for lambda compatibility
            weight_lambda=lambda_own,
            curve_type="sigmoid",  # Default curve
            pivot_p0=0.20,
            curve_alpha=2.0,
            clamp_min=0.01,
            clamp_max=0.80,
            shrink_gamma=1.0,
        )

    # Build Constraints object
    cpsat_constraints = Constraints(
        N_lineups=int(constraints["num_lineups"]),
        unique_players=int(constraints["num_uniques"]),
        min_salary=int(constraints.get("min_salary") or 0),
        max_salary=int(
            constraints.get("max_salary") or (50000 if site == "dk" else 60000)
        ),
        global_team_limit=None,  # Use DK default of 4 players per team
        team_limits={},
        randomness_pct=float(
            constraints.get("randomness_pct", constraints.get("randomness", 0.0))
        ),
        cp_sat_params={
            "max_time_seconds": float(time_limit_sec),
            "num_search_workers": int(threads_val),
            "random_seed": int(seed or 42),
            "relative_gap_limit": 0.001,
        },
        ownership_penalty=ownership_penalty or OwnershipPenaltySettings(enabled=False),
    )

    # Call CP-SAT solver
    try:
        if site == "dk":
            # Use counts-based solver for DK (more efficient)
            lineup_objs, diagnostics = cpsat_solver.solve_cpsat_iterative_counts(
                players_payload, cpsat_constraints, seed or 42, site
            )
        else:
            # Use slot-based solver for FD
            lineup_objs, diagnostics = cpsat_solver.solve_cpsat_iterative(
                players_payload, cpsat_constraints, seed or 42, site
            )

        # Log solver diagnostics
        if diagnostics:
            status = diagnostics.get("status", "UNKNOWN")
            n_lineups = diagnostics.get("N", 0)
            best_obj = diagnostics.get("best_obj")
            best_bound = diagnostics.get("best_bound")
            wall_time = diagnostics.get("wall_time_sec")

            obj_str = f"obj={best_obj:.1f}" if best_obj is not None else "obj=?"
            bound_str = (
                f"bound={best_bound:.1f}" if best_bound is not None else "bound=?"
            )
            time_str = f"time={wall_time:.2f}s" if wall_time is not None else "time=?s"

            print(
                f"CP-SAT: Generated {n_lineups}/{constraints['num_lineups']} | {obj_str} | {bound_str} | {time_str} | tl={time_limit_sec}s threads={threads_val}"
            )

    except Exception as e:
        logging.error(f"CP-SAT solver failed: {e}")
        return []

    # Convert results back to driver contract format
    selected_lineups = []

    # Create reverse lookup from pid to player_key
    pid_to_player_key = {
        player_obj.pid: player_key for player_key, player_obj in players.items()
    }

    for lineup_obj in lineup_objs:
        lineup_tuples = []
        for player in lineup_obj.players:
            pid = player.player_id
            final_slot = player.pos  # CP-SAT should return assigned slots

            # Find the original player_key
            if pid in pid_to_player_key:
                player_key = pid_to_player_key[pid]
                lineup_tuples.append((player_key, final_slot, pid))
            else:
                # Fallback: skip invalid players with warning
                logging.warning(f"CP-SAT returned unknown player ID: {pid}")
                continue

        if len(lineup_tuples) == constraints["lineup_size"]:
            selected_lineups.append(lineup_tuples)
        else:
            logging.warning(
                f"CP-SAT returned lineup with {len(lineup_tuples)} players, expected {constraints['lineup_size']}"
            )

    return selected_lineups
