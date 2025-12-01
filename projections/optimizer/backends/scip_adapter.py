"""
SCIP Backend Adapter for NBA DFS Optimizer

Builds a simple slot-assignment MIP using PySCIPOpt that mirrors the
legacy PuLP backend and the CP-SAT slot model:

- Binary x[p, s] indicates player p is assigned to slot s
- Binary y[p] indicates player p is used in the lineup
- Constraints:
  - Exactly one player per slot
  - Each player used at most once: sum_s x[p, s] == y[p]
  - Salary min/max using y[p]
- Objective per lineup iteration:
  Max sum_p y[p] * (proj_jittered[p] - lambda_own * own_frac[p] - lambda_var * stddev[p])

We solve sequentially for multiple lineups and add a uniqueness cut after
each solution: sum_{p in lineup_k} y[p] <= |lineup_k| - num_uniques.
"""

from __future__ import annotations

from typing import Dict, Tuple, Any, List, Optional
import math
import logging

try:
    from pyscipopt import Model, quicksum
except Exception as e:  # pragma: no cover - import error path
    Model = None  # type: ignore
    quicksum = None  # type: ignore

# Optional ownership penalty functional API (shared with CP-SAT path)
try:  # Top-level import when 'optimizer' is on sys.path
    from objective import get_active_ownership_penalty, build_ownership_penalty  # type: ignore
except Exception:  # pragma: no cover
    try:
        from ..objective import get_active_ownership_penalty, build_ownership_penalty  # type: ignore
    except Exception:  # pragma: no cover
        get_active_ownership_penalty = None  # type: ignore
        build_ownership_penalty = None  # type: ignore


def _detect_own_is_percent(players: Dict[Tuple[str, str, str], Any]) -> bool:
    try:
        max_own = max(abs(float(p.ownership or 0.0)) for p in players.values())
    except Exception:
        return True
    return bool(max_own > 1.0000001)


def solve(
    players: Dict[Tuple[str, str, str], Any],
    constraints: Dict[str, Any],
    seed: Optional[int] = None,
) -> List[List[Tuple[Tuple[str, str, str], str, str]]]:
    """
    Solve using SCIP backend and return lineups in the driver's expected format.

    Args:
        players: Dict with key=(name_norm, pos_str, team_norm), value=Player object
        constraints: Dict with solver parameters (site, lineup_size, min_salary, max_salary,
                     num_lineups, num_uniques, randomness, lambda_own, lambda_var, time_limit_sec, threads)
        seed: Optional random seed

    Returns:
        List of lineups; each lineup is a list of (player_key, final_slot, pid) tuples
    """
    if Model is None:
        raise ImportError(
            "PySCIPOpt is not installed. Please install pyscipopt>=4.5.0 to use the SCIP backend."
        )

    site = str(constraints.get("site", "dk")).lower()
    assert site in {"dk", "fd"}, "site must be 'dk' or 'fd'"

    # Slot labels in order
    if site == "dk":
        slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    else:
        # Duplicate labels are fine; we index by slot position
        slots = ["PG", "PG", "SG", "SG", "SF", "SF", "PF", "PF", "C"]

    # Build player index and attributes
    P: List[Tuple[str, str, str]] = list(players.keys())
    pid = {pkey: str(players[pkey].pid) for pkey in P}
    salary = {pkey: int(players[pkey].salary) for pkey in P}
    fpts = {pkey: float(players[pkey].fpts) for pkey in P}
    stddev = {pkey: float(players[pkey].stddev or 0.0) for pkey in P}
    positions = {pkey: set(players[pkey].positions) for pkey in P}
    own_is_percent = _detect_own_is_percent(players)
    own_frac = {
        pkey: (float(players[pkey].ownership or 0.0) / 100.0 if own_is_percent else float(players[pkey].ownership or 0.0))
        for pkey in P
    }

    min_salary = int(constraints.get("min_salary") or (49000 if site == "dk" else 59000))
    max_salary = int(constraints.get("max_salary") or (50000 if site == "dk" else 60000))
    num_lineups = int(constraints.get("num_lineups") or 0)
    num_uniques = int(constraints.get("num_uniques") or 1)
    randomness_pct = float(constraints.get("randomness") or 0.0)
    lambda_own = float(constraints.get("lambda_own") or 0.0)
    lambda_var = float(constraints.get("lambda_var") or 0.0)
    time_limit = constraints.get("time_limit_sec")
    threads = constraints.get("threads")

    rng_seed = int(seed or constraints.get("seed") or 42)

    # Reusable RNG and helper penalty builder
    import numpy as _np  # local import to avoid hard dep at import time
    import pandas as _pd
    _rng = _np.random.default_rng(rng_seed)

    def _ownership_penalties(lineup_idx: int) -> Dict[Tuple[str, str, str], float]:
        pen_map: Dict[Tuple[str, str, str], float] = {pkey: 0.0 for pkey in P}
        if build_ownership_penalty is None or get_active_ownership_penalty is None:
            return pen_map
        try:
            cfg, aggro = get_active_ownership_penalty()  # type: ignore
            if cfg is None:
                return pen_map
            # Pass ownership as fraction [0,1]; builder's pct_scale handles conversion (e.g., 100 → percent)
            own_vals = _np.array([own_frac[pkey] for pkey in P], dtype=float)
            df_tmp = _pd.DataFrame({"own": own_vals})
            pen_vec = build_ownership_penalty(df_tmp, cfg, own_col="own", lineup_idx=lineup_idx, aggro=aggro)  # type: ignore
            if pen_vec is not None and len(pen_vec) == len(P):
                for i, pkey in enumerate(P):
                    pen_map[pkey] = float(pen_vec[i])  # already negative
        except Exception:
            return pen_map
        return pen_map

    def _build_once() -> Tuple[Any, Dict[Tuple[str, int], Any], Dict[Tuple[str, str, str], Any]]:
        m = Model("nba_lineup_scip")
        try:
            m.setIntParam("randomseed", int(rng_seed))
        except Exception:
            pass
        if time_limit is not None:
            try:
                m.setRealParam("limits/time", float(time_limit))
            except Exception:
                pass
        if threads is not None:
            try:
                m.setIntParam("lp/threads", int(threads))
            except Exception:
                pass
            try:
                m.setIntParam("parallel/maxnthreads", int(threads))
            except Exception:
                pass
        try:
            m.setIntParam("presolving/maxrounds", 3)
            m.setBoolParam("branching/preferbinary", True)
        except Exception:
            pass

        x: Dict[Tuple[str, int], Any] = {}
        y: Dict[Tuple[str, str, str], Any] = {pkey: m.addVar(vtype="B", name=f"y_{players[pkey].pid}") for pkey in P}
        # Eligibility and slots
        for si, slabel in enumerate(slots):
            for pkey in P:
                if slabel in positions[pkey]:
                    x[(pkey, si)] = m.addVar(vtype="B", name=f"x_{players[pkey].pid}_{si}")
        # Slot fill
        for si, slabel in enumerate(slots):
            m.addCons(quicksum(x[(pkey, si)] for pkey in P if (pkey, si) in x) == 1, name=f"slot_{si}_{slabel}")
        # Link
        for pkey in P:
            m.addCons(quicksum(x[(pkey, si)] for si in range(len(slots)) if (pkey, si) in x) == y[pkey], name=f"link_{players[pkey].pid}")
        # Salary
        m.addCons(quicksum(y[pkey] * salary[pkey] for pkey in P) <= max_salary, name="salary_max")
        m.addCons(quicksum(y[pkey] * salary[pkey] for pkey in P) >= min_salary, name="salary_min")
        return m, x, y

    # Main loop: rebuild model per lineup and add all previous uniqueness cuts
    selected_lineups: List[List[Tuple[Tuple[str, str, str], str, str]]] = []
    prev_lineups: List[List[Tuple[str, str, str]]] = []  # list of used_players keys

    for k in range(num_lineups):
        m, x, y = _build_once()
        # Add uniqueness cuts accumulated so far
        for i, used in enumerate(prev_lineups):
            try:
                m.addCons(
                    quicksum(y[pkey] for pkey in used) <= len(used) - num_uniques,
                    name=f"uniq_{i}",
                )
            except Exception:
                # Skip faulty cut, continue
                pass

        # Build objective coefficients
        pen_own_map = _ownership_penalties(k)  # negative values
        coeff: Dict[Tuple[str, str, str], float] = {}
        for pkey in P:
            base = float(fpts[pkey])
            if randomness_pct > 0.0 and stddev[pkey] > 0.0:
                sd = stddev[pkey] * (randomness_pct / 100.0)
                try:
                    base = float(_rng.normal(loc=base, scale=sd))
                except Exception:
                    base = float(fpts[pkey])
            pen_own = pen_own_map.get(pkey, -float(lambda_own * own_frac[pkey]))
            pen_var = -float(lambda_var * stddev[pkey])
            eff = base + pen_own + pen_var
            if not math.isfinite(eff) or eff <= 0:
                eff = 1e-9
            coeff[pkey] = eff
        m.setObjective(quicksum(coeff[pkey] * y[pkey] for pkey in P), sense="maximize")

        # Optimize and extract
        m.optimize()
        status = str(m.getStatus())
        if status not in {"optimal", "timelimit", "bestsollimit", "gaplimit"}:
            logging.warning(
                f"SCIP infeasible or no solution (status={status}) at lineup {k}. Generated {len(selected_lineups)} / {num_lineups}."
            )
            break
        sol = m.getBestSol()
        if sol is None:
            logging.warning(f"SCIP returned no solution at lineup {k}.")
            break

        lineup: List[Tuple[Tuple[str, str, str], str, str]] = []
        used_players: List[Tuple[str, str, str]] = []
        for si, slabel in enumerate(slots):
            chosen_p: Optional[Tuple[str, str, str]] = None
            for pkey in P:
                var = x.get((pkey, si))
                if var is not None and m.getSolVal(sol, var) > 0.5:
                    chosen_p = pkey
                    break
            if chosen_p is None:
                logging.warning(f"SCIP: slot {si}={slabel} unfilled in solution {k}")
                continue
            lineup.append((chosen_p, slabel, pid[chosen_p]))
            used_players.append(chosen_p)

        if len(lineup) != len(slots):
            logging.warning(f"SCIP: lineup {k} incomplete ({len(lineup)} of {len(slots)}); stopping.")
            break

        selected_lineups.append(lineup)
        prev_lineups.append(used_players)

        if (k + 1) % 50 == 0 or k == 0:
            msg = [f"SCIP: Generated {k + 1}/{num_lineups}"]
            try:
                tsec = m.getSolvingTime()
                msg.append(f"time={tsec:.2f}s")
            except Exception:
                pass
            try:
                gap = m.getGap()
                if math.isfinite(gap):
                    msg.append(f"gap={gap:.4f}")
            except Exception:
                pass
            try:
                nnode = m.getNNodes()
                msg.append(f"nodes={nnode}")
            except Exception:
                pass
            print(" | ".join(msg))

    return selected_lineups
