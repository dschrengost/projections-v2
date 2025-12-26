import math
import hashlib
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

# New objective penalty wiring (global config getter)
try:  # local import
    from .objective import get_active_ownership_penalty
    from .objective.own_penalty import build_ownership_penalty
except Exception:  # fallback when relative import fails
    from objective import get_active_ownership_penalty  # type: ignore
    from objective.own_penalty import build_ownership_penalty  # type: ignore


# ============================================================================
# PRP-16: Ownership Penalty Helper Functions for CP-SAT
# ============================================================================


def _clamp(v: float, lo: float, hi: float) -> float:
    """Clamp value to range [lo, hi]"""
    return max(lo, min(hi, v))


def _effective_p(p: float, p0: float, gamma: float, lo: float, hi: float) -> float:
    """Calculate effective ownership percentage with shrinkage and clamping"""
    return _clamp(gamma * p + (1.0 - gamma) * p0, lo, hi)


def _g_curve(p_eff: float, settings: Dict) -> float:
    """Calculate penalty curve value for effective ownership percentage"""
    curve_type = settings.get("curve_type", "sigmoid")

    if curve_type == "linear":
        return p_eff
    elif curve_type == "power":
        power_k = settings.get("power_k", 1.5)
        return p_eff**power_k
    elif curve_type == "neglog":
        return -math.log(max(1e-9, 1.0 - p_eff))
    else:  # Default: sigmoid
        eps = 1e-9
        pivot_p0 = settings.get("pivot_p0", 0.20)
        curve_alpha = settings.get("curve_alpha", 2.0)
        ratio = pivot_p0 / max(eps, p_eff)
        return 1.0 / (1.0 + (ratio**curve_alpha))


def _effective_lambda(
    lambda_ui: float, mode: str, df, normalization_diag: Optional[Dict] = None
) -> Tuple[float, float]:
    """
    Compute effective lambda to use against normalized ownership [0,1].
    - If data appear normalized (own_max <= 1) OR normalization_diag indicates scaled_by=100, use UI value as-is.
    - Defensive fallback: if ownership appears in percent scale (>1), adapt ui by *0.01.
    Returns (lambda_eff, scale_used).
    """
    try:
        own_max = (
            float(df["own_proj"].max())
            if hasattr(df, "__getitem__") and "own_proj" in df.columns
            else None
        )
    except Exception:
        own_max = None

    scaled_by = None
    try:
        if normalization_diag and "ownership" in normalization_diag:
            scaled_by = normalization_diag["ownership"].get("scaled_by")
    except Exception:
        scaled_by = None

    # Prefer data signal; normalization path ensures df.own_proj in [0,1]
    own_is_normalized = (own_max is None) or (own_max <= 1.0 + 1e-9)
    if own_is_normalized or scaled_by in (100.0, 100):
        return float(lambda_ui), 1.0
    else:
        return float(lambda_ui) * 0.01, 0.01


def _calculate_ownership_penalty_term(own_pct: float, settings: Dict) -> float:
    """Calculate ownership penalty term for a single player"""
    p_eff = _effective_p(
        own_pct,
        settings.get("pivot_p0", 0.20),
        settings.get("shrink_gamma", 1.0),
        settings.get("clamp_min", 0.01),
        settings.get("clamp_max", 0.80),
    )
    return _g_curve(p_eff, settings)


def _player_level_jitter(pid: str, jitter: float, seed: Optional[int]) -> float:
    if jitter <= 0.0:
        return 0.0
    key = f"{pid}|{seed if seed is not None else 0}|qb_jitter"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    rnd = random.Random(int.from_bytes(digest, "big"))
    return rnd.uniform(-1.0, 1.0) * jitter


# ============================================================================
# PRP-OWN-19/22: Input Contract + Objective Telemetry (Wiring Check)
# ============================================================================
def _run_input_contract_and_objective_telemetry(
    spec: Any, ownership_penalty: Optional[Dict]
) -> Dict[str, Any]:
    """Validate the exact inputs used by CP-SAT and print objective scaling telemetry.

    Returns a small diagnostics dict that can be attached to solver diagnostics.
    """
    import pandas as _pd
    import hashlib as _hashlib
    import time as _time
    import os as _os

    RUN_ID = str(int(_time.time()))

    # Build a DataFrame from spec.players (this is the exact candidate set)
    rows = []
    for p in spec.players:
        rows.append(
            {
                "player_id": p.player_id,
                "name": p.name,
                "team": p.team,
                "position": (
                    "/".join(p.positions)
                    if isinstance(p.positions, (list, tuple))
                    else str(p.positions)
                ),
                "salary": p.salary,
                "FPts": float(p.proj),
                "own_proj": (
                    None if getattr(p, "own_proj", None) is None else float(p.own_proj)
                ),
            }
        )
    df = _pd.DataFrame(rows)
    # Ensure numeric ownership column for normalization logic
    try:
        df["own_proj"] = _pd.to_numeric(df["own_proj"], errors="coerce")
    except Exception:
        pass

    # Canonical required columns
    required = {"player_id", "name", "team", "position", "salary", "FPts", "own_proj"}
    missing = required - set(df.columns)
    if missing:
        raise AssertionError(f"[INPUT CONTRACT] Missing columns: {missing}")

    # Ownership normalization: normalize in-place on the exact inputs to CP-SAT
    own_max_before = (
        float(df["own_proj"].max()) if df["own_proj"].notna().any() else 0.0
    )
    scaled_by = 1.0
    if own_max_before > 1.5:
        # Percent scale detected → convert to [0,1]
        df["own_proj"] = (df["own_proj"] / 100.0).clip(0.0, 1.0)
        scaled_by = 100.0
    else:
        # Ensure values are clipped to [0,1]
        df["own_proj"] = df["own_proj"].clip(0.0, 1.0)
    # Strict: must be normalized
    if float(df["own_proj"].max()) > 1.000001:
        raise AssertionError("[INPUT CONTRACT] own_proj not normalized")
    # Stats after normalization (for logging)
    own_min = float(df["own_proj"].min()) if df["own_proj"].notna().any() else 0.0
    own_max = float(df["own_proj"].max()) if df["own_proj"].notna().any() else 0.0
    pct_over1 = float((df["own_proj"] > 1.0).mean()) * 100.0 if len(df) else 0.0

    # Propagate normalized ownership back into spec.players so the objective uses it
    try:
        own_map = {
            r["player_id"]: (None if _pd.isna(r["own_proj"]) else float(r["own_proj"]))
            for _, r in df.iterrows()
        }
        for p in spec.players:
            if p.player_id in own_map:
                p.own_proj = own_map[p.player_id]
    except Exception:
        pass

    # FPts sanity: numeric and non-constant
    if not _pd.api.types.is_numeric_dtype(df["FPts"]):
        raise AssertionError("[INPUT CONTRACT] FPts not numeric")
    if float(df["FPts"].std()) <= 0.0:
        raise AssertionError("[INPUT CONTRACT] FPts look constant/zero")

    # Contract hash of fields that impact solve
    key_cols = ["player_id", "FPts", "own_proj", "salary", "team", "position"]
    contract_key = df[key_cols].sort_values("player_id").to_csv(index=False)
    contract_hash = _hashlib.sha256(contract_key.encode()).hexdigest()[:12]

    print(
        f"[INPUT CONTRACT] run={RUN_ID} rows={df.shape[0]} "
        f"own_min={own_min:.3f} own_max={own_max:.3f} over1%={pct_over1:.2f}% "
        f"contract={contract_hash}"
    )

    # Objective telemetry (points vs lambda*penalty on same SCALE)
    SCALE = 1000

    points_scaled_total = int(df["FPts"].round(9).mul(SCALE).round().sum())
    lambda_penalty_scaled_total = 0
    lambda_ui = 0.0
    lambda_eff = 0.0
    lambda_scale_used = 1.0

    if ownership_penalty and ownership_penalty.get("enabled", False):
        lambda_ui = float(ownership_penalty.get("weight_lambda", 0.0) or 0.0)
        # Use effective lambda without mutating input dict
        lambda_eff, lambda_scale_used = _effective_lambda(
            lambda_ui,
            ownership_penalty.get("mode", "by_points"),
            df,
            {"ownership": {"scaled_by": scaled_by}},
        )
        if lambda_eff > 0 and df["own_proj"].notna().any():
            settings = ownership_penalty
            pen_sum = 0.0
            for _, row in df.iterrows():
                op = row["own_proj"]
                if _pd.isna(op):
                    continue
                own_pct = float(op)
                pen_sum += _calculate_ownership_penalty_term(own_pct, settings)
            lambda_penalty_scaled_total = int(round(SCALE * lambda_eff * pen_sum))

    ratio = (
        (lambda_penalty_scaled_total / (points_scaled_total + 1e-9))
        if points_scaled_total
        else 0.0
    )
    print(
        f"[OBJ] SCALE={SCALE} lambda_ui={lambda_ui:.3f} lambda_eff={lambda_eff:.3f} "
        f"points_scaled_total={points_scaled_total} "
        f"lambda_penalty_scaled_total={lambda_penalty_scaled_total} "
        f"ratio={ratio:.4%}"
    )

    payload = {
        "run_id": RUN_ID,
        "contract": {
            "hash": contract_hash,
            "rows": int(df.shape[0]),
            "own_min": own_min,
            "own_max": own_max,
            "pct_over_1": pct_over1,
            "own_max_before": own_max_before,
            "scaled_by": scaled_by,
        },
        "objective": {
            "scale": SCALE,
            "lambda_ui": lambda_ui,
            "lambda_eff": lambda_eff,
            "lambda_scale_used": lambda_scale_used,
            "points_scaled_total": points_scaled_total,
            "lambda_penalty_scaled_total": lambda_penalty_scaled_total,
            "ratio": ratio,
        },
    }
    # Optional artifact export (env-gated)
    try:
        if str(_os.environ.get("DFS_WRITE_CONTRACT_ARTIFACTS", "0")).lower() in (
            "1",
            "true",
            "yes",
        ):
            out_dir = _os.path.join(
                "src", "exports", f"run_{_time.strftime('%Y%m%d_%H%M%S')}"
            )
            _os.makedirs(out_dir, exist_ok=True)
            df.to_csv(_os.path.join(out_dir, "projections_used.csv"), index=False)
            with open(_os.path.join(out_dir, "contract.txt"), "w") as f:
                f.write(
                    f"""run={RUN_ID}
contract={contract_hash}
own_min={own_min}
own_max={own_max}
over1%={pct_over1}
SCALE={SCALE}
lambda_ui={lambda_ui}
lambda_eff={lambda_eff}
points_scaled_total={points_scaled_total}
lambda_penalty_scaled_total={lambda_penalty_scaled_total}
ratio={ratio}
"""
                )
            try:
                import json as _json

                with open(_os.path.join(out_dir, "telemetry.json"), "w") as jf:
                    _json.dump(payload, jf, indent=2)
            except Exception:
                pass
    except Exception:
        pass

    return payload


def _player_level_randomness(pid: str, stddev: Optional[float], randomness_pct: float, seed: Optional[int]) -> float:
    if randomness_pct <= 0.0 or stddev is None or stddev <= 0.0:
        return 0.0
    try:
        sigma = float(stddev) * float(randomness_pct) / 100.0
    except Exception:
        return 0.0
    key = f"{pid}|{seed if seed is not None else 0}|qb_randpct"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    rnd = random.Random(int.from_bytes(digest, "big"))
    return rnd.gauss(0.0, sigma)


def build_objective_weights(
    spec: Any,
    *,
    jitter: float = 0.0,
    seed: Optional[int] = None,
    scale: int = 1000,
    randomness_pct: float = 0.0,
) -> Tuple[Dict[str, int], int]:
    weights: Dict[str, int] = {}
    for p in spec.players:
        base = float(p.proj)
        noise_j = _player_level_jitter(p.player_id, jitter, seed)
        try:
            sd = float(getattr(p, "stddev", 0.0) or 0.0)
        except Exception:
            sd = 0.0
        noise_r = _player_level_randomness(p.player_id, sd, randomness_pct, seed)
        val = max(base + noise_j + noise_r, 0.0)
        weights[p.player_id] = int(round(val * scale))
    return weights, scale


def build_cpsat_model(
    spec: Any,
    *,
    jitter: float = 0.0,
    seed: Optional[int] = None,
    optimize: bool = True,
    randomness_pct: float = 0.0,
) -> "CPSATModelArtifacts":
    model, y = build_cpsat_counts(spec)
    # Use variance-aware randomness in objective if configured
    rp = float(getattr(spec, "randomness_pct", randomness_pct) or 0.0)
    weights, scale = build_objective_weights(spec, jitter=jitter, seed=seed, randomness_pct=rp)
    if optimize and weights:
        model.Maximize(
            sum(weights.get(pid, 0) * var for pid, var in y.items())
        )
    player_ids = [p.player_id for p in spec.players]
    player_vars = [y[pid] for pid in player_ids]
    return CPSATModelArtifacts(
        model=model,
        slot_vars={},
        player_vars=player_vars,
        player_ids=player_ids,
    )


def build_cpsat(spec: Any):
    """
    Build a CP-SAT model mirroring the CBC model.
    Variables: x[(player_id, pos)] ∈ {0,1}
    Constraints:
      - DK: exactly 1 per slot in [PG,SG,SF,PF,C,G,F,UTIL]
      - FD: PG/SG/SF/PF exactly 2 each, C exactly 1
      - Each player at most once
      - Salary cap (and optional min salary)
      - Team limits (global + specific)
      - Locks/Bans
    Objective: maximize projected points (scaled integers)
    """
    # Import orthogonally inside function to avoid hard dependency at import time
    from ortools.sat.python import cp_model

    m = cp_model.CpModel()

    # Index players by id for quick lookup
    pid_to_player: Dict[str, Any] = {p.player_id: p for p in spec.players}

    # Decision variables for eligible positions
    x: Dict[Tuple[str, str], Any] = {}

    # Slot list constant bound to spec (avoid hard-coding)
    DK_SLOTS = spec.roster_slots if spec.site == "dk" else ["PG", "SG", "SF", "PF", "C"]

    # Helper to build DK flex eligibility
    def eligible_positions(positions: List[str]) -> List[str]:
        base = [pos for pos in ["PG", "SG", "SF", "PF", "C"] if pos in positions]
        if spec.site == "dk":
            if any(p in ("PG", "SG") for p in base):
                base.append("G")
            if any(p in ("SF", "PF") for p in base):
                base.append("F")
            base.append("UTIL")
        return base

    # Create variables
    for p in spec.players:
        for pos in eligible_positions(p.positions):
            x[(p.player_id, pos)] = m.NewBoolVar(f"x_{p.player_id}_{pos}")

    # Pre-bucket once for speed & clarity
    slot_vars: Dict[str, List[Any]] = {pos: [] for pos in DK_SLOTS}
    from collections import defaultdict as _dd

    team_vars: Dict[str, List[Any]] = _dd(list)
    for (pid, pos), var in x.items():
        if pos in slot_vars:
            slot_vars[pos].append(var)
        team_vars[pid_to_player[pid].team].append(var)

    # Preflight infeasibility checks (fast fail before adding constraints)
    # 1) Ensure each roster slot has at least one eligible candidate
    slot_candidates = {
        pos: [pid for (pid, p) in x.keys() if p == pos] for pos in spec.roster_slots
    }
    for pos, cands in slot_candidates.items():
        if not cands:
            raise RuntimeError(
                f"No eligible players for slot {pos}. Check locks/bans/positions."
            )

    # 2) Cheap salary screen (not exact but fast):
    #    Sum of per-slot mins must not exceed cap; sum of per-slot maxes must meet min_salary (if set)
    min_sum = 0
    max_sum = 0
    for cands in slot_candidates.values():
        # candidates are pids eligible for the slot
        salaries = [pid_to_player[pid].salary for pid in cands]
        if not salaries:
            continue
        min_sum += min(salaries)
        max_sum += max(salaries)
    if min_sum > spec.salary_cap:
        raise RuntimeError(
            f"Infeasible: minimum possible salary {min_sum} exceeds cap {spec.salary_cap}."
        )
    if spec.min_salary is not None and max_sum < spec.min_salary:
        raise RuntimeError(
            f"Infeasible: even max possible salary {max_sum} < min_salary {spec.min_salary}."
        )

    # Position fill constraints (use pre-buckets)
    if spec.site == "dk":
        # Exactly one per DK slot
        for pos in DK_SLOTS:
            m.AddExactlyOne(slot_vars[pos])
    else:  # FanDuel counts
        for pos in ["PG", "SG", "SF", "PF"]:
            m.Add(sum(slot_vars.get(pos, [])) == 2)
        m.Add(sum(slot_vars.get("C", [])) == 1)

    # Each player at most once
    from collections import defaultdict

    per_player = defaultdict(list)
    for (pid, pos), var in x.items():
        per_player[pid].append(var)
    for pid, vars_ in per_player.items():
        m.AddAtMostOne(vars_)

    # Salary constraints
    salary_terms = []
    for (pid, pos), var in x.items():
        salary_terms.append(pid_to_player[pid].salary * var)
    m.Add(sum(salary_terms) <= spec.salary_cap)
    if spec.min_salary is not None:
        m.Add(sum(salary_terms) >= spec.min_salary)

    # Team limits (use pre-bucketed vars)
    if spec.team_max is not None:
        for t, vars_ in team_vars.items():
            m.Add(sum(vars_) <= spec.team_max)
    for t, cap in (spec.team_limits or {}).items():
        if t in team_vars:
            m.Add(sum(team_vars[t]) <= cap)

    # Locks / bans (use site-driven slot list)
    for pid in spec.lock_ids:
        vars_ = [x[(pid, pos)] for pos in DK_SLOTS if (pid, pos) in x]
        if not vars_:
            raise ValueError(
                f"Locked player {pid} has no eligible positions for site {spec.site}."
            )
        m.AddExactlyOne(vars_)
    for pid in spec.ban_ids:
        vars_ = [x[(pid, pos)] for pos in DK_SLOTS if (pid, pos) in x]
        if vars_:
            m.Add(sum(vars_) == 0)

    # Objective (PRP-16/OWN): projected points with optional ownership penalty
    # Initial objective; iterative solvers will override per lineup when aggro is enabled
    SCALE = 1000
    obj_terms = []
    # Prefer new functional API if a config is published; else fall back to legacy dict
    cfg, aggro = get_active_ownership_penalty()
    if cfg is not None:
        try:
            import pandas as _pd

            df = _pd.DataFrame(
                {
                    "player_id": [p.player_id for p in spec.players],
                    "proj": [float(p.proj) for p in spec.players],
                    "own_proj": [
                        None if getattr(p, "own_proj", None) is None else float(p.own_proj)
                        for p in spec.players
                    ],
                }
            )
            # lineup_idx=None here; iterative loop may override objective later
            pen = build_ownership_penalty(
                df, cfg, own_col="own_proj", lineup_idx=None, aggro=None
            )
            weights = (df["proj"].astype(float).to_numpy() + pen)
            pid_to_w = {row[0]: float(w) for row, w in zip(df[["player_id"]].itertuples(index=False, name=None), weights)}
            for (pid, pos), var in x.items():
                obj_terms.append(int(round(pid_to_w[pid] * SCALE)) * var)
            m.Maximize(sum(obj_terms))
        except Exception:
            obj_terms.clear()
    if not obj_terms:
        # Legacy path
        use_penalty = bool(
            spec.ownership_penalty and spec.ownership_penalty.get("enabled", False)
        )
        lam = 0.0
        settings = None
        if use_penalty:
            settings = spec.ownership_penalty
            try:
                lam = float(settings.get("weight_lambda", 0.0) or 0.0)
            except Exception:
                lam = 0.0
        for (pid, pos), var in x.items():
            base_proj = float(pid_to_player[pid].proj)
            eff_proj = base_proj
            if use_penalty and lam > 0:
                own_val = getattr(pid_to_player[pid], "own_proj", None)
                if own_val is not None:
                    own_pct = (own_val / 100.0) if own_val > 1.0 else float(own_val)
                    pen = lam * _calculate_ownership_penalty_term(own_pct, settings)  # type: ignore[arg-type]
                    eff_proj = base_proj - pen
            obj_terms.append(int(round(eff_proj * SCALE)) * var)
        m.Maximize(sum(obj_terms))

    return m, x


def build_cpsat_counts(spec):
    """
    Build a CP-SAT model using counts-only approach with one binary per player.
    Variables: y[player_id] ∈ {0,1} select 8 players.
    Constraints:
      - DK: at least one per base slot (PG/SG/SF/PF/C), plus one G (PG|SG), one F (SF|PF), one UTIL (any)
      - Roster size, salary cap/floor, team limits, locks/bans identical
    Objective: maximize projected points (scaled integers)
    """
    from ortools.sat.python import cp_model

    m = cp_model.CpModel()
    pid2 = {p.player_id: p for p in spec.players}
    y = {pid: m.NewBoolVar(f"y_{pid}") for pid in pid2}

    # roster size
    m.Add(sum(y.values()) == spec.lineup_size)

    # salary
    m.Add(sum(pid2[pid].salary * y[pid] for pid in y) <= spec.salary_cap)
    if spec.min_salary is not None:
        m.Add(sum(pid2[pid].salary * y[pid] for pid in y) >= spec.min_salary)

    # base elig sets
    PG = {pid for pid, p in pid2.items() if "PG" in p.positions}
    SG = {pid for pid, p in pid2.items() if "SG" in p.positions}
    SF = {pid for pid, p in pid2.items() if "SF" in p.positions}
    PF = {pid for pid, p in pid2.items() if "PF" in p.positions}
    C = {pid for pid, p in pid2.items() if "C" in p.positions}

    if spec.site == "dk":
        m.Add(sum(y[pid] for pid in PG) >= 1)
        m.Add(sum(y[pid] for pid in SG) >= 1)
        m.Add(sum(y[pid] for pid in SF) >= 1)
        m.Add(sum(y[pid] for pid in PF) >= 1)
        m.Add(sum(y[pid] for pid in C) >= 1)
        # G/F requirements
        m.Add(sum(y[pid] for pid in (PG | SG)) >= 1)
        m.Add(sum(y[pid] for pid in (SF | PF)) >= 1)
        # Strengthened DK feasibility counts to guarantee assignability
        UG = PG | SG
        UF = SF | PF
        # Need enough supply to fill (PG, SG, G) and (SF, PF, F)
        m.Add(sum(y[pid] for pid in UG) >= 3)
        m.Add(sum(y[pid] for pid in UF) >= 3)
        # Centers: at least one, at most two (C slot plus optionally UTIL)
        m.Add(sum(y[pid] for pid in C) >= 1)
        m.Add(sum(y[pid] for pid in C) <= 2)
        
        # NEW: Ensure enough coverage for flex slots
        # Without these, a lineup could have e.g. only 1 PF-eligible player
        # who must fill C (if also C-eligible), leaving no one for PF slot
        m.Add(sum(y[pid] for pid in PG) >= 2)  # PG slot + G flex backup
        m.Add(sum(y[pid] for pid in SG) >= 2)  # SG slot + G flex backup
        m.Add(sum(y[pid] for pid in SF) >= 2)  # SF slot + F flex backup
        m.Add(sum(y[pid] for pid in PF) >= 2)  # PF slot + F flex backup

    # team limits
    if spec.team_max is not None:
        teams = set(p.team for p in spec.players)
        for t in teams:
            m.Add(sum(y[pid] for pid in y if pid2[pid].team == t) <= spec.team_max)
    for t, cap in (spec.team_limits or {}).items():
        m.Add(sum(y[pid] for pid in y if pid2[pid].team == t) <= cap)

    # locks/bans
    for pid in spec.lock_ids:
        if pid not in y:
            raise ValueError(f"Locked player {pid} missing from candidate set.")
        m.Add(y[pid] == 1)
    for pid in spec.ban_ids:
        if pid in y:
            m.Add(y[pid] == 0)

    # Objective (PRP-16/OWN): default objective; iterative loop will override per lineup if cfg provided
    SCALE = 1000
    objective_terms = []
    cfg, aggro = get_active_ownership_penalty()
    if cfg is not None:
        try:
            import pandas as _pd

            df = _pd.DataFrame(
                {
                    "player_id": [p.player_id for p in spec.players],
                    "proj": [float(p.proj) for p in spec.players],
                    "own_proj": [
                        None if getattr(p, "own_proj", None) is None else float(p.own_proj)
                        for p in spec.players
                    ],
                }
            )
            pen = build_ownership_penalty(df, cfg, own_col="own_proj", lineup_idx=None, aggro=None)
            weights = (df["proj"].astype(float).to_numpy() + pen)
            pid_to_w = {row[0]: float(w) for row, w in zip(df[["player_id"]].itertuples(index=False, name=None), weights)}
            for pid in y:
                objective_terms.append(int(round(pid_to_w[pid] * SCALE)) * y[pid])
            m.Maximize(sum(objective_terms))
        except Exception:
            objective_terms.clear()
    if not objective_terms:
        for pid in y:
            player = pid2[pid]
            base_proj = player.proj
            # Apply ownership penalty if settings are provided (legacy path)
            if spec.ownership_penalty and spec.ownership_penalty.get("enabled", False):
                penalty_settings = spec.ownership_penalty
                lam = penalty_settings.get("weight_lambda", 0.0)
                if player.own_proj is not None and lam > 0:
                    own_pct = (
                        player.own_proj / 100.0
                        if player.own_proj > 1.0
                        else player.own_proj
                    )
                    penalty = lam * _calculate_ownership_penalty_term(own_pct, penalty_settings)
                    effective_proj = base_proj - penalty
                else:
                    effective_proj = base_proj
            else:
                effective_proj = base_proj
            objective_terms.append(int(round(effective_proj * SCALE)) * y[pid])
        m.Maximize(sum(objective_terms))
    return m, y


BASE = ("PG", "SG", "SF", "PF", "C")


def _player_level_jitter(pid: str, jitter: float, seed: Optional[int]) -> float:
    if jitter <= 0.0:
        return 0.0
    key = f"{pid}|{seed if seed is not None else 0}|qb_jitter"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    rnd = random.Random(int.from_bytes(digest, "big"))
    return rnd.uniform(-1.0, 1.0) * jitter


def _flex_degree(pos_list):
    pos = set(pos_list)
    deg = len(pos & set(BASE))
    if "PG" in pos or "SG" in pos:
        deg += 1  # G
    if "SF" in pos or "PF" in pos:
        deg += 1  # F
    return deg + 1  # UTIL


def assign_slots_dk(selected_pids, pid2player):
    """Deterministic DK slot assignment with matching fallback.

    Returns list of (pid, slot) on success; returns None if assignment is impossible.
    Strategy:
      1) Try fast greedy placement (previous behavior).
      2) If greedy fails, run maximum bipartite matching on the 8x8 eligibility graph.
    """

    # ---------------- Greedy attempt (fast path) ----------------
    def _flex_deg(pos_list):
        pos = set(pos_list)
        deg = len(pos & set(BASE))
        if ("PG" in pos) or ("SG" in pos):
            deg += 1  # eligible for G
        if ("SF" in pos) or ("PF" in pos):
            deg += 1  # eligible for F
        return deg + 1  # UTIL

    def _greedy():
        remaining = set(selected_pids)
        assigned = []

        # Base slots: least-flexible first
        for slot in ("PG", "SG", "SF", "PF", "C"):
            eligible = [pid for pid in remaining if slot in pid2player[pid].positions]
            if not eligible:
                return None
            eligible.sort(key=lambda pid: (_flex_deg(pid2player[pid].positions), pid))
            pick = eligible[0]
            assigned.append((pick, slot))
            remaining.remove(pick)

        # Flex slots: most-flexible first
        def _pick(slot, pred):
            cands = [pid for pid in remaining if pred(pid2player[pid].positions)]
            if not cands:
                return False
            cands.sort(key=lambda pid: (-_flex_deg(pid2player[pid].positions), pid))
            choice = cands[0]
            assigned.append((choice, slot))
            remaining.remove(choice)
            return True

        if not _pick("G", lambda pos: ("PG" in pos) or ("SG" in pos)):
            return None
        if not _pick("F", lambda pos: ("SF" in pos) or ("PF" in pos)):
            return None

        # UTIL: anyone remaining
        if not remaining:
            return None
        last = sorted(remaining)[0]
        assigned.append((last, "UTIL"))
        return assigned

    greedy = _greedy()
    if greedy is not None:
        return greedy

    # --------------- Matching fallback (robust) -----------------
    SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

    def _eligible(pid, slot):
        pos = set(pid2player[pid].positions)
        if slot in {"PG", "SG", "SF", "PF", "C"}:
            return slot in pos
        if slot == "G":
            return ("PG" in pos) or ("SG" in pos)
        if slot == "F":
            return ("SF" in pos) or ("PF" in pos)
        if slot == "UTIL":
            return True
        return False

    pids = list(dict.fromkeys(selected_pids))  # preserve order, dedup defensively
    if len(pids) != 8:
        # Expect exactly 8 selected for DK
        return None

    # Build adjacency: pid -> list of eligible slots
    adj = {pid: [s for s in SLOTS if _eligible(pid, s)] for pid in pids}

    # Quick infeasibility check: any pid with no eligible slots
    if any(len(v) == 0 for v in adj.values()):
        return None

    # Hopcroft–Karp is overkill; use simple DFS-based Kuhn since graph is tiny
    matchR = {s: None for s in SLOTS}

    def dfs(pid, seen):
        for s in adj[pid]:
            if s in seen:
                continue
            seen.add(s)
            if matchR[s] is None or dfs(matchR[s], seen):
                matchR[s] = pid
                return True
        return False

    # Try to match every pid
    for pid in pids:
        if not dfs(pid, set()):
            return None

    # Build assignment list from matchR (slots -> pid)
    assigned = []
    for slot in SLOTS:
        pid = matchR.get(slot)
        if pid is None:
            return None
        assigned.append((pid, slot))
    return assigned


@dataclass
class CPSATModelArtifacts:
    model: Any
    slot_vars: Dict[Tuple[str, str], Any]
    player_vars: List[Any]
    player_ids: List[str]


@dataclass
class SolveResult:
    lineups: List[List[Tuple[str, str]]]
    total_proj: List[float]
    total_salary: List[int]


def solve_cpsat_iterative(players: List[Dict], constraints: Any, seed: int, site: str):
    """
    Map players + constraints to Spec, build CP-SAT and generate N lineups with no-good cuts.
    Returns (lineup_objs, diagnostics)
    """
    # Import inside function to avoid hard dependency on ortools at import time
    from ortools.sat.python import cp_model

    # Apply safe pruning before building the model
    try:
        from .pruning import prune_safely
    except ImportError:
        from pruning import prune_safely
    import logging

    original_players = players.copy()  # Keep reference to original list
    original_count = len(players)
    locks_list = getattr(constraints, "lock_ids", [])
    players = prune_safely(
        players,
        locks=locks_list,
        proj_floor=getattr(constraints, "proj_min", None),
        k_per_pos=24,
        k_global=48,
        keep_value_per_pos=4,
    )
    pruned_count = len(players)

    # Log pruning summary
    reduction_pct = (
        (original_count - pruned_count) / original_count * 100
        if original_count > 0
        else 0
    )
    logging.info(
        f"Safe pruning: kept {pruned_count}/{original_count} players ({reduction_pct:.1f}% reduction)"
    )

    if locks_list:
        locks_kept = [
            pid for pid in locks_list if any(p["player_id"] == pid for p in players)
        ]
        logging.info(
            f"Locks auto-kept: {len(locks_kept)}/{len(locks_list)} ({locks_kept})"
        )

    # Show top pruned players (those with highest projections that were removed)
    if original_count > pruned_count:
        kept_ids = {p["player_id"] for p in players}
        pruned_players = [p for p in original_players if p["player_id"] not in kept_ids]
        if pruned_players:
            top_pruned = sorted(pruned_players, key=lambda x: x["proj"], reverse=True)[
                :3
            ]
            top_pruned_info = [f"{p['name']} ({p['proj']:.1f})" for p in top_pruned]
            logging.info(f"Top pruned players: {', '.join(top_pruned_info)}")

    # Build Spec
    try:
        from .model_spec import Spec, SpecPlayer
    except ImportError:
        from model_spec import Spec, SpecPlayer

    roster = (
        ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
        if site == "dk"
        else ["PG", "SG", "SF", "PF", "C"]
    )

    # PRP-16: Convert ownership penalty settings to dict for Spec
    ownership_penalty_dict = None
    if hasattr(constraints, "ownership_penalty") and constraints.ownership_penalty:
        ownership_penalty_dict = {
            "enabled": constraints.ownership_penalty.enabled,
            "mode": constraints.ownership_penalty.mode,
            "weight_lambda": constraints.ownership_penalty.weight_lambda,
            "curve_type": constraints.ownership_penalty.curve_type,
            "power_k": constraints.ownership_penalty.power_k,
            "pivot_p0": constraints.ownership_penalty.pivot_p0,
            "curve_alpha": constraints.ownership_penalty.curve_alpha,
            "clamp_min": constraints.ownership_penalty.clamp_min,
            "clamp_max": constraints.ownership_penalty.clamp_max,
            "shrink_gamma": constraints.ownership_penalty.shrink_gamma,
        }

    spec = Spec(
        site=site,  # type: ignore
        roster_slots=roster,
        salary_cap=constraints.max_salary or (50000 if site == "dk" else 60000),
        min_salary=constraints.min_salary,
        players=[
            SpecPlayer(
                player_id=p["player_id"],
                name=p["name"],
                team=p["team"],
                positions=p["positions"],
                salary=p["salary"],
                proj=p["proj"],
                dk_id=p.get("dk_id"),
                own_proj=p.get("own_proj"),  # PRP-16: Pass ownership data
            )
            for p in players
        ],
        team_max=(
            constraints.global_team_limit
            if constraints.global_team_limit is not None
            else (4 if site == "dk" else None)
        ),
        team_limits=constraints.team_limits or {},
        lock_ids=[pid for pid in getattr(constraints, "lock_ids", [])],
        ban_ids=[pid for pid in getattr(constraints, "ban_ids", [])],
        lineup_size=8 if site == "dk" else 9,
        N_lineups=constraints.N_lineups,
        unique_players=max(0, constraints.unique_players),
        cp_sat_params=getattr(constraints, "cp_sat_params", {}) or {},
        engine="cp_sat",
        ownership_penalty=ownership_penalty_dict,  # PRP-16: Pass ownership penalty settings
    )

    # Wiring check: Input contract + objective telemetry (prints + returns diagnostics)
    wiring_diag = None
    try:
        wiring_diag = _run_input_contract_and_objective_telemetry(
            spec, ownership_penalty_dict
        )
    except AssertionError as _ae:
        # Fail fast with a clear message
        raise RuntimeError(str(_ae))
    except Exception:
        # Non-fatal; continue solve
        wiring_diag = None

    # Build model
    model, x = build_cpsat(spec)
    num_bool_vars = len(x)

    # Slot locks (DraftKings late swap): force a specific player into a specific slot.
    lock_slots = getattr(constraints, "lock_slots", None) or {}
    if lock_slots:
        seen_players = set()
        for slot, pid in lock_slots.items():
            slot_s = str(slot)
            pid_s = str(pid)
            if pid_s in seen_players:
                raise ValueError(f"lock_slots assigns player {pid_s} to multiple slots")
            seen_players.add(pid_s)
            key = (pid_s, slot_s)
            if key not in x:
                raise ValueError(
                    f"Locked slot {slot_s} requires player {pid_s} but no eligible variable exists (player not in pool or not eligible)."
                )
            model.Add(x[key] == 1)

    # Configure solver with strict, safe defaults
    solver = cp_model.CpSolver()
    params = spec.cp_sat_params or {}

    def _flt(key: str, default: float) -> float:
        try:
            return float(params.get(key, default))
        except Exception:
            return default

    # Speed-safe defaults
    time_limit = _flt("max_time_seconds", 0.7)
    if time_limit <= 0:
        time_limit = 0.7
    solver.parameters.max_time_in_seconds = time_limit
    # Adaptive time cap (off by default unless enabled via params)
    adaptive_time = bool(params.get("adaptive_time", False))
    time_cap = time_limit

    rel_gap = _flt("relative_gap_limit", 0.001)
    if rel_gap > 0:
        # only set if > 0 to allow full optimality when requested
        try:
            solver.parameters.relative_gap_limit = rel_gap
        except Exception:
            pass

    det_time = _flt("max_deterministic_time", 0.0)
    if det_time > 0:
        solver.parameters.max_deterministic_time = det_time

    solver.parameters.num_search_workers = int(
        params.get("num_search_workers", 0)
    )  # 0=all cores
    solver.parameters.random_seed = int(params.get("random_seed", seed))
    # Prefer portfolio search for speed/robustness when available (Pylance-safe)
    try:
        from ortools.sat import sat_parameters_pb2 as sat_pb2

        solver.parameters.search_branching = sat_pb2.SatParameters.PORTFOLIO_SEARCH  # type: ignore[reportAttributeAccessIssue]
    except Exception:
        pass
    # Optional verbose logging controlled via params
    solver.parameters.log_search_progress = bool(
        params.get("log_search_progress", False)
    )

    # Helper indices
    pid_to_player: Dict[str, Any] = {p.player_id: p for p in spec.players}

    def extract_lineup():
        chosen = [(pid, pos) for (pid, pos), var in x.items() if solver.Value(var) == 1]
        # Sum salary and proj by players actually chosen
        total_salary = 0
        total_proj = 0.0
        # chosen contains one variable per player due to <=1 constraint
        for pid, pos in chosen:
            sp = pid_to_player[pid]
            total_salary += sp.salary
            total_proj += sp.proj
        return chosen, total_salary, total_proj

    results: List[Tuple[List[Tuple[str, str]], int, float]] = []
    built = 0
    last_status = None

    while built < spec.N_lineups:
        # Rebuild objective weights per iteration if new penalty config is active
        cfg_active, aggro_active = get_active_ownership_penalty()
        if cfg_active is not None:
            try:
                import pandas as _pd
                SCALE = 1000
                _df = _pd.DataFrame(
                    {
                        "player_id": [p.player_id for p in spec.players],
                        "proj": [float(p.proj) for p in spec.players],
                        "own_proj": [
                            None if getattr(p, "own_proj", None) is None else float(p.own_proj)
                            for p in spec.players
                        ],
                    }
                )
                _pen = build_ownership_penalty(
                    _df,
                    cfg_active,
                    own_col="own_proj",
                    lineup_idx=built,
                    aggro=aggro_active,
                )
                _w = (_df["proj"].astype(float).to_numpy() + _pen)
                _pid2w = {row[0]: float(w) for row, w in zip(_df[["player_id"]].itertuples(index=False, name=None), _w)}
                _terms = [int(round(_pid2w[pid] * SCALE)) * var for (pid, _pos), var in x.items()]
                model.Maximize(sum(_terms))
                # Minimal logging of curve + lambda
                try:
                    eff_lambda = (
                        aggro_active.lambda_for(built) if aggro_active is not None else cfg_active.lambda1
                    )
                    if aggro_active is not None:
                        print(
                            f"[own] curve={cfg_active.curve.value} lambda1={cfg_active.lambda1:.3f} eff_lambda={eff_lambda:.3f} (lineup {built})"
                        )
                except Exception:
                    pass
            except Exception:
                pass
        # set the per-iteration time budget (may be adapted)
        solver.parameters.max_time_in_seconds = time_cap
        status = solver.Solve(model)
        last_status = status
        # compute gap for adaptation/diagnostics
        try:
            curr_obj = float(solver.ObjectiveValue())
            curr_bound = float(solver.BestObjectiveBound())
            curr_gap = abs(curr_bound - curr_obj) / max(1.0, abs(curr_obj))
        except Exception:
            curr_obj, curr_bound, curr_gap = None, None, None

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break

        chosen, sal, proj = extract_lineup()
        results.append((chosen, sal, proj))
        # Adapt next iteration's time cap if enabled
        if adaptive_time and curr_gap is not None:
            if status == cp_model.OPTIMAL or curr_gap <= 0.002:  # ~0.2%
                time_cap = max(0.25, time_cap * 0.8)
            elif curr_gap > 0.01:  # >1%
                time_cap = min(1.2, time_cap + 0.2)
        built += 1

        # Player-level no-good cut with min-uniques
        selected_pids = {pid for (pid, _pos) in chosen}
        selected_vars_all_pos = [
            var for (pid, pos), var in x.items() if pid in selected_pids
        ]
        if spec.unique_players <= 0:
            # Forbid the exact lineup
            model.Add(sum(selected_vars_all_pos) <= len(selected_pids) - 1)
        else:
            # Enforce min-uniques vs previous solution
            model.Add(
                sum(selected_vars_all_pos) <= len(selected_pids) - spec.unique_players
            )

        # Optional: provide hint for next solution
        model.ClearHints()
        for pid, pos in chosen:
            model.AddHint(x[(pid, pos)], 1)
        # Lightly discourage reusing other positions of the same players
        for (pid, pos), var in x.items():
            if pid in selected_pids and (pid, pos) not in chosen:
                model.AddHint(var, 0)

    # Convert to application Lineup/Player
    try:
        from .optimizer_types import Player, Lineup
    except ImportError:  # pragma: no cover - fallback for loose execution contexts
        from projections.optimizer.optimizer_types import Player, Lineup  # type: ignore

    lineup_objs: List[Any] = []
    for idx, (chosen, sal, proj) in enumerate(results, start=1):
        # Keep chosen order as DK slot order
        if spec.site == "dk":
            slot_order_map = {s: i for i, s in enumerate(spec.roster_slots)}
        else:
            slot_order_map = {"PG": 0, "SG": 1, "SF": 2, "PF": 3, "C": 4}
        chosen_sorted = sorted(
            chosen, key=lambda t: (slot_order_map.get(t[1], 99), t[0])
        )
        players_out: List[Any] = []
        for pid, pos in chosen_sorted:
            sp = pid_to_player[pid]
            # Find original player dict to get optional fields
            orig_player = next((p for p in players if p["player_id"] == pid), {})
            players_out.append(
                Player(
                    player_id=pid,
                    name=sp.name,
                    pos=pos,
                    team=sp.team,
                    salary=sp.salary,
                    proj=sp.proj,
                    dk_id=sp.dk_id,
                    own_proj=orig_player.get("own_proj"),
                    minutes=orig_player.get("minutes"),
                    stddev=orig_player.get("stddev"),
                )
            )
        lineup_objs.append(
            Lineup(
                lineup_id=idx,
                total_proj=round(proj, 2),
                total_salary=sal,
                players=players_out,
            )
        )

    status_name = (
        solver.StatusName(last_status) if last_status is not None else "UNKNOWN"
    )
    try:
        best_obj = float(solver.ObjectiveValue())
    except Exception:
        best_obj = None
    try:
        best_bound = float(solver.BestObjectiveBound())
    except Exception:
        best_bound = None
    # Calculate pruning statistics for diagnostics
    reduction_pct = (
        (original_count - pruned_count) / original_count * 100
        if original_count > 0
        else 0
    )
    pruned_players_info = None
    if original_count > pruned_count:
        kept_ids = {p["player_id"] for p in players}
        pruned_players_list = [
            p for p in original_players if p["player_id"] not in kept_ids
        ]
        if pruned_players_list:
            top_pruned = sorted(
                pruned_players_list, key=lambda x: x["proj"], reverse=True
            )[:3]
            pruned_players_info = [f"{p['name']} ({p['proj']:.1f})" for p in top_pruned]

    diagnostics = {
        "engine": "cp_sat",
        "N": len(lineup_objs),
        "status": status_name,
        "best_obj": best_obj,
        "best_bound": best_bound,
        "achieved_gap": (
            None
            if best_obj is None or best_bound is None
            else abs(best_bound - best_obj) / max(1.0, abs(best_obj))
        ),
        "wall_time_sec": getattr(solver, "WallTime", lambda: None)(),
        "model": {
            "num_bool_vars": num_bool_vars,
            "num_slots": len(spec.roster_slots),
        },
        "params": {
            "max_time_in_seconds": solver.parameters.max_time_in_seconds,
            "relative_gap_limit": getattr(
                solver.parameters, "relative_gap_limit", None
            ),
            "max_deterministic_time": solver.parameters.max_deterministic_time,
            "num_search_workers": solver.parameters.num_search_workers,
            "random_seed": solver.parameters.random_seed,
            "adaptive_time": adaptive_time,
            "final_time_cap": time_cap,
        },
        # PRP-13: Safe Position-Aware Pruning diagnostics
        "pruning": {
            "enabled": True,
            "original_players": original_count,
            "kept_players": pruned_count,
            "reduction_pct": round(reduction_pct, 1),
            "top_pruned": pruned_players_info,
            "locks_kept": (
                len(
                    [
                        pid
                        for pid in locks_list
                        if any(p["player_id"] == pid for p in players)
                    ]
                )
                if locks_list
                else 0
            ),
        },
        "wiring_check": wiring_diag,
    }
    return lineup_objs, diagnostics


def solve_cpsat_iterative_counts(
    players: List[Dict], constraints: Any, seed: int, site: str
):
    """
    Map players + constraints to Spec, build counts-based CP-SAT and generate N lineups.
    Uses one binary variable per player and deterministic slot assignment post-solve.
    Returns (lineup_objs, diagnostics)
    """
    # Counts-only can't enforce slot-level locks; route to per-slot solver.
    if getattr(constraints, "lock_slots", None):
        return solve_cpsat_iterative(players, constraints, seed, site)

    # Import inside function to avoid hard dependency on ortools at import time
    from ortools.sat.python import cp_model

    # Apply safe pruning before building the model
    try:
        from .pruning import prune_safely
    except ImportError:
        from pruning import prune_safely
    import logging

    original_players = players.copy()  # Keep reference to original list
    original_count = len(players)
    locks_list = getattr(constraints, "lock_ids", [])
    players = prune_safely(
        players,
        locks=locks_list,
        proj_floor=getattr(constraints, "proj_min", None),
        k_per_pos=24,
        k_global=48,
        keep_value_per_pos=4,
    )
    pruned_count = len(players)

    # Log pruning summary
    reduction_pct = (
        (original_count - pruned_count) / original_count * 100
        if original_count > 0
        else 0
    )
    logging.info(
        f"Safe pruning: kept {pruned_count}/{original_count} players ({reduction_pct:.1f}% reduction)"
    )

    if locks_list:
        locks_kept = [
            pid for pid in locks_list if any(p["player_id"] == pid for p in players)
        ]
        logging.info(
            f"Locks auto-kept: {len(locks_kept)}/{len(locks_list)} ({locks_kept})"
        )

    # Show top pruned players (those with highest projections that were removed)
    if original_count > pruned_count:
        kept_ids = {p["player_id"] for p in players}
        pruned_players = [p for p in original_players if p["player_id"] not in kept_ids]
        if pruned_players:
            top_pruned = sorted(pruned_players, key=lambda x: x["proj"], reverse=True)[
                :3
            ]
            top_pruned_info = [f"{p['name']} ({p['proj']:.1f})" for p in top_pruned]
            logging.info(f"Top pruned players: {', '.join(top_pruned_info)}")

    # Build Spec
    try:
        from .model_spec import Spec, SpecPlayer
    except ImportError:
        from model_spec import Spec, SpecPlayer

    roster = (
        ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
        if site == "dk"
        else ["PG", "SG", "SF", "PF", "C"]
    )

    # PRP-16: Convert ownership penalty settings to dict for Spec
    ownership_penalty_dict = None
    if hasattr(constraints, "ownership_penalty") and constraints.ownership_penalty:
        ownership_penalty_dict = {
            "enabled": constraints.ownership_penalty.enabled,
            "mode": constraints.ownership_penalty.mode,
            "weight_lambda": constraints.ownership_penalty.weight_lambda,
            "curve_type": constraints.ownership_penalty.curve_type,
            "power_k": constraints.ownership_penalty.power_k,
            "pivot_p0": constraints.ownership_penalty.pivot_p0,
            "curve_alpha": constraints.ownership_penalty.curve_alpha,
            "clamp_min": constraints.ownership_penalty.clamp_min,
            "clamp_max": constraints.ownership_penalty.clamp_max,
            "shrink_gamma": constraints.ownership_penalty.shrink_gamma,
        }

    spec = Spec(
        site=site,  # type: ignore
        roster_slots=roster,
        salary_cap=constraints.max_salary or (50000 if site == "dk" else 60000),
        min_salary=constraints.min_salary,
        players=[
            SpecPlayer(
                player_id=p["player_id"],
                name=p["name"],
                team=p["team"],
                positions=p["positions"],
                salary=p["salary"],
                proj=p["proj"],
                dk_id=p.get("dk_id"),
                own_proj=p.get("own_proj"),  # PRP-16: Pass ownership data
            )
            for p in players
        ],
        team_max=(
            constraints.global_team_limit
            if constraints.global_team_limit is not None
            else (4 if site == "dk" else None)
        ),
        team_limits=constraints.team_limits or {},
        lock_ids=[pid for pid in getattr(constraints, "lock_ids", [])],
        ban_ids=[pid for pid in getattr(constraints, "ban_ids", [])],
        lineup_size=8 if site == "dk" else 9,
        N_lineups=constraints.N_lineups,
        unique_players=max(0, constraints.unique_players),
        cp_sat_params=getattr(constraints, "cp_sat_params", {}) or {},
        engine="cp_sat",
        ownership_penalty=ownership_penalty_dict,  # PRP-16: Pass ownership penalty settings
    )

    # Counts-only currently implemented for DK only; fallback to per-slot solver for others
    if site != "dk":
        return solve_cpsat_iterative(players, constraints, seed, site)

    # Wiring check: Input contract + objective telemetry (prints + returns diagnostics)
    wiring_diag = None
    try:
        wiring_diag = _run_input_contract_and_objective_telemetry(
            spec, ownership_penalty_dict
        )
    except AssertionError as _ae:
        # Fail fast with a clear message
        raise RuntimeError(str(_ae))
    except Exception:
        wiring_diag = None

    # Build counts model
    model, y = build_cpsat_counts(spec)
    num_bool_vars = len(y)

    # Configure solver with strict, safe defaults
    solver = cp_model.CpSolver()
    params = spec.cp_sat_params or {}

    def _flt(key: str, default: float) -> float:
        try:
            return float(params.get(key, default))
        except Exception:
            return default

    # Speed-safe defaults
    time_limit = _flt("max_time_seconds", 0.7)
    if time_limit <= 0:
        time_limit = 0.7
    solver.parameters.max_time_in_seconds = time_limit
    # Adaptive time cap (off by default unless enabled via params)
    adaptive_time = bool(params.get("adaptive_time", False))
    time_cap = time_limit

    rel_gap = _flt("relative_gap_limit", 0.001)
    if rel_gap > 0:
        # only set if > 0 to allow full optimality when requested
        try:
            solver.parameters.relative_gap_limit = rel_gap
        except Exception:
            pass

    det_time = _flt("max_deterministic_time", 0.0)
    if det_time > 0:
        solver.parameters.max_deterministic_time = det_time

    solver.parameters.num_search_workers = int(
        params.get("num_search_workers", 0)
    )  # 0=all cores
    solver.parameters.random_seed = int(params.get("random_seed", seed))
    # Prefer portfolio search for speed/robustness when available (Pylance-safe)
    try:
        from ortools.sat import sat_parameters_pb2 as sat_pb2

        solver.parameters.search_branching = sat_pb2.SatParameters.PORTFOLIO_SEARCH  # type: ignore[reportAttributeAccessIssue]
    except Exception:
        pass
    # Optional verbose logging controlled via params
    solver.parameters.log_search_progress = bool(
        params.get("log_search_progress", False)
    )

    # Helper indices
    pid_to_player: Dict[str, Any] = {p.player_id: p for p in spec.players}

    def extract_selected():
        chosen = [pid for pid, var in y.items() if solver.Value(var) == 1]
        sal = sum(
            next(pp for pp in spec.players if pp.player_id == pid).salary
            for pid in chosen
        )
        proj = sum(
            next(pp for pp in spec.players if pp.player_id == pid).proj
            for pid in chosen
        )
        return chosen, sal, proj

    assigned_lineups: List[Tuple[List[Tuple[str, str]], int, float]] = []
    built = 0
    last_status = None

    while built < spec.N_lineups:
        # Rebuild objective per iteration if active cfg present
        cfg_active, aggro_active = get_active_ownership_penalty()
        if cfg_active is not None:
            try:
                import pandas as _pd
                SCALE = 1000
                _df = _pd.DataFrame(
                    {
                        "player_id": [p.player_id for p in spec.players],
                        "proj": [float(p.proj) for p in spec.players],
                        "own_proj": [
                            None if getattr(p, "own_proj", None) is None else float(p.own_proj)
                            for p in spec.players
                        ],
                    }
                )
                _pen = build_ownership_penalty(
                    _df,
                    cfg_active,
                    own_col="own_proj",
                    lineup_idx=built,
                    aggro=aggro_active,
                )
                _w = (_df["proj"].astype(float).to_numpy() + _pen)
                _pid2w = {row[0]: float(w) for row, w in zip(_df[["player_id"]].itertuples(index=False, name=None), _w)}
                _terms = [int(round(_pid2w[pid] * SCALE)) * y[pid] for pid in y]
                model.Maximize(sum(_terms))
                # Minimal logging of curve + lambda when aggro is enabled
                try:
                    eff_lambda = (
                        aggro_active.lambda_for(built) if aggro_active is not None else cfg_active.lambda1
                    )
                    if aggro_active is not None:
                        print(
                            f"[own] curve={cfg_active.curve.value} lambda1={cfg_active.lambda1:.3f} eff_lambda={eff_lambda:.3f} (lineup {built})"
                        )
                except Exception:
                    pass
            except Exception:
                pass
        # set the per-iteration time budget (may be adapted)
        solver.parameters.max_time_in_seconds = time_cap
        status = solver.Solve(model)
        last_status = status
        # compute gap for adaptation/diagnostics
        try:
            curr_obj = float(solver.ObjectiveValue())
            curr_bound = float(solver.BestObjectiveBound())
            curr_gap = abs(curr_bound - curr_obj) / max(1.0, abs(curr_obj))
        except Exception:
            curr_obj, curr_bound, curr_gap = None, None, None

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break

        chosen, sal, proj = extract_selected()
        assigned = assign_slots_dk(chosen, pid_to_player)
        if assigned is None:
            # Unassignable set (rare with strengthened counts): cut & retry without incrementing built
            model.Add(sum(y[pid] for pid in chosen) <= spec.lineup_size - 1)
            continue

        assigned_lineups.append((assigned, sal, proj))

        # Adapt next iteration's time cap if enabled
        if adaptive_time and curr_gap is not None:
            if status == cp_model.OPTIMAL or curr_gap <= 0.002:  # ~0.2%
                time_cap = max(0.25, time_cap * 0.8)
            elif curr_gap > 0.01:  # >1%
                time_cap = min(1.2, time_cap + 0.2)
        built += 1

        # No-good cuts (on the selected set)
        if spec.unique_players <= 0:
            model.Add(sum(y[pid] for pid in chosen) <= spec.lineup_size - 1)
        else:
            model.Add(
                sum(y[pid] for pid in chosen) <= spec.lineup_size - spec.unique_players
            )

        # Hints
        model.ClearHints()
        for pid in chosen:
            model.AddHint(y[pid], 1)

    # Convert to application Lineup/Player using slot assignment
    try:
        from .optimizer_types import Player, Lineup
    except ImportError:  # pragma: no cover - fallback for loose execution contexts
        from projections.optimizer.optimizer_types import Player, Lineup  # type: ignore

    lineup_objs: List[Any] = []
    if not assigned_lineups:
        raise RuntimeError(
            "CP-SAT counts-only: no assignable lineups found. Check locks/bans and strengthened DK counts."
        )

    for idx, (assigned_slots, sal, proj) in enumerate(assigned_lineups, start=1):
        # Sort by slot order for consistent output (DK-only path here)
        slot_order_map = {s: i for i, s in enumerate(spec.roster_slots)}
        assigned_slots_sorted = sorted(
            assigned_slots, key=lambda t: (slot_order_map.get(t[1], 99), t[0])
        )

        players_out: List[Any] = []
        for pid, pos in assigned_slots_sorted:
            sp = pid_to_player[pid]
            # Find original player dict to get optional fields
            orig_player = next((p for p in players if p["player_id"] == pid), {})
            players_out.append(
                Player(
                    player_id=pid,
                    name=sp.name,
                    pos=pos,
                    team=sp.team,
                    salary=sp.salary,
                    proj=sp.proj,
                    dk_id=sp.dk_id,
                    own_proj=orig_player.get("own_proj"),
                    minutes=orig_player.get("minutes"),
                    stddev=orig_player.get("stddev"),
                )
            )
        lineup_objs.append(
            Lineup(
                lineup_id=idx,
                total_proj=round(proj, 2),
                total_salary=sal,
                players=players_out,
            )
        )

    status_name = (
        solver.StatusName(last_status) if last_status is not None else "UNKNOWN"
    )
    try:
        best_obj = float(solver.ObjectiveValue())
    except Exception:
        best_obj = None
    try:
        best_bound = float(solver.BestObjectiveBound())
    except Exception:
        best_bound = None
    # Calculate pruning statistics for diagnostics
    reduction_pct = (
        (original_count - pruned_count) / original_count * 100
        if original_count > 0
        else 0
    )
    pruned_players_info = None
    if original_count > pruned_count:
        kept_ids = {p["player_id"] for p in players}
        pruned_players_list = [
            p for p in original_players if p["player_id"] not in kept_ids
        ]
        if pruned_players_list:
            top_pruned = sorted(
                pruned_players_list, key=lambda x: x["proj"], reverse=True
            )[:3]
            pruned_players_info = [f"{p['name']} ({p['proj']:.1f})" for p in top_pruned]

    diagnostics = {
        "engine": "cp_sat_counts",
        "N": len(lineup_objs),
        "status": status_name,
        "best_obj": best_obj,
        "best_bound": best_bound,
        "achieved_gap": (
            None
            if best_obj is None or best_bound is None
            else abs(best_bound - best_obj) / max(1.0, abs(best_obj))
        ),
        "wall_time_sec": getattr(solver, "WallTime", lambda: None)(),
        "model": {
            "num_bool_vars": num_bool_vars,
            "num_slots": len(spec.roster_slots),
        },
        "params": {
            "max_time_in_seconds": solver.parameters.max_time_in_seconds,
            "relative_gap_limit": getattr(
                solver.parameters, "relative_gap_limit", None
            ),
            "max_deterministic_time": solver.parameters.max_deterministic_time,
            "num_search_workers": solver.parameters.num_search_workers,
            "random_seed": solver.parameters.random_seed,
            "adaptive_time": adaptive_time,
            "final_time_cap": time_cap,
        },
        # PRP-13: Safe Position-Aware Pruning diagnostics
        "pruning": {
            "enabled": True,
            "original_players": original_count,
            "kept_players": pruned_count,
            "reduction_pct": round(reduction_pct, 1),
            "top_pruned": pruned_players_info,
            "locks_kept": (
                len(
                    [
                        pid
                        for pid in locks_list
                        if any(p["player_id"] == pid for p in players)
                    ]
                )
                if locks_list
                else 0
            ),
        },
        "wiring_check": wiring_diag,
    }
    return lineup_objs, diagnostics
