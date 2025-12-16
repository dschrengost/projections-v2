"""Thin adapter used by the Sheets runner to call the CP-SAT optimizer.

The original version of this module exposed several backends (PuLP, SCIP) and
an ownership-penalty objective. After the refactor to a pure CP-SAT + fpts
optimizer we keep the public surface (`solve`, `SchemaError`, `OptimizerError`)
but streamline the implementation to match the new capabilities:

* DraftKings only
* Salary cap/min-salary, locks/bans, projection floor, min uniques
* Simple metrics and DataFrame output compatible with the Sheets UI
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from ortools.sat.python import cp_model


DK_POSITIONS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
BASE_POSITIONS = {"PG", "SG", "SF", "PF", "C"}


class SchemaError(ValueError):
    """Raised when projections are missing required columns or types."""


class OptimizerError(RuntimeError):
    """Raised when the optimizer cannot produce lineups."""


@dataclass
class Candidate:
    pid: str
    name: str
    team: str
    salary: int
    proj: float
    ownership: float
    base_positions: List[str]
    slots: List[str]


def solve(projections: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Entry point used by the sheets runner.

    Args:
        projections: DataFrame with columns [player_id, name, team, pos, proj, salary].
        config: mapping with optimizer options (mirrors Config sheet keys).

    Returns:
        (lineups_df, metrics): DataFrame with DK slot columns + summary dict.
    """

    start = time.time()
    df = _validate_and_normalize_projections(projections)

    site = str(config.get("site", "dk")).lower()
    if site != "dk":
        raise OptimizerError(f"Only DraftKings is supported (requested site='{site}').")

    backend = str(config.get("backend", "cpsat")).strip().lower()
    if backend not in {"cpsat", "cp-sat"}:
        raise OptimizerError(f"Unsupported backend '{backend}'. Use 'cpsat'.")

    proj_min = config.get("proj_min")
    if proj_min is not None:
        try:
            proj_min = float(proj_min)
        except Exception:
            proj_min = None

    candidates, match_pct = _build_candidates_from_df(
        df,
        proj_min=proj_min,
    )

    if not candidates:
        raise OptimizerError("No eligible players after filtering projections.")

    require_ids = bool(config.get("require_dk_ids", False))
    min_match_pct = config.get("min_dk_id_match_rate")
    if min_match_pct is not None:
        try:
            min_match_pct = float(min_match_pct)
        except Exception:
            min_match_pct = None
    if require_ids and min_match_pct is not None and match_pct < min_match_pct:
        raise OptimizerError(
            f"DK ID match rate {match_pct:.1f}% below required threshold {min_match_pct:.1f}%."
        )

    num_lineups = max(0, int(config.get("n_lineups") or 20))
    if num_lineups == 0:
        empty = _empty_results_df()
        return empty, _build_metrics(config, empty, start)

    min_uniques = max(0, int(config.get("min_uniques") or 1))
    lineup_size = len(DK_POSITIONS)
    if min_uniques > lineup_size:
        min_uniques = lineup_size

    salary_cap = int(config.get("dk_salary_cap") or config.get("salary_cap") or 50000)
    min_salary = config.get("min_salary")
    if min_salary is None:
        min_salary = 49000
    else:
        min_salary = int(min_salary)

    locks = _parse_id_list(
        config.get("lock_ids") or config.get("locks")
    )
    bans = _parse_id_list(
        config.get("ban_ids") or config.get("bans")
    )

    threads = config.get("threads")
    threads = None if threads in (None, "", []) else int(threads)
    time_limit = config.get("time_limit_sec")
    time_limit = None if time_limit in (None, "") else float(time_limit)
    seed = config.get("seed")
    seed = None if seed in (None, "") else int(seed)

    team_max = config.get("max_per_team")
    team_max = None if team_max in (None, "") else int(team_max)
    team_caps = _parse_team_limits(config.get("team_limits"))

    lineups = _solve_cp_sat(
        candidates=candidates,
        num_lineups=num_lineups,
        min_uniques=min_uniques,
        salary_cap=salary_cap,
        min_salary=min_salary,
        lock_ids=locks,
        ban_ids=bans,
        team_max=team_max,
        team_caps=team_caps,
        time_limit=time_limit,
        threads=threads,
        seed=seed,
    )

    df_out = _lineups_to_dataframe(lineups, candidates)
    metrics = _build_metrics(config, df_out, start)
    return df_out, metrics


# ---------------------------------------------------------------------------
# Projection preprocessing
# ---------------------------------------------------------------------------


def _validate_and_normalize_projections(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"player_id", "name", "team", "pos", "proj", "salary"}
    missing = required_cols - set(c.lower() for c in df.columns)
    if missing:
        raise SchemaError(f"Missing required columns: {sorted(missing)}")

    df_norm = df.copy()
    df_norm.columns = [c.lower() for c in df_norm.columns]

    df_norm["player_id"] = df_norm["player_id"].astype(str).str.strip()
    df_norm["name"] = df_norm["name"].astype(str).str.strip()
    df_norm["team"] = df_norm["team"].astype(str).str.strip().str.upper()

    try:
        df_norm["salary"] = pd.to_numeric(df_norm["salary"], errors="coerce")
    except Exception as exc:
        raise SchemaError("salary column must be numeric") from exc

    try:
        df_norm["proj"] = pd.to_numeric(df_norm["proj"], errors="coerce")
    except Exception as exc:
        raise SchemaError("proj column must be numeric") from exc

    if "ownership" in df_norm.columns:
        df_norm["ownership"] = pd.to_numeric(df_norm["ownership"], errors="coerce")
    else:
        df_norm["ownership"] = 0.0

    df_norm = df_norm[df_norm["name"].ne("") & df_norm["team"].ne("")]
    df_norm = df_norm.dropna(subset=["proj", "salary"])

    # Normalize positions to lists
    df_norm["positions"] = df_norm["pos"].astype(str).str.split("/")
    df_norm["positions"] = df_norm["positions"].apply(
        lambda items: [p.strip().upper() for p in items if p.strip()]
    )

    return df_norm.reset_index(drop=True)


def _build_candidates_from_df(
    df: pd.DataFrame,
    *,
    proj_min: Optional[float] = None,
) -> tuple[List[Candidate], float]:
    seen: set[str] = set()
    candidates: List[Candidate] = []
    matched_ids = 0

    for row in df.itertuples(index=False):
        try:
            proj_val = float(row.proj)
        except Exception:
            continue
        if proj_min is not None and proj_val < proj_min:
            continue

        base_positions = [p for p in row.positions if p in BASE_POSITIONS]
        if not base_positions:
            continue

        pid = str(row.player_id).strip()
        if not pid or pid.lower() == "nan":
            pid = _normalize_name(row.name)
        else:
            matched_ids += 1

        if pid in seen:
            continue

        try:
            salary = int(float(row.salary))
        except Exception:
            continue

        ownership = float(row.ownership) if row.ownership is not None else 0.0
        slots = _expand_positions(base_positions)
        candidates.append(
            Candidate(
                pid=pid,
                name=str(row.name).strip(),
                team=str(row.team).strip().upper(),
                salary=salary,
                proj=proj_val,
                ownership=ownership,
                base_positions=base_positions,
                slots=slots,
            )
        )
        seen.add(pid)

    total_rows = len(df)
    match_pct = (matched_ids / total_rows * 100.0) if total_rows else 0.0
    return candidates, match_pct


def _normalize_name(name: str) -> str:
    token = name.strip().lower()
    token = token.replace(" ", "_").replace("-", "_")
    return token


def _expand_positions(base_positions: Sequence[str]) -> List[str]:
    ordered = []
    for pos in ["PG", "SG", "SF", "PF", "C"]:
        if pos in base_positions and pos not in ordered:
            ordered.append(pos)
    if any(pos in ("PG", "SG") for pos in base_positions):
        ordered.append("G")
    if any(pos in ("SF", "PF") for pos in base_positions):
        ordered.append("F")
    ordered.append("UTIL")
    return ordered


def _parse_id_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        tokens = [t.strip() for t in value.replace(";", ",").split(",")]
        return [t for t in tokens if t]
    if isinstance(value, Iterable):
        return [str(t).strip() for t in value if str(t).strip()]
    return []


def _parse_team_limits(value: Any) -> Dict[str, int]:
    if not value:
        return {}
    if isinstance(value, dict):
        try:
            return {str(k).upper(): int(v) for k, v in value.items()}
        except Exception:
            return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        # Accept JSON-ish input first
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
                return _parse_team_limits(parsed)
            except Exception:
                pass
        parts = [p.strip() for p in text.split(",") if p.strip()]
        caps: Dict[str, int] = {}
        for part in parts:
            if ":" not in part:
                continue
            team, cap = part.split(":", 1)
            try:
                caps[team.strip().upper()] = int(cap)
            except Exception:
                continue
        return caps
    return {}


# ---------------------------------------------------------------------------
# CP-SAT solver
# ---------------------------------------------------------------------------


def _solve_cp_sat(
    *,
    candidates: Sequence[Candidate],
    num_lineups: int,
    min_uniques: int,
    salary_cap: int,
    min_salary: int,
    lock_ids: Sequence[str],
    ban_ids: Sequence[str],
    team_max: Optional[int],
    team_caps: Dict[str, int],
    time_limit: Optional[float],
    threads: Optional[int],
    seed: Optional[int],
) -> List[Dict[str, Candidate]]:
    model = cp_model.CpModel()

    player_vars: Dict[str, cp_model.IntVar] = {}
    slot_entries: Dict[str, List[tuple[str, cp_model.IntVar]]] = {slot: [] for slot in DK_POSITIONS}
    team_buckets: Dict[str, List[cp_model.IntVar]] = {}

    for cand in candidates:
        y = model.NewBoolVar(f"y_{cand.pid}")
        player_vars[cand.pid] = y
        team_buckets.setdefault(cand.team, []).append(y)

        child_vars: List[cp_model.IntVar] = []
        for slot in cand.slots:
            if slot not in slot_entries:
                continue
            var = model.NewBoolVar(f"x_{cand.pid}_{slot}")
            slot_entries[slot].append((cand.pid, var))
            child_vars.append(var)
        if not child_vars:
            # Candidate does not cover any active slot (should not happen for DK)
            model.Add(y == 0)
            continue
        model.Add(sum(child_vars) == y)

    for slot, entries in slot_entries.items():
        if not entries:
            raise OptimizerError(f"No candidates eligible for slot '{slot}'.")
        model.AddExactlyOne(var for _, var in entries)

    lineup_size = len(DK_POSITIONS)
    model.Add(sum(player_vars.values()) == lineup_size)

    salary_expr = sum(
        cand.salary * player_vars[cand.pid] for cand in candidates if cand.pid in player_vars
    )
    model.Add(salary_expr <= salary_cap)
    if min_salary and min_salary > 0:
        model.Add(salary_expr >= min_salary)

    lock_set = set(lock_ids)
    ban_set = set(ban_ids)

    for pid in lock_set:
        if pid not in player_vars:
            raise OptimizerError(f"Locked player '{pid}' is not in the candidate pool.")
        model.Add(player_vars[pid] == 1)

    for pid in ban_set:
        if pid in player_vars:
            model.Add(player_vars[pid] == 0)

    if team_max is not None and team_max >= 0:
        for team, vars_ in team_buckets.items():
            model.Add(sum(vars_) <= team_max)

    for team, cap in team_caps.items():
        vars_ = team_buckets.get(team)
        if vars_:
            model.Add(sum(vars_) <= cap)

    scale = 1000
    objective_terms = []
    for cand in candidates:
        if cand.pid in player_vars:
            objective_terms.append(int(round(cand.proj * scale)) * player_vars[cand.pid])
    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    if time_limit and time_limit > 0:
        solver.parameters.max_time_in_seconds = float(time_limit)
    if threads is not None:
        solver.parameters.num_search_workers = max(0, int(threads))
    if seed is not None:
        solver.parameters.random_seed = int(seed)

    results: List[Dict[str, Candidate]] = []
    cand_lookup = {cand.pid: cand for cand in candidates}

    for idx in range(num_lineups):
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break

        lineup: Dict[str, Candidate] = {}
        for slot in DK_POSITIONS:
            entries = slot_entries[slot]
            assigned_pid = None
            for pid, var in entries:
                if solver.BooleanValue(var):
                    assigned_pid = pid
                    break
            if assigned_pid is None:
                raise OptimizerError(f"Solver returned incomplete assignment for slot '{slot}'.")
            lineup[slot] = cand_lookup[assigned_pid]
        results.append(lineup)

        if min_uniques > 0:
            selected_pids = {cand.pid for cand in lineup.values() if cand.pid in player_vars}
            model.Add(
                sum(player_vars[pid] for pid in selected_pids)
                <= len(selected_pids) - min_uniques
            )

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _lineups_to_dataframe(
    lineups: Sequence[Dict[str, Candidate]],
    candidates: Sequence[Candidate],
) -> pd.DataFrame:
    columns = DK_POSITIONS + ["salary", "proj", "own_sum", "ev", "dup_risk"]
    rows: List[List[str]] = []

    for lineup in lineups:
        total_salary = 0
        total_proj = 0.0
        total_own = 0.0
        row: List[str] = []
        for slot in DK_POSITIONS:
            cand = lineup[slot]
            total_salary += cand.salary
            total_proj += cand.proj
            total_own += cand.ownership if cand.ownership is not None else 0.0
            display = f"{cand.name} ({cand.pid})"
            row.append(display)
        row.extend(
            [
                str(total_salary),
                f"{total_proj:.2f}",
                f"{total_own:.1f}",
                "",
                "",
            ]
        )
        rows.append(row)

    if not rows:
        return _empty_results_df()
    return pd.DataFrame(rows, columns=columns)


def _empty_results_df() -> pd.DataFrame:
    columns = DK_POSITIONS + ["salary", "proj", "own_sum", "ev", "dup_risk"]
    return pd.DataFrame(columns=columns)


def _build_metrics(config: dict, lineups_df: pd.DataFrame, start_time: float) -> dict:
    runtime = round(time.time() - start_time, 3)
    produced = int(len(lineups_df))

    if produced:
        avg_proj = float(pd.to_numeric(lineups_df["proj"], errors="coerce").astype(float).mean())
        avg_salary = float(pd.to_numeric(lineups_df["salary"], errors="coerce").astype(float).mean())
    else:
        avg_proj = 0.0
        avg_salary = 0.0

    metrics = {
        "Lineups Generated": produced,
        "Avg Proj": round(avg_proj, 2),
        "Avg Salary": round(avg_salary, 0),
        "Runtime (s)": runtime,
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    requested = config.get("n_lineups")
    if requested is not None:
        try:
            metrics["Requested lineups"] = int(requested)
        except Exception:
            pass

    return metrics


def export_lineups_csv(lineups_df: pd.DataFrame, filename: str = "lineups.csv") -> str:
    lineups_df.to_csv(filename, index=False)
    return filename
