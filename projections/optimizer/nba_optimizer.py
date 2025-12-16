#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import datetime as dt
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Any

import numpy as np
import pulp as plp


# -------------------------
# Data model
# -------------------------

TeamFix = {
    "PHO": "PHX",
    "GS": "GSW",
    "SA": "SAS",
    "NO": "NOP",
    "NY": "NYK",
}

DK_POSITIONS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
FD_POSITIONS = ["PG", "PG", "SG", "SG", "SF", "SF", "PF", "PF", "C"]

DK_LINEUP_SIZE = 8
FD_LINEUP_SIZE = 9


@dataclass
class Player:
    # Key for dicts: (name_norm, pos_str, team_norm) where name_norm replaces '-' with '#'
    name: str
    team: str
    pos_str: str
    salary: int
    fpts: float
    minutes: float
    ownership: float
    stddev: float
    positions: List[str]  # expanded DK G/F/UTIL too (for DK)
    pid: str  # DK: int-like str; FD: slate Id (may include '#')
    matchup: str
    gametime: Optional[dt.datetime] = None


# -------------------------
# Helpers
# -------------------------


def _lower_first(iterator: Iterable[str]) -> Iterable[str]:
    it = iter(iterator)
    first = next(it)
    yield first.lower()
    yield from it


def _normalize_team(t: str) -> str:
    return TeamFix.get(t, t)


def _norm_name(n: str) -> str:
    return n.replace("-", "#")


def _parse_gametime(s: str) -> Optional[dt.datetime]:
    # expects like "MM/DD/YYYY HH:MMAM/PM" — trimmed outside
    try:
        return dt.datetime.strptime(s, "%m/%d/%Y %I:%M%p")
    except Exception:
        return None


def _matchup_to_set(m: str) -> Tuple[str, str]:
    # handle "LAL@PHX" or "LAL-PHX" variants uniformly
    s = m.replace(" ", "").replace("@", "-")
    parts = [x for x in s.split("-") if x]
    if len(parts) == 2:
        return tuple(sorted((_normalize_team(parts[0]), _normalize_team(parts[1]))))
    return (m, m)


# -------------------------
# Optimizer
# -------------------------


class NBAOptimizer:
    def __init__(
        self,
        site: str,
        num_lineups: int,
        num_uniques: int,
        config_path: str,
        projection_path: Optional[str] = None,
        player_ids_path: Optional[str] = None,
        randomness: Optional[float] = None,
        min_lineup_salary: Optional[int] = None,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        lambda_own: Optional[float] = None,
        lambda_var: Optional[float] = None,
        time_limit_sec: Optional[int] = None,
        threads: Optional[int] = None,
        deterministic: bool = False,
        backend: str = "pulp",
    ):
        self.site = site.lower()
        assert self.site in {"dk", "fd"}, "site must be 'dk' or 'fd'"
        self.num_lineups = int(num_lineups)
        self.num_uniques = int(num_uniques)
        # RIPPED OUT: in-driver PuLP problem moved behind backend adapter
        self.output_dir = output_dir

        # CLI knobs (new)
        self.lambda_own: float = float(lambda_own or 0.0)
        self.lambda_var: float = float(lambda_var or 0.0)
        self.time_limit_sec: Optional[int] = (
            int(time_limit_sec) if time_limit_sec is not None else None
        )
        self.threads: Optional[int] = int(threads) if threads is not None else None
        self.deterministic: bool = bool(deterministic)

        # Backend selection (modularize solver)
        self.backend: str = (backend or "pulp").lower()
        if self.backend not in {"pulp", "cpsat", "scip"}:
            raise ValueError("backend must be 'pulp', 'cpsat', or 'scip'")

        # Load config, allow CLI overrides
        with open(config_path, encoding="utf-8-sig") as jf:
            self.config = json.load(jf)

        if projection_path:
            self.config["projection_path"] = projection_path
        if player_ids_path:
            self.config["player_path"] = player_ids_path
        if randomness is not None:
            self.config["randomness"] = float(randomness)
        if min_lineup_salary is not None:
            self.config["min_lineup_salary"] = int(min_lineup_salary)
        if self.deterministic:
            # Force deterministic behavior: no jitter from randomness
            self.config["randomness"] = 0.0

        self.at_most: Dict[str, List[List[str]]] = self.config.get("at_most", {})
        self.at_least: Dict[str, List[List[str]]] = self.config.get("at_least", {})
        self.team_limits: Dict[str, int] = self.config.get("team_limits", {})
        self.global_team_limit: Optional[int] = (
            int(self.config.get("global_team_limit", 0)) or None
        )
        self.projection_minimum: float = float(self.config.get("projection_minimum", 0))
        self.randomness: float = float(self.config.get("randomness", 0.0))
        self.matchup_limits: Dict[str, int] = self.config.get("matchup_limits", {})
        self.matchup_at_least: Dict[str, int] = self.config.get("matchup_at_least", {})
        self.min_lineup_salary: int = int(self.config.get("min_lineup_salary", 0))

        # NOTE: legacy group/team/matchup limits are still honored here for backward compatibility.
        # Roadmap scope will remove these later; Step 1 only updates CLI and objective weights.

        # Paths
        base_dir = os.path.dirname(os.path.abspath(config_path))
        proj_path = os.path.join(base_dir, self.config["projection_path"])
        pid_path = os.path.join(base_dir, self.config["player_path"])

        # RNG
        self.rng = np.random.default_rng(seed)

        # Load data
        self.players: Dict[Tuple[str, str, str], Player] = {}
        self.team_list: List[str] = []
        self.matchup_list: List[str] = []
        self._load_projections(proj_path)
        self._join_player_ids(pid_path)
        self._detect_ownership_units()

        # Pre-alloc
        self.lp_vars: Dict[Tuple[Tuple[str, str, str], str, str], plp.LpVariable] = {}
        self.selected_lineups: List[
            List[Tuple[Tuple[str, str, str], str, str]]
        ] = []  # per solve

    # ---------- IO ----------

    def _load_projections(self, path: str):
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(_lower_first(f))
            for row in reader:
                fpts = float(row["fpts"])
                if fpts < self.projection_minimum:
                    continue
                name = row["name"]
                team = _normalize_team(row["team"])
                pos_str = row["position"]
                salary = int(row["salary"].replace(",", ""))
                minutes = float(row.get("minutes", 0) or 0)
                ownership = float(row.get("own%", row.get("own", 0)) or 0)
                stddev = float(row.get("stddev", 0) or 0)
                positions = [p.strip() for p in pos_str.split("/") if p.strip()]

                # DK expands G/F/UTIL eligibility
                if self.site == "dk":
                    if "PG" in positions or "SG" in positions:
                        positions = list(dict.fromkeys(positions + ["G"]))
                    if "SF" in positions or "PF" in positions:
                        positions = list(dict.fromkeys(positions + ["F"]))
                    positions = list(dict.fromkeys(positions + ["UTIL"]))

                key = (_norm_name(name), pos_str, team)
                self.players[key] = Player(
                    name=name,
                    team=team,
                    pos_str=pos_str,
                    salary=salary,
                    fpts=fpts,
                    minutes=minutes,
                    ownership=ownership,
                    stddev=stddev,
                    positions=positions,
                    pid="",  # filled in join step
                    matchup="",
                )
                if team not in self.team_list:
                    self.team_list.append(team)

    def _join_player_ids(self, path: str):
        # Build simple index by (name, team, position-string)
        idx = {k: v for k, v in self.players.items()}

        missing: List[Tuple[str, str, str]] = []
        matched = 0

        with open(path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.site == "dk":
                    name_key = "Name"
                    team_key = "TeamAbbrev"
                    pos_key = "Position"
                    pid_key = "ID"
                    game_info = row.get("Game Info", "")
                    matchup = (game_info.split(" ")[0] if game_info else "").strip()
                    rest = " ".join(game_info.split()[1:]) if game_info else ""
                    # DK formats like "MM/DD/YYYY HH:MMAM" then "ET"
                    gt_raw = rest[:-3] if rest.endswith(" ET") else rest
                    gametime = _parse_gametime(gt_raw) if gt_raw else None
                else:
                    name_key = "Nickname"
                    team_key = "Team"
                    pos_key = "Position"
                    pid_key = "Id"
                    matchup = row.get("Game", "")
                    gametime = None

                pname = _norm_name(row[name_key])
                pteam = _normalize_team(row[team_key])
                ppos = row[pos_key]

                key = (pname, ppos, pteam)
                if key in idx:
                    p = idx[key]
                    p.pid = str(row[pid_key]).replace("-", "#")
                    p.matchup = matchup.replace("PHO", "PHX").replace("GS", "GSW")
                    p.gametime = gametime
                    matched += 1

                    if p.matchup and p.matchup not in self.matchup_list:
                        self.matchup_list.append(p.matchup)
                else:
                    # We don’t know yet whether this is a true mismatch or just a player we filtered out by fpts
                    continue

        # Validate that every kept projection has an ID
        for k, p in idx.items():
            if not p.pid:
                missing.append(k)

        if missing:
            msg = [
                "ERROR: Some projection rows could not be matched to player_ids.csv.",
                "Common causes: team abbrev mismatch, name mismatch, or position-string mismatch.",
                "Examples (name_norm, pos_str, team):",
            ]
            for ex in missing[:15]:
                msg.append(f"  - {ex}")
            msg.append(f"... ({len(missing)} total). Fix inputs and retry.")
            raise RuntimeError("\n".join(msg))

    # ---------- Units & weights ----------

    def _detect_ownership_units(self) -> None:
        """Detect whether ownership is provided as [0,1] fraction or [0,100] percent."""
        if not self.players:
            self.own_is_percent = True
            return
        max_own = max(abs(p.ownership) for p in self.players.values())
        # Heuristic: if anything clearly above 1.0, treat as percent
        self.own_is_percent: bool = bool(max_own > 1.0000001)
        unit = "percent [0,100]" if self.own_is_percent else "fraction [0,1]"
        print(f"[info] Detected ownership units: {unit}")

    def _own_to_fraction(self, v: float) -> float:
        """Convert ownership to [0,1] fraction regardless of input units."""
        if getattr(self, "own_is_percent", True):
            return float(v) / 100.0
        return float(v)

    # ---------- Constraints payload (backend-agnostic) ----------
    def _build_constraints_payload(self) -> Dict[str, object]:
        """Prepare a neutral constraints dict to pass into solver backends."""
        lineup_size = DK_LINEUP_SIZE if self.site == "dk" else FD_LINEUP_SIZE
        max_salary = 50000 if self.site == "dk" else 60000
        min_salary = 49000 if self.site == "dk" else 59000
        if self.min_lineup_salary:
            min_salary = self.min_lineup_salary
        return {
            "site": self.site,
            "lineup_size": lineup_size,
            "max_salary": max_salary,
            "min_salary": min_salary,
            "num_lineups": self.num_lineups,
            "num_uniques": self.num_uniques,
            "randomness": self.randomness,
            "lambda_own": self.lambda_own,
            "lambda_var": self.lambda_var,
            "time_limit_sec": self.time_limit_sec,
            "threads": self.threads,
            "deterministic": self.deterministic,
        }

    # ---------- QuickBuild helpers ----------

    def _quick_build_player_payload(self) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for player in self.players.values():
            pid = str(player.pid)
            if pid in seen:
                continue
            seen.add(pid)
            positions = []
            for pos in player.positions:
                if pos not in positions:
                    positions.append(pos)
            payload.append(
                {
                    "player_id": pid,
                    "name": player.name,
                    "team": player.team,
                    "positions": positions,
                    "salary": int(player.salary),
                    "proj": float(player.fpts),
                    "dk_id": pid,
                    "own_proj": self._own_to_fraction(player.ownership),
                    "stddev": float(player.stddev) if player.stddev is not None else None,
                }
            )
        return payload

    def _quick_build_constraints_payload(self) -> Dict[str, Any]:
        payload = self._build_constraints_payload().copy()
        payload.update(
            {
                "global_team_limit": self.global_team_limit,
                "team_limits": self.team_limits,
                "cp_sat_params": self.config.get("cp_sat_params"),
            }
        )
        return payload

    def _pid_lookup(self) -> Dict[str, Player]:
        lookup: Dict[str, Player] = {}
        for player in self.players.values():
            lookup[str(player.pid)] = player
        return lookup

    def _write_quick_build_pool_csv(
        self,
        pool_lineups: List[Tuple[str, ...]],
        cfg,
    ) -> int:
        if not cfg.output_path:
            return 0
        if self.site != "dk":
            raise NotImplementedError("QuickBuild export currently supports DraftKings only.")

        from .cpsat_solver import assign_slots_dk
        from .quick_build import fnv1a_64

        output_path = Path(cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pid_lookup = self._pid_lookup()
        wrote = 0
        skipped = 0
        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            header = ["LineupID", *DK_POSITIONS, "salary", "proj", "own_sum", "hash64"]
            writer.writerow(header)
            for idx, lineup in enumerate(pool_lineups):
                try:
                    assignment = assign_slots_dk(lineup, pid_lookup)
                except KeyError:
                    skipped += 1
                    continue
                if not assignment:
                    skipped += 1
                    continue
                slot_map = {slot: pid for (pid, slot) in assignment}
                row_names = []
                total_salary = 0
                total_proj = 0.0
                total_own = 0.0
                missing_slot = False
                for slot in DK_POSITIONS:
                    pid = slot_map.get(slot)
                    if not pid or pid not in pid_lookup:
                        missing_slot = True
                        break
                    pl = pid_lookup[pid]
                    row_names.append(f"{pl.name} ({pid})")
                    total_salary += int(pl.salary)
                    total_proj += float(pl.fpts)
                    total_own += self._own_to_fraction(pl.ownership)
                if missing_slot:
                    skipped += 1
                    continue
                hash64 = f"{fnv1a_64(lineup):016x}"
                row = [idx] + row_names + [total_salary, f"{total_proj:.4f}", f"{total_own:.4f}", hash64]
                writer.writerow(row)
                wrote += 1

        if skipped:
            print(f"[quick-build] Skipped {skipped} lineup(s) due to slot assignment issues.")
        print(f"[quick-build] Wrote {wrote} lineup(s) to {output_path}")
        return wrote

    def run_quick_build(self, qb_cfg) -> None:
        from .quick_build import quick_build_pool

        player_payload = self._quick_build_player_payload()
        constraints = self._quick_build_constraints_payload()
        result = quick_build_pool(player_payload, self.site, constraints, qb_cfg, run_id=qb_cfg.run_id)
        lineups = result.lineups
        print(f"[quick-build] Received {len(lineups)} unique lineup(s) from workers.")
        written = self._write_quick_build_pool_csv(lineups, result.config)
        print(
            "[quick-build] Stats: "
            + json.dumps(result.stats.to_dict(), indent=2)
        )
        if written == 0:
            print("[quick-build] WARNING: No lineups were written to the pool CSV.")

    # ---------- Model ----------
    def _create_variables(self):
        # RIPPED OUT: moved behind backend adapter (_PulpBackendAdapter)
        raise NotImplementedError("Use backend adapter via NBAOptimizer.solve()")

    def _set_objective(self, jitter: bool):
        # RIPPED OUT: moved behind backend adapter (_PulpBackendAdapter)
        raise NotImplementedError("Use backend adapter via NBAOptimizer.solve()")

    def _add_salary_constraints(self):
        # RIPPED OUT: moved behind backend adapter (_PulpBackendAdapter)
        raise NotImplementedError("Use backend adapter via NBAOptimizer.solve()")

    def _add_position_constraints(self):
        # RIPPED OUT: moved behind backend adapter (_PulpBackendAdapter)
        raise NotImplementedError("Use backend adapter via NBAOptimizer.solve()")

    def _build_model(self):
        # RIPPED OUT: moved behind backend adapter (_PulpBackendAdapter)
        raise NotImplementedError("Use backend adapter via NBAOptimizer.solve()")

    # RIPPED OUT (scope): team/matchup/group limits handled by legacy code, not used in lean scope.

    # ---------- Solve ----------

    def solve(self):
        """Backend dispatcher: assemble payload, then call the selected adapter."""
        # Clamp num_uniques to lineup size (guardrail stays in driver)
        lineup_size = DK_LINEUP_SIZE if self.site == "dk" else FD_LINEUP_SIZE
        if self.num_uniques > lineup_size:
            print(
                f"[warn] num_uniques {self.num_uniques} exceeds lineup size {lineup_size}; clamping."
            )
            self.num_uniques = lineup_size

        constraints = self._build_constraints_payload()
        self.selected_lineups.clear()

        if self.backend == "pulp":
            # Use internal PuLP adapter for now (legacy path)
            adapter = _PulpBackendAdapter(self, constraints)
            self.selected_lineups = adapter.solve()
            return

        if self.backend == "cpsat":
            # Local import to avoid hard dependency when using PuLP backend
            try:
                from .backends.cpsat_adapter import solve as cpsat_solve
            except ImportError:
                from backends.cpsat_adapter import solve as cpsat_solve
            self.selected_lineups = cpsat_solve(self.players, constraints, seed=None)
            return

        if self.backend == "scip":
            # Local import to avoid hard dependency otherwise
            try:
                from .backends.scip_adapter import solve as scip_solve
            except ImportError:
                from backends.scip_adapter import solve as scip_solve
            self.selected_lineups = scip_solve(self.players, constraints, seed=None)
            return

        raise ValueError(f"Unknown backend: {self.backend}")

    # ---------- Output ----------

    def write_output(self):
        if not self.selected_lineups:
            print("No lineups to write.")
            return

        out_dir = self.output_dir or os.path.join(
            os.path.dirname(__file__), "../output"
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            f"{self.site}_optimal_lineups_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)

            if self.site == "dk":
                writer.writerow(
                    [
                        "PG",
                        "SG",
                        "SF",
                        "PF",
                        "C",
                        "G",
                        "F",
                        "UTIL",
                        "Salary",
                        "Fpts Proj",
                        "Own. Prod.",
                        "Own. Sum.",
                        "Minutes",
                        "StdDev",
                    ]
                )
                for lineup in self.selected_lineups:
                    ordered = self._sort_lineup_for_export(list(lineup))
                    ordered = self._late_swap_adjust(ordered)
                    keys = [k for _, k in ordered]
                    salary = sum(self.players[k].salary for k in keys)
                    fpts_sum = round(sum(self.players[k].fpts for k in keys), 2)
                    own_prod = float(
                        np.prod(
                            [
                                self._own_to_fraction(self.players[k].ownership)
                                for k in keys
                            ]
                        )
                    )
                    own_sum = sum(self.players[k].ownership for k in keys)
                    mins = sum(self.players[k].minutes for k in keys)
                    sdev = sum(self.players[k].stddev for k in keys)

                    row = []
                    for _, key in ordered:
                        p = self.players[key]
                        row.append(f"{p.name} ({p.pid})")
                    row += [salary, fpts_sum, own_prod, own_sum, mins, sdev]
                    writer.writerow(row)

            else:
                writer.writerow(
                    [
                        "PG",
                        "PG",
                        "SG",
                        "SG",
                        "SF",
                        "SF",
                        "PF",
                        "PF",
                        "C",
                        "Salary",
                        "Fpts Proj",
                        "Own. Prod.",
                        "Own. Sum.",
                        "Minutes",
                        "StdDev",
                    ]
                )
                for lineup in self.selected_lineups:
                    ordered = self._sort_lineup_for_export(list(lineup))
                    keys = [k for _, k in ordered]
                    salary = sum(self.players[k].salary for k in keys)
                    fpts_sum = round(sum(self.players[k].fpts for k in keys), 2)
                    own_prod = float(
                        np.prod(
                            [
                                self._own_to_fraction(self.players[k].ownership)
                                for k in keys
                            ]
                        )
                    )
                    own_sum = sum(self.players[k].ownership for k in keys)
                    mins = sum(self.players[k].stddev for k in keys)
                    sdev = sum(self.players[k].stddev for k in keys)

                    row = []
                    for _, key in ordered:
                        p = self.players[key]
                        row.append(f"{p.pid.replace('#', '-')}:{p.name}")
                    row += [salary, fpts_sum, own_prod, own_sum, mins, sdev]
                    writer.writerow(row)

        print(f"Output written: {out_path}")

    def _sort_lineup_for_export(
        self, lineup: List[Tuple[Tuple[str, str, str], str, str]]
    ) -> List[Tuple[str, Tuple[str, str, str]]]:
        # returns ordered list of player-keys matching export order
        if self.site == "dk":
            order = DK_POSITIONS
            out = [None] * 8
            for key, pos, pid in lineup:
                idx = order.index(pos)
                if out[idx] is None:
                    out[idx] = (pos, key)
                else:
                    # UTIL/G/F duplicates can collide; place next slot
                    # Find next same-pos slot if any (DK has single each, except UTIL/G/F)
                    j = idx
                    while j < len(out) and out[j] is not None:
                        j += 1
                    if j < len(out):
                        out[j] = (pos, key)
            # compact None if any (shouldn't happen for valid solution)
            return [x for x in out if x is not None]
        else:
            order = FD_POSITIONS
            counts = {"PG": 0, "SG": 0, "SF": 0, "PF": 0, "C": 0}
            out = [None] * 9
            for pos_target in order:
                # pick a player with that pos that's not yet placed
                for idx, (key, pos, pid) in enumerate(lineup):
                    if pos == pos_target and lineup[idx] is not None:
                        out_index = order.index(pos_target, counts[pos_target])
                        out[out_index] = (pos, key)
                        counts[pos_target] += 1
                        lineup[idx] = None  # mark used
                        break
            return [x for x in out if x is not None]

    def _late_swap_adjust(
        self, ordered: List[Tuple[str, Tuple[str, str, str]]]
    ) -> List[Tuple[str, Tuple[str, str, str]]]:
        if self.site == "fd":
            return ordered

        pos_map = {
            0: ["PG"],
            1: ["SG"],
            2: ["SF"],
            3: ["PF"],
            4: ["C"],
            5: ["PG", "SG"],
            6: ["SF", "PF"],
            7: ["PG", "SG", "SF", "PF", "C"],
        }

        sorted_lineup = list(ordered)

        def swap_if_needed(i: int, j: int):
            pos_i, key_i = sorted_lineup[i]
            pos_j, key_j = sorted_lineup[j]
            pi = self.players[key_i]
            pj = self.players[key_j]
            if pi.gametime and pj.gametime and pi.gametime > pj.gametime:
                # ensure eligibility both ways
                if any(p in pos_map[i] for p in pj.positions) and any(
                    p in pos_map[j] for p in pi.positions
                ):
                    sorted_lineup[i], sorted_lineup[j] = (
                        sorted_lineup[j],
                        sorted_lineup[i],
                    )

        for i in range(5):
            for j in range(5, 8):
                swap_if_needed(i, j)

        return sorted_lineup


# -------------------------
# Backend Adapters
# -------------------------
class _PulpBackendAdapter:
    """
    Legacy PuLP backend kept as a pluggable adapter.
    Implements the minimal slot model + salary min/max and uniqueness cuts.
    """

    def __init__(self, opt: "NBAOptimizer", constraints: Dict[str, object]):
        self.opt = opt
        self.constraints = constraints
        self.problem = plp.LpProblem("NBA", plp.LpMaximize)
        self.lp_vars: Dict[Tuple[Tuple[str, str, str], str, str], plp.LpVariable] = {}

    def _create_variables(self):
        self.lp_vars.clear()
        for key, p in self.opt.players.items():
            for pos in p.positions:
                self.lp_vars[(key, pos, p.pid)] = plp.LpVariable(
                    name=f"{key[0]}_{pos}_{p.pid}", cat=plp.LpBinary
                )

    def _set_objective(self, jitter: bool):
        terms = []
        eps = 1e-9
        for (key, pos, pid), var in self.lp_vars.items():
            p = self.opt.players[key]
            base = float(p.fpts)
            if jitter and self.opt.randomness > 0 and p.stddev > 0:
                sd = p.stddev * (self.opt.randomness / 100.0)
                base = float(self.opt.rng.normal(loc=base, scale=sd))
            p_frac = self.opt._own_to_fraction(p.ownership)
            pen = (self.opt.lambda_own * p_frac) + (
                self.opt.lambda_var * float(p.stddev or 0.0)
            )
            eff = base - pen
            if eff <= 0:
                eff = eps
            terms.append(eff * var)
        self.problem.setObjective(plp.lpSum(terms))

    def _add_salary_constraints(self):
        max_salary = 50000 if self.opt.site == "dk" else 60000
        min_salary = 49000 if self.opt.site == "dk" else 59000
        if self.opt.min_lineup_salary:
            min_salary = self.opt.min_lineup_salary
        self.problem += (
            plp.lpSum(
                self.opt.players[key].salary * var
                for (key, _, _), var in self.lp_vars.items()
            )
            <= max_salary,
            "MaxSalary",
        )
        self.problem += (
            plp.lpSum(
                self.opt.players[key].salary * var
                for (key, _, _), var in self.lp_vars.items()
            )
            >= min_salary,
            "MinSalary",
        )

    def _add_position_constraints(self):
        if self.opt.site == "dk":
            for pos in DK_POSITIONS:
                self.problem += (
                    plp.lpSum(
                        var for (key, p, _), var in self.lp_vars.items() if p == pos
                    )
                    == 1,
                    f"MustHave1_{pos}",
                )
            # at most once per player
            for key in self.opt.players:
                pid = self.opt.players[key].pid
                self.problem += (
                    plp.lpSum(
                        self.lp_vars[(key, p, pid)]
                        for p in self.opt.players[key].positions
                    )
                    <= 1,
                    f"Once_{key}",
                )
        else:
            need = {"PG": 2, "SG": 2, "SF": 2, "PF": 2, "C": 1}
            for pos, cnt in need.items():
                self.problem += (
                    plp.lpSum(
                        var for (key, p, _), var in self.lp_vars.items() if p == pos
                    )
                    == cnt,
                    f"Count_{pos}_{cnt}",
                )
            # NOTE: RIPPED OUT: FanDuel 'max 4 per team' (out of scope)
            for key in self.opt.players:
                pid = self.opt.players[key].pid
                self.problem += (
                    plp.lpSum(
                        self.lp_vars[(key, p, pid)]
                        for p in self.opt.players[key].positions
                    )
                    <= 1,
                    f"Once_{key}",
                )

    def _build_model(self):
        self.problem = plp.LpProblem("NBA", plp.LpMaximize)
        self._create_variables()
        self._set_objective(jitter=False)
        self._add_salary_constraints()
        self._add_position_constraints()

    def solve(self) -> List[List[Tuple[Tuple[str, str, str], str, str]]]:
        self._build_model()
        selected_lineups: List[List[Tuple[Tuple[str, str, str], str, str]]] = []
        for i in range(int(self.constraints["num_lineups"])):
            # Refresh objective with jitter each iteration
            self._set_objective(jitter=True)
            try:
                self.problem.solve(plp.PULP_CBC_CMD(msg=0))
            except plp.PulpSolverError:
                print(
                    f"Infeasible at lineup {i}. Generated {len(selected_lineups)} / {self.constraints['num_lineups']}."
                )
                break
            if plp.LpStatus[self.problem.status] != "Optimal":
                print(
                    f"Infeasible (status={plp.LpStatus[self.problem.status]}) at lineup {i}. "
                    f"Generated {len(selected_lineups)} / {self.constraints['num_lineups']}."
                )
                break
            selected = [
                (key, pos, pid)
                for (key, pos, pid), var in self.lp_vars.items()
                if var.value() == 1
            ]
            if not selected:
                print(f"No selection at iteration {i}. Stopping.")
                break
            selected_lineups.append(selected)
            # Uniqueness cut for next iteration
            lineup_size = len(selected)
            cut_vars = [self.lp_vars[(key, pos, pid)] for (key, pos, pid) in selected]
            self.problem += (
                plp.lpSum(cut_vars)
                <= lineup_size - int(self.constraints["num_uniques"]),
                f"UniqCut_{i}",
            )
            if (i + 1) % 50 == 0 or i == 0:
                print(f"Generated {i + 1}/{self.constraints['num_lineups']}…")
        return selected_lineups

    # ---------- Output ----------

    def write_output(self):
        if not self.selected_lineups:
            print("No lineups to write.")
            return

        out_dir = self.output_dir or os.path.join(
            os.path.dirname(__file__), "../output"
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            f"{self.site}_optimal_lineups_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)

            if self.site == "dk":
                writer.writerow(
                    [
                        "PG",
                        "SG",
                        "SF",
                        "PF",
                        "C",
                        "G",
                        "F",
                        "UTIL",
                        "Salary",
                        "Fpts Proj",
                        "Own. Prod.",
                        "Own. Sum.",
                        "Minutes",
                        "StdDev",
                    ]
                )
                for lineup in self.selected_lineups:
                    ordered = self._sort_lineup_for_export(list(lineup))
                    ordered = self._late_swap_adjust(ordered)
                    keys = [k for _, k in ordered]
                    salary = sum(self.players[k].salary for k in keys)
                    fpts_sum = round(sum(self.players[k].fpts for k in keys), 2)
                    own_prod = float(
                        np.prod(
                            [
                                self._own_to_fraction(self.players[k].ownership)
                                for k in keys
                            ]
                        )
                    )
                    own_sum = sum(self.players[k].ownership for k in keys)
                    mins = sum(self.players[k].minutes for k in keys)
                    sdev = sum(self.players[k].stddev for k in keys)

                    row = []
                    for _, key in ordered:
                        p = self.players[key]
                        row.append(f"{p.name} ({p.pid})")
                    row += [salary, fpts_sum, own_prod, own_sum, mins, sdev]
                    writer.writerow(row)

            else:
                writer.writerow(
                    [
                        "PG",
                        "PG",
                        "SG",
                        "SG",
                        "SF",
                        "SF",
                        "PF",
                        "PF",
                        "C",
                        "Salary",
                        "Fpts Proj",
                        "Own. Prod.",
                        "Own. Sum.",
                        "Minutes",
                        "StdDev",
                    ]
                )
                for lineup in self.selected_lineups:
                    ordered = self._sort_lineup_for_export(list(lineup))
                    keys = [k for _, k in ordered]
                    salary = sum(self.players[k].salary for k in keys)
                    fpts_sum = round(sum(self.players[k].fpts for k in keys), 2)
                    own_prod = float(
                        np.prod(
                            [
                                self._own_to_fraction(self.players[k].ownership)
                                for k in keys
                            ]
                        )
                    )
                    own_sum = sum(self.players[k].ownership for k in keys)
                    mins = sum(self.players[k].minutes for k in keys)
                    sdev = sum(self.players[k].stddev for k in keys)

                    row = []
                    for _, key in ordered:
                        p = self.players[key]
                        row.append(f"{p.pid.replace('#', '-')}:{p.name}")
                    row += [salary, fpts_sum, own_prod, own_sum, mins, sdev]
                    writer.writerow(row)

        print(f"Output written: {out_path}")

    def _sort_lineup_for_export(
        self, lineup: List[Tuple[Tuple[str, str, str], str, str]]
    ) -> List[Tuple[str, Tuple[str, str, str]]]:
        # returns ordered list of player-keys matching export order
        if self.site == "dk":
            order = DK_POSITIONS
            out = [None] * 8
            for key, pos, pid in lineup:
                idx = order.index(pos)
                if out[idx] is None:
                    out[idx] = (pos, key)
                else:
                    # UTIL/G/F duplicates can collide; place next slot
                    # Find next same-pos slot if any (DK has single each, except UTIL/G/F)
                    j = idx
                    while j < len(out) and out[j] is not None:
                        j += 1
                    if j < len(out):
                        out[j] = (pos, key)
            # compact None if any (shouldn’t happen for valid solution)
            return [x for x in out if x is not None]
        else:
            order = FD_POSITIONS
            counts = {"PG": 0, "SG": 0, "SF": 0, "PF": 0, "C": 0}
            out = [None] * 9
            for pos_target in order:
                # pick a player with that pos that’s not yet placed
                for idx, (key, pos, pid) in enumerate(lineup):
                    if pos == pos_target and lineup[idx] is not None:
                        out_index = order.index(pos_target, counts[pos_target])
                        out[out_index] = (pos, key)
                        counts[pos_target] += 1
                        lineup[idx] = None  # mark used
                        break
            return [x for x in out if x is not None]

    def _late_swap_adjust(
        self, ordered: List[Tuple[str, Tuple[str, str, str]]]
    ) -> List[Tuple[str, Tuple[str, str, str]]]:
        if self.site == "fd":
            return ordered

        pos_map = {
            0: ["PG"],
            1: ["SG"],
            2: ["SF"],
            3: ["PF"],
            4: ["C"],
            5: ["PG", "SG"],
            6: ["SF", "PF"],
            7: ["PG", "SG", "SF", "PF", "C"],
        }

        sorted_lineup = list(ordered)

        def swap_if_needed(i: int, j: int):
            pos_i, key_i = sorted_lineup[i]
            pos_j, key_j = sorted_lineup[j]
            pi = self.players[key_i]
            pj = self.players[key_j]
            if pi.gametime and pj.gametime and pi.gametime > pj.gametime:
                # ensure eligibility both ways
                if any(p in pos_map[i] for p in pj.positions) and any(
                    p in pos_map[j] for p in pi.positions
                ):
                    sorted_lineup[i], sorted_lineup[j] = (
                        sorted_lineup[j],
                        sorted_lineup[i],
                    )

        for i in range(5):
            for j in range(5, 8):
                swap_if_needed(i, j)

        return sorted_lineup


# -------------------------
# CLI
# -------------------------


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="NBA DFS Optimizer (drop-in)")
    ap.add_argument("--site", required=True, choices=["dk", "fd"])
    ap.add_argument("--num-lineups", type=int, default=0)
    ap.add_argument("--num-uniques", type=int, default=1)
    ap.add_argument(
        "--config", default=os.path.join(os.path.dirname(__file__), "config.json")
    )
    ap.add_argument("--projection-path", default=None)
    ap.add_argument("--player-ids-path", default=None)
    ap.add_argument("--randomness", type=float, default=None)
    ap.add_argument("--min-lineup-salary", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument(
        "--lambda-own",
        type=float,
        default=0.0,
        help="Ownership penalty λ_own (applied to ownership in [0,1])",
    )
    ap.add_argument(
        "--lambda-var",
        type=float,
        default=0.0,
        help="Variance penalty λ_var (applied to stddev)",
    )
    ap.add_argument(
        "--time-limit-sec",
        type=int,
        default=None,
        help="Solver time limit per lineup (seconds)",
    )
    ap.add_argument("--threads", type=int, default=None, help="Solver threads/workers")
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable randomness/jitter and force deterministic solving",
    )
    ap.add_argument(
        "--backend",
        choices=["pulp", "cpsat", "scip"],
        default="cpsat",
        help="Optimization backend to use (pulp|cpsat|scip)",
    )
    # Ownership penalty wiring (new)
    ap.add_argument(
        "--own-curve",
        choices=["piecewise", "power", "logistic"],
        default="piecewise",
        help="Ownership curve kind (piecewise|power|logistic)",
    )
    ap.add_argument(
        "--own-lambda1",
        type=float,
        default=0.0,
        help="Base lambda1 for ownership penalty (0 disables)",
    )
    ap.add_argument(
        "--own-pct-scale",
        type=float,
        default=1.0,
        help="Scale applied to ownership values (e.g., 100 if inputs are fractions)",
    )
    # Per-curve extras
    ap.add_argument(
        "--own-breaks",
        type=str,
        default="10,20,30",
        help="Piecewise breaks in percent (comma-separated)",
    )
    ap.add_argument(
        "--own-weights",
        type=str,
        default="0.5,1.0,1.5,2.0",
        help="Piecewise weights (len = len(breaks)+1, comma-separated)",
    )
    ap.add_argument(
        "--own-power-gamma",
        type=float,
        default=1.6,
        help="Power curve gamma (>1 convex, <1 concave)",
    )
    ap.add_argument(
        "--own-logi-mid",
        type=float,
        default=22.0,
        help="Logistic knee (midpoint) in percent",
    )
    # Aggro schedule controls
    ap.add_argument(
        "--aggro-enable",
        action="store_true",
        help="Enable aggressiveness ramp for lambda1 across lineup iterations",
    )
    ap.add_argument(
        "--aggro-start-lambda",
        type=float,
        default=0.30,
        help="Aggro: starting lambda1",
    )
    ap.add_argument(
        "--aggro-max-lambda",
        type=float,
        default=0.60,
        help="Aggro: max lambda1 at end of ramp",
    )
    ap.add_argument(
        "--aggro-ramp",
        nargs=2,
        type=int,
        default=(20, 80),
        metavar=("START","END"),
        help="Aggro: ramp start and end lineup indices",
    )
    qb = ap.add_argument_group("QuickBuild pool builder")
    qb.add_argument(
        "--quick-build",
        action="store_true",
        help="Enable multi-process QuickBuild pool generation (experimental).",
    )
    qb.add_argument("--qb-builds", type=int, default=0, help="Number of QuickBuild worker processes.")
    qb.add_argument("--qb-per-build", type=int, default=1200, help="Target solutions emitted per worker.")
    qb.add_argument("--qb-threads", type=int, default=1, help="OR-Tools threads per worker.")
    qb.add_argument("--qb-timeout", type=float, default=75.0, help="Timeout (seconds) per worker solve run.")
    qb.add_argument("--qb-min-uniq", type=int, default=5, help="Minimum changes vs last emitted lineup within a worker.")
    qb.add_argument("--qb-nogood-rate", type=int, default=20, help="Add an exact no-good cut every Nth emission (0 disables).")
    qb.add_argument("--qb-jitter", type=float, default=5e-4, help="Objective jitter magnitude applied per worker.")
    qb.add_argument("--qb-seed", type=int, default=None, help="Global seed; defaults to wall time when omitted.")
    qb.add_argument("--qb-max-pool", type=int, default=20000, help="Legacy in-memory pool cap (before spill).")
    qb.add_argument("--qb-max-pool-ram", type=int, default=250000, help="Number of lineups to keep in RAM before spilling.")
    qb.add_argument("--qb-max-pool-disk", type=int, default=2000000, help="Hard cap on total pool size including spill files.")
    qb.add_argument("--qb-chunk-size", type=int, default=25000, help="Rows per spill chunk when writing to disk.")
    qb.add_argument("--qb-store", choices=["parquet", "memmap"], default="parquet", help="Spill backend for large pools.")
    qb.add_argument("--qb-spill-dir", default=None, help="Directory for spill chunks (defaults to runs/{run_id}/pool).")
    qb.add_argument("--qb-out", default=None, help="Final pool CSV path (defaults to runs/{run_id}/pool.csv).")
    qb.add_argument("--qb-stats", default=None, help="Optional qb_stats.json output path.")
    qb.add_argument("--qb-run-id", default=None, help="Override run identifier (defaults to timestamp).")
    qb.add_argument("--qb-quality-quantile", type=float, default=0.50, help="Per-archetype quantile floor for accepting a lineup.")
    qb.add_argument("--qb-archetype-key", default="stack,salbin", help="Comma-separated features used for archetype caps.")
    qb.add_argument("--qb-merge-topk-per-arch", type=int, default=2000, help="Cap kept per archetype before global merge.")
    qb.add_argument("--qb-bloom-fpp", type=float, default=0.01, help="Target false-positive rate for Bloom filter prefilter.")
    qb.add_argument("--qb-near-dup-jaccard", type=float, default=0.75, help="Drop lineup if approximate Jaccard >= threshold.")
    qb.add_argument("--qb-queue-size", type=int, default=10000, help="Max size of inter-process queue for lineup streaming.")
    qb.set_defaults(qb_approx_uniq=True)
    qb.add_argument(
        "--qb-approx-uniq",
        dest="qb_approx_uniq",
        action="store_true",
        help="Enable Bloom/LSH prefilter for approximate uniqueness (default).",
    )
    qb.add_argument(
        "--qb-no-approx-uniq",
        dest="qb_approx_uniq",
        action="store_false",
        help="Disable Bloom/LSH prefilter (exact hash only).",
    )
    return ap.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    qb_cfg = None
    if args.quick_build:
        from .quick_build import QuickBuildConfig
        site = (args.site or "dk").lower()
        lineup_size = 8 if site == "dk" else 9
        qb_cfg = QuickBuildConfig.from_namespace(args, lineup_size=lineup_size)
    # Lazy import to avoid making objective a hard dependency during import
    try:
        from .objective.own_penalty import (
            OwnershipPenaltyConfig,
            AggroSchedule,
            CurveKind,
        )
        from .objective import set_active_ownership_penalty
    except Exception:
        OwnershipPenaltyConfig = None  # type: ignore
        AggroSchedule = None  # type: ignore
        CurveKind = None  # type: ignore
        set_active_ownership_penalty = None  # type: ignore

    # Build ownership penalty config from CLI (if objective module is available)
    own_cfg = None
    aggro_sched = None
    if 'OwnershipPenaltyConfig' in globals() and OwnershipPenaltyConfig is not None:
        # Only publish ownership penalty if explicitly enabled (lambda1 > 0 or aggro enabled)
        should_publish = (float(getattr(args, 'own_lambda1', 0.0) or 0.0) > 0.0) or bool(getattr(args, 'aggro_enable', False))
        if should_publish:
            curve_map = {
                "piecewise": CurveKind.PIECEWISE,  # type: ignore
                "power": CurveKind.POWER,          # type: ignore
                "logistic": CurveKind.LOGISTIC,    # type: ignore
            }
            curve_kind = curve_map.get(args.own_curve, curve_map["piecewise"])  # type: ignore
            # parse piecewise params
            def _parse_csv_floats(s: str) -> tuple:
                parts = [p.strip() for p in (s or "").split(",") if p.strip()]
                return tuple(float(x) for x in parts)

            breaks = _parse_csv_floats(args.own_breaks)
            weights = _parse_csv_floats(args.own_weights)

            own_cfg = OwnershipPenaltyConfig(  # type: ignore
                lambda1=float(args.own_lambda1 or 0.0),
                pct_scale=float(args.own_pct_scale),
                curve=curve_kind,
                breaks=breaks if breaks else (10.0, 20.0, 30.0),
                weights=weights if weights else (0.5, 1.0, 1.5, 2.0),
                power_gamma=float(args.own_power_gamma),
                logi_mid=float(args.own_logi_mid),
            )
            if args.aggro_enable and AggroSchedule is not None:
                rs, re = args.aggro_ramp
                aggro_sched = AggroSchedule(  # type: ignore
                    start_lambda=float(args.aggro_start_lambda),
                    max_lambda=float(args.aggro_max_lambda),
                    ramp_start=int(rs),
                    ramp_end=int(re),
                )
            # Publish globally so CP-SAT objective can pick it up; remain silent (no logging)
            if set_active_ownership_penalty is not None:
                set_active_ownership_penalty(own_cfg, aggro_sched)  # type: ignore
    opt = NBAOptimizer(
        site=args.site,
        num_lineups=args.num_lineups,
        num_uniques=args.num_uniques,
        config_path=args.config,
        projection_path=args.projection_path,
        player_ids_path=args.player_ids_path,
        randomness=args.randomness,
        min_lineup_salary=args.min_lineup_salary,
        seed=args.seed,
        output_dir=args.output_dir,
        lambda_own=args.lambda_own,
        lambda_var=args.lambda_var,
        time_limit_sec=args.time_limit_sec,
        threads=args.threads,
        deterministic=args.deterministic,
        backend=args.backend,
    )
    if qb_cfg is not None:
        opt.run_quick_build(qb_cfg)
        return 0
    opt.solve()
    opt.write_output()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
