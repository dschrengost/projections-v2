"""Toy sanity checks for contest-sim select_score ranking.

This is a lightweight script to validate select_score monotonic behavior with
respect to tail metrics and dupe_penalty.

Usage:
  uv run python scripts/diagnostics/select_score_toy.py
"""

from __future__ import annotations

from dataclasses import dataclass


TAIL_WEIGHT_P90 = 0.6
TAIL_WEIGHT_UCV = 0.4
RANK_MODES = ("tail_only", "tail_times_dupe", "current")


@dataclass(frozen=True)
class ToyLineup:
    name: str
    mean: float
    p90: float
    ucv90: float
    dupe_penalty: float

    @property
    def tail_score(self) -> float:
        return TAIL_WEIGHT_P90 * self.p90 + TAIL_WEIGHT_UCV * self.ucv90

    def select_score(self, *, rank_mode: str) -> float:
        mode = str(rank_mode).strip().lower()
        if mode == "tail_only":
            return float(self.tail_score)
        if mode == "tail_times_dupe":
            return float(self.tail_score) * float(self.dupe_penalty)
        if mode == "current":
            penalty_impact = (1.0 - float(self.dupe_penalty)) * float(self.mean)
            return float(self.tail_score) - penalty_impact
        raise ValueError(f"Unknown rank_mode: {rank_mode!r}")


def _print_rank(lineups: list[ToyLineup], *, title: str, rank_mode: str) -> None:
    print("\n" + title + f" (rank_mode={rank_mode})")
    rows = sorted(lineups, key=lambda x: x.select_score(rank_mode=rank_mode), reverse=True)
    print("rank | lineup | tail_score | dupe_penalty | mean | select_score")
    for i, r in enumerate(rows, 1):
        print(
            f"{i:4d} | {r.name:6s} | {r.tail_score:9.2f} | {r.dupe_penalty:11.2f} | "
            f"{r.mean:4.0f} | {r.select_score(rank_mode=rank_mode):11.2f}"
        )


def main() -> None:
    base = [
        ToyLineup(name="A", mean=320, p90=390, ucv90=410, dupe_penalty=1.00),  # strong, unique
        ToyLineup(name="B", mean=320, p90=390, ucv90=410, dupe_penalty=0.85),  # same tail, duped
        ToyLineup(name="C", mean=320, p90=380, ucv90=400, dupe_penalty=1.00),  # worse tail, unique
        ToyLineup(name="D", mean=280, p90=385, ucv90=405, dupe_penalty=0.80),  # lower mean, more duped
        ToyLineup(name="E", mean=360, p90=395, ucv90=415, dupe_penalty=0.80),  # best tail, chalky/duped
    ]

    for mode in RANK_MODES:
        _print_rank(base, title="Scenario 1: Baseline", rank_mode=mode)

    # Scenario 2: degrade dupe_penalty for everyone to show monotonic effect.
    degrade = [
        ToyLineup(**{**lineup.__dict__, "dupe_penalty": max(0.0, lineup.dupe_penalty - 0.1)})
        for lineup in base
    ]
    for mode in RANK_MODES:
        _print_rank(degrade, title="Scenario 2: dupe_penalty - 0.10 (clipped)", rank_mode=mode)

    # Scenario 3: remove dupe penalty (all unique); ranking should be by tail_score only.
    all_unique = [ToyLineup(**{**lineup.__dict__, "dupe_penalty": 1.0}) for lineup in base]
    for mode in RANK_MODES:
        _print_rank(all_unique, title="Scenario 3: All unique (dupe_penalty=1.0)", rank_mode=mode)

    # Scenario 4: Dominance violation example for rank_mode=current.
    # Lineup A dominates B on (mean, p90, ucv90) with equal dupe_penalty, but can lose
    # because the penalty is scaled by mean.
    dom = [
        ToyLineup(name="A_dom", mean=500, p90=410, ucv90=420, dupe_penalty=0.90),
        ToyLineup(name="B_dom", mean=400, p90=405, ucv90=415, dupe_penalty=0.90),
    ]
    _print_rank(dom, title="Scenario 4: Dominance example", rank_mode="tail_only")
    _print_rank(dom, title="Scenario 4: Dominance example", rank_mode="tail_times_dupe")
    _print_rank(dom, title="Scenario 4: Dominance example (can violate)", rank_mode="current")


if __name__ == "__main__":
    main()
