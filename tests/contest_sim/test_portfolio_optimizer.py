from __future__ import annotations

import pytest

from projections.contest_sim.portfolio_optimizer import (
    DecorrelatedPortfolioConfig,
    ExposureBoundsPct,
    PortfolioCandidate,
    build_decorrelated_portfolio,
    build_portfolio,
)


def test_build_portfolio_selects_top_ev_without_constraints() -> None:
    candidates = [
        PortfolioCandidate(lineup_id=1, player_ids=("A",), expected_value=2.0),
        PortfolioCandidate(lineup_id=2, player_ids=("B",), expected_value=1.0),
        PortfolioCandidate(lineup_id=3, player_ids=("C",), expected_value=0.5),
    ]
    selection = build_portfolio(candidates, portfolio_size=2)
    assert [c.lineup_id for c in selection.selected] == [1, 2]


def test_build_portfolio_respects_min_uniques() -> None:
    candidates = [
        PortfolioCandidate(lineup_id=1, player_ids=("A", "B", "C"), expected_value=10.0),
        PortfolioCandidate(lineup_id=2, player_ids=("A", "B", "D"), expected_value=9.0),
        PortfolioCandidate(lineup_id=3, player_ids=("E", "F", "G"), expected_value=8.0),
    ]
    selection = build_portfolio(candidates, portfolio_size=2, min_uniques=2)
    assert [c.lineup_id for c in selection.selected] == [1, 3]


def test_build_portfolio_respects_max_exposure_caps() -> None:
    candidates = [
        PortfolioCandidate(lineup_id=1, player_ids=("A", "B"), expected_value=10.0),
        PortfolioCandidate(lineup_id=2, player_ids=("A", "C"), expected_value=9.0),
        PortfolioCandidate(lineup_id=3, player_ids=("D", "E"), expected_value=8.0),
    ]
    selection = build_portfolio(
        candidates,
        portfolio_size=2,
        exposure_bounds={"A": ExposureBoundsPct(max=50.0)},
    )
    assert [c.lineup_id for c in selection.selected] == [1, 3]


def test_build_portfolio_sorts_missing_metrics_to_bottom_for_ascending() -> None:
    candidates = [
        PortfolioCandidate(lineup_id=1, player_ids=("A",), total_own=55.0),
        PortfolioCandidate(lineup_id=2, player_ids=("B",), total_own=None),
        PortfolioCandidate(lineup_id=3, player_ids=("C",), total_own=25.0),
    ]

    selection = build_portfolio(
        candidates,
        portfolio_size=2,
        sort_key="total_own",
        sort_dir="asc",
    )

    assert [c.lineup_id for c in selection.selected] == [3, 1]


def test_build_portfolio_prioritizes_seed_lineups_when_feasible() -> None:
    candidates = [
        PortfolioCandidate(lineup_id=1, player_ids=("A",), expected_value=5.0),
        PortfolioCandidate(lineup_id=2, player_ids=("B",), expected_value=4.0),
        PortfolioCandidate(lineup_id=3, player_ids=("C",), expected_value=3.0),
        PortfolioCandidate(lineup_id=4, player_ids=("D",), expected_value=2.0),
    ]

    selection = build_portfolio(
        candidates,
        portfolio_size=2,
        seed_lineup_ids=[4, 3],
    )

    assert [c.lineup_id for c in selection.selected] == [3, 4]


def test_build_portfolio_repairs_seed_lineups_with_next_feasible_candidates() -> None:
    candidates = [
        PortfolioCandidate(lineup_id=1, player_ids=("A", "B"), expected_value=10.0),
        PortfolioCandidate(lineup_id=2, player_ids=("A", "C"), expected_value=9.0),
        PortfolioCandidate(lineup_id=3, player_ids=("D", "E"), expected_value=8.0),
    ]

    selection = build_portfolio(
        candidates,
        portfolio_size=2,
        exposure_bounds={"A": ExposureBoundsPct(max=50.0)},
        seed_lineup_ids=[1, 2],
    )

    assert [c.lineup_id for c in selection.selected] == [1, 3]


def test_build_portfolio_rejects_min_exposure_bounds() -> None:
    candidates = [
        PortfolioCandidate(lineup_id=1, player_ids=("A", "B"), expected_value=10.0),
        PortfolioCandidate(lineup_id=2, player_ids=("C", "D"), expected_value=9.0),
    ]

    with pytest.raises(ValueError, match="Minimum exposure is not supported yet"):
        build_portfolio(
            candidates,
            portfolio_size=1,
            exposure_bounds={"A": ExposureBoundsPct(min=25.0)},
        )


def test_build_portfolio_raises_when_constraints_exhaust_pool() -> None:
    candidates = [
        PortfolioCandidate(lineup_id=1, player_ids=("A",), expected_value=10.0),
        PortfolioCandidate(lineup_id=2, player_ids=("A",), expected_value=9.0),
    ]
    with pytest.raises(ValueError, match="pool exhausted"):
        build_portfolio(
            candidates,
            portfolio_size=2,
            exposure_bounds={"A": ExposureBoundsPct(max=0.0)},
        )


def test_build_decorrelated_portfolio_avoids_correlated_lineups() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    worlds = np.vstack([x, x + 0.01 * rng.normal(size=200), y]).T.astype(np.float64)  # (W, P)
    player_index = {"A": 0, "B": 1, "C": 2}

    candidates = [
        PortfolioCandidate(lineup_id=1, player_ids=("A",), expected_value=1.0),
        PortfolioCandidate(lineup_id=2, player_ids=("B",), expected_value=1.0),
        PortfolioCandidate(lineup_id=3, player_ids=("C",), expected_value=0.99),
    ]

    selection, diag = build_decorrelated_portfolio(
        candidates,
        portfolio_size=2,
        worlds_matrix=worlds,
        player_index=player_index,
        config=DecorrelatedPortfolioConfig(ev_retention=0.99, worlds_sample=200, seed=1),
    )

    picked = {c.lineup_id for c in selection.selected}
    assert 3 in picked  # include the independent option
    assert picked != {1, 2}  # avoid taking both correlated A/B
    assert diag.ev_selected >= diag.ev_target
    assert diag.risk_var_total_selected <= diag.risk_var_total_baseline
    assert diag.swaps_made >= 1


def test_build_decorrelated_portfolio_respects_exposure_caps() -> None:
    import numpy as np

    worlds = np.asarray(
        [
            [10.0, 5.0],
            [20.0, 5.0],
            [30.0, 5.0],
        ],
        dtype=np.float64,
    )
    player_index = {"A": 0, "C": 1}

    candidates = [
        PortfolioCandidate(lineup_id=1, player_ids=("A",), expected_value=1.0),
        PortfolioCandidate(lineup_id=2, player_ids=("A",), expected_value=0.9),
        PortfolioCandidate(lineup_id=3, player_ids=("C",), expected_value=0.8),
    ]

    selection, _diag = build_decorrelated_portfolio(
        candidates,
        portfolio_size=2,
        worlds_matrix=worlds,
        player_index=player_index,
        config=DecorrelatedPortfolioConfig(ev_retention=1.0, worlds_sample=3, seed=1),
        exposure_bounds={"A": ExposureBoundsPct(max=50.0)},
    )

    picked = {c.lineup_id for c in selection.selected}
    assert picked != {1, 2}  # would violate A max 50% in a 2-lineup portfolio


def test_build_decorrelated_portfolio_uses_seed_lineups_when_ev_retention_allows() -> None:
    import numpy as np

    worlds = np.zeros((50, 3), dtype=np.float64)
    player_index = {"A": 0, "B": 1, "C": 2}

    candidates = [
        PortfolioCandidate(lineup_id=1, player_ids=("A",), expected_value=3.0),
        PortfolioCandidate(lineup_id=2, player_ids=("B",), expected_value=2.9),
        PortfolioCandidate(lineup_id=3, player_ids=("C",), expected_value=2.8),
    ]

    selection, diag = build_decorrelated_portfolio(
        candidates,
        portfolio_size=2,
        worlds_matrix=worlds,
        player_index=player_index,
        config=DecorrelatedPortfolioConfig(ev_retention=0.9, worlds_sample=50, seed=7),
        seed_lineup_ids=[3, 2],
    )

    assert {c.lineup_id for c in selection.selected} == {2, 3}
    assert diag.ev_best == pytest.approx(5.9)
    assert diag.ev_selected == pytest.approx(5.7)


def test_build_decorrelated_portfolio_reports_exact_risk() -> None:
    import numpy as np

    rng = np.random.default_rng(123)
    W = 500
    P = 60
    K = 200
    N = 20

    worlds = rng.normal(size=(W, P)).astype(np.float64)
    player_ids = [f"P{i}" for i in range(P)]
    player_index = {pid: i for i, pid in enumerate(player_ids)}

    # Random lineup pool with EV noise (all finite).
    candidates = []
    for lineup_id in range(1, K + 1):
        ids = tuple(rng.choice(player_ids, size=8, replace=False))
        ev = float(rng.normal(loc=1.0, scale=0.2))
        candidates.append(PortfolioCandidate(lineup_id=lineup_id, player_ids=ids, expected_value=ev))

    selection, diag = build_decorrelated_portfolio(
        candidates,
        portfolio_size=N,
        worlds_matrix=worlds,
        player_index=player_index,
        config=DecorrelatedPortfolioConfig(ev_retention=0.95, worlds_sample=W, seed=1, max_passes=3, max_swaps=200),
    )

    # Recompute Σ and risk exactly using the same player inclusion logic.
    included: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        for pid in cand.player_ids:
            pid_s = str(pid).strip()
            if not pid_s or pid_s in seen:
                continue
            if pid_s not in player_index:
                continue
            seen.add(pid_s)
            included.append(pid_s)

    cols = np.asarray([player_index[pid] for pid in included], dtype=np.int64)
    worlds_sub = np.take(worlds.astype(np.float32), cols, axis=1)
    mu = worlds_sub.mean(axis=0, dtype=np.float32)
    X = worlds_sub - mu
    sigma = (X.T @ X) / float(W - 1)
    sigma = np.asarray(sigma, dtype=np.float64)

    pid_to_local = {pid: i for i, pid in enumerate(included)}
    counts = np.zeros((len(included),), dtype=np.float64)
    for cand in selection.selected:
        for pid in cand.player_ids:
            loc = pid_to_local.get(str(pid))
            if loc is not None:
                counts[loc] += 1.0

    risk_exact = float(counts @ (sigma @ counts))
    assert np.isfinite(risk_exact)
    assert abs(risk_exact - diag.risk_var_total_selected) <= 1e-6 * max(1.0, abs(risk_exact))

    assert diag.ev_selected + 1e-9 >= diag.ev_target


def test_build_decorrelated_portfolio_is_deterministic_for_train_split() -> None:
    import numpy as np

    rng = np.random.default_rng(321)
    worlds = rng.normal(size=(200, 12)).astype(np.float64)
    player_ids = [f"P{i}" for i in range(12)]
    player_index = {pid: i for i, pid in enumerate(player_ids)}
    candidates = [
        PortfolioCandidate(
            lineup_id=lineup_id,
            player_ids=tuple(player_ids[lineup_id - 1 : lineup_id + 3]),
            expected_value=2.0 - (lineup_id * 0.05),
        )
        for lineup_id in range(1, 7)
    ]

    cfg = DecorrelatedPortfolioConfig(
        ev_retention=0.97,
        worlds_sample=150,
        worlds_train_frac=0.6,
        seed=17,
        max_passes=2,
        max_swaps=20,
    )

    selection_a, diag_a = build_decorrelated_portfolio(
        candidates,
        portfolio_size=3,
        worlds_matrix=worlds,
        player_index=player_index,
        config=cfg,
    )
    selection_b, diag_b = build_decorrelated_portfolio(
        candidates,
        portfolio_size=3,
        worlds_matrix=worlds,
        player_index=player_index,
        config=cfg,
    )

    assert [c.lineup_id for c in selection_a.selected] == [c.lineup_id for c in selection_b.selected]
    assert diag_a.to_dict() == diag_b.to_dict()
