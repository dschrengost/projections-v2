import pytest


TAIL_WEIGHT_P90 = 0.6
TAIL_WEIGHT_UCV = 0.4


def _tail_score(*, p90: float, ucv90: float) -> float:
    tail_score = TAIL_WEIGHT_P90 * p90 + TAIL_WEIGHT_UCV * ucv90
    return tail_score


def _select_score(*, mean: float, p90: float, ucv90: float, dupe_penalty: float, rank_mode: str) -> float:
    tail_score = _tail_score(p90=p90, ucv90=ucv90)
    mode = str(rank_mode).strip().lower()
    if mode == "tail_only":
        return tail_score
    if mode == "tail_times_dupe":
        return tail_score * dupe_penalty
    if mode == "current":
        penalty_impact = (1.0 - dupe_penalty) * mean
        return tail_score - penalty_impact
    raise ValueError(f"invalid rank_mode: {rank_mode!r}")


def test_select_score_monotonic_in_tail_score() -> None:
    mean = 300.0
    dupe_penalty = 0.85
    for rank_mode in ("tail_only", "tail_times_dupe", "current"):
        worse = _select_score(mean=mean, p90=380.0, ucv90=395.0, dupe_penalty=dupe_penalty, rank_mode=rank_mode)
        better = _select_score(mean=mean, p90=390.0, ucv90=410.0, dupe_penalty=dupe_penalty, rank_mode=rank_mode)
        assert better > worse


def test_select_score_monotonic_in_dupe_penalty() -> None:
    mean = 320.0
    p90 = 390.0
    ucv90 = 410.0
    for rank_mode in ("tail_times_dupe", "current"):
        more_duped = _select_score(mean=mean, p90=p90, ucv90=ucv90, dupe_penalty=0.75, rank_mode=rank_mode)
        less_duped = _select_score(mean=mean, p90=p90, ucv90=ucv90, dupe_penalty=0.9, rank_mode=rank_mode)
        assert less_duped > more_duped


def test_select_score_no_penalty_when_unique() -> None:
    mean = 320.0
    p90 = 390.0
    ucv90 = 410.0
    expected_tail = _tail_score(p90=p90, ucv90=ucv90)
    for rank_mode in ("tail_only", "tail_times_dupe", "current"):
        score = _select_score(mean=mean, p90=p90, ucv90=ucv90, dupe_penalty=1.0, rank_mode=rank_mode)
        assert score == pytest.approx(expected_tail)


def test_tail_only_and_tail_times_dupe_respect_dominance() -> None:
    """If A dominates B on (mean, p90, ucv90) and dupe(A) >= dupe(B), score(A) >= score(B)."""
    a = {"mean": 500.0, "p90": 410.0, "ucv90": 420.0, "dupe_penalty": 0.90}
    b = {"mean": 400.0, "p90": 405.0, "ucv90": 415.0, "dupe_penalty": 0.85}
    for mode in ("tail_only", "tail_times_dupe"):
        sa = _select_score(**a, rank_mode=mode)
        sb = _select_score(**b, rank_mode=mode)
        assert sa >= sb


def test_current_rank_mode_can_violate_dominance_example() -> None:
    """Flag that the current formula can reverse dominated lineups (mean-scaled penalty)."""
    # A dominates B on tail metrics + mean, with equal dupe_penalty.
    a = {"mean": 500.0, "p90": 410.0, "ucv90": 420.0, "dupe_penalty": 0.90}
    b = {"mean": 400.0, "p90": 405.0, "ucv90": 415.0, "dupe_penalty": 0.90}
    sa = _select_score(**a, rank_mode="current")
    sb = _select_score(**b, rank_mode="current")
    assert sa < sb
