"""Candidate scoring helpers for late swap modes."""

from __future__ import annotations

from projections.late_swap.models import LateSwapCandidate, LateSwapPolicy


def _resolve_primary_score(candidate: LateSwapCandidate) -> float:
    if candidate.expected_value is not None:
        return float(candidate.expected_value)
    if candidate.remaining_score_mean is not None:
        return float(candidate.remaining_score_mean)
    if candidate.projected_score is not None:
        return float(candidate.projected_score)
    return 0.0


def _resolve_mode_adjustments(policy: LateSwapPolicy) -> tuple[float, float, float]:
    """Return swap-cost, ownership penalty, leverage boost coefficients."""
    swap_cost = float(policy.swap_cost_lambda)
    own_penalty = float(policy.ownership_penalty_lambda)
    leverage_boost = float(policy.leverage_boost_lambda)
    if policy.mode == "best_ev":
        return max(0.01, swap_cost), own_penalty, leverage_boost
    if policy.mode == "decorrelated_ev":
        return max(0.01, swap_cost), own_penalty + 0.04, leverage_boost
    if policy.mode == "catch_up":
        return max(0.0, swap_cost), max(own_penalty, 0.1), max(leverage_boost, 0.08)
    if policy.mode == "block":
        return max(swap_cost, 0.2), own_penalty, leverage_boost
    return swap_cost, own_penalty, leverage_boost


def apply_candidate_scores(
    *,
    candidates_by_entry_id: dict[str, list[LateSwapCandidate]],
    policy: LateSwapPolicy,
) -> dict[str, list[LateSwapCandidate]]:
    swap_cost_lambda, own_penalty_lambda, leverage_boost_lambda = _resolve_mode_adjustments(policy)
    for _entry_id, candidates in candidates_by_entry_id.items():
        hold = next((c for c in candidates if c.generated_by == "hold"), None)
        hold_score = _resolve_primary_score(hold) if hold else None
        for candidate in candidates:
            score = _resolve_primary_score(candidate)
            score -= swap_cost_lambda * float(candidate.swap_count)
            if candidate.total_own is not None:
                score -= own_penalty_lambda * float(candidate.total_own) / 100.0
                score += leverage_boost_lambda * max(0.0, 100.0 - float(candidate.total_own)) / 100.0
            if (
                hold_score is not None
                and candidate.generated_by != "hold"
                and candidate.projected_score is not None
            ):
                gain = float(candidate.projected_score) - float(hold_score)
                if gain < float(policy.min_gain_to_swap):
                    score -= 100.0
            candidate.objective_score = float(score)
    return candidates_by_entry_id
