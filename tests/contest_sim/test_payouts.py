import numpy as np

from projections.contest_sim.payouts import compute_expected_user_payouts_vectorized
from projections.contest_sim.scoring_models import PayoutTier


def _make_inputs() -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[PayoutTier]]:
    rng = np.random.default_rng(123)
    user_scores = rng.normal(loc=100.0, scale=15.0, size=(3, 17)).astype(np.float64)
    field_scores = rng.normal(loc=100.0, scale=15.0, size=(8, 17)).astype(np.float64)
    user_weights = [1, 2, 1]
    field_weights = [3, 1, 1, 2, 1, 1, 1, 1]
    payout_tiers = [
        PayoutTier(start_place=1, end_place=1, payout=10.0),
        PayoutTier(start_place=2, end_place=5, payout=5.0),
        PayoutTier(start_place=6, end_place=15, payout=1.0),
    ]
    return user_scores, field_scores, user_weights, field_weights, payout_tiers


def test_payouts_chunking_matches_single_chunk() -> None:
    user_scores, field_scores, user_weights, field_weights, payout_tiers = _make_inputs()

    res_small_chunks = compute_expected_user_payouts_vectorized(
        user_scores=user_scores,
        field_scores=field_scores,
        user_weights=user_weights,
        field_weights=field_weights,
        payout_tiers=payout_tiers,
        workers=1,
        compute_field_side=True,
        world_chunk_size=3,
    )
    res_large_chunks = compute_expected_user_payouts_vectorized(
        user_scores=user_scores,
        field_scores=field_scores,
        user_weights=user_weights,
        field_weights=field_weights,
        payout_tiers=payout_tiers,
        workers=1,
        compute_field_side=True,
        world_chunk_size=19,
    )

    np.testing.assert_allclose(res_small_chunks.expected_payouts, res_large_chunks.expected_payouts)
    np.testing.assert_allclose(res_small_chunks.win_rates, res_large_chunks.win_rates)
    np.testing.assert_allclose(res_small_chunks.cash_rates, res_large_chunks.cash_rates)
    np.testing.assert_allclose(res_small_chunks.top_1pct_rates, res_large_chunks.top_1pct_rates)
    np.testing.assert_allclose(res_small_chunks.top_5pct_rates, res_large_chunks.top_5pct_rates)
    np.testing.assert_allclose(res_small_chunks.top_10pct_rates, res_large_chunks.top_10pct_rates)
    np.testing.assert_allclose(res_small_chunks.field_expected_payouts, res_large_chunks.field_expected_payouts)
    np.testing.assert_allclose(res_small_chunks.field_win_rates, res_large_chunks.field_win_rates)
    np.testing.assert_allclose(res_small_chunks.field_cash_rates, res_large_chunks.field_cash_rates)


def test_payouts_compute_field_side_flag() -> None:
    user_scores, field_scores, user_weights, field_weights, payout_tiers = _make_inputs()

    res_with_field = compute_expected_user_payouts_vectorized(
        user_scores=user_scores,
        field_scores=field_scores,
        user_weights=user_weights,
        field_weights=field_weights,
        payout_tiers=payout_tiers,
        workers=1,
        compute_field_side=True,
        world_chunk_size=5,
    )
    res_without_field = compute_expected_user_payouts_vectorized(
        user_scores=user_scores,
        field_scores=field_scores,
        user_weights=user_weights,
        field_weights=field_weights,
        payout_tiers=payout_tiers,
        workers=1,
        compute_field_side=False,
        world_chunk_size=5,
    )

    np.testing.assert_allclose(res_with_field.expected_payouts, res_without_field.expected_payouts)
    np.testing.assert_allclose(res_with_field.win_rates, res_without_field.win_rates)
    np.testing.assert_allclose(res_with_field.cash_rates, res_without_field.cash_rates)
    np.testing.assert_allclose(res_with_field.top_1pct_rates, res_without_field.top_1pct_rates)
    np.testing.assert_allclose(res_with_field.top_5pct_rates, res_without_field.top_5pct_rates)
    np.testing.assert_allclose(res_with_field.top_10pct_rates, res_without_field.top_10pct_rates)

    np.testing.assert_allclose(res_without_field.field_expected_payouts, np.zeros_like(res_without_field.field_expected_payouts))
    np.testing.assert_allclose(res_without_field.field_win_rates, np.zeros_like(res_without_field.field_win_rates))
    np.testing.assert_allclose(res_without_field.field_cash_rates, np.zeros_like(res_without_field.field_cash_rates))
