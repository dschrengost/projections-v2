import pandas as pd

from projections.cli import score_minutes_v1 as score_cli


def test_rotation_prob_nullable_na_does_not_trigger_high_prob() -> None:
    df = pd.DataFrame(
        {
            "is_starter": [1, 0],
            # Nullable Float64 columns (parquet often yields these) should not propagate <NA>
            # through boolean masks and accidentally bump rotation_prob.
            "min_last1": pd.Series([pd.NA, pd.NA], dtype="Float64"),
            "min_last3": pd.Series([pd.NA, pd.NA], dtype="Float64"),
            "roll_mean_5": [float("nan"), float("nan")],
            "recent_start_pct_10": [0.0, 0.0],
            "days_since_last": pd.Series([pd.NA, pd.NA], dtype="Float64"),
        }
    )

    prob = score_cli._derive_rotation_prob(df)
    assert prob.iloc[0] == 0.98  # starter
    assert prob.iloc[1] == 0.20  # no history -> conservative default

