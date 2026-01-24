from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

import projections.builders.features_builder as features_builder_mod
from projections.builders.features_builder import FeatureBuildConfig, SharedFeaturesBuilder
from projections.builders.injuries_resolver import InjuriesResolutionResult


def test_build_passes_full_schedule_to_dnp_history(tmp_path: Path, monkeypatch) -> None:
    """Regression test: DNP history must see the full schedule, not label-only schedule.

    SharedFeaturesBuilder.build filters `schedule_for_builder` down to label game_ids
    for MinutesFeatureBuilder. DNP history needs the *full* schedule (team-game spine),
    otherwise it misses OUT games that are not present in boxscore labels.
    """

    target_day = date(2025, 12, 15)
    cfg = FeatureBuildConfig(
        data_root=tmp_path,
        season=2025,
        target_day=target_day,
        as_of_ts=pd.Timestamp("2025-12-15T18:30:00Z"),
    )
    builder = SharedFeaturesBuilder(cfg)

    # Avoid IO / injuries resolution work.
    def _fake_resolve(*, target_day, game_ids, tip_lookup, feature_as_of_ts, backfill_mode, allow_empty):  # noqa: ANN001
        return InjuriesResolutionResult(
            injuries=pd.DataFrame(),
            source="empty",
            games_with_injuries=set(),
            games_without_injuries=set(game_ids),
            warnings=[],
        )

    builder.injuries_resolver.resolve = _fake_resolve  # type: ignore[method-assign]

    # Avoid running the full MinutesFeatureBuilder pipeline.
    class DummyMinutesFeatureBuilder:
        def __init__(self, **kwargs):  # noqa: ANN003
            pass

        def build(self, labels: pd.DataFrame) -> pd.DataFrame:
            # Return a minimal frame that still allows SharedFeaturesBuilder.build to proceed.
            return labels.copy()

    monkeypatch.setattr(features_builder_mod, "MinutesFeatureBuilder", DummyMinutesFeatureBuilder)

    # No-op the expensive/enrichment steps; we only care about schedule passed to DNP history.
    monkeypatch.setattr(builder, "_attach_trend_features", lambda df, *, labels_source, warnings: df)
    monkeypatch.setattr(
        builder, "_attach_team_minutes_dispersion_prior", lambda df, *, labels_source, warnings: df
    )
    monkeypatch.setattr(builder, "_attach_team_context", lambda df, *, warnings: df)
    monkeypatch.setattr(
        builder,
        "_attach_vacancy_features",
        lambda df, *, labels_source, injuries_snapshot, roster_nightly, warnings: df,
    )

    captured: dict[str, object] = {}

    def _spy_dnp(df: pd.DataFrame, *, labels_source, schedule, warnings):  # noqa: ANN001
        captured["schedule_game_ids"] = set(pd.to_numeric(schedule["game_id"], errors="coerce").dropna().astype(int))
        return df

    monkeypatch.setattr(builder, "_attach_dnp_history_features", _spy_dnp)

    # Labels only reference game_id=1; schedule includes an extra game_id=999.
    labels = pd.DataFrame(
        {
            "game_id": [1],
            "team_id": [1610612751],
            "player_id": [101],
            "player_name": ["Player A"],
            "game_date": [pd.Timestamp(target_day)],
            "minutes": [pd.NA],
        }
    )
    schedule = pd.DataFrame(
        {
            "game_id": [1, 999],
            "game_date": [pd.Timestamp(target_day), pd.Timestamp(target_day)],
            "tip_ts": [pd.Timestamp("2025-12-15T23:00:00Z"), pd.Timestamp("2025-12-15T23:00:00Z")],
            "home_team_id": [1610612751, 1610612751],
            "away_team_id": [1610612744, 1610612744],
        }
    )

    _ = builder.build(
        labels=labels,
        schedule=schedule,
        game_ids=[1],
        roster=pd.DataFrame(),
        odds=pd.DataFrame(),
        coach=pd.DataFrame(),
        roles=pd.DataFrame(),
        archetype_deltas=pd.DataFrame(),
    )

    assert captured["schedule_game_ids"] == {1, 999}

