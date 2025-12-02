"""Notes on how Vegas odds are currently handled in the projections stack.

Data locations
--------------
- Bronze odds snapshots are written by ``projections.etl.odds`` to
  ``<DATA_ROOT>/bronze/odds_raw/season=YYYY/odds_{month}.parquet`` with the schema defined in
  :data:`projections.minutes_v1.schemas.ODDS_RAW_SCHEMA`
  (``game_id``, ``as_of_ts``, ``book``, ``market``, ``home_team_id``, ``away_team_id``,
  ``spread_home``, ``total``, ``ingested_ts``, ``source``).
- The canonical feed for features lives under ``<DATA_ROOT>/silver/odds_snapshot/season=YYYY/month=MM/odds_snapshot.parquet``
  (:data:`ODDS_SNAPSHOT_SCHEMA`) carrying one spread/total per ``game_id`` plus ``as_of_ts`` + book metadata.

Minutes features & model
------------------------
- ``MinutesFeatureBuilder`` calls :func:`projections.features.game_env.attach_game_environment_features`
  inside ``_attach_odds`` which merges the silver odds snapshot onto each player row and emits
  ``spread_home``, ``total`, ``odds_as_of_ts``, ``blowout_index``, ``blowout_risk_score``, and ``close_game_score``.
-  These columns are present in the gold feature parquet (e.g.
  ``/home/daniel/projections-data/gold/features_minutes_v1/season=2025/month=10/features.parquet``).
- The production minutes bundle (``artifacts/minutes_lgbm/lgbm_full_v1_no_p_play_20251202``) includes the Vegas signals in
  ``feature_columns.json``: ``spread_home``, ``total``, ``blowout_index``, ``blowout_risk_score``, ``close_game_score``.
  ``odds_as_of_ts`` is persisted for auditing but excluded from modeling.
- Minutes gold outputs (``<DATA_ROOT>/gold/projections_minutes_v1/.../minutes.parquet``) do **not** expose any odds fields.
  The public FastAPI (`/api/minutes`) only serializes the columns listed in ``PLAYER_COLUMNS``, so no Vegas data leaves the API.

FPTS dataset & model
--------------------
- ``projections.fpts_v1.datasets.FptsDatasetBuilder`` inherits the minutes feature columns and derives
  ``team_implied_total`` and ``opponent_implied_total`` whenever ``total``/``spread_home``/``home_flag`` are present.
- The active FPTS bundle (``artifacts/fpts_lgbm/fpts_lgbm_v2/model.joblib``) keeps
  ``spread_home``, ``total``, ``team_implied_total``, and ``opponent_implied_total`` in ``feature_columns``.
- Gold FPTS exports (e.g. ``<DATA_ROOT>/gold/projections_fpts_v1/2025-11-24/run=.../fpts.parquet``) only contain player-level
  projection outputs (`proj_fpts`, `minutes_*`, etc.) and omit the Vegas columns, so the public API likewise does not share odds.
"""
