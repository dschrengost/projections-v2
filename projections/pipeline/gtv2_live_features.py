"""Live feature builder for GameTransformerV2 with training-contract parity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pandas as pd

from projections.rotation.game_transformer_v2 import GameTransformerV2Config
from projections.rotation.live_features_v1 import (
    KEY_COLS,
    build_rotation_set_minutes_v1_features,
    load_rotation_priors_for_live_inference,
)
from projections.cli.score_minutes_rotation_set_v1 import (
    _load_rotation_historical_features_for_dnp,
)
from scripts.rotation.build_joint_rotation_rates_dataset_v1 import (
    DEFAULT_EST_POSSESSIONS_CLIP_MAX,
    DEFAULT_EST_POSSESSIONS_CLIP_MIN,
    DEFAULT_EST_POSSESSIONS_PACE_WEIGHT,
    DEFAULT_LEAGUE_PPP,
    _apply_game_context_feature_contract,
    _apply_lineup_feature_contract,
)


class GTV2LiveFeatureBuildError(RuntimeError):
    """Raised when live GTV2 feature building fails."""


@dataclass(frozen=True)
class GTV2FeatureSpec:
    """Feature contract spec loaded from a GameTransformerV2 bundle."""

    bundle_dir: Path
    feature_columns: list[str]
    game_feature_columns: list[str]
    team_feature_columns: list[str]


@dataclass(frozen=True)
class GTV2LiveFeaturesBuildResult:
    """Result payload for live GTV2 feature building."""

    features: pd.DataFrame
    transform_manifest: dict[str, Any]
    diagnostics: dict[str, Any]


def load_gtv2_feature_spec(bundle_dir: Path) -> GTV2FeatureSpec:
    """Load model feature spec from bundle config."""
    root = Path(bundle_dir).expanduser().resolve()
    config_path = root / "config.json"
    if not config_path.exists():
        raise GTV2LiveFeatureBuildError(f"missing bundle config: {config_path}")
    config = GameTransformerV2Config.load(config_path)
    if not config.feature_columns:
        raise GTV2LiveFeatureBuildError(f"bundle config has empty feature_columns: {config_path}")
    return GTV2FeatureSpec(
        bundle_dir=root,
        feature_columns=list(config.feature_columns),
        game_feature_columns=list(config.game_feature_columns),
        team_feature_columns=list(config.team_feature_columns),
    )


def _season_for_day(day: pd.Timestamp) -> int:
    return int(day.year) if int(day.month) >= 8 else int(day.year) - 1


def _dedupe_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if not frame.columns.duplicated().any():
        return frame
    return frame.loc[:, ~frame.columns.duplicated()].copy()


def build_gtv2_live_features(
    *,
    minutes_features: pd.DataFrame,
    spec: GTV2FeatureSpec,
    data_root: Path,
    game_date: str,
    allow_priors_fallback: bool = True,
    dnp_lookback_days: int | None = None,
    league_ppp: float = DEFAULT_LEAGUE_PPP,
    estimated_possessions_pace_weight: float = DEFAULT_EST_POSSESSIONS_PACE_WEIGHT,
    estimated_possessions_clip_min: float = DEFAULT_EST_POSSESSIONS_CLIP_MIN,
    estimated_possessions_clip_max: float = DEFAULT_EST_POSSESSIONS_CLIP_MAX,
    priors_loader: Callable[..., Any] = load_rotation_priors_for_live_inference,
    rotation_feature_builder: Callable[..., Any] = build_rotation_set_minutes_v1_features,
    dnp_history_loader: Callable[..., pd.DataFrame] = _load_rotation_historical_features_for_dnp,
) -> GTV2LiveFeaturesBuildResult:
    """Build live features matching the training contract used by GTV2."""
    if minutes_features.empty:
        raise GTV2LiveFeatureBuildError("minutes_features is empty")

    for col in KEY_COLS:
        if col not in minutes_features.columns:
            raise GTV2LiveFeatureBuildError(f"minutes_features missing key column: {col}")

    day = pd.Timestamp(game_date).normalize()
    season = _season_for_day(day)
    work = minutes_features.copy()
    if "game_date" not in work.columns:
        work["game_date"] = day
    else:
        work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce").dt.normalize()
        work["game_date"] = work["game_date"].fillna(day)

    # These transforms are the exact dataset-contract transforms used in training.
    work = _apply_lineup_feature_contract(work)
    work, game_context_meta = _apply_game_context_feature_contract(
        work,
        league_ppp=float(league_ppp),
        pace_weight=float(estimated_possessions_pace_weight),
        clip_min=float(estimated_possessions_clip_min),
        clip_max=float(estimated_possessions_clip_max),
    )

    team_ids = (
        pd.to_numeric(work["team_id"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    player_ids = (
        pd.to_numeric(work["player_id"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    game_ids = (
        pd.to_numeric(work["game_id"], errors="coerce")
        .dropna()
        .astype(int)
        .astype(str)
        .str.zfill(10)
        .unique()
        .tolist()
    )
    if not team_ids or not player_ids or not game_ids:
        raise GTV2LiveFeatureBuildError("failed to resolve game/team/player ids for priors join")

    priors_result = priors_loader(
        Path(data_root),
        season=season,
        game_date=day.strftime("%Y-%m-%d"),
        game_ids=game_ids,
        team_ids=team_ids,
        player_ids=player_ids,
        allow_priors_fallback=bool(allow_priors_fallback),
    )
    team_priors = getattr(priors_result, "team_priors", pd.DataFrame())
    player_priors = getattr(priors_result, "player_priors", pd.DataFrame())
    if team_priors.empty or player_priors.empty:
        raise GTV2LiveFeatureBuildError(
            "rotation priors missing for live GTV2 feature build "
            f"(teams={len(team_ids)} players={len(player_ids)} games={len(game_ids)})"
        )

    historical_kwargs: dict[str, Any] = {
        "data_root": Path(data_root),
        "season": season,
        "target_day": day,
        "team_ids": set(team_ids),
        "player_ids": set(player_ids),
    }
    if dnp_lookback_days is not None:
        historical_kwargs["lookback_days"] = int(dnp_lookback_days)

    historical_df = dnp_history_loader(
        **historical_kwargs,
    )

    built = rotation_feature_builder(
        work,
        team_priors=team_priors,
        player_priors=player_priors,
        feature_columns=list(spec.feature_columns),
        historical_features=historical_df,
    )

    built_ns = built if hasattr(built, "features") else SimpleNamespace(features=built, dropped_extra_columns=[])
    features = _dedupe_columns(built_ns.features.copy())
    missing_features = [c for c in spec.feature_columns if c not in features.columns]
    if missing_features:
        raise GTV2LiveFeatureBuildError(
            f"GTV2 live feature build missing model features (n={len(missing_features)}): {missing_features}"
        )
    missing_keys = [c for c in KEY_COLS if c not in features.columns]
    if missing_keys:
        raise GTV2LiveFeatureBuildError(f"GTV2 live feature build missing key columns: {missing_keys}")

    out = features.loc[:, [*KEY_COLS, *spec.feature_columns]].copy()
    dupes = int(out.duplicated(subset=list(KEY_COLS), keep=False).sum())
    if dupes > 0:
        raise GTV2LiveFeatureBuildError(f"duplicate key rows in GTV2 live features: duplicates={dupes}")

    priors_mode = (
        "game_id_partitions_plus_pre_game_entity_fallback"
        if bool(allow_priors_fallback)
        else "game_id_partitions_only"
    )
    transform_manifest = {
        "builder": "gtv2_live_features_v1",
        "lineup_contract": "joint_rotation_rates_v1_section_4_7",
        "game_context_contract": {
            "league_ppp": float(league_ppp),
            "estimated_possessions_pace_weight": float(estimated_possessions_pace_weight),
            "estimated_possessions_clip_min": float(estimated_possessions_clip_min),
            "estimated_possessions_clip_max": float(estimated_possessions_clip_max),
        },
        "priors": {
            "allow_priors_fallback": bool(allow_priors_fallback),
            "mode": priors_mode,
            "explanation": (
                "Live slates often have no same-day game_id priors partitions pre-tip. "
                "Fallback uses latest pre-game priors by team/player; training contract is recorded "
                "via priors.mode for parity auditing."
            ),
        },
        "dnp_history": (
            {
                "mode": "bounded_lookback",
                "lookback_days": int(dnp_lookback_days),
            }
            if dnp_lookback_days is not None
            else {
                "mode": "full_prior_history",
                "lookback_days": None,
            }
        ),
    }

    diagnostics = {
        "rows": int(len(out)),
        "players": int(out["player_id"].nunique()),
        "teams": int(out["team_id"].nunique()),
        "games": int(out["game_id"].nunique()),
        "game_context_meta": dict(game_context_meta),
        "dropped_extra_columns": list(getattr(built_ns, "dropped_extra_columns", [])),
        "priors_used_latest_fallback": bool(getattr(priors_result, "used_latest_fallback", False)),
        "priors_warning_message": getattr(priors_result, "warning_message", None),
    }
    return GTV2LiveFeaturesBuildResult(
        features=out,
        transform_manifest=transform_manifest,
        diagnostics=diagnostics,
    )
