"""Shared builders for feature engineering.

These modules provide unified logic for building features across both live
and training pipelines, enforcing strict as_of_ts semantics.

Key principles:
- Feature code NEVER calls now() - all temporal state is injected via as_of_ts
- Live builder supplies as_of_ts based on configured offset from tip
- Training builder supplies historical as_of_ts (typically tip_ts)
- Both paths use identical resolution logic and produce identical schema
"""

from projections.builders.injuries_resolver import (
    InjuriesResolver,
    InjuriesResolutionResult,
    resolve_injuries,
)
from projections.builders.features_builder import (
    FeatureBuildConfig,
    FeatureBuildResult,
    SharedFeaturesBuilder,
    build_features_unified,
    build_shared_features,
)

__all__ = [
    # Injuries resolver
    "InjuriesResolver",
    "InjuriesResolutionResult",
    "resolve_injuries",
    # Features builder
    "FeatureBuildConfig",
    "FeatureBuildResult",
    "SharedFeaturesBuilder",
    "build_features_unified",
    "build_shared_features",
]
