"""Allocation primitives shared across minutes/rotation workflows."""

from .bounded_projection import ProjectionInfeasibleError, project_sum_with_bounds

__all__ = ["project_sum_with_bounds", "ProjectionInfeasibleError"]
