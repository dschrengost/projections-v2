"""Data models for Late Swap V2 sessionized workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from math import ceil, floor
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def utc_now_iso() -> str:
    return (
        datetime.now(tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


class ExposureBoundsPct(BaseModel):
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ExposureBoundsPct":
        if self.min is not None and not (0.0 <= float(self.min) <= 100.0):
            raise ValueError("exposure min must be in [0, 100]")
        if self.max is not None and not (0.0 <= float(self.max) <= 100.0):
            raise ValueError("exposure max must be in [0, 100]")
        if self.min is not None and self.max is not None and float(self.min) > float(self.max):
            raise ValueError("exposure min cannot exceed max")
        return self


class SegmentPolicyOverride(BaseModel):
    exposure_bounds: dict[str, ExposureBoundsPct] = Field(default_factory=dict)
    team_exposure_bounds: dict[str, ExposureBoundsPct] = Field(default_factory=dict)
    game_exposure_bounds: dict[str, ExposureBoundsPct] = Field(default_factory=dict)
    min_uniques: int | None = None
    max_duplicate_lineups: int | None = None
    max_total_swaps: int | None = None


LateSwapMode = Literal[
    "preserve_targets",
    "best_ev",
    "decorrelated_ev",
    "catch_up",
    "block",
    "manual_review",
]

TargetSource = Literal["source_portfolio", "current_entries", "explicit", "none"]
SegmentMode = Literal["global", "by_contest", "by_tag"]
RerunAnchor = Literal["source_portfolio", "last_committed", "session_start"]
LateSwapStatus = Literal["draft", "preview_ready", "committed", "stale", "failed"]


class LateSwapPolicy(BaseModel):
    mode: LateSwapMode = "preserve_targets"
    target_source: TargetSource = "source_portfolio"
    exposure_bounds: dict[str, ExposureBoundsPct] = Field(default_factory=dict)
    team_exposure_bounds: dict[str, ExposureBoundsPct] = Field(default_factory=dict)
    game_exposure_bounds: dict[str, ExposureBoundsPct] = Field(default_factory=dict)
    min_uniques: int = 0
    max_duplicate_lineups: int = 1
    candidate_count_per_entry: int = 10
    max_swaps_per_entry: int | None = None
    max_total_swaps: int | None = None
    min_gain_to_swap: float = 0.0
    swap_cost_lambda: float = 0.15
    target_deviation_lambda: float = 0.25
    overlap_penalty_lambda: float = 0.0
    ownership_penalty_lambda: float = 0.0
    leverage_boost_lambda: float = 0.0
    segment_mode: SegmentMode = "global"
    segment_overrides: dict[str, SegmentPolicyOverride] = Field(default_factory=dict)
    rerun_anchor: RerunAnchor = "source_portfolio"

    @model_validator(mode="after")
    def _validate_counts(self) -> "LateSwapPolicy":
        if not (6 <= int(self.candidate_count_per_entry) <= 20):
            raise ValueError("candidate_count_per_entry must be in [6, 20]")
        if self.min_uniques < 0:
            raise ValueError("min_uniques must be >= 0")
        if self.max_duplicate_lineups < 1:
            raise ValueError("max_duplicate_lineups must be >= 1")
        if self.max_swaps_per_entry is not None and self.max_swaps_per_entry < 0:
            raise ValueError("max_swaps_per_entry must be >= 0")
        if self.max_total_swaps is not None and self.max_total_swaps < 0:
            raise ValueError("max_total_swaps must be >= 0")
        return self

    @classmethod
    def with_mode_defaults(cls, mode: LateSwapMode) -> "LateSwapPolicy":
        policy = cls(mode=mode)
        if mode == "best_ev":
            policy.swap_cost_lambda = 0.05
            policy.target_deviation_lambda = 0.05
        elif mode == "decorrelated_ev":
            policy.swap_cost_lambda = 0.05
            policy.target_deviation_lambda = 0.05
            policy.overlap_penalty_lambda = 0.2
        elif mode == "catch_up":
            policy.swap_cost_lambda = 0.03
            policy.target_deviation_lambda = 0.08
            policy.ownership_penalty_lambda = 0.18
            policy.leverage_boost_lambda = 0.12
        elif mode == "block":
            policy.swap_cost_lambda = 0.35
            policy.target_deviation_lambda = 0.45
            policy.ownership_penalty_lambda = 0.02
        elif mode == "manual_review":
            policy.swap_cost_lambda = 0.1
            policy.target_deviation_lambda = 0.15
        return policy


class LateSwapSourceProfile(BaseModel):
    entries_total: int = 0
    contests_total: int = 0
    source_build_ids: list[str] = Field(default_factory=list)
    source_portfolio_build_ids: list[str] = Field(default_factory=list)
    source_selection_modes: list[str] = Field(default_factory=list)
    mixed_sources: bool = False


class LateSwapLockStateSummary(BaseModel):
    entries_total: int = 0
    locked_slots_total: int = 0
    locked_slots_pct: float = 0.0
    entries_fully_locked: int = 0
    entries_with_unmapped_locked_slots: int = 0
    locked_slots_by_entry_id: dict[str, list[str]] = Field(default_factory=dict)
    unmapped_locked_by_entry_id: dict[str, list[str]] = Field(default_factory=dict)
    player_locked_floor_count: dict[str, int] = Field(default_factory=dict)


class LateSwapCandidateSummary(BaseModel):
    entries_total: int = 0
    entries_with_candidates: int = 0
    requested_total: int = 0
    generated_total: int = 0
    deduped_total: int = 0
    rejected_unassignable_total: int = 0
    rejected_salary_total: int = 0
    rejected_swap_limit_total: int = 0
    pass_counts: dict[str, int] = Field(default_factory=dict)


class LateSwapSelectionSummary(BaseModel):
    status: Literal["optimal", "feasible", "fallback_hold", "infeasible"] = "fallback_hold"
    entries_total: int = 0
    entries_swapped: int = 0
    entries_held: int = 0
    total_swaps: int = 0
    average_swaps_per_changed_entry: float = 0.0
    objective_value: float | None = None
    projected_delta_total: float | None = None
    projected_delta_avg: float | None = None
    max_exposure_before_pct: float | None = None
    max_exposure_after_pct: float | None = None
    duplicate_count_before: int = 0
    duplicate_count_after: int = 0
    infeasibility_count: int = 0


class ExposureStateRow(BaseModel):
    player_id: str
    player_name: str
    source_target_pct: float | None = None
    locked_floor_pct: float = 0.0
    current_committed_pct: float = 0.0
    proposed_final_pct: float = 0.0
    delta_vs_target_pct: float | None = None
    status: str = "within_target"
    hard_min_pct: float | None = None
    hard_max_pct: float | None = None
    forced_over_cap_by_locks: bool = False
    min_achievable_pct: float | None = None


class LateSwapDiagnostics(BaseModel):
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    infeasible_exposure_players: list[str] = Field(default_factory=list)
    forced_over_cap_players: list[str] = Field(default_factory=list)
    candidate_coverage_gaps: list[str] = Field(default_factory=list)
    stale_reasons: list[str] = Field(default_factory=list)
    exposure_states: list[ExposureStateRow] = Field(default_factory=list)
    selector_notes: list[str] = Field(default_factory=list)
    lock_notes: list[str] = Field(default_factory=list)


class LateSwapCandidate(BaseModel):
    candidate_id: str
    entry_id: str
    contest_id: str
    locked_slots: list[str] = Field(default_factory=list)
    slot_values: dict[str, str] = Field(default_factory=dict)
    player_ids: list[str] = Field(default_factory=list)
    unlocked_player_ids: list[str] = Field(default_factory=list)
    generated_by: Literal[
        "hold",
        "best_ev",
        "randomized",
        "cap_relief",
        "target_recovery",
        "low_own",
        "manual",
    ] = "hold"
    projected_score: float | None = None
    remaining_score_mean: float | None = None
    remaining_score_p90: float | None = None
    expected_value: float | None = None
    roi: float | None = None
    total_own: float | None = None
    swap_count: int = 0
    added_player_ids: list[str] = Field(default_factory=list)
    removed_player_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    lineup_hash: str = ""
    objective_score: float = 0.0


class LateSwapSession(BaseModel):
    session_id: str
    game_date: str
    site: str = "dk"
    contest_ids: list[str]
    draft_group_ids: list[int]
    created_at: str
    updated_at: str
    status: LateSwapStatus = "draft"
    source_entry_revisions: dict[str, int] = Field(default_factory=dict)
    source_profile: LateSwapSourceProfile = Field(default_factory=LateSwapSourceProfile)
    policy: LateSwapPolicy = Field(default_factory=LateSwapPolicy)
    lock_state: LateSwapLockStateSummary = Field(default_factory=LateSwapLockStateSummary)
    candidate_summary: LateSwapCandidateSummary = Field(default_factory=LateSwapCandidateSummary)
    selection_summary: LateSwapSelectionSummary | None = None
    diagnostics: LateSwapDiagnostics = Field(default_factory=LateSwapDiagnostics)
    selected_candidates_by_entry_id: dict[str, str] = Field(default_factory=dict)
    pinned_candidates_by_entry_id: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class LateSwapSessionCreateRequest(BaseModel):
    date: str = Field(alias="game_date")
    contest_ids: list[str]
    policy: LateSwapPolicy | None = None

    model_config = {"populate_by_name": True}


class LateSwapPreviewRequest(BaseModel):
    run_id: str | None = None
    ownership_mode: str = "renormalize"
    use_user_overrides: bool = True
    only_out_lineups: bool = False


class LateSwapPinCandidatesRequest(BaseModel):
    pins: dict[str, str] = Field(default_factory=dict)
    clear_existing: bool = False


class LateSwapPolicyUpdateRequest(BaseModel):
    policy: LateSwapPolicy


class LateSwapCommitRequest(BaseModel):
    note: str | None = None


class LateSwapExportRequest(BaseModel):
    include_uncommitted_preview: bool = False
    contest_ids: list[str] | None = None


class LateSwapPreviewResponse(BaseModel):
    session: LateSwapSession
    candidates_by_entry_id: dict[str, list[LateSwapCandidate]] = Field(default_factory=dict)
    selected_candidates_by_entry_id: dict[str, str] = Field(default_factory=dict)


def pct_to_min_count(pct: float, total: int) -> int:
    return max(0, min(total, int(ceil((float(pct) / 100.0) * float(total) - 1e-12))))


def pct_to_max_count(pct: float, total: int) -> int:
    return max(0, min(total, int(floor((float(pct) / 100.0) * float(total) + 1e-12))))
