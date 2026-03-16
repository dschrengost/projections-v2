import { apiUrl } from './client'

export type SiteCode = 'dk' | 'fd'

export type LateSwapMode =
    | 'preserve_targets'
    | 'best_ev'
    | 'decorrelated_ev'
    | 'catch_up'
    | 'block'
    | 'manual_review'

export interface ExposureBoundsPct {
    min?: number | null
    max?: number | null
}

export interface LateSwapPolicy {
    mode: LateSwapMode
    target_source: 'source_portfolio' | 'current_entries' | 'explicit' | 'none'
    exposure_bounds: Record<string, ExposureBoundsPct>
    team_exposure_bounds: Record<string, ExposureBoundsPct>
    game_exposure_bounds: Record<string, ExposureBoundsPct>
    min_uniques: number
    max_duplicate_lineups: number
    candidate_count_per_entry: number
    max_swaps_per_entry?: number | null
    max_total_swaps?: number | null
    min_gain_to_swap: number
    swap_cost_lambda: number
    target_deviation_lambda: number
    overlap_penalty_lambda: number
    ownership_penalty_lambda: number
    leverage_boost_lambda: number
    segment_mode: 'global' | 'by_contest' | 'by_tag'
    segment_overrides: Record<string, unknown>
    rerun_anchor: 'source_portfolio' | 'last_committed' | 'session_start'
}

export interface ExposureStateRow {
    player_id: string
    player_name: string
    source_target_pct?: number | null
    locked_floor_pct: number
    current_committed_pct: number
    proposed_final_pct: number
    delta_vs_target_pct?: number | null
    status: string
    hard_min_pct?: number | null
    hard_max_pct?: number | null
    forced_over_cap_by_locks: boolean
    min_achievable_pct?: number | null
}

export interface LateSwapDiagnostics {
    warnings: string[]
    errors: string[]
    infeasible_exposure_players: string[]
    forced_over_cap_players: string[]
    candidate_coverage_gaps: string[]
    stale_reasons: string[]
    selector_notes: string[]
    lock_notes: string[]
    exposure_states: ExposureStateRow[]
}

export interface LateSwapSelectionSummary {
    status: string
    entries_total: number
    entries_swapped: number
    entries_held: number
    total_swaps: number
    average_swaps_per_changed_entry: number
    objective_value?: number | null
    projected_delta_total?: number | null
    projected_delta_avg?: number | null
    max_exposure_before_pct?: number | null
    max_exposure_after_pct?: number | null
    duplicate_count_before: number
    duplicate_count_after: number
    infeasibility_count: number
}

export interface LateSwapSession {
    session_id: string
    game_date: string
    site: string
    contest_ids: string[]
    draft_group_ids: number[]
    created_at: string
    updated_at: string
    status: 'draft' | 'preview_ready' | 'committed' | 'stale' | 'failed'
    source_entry_revisions: Record<string, number>
    source_profile: {
        entries_total: number
        contests_total: number
        source_build_ids: string[]
        source_portfolio_build_ids: string[]
        source_selection_modes: string[]
        mixed_sources: boolean
    }
    policy: LateSwapPolicy
    lock_state: {
        entries_total: number
        locked_slots_total: number
        locked_slots_pct: number
        entries_fully_locked: number
        entries_with_unmapped_locked_slots: number
        locked_slots_by_entry_id: Record<string, string[]>
        unmapped_locked_by_entry_id: Record<string, string[]>
        player_locked_floor_count: Record<string, number>
    }
    candidate_summary: {
        entries_total: number
        entries_with_candidates: number
        requested_total: number
        generated_total: number
        deduped_total: number
        rejected_unassignable_total: number
        rejected_salary_total: number
        rejected_swap_limit_total: number
        pass_counts: Record<string, number>
    }
    selection_summary?: LateSwapSelectionSummary | null
    diagnostics: LateSwapDiagnostics
    selected_candidates_by_entry_id: Record<string, string>
    pinned_candidates_by_entry_id: Record<string, string>
    warnings: string[]
}

export interface LateSwapCandidate {
    candidate_id: string
    entry_id: string
    contest_id: string
    locked_slots: string[]
    slot_values: Record<string, string>
    player_ids: string[]
    projected_score?: number | null
    total_own?: number | null
    swap_count: number
    added_player_ids: string[]
    removed_player_ids: string[]
    generated_by: string
    reason_codes: string[]
    objective_score: number
}

export interface LateSwapPreviewResponse {
    session: LateSwapSession
    candidates_by_entry_id: Record<string, LateSwapCandidate[]>
    selected_candidates_by_entry_id: Record<string, string>
}

export function defaultLateSwapPolicy(mode: LateSwapMode = 'preserve_targets'): LateSwapPolicy {
    const base: LateSwapPolicy = {
        mode,
        target_source: 'source_portfolio',
        exposure_bounds: {},
        team_exposure_bounds: {},
        game_exposure_bounds: {},
        min_uniques: 0,
        max_duplicate_lineups: 1,
        candidate_count_per_entry: 10,
        max_swaps_per_entry: null,
        max_total_swaps: null,
        min_gain_to_swap: 0,
        swap_cost_lambda: 0.15,
        target_deviation_lambda: 0.25,
        overlap_penalty_lambda: 0,
        ownership_penalty_lambda: 0,
        leverage_boost_lambda: 0,
        segment_mode: 'global',
        segment_overrides: {},
        rerun_anchor: 'source_portfolio',
    }
    if (mode === 'best_ev') {
        base.swap_cost_lambda = 0.05
        base.target_deviation_lambda = 0.05
    } else if (mode === 'decorrelated_ev') {
        base.swap_cost_lambda = 0.05
        base.target_deviation_lambda = 0.05
        base.overlap_penalty_lambda = 0.2
    } else if (mode === 'catch_up') {
        base.swap_cost_lambda = 0.03
        base.target_deviation_lambda = 0.08
        base.ownership_penalty_lambda = 0.18
        base.leverage_boost_lambda = 0.12
    } else if (mode === 'block') {
        base.swap_cost_lambda = 0.35
        base.target_deviation_lambda = 0.45
    }
    return base
}

async function parseOrThrow(res: Response, fallback: string) {
    if (res.ok) return res.json()
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `${fallback}: ${res.status}`)
}

export async function createLateSwapSession(
    date: string,
    contestIds: string[],
    policy: LateSwapPolicy,
    site: SiteCode = 'dk',
): Promise<LateSwapSession> {
    const res = await fetch(apiUrl('/api/entry-manager/late-swap/sessions'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            date,
            site,
            contest_ids: contestIds,
            policy,
        }),
    })
    return parseOrThrow(res, 'Failed to create late swap session')
}

export async function listLateSwapSessions(date: string, site: SiteCode = 'dk'): Promise<LateSwapSession[]> {
    const res = await fetch(apiUrl(`/api/entry-manager/late-swap/sessions?date=${date}&site=${site}`))
    return parseOrThrow(res, 'Failed to list late swap sessions')
}

export async function getLateSwapSession(sessionId: string, date?: string): Promise<LateSwapPreviewResponse> {
    const url = date
        ? apiUrl(`/api/entry-manager/late-swap/sessions/${sessionId}?date=${date}`)
        : apiUrl(`/api/entry-manager/late-swap/sessions/${sessionId}`)
    const res = await fetch(url)
    return parseOrThrow(res, 'Failed to load late swap session')
}

export async function previewLateSwapSession(
    sessionId: string,
    request?: {
        runId?: string | null
        ownershipMode?: string
        useUserOverrides?: boolean
        onlyOutLineups?: boolean
    },
    date?: string,
): Promise<LateSwapPreviewResponse> {
    const url = date
        ? apiUrl(`/api/entry-manager/late-swap/sessions/${sessionId}/preview?date=${date}`)
        : apiUrl(`/api/entry-manager/late-swap/sessions/${sessionId}/preview`)
    const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            run_id: request?.runId ?? null,
            ownership_mode: request?.ownershipMode ?? 'renormalize',
            use_user_overrides: request?.useUserOverrides ?? true,
            only_out_lineups: request?.onlyOutLineups ?? false,
        }),
    })
    return parseOrThrow(res, 'Failed to preview late swap session')
}

export async function pinLateSwapCandidates(
    sessionId: string,
    pins: Record<string, string>,
    date?: string,
): Promise<LateSwapPreviewResponse> {
    const url = date
        ? apiUrl(`/api/entry-manager/late-swap/sessions/${sessionId}/pin-candidates?date=${date}`)
        : apiUrl(`/api/entry-manager/late-swap/sessions/${sessionId}/pin-candidates`)
    const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pins }),
    })
    return parseOrThrow(res, 'Failed to pin late swap candidate')
}

export async function updateLateSwapPolicy(
    sessionId: string,
    policy: LateSwapPolicy,
    date?: string,
): Promise<LateSwapSession> {
    const url = date
        ? apiUrl(`/api/entry-manager/late-swap/sessions/${sessionId}/policy?date=${date}`)
        : apiUrl(`/api/entry-manager/late-swap/sessions/${sessionId}/policy`)
    const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ policy }),
    })
    return parseOrThrow(res, 'Failed to update late swap policy')
}

export async function commitLateSwapSession(sessionId: string, date?: string): Promise<LateSwapSession> {
    const url = date
        ? apiUrl(`/api/entry-manager/late-swap/sessions/${sessionId}/commit?date=${date}`)
        : apiUrl(`/api/entry-manager/late-swap/sessions/${sessionId}/commit`)
    const res = await fetch(url, { method: 'POST' })
    return parseOrThrow(res, 'Failed to commit late swap session')
}

export async function exportLateSwapSession(
    sessionId: string,
    options?: { includeUncommittedPreview?: boolean; date?: string },
): Promise<Blob> {
    const dateQuery = options?.date ? `?date=${encodeURIComponent(options.date)}` : ''
    const res = await fetch(apiUrl(`/api/entry-manager/late-swap/sessions/${sessionId}/export${dateQuery}`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            include_uncommitted_preview: options?.includeUncommittedPreview ?? false,
        }),
    })
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Failed to export late swap session: ${res.status}`)
    }
    return res.blob()
}
