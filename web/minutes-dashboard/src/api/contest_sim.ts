/**
 * Contest Sim API client
 */

const API_BASE = '/api/contest-sim'

export interface LineupEVResult {
    lineup_id: number
    player_ids: string[]
    mean: number
    std: number
    p90: number
    p95: number
    expected_payout: number
    expected_value: number
    roi: number
    win_rate: number
    top_1pct_rate: number
    top_5pct_rate: number
    top_10pct_rate: number
    cash_rate: number
    dupe_penalty?: number
    unadjusted_expected_payout?: number | null
    adjusted_expected_payout?: number | null
    // Tail / upside selection metrics
    ucv90?: number | null  // Upper CVaR at 90th pctile (mean of top 10% scores)
    tail_score?: number | null  // Weighted combo: 0.6*p90 + 0.4*ucv90
    select_score?: number | null  // tail_score - dupe penalty impact
    score_lcb95?: number | null  // mean - 1.96*std
    score_cvar10?: number | null  // mean of worst 10% worlds
    robust_floor?: number | null  // min(score_lcb95, score_cvar10)
}

export interface ContestConfig {
    field_size: number
    entry_fee: number
    archetype: string
    rake: number
    prize_pool: number
}

export interface SummaryStats {
    lineup_count: number
    worlds_count: number
    avg_ev: number
    avg_roi: number
    positive_ev_count: number
    best_ev_lineup_id: number
    best_win_rate_lineup_id: number
    best_top1pct_lineup_id: number
    debug?: Record<string, unknown>
}

export interface ContestSimResponse {
    results: LineupEVResult[]
    config: ContestConfig
    stats: SummaryStats
    build_id?: string | null
}

export type SiteCode = 'dk' | 'fd'

export interface ContestSimRequest {
    game_date: string
    site?: SiteCode
    draft_group_id?: number | null
    lineups: string[][]
    field_mode?: 'self_play' | 'generated_field'
    field_library_version?: string
    field_library_k?: number
    field_candidate_pool_size?: number
    field_library_rebuild?: boolean
    field_library_rebuild_candidates?: boolean
    archetype?: string
    field_size_bucket?: string
    field_size_override?: number
    entry_fee?: number
    weights?: number[]
    ownership_mode?: 'full' | 'off' | 'dupe_only' | 'field_only'
    rank_mode?: 'current' | 'tail_only' | 'tail_times_dupe'
    use_strategy_overrides?: boolean
}

export interface FieldLibrarySummary {
    version: string
    path: string
    game_date: string
    draft_group_id: number
    method?: string | null
    generated_at?: string | null
    selected_k: number
    weights_sum: number
    meta?: Record<string, unknown>
}

export interface BuildFieldLibraryRequest {
    game_date: string
    site?: SiteCode
    draft_group_id: number
    version?: string
    k?: number
    candidate_pool_size?: number
    rebuild?: boolean
    rebuild_candidates?: boolean
    ownership_mode?: 'full' | 'off' | 'dupe_only' | 'field_only'
}

export interface FieldSizeOption {
    key: string
    label: string
    default: number
    range: number[]
}

export interface PayoutArchetypeOption {
    key: string
    label: string
    first_place_pct: number
    itm_pct: number
}

export interface ConfigResponse {
    field_sizes: FieldSizeOption[]
    payout_archetypes: PayoutArchetypeOption[]
    default_entry_fee: number
    default_archetype: string
    default_field_size_bucket: string
}

export interface SavedSimBuildSummary {
    build_id: string
    game_date: string
    draft_group_id?: number | null
    created_at: string
    lineups_count: number
    name?: string | null
    kind: 'run' | 'lineups' | 'portfolio'
    stats?: Record<string, unknown>
}

export interface SavedSimBuildDetail extends SavedSimBuildSummary {
    config?: Record<string, unknown> | null
    results?: LineupEVResult[]
    lineups: string[][]
    request?: Record<string, unknown> | null
}

export type PortfolioSelectionMode = 'greedy_constraints' | 'decorrelated_ev' | 'weighted_allocations'

export interface PortfolioExposureBounds {
    min?: number
    max?: number
}

export interface PortfolioSelectionRequest {
    game_date: string
    site?: SiteCode
    draft_group_id?: number | null
    source_build_id: string
    mode: PortfolioSelectionMode
    worlds_source?: 'gtv2' | 'sim_v2'
    sort_key?: keyof LineupEVResult | 'lineup_id' | 'total_own'
    sort_dir?: 'asc' | 'desc'
    portfolio_size: number
    ev_retention?: number
    worlds_sample?: number
    worlds_train_frac?: number | null
    world_indices?: number[] | null
    min_uniques?: number
    max_total_own?: number | null
    filter_positive_ev?: boolean
    top_n?: number | null
    candidate_lineup_ids?: number[] | null
    seed_lineup_ids?: number[] | null
    exposure_bounds?: Record<string, PortfolioExposureBounds>
    seed?: number
}

export interface PortfolioSelectionResponse {
    mode: PortfolioSelectionMode
    source_build_id: string
    candidate_count: number
    filtered_candidate_count: number
    selected_lineup_ids: number[]
    selected_results: LineupEVResult[]
    selected_lineups: string[][]
    weights?: number[] | null
    diagnostics: Record<string, unknown>
    warnings: string[]
}

export async function runContestSim(request: ContestSimRequest): Promise<ContestSimResponse> {
    const resp = await fetch(`${API_BASE}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
    })
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }))
        throw new Error(err.detail || 'Contest simulation failed')
    }
    return resp.json()
}

export async function getContestSimConfig(): Promise<ConfigResponse> {
    const resp = await fetch(`${API_BASE}/config`)
    if (!resp.ok) {
        throw new Error('Failed to load contest sim config')
    }
    return resp.json()
}

export async function listFieldLibraries(date: string, draft_group_id: number, site: SiteCode = 'dk'): Promise<FieldLibrarySummary[]> {
    const resp = await fetch(`${API_BASE}/field-libraries?date=${encodeURIComponent(date)}&draft_group_id=${draft_group_id}&site=${site}`)
    if (!resp.ok) {
        throw new Error('Failed to load field libraries')
    }
    return resp.json()
}

export async function buildFieldLibrary(req: BuildFieldLibraryRequest): Promise<FieldLibrarySummary> {
    const resp = await fetch(`${API_BASE}/field-libraries/build`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
    })
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }))
        throw new Error(err.detail || 'Failed to build field library')
    }
    return resp.json()
}

export async function getSavedSimBuilds(
    date: string,
    kind?: 'run' | 'lineups' | 'portfolio',
    site: SiteCode = 'dk',
): Promise<SavedSimBuildSummary[]> {
    const params = new URLSearchParams({ date, site })
    if (kind) params.set('kind', kind)
    const resp = await fetch(`${API_BASE}/saved-builds?${params.toString()}`)
    if (!resp.ok) {
        throw new Error('Failed to load saved sim builds')
    }
    return resp.json()
}

export async function loadSavedSimBuild(
    date: string,
    buildId: string,
    site: SiteCode = 'dk',
): Promise<SavedSimBuildDetail> {
    const resp = await fetch(`${API_BASE}/saved-builds/${buildId}?date=${encodeURIComponent(date)}&site=${site}`)
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }))
        throw new Error(err.detail || 'Failed to load saved sim build')
    }
    return resp.json()
}

export async function saveSimLineups(
    date: string,
    draftGroupId: number | null,
    name: string,
    lineups: string[][],
    results?: LineupEVResult[] | null,
    config?: ContestConfig | null,
    stats?: SummaryStats | null,
    options?: {
        site?: SiteCode
        kind?: 'lineups' | 'portfolio'
        sourceBuildId?: string | null
        selectionMode?: string | null
        selectionConfig?: Record<string, unknown> | null
        selectionDiagnostics?: Record<string, unknown> | null
        warnings?: string[]
    },
): Promise<SavedSimBuildSummary> {
    const resp = await fetch(`${API_BASE}/saved-lineups`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            game_date: date,
            site: options?.site ?? 'dk',
            draft_group_id: draftGroupId,
            name,
            lineups,
            kind: options?.kind ?? 'lineups',
            results: results ?? null,
            config: config ?? null,
            stats: stats ?? null,
            source_build_id: options?.sourceBuildId ?? null,
            selection_mode: options?.selectionMode ?? null,
            selection_config: options?.selectionConfig ?? null,
            selection_diagnostics: options?.selectionDiagnostics ?? null,
            warnings: options?.warnings ?? [],
        }),
    })
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }))
        throw new Error(err.detail || 'Failed to save sim lineups')
    }
    return resp.json()
}

export async function deleteSavedSimBuild(date: string, buildId: string, site: SiteCode = 'dk'): Promise<void> {
    const resp = await fetch(`${API_BASE}/saved-builds/${buildId}?date=${encodeURIComponent(date)}&site=${site}`, {
        method: 'DELETE',
    })
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }))
        throw new Error(err.detail || 'Failed to delete sim build')
    }
}

export async function selectPortfolio(
    request: PortfolioSelectionRequest,
): Promise<PortfolioSelectionResponse> {
    const resp = await fetch(`${API_BASE}/portfolio`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
    })
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }))
        throw new Error(err.detail || 'Failed to build portfolio')
    }
    return resp.json()
}
