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

export interface ContestSimRequest {
    game_date: string
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
    draft_group_id: number
    version?: string
    k?: number
    candidate_pool_size?: number
    rebuild?: boolean
    rebuild_candidates?: boolean
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
    kind: 'run' | 'lineups'
    stats?: Record<string, unknown>
}

export interface SavedSimBuildDetail extends SavedSimBuildSummary {
    config?: Record<string, unknown> | null
    results?: LineupEVResult[]
    lineups: string[][]
    request?: Record<string, unknown> | null
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

export async function listFieldLibraries(date: string, draft_group_id: number): Promise<FieldLibrarySummary[]> {
    const resp = await fetch(`${API_BASE}/field-libraries?date=${encodeURIComponent(date)}&draft_group_id=${draft_group_id}`)
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
    kind?: 'run' | 'lineups',
): Promise<SavedSimBuildSummary[]> {
    const params = new URLSearchParams({ date })
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
): Promise<SavedSimBuildDetail> {
    const resp = await fetch(`${API_BASE}/saved-builds/${buildId}?date=${encodeURIComponent(date)}`)
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
): Promise<SavedSimBuildSummary> {
    const resp = await fetch(`${API_BASE}/saved-lineups`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            game_date: date,
            draft_group_id: draftGroupId,
            name,
            lineups,
            results: results ?? null,
            config: config ?? null,
            stats: stats ?? null,
        }),
    })
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }))
        throw new Error(err.detail || 'Failed to save sim lineups')
    }
    return resp.json()
}

export async function deleteSavedSimBuild(date: string, buildId: string): Promise<void> {
    const resp = await fetch(`${API_BASE}/saved-builds/${buildId}?date=${encodeURIComponent(date)}`, {
        method: 'DELETE',
    })
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }))
        throw new Error(err.detail || 'Failed to delete sim build')
    }
}
