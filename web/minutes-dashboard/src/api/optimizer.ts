/**
 * Optimizer API client for QuickBuild lineup generation.
 */
import { apiUrl } from './client'

// Types

export interface GameInfo {
    matchup: string
    start_time?: string
}

export interface Slate {
    game_date: string
    slate_type: string
    draft_group_id: number
    n_contests: number
    earliest_start?: string
    latest_start?: string
    example_contest_name?: string
    games?: GameInfo[]
}

export interface PoolPlayer {
    player_id: string
    name: string
    team: string
    positions: string[]
    salary: number
    proj: number
    dk_id?: string
    own_proj?: number | null
    stddev?: number | null
    p90?: number | null
    game_matchup?: string
    game_start_utc?: string
    model_proj?: number | null
    model_minutes?: number | null
    model_own?: number | null
    effective_proj?: number | null
    effective_minutes?: number | null
    effective_own?: number | null
    effective_stddev?: number | null
    effective_p90?: number | null
    override_minutes_delta?: number | null
    override_fpts_delta?: number | null
    has_override?: boolean | null
    used_fppm_fallback?: boolean | null
    is_active?: boolean | null
    is_out?: boolean | null
    fppm?: number | null
    optimal_pct?: number | null
    ceiling_leverage?: number | null
    boom_pct?: number | null
    bust_pct?: number | null
}


export interface QuickBuildRequest {
    date: string
    draft_group_id: number
    site?: string
    run_id?: string | null
    max_pool?: number
    builds?: number
    per_build?: number
    min_uniq?: number
    max_exposure_pct?: number | null
    jitter?: number
    near_dup_jaccard?: number
    enum_enable?: boolean
    min_salary?: number | null
    max_salary?: number | null
    global_team_limit?: number
    lock_ids?: string[]
    ban_ids?: string[]
    max_offoptimal_pct?: number | null
    include_games?: string[]
    exclude_games?: string[]
    ownership_penalty_enabled?: boolean
    ownership_lambda?: number
    late_swap_enabled?: boolean
    late_swap_bonus_per_hour?: number
    late_swap_max_bonus?: number
    randomness_pct?: number | null
    use_strategy_overrides?: boolean
    use_user_overrides?: boolean
    ownership_mode?: string
    world_sample_enabled?: boolean
    worlds_source?: 'gtv2' | 'sim_v2'
}

export interface JobStatus {
    job_id: string
    status: 'pending' | 'running' | 'completed' | 'failed'
    created_at: string
    game_date: string
    draft_group_id: number
    site: string
    started_at?: string | null
    completed_at?: string | null
    progress: number
    target: number
    lineups_count: number
    wall_time_sec?: number | null
    error?: string | null
}

export interface LineupRow {
    lineup_id: number
    player_ids: string[]
    mean?: number | null
    p10?: number | null
    p50?: number | null
    p75?: number | null
    p90?: number | null
    stdev?: number | null
    ceiling_upside?: number | null
}

export interface LineupsResponse {
    job_id: string
    lineups_count: number
    lineups: LineupRow[]
    stats: Record<string, unknown>
}

// API Functions

export async function getSlates(date: string, slateType = 'all'): Promise<Slate[]> {
    const res = await fetch(apiUrl(`/api/optimizer/slates?date=${date}&slate_type=${slateType}`))
    if (!res.ok) throw new Error(`Failed to fetch slates: ${res.status}`)
    return res.json()
}

export async function getPlayerPool(
    date: string,
    draftGroupId: number,
    runId?: string | null,
    options?: { useStrategyOverrides?: boolean },
): Promise<PoolPlayer[]> {
    let url = apiUrl(`/api/optimizer/pool?date=${date}&draft_group_id=${draftGroupId}`)
    if (runId) url += `&run_id=${encodeURIComponent(runId)}`
    if (options?.useStrategyOverrides) url += '&use_strategy_overrides=true'
    const res = await fetch(url)
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Failed to fetch pool: ${res.status}`)
    }
    return res.json()
}

export async function startBuild(request: QuickBuildRequest): Promise<JobStatus> {
    const res = await fetch(apiUrl('/api/optimizer/build'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
    })
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        // Handle FastAPI validation errors which come as an array
        let errorMsg = `Failed to start build: ${res.status}`
        if (Array.isArray(body.detail)) {
            errorMsg = body.detail.map((e: { msg?: string; loc?: string[] }) =>
                `${e.loc?.slice(-1)[0] || 'field'}: ${e.msg || 'validation error'}`
            ).join('; ')
        } else if (typeof body.detail === 'string') {
            errorMsg = body.detail
        }
        throw new Error(errorMsg)
    }
    return res.json()
}

export async function getBuildStatus(jobId: string): Promise<JobStatus> {
    const res = await fetch(apiUrl(`/api/optimizer/build/${jobId}`))
    if (!res.ok) throw new Error(`Failed to get status: ${res.status}`)
    return res.json()
}

export async function getBuildLineups(jobId: string): Promise<LineupsResponse> {
    const res = await fetch(apiUrl(`/api/optimizer/build/${jobId}/lineups`))
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Failed to get lineups: ${res.status}`)
    }
    return res.json()
}

export async function exportLineupsCSV(jobId: string): Promise<Blob> {
    const res = await fetch(apiUrl(`/api/optimizer/build/${jobId}/export`))
    if (!res.ok) throw new Error(`Failed to export: ${res.status}`)
    return res.blob()
}

export async function exportCustomLineupsCSV(
    date: string,
    draftGroupId: number,
    lineups: string[][],
    filenamePrefix?: string,
): Promise<Blob> {
    const res = await fetch(apiUrl('/api/optimizer/export'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            date,
            draft_group_id: draftGroupId,
            site: 'dk',
            filename_prefix: filenamePrefix ?? null,
            lineups,
        }),
    })
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Failed to export: ${res.status}`)
    }
    return res.blob()
}

export async function getJobs(limit = 20): Promise<JobStatus[]> {
    const res = await fetch(apiUrl(`/api/optimizer/jobs?limit=${limit}`))
    if (!res.ok) throw new Error(`Failed to get jobs: ${res.status}`)
    return res.json()
}

// Saved Builds

export interface SavedBuild {
    job_id: string
    game_date: string
    draft_group_id: number
    site: string
    created_at: string
    completed_at?: string | null
    lineups_count: number
    config: Record<string, unknown>
    stats: Record<string, unknown>
    lineups?: LineupRow[]
}

export async function getSavedBuilds(date: string, draftGroupId?: number): Promise<SavedBuild[]> {
    let url = apiUrl(`/api/optimizer/saved-builds?date=${date}`)
    if (draftGroupId) url += `&draft_group_id=${draftGroupId}`
    const res = await fetch(url)
    if (!res.ok) throw new Error(`Failed to get saved builds: ${res.status}`)
    return res.json()
}

export async function loadSavedBuild(date: string, jobId: string): Promise<SavedBuild> {
    const res = await fetch(apiUrl(`/api/optimizer/saved-builds/${jobId}?date=${date}`))
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Failed to load build: ${res.status}`)
    }
    return res.json()
}

export async function deleteSavedBuild(date: string, jobId: string): Promise<void> {
    const res = await fetch(apiUrl(`/api/optimizer/saved-builds/${jobId}?date=${date}`), {
        method: 'DELETE',
    })
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Failed to delete build: ${res.status}`)
    }
}

export async function saveCustomBuild(
    date: string,
    draftGroupId: number,
    lineups: LineupRow[],
    name?: string,
    site = 'dk',
): Promise<SavedBuild> {
    const res = await fetch(apiUrl('/api/optimizer/saved-builds'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            date,
            draft_group_id: draftGroupId,
            site,
            name,
            lineups,
        }),
    })
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        // Handle FastAPI validation errors which come as an array
        let errorMsg = `Failed to save build: ${res.status}`
        if (Array.isArray(body.detail)) {
            errorMsg = body.detail.map((e: { msg?: string; loc?: string[] }) =>
                `${e.loc?.join('.') || 'field'}: ${e.msg || 'validation error'}`
            ).join('; ')
        } else if (typeof body.detail === 'string') {
            errorMsg = body.detail
        }
        throw new Error(errorMsg)
    }
    return res.json()
}

// ---------------------------------------------------------------------------
// Strategy Overrides
// ---------------------------------------------------------------------------

export interface PlayerStrategyOverride {
    player_id: string
    minutes_delta?: number | null
    fpts_delta?: number | null
    minutes?: number | null
    fpts?: number | null
    updated_at?: string | null
}

export interface SlateStrategyOverrides {
    game_date: string
    draft_group_id: number
    client_revision: number
    updated_at: string
    overrides: PlayerStrategyOverride[]
}

export async function getStrategyOverrides(
    date: string,
    draftGroupId: number,
): Promise<SlateStrategyOverrides> {
    const res = await fetch(
        apiUrl(`/api/optimizer/overrides?date=${date}&draft_group_id=${draftGroupId}`)
    )
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Failed to fetch overrides: ${res.status}`)
    }
    return res.json()
}

export async function saveStrategyOverrides(
    date: string,
    draftGroupId: number,
    overrides: PlayerStrategyOverride[],
    expectedRevision?: number,
): Promise<SlateStrategyOverrides> {
    const res = await fetch(apiUrl('/api/optimizer/overrides'), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            date,
            draft_group_id: draftGroupId,
            overrides,
            expected_revision: expectedRevision ?? null,
        }),
    })
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Failed to save overrides: ${res.status}`)
    }
    return res.json()
}

export async function clearStrategyOverrides(
    date: string,
    draftGroupId: number,
): Promise<{ status: string; count: number }> {
    const res = await fetch(
        apiUrl(`/api/optimizer/overrides?date=${date}&draft_group_id=${draftGroupId}`),
        { method: 'DELETE' }
    )
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Failed to clear overrides: ${res.status}`)
    }
    return res.json()
}

export async function getPlayerPoolWithStrategyOverrides(
    date: string,
    draftGroupId: number,
    runId?: string | null,
): Promise<PoolPlayer[]> {
    let url = apiUrl(
        `/api/optimizer/pool?date=${date}&draft_group_id=${draftGroupId}&use_strategy_overrides=true`
    )
    if (runId) url += `&run_id=${encodeURIComponent(runId)}`
    const res = await fetch(url)
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Failed to fetch pool with overrides: ${res.status}`)
    }
    return res.json()
}
