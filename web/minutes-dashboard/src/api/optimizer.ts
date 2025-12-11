/**
 * Optimizer API client for QuickBuild lineup generation.
 */
import { apiUrl } from './client'

// Types

export interface Slate {
    game_date: string
    slate_type: string
    draft_group_id: number
    n_contests: number
    earliest_start?: string
    latest_start?: string
    example_contest_name?: string
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
    jitter?: number
    near_dup_jaccard?: number
    enum_enable?: boolean
    min_salary?: number | null
    max_salary?: number | null
    global_team_limit?: number
    lock_ids?: string[]
    ban_ids?: string[]
    ownership_penalty_enabled?: boolean
    ownership_lambda?: number
    randomness_pct?: number | null
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
): Promise<PoolPlayer[]> {
    let url = apiUrl(`/api/optimizer/pool?date=${date}&draft_group_id=${draftGroupId}`)
    if (runId) url += `&run_id=${encodeURIComponent(runId)}`
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
        throw new Error(body.detail || `Failed to start build: ${res.status}`)
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
