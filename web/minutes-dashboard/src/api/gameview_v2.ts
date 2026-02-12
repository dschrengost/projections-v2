import { apiUrl } from './client'
import { PlayerRow } from '../types'

export type OverrideMode =
    | 'none'
    | 'lock'
    | 'band'
    | 'cap'
    | 'floor'
    | 'zero'
    | 'force_active'
    | 'force_inactive'

export type ProjectionStats = {
    minutes?: number | null
    fpts?: number | null
    pts?: number | null
    reb?: number | null
    ast?: number | null
    stl?: number | null
    blk?: number | null
    to?: number | null
}

export type ProjectionQuantiles = {
    minutes_p10?: number | null
    minutes_p50?: number | null
    minutes_p90?: number | null
    fpts_p10?: number | null
    fpts_p50?: number | null
    fpts_p90?: number | null
}

export type PlayerOverrideState = {
    mode: OverrideMode
    lock_value?: number | null
    min_value?: number | null
    max_value?: number | null
    cap_value?: number | null
    floor_value?: number | null
    protect_weight?: boolean
}

export type GameviewPlayerRow = {
    game_id: string
    team_id: string
    player_id: string
    name: string
    pos?: string
    status?: string
    projections: {
        baseline: ProjectionStats
        resolved: ProjectionStats
        quantiles?: ProjectionQuantiles
    }
    override: PlayerOverrideState
}

export type TeamDiagnostics = {
    game_id: string
    team_id: string
    sum_lb: number
    sum_ub: number
    sum_mu: number
    locked_minutes_total: number
    infeasibility_reason?: string | null
    infeasible_action?: string | null
    hit_floor_player_ids?: string[]
    hit_cap_player_ids?: string[]
    n_players?: number
    n_overrides?: number
}

export type TeamContract = {
    team_id: string
    team_name: string
    players: GameviewPlayerRow[]
    diagnostics?: TeamDiagnostics | null
}

export type GameContract = {
    game_id: string
    start_time?: string
    away_team: TeamContract
    home_team: TeamContract
}

export type SlateContract = {
    slate_date: string
    games: GameContract[]
}

export type MinutesApiResponse = {
    date: string
    count: number
    players: PlayerRow[]
}

export type V2OverrideStateItem = {
    game_id: string
    player_id: string
    mode: OverrideMode
    fields: Record<string, unknown>
    legacy_fields_present: string[]
    updated_at?: string
}

export type V2OverrideStateResponse = {
    date: string
    game_id?: string | null
    overrides: V2OverrideStateItem[]
}

export type ApplyOverridesResponse = {
    date: string
    game_id: string
    applied_at: string
    override_infeasible: 'error' | 'relax' | 'ignore'
    run_context?: {
        projections_run_id?: string | null
    }
    resolved_players: Array<{
        game_id: string
        team_id: string
        player_id: string
        b_minutes: number
        mu_minutes: number
        lb_minutes: number
        ub_minutes: number
        eligible: boolean
        force_active: boolean
        force_inactive: boolean
        weight: number
        constraint_kind: string
        override_present: boolean
    }>
    team_diagnostics: TeamDiagnostics[]
    diag?: {
        team_diagnostics?: TeamDiagnostics[]
    }
    overrides: V2OverrideStateItem[]
}

export type RunWorldsResponse = {
    status: 'triggered' | 'success'
    date: string
    game_id: number
    run_ts?: string
    projections_run_id?: string
    sim_run_id?: string
    message?: string
}

export type PipelineJobStatus = {
    job_name: string
    stage: string
    target_date: string
    run_ts: string
    status: 'running' | 'success' | 'error'
    message?: string | null
}

export type PollRunResult = {
    done: boolean
    ok: boolean
    status?: PipelineJobStatus
    projections_run_id?: string | null
}

export async function fetchSlate(date: string, runId?: string | null): Promise<{ slate: SlateContract; rows: PlayerRow[] }> {
    const runParam = runId ? `&run_id=${encodeURIComponent(runId)}` : ''
    const res = await fetch(apiUrl(`/api/minutes?date=${date}${runParam}`))
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error((body as { detail?: string }).detail || `Failed to fetch minutes: ${res.status}`)
    }
    const payload = (await res.json()) as MinutesApiResponse
    return {
        slate: {
            slate_date: payload.date,
            games: [],
        },
        rows: payload.players ?? [],
    }
}

export async function fetchGameProjections(date: string, runId?: string | null): Promise<PlayerRow[]> {
    const runParam = runId ? `&run_id=${encodeURIComponent(runId)}` : ''
    const res = await fetch(apiUrl(`/api/minutes?date=${date}${runParam}`))
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error((body as { detail?: string }).detail || `Failed to fetch game projections: ${res.status}`)
    }
    const payload = (await res.json()) as MinutesApiResponse
    return payload.players ?? []
}

export async function fetchOverrideState(date: string, gameId: string): Promise<V2OverrideStateResponse> {
    const res = await fetch(apiUrl(`/api/ops/overrides-v2?date=${encodeURIComponent(date)}&game_id=${encodeURIComponent(gameId)}`))
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error((body as { detail?: string }).detail || `Failed to fetch v2 overrides: ${res.status}`)
    }
    return (await res.json()) as V2OverrideStateResponse
}

export async function applyOverrides(params: {
    date: string
    gameId: string
    runId?: string | null
    overrideInfeasible?: 'error' | 'relax' | 'ignore'
    overrides: Array<{ player_id: string } & PlayerOverrideState>
}): Promise<ApplyOverridesResponse> {
    const res = await fetch(apiUrl('/api/ops/overrides-v2/apply'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            date: params.date,
            game_id: params.gameId,
            run_id: params.runId ?? null,
            override_infeasible: params.overrideInfeasible ?? 'error',
            overrides: params.overrides,
        }),
    })
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error((body as { detail?: string }).detail || `Failed to apply v2 overrides: ${res.status}`)
    }
    return (await res.json()) as ApplyOverridesResponse
}

export async function runWorldsWithOverrides(params: {
    date: string
    gameId: string
    pin?: boolean
    background?: boolean
    baseRunId?: string | null
    minutesOverrideMode?: 'legacy' | 'v2'
    overrideInfeasible?: 'error' | 'relax' | 'ignore'
}): Promise<RunWorldsResponse> {
    const res = await fetch(apiUrl('/api/ops/run-worlds'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            date: params.date,
            game_id: Number(params.gameId),
            base_run_id: params.baseRunId ?? null,
            pin: params.pin ?? true,
            background: params.background ?? true,
            minutes_override_mode: params.minutesOverrideMode ?? 'v2',
            override_infeasible: params.overrideInfeasible ?? 'error',
        }),
    })
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error((body as { detail?: string }).detail || `Failed to run worlds: ${res.status}`)
    }
    return (await res.json()) as RunWorldsResponse
}

export async function pollRun(date: string, runTs: string): Promise<PollRunResult> {
    const res = await fetch(apiUrl(`/api/pipeline/status?target_date=${encodeURIComponent(date)}&stage=ops`))
    if (!res.ok) {
        throw new Error(`Failed to poll run status: ${res.status}`)
    }
    const all = (await res.json()) as PipelineJobStatus[]
    const status = all.find(
        (item) =>
            item.job_name === 'ops_patch_worlds_matrix_game' &&
            item.target_date === date &&
            item.run_ts === runTs,
    )
    if (!status) {
        return { done: false, ok: false }
    }
    if (status.status === 'running') {
        return { done: false, ok: false, status }
    }

    let projectionsRunId: string | null = null
    const msg = status.message || ''
    const m = msg.match(/projections_run_id=([0-9TZ]+)/)
    if (m) projectionsRunId = m[1]

    return {
        done: true,
        ok: status.status === 'success',
        status,
        projections_run_id: projectionsRunId,
    }
}

export async function fetchArtifacts(runId: string | null | undefined, date: string): Promise<PlayerRow[]> {
    return fetchGameProjections(date, runId ?? null)
}
