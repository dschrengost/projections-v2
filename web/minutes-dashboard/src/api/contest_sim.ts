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
}

export interface ContestSimResponse {
    results: LineupEVResult[]
    config: ContestConfig
    stats: SummaryStats
}

export interface ContestSimRequest {
    game_date: string
    lineups: string[][]
    archetype?: string
    field_size_bucket?: string
    field_size_override?: number
    entry_fee?: number
    weights?: number[]
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
