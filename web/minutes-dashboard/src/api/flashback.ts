const API_BASE = '/api/flashback'

export interface FlashbackContestSummary {
  game_date: string
  contest_id: string
  contest_name: string
  draft_group_id?: number | null
  entry_fee?: number | null
  entry_count: number
  best_rank?: number | null
  best_prize?: number | null
  candidate_manifest_available: boolean
}

export interface FlashbackRunRequest {
  game_date: string
  contest_id: string
  user_pattern: string
  draft_group_id?: number | null
  run_id?: string | null
  entry_fee?: number | null
  archetype?: string
  worlds_source?: string
  ownership_mode?: string
  modeled_field_version?: string
  include_modeled_field?: boolean
}

export interface FlashbackRunResponse {
  summary: Record<string, unknown>
  previews: Record<string, Array<Record<string, unknown>>>
}

export interface FlashbackCalibrationResponse {
  summary: Record<string, unknown>
  previews: Record<string, Array<Record<string, unknown>>>
}

export async function listFlashbackContests(date: string, userPattern: string): Promise<FlashbackContestSummary[]> {
  const params = new URLSearchParams({ date, user_pattern: userPattern })
  const resp = await fetch(`${API_BASE}/contests?${params.toString()}`)
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }))
    throw new Error(err.detail || 'Failed to load flashback contests')
  }
  return resp.json()
}

export async function runFlashback(request: FlashbackRunRequest): Promise<FlashbackRunResponse> {
  const resp = await fetch(`${API_BASE}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }))
    throw new Error(err.detail || 'Failed to run flashback')
  }
  return resp.json()
}

export async function runFlashbackCalibration(): Promise<FlashbackCalibrationResponse> {
  const resp = await fetch(`${API_BASE}/calibration/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  })
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }))
    throw new Error(err.detail || 'Failed to run flashback calibration')
  }
  return resp.json()
}
