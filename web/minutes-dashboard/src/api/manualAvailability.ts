import { apiUrl } from './client'

export type ManualAvailabilityOverride = {
  override_id: string
  game_date: string
  game_id: string
  player_id: string
  player_name?: string | null
  team_id?: number | null
  team_tricode?: string | null
  override_type: 'force_out' | 'force_in'
  reason_code?: string | null
  reason_text?: string | null
  source_label?: string | null
  entered_by: string
  created_ts?: string | null
  effective_ts?: string | null
  expires_ts?: string | null
  active: boolean
  cleared_ts?: string | null
  cleared_by?: string | null
}

export type OpsGamePlayer = {
  player_id: string
  player_name?: string | null
  team_id?: number | string | null
  team_tricode?: string | null
  status?: string | null
  is_confirmed_starter?: boolean | null
  is_projected_starter?: boolean | null
  minutes_effective: {
    status?: string | null
    play_prob?: number | null
    minutes_final?: number | null
    minutes_p10?: number | null
    minutes_p50?: number | null
    minutes_p90?: number | null
    manual_override_id?: string | null
    manual_override_type?: 'force_out' | 'force_in' | null
    manual_override_reason_code?: string | null
    manual_override_reason_text?: string | null
    manual_override_source_label?: string | null
    manual_override_entered_by?: string | null
    manual_override_created_ts?: string | null
    manual_override_effective_ts?: string | null
    manual_override_expires_ts?: string | null
    manual_override_active?: boolean | null
    manual_override_used?: boolean | null
  }
  manual_override?: ManualAvailabilityOverride | null
}

export type OpsGameResponse = {
  date: string
  game_id: string
  players: OpsGamePlayer[]
}

export type UpsertManualAvailabilityPayload = {
  date: string
  game_id: string
  player_id: string
  player_name?: string | null
  team_id?: number | null
  team_tricode?: string | null
  override_type: 'force_out' | 'force_in'
  entered_by: string
  reason_code?: string | null
  reason_text?: string | null
  source_label?: string | null
}

const parseError = async (res: Response, fallback: string) => {
  const body = await res.json().catch(() => ({}))
  return (body as { detail?: string }).detail || fallback
}

export async function fetchOpsGame(date: string, gameId: string): Promise<OpsGameResponse> {
  const res = await fetch(
    apiUrl(`/api/ops/game?date=${encodeURIComponent(date)}&game_id=${encodeURIComponent(gameId)}`),
  )
  if (!res.ok) {
    throw new Error(await parseError(res, `Failed to fetch game view: ${res.status}`))
  }
  return (await res.json()) as OpsGameResponse
}

export async function upsertManualAvailabilityOverride(
  payload: UpsertManualAvailabilityPayload,
): Promise<ManualAvailabilityOverride> {
  const res = await fetch(apiUrl('/api/ops/manual-availability-overrides'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    throw new Error(await parseError(res, `Failed to save manual availability override: ${res.status}`))
  }
  const body = (await res.json()) as { override: ManualAvailabilityOverride }
  return body.override
}

export async function clearManualAvailabilityOverride(params: {
  date: string
  overrideId: string
  clearedBy: string
}): Promise<void> {
  const query = new URLSearchParams({
    date: params.date,
    cleared_by: params.clearedBy,
  }).toString()
  const res = await fetch(
    apiUrl(`/api/ops/manual-availability-overrides/${encodeURIComponent(params.overrideId)}?${query}`),
    { method: 'DELETE' },
  )
  if (!res.ok) {
    throw new Error(await parseError(res, `Failed to clear manual availability override: ${res.status}`))
  }
}
