import { apiUrl } from './client'

export type LiveSourceFreshness = {
  source_used?: string | null
  latest_as_of_ts?: string | null
  content_digest?: string | null
}

export type LiveGameStatus = {
  game_id: string
  tip_ts?: string | null
  minutes_to_tip?: number | null
  source_freshness: Record<string, LiveSourceFreshness>
  freshness_gates: {
    lock_window_ok: boolean
    report_window_active: boolean
    report_window_label?: string | null
    report_window_blocking?: boolean
  }
  affected_by_change_set: boolean
  rerun_targeted: boolean
  manual_override_active: boolean
  manual_override_count?: number
  changed_sources: string[]
  latest_effective_status_source_summary?: string
  warning_badges: string[]
  status_source_run_id?: string | null
  status_source_label?: 'published' | 'candidate' | string
}

export type LiveRunEvent = {
  run_id: string
  status: string
  reason?: string | null
  as_of_ts?: string | null
  updated_at?: string | null
  target_game_ids?: Array<string | number>
}

export type LiveStatusResponse = {
  game_date: string
  latest_published_run_id?: string | null
  latest_published_as_of_ts?: string | null
  candidate_run_id?: string | null
  candidate_status: string
  candidate_status_reason?: string | null
  publish_status?: string | null
  updated_at?: string | null
  status_source_run_id?: string | null
  games: LiveGameStatus[]
  run_event_strip: LiveRunEvent[]
}

export async function fetchLiveStatus(date: string): Promise<LiveStatusResponse> {
  const res = await fetch(apiUrl(`/api/live/status?date=${encodeURIComponent(date)}`))
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error((body as { detail?: string }).detail || `Failed to fetch live status: ${res.status}`)
  }
  return (await res.json()) as LiveStatusResponse
}
