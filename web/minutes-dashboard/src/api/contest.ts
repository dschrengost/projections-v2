import { apiUrl } from './client'

// Types matching backend Pydantic models

export type OwnershipMetrics = {
  total_own: number
  avg_own: number
  min_own: number
  max_own: number
  num_under_10: number
  num_under_5: number
  num_over_50: number
}

export type PlayerOwnership = {
  player: string
  ownership_pct: number
  fpts: number | null
  win_pct: number | null
  top10_pct: number | null
}

export type LeveragePlay = {
  player: string
  ownership_pct: number
  win_rate: number
  leverage_score: number
}

export type PlayerPair = {
  player1: string
  player2: string
  count: number
  pct: number
}

export type StackInfo = {
  team: string
  stack_size: number
  count: number
  pct_of_top: number
}

export type GameCorrelation = {
  matchup: string
  players_in_winners: number
  total_fpts: number
  avg_fpts: number
}

export type ContestSummary = {
  contest_id: string
  contest_name: string
  entry_fee: number | null
  total_entries: number
  contest_type: string
}

export type LineupOwnershipEntry = {
  rank: number
  entry_name: string
  points: number
  total_own: number  // Sum of all 8 players' ownership
  avg_own: number
  num_under_10: number
  num_under_5: number
  lineup_players: string[]
}

export type OwnershipDistribution = {
  bin_label: string  // e.g., "100-110%"
  count: number
  pct_of_total: number
}

export type DuplicateLineup = {
  lineup_key: string
  lineup_players: string[]
  entry_count: number
  entry_names: string[]
  ranks: number[]
  best_rank: number
  total_own: number
  avg_own: number
}

export type DupesByOwnershipBand = {
  ownership_band: string
  total_lineups: number
  unique_lineups: number
  duped_lineups: number
  dupe_rate: number
}

export type DupeAnalysis = {
  total_entries: number
  unique_lineups: number
  duplicate_entries: number
  duplicate_rate: number
  most_duped_lineup: DuplicateLineup | null
  top_duped_lineups: DuplicateLineup[]
  dupes_by_ownership: DupesByOwnershipBand[]
}

export type ContestDetail = {
  contest_id: string
  contest_name: string
  total_entries: number
  winner: OwnershipMetrics | null
  top10: OwnershipMetrics | null
  top1pct: OwnershipMetrics | null
  cash_line: OwnershipMetrics | null
  field_avg: OwnershipMetrics | null
  winner_score: number | null
  top1pct_score: number | null
  cash_line_score: number | null
  leverage_plays: LeveragePlay[]
  stacks: StackInfo[]
  game_correlations: GameCorrelation[]
  player_ownership: PlayerOwnership[]
  top_lineups: LineupOwnershipEntry[]  // Top 100 lineups
  ownership_distribution: OwnershipDistribution[]  // Histogram
  dupe_analysis: DupeAnalysis
}

export type UserLineupEntry = {
  contest_id: string
  contest_name: string
  entry_name: string
  rank: number
  points: number
  total_entries: number
  pct_finish: number
  lineup_players: string[]
  ownership_metrics: OwnershipMetrics
}

export type ContestDatesResponse = {
  dates: string[]
}

export type ContestListResponse = {
  date: string
  contests: ContestSummary[]
  total_contests: number
}

export type PairsResponse = {
  contest_id: string
  top_n_analyzed: number
  pairs: PlayerPair[]
}

export type UserLineupsResponse = {
  date: string
  pattern_used: string
  entries_found: number
  entries: UserLineupEntry[]
  avg_finish_pct: number | null
  best_finish_rank: number | null
  best_finish_pct: number | null
}

// API functions

export async function getContestDates(limit?: number): Promise<string[]> {
  const params = new URLSearchParams()
  if (limit) params.set('limit', limit.toString())
  const query = params.toString()
  const url = query ? apiUrl(`/api/contest/dates?${query}`) : apiUrl('/api/contest/dates')
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`Failed to fetch contest dates: ${res.status}`)
  }
  const data: ContestDatesResponse = await res.json()
  return data.dates
}

export async function getContests(date: string): Promise<ContestSummary[]> {
  const url = apiUrl(`/api/contest/contests?date=${encodeURIComponent(date)}`)
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`Failed to fetch contests: ${res.status}`)
  }
  const data: ContestListResponse = await res.json()
  return data.contests
}

export async function getContestDetail(contestId: string, date: string): Promise<ContestDetail> {
  const url = apiUrl(`/api/contest/contests/${encodeURIComponent(contestId)}?date=${encodeURIComponent(date)}`)
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`Failed to fetch contest detail: ${res.status}`)
  }
  return res.json()
}

export async function getPairsAnalysis(contestId: string, date: string, topN?: number): Promise<PairsResponse> {
  const params = new URLSearchParams()
  params.set('date', date)
  if (topN) params.set('top_n', topN.toString())
  const url = apiUrl(`/api/contest/contests/${encodeURIComponent(contestId)}/pairs?${params.toString()}`)
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`Failed to fetch pairs analysis: ${res.status}`)
  }
  return res.json()
}

export async function getUserLineups(date: string, pattern: string): Promise<UserLineupsResponse> {
  const params = new URLSearchParams()
  params.set('date', date)
  params.set('pattern', pattern)
  const url = apiUrl(`/api/contest/user-lineups?${params.toString()}`)
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`Failed to fetch user lineups: ${res.status}`)
  }
  return res.json()
}
