import { apiUrl } from './client'

export type FptsMiss = {
  player_id: string
  player_name?: string
  pred_fpts: number
  actual_fpts: number
  error: number
  pred_mins: number
  actual_mins: number
}

export type EvaluationDayMetrics = {
  date: string
  players_matched: number
  total_players_actual?: number
  total_players_pred?: number
  // FPTS metrics
  fpts_mae: number | null
  minutes_mae: number | null
  coverage_80: number | null
  coverage_90: number | null
  bias: number | null
  missed: number
  false_preds: number
  // Minutes by bucket
  mins_mae_starter_30plus?: number | null
  n_starter_30plus?: number
  mins_mae_heavy_20_30?: number | null
  n_heavy_20_30?: number
  mins_mae_rotation_10_20?: number | null
  n_rotation_10_20?: number
  mins_mae_light_5_10?: number | null
  n_light_5_10?: number
  // Top FPTS misses
  top_fpts_misses?: FptsMiss[]
  // Ownership metrics (optional - may not be available for all dates)
  own_players_matched?: number | null
  own_mae?: number | null
  own_smape?: number | null
  own_corr?: number | null
  chalk_top5_acc?: number | null
  own_bias?: number | null
  high_own_mae?: number | null
  own_top_misses?: Array<{ player: string; actual: number; pred: number; error: number }>
}

export type EvaluationSummary = {
  // FPTS aggregates
  avg_fpts_mae: number | null
  avg_minutes_mae: number | null
  avg_coverage_80: number | null
  avg_coverage_90: number | null
  avg_coverage_50?: number | null
  avg_bias: number | null
  total_missed: number
  total_false_preds: number

  // Calibration gaps
  avg_cal_gap_50?: number | null
  avg_cal_gap_80?: number | null
  avg_cal_gap_90?: number | null

  // Salary tier accuracy
  avg_fpts_mae_elite?: number | null
  avg_fpts_mae_mid?: number | null
  avg_fpts_mae_value?: number | null
  avg_fpts_mae_punt?: number | null

  // Starter/bench splits
  avg_fpts_mae_starters?: number | null
  avg_fpts_mae_bench?: number | null

  // Edge cases
  total_dnp_false_positives?: number
  total_starter_misses?: number
  total_blowup_misses?: number

  // Ownership aggregates
  avg_own_mae?: number | null
  avg_own_smape?: number | null
  avg_own_corr?: number | null
  avg_chalk_top5_acc?: number | null
  avg_own_bias?: number | null

  // Counts
  dates_evaluated: number
  total_players_matched: number
}

export type EvaluationResponse = {
  days: number
  end_date: string | null
  metrics: EvaluationDayMetrics[]
  summary: EvaluationSummary
}

export async function fetchEvaluation(params?: { days?: number }): Promise<EvaluationResponse> {
  const searchParams = new URLSearchParams()
  if (params?.days) searchParams.set('days', params.days.toString())

  const query = searchParams.toString()
  const url = query ? apiUrl(`/api/evaluation?${query}`) : apiUrl('/api/evaluation')
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`Failed to fetch evaluation: ${res.status}`)
  }
  return res.json()
}
