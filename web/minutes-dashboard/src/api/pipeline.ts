import { apiUrl } from './client'

export type JobStatus = {
  job_name: string
  stage: string
  target_date: string
  run_ts: string
  status: string
  rows_written: number
  expected_rows?: number | null
  nan_rate_key_cols?: number | null
  schema_version?: string | null
  message?: string | null
}

export type PipelineJobSummary = {
  job_name: string
  stage: string
  target_date: string
  status: string
  run_ts: string
  rows_written: number
  expected_rows?: number | null
  nan_rate_key_cols?: number | null
  message?: string | null
}

export type PipelineSummary = {
  target_date: string
  slate_id?: string | null
  all_jobs_healthy: boolean
  jobs: PipelineJobSummary[]
}

export async function fetchPipelineStatus(params?: { target_date?: string; stage?: string }): Promise<JobStatus[]> {
  const searchParams = new URLSearchParams()
  if (params?.target_date) searchParams.set('target_date', params.target_date)
  if (params?.stage) searchParams.set('stage', params.stage)

  const query = searchParams.toString()
  const url = query ? apiUrl(`/api/pipeline/status?${query}`) : apiUrl('/api/pipeline/status')
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`Failed to fetch pipeline status: ${res.status}`)
  }
  return res.json()
}

export async function fetchPipelineSummary(params?: { target_date?: string }): Promise<PipelineSummary> {
  const searchParams = new URLSearchParams()
  if (params?.target_date) searchParams.set('target_date', params.target_date)
  const query = searchParams.toString()
  const url = query ? apiUrl(`/api/pipeline/summary?${query}`) : apiUrl('/api/pipeline/summary')
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`Failed to fetch pipeline summary: ${res.status}`)
  }
  return res.json()
}
