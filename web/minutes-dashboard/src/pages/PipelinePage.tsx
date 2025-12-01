import { useCallback, useEffect, useMemo, useState } from 'react'
import { fetchPipelineSummary, type PipelineJobSummary, type PipelineSummary } from '../api/pipeline'

const todayLocalISO = () => {
  const now = new Date()
  const tzOffsetMs = now.getTimezoneOffset() * 60 * 1000
  const local = new Date(now.getTime() - tzOffsetMs)
  return local.toISOString().slice(0, 10)
}

const formatDateTime = (value: string | null | undefined) => {
  if (!value) return 'N/A'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    return value
  }
  return parsed.toLocaleString()
}

export function PipelinePage() {
  const [targetDate, setTargetDate] = useState(() => todayLocalISO())
  const [summary, setSummary] = useState<PipelineSummary | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadStatuses = useCallback(
    async (date: string) => {
      setLoading(true)
      setError(null)
      try {
        const data = await fetchPipelineSummary({ target_date: date })
        setSummary(data)
      } catch (err) {
        setError((err as Error).message)
        setSummary(null)
      } finally {
        setLoading(false)
      }
    },
    [],
  )

  useEffect(() => {
    void loadStatuses(targetDate)
  }, [loadStatuses, targetDate])

  const summaryStats = useMemo(() => {
    const jobs = summary?.jobs ?? []
    const total = jobs.length
    const errors = jobs.filter((item) => item.status !== 'success').length
    const lastUpdated = jobs.reduce<string | null>((latest, current) => {
      const ts = new Date(current.run_ts)
      if (Number.isNaN(ts.getTime())) return latest
      if (!latest) return current.run_ts
      return new Date(latest) < ts ? current.run_ts : latest
    }, null)
    return { total, errors, lastUpdated }
  }, [summary])

  const rowClass = (item: PipelineJobSummary) => {
    if (item.status !== 'success') return 'pipeline-row-error'
    const nanFlag = item.nan_rate_key_cols !== null && item.nan_rate_key_cols !== undefined && item.nan_rate_key_cols > 0
    const rowGap = item.expected_rows !== null && item.expected_rows !== undefined && item.rows_written < 0.8 * item.expected_rows
    if (nanFlag || rowGap) return 'pipeline-row-warn'
    return ''
  }

  const failingJobs = useMemo(
    () => (summary?.jobs ?? []).filter((job) => job.status !== 'success'),
    [summary],
  )

  return (
    <div className="pipeline-page">
      <div className="pipeline-header">
        <div>
          <h1>Pipeline Health</h1>
          <p className="subtitle">Live job outputs for the selected slate date.</p>
        </div>
        <label className="pipeline-date">
          Date
          <input
            type="date"
            value={targetDate}
            onChange={(event) => setTargetDate(event.target.value)}
          />
        </label>
      </div>

      {summary && (
        <div className={`pipeline-banner ${summary.all_jobs_healthy ? 'success' : 'warning'}`}>
          <div>
            {summary.all_jobs_healthy
              ? `Pipeline healthy for ${summary.target_date}`
              : `Pipeline NOT healthy for ${summary.target_date}`}
          </div>
          {!summary.all_jobs_healthy && failingJobs.length > 0 && (
            <div className="pipeline-banner-failures">
              Issues:{' '}
              {failingJobs.map((job) => `${job.job_name} (${job.status})`).join(', ')}
            </div>
          )}
        </div>
      )}

      <div className="pipeline-summary">
        <div className="card">
          <div className="label">Jobs</div>
          <div className="value">{loading ? '…' : summaryStats.total}</div>
        </div>
        <div className="card">
          <div className="label">Errors</div>
          <div className={`value ${summaryStats.errors > 0 ? 'text-error' : ''}`}>
            {loading ? '…' : summaryStats.errors}
          </div>
        </div>
        <div className="card">
          <div className="label">Last updated</div>
          <div className="value">{loading ? '…' : formatDateTime(summaryStats.lastUpdated)}</div>
        </div>
      </div>

      {error && <div className="pipeline-alert error">Failed to load status: {error}</div>}
      {!error && !loading && (summary?.jobs.length ?? 0) === 0 && (
        <div className="pipeline-alert muted">No status files found for {targetDate}.</div>
      )}

      <div className="table-wrapper pipeline-table">
        <table>
          <thead>
            <tr>
              <th>Job</th>
              <th>Stage</th>
              <th>Status</th>
              <th>Run time</th>
              <th>Rows written</th>
              <th>Expected rows</th>
              <th>NaN rate (key cols)</th>
              <th>Message</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={8}>Loading…</td>
              </tr>
            )}
            {!loading &&
              (summary?.jobs ?? []).map((item) => {
                const warnNan = item.nan_rate_key_cols !== null && item.nan_rate_key_cols !== undefined && item.nan_rate_key_cols > 0
                const warnRows =
                  item.expected_rows !== null &&
                  item.expected_rows !== undefined &&
                  item.rows_written < 0.8 * item.expected_rows
                return (
                  <tr key={`${item.job_name}-${item.target_date}-${item.run_ts}`} className={rowClass(item)}>
                    <td>
                      <div className="job-name">{item.job_name}</div>
                      <div className="muted">{item.target_date}</div>
                    </td>
                    <td>{item.stage}</td>
                    <td className={item.status !== 'success' ? 'text-error' : 'text-success'}>
                      {item.status}
                    </td>
                    <td>{formatDateTime(item.run_ts)}</td>
                    <td>{item.rows_written}</td>
                    <td className={warnRows ? 'text-warn' : ''}>
                      {item.expected_rows ?? '—'}
                    </td>
                    <td className={warnNan ? 'text-warn' : ''}>
                      {item.nan_rate_key_cols ?? '—'}
                    </td>
                    <td className="muted">{item.message ?? '—'}</td>
                  </tr>
                )
              })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default PipelinePage
