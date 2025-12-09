import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  CartesianGrid,
} from 'recharts'
import {
  fetchEvaluation,
  type EvaluationDayMetrics,
  type EvaluationResponse,
} from '../api/evaluation'

// Color thresholds for metrics
const getMetricClass = (
  value: number | null | undefined,
  thresholds: { good: number; warn: number },
  higherIsBetter: boolean = false
): string => {
  if (value === null || value === undefined) return ''
  if (higherIsBetter) {
    if (value >= thresholds.good) return 'metric-good'
    if (value >= thresholds.warn) return 'metric-warn'
    return 'metric-bad'
  } else {
    if (value <= thresholds.good) return 'metric-good'
    if (value <= thresholds.warn) return 'metric-warn'
    return 'metric-bad'
  }
}

const formatPercent = (value: number | null | undefined): string => {
  if (value === null || value === undefined) return '—'
  return `${(value * 100).toFixed(1)}%`
}

const formatNumber = (value: number | null | undefined, decimals: number = 2): string => {
  if (value === null || value === undefined) return '—'
  return value.toFixed(decimals)
}

const formatDateShort = (dateStr: string): string => {
  return dateStr.slice(5) // MM-DD
}

export function EvaluationPage() {
  const [days, setDays] = useState(7)
  const [data, setData] = useState<EvaluationResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadData = useCallback(async (numDays: number) => {
    setLoading(true)
    setError(null)
    try {
      const result = await fetchEvaluation({ days: numDays })
      setData(result)
    } catch (err) {
      setError((err as Error).message)
      setData(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadData(days)
  }, [loadData, days])

  const summary = data?.summary
  const metrics = data?.metrics ?? []

  // Chart data - ensure ascending date order
  const chartData = useMemo(() => {
    return [...metrics].sort((a, b) => a.date.localeCompare(b.date))
  }, [metrics])

  // Has ownership data?
  const hasOwnership = useMemo(() => {
    return metrics.some((m) => m.own_mae !== null && m.own_mae !== undefined)
  }, [metrics])

  return (
    <div className="evaluation-page">
      <div className="evaluation-header">
        <div>
          <h1>Prediction Accuracy</h1>
          <p className="subtitle">Rolling evaluation of FPTS and ownership predictions.</p>
        </div>
        <label className="evaluation-days">
          Days
          <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
            <option value={7}>7</option>
            <option value={14}>14</option>
            <option value={30}>30</option>
          </select>
        </label>
      </div>

      {error && <div className="evaluation-alert error">Failed to load: {error}</div>}

      {/* Summary Cards */}
      <div className="evaluation-summary">
        <div className="card">
          <div className="label">FPTS MAE</div>
          <div className={`value ${getMetricClass(summary?.avg_fpts_mae, { good: 7, warn: 9 })}`}>
            {loading ? '…' : formatNumber(summary?.avg_fpts_mae)}
          </div>
        </div>
        <div className="card">
          <div className="label">80% Coverage</div>
          <div
            className={`value ${getMetricClass(summary?.avg_coverage_80, { good: 0.65, warn: 0.55 }, true)}`}
          >
            {loading ? '…' : formatPercent(summary?.avg_coverage_80)}
          </div>
        </div>
        <div className="card">
          <div className="label">Minutes MAE</div>
          <div className={`value ${getMetricClass(summary?.avg_minutes_mae, { good: 4.5, warn: 6 })}`}>
            {loading ? '…' : formatNumber(summary?.avg_minutes_mae)}
          </div>
        </div>
        <div className="card">
          <div className="label">Bias</div>
          <div
            className={`value ${getMetricClass(Math.abs(summary?.avg_bias ?? 0), { good: 1, warn: 2 })}`}
          >
            {loading ? '…' : formatNumber(summary?.avg_bias)}
          </div>
        </div>
        {hasOwnership && (
          <>
            <div className="card">
              <div className="label">Own Corr</div>
              <div
                className={`value ${getMetricClass(summary?.avg_own_corr, { good: 0.8, warn: 0.65 }, true)}`}
              >
                {loading ? '…' : formatNumber(summary?.avg_own_corr, 3)}
              </div>
            </div>
            <div className="card">
              <div className="label">Chalk Top5</div>
              <div
                className={`value ${getMetricClass(summary?.avg_chalk_top5_acc, { good: 0.6, warn: 0.4 }, true)}`}
              >
                {loading ? '…' : formatPercent(summary?.avg_chalk_top5_acc)}
              </div>
            </div>
          </>
        )}
      </div>

      {/* FPTS Accuracy Trend */}
      {!loading && chartData.length > 0 && (
        <div className="evaluation-section">
          <h2>FPTS Accuracy Trend</h2>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis
                  dataKey="date"
                  tickFormatter={formatDateShort}
                  stroke="#94a3b8"
                  fontSize={12}
                />
                <YAxis yAxisId="left" stroke="#94a3b8" fontSize={12} domain={[0, 'auto']} />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  stroke="#94a3b8"
                  fontSize={12}
                  domain={[0, 1]}
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  labelStyle={{ color: '#f8fafc' }}
                  formatter={(value: number, name: string) => {
                    if (name.includes('Coverage')) return `${(value * 100).toFixed(1)}%`
                    return value?.toFixed(2)
                  }}
                />
                <Legend />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="fpts_mae"
                  stroke="#4f46e5"
                  strokeWidth={2}
                  name="FPTS MAE"
                  dot={{ fill: '#4f46e5' }}
                  connectNulls
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="coverage_80"
                  stroke="#14b8a6"
                  strokeWidth={2}
                  name="Coverage 80%"
                  dot={{ fill: '#14b8a6' }}
                  connectNulls
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Ownership Accuracy Trend */}
      {!loading && hasOwnership && chartData.length > 0 && (
        <div className="evaluation-section">
          <h2>Ownership Accuracy Trend</h2>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis
                  dataKey="date"
                  tickFormatter={formatDateShort}
                  stroke="#94a3b8"
                  fontSize={12}
                />
                <YAxis yAxisId="left" stroke="#94a3b8" fontSize={12} domain={[0, 'auto']} />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  stroke="#94a3b8"
                  fontSize={12}
                  domain={[-1, 1]}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  labelStyle={{ color: '#f8fafc' }}
                  formatter={(value: number, name: string) => {
                    if (name.includes('Corr')) return value?.toFixed(3)
                    return value?.toFixed(2)
                  }}
                />
                <Legend />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="own_mae"
                  stroke="#a78bfa"
                  strokeWidth={2}
                  name="Own MAE"
                  dot={{ fill: '#a78bfa' }}
                  connectNulls
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="own_corr"
                  stroke="#10b981"
                  strokeWidth={2}
                  name="Own Corr"
                  dot={{ fill: '#10b981' }}
                  connectNulls
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Roster Accuracy (Missed / False Preds) */}
      {!loading && chartData.length > 0 && (
        <div className="evaluation-section">
          <h2>Roster Accuracy</h2>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis
                  dataKey="date"
                  tickFormatter={formatDateShort}
                  stroke="#94a3b8"
                  fontSize={12}
                />
                <YAxis stroke="#94a3b8" fontSize={12} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  labelStyle={{ color: '#f8fafc' }}
                />
                <Legend />
                <Bar dataKey="missed" fill="#f87171" name="Missed Players" />
                <Bar dataKey="false_preds" fill="#fbbf24" name="False Predictions" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Details Table */}
      {!loading && metrics.length > 0 && (
        <div className="evaluation-section">
          <h2>Daily Details</h2>
          <div className="table-wrapper evaluation-table">
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Players</th>
                  <th>FPTS MAE</th>
                  <th>Min MAE</th>
                  <th>Cov 80%</th>
                  <th>Cov 90%</th>
                  <th>Bias</th>
                  <th>Missed</th>
                  <th>False</th>
                  {hasOwnership && (
                    <>
                      <th>Own MAE</th>
                      <th>Own Corr</th>
                      <th>Top5 Acc</th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody>
                {[...metrics].reverse().map((row) => (
                  <tr key={row.date}>
                    <td className="date-cell">{row.date}</td>
                    <td>{row.players_matched}</td>
                    <td className={getMetricClass(row.fpts_mae, { good: 7, warn: 9 })}>
                      {formatNumber(row.fpts_mae)}
                    </td>
                    <td className={getMetricClass(row.minutes_mae, { good: 4.5, warn: 6 })}>
                      {formatNumber(row.minutes_mae)}
                    </td>
                    <td className={getMetricClass(row.coverage_80, { good: 0.65, warn: 0.55 }, true)}>
                      {formatPercent(row.coverage_80)}
                    </td>
                    <td className={getMetricClass(row.coverage_90, { good: 0.75, warn: 0.68 }, true)}>
                      {formatPercent(row.coverage_90)}
                    </td>
                    <td className={getMetricClass(Math.abs(row.bias ?? 0), { good: 1, warn: 2 })}>
                      {formatNumber(row.bias)}
                    </td>
                    <td>{row.missed}</td>
                    <td>{row.false_preds}</td>
                    {hasOwnership && (
                      <>
                        <td className={getMetricClass(row.own_mae, { good: 3, warn: 5 })}>
                          {formatNumber(row.own_mae)}
                        </td>
                        <td className={getMetricClass(row.own_corr, { good: 0.8, warn: 0.65 }, true)}>
                          {formatNumber(row.own_corr, 3)}
                        </td>
                        <td
                          className={getMetricClass(row.chalk_top5_acc, { good: 0.6, warn: 0.4 }, true)}
                        >
                          {formatPercent(row.chalk_top5_acc)}
                        </td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!loading && metrics.length === 0 && !error && (
        <div className="evaluation-alert muted">No evaluation data available yet.</div>
      )}

      {/* Footer Stats */}
      {!loading && summary && (
        <div className="evaluation-footer">
          <span>
            {summary.dates_evaluated} days evaluated · {summary.total_players_matched} total
            player-games
          </span>
          <span>
            Missed: {summary.total_missed} · False Preds: {summary.total_false_preds}
          </span>
        </div>
      )}
    </div>
  )
}

export default EvaluationPage
