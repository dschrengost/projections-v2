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
            <option value={1}>1</option>
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
            <div className="card">
              <div className="label">Own MAE</div>
              <div
                className={`value ${getMetricClass(summary?.avg_own_mae, { good: 4, warn: 6 })}`}
              >
                {loading ? '…' : formatNumber(summary?.avg_own_mae)}
              </div>
            </div>
            <div className="card">
              <div className="label">High Own MAE</div>
              <div
                className={`value ${getMetricClass(summary?.avg_high_own_mae, { good: 10, warn: 20 })}`}
              >
                {loading ? '…' : formatNumber(summary?.avg_high_own_mae)}
              </div>
            </div>
          </>
        )}
      </div>

      {/* Salary Tier Accuracy */}
      {!loading && summary?.avg_fpts_mae_elite != null && (
        <div className="evaluation-section">
          <h2>Accuracy by Salary Tier</h2>
          <div className="salary-tier-grid">
            <div className="tier-card elite">
              <div className="tier-label">Elite ($8K+)</div>
              <div className="tier-value">{formatNumber(summary?.avg_fpts_mae_elite)}</div>
            </div>
            <div className="tier-card mid">
              <div className="tier-label">Mid ($5.5-8K)</div>
              <div className="tier-value">{formatNumber(summary?.avg_fpts_mae_mid)}</div>
            </div>
            <div className="tier-card value">
              <div className="tier-label">Value ($3.5-5.5K)</div>
              <div className="tier-value">{formatNumber(summary?.avg_fpts_mae_value)}</div>
            </div>
            <div className="tier-card punt">
              <div className="tier-label">Punt ($3-3.5K)</div>
              <div className="tier-value">{formatNumber(summary?.avg_fpts_mae_punt)}</div>
            </div>
          </div>
          <p className="section-note">FPTS Mean Absolute Error by salary tier</p>
        </div>
      )}

      {/* Edge Cases Summary */}
      {!loading && (summary?.total_dnp_false_positives > 0 || summary?.total_starter_misses > 0) && (
        <div className="evaluation-section">
          <h2>Prediction Failures</h2>
          <p className="section-note" style={{ marginTop: 0, marginBottom: '1rem' }}>
            High-impact prediction errors that would hurt lineup construction
          </p>
          <div className="edge-case-grid">
            <div className="edge-case-card">
              <div className="edge-case-value">{summary?.total_dnp_false_positives ?? 0}</div>
              <div className="edge-case-label">Injured/DNP Misses</div>
              <div className="edge-case-desc">
                Players we projected to play 10+ min who didn't suit up.
                Likely late scratches or injury news we missed.
              </div>
            </div>
            <div className="edge-case-card">
              <div className="edge-case-value">{summary?.total_starter_misses ?? 0}</div>
              <div className="edge-case-label">Minutes Underestimates</div>
              <div className="edge-case-desc">
                Players who played 30+ min but we predicted under 20.
                Missed lineup changes or role expansions.
              </div>
            </div>
            <div className="edge-case-card">
              <div className="edge-case-value">{summary?.total_blowup_misses ?? 0}</div>
              <div className="edge-case-label">Ceiling Misses</div>
              <div className="edge-case-desc">
                Players who scored 50+ FPTS but we projected under 30.
                Underestimated upside on big performances.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Minutes Accuracy by Bucket */}
      {!loading && metrics.length > 0 && metrics[metrics.length - 1]?.mins_mae_starter_30plus != null && (
        <div className="evaluation-section">
          <h2>Minutes Accuracy by Bucket</h2>
          <p className="section-note" style={{ marginTop: 0, marginBottom: '1rem' }}>
            Minutes MAE (Mean Absolute Error) grouped by actual playing time
          </p>
          <div className="salary-tier-grid">
            <div className="tier-card elite">
              <div className="tier-label">Starter (30+)</div>
              <div className="tier-value">{formatNumber(metrics[metrics.length - 1]?.mins_mae_starter_30plus)}</div>
              <div className="tier-count">n={metrics[metrics.length - 1]?.n_starter_30plus ?? 0}</div>
            </div>
            <div className="tier-card mid">
              <div className="tier-label">Heavy (20-30)</div>
              <div className="tier-value">{formatNumber(metrics[metrics.length - 1]?.mins_mae_heavy_20_30)}</div>
              <div className="tier-count">n={metrics[metrics.length - 1]?.n_heavy_20_30 ?? 0}</div>
            </div>
            <div className="tier-card value">
              <div className="tier-label">Rotation (10-20)</div>
              <div className="tier-value">{formatNumber(metrics[metrics.length - 1]?.mins_mae_rotation_10_20)}</div>
              <div className="tier-count">n={metrics[metrics.length - 1]?.n_rotation_10_20 ?? 0}</div>
            </div>
            <div className="tier-card punt">
              <div className="tier-label">Light (5-10)</div>
              <div className="tier-value">{formatNumber(metrics[metrics.length - 1]?.mins_mae_light_5_10)}</div>
              <div className="tier-count">n={metrics[metrics.length - 1]?.n_light_5_10 ?? 0}</div>
            </div>
          </div>
        </div>
      )}

      {/* Top FPTS Misses */}
      {!loading && metrics.length > 0 && metrics[metrics.length - 1]?.top_fpts_misses?.length && (
        <div className="evaluation-section">
          <h2>Top FPTS Misses (Latest Day)</h2>
          <p className="section-note" style={{ marginTop: 0, marginBottom: '1rem' }}>
            Largest absolute errors between predicted and actual FPTS
          </p>
          <div className="table-wrapper evaluation-table">
            <table>
              <thead>
                <tr>
                  <th>Player</th>
                  <th>Pred FPTS</th>
                  <th>Actual FPTS</th>
                  <th>Error</th>
                  <th>Pred Min</th>
                  <th>Actual Min</th>
                </tr>
              </thead>
              <tbody>
                {metrics[metrics.length - 1].top_fpts_misses?.slice(0, 10).map((miss, i) => (
                  <tr key={i}>
                    <td className="date-cell">{miss.player_name ?? miss.player_id}</td>
                    <td>{formatNumber(miss.pred_fpts, 1)}</td>
                    <td>{formatNumber(miss.actual_fpts, 1)}</td>
                    <td className={miss.error > 0 ? 'metric-good' : 'metric-warn'}>
                      {miss.error > 0 ? '+' : ''}{formatNumber(miss.error, 1)}
                    </td>
                    <td>{formatNumber(miss.pred_mins, 1)}</td>
                    <td>{formatNumber(miss.actual_mins, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

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

      {/* Top Ownership Misses */}
      {!loading && hasOwnership && metrics.length > 0 && metrics[metrics.length - 1]?.own_top_misses && (
        <div className="evaluation-section">
          <h2>Top Ownership Misses (Latest Day)</h2>
          <p className="section-note" style={{ marginTop: 0, marginBottom: '1rem' }}>
            Largest absolute errors between predicted and actual ownership
          </p>
          <div className="ownership-misses-grid">
            {metrics[metrics.length - 1].own_top_misses?.map((miss, i) => (
              <div key={i} className="ownership-miss-card">
                <div className="miss-player">{miss.player}</div>
                <div className="miss-details">
                  <span className="miss-actual">Actual: {miss.actual}%</span>
                  <span className="miss-pred">Pred: {miss.pred}%</span>
                  <span className={`miss-error ${miss.error > 0 ? 'under' : 'over'}`}>
                    {miss.error > 0 ? '↓' : '↑'} {Math.abs(miss.error).toFixed(1)}pts
                  </span>
                </div>
              </div>
            ))}
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
