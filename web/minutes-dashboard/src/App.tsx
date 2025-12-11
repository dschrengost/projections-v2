import { useCallback, useEffect, useMemo, useState } from 'react'
import './App.css'
import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { apiUrl } from './api/client'
import PipelinePage from './pages/PipelinePage'
import EvaluationPage from './pages/EvaluationPage'
import OptimizerPage from './pages/OptimizerPage'
import { MinutesResponse, PlayerRow } from './types'
import {
  formatFpts,
  formatMinutes,
  formatMinutesSim,
  formatOwnership,
  formatPercent,
  formatSalary,
  formatStat,
  formatTime,
  formatValue,
  getStatusBadge,
  StatusBadge,
} from './utils'
import { GameView } from './components/GameView'

type FptsMeta = {
  fpts_model_run_id?: string
  generated_at?: string
  scoring_system?: string
}

type SummaryResponse = {
  date: string
  counts?: {
    rows: number
    players: number
    teams: number
  }
  model_run_id?: string
  bundle_dir?: string
  run_id?: string | null
  run_as_of_ts?: string | null
  fpts_available?: boolean
  fpts_meta?: FptsMeta
  sim_available?: boolean
}

type SortKey =
  | 'player_id'
  | 'team_id'
  | 'minutes_p10'
  | 'minutes_p50'
  | 'minutes_p90'
  | 'play_prob'
  | 'proj_fpts'
  | 'fpts_per_min_pred'
  | 'starter_status'
  | 'salary'
  | 'pred_own_pct'
  | 'value'
  | 'sim_dk_fpts_mean'
  | 'sim_dk_fpts_p05'
  | 'sim_dk_fpts_p10'
  | 'sim_dk_fpts_p25'
  | 'sim_dk_fpts_p50'
  | 'sim_dk_fpts_p75'
  | 'sim_dk_fpts_p95'
  | 'sim_pts_mean'
  | 'sim_reb_mean'
  | 'sim_ast_mean'
  | 'sim_stl_mean'
  | 'sim_blk_mean'
  | 'sim_tov_mean'
  | 'sim_minutes_sim_mean'

type RunOption = {
  run_id: string
  run_as_of_ts?: string | null
  generated_at?: string | null
}

const SORT_LABELS: Record<SortKey, string> = {
  player_id: 'Player',
  team_id: 'Team',
  minutes_p10: 'P10',
  minutes_p50: 'P50',
  minutes_p90: 'P90',
  play_prob: 'Play Prob',
  proj_fpts: 'FPTS',
  fpts_per_min_pred: 'FPTS/min',
  starter_status: 'Starter',
  salary: 'Salary',
  pred_own_pct: 'Own%',
  value: 'Value',
  sim_dk_fpts_mean: 'Sim Mean',
  sim_dk_fpts_p05: 'Sim p05',
  sim_dk_fpts_p10: 'Sim p10',
  sim_dk_fpts_p25: 'Sim p25',
  sim_dk_fpts_p50: 'Sim p50',
  sim_dk_fpts_p75: 'Sim p75',
  sim_dk_fpts_p95: 'Sim p95',
  sim_pts_mean: 'Sim PTS',
  sim_reb_mean: 'Sim REB',
  sim_ast_mean: 'Sim AST',
  sim_stl_mean: 'Sim STL',
  sim_blk_mean: 'Sim BLK',
  sim_tov_mean: 'Sim TOV',
  sim_minutes_sim_mean: 'Sim MIN',
}



const todayISO = () => new Date().toISOString().slice(0, 10)

const initialTab = (): 'minutes' | 'pipeline' | 'evaluation' | 'optimizer' => {
  if (typeof window === 'undefined') {
    return 'minutes'
  }
  if (window.location.pathname.includes('optimizer')) return 'optimizer'
  if (window.location.pathname.includes('pipeline')) return 'pipeline'
  if (window.location.pathname.includes('evaluation')) return 'evaluation'
  return 'minutes'
}



function App() {
  const [activeTab, setActiveTab] = useState<'minutes' | 'pipeline' | 'evaluation' | 'optimizer'>(initialTab)
  const [selectedDate, setSelectedDate] = useState(todayISO())
  const [rows, setRows] = useState<PlayerRow[]>([])
  const [summary, setSummary] = useState<SummaryResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState('')
  const [sortKey, setSortKey] = useState<SortKey>('minutes_p50')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
  const [statusMessage, setStatusMessage] = useState<string | null>(null)
  const [runId, setRunId] = useState<string | null>(null)
  const [runOptions, setRunOptions] = useState<RunOption[]>([])
  const [latestRunId, setLatestRunId] = useState<string | null>(null)

  const [showSim, setShowSim] = useState(true)
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null)

  useEffect(() => {
    if (typeof window === 'undefined') return
    const pathMap = { minutes: '/', pipeline: '/pipeline', evaluation: '/evaluation', optimizer: '/optimizer' }
    window.history.replaceState({}, '', pathMap[activeTab])
  }, [activeTab])

  const fetchData = useCallback(
    async (date: string, currentRunId: string | null) => {
      setLoading(true)
      setError(null)
      setStatusMessage(null)
      try {
        const runParam = currentRunId ? `&run_id=${encodeURIComponent(currentRunId)}` : ''
        const minutesRes = await fetch(
          apiUrl(`/api/minutes?date=${date}${runParam}`),
        )
        if (minutesRes.status === 404) {
          let detail = 'No artifact for selected date.'
          try {
            const body = await minutesRes.json()
            detail = body?.detail ?? detail
          } catch {
            // ignore parse errors
          }
          setRows([])
          setStatusMessage(detail)
        } else if (!minutesRes.ok) {
          throw new Error(`API error (${minutesRes.status})`)
        }
        if (minutesRes.ok) {
          const minutesJson: MinutesResponse = await minutesRes.json()
          setRows(minutesJson.players ?? [])
          setStatusMessage(null)
        }

        const summaryRes = await fetch(
          apiUrl(`/api/minutes/meta?date=${date}${runParam}`),
        )
        if (summaryRes.status === 404) {
          setSummary(null)
        } else if (summaryRes.ok) {
          setSummary(await summaryRes.json())
        } else {
          throw new Error(`Meta API error (${summaryRes.status})`)
        }
      } catch (err) {
        setError((err as Error).message)
        setSummary(null)
        setRows([])
        setStatusMessage(null)
      } finally {
        setLoading(false)
      }
    },
    [setRows],
  )

  useEffect(() => {
    const controller = new AbortController()
    const loadRuns = async () => {
      try {
        const res = await fetch(
          apiUrl(`/api/minutes/runs?date=${selectedDate}`),
          {
            signal: controller.signal,
          },
        )
        if (!res.ok) {
          setRunOptions([])
          setLatestRunId(null)
          setRunId(null)
          return
        }
        const data = await res.json()
        const runs: RunOption[] = data?.runs ?? []
        setRunOptions(runs)
        setLatestRunId(data?.latest ?? null)
        setRunId((prev) => {
          if (prev && runs.some((run) => run.run_id === prev)) {
            return prev
          }
          return data?.latest ?? null
        })
      } catch {
        setRunOptions([])
        setLatestRunId(null)
        setRunId(null)
      }
    }
    void loadRuns()
    return () => controller.abort()
  }, [selectedDate])

  useEffect(() => {
    void fetchData(selectedDate, runId)
  }, [selectedDate, runId, fetchData])

  const matchups = useMemo(() => {
    const games = new Map<string, { home: string; away: string; id: string }>()
    rows.forEach((row) => {
      if (!row.game_id) return
      const gameId = String(row.game_id)
      if (!games.has(gameId)) {
        // Infer home/away from team_id/opponent_team_id logic or just use tricode
        // Since we don't have explicit home/away flags in PlayerRow easily accessible here without parsing,
        // we'll just use the first row we see for a game.
        // A better way: The API returns all players. We can find unique team_tricode vs opponent_team_tricode pairs.
        // Let's just store the match string "TEAM vs OPP"
        const team = row.team_tricode || 'UNK'
        const opp = row.opponent_team_tricode || 'UNK'
        // Sort to ensure consistent key regardless of which team's row we hit first
        const [t1, t2] = [team, opp].sort()
        games.set(gameId, { home: t2, away: t1, id: gameId }) // Arbitrary assignment for now, or use row data if we knew home/away
      }
    })
    return Array.from(games.values())
  }, [rows])

  const filteredRows = useMemo(() => {
    let filtered = rows.slice()

    if (selectedGameId) {
      filtered = filtered.filter(row => String(row.game_id) === selectedGameId)
    }

    const text = filter.trim().toLowerCase()
    if (text) {
      filtered = filtered.filter((row) => {
        const values = [
          row.player_name,
          row.player_id,
          row.team_tricode,
          row.team_name,
          row.team_id,
          row.opponent_team_tricode,
          row.opponent_team_name,
          row.opponent_team_id,
          row.game_id,
        ]
          .filter(Boolean)
          .map((value) => String(value).toLowerCase())
        return values.some((value) => value.includes(text))
      })
    }

    const sorted = filtered.sort((a, b) => {
      const left = a[sortKey]
      const right = b[sortKey]
      if (typeof left === 'number' && typeof right === 'number') {
        return sortDir === 'asc' ? left - right : right - left
      }
      const leftText = String(left ?? '')
      const rightText = String(right ?? '')
      return sortDir === 'asc'
        ? leftText.localeCompare(rightText)
        : rightText.localeCompare(leftText)
    })
    return sorted
  }, [rows, filter, sortKey, sortDir, selectedGameId])

  const starterSortValue = (row: PlayerRow) =>
    row.is_confirmed_starter ? 2 : row.is_projected_starter ? 1 : 0

  const chartData = useMemo(() => {
    const totals = new Map<string, number>()
    filteredRows.forEach((row) => {
      const team =
        String(row.team_tricode ?? row.team_name ?? row.team_id ?? 'NA')
      const minutes = typeof row.minutes_p50 === 'number' ? row.minutes_p50 : 0
      totals.set(team, (totals.get(team) ?? 0) + minutes)
    })
    return Array.from(totals.entries()).map(([team, minutes]) => ({
      team,
      minutes: Number(minutes.toFixed(1)),
    }))
  }, [filteredRows])

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key)
      setSortDir(key === 'player_id' || key === 'team_id' ? 'asc' : 'desc')
    }
  }



  const simFptsColumns: Array<{ key: SortKey; label: string }> = [
    { key: 'sim_dk_fpts_p05', label: 'Sim p05' },
    { key: 'sim_dk_fpts_p10', label: 'Sim p10' },
    { key: 'sim_dk_fpts_p25', label: 'Sim p25' },
    { key: 'sim_dk_fpts_p50', label: 'Sim p50' },
    { key: 'sim_dk_fpts_p75', label: 'Sim p75' },
    { key: 'sim_dk_fpts_p95', label: 'Sim p95' },
    { key: 'sim_dk_fpts_mean', label: 'Sim mean' },
  ]

  const simStatColumns: Array<{ key: SortKey; label: string }> = [
    { key: 'sim_pts_mean', label: 'Sim PTS' },
    { key: 'sim_reb_mean', label: 'Sim REB' },
    { key: 'sim_ast_mean', label: 'Sim AST' },
    { key: 'sim_stl_mean', label: 'Sim STL' },
    { key: 'sim_blk_mean', label: 'Sim BLK' },
    { key: 'sim_tov_mean', label: 'Sim TOV' },
    { key: 'sim_minutes_sim_mean', label: 'Sim MIN' },
  ]

  const nav = (
    <nav className="app-nav">
      <button
        className={activeTab === 'minutes' ? 'active' : ''}
        onClick={() => setActiveTab('minutes')}
      >
        Minutes
      </button>
      <button
        className={activeTab === 'pipeline' ? 'active' : ''}
        onClick={() => setActiveTab('pipeline')}
      >
        Pipeline
      </button>
      <button
        className={activeTab === 'evaluation' ? 'active' : ''}
        onClick={() => setActiveTab('evaluation')}
      >
        Evaluation
      </button>
      <button
        className={activeTab === 'optimizer' ? 'active' : ''}
        onClick={() => setActiveTab('optimizer')}
      >
        Optimizer
      </button>
    </nav>
  )

  if (activeTab === 'pipeline') {
    return (
      <div className="app-shell">
        {nav}
        <PipelinePage />
      </div>
    )
  }

  if (activeTab === 'evaluation') {
    return (
      <div className="app-shell">
        {nav}
        <EvaluationPage />
      </div>
    )
  }

  if (activeTab === 'optimizer') {
    return (
      <div className="app-shell">
        {nav}
        <OptimizerPage />
      </div>
    )
  }

  return (
    <div className="app-shell">
      {nav}
      <header className="app-header">
        <div>
          <h1>Minutes Dashboard</h1>
          <p className="subtitle">
            Two-stage minutes_v1 predictions with conformal intervals.
          </p>
        </div>
        <div className="controls">
          <label>
            Date
            <input
              type="date"
              value={selectedDate}
              onChange={(event) => setSelectedDate(event.target.value)}
            />
          </label>
          <label>
            Run
            <select
              value={runId ?? latestRunId ?? ''}
              onChange={(event) => {
                const value = event.target.value
                setRunId(value || null)
              }}
            >
              {latestRunId && (
                <option value={latestRunId}>
                  {`Latest (${latestRunId})`}
                </option>
              )}
              {runOptions
                .filter((option) => option.run_id !== latestRunId)
                .map((option) => (
                  <option key={option.run_id} value={option.run_id}>
                    {option.run_id}
                    {option.run_as_of_ts ? ` · ${option.run_as_of_ts}` : ''}
                  </option>
                ))}
              {!latestRunId && !runOptions.length && <option value="">No runs</option>}
            </select>
          </label>
          <button onClick={() => fetchData(selectedDate, runId)} disabled={loading}>
            Refresh
          </button>
          <button
            className="trigger-btn"
            onClick={async () => {
              if (!confirm('Trigger manual pipeline run? This will take a few minutes.')) return
              try {
                const res = await fetch(apiUrl('/api/trigger'), { method: 'POST' })
                if (res.ok) {
                  alert('Pipeline triggered! Check back in a few minutes.')
                } else {
                  alert('Failed to trigger pipeline.')
                }
              } catch (err) {
                alert('Error triggering pipeline.')
              }
            }}
          >
            Run Pipeline
          </button>
        </div>
      </header>

      <section className="status-bar">
        {loading && <span>Loading projections…</span>}
        {error && <span className="error">Error: {error}</span>}
        {!loading && !error && statusMessage && (
          <span>{statusMessage}</span>
        )}
        {!loading && !error && !statusMessage && summary?.counts && (
          <span>
            Rows: {summary.counts.rows} · Players: {summary.counts.players} ·
            Teams: {summary.counts.teams}{' '}
            {summary.model_run_id && `· Model run: ${summary.model_run_id}`}{' '}
            {summary.run_id && `· Artifact: ${summary.run_id}`}{' '}
            {summary.run_as_of_ts && `· As of: ${formatTime(summary.run_as_of_ts)}`}{' '}
            {showSim && summary.sim_available === false && '· sim_v2 unavailable'}
          </span>
        )}
      </section>

      {matchups.length > 0 && (
        <section className="matchup-selector">
          <button
            className={`matchup-card ${selectedGameId === null ? 'active' : ''}`}
            onClick={() => setSelectedGameId(null)}
          >
            All Games
          </button>
          {matchups.map(m => (
            <button
              key={m.id}
              className={`matchup-card ${selectedGameId === m.id ? 'active' : ''}`}
              onClick={() => setSelectedGameId(m.id)}
            >
              {m.away} @ {m.home}
            </button>
          ))}
        </section>
      )}

      <section className="filters">
        <input
          type="text"
          placeholder="Filter by player/team id"
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
        />

        <label className="fpts-toggle">
          <input
            type="checkbox"
            checked={showSim}
            onChange={(event) => setShowSim(event.target.checked)}
          />
          Show sim_v2 columns
        </label>
      </section>

      <section className="content">
        {selectedGameId ? (
          <GameView rows={filteredRows} gameId={selectedGameId} />
        ) : (
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th onClick={() => toggleSort('player_id')} className="sortable">
                    Player
                    {sortKey === 'player_id' && (
                      <span>{sortDir === 'asc' ? ' ▲' : ' ▼'}</span>
                    )}
                  </th>
                  <th onClick={() => toggleSort('team_id')} className="sortable">
                    Team
                    {sortKey === 'team_id' && (
                      <span>{sortDir === 'asc' ? ' ▲' : ' ▼'}</span>
                    )}
                  </th>
                  <th>Opponent</th>
                  {(['minutes_p10', 'minutes_p50', 'minutes_p90', 'play_prob'] as SortKey[]).map(
                    (key) => (
                      <th
                        key={key}
                        onClick={() => toggleSort(key)}
                        className="sortable"
                      >
                        {SORT_LABELS[key]}
                        {sortKey === key && (
                          <span>{sortDir === 'asc' ? ' ▲' : ' ▼'}</span>
                        )}
                      </th>
                    ),
                  )}
                  {/* Ownership Columns */}
                  {(['salary', 'pred_own_pct', 'value'] as SortKey[]).map((key) => (
                    <th key={key} onClick={() => toggleSort(key)} className="sortable">
                      {SORT_LABELS[key]}
                      {sortKey === key && <span>{sortDir === 'asc' ? ' ▲' : ' ▼'}</span>}
                    </th>
                  ))}
                  {showSim && (
                    <>
                      {simFptsColumns.map(({ key, label }) => (
                        <th key={key} onClick={() => toggleSort(key)} className="sortable">
                          {label}
                          {sortKey === key && <span>{sortDir === 'asc' ? ' ▲' : ' ▼'}</span>}
                        </th>
                      ))}
                      {simStatColumns.map(({ key, label }) => (
                        <th key={key} onClick={() => toggleSort(key)} className="sortable">
                          {label}
                          {sortKey === key && <span>{sortDir === 'asc' ? ' ▲' : ' ▼'}</span>}
                        </th>
                      ))}
                    </>
                  )}
                </tr>
              </thead>
              <tbody>
                {filteredRows.map((row) => {
                  const playerLabel = row.player_name || row.player_id
                  const teamLabel =
                    row.team_tricode || row.team_name || row.team_id
                  const opponentLabel =
                    row.opponent_team_tricode ||
                    row.opponent_team_name ||
                    row.opponent_team_id
                  const statusBadge = getStatusBadge(row.status)
                  return (
                    <tr key={`${row.game_id}-${row.player_id}`}>
                      <td>
                        <div className="player-header">
                          <strong>{playerLabel}</strong>
                          {statusBadge && (
                            <span
                              className={`status-tag ${statusBadge.className}`}
                              title={statusBadge.title}
                            >
                              {statusBadge.label}
                            </span>
                          )}
                          {row.is_confirmed_starter && (
                            <span className="status-tag badge-confirmed" title="Confirmed Starter">Start</span>
                          )}
                          {!row.is_confirmed_starter && row.is_projected_starter && (
                            <span className="status-tag badge-projected" title="Projected Starter">Proj</span>
                          )}
                        </div>
                        <div className="muted">{formatTime(row.tip_ts, row.game_date)}</div>
                      </td>
                      <td>{teamLabel}</td>
                      <td>{opponentLabel}</td>
                      <td>{formatMinutes(row.minutes_p10)}</td>
                      <td>{formatMinutes(row.minutes_p50)}</td>
                      <td>{formatMinutes(row.minutes_p90)}</td>
                      <td>{formatPercent(row.play_prob)}</td>
                      {/* Ownership Columns */}
                      <td>{formatSalary(row.salary)}</td>
                      <td className="ownership-cell">{formatOwnership(row.pred_own_pct)}</td>
                      <td className="value-cell">{formatValue(row.value)}</td>
                      {showSim && (
                        <>
                          {simFptsColumns.map(({ key }) => (
                            <td key={key}>
                              {formatFpts(row[key as keyof PlayerRow] as number | undefined)}
                            </td>
                          ))}
                          {simStatColumns.map(({ key }) => (
                            <td key={key}>
                              {key === 'sim_minutes_sim_mean'
                                ? formatMinutesSim(row[key as keyof PlayerRow] as number | undefined)
                                : formatStat(row[key as keyof PlayerRow] as number | undefined)}
                            </td>
                          ))}
                        </>
                      )}
                    </tr>
                  )
                })}
                {!filteredRows.length && !loading && (
                  <tr>
                    <td colSpan={7} className="muted">
                      No rows available for {selectedDate}.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
        <div className="chart-wrapper">
          <h2>Team minutes (p50)</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <XAxis dataKey="team" />
              <YAxis />
              <Tooltip formatter={(value: number) => `${value.toFixed(1)} min`} />
              <Bar dataKey="minutes" fill="#4f46e5" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  )
}

export default App
