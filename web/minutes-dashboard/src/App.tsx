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

type PlayerRow = {
  game_date?: string
  tip_ts?: string
  game_id?: number | string
  player_id?: number | string
  player_name?: string
  team_id?: number | string
  status?: string
  team_name?: string
  team_tricode?: string
  opponent_team_id?: number | string
  opponent_team_name?: string
  opponent_team_tricode?: string
  starter_flag?: boolean
  play_prob?: number
  minutes_p10?: number
  minutes_p50?: number
  minutes_p90?: number
  minutes_p10_cond?: number
  minutes_p90_cond?: number
}

type MinutesResponse = {
  date: string
  count: number
  players: PlayerRow[]
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
}

type SortKey = 'player_id' | 'team_id' | 'minutes_p10' | 'minutes_p50' | 'minutes_p90' | 'play_prob'

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
}

type StatusBadge = {
  label: string
  title: string
  className: string
}

const STATUS_BADGE_DEFINITIONS: Array<{
  matcher: (value: string) => boolean
  badge: StatusBadge
}> = [
  {
    matcher: (value) => value === 'q' || value === 'questionable',
    badge: {
      label: 'Q',
      title: 'Questionable',
      className: 'status-questionable',
    },
  },
  {
    matcher: (value) => value.startsWith('prob'),
    badge: {
      label: 'Prob',
      title: 'Probable',
      className: 'status-probable',
    },
  },
  {
    matcher: (value) => value.includes('doubt'),
    badge: {
      label: 'D',
      title: 'Doubtful',
      className: 'status-doubtful',
    },
  },
  {
    matcher: (value) => value.includes('gtd') || value.includes('game time'),
    badge: {
      label: 'GTD',
      title: 'Game time decision',
      className: 'status-gtd',
    },
  },
  {
    matcher: (value) => value === 'out',
    badge: {
      label: 'Out',
      title: 'Out',
      className: 'status-out',
    },
  },
]

const getStatusBadge = (status?: string): StatusBadge | null => {
  const normalized = status?.trim().toLowerCase()
  if (!normalized) {
    return null
  }
  for (const definition of STATUS_BADGE_DEFINITIONS) {
    if (definition.matcher(normalized)) {
      return definition.badge
    }
  }
  return null
}

const todayISO = () => new Date().toISOString().slice(0, 10)

function App() {
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

  const fetchData = useCallback(
    async (date: string, currentRunId: string | null) => {
      setLoading(true)
      setError(null)
      setStatusMessage(null)
      try {
        const runParam = currentRunId ? `&run_id=${encodeURIComponent(currentRunId)}` : ''
        const minutesRes = await fetch(`/api/minutes?date=${date}${runParam}`)
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

        const summaryRes = await fetch(`/api/minutes/meta?date=${date}${runParam}`)
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
        const res = await fetch(`/api/minutes/runs?date=${selectedDate}`, {
          signal: controller.signal,
        })
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

  const filteredRows = useMemo(() => {
    const text = filter.trim().toLowerCase()
    const filtered = text
      ? rows.filter((row) => {
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
      : rows.slice()

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
  }, [rows, filter, sortKey, sortDir])

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

  const formatMinutes = (value?: number) =>
    typeof value === 'number' ? value.toFixed(1) : '—'
  const formatPercent = (value?: number) =>
    typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—'

  return (
    <div className="app-shell">
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
            {summary.run_as_of_ts && `· As of: ${summary.run_as_of_ts}`}
          </span>
        )}
      </section>

      <section className="filters">
        <input
          type="text"
          placeholder="Filter by player/team id"
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
        />
      </section>

      <section className="content">
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
                      </div>
                      <div className="muted">{row.tip_ts}</div>
                    </td>
                    <td>{teamLabel}</td>
                    <td>{opponentLabel}</td>
                  <td>{formatMinutes(row.minutes_p10)}</td>
                  <td>{formatMinutes(row.minutes_p50)}</td>
                  <td>{formatMinutes(row.minutes_p90)}</td>
                  <td>{formatPercent(row.play_prob)}</td>
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
