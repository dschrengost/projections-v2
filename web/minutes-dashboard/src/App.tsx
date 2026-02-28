import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
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
import ContestPage from './pages/ContestPage'
import ContestSimPage from './pages/ContestSimPage'
import DiagnosticsPage from './pages/DiagnosticsPage'
import EntryManagerPage from './pages/EntryManagerPage'
import LivePage from './pages/LivePage'
import PropsPage from './pages/PropsPage'
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
  formatRunIdToEST,
  formatTimestampToEST,
  getStatusBadge,
  StatusBadge,
} from './utils'
import { GameView } from './components/GameView'
import { PlayerChangeAlerts } from './components/PlayerChangeAlerts'

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
  projections_run_id?: string | null
  minutes_run_id?: string | null
  rates_run_id?: string | null
  sim_run_id?: string | null
  sim_profile?: string | null
  n_worlds?: number | null
  pinned_run_id?: string | null
  latest_run_id?: string | null
  blessed_run_id?: string | null
  generated_at?: string | null
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
  | 'minutes_final'
  | 'play_prob'
  | 'p_in_rotation'
  | 'proj_fpts'
  | 'fpts_per_min_pred'
  | 'starter_status'
  | 'salary'
  | 'pred_own_pct'
  | 'value'
  | 'sim_dk_fpts_mean'
  | 'sim_dk_fpts_p95'
  | 'sim_minutes_sim_mean'
  | 'sim_minutes_sim_mean_uncond'
  // Canonical sim fields (explicit conditioning)
  | 'fpts_sim_uncond_mean'
  | 'fpts_sim_uncond_p95'
  | 'minutes_sim_uncond_mean'
  | 'play_prob_eff'
  | 'sim_p_available'
  | 'sim_p_rotation'
  | 'bench_zero_p_zero'
  | 'minutes_sim_p_active'

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
  minutes_final: 'MIN',
  play_prob: 'p(avail)',
  p_in_rotation: 'In Rotation',
  proj_fpts: 'FPTS',
  fpts_per_min_pred: 'FPTS/min',
  starter_status: 'Starter',
  salary: 'Salary',
  pred_own_pct: 'Own%',
  value: 'Value',
  sim_dk_fpts_mean: 'Sim Mean',
  sim_dk_fpts_p95: 'Sim p95',
  sim_minutes_sim_mean: 'Sim MIN',
  sim_minutes_sim_mean_uncond: 'Sim MIN (Uncond)',
  fpts_sim_uncond_mean: 'FPTS (Sim Mean, Uncond)',
  fpts_sim_uncond_p95: 'FPTS (Sim p95, Uncond)',
  minutes_sim_uncond_mean: 'MIN (Sim Mean, Uncond)',
  play_prob_eff: 'p(avail used)',
  sim_p_available: 'p(avail realized)',
  sim_p_rotation: 'p(MIN≥5)',
  bench_zero_p_zero: 'p(drop|avail)',
  minutes_sim_p_active: 'p(MIN≥1)',
}

type ModelOption = {
  model_id: string
  label: string
  meta?: {
    play_threshold?: number
  }
}



import { useSlateDate } from './hooks/useSlateDate'

type TabKey =
  | 'live'
  | 'minutes'
  | 'pipeline'
  | 'evaluation'
  | 'optimizer'
  | 'entry-manager'
  | 'contest'
  | 'contest-sim'
  | 'diagnostics'
  | 'props'

const initialTab = (): TabKey => {
  if (typeof window === 'undefined') {
    return 'live'
  }
  if (window.location.pathname.includes('/live/game/')) return 'live'
  if (window.location.pathname === '/' || window.location.pathname.includes('/live')) return 'live'
  if (window.location.pathname.includes('/projections')) return 'minutes'
  if (window.location.pathname.includes('/runs')) return 'pipeline'
  if (window.location.pathname.includes('props')) return 'props'
  if (window.location.pathname.includes('diagnostics')) return 'diagnostics'
  if (window.location.pathname.includes('contest-sim')) return 'contest-sim'
  if (window.location.pathname.includes('contest')) return 'contest'
  if (window.location.pathname.includes('entry-manager')) return 'entry-manager'
  if (window.location.pathname.includes('optimizer')) return 'optimizer'
  if (window.location.pathname.includes('pipeline')) return 'pipeline'
  if (window.location.pathname.includes('evaluation')) return 'evaluation'
  return 'live'
}

const initialLiveGameId = (): string | null => {
  if (typeof window === 'undefined') {
    return null
  }
  const match = window.location.pathname.match(/\/live\/game\/([^/]+)/)
  return match?.[1] ? decodeURIComponent(match[1]) : null
}



function App() {
  const [activeTab, setActiveTab] = useState<TabKey>(initialTab)
  const [liveGameId, setLiveGameId] = useState<string | null>(initialLiveGameId)
  const [selectedDate, setSelectedDate] = useSlateDate()
  const [rows, setRows] = useState<PlayerRow[]>([])
  const [summary, setSummary] = useState<SummaryResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState('')
  const [sortKey, setSortKey] = useState<SortKey>('minutes_final')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
  const [statusMessage, setStatusMessage] = useState<string | null>(null)
  const [runId, setRunId] = useState<string | null>(null)
  const [runOptions, setRunOptions] = useState<RunOption[]>([])
  const [latestRunId, setLatestRunId] = useState<string | null>(null)
  const [pinnedRunId, setPinnedRunId] = useState<string | null>(null)
  const [blessedRunId, setBlessedRunId] = useState<string | null>(null)
  const pinnedRunRef = useRef<string | null>(null)

  const [modelId, setModelId] = useState<string>('prod')
  const [modelOptions, setModelOptions] = useState<ModelOption[]>([
    { model_id: 'prod', label: 'Production' },
  ])

  const [showSim, setShowSim] = useState(true)
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null)

  // Player override state: player_id -> { minutes?: number, fpts?: number, own?: number }
  type PlayerOverride = { minutes?: number | null; fpts?: number | null; own?: number | null }
  const [overrides, setOverrides] = useState<Map<string, PlayerOverride>>(new Map())

  const resolveSimMinutesValue = (row: PlayerRow, key: SortKey) => {
    if (key === 'minutes_sim_uncond_mean') {
      return (
        row.minutes_sim_uncond_mean ??
        row.sim_minutes_sim_mean_uncond ??
        row.sim_minutes_sim_mean
      )
    }
    if (key === 'sim_minutes_sim_mean_uncond') {
      return row.sim_minutes_sim_mean_uncond ?? row.sim_minutes_sim_mean
    }
    return row[key as keyof PlayerRow] as number | undefined
  }

  const updateOverride = (playerId: string, field: keyof PlayerOverride, value: number | null) => {
    setOverrides(prev => {
      const next = new Map(prev)
      const current = next.get(playerId) ?? {}
      next.set(playerId, { ...current, [field]: value })
      return next
    })
  }

  useEffect(() => {
    if (typeof window === 'undefined') return
    const pathMap = {
      live: '/live',
      minutes: '/projections',
      pipeline: '/runs',
      evaluation: '/evaluation',
      optimizer: '/optimizer',
      'entry-manager': '/entry-manager',
      contest: '/contest',
      'contest-sim': '/contest-sim',
      diagnostics: '/diagnostics',
      props: '/props',
    }
    const nextPath =
      activeTab === 'live'
        ? liveGameId
          ? `/live/game/${encodeURIComponent(liveGameId)}`
          : '/'
        : pathMap[activeTab]
    window.history.replaceState({}, '', nextPath)
  }, [activeTab, liveGameId])

  const fetchData = useCallback(
    async (date: string, currentRunId: string | null, currentModelId: string) => {
      setLoading(true)
      setError(null)
      setStatusMessage(null)
      try {
        const runParam = currentRunId ? `&run_id=${encodeURIComponent(currentRunId)}` : ''
        const modelParam = currentModelId && currentModelId !== 'prod'
          ? `&model_id=${encodeURIComponent(currentModelId)}`
          : ''
        const minutesRes = await fetch(
          apiUrl(`/api/minutes?date=${date}${runParam}${modelParam}`),
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
          apiUrl(`/api/minutes/meta?date=${date}${runParam}${modelParam}`),
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
    const loadModels = async () => {
      try {
        const res = await fetch(apiUrl(`/api/minutes/models?date=${selectedDate}`), {
          signal: controller.signal,
        })
        if (!res.ok) {
          setModelOptions([{ model_id: 'prod', label: 'Production' }])
          setModelId('prod')
          return
        }
        const models: ModelOption[] = await res.json()
        const safeModels =
          Array.isArray(models) && models.length
            ? models
            : [{ model_id: 'prod', label: 'Production' }]
        setModelOptions(safeModels)
        setModelId((prev) => (safeModels.some((m) => m.model_id === prev) ? prev : 'prod'))
      } catch {
        setModelOptions([{ model_id: 'prod', label: 'Production' }])
        setModelId('prod')
      }
    }
    void loadModels()
    return () => controller.abort()
  }, [selectedDate])

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
        setPinnedRunId(data?.pinned ?? null)
        setBlessedRunId(data?.blessed ?? null)
        setRunId((prev) => {
          const prevPinned = pinnedRunRef.current
          pinnedRunRef.current = data?.pinned ?? null
          if (prev && prevPinned && prev === prevPinned && data?.pinned) {
            return data.pinned
          }
          if (prev && runs.some((run) => run.run_id === prev)) {
            return prev
          }
          // Priority: blessed > pinned > latest
          if (data?.blessed && runs.some((run) => run.run_id === data.blessed)) {
            return data.blessed
          }
          if (data?.pinned && runs.some((run) => run.run_id === data.pinned)) {
            return data.pinned
          }
          return data?.latest ?? null
        })
      } catch {
        setRunOptions([])
        setLatestRunId(null)
        setRunId(null)
        setPinnedRunId(null)
        setBlessedRunId(null)
      }
    }
    void loadRuns()
    return () => controller.abort()
  }, [selectedDate])

  useEffect(() => {
    if (activeTab !== 'minutes') {
      return
    }
    void fetchData(selectedDate, runId, modelId)
  }, [activeTab, selectedDate, runId, modelId, fetchData])

  useEffect(() => {
    setSortKey((prev) => {
      if (modelId === 'rmh_v1_1' && prev === 'play_prob') return 'p_in_rotation'
      if (modelId !== 'rmh_v1_1' && prev === 'p_in_rotation') return 'play_prob'
      return prev
    })
  }, [modelId])

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
      const left = resolveSimMinutesValue(a, sortKey)
      const right = resolveSimMinutesValue(b, sortKey)
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
      const minutes = typeof row.minutes_final === 'number'
        ? row.minutes_final
        : typeof row.minutes_p50 === 'number'
          ? row.minutes_p50
          : 0
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

  const probSortKey: SortKey = modelId === 'rmh_v1_1' ? 'p_in_rotation' : 'play_prob'
  const selectedModel = useMemo(
    () => modelOptions.find((m) => m.model_id === modelId),
    [modelOptions, modelId],
  )



  const simFptsColumns: Array<{ key: SortKey; label: string }> = [
    { key: 'fpts_sim_uncond_mean', label: 'FPTS (Uncond Mean)' },
    { key: 'fpts_sim_uncond_p95', label: 'FPTS (Uncond p95)' },
  ]

  const simStatColumns: Array<{ key: SortKey; label: string }> = [
    { key: 'minutes_sim_uncond_mean', label: 'MIN (Uncond Mean)' },
    { key: 'play_prob_eff', label: 'p(avail used)' },
    { key: 'bench_zero_p_zero', label: 'p(drop|avail)' },
    { key: 'sim_p_rotation', label: 'p(MIN≥5)' },
    { key: 'minutes_sim_p_active', label: 'p(MIN≥1)' },
  ]

  const nav = (
    <nav className="app-nav">
      <button
        className={activeTab === 'live' ? 'active' : ''}
        onClick={() => {
          setLiveGameId(null)
          setActiveTab('live')
        }}
      >
        Live
      </button>
      <button
        className={activeTab === 'minutes' ? 'active' : ''}
        onClick={() => setActiveTab('minutes')}
      >
        Projections
      </button>
      <button
        className={activeTab === 'pipeline' ? 'active' : ''}
        onClick={() => setActiveTab('pipeline')}
      >
        Runs
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
      <button
        className={activeTab === 'entry-manager' ? 'active' : ''}
        onClick={() => setActiveTab('entry-manager')}
      >
        Entry Manager
      </button>
      <button
        className={activeTab === 'contest' ? 'active' : ''}
        onClick={() => setActiveTab('contest')}
      >
        Contest
      </button>
      <button
        className={activeTab === 'contest-sim' ? 'active' : ''}
        onClick={() => setActiveTab('contest-sim')}
      >
        Contest Sim
      </button>
      <button
        className={activeTab === 'props' ? 'active' : ''}
        onClick={() => setActiveTab('props')}
      >
        Props
      </button>
      <button
        className={activeTab === 'diagnostics' ? 'active' : ''}
        onClick={() => setActiveTab('diagnostics')}
      >
        Diagnostics
      </button>
    </nav>
  )

  if (activeTab === 'live') {
    return (
      <div className="app-shell">
        {nav}
        <LivePage
          date={selectedDate}
          onDateChange={setSelectedDate}
          selectedGameId={liveGameId}
          onOpenGame={(gameId) => {
            setLiveGameId(gameId)
            setActiveTab('live')
          }}
          onCloseGame={() => setLiveGameId(null)}
          onOpenLateSwap={() => setActiveTab('entry-manager')}
        />
      </div>
    )
  }

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

  if (activeTab === 'entry-manager') {
    return (
      <div className="app-shell">
        {nav}
        <EntryManagerPage />
      </div>
    )
  }

  if (activeTab === 'contest') {
    return (
      <div className="app-shell">
        {nav}
        <ContestPage />
      </div>
    )
  }

  if (activeTab === 'contest-sim') {
    return (
      <div className="app-shell">
        {nav}
        <ContestSimPage />
      </div>
    )
  }

  if (activeTab === 'props') {
    return (
      <div className="app-shell">
        {nav}
        <PropsPage />
      </div>
    )
  }

  if (activeTab === 'diagnostics') {
    return (
      <div className="app-shell">
        {nav}
        <DiagnosticsPage />
      </div>
    )
  }

  return (
    <div className="app-shell">
      {nav}
      <header className="app-header">
        <div>
          <h1>Projections</h1>
          <p className="subtitle">
            Player-level projections and run comparison for the selected slate.
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
              value={runId ?? blessedRunId ?? pinnedRunId ?? latestRunId ?? ''}
              onChange={(event) => {
                const value = event.target.value
                setRunId(value || null)
              }}
            >
              {/* Blessed run - validated production run */}
              {blessedRunId && (
                <option value={blessedRunId}>
                  ✅ Blessed · {formatRunIdToEST(blessedRunId)}{blessedRunId === latestRunId ? ' (Latest)' : ''}
                </option>
              )}
              {/* Pinned run - manual override for debugging */}
              {pinnedRunId && pinnedRunId !== blessedRunId && (
                <option value={pinnedRunId}>
                  📌 Pinned · {formatRunIdToEST(pinnedRunId)}{pinnedRunId === latestRunId ? ' (Latest)' : ''}
                </option>
              )}
              {/* Latest run - most recent */}
              {latestRunId && latestRunId !== pinnedRunId && latestRunId !== blessedRunId && (
                <option value={latestRunId}>
                  🕑 Latest · {formatRunIdToEST(latestRunId)}
                </option>
              )}
              {/* Other runs */}
              {runOptions
                .filter((option) => option.run_id !== latestRunId && option.run_id !== pinnedRunId && option.run_id !== blessedRunId)
                .map((option) => (
                  <option key={option.run_id} value={option.run_id}>
                    {formatRunIdToEST(option.run_id) || option.run_id}
                    {(option.run_as_of_ts || option.generated_at) ? ` · ${formatTimestampToEST(option.run_as_of_ts ?? option.generated_at ?? '')}` : ''}
                  </option>
                ))}
              {!latestRunId && !runOptions.length && <option value="">No runs</option>}
            </select>
          </label>
          {modelOptions.length > 1 && (
            <label>
              Model
              <select
                value={modelId}
                onChange={(event) => setModelId(event.target.value || 'prod')}
              >
                {modelOptions.map((option) => (
                  <option key={option.model_id} value={option.model_id}>
                    {option.label}
                  </option>
                ))}
              </select>
              {modelId === 'rmh_v1_1' && selectedModel?.meta?.play_threshold != null && (
                <div
                  className="muted model-caption"
                  title="RMH v1.1: play_threshold (minutes) defines 'in rotation' for p_in_rotation"
                >
                  T={selectedModel.meta.play_threshold}
                </div>
              )}
            </label>
          )}
          <button onClick={() => fetchData(selectedDate, runId, modelId)} disabled={loading}>
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
            {summary.run_id && `· Run: ${summary.run_id}`}{' '}
            {summary.n_worlds != null && `· Worlds: ${summary.n_worlds}`}{' '}
            {summary.sim_profile && `· Sim: ${summary.sim_profile}`}{' '}
            {summary.minutes_run_id && `· Minutes: ${summary.minutes_run_id}`}{' '}
            {summary.sim_run_id && `· Sim run: ${summary.sim_run_id}`}{' '}
            {summary.rates_run_id && `· Rates: ${summary.rates_run_id}`}{' '}
            {summary.run_as_of_ts && `· As of: ${formatTime(summary.run_as_of_ts)}`}{' '}
            {pinnedRunId && `· Pinned: ${pinnedRunId}`}{' '}
            {latestRunId && `· Latest: ${latestRunId}`}{' '}
            {showSim && summary.sim_available === false && '· sim_v2 unavailable'}
          </span>
        )}
      </section>

      {/* Player Change Alerts */}
      <PlayerChangeAlerts
        date={selectedDate}
        currentRunId={runId}
        blessedRunId={blessedRunId}
        latestRunId={latestRunId}
      />
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
          <GameView
            rows={rows}
            gameId={selectedGameId}
            date={selectedDate}
            runId={runId}
            readOnly
            onGameChange={setSelectedGameId}
            onOpenLateSwap={() => setActiveTab('entry-manager')}
            onRunCompleted={(nextRunId) => {
              if (nextRunId) {
                setRunId(nextRunId)
              }
              void fetchData(selectedDate, nextRunId ?? runId, modelId)
            }}
            onOverridesSaved={() => {
              void fetchData(selectedDate, runId, modelId)
            }}
          />
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
	                  {(['minutes_p10', 'minutes_final', 'minutes_p90', probSortKey] as SortKey[]).map(
	                    (key) => (
                      <th
                        key={key}
                        onClick={() => toggleSort(key)}
                        className="sortable"
                        title={
                          key === 'p_in_rotation' && modelId === 'rmh_v1_1'
                            ? `RMH: In rotation probability (minutes >= T=${selectedModel?.meta?.play_threshold ?? 5})`
                            : key === 'play_prob'
                              ? 'Model play_prob: base availability input to sim; sim_v3 also applies core locks and bench-zero rotation gating. Compare with p(MIN≥1) and p(MIN≥5) in sim columns.'
                            : undefined
                        }
                      >
                        {SORT_LABELS[key]}
                        {sortKey === key && (
                          <span>{sortDir === 'asc' ? ' ▲' : ' ▼'}</span>
                        )}
	                      </th>
	                    ),
	                  )}
                  {/* Ownership Columns - add My Own% before Own% */}
                  <th>Salary</th>
                  <th className="override-col">My Own%</th>
                  <th onClick={() => toggleSort('pred_own_pct')} className="sortable">
                    {SORT_LABELS['pred_own_pct']}
                    {sortKey === 'pred_own_pct' && <span>{sortDir === 'asc' ? ' ▲' : ' ▼'}</span>}
                  </th>
                  <th onClick={() => toggleSort('value')} className="sortable">
                    {SORT_LABELS['value']}
                    {sortKey === 'value' && <span>{sortDir === 'asc' ? ' ▲' : ' ▼'}</span>}
                  </th>
                  {showSim && (
                    <>
                      {/* My FPTS before sim FPTS columns */}
                      <th className="override-col">My FPTS</th>
                      {simFptsColumns.map(({ key, label }) => (
                        <th key={key} onClick={() => toggleSort(key)} className="sortable">
                          {label}
                          {sortKey === key && <span>{sortDir === 'asc' ? ' ▲' : ' ▼'}</span>}
                        </th>
                      ))}
                      {/* My MIN before sim MIN */}
                      <th className="override-col">My MIN</th>
                      {simStatColumns.map(({ key, label }) => (
                        <th
                          key={key}
                          onClick={() => toggleSort(key)}
                          className="sortable"
                          title={
                            key === 'minutes_sim_p_active'
                              ? 'Sim played probability: share of worlds with minutes >= 1.0.'
                              : key === 'sim_p_rotation'
                                ? 'Sim rotation probability: share of worlds with minutes >= 5.0.'
                                : key === 'bench_zero_p_zero'
                                  ? 'Bench-zero regime: per-player drop probability (conditional on being available) for low-minute players.'
                                  : key === 'play_prob_eff'
                                    ? 'Effective availability probability used for sim availability sampling after policy floors (core locks, probable floor, OUT=0).'
                                    : undefined
                          }
                        >
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
	                      <td>{formatMinutes(row.minutes_final ?? row.minutes_p50)}</td>
	                      <td>{formatMinutes(row.minutes_p90)}</td>
	                      <td>
	                        {formatPercent(
	                          (row as any)[probSortKey] as number | undefined,
	                        )}
                      </td>
                      {/* Ownership Columns */}
                      <td>{formatSalary(row.salary)}</td>
                      {/* My Own% input */}
                      <td className="override-input-cell">
                        <input
                          type="number"
                          className="override-input"
                          placeholder="—"
                          step={0.5}
                          min={0}
                          max={100}
                          value={overrides.get(String(row.player_id))?.own ?? ''}
                          onChange={e => updateOverride(
                            String(row.player_id),
                            'own',
                            e.target.value ? Number(e.target.value) : null
                          )}
                        />
                      </td>
                      <td className="ownership-cell">
                        {row.is_locked && <span className="lock-icon" title="Predictions locked">🔒</span>}
                        {formatOwnership(row.pred_own_pct)}
                      </td>
                      <td className="value-cell">{formatValue(row.value)}</td>
                      {showSim && (
                        <>
                          {/* My FPTS input */}
                          <td className="override-input-cell">
                            <input
                              type="number"
                              className="override-input"
                              placeholder="—"
                              step={0.5}
                              min={0}
                              max={100}
                              value={overrides.get(String(row.player_id))?.fpts ?? ''}
                              onChange={e => updateOverride(
                                String(row.player_id),
                                'fpts',
                                e.target.value ? Number(e.target.value) : null
                              )}
                            />
                          </td>
                          {simFptsColumns.map(({ key }) => (
                            <td key={key}>
                              {formatFpts(row[key as keyof PlayerRow] as number | undefined)}
                            </td>
                          ))}
                          {/* My MIN input */}
                          <td className="override-input-cell">
                            <input
                              type="number"
                              className="override-input"
                              placeholder="—"
                              step={1}
                              min={0}
                              max={48}
                              value={overrides.get(String(row.player_id))?.minutes ?? ''}
                              onChange={e => updateOverride(
                                String(row.player_id),
                                'minutes',
                                e.target.value ? Number(e.target.value) : null
                              )}
                            />
                          </td>
	                          {simStatColumns.map(({ key }) => (
	                            <td key={key}>
	                              {key === 'minutes_sim_uncond_mean'
	                                ? formatMinutesSim(resolveSimMinutesValue(row, key))
	                                : key === 'minutes_sim_p_active' ||
	                                    key === 'play_prob_eff' ||
	                                    key === 'sim_p_available' ||
	                                    key === 'sim_p_rotation' ||
	                                    key === 'bench_zero_p_zero'
	                                  ? formatPercent(row[key as keyof PlayerRow] as number | undefined)
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
