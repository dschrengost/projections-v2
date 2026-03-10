import { useCallback, useEffect, useMemo, useState } from 'react'
import { apiUrl } from '../api/client'
import {
  fetchLiveSlateAnalytics,
  fetchLiveStatus,
  triggerLiveGameRerun,
  type LiveGameStatus,
  type LiveRunEvent,
  type LiveSlateAnalyticsPlayer,
  type LiveSlateAnalyticsResponse,
  type LiveStatusResponse,
} from '../api/live'
import { GameView } from '../components/GameView'
import type { MinutesResponse, PlayerRow } from '../types'
import { formatRunIdToEST, formatTime } from '../utils'
import './live.css'

type LivePageProps = {
  date: string
  onDateChange: (nextDate: string) => void
  selectedGameId: string | null
  onOpenGame: (gameId: string) => void
  onCloseGame: () => void
  onOpenLateSwap?: () => void
}

type GameCardData = {
  gameId: string
  label: string
  awayTeam: string
  homeTeam: string
  tipTs?: string
  status?: LiveGameStatus
}

type GameRerunState = {
  phase: 'triggering' | 'queued' | 'failed'
  message: string
  flowRunId?: string
}

const toId = (value: unknown) => String(value ?? '')

const toMaybeNum = (value: unknown): number | null => {
  if (value == null) return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

const inferAwayHome = (teams: Array<{ teamId: string; teamName: string; players: PlayerRow[] }>, gameRows: PlayerRow[]) => {
  if (teams.length !== 2) {
    const sorted = [...teams].sort((a, b) => a.teamName.localeCompare(b.teamName))
    return { away: sorted[0], home: sorted[1] }
  }

  const sample = gameRows[0]
  const total = toMaybeNum(sample?.total)
  const spreadHome = toMaybeNum(sample?.spread_home)

  if (total != null && spreadHome != null) {
    const homeImplied = total / 2 - spreadHome / 2
    const dist = (team: { players: PlayerRow[] }) => {
      const implied = toMaybeNum(team.players[0]?.team_implied_total)
      if (implied == null) return Number.POSITIVE_INFINITY
      return Math.abs(implied - homeImplied)
    }
    const [a, b] = teams
    const aDist = dist(a)
    const bDist = dist(b)
    if (Number.isFinite(aDist) && Number.isFinite(bDist) && aDist !== bDist) {
      return aDist < bDist ? { away: b, home: a } : { away: a, home: b }
    }
  }

  const sorted = [...teams].sort((a, b) => a.teamName.localeCompare(b.teamName))
  return { away: sorted[0], home: sorted[1] }
}

const deriveGames = (rows: PlayerRow[], statuses: LiveGameStatus[]): GameCardData[] => {
  const statusByGame = new Map(statuses.map((status) => [status.game_id, status]))
  const byGame = new Map<string, PlayerRow[]>()

  rows.forEach((row) => {
    const gameId = toId(row.game_id)
    if (!gameId) return
    const next = byGame.get(gameId) ?? []
    next.push(row)
    byGame.set(gameId, next)
  })

  const cards: GameCardData[] = []
  byGame.forEach((gameRows, gameId) => {
    const byTeam = new Map<string, { teamId: string; teamName: string; players: PlayerRow[] }>()
    gameRows.forEach((row) => {
      const teamId = toId(row.team_id)
      if (!teamId) return
      const existing = byTeam.get(teamId)
      if (existing) {
        existing.players.push(row)
        return
      }
      byTeam.set(teamId, {
        teamId,
        teamName: row.team_tricode || row.team_name || teamId,
        players: [row],
      })
    })

    const teams = Array.from(byTeam.values())
    if (!teams.length) return
    const { away, home } = inferAwayHome(teams, gameRows)
    cards.push({
      gameId,
      label: `${away.teamName} @ ${home.teamName}`,
      awayTeam: away.teamName,
      homeTeam: home.teamName,
      tipTs: gameRows[0]?.tip_ts || statusByGame.get(gameId)?.tip_ts || undefined,
      status: statusByGame.get(gameId),
    })
  })

  statusByGame.forEach((status, gameId) => {
    if (byGame.has(gameId)) return
    cards.push({
      gameId,
      label: `Game ${gameId}`,
      awayTeam: 'Away',
      homeTeam: 'Home',
      tipTs: status.tip_ts ?? undefined,
      status,
    })
  })

  return cards.sort((left, right) => (left.tipTs || '').localeCompare(right.tipTs || ''))
}

const formatTipCountdown = (minutesToTip?: number | null) => {
  if (minutesToTip == null) return 'n/a'
  if (minutesToTip <= 0) return 'Locked'
  const hours = Math.floor(minutesToTip / 60)
  const minutes = minutesToTip % 60
  return `${hours}h ${minutes}m`
}

const formatSourceValue = (label: string, status: LiveGameStatus) => {
  const freshness = status.source_freshness?.[label]
  if (!freshness) return 'missing'
  const source = freshness.source_used || 'unknown'
  const ts = freshness.latest_as_of_ts ? formatTime(freshness.latest_as_of_ts) : null
  return ts ? `${source} · ${ts}` : source
}

const eventTone = (status: string) => {
  if (status === 'published') return 'live-good'
  if (status === 'in_progress') return 'live-info'
  if (status === 'waiting_for_fresh_input') return 'live-muted'
  return 'live-warn'
}

const statusTone = (status: string) => {
  if (status === 'published') return 'live-good'
  if (status === 'in_progress') return 'live-info'
  if (status === 'waiting_for_fresh_input') return 'live-muted'
  if (status === 'blocked' || status === 'stale_relative_to_newer_input' || status === 'superseded') return 'live-warn'
  return 'live-muted'
}

const prettyStatus = (value: string) => value.replaceAll('_', ' ')
const formatPct = (value?: number | null) => (value == null ? '—' : `${value.toFixed(1)}%`)
const formatSigned = (value?: number | null) => (value == null ? '—' : `${value >= 0 ? '+' : ''}${value.toFixed(1)}`)

export default function LivePage({
  date,
  onDateChange,
  selectedGameId,
  onOpenGame,
  onCloseGame,
  onOpenLateSwap,
}: LivePageProps) {
  const [rows, setRows] = useState<PlayerRow[]>([])
  const [status, setStatus] = useState<LiveStatusResponse | null>(null)
  const [slateAnalytics, setSlateAnalytics] = useState<LiveSlateAnalyticsResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [rerunStateByGame, setRerunStateByGame] = useState<Record<string, GameRerunState>>({})

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const liveStatus = await fetchLiveStatus(date)
      const runParam = liveStatus.latest_published_run_id
        ? `&run_id=${encodeURIComponent(liveStatus.latest_published_run_id)}`
        : ''
      const minutesRes = await fetch(apiUrl(`/api/minutes?date=${encodeURIComponent(date)}${runParam}`))
      if (!minutesRes.ok) {
        const body = await minutesRes.json().catch(() => ({}))
        throw new Error((body as { detail?: string }).detail || `Failed to fetch minutes: ${minutesRes.status}`)
      }
      const minutesPayload = (await minutesRes.json()) as MinutesResponse
      let analyticsPayload: LiveSlateAnalyticsResponse | null = null
      try {
        analyticsPayload = await fetchLiveSlateAnalytics(date, {
          runId: liveStatus.latest_published_run_id,
          topN: 8,
        })
      } catch (analyticsErr) {
        console.warn('Failed to load slate analytics', analyticsErr)
      }
      setRows(minutesPayload.players ?? [])
      setStatus(liveStatus)
      setSlateAnalytics(analyticsPayload)
    } catch (err) {
      setRows([])
      setStatus(null)
      setSlateAnalytics(null)
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }, [date])

  const triggerRerunForGame = useCallback(
    async (game: GameCardData) => {
      const gameId = String(game.gameId || '').trim()
      if (!gameId) return
      const confirmed = window.confirm(`Trigger targeted rerun for ${game.label}?`)
      if (!confirmed) return
      setRerunStateByGame((prev) => ({
        ...prev,
        [gameId]: {
          phase: 'triggering',
          message: 'Queuing targeted rerun…',
        },
      }))
      try {
        const response = await triggerLiveGameRerun(date, gameId)
        const warning = response.validation_warning ? ` · ${response.validation_warning}` : ''
        setRerunStateByGame((prev) => ({
          ...prev,
          [gameId]: {
            phase: 'queued',
            message: `Queued targeted rerun${warning}`,
            flowRunId: response.flow_run_id,
          },
        }))
        window.setTimeout(() => {
          void load()
        }, 1000)
      } catch (err) {
        setRerunStateByGame((prev) => ({
          ...prev,
          [gameId]: {
            phase: 'failed',
            message: (err as Error).message,
          },
        }))
      }
    },
    [date, load],
  )

  useEffect(() => {
    void load()
  }, [load])

  const games = useMemo(() => deriveGames(rows, status?.games ?? []), [rows, status])
  const activeGame = useMemo(
    () => games.find((game) => game.gameId === selectedGameId) ?? null,
    [games, selectedGameId],
  )
  const playerAnalyticsById = useMemo(() => {
    const byId: Record<string, LiveSlateAnalyticsPlayer> = {}
    for (const player of slateAnalytics?.players ?? []) {
      if (!player.player_id) continue
      byId[String(player.player_id)] = player
    }
    return byId
  }, [slateAnalytics])

  const currentTruthLabel = status?.latest_published_run_id
    ? formatRunIdToEST(status.latest_published_run_id) || status.latest_published_run_id
    : 'None'

  const renderRunEvent = (event: LiveRunEvent) => (
    <div key={event.run_id} className="live-event">
      <div className="live-event-head">
        <span className={`live-chip ${eventTone(event.status)}`}>{prettyStatus(event.status)}</span>
        <strong>{formatRunIdToEST(event.run_id) || event.run_id}</strong>
      </div>
      <div className="live-event-meta">
        <span>{event.as_of_ts ? formatTime(event.as_of_ts) : 'No as-of'}</span>
        <span>{event.reason ? prettyStatus(event.reason) : 'No reason'}</span>
      </div>
    </div>
  )

  if (selectedGameId && activeGame) {
    return (
      <div className="live-page">
        <section className="live-game-shell">
          <div className="live-game-toolbar">
            <button type="button" className="live-back-button" onClick={onCloseGame}>
              Back to Live
            </button>
            <div className="live-game-context">
              <div className="live-game-title">{activeGame.label}</div>
              <div className="muted">
                Published truth: {currentTruthLabel}
                {status?.candidate_run_id ? ` · Candidate: ${formatRunIdToEST(status.candidate_run_id) || status.candidate_run_id}` : ''}
              </div>
            </div>
            <button type="button" className="live-refresh-button" onClick={() => void load()} disabled={loading}>
              {loading ? 'Refreshing…' : 'Refresh'}
            </button>
          </div>
          <GameView
            rows={rows}
            gameId={selectedGameId}
            date={date}
            playerAnalyticsById={playerAnalyticsById}
            onGameChange={onOpenGame}
            onOverridesSaved={() => void load()}
            onOpenLateSwap={onOpenLateSwap}
          />
        </section>
      </div>
    )
  }

  return (
    <div className="live-page">
      <header className="live-header">
        <div>
          <h1>Live</h1>
          <p className="subtitle">Current truth first. Candidate state stays visible, but secondary.</p>
        </div>
        <div className="live-header-actions">
          <label>
            Date
            <input type="date" value={date} onChange={(event) => onDateChange(event.target.value)} />
          </label>
          <button type="button" className="live-refresh-button" onClick={() => void load()} disabled={loading}>
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </header>

      {error ? <div className="live-alert live-alert-error">Failed to load live status: {error}</div> : null}

      <section className="live-rail">
        <div className="live-rail-card">
          <span className="live-rail-label">Published run</span>
          <strong>{currentTruthLabel}</strong>
          <span className="muted">{status?.latest_published_as_of_ts ? formatTime(status.latest_published_as_of_ts) : 'No published pointer'}</span>
        </div>
        <div className="live-rail-card">
          <span className="live-rail-label">Candidate</span>
          <strong>{status?.candidate_run_id ? formatRunIdToEST(status.candidate_run_id) || status.candidate_run_id : 'None'}</strong>
          <span className={`live-chip ${statusTone(status?.candidate_status || 'waiting_for_fresh_input')}`}>
            {prettyStatus(status?.candidate_status || 'waiting_for_fresh_input')}
          </span>
        </div>
        <div className="live-rail-card">
          <span className="live-rail-label">Publish status</span>
          <strong>{prettyStatus(status?.publish_status || 'unknown')}</strong>
          <span className="muted">{status?.candidate_status_reason ? prettyStatus(status.candidate_status_reason) : 'No candidate reason'}</span>
        </div>
        <div className="live-rail-card">
          <span className="live-rail-label">Last update</span>
          <strong>{status?.updated_at ? formatTime(status.updated_at) : 'Unknown'}</strong>
          <span className="muted">{games.length} games on board</span>
        </div>
      </section>

      <section className="live-run-strip">
        <div className="live-section-head">
          <h2>Recent runs</h2>
          <span className="muted">Published, waiting, and in-flight runs from the live control plane.</span>
        </div>
        <div className="live-event-strip">
          {(status?.run_event_strip ?? []).map(renderRunEvent)}
          {!loading && (status?.run_event_strip.length ?? 0) === 0 ? <div className="muted">No run history found.</div> : null}
        </div>
      </section>

      <section className="live-analytics">
        <div className="live-section-head">
          <h2>Slate Analytics</h2>
          <span className="muted">
            {slateAnalytics
              ? `DG ${slateAnalytics.draft_group_id} · ${
                  slateAnalytics.generated_at ? `Updated ${formatTime(slateAnalytics.generated_at)}` : 'Cached'
                }`
              : 'No slate analytics available'}
          </span>
        </div>
        <div className="live-analytics-grid">
          {([
            { key: 'optimal_pct', label: 'Optimal %', formatter: formatPct },
            { key: 'ceiling_leverage', label: 'Ceiling Leverage', formatter: formatSigned },
            { key: 'boom_pct', label: 'Boom %', formatter: formatPct },
            { key: 'bust_pct', label: 'Bust %', formatter: formatPct },
          ] as const).map((metric) => {
            const rowsForMetric = slateAnalytics?.leaders?.[metric.key] ?? []
            return (
              <div key={metric.key} className="live-analytics-card">
                <div className="live-analytics-card-head">
                  <strong>{metric.label}</strong>
                </div>
                <div className="live-analytics-list">
                  {rowsForMetric.slice(0, 8).map((player, idx) => (
                    <div key={`${metric.key}-${player.player_id}-${idx}`} className="live-analytics-row">
                      <span className="live-analytics-rank">#{idx + 1}</span>
                      <span className="live-analytics-name">
                        {player.name}
                        <span className="muted"> {player.team ?? ''}</span>
                      </span>
                      <span className="live-analytics-value">{metric.formatter(player[metric.key])}</span>
                    </div>
                  ))}
                  {!loading && rowsForMetric.length === 0 ? <div className="muted">No data.</div> : null}
                </div>
              </div>
            )
          })}
        </div>
      </section>

      <section className="live-board">
        <div className="live-section-head">
          <h2>Game board</h2>
          <span className="muted">Highest-risk games should stand out without opening a drawer.</span>
        </div>

        <div className="live-board-grid">
          {games.map((game) => {
            const gameStatus = game.status
            const tone = statusTone(status?.candidate_status || 'waiting_for_fresh_input')
            const rerunState = rerunStateByGame[game.gameId]
            const rerunBusy = rerunState?.phase === 'triggering'
            return (
              <article
                key={game.gameId}
                className="live-game-card"
                role="button"
                tabIndex={0}
                onClick={() => onOpenGame(game.gameId)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault()
                    onOpenGame(game.gameId)
                  }
                }}
              >
                <div className="live-game-card-head">
                  <div>
                    <div className="live-matchup">{game.label}</div>
                    <div className="muted">
                      {game.tipTs ? formatTime(game.tipTs) : 'Tip unknown'} · {formatTipCountdown(gameStatus?.minutes_to_tip)}
                    </div>
                  </div>
                  <div className="live-game-card-actions">
                    <button
                      type="button"
                      className="live-game-rerun-btn"
                      onClick={(event) => {
                        event.stopPropagation()
                        void triggerRerunForGame(game)
                      }}
                      disabled={rerunBusy}
                    >
                      {rerunBusy ? 'Queuing…' : 'Rebuild game'}
                    </button>
                    <span className={`live-chip ${tone}`}>
                      {prettyStatus(status?.candidate_status || 'waiting_for_fresh_input')}
                    </span>
                  </div>
                </div>

                <div className="live-game-card-badges">
                  {gameStatus?.manual_override_active ? (
                    <span className="live-badge live-badge-override">
                      Override{(gameStatus.manual_override_count ?? 0) > 1 ? ` x${gameStatus.manual_override_count}` : ''}
                    </span>
                  ) : null}
                  {gameStatus?.warning_badges.map((badge) => (
                    <span key={`${game.gameId}-${badge}`} className="live-badge">
                      {prettyStatus(badge)}
                    </span>
                    ))}
                </div>
                {rerunState ? (
                  <div className={`live-rerun-status ${rerunState.phase === 'failed' ? 'live-rerun-status-error' : ''}`}>
                    <span>{rerunState.message}</span>
                    {rerunState.flowRunId ? <span className="muted">Run {formatRunIdToEST(rerunState.flowRunId) || rerunState.flowRunId}</span> : null}
                  </div>
                ) : null}

                <div className="live-source-grid">
                  <div>
                    <span className="live-source-label">Injuries</span>
                    <span>{gameStatus ? formatSourceValue('injuries', gameStatus) : 'missing'}</span>
                  </div>
                  <div>
                    <span className="live-source-label">Lineups</span>
                    <span>{gameStatus ? formatSourceValue('lineups', gameStatus) : 'missing'}</span>
                  </div>
                  <div>
                    <span className="live-source-label">Odds</span>
                    <span>{gameStatus ? formatSourceValue('odds', gameStatus) : 'missing'}</span>
                  </div>
                  <div>
                    <span className="live-source-label">Props</span>
                    <span>{gameStatus ? formatSourceValue('props', gameStatus) : 'missing'}</span>
                  </div>
                </div>

                <div className="live-card-footer">
                  <span>{gameStatus?.latest_effective_status_source_summary || 'No source summary'}</span>
                  <span>
                    {gameStatus?.affected_by_change_set ? 'Changed' : 'Unchanged'}
                    {gameStatus?.rerun_targeted ? ' · Targeted' : ''}
                  </span>
                </div>
              </article>
            )
          })}

          {!loading && games.length === 0 ? (
            <div className="live-alert">No live slate artifacts found for {date}.</div>
          ) : null}
        </div>
      </section>
    </div>
  )
}
