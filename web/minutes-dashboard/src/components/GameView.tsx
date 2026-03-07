import React, { useEffect, useMemo, useState } from 'react'
import {
    clearManualAvailabilityOverride,
    fetchOpsGame,
    fetchPlayerLastGames,
    OpsGamePlayer,
    OpsGameResponse,
    PlayerLastGameStat,
    upsertManualAvailabilityOverride,
} from '../api/manualAvailability'
import type { LiveSlateAnalyticsPlayer } from '../api/live'
import { PlayerDetailsDrawer, PlayerDrawerData } from './PlayerDetailsDrawer'
import { PlayerRow } from '../types'
import { formatFpts, formatMinutes, formatPercent, formatTime, getStatusBadge } from '../utils'
import './gameview.css'

type GameViewProps = {
    rows: PlayerRow[]
    gameId: string
    date?: string
    runId?: string | null
    readOnly?: boolean
    onGameChange?: (gameId: string) => void
    onOverridesSaved?: () => void
    onOpenLateSwap?: () => void
    onRunCompleted?: (runId: string | null) => void
    playerAnalyticsById?: Record<string, LiveSlateAnalyticsPlayer>
}

const toId = (value: unknown) => String(value ?? '')

const toMaybeNum = (value: unknown): number | null => {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
}

const projectedPointsForRow = (row: PlayerRow): number | null => {
    const baseline =
        toMaybeNum(row.sim_pts_mean_uncond) ??
        toMaybeNum(row.pts_mean_uncond) ??
        toMaybeNum(row.sim_pts_mean) ??
        toMaybeNum(row.pts_mean)
    if (baseline == null) return null
    return row.manual_override_type === 'force_out' ? 0 : baseline
}

export const GameView: React.FC<GameViewProps> = ({
    rows,
    gameId,
    date,
    readOnly,
    onOverridesSaved,
    onOpenLateSwap,
    playerAnalyticsById,
}) => {
    const targetDate = date || rows[0]?.game_date || new Date().toISOString().slice(0, 10)
    const [game, setGame] = useState<OpsGameResponse | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [message, setMessage] = useState<string | null>(null)
    const [busyPlayerId, setBusyPlayerId] = useState<string | null>(null)
    const [operator, setOperator] = useState(() => {
        if (typeof window === 'undefined') return ''
        return window.localStorage.getItem('manualAvailability.operator') || ''
    })
    const [reasonCode, setReasonCode] = useState('operator_report')
    const [reasonText, setReasonText] = useState('')
    const [sourceLabel, setSourceLabel] = useState('')
    const [selectedPlayerId, setSelectedPlayerId] = useState<string | null>(null)
    const [lastGames, setLastGames] = useState<PlayerLastGameStat[]>([])
    const [lastGamesLoading, setLastGamesLoading] = useState(false)
    const [lastGamesError, setLastGamesError] = useState<string | null>(null)
    const [manualDrawerOpen, setManualDrawerOpen] = useState(false)

    const asNumber = (value: unknown): number | null => {
        const parsed = Number(value)
        return Number.isFinite(parsed) ? parsed : null
    }

    const statValue = (record: Record<string, unknown> | undefined, keys: string[]) => {
        for (const key of keys) {
            if (record && key in record) {
                const parsed = asNumber(record[key])
                if (parsed != null) return parsed
            }
        }
        return null
    }

    const resolvedStat = (baseline: number | null, player: OpsGamePlayer) =>
        player.manual_override?.override_type === 'force_out' ? 0 : baseline

    useEffect(() => {
        if (typeof window === 'undefined') return
        window.localStorage.setItem('manualAvailability.operator', operator)
    }, [operator])

    useEffect(() => {
        let cancelled = false

        const load = async () => {
            if (!targetDate || !gameId) return
            setLoading(true)
            setError(null)
            try {
                const payload = await fetchOpsGame(targetDate, gameId)
                if (!cancelled) {
                    setGame(payload)
                }
            } catch (err) {
                if (!cancelled) {
                    setGame(null)
                    setError((err as Error).message)
                }
            } finally {
                if (!cancelled) {
                    setLoading(false)
                }
            }
        }

        void load()
        return () => {
            cancelled = true
        }
    }, [targetDate, gameId])

    const refresh = async () => {
        setLoading(true)
        setError(null)
        try {
            const payload = await fetchOpsGame(targetDate, gameId)
            setGame(payload)
        } catch (err) {
            setGame(null)
            setError((err as Error).message)
        } finally {
            setLoading(false)
        }
    }

    const teams = useMemo(() => {
        const byTeam = new Map<string, { teamKey: string; teamName: string; players: OpsGamePlayer[] }>()
        for (const player of game?.players ?? []) {
            const teamKey = String(player.team_id ?? player.team_tricode ?? 'team')
            const teamName = player.team_tricode || teamKey
            const bucket = byTeam.get(teamKey) ?? { teamKey, teamName, players: [] }
            bucket.players.push(player)
            byTeam.set(teamKey, bucket)
        }
        return Array.from(byTeam.values())
            .map((team) => ({
                ...team,
                players: [...team.players].sort((left, right) => {
                    const leftStarter = left.minutes_effective?.status?.toLowerCase() !== 'out' && Boolean(left.is_confirmed_starter || left.is_projected_starter)
                    const rightStarter = right.minutes_effective?.status?.toLowerCase() !== 'out' && Boolean(right.is_confirmed_starter || right.is_projected_starter)
                    if (leftStarter !== rightStarter) return leftStarter ? -1 : 1
                    return (right.minutes_effective?.minutes_final ?? right.minutes_effective?.minutes_p50 ?? 0) - (left.minutes_effective?.minutes_final ?? left.minutes_effective?.minutes_p50 ?? 0)
                }),
            }))
            .sort((left, right) => left.teamName.localeCompare(right.teamName))
    }, [game])

    const activeOverrides = useMemo(
        () => (game?.players ?? []).filter((player) => Boolean(player.manual_override?.active)),
        [game],
    )

    const selectedPlayer = useMemo(
        () => (game?.players ?? []).find((player) => player.player_id === selectedPlayerId) ?? null,
        [game, selectedPlayerId],
    )

    const gameContext = useMemo(() => {
        const gameRows = rows.filter((row) => toId(row.game_id) === gameId)
        if (!gameRows.length) return null

        const byTeam = new Map<string, { teamId: string; teamName: string; players: PlayerRow[] }>()
        for (const row of gameRows) {
            const teamId = toId(row.team_id)
            if (!teamId) continue
            const teamName = row.team_tricode || row.team_name || teamId
            const bucket = byTeam.get(teamId) ?? { teamId, teamName, players: [] }
            bucket.players.push(row)
            byTeam.set(teamId, bucket)
        }
        const teamsList = Array.from(byTeam.values())
        if (!teamsList.length) return null

        let away = teamsList[0]
        let home = teamsList[teamsList.length - 1]
        if (teamsList.length === 2) {
            const sample = gameRows[0]
            const total = toMaybeNum(sample?.total)
            const spreadHome = toMaybeNum(sample?.spread_home)
            if (total != null && spreadHome != null) {
                const homeImplied = total / 2 - spreadHome / 2
                const distance = (team: { players: PlayerRow[] }) => {
                    const implied = toMaybeNum(team.players[0]?.team_implied_total)
                    if (implied == null) return Number.POSITIVE_INFINITY
                    return Math.abs(implied - homeImplied)
                }
                const [a, b] = teamsList
                const aDist = distance(a)
                const bDist = distance(b)
                if (Number.isFinite(aDist) && Number.isFinite(bDist) && aDist !== bDist) {
                    home = aDist < bDist ? a : b
                    away = home === a ? b : a
                } else {
                    const sorted = [...teamsList].sort((left, right) => left.teamName.localeCompare(right.teamName))
                    away = sorted[0]
                    home = sorted[1]
                }
            }
        }

        const sample = gameRows[0]
        const total = toMaybeNum(sample?.total)
        const spreadHome = toMaybeNum(sample?.spread_home)
        const homeImplied = toMaybeNum(home.players[0]?.team_implied_total)
        const awayImplied = toMaybeNum(away.players[0]?.team_implied_total)
        const ourHomeImpliedRaw = home.players.reduce((sum, player) => sum + (projectedPointsForRow(player) ?? 0), 0)
        const ourAwayImpliedRaw = away.players.reduce((sum, player) => sum + (projectedPointsForRow(player) ?? 0), 0)
        const hasOurHomeImplied = home.players.some((player) => projectedPointsForRow(player) != null)
        const hasOurAwayImplied = away.players.some((player) => projectedPointsForRow(player) != null)
        const ourHomeImplied = hasOurHomeImplied ? ourHomeImpliedRaw : null
        const ourAwayImplied = hasOurAwayImplied ? ourAwayImpliedRaw : null
        const ourTotal =
            ourHomeImplied != null && ourAwayImplied != null ? ourHomeImplied + ourAwayImplied : null
        const ourSpreadHome =
            ourHomeImplied != null && ourAwayImplied != null ? ourAwayImplied - ourHomeImplied : null
        return {
            away,
            home,
            total,
            spreadHome,
            homeImplied,
            awayImplied,
            ourTotal,
            ourSpreadHome,
            ourHomeImplied,
            ourAwayImplied,
            tipTs: sample?.tip_ts,
        }
    }, [gameId, rows])

    const drawerData = useMemo<PlayerDrawerData | null>(() => {
        if (!selectedPlayer) return null
        const baseline = selectedPlayer.effective || {}
        const minutesBaseline = selectedPlayer.minutes_baseline || {}
        const minutesEffective = selectedPlayer.minutes_effective || {}

        const baselineMinutes = statValue(minutesBaseline, ['minutes_final', 'minutes_p50', 'minutes_p50_cond']) ?? statValue(baseline, ['minutes_p50', 'minutes_p50_cond', 'minutes_final'])
        const resolvedMinutes = statValue(minutesEffective as Record<string, unknown>, ['minutes_final', 'minutes_p50', 'minutes_p50_cond']) ?? baselineMinutes

        return {
            player: selectedPlayer,
            metrics: {
                minutes: {
                    baseline: baselineMinutes,
                    resolved: resolvedMinutes,
                    p10: statValue(minutesEffective as Record<string, unknown>, ['minutes_p10', 'minutes_p10_cond']),
                    p50: statValue(minutesEffective as Record<string, unknown>, ['minutes_p50', 'minutes_p50_cond']),
                    p90: statValue(minutesEffective as Record<string, unknown>, ['minutes_p90', 'minutes_p90_cond']),
                },
                fpts: {
                    baseline: statValue(baseline, ['fpts_sim_uncond_mean', 'dk_fpts_mean', 'sim_dk_fpts_mean']),
                    resolved: resolvedStat(statValue(baseline, ['fpts_sim_uncond_mean', 'dk_fpts_mean', 'sim_dk_fpts_mean']), selectedPlayer),
                    p10: statValue(baseline, ['fpts_sim_uncond_p10', 'dk_fpts_p10', 'sim_dk_fpts_p10']),
                    p50: statValue(baseline, ['fpts_sim_uncond_p50', 'dk_fpts_p50', 'sim_dk_fpts_p50']),
                    p90: statValue(baseline, ['fpts_sim_uncond_p90', 'dk_fpts_p90', 'sim_dk_fpts_p90']),
                },
                pts: {
                    baseline: statValue(baseline, ['pts_mean_uncond', 'pts_mean', 'sim_pts_mean_uncond', 'sim_pts_mean']),
                    resolved: resolvedStat(statValue(baseline, ['pts_mean_uncond', 'pts_mean', 'sim_pts_mean_uncond', 'sim_pts_mean']), selectedPlayer),
                },
                reb: {
                    baseline: statValue(baseline, ['reb_mean_uncond', 'reb_mean', 'sim_reb_mean_uncond', 'sim_reb_mean']),
                    resolved: resolvedStat(statValue(baseline, ['reb_mean_uncond', 'reb_mean', 'sim_reb_mean_uncond', 'sim_reb_mean']), selectedPlayer),
                },
                ast: {
                    baseline: statValue(baseline, ['ast_mean_uncond', 'ast_mean', 'sim_ast_mean_uncond', 'sim_ast_mean']),
                    resolved: resolvedStat(statValue(baseline, ['ast_mean_uncond', 'ast_mean', 'sim_ast_mean_uncond', 'sim_ast_mean']), selectedPlayer),
                },
                stl: {
                    baseline: statValue(baseline, ['stl_mean_uncond', 'stl_mean', 'sim_stl_mean_uncond', 'sim_stl_mean']),
                    resolved: resolvedStat(statValue(baseline, ['stl_mean_uncond', 'stl_mean', 'sim_stl_mean_uncond', 'sim_stl_mean']), selectedPlayer),
                },
                blk: {
                    baseline: statValue(baseline, ['blk_mean_uncond', 'blk_mean', 'sim_blk_mean_uncond', 'sim_blk_mean']),
                    resolved: resolvedStat(statValue(baseline, ['blk_mean_uncond', 'blk_mean', 'sim_blk_mean_uncond', 'sim_blk_mean']), selectedPlayer),
                },
                to: {
                    baseline: statValue(baseline, ['tov_mean_uncond', 'tov_mean', 'sim_tov_mean_uncond', 'sim_tov_mean']),
                    resolved: resolvedStat(statValue(baseline, ['tov_mean_uncond', 'tov_mean', 'sim_tov_mean_uncond', 'sim_tov_mean']), selectedPlayer),
                },
            },
            lastGames,
            lastGamesLoading,
            lastGamesError,
        }
    }, [selectedPlayer, lastGames, lastGamesLoading, lastGamesError])

    useEffect(() => {
        let cancelled = false
        if (!selectedPlayerId) {
            setLastGames([])
            setLastGamesError(null)
            setLastGamesLoading(false)
            return
        }
        setLastGamesLoading(true)
        setLastGamesError(null)
        void fetchPlayerLastGames(targetDate, selectedPlayerId, 5)
            .then((payload) => {
                if (cancelled) return
                setLastGames(payload.games ?? [])
            })
            .catch((err) => {
                if (cancelled) return
                setLastGames([])
                setLastGamesError((err as Error).message)
            })
            .finally(() => {
                if (!cancelled) setLastGamesLoading(false)
            })
        return () => {
            cancelled = true
        }
    }, [targetDate, selectedPlayerId])

    const actionDisabled = Boolean(readOnly) || !operator.trim()
    const hasSlateAnalytics = Boolean(playerAnalyticsById && Object.keys(playerAnalyticsById).length > 0)

    const formatSlatePct = (value?: number | null) =>
        value == null ? '—' : `${value.toFixed(1)}%`

    const formatSigned = (value?: number | null) =>
        value == null ? '—' : `${value >= 0 ? '+' : ''}${value.toFixed(1)}`

    const submitOverride = async (player: OpsGamePlayer, overrideType: 'force_out' | 'force_in') => {
        if (actionDisabled) return
        setBusyPlayerId(player.player_id)
        setError(null)
        setMessage(null)
        try {
            await upsertManualAvailabilityOverride({
                date: targetDate,
                game_id: gameId,
                player_id: player.player_id,
                player_name: player.player_name ?? undefined,
                team_id: player.team_id == null ? undefined : Number(player.team_id),
                team_tricode: player.team_tricode ?? undefined,
                override_type: overrideType,
                entered_by: operator.trim(),
                reason_code: reasonCode || undefined,
                reason_text: reasonText.trim() || undefined,
                source_label: sourceLabel.trim() || undefined,
            })
            setMessage(`${player.player_name || player.player_id} marked ${overrideType === 'force_out' ? 'OUT' : 'IN'}.`)
            await refresh()
            onOverridesSaved?.()
        } catch (err) {
            setError((err as Error).message)
        } finally {
            setBusyPlayerId(null)
        }
    }

    const clearOverride = async (player: OpsGamePlayer) => {
        const overrideId = player.manual_override?.override_id
        if (!overrideId || actionDisabled) return
        setBusyPlayerId(player.player_id)
        setError(null)
        setMessage(null)
        try {
            await clearManualAvailabilityOverride({
                date: targetDate,
                overrideId,
                clearedBy: operator.trim(),
            })
            setMessage(`Cleared manual override for ${player.player_name || player.player_id}.`)
            await refresh()
            onOverridesSaved?.()
        } catch (err) {
            setError((err as Error).message)
        } finally {
            setBusyPlayerId(null)
        }
    }

    return (
            <div className="gv-root">
                <div className="gv-header">
                    <div>
                        <h2>Manual Availability</h2>
                    <div className="gv-toolbar-note">
                        Only manual <code>IN</code> / <code>OUT</code> changes apply to canonical live inputs.
                    </div>
                </div>
                <div className="gv-chip-row">
                    {readOnly ? <span className="gv-chip gv-chip-readonly">Read only</span> : null}
                    {onOpenLateSwap ? (
                        <button type="button" className="gv-button" onClick={onOpenLateSwap}>
                            Open Late Swap
                        </button>
                    ) : null}
                    <button type="button" className="gv-button gv-button-small" onClick={() => setManualDrawerOpen(true)}>
                        Edit Manual Availability
                    </button>
                    <button type="button" className="gv-button gv-button-refresh" onClick={() => void refresh()} disabled={loading}>
                        {loading ? 'Refreshing…' : 'Refresh'}
                    </button>
                </div>
            </div>

            {manualDrawerOpen ? (
                <>
                    <div className="gv-manual-backdrop" onClick={() => setManualDrawerOpen(false)} />
                    <section className="gv-manual-drawer">
                        <div className="gv-manual-drawer-header">
                            <div>
                                <h3>Manual Availability Controls</h3>
                                <div className="gv-toolbar-note">
                                    Only manual <code>IN</code> / <code>OUT</code> changes apply to canonical live inputs.
                                </div>
                            </div>
                            <button type="button" className="gv-button gv-button-small" onClick={() => setManualDrawerOpen(false)}>
                                Close
                            </button>
                        </div>
                        <div className="gv-manual-drawer-fields">
                            <div className="gv-toolbar-fields">
                                <div className="gv-field">
                                    <label htmlFor="manual-availability-operator">Operator</label>
                                    <input
                                        id="manual-availability-operator"
                                        value={operator}
                                        onChange={(event) => setOperator(event.target.value)}
                                        placeholder="daniel"
                                        disabled={Boolean(readOnly)}
                                    />
                                </div>
                                <div className="gv-field">
                                    <label htmlFor="manual-availability-reason-code">Reason Code</label>
                                    <select
                                        id="manual-availability-reason-code"
                                        value={reasonCode}
                                        onChange={(event) => setReasonCode(event.target.value)}
                                        disabled={Boolean(readOnly)}
                                    >
                                        <option value="operator_report">Operator report</option>
                                        <option value="source_correction">Source correction</option>
                                        <option value="late_scratch">Late scratch</option>
                                        <option value="manual_reversal">Manual reversal</option>
                                    </select>
                                </div>
                                <div className="gv-field">
                                    <label htmlFor="manual-availability-source">Source Label</label>
                                    <input
                                        id="manual-availability-source"
                                        value={sourceLabel}
                                        onChange={(event) => setSourceLabel(event.target.value)}
                                        placeholder="twitter, beat writer, rotowire"
                                        disabled={Boolean(readOnly)}
                                    />
                                </div>
                                <div className="gv-field gv-field-wide">
                                    <label htmlFor="manual-availability-reason-text">Reason Note</label>
                                    <input
                                        id="manual-availability-reason-text"
                                        value={reasonText}
                                        onChange={(event) => setReasonText(event.target.value)}
                                        placeholder="Optional provenance or note"
                                        disabled={Boolean(readOnly)}
                                    />
                                </div>
                            </div>
                            <div className="gv-toolbar-note">
                                {operator.trim()
                                    ? `Actions will be recorded under ${operator.trim()}.`
                                    : 'Set an operator name before submitting manual overrides.'}
                            </div>
                        </div>
                    </section>
                </>
            ) : null}

            <section className="gv-summary">
                <div className="gv-summary-card gv-summary-combo">
                    <div className="gv-summary-combo-stats">
                        <div className="gv-summary-item">
                            <span className="gv-summary-label">Active Overrides</span>
                            <span className="gv-summary-value">{activeOverrides.length}</span>
                            <span className="gv-summary-subtle">
                                {activeOverrides.length ? 'Manual availability is currently affecting this game.' : 'No active manual availability overrides.'}
                            </span>
                        </div>
                        <div className="gv-summary-item">
                            <span className="gv-summary-label">Players</span>
                            <span className="gv-summary-value">{game?.players.length ?? 0}</span>
                            <span className="gv-summary-subtle">Immediate effective view from <code>/api/ops/game</code>.</span>
                        </div>
                    </div>
                </div>
            </section>

            {gameContext ? (
                <section className="gv-summary gv-summary-lines">
                    <div className="gv-summary-card">
                        <span className="gv-summary-label">Matchup</span>
                        <span className="gv-summary-value gv-summary-value-small">
                            {gameContext.away.teamName} @ {gameContext.home.teamName}
                        </span>
                        <span className="gv-summary-subtle">
                            {gameContext.tipTs ? formatTime(gameContext.tipTs, targetDate) : 'Tip TBD'}
                        </span>
                    </div>
                    <div className="gv-summary-card">
                        <span className="gv-summary-label">Game Total</span>
                        <span className="gv-summary-value">
                            {gameContext.ourTotal == null ? '—' : gameContext.ourTotal.toFixed(1)}
                        </span>
                        <span className="gv-summary-subtle">
                            Ours
                            <strong>{gameContext.ourTotal == null ? ' —' : ` ${gameContext.ourTotal.toFixed(1)}`}</strong>
                            {' · '}
                            Mkt
                            <strong>{gameContext.total == null ? ' —' : ` ${gameContext.total.toFixed(1)}`}</strong>
                        </span>
                    </div>
                    <div className="gv-summary-card">
                        <span className="gv-summary-label">Home Spread</span>
                        <span className="gv-summary-value">
                            {gameContext.ourSpreadHome == null ? '—' : gameContext.ourSpreadHome.toFixed(1)}
                        </span>
                        <span className="gv-summary-subtle">
                            Ours
                            <strong>{gameContext.ourSpreadHome == null ? ' —' : ` ${gameContext.ourSpreadHome.toFixed(1)}`}</strong>
                            {' · '}
                            Mkt
                            <strong>{gameContext.spreadHome == null ? ' —' : ` ${gameContext.spreadHome.toFixed(1)}`}</strong>
                            {' · '}
                            {gameContext.home.teamName}
                        </span>
                    </div>
                    <div className="gv-summary-card">
                        <span className="gv-summary-label">Implied Totals</span>
                        <span className="gv-summary-value gv-summary-value-small">
                            {gameContext.away.teamName} {gameContext.ourAwayImplied == null ? '—' : gameContext.ourAwayImplied.toFixed(1)}
                            {' · '}
                            {gameContext.home.teamName} {gameContext.ourHomeImplied == null ? '—' : gameContext.ourHomeImplied.toFixed(1)}
                        </span>
                        <span className="gv-summary-subtle">
                            Mkt: {gameContext.away.teamName} {gameContext.awayImplied == null ? '—' : gameContext.awayImplied.toFixed(1)} · {gameContext.home.teamName} {gameContext.homeImplied == null ? '—' : gameContext.homeImplied.toFixed(1)}
                        </span>
                    </div>
                </section>
            ) : null}

            {loading && !game ? <div className="gv-message">Loading game view…</div> : null}
            {!loading && !error && !teams.length ? <div className="gv-message gv-empty">No players found for this game.</div> : null}

            <section className="gv-team-grid">
                {teams.map((team) => (
                    <div key={team.teamKey} className="gv-team-panel">
                        <div className="gv-team-head">
                            <div>
                                <h3>{team.teamName}</h3>
                                <div className="gv-team-subtitle">{team.players.length} players</div>
                            </div>
                            <div className="gv-chip-row">
                                <span className="gv-chip gv-chip-active">
                                    {team.players.filter((player) => Boolean(player.manual_override?.active)).length} manual
                                </span>
                            </div>
                        </div>
                        <div className="gv-player-table-wrap">
                            <table className="gv-player-table">
                                <thead>
                                    <tr>
                                        <th>Player</th>
                                        <th>Status</th>
                                        <th>P50</th>
                                        <th>P90</th>
                                        <th>FPTS</th>
                                        {hasSlateAnalytics ? <th>Opt%</th> : null}
                                        {hasSlateAnalytics ? <th>Ceil Lev</th> : null}
                                        {hasSlateAnalytics ? <th>Boom%</th> : null}
                                        {hasSlateAnalytics ? <th>Bust%</th> : null}
                                        <th>Play</th>
                                        <th>Override</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {team.players.map((player) => {
                                        const effective = player.minutes_effective || {}
                                        const statusBadge = getStatusBadge(String(effective.status || player.status || ''))
                                        const isBusy = busyPlayerId === player.player_id
                                        const manualOverride = player.manual_override
                                        const baseline = player.effective || {}
                                        const slateMetrics = playerAnalyticsById?.[String(player.player_id)]
                                        return (
                                            <tr key={player.player_id} onClick={() => setSelectedPlayerId(player.player_id)}>
                                                <td>
                                                    <div className="gv-player-cell">
                                                        <div className="gv-player-name-row">
                                                            <strong>{player.player_name || player.player_id}</strong>
                                                            {player.is_confirmed_starter ? <span className="gv-chip gv-chip-starter">Confirmed</span> : null}
                                                            {!player.is_confirmed_starter && player.is_projected_starter ? <span className="gv-chip gv-chip-starter">Projected</span> : null}
                                                        </div>
                                                        {manualOverride ? (
                                                            <div className="gv-row-note">
                                                                {manualOverride.override_type === 'force_out' ? 'Force OUT' : 'Force IN'}
                                                                {manualOverride.entered_by ? ` · ${manualOverride.entered_by}` : ''}
                                                                {manualOverride.created_ts ? ` · ${formatTime(manualOverride.created_ts)}` : ''}
                                                            </div>
                                                        ) : null}
                                                    </div>
                                                </td>
                                                <td>{statusBadge ? <span className={`status-tag ${statusBadge.className}`}>{statusBadge.label}</span> : (effective.status || player.status || '-')}</td>
                                                <td>{formatMinutes(effective.minutes_p50 ?? undefined)}</td>
                                                <td>{formatMinutes(effective.minutes_p90 ?? undefined)}</td>
                                                <td>{formatFpts(asNumber(baseline.fpts_sim_uncond_mean ?? baseline.dk_fpts_mean ?? baseline.sim_dk_fpts_mean) ?? undefined)}</td>
                                                {hasSlateAnalytics ? <td>{formatSlatePct(slateMetrics?.optimal_pct)}</td> : null}
                                                {hasSlateAnalytics ? <td>{formatSigned(slateMetrics?.ceiling_leverage)}</td> : null}
                                                {hasSlateAnalytics ? <td>{formatSlatePct(slateMetrics?.boom_pct)}</td> : null}
                                                {hasSlateAnalytics ? <td>{formatSlatePct(slateMetrics?.bust_pct)}</td> : null}
                                                <td>{formatPercent(effective.play_prob ?? undefined)}</td>
                                                <td>
                                                    <div className="gv-inline-actions" onClick={(event) => event.stopPropagation()}>
                                                        <button
                                                            type="button"
                                                            className="gv-button gv-button-small gv-button-out"
                                                            disabled={actionDisabled || isBusy}
                                                            onClick={() => void submitOverride(player, 'force_out')}
                                                        >
                                                            OUT
                                                        </button>
                                                        <button
                                                            type="button"
                                                            className="gv-button gv-button-small gv-button-in"
                                                            disabled={actionDisabled || isBusy}
                                                            onClick={() => void submitOverride(player, 'force_in')}
                                                        >
                                                            IN
                                                        </button>
                                                        <button
                                                            type="button"
                                                            className="gv-button gv-button-small gv-button-clear"
                                                            disabled={actionDisabled || isBusy || !manualOverride?.override_id}
                                                            onClick={() => void clearOverride(player)}
                                                        >
                                                            Clear
                                                        </button>
                                                    </div>
                                                </td>
                                            </tr>
                                        )
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </div>
                ))}
            </section>
            <PlayerDetailsDrawer
                open={Boolean(drawerData)}
                data={drawerData}
                onClose={() => setSelectedPlayerId(null)}
            />
        </div>
    )
}
