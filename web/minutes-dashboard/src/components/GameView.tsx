import React, { useEffect, useMemo, useState } from 'react'
import {
    clearManualAvailabilityOverride,
    fetchOpsGame,
    OpsGamePlayer,
    OpsGameResponse,
    upsertManualAvailabilityOverride,
} from '../api/manualAvailability'
import { PlayerRow } from '../types'
import { formatMinutes, formatPercent, formatTime, getStatusBadge } from '../utils'
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
}

export const GameView: React.FC<GameViewProps> = ({
    rows,
    gameId,
    date,
    readOnly,
    onOverridesSaved,
    onOpenLateSwap,
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

    const actionDisabled = Boolean(readOnly) || !operator.trim()

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
                    <button type="button" className="gv-button gv-button-refresh" onClick={() => void refresh()} disabled={loading}>
                        {loading ? 'Refreshing…' : 'Refresh'}
                    </button>
                </div>
            </div>

            <section className="gv-toolbar">
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
            </section>

            {error ? <div className="gv-message gv-message-error">{error}</div> : null}
            {message ? <div className="gv-message">{message}</div> : null}

            <section className="gv-summary">
                <div className="gv-summary-card">
                    <span className="gv-summary-label">Active Overrides</span>
                    <span className="gv-summary-value">{activeOverrides.length}</span>
                    <span className="gv-summary-subtle">
                        {activeOverrides.length ? 'Manual availability is currently affecting this game.' : 'No active manual availability overrides.'}
                    </span>
                </div>
                <div className="gv-summary-card">
                    <span className="gv-summary-label">Players</span>
                    <span className="gv-summary-value">{game?.players.length ?? 0}</span>
                    <span className="gv-summary-subtle">Immediate effective view from <code>/api/ops/game</code>.</span>
                </div>
            </section>

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
                        <div className="gv-player-list">
                            {team.players.map((player) => {
                                const effective = player.minutes_effective || {}
                                const statusBadge = getStatusBadge(String(effective.status || player.status || ''))
                                const isBusy = busyPlayerId === player.player_id
                                const manualOverride = player.manual_override
                                return (
                                    <article key={player.player_id} className="gv-player-card">
                                        <div className="gv-player-top">
                                            <div>
                                                <div className="gv-player-name">{player.player_name || player.player_id}</div>
                                                <div className="gv-player-meta">
                                                    <span>P50 {formatMinutes(effective.minutes_p50 ?? undefined)}</span>
                                                    <span>Final {formatMinutes(effective.minutes_final ?? undefined)}</span>
                                                    <span>Play {formatPercent(effective.play_prob ?? undefined)}</span>
                                                </div>
                                            </div>
                                            <div className="gv-chip-row">
                                                {statusBadge ? (
                                                    <span className={`status-tag ${statusBadge.className}`} title={statusBadge.title}>
                                                        {statusBadge.label}
                                                    </span>
                                                ) : null}
                                                {player.is_confirmed_starter ? <span className="gv-chip gv-chip-starter">Confirmed starter</span> : null}
                                                {!player.is_confirmed_starter && player.is_projected_starter ? <span className="gv-chip gv-chip-starter">Projected starter</span> : null}
                                                {manualOverride?.override_type === 'force_out' ? <span className="gv-chip gv-chip-out">Manual out</span> : null}
                                                {manualOverride?.override_type === 'force_in' ? <span className="gv-chip gv-chip-manual">Manual in</span> : null}
                                            </div>
                                        </div>

                                        {manualOverride ? (
                                            <div className="gv-override-copy">
                                                <span>{manualOverride.override_type === 'force_out' ? 'Force OUT' : 'Force IN'}</span>
                                                {manualOverride.entered_by ? <span>by {manualOverride.entered_by}</span> : null}
                                                {manualOverride.reason_code ? <span>{manualOverride.reason_code}</span> : null}
                                                {manualOverride.source_label ? <span>{manualOverride.source_label}</span> : null}
                                                {manualOverride.created_ts ? <span>{formatTime(manualOverride.created_ts)}</span> : null}
                                            </div>
                                        ) : null}
                                        {effective.manual_override_reason_text ? (
                                            <div className="gv-card-note">
                                                <span>{effective.manual_override_reason_text}</span>
                                            </div>
                                        ) : null}

                                        <div className="gv-player-actions">
                                            <button
                                                type="button"
                                                className="gv-button gv-button-out"
                                                disabled={actionDisabled || isBusy}
                                                onClick={() => void submitOverride(player, 'force_out')}
                                            >
                                                {isBusy && manualOverride?.override_type !== 'force_in' ? 'Saving…' : 'Mark OUT'}
                                            </button>
                                            <button
                                                type="button"
                                                className="gv-button gv-button-in"
                                                disabled={actionDisabled || isBusy}
                                                onClick={() => void submitOverride(player, 'force_in')}
                                            >
                                                {isBusy && manualOverride?.override_type === 'force_in' ? 'Saving…' : 'Mark IN'}
                                            </button>
                                            <button
                                                type="button"
                                                className="gv-button gv-button-clear"
                                                disabled={actionDisabled || isBusy || !manualOverride?.override_id}
                                                onClick={() => void clearOverride(player)}
                                            >
                                                Clear
                                            </button>
                                        </div>
                                    </article>
                                )
                            })}
                        </div>
                    </div>
                ))}
            </section>
        </div>
    )
}
