import React, { useState, useCallback } from 'react'
import { PlayerRow } from '../types'
import { apiUrl } from '../api/client'

type PlayerOpsPanelProps = {
    players: PlayerRow[]
    teams: string[]
    gameId: string
    date: string
    onOverridesSaved?: () => void
}

type SaveStatus = 'idle' | 'saving' | 'saved' | 'error'

export const PlayerOpsPanel: React.FC<PlayerOpsPanelProps> = ({
    players,
    teams,
    gameId,
    date,
    onOverridesSaved
}) => {
    const [deltas, setDeltas] = useState<Record<string, number>>({})
    const [roles, setRoles] = useState<Record<string, string>>({})
    const [saveStatus, setSaveStatus] = useState<SaveStatus>('idle')
    const [runningWorlds, setRunningWorlds] = useState(false)
    const [worldsMessage, setWorldsMessage] = useState<string | null>(null)

    const handleDeltaChange = (playerId: string | number, val: string) => {
        const num = parseFloat(val)
        const pid = String(playerId)
        setDeltas(prev => {
            const next = { ...prev }
            if (isNaN(num) || num === 0) {
                delete next[pid]
            } else {
                next[pid] = num
            }
            return next
        })
        setSaveStatus('idle')
    }

    const handleRoleChange = (playerId: string | number, val: string) => {
        const pid = String(playerId)
        const normalized = val.trim().toLowerCase()
        setRoles(prev => {
            const next = { ...prev }
            if (!normalized) {
                delete next[pid]
            } else {
                next[pid] = normalized
            }
            return next
        })
        setSaveStatus('idle')
    }

    const saveOverrides = useCallback(async () => {
        if (Object.keys(deltas).length === 0 && Object.keys(roles).length === 0) return

        setSaveStatus('saving')
        try {
            const byPlayer: Record<string, Record<string, unknown>> = {}
            for (const [playerId, delta] of Object.entries(deltas)) {
                byPlayer[playerId] = { ...(byPlayer[playerId] ?? {}), minutes_delta: delta }
            }
            for (const [playerId, role] of Object.entries(roles)) {
                byPlayer[playerId] = { ...(byPlayer[playerId] ?? {}), ops_depth_role: role }
            }

            const updates = Object.entries(byPlayer).map(([playerId, fields]) => ({
                game_id: gameId,
                player_id: playerId,
                ...fields,
            }))

            const res = await fetch(apiUrl('/api/ops/overrides'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ date, updates }),
            })

            if (!res.ok) {
                const errText = await res.text()
                throw new Error(errText || 'Failed to save overrides')
            }

            setSaveStatus('saved')
            onOverridesSaved?.()
        } catch (err) {
            console.error('Failed to save overrides:', err)
            setSaveStatus('error')
        }
    }, [deltas, roles, gameId, date, onOverridesSaved])

    const runWorlds = useCallback(async () => {
        setRunningWorlds(true)
        setWorldsMessage(null)
        try {
            // game_id needs to be numeric for the API
            const numericGameId = parseInt(gameId, 10)
            if (isNaN(numericGameId)) {
                throw new Error('Invalid game_id')
            }

            const res = await fetch(apiUrl('/api/ops/run-worlds'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    date,
                    game_id: numericGameId,
                    pin: true,
                    background: true,
                }),
            })

            if (!res.ok) {
                const errText = await res.text()
                throw new Error(errText || 'Failed to trigger worlds patch')
            }

            const data = await res.json()
            setWorldsMessage(data.message || 'Sim re-run triggered')
        } catch (err) {
            console.error('Failed to run worlds:', err)
            setWorldsMessage(`Error: ${(err as Error).message}`)
        } finally {
            setRunningWorlds(false)
        }
    }, [gameId, date])

    // Group players by team
    const groupedPlayers = React.useMemo(() => {
        const groups: Record<string, PlayerRow[]> = {}
        teams.forEach(t => groups[t] = [])
        players.forEach(p => {
            const team = p.team_tricode || p.team_name || String(p.team_id)
            if (groups[team]) {
                groups[team].push(p)
            }
        })
        // Sort by minutes within each team
        Object.values(groups).forEach(arr => {
            arr.sort((a, b) => (b.minutes_final ?? b.minutes_p50 ?? 0) - (a.minutes_final ?? a.minutes_p50 ?? 0))
        })
        return groups
    }, [players, teams])

    const totalDeltas = Object.values(deltas).reduce((sum, d) => sum + d, 0)
    const hasPendingChanges = Object.keys(deltas).length > 0 || Object.keys(roles).length > 0

    return (
        <div className="sidebar-card ops-panel">
            <div className="ops-header">
                <h3 className="sidebar-card-title">⚙️ Manual Overrides</h3>
                <div className="ops-header-actions">
                    {totalDeltas !== 0 && (
                        <span className={`delta-total ${totalDeltas > 0 ? 'positive' : 'negative'}`}>
                            {totalDeltas > 0 ? '+' : ''}{totalDeltas.toFixed(1)}
                        </span>
                    )}
                </div>
            </div>

            <div className="ops-content">
                {teams.map(team => {
                    const teamPlayers = groupedPlayers[team] || []
                    const teamDelta = teamPlayers.reduce((sum, p) => {
                        const d = deltas[String(p.player_id)] || 0
                        return sum + d
                    }, 0)
                    const teamTotal = teamPlayers.reduce((sum, p) => sum + (p.minutes_final ?? p.minutes_p50 ?? 0), 0) + teamDelta

                    return (
                        <div key={team} className="ops-team-group">
                            <div className="ops-team-header">
                                <span>{team}</span>
                                <span className={`team-total ${Math.abs(teamTotal - 240) < 1 ? 'balanced' : teamTotal > 240 ? 'over' : 'under'}`}>
                                    {teamTotal.toFixed(0)} min
                                </span>
                            </div>
                            <table className="ops-player-table">
                                <tbody>
                                    {teamPlayers.map(p => {
                                        const pid = String(p.player_id)
                                        const delta = deltas[pid]
                                        const role = roles[pid] ?? (p.ops_depth_role ?? '')
                                        const baseMin = p.minutes_final ?? p.minutes_p50 ?? 0
                                        const finalMin = baseMin + (delta || 0)
                                        const isStarter = p.is_confirmed_starter || p.is_projected_starter

                                        return (
                                            <tr key={pid} className={(delta !== undefined || roles[pid] !== undefined) ? 'modified' : ''}>
                                                <td className="name-cell">
                                                    <div className="name-row">
                                                        <span className="name" title={p.player_name}>{p.player_name}</span>
                                                        {isStarter && <span className="starter-dot" title="Starter">S</span>}
                                                    </div>
                                                </td>
                                                <td className="role-cell">
                                                    <select
                                                        className="role-select"
                                                        value={role}
                                                        onChange={(e) => handleRoleChange(p.player_id ?? pid, e.target.value)}
                                                        title="Manual role override"
                                                    >
                                                        <option value="">Role…</option>
                                                        <option value="starter">Starter</option>
                                                        <option value="rotation">Rotation</option>
                                                        <option value="deep_bench">Deep bench</option>
                                                        <option value="out">OUT</option>
                                                    </select>
                                                </td>
                                                <td className="base-minutes-cell">
                                                    {baseMin.toFixed(1)}
                                                </td>
                                                <td className="delta-cell">
                                                    <input
                                                        type="number"
                                                        className="delta-input"
                                                        placeholder="±"
                                                        step="0.5"
                                                        value={delta ?? ''}
                                                        onChange={(e) => handleDeltaChange(p.player_id, e.target.value)}
                                                    />
                                                </td>
                                                <td className={`final-minutes-cell ${delta !== undefined ? 'adjusted' : ''}`}>
                                                    {delta !== undefined ? finalMin.toFixed(1) : ''}
                                                </td>
                                            </tr>
                                        )
                                    })}
                                </tbody>
                            </table>
                        </div>
                    )
                })}
            </div>

            {/* Action Buttons */}
            <div className="ops-actions">
                <button
                    className={`ops-save-btn ${saveStatus}`}
                    onClick={saveOverrides}
                    disabled={!hasPendingChanges || saveStatus === 'saving'}
                >
                    {saveStatus === 'saving' ? 'Saving...' :
                     saveStatus === 'saved' ? '✓ Saved' :
                     saveStatus === 'error' ? '✗ Error' :
                     'Save Overrides'}
                </button>
                <button
                    className="ops-run-btn"
                    onClick={runWorlds}
                    disabled={runningWorlds}
                    title="Re-run sim for this game with current overrides"
                >
                    {runningWorlds ? 'Running...' : '▶ Re-run Sim'}
                </button>
            </div>
            {worldsMessage && (
                <div className={`ops-worlds-message ${worldsMessage.startsWith('Error') ? 'error' : 'success'}`}>
                    {worldsMessage}
                </div>
            )}
        </div>
    )
}
