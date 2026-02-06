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
    const [targets, setTargets] = useState<Record<string, number>>({})
    const [locks, setLocks] = useState<Record<string, boolean>>({})
    const [roles, setRoles] = useState<Record<string, string>>({})
    const [saveStatus, setSaveStatus] = useState<SaveStatus>('idle')
    const [saveError, setSaveError] = useState<string | null>(null)
    const [resetting, setResetting] = useState(false)
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

    const handleTargetChange = (playerId: string | number, val: string) => {
        const num = parseFloat(val)
        const pid = String(playerId)
        setTargets(prev => {
            const next = { ...prev }
            if (isNaN(num)) {
                delete next[pid]
            } else {
                next[pid] = num
            }
            return next
        })
        setSaveStatus('idle')
    }

    const handleLockToggle = (playerId: string | number, checked: boolean) => {
        const pid = String(playerId)
        setLocks(prev => ({ ...prev, [pid]: checked }))
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
        const hasChanges =
            Object.keys(deltas).length > 0 ||
            Object.keys(targets).length > 0 ||
            Object.keys(locks).length > 0 ||
            Object.keys(roles).length > 0
        if (!hasChanges) return

        setSaveStatus('saving')
        setSaveError(null)
        try {
            const baseMinutesById: Record<string, number> = {}
            for (const p of players) {
                const pid = String(p.player_id)
                baseMinutesById[pid] = p.minutes_final ?? p.minutes_p50 ?? 0
            }

            const byPlayer: Record<string, Record<string, unknown>> = {}
            for (const [playerId, delta] of Object.entries(deltas)) {
                const baseMin = baseMinutesById[playerId] ?? 0
                byPlayer[playerId] = {
                    ...(byPlayer[playerId] ?? {}),
                    minutes_target: baseMin + delta,
                    minutes_lock: true,
                }
            }
            for (const [playerId, target] of Object.entries(targets)) {
                byPlayer[playerId] = {
                    ...(byPlayer[playerId] ?? {}),
                    minutes_target: target,
                    minutes_lock: true,
                }
            }
            for (const [playerId, lockVal] of Object.entries(locks)) {
                byPlayer[playerId] = { ...(byPlayer[playerId] ?? {}), minutes_lock: lockVal }
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
                try {
                    const parsed = JSON.parse(errText) as { detail?: string }
                    throw new Error(parsed?.detail || errText || 'Failed to save overrides')
                } catch {
                    throw new Error(errText || 'Failed to save overrides')
                }
            }

            setSaveStatus('saved')
            setDeltas({})
            setTargets({})
            setLocks({})
            setRoles({})
            onOverridesSaved?.()
        } catch (err) {
            console.error('Failed to save overrides:', err)
            setSaveStatus('error')
            setSaveError((err as Error).message)
        }
    }, [deltas, targets, locks, roles, gameId, date, players, onOverridesSaved])

    const hasSavedOverrides = React.useMemo(() => {
        return players.some(p => {
            if (p.ops_override_applied) return true
            if ((p.ops_depth_role ?? '').trim() !== '') return true
            if (p.minutes_target !== null && p.minutes_target !== undefined) return true
            if (p.minutes_lock !== null && p.minutes_lock !== undefined) return true
            if (p.minutes_delta !== null && p.minutes_delta !== undefined) return true
            return false
        })
    }, [players])

    const resetOverrides = useCallback(async () => {
        const confirmed = window.confirm('Reset ALL manual overrides for this game?\n\nThis clears role overrides + minutes targets/locks and cannot be undone.')
        if (!confirmed) return

        setResetting(true)
        setSaveError(null)
        setWorldsMessage(null)

        try {
            const qs = new URLSearchParams({ date, game_id: String(gameId) }).toString()
            const res = await fetch(apiUrl(`/api/ops/overrides?${qs}`), { method: 'DELETE' })
            if (!res.ok) {
                const errText = await res.text()
                try {
                    const parsed = JSON.parse(errText) as { detail?: string }
                    throw new Error(parsed?.detail || errText || 'Failed to reset overrides')
                } catch {
                    throw new Error(errText || 'Failed to reset overrides')
                }
            }

            setDeltas({})
            setTargets({})
            setLocks({})
            setRoles({})
            setSaveStatus('idle')
            setWorldsMessage('Overrides cleared. Re-run sim if needed.')
            onOverridesSaved?.()
        } catch (err) {
            console.error('Failed to reset overrides:', err)
            setSaveError((err as Error).message)
        } finally {
            setResetting(false)
        }
    }, [date, gameId, onOverridesSaved])

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
    const hasPendingChanges =
        Object.keys(deltas).length > 0 ||
        Object.keys(targets).length > 0 ||
        Object.keys(locks).length > 0 ||
        Object.keys(roles).length > 0
    const canReset = hasPendingChanges || hasSavedOverrides

    const plannedMinutes = (p: PlayerRow) => {
        const pid = String(p.player_id)
        const baseMin = p.minutes_final ?? p.minutes_p50 ?? 0
        const role = roles[pid] ?? (p.ops_depth_role ?? '')
        const roleNorm = role.trim().toLowerCase()
        if (roleNorm === 'out') return 0

        const targetVal = targets[pid]
        if (targetVal !== undefined && !isNaN(targetVal)) return targetVal

        const deltaVal = deltas[pid]
        if (deltaVal !== undefined) return baseMin + deltaVal

        return baseMin
    }

    return (
        <div className="sidebar-card ops-panel">
            <div className="ops-header">
                <h3 className="sidebar-card-title">⚙️ Manual Overrides</h3>
                <div className="ops-header-actions">
                    <span
                        className="ops-help-icon"
                        title={
                            [
                                'Policy (Stage 1A):',
                                '• Target + Lock are authoritative and are enforced in effective minutes + sim allocator.',
                                '• Availability overrides locks: if a locked player is inactive in a sim world, their minutes become 0.',
                                '• Δ is sugar: on Save it becomes Target = Base + Δ and Lock = true.',
                                '• OUT implies Target = 0 and Lock = true.',
                                '• Infeasible locks (sum locked targets > 240) will error on save.',
                            ].join('\n')
                        }
                    >
                        ⓘ
                    </span>
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
                    const teamTotal = teamPlayers.reduce((sum, p) => sum + plannedMinutes(p), 0)

                    return (
                        <div key={team} className="ops-team-group">
                            <div
                                className="ops-team-header"
                                title="Preview total for this team using Base + local edits. Effective layer + sim will keep team totals at 240 by redistributing minutes across unlocked eligible players."
                            >
                                <span>{team}</span>
                                <span className={`team-total ${Math.abs(teamTotal - 240) < 1 ? 'balanced' : teamTotal > 240 ? 'over' : 'under'}`}>
                                    {teamTotal.toFixed(0)} min
                                </span>
                            </div>
                            <div className="ops-table-scroll">
                                <table className="ops-player-table">
                                    <tbody>
                                        {teamPlayers.map(p => {
                                            const pid = String(p.player_id)
                                            const delta = deltas[pid]
                                            const role = roles[pid] ?? (p.ops_depth_role ?? '')
                                            const baseMin = p.minutes_final ?? p.minutes_p50 ?? 0
                                            const targetValue = targets[pid]
                                            const savedTarget = p.minutes_target ?? null
                                            const savedLock = p.minutes_lock ?? null
                                            const effectiveTarget = p.minutes_target_eff
                                            const effectiveLock = p.minutes_lock_eff ?? false

                                            const outRole = role.trim().toLowerCase() === 'out'
                                            const previewTarget =
                                                outRole
                                                    ? 0
                                                    : (targetValue !== undefined && !isNaN(targetValue))
                                                        ? targetValue
                                                        : (delta !== undefined)
                                                            ? baseMin + delta
                                                            : (savedTarget !== null && savedTarget !== undefined)
                                                                ? savedTarget
                                                                : (effectiveLock && effectiveTarget !== undefined)
                                                                    ? effectiveTarget
                                                                    : null
                                            const lockChecked =
                                                (targetValue !== undefined && !isNaN(targetValue)) ||
                                                delta !== undefined ||
                                                (locks[pid] ?? (savedLock ?? effectiveLock))
                                            const lockDisabled =
                                                (targetValue !== undefined && !isNaN(targetValue)) || delta !== undefined
                                            const isStarter = p.is_confirmed_starter || p.is_projected_starter

                                            return (
                                                <tr
                                                    key={pid}
                                                    className={(delta !== undefined || targets[pid] !== undefined || locks[pid] !== undefined || roles[pid] !== undefined) ? 'modified' : ''}
                                                >
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
                                                            title="Role override. OUT implies Target=0 and Lock=true."
                                                        >
                                                            <option value="">Role…</option>
                                                            <option value="starter">Starter</option>
                                                            <option value="rotation">Rotation</option>
                                                            <option value="deep_bench">Deep bench</option>
                                                            <option value="out">OUT</option>
                                                        </select>
                                                    </td>
                                                    <td className="base-minutes-cell" title="Base minutes from model/effective layer (before your local edits).">
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
                                                            title="Minutes Δ (sugar). On Save, this becomes Target = Base + Δ and Lock = true."
                                                        />
                                                    </td>
                                                    <td className="target-cell">
                                                        <input
                                                            type="number"
                                                            className="target-input"
                                                            placeholder="Min"
                                                            step="0.5"
                                                            min="0"
                                                            max="48"
                                                            value={targetValue ?? ''}
                                                            onChange={(e) => handleTargetChange(p.player_id, e.target.value)}
                                                            title="Minutes Target (absolute). On Save, Lock = true and minutes are held fixed through reconcile + sim allocation (when active)."
                                                        />
                                                    </td>
                                                    <td className="lock-cell">
                                                        <input
                                                            type="checkbox"
                                                            className="lock-checkbox"
                                                            checked={Boolean(lockChecked)}
                                                            disabled={lockDisabled}
                                                            onChange={(e) => handleLockToggle(p.player_id, e.target.checked)}
                                                            title={
                                                                lockDisabled
                                                                    ? 'Target/Δ implies lock'
                                                                    : 'Lock minutes through effective reconcile + sim allocator. Availability overrides locks: if inactive in a sim world, minutes become 0.'
                                                            }
                                                        />
                                                    </td>
                                                    <td className={`final-minutes-cell ${(delta !== undefined || targetValue !== undefined) ? 'adjusted' : ''}`} title="Preview minutes (local). Effective layer will reconcile unlocked players to team=240.">
                                                        {previewTarget !== null ? Number(previewTarget).toFixed(1) : ''}
                                                    </td>
                                                </tr>
                                            )
                                        })}
                                    </tbody>
                                </table>
                            </div>
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
                    className="ops-reset-btn"
                    onClick={resetOverrides}
                    disabled={!canReset || resetting || saveStatus === 'saving'}
                    title="Clear ALL saved overrides for this game (minutes + roles). This also clears any unsaved local edits."
                >
                    {resetting ? 'Resetting...' : 'Reset All'}
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
            {saveError && (
                <div className="ops-worlds-message error">
                    {saveError}
                </div>
            )}
        </div>
    )
}
