import { useMemo, useState } from 'react'
import { LineupEVResult } from '../api/contest_sim'
import { PoolPlayer } from '../api/optimizer'

export interface PlayerExposure {
    player_id: string
    player: PoolPlayer | undefined
    count: number
    exposure_pct: number
}

export interface ExposureBounds {
    min?: number
    max?: number
}

interface PlayerExposurePanelProps {
    lineupResults: (LineupEVResult & { total_own?: number })[]
    playerMap: Map<string, PoolPlayer>
    minUniques: number
    onMinUniquesChange: (n: number) => void
    minUniquesPassCount: number
    exposureBounds: Map<string, ExposureBounds>
    onExposureBoundsChange: (playerId: string, bounds: ExposureBounds | null) => void
    exposureCapError: string | null
}

export default function PlayerExposurePanel({
    lineupResults,
    playerMap,
    minUniques,
    onMinUniquesChange,
    minUniquesPassCount,
    exposureBounds,
    onExposureBoundsChange,
    exposureCapError,
}: PlayerExposurePanelProps) {
    const [isCollapsed, setIsCollapsed] = useState(false)
    const [sortBy, setSortBy] = useState<'exposure' | 'salary' | 'ownership'>('exposure')
    const [showOnlyHighExposure, setShowOnlyHighExposure] = useState(false)
    const [editingPlayer, setEditingPlayer] = useState<{ playerId: string; field: 'min' | 'max' } | null>(null)
    const [editValue, setEditValue] = useState('')

    // Compute exposure data
    const exposureData = useMemo(() => {
        if (lineupResults.length === 0) return []

        const counts = new Map<string, number>()
        lineupResults.forEach(r => {
            r.player_ids.forEach(pid => counts.set(pid, (counts.get(pid) || 0) + 1))
        })

        return Array.from(counts.entries())
            .map(([pid, count]) => ({
                player_id: pid,
                player: playerMap.get(pid),
                count,
                exposure_pct: (count / lineupResults.length) * 100,
            }))
            .sort((a, b) => {
                if (sortBy === 'exposure') return b.exposure_pct - a.exposure_pct
                if (sortBy === 'salary') return (b.player?.salary ?? 0) - (a.player?.salary ?? 0)
                if (sortBy === 'ownership') return (b.player?.own_proj ?? 0) - (a.player?.own_proj ?? 0)
                return 0
            })
    }, [lineupResults, playerMap, sortBy])

    // Filter if needed
    const filteredExposure = useMemo(() => {
        if (showOnlyHighExposure) {
            return exposureData.filter(e => e.exposure_pct >= 50)
        }
        return exposureData
    }, [exposureData, showOnlyHighExposure])

    const getExposureClass = (pct: number, bounds: ExposureBounds | undefined) => {
        // If over max or under min, highlight as error
        if (bounds?.max !== undefined && pct > bounds.max) return 'exposure-over-cap'
        if (bounds?.min !== undefined && pct < bounds.min) return 'exposure-under-min'
        if (pct >= 75) return 'exposure-very-high'
        if (pct >= 50) return 'exposure-high'
        if (pct >= 25) return 'exposure-medium'
        if (pct >= 10) return 'exposure-low'
        return 'exposure-very-low'
    }

    const handleBoundsClick = (playerId: string, field: 'min' | 'max') => {
        const bounds = exposureBounds.get(playerId)
        const currentValue = field === 'min' ? bounds?.min : bounds?.max
        setEditingPlayer({ playerId, field })
        setEditValue(currentValue !== undefined ? currentValue.toString() : '')
    }

    const handleBoundsBlur = () => {
        if (editingPlayer) {
            const { playerId, field } = editingPlayer
            const val = editValue.trim()
            const currentBounds = exposureBounds.get(playerId) || {}

            if (val === '' || val === '-') {
                // Clear this bound
                const newBounds = { ...currentBounds }
                delete newBounds[field]
                if (Object.keys(newBounds).length === 0) {
                    onExposureBoundsChange(playerId, null)
                } else {
                    onExposureBoundsChange(playerId, newBounds)
                }
            } else {
                const num = parseFloat(val)
                if (!isNaN(num) && num >= 0 && num <= 100) {
                    onExposureBoundsChange(playerId, { ...currentBounds, [field]: num })
                }
            }
            setEditingPlayer(null)
            setEditValue('')
        }
    }

    const handleBoundsKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            handleBoundsBlur()
        } else if (e.key === 'Escape') {
            setEditingPlayer(null)
            setEditValue('')
        }
    }

    const clearAllBounds = () => {
        exposureBounds.forEach((_, pid) => onExposureBoundsChange(pid, null))
    }

    if (lineupResults.length === 0) {
        return null
    }

    const boundsCount = exposureBounds.size

    return (
        <div className={`exposure-panel ${isCollapsed ? 'collapsed' : ''}`}>
            <div className="exposure-header" onClick={() => setIsCollapsed(!isCollapsed)}>
                <h4>
                    <span className="collapse-icon">{isCollapsed ? '▶' : '▼'}</span>
                    Player Exposure
                    <span className="exposure-badge">{exposureData.length} players</span>
                    {boundsCount > 0 && <span className="exposure-badge caps-badge">{boundsCount} constraints</span>}
                </h4>
            </div>

            {!isCollapsed && (
                <div className="exposure-content">
                    {/* Error Banner */}
                    {exposureCapError && (
                        <div className="exposure-error">
                            ⚠️ {exposureCapError}
                        </div>
                    )}

                    {/* Min Uniques Control */}
                    <div className="exposure-controls">
                        <div className="control-group">
                            <label>Min Uniques:</label>
                            <input
                                type="number"
                                min={0}
                                max={8}
                                value={minUniques}
                                onChange={e => onMinUniquesChange(Math.max(0, Math.min(8, Number(e.target.value))))}
                                className="min-uniques-input"
                            />
                            <span className="min-uniques-info">
                                {minUniquesPassCount} / {lineupResults.length} pass
                            </span>
                        </div>

                        <div className="control-group">
                            <label>Sort:</label>
                            <select
                                value={sortBy}
                                onChange={e => setSortBy(e.target.value as typeof sortBy)}
                            >
                                <option value="exposure">Exposure %</option>
                                <option value="salary">Salary</option>
                                <option value="ownership">Ownership %</option>
                            </select>
                        </div>

                        <div className="control-group">
                            <label>
                                <input
                                    type="checkbox"
                                    checked={showOnlyHighExposure}
                                    onChange={e => setShowOnlyHighExposure(e.target.checked)}
                                />
                                High exposure only (≥50%)
                            </label>
                        </div>

                        {boundsCount > 0 && (
                            <div className="control-group">
                                <button
                                    className="clear-caps-btn"
                                    onClick={clearAllBounds}
                                    title="Clear all exposure constraints"
                                >
                                    Clear All
                                </button>
                            </div>
                        )}
                    </div>

                    {/* Exposure Table */}
                    <div className="exposure-table-container">
                        <table className="exposure-table">
                            <thead>
                                <tr>
                                    <th>Player</th>
                                    <th>Pos</th>
                                    <th>Salary</th>
                                    <th>Own%</th>
                                    <th>Exp%</th>
                                    <th>Min%</th>
                                    <th>Max%</th>
                                    <th>Count</th>
                                </tr>
                            </thead>
                            <tbody>
                                {filteredExposure.map(e => {
                                    const bounds = exposureBounds.get(e.player_id)
                                    const isEditingMin = editingPlayer?.playerId === e.player_id && editingPlayer.field === 'min'
                                    const isEditingMax = editingPlayer?.playerId === e.player_id && editingPlayer.field === 'max'
                                    const isOverMax = bounds?.max !== undefined && e.exposure_pct > bounds.max
                                    const isUnderMin = bounds?.min !== undefined && e.exposure_pct < bounds.min

                                    return (
                                        <tr
                                            key={e.player_id}
                                            className={`${getExposureClass(e.exposure_pct, bounds)} ${isOverMax ? 'over-cap' : ''} ${isUnderMin ? 'under-min' : ''}`}
                                        >
                                            <td className="player-name" title={e.player?.name ?? e.player_id}>
                                                {e.player?.name ?? e.player_id}
                                            </td>
                                            <td className="player-pos">
                                                {e.player?.positions?.join('/') ?? '?'}
                                            </td>
                                            <td className="player-salary">
                                                {e.player ? `$${(e.player.salary / 1000).toFixed(1)}k` : '-'}
                                            </td>
                                            <td className="player-own">
                                                {e.player?.own_proj?.toFixed(0) ?? '-'}%
                                            </td>
                                            <td className="player-exposure">
                                                <div className="exposure-bar-container">
                                                    <div
                                                        className="exposure-bar"
                                                        style={{ width: `${Math.min(100, e.exposure_pct)}%` }}
                                                    />
                                                    {bounds?.min !== undefined && (
                                                        <div
                                                            className="exposure-min-marker"
                                                            style={{ left: `${Math.min(100, bounds.min)}%` }}
                                                            title={`Min: ${bounds.min}%`}
                                                        />
                                                    )}
                                                    {bounds?.max !== undefined && (
                                                        <div
                                                            className="exposure-cap-marker"
                                                            style={{ left: `${Math.min(100, bounds.max)}%` }}
                                                            title={`Max: ${bounds.max}%`}
                                                        />
                                                    )}
                                                    <span className="exposure-value">{e.exposure_pct.toFixed(1)}%</span>
                                                </div>
                                            </td>
                                            <td className="player-cap" onClick={() => handleBoundsClick(e.player_id, 'min')}>
                                                {isEditingMin ? (
                                                    <input
                                                        type="number"
                                                        className="cap-input min-input"
                                                        value={editValue}
                                                        onChange={ev => setEditValue(ev.target.value)}
                                                        onBlur={handleBoundsBlur}
                                                        onKeyDown={handleBoundsKeyDown}
                                                        autoFocus
                                                        min={0}
                                                        max={100}
                                                        placeholder="-"
                                                    />
                                                ) : (
                                                    <span className={`cap-value ${bounds?.min !== undefined ? 'has-min' : ''}`}>
                                                        {bounds?.min !== undefined ? `${bounds.min}%` : '-'}
                                                    </span>
                                                )}
                                            </td>
                                            <td className="player-cap" onClick={() => handleBoundsClick(e.player_id, 'max')}>
                                                {isEditingMax ? (
                                                    <input
                                                        type="number"
                                                        className="cap-input"
                                                        value={editValue}
                                                        onChange={ev => setEditValue(ev.target.value)}
                                                        onBlur={handleBoundsBlur}
                                                        onKeyDown={handleBoundsKeyDown}
                                                        autoFocus
                                                        min={0}
                                                        max={100}
                                                        placeholder="-"
                                                    />
                                                ) : (
                                                    <span className={`cap-value ${bounds?.max !== undefined ? 'has-cap' : ''}`}>
                                                        {bounds?.max !== undefined ? `${bounds.max}%` : '-'}
                                                    </span>
                                                )}
                                            </td>
                                            <td className="player-count">
                                                {e.count} / {lineupResults.length}
                                            </td>
                                        </tr>
                                    )
                                })}
                            </tbody>
                        </table>
                    </div>

                    {filteredExposure.length === 0 && (
                        <div className="exposure-empty">
                            No players match the current filter.
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}
