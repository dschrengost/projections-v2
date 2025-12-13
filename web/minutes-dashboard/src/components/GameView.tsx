import React, { useMemo, useState } from 'react'
import { PlayerRow } from '../types'
import {
    formatFpts,
    formatMinutes,
    formatOwnership,
    formatSalary,
    formatValue,
    getStatusBadge,
} from '../utils'

type GameViewProps = {
    rows: PlayerRow[]
    gameId: string
}

type PlayerGroup = {
    starters: PlayerRow[]
    rotation: PlayerRow[]
    bench: PlayerRow[]
    injured: PlayerRow[]
}

const ROTATION_MINUTES_THRESHOLD = 10

// Get Vegas info from first player row (all players in same game have same vegas data)
const getVegasInfo = (rows: PlayerRow[]) => {
    const row = rows.find(r => r.total !== undefined || r.spread_home !== undefined)
    if (!row) return null

    return {
        total: row.total,
        spreadHome: row.spread_home,
        teamImpliedTotal: row.team_implied_total,
        opponentImpliedTotal: row.opponent_implied_total,
    }
}

// Compute team-level summary stats
const computeTeamSummary = (players: PlayerRow[]) => {
    const active = players.filter(p => p.status?.toLowerCase() !== 'out')
    const starters = active.filter(p => p.is_confirmed_starter || p.is_projected_starter)
    const injured = players.filter(p => p.status?.toLowerCase() === 'out')
    const gtd = players.filter(p => ['gtd', 'questionable', 'doubtful'].includes(p.status?.toLowerCase() || ''))

    const totalFpts = active.reduce((sum, p) => sum + (p.sim_dk_fpts_mean || 0), 0)
    const totalMinutes = active.reduce((sum, p) => sum + (p.minutes_p50 || 0), 0)
    const avgFptsPerMin = totalMinutes > 0 ? totalFpts / totalMinutes : 0

    return {
        playerCount: active.length,
        starterCount: starters.length,
        injuredCount: injured.length,
        gtdCount: gtd.length,
        totalFpts,
        totalMinutes,
        avgFptsPerMin,
    }
}

export const GameView: React.FC<GameViewProps> = ({ rows, gameId }) => {
    const [selectedPlayer, setSelectedPlayer] = useState<PlayerRow | null>(null)

    const teams = useMemo(() => {
        const teamMap = new Map<string, { tricode: string; players: PlayerRow[] }>()
        rows.forEach((row) => {
            const team = row.team_tricode || row.team_name || String(row.team_id)
            if (!teamMap.has(team)) {
                teamMap.set(team, {
                    tricode: row.team_tricode || team,
                    players: []
                })
            }
            teamMap.get(team)?.players.push(row)
        })
        return Array.from(teamMap.entries())
    }, [rows])

    const vegasInfo = useMemo(() => getVegasInfo(rows), [rows])

    const categorizePlayers = (players: PlayerRow[]): PlayerGroup => {
        const starters: PlayerRow[] = []
        const rotation: PlayerRow[] = []
        const bench: PlayerRow[] = []
        const injured: PlayerRow[] = []

        // Sort by minutes descending first
        const sorted = [...players].sort((a, b) => (b.minutes_p50 || 0) - (a.minutes_p50 || 0))

        sorted.forEach((p) => {
            const status = p.status?.toLowerCase() || ''
            if (status === 'out') {
                injured.push(p)
            } else if (p.is_confirmed_starter || p.is_projected_starter) {
                starters.push(p)
            } else if ((p.minutes_p50 || 0) >= ROTATION_MINUTES_THRESHOLD) {
                rotation.push(p)
            } else {
                bench.push(p)
            }
        })

        return { starters, rotation, bench, injured }
    }

    if (teams.length === 0) return <div>No data for game {gameId}</div>

    // Get team names for header
    const teamNames = teams.map(([name]) => name)
    const tipTime = rows[0]?.tip_ts ? new Date(rows[0].tip_ts).toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true,
        timeZoneName: 'short'
    }) : null

    return (
        <div className="game-view-container">
            {/* Left Sidebar */}
            <div className="game-sidebar">
                {/* Game Environment Panel */}
                <GameEnvironmentPanel vegasInfo={vegasInfo} tipTime={tipTime} teamNames={teamNames} />

                {/* Team Summary Cards */}
                {teams.map(([teamName, { players }]) => (
                    <TeamSummaryCard key={teamName} teamName={teamName} players={players} />
                ))}

                {/* Minutes Distribution Chart */}
                {selectedPlayer && (
                    <MinutesDistributionChart player={selectedPlayer} />
                )}
            </div>

            {/* Main Content - Player Tables */}
            <div className="game-main">
                {/* Vegas Header */}
                {vegasInfo && (
                    <div className="vegas-header">
                        <div className="vegas-matchup">
                            {teamNames.join(' vs ')}
                            {tipTime && <span className="tip-time">• {tipTime}</span>}
                        </div>
                        <div className="vegas-lines">
                            {vegasInfo.total && (
                                <span className="vegas-stat">
                                    Total: <strong>{vegasInfo.total}</strong>
                                </span>
                            )}
                            {vegasInfo.spreadHome !== undefined && (
                                <span className="vegas-stat">
                                    Spread: <strong>{vegasInfo.spreadHome > 0 ? '+' : ''}{vegasInfo.spreadHome}</strong>
                                </span>
                            )}
                        </div>
                    </div>
                )}

                {/* Team Columns */}
                <div className="teams-container">
                    {teams.map(([teamName, { tricode, players }]) => {
                        const { starters, rotation, bench, injured } = categorizePlayers(players)
                        const teamImplied = players[0]?.team_implied_total

                        return (
                            <div key={teamName} className="team-column">
                                <h2 className="team-header">
                                    {teamName}
                                    {teamImplied && <span className="implied-total">IT: {teamImplied.toFixed(1)}</span>}
                                </h2>

                                <div className="player-section">
                                    <h3>Starters</h3>
                                    <ExpandablePlayerList players={starters} onPlayerSelect={setSelectedPlayer} />
                                </div>

                                <div className="player-section">
                                    <h3>Rotation</h3>
                                    <ExpandablePlayerList players={rotation} onPlayerSelect={setSelectedPlayer} />
                                </div>

                                <div className="player-section">
                                    <h3>Deep Bench</h3>
                                    <ExpandablePlayerList players={bench} onPlayerSelect={setSelectedPlayer} />
                                </div>

                                <div className="player-section">
                                    <h3>Injured / Out</h3>
                                    <ExpandablePlayerList players={injured} onPlayerSelect={setSelectedPlayer} />
                                </div>
                            </div>
                        )
                    })}
                </div>
            </div>
        </div>
    )
}

// === Game Environment Panel ===
type GameEnvironmentProps = {
    vegasInfo: ReturnType<typeof getVegasInfo>
    tipTime: string | null
    teamNames: string[]
}

const GameEnvironmentPanel: React.FC<GameEnvironmentProps> = ({ vegasInfo, tipTime, teamNames }) => {
    // Derive game environment from Vegas data
    const getGameScript = () => {
        if (!vegasInfo?.spreadHome) return null
        const spread = Math.abs(vegasInfo.spreadHome)
        if (spread >= 10) return { label: 'Blowout', color: 'text-downside' }
        if (spread >= 6) return { label: 'Comfortable', color: 'text-muted' }
        if (spread >= 3) return { label: 'Close', color: 'text-upside' }
        return { label: 'Toss-up', color: 'text-upside' }
    }

    const getPaceRating = () => {
        if (!vegasInfo?.total) return null
        const total = vegasInfo.total
        if (total >= 235) return { label: 'Very Fast', value: '🔥', color: 'text-upside' }
        if (total >= 225) return { label: 'Fast', value: '⬆️', color: 'text-upside' }
        if (total >= 215) return { label: 'Average', value: '➡️', color: 'text-muted' }
        if (total >= 205) return { label: 'Slow', value: '⬇️', color: 'text-downside' }
        return { label: 'Very Slow', value: '🐢', color: 'text-downside' }
    }

    const gameScript = getGameScript()
    const paceRating = getPaceRating()

    return (
        <div className="sidebar-card">
            <h3 className="sidebar-card-title">🎮 Game Environment</h3>
            <div className="environment-grid">
                {vegasInfo?.total && (
                    <div className="env-stat">
                        <span className="env-label">Total</span>
                        <span className="env-value">{vegasInfo.total}</span>
                    </div>
                )}
                {vegasInfo?.spreadHome !== undefined && (
                    <div className="env-stat">
                        <span className="env-label">Spread</span>
                        <span className="env-value">{vegasInfo.spreadHome > 0 ? '+' : ''}{vegasInfo.spreadHome}</span>
                    </div>
                )}
                {paceRating && (
                    <div className="env-stat">
                        <span className="env-label">Pace</span>
                        <span className={`env-value ${paceRating.color}`}>{paceRating.value} {paceRating.label}</span>
                    </div>
                )}
                {gameScript && (
                    <div className="env-stat">
                        <span className="env-label">Script</span>
                        <span className={`env-value ${gameScript.color}`}>{gameScript.label}</span>
                    </div>
                )}
            </div>
            {(!vegasInfo?.total && !vegasInfo?.spreadHome) && (
                <div className="env-empty">Vegas lines not available</div>
            )}
        </div>
    )
}

// === Team Summary Card ===
type TeamSummaryProps = {
    teamName: string
    players: PlayerRow[]
}

const TeamSummaryCard: React.FC<TeamSummaryProps> = ({ teamName, players }) => {
    const summary = useMemo(() => computeTeamSummary(players), [players])
    const impliedTotal = players[0]?.team_implied_total

    return (
        <div className="sidebar-card team-summary">
            <h3 className="sidebar-card-title">{teamName}</h3>
            <div className="summary-grid">
                <div className="summary-stat">
                    <span className="summary-value">{summary.totalFpts.toFixed(1)}</span>
                    <span className="summary-label">Team FPTS</span>
                </div>
                {impliedTotal && (
                    <div className="summary-stat">
                        <span className="summary-value">{impliedTotal.toFixed(1)}</span>
                        <span className="summary-label">Implied</span>
                    </div>
                )}
                <div className="summary-stat">
                    <span className="summary-value">{summary.starterCount}</span>
                    <span className="summary-label">Starters</span>
                </div>
                <div className="summary-stat">
                    <span className="summary-value text-downside">{summary.injuredCount}</span>
                    <span className="summary-label">Out</span>
                </div>
                {summary.gtdCount > 0 && (
                    <div className="summary-stat">
                        <span className="summary-value text-warning">{summary.gtdCount}</span>
                        <span className="summary-label">GTD</span>
                    </div>
                )}
            </div>
        </div>
    )
}

// === Minutes Distribution Chart ===
type MinutesChartProps = {
    player: PlayerRow
}

const MinutesDistributionChart: React.FC<MinutesChartProps> = ({ player }) => {
    const p10 = player.minutes_p10 || 0
    const p50 = player.minutes_p50 || 0
    const p90 = player.minutes_p90 || 0
    const simP50 = player.sim_minutes_sim_p50 ?? player.sim_minutes_sim_mean ?? p50

    const maxMinutes = 48
    const scale = (val: number) => (val / maxMinutes) * 100

    return (
        <div className="sidebar-card">
            <h3 className="sidebar-card-title">📊 {player.player_name}</h3>
            <div className="minutes-chart">
                <div className="minutes-bar-container">
                    {/* Range bar from p10 to p90 */}
                    <div
                        className="minutes-range-bar"
                        style={{
                            left: `${scale(p10)}%`,
                            width: `${scale(p90 - p10)}%`
                        }}
                    />
                    {/* p50 marker */}
                    <div
                        className="minutes-marker p50"
                        style={{ left: `${scale(p50)}%` }}
                        title={`p50: ${p50.toFixed(1)} min`}
                    />
                    {/* Sim mean marker */}
                    {simP50 !== p50 && (
                        <div
                            className="minutes-marker sim"
                            style={{ left: `${scale(simP50)}%` }}
                            title={`Sim p50: ${simP50.toFixed(1)} min`}
                        />
                    )}
                </div>
                <div className="minutes-labels">
                    <span>{p10.toFixed(0)}</span>
                    <span className="center">{p50.toFixed(0)} min</span>
                    <span>{p90.toFixed(0)}</span>
                </div>
                <div className="minutes-legend">
                    <span className="legend-item"><span className="dot p50-dot"></span> p50</span>
                    <span className="legend-item"><span className="dot sim-dot"></span> Sim</span>
                </div>
            </div>
        </div>
    )
}

// === Expandable Player List ===
const ExpandablePlayerList: React.FC<{
    players: PlayerRow[]
    onPlayerSelect?: (player: PlayerRow) => void
}> = ({ players, onPlayerSelect }) => {
    const [expandedIds, setExpandedIds] = useState<Set<string | number>>(new Set())

    const toggleExpand = (player: PlayerRow) => {
        const playerId = player.player_id
        if (!playerId) return
        setExpandedIds(prev => {
            const next = new Set(prev)
            if (next.has(playerId)) {
                next.delete(playerId)
            } else {
                next.add(playerId)
            }
            return next
        })
        onPlayerSelect?.(player)
    }

    if (players.length === 0) return <div className="empty-section">None</div>

    return (
        <table className="player-list-table expandable">
            <thead>
                <tr>
                    <th></th>
                    <th>Player</th>
                    <th>Salary</th>
                    <th>Min</th>
                    <th>FPTS</th>
                    <th title="Ownership %">Own%</th>
                    <th title="Value (FPTS/$1k)">Val</th>
                    <th title="Ceiling (p95)">Ceil</th>
                    <th title="Floor (p05)">Floor</th>
                </tr>
            </thead>
            <tbody>
                {players.map((p) => {
                    const statusBadge = getStatusBadge(p.status)
                    const playerId = p.player_id
                    const isExpanded = playerId ? expandedIds.has(playerId) : false
                    const hasStats = p.sim_pts_mean !== undefined || p.sim_reb_mean !== undefined

                    return (
                        <React.Fragment key={playerId}>
                            <tr
                                className={`player-row ${isExpanded ? 'expanded' : ''} ${hasStats ? 'expandable-row' : ''}`}
                                onClick={() => hasStats && toggleExpand(p)}
                            >
                                <td className="expand-cell">
                                    {hasStats && (
                                        <span className="expand-icon">{isExpanded ? '▼' : '▶'}</span>
                                    )}
                                </td>
                                <td>
                                    <div className="player-cell">
                                        <span className="player-name">{p.player_name}</span>
                                        {statusBadge && (
                                            <span
                                                className={`status-tag ${statusBadge.className}`}
                                                title={statusBadge.title}
                                            >
                                                {statusBadge.label}
                                            </span>
                                        )}
                                        {p.is_confirmed_starter && (
                                            <span className="status-tag badge-confirmed" title="Confirmed">S</span>
                                        )}
                                        {!p.is_confirmed_starter && p.is_projected_starter && (
                                            <span className="status-tag badge-projected" title="Projected">P</span>
                                        )}
                                    </div>
                                </td>
                                <td className="salary-cell">{formatSalary(p.salary)}</td>
                                <td>{formatMinutes(p.minutes_p50)}</td>
                                <td className="font-bold">{formatFpts(p.sim_dk_fpts_mean)}</td>
                                <td className="ownership-cell">
                                    {p.is_locked && <span className="lock-icon" title="Predictions locked">🔒</span>}
                                    {formatOwnership(p.pred_own_pct)}
                                </td>
                                <td className="value-cell">{formatValue(p.value)}</td>
                                <td className="text-upside">{formatFpts(p.sim_dk_fpts_p95)}</td>
                                <td className="text-downside">{formatFpts(p.sim_dk_fpts_p05)}</td>
                            </tr>
                            {isExpanded && (
                                <tr className="expanded-details">
                                    <td colSpan={9}>
                                        <div className="stats-breakdown">
                                            <div className="stats-row">
                                                <StatItem label="PTS" value={p.sim_pts_mean} />
                                                <StatItem label="REB" value={p.sim_reb_mean} />
                                                <StatItem label="AST" value={p.sim_ast_mean} />
                                                <StatItem label="STL" value={p.sim_stl_mean} />
                                                <StatItem label="BLK" value={p.sim_blk_mean} />
                                                <StatItem label="TOV" value={p.sim_tov_mean} />
                                            </div>
                                            <div className="stats-row secondary">
                                                <StatItem
                                                    label="Sim Min p50"
                                                    value={p.sim_minutes_sim_p50 ?? p.sim_minutes_sim_mean}
                                                    compare={p.minutes_p50}
                                                    format="minutes"
                                                />
                                                <StatItem label="p10" value={p.sim_dk_fpts_p10} />
                                                <StatItem label="p25" value={p.sim_dk_fpts_p25} />
                                                <StatItem label="p50" value={p.sim_dk_fpts_p50} />
                                                <StatItem label="p75" value={p.sim_dk_fpts_p75} />
                                                <StatItem label="p90" value={p.sim_dk_fpts_p90} />
                                                <StatItem label="p95" value={p.sim_dk_fpts_p95} />
                                            </div>
                                        </div>
                                    </td>
                                </tr>
                            )}
                        </React.Fragment>
                    )
                })}
            </tbody>
        </table>
    )
}

type StatItemProps = {
    label: string
    value: number | undefined
    compare?: number
    format?: 'fpts' | 'minutes'
}

const StatItem: React.FC<StatItemProps> = ({ label, value, compare, format = 'fpts' }) => {
    if (value === undefined || value === null) return null

    const formatted = format === 'minutes' ? formatMinutes(value) : value.toFixed(1)

    // Show comparison arrow if compare value provided
    let compareClass = ''
    if (compare !== undefined && value !== undefined) {
        const diff = value - compare
        if (Math.abs(diff) > 0.5) {
            compareClass = diff > 0 ? 'stat-up' : 'stat-down'
        }
    }

    return (
        <div className={`stat-item ${compareClass}`}>
            <span className="stat-label">{label}</span>
            <span className="stat-value">{formatted}</span>
        </div>
    )
}
