import React from 'react'
import { PlayerLastGameStat, PlayerOverrideState } from '../api/gameview_v2'
import { OverrideControl } from './OverrideControl'

export type DrawerMetric = {
    baseline: number | null
    resolved: number | null
    p10?: number | null
    p50?: number | null
    p90?: number | null
}

export type PlayerDrawerData = {
    player_id: string
    name: string
    team: string
    pos?: string
    status?: string
    isProjectedStarter?: boolean
    isConfirmedStarter?: boolean
    baselineMinutes: number
    resolvedMinutes: number
    override: PlayerOverrideState
    whyChanged?: string | null
    metrics: {
        minutes: DrawerMetric
        fpts: DrawerMetric
        pts: DrawerMetric
        reb: DrawerMetric
        ast: DrawerMetric
        stl: DrawerMetric
        blk: DrawerMetric
        to: DrawerMetric
    }
    lastGames: PlayerLastGameStat[]
    lastGamesLoading: boolean
    lastGamesError?: string | null
    minutesBandLabel?: string | null
}

type PlayerDetailsDrawerProps = {
    open: boolean
    player: PlayerDrawerData | null
    onClose: () => void
    onOverrideChange: (playerId: string, next: PlayerOverrideState) => void
}

const MetricRow: React.FC<{ label: string; metric: DrawerMetric }> = ({ label, metric }) => {
    const hasQuantiles = metric.p10 != null || metric.p50 != null || metric.p90 != null
    return (
        <div className="gv2-metric-row">
            <div className="gv2-metric-head">
                <span>{label}</span>
                <span>
                    {metric.baseline == null ? '-' : metric.baseline.toFixed(1)} → {metric.resolved == null ? '-' : metric.resolved.toFixed(1)}
                </span>
            </div>
            {hasQuantiles && (
                <div className="gv2-metric-quantiles">
                    <span>p10 {metric.p10 == null ? '-' : metric.p10.toFixed(1)}</span>
                    <span>p50 {metric.p50 == null ? '-' : metric.p50.toFixed(1)}</span>
                    <span>p90 {metric.p90 == null ? '-' : metric.p90.toFixed(1)}</span>
                </div>
            )}
        </div>
    )
}

export const PlayerDetailsDrawer: React.FC<PlayerDetailsDrawerProps> = ({
    open,
    player,
    onClose,
    onOverrideChange,
}) => {
    if (!open || !player) return null

    return (
        <>
            <div className="gv2-drawer-backdrop" onClick={onClose} />
            <aside className="gv2-drawer" role="dialog" aria-label="Player details">
                <div className="gv2-drawer-header">
                    <div>
                        <h3 className="gv2-drawer-player-title">
                            {player.name}
                            {player.isConfirmedStarter ? (
                                <span className="status-tag badge-confirmed" title="Confirmed Starter">
                                    Confirmed
                                </span>
                            ) : player.isProjectedStarter ? (
                                <span className="status-tag badge-projected" title="Projected Starter">
                                    Projected
                                </span>
                            ) : null}
                        </h3>
                        <div className="muted">{player.team} {player.pos ? `· ${player.pos}` : ''} {player.status ? `· ${player.status}` : ''}</div>
                    </div>
                    <button type="button" onClick={onClose}>Close</button>
                </div>

                <section className="gv2-drawer-section">
                    <h4>Minutes Band</h4>
                    <div className="muted gv2-band-note">Constrains resolved mean μ only; world tails remain stochastic.</div>
                    <OverrideControl
                        value={player.override}
                        baselineMinutes={player.baselineMinutes}
                        resolvedMinutes={player.resolvedMinutes}
                        onChange={(next) => onOverrideChange(player.player_id, next)}
                    />
                </section>

                <section className="gv2-drawer-section">
                    <h4>Projections</h4>
                    <div className="muted gv2-band-note">
                        Minutes p10/p50/p90 source: {player.minutesBandLabel ?? 'Sim (uncond)'}
                    </div>
                    <MetricRow label="Minutes" metric={player.metrics.minutes} />
                    <MetricRow label="FPTS" metric={player.metrics.fpts} />
                    <MetricRow label="PTS" metric={player.metrics.pts} />
                    <MetricRow label="REB" metric={player.metrics.reb} />
                    <MetricRow label="AST" metric={player.metrics.ast} />
                    <MetricRow label="STL" metric={player.metrics.stl} />
                    <MetricRow label="BLK" metric={player.metrics.blk} />
                    <MetricRow label="TO" metric={player.metrics.to} />
                </section>

                {player.whyChanged ? (
                    <section className="gv2-drawer-section">
                        <h4>Why Changed</h4>
                        <div className="gv2-why-changed">{player.whyChanged}</div>
                    </section>
                ) : null}

                <section className="gv2-drawer-section">
                    <h4>Last 5 Games (Actuals)</h4>
                    {player.lastGamesLoading ? (
                        <div className="muted">Loading game log...</div>
                    ) : player.lastGamesError ? (
                        <div className="gv2-why-changed">{player.lastGamesError}</div>
                    ) : player.lastGames.length === 0 ? (
                        <div className="muted">No recent games found.</div>
                    ) : (
                        <div className="gv2-last-games-wrap">
                            <table className="gv2-last-games-table">
                                <thead>
                                    <tr>
                                        <th>Date</th>
                                        <th>Opp</th>
                                        <th>MIN</th>
                                        <th>PTS</th>
                                        <th>REB</th>
                                        <th>AST</th>
                                        <th>FPTS</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {player.lastGames.map((game) => (
                                        <tr key={`${game.game_id}-${game.game_date}`}>
                                            <td>{game.game_date}</td>
                                            <td>
                                                {game.team_tricode && game.opponent_tricode
                                                    ? `${game.team_tricode} vs ${game.opponent_tricode}`
                                                    : game.opponent_tricode || '-'}
                                            </td>
                                            <td>{game.minutes.toFixed(1)}</td>
                                            <td>{game.pts.toFixed(0)}</td>
                                            <td>{game.reb.toFixed(0)}</td>
                                            <td>{game.ast.toFixed(0)}</td>
                                            <td>{game.fpts.toFixed(1)}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </section>
            </aside>
        </>
    )
}
