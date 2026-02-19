import React from 'react'
import { PlayerOverrideState } from '../api/gameview_v2'
import { OverrideControl } from './OverrideControl'

export type PlayerTableRow = {
    game_id: string
    team_id: string
    player_id: string
    name: string
    pos?: string
    status?: string
    isProjectedStarter?: boolean
    isConfirmedStarter?: boolean
    baselineMinutes: number
    resolvedMinutes: number
    baselineFpts: number | null
    resolvedFpts: number | null
    override: PlayerOverrideState
    minutesP10: number | null
    minutesP50: number | null
    minutesP90: number | null
}

type PlayerTableProps = {
    rows: PlayerTableRow[]
    onSelectPlayer: (playerId: string) => void
    onOverrideChange: (playerId: string, next: PlayerOverrideState) => void
}

export const PlayerTable: React.FC<PlayerTableProps> = ({ rows, onSelectPlayer, onOverrideChange }) => {
    if (!rows.length) {
        return <div className="muted">No players found.</div>
    }
    const fmt = (value: number | null) => (value == null ? '-' : value.toFixed(1))

    return (
        <div className="gv2-player-table-wrap">
            <table className="gv2-player-table">
                <thead>
                    <tr>
                        <th>Player</th>
                        <th>Status</th>
                        <th>Baseline Min</th>
                        <th>Resolved μ</th>
                        <th>Band P10</th>
                        <th>Band P50</th>
                        <th>Band P90</th>
                        <th>Minutes Band</th>
                        <th>FPTS</th>
                    </tr>
                </thead>
                <tbody>
                    {rows.map((row) => (
                        <tr key={`${row.game_id}-${row.player_id}`} onClick={() => onSelectPlayer(row.player_id)}>
                            <td>
                                <div className="gv2-player-cell">
                                    <div className="gv2-player-name-row">
                                        <strong>{row.name}</strong>
                                        {row.isConfirmedStarter ? (
                                            <span className="status-tag badge-confirmed" title="Confirmed Starter">
                                                Confirmed
                                            </span>
                                        ) : row.isProjectedStarter ? (
                                            <span className="status-tag badge-projected" title="Projected Starter">
                                                Projected
                                            </span>
                                        ) : null}
                                    </div>
                                    {row.pos ? <span className="muted">{row.pos}</span> : null}
                                </div>
                            </td>
                            <td>{row.status || '-'}</td>
                            <td>{row.baselineMinutes.toFixed(1)}</td>
                            <td>{row.resolvedMinutes.toFixed(1)}</td>
                            <td>{fmt(row.minutesP10)}</td>
                            <td>{fmt(row.minutesP50)}</td>
                            <td>{fmt(row.minutesP90)}</td>
                            <td>
                                <OverrideControl
                                    compact
                                    value={row.override}
                                    baselineMinutes={row.baselineMinutes}
                                    resolvedMinutes={row.resolvedMinutes}
                                    onChange={(next) => onOverrideChange(row.player_id, next)}
                                />
                            </td>
                            <td>{row.resolvedFpts == null ? '-' : row.resolvedFpts.toFixed(1)}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    )
}
