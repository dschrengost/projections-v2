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
    baselineMinutes: number
    resolvedMinutes: number
    baselineFpts: number | null
    resolvedFpts: number | null
    override: PlayerOverrideState
    minBound: number
    maxBound: number
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

    return (
        <div className="gv2-player-table-wrap">
            <table className="gv2-player-table">
                <thead>
                    <tr>
                        <th>Player</th>
                        <th>Status</th>
                        <th>Baseline Min</th>
                        <th>Resolved μ</th>
                        <th>Override Control</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>FPTS</th>
                    </tr>
                </thead>
                <tbody>
                    {rows.map((row) => (
                        <tr key={`${row.game_id}-${row.player_id}`} onClick={() => onSelectPlayer(row.player_id)}>
                            <td>
                                <div className="gv2-player-cell">
                                    <strong>{row.name}</strong>
                                    {row.pos ? <span className="muted">{row.pos}</span> : null}
                                </div>
                            </td>
                            <td>{row.status || '-'}</td>
                            <td>{row.baselineMinutes.toFixed(1)}</td>
                            <td>{row.resolvedMinutes.toFixed(1)}</td>
                            <td>
                                <OverrideControl
                                    compact
                                    value={row.override}
                                    baselineMinutes={row.baselineMinutes}
                                    resolvedMinutes={row.resolvedMinutes}
                                    onChange={(next) => onOverrideChange(row.player_id, next)}
                                />
                            </td>
                            <td>{row.minBound.toFixed(1)}</td>
                            <td>{row.maxBound.toFixed(1)}</td>
                            <td>{row.resolvedFpts == null ? '-' : row.resolvedFpts.toFixed(1)}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    )
}
