import React from 'react'
import { TeamDiagnostics } from '../api/gameview_v2'
import { PlayerOverrideState } from '../api/gameview_v2'
import { PlayerTable, PlayerTableRow } from './PlayerTable'
import { TeamBudgetBar } from './TeamBudgetBar'

type TeamPanelProps = {
    teamName: string
    diagnostics?: TeamDiagnostics | null
    rows: PlayerTableRow[]
    minutesBandLabel: string
    onSelectPlayer: (playerId: string) => void
    onOverrideChange: (playerId: string, next: PlayerOverrideState) => void
}

export const TeamPanel: React.FC<TeamPanelProps> = ({
    teamName,
    diagnostics,
    rows,
    minutesBandLabel,
    onSelectPlayer,
    onOverrideChange,
}) => {
    const isOutLike = (status?: string) => {
        const s = (status || '').trim().toLowerCase()
        return s === 'out' || s === 'inactive' || s === 'dnp' || s === 'suspended'
    }

    const outPlayers = rows.filter((row) => isOutLike(row.status))
    const activePlayers = rows.filter((row) => !isOutLike(row.status))

    const starters = activePlayers.filter((row) => row.isConfirmedStarter || row.isProjectedStarter)
    const bench = activePlayers.filter(
        (row) => !(row.isConfirmedStarter || row.isProjectedStarter) && row.resolvedMinutes >= 10,
    )
    const fringe = activePlayers.filter(
        (row) => !(row.isConfirmedStarter || row.isProjectedStarter) && row.resolvedMinutes < 10,
    )

    const sections: Array<{ label: string; rows: PlayerTableRow[] }> = [
        { label: 'Starters', rows: starters },
        { label: 'Bench', rows: bench },
        { label: 'Fringe', rows: fringe },
        { label: 'Out', rows: outPlayers },
    ]

    return (
        <section className="gv2-team-panel">
            <header className="gv2-team-header">
                <div>
                    <h3>{teamName}</h3>
                    <div className="muted gv2-band-source">Band source: {minutesBandLabel}</div>
                </div>
            </header>
            <TeamBudgetBar diagnostics={diagnostics ?? null} />
            {sections.filter((section) => section.rows.length > 0).map((section) => (
                <div key={section.label} className="gv2-team-section">
                    <div className="gv2-team-section-title">
                        {section.label}
                        <span>{section.rows.length}</span>
                    </div>
                    <PlayerTable
                        rows={section.rows}
                        onSelectPlayer={onSelectPlayer}
                        onOverrideChange={onOverrideChange}
                    />
                </div>
            ))}
        </section>
    )
}
