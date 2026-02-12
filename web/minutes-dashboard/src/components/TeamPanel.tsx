import React from 'react'
import { TeamDiagnostics } from '../api/gameview_v2'
import { PlayerOverrideState } from '../api/gameview_v2'
import { PlayerTable, PlayerTableRow } from './PlayerTable'
import { TeamBudgetBar } from './TeamBudgetBar'

type TeamPanelProps = {
    teamName: string
    diagnostics?: TeamDiagnostics | null
    rows: PlayerTableRow[]
    onSelectPlayer: (playerId: string) => void
    onOverrideChange: (playerId: string, next: PlayerOverrideState) => void
}

export const TeamPanel: React.FC<TeamPanelProps> = ({
    teamName,
    diagnostics,
    rows,
    onSelectPlayer,
    onOverrideChange,
}) => {
    return (
        <section className="gv2-team-panel">
            <header className="gv2-team-header">
                <h3>{teamName}</h3>
            </header>
            <TeamBudgetBar diagnostics={diagnostics ?? null} />
            <PlayerTable rows={rows} onSelectPlayer={onSelectPlayer} onOverrideChange={onOverrideChange} />
        </section>
    )
}
