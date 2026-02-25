import React from 'react'
import { PlayerRow } from '../types'
import { GameviewV2Page } from './GameviewV2Page'

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
    runId,
    readOnly,
    onGameChange,
    onOverridesSaved,
    onOpenLateSwap,
    onRunCompleted,
}) => {
    const targetDate = date || rows[0]?.game_date || new Date().toISOString().slice(0, 10)

    return (
        <GameviewV2Page
            rows={rows}
            date={targetDate}
            runId={runId}
            readOnly={Boolean(readOnly)}
            initialGameId={gameId}
            onGameChange={onGameChange}
            onRefresh={onOverridesSaved}
            onOpenLateSwap={onOpenLateSwap}
            onRunCompleted={onRunCompleted}
        />
    )
}
