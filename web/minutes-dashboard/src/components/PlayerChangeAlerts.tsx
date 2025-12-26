import { useEffect, useState } from 'react'
import { apiUrl } from '../api/client'

type PlayerChange = {
    player_id: number
    player_name: string
    fpts_old: number
    fpts_new: number
    fpts_delta: number
    minutes_old: number
    minutes_new: number
    minutes_delta: number
}

type PlayerDiffResponse = {
    date: string
    run_a: string | null
    run_b: string | null
    threshold: number
    total_players: number
    changed_count: number
    changes: PlayerChange[]
    error?: string
    message?: string
}

type Props = {
    date: string
    currentRunId: string | null
    blessedRunId: string | null
    latestRunId: string | null
}

export function PlayerChangeAlerts({ date, currentRunId, blessedRunId, latestRunId }: Props) {
    const [data, setData] = useState<PlayerDiffResponse | null>(null)
    const [loading, setLoading] = useState(false)
    const [expanded, setExpanded] = useState(false)

    useEffect(() => {
        if (!date) return

        const fetchDiff = async () => {
            setLoading(true)
            try {
                // Compare latest vs blessed
                const params = new URLSearchParams({ date, threshold: '3' })
                if (latestRunId) params.set('run_a', latestRunId)
                if (blessedRunId) params.set('run_b', blessedRunId)

                const res = await fetch(apiUrl(`/api/player-diff?${params}`))
                if (res.ok) {
                    setData(await res.json())
                }
            } catch {
                setData(null)
            } finally {
                setLoading(false)
            }
        }

        void fetchDiff()
    }, [date, latestRunId, blessedRunId])

    // Don't show if same run or no changes
    if (!data || data.changed_count === 0 || data.run_a === data.run_b) {
        return null
    }

    if (data.error) {
        return null
    }

    const hasSignificantChanges = data.changes.some(c => Math.abs(c.fpts_delta) >= 5)

    return (
        <div className={`player-change-alerts ${hasSignificantChanges ? 'alert-warning' : 'alert-info'}`}>
            <div
                className="alert-header"
                onClick={() => setExpanded(!expanded)}
                style={{ cursor: 'pointer' }}
            >
                <span className="alert-icon">
                    {hasSignificantChanges ? '⚠️' : 'ℹ️'}
                </span>
                <span className="alert-title">
                    {data.changed_count} player{data.changed_count !== 1 ? 's' : ''} changed
                    {data.run_a !== currentRunId && ' (vs latest)'}
                </span>
                <span className="alert-toggle">{expanded ? '▼' : '▶'}</span>
            </div>

            {expanded && (
                <div className="alert-body">
                    <table className="change-table">
                        <thead>
                            <tr>
                                <th>Player</th>
                                <th>FPTS (Old)</th>
                                <th>FPTS (New)</th>
                                <th>Δ FPTS</th>
                                <th>MIN Δ</th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.changes.slice(0, 15).map((change) => (
                                <tr
                                    key={change.player_id}
                                    className={Math.abs(change.fpts_delta) >= 5 ? 'significant-change' : ''}
                                >
                                    <td>{change.player_name}</td>
                                    <td>{change.fpts_old.toFixed(1)}</td>
                                    <td>{change.fpts_new.toFixed(1)}</td>
                                    <td className={change.fpts_delta > 0 ? 'positive' : 'negative'}>
                                        {change.fpts_delta > 0 ? '+' : ''}{change.fpts_delta.toFixed(1)}
                                    </td>
                                    <td className={change.minutes_delta > 0 ? 'positive' : 'negative'}>
                                        {change.minutes_delta > 0 ? '+' : ''}{change.minutes_delta.toFixed(0)}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    <div className="alert-meta">
                        Comparing {data.run_a?.slice(0, 15)} → {data.run_b?.slice(0, 15)}
                    </div>
                </div>
            )}
        </div>
    )
}
