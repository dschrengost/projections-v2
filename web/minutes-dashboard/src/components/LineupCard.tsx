import { LineupEVResult } from '../api/contest_sim'
import { PoolPlayer } from '../api/optimizer'

interface LineupCardProps {
    result: LineupEVResult
    players: Map<string, PoolPlayer>
    selected: boolean
    onToggleSelect: () => void
    highlighted?: 'best-ev' | 'best-ceiling' | null
}

export default function LineupCard({
    result,
    players,
    selected,
    onToggleSelect,
    highlighted,
}: LineupCardProps) {
    const formatMoney = (val: number) =>
        val >= 0 ? `$${val.toFixed(2)}` : `-$${Math.abs(val).toFixed(2)}`
    const formatPct = (val: number) => `${(val * 100).toFixed(1)}%`
    const formatRoi = (val: number) => {
        const pct = (val * 100).toFixed(1)
        return val >= 0 ? `+${pct}%` : `${pct}%`
    }
    const formatRiskAdj = (val: number) => {
        if (!Number.isFinite(val)) return '—'
        const s = val.toFixed(2)
        return val >= 0 ? `+${s}` : s
    }

    // Get player info with fallbacks
    const getPlayer = (pid: string) => players.get(pid)

    // Calculate total ownership
    const totalOwn = result.player_ids.reduce((sum, pid) => {
        const p = getPlayer(pid)
        return sum + (p?.own_proj ?? 0)
    }, 0)

    // Calculate total salary
    const totalSalary = result.player_ids.reduce((sum, pid) => {
        const p = getPlayer(pid)
        return sum + (p?.salary ?? 0)
    }, 0)

    const isPositiveEV = result.expected_value >= 0

    return (
        <div
            className={`lineup-card compact ${selected ? 'selected' : ''} ${isPositiveEV ? 'positive-ev' : 'negative-ev'} ${highlighted ? highlighted : ''}`}
            onClick={onToggleSelect}
        >
            <div className="lineup-card-content">
                {/* Left Column: Player List */}
                <div className="lineup-players-column">
                    <div className="lineup-header-compact">
                        <div className="lineup-check">
                            <input
                                type="checkbox"
                                checked={selected}
                                onChange={onToggleSelect}
                                onClick={(e) => e.stopPropagation()}
                            />
                            <span className="lineup-id">#{result.lineup_id + 1}</span>
                        </div>
                        <div className="lineup-total-salary">
                            ${totalSalary.toLocaleString()}
                        </div>
                    </div>

                    <div className="players-list-compact">
                        {result.player_ids.map((pid, idx) => {
                            const player = getPlayer(pid)
                            const posLabel = player?.positions?.join('/') ?? '?'

                            return (
                                <div key={`${pid}-${idx}`} className="player-row-compact">
                                    <span className="p-pos">{posLabel}</span>
                                    <span className="p-name" title={player?.name}>{player?.name ?? pid}</span>
                                    <span className="p-salary">{player ? `$${(player.salary / 1000).toFixed(1)}k` : '-'}</span>
                                    <span className={`p-own ${player && (player.own_proj || 0) > 20 ? 'high-own' : ''}`}>
                                        {player?.own_proj?.toFixed(0) ?? '-'}%
                                    </span>
                                </div>
                            )
                        })}
                    </div>
                </div>

                {/* Right Column: Stats Grid */}
                <div className="lineup-stats-column">
                    <div className="stats-primary">
                        <div className={`stat-box primary ${isPositiveEV ? 'positive' : 'negative'}`}>
                            <span className="label">Exp. Value</span>
                            <span className="value">{formatMoney(result.expected_value)}</span>
                        </div>
                        <div className={`stat-box primary ${result.roi > 0 ? 'positive' : 'negative'}`}>
                            <span className="label">ROI</span>
                            <span className="value">{formatRoi(result.roi)}</span>
                        </div>
                    </div>

                    <div className="stats-grid-compact">
                        <div className="stat-box">
                            <span className="label">Win %</span>
                            <span className="value highlight">{formatPct(result.win_rate)}</span>
                        </div>
                        <div className="stat-box">
                            <span className="label">Top 1%</span>
                            <span className="value">{formatPct(result.top_1pct_rate)}</span>
                        </div>
                        <div className="stat-box">
                            <span className="label">Cash %</span>
                            <span className="value">{formatPct(result.cash_rate)}</span>
                        </div>
                        <div className="stat-box">
                            <span className="label">Own %</span>
                            <span className="value">{totalOwn.toFixed(0)}%</span>
                        </div>

                        <div className="stat-box">
                            <span className="label">Mean</span>
                            <span className="value">{result.mean.toFixed(1)}</span>
                        </div>
                        <div className="stat-box">
                            <span className="label">Ceiling</span>
                            <span className="value">{result.p90.toFixed(1)}</span>
                        </div>
                        <div className="stat-box">
                            <span className="label">Top 5%</span>
                            <span className="value">{formatPct(result.top_5pct_rate)}</span>
                        </div>
                        <div className="stat-box">
                            <span className="label">p95</span>
                            <span className="value">{result.p95.toFixed(1)}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
