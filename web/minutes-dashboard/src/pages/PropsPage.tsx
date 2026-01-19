import { useCallback, useEffect, useMemo, useState } from 'react'
import {
    getPropsLines,
    getPropsSummary,
    PROP_TYPE_LABELS,
    BOOK_LABELS,
    type PropLine,
    type PropsSummary,
    type BookLine,
} from '../api/props'
import { useSlateDate } from '../hooks/useSlateDate'

type SortKey = 'player_name' | 'team' | 'prop_type' | 'prediction' | 'best_over_line' | 'over_ev' | 'under_ev' | 'max_ev'

const SORT_LABELS: Record<SortKey, string> = {
    player_name: 'Player',
    team: 'Team',
    prop_type: 'Prop',
    prediction: 'Prediction',
    best_over_line: 'Line',
    over_ev: 'Over EV',
    under_ev: 'Under EV',
    max_ev: 'Best EV',
}

function formatEV(ev: number | null): string {
    if (ev === null) return '—'
    const pct = (ev * 100).toFixed(1)
    return ev >= 0 ? `+${pct}%` : `${pct}%`
}

function formatOdds(odds: number | null): string {
    if (odds === null) return '—'
    return odds >= 0 ? `+${odds}` : `${odds}`
}

function formatProb(prob: number | null): string {
    if (prob === null) return '—'
    return `${(prob * 100).toFixed(1)}%`
}

function getEdgeClass(edge: string | null): string {
    switch (edge) {
        case 'strong_over':
        case 'strong_under':
            return 'edge-strong'
        case 'slight_over':
        case 'slight_under':
            return 'edge-slight'
        case 'fair':
            return 'edge-fair'
        default:
            return ''
    }
}

function EdgeBadge({ edge, ev }: { edge: string | null; ev: number | null }) {
    if (!edge || edge === 'no_edge') return <span className="edge-badge edge-none">—</span>

    const label = edge.replace('_', ' ').toUpperCase()
    const evStr = formatEV(ev)

    return (
        <span className={`edge-badge ${getEdgeClass(edge)}`}>
            {label} ({evStr})
        </span>
    )
}

function BookLinesTable({ lines }: { lines: BookLine[] }) {
    return (
        <table className="book-lines-table">
            <thead>
                <tr>
                    <th>Book</th>
                    <th>Line</th>
                    <th>Over</th>
                    <th>Under</th>
                </tr>
            </thead>
            <tbody>
                {lines.map((line) => (
                    <tr key={line.book}>
                        <td>{BOOK_LABELS[line.book] ?? line.book}</td>
                        <td>{line.line}</td>
                        <td>{formatOdds(line.over_odds)}</td>
                        <td>{formatOdds(line.under_odds)}</td>
                    </tr>
                ))}
            </tbody>
        </table>
    )
}

export function PropsPage() {
    const [targetDate, setTargetDate] = useSlateDate()
    const [lines, setLines] = useState<PropLine[]>([])
    const [summary, setSummary] = useState<PropsSummary | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    // Filters
    const [propTypeFilter, setPropTypeFilter] = useState<string>('')
    const [teamFilter, setTeamFilter] = useState<string>('')
    const [searchFilter, setSearchFilter] = useState<string>('')
    const [minEvFilter, setMinEvFilter] = useState<string>('')
    const [showEdgesOnly, setShowEdgesOnly] = useState(false)

    // Sorting
    const [sortKey, setSortKey] = useState<SortKey>('max_ev')
    const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

    // Expanded rows
    const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set())

    const loadData = useCallback(async (date: string) => {
        setLoading(true)
        setError(null)
        try {
            const [linesData, summaryData] = await Promise.all([
                getPropsLines(date),
                getPropsSummary(date),
            ])
            setLines(linesData)
            setSummary(summaryData)
        } catch (err) {
            setError((err as Error).message)
            setLines([])
            setSummary(null)
        } finally {
            setLoading(false)
        }
    }, [])

    useEffect(() => {
        void loadData(targetDate)
    }, [loadData, targetDate])

    // Get unique values for filters
    const propTypes = useMemo(() => {
        const types = new Set(lines.map((l) => l.prop_type))
        return Array.from(types).sort()
    }, [lines])

    const teams = useMemo(() => {
        const teamSet = new Set(lines.map((l) => l.team))
        return Array.from(teamSet).sort()
    }, [lines])

    // Filter and sort lines
    const filteredLines = useMemo(() => {
        let result = lines

        // Prop type filter
        if (propTypeFilter) {
            result = result.filter((l) => l.prop_type === propTypeFilter)
        }

        // Team filter
        if (teamFilter) {
            result = result.filter((l) => l.team === teamFilter)
        }

        // Search filter
        if (searchFilter.trim()) {
            const search = searchFilter.trim().toLowerCase()
            result = result.filter(
                (l) =>
                    l.player_name.toLowerCase().includes(search) ||
                    l.team.toLowerCase().includes(search) ||
                    l.opponent.toLowerCase().includes(search)
            )
        }

        // Min EV filter
        const minEv = parseFloat(minEvFilter)
        if (!isNaN(minEv)) {
            result = result.filter((l) => {
                const maxEv = Math.max(l.over_ev ?? -999, l.under_ev ?? -999)
                return maxEv >= minEv / 100 // Convert percentage to decimal
            })
        }

        // Edges only filter
        if (showEdgesOnly) {
            result = result.filter(
                (l) =>
                    (l.over_edge && l.over_edge !== 'no_edge' && l.over_edge !== 'fair') ||
                    (l.under_edge && l.under_edge !== 'no_edge' && l.under_edge !== 'fair')
            )
        }

        // Sort
        result = [...result].sort((a, b) => {
            let aVal: number | string = 0
            let bVal: number | string = 0

            switch (sortKey) {
                case 'player_name':
                    aVal = a.player_name
                    bVal = b.player_name
                    break
                case 'team':
                    aVal = a.team
                    bVal = b.team
                    break
                case 'prop_type':
                    aVal = a.prop_type
                    bVal = b.prop_type
                    break
                case 'prediction':
                    aVal = a.prediction ?? -999
                    bVal = b.prediction ?? -999
                    break
                case 'best_over_line':
                    aVal = a.best_over_line ?? -999
                    bVal = b.best_over_line ?? -999
                    break
                case 'over_ev':
                    aVal = a.over_ev ?? -999
                    bVal = b.over_ev ?? -999
                    break
                case 'under_ev':
                    aVal = a.under_ev ?? -999
                    bVal = b.under_ev ?? -999
                    break
                case 'max_ev':
                    aVal = Math.max(a.over_ev ?? -999, a.under_ev ?? -999)
                    bVal = Math.max(b.over_ev ?? -999, b.under_ev ?? -999)
                    break
            }

            if (typeof aVal === 'string' && typeof bVal === 'string') {
                return sortDir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal)
            }
            return sortDir === 'asc' ? (aVal as number) - (bVal as number) : (bVal as number) - (aVal as number)
        })

        return result
    }, [lines, propTypeFilter, teamFilter, searchFilter, minEvFilter, showEdgesOnly, sortKey, sortDir])

    const toggleSort = (key: SortKey) => {
        if (sortKey === key) {
            setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
        } else {
            setSortKey(key)
            setSortDir(key === 'player_name' || key === 'team' || key === 'prop_type' ? 'asc' : 'desc')
        }
    }

    const toggleExpanded = (rowId: string) => {
        setExpandedRows((prev) => {
            const next = new Set(prev)
            if (next.has(rowId)) {
                next.delete(rowId)
            } else {
                next.add(rowId)
            }
            return next
        })
    }

    const getRowId = (line: PropLine) => `${line.player_id}-${line.prop_type}`

    return (
        <div className="props-page">
            <div className="props-header">
                <div>
                    <h1>Props & EV Analysis</h1>
                    <p className="subtitle">Player prop lines with model predictions and edge detection.</p>
                </div>
                <div className="props-controls">
                    <label>
                        Date
                        <input
                            type="date"
                            value={targetDate}
                            onChange={(e) => setTargetDate(e.target.value)}
                        />
                    </label>
                    <button onClick={() => loadData(targetDate)} disabled={loading}>
                        Refresh
                    </button>
                </div>
            </div>

            {/* Summary Bar */}
            {summary && (
                <div className="props-summary">
                    <div className="card">
                        <div className="label">Total Props</div>
                        <div className="value">{summary.total_props}</div>
                    </div>
                    <div className="card">
                        <div className="label">Players</div>
                        <div className="value">{summary.players_with_props}</div>
                    </div>
                    <div className="card">
                        <div className="label">With Edge</div>
                        <div className="value edge-highlight">{summary.props_with_edge}</div>
                    </div>
                    {summary.best_edges.length > 0 && (
                        <div className="card best-edge">
                            <div className="label">Top Edge</div>
                            <div className="value">
                                {summary.best_edges[0].player} {summary.best_edges[0].prop.toUpperCase()}{' '}
                                {summary.best_edges[0].side} {formatEV(summary.best_edges[0].ev)}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Filters */}
            <div className="props-filters">
                <select
                    value={propTypeFilter}
                    onChange={(e) => setPropTypeFilter(e.target.value)}
                >
                    <option value="">All Prop Types</option>
                    {propTypes.map((pt) => (
                        <option key={pt} value={pt}>
                            {PROP_TYPE_LABELS[pt] ?? pt}
                        </option>
                    ))}
                </select>

                <select value={teamFilter} onChange={(e) => setTeamFilter(e.target.value)}>
                    <option value="">All Teams</option>
                    {teams.map((team) => (
                        <option key={team} value={team}>
                            {team}
                        </option>
                    ))}
                </select>

                <input
                    type="text"
                    placeholder="Search player..."
                    value={searchFilter}
                    onChange={(e) => setSearchFilter(e.target.value)}
                />

                <input
                    type="number"
                    placeholder="Min EV %"
                    value={minEvFilter}
                    onChange={(e) => setMinEvFilter(e.target.value)}
                    step="1"
                    style={{ width: '100px' }}
                />

                <label className="checkbox-label">
                    <input
                        type="checkbox"
                        checked={showEdgesOnly}
                        onChange={(e) => setShowEdgesOnly(e.target.checked)}
                    />
                    Edges only
                </label>
            </div>

            {/* Error/Loading states */}
            {error && <div className="props-alert error">Error: {error}</div>}
            {!error && !loading && lines.length === 0 && (
                <div className="props-alert muted">No props data found for {targetDate}.</div>
            )}

            {/* Props Table */}
            <div className="table-wrapper props-table">
                <table>
                    <thead>
                        <tr>
                            <th className="expand-col"></th>
                            {(['player_name', 'team', 'prop_type', 'prediction', 'best_over_line'] as SortKey[]).map(
                                (key) => (
                                    <th key={key} onClick={() => toggleSort(key)} className="sortable">
                                        {SORT_LABELS[key]}
                                        {sortKey === key && <span>{sortDir === 'asc' ? ' ▲' : ' ▼'}</span>}
                                    </th>
                                )
                            )}
                            <th>Best Over</th>
                            <th onClick={() => toggleSort('over_ev')} className="sortable">
                                Over EV
                                {sortKey === 'over_ev' && <span>{sortDir === 'asc' ? ' ▲' : ' ▼'}</span>}
                            </th>
                            <th>Best Under</th>
                            <th onClick={() => toggleSort('under_ev')} className="sortable">
                                Under EV
                                {sortKey === 'under_ev' && <span>{sortDir === 'asc' ? ' ▲' : ' ▼'}</span>}
                            </th>
                            <th onClick={() => toggleSort('max_ev')} className="sortable">
                                Edge
                                {sortKey === 'max_ev' && <span>{sortDir === 'asc' ? ' ▲' : ' ▼'}</span>}
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        {loading && (
                            <tr>
                                <td colSpan={11}>Loading...</td>
                            </tr>
                        )}
                        {!loading &&
                            filteredLines.map((line) => {
                                const rowId = getRowId(line)
                                const isExpanded = expandedRows.has(rowId)
                                const maxEv = Math.max(line.over_ev ?? -999, line.under_ev ?? -999)
                                const bestEdge = maxEv === (line.over_ev ?? -999) ? line.over_edge : line.under_edge

                                return (
                                    <>
                                        <tr key={rowId} className={isExpanded ? 'expanded' : ''}>
                                            <td className="expand-col">
                                                <button
                                                    className="expand-btn"
                                                    onClick={() => toggleExpanded(rowId)}
                                                    title={isExpanded ? 'Collapse' : 'Expand to see all books'}
                                                >
                                                    {isExpanded ? '−' : '+'}
                                                </button>
                                            </td>
                                            <td>
                                                <div className="player-name">{line.player_name}</div>
                                                <div className="muted">vs {line.opponent}</div>
                                            </td>
                                            <td>{line.team}</td>
                                            <td>{PROP_TYPE_LABELS[line.prop_type] ?? line.prop_type}</td>
                                            <td>{line.prediction?.toFixed(1) ?? '—'}</td>
                                            <td>{line.best_over_line ?? '—'}</td>
                                            <td>
                                                <span className={line.best_over_book === 'draftkings' ? 'best-book' : ''}>
                                                    {formatOdds(line.best_over_odds)}
                                                </span>
                                                <span className="muted book-label">
                                                    {line.best_over_book ? ` ${BOOK_LABELS[line.best_over_book] ?? line.best_over_book}` : ''}
                                                </span>
                                            </td>
                                            <td className={line.over_ev && line.over_ev > 0.03 ? 'positive-ev' : ''}>
                                                {formatEV(line.over_ev)}
                                                <span className="muted prob-label">
                                                    {line.over_true_prob ? ` (${formatProb(line.over_true_prob)})` : ''}
                                                </span>
                                            </td>
                                            <td>
                                                <span className={line.best_under_book === 'draftkings' ? 'best-book' : ''}>
                                                    {formatOdds(line.best_under_odds)}
                                                </span>
                                                <span className="muted book-label">
                                                    {line.best_under_book ? ` ${BOOK_LABELS[line.best_under_book] ?? line.best_under_book}` : ''}
                                                </span>
                                            </td>
                                            <td className={line.under_ev && line.under_ev > 0.03 ? 'positive-ev' : ''}>
                                                {formatEV(line.under_ev)}
                                                <span className="muted prob-label">
                                                    {line.under_true_prob ? ` (${formatProb(line.under_true_prob)})` : ''}
                                                </span>
                                            </td>
                                            <td>
                                                <EdgeBadge edge={bestEdge} ev={maxEv > -999 ? maxEv : null} />
                                            </td>
                                        </tr>
                                        {isExpanded && (
                                            <tr key={`${rowId}-expanded`} className="expanded-row">
                                                <td colSpan={11}>
                                                    <div className="expanded-content">
                                                        <h4>All Book Lines</h4>
                                                        <BookLinesTable lines={line.all_lines} />
                                                    </div>
                                                </td>
                                            </tr>
                                        )}
                                    </>
                                )
                            })}
                    </tbody>
                </table>
            </div>
        </div>
    )
}

export default PropsPage
