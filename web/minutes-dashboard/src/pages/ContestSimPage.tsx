import { useEffect, useMemo, useState } from 'react'
import {
    runContestSim,
    getContestSimConfig,
    ContestSimResponse,
    ConfigResponse,
} from '../api/contest_sim'
import {
    getSavedBuilds,
    loadSavedBuild,
    getPlayerPool,
    getSlates,
    exportCustomLineupsCSV,
    SavedBuild,
    PoolPlayer,
    Slate,
} from '../api/optimizer'
import LineupCard from '../components/LineupCard'

const todayISO = () => new Date().toISOString().slice(0, 10)

type SortKey = 'lineup_id' | 'mean' | 'p90' | 'p95' | 'expected_value' | 'roi' | 'win_rate' | 'top_1pct_rate' | 'top_10pct_rate' | 'cash_rate' | 'total_own'

export default function ContestSimPage() {
    // Date and slate selection
    const [selectedDate, setSelectedDate] = useState(todayISO())
    const [slates, setSlates] = useState<Slate[]>([])
    const [selectedSlate, setSelectedSlate] = useState<number | null>(null)
    const [slatesLoading, setSlatesLoading] = useState(false)

    // Saved builds
    const [savedBuilds, setSavedBuilds] = useState<SavedBuild[]>([])
    const [selectedBuildId, setSelectedBuildId] = useState<string | null>(null)
    const [buildsLoading, setBuildsLoading] = useState(false)

    // Player pool for name resolution
    const [pool, setPool] = useState<PoolPlayer[]>([])

    // Configuration options
    const [config, setConfig] = useState<ConfigResponse | null>(null)
    const [archetype, setArchetype] = useState('medium')
    const [fieldSizeBucket, setFieldSizeBucket] = useState('medium')
    const [entryFee, setEntryFee] = useState(3.0)

    // Simulation state
    const [lineups, setLineups] = useState<string[][]>([])
    const [simResult, setSimResult] = useState<ContestSimResponse | null>(null)
    const [simLoading, setSimLoading] = useState(false)
    const [simError, setSimError] = useState<string | null>(null)

    // Sorting and filtering
    const [sortKey, setSortKey] = useState<SortKey>('expected_value')
    const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
    const [filterPositiveEV, setFilterPositiveEV] = useState(false)
    const [maxOwnership, setMaxOwnership] = useState<number | null>(null)

    const [selectedLineups, setSelectedLineups] = useState<Set<number>>(new Set())

    // Pagination & Top N Filter
    const [page, setPage] = useState(1)
    const [pageSize, setPageSize] = useState(50)
    const [topN, setTopN] = useState<number | null>(null) // null = all


    // Load slates when date changes
    useEffect(() => {
        const load = async () => {
            setSlatesLoading(true)
            try {
                const data = await getSlates(selectedDate)
                setSlates(data)
                const mainSlate = data.find(s => s.slate_type !== 'showdown')
                setSelectedSlate(mainSlate?.draft_group_id ?? data[0]?.draft_group_id ?? null)
            } catch {
                setSlates([])
                setSelectedSlate(null)
            } finally {
                setSlatesLoading(false)
            }
        }
        void load()
    }, [selectedDate])

    // Load saved builds when slate changes
    useEffect(() => {
        if (!selectedSlate) {
            setSavedBuilds([])
            return
        }
        const load = async () => {
            setBuildsLoading(true)
            try {
                const builds = await getSavedBuilds(selectedDate, selectedSlate)
                setSavedBuilds(builds)
                setSelectedBuildId(builds[0]?.job_id ?? null)
            } catch {
                setSavedBuilds([])
            } finally {
                setBuildsLoading(false)
            }
        }
        void load()
    }, [selectedDate, selectedSlate])

    // Load player pool for name resolution
    useEffect(() => {
        if (!selectedSlate) {
            setPool([])
            return
        }
        const load = async () => {
            try {
                const data = await getPlayerPool(selectedDate, selectedSlate)
                setPool(data)
            } catch {
                setPool([])
            }
        }
        void load()
    }, [selectedDate, selectedSlate])

    // Load config on mount
    useEffect(() => {
        const load = async () => {
            try {
                const cfg = await getContestSimConfig()
                setConfig(cfg)
                setArchetype(cfg.default_archetype)
                setFieldSizeBucket(cfg.default_field_size_bucket)
                setEntryFee(cfg.default_entry_fee)
            } catch {
                // Use defaults
            }
        }
        void load()
    }, [])

    // Load lineups when build selection changes
    useEffect(() => {
        if (!selectedBuildId) {
            setLineups([])
            return
        }
        const load = async () => {
            try {
                const build = await loadSavedBuild(selectedDate, selectedBuildId)
                if (build.lineups) {
                    setLineups(build.lineups.map(lu => lu.player_ids))
                }
            } catch {
                setLineups([])
            }
        }
        void load()
    }, [selectedDate, selectedBuildId])

    // Clear selection when results change
    useEffect(() => {
        setSelectedLineups(new Set())
    }, [simResult])

    // Player name lookup
    const playerMap = useMemo(() => {
        const map = new Map<string, PoolPlayer>()
        pool.forEach(p => map.set(p.player_id, p))
        return map
    }, [pool])

    // Calculate total ownership for each lineup result
    const resultsWithOwnership = useMemo(() => {
        if (!simResult) return []
        return simResult.results.map(r => {
            const totalOwn = r.player_ids.reduce((sum, pid) => {
                const p = playerMap.get(pid)
                return sum + (p?.own_proj ?? 0)
            }, 0)
            return { ...r, total_own: totalOwn }
        })
    }, [simResult, playerMap])

    // Filter and sort results
    const filteredResults = useMemo(() => {
        let results = [...resultsWithOwnership]

        // Apply filters
        if (filterPositiveEV) {
            results = results.filter(r => r.expected_value >= 0)
        }
        if (maxOwnership !== null) {
            results = results.filter(r => r.total_own <= maxOwnership)
        }

        // Sort
        results.sort((a, b) => {
            const aVal = a[sortKey as keyof typeof a]
            const bVal = b[sortKey as keyof typeof b]
            if (typeof aVal === 'number' && typeof bVal === 'number') {
                return sortDir === 'asc' ? aVal - bVal : bVal - aVal
            }
            return 0
        })

        // Apply Top N filter (after sorting)
        if (topN !== null) {
            results = results.slice(0, topN)
        }

        return results
    }, [resultsWithOwnership, filterPositiveEV, maxOwnership, topN, sortKey, sortDir])

    // Paginated results
    const paginatedResults = useMemo(() => {
        const start = (page - 1) * pageSize
        return filteredResults.slice(start, start + pageSize)
    }, [filteredResults, page, pageSize])

    const totalPages = Math.ceil(filteredResults.length / pageSize)

    // Reset page when filters change
    useEffect(() => {
        setPage(1)
    }, [filterPositiveEV, maxOwnership, topN, sortKey, sortDir])

    // Run simulation
    const handleRunSim = async () => {
        if (lineups.length === 0) {
            setSimError('No lineups loaded. Select a build first.')
            return
        }
        setSimLoading(true)
        setSimError(null)
        setSimResult(null)
        try {
            const result = await runContestSim({
                game_date: selectedDate,
                lineups,
                archetype,
                field_size_bucket: fieldSizeBucket,
                entry_fee: entryFee,
            })
            setSimResult(result)
        } catch (err) {
            setSimError((err as Error).message)
        } finally {
            setSimLoading(false)
        }
    }

    // Selection handlers
    const toggleLineupSelection = (lineupId: number) => {
        setSelectedLineups(prev => {
            const next = new Set(prev)
            if (next.has(lineupId)) {
                next.delete(lineupId)
            } else {
                next.add(lineupId)
            }
            return next
        })
    }

    const selectAll = () => {
        setSelectedLineups(new Set(filteredResults.map(r => r.lineup_id)))
    }

    const clearSelection = () => {
        setSelectedLineups(new Set())
    }

    // Export handler
    const handleExport = async (type: 'selected' | 'view') => {
        if (!selectedSlate) return

        let lineupsToExport = []

        if (type === 'selected') {
            if (selectedLineups.size === 0) return
            lineupsToExport = resultsWithOwnership.filter(r => selectedLineups.has(r.lineup_id))
        } else {
            // Export current filtered view (Top N)
            lineupsToExport = filteredResults
        }

        const selectedPlayerIds = lineupsToExport.map(r => r.player_ids)
        const filename = `contest_sim_${selectedDate}_${lineupsToExport.length}lineups.csv`

        try {
            const blob = await exportCustomLineupsCSV(
                selectedDate,
                selectedSlate,
                selectedPlayerIds,
                `contest_sim_${selectedDate}`
            )
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = filename
            a.click()
            URL.revokeObjectURL(url)
        } catch (err) {
            alert('Export failed: ' + (err as Error).message)
        }
    }

    return (
        <div className="contest-sim-page">
            <header className="app-header">
                <div>
                    <h1>Contest Simulator</h1>
                    <p className="subtitle">EV calculation for optimizer lineups</p>
                </div>
                <div className="controls">
                    <label>
                        Date
                        <input
                            type="date"
                            value={selectedDate}
                            onChange={e => setSelectedDate(e.target.value)}
                        />
                    </label>
                    <label>
                        Slate
                        <select
                            value={selectedSlate ?? ''}
                            onChange={e => setSelectedSlate(Number(e.target.value) || null)}
                            disabled={slatesLoading}
                        >
                            {slates.length === 0 && <option value="">No slates</option>}
                            {slates.map(s => (
                                <option key={s.draft_group_id} value={s.draft_group_id}>
                                    {s.slate_type} ({s.n_contests} contests)
                                </option>
                            ))}
                        </select>
                    </label>
                </div>
            </header>

            <div className="sim-layout">
                {/* Configuration Sidebar */}
                <aside className="sim-sidebar">
                    <h3>Simulation Settings</h3>

                    <label>
                        Saved Build
                        <select
                            value={selectedBuildId ?? ''}
                            onChange={e => setSelectedBuildId(e.target.value || null)}
                            disabled={buildsLoading}
                        >
                            {savedBuilds.length === 0 && <option value="">No builds</option>}
                            {savedBuilds.map(b => (
                                <option key={b.job_id} value={b.job_id}>
                                    {b.job_id.slice(0, 8)} ({b.lineups_count} lineups)
                                </option>
                            ))}
                        </select>
                    </label>

                    <div className="lineup-count">
                        {lineups.length > 0 && <span>{lineups.length} lineups loaded</span>}
                    </div>

                    <hr />

                    <label>
                        Payout Archetype
                        <select
                            value={archetype}
                            onChange={e => setArchetype(e.target.value)}
                        >
                            {config?.payout_archetypes.map(a => (
                                <option key={a.key} value={a.key}>
                                    {a.label} ({(a.first_place_pct * 100).toFixed(0)}% to 1st)
                                </option>
                            )) ?? (
                                    <>
                                        <option value="top_heavy">Top Heavy</option>
                                        <option value="medium">Medium</option>
                                        <option value="flat">Flat</option>
                                    </>
                                )}
                        </select>
                    </label>

                    <label>
                        Field Size
                        <select
                            value={fieldSizeBucket}
                            onChange={e => setFieldSizeBucket(e.target.value)}
                        >
                            {config?.field_sizes.map(f => (
                                <option key={f.key} value={f.key}>
                                    {f.label}
                                </option>
                            )) ?? (
                                    <>
                                        <option value="small">Small (1-10K)</option>
                                        <option value="medium">Medium (10-50K)</option>
                                        <option value="massive">Massive (50K+)</option>
                                    </>
                                )}
                        </select>
                    </label>

                    <label>
                        Entry Fee
                        <input
                            type="number"
                            value={entryFee}
                            onChange={e => setEntryFee(Number(e.target.value))}
                            min={0.25}
                            max={1000}
                            step={0.25}
                        />
                    </label>

                    <button
                        className="run-sim-btn"
                        onClick={handleRunSim}
                        disabled={simLoading || lineups.length === 0}
                    >
                        {simLoading ? 'Running...' : '▶ Run Simulation'}
                    </button>

                    {simError && <div className="sim-error">{simError}</div>}
                </aside>

                {/* Results Area */}
                <main className="sim-main">
                    {simResult && (
                        <div className="lineup-cards-container">
                            {/* Summary Cards */}
                            <div className="sim-summary">
                                <div className="summary-card">
                                    <div className="card-label">Lineups</div>
                                    <div className="card-value">{simResult.stats.lineup_count}</div>
                                </div>
                                <div className="summary-card">
                                    <div className="card-label">Worlds</div>
                                    <div className="card-value">{simResult.stats.worlds_count.toLocaleString()}</div>
                                </div>
                                <div className="summary-card">
                                    <div className="card-label">Avg EV</div>
                                    <div className={`card-value ${simResult.stats.avg_ev >= 0 ? 'positive' : 'negative'}`}>
                                        {simResult.stats.avg_ev >= 0 ? '$' : '-$'}{Math.abs(simResult.stats.avg_ev).toFixed(2)}
                                    </div>
                                </div>
                                <div className="summary-card">
                                    <div className="card-label">Avg ROI</div>
                                    <div className={`card-value ${simResult.stats.avg_roi >= 0 ? 'positive' : 'negative'}`}>
                                        {simResult.stats.avg_roi >= 0 ? '+' : ''}{(simResult.stats.avg_roi * 100).toFixed(1)}%
                                    </div>
                                </div>
                                <div className="summary-card">
                                    <div className="card-label">+EV Lineups</div>
                                    <div className="card-value">
                                        {simResult.stats.positive_ev_count} / {simResult.stats.lineup_count}
                                    </div>
                                </div>
                                <div className="summary-card">
                                    <div className="card-label">Prize Pool</div>
                                    <div className="card-value">
                                        ${simResult.config.prize_pool.toLocaleString()}
                                    </div>
                                </div>
                            </div>

                            {/* Toolbar */}
                            <div className="lineup-cards-toolbar">
                                <div className="toolbar-group">
                                    <label>Sort:</label>
                                    <select
                                        value={sortKey}
                                        onChange={e => setSortKey(e.target.value as SortKey)}
                                    >
                                        <option value="expected_value">EV</option>
                                        <option value="roi">ROI</option>
                                        <option value="win_rate">Win%</option>
                                        <option value="top_1pct_rate">Top 1%</option>
                                        <option value="cash_rate">Cash%</option>
                                        <option value="p90">Ceiling (p90)</option>
                                        <option value="mean">Mean</option>
                                        <option value="total_own">Total Own%</option>
                                        <option value="lineup_id">Lineup #</option>
                                    </select>
                                    <button
                                        onClick={() => setSortDir(d => d === 'asc' ? 'desc' : 'asc')}
                                        style={{ padding: '0.35rem 0.5rem', background: '#0f172a', border: '1px solid #334155', borderRadius: '4px', color: '#f8fafc', cursor: 'pointer' }}
                                    >
                                        {sortDir === 'desc' ? '↓' : '↑'}
                                    </button>
                                </div>

                                <div className="toolbar-divider" />

                                <div className="toolbar-group">
                                    <input
                                        type="checkbox"
                                        id="filter-positive-ev"
                                        checked={filterPositiveEV}
                                        onChange={e => setFilterPositiveEV(e.target.checked)}
                                    />
                                    <label htmlFor="filter-positive-ev">+EV Only</label>
                                </div>

                                <div className="toolbar-group">
                                    <label>Max Own%:</label>
                                    <select
                                        value={maxOwnership ?? ''}
                                        onChange={e => setMaxOwnership(e.target.value ? Number(e.target.value) : null)}
                                    >
                                        <option value="">All</option>
                                        <option value="50">≤50%</option>
                                        <option value="75">≤75%</option>
                                        <option value="100">≤100%</option>
                                        <option value="150">≤150%</option>
                                    </select>
                                </div>

                                <div className="toolbar-divider" />

                                <div className="toolbar-group">
                                    <button onClick={selectAll} style={{ padding: '0.35rem 0.5rem', background: '#0f172a', border: '1px solid #334155', borderRadius: '4px', color: '#f8fafc', cursor: 'pointer' }}>
                                        Select All ({filteredResults.length})
                                    </button>
                                    <button onClick={clearSelection} style={{ padding: '0.35rem 0.5rem', background: '#0f172a', border: '1px solid #334155', borderRadius: '4px', color: '#f8fafc', cursor: 'pointer' }}>
                                        Clear
                                    </button>
                                </div>

                                <div className="toolbar-group">
                                    <label>Top N:</label>
                                    <select
                                        value={topN ?? ''}
                                        onChange={e => setTopN(e.target.value ? Number(e.target.value) : null)}
                                    >
                                        <option value="">All</option>
                                        <option value="150">Top 150</option>
                                        <option value="500">Top 500</option>
                                        <option value="1000">Top 1000</option>
                                        <option value="2000">Top 2000</option>
                                    </select>
                                </div>

                                <div className="toolbar-divider" />

                                <div className="toolbar-group">
                                    <button
                                        onClick={() => handleExport('view')}
                                        disabled={filteredResults.length === 0}
                                        style={{ padding: '0.35rem 0.5rem', background: '#0f172a', border: '1px solid #334155', borderRadius: '4px', color: '#f8fafc', cursor: 'pointer', marginRight: '0.5rem' }}
                                    >
                                        Export View ({filteredResults.length})
                                    </button>
                                    <button
                                        className="export-btn"
                                        onClick={() => handleExport('selected')}
                                        disabled={selectedLineups.size === 0}
                                    >
                                        Export Selected ({selectedLineups.size})
                                    </button>
                                </div>
                            </div>

                            {/* Pagination */}
                            <div className="pagination-controls">
                                <div className="page-info">
                                    Showing {((page - 1) * pageSize) + 1}-{Math.min(page * pageSize, filteredResults.length)} of {filteredResults.length}
                                </div>
                                <div className="page-buttons">
                                    <button
                                        className="page-btn"
                                        disabled={page === 1}
                                        onClick={() => setPage(p => Math.max(1, p - 1))}
                                    >
                                        Previous
                                    </button>
                                    <span style={{ margin: '0 0.5rem', color: '#94a3b8' }}>Page {page} of {totalPages}</span>
                                    <button
                                        className="page-btn"
                                        disabled={page >= totalPages}
                                        onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                                    >
                                        Next
                                    </button>

                                    <select
                                        value={pageSize}
                                        onChange={e => setPageSize(Number(e.target.value))}
                                        style={{ marginLeft: '1rem', padding: '0.2rem', background: '#0f172a', color: '#e2e8f0', border: '1px solid #334155', borderRadius: '4px' }}
                                    >
                                        <option value="20">20 / page</option>
                                        <option value="50">50 / page</option>
                                        <option value="100">100 / page</option>
                                        <option value="500">500 / page</option>
                                    </select>
                                </div>
                            </div>

                            {/* Cards Grid - Use paginatedResults */}
                            <div className="lineup-cards-grid">
                                {paginatedResults.map(result => (
                                    <LineupCard
                                        key={result.lineup_id}
                                        result={result}
                                        players={playerMap}
                                        selected={selectedLineups.has(result.lineup_id)}
                                        onToggleSelect={() => toggleLineupSelection(result.lineup_id)}
                                        highlighted={
                                            result.lineup_id === simResult.stats.best_ev_lineup_id
                                                ? 'best-ev'
                                                : result.lineup_id === simResult.stats.best_top1pct_lineup_id
                                                    ? 'best-ceiling'
                                                    : null
                                        }
                                    />
                                ))}
                            </div>
                        </div>
                    )}

                    {!simResult && !simLoading && (
                        <div className="sim-placeholder">
                            <h2>No Simulation Results</h2>
                            <p>Select a saved build and configure simulation settings, then click "Run Simulation".</p>
                        </div>
                    )}

                    {simLoading && (
                        <div className="sim-loading">
                            <div className="spinner"></div>
                            <p>Running simulation across all worlds...</p>
                        </div>
                    )}
                </main>
            </div>
        </div>
    )
}
