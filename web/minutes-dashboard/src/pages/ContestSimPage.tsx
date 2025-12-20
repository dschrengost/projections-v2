import { useCallback, useEffect, useMemo, useState } from 'react'
import {
    runContestSim,
    getContestSimConfig,
    ContestSimResponse,
    ConfigResponse,
    FieldLibrarySummary,
    buildFieldLibrary,
    getSavedSimBuilds,
    listFieldLibraries,
    loadSavedSimBuild,
    saveSimLineups,
    deleteSavedSimBuild,
    SavedSimBuildSummary,
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

    // Saved contest sim builds
    const [savedSimBuilds, setSavedSimBuilds] = useState<SavedSimBuildSummary[]>([])
    const [selectedSimBuildId, setSelectedSimBuildId] = useState<string | null>(null)
    const [selectedSimLineupId, setSelectedSimLineupId] = useState<string | null>(null)
    const [simBuildsLoading, setSimBuildsLoading] = useState(false)

    // Player pool for name resolution
    const [pool, setPool] = useState<PoolPlayer[]>([])

    // Configuration options
    const [config, setConfig] = useState<ConfigResponse | null>(null)
    const [archetype, setArchetype] = useState('medium')
    const [fieldSizeBucket, setFieldSizeBucket] = useState('medium')
    const [entryFee, setEntryFee] = useState(3.0)
    const [fieldMode, setFieldMode] = useState<'self_play' | 'generated_field'>('self_play')
    const [fieldLibraryVersion, setFieldLibraryVersion] = useState('v0')
    const [fieldLibraryK, setFieldLibraryK] = useState(2500)
    const [fieldCandidatePoolSize, setFieldCandidatePoolSize] = useState(40000)
    const [fieldLibraryRebuild, setFieldLibraryRebuild] = useState(false)
    const [fieldLibraryRebuildCandidates, setFieldLibraryRebuildCandidates] = useState(false)
    const [fieldLibraries, setFieldLibraries] = useState<FieldLibrarySummary[]>([])
    const [fieldLibrariesLoading, setFieldLibrariesLoading] = useState(false)
    const [fieldLibraryError, setFieldLibraryError] = useState<string | null>(null)

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

    // Load saved contest sim builds when date/slate changes
    useEffect(() => {
        const load = async () => {
            setSimBuildsLoading(true)
            try {
                const builds = await getSavedSimBuilds(selectedDate)
                setSavedSimBuilds(builds)
                const latestRun = builds.find(b => b.kind === 'run')?.build_id ?? null
                setSelectedSimBuildId(latestRun)
                const latestLineup = builds.find(b => b.kind === 'lineups')?.build_id ?? null
                setSelectedSimLineupId(latestLineup)
            } catch {
                setSavedSimBuilds([])
                setSelectedSimBuildId(null)
                setSelectedSimLineupId(null)
            } finally {
                setSimBuildsLoading(false)
            }
        }
        void load()
    }, [selectedDate, selectedSlate])

    // Load cached field libraries when date/slate changes
    useEffect(() => {
        if (!selectedSlate) {
            setFieldLibraries([])
            return
        }
        const load = async () => {
            setFieldLibrariesLoading(true)
            setFieldLibraryError(null)
            try {
                const libs = await listFieldLibraries(selectedDate, selectedSlate)
                setFieldLibraries(libs)
                if (libs.length > 0 && !libs.some(l => l.version === fieldLibraryVersion)) {
                    setFieldLibraryVersion(libs[0].version)
                }
            } catch (err) {
                setFieldLibraries([])
                setFieldLibraryError((err as Error).message)
            } finally {
                setFieldLibrariesLoading(false)
            }
        }
        void load()
    }, [selectedDate, selectedSlate, fieldLibraryVersion])

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

    // Load sim run when selection changes
    useEffect(() => {
        if (!selectedSimBuildId) {
            return
        }
        const load = async () => {
            try {
                const build = await loadSavedSimBuild(selectedDate, selectedSimBuildId)
                if (build.kind === 'run' && build.results && build.config && build.stats) {
                    setSimResult({
                        results: build.results,
                        config: build.config as unknown as ContestSimResponse['config'],
                        stats: build.stats as unknown as ContestSimResponse['stats'],
                        build_id: build.build_id,
                    })
                    setLineups(build.lineups ?? [])
                }
            } catch {
                setSimResult(null)
            }
        }
        void load()
    }, [selectedDate, selectedSimBuildId])

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

    const activeResults = useMemo(() => {
        if (selectedLineups.size === 0) return filteredResults
        return filteredResults.filter(r => selectedLineups.has(r.lineup_id))
    }, [filteredResults, selectedLineups])

    const activeSummary = useMemo(() => {
        if (!simResult || activeResults.length === 0) {
            return null
        }
        const avgEv = activeResults.reduce((sum, r) => sum + r.expected_value, 0) / activeResults.length
        const avgRoi = activeResults.reduce((sum, r) => sum + r.roi, 0) / activeResults.length
        const positiveEv = activeResults.filter(r => r.expected_value >= 0).length
        return {
            lineupCount: activeResults.length,
            avgEv,
            avgRoi,
            positiveEv,
            worlds: simResult.stats.worlds_count,
            prizePool: simResult.config.prize_pool,
        }
    }, [activeResults, simResult])

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

    const runSimWithLineups = useCallback(async (lineupsToRun: string[][]) => {
        if (lineupsToRun.length === 0) {
            setSimError('No lineups loaded. Select a build first.')
            return
        }
        setSimLoading(true)
        setSimError(null)
        setSimResult(null)
        try {
            const result = await runContestSim({
                game_date: selectedDate,
                draft_group_id: selectedSlate ?? undefined,
                lineups: lineupsToRun,
                archetype,
                field_size_bucket: fieldSizeBucket,
                entry_fee: entryFee,
                field_mode: fieldMode,
                field_library_version: fieldMode === 'generated_field' ? fieldLibraryVersion : undefined,
                field_library_k: fieldMode === 'generated_field' ? fieldLibraryK : undefined,
                field_candidate_pool_size: fieldMode === 'generated_field' ? fieldCandidatePoolSize : undefined,
                field_library_rebuild: fieldMode === 'generated_field' ? fieldLibraryRebuild : undefined,
                field_library_rebuild_candidates: fieldMode === 'generated_field' ? fieldLibraryRebuildCandidates : undefined,
            })
            setSimResult(result)
            const builds = await getSavedSimBuilds(selectedDate)
            setSavedSimBuilds(builds)
            setSelectedSimBuildId(result.build_id ?? builds.find(b => b.kind === 'run')?.build_id ?? null)
        } catch (err) {
            setSimError((err as Error).message)
        } finally {
            setSimLoading(false)
        }
    }, [
        selectedDate,
        selectedSlate,
        archetype,
        fieldSizeBucket,
        entryFee,
        fieldMode,
        fieldLibraryVersion,
        fieldLibraryK,
        fieldCandidatePoolSize,
        fieldLibraryRebuild,
        fieldLibraryRebuildCandidates,
    ])

    const handleBuildFieldLibrary = useCallback(async () => {
        if (!selectedSlate) {
            setFieldLibraryError('Select a slate first')
            return
        }
        setFieldLibrariesLoading(true)
        setFieldLibraryError(null)
        try {
            await buildFieldLibrary({
                game_date: selectedDate,
                draft_group_id: selectedSlate,
                version: fieldLibraryVersion,
                k: fieldLibraryK,
                candidate_pool_size: fieldCandidatePoolSize,
                rebuild: fieldLibraryRebuild,
                rebuild_candidates: fieldLibraryRebuildCandidates,
            })
            const libs = await listFieldLibraries(selectedDate, selectedSlate)
            setFieldLibraries(libs)
        } catch (err) {
            setFieldLibraryError((err as Error).message)
        } finally {
            setFieldLibrariesLoading(false)
        }
    }, [
        selectedDate,
        selectedSlate,
        fieldLibraryVersion,
        fieldLibraryK,
        fieldCandidatePoolSize,
        fieldLibraryRebuild,
        fieldLibraryRebuildCandidates,
    ])

    // Run simulation
    const handleRunSim = async () => {
        await runSimWithLineups(lineups)
    }

    // Load saved sim lineups when selection changes
    useEffect(() => {
        if (!selectedSimLineupId) {
            return
        }
        const load = async () => {
            try {
                const build = await loadSavedSimBuild(selectedDate, selectedSimLineupId)
                if (build.kind === 'lineups') {
                    setLineups(build.lineups ?? [])
                    if (build.results && build.config && build.stats) {
                        setSimResult({
                            results: build.results,
                            config: build.config as unknown as ContestSimResponse['config'],
                            stats: build.stats as unknown as ContestSimResponse['stats'],
                            build_id: build.build_id,
                        })
                    } else {
                        setSimResult(null)
                        setSimError('Saved lineups missing snapshot results. Re-save from a sim run.')
                    }
                }
            } catch {
                // ignore
            }
        }
        void load()
    }, [selectedDate, selectedSimLineupId])

    const handleSaveSimLineups = async () => {
        if (!selectedSlate) return
        if (filteredResults.length === 0) return
        const defaultName = `Sim lineups (${filteredResults.length})`
        const name = prompt('Save lineups as:', defaultName)?.trim()
        if (!name) return
        try {
            const lineupsToSave = filteredResults.map(r => r.player_ids)
            const resultIds = new Set(filteredResults.map(r => r.lineup_id))
            const resultsToSave = simResult?.results.filter(r => resultIds.has(r.lineup_id)) ?? null
            const saved = await saveSimLineups(
                selectedDate,
                selectedSlate,
                name,
                lineupsToSave,
                resultsToSave,
                simResult?.config ?? null,
                simResult?.stats ?? null,
            )
            const builds = await getSavedSimBuilds(selectedDate)
            setSavedSimBuilds(builds)
            setSelectedSimLineupId(saved.build_id)
        } catch (err) {
            alert('Failed to save sim lineups: ' + (err as Error).message)
        }
    }

    const handleDeleteSimBuild = async (buildId: string) => {
        if (!confirm('Delete this saved sim build?')) return
        try {
            await deleteSavedSimBuild(selectedDate, buildId)
            const builds = await getSavedSimBuilds(selectedDate)
            setSavedSimBuilds(builds)
            if (selectedSimBuildId === buildId) {
                setSelectedSimBuildId(null)
                setSimResult(null)
            }
            if (selectedSimLineupId === buildId) {
                setSelectedSimLineupId(null)
                setSimResult(null)
            }
        } catch (err) {
            alert('Failed to delete sim build: ' + (err as Error).message)
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
                                    {s.slate_type} ({s.draft_group_id}) - {s.n_contests} contests
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
                        Field Model
                        <select
                            value={fieldMode}
                            onChange={e => setFieldMode(e.target.value as 'self_play' | 'generated_field')}
                        >
                            <option value="self_play">Self-play (your lineups as field)</option>
                            <option value="generated_field">Representative field (QuickBuild)</option>
                        </select>
                    </label>

                    {fieldMode === 'generated_field' && (
                        <>
                            <label>
                                Field Library Version
                                <select
                                    value={fieldLibraryVersion}
                                    onChange={e => setFieldLibraryVersion(e.target.value)}
                                >
                                    {fieldLibraries.length === 0 && <option value="v0">v0</option>}
                                    {fieldLibraries.map(l => (
                                        <option key={l.version} value={l.version}>
                                            {l.version} ({l.selected_k} lineups)
                                        </option>
                                    ))}
                                </select>
                            </label>

                            <label>
                                Field K (unique lineups)
                                <input
                                    type="number"
                                    value={fieldLibraryK}
                                    onChange={e => setFieldLibraryK(Number(e.target.value))}
                                    min={100}
                                    max={5000}
                                    step={100}
                                />
                            </label>

                            <label>
                                Candidate Pool Size
                                <input
                                    type="number"
                                    value={fieldCandidatePoolSize}
                                    onChange={e => setFieldCandidatePoolSize(Number(e.target.value))}
                                    min={5000}
                                    max={100000}
                                    step={5000}
                                />
                            </label>

                            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <input
                                    type="checkbox"
                                    checked={fieldLibraryRebuild}
                                    onChange={e => setFieldLibraryRebuild(e.target.checked)}
                                />
                                Force rebuild field library
                            </label>

                            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <input
                                    type="checkbox"
                                    checked={fieldLibraryRebuildCandidates}
                                    onChange={e => setFieldLibraryRebuildCandidates(e.target.checked)}
                                />
                                Force rebuild candidate pool (slow)
                            </label>

                            <button
                                className="run-sim-btn"
                                onClick={handleBuildFieldLibrary}
                                disabled={fieldLibrariesLoading || !selectedSlate}
                            >
                                {fieldLibrariesLoading ? 'Building...' : 'Build Field Library'}
                            </button>

                            {fieldLibraryError && <div className="sim-error">{fieldLibraryError}</div>}
                        </>
                    )}

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
                    <section className="saved-builds-section">
                        <div className="saved-builds-header">
                            <h3>Saved Sim Runs</h3>
                            <div className="saved-builds-header-actions">
                                {simBuildsLoading && <span className="muted">Loading...</span>}
                            </div>
                        </div>
                        <div className="saved-builds-list">
                            {savedSimBuilds.filter(b => b.kind === 'run').length === 0 && (
                                <span className="muted">No sim runs yet.</span>
                            )}
                            {savedSimBuilds.filter(b => b.kind === 'run').map(b => (
                                <div key={b.build_id} className={`saved-build-card ${selectedSimBuildId === b.build_id ? 'selected' : ''}`}>
                                    <div className="saved-build-info">
                                        <span className="saved-build-count">{b.lineups_count} lineups</span>
                                        <span className="saved-build-time">{new Date(b.created_at).toLocaleTimeString()}</span>
                                        {typeof b.stats?.avg_ev === 'number' && (
                                            <span className="saved-build-stats">EV {b.stats.avg_ev.toFixed(2)}</span>
                                        )}
                                    </div>
                                    <div className="saved-build-actions">
                                        <button
                                            className="load-btn"
                                            onClick={() => {
                                                setSelectedSimLineupId(null)
                                                setSelectedSimBuildId(b.build_id)
                                            }}
                                        >
                                            Load
                                        </button>
                                        <button
                                            className="delete-btn"
                                            onClick={() => handleDeleteSimBuild(b.build_id)}
                                            title="Delete"
                                        >
                                            ×
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>

                    <section className="saved-builds-section">
                        <div className="saved-builds-header">
                            <h3>Saved Sim Lineups</h3>
                        </div>
                        <div className="saved-builds-list">
                            {savedSimBuilds.filter(b => b.kind === 'lineups').length === 0 && (
                                <span className="muted">No saved lineups yet.</span>
                            )}
                            {savedSimBuilds.filter(b => b.kind === 'lineups').map(b => (
                                <div key={b.build_id} className={`saved-build-card ${selectedSimLineupId === b.build_id ? 'selected' : ''}`}>
                                    <div className="saved-build-info">
                                        <span className="saved-build-count">{b.lineups_count} lineups</span>
                                        <span className="saved-build-time">{b.name ?? b.build_id.slice(0, 8)}</span>
                                    </div>
                                    <div className="saved-build-actions">
                                        <button
                                            className="load-btn"
                                            onClick={() => {
                                                setSelectedSimBuildId(null)
                                                setSelectedSimLineupId(b.build_id)
                                            }}
                                        >
                                            Load
                                        </button>
                                        <button
                                            className="delete-btn"
                                            onClick={() => handleDeleteSimBuild(b.build_id)}
                                            title="Delete"
                                        >
                                            ×
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>

                    {simResult && (
                        <div className="lineup-cards-container">
                            {/* Summary Cards */}
                            <div className="sim-summary compact">
                                <div className="summary-card">
                                    <div className="card-label">Lineups</div>
                                    <div className="card-value">{activeSummary?.lineupCount ?? simResult.stats.lineup_count}</div>
                                </div>
                                <div className="summary-card">
                                    <div className="card-label">Worlds</div>
                                    <div className="card-value">{(activeSummary?.worlds ?? simResult.stats.worlds_count).toLocaleString()}</div>
                                </div>
                                <div className="summary-card">
                                    <div className="card-label">Avg EV</div>
                                    <div className={`card-value ${(activeSummary?.avgEv ?? simResult.stats.avg_ev) >= 0 ? 'positive' : 'negative'}`}>
                                        {(activeSummary?.avgEv ?? simResult.stats.avg_ev) >= 0 ? '$' : '-$'}
                                        {Math.abs(activeSummary?.avgEv ?? simResult.stats.avg_ev).toFixed(2)}
                                    </div>
                                </div>
                                <div className="summary-card">
                                    <div className="card-label">Avg ROI</div>
                                    <div className={`card-value ${(activeSummary?.avgRoi ?? simResult.stats.avg_roi) >= 0 ? 'positive' : 'negative'}`}>
                                        {(activeSummary?.avgRoi ?? simResult.stats.avg_roi) >= 0 ? '+' : ''}
                                        {((activeSummary?.avgRoi ?? simResult.stats.avg_roi) * 100).toFixed(1)}%
                                    </div>
                                </div>
                                <div className="summary-card">
                                    <div className="card-label">+EV Lineups</div>
                                    <div className="card-value">
                                        {activeSummary?.positiveEv ?? simResult.stats.positive_ev_count} / {activeSummary?.lineupCount ?? simResult.stats.lineup_count}
                                    </div>
                                </div>
                                <div className="summary-card">
                                    <div className="card-label">Prize Pool</div>
                                    <div className="card-value">
                                        ${(activeSummary?.prizePool ?? simResult.config.prize_pool).toLocaleString()}
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
                                    <input
                                        type="number"
                                        min={1}
                                        placeholder="All"
                                        value={topN ?? ''}
                                        onChange={e => setTopN(e.target.value ? Number(e.target.value) : null)}
                                        style={{ width: '90px', padding: '0.25rem 0.4rem', background: '#0f172a', color: '#e2e8f0', border: '1px solid #334155', borderRadius: '4px' }}
                                    />
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
                                    <button
                                        onClick={handleSaveSimLineups}
                                        disabled={filteredResults.length === 0}
                                        style={{ padding: '0.35rem 0.5rem', background: '#1e3a5f', border: '1px solid #3b82f6', borderRadius: '4px', color: '#60a5fa', cursor: 'pointer', marginLeft: '0.5rem' }}
                                    >
                                        Save Sim Lineups ({filteredResults.length})
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
