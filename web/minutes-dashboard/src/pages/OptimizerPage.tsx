import { useEffect, useMemo, useState } from 'react'
import {
    getSlates,
    getPlayerPool,
    startBuild,
    getBuildStatus,
    getBuildLineups,
    exportLineupsCSV,
    exportCustomLineupsCSV,
    getSavedBuilds,
    loadSavedBuild,
    deleteSavedBuild,
    Slate,
    PoolPlayer,
    JobStatus,
    LineupRow,
    QuickBuildRequest,
    SavedBuild,
    GameInfo,
} from '../api/optimizer'
import { formatSalary } from '../utils'

type SortKey = 'name' | 'team' | 'salary' | 'proj' | 'own_proj' | 'value'

type LineupGroup = {
    id: string
    name: string
    lineup_ids: number[]
    created_at: string
}

const todayISO = () => new Date().toISOString().slice(0, 10)

export default function OptimizerPage() {
    // Date and slate selection
    const [selectedDate, setSelectedDate] = useState(todayISO())
    const [slates, setSlates] = useState<Slate[]>([])
    const [selectedSlate, setSelectedSlate] = useState<number | null>(null)
    const [slatesLoading, setSlatesLoading] = useState(false)
    const [slatesError, setSlatesError] = useState<string | null>(null)

    // Player pool
    const [pool, setPool] = useState<PoolPlayer[]>([])
    const [poolLoading, setPoolLoading] = useState(false)
    const [poolError, setPoolError] = useState<string | null>(null)

    // Lock/ban players
    const [lockedIds, setLockedIds] = useState<Set<string>>(new Set())
    const [bannedIds, setBannedIds] = useState<Set<string>>(new Set())

    // Filter and sort
    const [filter, setFilter] = useState('')
    const [sortKey, setSortKey] = useState<SortKey>('proj')
    const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

    // Build config
    const [maxPool, setMaxPool] = useState(5000)
    const [builds, setBuilds] = useState(22)
    const [minUniq, setMinUniq] = useState(1)
    const [globalTeamLimit, setGlobalTeamLimit] = useState(4)
    const [minSalary, setMinSalary] = useState<number | null>(null)
    const [maxSalary, setMaxSalary] = useState<number | null>(50000)
    const [minProj, setMinProj] = useState<number | null>(null)
    const [randomnessPct, setRandomnessPct] = useState(0)

    // Job state
    const [currentJob, setCurrentJob] = useState<JobStatus | null>(null)
    const [lineups, setLineups] = useState<LineupRow[]>([])
    const [buildError, setBuildError] = useState<string | null>(null)

    // Lineup filter
    const [lineupFilter, setLineupFilter] = useState('')
    const [showCount, setShowCount] = useState(50)
    const [lineupSort, setLineupSort] = useState<'default' | 'proj-desc' | 'proj-asc' | 'salary-desc' | 'salary-asc' | 'p90-desc' | 'p90-asc' | 'own-desc' | 'own-asc'>('default')
    const [minLineupProj, setMinLineupProj] = useState<number | null>(null)
    const [maxLineupOwn, setMaxLineupOwn] = useState<number | null>(null)
    const [minLineupP90, setMinLineupP90] = useState<number | null>(null)
    const [selectedLineupIds, setSelectedLineupIds] = useState<Set<number>>(new Set())
    const [lineupGroups, setLineupGroups] = useState<LineupGroup[]>([])
    const [activeLineupGroupId, setActiveLineupGroupId] = useState<string>('')

    // Saved builds
    const [savedBuilds, setSavedBuilds] = useState<SavedBuild[]>([])
    const [savedBuildsLoading, setSavedBuildsLoading] = useState(false)
    const [selectedBuildIds, setSelectedBuildIds] = useState<Set<string>>(new Set())

    // Game exclusion
    const [excludedGames, setExcludedGames] = useState<Set<string>>(new Set())

    // Get current slate's games
    const currentSlateGames = useMemo(() => {
        const slate = slates.find(s => s.draft_group_id === selectedSlate)
        return slate?.games ?? []
    }, [slates, selectedSlate])

    // Check if stddev is available in pool (needed for randomness feature)
    const hasStddev = useMemo(() =>
        pool.some(p => p.stddev != null && p.stddev > 0), [pool])

    // Load slates when date changes
    useEffect(() => {
        const loadSlates = async () => {
            setSlatesLoading(true)
            setSlatesError(null)
            try {
                const data = await getSlates(selectedDate)
                setSlates(data)
                // Auto-select first non-showdown slate
                const mainSlate = data.find(s => s.slate_type !== 'showdown')
                setSelectedSlate(mainSlate?.draft_group_id ?? data[0]?.draft_group_id ?? null)
            } catch (err) {
                setSlatesError((err as Error).message)
                setSlates([])
                setSelectedSlate(null)
            } finally {
                setSlatesLoading(false)
            }
        }
        void loadSlates()
    }, [selectedDate])

    // Load player pool when slate changes
    useEffect(() => {
        if (!selectedSlate) {
            setPool([])
            return
        }
        const loadPool = async () => {
            setPoolLoading(true)
            setPoolError(null)
            try {
                const data = await getPlayerPool(selectedDate, selectedSlate)
                setPool(data)
                // Reset locks/bans on new pool
                setLockedIds(new Set())
                setBannedIds(new Set())
            } catch (err) {
                setPoolError((err as Error).message)
                setPool([])
            } finally {
                setPoolLoading(false)
            }
        }
        void loadPool()
    }, [selectedDate, selectedSlate])

    // Poll job status
    useEffect(() => {
        if (!currentJob || currentJob.status === 'completed' || currentJob.status === 'failed') {
            return
        }
        const interval = setInterval(async () => {
            try {
                const status = await getBuildStatus(currentJob.job_id)
                setCurrentJob(status)
                if (status.status === 'completed') {
                    const result = await getBuildLineups(status.job_id)
                    setLineups(result.lineups)
                    // Refresh saved builds list
                    refreshSavedBuilds()
                }
            } catch (err) {
                setBuildError((err as Error).message)
            }
        }, 1000)
        return () => clearInterval(interval)
    }, [currentJob])

    // Load saved builds when date/slate changes
    const refreshSavedBuilds = async () => {
        if (!selectedSlate) return
        setSavedBuildsLoading(true)
        try {
            const builds = await getSavedBuilds(selectedDate, selectedSlate)
            setSavedBuilds(builds)
        } catch (err) {
            console.error('Failed to load saved builds:', err)
            setSavedBuilds([])
        } finally {
            setSavedBuildsLoading(false)
        }
    }

    useEffect(() => {
        void refreshSavedBuilds()
    }, [selectedDate, selectedSlate])

    // Load a saved build
    const handleLoadSavedBuild = async (jobId: string) => {
        try {
            const build = await loadSavedBuild(selectedDate, jobId)
            if (build.lineups) {
                setLineups(build.lineups)
                setCurrentJob(null) // Clear any running job state
            }
        } catch (err) {
            setBuildError((err as Error).message)
        }
    }

    // Delete a saved build
    const handleDeleteSavedBuild = async (jobId: string) => {
        if (!confirm('Delete this saved build?')) return
        try {
            await deleteSavedBuild(selectedDate, jobId)
            await refreshSavedBuilds()
            // Remove from selection if selected
            setSelectedBuildIds(prev => {
                const next = new Set(prev)
                next.delete(jobId)
                return next
            })
        } catch (err) {
            alert('Failed to delete: ' + (err as Error).message)
        }
    }

    // Toggle build selection for joining
    const toggleBuildSelection = (jobId: string) => {
        setSelectedBuildIds(prev => {
            const next = new Set(prev)
            if (next.has(jobId)) {
                next.delete(jobId)
            } else {
                next.add(jobId)
            }
            return next
        })
    }

    // Join selected builds
    const handleJoinBuilds = async () => {
        if (selectedBuildIds.size < 2) return

        const buildName = prompt('Name for merged build:', `Merged (${selectedBuildIds.size} builds)`)
        if (!buildName) return

        try {
            // Load all selected builds
            const allLineups: LineupRow[] = []
            for (const jobId of selectedBuildIds) {
                const build = await loadSavedBuild(selectedDate, jobId)
                if (build.lineups) {
                    allLineups.push(...build.lineups)
                }
            }

            // De-duplicate by player_ids set
            const seen = new Set<string>()
            const unique: LineupRow[] = []
            for (const lu of allLineups) {
                const key = [...lu.player_ids].sort().join(',')
                if (!seen.has(key)) {
                    seen.add(key)
                    unique.push({ ...lu, lineup_id: unique.length })
                }
            }

            setLineups(unique)
            setCurrentJob(null)
            setSelectedBuildIds(new Set())
            alert(`Merged ${selectedBuildIds.size} builds into "${buildName}": ${unique.length} unique lineups`)
        } catch (err) {
            alert('Failed to join builds: ' + (err as Error).message)
        }
    }

    // Filtered and sorted pool
    const filteredPool = useMemo(() => {
        let filtered = pool.slice()

        // Filter by minimum projection
        if (minProj != null) {
            filtered = filtered.filter(p => p.proj >= minProj)
        }

        const text = filter.trim().toLowerCase()
        if (text) {
            filtered = filtered.filter(p =>
                p.name.toLowerCase().includes(text) ||
                p.team.toLowerCase().includes(text) ||
                p.positions.some(pos => pos.toLowerCase().includes(text))
            )
        }
        filtered.sort((a, b) => {
            let left: number | string
            let right: number | string
            switch (sortKey) {
                case 'name': left = a.name; right = b.name; break
                case 'team': left = a.team; right = b.team; break
                case 'salary': left = a.salary; right = b.salary; break
                case 'proj': left = a.proj; right = b.proj; break
                case 'own_proj': left = a.own_proj ?? 0; right = b.own_proj ?? 0; break
                case 'value': left = a.proj / (a.salary / 1000); right = b.proj / (b.salary / 1000); break
                default: left = a.proj; right = b.proj
            }
            if (typeof left === 'number' && typeof right === 'number') {
                return sortDir === 'asc' ? left - right : right - left
            }
            return sortDir === 'asc' ? String(left).localeCompare(String(right)) : String(right).localeCompare(String(left))
        })
        return filtered
    }, [pool, filter, sortKey, sortDir, minProj])

    // Toggle lock/ban
    const toggleLock = (id: string) => {
        setLockedIds(prev => {
            const next = new Set(prev)
            if (next.has(id)) next.delete(id)
            else next.add(id)
            return next
        })
        // Remove from banned if locking
        setBannedIds(prev => {
            const next = new Set(prev)
            next.delete(id)
            return next
        })
    }

    const toggleBan = (id: string) => {
        setBannedIds(prev => {
            const next = new Set(prev)
            if (next.has(id)) next.delete(id)
            else next.add(id)
            return next
        })
        // Remove from locked if banning
        setLockedIds(prev => {
            const next = new Set(prev)
            next.delete(id)
            return next
        })
    }

    // Start build
    const handleStartBuild = async () => {
        if (!selectedSlate) return
        setBuildError(null)
        setLineups([])
        try {
            const request: QuickBuildRequest = {
                date: selectedDate,
                draft_group_id: selectedSlate,
                site: 'dk',
                max_pool: maxPool,
                builds,
                per_build: Math.ceil(maxPool / builds) + 500,
                min_uniq: minUniq,
                global_team_limit: globalTeamLimit,
                min_salary: minSalary,
                max_salary: maxSalary,
                lock_ids: Array.from(lockedIds),
                ban_ids: Array.from(bannedIds),
                exclude_games: Array.from(excludedGames),
                enum_enable: maxPool >= 5000,
                randomness_pct: randomnessPct > 0 && hasStddev ? randomnessPct : undefined,
            }
            const job = await startBuild(request)
            setCurrentJob(job)
        } catch (err) {
            setBuildError((err as Error).message)
        }
    }

    // Export CSV
    const handleExport = async () => {
        if (!currentJob?.job_id) return
        try {
            const blob = await exportLineupsCSV(currentJob.job_id)
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = `lineups_${selectedDate}_${currentJob.job_id.slice(0, 8)}.csv`
            a.click()
            URL.revokeObjectURL(url)
        } catch (err) {
            alert('Export failed: ' + (err as Error).message)
        }
    }

    // Export saved build CSV
    const handleExportBuild = async (jobId: string) => {
        try {
            const blob = await exportLineupsCSV(jobId)
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = `lineups_${selectedDate}_${jobId.slice(0, 8)}.csv`
            a.click()
            URL.revokeObjectURL(url)
        } catch (err) {
            alert('Export failed: ' + (err as Error).message)
        }
    }

    const makeLineupGroupId = (): string => {
        const id = globalThis.crypto?.randomUUID?.()
        if (id) return id
        return `${Date.now()}_${Math.random().toString(16).slice(2)}`
    }

    // Toggle lineup selection
    const toggleLineupSelection = (lineupId: number) => {
        setSelectedLineupIds(prev => {
            const next = new Set(prev)
            if (next.has(lineupId)) {
                next.delete(lineupId)
            } else {
                next.add(lineupId)
            }
            return next
        })
    }

    // Select all visible lineups
    const selectAllVisible = () => {
        const visibleIds = filteredLineups.slice(0, showCount).map(lu => lu.lineup_id)
        setSelectedLineupIds(new Set(visibleIds))
    }

    const selectAllFiltered = () => {
        const filteredIds = filteredLineups.map(lu => lu.lineup_id)
        setSelectedLineupIds(new Set(filteredIds))
    }

    // Clear selection
    const clearSelection = () => {
        setSelectedLineupIds(new Set())
    }

    const safeFilenamePart = (input: string): string => {
        const cleaned = input
            .trim()
            .replace(/\s+/g, '_')
            .replace(/[^a-zA-Z0-9._-]/g, '')
        return cleaned || 'group'
    }

    const downloadCSVBlob = (blob: Blob, filename: string) => {
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = filename
        a.click()
        URL.revokeObjectURL(url)
    }

    const exportLineupsByIds = async (lineupIds: number[], filenamePrefix: string) => {
        if (!selectedSlate) return
        if (lineupIds.length === 0) return

        const byId = new Map(lineups.map(lu => [lu.lineup_id, lu]))
        const exportLineups = lineupIds.map(id => byId.get(id)).filter(Boolean) as LineupRow[]
        const payload = exportLineups.map(lu => lu.player_ids)
        if (payload.length === 0) return

        try {
            const blob = await exportCustomLineupsCSV(selectedDate, selectedSlate, payload, filenamePrefix)
            downloadCSVBlob(
                blob,
                `${safeFilenamePart(filenamePrefix)}_${selectedDate}_${payload.length}.csv`,
            )
        } catch (err) {
            alert('Export failed: ' + (err as Error).message)
        }
    }

    // Export selected lineups as CSV
    const exportSelectedCSV = async () => {
        if (selectedLineupIds.size === 0) return
        await exportLineupsByIds(
            Array.from(selectedLineupIds),
            `selected_lineups_${selectedLineupIds.size}`,
        )
    }

    const createGroupFromSelection = () => {
        if (selectedLineupIds.size === 0) return
        const defaultName = `Group ${lineupGroups.length + 1} (${selectedLineupIds.size})`
        const name = prompt('Group name:', defaultName)?.trim()
        if (!name) return
        const group: LineupGroup = {
            id: makeLineupGroupId(),
            name,
            lineup_ids: Array.from(selectedLineupIds),
            created_at: new Date().toISOString(),
        }
        setLineupGroups(prev => [...prev, group])
        setActiveLineupGroupId(group.id)
    }

    const deleteActiveGroup = () => {
        if (!activeLineupGroupId) return
        const g = lineupGroups.find(gr => gr.id === activeLineupGroupId)
        if (!g) return
        if (!confirm(`Delete group "${g.name}"?`)) return
        setLineupGroups(prev => prev.filter(gr => gr.id !== activeLineupGroupId))
        setActiveLineupGroupId('')
    }

    const exportActiveGroupCSV = async () => {
        if (!activeLineupGroupId) return
        const g = lineupGroups.find(gr => gr.id === activeLineupGroupId)
        if (!g) return
        await exportLineupsByIds(g.lineup_ids, `group_${g.name}`)
    }

    const selectActiveGroupLineups = () => {
        if (!activeLineupGroupId) return
        const g = lineupGroups.find(gr => gr.id === activeLineupGroupId)
        if (!g) return
        setSelectedLineupIds(new Set(g.lineup_ids))
    }

    // Get player name by ID for lineup display
    const playerMap = useMemo(() => {
        const map = new Map<string, PoolPlayer>()
        pool.forEach(p => map.set(p.player_id, p))
        return map
    }, [pool])

    useEffect(() => {
        setSelectedLineupIds(new Set())
        setLineupGroups([])
        setActiveLineupGroupId('')
    }, [lineups])

    // Filter and sort lineups
    const filteredLineups = useMemo(() => {
        let result = lineups.slice()

        // Filter by player name
        if (lineupFilter.trim()) {
            const text = lineupFilter.trim().toLowerCase()
            result = result.filter(lu =>
                lu.player_ids.some(id => {
                    const p = playerMap.get(id)
                    return p && p.name.toLowerCase().includes(text)
                })
            )
        }

        // Filter by lineup metrics
        result = result.filter(lu => {
            const proj = lu.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.proj ?? 0), 0)
            const own = lu.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.own_proj ?? 0), 0)
            const p90 = lu.p90 ?? lu.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.p90 ?? 0), 0)

            if (minLineupProj != null && proj < minLineupProj) return false
            if (maxLineupOwn != null && own > maxLineupOwn) return false
            if (minLineupP90 != null && p90 < minLineupP90) return false
            return true
        })

        // Sort
        if (lineupSort !== 'default') {
            result.sort((a, b) => {
                const aProj = a.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.proj ?? 0), 0)
                const bProj = b.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.proj ?? 0), 0)
                const aSal = a.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.salary ?? 0), 0)
                const bSal = b.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.salary ?? 0), 0)
                const aP90 = a.p90 ?? a.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.p90 ?? 0), 0)
                const bP90 = b.p90 ?? b.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.p90 ?? 0), 0)
                const aOwn = a.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.own_proj ?? 0), 0)
                const bOwn = b.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.own_proj ?? 0), 0)

                switch (lineupSort) {
                    case 'proj-desc': return bProj - aProj
                    case 'proj-asc': return aProj - bProj
                    case 'salary-desc': return bSal - aSal
                    case 'salary-asc': return aSal - bSal
                    case 'p90-desc': return bP90 - aP90
                    case 'p90-asc': return aP90 - bP90
                    case 'own-desc': return bOwn - aOwn
                    case 'own-asc': return aOwn - bOwn
                    default: return 0
                }
            })
        }

        return result
    }, [lineups, lineupFilter, lineupSort, playerMap, minLineupProj, maxLineupOwn, minLineupP90])


    const toggleSort = (key: SortKey) => {
        if (sortKey === key) {
            setSortDir(prev => prev === 'asc' ? 'desc' : 'asc')
        } else {
            setSortKey(key)
            setSortDir(key === 'name' || key === 'team' ? 'asc' : 'desc')
        }
    }

    const formatProj = (val: number | undefined | null) =>
        val != null ? val.toFixed(1) : '-'

    const formatOwn = (val: number | undefined | null) =>
        val != null ? val.toFixed(1) + '%' : '-'

    const formatValue = (p: PoolPlayer) =>
        (p.proj / (p.salary / 1000)).toFixed(2)

    return (
        <div className="optimizer-page">
            {/* Header */}
            <header className="app-header">
                <div>
                    <h1>Lineup Optimizer</h1>
                    <p className="subtitle">QuickBuild lineup pool generation</p>
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

            {(slatesError || poolError) && (
                <section className="status-bar">
                    <span className="error">{slatesError || poolError}</span>
                </section>
            )}

            <div className="optimizer-layout">
                {/* Settings Sidebar */}
                <aside className="optimizer-sidebar">
                    <h3>Build Settings</h3>

                    <label>
                        Max Lineups
                        <input
                            type="number"
                            value={maxPool}
                            onChange={e => setMaxPool(Number(e.target.value))}
                            min={100}
                            max={100000}
                            step={500}
                        />
                    </label>

                    <label>
                        Workers
                        <input
                            type="number"
                            value={builds}
                            onChange={e => setBuilds(Number(e.target.value))}
                            min={1}
                            max={24}
                        />
                    </label>

                    <label>
                        Min Uniques
                        <input
                            type="number"
                            value={minUniq}
                            onChange={e => setMinUniq(Number(e.target.value))}
                            min={1}
                            max={8}
                        />
                    </label>

                    <label>
                        Team Limit
                        <input
                            type="number"
                            value={globalTeamLimit}
                            onChange={e => setGlobalTeamLimit(Number(e.target.value))}
                            min={1}
                            max={8}
                        />
                    </label>

                    <div className="salary-range">
                        <label>
                            Min Salary
                            <input
                                type="number"
                                value={minSalary ?? ''}
                                onChange={e => setMinSalary(e.target.value ? Number(e.target.value) : null)}
                                placeholder="No min"
                                min={0}
                                max={50000}
                                step={100}
                            />
                        </label>
                        <label>
                            Max Salary
                            <input
                                type="number"
                                value={maxSalary ?? ''}
                                onChange={e => setMaxSalary(e.target.value ? Number(e.target.value) : null)}
                                placeholder="No max"
                                min={0}
                                max={50000}
                                step={100}
                            />
                        </label>
                    </div>

                    <label>
                        Min Projection
                        <input
                            type="number"
                            value={minProj ?? ''}
                            onChange={e => setMinProj(e.target.value ? Number(e.target.value) : null)}
                            placeholder="No min"
                            min={0}
                            step={5}
                        />
                    </label>

                    {/* Randomness Control */}
                    <div className={`randomness-control ${!hasStddev ? 'disabled' : ''}`}>
                        <label>
                            Randomness: {randomnessPct}%
                            <input
                                type="range"
                                min={0}
                                max={100}
                                step={5}
                                value={randomnessPct}
                                onChange={e => setRandomnessPct(Number(e.target.value))}
                                disabled={!hasStddev}
                            />
                        </label>
                        {!hasStddev && (
                            <small className="hint">Requires sim projections with stddev</small>
                        )}
                        {hasStddev && randomnessPct > 0 && (
                            <small className="hint">Adds variance-aware noise for diversity</small>
                        )}
                    </div>

                    {/* Game Filters */}

                    {currentSlateGames.length > 0 && (
                        <div className="game-filters">
                            <h4>Games ({currentSlateGames.length - excludedGames.size} of {currentSlateGames.length})</h4>
                            <div className="game-filter-list">
                                {currentSlateGames.map(game => {
                                    const isExcluded = excludedGames.has(game.matchup)
                                    const startTime = game.start_time
                                        ? new Date(game.start_time).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
                                        : ''
                                    return (
                                        <label key={game.matchup} className={`game-filter-item ${isExcluded ? 'excluded' : ''}`}>
                                            <input
                                                type="checkbox"
                                                checked={!isExcluded}
                                                onChange={() => {
                                                    setExcludedGames(prev => {
                                                        const next = new Set(prev)
                                                        if (next.has(game.matchup)) {
                                                            next.delete(game.matchup)
                                                        } else {
                                                            next.add(game.matchup)
                                                        }
                                                        return next
                                                    })
                                                }}
                                            />
                                            <span className="game-matchup">{game.matchup}</span>
                                            {startTime && <span className="game-time">{startTime}</span>}
                                        </label>
                                    )
                                })}
                            </div>
                            {excludedGames.size > 0 && (
                                <button
                                    className="clear-exclusions"
                                    onClick={() => setExcludedGames(new Set())}
                                >
                                    Include All Games
                                </button>
                            )}
                        </div>
                    )}

                    <div className="lock-ban-summary">
                        <span>🔒 Locked: {lockedIds.size}</span>
                        <span>🚫 Banned: {bannedIds.size}</span>
                    </div>

                    <button
                        className="build-btn"
                        onClick={handleStartBuild}
                        disabled={!selectedSlate || poolLoading || (currentJob?.status === 'running')}
                    >
                        {currentJob?.status === 'running' ? 'Building...' : 'Generate Lineups'}
                    </button>

                    {/* Progress */}
                    {currentJob && (
                        <div className="build-status">
                            <div className="status-label">
                                {currentJob.status === 'running' && `Generating... ${currentJob.lineups_count}/${currentJob.target}`}
                                {currentJob.status === 'completed' && `✓ ${currentJob.lineups_count} lineups in ${currentJob.wall_time_sec?.toFixed(1)}s`}
                                {currentJob.status === 'failed' && `✗ ${currentJob.error}`}
                                {currentJob.status === 'pending' && 'Starting...'}
                            </div>
                            {currentJob.status === 'running' && (
                                <div className="progress-bar">
                                    <div
                                        className="progress-fill"
                                        style={{ width: `${Math.min(100, (currentJob.lineups_count / currentJob.target) * 100)}%` }}
                                    />
                                </div>
                            )}
                        </div>
                    )}

                    {buildError && <div className="error">{buildError}</div>}

                    {currentJob?.status === 'completed' && (
                        <button className="export-btn" onClick={handleExport}>
                            Export CSV
                        </button>
                    )}
                </aside>

                {/* Player Pool Table */}
                <section className="optimizer-pool">
                    <div className="pool-header">
                        <h3>Player Pool ({pool.length} players)</h3>
                        <input
                            type="text"
                            placeholder="Filter players..."
                            value={filter}
                            onChange={e => setFilter(e.target.value)}
                        />
                    </div>

                    {poolLoading ? (
                        <div className="loading">Loading player pool...</div>
                    ) : (
                        <div className="table-wrapper">
                            <table>
                                <thead>
                                    <tr>
                                        <th>Lock</th>
                                        <th>Ban</th>
                                        <th onClick={() => toggleSort('name')} className="sortable">
                                            Player {sortKey === 'name' && (sortDir === 'asc' ? '▲' : '▼')}
                                        </th>
                                        <th onClick={() => toggleSort('team')} className="sortable">
                                            Team {sortKey === 'team' && (sortDir === 'asc' ? '▲' : '▼')}
                                        </th>
                                        <th>Pos</th>
                                        <th onClick={() => toggleSort('salary')} className="sortable">
                                            Salary {sortKey === 'salary' && (sortDir === 'asc' ? '▲' : '▼')}
                                        </th>
                                        <th onClick={() => toggleSort('proj')} className="sortable">
                                            Proj {sortKey === 'proj' && (sortDir === 'asc' ? '▲' : '▼')}
                                        </th>
                                        <th onClick={() => toggleSort('own_proj')} className="sortable">
                                            Own% {sortKey === 'own_proj' && (sortDir === 'asc' ? '▲' : '▼')}
                                        </th>
                                        <th onClick={() => toggleSort('value')} className="sortable">
                                            Value {sortKey === 'value' && (sortDir === 'asc' ? '▲' : '▼')}
                                        </th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {filteredPool.map(p => {
                                        const isLocked = lockedIds.has(p.player_id)
                                        const isBanned = bannedIds.has(p.player_id)
                                        return (
                                            <tr
                                                key={p.player_id}
                                                className={isLocked ? 'player-locked' : isBanned ? 'player-banned' : ''}
                                            >
                                                <td>
                                                    <input
                                                        type="checkbox"
                                                        checked={isLocked}
                                                        onChange={() => toggleLock(p.player_id)}
                                                        title="Lock this player"
                                                    />
                                                </td>
                                                <td>
                                                    <input
                                                        type="checkbox"
                                                        checked={isBanned}
                                                        onChange={() => toggleBan(p.player_id)}
                                                        title="Ban this player"
                                                    />
                                                </td>
                                                <td><strong>{p.name}</strong></td>
                                                <td>{p.team}</td>
                                                <td>{p.positions.join('/')}</td>
                                                <td>{formatSalary(p.salary)}</td>
                                                <td>{formatProj(p.proj)}</td>
                                                <td>{formatOwn(p.own_proj)}</td>
                                                <td>{formatValue(p)}</td>
                                            </tr>
                                        )
                                    })}
                                    {filteredPool.length === 0 && !poolLoading && (
                                        <tr>
                                            <td colSpan={9} className="muted">No players found</td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    )}
                </section>
            </div>

            {/* Saved Builds Panel */}
            {savedBuilds.length > 0 && (
                <section className="saved-builds-section">
                    <div className="saved-builds-header">
                        <h3>Saved Builds ({savedBuilds.length})</h3>
                        <div className="saved-builds-header-actions">
                            {selectedBuildIds.size >= 2 && (
                                <button className="join-builds-btn" onClick={handleJoinBuilds}>
                                    Join {selectedBuildIds.size} Builds
                                </button>
                            )}
                            {savedBuildsLoading && <span className="muted">Loading...</span>}
                        </div>
                    </div>
                    <div className="saved-builds-list">
                        {savedBuilds.map(build => {
                            const isSelected = selectedBuildIds.has(build.job_id)
                            return (
                                <div key={build.job_id} className={`saved-build-card ${isSelected ? 'selected' : ''}`}>
                                    <input
                                        type="checkbox"
                                        className="saved-build-checkbox"
                                        checked={isSelected}
                                        onChange={() => toggleBuildSelection(build.job_id)}
                                        title="Select to join"
                                    />
                                    <div className="saved-build-info">
                                        <span className="saved-build-count">{build.lineups_count} lineups</span>
                                        <span className="saved-build-time">
                                            {new Date(build.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                        </span>
                                        {build.stats?.wall_time_s && (
                                            <span className="saved-build-stats">
                                                {(build.stats.wall_time_s as number).toFixed(1)}s
                                            </span>
                                        )}
                                    </div>
                                    <div className="saved-build-actions">
                                        <button
                                            className="export-btn-sm"
                                            onClick={() => handleExportBuild(build.job_id)}
                                            title="Export CSV"
                                        >
                                            ⬇
                                        </button>
                                        <button
                                            className="load-btn"
                                            onClick={() => handleLoadSavedBuild(build.job_id)}
                                        >
                                            Load
                                        </button>
                                        <button
                                            className="delete-btn"
                                            onClick={() => handleDeleteSavedBuild(build.job_id)}
                                        >
                                            ×
                                        </button>
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                </section>
            )}

            {/* Lineups Section - Below the main layout */}
            {lineups.length > 0 && (
                <section className="lineups-section">
                    {/* Filter taskbar */}
                    <div className="lineups-toolbar">
                        <h3>Lineups ({filteredLineups.length} of {lineups.length})</h3>
                        <div className="lineups-filters">
                            <input
                                type="text"
                                placeholder="Filter by player name..."
                                value={lineupFilter}
                                onChange={e => setLineupFilter(e.target.value)}
                            />
                            <input
                                type="number"
                                placeholder="Min proj"
                                value={minLineupProj ?? ''}
                                onChange={e => setMinLineupProj(e.target.value ? Number(e.target.value) : null)}
                                style={{ width: '80px' }}
                                title="Minimum projection total"
                            />
                            <input
                                type="number"
                                placeholder="Max own%"
                                value={maxLineupOwn ?? ''}
                                onChange={e => setMaxLineupOwn(e.target.value ? Number(e.target.value) : null)}
                                style={{ width: '80px' }}
                                title="Maximum ownership total"
                            />
                            <input
                                type="number"
                                placeholder="Min p90"
                                value={minLineupP90 ?? ''}
                                onChange={e => setMinLineupP90(e.target.value ? Number(e.target.value) : null)}
                                style={{ width: '80px' }}
                                title="Minimum p90 ceiling"
                            />
                            <select
                                value={showCount}
                                onChange={e => setShowCount(Number(e.target.value))}
                            >
                                <option value={25}>Show 25</option>
                                <option value={50}>Show 50</option>
                                <option value={100}>Show 100</option>
                                <option value={200}>Show 200</option>
                            </select>
                            <select
                                value={lineupSort}
                                onChange={e => setLineupSort(e.target.value as typeof lineupSort)}
                            >
                                <option value="default">Original Order</option>
                                <option value="proj-desc">Proj ↓</option>
                                <option value="proj-asc">Proj ↑</option>
                                <option value="p90-desc">p90 ↓</option>
                                <option value="p90-asc">p90 ↑</option>
                                <option value="own-desc">Own% ↓</option>
                                <option value="own-asc">Own% ↑</option>
                                <option value="salary-desc">Salary ↓</option>
                                <option value="salary-asc">Salary ↑</option>
                            </select>
                            {(lineupFilter || minLineupProj || maxLineupOwn || minLineupP90) && (
                                <button className="clear-filter" onClick={() => {
                                    setLineupFilter('')
                                    setMinLineupProj(null)
                                    setMaxLineupOwn(null)
                                    setMinLineupP90(null)
                                }}>
                                    Clear Filters
                                </button>
                            )}
                            <span className="lineups-selected-count">
                                {selectedLineupIds.size} selected
                            </span>
                            <button
                                className="lineups-action-btn"
                                onClick={selectAllVisible}
                                disabled={filteredLineups.length === 0}
                                title="Select currently shown lineups"
                            >
                                Select showing
                            </button>
                            <button
                                className="lineups-action-btn"
                                onClick={selectAllFiltered}
                                disabled={filteredLineups.length === 0}
                                title="Select all filtered lineups"
                            >
                                Select filtered
                            </button>
                            <button
                                className="lineups-action-btn"
                                onClick={clearSelection}
                                disabled={selectedLineupIds.size === 0}
                            >
                                Clear selection
                            </button>
                            <button
                                className="lineups-action-btn primary"
                                onClick={exportSelectedCSV}
                                disabled={selectedLineupIds.size === 0}
                                title="Export selected lineups"
                            >
                                Export selected CSV
                            </button>
                            <select
                                value={activeLineupGroupId}
                                onChange={e => setActiveLineupGroupId(e.target.value)}
                                title="Lineup groups"
                            >
                                <option value="">No group</option>
                                {lineupGroups.map(g => (
                                    <option key={g.id} value={g.id}>
                                        {g.name} ({g.lineup_ids.length})
                                    </option>
                                ))}
                            </select>
                            <button
                                className="lineups-action-btn"
                                onClick={createGroupFromSelection}
                                disabled={selectedLineupIds.size === 0}
                                title="Create a group from the current selection"
                            >
                                Save group
                            </button>
                            <button
                                className="lineups-action-btn"
                                onClick={selectActiveGroupLineups}
                                disabled={!activeLineupGroupId}
                                title="Load the active group into selection"
                            >
                                Select group
                            </button>
                            <button
                                className="lineups-action-btn primary"
                                onClick={exportActiveGroupCSV}
                                disabled={!activeLineupGroupId}
                                title="Export active group lineups"
                            >
                                Export group CSV
                            </button>
                            <button
                                className="lineups-action-btn danger"
                                onClick={deleteActiveGroup}
                                disabled={!activeLineupGroupId}
                                title="Delete active group"
                            >
                                Delete group
                            </button>
                        </div>
                    </div>

                    {/* Lineups grid */}
                    <div className="lineups-grid">
                        {filteredLineups.slice(0, showCount).map((lu, idx) => (
                            <div
                                key={lu.lineup_id}
                                className={`lineup-card ${selectedLineupIds.has(lu.lineup_id) ? 'selected' : ''}`}
                            >
                                <div className="lineup-header">
                                    <label className="lineup-select">
                                        <input
                                            type="checkbox"
                                            checked={selectedLineupIds.has(lu.lineup_id)}
                                            onChange={() => toggleLineupSelection(lu.lineup_id)}
                                            title="Select lineup"
                                        />
                                    </label>
                                    <span>#{idx + 1}</span>
                                    <span className="lineup-salary">
                                        ${lu.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.salary ?? 0), 0).toLocaleString()}
                                    </span>
                                    <span className="lineup-proj">
                                        {lu.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.proj ?? 0), 0).toFixed(1)} pts
                                    </span>
                                    <span className="lineup-p90">
                                        p90: {(() => {
                                            const p90 = lu.p90 ?? lu.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.p90 ?? 0), 0)
                                            return p90 > 0 ? p90.toFixed(1) : '—'
                                        })()}
                                    </span>
                                    <span className="lineup-ownership">
                                        {lu.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.own_proj ?? 0), 0).toFixed(1)}% own
                                    </span>
                                </div>
                                <div className="lineup-players">
                                    {lu.player_ids.map(id => {
                                        const p = playerMap.get(id)
                                        const isFiltered = lineupFilter && p && p.name.toLowerCase().includes(lineupFilter.toLowerCase())
                                        return (
                                            <span key={id} className={`lineup-player ${isFiltered ? 'highlight' : ''}`}>
                                                {p ? `${p.name} (${p.positions[0]})` : id}
                                            </span>
                                        )
                                    })}
                                </div>
                            </div>
                        ))}
                    </div>
                    {filteredLineups.length > showCount && (
                        <div className="lineups-footer">
                            <button onClick={() => setShowCount(prev => prev + 50)}>
                                Load More ({filteredLineups.length - showCount} remaining)
                            </button>
                        </div>
                    )}
                </section>
            )}
        </div>
    )
}
