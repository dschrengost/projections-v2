import { useEffect, useMemo, useState } from 'react'
import {
    applyBuildToEntries,
    exportEntryFile,
    exportEntriesBatch,
    getEntryFile,
    listEntryFiles,
    runLateSwap,
    selectLateSwapAlternative,
    uploadEntries,
    EntryFileState,
    EntryFileSummary,
    EntryAlternatives,
} from '../api/entry_manager'
import { getSavedBuilds, getSlates, loadSavedBuild, SavedBuild, Slate } from '../api/optimizer'
import { getSavedSimBuilds, loadSavedSimBuild, SavedSimBuildSummary } from '../api/contest_sim'

const todayISO = () => new Date().toISOString().slice(0, 10)
const DK_SLOTS = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
const ENTRIES_PER_PAGE = 25

interface ContestSwapResult {
    contestId: string
    contestName: string
    entryCount: number
    lockedCount: number
    swappedCount: number
    totalProjDelta: number
    missingLockedCount: number
    heldCount: number
    avgGap: number | null
    maxGap: number | null
    statusSummary: string | null
    unmappedEntries: number
}

const extractPlayerName = (playerStr: string): string => {
    // "Player Name (12345)" -> "Player Name"
    const match = playerStr.match(/^(.+?)\s*\(\d+\)$/)
    return match ? match[1] : playerStr
}

export default function EntryManagerPage() {
    const [selectedDate, setSelectedDate] = useState(todayISO())
    const [slates, setSlates] = useState<Slate[]>([])
    const [selectedSlate, setSelectedSlate] = useState<number | null>(null)
    const [slatesLoading, setSlatesLoading] = useState(false)

    const [entryFiles, setEntryFiles] = useState<EntryFileSummary[]>([])
    const [entriesLoading, setEntriesLoading] = useState(false)
    const [selectedContestId, setSelectedContestId] = useState<string | null>(null)
    const [selectedContestIds, setSelectedContestIds] = useState<Set<string>>(new Set())
    const [contestOrder, setContestOrder] = useState<string[]>([])
    const [entryFile, setEntryFile] = useState<EntryFileState | null>(null)
    const [entryError, setEntryError] = useState<string | null>(null)
    const [entryDiffs, setEntryDiffs] = useState<Record<string, string[]>>({})
    const [lockedSlotsByEntryId, setLockedSlotsByEntryId] = useState<Record<string, string[]>>({})
    const [entryPage, setEntryPage] = useState(1)

    const [savedBuilds, setSavedBuilds] = useState<SavedBuild[]>([])
    const [savedSimBuilds, setSavedSimBuilds] = useState<SavedSimBuildSummary[]>([])
    const [buildSource, setBuildSource] = useState<'optimizer' | 'contest-sim'>('optimizer')
    const [selectedBuildId, setSelectedBuildId] = useState<string | null>(null)

    const [uploading, setUploading] = useState(false)
    const [applyLoading, setApplyLoading] = useState(false)
    const [lateSwapLoading, setLateSwapLoading] = useState(false)

    // Late swap alternatives state
    const [alternativesByEntryId, setAlternativesByEntryId] = useState<
        Record<string, EntryAlternatives>
    >({})
    const [selectedAlternatives, setSelectedAlternatives] = useState<
        Record<string, number>
    >({})
    const [contestResults, setContestResults] = useState<ContestSwapResult[]>([])
    const [showResultsPanel, setShowResultsPanel] = useState(false)
    const [randomnessPct, setRandomnessPct] = useState(0)

    useEffect(() => {
        const loadSlates = async () => {
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
        void loadSlates()
    }, [selectedDate])

    useEffect(() => {
        const loadEntries = async () => {
            setEntriesLoading(true)
            setEntryError(null)
            try {
                const data = await listEntryFiles(selectedDate)
                setEntryFiles(data)
                if (data.length && !selectedContestId) {
                    setSelectedContestId(data[0].contest_id)
                }
                setContestOrder(data.map(entry => entry.contest_id))
            } catch (err) {
                setEntryError((err as Error).message)
                setEntryFiles([])
            } finally {
                setEntriesLoading(false)
            }
        }
        void loadEntries()
    }, [selectedDate])

    useEffect(() => {
        if (!selectedContestId) {
            setEntryFile(null)
            return
        }
        setEntryDiffs({})
        setLockedSlotsByEntryId({})
        setEntryPage(1)
        const load = async () => {
            try {
                const data = await getEntryFile(selectedDate, selectedContestId)
                setEntryFile(data)
            } catch (err) {
                setEntryError((err as Error).message)
                setEntryFile(null)
            }
        }
        void load()
    }, [selectedDate, selectedContestId])

    useEffect(() => {
        if (!selectedSlate) {
            setSavedBuilds([])
            return
        }
        const load = async () => {
            try {
                const builds = await getSavedBuilds(selectedDate, selectedSlate)
                setSavedBuilds(builds)
                setSelectedBuildId(builds[0]?.job_id ?? null)
            } catch {
                setSavedBuilds([])
            }
        }
        void load()
    }, [selectedDate, selectedSlate])

    useEffect(() => {
        const load = async () => {
            try {
                const builds = await getSavedSimBuilds(selectedDate, 'lineups')
                setSavedSimBuilds(builds)
            } catch {
                setSavedSimBuilds([])
            }
        }
        void load()
    }, [selectedDate, selectedSlate])

    useEffect(() => {
        if (buildSource === 'optimizer') {
            setSelectedBuildId(savedBuilds[0]?.job_id ?? null)
        } else {
            setSelectedBuildId(savedSimBuilds[0]?.build_id ?? null)
        }
    }, [buildSource, savedBuilds, savedSimBuilds])

    const handleUpload = async (file: File) => {
        if (!selectedSlate) return
        setUploading(true)
        setEntryError(null)
        try {
            const summaries = await uploadEntries(selectedDate, selectedSlate, file)
            setEntryFiles(summaries)
            setSelectedContestId(summaries[0]?.contest_id ?? null)
        } catch (err) {
            setEntryError((err as Error).message)
        } finally {
            setUploading(false)
        }
    }

    const handleApplyBuild = async () => {
        if (!selectedBuildId) return
        const targetIds = selectedContestIds.size > 0
            ? contestOrder.filter(id => selectedContestIds.has(id))
            : selectedContestId
                ? [selectedContestId]
                : []
        if (targetIds.length === 0) return
        setApplyLoading(true)
        setEntryError(null)
        try {
            let lineups: string[][] = []
            if (buildSource === 'optimizer') {
                const build = await loadSavedBuild(selectedDate, selectedBuildId)
                lineups = build.lineups?.map(lu => lu.player_ids) ?? []
            } else {
                const build = await loadSavedSimBuild(selectedDate, selectedBuildId)
                lineups = build.lineups ?? []
            }
            let offset = 0
            for (const contestId of targetIds) {
                const entrySummary = entryFiles.find(entry => entry.contest_id === contestId)
                const entryCount = entrySummary?.entry_count ?? 0
                if (entryCount <= 0) continue
                const slice = lineups.slice(offset, offset + entryCount)
                if (slice.length < entryCount) {
                    throw new Error('Not enough lineups to cover selected contests')
                }
                const updated = await applyBuildToEntries(
                    selectedDate,
                    contestId,
                    buildSource,
                    selectedBuildId,
                    slice,
                )
                if (contestId === selectedContestId) {
                    setEntryFile(updated)
                }
                offset += entryCount
            }
        } catch (err) {
            setEntryError((err as Error).message)
        } finally {
            setApplyLoading(false)
        }
    }

    const handleExport = async () => {
        if (!selectedContestId) return
        try {
            const blob = await exportEntryFile(selectedDate, selectedContestId)
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = `entries_${selectedDate}_${selectedContestId}.csv`
            a.click()
            URL.revokeObjectURL(url)
        } catch (err) {
            alert('Export failed: ' + (err as Error).message)
        }
    }

    const handleExportSelected = async () => {
        const targetIds = selectedContestIds.size > 0
            ? contestOrder.filter(id => selectedContestIds.has(id))
            : entryFiles.map(entry => entry.contest_id)
        if (targetIds.length === 0) return
        try {
            const blob = await exportEntriesBatch(selectedDate, targetIds)
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = `entries_${selectedDate}_combined.csv`
            a.click()
            URL.revokeObjectURL(url)
        } catch (err) {
            alert('Export failed: ' + (err as Error).message)
        }
    }

    const computeEntryDiffs = (
        prevEntries: Record<string, string>[],
        nextEntries: Record<string, string>[],
    ): Record<string, string[]> => {
        const prevMap = new Map(
            prevEntries.map((entry, idx) => [
                String(entry.entry_key || entry.entry_id || `row-${idx + 1}`),
                entry,
            ]),
        )
        const diffs: Record<string, string[]> = {}
        for (const [idx, entry] of nextEntries.entries()) {
            const entryId = String(entry.entry_key || entry.entry_id || `row-${idx + 1}`)
            const prev = prevMap.get(entryId)
            if (!prev) continue
            const changedSlots = DK_SLOTS.filter(slot => (prev[slot] || '') !== (entry[slot] || ''))
            if (changedSlots.length) {
                diffs[entryId] = changedSlots
            }
        }
        return diffs
    }

    const handleLateSwap = async () => {
        const selectedIds = Array.from(selectedContestIds)
        const orderedSelectedIds = selectedIds.length
            ? contestOrder.length
                ? contestOrder.filter(id => selectedContestIds.has(id))
                : selectedIds
            : []
        const targetIds = orderedSelectedIds.length > 0
            ? orderedSelectedIds
            : selectedContestId
                ? [selectedContestId]
                : []
        if (targetIds.length === 0) return
        setLateSwapLoading(true)
        setEntryError(null)

        const results: ContestSwapResult[] = []

        try {
            const prevEntries = selectedContestId && entryFile && entryFile.contest_id === selectedContestId
                ? entryFile.entries
                : null

            for (const contestId of targetIds) {
                const contestSummary = entryFiles.find(e => e.contest_id === contestId)
                const result = await runLateSwap(selectedDate, contestId, {
                    randomnessPct: randomnessPct > 0 ? randomnessPct : undefined,
                })

                // Calculate stats for results panel
                let swappedCount = 0
                let totalProjDelta = 0

                if (result.alternatives_by_entry_id) {
                    for (const alts of Object.values(result.alternatives_by_entry_id)) {
                        const selectedAlt = alts.alternatives[alts.selected_idx]
                        if (selectedAlt?.player_swaps?.length > 0) {
                            swappedCount++
                            for (const swap of selectedAlt.player_swaps) {
                                if (swap.old_proj !== null && swap.new_proj !== null) {
                                    totalProjDelta += swap.new_proj - swap.old_proj
                                }
                            }
                        }
                    }

                    setAlternativesByEntryId(prev => ({
                        ...prev,
                        ...result.alternatives_by_entry_id,
                    }))

                    const newSelected: Record<string, number> = {}
                    for (const [entryId, alts] of Object.entries(result.alternatives_by_entry_id)) {
                        newSelected[entryId] = alts.selected_idx
                    }
                    setSelectedAlternatives(prev => ({ ...prev, ...newSelected }))
                }

                results.push({
                    contestId,
                    contestName: contestSummary?.contest_name || contestId,
                    entryCount: result.updated_entries,
                    lockedCount: result.locked_count,
                    swappedCount,
                    totalProjDelta,
                    missingLockedCount: result.missing_locked_ids?.length ?? 0,
                    heldCount: result.selection_summary?.entries_held ?? 0,
                    avgGap: result.solver_summary?.avg_gap ?? null,
                    maxGap: result.solver_summary?.max_gap ?? null,
                    statusSummary: result.solver_summary?.status_counts
                        ? Object.entries(result.solver_summary.status_counts)
                            .map(([status, count]) => `${status}: ${count}`)
                            .join(', ')
                        : null,
                    unmappedEntries: result.selection_summary?.entries_unmapped ?? 0,
                })

                if (contestId === selectedContestId) {
                    setEntryFile(result.entry_state)
                    if (prevEntries) {
                        const diffs = computeEntryDiffs(prevEntries, result.entry_state.entries)
                        setEntryDiffs(diffs)
                        setEntryPage(1)
                    }
                    setLockedSlotsByEntryId(result.locked_slots_by_entry_id || {})
                }
            }

            setContestResults(results)
            if (results.length > 0) {
                setShowResultsPanel(true)
            }
        } catch (err) {
            setEntryError((err as Error).message)
        } finally {
            setLateSwapLoading(false)
        }
    }

    const currentEntriesCount = entryFile?.entries.length ?? 0
    const entryPages = Math.max(1, Math.ceil(currentEntriesCount / ENTRIES_PER_PAGE))
    const entryPageIndex = Math.min(entryPage, entryPages)
    const entryPageStart = (entryPageIndex - 1) * ENTRIES_PER_PAGE
    const entryPageEntries = entryFile?.entries.slice(
        entryPageStart,
        entryPageStart + ENTRIES_PER_PAGE,
    ) ?? []

    const buildOptions = useMemo(() => {
        return buildSource === 'optimizer'
            ? savedBuilds.map(b => ({ id: b.job_id, label: `${b.job_id.slice(0, 8)} (${b.lineups_count})` }))
            : savedSimBuilds.map(b => ({ id: b.build_id, label: `${b.name ?? b.build_id.slice(0, 8)} (${b.lineups_count})` }))
    }, [buildSource, savedBuilds, savedSimBuilds])

    const orderedEntries = useMemo(() => {
        const map = new Map(entryFiles.map(entry => [entry.contest_id, entry]))
        return contestOrder.map(id => map.get(id)).filter(Boolean) as EntryFileSummary[]
    }, [contestOrder, entryFiles])

    const toggleContestSelection = (contestId: string) => {
        setSelectedContestIds(prev => {
            const next = new Set(prev)
            if (next.has(contestId)) {
                next.delete(contestId)
            } else {
                next.add(contestId)
            }
            return next
        })
    }

    const selectAllContests = () => {
        setSelectedContestIds(new Set(contestOrder))
    }

    const clearAllContests = () => {
        setSelectedContestIds(new Set())
    }

    const moveContest = (contestId: string, direction: 'up' | 'down') => {
        setContestOrder(prev => {
            const idx = prev.indexOf(contestId)
            if (idx === -1) return prev
            const next = [...prev]
            const target = direction === 'up' ? idx - 1 : idx + 1
            if (target < 0 || target >= next.length) return prev
            const [item] = next.splice(idx, 1)
            next.splice(target, 0, item)
            return next
        })
    }

    return (
        <div className="optimizer-page">
            <header className="app-header">
                <div>
                    <h1>Entry Manager</h1>
                    <p className="subtitle">Upload DK entries, apply builds, export CSV</p>
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
                                    {s.slate_type} ({s.draft_group_id})
                                </option>
                            ))}
                        </select>
                    </label>
                </div>
            </header>

            {entryError && (
                <section className="status-bar">
                    <span className="error">{entryError}</span>
                </section>
            )}

            {/* Late Swap Results Panel */}
            {showResultsPanel && contestResults.length > 0 && (
                <section className="late-swap-results-panel">
                    <div className="results-header">
                        <h4>Late Swap Results</h4>
                        <button onClick={() => setShowResultsPanel(false)} className="close-btn">×</button>
                    </div>
                    <div className="results-list">
                        {contestResults.map(result => (
                            <div key={result.contestId} className="result-item">
                                <div className="result-contest">{result.contestName}</div>
                                <div className="result-stats">
                                    <span>{result.entryCount} entries</span>
                                    <span className="muted">|</span>
                                    <span>{result.lockedCount} locked</span>
                                    <span className="muted">|</span>
                                    <span>{result.swappedCount} swapped</span>
                                    <span className="muted">|</span>
                                    <span>{result.heldCount} held</span>
                                    <span className="muted">|</span>
                                    <span className={result.totalProjDelta >= 0 ? 'positive' : 'negative'}>
                                        {result.totalProjDelta >= 0 ? '+' : ''}{result.totalProjDelta.toFixed(1)} proj
                                    </span>
                                    {result.avgGap !== null && (
                                        <>
                                            <span className="muted">|</span>
                                            <span className="muted">
                                                gap {result.avgGap.toFixed(4)} avg
                                                {result.maxGap !== null ? ` / ${result.maxGap.toFixed(4)} max` : ''}
                                            </span>
                                        </>
                                    )}
                                    {result.missingLockedCount > 0 && (
                                        <>
                                            <span className="muted">|</span>
                                            <span className="negative">{result.missingLockedCount} unmapped locks</span>
                                        </>
                                    )}
                                    {result.unmappedEntries > 0 && (
                                        <>
                                            <span className="muted">|</span>
                                            <span className="negative">{result.unmappedEntries} entries skipped</span>
                                        </>
                                    )}
                                    {result.statusSummary && (
                                        <>
                                            <span className="muted">|</span>
                                            <span className="muted">{result.statusSummary}</span>
                                        </>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </section>
            )}

            <div className="optimizer-layout">
                <aside className="optimizer-sidebar">
                    <h3>Upload Entry CSV</h3>
                    <input
                        type="file"
                        accept=".csv"
                        disabled={uploading}
                        onChange={e => {
                            const file = e.target.files?.[0]
                            if (file) {
                                void handleUpload(file)
                                e.currentTarget.value = ''
                            }
                        }}
                    />
                    {uploading && <div className="muted">Uploading...</div>}

                    <hr />

                    <h3>Apply Build</h3>
                    <label>
                        Source
                        <select
                            value={buildSource}
                            onChange={e => setBuildSource(e.target.value as 'optimizer' | 'contest-sim')}
                        >
                            <option value="optimizer">Optimizer Builds</option>
                            <option value="contest-sim">Sim Lineups</option>
                        </select>
                    </label>
                    <label>
                        Build
                        <select
                            value={selectedBuildId ?? ''}
                            onChange={e => setSelectedBuildId(e.target.value || null)}
                        >
                            {buildOptions.length === 0 && <option value="">No builds</option>}
                            {buildOptions.map(b => (
                                <option key={b.id} value={b.id}>
                                    {b.label}
                                </option>
                            ))}
                        </select>
                    </label>
                    <button
                        className="build-btn"
                        onClick={handleApplyBuild}
                        disabled={!selectedBuildId || applyLoading}
                    >
                        {applyLoading ? 'Applying...' : 'Apply Build'}
                    </button>
                    <button
                        className="build-btn"
                        onClick={handleLateSwap}
                        disabled={lateSwapLoading || entryFiles.length === 0}
                    >
                        {lateSwapLoading ? 'Late Swapping...' : 'Late Swap (Now)'}
                    </button>
                    <div className="muted">Uses server time to lock started games.</div>

                    {/* Randomness Control */}
                    <div className="randomness-control">
                        <label>
                            Randomness: {randomnessPct}%
                            <input
                                type="range"
                                min={0}
                                max={100}
                                step={5}
                                value={randomnessPct}
                                onChange={e => setRandomnessPct(Number(e.target.value))}
                            />
                        </label>
                        {randomnessPct > 0 && (
                            <small className="hint">Adds variance-aware noise for lineup diversity</small>
                        )}
                    </div>
                    <button
                        className="export-btn"
                        onClick={handleExport}
                        disabled={!selectedContestId || currentEntriesCount === 0}
                    >
                        Export CSV
                    </button>
                    <button
                        className="export-btn"
                        onClick={handleExportSelected}
                        disabled={entryFiles.length === 0}
                    >
                        Export Selected
                    </button>
                </aside>

                <section className="optimizer-pool">
                    <div className="pool-header">
                        <h3>Entry Files ({entryFiles.length})</h3>
                        <div className="saved-build-actions">
                            <button className="load-btn" onClick={selectAllContests}>
                                Select All
                            </button>
                            <button className="delete-btn" onClick={clearAllContests}>
                                Clear
                            </button>
                            {entriesLoading && <span className="muted">Loading...</span>}
                        </div>
                    </div>
                    <div className="saved-builds-list">
                        {entryFiles.length === 0 && !entriesLoading && (
                            <span className="muted">No entry files yet.</span>
                        )}
                        {orderedEntries.map(entry => (
                            <div
                                key={entry.contest_id}
                                className={`saved-build-card ${selectedContestId === entry.contest_id ? 'selected' : ''}`}
                            >
                                <div className="saved-build-info">
                                    <input
                                        type="checkbox"
                                        checked={selectedContestIds.has(entry.contest_id)}
                                        onChange={() => toggleContestSelection(entry.contest_id)}
                                        title="Select contest"
                                    />
                                    <span className="saved-build-count">{entry.entry_count} entries</span>
                                    <span className="saved-build-time">{entry.contest_name}</span>
                                </div>
                                <div className="saved-build-actions">
                                    <button className="load-btn" onClick={() => setSelectedContestId(entry.contest_id)}>
                                        Load
                                    </button>
                                    <button
                                        className="delete-btn"
                                        onClick={() => moveContest(entry.contest_id, 'up')}
                                        title="Move up"
                                    >
                                        ↑
                                    </button>
                                    <button
                                        className="delete-btn"
                                        onClick={() => moveContest(entry.contest_id, 'down')}
                                        title="Move down"
                                    >
                                        ↓
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>

                    {entryFile && (
                        <div className="lineups-section">
                            <div className="lineups-toolbar">
                                <h3>{entryFile.contest_name}</h3>
                                <div className="lineups-filters">
                                    <span className="muted">
                                        {entryFile.entries.length} entries · Draft group {entryFile.draft_group_id}
                                    </span>
                                    <button
                                        className="lineups-action-btn"
                                        onClick={() => setEntryPage(p => Math.max(1, p - 1))}
                                        disabled={entryPageIndex <= 1}
                                    >
                                        Prev
                                    </button>
                                    <button
                                        className="lineups-action-btn"
                                        onClick={() => setEntryPage(p => Math.min(entryPages, p + 1))}
                                        disabled={entryPageIndex >= entryPages}
                                    >
                                        Next
                                    </button>
                                    <span className="muted">Page {entryPageIndex} / {entryPages}</span>
                                </div>
                            </div>
                            <div className="lineups-grid">
                                {entryPageEntries.map((entry, idx) => {
                                    const entryId = String(
                                        entry.entry_key
                                        || entry.entry_id
                                        || `row-${entryPageStart + idx + 1}`,
                                    )
                                    const changedSlots = new Set(entryDiffs[entryId] ?? [])
                                    const lockedSlots = new Set(lockedSlotsByEntryId[entryId] ?? [])
                                    const alternatives = alternativesByEntryId[entryId]
                                    const selectedAltIdx = selectedAlternatives[entryId] ?? 0

                                    const handleAlternativeChange = async (newIdx: number) => {
                                        if (!alternatives || !entryFile) return
                                        const alt = alternatives.alternatives[newIdx]
                                        if (!alt) return

                                        setSelectedAlternatives(prev => ({
                                            ...prev,
                                            [entryId]: newIdx,
                                        }))

                                        try {
                                            const updated = await selectLateSwapAlternative(
                                                selectedDate,
                                                entryFile.contest_id,
                                                entryId,
                                                newIdx,
                                                alt.slot_values,
                                            )
                                            setEntryFile(updated)
                                        } catch (err) {
                                            setEntryError('Failed to apply alternative: ' + (err as Error).message)
                                        }
                                    }

                                    const currentAlt = alternatives?.alternatives[selectedAltIdx]

                                    return (
                                        <div key={entryId} className="lineup-card">
                                            <div className="lineup-header">
                                                <span>Entry {entryId}</span>
                                                {alternatives && alternatives.alternatives.length > 1 && (
                                                    <select
                                                        value={selectedAltIdx}
                                                        onChange={(e) => handleAlternativeChange(Number(e.target.value))}
                                                        className="alternative-select"
                                                    >
                                                        {alternatives.alternatives.map((alt, i) => (
                                                            <option key={i} value={i}>
                                                                Alt {i + 1}: {alt.projected_score.toFixed(1)} pts
                                                            </option>
                                                        ))}
                                                    </select>
                                                )}
                                                <span className="lineup-salary">{entry.entry_fee || ''}</span>
                                            </div>
                                            <div className="lineup-players">
                                                {DK_SLOTS.map(slot => {
                                                    const val = entry[slot] || '-'
                                                    const highlight = changedSlots.has(slot)
                                                    const locked = lockedSlots.has(slot)
                                                    return (
                                                        <div
                                                            key={`${entryId}-${slot}`}
                                                            className={`lineup-player ${highlight ? 'highlight' : ''}`}
                                                        >
                                                            <span className="lineup-slot-label">{slot}</span>
                                                            {locked && <span className="lineup-lock" title="Locked">🔒</span>}
                                                            {val}
                                                        </div>
                                                    )
                                                })}
                                            </div>
                                            {/* Player swaps display */}
                                            {currentAlt?.player_swaps && currentAlt.player_swaps.length > 0 && (
                                                <div className="player-swaps">
                                                    {currentAlt.player_swaps.map((swap, i) => (
                                                        <div key={i} className="swap-item">
                                                            <span className="swap-slot">{swap.slot}:</span>
                                                            <span className="swap-old">{extractPlayerName(swap.old_player)}</span>
                                                            <span className="swap-arrow"> → </span>
                                                            <span className="swap-new">{extractPlayerName(swap.new_player)}</span>
                                                            {swap.old_proj !== null && swap.new_proj !== null && (
                                                                <span className={`swap-delta ${swap.new_proj > swap.old_proj ? 'positive' : 'negative'}`}>
                                                                    ({swap.new_proj > swap.old_proj ? '+' : ''}{(swap.new_proj - swap.old_proj).toFixed(1)})
                                                                </span>
                                                            )}
                                                        </div>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    )
                                })}
                            </div>
                        </div>
                    )}
                </section>
            </div>
        </div>
    )
}
