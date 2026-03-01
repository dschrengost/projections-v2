import { useEffect, useMemo, useState } from 'react'
import {
    applyBuildToEntries,
    deleteEntryFile,
    exportEntryFile,
    exportEntriesBatch,
    getEntryFile,
    listEntryFiles,
    repairEntryFileDraftGroups,
    runLateSwap,
    selectLateSwapAlternative,
    uploadEntries,
    EntryFileState,
    EntryFileSummary,
    EntryAlternatives,
} from '../api/entry_manager'
import { getSavedBuilds, getSlates, loadSavedBuild, SavedBuild, Slate } from '../api/optimizer'
import { getSavedSimBuilds, loadSavedSimBuild, SavedSimBuildSummary } from '../api/contest_sim'
import { useSlateDateAndSlate } from '../hooks/useSlateDate'
import { formatSlateLabel } from '../utils/slateFormat'
import { computeRealSwaps, LineupSlots } from '../utils/lateSwapUtils'

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

interface ApplyBuildResult {
    contestId: string
    contestName: string
    lineupsApplied: number
    lineupRangeStart: number
    lineupRangeEnd: number
}

interface ApplyBuildTargetInfo {
    contestCount: number
    totalEntries: number
    targetIds: string[]
    applyingAllByDefault: boolean
}

interface LineupState {
    baselineLineup: LineupSlots
    currentLineup: LineupSlots
    isSelected: boolean
}

const extractPlayerName = (playerStr: string): string => {
    // "Player Name (12345)" -> "Player Name"
    const match = playerStr.match(/^(.+?)\s*\(\d+\)$/)
    return match ? match[1] : playerStr
}

const getEntryKey = (entry: Record<string, string>, idx: number): string => {
    return String(entry.entry_key || entry.entry_id || `row-${idx + 1}`)
}

const buildLineupSlots = (entry: Record<string, string>): LineupSlots => {
    const slots: LineupSlots = {}
    for (const slot of DK_SLOTS) {
        slots[slot] = entry[slot] || ''
    }
    return slots
}

const buildLineupStates = (
    baselineEntries: Record<string, string>[],
    currentEntries: Record<string, string>[],
    isSelected = true,
): Record<string, LineupState> => {
    const baselineById = new Map<string, LineupSlots>()
    baselineEntries.forEach((entry, idx) => {
        baselineById.set(getEntryKey(entry, idx), buildLineupSlots(entry))
    })

    const next: Record<string, LineupState> = {}
    currentEntries.forEach((entry, idx) => {
        const entryId = getEntryKey(entry, idx)
        const currentLineup = buildLineupSlots(entry)
        next[entryId] = {
            baselineLineup: baselineById.get(entryId) ?? currentLineup,
            currentLineup,
            isSelected,
        }
    })
    return next
}

export default function EntryManagerPage() {
    const [selectedDate, setSelectedDate, selectedSlate, setSelectedSlate] = useSlateDateAndSlate()
    const [slates, setSlates] = useState<Slate[]>([])
    const [slatesLoading, setSlatesLoading] = useState(false)

    const [entryFiles, setEntryFiles] = useState<EntryFileSummary[]>([])
    const [entriesLoading, setEntriesLoading] = useState(false)
    const [selectedContestId, setSelectedContestId] = useState<string | null>(null)
    const [selectedContestIds, setSelectedContestIds] = useState<Set<string>>(new Set())
    const [contestOrder, setContestOrder] = useState<string[]>([])
    const [entryFile, setEntryFile] = useState<EntryFileState | null>(null)
    const [entryError, setEntryError] = useState<string | null>(null)
    const [lockedSlotsByEntryId, setLockedSlotsByEntryId] = useState<Record<string, string[]>>({})
    const [entryPage, setEntryPage] = useState(1)
    const [lineupStates, setLineupStates] = useState<Record<string, LineupState>>({})
    const [lineupStateKey, setLineupStateKey] = useState<string | null>(null)
    const [lineupBaselineLocked, setLineupBaselineLocked] = useState(false)

    const [savedBuilds, setSavedBuilds] = useState<SavedBuild[]>([])
    const [savedSimBuilds, setSavedSimBuilds] = useState<SavedSimBuildSummary[]>([])
    const [buildSource, setBuildSource] = useState<'optimizer' | 'contest-sim'>('optimizer')
    const [selectedBuildId, setSelectedBuildId] = useState<string | null>(null)

    const [uploading, setUploading] = useState(false)
    const [repairing, setRepairing] = useState(false)
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
    const [onlyOutLineups, setOnlyOutLineups] = useState(false)
    const [applyBuildResults, setApplyBuildResults] = useState<ApplyBuildResult[]>([])
    const [showApplyResultsPanel, setShowApplyResultsPanel] = useState(false)

    useEffect(() => {
        const loadSlates = async () => {
            setSlatesLoading(true)
            try {
                const data = await getSlates(selectedDate)
                setSlates(data)
                // If URL has a slate and it exists in the data, keep it; otherwise auto-select
                const urlSlateExists = selectedSlate && data.some(s => s.draft_group_id === selectedSlate)
                if (!urlSlateExists) {
                    const mainSlate = data.find(s => s.slate_type !== 'showdown')
                    setSelectedSlate(mainSlate?.draft_group_id ?? data[0]?.draft_group_id ?? null)
                }
            } catch {
                setSlates([])
                setSelectedSlate(null)
            } finally {
                setSlatesLoading(false)
            }
        }
        void loadSlates()
    }, [selectedDate]) // eslint-disable-line react-hooks/exhaustive-deps

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
        setLockedSlotsByEntryId({})
        setEntryPage(1)
        const load = async () => {
            try {
                const data = await getEntryFile(selectedDate, selectedContestId)
                setEntryFile(data)
                if (data.draft_group_id && data.draft_group_id !== selectedSlate) {
                    setSelectedSlate(data.draft_group_id)
                }
            } catch (err) {
                setEntryError((err as Error).message)
                setEntryFile(null)
            }
        }
        void load()
    }, [selectedDate, selectedContestId])

    useEffect(() => {
        if (!entryFile) {
            setLineupStates({})
            setLineupStateKey(null)
            setLineupBaselineLocked(false)
            return
        }
        const contestKey = `${selectedDate}:${entryFile.contest_id}`
        if (contestKey !== lineupStateKey) {
            setLineupStates(buildLineupStates(entryFile.entries, entryFile.entries))
            setLineupStateKey(contestKey)
            setLineupBaselineLocked(false)
            return
        }
        setLineupStates(prev => {
            if (Object.keys(prev).length === 0) {
                return buildLineupStates(entryFile.entries, entryFile.entries)
            }
            const next = { ...prev }
            entryFile.entries.forEach((entry, idx) => {
                const entryId = getEntryKey(entry, idx)
                const currentLineup = buildLineupSlots(entry)
                const existing = next[entryId]
                if (existing) {
                    next[entryId] = {
                        baselineLineup: lineupBaselineLocked ? existing.baselineLineup : currentLineup,
                        isSelected: existing.isSelected,
                        currentLineup,
                    }
                } else {
                    next[entryId] = {
                        baselineLineup: currentLineup,
                        currentLineup,
                        isSelected: true,
                    }
                }
            })
            return next
        })
    }, [entryFile, lineupBaselineLocked, lineupStateKey, selectedDate])

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
        const targetDraftGroupId = entryFile?.draft_group_id ?? selectedSlate ?? null
        if (buildSource === 'optimizer') {
            const match = targetDraftGroupId
                ? savedBuilds.find(b => b.draft_group_id === targetDraftGroupId)
                : savedBuilds[0]
            setSelectedBuildId(match?.job_id ?? savedBuilds[0]?.job_id ?? null)
        } else {
            const match = targetDraftGroupId
                ? savedSimBuilds.find(b => b.draft_group_id === targetDraftGroupId)
                : savedSimBuilds[0]
            setSelectedBuildId(match?.build_id ?? savedSimBuilds[0]?.build_id ?? null)
        }
    }, [buildSource, entryFile?.draft_group_id, savedBuilds, savedSimBuilds, selectedSlate])

    const slateOptions = useMemo(() => {
        const opts = [...slates]
        if (
            selectedSlate !== null
            && selectedSlate !== undefined
            && !opts.some(s => s.draft_group_id === selectedSlate)
        ) {
            opts.unshift({
                game_date: selectedDate,
                slate_type: 'selected',
                draft_group_id: selectedSlate,
                n_contests: 0,
            })
        }
        return opts
    }, [selectedDate, selectedSlate, slates])

    const handleUpload = async (file: File) => {
        setUploading(true)
        setEntryError(null)
        try {
            const summaries = await uploadEntries(selectedDate, selectedSlate, file)
            setEntryFiles(summaries)
            setContestOrder(summaries.map(s => s.contest_id))
            setSelectedContestId(summaries[0]?.contest_id ?? null)
            if (summaries.length > 0) {
                setSelectedSlate(summaries[0].draft_group_id)
            }
        } catch (err) {
            setEntryError((err as Error).message)
        } finally {
            setUploading(false)
        }
    }

    const handleRepairDraftGroups = async () => {
        setRepairing(true)
        setEntryError(null)
        try {
            const summaries = await repairEntryFileDraftGroups(selectedDate)
            setEntryFiles(summaries)
            setContestOrder(summaries.map(s => s.contest_id))
            if (selectedContestId && !summaries.some(s => s.contest_id === selectedContestId)) {
                setSelectedContestId(summaries[0]?.contest_id ?? null)
            }
        } catch (err) {
            setEntryError((err as Error).message)
        } finally {
            setRepairing(false)
        }
    }

    const handleDeleteContest = async (contestId: string) => {
        const confirmDelete = window.confirm('Delete this entry file? This cannot be undone.')
        if (!confirmDelete) return
        setEntryError(null)
        try {
            await deleteEntryFile(selectedDate, contestId)
            const data = await listEntryFiles(selectedDate)
            setEntryFiles(data)
            setContestOrder(data.map(entry => entry.contest_id))
            setSelectedContestIds(prev => {
                const next = new Set(prev)
                next.delete(contestId)
                return next
            })
            if (selectedContestId === contestId) {
                setSelectedContestId(data[0]?.contest_id ?? null)
            }
        } catch (err) {
            setEntryError((err as Error).message)
        }
    }

    const handleApplyBuild = async () => {
        if (!selectedBuildId) return
        const targetIds = applyBuildTargetInfo.targetIds
        if (targetIds.length === 0) return
        setApplyLoading(true)
        setEntryError(null)
        const results: ApplyBuildResult[] = []
        try {
            let lineups: string[][] = []
            let buildDraftGroupId: number | null = null
            if (buildSource === 'optimizer') {
                const build = await loadSavedBuild(selectedDate, selectedBuildId)
                lineups = build.lineups?.map(lu => lu.player_ids) ?? []
                buildDraftGroupId = build.draft_group_id ?? null
            } else {
                const build = await loadSavedSimBuild(selectedDate, selectedBuildId)
                lineups = build.lineups ?? []
                buildDraftGroupId = build.draft_group_id ?? null
            }
            let offset = 0
            const applyRequests = targetIds.flatMap(contestId => {
                const entrySummary = entryFiles.find(entry => entry.contest_id === contestId)
                const entryCount = entrySummary?.entry_count ?? 0
                if (
                    buildDraftGroupId !== null
                    && entrySummary?.draft_group_id
                    && entrySummary.draft_group_id !== buildDraftGroupId
                ) {
                    throw new Error(
                        `Build slate DG${buildDraftGroupId} does not match contest slate DG${entrySummary.draft_group_id} (${entrySummary.contest_name})`,
                    )
                }
                if (entryCount <= 0) return []
                const slice = lineups.slice(offset, offset + entryCount)
                if (slice.length < entryCount) {
                    throw new Error('Not enough lineups to cover selected contests')
                }
                const request = {
                    contestId,
                    contestName: entrySummary?.contest_name || contestId,
                    entryCount,
                    lineupRangeStart: offset + 1,
                    lineupRangeEnd: offset + entryCount,
                    promise: applyBuildToEntries(
                        selectedDate,
                        contestId,
                        buildSource,
                        selectedBuildId,
                        slice,
                    ),
                }
                offset += entryCount
                return [request]
            })

            const updates = await Promise.all(
                applyRequests.map(async request => ({
                    contestId: request.contestId,
                    contestName: request.contestName,
                    entryCount: request.entryCount,
                    lineupRangeStart: request.lineupRangeStart,
                    lineupRangeEnd: request.lineupRangeEnd,
                    updated: await request.promise,
                })),
            )

            for (const update of updates) {
                results.push({
                    contestId: update.contestId,
                    contestName: update.contestName,
                    lineupsApplied: update.entryCount,
                    lineupRangeStart: update.lineupRangeStart,
                    lineupRangeEnd: update.lineupRangeEnd,
                })
                if (update.contestId === selectedContestId) {
                    setEntryFile(update.updated)
                }
            }
            setApplyBuildResults(results)
            if (results.length > 0) {
                setShowApplyResultsPanel(true)
            }
        } catch (err) {
            setEntryError((err as Error).message)
        } finally {
            setApplyLoading(false)
        }
    }

    const handleExport = async () => {
        const targetIds = applyBuildTargetInfo.targetIds
        if (targetIds.length === 0) return
        try {
            const blob = targetIds.length === 1
                ? await exportEntryFile(selectedDate, targetIds[0])
                : await exportEntriesBatch(selectedDate, targetIds)
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = targetIds.length === 1
                ? `entries_${selectedDate}_${targetIds[0]}.csv`
                : `entries_${selectedDate}_combined.csv`
            a.click()
            URL.revokeObjectURL(url)
        } catch (err) {
            alert('Export failed: ' + (err as Error).message)
        }
    }

    const handleExportCheckedContests = async () => {
        const targetIds = contestOrder.filter(id => selectedContestIds.has(id))
        if (targetIds.length === 0) return
        try {
            const blob = targetIds.length === 1
                ? await exportEntryFile(selectedDate, targetIds[0])
                : await exportEntriesBatch(selectedDate, targetIds)
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = targetIds.length === 1
                ? `entries_${selectedDate}_${targetIds[0]}.csv`
                : `entries_${selectedDate}_selected_contests.csv`
            a.click()
            URL.revokeObjectURL(url)
        } catch (err) {
            alert('Export failed: ' + (err as Error).message)
        }
    }

    const handleExportCurrentContest = async () => {
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

    const handleExportSelectedLineups = async () => {
        if (!entryFile) return
        if (selectedLineupIds.length === 0) return
        try {
            const blob = await exportEntryFile(selectedDate, entryFile.contest_id, selectedLineupIds)
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = `entries_${selectedDate}_${entryFile.contest_id}_selected.csv`
            a.click()
            URL.revokeObjectURL(url)
        } catch (err) {
            alert('Export failed: ' + (err as Error).message)
        }
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
                    onlyOutLineups,
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
                    const baselineEntries = prevEntries ?? result.entry_state.entries
                    setLineupStates(buildLineupStates(baselineEntries, result.entry_state.entries, false))
                    setLineupStateKey(`${selectedDate}:${result.entry_state.contest_id}`)
                    setLineupBaselineLocked(true)
                    setEntryPage(1)
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

    const selectedLineupIds = useMemo(() => {
        return Object.entries(lineupStates)
            .filter(([, state]) => state.isSelected)
            .map(([entryId]) => entryId)
    }, [lineupStates])

    const selectedLineupCount = selectedLineupIds.length

    const selectedBuildInfo = useMemo(() => {
        if (!selectedBuildId) return null
        if (buildSource === 'optimizer') {
            return savedBuilds.find(b => b.job_id === selectedBuildId) ?? null
        }
        return savedSimBuilds.find(b => b.build_id === selectedBuildId) ?? null
    }, [buildSource, savedBuilds, savedSimBuilds, selectedBuildId])

    const buildOptions = useMemo(() => {
        return buildSource === 'optimizer'
            ? savedBuilds.map(b => ({
                id: b.job_id,
                label: `${b.job_id.slice(0, 8)} (DG${b.draft_group_id}, ${b.lineups_count})`,
            }))
            : savedSimBuilds.map(b => ({
                id: b.build_id,
                label: `${b.name ?? b.build_id.slice(0, 8)} (DG${b.draft_group_id ?? '?'}, ${b.lineups_count})`,
            }))
    }, [buildSource, savedBuilds, savedSimBuilds])

    const orderedEntries = useMemo(() => {
        const map = new Map(entryFiles.map(entry => [entry.contest_id, entry]))
        return contestOrder.map(id => map.get(id)).filter(Boolean) as EntryFileSummary[]
    }, [contestOrder, entryFiles])

    const checkedContestIds = useMemo(() => {
        return contestOrder.filter(id => selectedContestIds.has(id))
    }, [contestOrder, selectedContestIds])

    const checkedContestEntryCount = useMemo(() => {
        return checkedContestIds.reduce((sum, id) => {
            const entry = entryFiles.find(e => e.contest_id === id)
            return sum + (entry?.entry_count ?? 0)
        }, 0)
    }, [checkedContestIds, entryFiles])

    const applyBuildTargetInfo = useMemo<ApplyBuildTargetInfo>(() => {
        const explicitTargets = contestOrder.filter(id => selectedContestIds.has(id))
        let targetIds: string[]
        let applyingAllByDefault = false

        if (explicitTargets.length > 0) {
            targetIds = explicitTargets
        } else if (entryFiles.length > 1) {
            // Default to all uploaded contests when multiple are present.
            targetIds = contestOrder.length > 0
                ? [...contestOrder]
                : entryFiles.map(entry => entry.contest_id)
            applyingAllByDefault = true
        } else if (selectedContestId) {
            targetIds = [selectedContestId]
        } else if (entryFiles.length === 1) {
            targetIds = [entryFiles[0].contest_id]
        } else {
            targetIds = []
        }

        const totalEntries = targetIds.reduce((sum, id) => {
            const entry = entryFiles.find(e => e.contest_id === id)
            return sum + (entry?.entry_count ?? 0)
        }, 0)
        return { contestCount: targetIds.length, totalEntries, targetIds, applyingAllByDefault }
    }, [selectedContestIds, selectedContestId, contestOrder, entryFiles])

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

    const toggleLineupSelection = (entryId: string, fallbackLineup: LineupSlots) => {
        setLineupStates(prev => {
            const next = { ...prev }
            const existing = next[entryId]
            if (existing) {
                next[entryId] = { ...existing, isSelected: !existing.isSelected }
            } else {
                next[entryId] = {
                    baselineLineup: fallbackLineup,
                    currentLineup: fallbackLineup,
                    isSelected: true,
                }
            }
            return next
        })
    }

    const selectAllLineups = () => {
        setLineupStates(prev => {
            const next: Record<string, LineupState> = {}
            for (const [entryId, state] of Object.entries(prev)) {
                next[entryId] = { ...state, isSelected: true }
            }
            return next
        })
    }

    const clearAllLineups = () => {
        setLineupStates(prev => {
            const next: Record<string, LineupState> = {}
            for (const [entryId, state] of Object.entries(prev)) {
                next[entryId] = { ...state, isSelected: false }
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

    const moveContestToIndex = (contestId: string, targetIdx: number) => {
        setContestOrder(prev => {
            const idx = prev.indexOf(contestId)
            if (idx === -1) return prev
            const boundedTarget = Math.max(0, Math.min(prev.length - 1, targetIdx))
            if (boundedTarget === idx) return prev
            const next = [...prev]
            const [item] = next.splice(idx, 1)
            next.splice(boundedTarget, 0, item)
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
                            <option value="">Auto (detect from upload)</option>
                            {slateOptions.map(s => (
                                <option key={s.draft_group_id} value={s.draft_group_id}>
                                    {formatSlateLabel(s)} (DG{s.draft_group_id})
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

            {/* Apply Build Results Panel */}
            {showApplyResultsPanel && applyBuildResults.length > 0 && (
                <section className="late-swap-results-panel">
                    <div className="results-header">
                        <h4>Apply Build Results</h4>
                        <button onClick={() => setShowApplyResultsPanel(false)} className="close-btn">×</button>
                    </div>
                    <div className="results-list">
                        {applyBuildResults.map(result => (
                            <div key={result.contestId} className="result-item">
                                <div className="result-contest">{result.contestName}</div>
                                <div className="result-stats">
                                    <span>{result.lineupsApplied} lineups</span>
                                    <span className="muted">|</span>
                                    <span>#{result.lineupRangeStart}-{result.lineupRangeEnd} from build</span>
                                </div>
                            </div>
                        ))}
                    </div>
                    <div className="results-summary">
                        Total: {applyBuildResults.reduce((sum, r) => sum + r.lineupsApplied, 0)} lineups across {applyBuildResults.length} contest{applyBuildResults.length !== 1 ? 's' : ''}
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
                    <button
                        className="build-btn"
                        onClick={handleRepairDraftGroups}
                        disabled={repairing || entryFiles.length === 0}
                        style={{ marginTop: '0.5rem' }}
                        title="Fix slate IDs for existing entry files (uses DK Contest ID mapping)"
                    >
                        {repairing ? 'Fixing...' : 'Fix Slate IDs'}
                    </button>
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
                    {selectedBuildInfo && entryFile && (
                        <div className="muted">
                            Build slate DG{selectedBuildInfo.draft_group_id ?? '?'} | Contest slate DG{entryFile.draft_group_id}
                        </div>
                    )}
                    <button
                        className="build-btn"
                        onClick={handleApplyBuild}
                        disabled={!selectedBuildId || applyLoading || applyBuildTargetInfo.contestCount === 0}
                    >
                        {applyLoading
                            ? 'Applying...'
                            : applyBuildTargetInfo.contestCount > 1
                                ? applyBuildTargetInfo.applyingAllByDefault
                                    ? `Apply to All Contests (${applyBuildTargetInfo.totalEntries} entries)`
                                    : `Apply to ${applyBuildTargetInfo.contestCount} Contests (${applyBuildTargetInfo.totalEntries} entries)`
                                : 'Apply Build'}
                    </button>
                    {applyBuildTargetInfo.applyingAllByDefault && (
                        <div className="muted">No contests selected; applying across all uploaded contests.</div>
                    )}
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
                    {/* Only OUT Lineups Toggle */}
                    <div className="only-out-toggle">
                        <label>
                            <input
                                type="checkbox"
                                checked={onlyOutLineups}
                                onChange={e => setOnlyOutLineups(e.target.checked)}
                            />
                            Only swap OUT lineups
                        </label>
                        <small className="hint">Only optimize lineups containing an OUT player</small>
                    </div>
                    <button
                        className="export-btn primary"
                        onClick={handleExportSelectedLineups}
                        disabled={!entryFile || selectedLineupCount === 0}
                    >
                        Export Selected
                    </button>
                    <button
                        className="export-btn"
                        onClick={handleExportCheckedContests}
                        disabled={checkedContestIds.length === 0}
                    >
                        {checkedContestIds.length > 1
                            ? `Export Checked Contests (${checkedContestEntryCount} lineups)`
                            : 'Export Checked Contest'}
                    </button>
                    <div className="lineup-export-actions">
                        <button
                            className="lineups-action-btn"
                            onClick={selectAllLineups}
                            disabled={!entryFile}
                        >
                            Check All
                        </button>
                        <button
                            className="lineups-action-btn"
                            onClick={clearAllLineups}
                            disabled={!entryFile}
                        >
                            Uncheck All
                        </button>
                        <span className="muted">{selectedLineupCount} selected</span>
                    </div>
                    <button
                        className="export-btn"
                        onClick={handleExportCurrentContest}
                        disabled={!selectedContestId || currentEntriesCount === 0}
                    >
                        Export Current Contest
                    </button>
                    <button
                        className="export-btn"
                        onClick={handleExport}
                        disabled={applyBuildTargetInfo.contestCount === 0 || applyBuildTargetInfo.totalEntries === 0}
                    >
                        {applyBuildTargetInfo.contestCount > 1
                            ? `Export All Uploaded (${applyBuildTargetInfo.totalEntries} lineups)`
                            : 'Export All Lineups'}
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
                    {entryFiles.length > 1 && (
                        <div className="muted" style={{ marginBottom: '0.5rem' }}>
                            Apply order is top to bottom. Earlier contests receive earlier (higher-ranked) build lineups.
                        </div>
                    )}
                    <div className="saved-builds-list">
                        {entryFiles.length === 0 && !entriesLoading && (
                            <span className="muted">No entry files yet.</span>
                        )}
                        {orderedEntries.map((entry, orderIdx) => (
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
                                    <input
                                        type="number"
                                        min={1}
                                        max={Math.max(1, orderedEntries.length)}
                                        value={orderIdx + 1}
                                        onChange={e => {
                                            const value = Number(e.target.value)
                                            if (Number.isFinite(value)) {
                                                moveContestToIndex(entry.contest_id, Math.round(value) - 1)
                                            }
                                        }}
                                        title="Populate order (1 gets top lineups first)"
                                        style={{ width: '3.25rem' }}
                                    />
                                    <span className="saved-build-count">{entry.entry_count} entries</span>
                                    <span className="saved-build-time">DG{entry.draft_group_id}</span>
                                    <span className="saved-build-time">{entry.contest_name}</span>
                                </div>
                                <div className="saved-build-actions">
                                    <button className="load-btn" onClick={() => setSelectedContestId(entry.contest_id)}>
                                        Load
                                    </button>
                                    <button
                                        className="delete-btn"
                                        onClick={() => handleDeleteContest(entry.contest_id)}
                                        title="Delete entry file"
                                    >
                                        Delete
                                    </button>
                                    <button
                                        className="delete-btn"
                                        onClick={() => moveContest(entry.contest_id, 'up')}
                                        title="Move earlier in apply order"
                                    >
                                        ↑
                                    </button>
                                    <button
                                        className="delete-btn"
                                        onClick={() => moveContest(entry.contest_id, 'down')}
                                        title="Move later in apply order"
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
                                    const entryId = getEntryKey(entry, entryPageStart + idx)
                                    const lineupState = lineupStates[entryId]
                                    const baselineLineup = lineupState?.baselineLineup ?? buildLineupSlots(entry)
                                    const currentLineup = lineupState?.currentLineup ?? buildLineupSlots(entry)
                                    const isSelected = lineupState?.isSelected ?? true
                                    const changedSlots = new Set(
                                        DK_SLOTS.filter(slot => (baselineLineup[slot] || '') !== (currentLineup[slot] || '')),
                                    )
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

                                    const realSwaps = computeRealSwaps(baselineLineup, currentLineup, DK_SLOTS)

                                    return (
                                        <div key={entryId} className={`lineup-card ${isSelected ? 'selected' : ''}`}>
                                            <div className="lineup-header">
                                                <div className="lineup-header-left">
                                                    <label className="lineup-select" title="Select lineup for export">
                                                        <input
                                                            type="checkbox"
                                                            checked={isSelected}
                                                            onChange={() => toggleLineupSelection(entryId, currentLineup)}
                                                        />
                                                    </label>
                                                    <span>Entry {entryId}</span>
                                                </div>
                                                <div className="lineup-header-right">
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
                                            </div>
                                            <div className="lineup-players">
                                                {DK_SLOTS.map(slot => {
                                                    const val = currentLineup[slot] || '-'
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
                                            <div className="player-swaps">
                                                {realSwaps.outs.length === 0 && realSwaps.ins.length === 0 ? (
                                                    <div className="swap-item muted">No player swaps</div>
                                                ) : (
                                                    <>
                                                        {realSwaps.outs.length > 0 && (
                                                            <div className="swap-item">
                                                                <span className="swap-slot">OUT:</span>
                                                                <span className="swap-old">
                                                                    {realSwaps.outs.map(extractPlayerName).join(', ')}
                                                                </span>
                                                            </div>
                                                        )}
                                                        {realSwaps.ins.length > 0 && (
                                                            <div className="swap-item">
                                                                <span className="swap-slot">IN:</span>
                                                                <span className="swap-new">
                                                                    {realSwaps.ins.map(extractPlayerName).join(', ')}
                                                                </span>
                                                            </div>
                                                        )}
                                                    </>
                                                )}
                                            </div>
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
