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
import PlayerExposurePanel, { ExposureBounds } from '../components/PlayerExposurePanel'
import { useSlateDateAndSlate } from '../hooks/useSlateDate'
import { formatSlateLabel } from '../utils/slateFormat'

const ownershipStorageKey = 'contestSim.useOwnership'

type SortKey = 'lineup_id' | 'mean' | 'p90' | 'p95' | 'expected_value' | 'roi' | 'win_rate' | 'top_1pct_rate' | 'top_10pct_rate' | 'cash_rate' | 'total_own' | 'ucv90' | 'tail_score' | 'select_score' | 'score_lcb95' | 'score_cvar10' | 'robust_floor'

type LineupResultWithOwnership = ContestSimResponse['results'][number] & { total_own: number }

interface ExposureCountBounds {
    minCount?: number
    maxCount?: number
    minPct?: number
    maxPct?: number
}

interface ConstrainedSelectionResult {
    constrainedResults: LineupResultWithOwnership[]
    minUniquesPassCount: number
    exposureCapError: string | null
    targetCount: number
}

interface SetAndForgetSelection {
    orderedIds: number[]
    coreIds: Set<number>
    upsideIds: Set<number>
    safetyFloor: number
}

const DK_EDITOR_SLOTS = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'] as const

function isEligibleForSlot(player: PoolPlayer, slot: string): boolean {
    const posSet = new Set((player.positions ?? []).map(pos => pos.trim().toUpperCase()))
    switch (slot) {
        case 'PG':
        case 'SG':
        case 'SF':
        case 'PF':
        case 'C':
            return posSet.has(slot)
        case 'G':
            return posSet.has('PG') || posSet.has('SG') || posSet.has('G')
        case 'F':
            return posSet.has('SF') || posSet.has('PF') || posSet.has('F')
        case 'UTIL':
            return posSet.has('PG')
                || posSet.has('SG')
                || posSet.has('SF')
                || posSet.has('PF')
                || posSet.has('C')
                || posSet.has('G')
                || posSet.has('F')
        default:
            return true
    }
}

function editorSlotLabels(slotCount: number): string[] {
    if (slotCount === DK_EDITOR_SLOTS.length) {
        return [...DK_EDITOR_SLOTS]
    }
    return Array.from({ length: slotCount }, (_, idx) => `Slot ${idx + 1}`)
}

function toMinCount(pct: number, targetCount: number): number {
    return Math.ceil((pct / 100) * targetCount)
}

function toMaxCount(pct: number, targetCount: number): number {
    return Math.floor((pct / 100) * targetCount)
}

function getSharedCount(a: string[], b: Set<string>): number {
    let shared = 0
    for (const pid of a) {
        if (b.has(pid)) {
            shared += 1
        }
    }
    return shared
}

function getNumericOrNegInf(value: unknown): number {
    return typeof value === 'number' && Number.isFinite(value) ? value : -Infinity
}

function buildSetAndForgetSelection(
    pool: LineupResultWithOwnership[],
    targetSize: number,
    upsidePct: number,
): SetAndForgetSelection {
    const size = Math.max(0, Math.min(targetSize, pool.length))
    if (size <= 0) {
        return {
            orderedIds: [],
            coreIds: new Set<number>(),
            upsideIds: new Set<number>(),
            safetyFloor: -Infinity,
        }
    }

    const byRobust = [...pool].sort((a, b) => {
        const robustDelta = getNumericOrNegInf(b.robust_floor) - getNumericOrNegInf(a.robust_floor)
        if (robustDelta !== 0) return robustDelta
        return b.expected_value - a.expected_value
    })

    const robustValsAsc = byRobust
        .map(r => getNumericOrNegInf(r.robust_floor))
        .filter(v => Number.isFinite(v))
        .sort((a, b) => a - b)
    const floorIdx = Math.min(
        Math.max(0, Math.floor(0.3 * Math.max(0, robustValsAsc.length - 1))),
        Math.max(0, robustValsAsc.length - 1),
    )
    const safetyFloor = robustValsAsc.length > 0 ? robustValsAsc[floorIdx] : -Infinity

    const clippedUpsidePct = Math.max(0, Math.min(100, upsidePct))
    let upsideCount = Math.round((size * clippedUpsidePct) / 100)
    if (size > 1 && clippedUpsidePct > 0) {
        upsideCount = Math.max(1, upsideCount)
    }
    if (size > 1 && clippedUpsidePct < 100) {
        upsideCount = Math.min(size - 1, upsideCount)
    }
    const coreCount = size - upsideCount

    const core = byRobust.slice(0, coreCount)
    const coreIds = new Set(core.map(r => r.lineup_id))

    const upsideCandidates = byRobust
        .filter(r => !coreIds.has(r.lineup_id))
        .filter(r => getNumericOrNegInf(r.robust_floor) >= safetyFloor)
        .sort((a, b) => {
            const topDelta = b.top_1pct_rate - a.top_1pct_rate
            if (topDelta !== 0) return topDelta
            return getNumericOrNegInf(b.robust_floor) - getNumericOrNegInf(a.robust_floor)
        })

    const upside = upsideCandidates.slice(0, upsideCount)
    const upsideIds = new Set(upside.map(r => r.lineup_id))

    const orderedIds: number[] = [...core.map(r => r.lineup_id), ...upside.map(r => r.lineup_id)]
    if (orderedIds.length < size) {
        for (const r of byRobust) {
            if (orderedIds.length >= size) break
            if (!coreIds.has(r.lineup_id) && !upsideIds.has(r.lineup_id)) {
                orderedIds.push(r.lineup_id)
            }
        }
    }

    return {
        orderedIds,
        coreIds,
        upsideIds,
        safetyFloor,
    }
}

function selectConstrainedLineups(
    sortedResults: LineupResultWithOwnership[],
    requestedTargetCount: number,
    minUniques: number,
    exposureBounds: Map<string, ExposureBounds>,
    playerMap: Map<string, PoolPlayer>,
): ConstrainedSelectionResult {
    const targetCount = Math.max(0, Math.min(requestedTargetCount, sortedResults.length))
    if (targetCount === 0 || sortedResults.length === 0) {
        return {
            constrainedResults: [],
            minUniquesPassCount: 0,
            exposureCapError: null,
            targetCount,
        }
    }

    const normalizedBounds = new Map<string, ExposureCountBounds>()
    const boundErrors: string[] = []
    for (const [pid, bounds] of exposureBounds.entries()) {
        const minPct = bounds.min !== undefined && Number.isFinite(bounds.min)
            ? Math.max(0, Math.min(100, bounds.min))
            : undefined
        const maxPct = bounds.max !== undefined && Number.isFinite(bounds.max)
            ? Math.max(0, Math.min(100, bounds.max))
            : undefined
        if (minPct === undefined && maxPct === undefined) {
            continue
        }
        const minCount = minPct !== undefined ? toMinCount(minPct, targetCount) : undefined
        const maxCount = maxPct !== undefined ? toMaxCount(maxPct, targetCount) : undefined
        if (minCount !== undefined && maxCount !== undefined && minCount > maxCount) {
            const playerName = playerMap.get(pid)?.name ?? pid
            boundErrors.push(`${playerName}: min ${minPct}% > max ${maxPct}%`)
        }
        normalizedBounds.set(pid, { minCount, maxCount, minPct, maxPct })
    }

    const hasExposureConstraints = normalizedBounds.size > 0
    const hasConstraints = hasExposureConstraints || minUniques > 0
    if (!hasConstraints) {
        return {
            constrainedResults: sortedResults.slice(0, targetCount),
            minUniquesPassCount: sortedResults.length,
            exposureCapError: null,
            targetCount,
        }
    }

    const minEntries: Array<{ pid: string; minCount: number }> = []
    const maxByPid = new Map<string, number>()
    for (const [pid, bounds] of normalizedBounds.entries()) {
        if (bounds.minCount !== undefined && bounds.minCount > 0) {
            minEntries.push({ pid, minCount: bounds.minCount })
        }
        if (bounds.maxCount !== undefined) {
            maxByPid.set(pid, bounds.maxCount)
        }
    }

    // Fast path: when there are no exposure minimum constraints, a single-pass greedy
    // keeps the page responsive even for large candidate pools.
    if (minEntries.length === 0) {
        const maxFastMinUniquesTarget = 200
        if (minUniques > 0 && targetCount > maxFastMinUniquesTarget) {
            return {
                constrainedResults: sortedResults.slice(0, targetCount),
                minUniquesPassCount: sortedResults.length,
                exposureCapError: `Min uniques requires Top N ≤ ${maxFastMinUniquesTarget}. Lower Top N to enforce min uniques safely.`,
                targetCount,
            }
        }

        const result: LineupResultWithOwnership[] = []
        const selectedSets: Set<string>[] = []
        const counts = new Map<string, number>()
        let minUniquesPassCount = sortedResults.length
        if (minUniques > 0) {
            minUniquesPassCount = 0
        }

        for (const lineup of sortedResults) {
            if (result.length >= targetCount) {
                break
            }

            let valid = true
            if (minUniques > 0) {
                const lineupSet = new Set(lineup.player_ids)
                for (const existingSet of selectedSets) {
                    const shared = getSharedCount(lineup.player_ids, existingSet)
                    if (lineup.player_ids.length - shared < minUniques) {
                        valid = false
                        break
                    }
                }
                if (valid) {
                    minUniquesPassCount += 1
                }
            }
            if (!valid) {
                continue
            }

            for (const pid of lineup.player_ids) {
                const maxCount = maxByPid.get(pid)
                if (maxCount !== undefined && (counts.get(pid) ?? 0) + 1 > maxCount) {
                    valid = false
                    break
                }
            }
            if (!valid) {
                continue
            }

            result.push(lineup)
            selectedSets.push(new Set(lineup.player_ids))
            for (const pid of lineup.player_ids) {
                if (maxByPid.has(pid)) {
                    counts.set(pid, (counts.get(pid) ?? 0) + 1)
                }
            }
        }

        let error: string | null = null
        if (result.length < targetCount) {
            error = `Only ${result.length} of ${targetCount} lineups meet constraints (pool exhausted)`
        }
        return {
            constrainedResults: result,
            minUniquesPassCount,
            exposureCapError: error,
            targetCount,
        }
    }

    // Exposure minimums are combinatorial; cap exact solve size to avoid UI lockups.
    const maxExactSolveTarget = 120
    if (targetCount > maxExactSolveTarget) {
        return {
            constrainedResults: sortedResults.slice(0, targetCount),
            minUniquesPassCount: sortedResults.length,
            exposureCapError: `Exposure mins require Top N ≤ ${maxExactSolveTarget}. Lower Top N to apply min exposures.`,
            targetCount,
        }
    }

    const constrainedPidSet = new Set(normalizedBounds.keys())
    const lineupSets = sortedResults.map(r => new Set(r.player_ids))
    const lineupConstrainedPids = sortedResults.map(r => r.player_ids.filter(pid => constrainedPidSet.has(pid)))
    const lineupConstrainedPidSet = lineupConstrainedPids.map(pids => new Set(pids))
    const lineupRankScore = sortedResults.map((_, idx) => sortedResults.length - idx)

    for (const { pid, minCount } of minEntries) {
        const coverage = lineupConstrainedPidSet.reduce((count, pidSet) => count + (pidSet.has(pid) ? 1 : 0), 0)
        if (coverage < minCount) {
            const playerName = playerMap.get(pid)?.name ?? pid
            boundErrors.push(
                `${playerName}: needs ${minCount}/${targetCount} lineups, only ${coverage} available`,
            )
        }
    }

    const selectedIndices: number[] = []
    const selectedIndexSet = new Set<number>()
    const selectedCounts = new Map<string, number>()

    const addIndex = (idx: number) => {
        selectedIndices.push(idx)
        selectedIndexSet.add(idx)
        for (const pid of lineupConstrainedPids[idx]) {
            selectedCounts.set(pid, (selectedCounts.get(pid) ?? 0) + 1)
        }
    }

    const canAddIndex = (idx: number): boolean => {
        for (const pid of lineupConstrainedPids[idx]) {
            const maxCount = maxByPid.get(pid)
            if (maxCount !== undefined && (selectedCounts.get(pid) ?? 0) + 1 > maxCount) {
                return false
            }
        }
        if (minUniques <= 0) {
            return true
        }
        for (const existingIdx of selectedIndices) {
            const shared = getSharedCount(sortedResults[idx].player_ids, lineupSets[existingIdx])
            if (sortedResults[idx].player_ids.length - shared < minUniques) {
                return false
            }
        }
        return true
    }

    const computeDeficits = (): Array<{ pid: string; deficit: number }> => {
        if (minEntries.length === 0) {
            return []
        }
        const deficits: Array<{ pid: string; deficit: number }> = []
        for (const { pid, minCount } of minEntries) {
            const deficit = minCount - (selectedCounts.get(pid) ?? 0)
            if (deficit > 0) {
                deficits.push({ pid, deficit })
            }
        }
        return deficits
    }

    const computeTotalDeficit = (): number => {
        return computeDeficits().reduce((sum, d) => sum + d.deficit, 0)
    }

    let minUniquesPassCount = sortedResults.length
    if (minUniques > 0) {
        const uniquesOnlySelected: number[] = []
        minUniquesPassCount = 0
        for (let idx = 0; idx < sortedResults.length; idx += 1) {
            const compatible = uniquesOnlySelected.every(existingIdx => {
                const shared = getSharedCount(sortedResults[idx].player_ids, lineupSets[existingIdx])
                return sortedResults[idx].player_ids.length - shared >= minUniques
            })
            if (compatible) {
                uniquesOnlySelected.push(idx)
                minUniquesPassCount += 1
            }
        }
    }

    while (selectedIndices.length < targetCount) {
        const deficits = computeDeficits()
        let bestIdx = -1
        let bestScore = -Infinity

        for (let idx = 0; idx < sortedResults.length; idx += 1) {
            if (selectedIndexSet.has(idx) || !canAddIndex(idx)) {
                continue
            }
            let gain = 0
            if (deficits.length > 0) {
                for (const { pid } of deficits) {
                    if (lineupConstrainedPidSet[idx].has(pid)) {
                        gain += 1
                    }
                }
                if (gain === 0) {
                    continue
                }
            }
            const score = gain * 1_000_000 + lineupRankScore[idx]
            if (score > bestScore) {
                bestScore = score
                bestIdx = idx
            }
        }

        if (bestIdx < 0) {
            break
        }
        addIndex(bestIdx)
    }

    if (selectedIndices.length < targetCount) {
        for (let idx = 0; idx < sortedResults.length && selectedIndices.length < targetCount; idx += 1) {
            if (selectedIndexSet.has(idx) || !canAddIndex(idx)) {
                continue
            }
            addIndex(idx)
        }
    }

    const canSwap = (addIdx: number, removeIdx: number): boolean => {
        for (const [pid, maxCount] of maxByPid.entries()) {
            let nextCount = selectedCounts.get(pid) ?? 0
            if (lineupConstrainedPidSet[removeIdx].has(pid)) {
                nextCount -= 1
            }
            if (lineupConstrainedPidSet[addIdx].has(pid)) {
                nextCount += 1
            }
            if (nextCount > maxCount) {
                return false
            }
        }

        if (minUniques > 0) {
            for (const existingIdx of selectedIndices) {
                if (existingIdx === removeIdx) {
                    continue
                }
                const shared = getSharedCount(sortedResults[addIdx].player_ids, lineupSets[existingIdx])
                if (sortedResults[addIdx].player_ids.length - shared < minUniques) {
                    return false
                }
            }
        }
        return true
    }

    const deficitAfterSwap = (addIdx: number, removeIdx: number): number => {
        let deficit = 0
        for (const { pid, minCount } of minEntries) {
            let nextCount = selectedCounts.get(pid) ?? 0
            if (lineupConstrainedPidSet[removeIdx].has(pid)) {
                nextCount -= 1
            }
            if (lineupConstrainedPidSet[addIdx].has(pid)) {
                nextCount += 1
            }
            deficit += Math.max(0, minCount - nextCount)
        }
        return deficit
    }

    let repairIterations = 0
    while (computeTotalDeficit() > 0 && repairIterations < 200) {
        repairIterations += 1
        const deficits = computeDeficits()
        let bestSwap: { addIdx: number; removeIdx: number; deficit: number; scoreDelta: number } | null = null

        for (let addIdx = 0; addIdx < sortedResults.length; addIdx += 1) {
            if (selectedIndexSet.has(addIdx)) {
                continue
            }
            const hasDeficitPlayer = deficits.some(({ pid }) => lineupConstrainedPidSet[addIdx].has(pid))
            if (!hasDeficitPlayer) {
                continue
            }
            for (const removeIdx of selectedIndices) {
                if (!canSwap(addIdx, removeIdx)) {
                    continue
                }
                const deficit = deficitAfterSwap(addIdx, removeIdx)
                const scoreDelta = lineupRankScore[addIdx] - lineupRankScore[removeIdx]
                if (
                    bestSwap === null
                    || deficit < bestSwap.deficit
                    || (deficit === bestSwap.deficit && scoreDelta > bestSwap.scoreDelta)
                ) {
                    bestSwap = { addIdx, removeIdx, deficit, scoreDelta }
                }
            }
        }

        if (bestSwap === null) {
            break
        }
        const currentDeficit = computeTotalDeficit()
        if (bestSwap.deficit > currentDeficit) {
            break
        }
        if (bestSwap.deficit === currentDeficit && bestSwap.scoreDelta <= 0) {
            break
        }

        const swap = bestSwap
        if (!swap) {
            continue
        }
        const removePos = selectedIndices.findIndex(idx => idx === swap.removeIdx)
        if (removePos < 0) {
            continue
        }
        selectedIndices[removePos] = swap.addIdx
        selectedIndexSet.delete(swap.removeIdx)
        selectedIndexSet.add(swap.addIdx)

        for (const pid of lineupConstrainedPids[swap.removeIdx]) {
            selectedCounts.set(pid, (selectedCounts.get(pid) ?? 0) - 1)
        }
        for (const pid of lineupConstrainedPids[swap.addIdx]) {
            selectedCounts.set(pid, (selectedCounts.get(pid) ?? 0) + 1)
        }
    }

    selectedIndices.sort((a, b) => a - b)
    const constrainedResults = selectedIndices.map(idx => sortedResults[idx])

    const unmetMin: string[] = []
    for (const { pid, minCount } of minEntries) {
        const currentCount = selectedCounts.get(pid) ?? 0
        if (currentCount < minCount) {
            const playerName = playerMap.get(pid)?.name ?? pid
            const currentPct = constrainedResults.length > 0 ? (currentCount / constrainedResults.length) * 100 : 0
            const targetPct = normalizedBounds.get(pid)?.minPct ?? 0
            unmetMin.push(`${playerName}: ${currentPct.toFixed(1)}% < ${targetPct}%`)
        }
    }

    const errors = [...boundErrors]
    if (selectedIndices.length < targetCount) {
        errors.push(`Only ${selectedIndices.length} of ${targetCount} lineups meet constraints (pool exhausted)`)
    }
    if (unmetMin.length > 0) {
        errors.push(`Min not met: ${unmetMin.join(', ')}`)
    }

    return {
        constrainedResults,
        minUniquesPassCount,
        exposureCapError: errors.length > 0 ? errors.join(' | ') : null,
        targetCount,
    }
}

export default function ContestSimPage() {
    // Date and slate selection (persisted in URL)
    const [selectedDate, setSelectedDate, selectedSlate, setSelectedSlate] = useSlateDateAndSlate()
    const [slates, setSlates] = useState<Slate[]>([])
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
    const [useOwnership, setUseOwnership] = useState(() => {
        if (typeof window === 'undefined') {
            return false
        }
        const stored = window.localStorage.getItem(ownershipStorageKey)
        return stored ? stored === 'true' : false
    })

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
    const [playerSearch, setPlayerSearch] = useState('')
    const [requiredPlayerIds, setRequiredPlayerIds] = useState<string[]>([])
    const [finalSetSize, setFinalSetSize] = useState(40)
    const [finalUpsidePct, setFinalUpsidePct] = useState(20)

    const [selectedLineups, setSelectedLineups] = useState<Set<number>>(new Set())
    const [manualIncludeFinal, setManualIncludeFinal] = useState<Set<number>>(new Set())
    const [manualExcludeFinal, setManualExcludeFinal] = useState<Set<number>>(new Set())
    const [editedLineupsById, setEditedLineupsById] = useState<Record<number, string[]>>({})
    const [editingLineupId, setEditingLineupId] = useState<number | null>(null)
    const [editingLineupInputs, setEditingLineupInputs] = useState<string[]>([])
    const [editingActiveSlotIndex, setEditingActiveSlotIndex] = useState(0)
    const [editingEligibleSearch, setEditingEligibleSearch] = useState('')
    const [editingLineupError, setEditingLineupError] = useState<string | null>(null)

    // Pagination & Top N Filter
    const [page, setPage] = useState(1)
    const [pageSize, setPageSize] = useState(50)
    const [topN, setTopN] = useState<number | null>(null) // null = all
    const [minUniques, setMinUniques] = useState(0)
    const [exposureBounds, setExposureBounds] = useState<Map<string, ExposureBounds>>(new Map())

    const ownershipMode = useOwnership ? 'full' : 'off'
    const rankMode = useOwnership ? 'current' : 'tail_only'
    const displayOwnershipMode = (() => {
        const debugMode = simResult?.stats?.debug?.ownership_mode
        return typeof debugMode === 'string' ? debugMode : ownershipMode
    })()

    // Handler for changing exposure bounds
    const handleExposureBoundsChange = useCallback((playerId: string, bounds: ExposureBounds | null) => {
        setExposureBounds(prev => {
            const next = new Map(prev)
            if (bounds === null) {
                next.delete(playerId)
            } else {
                next.set(playerId, bounds)
            }
            return next
        })
    }, [])


    // Load slates when date changes
    useEffect(() => {
        const load = async () => {
            setSlatesLoading(true)
            try {
                const data = await getSlates(selectedDate)
                setSlates(data)
                // If URL has a slate and it exists in the data, keep it; otherwise auto-select
                const urlSlateExists = selectedSlate && data.some(s => s.draft_group_id === selectedSlate)
                if (!urlSlateExists) {
                    const mainSlates = data.filter(s => s.slate_type === 'main')
                    const bestMain = [...mainSlates].sort((a, b) => {
                        const aContests = a.n_contests ?? 0
                        const bContests = b.n_contests ?? 0
                        if (aContests !== bContests) return bContests - aContests
                        const aGames = a.games?.length ?? 0
                        const bGames = b.games?.length ?? 0
                        if (aGames !== bGames) return bGames - aGames
                        return b.draft_group_id - a.draft_group_id
                    })[0]
                    const fallback = data[0]
                    setSelectedSlate(bestMain?.draft_group_id ?? fallback?.draft_group_id ?? null)
                }
            } catch {
                setSlates([])
                setSelectedSlate(null)
            } finally {
                setSlatesLoading(false)
            }
        }
        void load()
    }, [selectedDate]) // eslint-disable-line react-hooks/exhaustive-deps

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
                const latestLineup = builds.find(b => b.kind === 'lineups')?.build_id ?? null
                if (latestRun) {
                    setSelectedSimBuildId(latestRun)
                    setSelectedSimLineupId(null)
                } else {
                    setSelectedSimBuildId(null)
                    setSelectedSimLineupId(latestLineup)
                }
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

    useEffect(() => {
        if (typeof window === 'undefined') {
            return
        }
        window.localStorage.setItem(ownershipStorageKey, String(useOwnership))
    }, [useOwnership])

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
                    if (build.draft_group_id && build.draft_group_id !== selectedSlate) {
                        setSelectedSlate(build.draft_group_id)
                    }
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
    }, [selectedDate, selectedSimBuildId, selectedSlate, setSelectedSlate])

    // Clear selection when results change
    useEffect(() => {
        setSelectedLineups(new Set())
        setManualIncludeFinal(new Set())
        setManualExcludeFinal(new Set())
        setEditedLineupsById({})
        setEditingLineupId(null)
        setEditingLineupInputs([])
        setEditingLineupError(null)
    }, [simResult])

    // Player name lookup
    const playerMap = useMemo(() => {
        const map = new Map<string, PoolPlayer>()
        pool.forEach(p => map.set(p.player_id, p))
        return map
    }, [pool])

    const sortedPlayers = useMemo(() => {
        return [...pool].sort((a, b) => a.name.localeCompare(b.name))
    }, [pool])

    const resolvePlayerFromSearch = useCallback((query: string): PoolPlayer | null => {
        const raw = query.trim()
        if (!raw) return null
        const q = raw.toLowerCase()
        const exactId = sortedPlayers.find(p => p.player_id === raw)
        if (exactId) return exactId
        const exactName = sortedPlayers.find(p => p.name.toLowerCase() === q)
        if (exactName) return exactName
        const prefixMatch = sortedPlayers.find(p => p.name.toLowerCase().startsWith(q))
        if (prefixMatch) return prefixMatch
        return sortedPlayers.find(p => p.name.toLowerCase().includes(q)) ?? null
    }, [sortedPlayers])

    const addRequiredPlayer = useCallback((query: string) => {
        const player = resolvePlayerFromSearch(query)
        if (!player) return false
        setRequiredPlayerIds(prev => (prev.includes(player.player_id) ? prev : [...prev, player.player_id]))
        return true
    }, [resolvePlayerFromSearch])

    const getEffectivePlayerIds = useCallback((lineup: { lineup_id: number; player_ids: string[] }): string[] => {
        return editedLineupsById[lineup.lineup_id] ?? lineup.player_ids
    }, [editedLineupsById])

    const editingSlotNames = useMemo(() => editorSlotLabels(editingLineupInputs.length), [editingLineupInputs.length])

    const editingResolvedPlayers = useMemo(
        () => editingLineupInputs.map(input => resolvePlayerFromSearch(input)),
        [editingLineupInputs, resolvePlayerFromSearch],
    )

    const editingTotalSalary = useMemo(
        () => editingResolvedPlayers.reduce((sum, player) => sum + (player?.salary ?? 0), 0),
        [editingResolvedPlayers],
    )

    const editingTotalProjection = useMemo(
        () => editingResolvedPlayers.reduce((sum, player) => sum + (player?.proj ?? 0), 0),
        [editingResolvedPlayers],
    )

    const activeEditorSlotName = editingSlotNames[editingActiveSlotIndex] ?? null

    const eligiblePlayersForActiveSlot = useMemo(() => {
        if (editingLineupId === null || !activeEditorSlotName) {
            return [] as PoolPlayer[]
        }

        const enforceSlotEligibility = editingSlotNames.length === DK_EDITOR_SLOTS.length
        const activePlayerId = editingResolvedPlayers[editingActiveSlotIndex]?.player_id
        const usedIds = new Set(
            editingResolvedPlayers
                .map(player => player?.player_id)
                .filter((pid): pid is string => Boolean(pid)),
        )
        if (activePlayerId) {
            usedIds.delete(activePlayerId)
        }
        const query = editingEligibleSearch.trim().toLowerCase()

        return sortedPlayers
            .filter(player => !usedIds.has(player.player_id))
            .filter(player => !enforceSlotEligibility || isEligibleForSlot(player, activeEditorSlotName))
            .filter(player => {
                if (!query) return true
                const pid = player.player_id.toLowerCase()
                const name = player.name.toLowerCase()
                const team = (player.team ?? '').toLowerCase()
                return name.includes(query) || pid.includes(query) || team.includes(query)
            })
            .sort((a, b) => {
                if (b.proj !== a.proj) {
                    return b.proj - a.proj
                }
                return a.name.localeCompare(b.name)
            })
    }, [
        activeEditorSlotName,
        editingActiveSlotIndex,
        editingEligibleSearch,
        editingLineupId,
        editingResolvedPlayers,
        editingSlotNames,
        sortedPlayers,
    ])

    // Calculate total ownership for each lineup result
    const resultsWithOwnership = useMemo(() => {
        if (!simResult) return []
        return simResult.results.map(r => {
            const effectivePlayerIds = getEffectivePlayerIds(r)
            const totalOwn = effectivePlayerIds.reduce((sum, pid) => {
                const p = playerMap.get(pid)
                return sum + (p?.own_proj ?? 0)
            }, 0)
            return { ...r, player_ids: effectivePlayerIds, total_own: totalOwn }
        })
    }, [simResult, playerMap, getEffectivePlayerIds])

    // Filter and sort results (NO Top N here - that's applied after constraint filters)
    const sortedResults = useMemo(() => {
        let results = [...resultsWithOwnership]

        // Apply filters
        if (filterPositiveEV) {
            results = results.filter(r => r.expected_value >= 0)
        }
        if (maxOwnership !== null) {
            results = results.filter(r => r.total_own <= maxOwnership)
        }

        // Sort (handle null/undefined values by treating them as -Infinity for desc, +Infinity for asc)
        results.sort((a, b) => {
            const aVal = a[sortKey as keyof typeof a]
            const bVal = b[sortKey as keyof typeof b]
            const aNum = typeof aVal === 'number' && !isNaN(aVal) ? aVal : (sortDir === 'desc' ? -Infinity : Infinity)
            const bNum = typeof bVal === 'number' && !isNaN(bVal) ? bVal : (sortDir === 'desc' ? -Infinity : Infinity)
            return sortDir === 'asc' ? aNum - bNum : bNum - aNum
        })

        return results
    }, [resultsWithOwnership, filterPositiveEV, maxOwnership, sortKey, sortDir])
    // Combined constraint filter: top-N + min uniques + exposure min/max.
    const { constrainedResults, minUniquesPassCount, exposureCapError, targetCount } = useMemo(() => {
        return selectConstrainedLineups(
            sortedResults,
            topN ?? sortedResults.length,
            minUniques,
            exposureBounds,
            playerMap,
        )
    }, [sortedResults, topN, minUniques, exposureBounds, playerMap])

    const filteredByPlayersResults = useMemo(() => {
        if (requiredPlayerIds.length === 0) {
            return constrainedResults
        }
        return constrainedResults.filter(r =>
            requiredPlayerIds.every(pid => getEffectivePlayerIds(r).includes(pid)),
        )
    }, [constrainedResults, requiredPlayerIds, getEffectivePlayerIds])

    const poolByLineupId = useMemo(() => {
        return new Map(filteredByPlayersResults.map(r => [r.lineup_id, r] as const))
    }, [filteredByPlayersResults])

    const setAndForgetAuto = useMemo(() => {
        return buildSetAndForgetSelection(filteredByPlayersResults, finalSetSize, finalUpsidePct)
    }, [filteredByPlayersResults, finalSetSize, finalUpsidePct])

    const finalSetLineupIds = useMemo(() => {
        const poolIds = new Set(filteredByPlayersResults.map(r => r.lineup_id))
        const included = Array.from(manualIncludeFinal).filter(id => poolIds.has(id) && !manualExcludeFinal.has(id))
        const desiredSize = Math.min(filteredByPlayersResults.length, Math.max(finalSetSize, included.length))
        const ordered: number[] = [...included]
        for (const id of setAndForgetAuto.orderedIds) {
            if (ordered.length >= desiredSize) break
            if (manualExcludeFinal.has(id)) continue
            if (!ordered.includes(id)) {
                ordered.push(id)
            }
        }
        return ordered
    }, [filteredByPlayersResults, finalSetSize, manualIncludeFinal, manualExcludeFinal, setAndForgetAuto])

    const finalSetIdSet = useMemo(() => new Set(finalSetLineupIds), [finalSetLineupIds])
    const finalSetResults = useMemo(() => {
        return finalSetLineupIds
            .map(id => poolByLineupId.get(id))
            .filter((r): r is LineupResultWithOwnership => Boolean(r))
    }, [finalSetLineupIds, poolByLineupId])
    const editedFinalCount = useMemo(() => {
        return finalSetResults.filter(r => Boolean(editedLineupsById[r.lineup_id])).length
    }, [finalSetResults, editedLineupsById])

    const activeResults = useMemo(() => {
        if (selectedLineups.size === 0) return filteredByPlayersResults
        return filteredByPlayersResults.filter(r => selectedLineups.has(r.lineup_id))
    }, [filteredByPlayersResults, selectedLineups])

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
        return filteredByPlayersResults.slice(start, start + pageSize)
    }, [filteredByPlayersResults, page, pageSize])

    const totalPages = Math.max(1, Math.ceil(filteredByPlayersResults.length / pageSize))
    const hasVisibleResults = filteredByPlayersResults.length > 0

    // Reset page when filters change
    useEffect(() => {
        setPage(1)
    }, [filterPositiveEV, maxOwnership, topN, sortKey, sortDir, minUniques, exposureBounds, requiredPlayerIds])

    useEffect(() => {
        const visibleIds = new Set(filteredByPlayersResults.map(r => r.lineup_id))
        setSelectedLineups(prev => {
            let changed = false
            const next = new Set<number>()
            prev.forEach(id => {
                if (visibleIds.has(id)) {
                    next.add(id)
                } else {
                    changed = true
                }
            })
            return changed ? next : prev
        })
        setManualIncludeFinal(prev => new Set(Array.from(prev).filter(id => visibleIds.has(id))))
        setManualExcludeFinal(prev => new Set(Array.from(prev).filter(id => visibleIds.has(id))))
        setEditedLineupsById(prev => {
            const next: Record<number, string[]> = {}
            Object.entries(prev).forEach(([id, pids]) => {
                if (visibleIds.has(Number(id))) {
                    next[Number(id)] = pids
                }
            })
            return next
        })
    }, [filteredByPlayersResults])

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
                ownership_mode: ownershipMode,
                rank_mode: rankMode,
            })
            setSimResult(result)
            const builds = await getSavedSimBuilds(selectedDate)
            setSavedSimBuilds(builds)
            setSelectedSimBuildId(result.build_id ?? builds.find(b => b.kind === 'run')?.build_id ?? null)
            setSelectedSimLineupId(null)
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
        ownershipMode,
        rankMode,
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
                ownership_mode: ownershipMode,
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
        ownershipMode,
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
                    if (build.draft_group_id && build.draft_group_id !== selectedSlate) {
                        setSelectedSlate(build.draft_group_id)
                    }
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
    }, [selectedDate, selectedSimLineupId, selectedSlate, setSelectedSlate])

    const handleSaveSimLineups = async () => {
        if (!selectedSlate) return
        if (filteredByPlayersResults.length === 0) return
        const editedCount = filteredByPlayersResults.filter(r => Boolean(editedLineupsById[r.lineup_id])).length
        if (editedCount > 0) {
            alert(`There are ${editedCount} edited lineups in view. Run sim on the edited set first, then save.`)
            return
        }
        const defaultName = `Sim lineups (${filteredByPlayersResults.length})`
        const name = prompt('Save lineups as:', defaultName)?.trim()
        if (!name) return
        try {
            const lineupsToSave = filteredByPlayersResults.map(getEffectivePlayerIds)
            const resultIds = new Set(filteredByPlayersResults.map(r => r.lineup_id))
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
            setSelectedSimBuildId(null)
            setSelectedSimLineupId(saved.build_id)
        } catch (err) {
            alert('Failed to save sim lineups: ' + (err as Error).message)
        }
    }

    const handleSaveFinalSet = async () => {
        if (!selectedSlate) return
        if (finalSetResults.length === 0) return
        if (editedFinalCount > 0) {
            alert(`There are ${editedFinalCount} edited lineups in the final set. Run sim on final set first, then save.`)
            return
        }
        const defaultName = `Final set (${finalSetResults.length})`
        const name = prompt('Save final set as:', defaultName)?.trim()
        if (!name) return
        try {
            const lineupsToSave = finalSetResults.map(getEffectivePlayerIds)
            const resultIds = new Set(finalSetResults.map(r => r.lineup_id))
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
            setSelectedSimBuildId(null)
            setSelectedSimLineupId(saved.build_id)
        } catch (err) {
            alert('Failed to save final set: ' + (err as Error).message)
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
        setSelectedLineups(new Set(filteredByPlayersResults.map(r => r.lineup_id)))
    }

    const clearSelection = () => {
        setSelectedLineups(new Set())
    }

    // Export handler
    const handleExport = async (type: 'selected' | 'view' | 'final') => {
        if (!selectedSlate) return

        let lineupsToExport: LineupResultWithOwnership[] = []

        if (type === 'selected') {
            if (selectedLineups.size === 0) return
            lineupsToExport = filteredByPlayersResults.filter(r => selectedLineups.has(r.lineup_id))
        } else if (type === 'final') {
            if (finalSetResults.length === 0) return
            lineupsToExport = finalSetResults
        } else {
            // Export current filtered view (Top N + constraints + player filters)
            lineupsToExport = filteredByPlayersResults
        }

        const selectedPlayerIds = lineupsToExport.map(getEffectivePlayerIds)
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

    const handleAddPlayerFilter = () => {
        if (!playerSearch.trim()) return
        const added = addRequiredPlayer(playerSearch)
        if (added) {
            setPlayerSearch('')
        }
    }

    const removePlayerFilter = (playerId: string) => {
        setRequiredPlayerIds(prev => prev.filter(pid => pid !== playerId))
    }

    const toggleFinalSetMembership = (lineupId: number) => {
        if (finalSetIdSet.has(lineupId)) {
            setManualExcludeFinal(prev => new Set(prev).add(lineupId))
            setManualIncludeFinal(prev => {
                const next = new Set(prev)
                next.delete(lineupId)
                return next
            })
        } else {
            setManualIncludeFinal(prev => new Set(prev).add(lineupId))
            setManualExcludeFinal(prev => {
                const next = new Set(prev)
                next.delete(lineupId)
                return next
            })
        }
    }

    const clearFinalOverride = (lineupId: number) => {
        setManualIncludeFinal(prev => {
            const next = new Set(prev)
            next.delete(lineupId)
            return next
        })
        setManualExcludeFinal(prev => {
            const next = new Set(prev)
            next.delete(lineupId)
            return next
        })
    }

    const applyFinalSetToSelection = () => {
        setSelectedLineups(new Set(finalSetLineupIds))
    }

    const runFinalSet = async () => {
        if (finalSetResults.length === 0) {
            setSimError('No final-set lineups available.')
            return
        }
        const lineupsToRun = finalSetResults.map(getEffectivePlayerIds)
        await runSimWithLineups(lineupsToRun)
    }

    const openLineupEditor = (lineup: LineupResultWithOwnership) => {
        const current = getEffectivePlayerIds(lineup)
        setEditingLineupId(lineup.lineup_id)
        setEditingLineupInputs(current.map(pid => playerMap.get(pid)?.name ?? pid))
        setEditingActiveSlotIndex(0)
        setEditingEligibleSearch('')
        setEditingLineupError(null)
    }

    const closeLineupEditor = () => {
        setEditingLineupId(null)
        setEditingLineupInputs([])
        setEditingActiveSlotIndex(0)
        setEditingEligibleSearch('')
        setEditingLineupError(null)
    }

    const setEditingSlotToPlayer = (slotIdx: number, player: PoolPlayer) => {
        setEditingLineupInputs(prev => {
            if (slotIdx < 0 || slotIdx >= prev.length) {
                return prev
            }
            const next = [...prev]
            next[slotIdx] = player.name
            return next
        })
        setEditingLineupError(null)
    }

    const saveLineupEditor = () => {
        if (editingLineupId === null) return
        if (editingLineupInputs.length !== 8) {
            setEditingLineupError('Lineup must have exactly 8 players.')
            return
        }

        const resolved = editingLineupInputs.map(input => resolvePlayerFromSearch(input))
        if (resolved.some(p => p === null)) {
            setEditingLineupError('One or more player names were not recognized.')
            return
        }
        const pids = resolved.filter((p): p is PoolPlayer => p !== null).map(p => p.player_id)
        const unique = new Set(pids)
        if (unique.size !== pids.length) {
            setEditingLineupError('Lineup contains duplicate players.')
            return
        }

        const enforceSlotEligibility = editingSlotNames.length === DK_EDITOR_SLOTS.length
        if (enforceSlotEligibility) {
            for (let idx = 0; idx < resolved.length; idx += 1) {
                const player = resolved[idx]
                const slot = editingSlotNames[idx]
                if (!player) continue
                if (!isEligibleForSlot(player, slot)) {
                    setEditingLineupError(`${player.name} is not eligible for ${slot}.`)
                    return
                }
            }
        }

        const salary = pids.reduce((sum, pid) => sum + (playerMap.get(pid)?.salary ?? 0), 0)
        if (salary > 50000) {
            setEditingLineupError(`Salary cap exceeded: $${salary.toLocaleString()} > $50,000`)
            return
        }
        setEditedLineupsById(prev => ({ ...prev, [editingLineupId]: pids }))
        closeLineupEditor()
    }

    const resetEditedLineup = (lineupId: number) => {
        setEditedLineupsById(prev => {
            const next = { ...prev }
            delete next[lineupId]
            return next
        })
    }

    const requiredPlayerNames = requiredPlayerIds.map(pid => playerMap.get(pid)?.name ?? pid)

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
                            {slateOptions.length === 0 && <option value="">No slates</option>}
                            {slateOptions.map(s => (
                                <option key={s.draft_group_id} value={s.draft_group_id}>
                                    {formatSlateLabel(s)} (DG{s.draft_group_id})
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
                                    {b.job_id.slice(0, 8)} (DG{b.draft_group_id}, {b.lineups_count} lineups)
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

                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <input
                            type="checkbox"
                            checked={useOwnership}
                            onChange={e => setUseOwnership(e.target.checked)}
                        />
                        Use ownership + dupe penalty
                    </label>
                    {!useOwnership && (
                        <div className="muted" style={{ fontSize: '0.85rem' }}>
                            Ownership is disabled for field weights and dupe penalties.
                        </div>
                    )}

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
                                        <span className="saved-build-count">DG{b.draft_group_id}</span>
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
                                        <span className="saved-build-count">DG{b.draft_group_id ?? '?'}</span>
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
                            {displayOwnershipMode === 'off' && (
                                <div style={{ marginBottom: '0.5rem' }}>
                                    <span style={{ padding: '0.2rem 0.5rem', borderRadius: '999px', background: '#0f172a', border: '1px solid #334155', color: '#f8fafc', fontSize: '0.75rem' }}>
                                        Ownership: OFF
                                    </span>
                                </div>
                            )}
                            {/* Player Exposure Panel */}
                            <PlayerExposurePanel
                                lineupResults={constrainedResults}
                                playerMap={playerMap}
                                minUniques={minUniques}
                                onMinUniquesChange={setMinUniques}
                                minUniquesPassCount={minUniquesPassCount}
                                candidateLineupCount={sortedResults.length}
                                exposureBounds={exposureBounds}
                                onExposureBoundsChange={handleExposureBoundsChange}
                                exposureCapError={exposureCapError}
                            />

                            {/* Summary Cards */}
                            <div className="sim-summary compact">
                                <div className="summary-card">
                                    <div className="card-label">Lineups</div>
                                    <div className="card-value">{activeSummary?.lineupCount ?? filteredByPlayersResults.length}</div>
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
                                        {activeSummary?.positiveEv ?? filteredByPlayersResults.filter(r => r.expected_value >= 0).length} / {activeSummary?.lineupCount ?? filteredByPlayersResults.length}
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
                                        <option value="robust_floor">Robust Floor</option>
                                        <option value="score_lcb95">Score LCB95</option>
                                        <option value="score_cvar10">Score CVaR10</option>
                                        <option value="select_score">Tail Select</option>
                                        <option value="tail_score">Tail Score</option>
                                        <option value="ucv90">UCVaR90</option>
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
                                    <label>Players:</label>
                                    <input
                                        type="text"
                                        list="contest-sim-player-options"
                                        placeholder="Add player..."
                                        value={playerSearch}
                                        onChange={e => setPlayerSearch(e.target.value)}
                                        onKeyDown={e => {
                                            if (e.key === 'Enter') {
                                                e.preventDefault()
                                                handleAddPlayerFilter()
                                            }
                                        }}
                                        style={{ width: '170px', padding: '0.25rem 0.4rem', background: '#0f172a', color: '#e2e8f0', border: '1px solid #334155', borderRadius: '4px' }}
                                    />
                                    <datalist id="contest-sim-player-options">
                                        {sortedPlayers.map(p => (
                                            <option key={p.player_id} value={p.name} />
                                        ))}
                                    </datalist>
                                    <button
                                        onClick={handleAddPlayerFilter}
                                        style={{ padding: '0.35rem 0.5rem', background: '#0f172a', border: '1px solid #334155', borderRadius: '4px', color: '#f8fafc', cursor: 'pointer' }}
                                    >
                                        Add
                                    </button>
                                    {requiredPlayerIds.length > 0 && (
                                        <button
                                            onClick={() => setRequiredPlayerIds([])}
                                            style={{ padding: '0.35rem 0.5rem', background: '#0f172a', border: '1px solid #334155', borderRadius: '4px', color: '#f8fafc', cursor: 'pointer' }}
                                        >
                                            Clear
                                        </button>
                                    )}
                                </div>

                                <div className="toolbar-group">
                                    <button onClick={selectAll} style={{ padding: '0.35rem 0.5rem', background: '#0f172a', border: '1px solid #334155', borderRadius: '4px', color: '#f8fafc', cursor: 'pointer' }}>
                                        Select All ({filteredByPlayersResults.length})
                                    </button>
                                    <button onClick={clearSelection} style={{ padding: '0.35rem 0.5rem', background: '#0f172a', border: '1px solid #334155', borderRadius: '4px', color: '#f8fafc', cursor: 'pointer' }}>
                                        Clear
                                    </button>
                                </div>

                                <div className="toolbar-group">
                                    <label>Final:</label>
                                    <input
                                        type="number"
                                        min={1}
                                        max={Math.max(1, filteredByPlayersResults.length)}
                                        value={finalSetSize}
                                        onChange={e => setFinalSetSize(Math.max(1, Number(e.target.value) || 1))}
                                        style={{ width: '70px', padding: '0.25rem 0.4rem', background: '#0f172a', color: '#e2e8f0', border: '1px solid #334155', borderRadius: '4px' }}
                                    />
                                    <label>Upside %:</label>
                                    <input
                                        type="number"
                                        min={0}
                                        max={100}
                                        value={finalUpsidePct}
                                        onChange={e => setFinalUpsidePct(Math.max(0, Math.min(100, Number(e.target.value) || 0)))}
                                        style={{ width: '60px', padding: '0.25rem 0.4rem', background: '#0f172a', color: '#e2e8f0', border: '1px solid #334155', borderRadius: '4px' }}
                                    />
                                    <button
                                        onClick={applyFinalSetToSelection}
                                        disabled={finalSetResults.length === 0}
                                        style={{ padding: '0.35rem 0.5rem', background: '#0f172a', border: '1px solid #334155', borderRadius: '4px', color: '#f8fafc', cursor: 'pointer' }}
                                    >
                                        Select Final ({finalSetResults.length})
                                    </button>
                                    <button
                                        onClick={runFinalSet}
                                        disabled={simLoading || finalSetResults.length === 0}
                                        style={{ padding: '0.35rem 0.5rem', background: '#1e3a5f', border: '1px solid #3b82f6', borderRadius: '4px', color: '#60a5fa', cursor: 'pointer' }}
                                    >
                                        Run Final
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
                                    <span style={{ color: '#64748b', fontSize: '0.78rem' }}>
                                        {filteredByPlayersResults.length}/{targetCount}
                                    </span>
                                </div>

                                <div className="toolbar-divider" />

                                <div className="toolbar-group">
                                    <button
                                        onClick={() => handleExport('view')}
                                        disabled={filteredByPlayersResults.length === 0}
                                        style={{ padding: '0.35rem 0.5rem', background: '#0f172a', border: '1px solid #334155', borderRadius: '4px', color: '#f8fafc', cursor: 'pointer', marginRight: '0.5rem' }}
                                    >
                                        Export View ({filteredByPlayersResults.length})
                                    </button>
                                    <button
                                        onClick={() => handleExport('final')}
                                        disabled={finalSetResults.length === 0}
                                        style={{ padding: '0.35rem 0.5rem', background: '#0f172a', border: '1px solid #334155', borderRadius: '4px', color: '#f8fafc', cursor: 'pointer', marginRight: '0.5rem' }}
                                    >
                                        Export Final ({finalSetResults.length})
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
                                        disabled={filteredByPlayersResults.length === 0}
                                        style={{ padding: '0.35rem 0.5rem', background: '#1e3a5f', border: '1px solid #3b82f6', borderRadius: '4px', color: '#60a5fa', cursor: 'pointer', marginLeft: '0.5rem' }}
                                    >
                                        Save Sim Lineups ({filteredByPlayersResults.length})
                                    </button>
                                    <button
                                        onClick={handleSaveFinalSet}
                                        disabled={finalSetResults.length === 0}
                                        style={{ padding: '0.35rem 0.5rem', background: '#1e3a5f', border: '1px solid #334155', borderRadius: '4px', color: '#cbd5e1', cursor: 'pointer', marginLeft: '0.5rem' }}
                                    >
                                        Save Final ({finalSetResults.length})
                                    </button>
                                </div>
                            </div>

                            <div className="contest-sim-finalset-banner">
                                <span className="muted">Set &amp; Forget</span>
                                <span>Final {finalSetResults.length}</span>
                                <span>Core {finalSetResults.filter(r => setAndForgetAuto.coreIds.has(r.lineup_id)).length}</span>
                                <span>Upside {finalSetResults.filter(r => setAndForgetAuto.upsideIds.has(r.lineup_id)).length}</span>
                                <span>Safety floor {Number.isFinite(setAndForgetAuto.safetyFloor) ? setAndForgetAuto.safetyFloor.toFixed(1) : '—'}</span>
                                {(manualIncludeFinal.size > 0 || manualExcludeFinal.size > 0) && (
                                    <span>Manual ± {manualIncludeFinal.size}/{manualExcludeFinal.size}</span>
                                )}
                                {editedFinalCount > 0 && (
                                    <span className="warning">Edited in final: {editedFinalCount} (rerun before save)</span>
                                )}
                            </div>

                            {requiredPlayerNames.length > 0 && (
                                <div className="contest-sim-player-filters">
                                    <span className="muted">Lineups must include:</span>
                                    {requiredPlayerIds.map((pid, idx) => (
                                        <button
                                            key={pid}
                                            className="contest-sim-player-chip"
                                            onClick={() => removePlayerFilter(pid)}
                                            title="Remove player filter"
                                        >
                                            {requiredPlayerNames[idx]} ×
                                        </button>
                                    ))}
                                </div>
                            )}

                            {/* Pagination */}
                            <div className="pagination-controls">
                                <div className="page-info">
                                    {hasVisibleResults
                                        ? `Showing ${((page - 1) * pageSize) + 1}-${Math.min(page * pageSize, filteredByPlayersResults.length)} of ${filteredByPlayersResults.length}`
                                        : 'Showing 0 of 0'}
                                </div>
                                <div className="page-buttons">
                                    <button
                                        className="page-btn"
                                        disabled={!hasVisibleResults || page === 1}
                                        onClick={() => setPage(p => Math.max(1, p - 1))}
                                    >
                                        Previous
                                    </button>
                                    <span style={{ margin: '0 0.5rem', color: '#94a3b8' }}>
                                        Page {hasVisibleResults ? page : 0} of {hasVisibleResults ? totalPages : 0}
                                    </span>
                                    <button
                                        className="page-btn"
                                        disabled={!hasVisibleResults || page >= totalPages}
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
                                        playerIdsOverride={getEffectivePlayerIds(result)}
                                        selected={selectedLineups.has(result.lineup_id)}
                                        onToggleSelect={() => toggleLineupSelection(result.lineup_id)}
                                        isInFinalSet={finalSetIdSet.has(result.lineup_id)}
                                        finalTag={
                                            setAndForgetAuto.coreIds.has(result.lineup_id)
                                                ? 'core'
                                                : setAndForgetAuto.upsideIds.has(result.lineup_id)
                                                    ? 'upside'
                                                    : null
                                        }
                                        isEdited={Boolean(editedLineupsById[result.lineup_id])}
                                        onToggleFinalSet={() => toggleFinalSetMembership(result.lineup_id)}
                                        onClearFinalOverride={() => clearFinalOverride(result.lineup_id)}
                                        hasFinalOverride={manualIncludeFinal.has(result.lineup_id) || manualExcludeFinal.has(result.lineup_id)}
                                        onEditLineup={() => openLineupEditor(result)}
                                        onResetEditLineup={() => resetEditedLineup(result.lineup_id)}
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

                            {editingLineupId !== null && (
                                <div className="contest-sim-editor-backdrop" onClick={closeLineupEditor}>
                                    <div className="contest-sim-editor-modal" onClick={e => e.stopPropagation()}>
                                        <h4>Edit Lineup #{editingLineupId + 1}</h4>
                                        <p className="muted">Pick players by slot. Slot legality, duplicates, and salary cap are enforced on save.</p>

                                        <div className="contest-sim-editor-summary">
                                            <span>Salary: ${editingTotalSalary.toLocaleString()} / $50,000</span>
                                            <span className={editingTotalSalary > 50000 ? 'bad' : ''}>
                                                Remaining: ${(50000 - editingTotalSalary).toLocaleString()}
                                            </span>
                                            <span>Proj: {editingTotalProjection.toFixed(1)}</span>
                                        </div>

                                        <div className="contest-sim-editor-layout">
                                            <div className="contest-sim-editor-grid">
                                                {editingLineupInputs.map((value, idx) => {
                                                    const slotName = editingSlotNames[idx] ?? `Slot ${idx + 1}`
                                                    const resolvedPlayer = editingResolvedPlayers[idx]
                                                    const hasTypedValue = value.trim().length > 0
                                                    const enforceSlotEligibility = editingSlotNames.length === DK_EDITOR_SLOTS.length
                                                    const invalidSlot = Boolean(
                                                        resolvedPlayer
                                                        && enforceSlotEligibility
                                                        && !isEligibleForSlot(resolvedPlayer, slotName),
                                                    )
                                                    const unresolved = hasTypedValue && !resolvedPlayer

                                                    return (
                                                        <div
                                                            key={`${slotName}-${idx}`}
                                                            className={`contest-sim-editor-slot ${editingActiveSlotIndex === idx ? 'active' : ''} ${invalidSlot || unresolved ? 'invalid' : ''}`}
                                                            onClick={() => setEditingActiveSlotIndex(idx)}
                                                            role="button"
                                                            tabIndex={0}
                                                            onKeyDown={e => {
                                                                if (e.key === 'Enter' || e.key === ' ') {
                                                                    e.preventDefault()
                                                                    setEditingActiveSlotIndex(idx)
                                                                }
                                                            }}
                                                        >
                                                            <div className="contest-sim-editor-slot-title">{slotName}</div>
                                                            <input
                                                                type="text"
                                                                list="contest-sim-player-options"
                                                                value={value}
                                                                onFocus={() => setEditingActiveSlotIndex(idx)}
                                                                onClick={e => e.stopPropagation()}
                                                                onChange={e => {
                                                                    const next = [...editingLineupInputs]
                                                                    next[idx] = e.target.value
                                                                    setEditingLineupInputs(next)
                                                                    setEditingActiveSlotIndex(idx)
                                                                    setEditingLineupError(null)
                                                                }}
                                                            />
                                                            {resolvedPlayer && (
                                                                <div className="contest-sim-editor-slot-meta">
                                                                    {resolvedPlayer.positions.join('/')} • {resolvedPlayer.team} • {resolvedPlayer.proj.toFixed(1)} proj • ${resolvedPlayer.salary.toLocaleString()}
                                                                </div>
                                                            )}
                                                            {invalidSlot && (
                                                                <div className="contest-sim-editor-slot-warning">
                                                                    Not eligible for {slotName}
                                                                </div>
                                                            )}
                                                            {unresolved && (
                                                                <div className="contest-sim-editor-slot-warning">
                                                                    Player not recognized
                                                                </div>
                                                            )}
                                                        </div>
                                                    )
                                                })}
                                            </div>

                                            <aside className="contest-sim-editor-eligible-panel">
                                                <div className="contest-sim-editor-eligible-header">
                                                    <strong>
                                                        Eligible for {activeEditorSlotName ?? 'Slot'}
                                                    </strong>
                                                    <input
                                                        type="text"
                                                        placeholder="Search eligible players"
                                                        value={editingEligibleSearch}
                                                        onChange={e => setEditingEligibleSearch(e.target.value)}
                                                    />
                                                </div>
                                                <div className="contest-sim-editor-eligible-list">
                                                    {eligiblePlayersForActiveSlot.slice(0, 120).map(player => {
                                                        const selectedHere = editingResolvedPlayers[editingActiveSlotIndex]?.player_id === player.player_id
                                                        return (
                                                            <button
                                                                key={player.player_id}
                                                                type="button"
                                                                className={`contest-sim-editor-eligible-item ${selectedHere ? 'selected' : ''}`}
                                                                onClick={() => setEditingSlotToPlayer(editingActiveSlotIndex, player)}
                                                            >
                                                                <span className="name">{player.name}</span>
                                                                <span className="meta">{player.positions.join('/')} • {player.team}</span>
                                                                <span className="proj">{player.proj.toFixed(1)} proj</span>
                                                                <span className="salary">${player.salary.toLocaleString()}</span>
                                                            </button>
                                                        )
                                                    })}
                                                    {eligiblePlayersForActiveSlot.length === 0 && (
                                                        <div className="contest-sim-editor-empty">No eligible players match this filter.</div>
                                                    )}
                                                </div>
                                            </aside>
                                        </div>

                                        {editingLineupError && <div className="sim-error">{editingLineupError}</div>}
                                        <div className="contest-sim-editor-actions">
                                            <button onClick={closeLineupEditor}>Cancel</button>
                                            <button onClick={saveLineupEditor}>Save Edit</button>
                                        </div>
                                    </div>
                                </div>
                            )}
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
