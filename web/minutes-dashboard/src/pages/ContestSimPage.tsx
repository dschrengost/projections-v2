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
    selectPortfolio,
    PortfolioSelectionResponse,
    PortfolioSelectionMode,
    SiteCode,
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
import NumericTextInput from '../components/NumericTextInput'
import PlayerExposurePanel, { ExposureBounds, ExposureScope } from '../components/PlayerExposurePanel'
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '../components/ui/select'
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
const FD_EDITOR_SLOTS = ['PG', 'PG', 'SG', 'SG', 'SF', 'SF', 'PF', 'PF', 'C'] as const
type ContestSimEditorSlot = (typeof DK_EDITOR_SLOTS)[number]

function getEditorSlotsForSite(site: SiteCode): readonly string[] {
    return site === 'fd' ? FD_EDITOR_SLOTS : DK_EDITOR_SLOTS
}

function getContestSimSlotFlexDegree(player: PoolPlayer): number {
    const posSet = new Set((player.positions ?? []).map(pos => pos.trim().toUpperCase()))
    let flex = 0
    if (posSet.has('PG')) flex += 1
    if (posSet.has('SG')) flex += 1
    if (posSet.has('SF')) flex += 1
    if (posSet.has('PF')) flex += 1
    if (posSet.has('C')) flex += 1
    if (posSet.has('PG') || posSet.has('SG') || posSet.has('G')) flex += 1
    if (posSet.has('SF') || posSet.has('PF') || posSet.has('F')) flex += 1
    return flex + 1
}

function getContestSimLineupSlotAssignments(
    playerIds: string[],
    playerMap: Map<string, PoolPlayer>,
): { playerId: string; slot: ContestSimEditorSlot }[] | null {
    if (playerIds.length !== DK_EDITOR_SLOTS.length) {
        return null
    }

    const uniquePlayerIds = Array.from(new Set(playerIds))
    if (uniquePlayerIds.length !== DK_EDITOR_SLOTS.length) {
        return null
    }

    const baseSlots = DK_EDITOR_SLOTS.slice(0, 5)
    const getPlayer = (id: string) => playerMap.get(id)

    const greedy = () => {
        const remaining = new Set(uniquePlayerIds)
        const assigned: { playerId: string; slot: ContestSimEditorSlot }[] = []

        for (const slot of baseSlots) {
            const candidates = Array.from(remaining).filter(id => {
                const player = getPlayer(id)
                return Boolean(player && isEligibleForSlot(player, slot))
            })
            if (candidates.length === 0) return null

            candidates.sort((a, b) => {
                const aPlayer = getPlayer(a)
                const bPlayer = getPlayer(b)
                const aFlex = aPlayer ? getContestSimSlotFlexDegree(aPlayer) : Number.MAX_VALUE
                const bFlex = bPlayer ? getContestSimSlotFlexDegree(bPlayer) : Number.MAX_VALUE
                if (aFlex !== bFlex) return aFlex - bFlex
                return a.localeCompare(b)
            })

            const picked = candidates[0]
            if (!picked) return null
            assigned.push({ playerId: picked, slot })
            remaining.delete(picked)
        }

        const pickMostFlexible = (slot: 'G' | 'F') => {
            const candidates = Array.from(remaining).filter(id => {
                const player = getPlayer(id)
                return Boolean(player && isEligibleForSlot(player, slot))
            })
            if (candidates.length === 0) return false
            candidates.sort((a, b) => {
                const aPlayer = getPlayer(a)
                const bPlayer = getPlayer(b)
                const aFlex = aPlayer ? getContestSimSlotFlexDegree(aPlayer) : -Number.MAX_VALUE
                const bFlex = bPlayer ? getContestSimSlotFlexDegree(bPlayer) : -Number.MAX_VALUE
                if (aFlex !== bFlex) return bFlex - aFlex
                return a.localeCompare(b)
            })
            const picked = candidates[0]
            if (!picked) return false
            assigned.push({ playerId: picked, slot })
            remaining.delete(picked)
            return true
        }

        if (!pickMostFlexible('G')) return null
        if (!pickMostFlexible('F')) return null
        if (remaining.size !== 1) return null

            const [utilPlayerId] = Array.from(remaining).sort()
            assigned.push({ playerId: utilPlayerId, slot: 'UTIL' })
            return assigned
    }

    const greedyAssigned = greedy()
    if (greedyAssigned) {
        return greedyAssigned
    }

    const candidatesByPlayer = new Map<string, ContestSimEditorSlot[]>()
    for (const playerId of uniquePlayerIds) {
        const player = getPlayer(playerId)
        if (!player) return null
        const eligibleSlots = DK_EDITOR_SLOTS.filter(slot => isEligibleForSlot(player, slot))
        if (eligibleSlots.length === 0) return null
        candidatesByPlayer.set(playerId, eligibleSlots)
    }

    const matchBySlot = new Map<ContestSimEditorSlot, string>()

    const assignPlayerToSlot = (playerId: string, seen: Set<ContestSimEditorSlot>): boolean => {
        for (const slot of candidatesByPlayer.get(playerId) ?? []) {
            if (seen.has(slot)) continue
            seen.add(slot)
            const occupyingPlayerId = matchBySlot.get(slot)
            if (occupyingPlayerId === undefined || assignPlayerToSlot(occupyingPlayerId, seen)) {
                matchBySlot.set(slot, playerId)
                return true
            }
        }
        return false
    }

    for (const playerId of uniquePlayerIds) {
        if (!assignPlayerToSlot(playerId, new Set())) {
            return null
        }
    }

    const ordered = [] as { playerId: string; slot: ContestSimEditorSlot }[]
    for (const slot of DK_EDITOR_SLOTS) {
        const playerId = matchBySlot.get(slot)
        if (!playerId) return null
        ordered.push({ playerId, slot })
    }

    return ordered
}

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

function orderLineupForEditorSlots(
    playerIds: string[],
    playerMap: Map<string, PoolPlayer>,
): string[] {
    if (playerIds.length !== DK_EDITOR_SLOTS.length) {
        return playerIds
    }

    const players = playerIds.map(pid => playerMap.get(pid))
    if (players.some(player => !player)) {
        return playerIds
    }

    const eligibleSlotsByPlayer = players.map((player) => {
        const p = player as PoolPlayer
        return DK_EDITOR_SLOTS
            .map((slot, slotIdx) => (isEligibleForSlot(p, slot) ? slotIdx : -1))
            .filter(slotIdx => slotIdx >= 0)
    })

    if (eligibleSlotsByPlayer.some(slots => slots.length === 0)) {
        return playerIds
    }

    const playerOrder = playerIds
        .map((_, idx) => idx)
        .sort((a, b) => {
            const flexDelta = eligibleSlotsByPlayer[a].length - eligibleSlotsByPlayer[b].length
            if (flexDelta !== 0) return flexDelta
            return a - b
        })

    const slotToPlayer = Array<number>(DK_EDITOR_SLOTS.length).fill(-1)

    const tryAssign = (playerIdx: number, seen: boolean[]): boolean => {
        for (const slotIdx of eligibleSlotsByPlayer[playerIdx]) {
            if (seen[slotIdx]) continue
            seen[slotIdx] = true
            const occupyingPlayerIdx = slotToPlayer[slotIdx]
            if (occupyingPlayerIdx === -1 || tryAssign(occupyingPlayerIdx, seen)) {
                slotToPlayer[slotIdx] = playerIdx
                return true
            }
        }
        return false
    }

    for (const playerIdx of playerOrder) {
        const seen = Array<boolean>(DK_EDITOR_SLOTS.length).fill(false)
        if (!tryAssign(playerIdx, seen)) {
            return playerIds
        }
    }

    if (slotToPlayer.some(playerIdx => playerIdx < 0)) {
        return playerIds
    }

    return slotToPlayer.map(playerIdx => playerIds[playerIdx])
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
    const [site, setSite] = useState<SiteCode>('dk')
    const [slates, setSlates] = useState<Slate[]>([])
    const [slatesLoading, setSlatesLoading] = useState(false)

    // Site constants
    const lineupSize = site === 'fd' ? 9 : 8

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
    const [useStrategyOverrides, setUseStrategyOverrides] = useState(false)

    // Simulation state
    const [lineups, setLineups] = useState<string[][]>([])
    const [simResult, setSimResult] = useState<ContestSimResponse | null>(null)
    const [simLoading, setSimLoading] = useState(false)
    const [simError, setSimError] = useState<string | null>(null)
    const [simResultTruncated, setSimResultTruncated] = useState<{ displayed: number; total: number } | null>(null)

    // Sorting and filtering
    const [sortKey, setSortKey] = useState<SortKey>('expected_value')
    const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
    const [filterPositiveEV, setFilterPositiveEV] = useState(false)
    const [maxOwnership, setMaxOwnership] = useState<number | null>(null)
    const [playerSearch, setPlayerSearch] = useState('')
    const [requiredPlayerIds, setRequiredPlayerIds] = useState<string[]>([])
    const [portfolioMode, setPortfolioMode] = useState<'browse_select' | PortfolioSelectionMode>('browse_select')
    const [finalSetSize, setFinalSetSize] = useState(40)
    const [portfolioWorldsSource, setPortfolioWorldsSource] = useState<'gtv2' | 'sim_v2'>('gtv2')
    const [portfolioEvRetention, setPortfolioEvRetention] = useState(0.99)
    const [portfolioWorldsTrainFrac, setPortfolioWorldsTrainFrac] = useState(0.8)
    const [portfolioWorldsSample, setPortfolioWorldsSample] = useState(5000)
    const [portfolioLoading, setPortfolioLoading] = useState(false)
    const [portfolioError, setPortfolioError] = useState<string | null>(null)
    const [portfolioResponse, setPortfolioResponse] = useState<PortfolioSelectionResponse | null>(null)
    const [exposureScope, setExposureScope] = useState<ExposureScope>('visible')

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

    const ownershipMode = useOwnership ? 'field_only' : 'off'
    const rankMode = 'tail_only'
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


    // Load stored site from localStorage
    useEffect(() => {
        if (typeof window === 'undefined') return
        const storedSite = window.localStorage.getItem('contestSim.site')
        if (storedSite === 'dk' || storedSite === 'fd') {
            setSite(storedSite)
        }
    }, [])

    // Persist site to localStorage
    useEffect(() => {
        if (typeof window === 'undefined') return
        window.localStorage.setItem('contestSim.site', site)
    }, [site])

    // Load slates when date or site changes
    useEffect(() => {
        const load = async () => {
            setSlatesLoading(true)
            try {
                const data = await getSlates(selectedDate, 'all', site)
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
    }, [selectedDate, site]) // eslint-disable-line react-hooks/exhaustive-deps

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

    const savedSimBuildsForSlate = useMemo(() => {
        if (!selectedSlate) {
            return savedSimBuilds
        }
        return savedSimBuilds.filter(build => build.draft_group_id == null || build.draft_group_id === selectedSlate)
    }, [savedSimBuilds, selectedSlate])

    // Load saved builds when slate changes
    useEffect(() => {
        if (!selectedSlate) {
            setSavedBuilds([])
            return
        }
        const load = async () => {
            setBuildsLoading(true)
            try {
                const builds = await getSavedBuilds(selectedDate, selectedSlate, site)
                setSavedBuilds(builds)
                setSelectedBuildId(builds[0]?.job_id ?? null)
            } catch {
                setSavedBuilds([])
            } finally {
                setBuildsLoading(false)
            }
        }
        void load()
    }, [selectedDate, selectedSlate, site])

    // Load saved contest sim builds when date/slate/site changes
    useEffect(() => {
        const load = async () => {
            setSimBuildsLoading(true)
            try {
                const builds = await getSavedSimBuilds(selectedDate, undefined, site)
                setSavedSimBuilds(builds)
                const scopedBuilds = selectedSlate
                    ? builds.filter(b => b.draft_group_id == null || b.draft_group_id === selectedSlate)
                    : builds
                const latestRun = scopedBuilds.find(b => b.kind === 'run')?.build_id ?? null
                const latestLineup = scopedBuilds.find(b => b.kind === 'lineups' || b.kind === 'portfolio')?.build_id ?? null
                if (latestRun) {
                    setSelectedSimBuildId(latestRun)
                    setSelectedSimLineupId(null)
                } else if (latestLineup) {
                    setSelectedSimBuildId(null)
                    setSelectedSimLineupId(latestLineup)
                } else {
                    setSelectedSimBuildId(null)
                    setSelectedSimLineupId(null)
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
    }, [selectedDate, selectedSlate, site])

    // Load cached field libraries when date/slate/site changes
    useEffect(() => {
        if (!selectedSlate) {
            setFieldLibraries([])
            return
        }
        const load = async () => {
            setFieldLibrariesLoading(true)
            setFieldLibraryError(null)
            try {
                const libs = await listFieldLibraries(selectedDate, selectedSlate, site)
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
    }, [selectedDate, selectedSlate, fieldLibraryVersion, site])

    // Load player pool for name resolution
    useEffect(() => {
        if (!selectedSlate) {
            setPool([])
            return
        }
        const load = async () => {
            try {
                const data = await getPlayerPool(
                    selectedDate,
                    selectedSlate,
                    undefined,
                    site,
                    { useStrategyOverrides },
                )
                setPool(data)
            } catch {
                setPool([])
            }
        }
        void load()
    }, [selectedDate, selectedSlate, useStrategyOverrides, site])

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
                    if (selectedSlate && build.draft_group_id && build.draft_group_id !== selectedSlate) {
                        return
                    }
                    setPortfolioResponse(null)
                    setSimResult({
                        results: build.results,
                        config: build.config as unknown as ContestSimResponse['config'],
                        stats: build.stats as unknown as ContestSimResponse['stats'],
                        build_id: build.build_id,
                    })
                    setLineups(build.lineups ?? [])
                    setSimResultTruncated(
                        build.results_truncated
                            ? { displayed: build.results.length, total: build.lineups_count }
                            : null
                    )
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

    // Browse-select path: explicit filters, visible sort, optional Top N shortlist.
    const sortedResults = useMemo(() => {
        let results = [...resultsWithOwnership]

        if (filterPositiveEV) {
            results = results.filter(r => r.expected_value >= 0)
        }
        if (maxOwnership !== null) {
            results = results.filter(r => r.total_own <= maxOwnership)
        }
        if (requiredPlayerIds.length > 0) {
            results = results.filter(r =>
                requiredPlayerIds.every(pid => getEffectivePlayerIds(r).includes(pid)),
            )
        }

        results.sort((a, b) => {
            const aVal = a[sortKey as keyof typeof a]
            const bVal = b[sortKey as keyof typeof b]
            const aNum = typeof aVal === 'number' && Number.isFinite(aVal) ? aVal : null
            const bNum = typeof bVal === 'number' && Number.isFinite(bVal) ? bVal : null
            if (aNum === null && bNum === null) return a.lineup_id - b.lineup_id
            if (aNum === null) return 1
            if (bNum === null) return -1
            return sortDir === 'asc' ? aNum - bNum : bNum - aNum
        })

        return results
    }, [resultsWithOwnership, filterPositiveEV, maxOwnership, requiredPlayerIds, getEffectivePlayerIds, sortKey, sortDir])

    const targetCount = topN ?? sortedResults.length
    const filteredByPlayersResults = useMemo(() => {
        if (topN === null) {
            return sortedResults
        }
        return sortedResults.slice(0, topN)
    }, [sortedResults, topN])

    const minUniquesPassCount = minUniques === 0 ? filteredByPlayersResults.length : 0
    const exposureCapError = portfolioMode === 'browse_select'
        ? 'Exposure caps and min-uniques apply only in optimizer modes.'
        : null

    const poolByLineupId = useMemo(() => {
        return new Map(filteredByPlayersResults.map(r => [r.lineup_id, r] as const))
    }, [filteredByPlayersResults])

    const portfolioBaseLineupIds = useMemo(() => {
        return portfolioResponse?.selected_lineup_ids ?? []
    }, [portfolioResponse])

    const finalSetLineupIds = useMemo(() => {
        const poolIds = new Set(filteredByPlayersResults.map(r => r.lineup_id))
        if (portfolioMode === 'browse_select') {
            return []
        }
        const included = Array.from(manualIncludeFinal).filter(id => poolIds.has(id) && !manualExcludeFinal.has(id))
        const desiredSize = Math.min(
            filteredByPlayersResults.length,
            Math.max(portfolioBaseLineupIds.length, included.length),
        )
        const ordered: number[] = [...included]
        for (const id of portfolioBaseLineupIds) {
            if (ordered.length >= desiredSize) break
            if (manualExcludeFinal.has(id)) continue
            if (!ordered.includes(id)) {
                ordered.push(id)
            }
        }
        return ordered
    }, [filteredByPlayersResults, manualIncludeFinal, manualExcludeFinal, portfolioBaseLineupIds, portfolioMode])

    const finalSetIdSet = useMemo(() => new Set(finalSetLineupIds), [finalSetLineupIds])
    const finalSetResults = useMemo(() => {
        return finalSetLineupIds
            .map(id => poolByLineupId.get(id))
            .filter((r): r is LineupResultWithOwnership => Boolean(r))
    }, [finalSetLineupIds, poolByLineupId])
    const selectedResults = useMemo(() => {
        return filteredByPlayersResults.filter(r => selectedLineups.has(r.lineup_id))
    }, [filteredByPlayersResults, selectedLineups])
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
        setPortfolioResponse(null)
        setPortfolioError(null)
        setManualIncludeFinal(new Set())
        setManualExcludeFinal(new Set())
    }, [
        filterPositiveEV,
        maxOwnership,
        topN,
        sortKey,
        sortDir,
        requiredPlayerIds,
    ])

    useEffect(() => {
        setExposureScope(prev => {
            if (prev === 'portfolio' && finalSetResults.length === 0) {
                return selectedResults.length > 0 ? 'selected' : 'visible'
            }
            if (prev === 'selected' && selectedResults.length === 0) {
                return finalSetResults.length > 0 ? 'portfolio' : 'visible'
            }
            return prev
        })
    }, [finalSetResults.length, selectedResults.length])

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
        setPortfolioResponse(null)
        try {
            const result = await runContestSim({
                game_date: selectedDate,
                site,
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
                use_strategy_overrides: useStrategyOverrides,
            })
            setSimResult(result)
            setSimResultTruncated(null)
            const builds = await getSavedSimBuilds(selectedDate, undefined, site)
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
        useStrategyOverrides,
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
                site,
                draft_group_id: selectedSlate,
                version: fieldLibraryVersion,
                k: fieldLibraryK,
                candidate_pool_size: fieldCandidatePoolSize,
                rebuild: fieldLibraryRebuild,
                rebuild_candidates: fieldLibraryRebuildCandidates,
                ownership_mode: ownershipMode,
            })
            const libs = await listFieldLibraries(selectedDate, selectedSlate, site)
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
                if (build.kind === 'lineups' || build.kind === 'portfolio') {
                    if (selectedSlate && build.draft_group_id && build.draft_group_id !== selectedSlate) {
                        return
                    }
                    setLineups(build.lineups ?? [])
                    if (build.results && build.config && build.stats) {
                        setSimResult({
                            results: build.results,
                            config: build.config as unknown as ContestSimResponse['config'],
                            stats: build.stats as unknown as ContestSimResponse['stats'],
                            build_id: build.build_id,
                        })
                        const requestMeta = (build.request ?? {}) as Record<string, unknown>
                        const selectionMode = requestMeta.selection_mode
                        if (build.kind === 'portfolio' && Array.isArray(build.results)) {
                            const diagnostics = (
                                typeof build.stats?.debug === 'object' && build.stats?.debug && 'selection' in build.stats.debug
                                    ? (build.stats.debug as Record<string, unknown>).selection
                                    : requestMeta.selection_diagnostics
                            ) as Record<string, unknown> | undefined
                            const warnings = (
                                Array.isArray(requestMeta.warnings)
                                    ? requestMeta.warnings
                                    : []
                            ) as string[]
                            setPortfolioResponse({
                                mode: (typeof selectionMode === 'string' ? selectionMode : 'decorrelated_ev') as PortfolioSelectionMode,
                                source_build_id: typeof requestMeta.source_build_id === 'string' ? requestMeta.source_build_id : build.build_id,
                                candidate_count: build.results.length,
                                filtered_candidate_count: build.results.length,
                                selected_lineup_ids: build.results.map(r => r.lineup_id),
                                selected_results: build.results,
                                selected_lineups: build.lineups ?? [],
                                diagnostics: diagnostics ?? {},
                                warnings,
                            })
                            if (typeof selectionMode === 'string' && selectionMode !== 'browse_select') {
                                setPortfolioMode(selectionMode as PortfolioSelectionMode)
                            }
                            setExposureScope('portfolio')
                        } else {
                            setPortfolioResponse(null)
                        }
                    } else {
                        setSimResult(null)
                        setPortfolioResponse(null)
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
                {
                    site,
                    kind: 'lineups',
                    sourceBuildId: simResult?.build_id ?? null,
                    selectionMode: 'browse_select',
                    selectionConfig: {
                        sortKey,
                        sortDir,
                        filterPositiveEV,
                        maxOwnership,
                        requiredPlayerIds,
                        topN,
                    },
                },
            )
            const builds = await getSavedSimBuilds(selectedDate, undefined, site)
            setSavedSimBuilds(builds)
            setSelectedSimBuildId(null)
            setSelectedSimLineupId(saved.build_id)
        } catch (err) {
            alert('Failed to save sim lineups: ' + (err as Error).message)
        }
    }

    const handleSaveSelectedLineups = async () => {
        if (!selectedSlate) return
        if (selectedLineups.size === 0) return
        const selectedResults = filteredByPlayersResults.filter(r => selectedLineups.has(r.lineup_id))
        const editedCount = selectedResults.filter(r => Boolean(editedLineupsById[r.lineup_id])).length
        if (editedCount > 0) {
            alert(`There are ${editedCount} edited selected lineups. Run sim on the edited set first, then save.`)
            return
        }
        const defaultName = `Selected lineups (${selectedResults.length})`
        const name = prompt('Save selected lineups as:', defaultName)?.trim()
        if (!name) return
        try {
            const lineupsToSave = selectedResults.map(getEffectivePlayerIds)
            const resultIds = new Set(selectedResults.map(r => r.lineup_id))
            const resultsToSave = simResult?.results.filter(r => resultIds.has(r.lineup_id)) ?? null
            const saved = await saveSimLineups(
                selectedDate,
                selectedSlate,
                name,
                lineupsToSave,
                resultsToSave,
                simResult?.config ?? null,
                simResult?.stats ?? null,
                {
                    site,
                    kind: 'lineups',
                    sourceBuildId: simResult?.build_id ?? null,
                    selectionMode: 'browse_select',
                    selectionConfig: {
                        selectedLineupIds: Array.from(selectedLineups),
                        sortKey,
                        sortDir,
                        filterPositiveEV,
                        maxOwnership,
                        requiredPlayerIds,
                        topN,
                    },
                },
            )
            const builds = await getSavedSimBuilds(selectedDate, undefined, site)
            setSavedSimBuilds(builds)
            setSelectedSimBuildId(null)
            setSelectedSimLineupId(saved.build_id)
        } catch (err) {
            alert('Failed to save selected lineups: ' + (err as Error).message)
        }
    }

    const handleBuildPortfolio = async () => {
        if (portfolioMode === 'browse_select') {
            setPortfolioError('Switch to an optimizer mode to build a portfolio.')
            return
        }
        if (!simResult?.build_id) {
            setPortfolioError('Load or run a saved contest-sim build first.')
            return
        }
        if (filteredByPlayersResults.length === 0) {
            setPortfolioError('No visible candidates remain after filters.')
            return
        }
        const editedCount = filteredByPlayersResults.filter(r => Boolean(editedLineupsById[r.lineup_id])).length
        if (editedCount > 0) {
            setPortfolioError(`There are ${editedCount} edited lineups in view. Re-run sim on the edited set first.`)
            return
        }
        setPortfolioLoading(true)
        setPortfolioError(null)
        try {
            const response = await selectPortfolio({
                game_date: selectedDate,
                site,
                draft_group_id: selectedSlate ?? undefined,
                source_build_id: simResult.build_id,
                mode: portfolioMode,
                worlds_source: portfolioWorldsSource,
                sort_key: sortKey,
                sort_dir: sortDir,
                portfolio_size: Math.max(1, Math.min(finalSetSize, filteredByPlayersResults.length)),
                ev_retention: portfolioEvRetention,
                worlds_sample: portfolioWorldsSample,
                worlds_train_frac: portfolioMode === 'decorrelated_ev' ? portfolioWorldsTrainFrac : null,
                min_uniques: minUniques,
                max_total_own: maxOwnership,
                filter_positive_ev: filterPositiveEV,
                candidate_lineup_ids: filteredByPlayersResults.map(r => r.lineup_id),
                seed_lineup_ids: portfolioResponse?.selected_lineup_ids ?? undefined,
                exposure_bounds: Object.fromEntries(Array.from(exposureBounds.entries())),
            })
            setPortfolioResponse(response)
            setManualIncludeFinal(new Set())
            setManualExcludeFinal(new Set())
            setExposureScope('portfolio')
        } catch (err) {
            setPortfolioError((err as Error).message)
        } finally {
            setPortfolioLoading(false)
        }
    }

    const handleSaveFinalSet = async () => {
        if (!selectedSlate) return
        if (finalSetResults.length === 0) return
        if (portfolioMode === 'browse_select' || !portfolioResponse) {
            alert('Build a portfolio first, or use Save Selected / Save View.')
            return
        }
        if (editedFinalCount > 0) {
            alert(`There are ${editedFinalCount} edited lineups in the portfolio. Run sim on the edited set first, then save.`)
            return
        }
        const defaultName = `Portfolio (${finalSetResults.length})`
        const name = prompt('Save portfolio as:', defaultName)?.trim()
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
                {
                    site,
                    kind: 'portfolio',
                    sourceBuildId: portfolioResponse.source_build_id,
                    selectionMode: portfolioResponse.mode,
                    selectionConfig: {
                        sortKey,
                        sortDir,
                        filterPositiveEV,
                        maxOwnership,
                        requiredPlayerIds,
                        topN,
                        portfolioSize: finalSetSize,
                        minUniques,
                        exposureBounds: Object.fromEntries(Array.from(exposureBounds.entries())),
                        worldsSource: portfolioWorldsSource,
                        evRetention: portfolioEvRetention,
                        worldsTrainFrac: portfolioWorldsTrainFrac,
                        worldsSample: portfolioWorldsSample,
                        manualIncludeIds: Array.from(manualIncludeFinal),
                        manualExcludeIds: Array.from(manualExcludeFinal),
                    },
                    selectionDiagnostics: portfolioResponse.diagnostics,
                    warnings: portfolioResponse.warnings,
                },
            )
            const builds = await getSavedSimBuilds(selectedDate, undefined, site)
            setSavedSimBuilds(builds)
            setSelectedSimBuildId(null)
            setSelectedSimLineupId(saved.build_id)
        } catch (err) {
            alert('Failed to save portfolio: ' + (err as Error).message)
        }
    }

    const handleDeleteSimBuild = async (buildId: string) => {
        if (!confirm('Delete this saved sim build?')) return
        try {
            await deleteSavedSimBuild(selectedDate, buildId, site)
            const builds = await getSavedSimBuilds(selectedDate, undefined, site)
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

    const runSelectedLineups = async () => {
        const selectedResults = filteredByPlayersResults.filter(r => selectedLineups.has(r.lineup_id))
        if (selectedResults.length === 0) {
            setSimError('No selected lineups available.')
            return
        }
        const lineupsToRun = selectedResults.map(getEffectivePlayerIds)
        await runSimWithLineups(lineupsToRun)
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
            lineupsToExport = filteredByPlayersResults
        }

        const selectedPlayerIds = lineupsToExport.map(getEffectivePlayerIds)
        const filename = `contest_sim_${selectedDate}_${lineupsToExport.length}lineups.csv`

        try {
            const blob = await exportCustomLineupsCSV(
                selectedDate,
                selectedSlate,
                selectedPlayerIds,
                site,
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
        await runSelectedLineups()
    }

    const openLineupEditor = (lineup: LineupResultWithOwnership) => {
        const current = orderLineupForEditorSlots(getEffectivePlayerIds(lineup), playerMap)
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
                        Site
                        <Select value={site} onValueChange={v => setSite(v === 'fd' ? 'fd' : 'dk')}>
                            <SelectTrigger className="contest-sim-select w-full">
                                <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="dk">DraftKings</SelectItem>
                                <SelectItem value="fd">FanDuel</SelectItem>
                            </SelectContent>
                        </Select>
                    </label>
                    <label>
                        Slate
                        <Select
                            value={selectedSlate?.toString()}
                            onValueChange={(value) => setSelectedSlate(value ? Number(value) : null)}
                            disabled={slatesLoading}
                        >
                            <SelectTrigger className="contest-sim-select w-full">
                                <SelectValue placeholder={slatesLoading ? 'Loading…' : 'Select slate'} />
                            </SelectTrigger>
                            <SelectContent>
                                {slateOptions.length === 0 ? (
                                    <SelectItem value="_none" disabled>
                                        No slates
                                    </SelectItem>
                                ) : (
                                    slateOptions.map(s => (
                                        <SelectItem key={s.draft_group_id} value={s.draft_group_id.toString()}>
                                            {formatSlateLabel(s)} (DG{s.draft_group_id})
                                        </SelectItem>
                                    ))
                                )}
                            </SelectContent>
                        </Select>
                    </label>
                </div>
            </header>

            <div className="sim-layout">
                {/* Configuration Sidebar */}
                <aside className="sim-sidebar">
                    <h3>Simulation Settings</h3>

                    <label>
                        Saved Build
                        <Select
                            value={selectedBuildId ?? undefined}
                            onValueChange={value => setSelectedBuildId(value || null)}
                            disabled={buildsLoading}
                        >
                            <SelectTrigger className="contest-sim-select w-full">
                                <SelectValue placeholder={buildsLoading ? 'Loading…' : 'Select saved build'} />
                            </SelectTrigger>
                            <SelectContent>
                                {savedBuilds.length === 0 ? (
                                    <SelectItem value="_none" disabled>
                                        No builds
                                    </SelectItem>
                                ) : (
                                    savedBuilds.map(b => (
                                        <SelectItem key={b.job_id} value={b.job_id}>
                                            {b.job_id.slice(0, 8)} (DG{b.draft_group_id}, {b.lineups_count} lineups)
                                        </SelectItem>
                                    ))
                                )}
                            </SelectContent>
                        </Select>
                    </label>

                    <div className="lineup-count">
                        {lineups.length > 0 && <span>{lineups.length} lineups loaded</span>}
                    </div>

                    <hr />

                    <label>
                        Payout Archetype
                        <Select value={archetype} onValueChange={setArchetype}>
                            <SelectTrigger className="contest-sim-select w-full">
                                <SelectValue placeholder="Select archetype" />
                            </SelectTrigger>
                            <SelectContent>
                                {config?.payout_archetypes.length ? (
                                    config.payout_archetypes.map(a => (
                                        <SelectItem key={a.key} value={a.key}>
                                            {a.label} ({(a.first_place_pct * 100).toFixed(0)}% to 1st)
                                        </SelectItem>
                                    ))
                                ) : (
                                    <>
                                        <SelectItem value="top_heavy">Top Heavy</SelectItem>
                                        <SelectItem value="medium">Medium</SelectItem>
                                        <SelectItem value="flat">Flat</SelectItem>
                                    </>
                                )}
                            </SelectContent>
                        </Select>
                    </label>

                    <label>
                        Field Size
                        <Select value={fieldSizeBucket} onValueChange={setFieldSizeBucket}>
                            <SelectTrigger className="contest-sim-select w-full">
                                <SelectValue placeholder="Field size" />
                            </SelectTrigger>
                            <SelectContent>
                                {config?.field_sizes.length ? (
                                    config.field_sizes.map(f => (
                                        <SelectItem key={f.key} value={f.key}>
                                            {f.label}
                                        </SelectItem>
                                    ))
                                ) : (
                                    <>
                                        <SelectItem value="small">Small (1-10K)</SelectItem>
                                        <SelectItem value="medium">Medium (10-50K)</SelectItem>
                                        <SelectItem value="massive">Massive (50K+)</SelectItem>
                                    </>
                                )}
                            </SelectContent>
                        </Select>
                    </label>

                    <label>
                        Field Model
                        <Select value={fieldMode} onValueChange={(value) => setFieldMode(value as typeof fieldMode)}>
                            <SelectTrigger className="contest-sim-select w-full">
                                <SelectValue placeholder="Field model" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="self_play">Self-play (your lineups as field)</SelectItem>
                                <SelectItem value="generated_field">Representative field (QuickBuild)</SelectItem>
                            </SelectContent>
                        </Select>
                    </label>

                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <input
                            type="checkbox"
                            checked={useStrategyOverrides}
                            onChange={e => setUseStrategyOverrides(e.target.checked)}
                        />
                        Use persistent strategy overrides
                    </label>

                    {fieldMode === 'generated_field' && (
                        <>
                            <label>
                                Field Library Version
                                <Select value={fieldLibraryVersion} onValueChange={setFieldLibraryVersion}>
                                    <SelectTrigger className="contest-sim-select w-full">
                                        <SelectValue placeholder="Field library version" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        {fieldLibraries.length === 0 ? (
                                            <SelectItem value="v0">v0</SelectItem>
                                        ) : (
                                            fieldLibraries.map(l => (
                                                <SelectItem key={l.version} value={l.version}>
                                                    {l.version} ({l.selected_k} lineups)
                                                </SelectItem>
                                            ))
                                        )}
                                    </SelectContent>
                                </Select>
                            </label>

                            <label>
                                Field K (unique lineups)
                                <NumericTextInput
                                    value={fieldLibraryK}
                                    onChangeValue={n => setFieldLibraryK(Math.max(100, Math.min(5000, Math.round(n ?? 2500))))}
                                    min={100}
                                    max={5000}
                                    step={100}
                                    integerOnly
                                />
                            </label>

                            <label>
                                Candidate Pool Size
                                <NumericTextInput
                                    value={fieldCandidatePoolSize}
                                    onChangeValue={n => setFieldCandidatePoolSize(Math.max(5000, Math.min(100000, Math.round(n ?? 40000))))}
                                    min={5000}
                                    max={100000}
                                    step={5000}
                                    integerOnly
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
                        <NumericTextInput
                            value={entryFee}
                            onChangeValue={n => setEntryFee(Math.max(0.25, Math.min(1000, n ?? 3.0)))}
                            min={0.25}
                            max={1000}
                            step={0.25}
                            inputMode="decimal"
                        />
                    </label>

                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <input
                            type="checkbox"
                            checked={useOwnership}
                            onChange={e => setUseOwnership(e.target.checked)}
                        />
                        Use ownership-informed field
                    </label>
                    {useOwnership ? (
                        <div className="muted" style={{ fontSize: '0.85rem' }}>
                            Ownership informs generated-field weights. Dupe penalty is disabled.
                        </div>
                    ) : (
                        <div className="muted" style={{ fontSize: '0.85rem' }}>
                            Ownership is disabled for generated-field weights. Dupe penalty remains disabled.
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
                            {savedSimBuildsForSlate.filter(b => b.kind === 'run').length === 0 && (
                                <span className="muted">No sim runs yet.</span>
                            )}
                            {savedSimBuildsForSlate.filter(b => b.kind === 'run').map(b => (
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
                            <h3>Saved Lineup Sets</h3>
                        </div>
                        <div className="saved-builds-list">
                            {savedSimBuildsForSlate.filter(b => b.kind === 'lineups').length === 0 && (
                                <span className="muted">No saved lineups yet.</span>
                            )}
                            {savedSimBuildsForSlate.filter(b => b.kind === 'lineups').map(b => (
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

                    <section className="saved-builds-section">
                        <div className="saved-builds-header">
                            <h3>Saved Portfolios</h3>
                        </div>
                        <div className="saved-builds-list">
                            {savedSimBuildsForSlate.filter(b => b.kind === 'portfolio').length === 0 && (
                                <span className="muted">No saved portfolios yet.</span>
                            )}
                            {savedSimBuildsForSlate.filter(b => b.kind === 'portfolio').map(b => (
                                <div key={b.build_id} className={`saved-build-card ${selectedSimLineupId === b.build_id ? 'selected' : ''}`}>
                                    <div className="saved-build-info">
                                        <span className="saved-build-count">DG{b.draft_group_id ?? '?'}</span>
                                        <span className="saved-build-count">{b.lineups_count} lineups</span>
                                        <span className="saved-build-time">{b.name ?? b.build_id.slice(0, 8)}</span>
                                        {typeof b.stats?.debug === 'object' && b.stats?.debug && 'selection_mode' in b.stats.debug && (
                                            <span className="saved-build-stats">{String((b.stats.debug as Record<string, unknown>).selection_mode)}</span>
                                        )}
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
                                visibleLineupResults={filteredByPlayersResults}
                                portfolioLineupResults={finalSetResults}
                                selectedLineupResults={selectedResults}
                                scope={exposureScope}
                                onScopeChange={setExposureScope}
                                playerMap={playerMap}
                                minUniques={minUniques}
                                onMinUniquesChange={setMinUniques}
                                minUniquesPassCount={minUniquesPassCount}
                                candidateLineupCount={sortedResults.length}
                                exposureBounds={exposureBounds}
                                onExposureBoundsChange={handleExposureBoundsChange}
                                exposureCapError={exposureCapError}
                            />

                            {/* Truncation notice */}
                            {simResultTruncated && (
                                <div className="sim-truncation-notice">
                                    Showing {simResultTruncated.displayed.toLocaleString()} of {simResultTruncated.total.toLocaleString()} lineups — portfolio optimize and top-N run on the full pool server-side.
                                </div>
                            )}

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
                                    <Select value={sortKey} onValueChange={value => setSortKey(value as SortKey)}>
                                        <SelectTrigger className="contest-sim-select w-44">
                                            <SelectValue placeholder="Sort metric" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="expected_value">EV</SelectItem>
                                            <SelectItem value="roi">ROI</SelectItem>
                                            <SelectItem value="robust_floor">Robust Floor</SelectItem>
                                            <SelectItem value="score_lcb95">Score LCB95</SelectItem>
                                            <SelectItem value="score_cvar10">Score CVaR10</SelectItem>
                                            <SelectItem value="select_score">Tail Select</SelectItem>
                                            <SelectItem value="tail_score">Tail Score</SelectItem>
                                            <SelectItem value="ucv90">UCVaR90</SelectItem>
                                            <SelectItem value="win_rate">Win%</SelectItem>
                                            <SelectItem value="top_1pct_rate">Top 1%</SelectItem>
                                            <SelectItem value="cash_rate">Cash%</SelectItem>
                                            <SelectItem value="p90">Ceiling (p90)</SelectItem>
                                            <SelectItem value="mean">Mean</SelectItem>
                                            <SelectItem value="total_own">Total Own%</SelectItem>
                                            <SelectItem value="lineup_id">Lineup #</SelectItem>
                                        </SelectContent>
                                    </Select>
                                    <button
                                        className="contest-sim-btn contest-sim-btn-ghost"
                                        onClick={() => setSortDir(d => d === 'asc' ? 'desc' : 'asc')}
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
                                    <Select
                                        value={maxOwnership == null ? 'all' : maxOwnership.toString()}
                                        onValueChange={(value) => setMaxOwnership(value === 'all' ? null : Number(value))}
                                    >
                                        <SelectTrigger className="contest-sim-select w-36">
                                            <SelectValue placeholder="Any" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="all">All</SelectItem>
                                            <SelectItem value="50">≤50%</SelectItem>
                                            <SelectItem value="75">≤75%</SelectItem>
                                            <SelectItem value="100">≤100%</SelectItem>
                                            <SelectItem value="150">≤150%</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>

                                <div className="toolbar-divider" />

                                <div className="toolbar-group">
                                    <label>Players:</label>
                                    <input
                                        className="contest-sim-input contest-sim-input-md"
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
                                    />
                                    <datalist id="contest-sim-player-options">
                                        {sortedPlayers.map(p => (
                                            <option key={p.player_id} value={p.name} />
                                        ))}
                                    </datalist>
                                    <button
                                        className="contest-sim-btn contest-sim-btn-ghost"
                                        onClick={handleAddPlayerFilter}
                                    >
                                        Add
                                    </button>
                                    {requiredPlayerIds.length > 0 && (
                                        <button
                                            className="contest-sim-btn contest-sim-btn-ghost"
                                            onClick={() => setRequiredPlayerIds([])}
                                        >
                                            Clear
                                        </button>
                                    )}
                                </div>

                                <div className="toolbar-group">
                                    <button className="contest-sim-btn contest-sim-btn-ghost" onClick={selectAll}>
                                        Select Visible ({filteredByPlayersResults.length})
                                    </button>
                                    <button className="contest-sim-btn contest-sim-btn-ghost" onClick={clearSelection}>
                                        Clear
                                    </button>
                                    <button
                                        className="contest-sim-btn contest-sim-btn-primary"
                                        onClick={runSelectedLineups}
                                        disabled={simLoading || selectedLineups.size === 0}
                                    >
                                        Run Selected ({selectedLineups.size})
                                    </button>
                                </div>

                                <div className="toolbar-group">
                                    <label>Mode:</label>
                                    <Select
                                        value={portfolioMode}
                                        onValueChange={(value) => setPortfolioMode(value as typeof portfolioMode)}
                                    >
                                        <SelectTrigger className="contest-sim-select w-56">
                                            <SelectValue placeholder="Mode" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="browse_select">Browse / Select</SelectItem>
                                            <SelectItem value="greedy_constraints">Greedy Constraints</SelectItem>
                                            <SelectItem value="decorrelated_ev">Decorrelated EV</SelectItem>
                                        </SelectContent>
                                    </Select>
                                    <label>Portfolio:</label>
                                    <NumericTextInput
                                        value={finalSetSize}
                                        onChangeValue={n => setFinalSetSize(Math.max(1, Math.min(Math.max(1, filteredByPlayersResults.length), Math.round(n ?? 1))))}
                                        min={1}
                                        max={Math.max(1, filteredByPlayersResults.length)}
                                        integerOnly
                                        className="contest-sim-input contest-sim-input-sm"
                                    />
                                    <button
                                        className="contest-sim-btn contest-sim-btn-primary"
                                        onClick={handleBuildPortfolio}
                                        disabled={portfolioMode === 'browse_select' || portfolioLoading || filteredByPlayersResults.length === 0}
                                    >
                                        {portfolioLoading ? 'Building...' : `Build Portfolio (${Math.min(finalSetSize, filteredByPlayersResults.length)})`}
                                    </button>
                                    <button
                                        className="contest-sim-btn contest-sim-btn-ghost"
                                        onClick={applyFinalSetToSelection}
                                        disabled={finalSetResults.length === 0}
                                    >
                                        Select Portfolio ({finalSetResults.length})
                                    </button>
                                </div>

                                {portfolioMode !== 'browse_select' && (
                                    <div className="toolbar-group">
                                        <label>Worlds:</label>
                                        <Select
                                            value={portfolioWorldsSource}
                                            onValueChange={(value) => setPortfolioWorldsSource(value as typeof portfolioWorldsSource)}
                                        >
                                            <SelectTrigger className="contest-sim-select w-32">
                                                <SelectValue placeholder="World source" />
                                            </SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="gtv2">gtv2</SelectItem>
                                                <SelectItem value="sim_v2">sim_v2</SelectItem>
                                            </SelectContent>
                                        </Select>
                                        {portfolioMode === 'decorrelated_ev' && (
                                            <>
                                                <label>EV Ret:</label>
                                                <NumericTextInput
                                                    value={portfolioEvRetention}
                                                    onChangeValue={n => setPortfolioEvRetention(Math.max(0.5, Math.min(1, n ?? 0.99)))}
                                                    min={0.5}
                                                    max={1}
                                                    step={0.01}
                                                    inputMode="decimal"
                                                    className="contest-sim-input contest-sim-input-sm"
                                                />
                                                <label>Train:</label>
                                                <NumericTextInput
                                                    value={portfolioWorldsTrainFrac}
                                                    onChangeValue={n => setPortfolioWorldsTrainFrac(Math.max(0.5, Math.min(0.95, n ?? 0.8)))}
                                                    min={0.5}
                                                    max={0.95}
                                                    step={0.05}
                                                    inputMode="decimal"
                                                    className="contest-sim-input contest-sim-input-sm"
                                                />
                                                <label>Sample:</label>
                                                <NumericTextInput
                                                    value={portfolioWorldsSample}
                                                    onChangeValue={n => setPortfolioWorldsSample(Math.max(100, Math.round(n ?? 5000)))}
                                                    min={100}
                                                    step={100}
                                                    integerOnly
                                                    className="contest-sim-input contest-sim-input-md"
                                                />
                                            </>
                                        )}
                                    </div>
                                )}

                                <div className="toolbar-group">
                                    <label>Top N:</label>
                                    <NumericTextInput
                                        value={topN}
                                        onChangeValue={n => setTopN(n === null ? null : Math.max(1, Math.round(n)))}
                                        min={1}
                                        placeholder="All"
                                        allowNull
                                        integerOnly
                                        className="contest-sim-input contest-sim-input-md"
                                    />
                                    <span className="toolbar-metric">
                                        {filteredByPlayersResults.length}/{targetCount}
                                    </span>
                                </div>

                                <div className="toolbar-divider" />

                                <div className="toolbar-group">
                                    <button
                                        className="contest-sim-btn contest-sim-btn-ghost"
                                        onClick={() => handleExport('view')}
                                        disabled={filteredByPlayersResults.length === 0}
                                    >
                                        Export View ({filteredByPlayersResults.length})
                                    </button>
                                    <button
                                        className="contest-sim-btn contest-sim-btn-ghost"
                                        onClick={() => handleExport('final')}
                                        disabled={finalSetResults.length === 0}
                                    >
                                        Export Portfolio ({finalSetResults.length})
                                    </button>
                                    <button
                                        className="export-btn"
                                        onClick={() => handleExport('selected')}
                                        disabled={selectedLineups.size === 0}
                                    >
                                        Export Selected ({selectedLineups.size})
                                    </button>
                                    <button
                                        className="contest-sim-btn contest-sim-btn-primary"
                                        onClick={handleSaveSimLineups}
                                        disabled={filteredByPlayersResults.length === 0}
                                    >
                                        Save View ({filteredByPlayersResults.length})
                                    </button>
                                    <button
                                        className="contest-sim-btn contest-sim-btn-secondary"
                                        onClick={handleSaveSelectedLineups}
                                        disabled={selectedLineups.size === 0}
                                    >
                                        Save Selected ({selectedLineups.size})
                                    </button>
                                    <button
                                        className="contest-sim-btn contest-sim-btn-secondary"
                                        onClick={handleSaveFinalSet}
                                        disabled={finalSetResults.length === 0}
                                    >
                                        Save Portfolio ({finalSetResults.length})
                                    </button>
                                </div>
                            </div>

                            <div className="contest-sim-finalset-banner">
                                <span className="muted">
                                    {portfolioMode === 'browse_select' ? 'Browse / Select' : 'Portfolio'}
                                </span>
                                <span>Visible {filteredByPlayersResults.length}</span>
                                <span>Selected {selectedLineups.size}</span>
                                {portfolioMode !== 'browse_select' && (
                                    <span>Portfolio {finalSetResults.length}</span>
                                )}
                                {portfolioResponse && typeof portfolioResponse.diagnostics.ev_selected === 'number' && (
                                    <span>EV {Number(portfolioResponse.diagnostics.ev_selected).toFixed(2)}</span>
                                )}
                                {portfolioResponse && typeof portfolioResponse.diagnostics.risk_var_total_reduction_pct === 'number' && (
                                    <span>Risk ↓ {Number(portfolioResponse.diagnostics.risk_var_total_reduction_pct).toFixed(1)}%</span>
                                )}
                                {portfolioResponse && typeof portfolioResponse.diagnostics.world_selection_policy === 'string' && (
                                    <span>{String(portfolioResponse.diagnostics.world_selection_policy)}</span>
                                )}
                                {(manualIncludeFinal.size > 0 || manualExcludeFinal.size > 0) && portfolioMode !== 'browse_select' && (
                                    <span>Manual ± {manualIncludeFinal.size}/{manualExcludeFinal.size}</span>
                                )}
                                {editedFinalCount > 0 && (
                                    <span className="warning">Edited in portfolio: {editedFinalCount} (rerun before save)</span>
                                )}
                            </div>

                            {(portfolioError || (portfolioResponse?.warnings?.length ?? 0) > 0) && (
                                <div className="contest-sim-finalset-banner">
                                    {portfolioError && <span className="warning">{portfolioError}</span>}
                                    {(portfolioResponse?.warnings ?? []).map((warning) => (
                                        <span key={warning} className="warning">{warning}</span>
                                    ))}
                                </div>
                            )}

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

                                    <Select
                                        value={pageSize.toString()}
                                        onValueChange={(value) => setPageSize(Number(value))}
                                    >
                                        <SelectTrigger className="contest-sim-select w-36" style={{ marginLeft: '1rem' }}>
                                            <SelectValue placeholder="Page size" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="20">20 / page</SelectItem>
                                            <SelectItem value="50">50 / page</SelectItem>
                                            <SelectItem value="100">100 / page</SelectItem>
                                            <SelectItem value="500">500 / page</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                            </div>

                            {/* Cards Grid - Use paginatedResults */}
                            <div className="lineup-cards-grid">
                                {paginatedResults.map(result => {
                                    const effectivePlayerIds = getEffectivePlayerIds(result)
                                    const slotAssignments = getContestSimLineupSlotAssignments(effectivePlayerIds, playerMap)
                                    const orderedPlayerIds = slotAssignments
                                        ? slotAssignments.map(({ playerId }) => playerId)
                                        : effectivePlayerIds

                                    return (
                                        <LineupCard
                                            key={result.lineup_id}
                                            result={result}
                                            players={playerMap}
                                            playerIdsOverride={orderedPlayerIds}
                                            slotAssignments={slotAssignments ?? undefined}
                                            selected={selectedLineups.has(result.lineup_id)}
                                            onToggleSelect={() => toggleLineupSelection(result.lineup_id)}
                                            isInFinalSet={finalSetIdSet.has(result.lineup_id)}
                                            finalTag={null}
                                            isEdited={Boolean(editedLineupsById[result.lineup_id])}
                                            onToggleFinalSet={portfolioMode === 'browse_select' ? undefined : () => toggleFinalSetMembership(result.lineup_id)}
                                            onClearFinalOverride={portfolioMode === 'browse_select' ? undefined : () => clearFinalOverride(result.lineup_id)}
                                            hasFinalOverride={portfolioMode === 'browse_select' ? false : (manualIncludeFinal.has(result.lineup_id) || manualExcludeFinal.has(result.lineup_id))}
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
                                    )
                                })}
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
