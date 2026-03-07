import { useCallback, useEffect, useMemo, useState } from 'react'
import {
    getSlates,
    getPlayerPool,
    getStrategyOverrides,
    startBuild,
    getBuildStatus,
    getBuildLineups,
    exportLineupsCSV,
    exportCustomLineupsCSV,
    getSavedBuilds,
    loadSavedBuild,
    deleteSavedBuild,
    saveCustomBuild,
    saveStrategyOverrides,
    clearStrategyOverrides,
    Slate,
    PoolPlayer,
    PlayerStrategyOverride,
    JobStatus,
    LineupRow,
    QuickBuildRequest,
    SavedBuild,
} from '../api/optimizer'
import { formatSalary } from '../utils'
import { useSlateDateAndSlate } from '../hooks/useSlateDate'
import { formatSlateLabel } from '../utils/slateFormat'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select'

const DK_SLOT_ORDER = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'] as const
const DK_SLOT_PRIORITY = DK_SLOT_ORDER.reduce<Record<string, number>>((acc, slot, idx) => {
    acc[slot] = idx
    return acc
}, {} as Record<string, number>)
const DK_BASE_SLOTS = ['PG', 'SG', 'SF', 'PF', 'C'] as const
const DK_ALL_SLOTS = [...DK_SLOT_ORDER] as const

type SortKey =
    | 'name'
    | 'team'
    | 'salary'
    | 'proj'
    | 'own_proj'
    | 'value'
    | 'min'
    | 'fppm'
    | 'optimal_pct'
    | 'ceiling_leverage'
    | 'boom_pct'
    | 'bust_pct'

type LineupGroup = {
    id: string
    name: string
    lineup_ids: number[]
    created_at: string
}

const parseMatchupTeams = (matchup?: string | null): string[] => {
    if (!matchup) return []
    const normalized = matchup
        .replace(/\s+/g, ' ')
        .trim()
        .replace(/\s+at\s+/gi, '@')
        .replace(/\s+vs\.?\s+/gi, '@')
        .replace(/\s+v\.?\s+/gi, '@')
        .replace(/\s*-\s*/g, '@')
        .replace(/\//g, '@')
    return normalized
        .split('@')
        .map(part => part.trim().toUpperCase())
        .filter(Boolean)
}

const getNormalizedPositions = (positions: string[] = []): string[] =>
    positions
        .flatMap(pos => pos.toUpperCase().split(/[^A-Z]/))
        .filter(Boolean)

const getLineupSlotOrder = (positions: string[] = []): number => {
    const normalized = getNormalizedPositions(positions)
    for (const slot of DK_SLOT_ORDER) {
        if (normalized.includes(slot)) return DK_SLOT_PRIORITY[slot]
    }
    return DK_SLOT_ORDER.length
}

const getLineupSlotFlexDegree = (positions: string[] = []) => {
    const normalized = new Set(getNormalizedPositions(positions))
    let deg = 0
    for (const slot of DK_BASE_SLOTS) {
        if (normalized.has(slot)) deg += 1
    }
    if (normalized.has('PG') || normalized.has('SG')) deg += 1 // G
    if (normalized.has('SF') || normalized.has('PF')) deg += 1 // F
    return deg + 1 // UTIL
}

const isEligibleForDkSlot = (positions: string[] = [], slot: (typeof DK_ALL_SLOTS)[number]) => {
    const normalized = new Set(getNormalizedPositions(positions))
    if (slot === 'PG' || slot === 'SG' || slot === 'SF' || slot === 'PF' || slot === 'C') {
        return normalized.has(slot)
    }
    if (slot === 'G') return normalized.has('PG') || normalized.has('SG')
    if (slot === 'F') return normalized.has('SF') || normalized.has('PF')
    if (slot === 'UTIL') return true
    return false
}

const getDisplaySlotByAssignment = (
    playerIds: string[],
    map: Map<string, PoolPlayer>,
): { playerId: string; slot: (typeof DK_SLOT_ORDER)[number] }[] | null => {
    if (playerIds.length !== 8) return null
    const unique = Array.from(new Set(playerIds))
    if (unique.length !== 8) return null

    const canUse = (id: string) => {
        const p = map.get(id)
        return p ? p.positions && p.positions.length > 0 : false
    }
    if (!unique.every(canUse)) return null

    const greedy = () => {
        const remaining = new Set(unique)
        const assigned: { playerId: string; slot: typeof DK_SLOT_ORDER[number] }[] = []

        for (const slot of DK_BASE_SLOTS) {
            const candidates = Array.from(remaining).filter(id => {
                const p = map.get(id)
                return p ? isEligibleForDkSlot(p.positions, slot) : false
            })
            if (candidates.length === 0) return null
            candidates.sort((a, b) => {
                const aPos = map.get(a)?.positions
                const bPos = map.get(b)?.positions
                const aFlex = getLineupSlotFlexDegree(aPos)
                const bFlex = getLineupSlotFlexDegree(bPos)
                if (aFlex !== bFlex) return aFlex - bFlex
                return a.localeCompare(b)
            })
            const pick = candidates[0]
            assigned.push({ playerId: pick, slot })
            remaining.delete(pick)
        }

        const pickMostFlexible = (slot: 'G' | 'F') => {
            const isPred = slot === 'G'
                ? (pos: string[]) => (getNormalizedPositions(pos).includes('PG') || getNormalizedPositions(pos).includes('SG'))
                : (pos: string[]) => (getNormalizedPositions(pos).includes('SF') || getNormalizedPositions(pos).includes('PF'))
            const candidates = Array.from(remaining).filter(id => {
                const p = map.get(id)
                return p ? isPred(p.positions) : false
            })
            if (candidates.length === 0) return false
            candidates.sort((a, b) => {
                const aPos = map.get(a)?.positions
                const bPos = map.get(b)?.positions
                const aFlex = getLineupSlotFlexDegree(aPos)
                const bFlex = getLineupSlotFlexDegree(bPos)
                if (aFlex !== bFlex) return bFlex - aFlex
                return a.localeCompare(b)
            })
            const pick = candidates[0]
            assigned.push({ playerId: pick, slot })
            remaining.delete(pick)
            return true
        }

        if (!pickMostFlexible('G')) return null
        if (!pickMostFlexible('F')) return null
        if (remaining.size === 0) return null
        const [util] = Array.from(remaining).sort()
        assigned.push({ playerId: util, slot: 'UTIL' })
        return assigned
    }

    const greedyAssigned = greedy()
    if (greedyAssigned) return greedyAssigned

    const adj = new Map<string, (typeof DK_SLOT_ORDER)[number][]>()
    for (const playerId of unique) {
        const p = map.get(playerId)
        if (!p) return null
        const eligible: (typeof DK_SLOT_ORDER)[number][] = DK_ALL_SLOTS.filter(slot => isEligibleForDkSlot(p.positions, slot))
        if (eligible.length === 0) return null
        adj.set(playerId, eligible)
    }

    const matchR = new Map<(typeof DK_SLOT_ORDER)[number], string>()

    const dfs = (playerId: string, seen: Set<(typeof DK_SLOT_ORDER)[number]>): boolean => {
        const options = adj.get(playerId) || []
        for (const slot of options) {
            if (seen.has(slot)) continue
            seen.add(slot)
            const assignedPid = matchR.get(slot)
            if (!assignedPid || dfs(assignedPid, seen)) {
                matchR.set(slot, playerId)
                return true
            }
        }
        return false
    }

    for (const playerId of unique) {
        if (!dfs(playerId, new Set())) return null
    }

    const assigned = [] as { playerId: string; slot: (typeof DK_SLOT_ORDER)[number] }[]
    for (const slot of DK_SLOT_ORDER) {
        const playerId = matchR.get(slot)
        if (!playerId) return null
        assigned.push({ playerId, slot })
    }
    return assigned
}

export default function OptimizerPage() {
    // Date and slate selection (persisted in URL)
    const [selectedDate, setSelectedDate, selectedSlate, setSelectedSlate] = useSlateDateAndSlate()
    const [slates, setSlates] = useState<Slate[]>([])
    const [slatesLoading, setSlatesLoading] = useState(false)
    const [slatesError, setSlatesError] = useState<string | null>(null)

    // Player pool
    const [pool, setPool] = useState<PoolPlayer[]>([])
    const [poolLoading, setPoolLoading] = useState(false)
    const [poolError, setPoolError] = useState<string | null>(null)
    const [useStrategyOverrides, setUseStrategyOverrides] = useState(false)
    const [strategyOverrides, setStrategyOverrides] = useState<Map<string, PlayerStrategyOverride>>(new Map())
    const [savedStrategyOverrides, setSavedStrategyOverrides] = useState<Map<string, PlayerStrategyOverride>>(new Map())
    const [overrideRevision, setOverrideRevision] = useState<number | null>(null)
    const [overrideLoading, setOverrideLoading] = useState(false)
    const [overrideSaving, setOverrideSaving] = useState(false)
    const [overrideError, setOverrideError] = useState<string | null>(null)

    // Lock/ban players
    const [lockedIds, setLockedIds] = useState<Set<string>>(new Set())
    const [bannedIds, setBannedIds] = useState<Set<string>>(new Set())

    // Filter and sort
    const [filter, setFilter] = useState('')
    const [sortKey, setSortKey] = useState<SortKey>('proj')
    const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

    // Build config
    const [maxPool, setMaxPool] = useState(5000)
    const [builds] = useState(22)
    const [minUniq, setMinUniq] = useState(1)
    const [maxExposurePct, setMaxExposurePct] = useState(0)
    const [nearDupJaccard, setNearDupJaccard] = useState(0)
    const [globalTeamLimit, setGlobalTeamLimit] = useState(4)
    const [minSalary, setMinSalary] = useState<number | null>(null)
    const [maxSalary, setMaxSalary] = useState<number | null>(50000)
    const [minProj, setMinProj] = useState<number | null>(null)
    const [maxOffoptimalPct, setMaxOffoptimalPct] = useState(0)
    const [randomnessPct, setRandomnessPct] = useState(0)
    const [lateSwapEnabled, setLateSwapEnabled] = useState(false)
    const [worldSampleEnabled, setWorldSampleEnabled] = useState(false)

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
    const [excludedTeams, setExcludedTeams] = useState<Set<string>>(new Set())
    const [showGameFilterDrawer, setShowGameFilterDrawer] = useState(false)
    const [settingsDrawerOpen, setSettingsDrawerOpen] = useState(false)

    // Get current slate's games
    const currentSlateGames = useMemo(() => {
        const slate = slates.find(s => s.draft_group_id === selectedSlate)
        return slate?.games ?? []
    }, [slates, selectedSlate])

    const currentSlateGameFilters = useMemo(() => {
        return currentSlateGames.map(game => ({
            ...game,
            teams: parseMatchupTeams(game.matchup),
        }))
    }, [currentSlateGames])

    const currentSlateTeams = useMemo(() => {
        const teamSet = new Set<string>()
        for (const game of currentSlateGameFilters) {
            for (const team of game.teams) {
                if (team) teamSet.add(team.toUpperCase())
            }
        }
        if (teamSet.size === 0) {
            for (const player of pool) {
                if (player.team) teamSet.add(player.team.toUpperCase())
            }
        }
        return Array.from(teamSet).sort()
    }, [currentSlateGameFilters, pool])

    const fadedTeamBanIds = useMemo(() => {
        const ids = new Set<string>()
        if (excludedTeams.size === 0) return ids
        for (const player of pool) {
            const team = player.team?.toUpperCase()
            if (!team || !excludedTeams.has(team) || lockedIds.has(player.player_id)) continue
            ids.add(player.player_id)
        }
        return ids
    }, [excludedTeams, lockedIds, pool])

    const combinedBanCount = useMemo(() => {
        const ids = new Set(bannedIds)
        for (const playerId of fadedTeamBanIds) {
            ids.add(playerId)
        }
        return ids.size
    }, [bannedIds, fadedTeamBanIds])

    const clearFilters = () => {
        setExcludedGames(new Set())
        setExcludedTeams(new Set())
    }

    const toggleExcludedGame = (matchup: string) => {
        setExcludedGames(prev => {
            const next = new Set(prev)
            if (next.has(matchup)) {
                next.delete(matchup)
            } else {
                next.add(matchup)
            }
            return next
        })
    }

    const toggleExcludedTeam = (team: string) => {
        setExcludedTeams(prev => {
            const next = new Set(prev)
            if (next.has(team)) {
                next.delete(team)
            } else {
                next.add(team)
            }
            return next
        })
    }

    // Check if stddev is available in pool (needed for randomness feature)
    const hasStddev = useMemo(() =>
        pool.some(p => p.stddev != null && p.stddev > 0), [pool])

    useEffect(() => {
        if (!showGameFilterDrawer) return
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                setShowGameFilterDrawer(false)
            }
        }
        window.addEventListener('keydown', onKeyDown)
        return () => window.removeEventListener('keydown', onKeyDown)
    }, [showGameFilterDrawer])

    useEffect(() => {
        if (typeof window === 'undefined') return
        const stored = window.localStorage.getItem('optimizer.useStrategyOverrides')
        if (stored != null) {
            setUseStrategyOverrides(stored === 'true')
        }
    }, [])

    useEffect(() => {
        if (typeof window === 'undefined') return
        window.localStorage.setItem('optimizer.useStrategyOverrides', String(useStrategyOverrides))
    }, [useStrategyOverrides])

    useEffect(() => {
        setExcludedGames(new Set())
        setExcludedTeams(new Set())
        setShowGameFilterDrawer(false)
    }, [selectedDate, selectedSlate])
    // Load slates when date changes
    useEffect(() => {
        const loadSlates = async () => {
            setSlatesLoading(true)
            setSlatesError(null)
            try {
                const data = await getSlates(selectedDate)
                setSlates(data)
                const urlSlateExists = selectedSlate && data.some(s => s.draft_group_id === selectedSlate)
                if (!urlSlateExists) {
                    const mainSlate = data.find(s => s.slate_type !== 'showdown')
                    setSelectedSlate(mainSlate?.draft_group_id ?? data[0]?.draft_group_id ?? null)
                }
            } catch (err) {
                setSlatesError((err as Error).message)
                setSlates([])
                setSelectedSlate(null)
            } finally {
                setSlatesLoading(false)
            }
        }
        void loadSlates()
    }, [selectedDate]) // eslint-disable-line react-hooks/exhaustive-deps

    const loadPool = useCallback(async () => {
        if (!selectedSlate) {
            setPool([])
            return
        }
        setPoolLoading(true)
        setPoolError(null)
        try {
            const data = await getPlayerPool(
                selectedDate,
                selectedSlate,
                undefined,
                { useStrategyOverrides },
            )
            setPool(data)
            setLockedIds(new Set())
            setBannedIds(new Set())
        } catch (err) {
            setPoolError((err as Error).message)
            setPool([])
        } finally {
            setPoolLoading(false)
        }
    }, [selectedDate, selectedSlate, useStrategyOverrides])

    // Load player pool when slate changes
    useEffect(() => {
        void loadPool()
    }, [loadPool])

    const loadOverrides = useCallback(async () => {
        if (!selectedSlate) {
            setStrategyOverrides(new Map())
            setSavedStrategyOverrides(new Map())
            setOverrideRevision(null)
            return
        }
        setOverrideLoading(true)
        setOverrideError(null)
        try {
            const data = await getStrategyOverrides(selectedDate, selectedSlate)
            const next = new Map<string, PlayerStrategyOverride>()
            data.overrides.forEach((override) => {
                next.set(override.player_id, override)
            })
            setStrategyOverrides(new Map(next))
            setSavedStrategyOverrides(new Map(next))
            setOverrideRevision(data.client_revision)
        } catch (err) {
            setOverrideError((err as Error).message)
            setStrategyOverrides(new Map())
            setSavedStrategyOverrides(new Map())
            setOverrideRevision(null)
        } finally {
            setOverrideLoading(false)
        }
    }, [selectedDate, selectedSlate])

    useEffect(() => {
        void loadOverrides()
    }, [loadOverrides])

    const overrideSignature = (override?: PlayerStrategyOverride) => JSON.stringify({
        minutes_delta: override?.minutes_delta ?? null,
        fpts_delta: override?.fpts_delta ?? null,
        minutes: override?.minutes ?? null,
        fpts: override?.fpts ?? null,
    })

    const unsavedOverrideCount = useMemo(() => {
        const ids = new Set<string>([
            ...Array.from(strategyOverrides.keys()),
            ...Array.from(savedStrategyOverrides.keys()),
        ])
        let count = 0
        ids.forEach((playerId) => {
            if (overrideSignature(strategyOverrides.get(playerId)) !== overrideSignature(savedStrategyOverrides.get(playerId))) {
                count += 1
            }
        })
        return count
    }, [savedStrategyOverrides, strategyOverrides])

    const activeOverrideCount = useMemo(() => savedStrategyOverrides.size, [savedStrategyOverrides])

    const updateStrategyOverride = (
        playerId: string,
        updates: Partial<PlayerStrategyOverride>,
    ) => {
        setStrategyOverrides((prev) => {
            const next = new Map(prev)
            const current = next.get(playerId) ?? { player_id: playerId }
            const updated: PlayerStrategyOverride = {
                ...current,
                ...updates,
                player_id: playerId,
            }
            const hasAnyValue =
                updated.minutes_delta != null ||
                updated.fpts_delta != null ||
                updated.minutes != null ||
                updated.fpts != null
            if (hasAnyValue) {
                next.set(playerId, updated)
            } else {
                next.delete(playerId)
            }
            return next
        })
    }

    const saveOverrideChanges = useCallback(async () => {
        if (!selectedSlate) return
        setOverrideSaving(true)
        setOverrideError(null)
        try {
            const payload = Array.from(strategyOverrides.values()).map((override) => ({
                player_id: override.player_id,
                minutes_delta: override.minutes_delta ?? null,
                fpts_delta: override.fpts_delta ?? null,
                minutes: override.minutes ?? null,
                fpts: override.fpts ?? null,
            }))
            const response = await saveStrategyOverrides(
                selectedDate,
                selectedSlate,
                payload,
                overrideRevision ?? undefined,
            )
            const next = new Map<string, PlayerStrategyOverride>()
            response.overrides.forEach((override) => {
                next.set(override.player_id, override)
            })
            setStrategyOverrides(new Map(next))
            setSavedStrategyOverrides(new Map(next))
            setOverrideRevision(response.client_revision)
            await loadPool()
        } catch (err) {
            setOverrideError((err as Error).message)
        } finally {
            setOverrideSaving(false)
        }
    }, [loadPool, overrideRevision, selectedDate, selectedSlate, strategyOverrides])

    const discardOverrideChanges = () => {
        setStrategyOverrides(new Map(savedStrategyOverrides))
        setOverrideError(null)
    }

    const resetAllOverrides = useCallback(async () => {
        if (!selectedSlate) return
        setOverrideSaving(true)
        setOverrideError(null)
        try {
            await clearStrategyOverrides(selectedDate, selectedSlate)
            await loadOverrides()
            await loadPool()
        } catch (err) {
            setOverrideError((err as Error).message)
        } finally {
            setOverrideSaving(false)
        }
    }, [loadOverrides, loadPool, selectedDate, selectedSlate])

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

    const handleLoadSavedBuild = async (jobId: string) => {
        try {
            const build = await loadSavedBuild(selectedDate, jobId)
            if (build.lineups) {
                setLineups(build.lineups)
                setCurrentJob(null)
            }
        } catch (err) {
            setBuildError((err as Error).message)
        }
    }

    const handleDeleteSavedBuild = async (jobId: string) => {
        if (!confirm('Delete this saved build?')) return
        try {
            await deleteSavedBuild(selectedDate, jobId)
            await refreshSavedBuilds()
            setSelectedBuildIds(prev => {
                const next = new Set(prev)
                next.delete(jobId)
                return next
            })
        } catch (err) {
            alert('Failed to delete: ' + (err as Error).message)
        }
    }

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

    const handleJoinBuilds = async () => {
        if (selectedBuildIds.size < 2 || !selectedSlate) return
        const buildName = prompt('Name for merged build:', `Merged (${selectedBuildIds.size} builds)`)
        if (!buildName) return
        try {
            const allLineups: LineupRow[] = []
            for (const jobId of selectedBuildIds) {
                const build = await loadSavedBuild(selectedDate, jobId)
                if (build.lineups) {
                    allLineups.push(...build.lineups)
                }
            }
            const seen = new Set<string>()
            const unique: LineupRow[] = []
            for (const lu of allLineups) {
                const key = [...lu.player_ids].sort().join(',')
                if (!seen.has(key)) {
                    seen.add(key)
                    unique.push({ ...lu, lineup_id: unique.length })
                }
            }
            await saveCustomBuild(selectedDate, selectedSlate, unique, buildName)
            setLineups(unique)
            setCurrentJob(null)
            setSelectedBuildIds(new Set())
            await refreshSavedBuilds()
            alert(`Merged ${selectedBuildIds.size} builds into "${buildName}": ${unique.length} unique lineups`)
        } catch (err) {
            alert('Failed to join builds: ' + (err as Error).message)
        }
    }

    // Filtered and sorted pool
    const filteredPool = useMemo(() => {
        let filtered = pool.slice()
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
                case 'min':
                    left = useStrategyOverrides ? a.effective_minutes ?? a.model_minutes ?? 0 : a.model_minutes ?? 0
                    right = useStrategyOverrides ? b.effective_minutes ?? b.model_minutes ?? 0 : b.model_minutes ?? 0
                    break
                case 'fppm': left = a.fppm ?? 0; right = b.fppm ?? 0; break
                case 'optimal_pct': left = a.optimal_pct ?? 0; right = b.optimal_pct ?? 0; break
                case 'ceiling_leverage': left = a.ceiling_leverage ?? 0; right = b.ceiling_leverage ?? 0; break
                case 'boom_pct': left = a.boom_pct ?? 0; right = b.boom_pct ?? 0; break
                case 'bust_pct': left = a.bust_pct ?? 0; right = b.bust_pct ?? 0; break
                default: left = a.proj; right = b.proj
            }
            if (typeof left === 'number' && typeof right === 'number') {
                return sortDir === 'asc' ? left - right : right - left
            }
            return sortDir === 'asc' ? String(left).localeCompare(String(right)) : String(right).localeCompare(String(left))
        })
        return filtered
    }, [pool, filter, sortKey, sortDir, minProj, useStrategyOverrides])

    // Toggle lock/ban
    const toggleLock = (id: string) => {
        setLockedIds(prev => {
            const next = new Set(prev)
            if (next.has(id)) next.delete(id)
            else next.add(id)
            return next
        })
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
        setLockedIds(prev => {
            const next = new Set(prev)
            next.delete(id)
            return next
        })
    }

    // Start build
    const handleStartBuild = async () => {
        if (!selectedSlate) return
        if (useStrategyOverrides && unsavedOverrideCount > 0) {
            setBuildError('Save or discard strategy override changes before building.')
            return
        }
        setBuildError(null)
        setLineups([])
        try {
            const combinedBanIds = new Set(bannedIds)
            for (const playerId of fadedTeamBanIds) {
                combinedBanIds.add(playerId)
            }
            const request: QuickBuildRequest = {
                date: selectedDate,
                draft_group_id: selectedSlate,
                site: 'dk',
                max_pool: maxPool,
                builds,
                per_build: Math.ceil(maxPool / builds) + 500,
                min_uniq: minUniq,
                max_exposure_pct: maxExposurePct > 0 ? maxExposurePct : null,
                near_dup_jaccard: nearDupJaccard > 0 ? nearDupJaccard : undefined,
                global_team_limit: globalTeamLimit,
                min_salary: minSalary,
                max_salary: maxSalary,
                lock_ids: Array.from(lockedIds),
                ban_ids: Array.from(combinedBanIds),
                max_offoptimal_pct: maxOffoptimalPct > 0 ? maxOffoptimalPct / 100 : undefined,
                exclude_games: Array.from(excludedGames),
                enum_enable: maxPool >= 5000,
                randomness_pct: randomnessPct > 0 && hasStddev ? randomnessPct : undefined,
                use_strategy_overrides: useStrategyOverrides,
                late_swap_enabled: lateSwapEnabled,
                world_sample_enabled: worldSampleEnabled,
            }
            const job = await startBuild(request)
            setCurrentJob(job)
        } catch (err) {
            setBuildError((err as Error).message)
        }
    }

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

    const selectAllVisible = () => {
        const visibleIds = filteredLineups.slice(0, showCount).map(lu => lu.lineup_id)
        setSelectedLineupIds(new Set(visibleIds))
    }

    const selectAllFiltered = () => {
        const filteredIds = filteredLineups.map(lu => lu.lineup_id)
        setSelectedLineupIds(new Set(filteredIds))
    }

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

    const filteredLineups = useMemo(() => {
        let result = lineups.slice()
        if (lineupFilter.trim()) {
            const text = lineupFilter.trim().toLowerCase()
            result = result.filter(lu =>
                lu.player_ids.some(id => {
                    const p = playerMap.get(id)
                    return p && p.name.toLowerCase().includes(text)
                })
            )
        }
        result = result.filter(lu => {
            const proj = lu.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.proj ?? 0), 0)
            const own = lu.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.own_proj ?? 0), 0)
            const p90 = lu.p90 ?? lu.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.p90 ?? 0), 0)
            if (minLineupProj != null && proj < minLineupProj) return false
            if (maxLineupOwn != null && own > maxLineupOwn) return false
            if (minLineupP90 != null && p90 < minLineupP90) return false
            return true
        })
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

    const showStrategyColumns = useMemo(
        () => useStrategyOverrides || strategyOverrides.size > 0 || savedStrategyOverrides.size > 0,
        [savedStrategyOverrides.size, strategyOverrides.size, useStrategyOverrides],
    )

    const getCurrentOverride = (playerId: string) => strategyOverrides.get(playerId)

    const getDisplayMinutes = (player: PoolPlayer) =>
        useStrategyOverrides ? player.effective_minutes ?? player.model_minutes : player.model_minutes

    const getDisplayStddev = (player: PoolPlayer) =>
        useStrategyOverrides ? player.effective_stddev ?? player.stddev : player.stddev

    const getDisplayP90 = (player: PoolPlayer) =>
        useStrategyOverrides ? player.effective_p90 ?? player.p90 : player.p90

    const formatDelta = (val: number | undefined | null) =>
        val != null ? `${val > 0 ? '+' : ''}${val.toFixed(1)}` : '—'

    const toggleSort = (key: SortKey) => {
        if (sortKey === key) {
            setSortDir(prev => prev === 'asc' ? 'desc' : 'asc')
        } else {
            setSortKey(key)
            setSortDir(key === 'name' || key === 'team' ? 'asc' : 'desc')
        }
    }

    const formatProj = (val: number | undefined | null) =>
        val != null ? val.toFixed(1) : '—'

    const formatOwn = (val: number | undefined | null) =>
        val != null ? val.toFixed(1) + '%' : '—'

    const formatPct = (val: number | undefined | null) =>
        val != null ? `${val.toFixed(1)}%` : '—'

    const formatSigned = (val: number | undefined | null) =>
        val != null ? `${val >= 0 ? '+' : ''}${val.toFixed(1)}` : '—'

    const formatMin = (val: number | undefined | null) =>
        val != null ? val.toFixed(1) : '—'

    const formatFppm = (val: number | undefined | null) =>
        val != null ? val.toFixed(2) : '—'

    const formatValue = (p: PoolPlayer) =>
        (p.proj / (p.salary / 1000)).toFixed(2)

    const SortIcon = ({ col }: { col: SortKey }) => {
        if (sortKey !== col) return <span className="ml-1 opacity-30">↕</span>
        return <span className="ml-1">{sortDir === 'asc' ? '▲' : '▼'}</span>
    }

    return (
        <div className="optimizer-page flex flex-col gap-5">
            {/* Header */}
            <div className="flex items-end justify-between gap-4 pb-4 border-b border-[hsl(var(--border))]">
                <div>
                    <h1 className="text-2xl font-bold tracking-tight">Lineup Optimizer</h1>
                    <p className="text-sm text-[hsl(var(--muted-foreground))] mt-0.5">QuickBuild lineup pool generation</p>
                </div>
                <div className="flex items-end gap-3">
                    <div className="flex flex-col gap-1.5">
                        <span className="text-xs font-medium text-[hsl(var(--muted-foreground))] uppercase tracking-wider">Date</span>
                        <Input
                            type="date"
                            value={selectedDate}
                            onChange={e => setSelectedDate(e.target.value)}
                            className="w-[145px]"
                        />
                    </div>
                    <div className="flex flex-col gap-1.5">
                        <span className="text-xs font-medium text-[hsl(var(--muted-foreground))] uppercase tracking-wider">Slate</span>
                        <Select
                            value={selectedSlate?.toString() ?? ''}
                            onValueChange={v => setSelectedSlate(Number(v) || null)}
                            disabled={slatesLoading}
                        >
                            <SelectTrigger className="w-[360px]">
                                <SelectValue placeholder={slatesLoading ? 'Loading slates…' : 'Select a slate…'} />
                            </SelectTrigger>
                            <SelectContent>
                                {slates.length === 0 && (
                                    <SelectItem value="_none" disabled>No slates available</SelectItem>
                                )}
                                {slates.map(s => (
                                    <SelectItem key={s.draft_group_id} value={s.draft_group_id.toString()}>
                                        {formatSlateLabel(s)}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>
                    <Button
                        variant="outline"
                        className={`settings-drawer-toggle ${settingsDrawerOpen ? 'active' : ''}`}
                        onClick={() => setSettingsDrawerOpen(prev => !prev)}
                    >
                        {settingsDrawerOpen ? 'Hide settings' : 'Build settings'}
                    </Button>
                </div>
            </div>

            {(slatesError || poolError) && (
                <div className="rounded-md bg-red-950/40 border border-red-800/50 px-4 py-2 text-sm text-red-400">
                    {slatesError || poolError}
                </div>
            )}

            <div className="optimizer-layout">
                {/* Settings Sidebar */}
                <aside className={`optimizer-sidebar space-y-4 ${settingsDrawerOpen ? 'open' : ''}`}>
                    <div className="optimizer-sidebar-header">
                        <h3 className="text-sm font-semibold text-[hsl(var(--muted-foreground))] uppercase tracking-wider">Build Settings</h3>
                        <button
                            type="button"
                            className="settings-drawer-close"
                            onClick={() => setSettingsDrawerOpen(false)}
                            aria-label="Close settings drawer"
                        >
                            ×
                        </button>
                    </div>

                    <div className="space-y-1">
                        <label className="text-xs font-medium text-[hsl(var(--muted-foreground))]">Max Lineups</label>
                        <Input
                            type="number"
                            value={maxPool}
                            onChange={e => setMaxPool(Number(e.target.value))}
                            min={100}
                            max={100000}
                            step={500}
                        />
                    </div>

                    <div className="space-y-1">
                        <label className="text-xs font-medium text-[hsl(var(--muted-foreground))]">Min Uniques</label>
                        <Input
                            type="number"
                            value={minUniq}
                            onChange={e => setMinUniq(Number(e.target.value))}
                            min={0}
                            max={8}
                        />
                    </div>

                    <div className="space-y-1.5">
                        <div className="flex justify-between">
                            <label className="text-xs font-medium text-[hsl(var(--muted-foreground))]">Max Exposure</label>
                            <span className="text-xs text-[hsl(var(--foreground))]">{maxExposurePct}%</span>
                        </div>
                        <input
                            type="range"
                            min={0}
                            max={100}
                            step={5}
                            value={maxExposurePct}
                            onChange={e => setMaxExposurePct(Number(e.target.value))}
                            className="w-full accent-[hsl(var(--primary))]"
                        />
                        <p className="text-xs text-[hsl(var(--muted-foreground))]">0% disables exposure cap</p>
                    </div>

                    <div className="space-y-1.5">
                        <div className="flex justify-between">
                            <label className="text-xs font-medium text-[hsl(var(--muted-foreground))]">Near-Dup Jaccard</label>
                            <span className="text-xs text-[hsl(var(--foreground))]">{nearDupJaccard.toFixed(2)}</span>
                        </div>
                        <input
                            type="range"
                            min={0}
                            max={1}
                            step={0.05}
                            value={nearDupJaccard}
                            onChange={e => setNearDupJaccard(Number(e.target.value))}
                            className="w-full accent-[hsl(var(--primary))]"
                        />
                        <p className="text-xs text-[hsl(var(--muted-foreground))]">0.75 ≈ 7/8 overlap for 8-man DK; 0 disables</p>
                    </div>

                    <div className="space-y-1">
                        <label className="text-xs font-medium text-[hsl(var(--muted-foreground))]">Team Limit</label>
                        <Input
                            type="number"
                            value={globalTeamLimit}
                            onChange={e => setGlobalTeamLimit(Number(e.target.value))}
                            min={1}
                            max={8}
                        />
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                        <div className="space-y-1">
                            <label className="text-xs font-medium text-[hsl(var(--muted-foreground))]">Min Salary</label>
                            <Input
                                type="number"
                                value={minSalary ?? ''}
                                onChange={e => setMinSalary(e.target.value ? Number(e.target.value) : null)}
                                placeholder="No min"
                                min={0}
                                max={50000}
                                step={100}
                            />
                        </div>
                        <div className="space-y-1">
                            <label className="text-xs font-medium text-[hsl(var(--muted-foreground))]">Max Salary</label>
                            <Input
                                type="number"
                                value={maxSalary ?? ''}
                                onChange={e => setMaxSalary(e.target.value ? Number(e.target.value) : null)}
                                placeholder="No max"
                                min={0}
                                max={50000}
                                step={100}
                            />
                        </div>
                    </div>

                    <div className="space-y-1">
                        <label className="text-xs font-medium text-[hsl(var(--muted-foreground))]">Min Projection</label>
                        <Input
                            type="number"
                            value={minProj ?? ''}
                            onChange={e => setMinProj(e.target.value ? Number(e.target.value) : null)}
                            placeholder="No min"
                            min={0}
                            step={5}
                        />
                    </div>

                    <div className="space-y-1.5">
                        <div className="flex justify-between">
                            <label className="text-xs font-medium text-[hsl(var(--muted-foreground))]">Max % Off Optimal</label>
                            <span className="text-xs text-[hsl(var(--foreground))]">{maxOffoptimalPct}%</span>
                        </div>
                        <input
                            type="range"
                            min={0}
                            max={50}
                            step={0.5}
                            value={maxOffoptimalPct}
                            onChange={e => setMaxOffoptimalPct(Number(e.target.value))}
                            className="w-full accent-[hsl(var(--primary))]"
                        />
                        <p className="text-xs text-[hsl(var(--muted-foreground))]">0% disables; tighter caps may return fewer lineups</p>
                    </div>

                    <div className={`space-y-1.5 ${!hasStddev ? 'opacity-50' : ''}`}>
                        <div className="flex justify-between">
                            <label className="text-xs font-medium text-[hsl(var(--muted-foreground))]">Randomness</label>
                            <span className="text-xs text-[hsl(var(--foreground))]">{randomnessPct}%</span>
                        </div>
                        <input
                            type="range"
                            min={0}
                            max={100}
                            step={5}
                            value={randomnessPct}
                            onChange={e => setRandomnessPct(Number(e.target.value))}
                            disabled={!hasStddev}
                            className="w-full accent-[hsl(var(--primary))]"
                        />
                        <p className="text-xs text-[hsl(var(--muted-foreground))]">
                            {!hasStddev ? 'Requires sim projections with stddev' : 'Variance-aware noise for diversity'}
                        </p>
                    </div>

                    {/* Toggles */}
                    <div className="space-y-2 pt-1">
                        <label className="flex items-center gap-2.5 cursor-pointer">
                            <div className="toggle-switch">
                                <input
                                    type="checkbox"
                                    checked={lateSwapEnabled}
                                    onChange={e => setLateSwapEnabled(e.target.checked)}
                                />
                                <span className="toggle-slider"></span>
                            </div>
                            <span className="text-sm">{lateSwapEnabled ? 'Late Swap Mode' : 'Standard Mode'}</span>
                        </label>
                        <label className="flex items-center gap-2.5 cursor-pointer">
                            <div className="toggle-switch">
                                <input
                                    type="checkbox"
                                    checked={worldSampleEnabled}
                                    onChange={e => setWorldSampleEnabled(e.target.checked)}
                                />
                                <span className="toggle-slider"></span>
                            </div>
                            <span className="text-sm">{worldSampleEnabled ? 'World Sample Mode' : 'Mean Projections'}</span>
                        </label>
                        <label className="flex items-center gap-2.5 cursor-pointer">
                            <div className="toggle-switch">
                                <input
                                    type="checkbox"
                                    checked={useStrategyOverrides}
                                    onChange={e => setUseStrategyOverrides(e.target.checked)}
                                />
                                <span className="toggle-slider"></span>
                            </div>
                            <span className="text-sm">{useStrategyOverrides ? 'Strategy Overrides On' : 'Strategy Overrides Off'}</span>
                        </label>
                    </div>

                    <div className="rounded-md border border-[hsl(var(--border))] p-3 space-y-2">
                        <div className="flex items-center justify-between">
                            <span className="text-xs font-semibold uppercase tracking-wider text-[hsl(var(--muted-foreground))]">
                                Strategy Overrides
                            </span>
                            <Badge variant="secondary">{activeOverrideCount} saved</Badge>
                        </div>
                        <p className="text-xs text-[hsl(var(--muted-foreground))]">
                            Minutes and FPTS deltas persist by slate and only apply downstream to optimizer and contest sim.
                        </p>
                        {overrideLoading && (
                            <p className="text-xs text-[hsl(var(--muted-foreground))]">Loading overrides…</p>
                        )}
                        {overrideError && (
                            <p className="text-xs text-red-400">{overrideError}</p>
                        )}
                        <div className="flex items-center justify-between text-xs text-[hsl(var(--muted-foreground))]">
                            <span>{unsavedOverrideCount} unsaved</span>
                            {overrideSaving && <span>Saving…</span>}
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                            <Button
                                variant="default"
                                onClick={() => void saveOverrideChanges()}
                                disabled={!selectedSlate || overrideSaving || unsavedOverrideCount === 0}
                            >
                                Save
                            </Button>
                            <Button
                                variant="outline"
                                onClick={discardOverrideChanges}
                                disabled={overrideSaving || unsavedOverrideCount === 0}
                            >
                                Discard
                            </Button>
                        </div>
                        <Button
                            variant="outline"
                            className="w-full"
                            onClick={() => void resetAllOverrides()}
                            disabled={!selectedSlate || overrideSaving || activeOverrideCount === 0}
                        >
                            Reset Slate Overrides
                        </Button>
                    </div>

                    {/* Game Filters */}
                    {(currentSlateGameFilters.length > 0 || currentSlateTeams.length > 0) && (
                        <>
                            <div className="game-filters">
                                <Button
                                    variant="outline"
                                    onClick={() => setShowGameFilterDrawer(true)}
                                    className="w-full justify-between"
                                    type="button"
                                >
                                    <span>
                                        Filters ({currentSlateGameFilters.length - excludedGames.size} of {currentSlateGameFilters.length} games, {currentSlateTeams.length} teams)
                                        {excludedTeams.size > 0 ? ` · ${excludedTeams.size} teams faded` : ''}
                                    </span>
                                    <span>Open filters</span>
                                </Button>
                            </div>
                            {showGameFilterDrawer && (
                                <div
                                    className="game-filter-drawer-backdrop"
                                    onClick={() => setShowGameFilterDrawer(false)}
                                >
                                    <div
                                        className="game-filter-drawer"
                                        role="dialog"
                                        aria-modal="true"
                                        onClick={e => e.stopPropagation()}
                                    >
                                        <div className="game-filter-drawer-header">
                                            <h4>Game &amp; Team Filters</h4>
                                            <Button
                                                variant="outline"
                                                size="sm"
                                                onClick={() => setShowGameFilterDrawer(false)}
                                            >
                                                Close
                                            </Button>
                                        </div>
                                        <p className="text-xs text-[hsl(var(--muted-foreground))] mb-2">
                                            Fade teams, toggle games, and then regenerate your lineup build.
                                        </p>
                                        {currentSlateGameFilters.length > 0 && (
                                            <div className="game-filter-list">
                                                {currentSlateGameFilters.map(game => {
                                                    const isExcluded = excludedGames.has(game.matchup)
                                                    const startTime = game.start_time
                                                        ? new Date(game.start_time).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
                                                        : ''
                                                    return (
                                                        <div key={game.matchup} className={`game-filter-card ${isExcluded ? 'excluded' : ''}`}>
                                                            <label className={`game-filter-item ${isExcluded ? 'excluded' : ''}`}>
                                                                <input
                                                                    type="checkbox"
                                                                    checked={!isExcluded}
                                                                    onChange={() => toggleExcludedGame(game.matchup)}
                                                                />
                                                                <span className="game-matchup">{game.matchup}</span>
                                                                {startTime && <span className="game-time">{startTime}</span>}
                                                            </label>
                                                        </div>
                                                    )
                                                })}
                                            </div>
                                        )}
                                        <div>
                                            <div className="text-xs uppercase tracking-wider text-[hsl(var(--muted-foreground))] mb-1.5">Team fades</div>
                                            <div className="team-filter-row">
                                                {currentSlateTeams.map(team => {
                                                    const isTeamExcluded = excludedTeams.has(team)
                                                    return (
                                                        <button
                                                            key={team}
                                                            type="button"
                                                            className={`team-filter-chip ${isTeamExcluded ? 'excluded' : ''}`}
                                                            onClick={() => toggleExcludedTeam(team)}
                                                        >
                                                            {isTeamExcluded ? `Fade ${team}` : team}
                                                        </button>
                                                    )
                                                })}
                                            </div>
                                        </div>
                                        <div className="game-filter-actions">
                                            {(excludedGames.size > 0 || excludedTeams.size > 0) ? (
                                                <Button
                                                    variant="outline"
                                                    onClick={clearFilters}
                                                >
                                                    Include all games and teams
                                                </Button>
                                            ) : (
                                                <span className="text-xs text-[hsl(var(--muted-foreground))]">No filters currently applied.</span>
                                            )}
                                            <Button
                                                onClick={() => setShowGameFilterDrawer(false)}
                                            >
                                                Done
                                            </Button>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </>
                    )}

                    <div className="flex gap-3 text-sm text-[hsl(var(--muted-foreground))]">
                        <span>🔒 {lockedIds.size} locked</span>
                        <span>🚫 Banned: {combinedBanCount}</span>
                        <span>🏁 {excludedTeams.size} team fades</span>
                    </div>

                    <Button
                        className="w-full"
                        onClick={handleStartBuild}
                        disabled={!selectedSlate || poolLoading || (currentJob?.status === 'running')}
                    >
                        {currentJob?.status === 'running' ? 'Building…' : 'Generate Lineups'}
                    </Button>

                    {/* Progress */}
                    {currentJob && (
                        <div className="space-y-2">
                            <p className="text-sm text-[hsl(var(--muted-foreground))]">
                                {currentJob.status === 'running' && `Generating… ${currentJob.lineups_count}/${currentJob.target}`}
                                {currentJob.status === 'completed' && `✓ ${currentJob.lineups_count} lineups in ${currentJob.wall_time_sec?.toFixed(1)}s`}
                                {currentJob.status === 'failed' && `✗ ${currentJob.error}`}
                                {currentJob.status === 'pending' && 'Starting…'}
                            </p>
                            {currentJob.status === 'running' && (
                                <div className="h-1.5 rounded-full bg-[hsl(var(--secondary))]">
                                    <div
                                        className="h-1.5 rounded-full bg-[hsl(var(--primary))] transition-all"
                                        style={{ width: `${Math.min(100, (currentJob.lineups_count / currentJob.target) * 100)}%` }}
                                    />
                                </div>
                            )}
                        </div>
                    )}

                    {buildError && (
                        <p className="text-sm text-red-400">{buildError}</p>
                    )}

                    {currentJob?.status === 'completed' && (
                        <Button variant="outline" className="w-full" onClick={handleExport}>
                            Export CSV
                        </Button>
                    )}
                </aside>
                {settingsDrawerOpen && (
                    <button
                        type="button"
                        className="optimizer-settings-backdrop"
                        aria-label="Close build settings"
                        onClick={() => setSettingsDrawerOpen(false)}
                    />
                )}

                <div className="optimizer-main-column">
                    {/* Player Pool Table */}
                    <section className="optimizer-pool">
                        <div className="flex items-center justify-between mb-3">
                            <h3 className="text-sm font-semibold text-[hsl(var(--muted-foreground))] uppercase tracking-wider">
                                Player Pool
                                <span className="ml-2 font-normal text-[hsl(var(--foreground))]">({pool.length})</span>
                            </h3>
                            <Input
                                type="text"
                                placeholder="Filter players…"
                                value={filter}
                                onChange={e => setFilter(e.target.value)}
                                className="w-[220px]"
                            />
                        </div>

                        {poolLoading ? (
                            <div className="text-sm text-[hsl(var(--muted-foreground))] py-8 text-center">Loading player pool…</div>
                        ) : (
                            <div className="table-wrapper">
                                <table>
                                <thead>
                                    <tr>
                                        <th className="w-10 text-center">Lock</th>
                                        <th className="w-10 text-center">Ban</th>
                                        <th
                                            onClick={() => toggleSort('name')}
                                            className="sortable cursor-pointer select-none"
                                        >
                                            Player<SortIcon col="name" />
                                        </th>
                                        <th
                                            onClick={() => toggleSort('team')}
                                            className="sortable cursor-pointer select-none"
                                        >
                                            Team<SortIcon col="team" />
                                        </th>
                                        <th>Pos</th>
                                        <th
                                            onClick={() => toggleSort('salary')}
                                            className="sortable cursor-pointer select-none"
                                        >
                                            Salary<SortIcon col="salary" />
                                        </th>
                                        <th
                                            onClick={() => toggleSort('min')}
                                            className="sortable cursor-pointer select-none"
                                            title="Projected minutes (P50)"
                                        >
                                            {showStrategyColumns ? 'Model Min' : 'Min'}<SortIcon col="min" />
                                        </th>
                                        {showStrategyColumns && <th className="text-center">Min Δ</th>}
                                        {showStrategyColumns && <th className="text-center">Eff Min</th>}
                                        <th
                                            onClick={() => toggleSort('fppm')}
                                            className="sortable cursor-pointer select-none"
                                            title="Fantasy points per minute"
                                        >
                                            FPPM<SortIcon col="fppm" />
                                        </th>
                                        {showStrategyColumns && <th className="text-center">Model Proj</th>}
                                        {showStrategyColumns && <th className="text-center">FPTS Δ</th>}
                                        <th
                                            onClick={() => toggleSort('proj')}
                                            className="sortable cursor-pointer select-none"
                                        >
                                            {showStrategyColumns ? 'Eff Proj' : 'Proj'}<SortIcon col="proj" />
                                        </th>
                                        <th
                                            onClick={() => toggleSort('own_proj')}
                                            className="sortable cursor-pointer select-none"
                                        >
                                            Own%<SortIcon col="own_proj" />
                                        </th>
                                        <th
                                            onClick={() => toggleSort('optimal_pct')}
                                            className="sortable cursor-pointer select-none"
                                        >
                                            Opt%<SortIcon col="optimal_pct" />
                                        </th>
                                        <th
                                            onClick={() => toggleSort('ceiling_leverage')}
                                            className="sortable cursor-pointer select-none"
                                        >
                                            Ceil Lev<SortIcon col="ceiling_leverage" />
                                        </th>
                                        <th
                                            onClick={() => toggleSort('boom_pct')}
                                            className="sortable cursor-pointer select-none"
                                        >
                                            Boom%<SortIcon col="boom_pct" />
                                        </th>
                                        <th
                                            onClick={() => toggleSort('bust_pct')}
                                            className="sortable cursor-pointer select-none"
                                        >
                                            Bust%<SortIcon col="bust_pct" />
                                        </th>
                                        {showStrategyColumns && <th className="text-center">Override</th>}
                                        <th
                                            onClick={() => toggleSort('value')}
                                            className="sortable cursor-pointer select-none"
                                        >
                                            Value<SortIcon col="value" />
                                        </th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {filteredPool.map(p => {
                                        const isLocked = lockedIds.has(p.player_id)
                                        const isBanned = bannedIds.has(p.player_id)
                                        const currentOverride = getCurrentOverride(p.player_id)
                                        return (
                                            <tr
                                                key={p.player_id}
                                                className={[
                                                    isLocked ? 'player-locked' : '',
                                                    isBanned ? 'player-banned' : '',
                                                    p.has_override ? 'player-override' : '',
                                                ].filter(Boolean).join(' ')}
                                            >
                                                <td className="text-center">
                                                    <input
                                                        type="checkbox"
                                                        checked={isLocked}
                                                        onChange={() => toggleLock(p.player_id)}
                                                        title="Lock this player"
                                                    />
                                                </td>
                                                <td className="text-center">
                                                    <input
                                                        type="checkbox"
                                                        checked={isBanned}
                                                        onChange={() => toggleBan(p.player_id)}
                                                        title="Ban this player"
                                                    />
                                                </td>
                                                <td className="font-medium">{p.name}</td>
                                                <td className="text-[hsl(var(--muted-foreground))]">{p.team}</td>
                                                <td>
                                                    <div className="flex flex-wrap gap-0.5">
                                                        {p.positions.join('/').split('/').map(pos => (
                                                            <Badge key={pos} variant="secondary" className="text-[10px] py-0 px-1">
                                                                {pos}
                                                            </Badge>
                                                        ))}
                                                    </div>
                                                </td>
                                                <td className="tabular-nums">{formatSalary(p.salary)}</td>
                                                <td className="tabular-nums text-[hsl(var(--muted-foreground))]">
                                                    {formatMin(p.model_minutes)}
                                                </td>
                                                {showStrategyColumns && (
                                                    <td className="min-w-[92px]">
                                                        <Input
                                                            type="number"
                                                            step="0.5"
                                                            min={-24}
                                                            max={24}
                                                            value={currentOverride?.minutes_delta ?? ''}
                                                            onChange={(e) => {
                                                                const value = e.target.value
                                                                updateStrategyOverride(p.player_id, {
                                                                    minutes_delta: value === '' ? null : Number(value),
                                                                })
                                                            }}
                                                            className="h-8 tabular-nums"
                                                            placeholder="0.0"
                                                        />
                                                    </td>
                                                )}
                                                {showStrategyColumns && (
                                                    <td className="tabular-nums text-[hsl(var(--muted-foreground))]">
                                                        {formatMin(getDisplayMinutes(p))}
                                                    </td>
                                                )}
                                                <td className="tabular-nums text-[hsl(var(--muted-foreground))]">
                                                    {formatFppm(p.fppm)}
                                                </td>
                                                {showStrategyColumns && (
                                                    <td className="tabular-nums text-[hsl(var(--muted-foreground))]">
                                                        {formatProj(p.model_proj)}
                                                    </td>
                                                )}
                                                {showStrategyColumns && (
                                                    <td className="min-w-[92px]">
                                                        <Input
                                                            type="number"
                                                            step="0.5"
                                                            min={-40}
                                                            max={40}
                                                            value={currentOverride?.fpts_delta ?? ''}
                                                            onChange={(e) => {
                                                                const value = e.target.value
                                                                updateStrategyOverride(p.player_id, {
                                                                    fpts_delta: value === '' ? null : Number(value),
                                                                })
                                                            }}
                                                            className="h-8 tabular-nums"
                                                            placeholder="0.0"
                                                        />
                                                    </td>
                                                )}
                                                <td className="tabular-nums font-medium text-[hsl(var(--primary))]">
                                                    {formatProj(p.proj)}
                                                </td>
                                                <td className="tabular-nums">{formatOwn(p.own_proj)}</td>
                                                <td className="tabular-nums">{formatPct(p.optimal_pct)}</td>
                                                <td className="tabular-nums">{formatSigned(p.ceiling_leverage)}</td>
                                                <td className="tabular-nums">{formatPct(p.boom_pct)}</td>
                                                <td className="tabular-nums">{formatPct(p.bust_pct)}</td>
                                                {showStrategyColumns && (
                                                    <td className="text-center">
                                                        <Button
                                                            variant="outline"
                                                            className="h-8 px-2"
                                                            disabled={!currentOverride}
                                                            onClick={() => {
                                                                updateStrategyOverride(p.player_id, {
                                                                    minutes_delta: null,
                                                                    fpts_delta: null,
                                                                    minutes: null,
                                                                    fpts: null,
                                                                })
                                                            }}
                                                        >
                                                            Clear
                                                        </Button>
                                                    </td>
                                                )}
                                                <td className="tabular-nums">{formatValue(p)}</td>
                                            </tr>
                                        )
                                    })}
                                    {filteredPool.length === 0 && !poolLoading && (
                                        <tr>
                                            <td colSpan={showStrategyColumns ? 20 : 15} className="text-center text-[hsl(var(--muted-foreground))] py-8">
                                                No players found
                                            </td>
                                        </tr>
                                    )}
                                </tbody>
                                </table>
                            </div>
                        )}
                    </section>

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

                    {/* Lineups Section */}
                    {lineups.length > 0 && (
                        <section className="lineups-section">
                            <div className="lineups-toolbar">
                                <div className="lineups-toolbar-top">
                                    <h3>Lineups ({filteredLineups.length} of {lineups.length})</h3>
                                    <div className="lineups-toolbar-top-right">
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
                                    </div>
                                </div>
                                <div className="lineups-filter-row">
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
                                        title="Minimum projection total"
                                    />
                                    <input
                                        type="number"
                                        placeholder="Max own%"
                                        value={maxLineupOwn ?? ''}
                                        onChange={e => setMaxLineupOwn(e.target.value ? Number(e.target.value) : null)}
                                        title="Maximum ownership total"
                                    />
                                    <input
                                        type="number"
                                        placeholder="Min p90"
                                        value={minLineupP90 ?? ''}
                                        onChange={e => setMinLineupP90(e.target.value ? Number(e.target.value) : null)}
                                        title="Minimum p90 ceiling"
                                    />
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
                                </div>
                                <div className="lineups-action-row">
                                    <span className="lineups-selected-count">
                                        {selectedLineupIds.size} selected
                                    </span>
                                    <button
                                        className="lineups-action-btn"
                                        onClick={selectAllVisible}
                                        disabled={filteredLineups.length === 0}
                                    >
                                        Select showing
                                    </button>
                                    <button
                                        className="lineups-action-btn"
                                        onClick={selectAllFiltered}
                                        disabled={filteredLineups.length === 0}
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
                                    >
                                        Export selected CSV
                                    </button>
                                </div>
                                <div className="lineups-group-row">
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
                                    >
                                        Save group
                                    </button>
                                    <button
                                        className="lineups-action-btn"
                                        onClick={selectActiveGroupLineups}
                                        disabled={!activeLineupGroupId}
                                    >
                                        Select group
                                    </button>
                                    <button
                                        className="lineups-action-btn primary"
                                        onClick={exportActiveGroupCSV}
                                        disabled={!activeLineupGroupId}
                                    >
                                        Export group CSV
                                    </button>
                                    <button
                                        className="lineups-action-btn danger"
                                        onClick={deleteActiveGroup}
                                        disabled={!activeLineupGroupId}
                                    >
                                        Delete group
                                    </button>
                                </div>
                            </div>

                            <div className="lineups-grid">
                                {filteredLineups.slice(0, showCount).map((lu, idx) => {
                                    const totalSalary = lu.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.salary ?? 0), 0)
                                    const totalProj = lu.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.proj ?? 0), 0)
                                    const totalP90 = lu.p90 ?? lu.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.p90 ?? 0), 0)
                                    const totalOwn = lu.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.own_proj ?? 0), 0)
                                    const isSelected = selectedLineupIds.has(lu.lineup_id)
                                    const filterValue = lineupFilter.trim().toLowerCase()
                                    const assignedSlots = getDisplaySlotByAssignment(lu.player_ids, playerMap)
                                    const orderedPlayers = assignedSlots
                                        ? assignedSlots.map(({ playerId, slot }) => ({
                                            playerId,
                                            p: playerMap.get(playerId),
                                            slot,
                                            slotOrder: DK_SLOT_PRIORITY[slot],
                                        }))
                                        : lu.player_ids
                                            .map((id, index) => {
                                                const p = playerMap.get(id)
                                                return {
                                                    playerId: id,
                                                    p,
                                                    slotOrder: getLineupSlotOrder(p?.positions),
                                                    slot: 'N/A',
                                                    index,
                                                    sortName: p?.name?.toLowerCase() ?? '',
                                                }
                                            })
                                            .sort((a, b) => {
                                                if (a.slotOrder !== b.slotOrder) return a.slotOrder - b.slotOrder
                                                if (a.sortName !== b.sortName) return a.sortName.localeCompare(b.sortName)
                                                return a.index - b.index
                                            })
                                            .map(({ playerId, p, slot }) => ({ playerId, p, slot }))

                                    return (
                                        <Card
                                            key={lu.lineup_id}
                                            className={`lineup-card ${isSelected ? 'selected' : ''}`}
                                        >
                                            <CardHeader className="lineup-header">
                                                <div className="lineup-card-meta">
                                                    <label className="lineup-select">
                                                        <input
                                                            type="checkbox"
                                                            checked={isSelected}
                                                            onChange={() => toggleLineupSelection(lu.lineup_id)}
                                                            title="Select lineup"
                                                        />
                                                    </label>
                                                    <span className="lineup-rank">#{idx + 1}</span>
                                                </div>
                                                <div className="lineup-metrics-grid">
                                                    <span className="lineup-metric">
                                                        <span className="lineup-metric-label">Salary</span>
                                                        <span className="lineup-salary">${totalSalary.toLocaleString()}</span>
                                                    </span>
                                                    <span className="lineup-metric">
                                                        <span className="lineup-metric-label">Proj</span>
                                                        <span className="lineup-proj">{totalProj.toFixed(1)}</span>
                                                    </span>
                                                    <span className="lineup-metric">
                                                        <span className="lineup-metric-label">p90</span>
                                                        <span className="lineup-p90">{totalP90 > 0 ? totalP90.toFixed(1) : '—'}</span>
                                                    </span>
                                                    <span className="lineup-metric">
                                                        <span className="lineup-metric-label">Own%</span>
                                                        <span className="lineup-ownership">{totalOwn.toFixed(1)}</span>
                                                    </span>
                                                </div>
                                            </CardHeader>
                                            <CardContent className="lineup-body">
                                                <div className="lineup-players">
                                                    {orderedPlayers.map(({ playerId, p, slot }) => {
                                                        const isFiltered = filterValue && p && p.name.toLowerCase().includes(filterValue)
                                                        return (
                                                            <span
                                                                key={playerId}
                                                                className={`lineup-player ${isFiltered ? 'highlight' : ''}`}
                                                            >
                                                                {p ? (
                                                                    <>
                                                                        <span className="lineup-slot-tag">{slot}</span>
                                                                        <span>{p.name}</span>
                                                                    </>
                                                                ) : (
                                                                    playerId
                                                                )}
                                                            </span>
                                                        )
                                                    })}
                                                </div>
                                            </CardContent>
                                        </Card>
                                    )
                                })}
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
            </div>
        </div>
    )
}
