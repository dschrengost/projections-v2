import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  getSlates,
  getPlayerPool,
  getPlayerPoolWithOverrides,
  startBuild,
  getBuildStatus,
  getBuildLineups,
  exportLineupsCSV,
  exportCustomLineupsCSV,
  getSavedBuilds,
  loadSavedBuild,
  deleteSavedBuild,
  saveCustomBuild,
  getOverrides,
  saveOverrides,
  clearOverrides,
  Slate,
  PoolPlayerWithOverrides,
  PlayerOverride,
  JobStatus,
  LineupRow,
  QuickBuildRequest,
  SavedBuild,
} from '../api/optimizer'
import { useSlateDateAndSlate } from '../hooks/useSlateDate'
import { formatSlateLabel } from '../utils/slateFormat'
import { Button, Card, CardTitle, Input, Select, Badge, Drawer } from '../components/ui'
import { PlayerPoolTable } from '../components/optimizer/PlayerPoolTable'
import { SettingsPanel } from '../components/optimizer/SettingsPanel'
import { LineupCardNew } from '../components/optimizer/LineupCardNew'

type SortKey = 'name' | 'team' | 'salary' | 'proj' | 'own_proj' | 'value'

type LineupGroup = {
  id: string
  name: string
  lineup_ids: number[]
  created_at: string
}

export default function OptimizerPageNew() {
  // Date and slate selection (persisted in URL)
  const [selectedDate, setSelectedDate, selectedSlate, setSelectedSlate] = useSlateDateAndSlate()
  const [slates, setSlates] = useState<Slate[]>([])
  const [slatesLoading, setSlatesLoading] = useState(false)
  const [slatesError, setSlatesError] = useState<string | null>(null)

  // Player pool
  const [pool, setPool] = useState<PoolPlayerWithOverrides[]>([])
  const [poolLoading, setPoolLoading] = useState(false)
  const [poolError, setPoolError] = useState<string | null>(null)

  // User overrides
  const [overrides, setOverrides] = useState<Map<string, PlayerOverride>>(new Map())
  const [savedOverrides, setSavedOverrides] = useState<Map<string, PlayerOverride>>(new Map())
  const [overrideRevision, setOverrideRevision] = useState<number | null>(null)
  const [overrideLoading, setOverrideLoading] = useState(false)
  const [overrideSaving, setOverrideSaving] = useState(false)
  const [overrideError, setOverrideError] = useState<string | null>(null)
  const [pendingOverrideIds, setPendingOverrideIds] = useState<Set<string>>(new Set())
  const [pendingOutApply, setPendingOutApply] = useState(false)

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
  const [maxExposurePct, setMaxExposurePct] = useState(0)
  const [nearDupJaccard, setNearDupJaccard] = useState(0)
  const [globalTeamLimit, setGlobalTeamLimit] = useState(4)
  const [minSalary, setMinSalary] = useState<number | null>(null)
  const [maxSalary, setMaxSalary] = useState<number | null>(50000)
  const [minProj, setMinProj] = useState<number | null>(null)
  const [maxOffoptimalPct, setMaxOffoptimalPct] = useState(0)
  const [randomnessPct, setRandomnessPct] = useState(0)
  const [useUserOverrides, setUseUserOverrides] = useState(false)
  const [lateSwapEnabled, setLateSwapEnabled] = useState(false)
  const [worldSampleEnabled, setWorldSampleEnabled] = useState(false)

  // Refs for async operations
  const overridesRef = useRef(overrides)
  const savedOverridesRef = useRef(savedOverrides)
  const pendingOverrideIdsRef = useRef(pendingOverrideIds)
  const overrideRevisionRef = useRef(overrideRevision)
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

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

  // Mobile UI state
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [activeView, setActiveView] = useState<'players' | 'lineups'>('players')

  // Get current slate's games
  const currentSlateGames = useMemo(() => {
    const slate = slates.find(s => s.draft_group_id === selectedSlate)
    return slate?.games ?? []
  }, [slates, selectedSlate])

  // Check if stddev is available in pool
  const hasStddev = useMemo(() =>
    pool.some(p => p.stddev != null && p.stddev > 0), [pool])

  // Sync refs
  useEffect(() => { overridesRef.current = overrides }, [overrides])
  useEffect(() => { savedOverridesRef.current = savedOverrides }, [savedOverrides])
  useEffect(() => { pendingOverrideIdsRef.current = pendingOverrideIds }, [pendingOverrideIds])
  useEffect(() => { overrideRevisionRef.current = overrideRevision }, [overrideRevision])

  // Load user override preference from localStorage
  useEffect(() => {
    if (typeof window === 'undefined') return
    const stored = window.localStorage.getItem('optimizer.useUserOverrides')
    if (stored != null) setUseUserOverrides(stored === 'true')
  }, [])

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem('optimizer.useUserOverrides', String(useUserOverrides))
  }, [useUserOverrides])

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
      const data = useUserOverrides
        ? await getPlayerPoolWithOverrides(selectedDate, selectedSlate)
        : await getPlayerPool(selectedDate, selectedSlate)
      setPool(data)
      setLockedIds(new Set())
      setBannedIds(new Set())
    } catch (err) {
      setPoolError((err as Error).message)
      setPool([])
    } finally {
      setPoolLoading(false)
    }
  }, [selectedDate, selectedSlate, useUserOverrides])

  useEffect(() => { void loadPool() }, [loadPool])

  const loadOverrides = useCallback(async () => {
    if (!selectedSlate) {
      setOverrides(new Map())
      setSavedOverrides(new Map())
      setOverrideRevision(null)
      setPendingOverrideIds(new Set())
      setPendingOutApply(false)
      return
    }
    setOverrideLoading(true)
    setOverrideError(null)
    try {
      const data = await getOverrides(selectedDate, selectedSlate)
      const next = new Map<string, PlayerOverride>()
      data.overrides.forEach((override) => next.set(override.player_id, override))
      setOverrides(next)
      setSavedOverrides(new Map(next))
      setOverrideRevision(data.client_revision)
      setPendingOverrideIds(new Set())
      setPendingOutApply(false)
    } catch (err) {
      setOverrideError((err as Error).message)
      setOverrides(new Map())
      setSavedOverrides(new Map())
      setOverrideRevision(null)
    } finally {
      setOverrideLoading(false)
    }
  }, [selectedDate, selectedSlate])

  useEffect(() => {
    void loadOverrides()
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current)
      saveTimerRef.current = null
    }
  }, [loadOverrides])

  type OverrideField = 'minutes' | 'fpts' | 'own' | 'is_out'

  const updateOverrideFields = (playerId: string, updates: Partial<Record<OverrideField, number | boolean | null>>) => {
    setOverrides(prev => {
      const next = new Map(prev)
      const current = next.get(playerId) ?? { player_id: playerId, minutes: null, fpts: null, own: null, is_out: false }
      const updated = { ...current, player_id: playerId, ...updates } as PlayerOverride
      const hasOverride = updated.is_out || updated.minutes != null || updated.fpts != null || updated.own != null
      if (hasOverride) {
        next.set(playerId, updated)
      } else {
        next.delete(playerId)
      }
      return next
    })
    setPendingOverrideIds(prev => {
      const next = new Set(prev)
      next.add(playerId)
      return next
    })
    if (updates.is_out !== undefined) setPendingOutApply(true)
  }

  const flushOverrides = useCallback(async () => {
    if (!selectedSlate) return
    const pending = Array.from(pendingOverrideIdsRef.current)
    if (pending.length === 0) return
    setOverrideSaving(true)
    setOverrideError(null)
    const payload = pending.map((playerId) => {
      const current = overridesRef.current.get(playerId)
      return {
        player_id: playerId,
        minutes: current?.minutes ?? null,
        fpts: current?.fpts ?? null,
        own: current?.own ?? null,
        is_out: current?.is_out ?? false,
      }
    })
    try {
      const response = await saveOverrides(selectedDate, selectedSlate, payload, overrideRevisionRef.current ?? undefined)
      const responseMap = new Map(response.overrides.map(o => [o.player_id, o]))
      setOverrides(prev => {
        const next = new Map(prev)
        pending.forEach((playerId) => {
          const updated = responseMap.get(playerId)
          if (updated) next.set(playerId, updated)
          else next.delete(playerId)
        })
        return next
      })
      setSavedOverrides(prev => {
        const next = new Map(prev)
        pending.forEach((playerId) => {
          const updated = responseMap.get(playerId)
          if (updated) next.set(playerId, updated)
          else next.delete(playerId)
        })
        return next
      })
      setOverrideRevision(response.client_revision)
      setPendingOverrideIds(prev => {
        const next = new Set(prev)
        pending.forEach((playerId) => next.delete(playerId))
        return next
      })
    } catch (err) {
      setOverrideError((err as Error).message)
    } finally {
      setOverrideSaving(false)
    }
  }, [selectedDate, selectedSlate])

  const scheduleOverrideSave = useCallback((delayMs = 500) => {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
    saveTimerRef.current = setTimeout(() => {
      saveTimerRef.current = null
      void flushOverrides()
    }, delayMs)
  }, [flushOverrides])

  // Poll job status
  useEffect(() => {
    if (!currentJob || currentJob.status === 'completed' || currentJob.status === 'failed') return
    const interval = setInterval(async () => {
      try {
        const status = await getBuildStatus(currentJob.job_id)
        setCurrentJob(status)
        if (status.status === 'completed') {
          const result = await getBuildLineups(status.job_id)
          setLineups(result.lineups)
          setActiveView('lineups')
          refreshSavedBuilds()
        }
      } catch (err) {
        setBuildError((err as Error).message)
      }
    }, 1000)
    return () => clearInterval(interval)
  }, [currentJob])

  // Load saved builds
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

  useEffect(() => { void refreshSavedBuilds() }, [selectedDate, selectedSlate])

  // Filtered and sorted pool
  const filteredPool = useMemo(() => {
    let filtered = pool.slice()
    if (minProj != null) filtered = filtered.filter(p => p.proj >= minProj)
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
    setBannedIds(prev => { const next = new Set(prev); next.delete(id); return next })
  }

  const toggleBan = (id: string) => {
    setBannedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
    setLockedIds(prev => { const next = new Set(prev); next.delete(id); return next })
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
        max_exposure_pct: maxExposurePct > 0 ? maxExposurePct : null,
        near_dup_jaccard: nearDupJaccard > 0 ? nearDupJaccard : undefined,
        global_team_limit: globalTeamLimit,
        min_salary: minSalary,
        max_salary: maxSalary,
        lock_ids: Array.from(lockedIds),
        ban_ids: Array.from(bannedIds),
        max_offoptimal_pct: maxOffoptimalPct > 0 ? maxOffoptimalPct / 100 : undefined,
        exclude_games: Array.from(excludedGames),
        enum_enable: maxPool >= 5000,
        randomness_pct: randomnessPct > 0 && hasStddev ? randomnessPct : undefined,
        use_user_overrides: useUserOverrides,
        late_swap_enabled: lateSwapEnabled,
        world_sample_enabled: worldSampleEnabled,
      }
      const job = await startBuild(request)
      setCurrentJob(job)
      setSettingsOpen(false) // Close settings drawer on mobile
    } catch (err) {
      setBuildError((err as Error).message)
    }
  }

  // Player map for lineup display
  const playerMap = useMemo(() => {
    const map = new Map<string, PoolPlayerWithOverrides>()
    pool.forEach(p => map.set(p.player_id, p))
    return map
  }, [pool])

  // Helper functions
  function getModelFppm(player: PoolPlayerWithOverrides) {
    const baseMinutes = player.model_minutes ?? null
    const baseProj = player.model_proj ?? player.proj
    if (baseMinutes && baseMinutes > 0 && baseProj != null) return baseProj / baseMinutes
    return 1.0
  }

  function getEffectiveProj(player: PoolPlayerWithOverrides | undefined) {
    if (!player) return 0
    if (!useUserOverrides) return player.proj ?? 0
    const override = overrides.get(player.player_id)
    if (override?.is_out) return 0
    if (override?.fpts != null) return override.fpts
    if (override?.minutes != null) {
      const fppm = getModelFppm(player)
      return Number((override.minutes * fppm).toFixed(1))
    }
    return player.proj ?? 0
  }

  // Filter lineups
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
      const proj = lu.player_ids.reduce((sum, id) => sum + getEffectiveProj(playerMap.get(id)), 0)
      const own = lu.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.own_proj ?? 0), 0)
      const p90 = lu.p90 ?? lu.player_ids.reduce((sum, id) => sum + (playerMap.get(id)?.p90 ?? 0), 0)
      if (minLineupProj != null && proj < minLineupProj) return false
      if (maxLineupOwn != null && own > maxLineupOwn) return false
      if (minLineupP90 != null && p90 < minLineupP90) return false
      return true
    })
    if (lineupSort !== 'default') {
      result.sort((a, b) => {
        const aProj = a.player_ids.reduce((sum, id) => sum + getEffectiveProj(playerMap.get(id)), 0)
        const bProj = b.player_ids.reduce((sum, id) => sum + getEffectiveProj(playerMap.get(id)), 0)
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
  }, [lineups, lineupFilter, lineupSort, playerMap, minLineupProj, maxLineupOwn, minLineupP90, overrides, useUserOverrides])

  const toggleLineupSelection = (lineupId: number) => {
    setSelectedLineupIds(prev => {
      const next = new Set(prev)
      if (next.has(lineupId)) next.delete(lineupId)
      else next.add(lineupId)
      return next
    })
  }

  const selectAllFiltered = () => {
    setSelectedLineupIds(new Set(filteredLineups.map(lu => lu.lineup_id)))
  }

  const clearSelection = () => {
    setSelectedLineupIds(new Set())
  }

  // Export
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

  const showOverrideColumns = useUserOverrides || overrides.size > 0 || pendingOverrideIds.size > 0

  return (
    <div className="min-h-screen bg-slate-900">
      {/* Header */}
      <header className="sticky top-0 z-30 bg-slate-900/95 backdrop-blur border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-4 py-3">
          <div className="flex items-center justify-between gap-4">
            {/* Title */}
            <div>
              <h1 className="text-xl font-bold text-slate-100">Lineup Optimizer</h1>
              <p className="text-xs text-slate-500 hidden sm:block">QuickBuild lineup generation</p>
            </div>

            {/* Date/Slate Selectors */}
            <div className="flex items-center gap-2 sm:gap-3">
              <input
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="px-2 py-1.5 text-sm bg-slate-800 border border-slate-700 rounded-lg text-slate-100"
              />
              <select
                value={selectedSlate ?? ''}
                onChange={(e) => setSelectedSlate(Number(e.target.value) || null)}
                disabled={slatesLoading}
                className="px-2 py-1.5 text-sm bg-slate-800 border border-slate-700 rounded-lg text-slate-100 max-w-[150px] sm:max-w-none"
              >
                {slates.length === 0 && <option value="">No slates</option>}
                {slates.map(s => (
                  <option key={s.draft_group_id} value={s.draft_group_id}>
                    {formatSlateLabel(s)}
                  </option>
                ))}
              </select>

              {/* Settings Button (Mobile) */}
              <button
                onClick={() => setSettingsOpen(true)}
                className="lg:hidden p-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-100 hover:bg-slate-700"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                </svg>
              </button>
            </div>
          </div>

          {/* Error Banner */}
          {(slatesError || poolError || buildError) && (
            <div className="mt-2 px-3 py-2 bg-danger/10 border border-red-500/30 rounded-lg text-danger text-sm">
              {slatesError || poolError || buildError}
            </div>
          )}

          {/* Build Status */}
          {currentJob && (
            <div className="mt-3">
              <div className="flex items-center gap-3">
                <span className="text-sm text-slate-400">
                  {currentJob.status === 'running' && `Generating... ${currentJob.lineups_count}/${currentJob.target}`}
                  {currentJob.status === 'completed' && `${currentJob.lineups_count} lineups in ${currentJob.wall_time_sec?.toFixed(1)}s`}
                  {currentJob.status === 'failed' && `Failed: ${currentJob.error}`}
                  {currentJob.status === 'pending' && 'Starting...'}
                </span>
                {currentJob.status === 'completed' && (
                  <Button size="sm" variant="secondary" onClick={handleExport}>
                    Export CSV
                  </Button>
                )}
              </div>
              {currentJob.status === 'running' && (
                <div className="mt-2 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-accent-primary to-accent-secondary transition-all duration-300"
                    style={{ width: `${Math.min(100, (currentJob.lineups_count / currentJob.target) * 100)}%` }}
                  />
                </div>
              )}
            </div>
          )}
        </div>
      </header>

      {/* Mobile View Toggle */}
      {lineups.length > 0 && (
        <div className="lg:hidden sticky top-[73px] z-20 bg-slate-900 border-b border-slate-700">
          <div className="flex">
            <button
              onClick={() => setActiveView('players')}
              className={`flex-1 py-2.5 text-sm font-medium border-b-2 transition-colors ${
                activeView === 'players'
                  ? 'border-indigo-500 text-indigo-500'
                  : 'border-transparent text-slate-500 hover:text-slate-100'
              }`}
            >
              Players ({pool.length})
            </button>
            <button
              onClick={() => setActiveView('lineups')}
              className={`flex-1 py-2.5 text-sm font-medium border-b-2 transition-colors ${
                activeView === 'lineups'
                  ? 'border-indigo-500 text-indigo-500'
                  : 'border-transparent text-slate-500 hover:text-slate-100'
              }`}
            >
              Lineups ({filteredLineups.length})
            </button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-4">
        <div className="flex gap-6">
          {/* Settings Sidebar (Desktop) */}
          <aside className="hidden lg:block w-80 flex-shrink-0">
            <div className="sticky top-24">
              <SettingsPanel
                maxPool={maxPool}
                setMaxPool={setMaxPool}
                builds={builds}
                setBuilds={setBuilds}
                minUniq={minUniq}
                setMinUniq={setMinUniq}
                maxExposurePct={maxExposurePct}
                setMaxExposurePct={setMaxExposurePct}
                nearDupJaccard={nearDupJaccard}
                setNearDupJaccard={setNearDupJaccard}
                globalTeamLimit={globalTeamLimit}
                setGlobalTeamLimit={setGlobalTeamLimit}
                minSalary={minSalary}
                setMinSalary={setMinSalary}
                maxSalary={maxSalary}
                setMaxSalary={setMaxSalary}
                minProj={minProj}
                setMinProj={setMinProj}
                maxOffoptimalPct={maxOffoptimalPct}
                setMaxOffoptimalPct={setMaxOffoptimalPct}
                randomnessPct={randomnessPct}
                setRandomnessPct={setRandomnessPct}
                hasStddev={hasStddev}
                useUserOverrides={useUserOverrides}
                setUseUserOverrides={setUseUserOverrides}
                lateSwapEnabled={lateSwapEnabled}
                setLateSwapEnabled={setLateSwapEnabled}
                worldSampleEnabled={worldSampleEnabled}
                setWorldSampleEnabled={setWorldSampleEnabled}
                games={currentSlateGames}
                excludedGames={excludedGames}
                setExcludedGames={setExcludedGames}
                lockedCount={lockedIds.size}
                bannedCount={bannedIds.size}
                onStartBuild={handleStartBuild}
                isBuilding={currentJob?.status === 'running'}
                canBuild={!!selectedSlate && !poolLoading}
              />
            </div>
          </aside>

          {/* Main Content Area */}
          <main className="flex-1 min-w-0">
            {/* Players View */}
            <div className={lineups.length > 0 && activeView !== 'players' ? 'hidden lg:block' : ''}>
              {/* Player Pool Header */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                <h2 className="text-lg font-semibold text-slate-100">
                  Player Pool <span className="text-slate-500 font-normal">({filteredPool.length})</span>
                </h2>
                <Input
                  type="text"
                  placeholder="Filter players..."
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  className="w-full sm:w-48"
                />
              </div>

              {/* Player Pool Table */}
              {poolLoading ? (
                <div className="flex items-center justify-center py-12">
                  <div className="animate-spin rounded-full h-8 w-8 border-2 border-indigo-500 border-t-transparent" />
                </div>
              ) : (
                <div className="max-h-[600px] overflow-y-auto">
                  <PlayerPoolTable
                    players={filteredPool}
                    lockedIds={lockedIds}
                    bannedIds={bannedIds}
                    overrides={overrides}
                    showOverrides={showOverrideColumns}
                    sortKey={sortKey}
                    sortDir={sortDir}
                    onToggleSort={(key) => {
                      if (sortKey === key) {
                        setSortDir(prev => prev === 'asc' ? 'desc' : 'asc')
                      } else {
                        setSortKey(key)
                        setSortDir(key === 'name' || key === 'team' ? 'asc' : 'desc')
                      }
                    }}
                    onToggleLock={toggleLock}
                    onToggleBan={toggleBan}
                    onUpdateOverride={(playerId, field, value) => {
                      updateOverrideFields(playerId, { [field]: value })
                      scheduleOverrideSave()
                    }}
                    getModelFppm={getModelFppm}
                  />
                </div>
              )}
            </div>

            {/* Lineups View */}
            {lineups.length > 0 && (
              <div className={activeView !== 'lineups' ? 'hidden lg:block lg:mt-8' : 'mt-4 lg:mt-8'}>
                {/* Lineups Header */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                  <h2 className="text-lg font-semibold text-slate-100">
                    Lineups <span className="text-slate-500 font-normal">({filteredLineups.length} of {lineups.length})</span>
                  </h2>
                  <div className="flex items-center gap-2 flex-wrap">
                    <Input
                      type="text"
                      placeholder="Filter by player..."
                      value={lineupFilter}
                      onChange={(e) => setLineupFilter(e.target.value)}
                      className="w-32 sm:w-40"
                    />
                    <select
                      value={lineupSort}
                      onChange={(e) => setLineupSort(e.target.value as typeof lineupSort)}
                      className="px-2 py-1.5 text-sm bg-slate-900 border border-slate-700 rounded-lg text-slate-100"
                    >
                      <option value="default">Original</option>
                      <option value="proj-desc">Proj ↓</option>
                      <option value="proj-asc">Proj ↑</option>
                      <option value="p90-desc">p90 ↓</option>
                      <option value="own-asc">Own% ↑</option>
                    </select>
                    <span className="text-xs text-slate-500">
                      {selectedLineupIds.size} selected
                    </span>
                    <Button size="sm" variant="ghost" onClick={selectAllFiltered} disabled={filteredLineups.length === 0}>
                      Select All
                    </Button>
                    <Button size="sm" variant="ghost" onClick={clearSelection} disabled={selectedLineupIds.size === 0}>
                      Clear
                    </Button>
                  </div>
                </div>

                {/* Lineup Cards Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {filteredLineups.slice(0, showCount).map((lineup, idx) => (
                    <LineupCardNew
                      key={lineup.lineup_id}
                      lineup={lineup}
                      index={idx}
                      playerMap={playerMap}
                      isSelected={selectedLineupIds.has(lineup.lineup_id)}
                      onToggleSelect={() => toggleLineupSelection(lineup.lineup_id)}
                      filterText={lineupFilter}
                      getEffectiveProj={getEffectiveProj}
                    />
                  ))}
                </div>

                {/* Load More */}
                {filteredLineups.length > showCount && (
                  <div className="flex justify-center mt-4">
                    <Button
                      variant="secondary"
                      onClick={() => setShowCount(prev => prev + 50)}
                    >
                      Load More ({filteredLineups.length - showCount} remaining)
                    </Button>
                  </div>
                )}
              </div>
            )}
          </main>
        </div>
      </div>

      {/* Mobile Settings Drawer */}
      <Drawer
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        title="Build Settings"
        position="bottom"
      >
        <SettingsPanel
          maxPool={maxPool}
          setMaxPool={setMaxPool}
          builds={builds}
          setBuilds={setBuilds}
          minUniq={minUniq}
          setMinUniq={setMinUniq}
          maxExposurePct={maxExposurePct}
          setMaxExposurePct={setMaxExposurePct}
          nearDupJaccard={nearDupJaccard}
          setNearDupJaccard={setNearDupJaccard}
          globalTeamLimit={globalTeamLimit}
          setGlobalTeamLimit={setGlobalTeamLimit}
          minSalary={minSalary}
          setMinSalary={setMinSalary}
          maxSalary={maxSalary}
          setMaxSalary={setMaxSalary}
          minProj={minProj}
          setMinProj={setMinProj}
          maxOffoptimalPct={maxOffoptimalPct}
          setMaxOffoptimalPct={setMaxOffoptimalPct}
          randomnessPct={randomnessPct}
          setRandomnessPct={setRandomnessPct}
          hasStddev={hasStddev}
          useUserOverrides={useUserOverrides}
          setUseUserOverrides={setUseUserOverrides}
          lateSwapEnabled={lateSwapEnabled}
          setLateSwapEnabled={setLateSwapEnabled}
          worldSampleEnabled={worldSampleEnabled}
          setWorldSampleEnabled={setWorldSampleEnabled}
          games={currentSlateGames}
          excludedGames={excludedGames}
          setExcludedGames={setExcludedGames}
          lockedCount={lockedIds.size}
          bannedCount={bannedIds.size}
          onStartBuild={handleStartBuild}
          isBuilding={currentJob?.status === 'running'}
          canBuild={!!selectedSlate && !poolLoading}
        />
      </Drawer>

      {/* Mobile FAB for Generate */}
      <div className="lg:hidden fixed bottom-4 right-4 z-30">
        <Button
          variant="primary"
          size="lg"
          onClick={handleStartBuild}
          disabled={!selectedSlate || poolLoading || currentJob?.status === 'running'}
          loading={currentJob?.status === 'running'}
          className="shadow-lg"
        >
          {currentJob?.status === 'running' ? 'Building...' : 'Generate'}
        </Button>
      </div>
    </div>
  )
}
