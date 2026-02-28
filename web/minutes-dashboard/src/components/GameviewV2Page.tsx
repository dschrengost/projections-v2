import React, { useEffect, useMemo, useRef, useState } from 'react'
import {
    ApplyOverridesResponse,
    OverrideMode,
    PlayerLastGameStat,
    PlayerOverrideState,
    TeamDiagnostics,
    applyOverrides,
    fetchPlayerLastGames,
    fetchOverrideState,
    pollRun,
    runWorldsWithOverrides,
} from '../api/gameview_v2'
import { PlayerRow } from '../types'
import { GameTabs } from './GameTabs'
import { PlayerDetailsDrawer, PlayerDrawerData } from './PlayerDetailsDrawer'
import { PlayerTableRow } from './PlayerTable'
import { TeamPanel } from './TeamPanel'
import './gameview-v2.css'

type GameviewV2PageProps = {
    rows: PlayerRow[]
    date: string
    runId?: string | null
    readOnly?: boolean
    initialGameId?: string | null
    onGameChange?: (gameId: string) => void
    onRefresh?: () => void
    onOpenLateSwap?: () => void
    onRunCompleted?: (runId: string | null) => void
}

type ResolvedPlayer = ApplyOverridesResponse['resolved_players'][number]

type MinutesBandSource = 'base' | 'sim_uncond' | 'sim_cond'

type TeamData = {
    teamId: string
    teamName: string
    players: PlayerRow[]
}

type GameData = {
    gameId: string
    label: string
    startTime?: string
    awayTeam: TeamData
    homeTeam: TeamData
}

const toId = (value: unknown) => String(value ?? '')

const toNum = (value: unknown, fallback = 0): number => {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : fallback
}

const toMaybeNum = (value: unknown): number | null => {
    if (value == null) return null
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
}

const pickNum = (...values: unknown[]): number | null => {
    for (const value of values) {
        const parsed = Number(value)
        if (Number.isFinite(parsed)) return parsed
    }
    return null
}

const resolveUncondPts = (player: PlayerRow): number => {
    const direct = pickNum(player.sim_pts_mean_uncond, player.pts_mean_uncond)
    if (direct != null) return direct
    const cond = pickNum(player.sim_pts_mean, player.pts_mean)
    if (cond == null) return 0
    const pActive = pickNum(player.minutes_sim_p_active, player.play_prob_eff, player.p_play_eff, player.play_prob)
    return pActive == null ? cond : cond * pActive
}

const minutesBandLabel = (source: MinutesBandSource): string => {
    if (source === 'sim_uncond') return 'Sim (uncond)'
    if (source === 'sim_cond') return 'Sim (cond)'
    return 'Base (model)'
}

const resolveMinutesBand = (player: PlayerRow, source: MinutesBandSource) => {
    const base = {
        p10: pickNum(player.minutes_p10),
        p50: pickNum(player.minutes_p50),
        p90: pickNum(player.minutes_p90),
    }
    const simUncond = {
        p10: pickNum(player.minutes_sim_uncond_p10, player.sim_minutes_sim_p10_uncond),
        p50: pickNum(player.minutes_sim_uncond_p50, player.sim_minutes_sim_p50_uncond),
        p90: pickNum(player.minutes_sim_uncond_p90, player.sim_minutes_sim_p90_uncond),
    }
    const simCond = {
        p10: pickNum(player.sim_minutes_sim_p10),
        p50: pickNum(player.sim_minutes_sim_p50),
        p90: pickNum(player.sim_minutes_sim_p90),
    }

    if (source === 'sim_uncond') {
        return {
            p10: simUncond.p10 ?? base.p10,
            p50: simUncond.p50 ?? base.p50,
            p90: simUncond.p90 ?? base.p90,
        }
    }
    if (source === 'sim_cond') {
        return {
            p10: simCond.p10 ?? base.p10,
            p50: simCond.p50 ?? base.p50,
            p90: simCond.p90 ?? base.p90,
        }
    }
    return {
        p10: base.p10 ?? simUncond.p10 ?? simCond.p10,
        p50: base.p50 ?? simUncond.p50 ?? simCond.p50,
        p90: base.p90 ?? simUncond.p90 ?? simCond.p90,
    }
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

const inferAwayHome = (teams: TeamData[], gameRows: PlayerRow[]): { away: TeamData; home: TeamData } => {
    if (teams.length !== 2) {
        const sorted = [...teams].sort((a, b) => a.teamName.localeCompare(b.teamName))
        return { away: sorted[0], home: sorted[1] }
    }

    const sample = gameRows[0]
    const total = toMaybeNum(sample?.total)
    const spreadHome = toMaybeNum(sample?.spread_home)

    if (total != null && spreadHome != null) {
        const homeImplied = total / 2 - spreadHome / 2
        const dist = (team: TeamData) => {
            const implied = toMaybeNum(team.players[0]?.team_implied_total)
            if (implied == null) return Number.POSITIVE_INFINITY
            return Math.abs(implied - homeImplied)
        }
        const [a, b] = teams
        const aDist = dist(a)
        const bDist = dist(b)
        if (Number.isFinite(aDist) && Number.isFinite(bDist) && aDist !== bDist) {
            return aDist < bDist ? { away: b, home: a } : { away: a, home: b }
        }
    }

    const sorted = [...teams].sort((a, b) => a.teamName.localeCompare(b.teamName))
    return { away: sorted[0], home: sorted[1] }
}

const deriveGames = (rows: PlayerRow[]): GameData[] => {
    const byGame = new Map<string, PlayerRow[]>()
    rows.forEach((row) => {
        const gameId = toId(row.game_id)
        if (!gameId) return
        const arr = byGame.get(gameId) ?? []
        arr.push(row)
        byGame.set(gameId, arr)
    })

    const games: GameData[] = []
    byGame.forEach((gameRows, gameId) => {
        const byTeam = new Map<string, TeamData>()
        gameRows.forEach((row) => {
            const teamId = toId(row.team_id)
            if (!teamId) return
            const existing = byTeam.get(teamId)
            if (existing) {
                existing.players.push(row)
                return
            }
            byTeam.set(teamId, {
                teamId,
                teamName: row.team_tricode || row.team_name || teamId,
                players: [row],
            })
        })

        const teams = Array.from(byTeam.values())
        if (teams.length < 2) return
        const { away, home } = inferAwayHome(teams.slice(0, 2), gameRows)
        games.push({
            gameId,
            label: `${away.teamName} @ ${home.teamName}`,
            startTime: gameRows[0]?.tip_ts,
            awayTeam: away,
            homeTeam: home,
        })
    })

    return games.sort((a, b) => (a.startTime || '').localeCompare(b.startTime || ''))
}

const defaultOverride = (): PlayerOverrideState => ({ mode: 'none' })

const toTargetOnlyOverride = (override: PlayerOverrideState): PlayerOverrideState => {
    const mode = override.mode ?? 'none'
    const protect_weight = override.protect_weight
    if (mode === 'none') return { mode: 'none', protect_weight }
    if (mode === 'zero' || mode === 'force_inactive') return { mode: 'zero', protect_weight }

    const asNum = (value: unknown): number | null => {
        if (value == null) return null
        const parsed = Number(value)
        return Number.isFinite(parsed) ? Math.max(0, Math.min(48, parsed)) : null
    }

    const lock = asNum(override.lock_value)
    if (mode === 'lock' && lock != null) {
        if (lock <= 0) return { mode: 'zero', protect_weight }
        return { mode: 'lock', lock_value: Number(lock.toFixed(1)), protect_weight }
    }

    const min = asNum(override.min_value)
    const max = asNum(override.max_value)
    if (min != null && max != null) {
        const mid = 0.5 * (min + max)
        if (mid <= 0) return { mode: 'zero', protect_weight }
        return { mode: 'lock', lock_value: Number(mid.toFixed(1)), protect_weight }
    }

    const cap = asNum(override.cap_value)
    const floor = asNum(override.floor_value)
    const fallback = lock ?? cap ?? floor
    if (fallback == null) return { mode: 'none', protect_weight }
    if (fallback <= 0) return { mode: 'zero', protect_weight }
    return { mode: 'lock', lock_value: Number(fallback.toFixed(1)), protect_weight }
}

const overrideFromServer = (item: { mode: OverrideMode; fields: Record<string, unknown> }): PlayerOverrideState => {
    const fields = item.fields || {}
    const meanLb = toMaybeNum(fields['mean_lb_minutes'] ?? fields['lb_minutes'])
    const meanUb = toMaybeNum(fields['mean_ub_minutes'] ?? fields['ub_minutes'])
    const worldUb = toMaybeNum(fields['world_ub_minutes'] ?? fields['ub_minutes'])
    return {
        mode: item.mode,
        lock_value: meanLb,
        min_value: meanLb,
        max_value: meanUb,
        cap_value: worldUb,
        floor_value: meanLb,
        protect_weight: Boolean(fields['protect_weight']),
    }
}

const scaleByMinutes = (stat: number | null, baselineMinutes: number, resolvedMinutes: number): number | null => {
    if (stat == null) return null
    if (baselineMinutes <= 1e-6) return stat
    return stat * (resolvedMinutes / baselineMinutes)
}

const shiftMinutesBandToResolved = (band: { p10: number | null; p50: number | null; p90: number | null }, resolved: number, fallbackCenter: number) => {
    const clamp = (n: number) => Math.max(0, Math.min(48, n))
    const center = band.p50 ?? fallbackCenter
    const delta = resolved - center
    const shiftedP50 = clamp(resolved)
    const shiftedP10 = band.p10 == null ? null : clamp(band.p10 + delta)
    const shiftedP90 = band.p90 == null ? null : clamp(band.p90 + delta)
    return {
        p10: shiftedP10 == null ? null : Math.min(shiftedP10, shiftedP50),
        p50: shiftedP50,
        p90: shiftedP90 == null ? null : Math.max(shiftedP90, shiftedP50),
    }
}

export const GameviewV2Page: React.FC<GameviewV2PageProps> = ({
    rows,
    date,
    runId,
    readOnly = false,
    initialGameId,
    onGameChange,
    onRefresh,
    onOpenLateSwap,
    onRunCompleted,
}) => {
    const games = useMemo(() => deriveGames(rows), [rows])
    const [activeGameId, setActiveGameId] = useState<string>(() => initialGameId || games[0]?.gameId || '')
    const [selectedPlayerId, setSelectedPlayerId] = useState<string | null>(null)

    const [localOverrides, setLocalOverrides] = useState<Record<string, Record<string, PlayerOverrideState>>>({})
    const [savedOverrides, setSavedOverrides] = useState<Record<string, Record<string, PlayerOverrideState>>>({})
    const [legacyFieldsByGame, setLegacyFieldsByGame] = useState<Record<string, Record<string, string[]>>>({})

    const [resolvedByGame, setResolvedByGame] = useState<Record<string, Record<string, ResolvedPlayer>>>({})
    const [teamDiagByGame, setTeamDiagByGame] = useState<Record<string, TeamDiagnostics[]>>({})

    const [isApplying, setIsApplying] = useState(false)
    const [isRunning, setIsRunning] = useState(false)
    const [message, setMessage] = useState<string | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [bandSource, setBandSource] = useState<MinutesBandSource>('base')
    const [lastAppliedAt, setLastAppliedAt] = useState<string | null>(null)
    const [lastRunId, setLastRunId] = useState<string | null>(null)
    const [lastGamesByKey, setLastGamesByKey] = useState<Record<string, PlayerLastGameStat[]>>({})
    const [lastGamesLoadingByKey, setLastGamesLoadingByKey] = useState<Record<string, boolean>>({})
    const [lastGamesErrorByKey, setLastGamesErrorByKey] = useState<Record<string, string | null>>({})
    const lastGamesInFlight = useRef<Set<string>>(new Set())

    const hydratedGames = useRef<Set<string>>(new Set())
    const [clockNow, setClockNow] = useState(Date.now())

    useEffect(() => {
        const timer = window.setInterval(() => setClockNow(Date.now()), 15000)
        return () => window.clearInterval(timer)
    }, [])

    useEffect(() => {
        if (!games.length) {
            setActiveGameId('')
            return
        }
        if (initialGameId && games.some((g) => g.gameId === initialGameId)) {
            setActiveGameId(initialGameId)
            return
        }
        if (!games.some((g) => g.gameId === activeGameId)) {
            setActiveGameId(games[0].gameId)
        }
    }, [games, initialGameId, activeGameId])

    useEffect(() => {
        if (!activeGameId || hydratedGames.current.has(activeGameId)) return
        let cancelled = false

        const load = async () => {
            try {
                const response = await fetchOverrideState(date, activeGameId)
                if (cancelled) return
                const nextLocal: Record<string, PlayerOverrideState> = {}
                const nextLegacy: Record<string, string[]> = {}
                response.overrides.forEach((item) => {
                    if (Object.keys(item.fields || {}).length > 0) {
                        nextLocal[item.player_id] = overrideFromServer({ mode: item.mode, fields: item.fields })
                    }
                    if (item.legacy_fields_present?.length) {
                        nextLegacy[item.player_id] = item.legacy_fields_present
                    }
                })

                setSavedOverrides((prev) => ({ ...prev, [activeGameId]: nextLocal }))
                setLocalOverrides((prev) => {
                    if (Object.prototype.hasOwnProperty.call(prev, activeGameId)) return prev
                    return { ...prev, [activeGameId]: nextLocal }
                })
                setLegacyFieldsByGame((prev) => ({ ...prev, [activeGameId]: nextLegacy }))
                hydratedGames.current.add(activeGameId)
            } catch (err) {
                if (!cancelled) {
                    setError((err as Error).message)
                }
            }
        }

        void load()
        return () => {
            cancelled = true
        }
    }, [activeGameId, date])

    useEffect(() => {
        if (!selectedPlayerId) return
        const cacheKey = `${date}:${selectedPlayerId}`
        if (Object.prototype.hasOwnProperty.call(lastGamesByKey, cacheKey)) return
        if (lastGamesInFlight.current.has(cacheKey)) return

        lastGamesInFlight.current.add(cacheKey)
        setLastGamesLoadingByKey((prev) => ({ ...prev, [cacheKey]: true }))
        setLastGamesErrorByKey((prev) => ({ ...prev, [cacheKey]: null }))

        const load = async () => {
            try {
                const response = await fetchPlayerLastGames(date, selectedPlayerId, 5)
                setLastGamesByKey((prev) => ({ ...prev, [cacheKey]: response.games || [] }))
            } catch (err) {
                setLastGamesErrorByKey((prev) => ({ ...prev, [cacheKey]: (err as Error).message }))
            } finally {
                lastGamesInFlight.current.delete(cacheKey)
                setLastGamesLoadingByKey((prev) => ({ ...prev, [cacheKey]: false }))
            }
        }

        void load()
    }, [selectedPlayerId, date, lastGamesByKey])

    const activeGame = useMemo(() => games.find((game) => game.gameId === activeGameId) || null, [games, activeGameId])
    const activeGameTotals = useMemo(() => {
        if (!activeGame) return null

        const teamSummary = (team: TeamData) => {
            const simPtsUncond = team.players.reduce((acc, player) => {
                return acc + resolveUncondPts(player)
            }, 0)

            const sample = team.players[0]
            let vegasImplied = pickNum(sample?.team_implied_total)
            if (vegasImplied == null) {
                const total = pickNum(sample?.total)
                const spreadHome = pickNum(sample?.spread_home)
                if (total != null && spreadHome != null) {
                    const homeImplied = total / 2 - spreadHome / 2
                    vegasImplied = team.teamId === activeGame.homeTeam.teamId ? homeImplied : (total - homeImplied)
                }
            }
            return {
                teamName: team.teamName,
                simPtsUncond,
                vegasImplied,
                delta: vegasImplied == null ? null : simPtsUncond - vegasImplied,
            }
        }

        const away = teamSummary(activeGame.awayTeam)
        const home = teamSummary(activeGame.homeTeam)
        const simGameTotal = away.simPtsUncond + home.simPtsUncond
        const vegasGameTotal = pickNum(
            activeGame.awayTeam.players[0]?.total,
            activeGame.homeTeam.players[0]?.total,
        )
        return {
            away,
            home,
            simGameTotal,
            vegasGameTotal,
            gameDelta: vegasGameTotal == null ? null : simGameTotal - vegasGameTotal,
        }
    }, [activeGame])

    const setOverride = (gameId: string, playerId: string, next: PlayerOverrideState) => {
        setLocalOverrides((prev) => ({
            ...prev,
            [gameId]: {
                ...(prev[gameId] || {}),
                [playerId]: next,
            },
        }))
    }

    const applyForGame = async (targetGameId: string): Promise<ApplyOverridesResponse> => {
        const overrides = localOverrides[targetGameId] || {}
        const payload = Object.entries(overrides).map(([playerId, override]) => ({
            player_id: playerId,
            ...toTargetOnlyOverride(override),
        }))

        const response = await applyOverrides({
            date,
            gameId: targetGameId,
            runId,
            overrideInfeasible: 'error',
            overrides: payload,
        })

        const resolvedMap: Record<string, ResolvedPlayer> = {}
        response.resolved_players.forEach((player) => {
            resolvedMap[toId(player.player_id)] = player
        })

        const nextSaved: Record<string, PlayerOverrideState> = {}
        const nextLegacy: Record<string, string[]> = {}
        response.overrides.forEach((item) => {
            if (Object.keys(item.fields || {}).length > 0) {
                nextSaved[item.player_id] = overrideFromServer({ mode: item.mode, fields: item.fields })
            }
            if (item.legacy_fields_present?.length) {
                nextLegacy[item.player_id] = item.legacy_fields_present
            }
        })

        setSavedOverrides((prev) => ({ ...prev, [targetGameId]: nextSaved }))
        setLocalOverrides((prev) => ({ ...prev, [targetGameId]: nextSaved }))
        setLegacyFieldsByGame((prev) => ({ ...prev, [targetGameId]: nextLegacy }))
        setResolvedByGame((prev) => ({ ...prev, [targetGameId]: resolvedMap }))
        setTeamDiagByGame((prev) => ({ ...prev, [targetGameId]: response.team_diagnostics || [] }))
        setLastAppliedAt(response.applied_at)
        return response
    }

    const onApply = async () => {
        if (!activeGameId) return
        setIsApplying(true)
        setError(null)
        setMessage(null)
        try {
            const response = await applyForGame(activeGameId)
            setMessage(`Applied v2 overrides at ${new Date(response.applied_at).toLocaleTimeString()}`)
        } catch (err) {
            setError((err as Error).message)
        } finally {
            setIsApplying(false)
        }
    }

    const onApplyAndRun = async () => {
        if (!activeGameId) return
        setIsRunning(true)
        setError(null)
        setMessage(null)
        try {
            await applyForGame(activeGameId)
            const run = await runWorldsWithOverrides({
                date,
                gameId: activeGameId,
                baseRunId: runId,
                pin: true,
                background: true,
                minutesOverrideMode: 'v2',
                overrideInfeasible: 'error',
            })
            if (!run.run_ts) {
                throw new Error('Missing run token for worlds poll.')
            }

            setMessage('Worlds run started. Polling status...')

            let pollResult = null
            for (let attempt = 0; attempt < 180; attempt += 1) {
                await sleep(2000)
                const status = await pollRun(date, run.run_ts)
                if (status.done) {
                    pollResult = status
                    break
                }
            }
            if (!pollResult) {
                throw new Error('Timed out waiting for worlds run to complete.')
            }
            if (!pollResult.ok) {
                throw new Error(pollResult.status?.message || 'Worlds run failed.')
            }

            setLastRunId(pollResult.projections_run_id ?? null)
            setMessage('Apply & Run Worlds completed.')
            if (onRunCompleted) {
                onRunCompleted(pollResult.projections_run_id ?? null)
            } else if (onRefresh) {
                onRefresh()
            }
        } catch (err) {
            setError((err as Error).message)
        } finally {
            setIsRunning(false)
        }
    }

    const onRevert = () => {
        if (!activeGameId) return
        setLocalOverrides((prev) => ({
            ...prev,
            [activeGameId]: savedOverrides[activeGameId] || {},
        }))
        setMessage('Local edits reverted.')
        setError(null)
    }

    const onTabChange = (gameId: string) => {
        setActiveGameId(gameId)
        setSelectedPlayerId(null)
        onGameChange?.(gameId)
    }

    const formatCountdown = (startTime?: string) => {
        if (!startTime) return 'n/a'
        const tipMs = new Date(startTime).getTime()
        if (!Number.isFinite(tipMs)) return 'n/a'
        const diffMs = tipMs - clockNow
        if (diffMs <= 0) return 'Locked'
        const totalMinutes = Math.floor(diffMs / 60000)
        const hours = Math.floor(totalMinutes / 60)
        const minutes = totalMinutes % 60
        return `${hours}h ${minutes}m`
    }

    const teamRows = (team: TeamData): PlayerTableRow[] => {
        const gameId = activeGameId
        const overrideMap = localOverrides[gameId] || {}
        const resolvedMap = resolvedByGame[gameId] || {}

        const out = team.players.map((player) => {
            const playerId = toId(player.player_id)
            const baselineMinutes = toNum(player.minutes_final ?? player.minutes_p50 ?? player.minutes_sim_uncond_mean)
            const resolvedInfo = resolvedMap[playerId]
            const resolvedMinutes = toNum(resolvedInfo?.mu_minutes, baselineMinutes)
            const baselineFpts = toMaybeNum(player.fpts_sim_uncond_mean ?? player.sim_dk_fpts_mean ?? player.proj_fpts)
            const resolvedFpts = scaleByMinutes(baselineFpts, baselineMinutes, resolvedMinutes)
            const band = resolveMinutesBand(player, bandSource)
            const shiftedBand = shiftMinutesBandToResolved(
                {
                    p10: toMaybeNum(band.p10),
                    p50: toMaybeNum(band.p50),
                    p90: toMaybeNum(band.p90),
                },
                resolvedMinutes,
                baselineMinutes,
            )

            const override = overrideMap[playerId] || defaultOverride()

            return {
                game_id: gameId,
                team_id: toId(player.team_id),
                player_id: playerId,
                name: player.player_name || playerId,
                pos: undefined,
                status: player.status,
                isProjectedStarter: Boolean(player.is_projected_starter),
                isConfirmedStarter: Boolean(player.is_confirmed_starter),
                baselineMinutes,
                resolvedMinutes,
                baselineFpts,
                resolvedFpts,
                minutesP10: shiftedBand.p10,
                minutesP50: shiftedBand.p50,
                minutesP90: shiftedBand.p90,
                override,
            }
        })

        return out.sort((a, b) => b.resolvedMinutes - a.resolvedMinutes)
    }

    const getTeamDiagnostics = (teamId: string): TeamDiagnostics | null => {
        const list = teamDiagByGame[activeGameId] || []
        return list.find((diag) => toId(diag.team_id) === teamId) || null
    }

    const selectedDrawerPlayer: PlayerDrawerData | null = useMemo(() => {
        if (!activeGame || !selectedPlayerId) return null
        const allPlayers = [...activeGame.awayTeam.players, ...activeGame.homeTeam.players]
        const player = allPlayers.find((p) => toId(p.player_id) === selectedPlayerId)
        if (!player) return null

        const gameOverride = localOverrides[activeGameId] || {}
        const override = gameOverride[selectedPlayerId] || defaultOverride()
        const resolvedInfo = (resolvedByGame[activeGameId] || {})[selectedPlayerId]

        const baselineMinutes = toNum(player.minutes_final ?? player.minutes_p50 ?? player.minutes_sim_uncond_mean)
        const resolvedMinutes = toNum(resolvedInfo?.mu_minutes, baselineMinutes)
        const band = resolveMinutesBand(player, bandSource)
        const shiftedBand = shiftMinutesBandToResolved(
            {
                p10: toMaybeNum(band.p10),
                p50: toMaybeNum(band.p50),
                p90: toMaybeNum(band.p90),
            },
            resolvedMinutes,
            baselineMinutes,
        )

        const baselineFpts = toMaybeNum(player.fpts_sim_uncond_mean ?? player.sim_dk_fpts_mean ?? player.proj_fpts)
        const resolvedFpts = scaleByMinutes(baselineFpts, baselineMinutes, resolvedMinutes)

        const baselinePts = pickNum(
            player.sim_pts_mean_uncond,
            player.pts_mean_uncond,
            player.sim_pts_mean,
            player.pts_mean,
        )
        const baselineReb = pickNum(player.sim_reb_mean_uncond, player.reb_mean_uncond, player.sim_reb_mean, player.reb_mean)
        const baselineAst = pickNum(player.sim_ast_mean_uncond, player.ast_mean_uncond, player.sim_ast_mean, player.ast_mean)
        const baselineStl = pickNum(player.sim_stl_mean_uncond, player.stl_mean_uncond, player.sim_stl_mean, player.stl_mean)
        const baselineBlk = pickNum(player.sim_blk_mean_uncond, player.blk_mean_uncond, player.sim_blk_mean, player.blk_mean)
        const baselineTo = pickNum(player.sim_tov_mean_uncond, player.tov_mean_uncond, player.sim_tov_mean, player.tov_mean)

        const ratioScaled = (value: number | null) => scaleByMinutes(value, baselineMinutes, resolvedMinutes)

        const teamDiag = getTeamDiagnostics(toId(player.team_id))
        const reasons: string[] = []
        if (teamDiag?.hit_floor_player_ids?.includes(selectedPlayerId)) reasons.push('binding floor')
        if (teamDiag?.hit_cap_player_ids?.includes(selectedPlayerId)) reasons.push('binding cap')
        if (override.mode !== 'none') reasons.push('target minutes active')
        if (teamDiag?.infeasibility_reason) reasons.push(`team infeasible: ${teamDiag.infeasibility_reason}`)
        const historyKey = `${date}:${selectedPlayerId}`
        const lastGames = lastGamesByKey[historyKey] || []
        const lastGamesLoading = Boolean(lastGamesLoadingByKey[historyKey])
        const lastGamesError = lastGamesErrorByKey[historyKey] ?? null

        return {
            player_id: selectedPlayerId,
            name: player.player_name || selectedPlayerId,
            team: player.team_tricode || player.team_name || toId(player.team_id),
            status: player.status,
            isProjectedStarter: Boolean(player.is_projected_starter),
            isConfirmedStarter: Boolean(player.is_confirmed_starter),
            baselineMinutes,
            resolvedMinutes,
            override,
            whyChanged: reasons.length ? reasons.join(' · ') : null,
            metrics: {
                minutes: {
                    baseline: baselineMinutes,
                    resolved: resolvedMinutes,
                    p10: shiftedBand.p10,
                    p50: shiftedBand.p50,
                    p90: shiftedBand.p90,
                },
                fpts: {
                    baseline: baselineFpts,
                    resolved: resolvedFpts,
                    p10: ratioScaled(toMaybeNum(player.sim_dk_fpts_p10)),
                    p50: ratioScaled(toMaybeNum(player.sim_dk_fpts_p50 ?? player.fpts_sim_uncond_p50)),
                    p90: ratioScaled(toMaybeNum(player.sim_dk_fpts_p90 ?? player.fpts_sim_uncond_p90)),
                },
                pts: { baseline: baselinePts, resolved: ratioScaled(baselinePts) },
                reb: { baseline: baselineReb, resolved: ratioScaled(baselineReb) },
                ast: { baseline: baselineAst, resolved: ratioScaled(baselineAst) },
                stl: { baseline: baselineStl, resolved: ratioScaled(baselineStl) },
                blk: { baseline: baselineBlk, resolved: ratioScaled(baselineBlk) },
                to: { baseline: baselineTo, resolved: ratioScaled(baselineTo) },
            },
            lastGames,
            lastGamesLoading,
            lastGamesError,
            minutesBandLabel: minutesBandLabel(bandSource),
        }
    }, [
        activeGame,
        selectedPlayerId,
        localOverrides,
        activeGameId,
        resolvedByGame,
        teamDiagByGame,
        date,
        lastGamesByKey,
        lastGamesLoadingByKey,
        lastGamesErrorByKey,
        bandSource,
    ])

    if (!games.length) {
        return <div className="muted">No games available for {date}.</div>
    }

    const tabs = games.map((game) => ({ game_id: game.gameId, label: game.label }))
    const activeLegacyConflicts = legacyFieldsByGame[activeGameId] || {}
    const legacyCount = Object.keys(activeLegacyConflicts).length

    return (
        <div className="gv2-root">
            <header className="gv2-header">
                <div>
                    <h2>{readOnly ? 'Gameview (Read-only)' : 'Gameview v2 Patch Layer'}</h2>
                    <div className="muted">Slate {date} · Time to lock {formatCountdown(activeGame?.startTime)}</div>
                </div>
                {!readOnly ? (
                    <div className="gv2-actions">
                        <button type="button" onClick={onApply} disabled={isApplying || isRunning}>
                            {isApplying ? 'Applying...' : 'Apply'}
                        </button>
                        <button type="button" onClick={onApplyAndRun} disabled={isApplying || isRunning}>
                            {isRunning ? 'Running...' : 'Apply & Run Worlds'}
                        </button>
                        <button type="button" onClick={() => onOpenLateSwap?.()}>
                            Export Late Swap
                        </button>
                        <button type="button" onClick={onRevert} disabled={isApplying || isRunning}>
                            Revert
                        </button>
                    </div>
                ) : (
                    <div className="gv2-readonly-pill">Overrides disabled</div>
                )}
            </header>

            <div className="gv2-status-line">
                {!readOnly ? (
                    <>
                        {lastAppliedAt ? <span>Last applied: {new Date(lastAppliedAt).toLocaleString()}</span> : <span>No applied overrides yet</span>}
                        {lastRunId ? <span>Last run id: {lastRunId}</span> : null}
                        {legacyCount > 0 ? <span className="warning">Legacy fields present on {legacyCount} players (read-only in v2 UI)</span> : null}
                    </>
                ) : (
                    <span>Read-only projection view (unconditional stat counts).</span>
                )}
                {message ? <span>{message}</span> : null}
                {error ? <span className="error">{error}</span> : null}
            </div>

            {!readOnly ? (
                <div className="gv2-controls">
                    <label>
                        Minutes tails
                        <select value={bandSource} onChange={(event) => setBandSource(event.target.value as MinutesBandSource)}>
                            <option value="base">{minutesBandLabel('base')}</option>
                            <option value="sim_uncond">{minutesBandLabel('sim_uncond')}</option>
                            <option value="sim_cond">{minutesBandLabel('sim_cond')}</option>
                        </select>
                    </label>
                    <span className="muted">Source for displayed p10/p50/p90 (shifted to match resolved μ).</span>
                </div>
            ) : null}

            <GameTabs tabs={tabs} activeGameId={activeGameId} onChange={onTabChange} />

            {activeGameTotals ? (
                <section className="gv2-game-totals">
                    <div className="gv2-game-totals-header">Implied Totals: Sim (Uncond) vs Vegas</div>
                    <div className="gv2-game-totals-grid">
                        <div className="gv2-game-total-card">
                            <div className="label">{activeGameTotals.away.teamName}</div>
                            <div className="line">Sim: {activeGameTotals.away.simPtsUncond.toFixed(1)}</div>
                            <div className="line">Vegas: {activeGameTotals.away.vegasImplied == null ? '-' : activeGameTotals.away.vegasImplied.toFixed(1)}</div>
                            <div className={`delta ${(activeGameTotals.away.delta ?? 0) >= 0 ? 'up' : 'down'}`}>
                                Δ {activeGameTotals.away.delta == null ? '-' : activeGameTotals.away.delta.toFixed(1)}
                            </div>
                        </div>
                        <div className="gv2-game-total-card">
                            <div className="label">{activeGameTotals.home.teamName}</div>
                            <div className="line">Sim: {activeGameTotals.home.simPtsUncond.toFixed(1)}</div>
                            <div className="line">Vegas: {activeGameTotals.home.vegasImplied == null ? '-' : activeGameTotals.home.vegasImplied.toFixed(1)}</div>
                            <div className={`delta ${(activeGameTotals.home.delta ?? 0) >= 0 ? 'up' : 'down'}`}>
                                Δ {activeGameTotals.home.delta == null ? '-' : activeGameTotals.home.delta.toFixed(1)}
                            </div>
                        </div>
                        <div className="gv2-game-total-card">
                            <div className="label">Game Total</div>
                            <div className="line">Sim: {activeGameTotals.simGameTotal.toFixed(1)}</div>
                            <div className="line">Vegas: {activeGameTotals.vegasGameTotal == null ? '-' : activeGameTotals.vegasGameTotal.toFixed(1)}</div>
                            <div className={`delta ${(activeGameTotals.gameDelta ?? 0) >= 0 ? 'up' : 'down'}`}>
                                Δ {activeGameTotals.gameDelta == null ? '-' : activeGameTotals.gameDelta.toFixed(1)}
                            </div>
                        </div>
                    </div>
                </section>
            ) : null}

            {activeGame ? (
                <div className="gv2-teams-grid">
                    <TeamPanel
                        teamName={activeGame.awayTeam.teamName}
                        diagnostics={getTeamDiagnostics(activeGame.awayTeam.teamId)}
                        rows={teamRows(activeGame.awayTeam)}
                        minutesBandLabel={minutesBandLabel(bandSource)}
                        readOnly={readOnly}
                        onSelectPlayer={(playerId) => setSelectedPlayerId(playerId)}
                        onOverrideChange={(playerId, next) => setOverride(activeGame.gameId, playerId, next)}
                    />
                    <TeamPanel
                        teamName={activeGame.homeTeam.teamName}
                        diagnostics={getTeamDiagnostics(activeGame.homeTeam.teamId)}
                        rows={teamRows(activeGame.homeTeam)}
                        minutesBandLabel={minutesBandLabel(bandSource)}
                        readOnly={readOnly}
                        onSelectPlayer={(playerId) => setSelectedPlayerId(playerId)}
                        onOverrideChange={(playerId, next) => setOverride(activeGame.gameId, playerId, next)}
                    />
                </div>
            ) : null}

            <PlayerDetailsDrawer
                open={Boolean(selectedDrawerPlayer)}
                player={selectedDrawerPlayer}
                readOnly={readOnly}
                onClose={() => setSelectedPlayerId(null)}
                onOverrideChange={(playerId, next) => setOverride(activeGameId, playerId, next)}
            />
        </div>
    )
}
