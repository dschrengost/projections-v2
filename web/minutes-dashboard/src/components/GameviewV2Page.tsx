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
    initialGameId?: string | null
    onGameChange?: (gameId: string) => void
    onRefresh?: () => void
    onOpenLateSwap?: () => void
    onRunCompleted?: (runId: string | null) => void
}

type ResolvedPlayer = ApplyOverridesResponse['resolved_players'][number]

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

const toBandOnlyOverride = (override: PlayerOverrideState): PlayerOverrideState => {
    const mode = override.mode
    if (mode === 'none') return { mode: 'none', protect_weight: override.protect_weight }

    const asNum = (value: unknown): number | null => {
        if (value == null) return null
        const parsed = Number(value)
        return Number.isFinite(parsed) ? Math.max(0, Math.min(48, parsed)) : null
    }

    let min = asNum(override.min_value)
    let max = asNum(override.max_value)

    if (mode === 'lock') {
        const lock = asNum(override.lock_value)
        if (lock != null) {
            min = lock
            max = lock
        }
    } else if (mode === 'cap') {
        const cap = asNum(override.cap_value)
        min = 0
        max = cap ?? 48
    } else if (mode === 'floor') {
        const floor = asNum(override.floor_value)
        min = floor ?? 0
        max = 48
    } else if (mode === 'zero' || mode === 'force_inactive') {
        min = 0
        max = 0
    } else if (mode === 'force_active') {
        return { mode: 'none', protect_weight: override.protect_weight }
    }

    if (min == null && max == null) {
        return { mode: 'none', protect_weight: override.protect_weight }
    }
    if (min == null) min = Math.max(0, Math.min(48, max ?? 48))
    if (max == null) max = Math.max(0, Math.min(48, min))

    const lb = Math.min(min, max)
    const ub = Math.max(min, max)
    return {
        mode: 'band',
        min_value: Number(lb.toFixed(1)),
        max_value: Number(ub.toFixed(1)),
        protect_weight: override.protect_weight,
    }
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

export const GameviewV2Page: React.FC<GameviewV2PageProps> = ({
    rows,
    date,
    runId,
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
    const [lastAppliedAt, setLastAppliedAt] = useState<string | null>(null)
    const [lastRunId, setLastRunId] = useState<string | null>(null)
    const [lastGamesByKey, setLastGamesByKey] = useState<Record<string, PlayerLastGameStat[]>>({})
    const [lastGamesLoadingByKey, setLastGamesLoadingByKey] = useState<Record<string, boolean>>({})
    const [lastGamesErrorByKey, setLastGamesErrorByKey] = useState<Record<string, string | null>>({})

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
        if (Object.prototype.hasOwnProperty.call(lastGamesByKey, cacheKey) || lastGamesLoadingByKey[cacheKey]) return

        let cancelled = false
        setLastGamesLoadingByKey((prev) => ({ ...prev, [cacheKey]: true }))
        setLastGamesErrorByKey((prev) => ({ ...prev, [cacheKey]: null }))

        const load = async () => {
            try {
                const response = await fetchPlayerLastGames(date, selectedPlayerId, 5)
                if (cancelled) return
                setLastGamesByKey((prev) => ({ ...prev, [cacheKey]: response.games || [] }))
            } catch (err) {
                if (cancelled) return
                setLastGamesErrorByKey((prev) => ({ ...prev, [cacheKey]: (err as Error).message }))
            } finally {
                if (!cancelled) {
                    setLastGamesLoadingByKey((prev) => ({ ...prev, [cacheKey]: false }))
                }
            }
        }

        void load()
        return () => {
            cancelled = true
        }
    }, [selectedPlayerId, date, lastGamesByKey, lastGamesLoadingByKey])

    const activeGame = useMemo(() => games.find((game) => game.gameId === activeGameId) || null, [games, activeGameId])

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
            ...toBandOnlyOverride(override),
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
            const minutesP10 = toMaybeNum(
                player.sim_minutes_sim_p10_uncond ?? player.sim_minutes_sim_p10 ?? player.minutes_p10_cond ?? player.minutes_p10,
            )
            const minutesP50 = toMaybeNum(
                player.sim_minutes_sim_p50_uncond ?? player.sim_minutes_sim_p50 ?? player.minutes_p50,
            )
            const minutesP90 = toMaybeNum(
                player.sim_minutes_sim_p90_uncond ?? player.sim_minutes_sim_p90 ?? player.minutes_p90_cond ?? player.minutes_p90,
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
                minutesP10,
                minutesP50,
                minutesP90,
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
        const minutesP10 = toMaybeNum(
            player.sim_minutes_sim_p10_uncond ?? player.sim_minutes_sim_p10 ?? player.minutes_p10_cond ?? player.minutes_p10,
        )
        const minutesP50 = toMaybeNum(
            player.sim_minutes_sim_p50_uncond ?? player.sim_minutes_sim_p50 ?? player.minutes_p50,
        )
        const minutesP90 = toMaybeNum(
            player.sim_minutes_sim_p90_uncond ?? player.sim_minutes_sim_p90 ?? player.minutes_p90_cond ?? player.minutes_p90,
        )

        const baselineFpts = toMaybeNum(player.fpts_sim_uncond_mean ?? player.sim_dk_fpts_mean ?? player.proj_fpts)
        const resolvedFpts = scaleByMinutes(baselineFpts, baselineMinutes, resolvedMinutes)

        const baselinePts = toMaybeNum(player.sim_pts_mean)
        const baselineReb = toMaybeNum(player.sim_reb_mean)
        const baselineAst = toMaybeNum(player.sim_ast_mean)
        const baselineStl = toMaybeNum(player.sim_stl_mean)
        const baselineBlk = toMaybeNum(player.sim_blk_mean)
        const baselineTo = toMaybeNum(player.sim_tov_mean)

        const ratioScaled = (value: number | null) => scaleByMinutes(value, baselineMinutes, resolvedMinutes)

        const teamDiag = getTeamDiagnostics(toId(player.team_id))
        const reasons: string[] = []
        if (teamDiag?.hit_floor_player_ids?.includes(selectedPlayerId)) reasons.push('binding floor')
        if (teamDiag?.hit_cap_player_ids?.includes(selectedPlayerId)) reasons.push('binding cap')
        if (override.mode !== 'none') reasons.push('minutes band active')
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
                    p10: minutesP10,
                    p50: minutesP50 ?? resolvedMinutes,
                    p90: minutesP90,
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
                    <h2>Gameview v2 Patch Layer</h2>
                    <div className="muted">Slate {date} · Time to lock {formatCountdown(activeGame?.startTime)}</div>
                </div>
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
            </header>

            <div className="gv2-status-line">
                {lastAppliedAt ? <span>Last applied: {new Date(lastAppliedAt).toLocaleString()}</span> : <span>No applied overrides yet</span>}
                {lastRunId ? <span>Last run id: {lastRunId}</span> : null}
                {legacyCount > 0 ? <span className="warning">Legacy fields present on {legacyCount} players (read-only in v2 UI)</span> : null}
                {message ? <span>{message}</span> : null}
                {error ? <span className="error">{error}</span> : null}
            </div>

            <GameTabs tabs={tabs} activeGameId={activeGameId} onChange={onTabChange} />

            {activeGame ? (
                <div className="gv2-teams-grid">
                    <TeamPanel
                        teamName={activeGame.awayTeam.teamName}
                        diagnostics={getTeamDiagnostics(activeGame.awayTeam.teamId)}
                        rows={teamRows(activeGame.awayTeam)}
                        onSelectPlayer={(playerId) => setSelectedPlayerId(playerId)}
                        onOverrideChange={(playerId, next) => setOverride(activeGame.gameId, playerId, next)}
                    />
                    <TeamPanel
                        teamName={activeGame.homeTeam.teamName}
                        diagnostics={getTeamDiagnostics(activeGame.homeTeam.teamId)}
                        rows={teamRows(activeGame.homeTeam)}
                        onSelectPlayer={(playerId) => setSelectedPlayerId(playerId)}
                        onOverrideChange={(playerId, next) => setOverride(activeGame.gameId, playerId, next)}
                    />
                </div>
            ) : null}

            <PlayerDetailsDrawer
                open={Boolean(selectedDrawerPlayer)}
                player={selectedDrawerPlayer}
                onClose={() => setSelectedPlayerId(null)}
                onOverrideChange={(playerId, next) => setOverride(activeGameId, playerId, next)}
            />
        </div>
    )
}
