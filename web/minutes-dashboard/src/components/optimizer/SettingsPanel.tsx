import { Button, Input, Slider, Toggle, Card, CardTitle } from '../ui'
import type { GameInfo } from '../../api/optimizer'

interface SettingsPanelProps {
  // Build config
  maxPool: number
  setMaxPool: (v: number) => void
  builds: number
  setBuilds: (v: number) => void
  minUniq: number
  setMinUniq: (v: number) => void
  maxExposurePct: number
  setMaxExposurePct: (v: number) => void
  nearDupJaccard: number
  setNearDupJaccard: (v: number) => void
  globalTeamLimit: number
  setGlobalTeamLimit: (v: number) => void
  minSalary: number | null
  setMinSalary: (v: number | null) => void
  maxSalary: number | null
  setMaxSalary: (v: number | null) => void
  minProj: number | null
  setMinProj: (v: number | null) => void
  maxOffoptimalPct: number
  setMaxOffoptimalPct: (v: number) => void
  randomnessPct: number
  setRandomnessPct: (v: number) => void
  hasStddev: boolean

  // Toggles
  useUserOverrides: boolean
  setUseUserOverrides: (v: boolean) => void
  lateSwapEnabled: boolean
  setLateSwapEnabled: (v: boolean) => void
  worldSampleEnabled: boolean
  setWorldSampleEnabled: (v: boolean) => void

  // Game filters
  games: GameInfo[]
  excludedGames: Set<string>
  setExcludedGames: (fn: (prev: Set<string>) => Set<string>) => void

  // Lock/ban counts
  lockedCount: number
  bannedCount: number

  // Actions
  onStartBuild: () => void
  isBuilding: boolean
  canBuild: boolean
}

export function SettingsPanel({
  maxPool,
  setMaxPool,
  builds,
  setBuilds,
  minUniq,
  setMinUniq,
  maxExposurePct,
  setMaxExposurePct,
  nearDupJaccard,
  setNearDupJaccard,
  globalTeamLimit,
  setGlobalTeamLimit,
  minSalary,
  setMinSalary,
  maxSalary,
  setMaxSalary,
  minProj,
  setMinProj,
  maxOffoptimalPct,
  setMaxOffoptimalPct,
  randomnessPct,
  setRandomnessPct,
  hasStddev,
  useUserOverrides,
  setUseUserOverrides,
  lateSwapEnabled,
  setLateSwapEnabled,
  worldSampleEnabled,
  setWorldSampleEnabled,
  games,
  excludedGames,
  setExcludedGames,
  lockedCount,
  bannedCount,
  onStartBuild,
  isBuilding,
  canBuild,
}: SettingsPanelProps) {
  const toggleGameExclusion = (matchup: string) => {
    setExcludedGames((prev) => {
      const next = new Set(prev)
      if (next.has(matchup)) {
        next.delete(matchup)
      } else {
        next.add(matchup)
      }
      return next
    })
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Build Parameters */}
      <Card padding="md">
        <CardTitle className="mb-4">Build Settings</CardTitle>

        <div className="grid grid-cols-2 gap-3">
          <Input
            label="Max Lineups"
            type="number"
            value={maxPool}
            onChange={(e) => setMaxPool(Number(e.target.value))}
            min={100}
            max={100000}
            step={500}
          />
          <Input
            label="Workers"
            type="number"
            value={builds}
            onChange={(e) => setBuilds(Number(e.target.value))}
            min={1}
            max={24}
          />
          <Input
            label="Min Uniques"
            type="number"
            value={minUniq}
            onChange={(e) => setMinUniq(Number(e.target.value))}
            min={0}
            max={8}
          />
          <Input
            label="Team Limit"
            type="number"
            value={globalTeamLimit}
            onChange={(e) => setGlobalTeamLimit(Number(e.target.value))}
            min={1}
            max={8}
          />
        </div>

        <div className="mt-4 space-y-4">
          <Slider
            label="Max Exposure"
            value={maxExposurePct}
            onChange={(e) => setMaxExposurePct(Number(e.target.value))}
            min={0}
            max={100}
            step={5}
            valueFormatter={(v) => `${v}%`}
            hint={maxExposurePct === 0 ? 'Disabled' : undefined}
          />

          <Slider
            label="Near-Dup Jaccard"
            value={nearDupJaccard}
            onChange={(e) => setNearDupJaccard(Number(e.target.value))}
            min={0}
            max={1}
            step={0.05}
            valueFormatter={(v) => v.toFixed(2)}
            hint="Higher = stricter duplicate detection"
          />

          <Slider
            label="Max % Off Optimal"
            value={maxOffoptimalPct}
            onChange={(e) => setMaxOffoptimalPct(Number(e.target.value))}
            min={0}
            max={50}
            step={0.5}
            valueFormatter={(v) => `${v}%`}
            hint={maxOffoptimalPct === 0 ? 'Disabled' : undefined}
          />

          <Slider
            label="Randomness"
            value={randomnessPct}
            onChange={(e) => setRandomnessPct(Number(e.target.value))}
            min={0}
            max={100}
            step={5}
            valueFormatter={(v) => `${v}%`}
            hint={!hasStddev ? 'Requires sim stddev' : undefined}
            disabled={!hasStddev}
          />
        </div>
      </Card>

      {/* Salary & Projection Filters */}
      <Card padding="md">
        <CardTitle className="mb-4">Filters</CardTitle>
        <div className="grid grid-cols-2 gap-3">
          <Input
            label="Min Salary"
            type="number"
            value={minSalary ?? ''}
            onChange={(e) => setMinSalary(e.target.value ? Number(e.target.value) : null)}
            placeholder="None"
            min={0}
            max={50000}
            step={100}
          />
          <Input
            label="Max Salary"
            type="number"
            value={maxSalary ?? ''}
            onChange={(e) => setMaxSalary(e.target.value ? Number(e.target.value) : null)}
            placeholder="None"
            min={0}
            max={50000}
            step={100}
          />
          <Input
            label="Min Projection"
            type="number"
            value={minProj ?? ''}
            onChange={(e) => setMinProj(e.target.value ? Number(e.target.value) : null)}
            placeholder="None"
            min={0}
            step={5}
            className="col-span-2"
          />
        </div>
      </Card>

      {/* Mode Toggles */}
      <Card padding="md">
        <CardTitle className="mb-4">Mode</CardTitle>
        <div className="space-y-3">
          <Toggle
            checked={useUserOverrides}
            onChange={setUseUserOverrides}
            label={useUserOverrides ? 'My Projections' : 'Model Projections'}
            description={useUserOverrides ? 'Using your manual overrides' : 'Using model projections'}
          />
          <Toggle
            checked={lateSwapEnabled}
            onChange={setLateSwapEnabled}
            label={lateSwapEnabled ? 'Late Swap Mode' : 'Standard Mode'}
            description={lateSwapEnabled ? 'Favors later games' : 'No late swap preference'}
          />
          <Toggle
            checked={worldSampleEnabled}
            onChange={setWorldSampleEnabled}
            label={worldSampleEnabled ? 'World Sample Mode' : 'Mean Projections'}
            description={worldSampleEnabled ? 'Optimizes against sampled worlds' : 'Uses mean projections'}
          />
        </div>
      </Card>

      {/* Game Filters */}
      {games.length > 0 && (
        <Card padding="md">
          <div className="flex items-center justify-between mb-3">
            <CardTitle>
              Games ({games.length - excludedGames.size}/{games.length})
            </CardTitle>
            {excludedGames.size > 0 && (
              <button
                onClick={() => setExcludedGames(() => new Set())}
                className="text-xs text-indigo-500 hover:underline"
              >
                Include All
              </button>
            )}
          </div>
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {games.map((game) => {
              const isExcluded = excludedGames.has(game.matchup)
              const startTime = game.start_time
                ? new Date(game.start_time).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
                : ''
              return (
                <label
                  key={game.matchup}
                  className={`
                    flex items-center gap-2 px-2 py-1.5 rounded-md cursor-pointer
                    hover:bg-slate-700/50 transition-colors
                    ${isExcluded ? 'opacity-50' : ''}
                  `}
                >
                  <input
                    type="checkbox"
                    checked={!isExcluded}
                    onChange={() => toggleGameExclusion(game.matchup)}
                    className="w-3.5 h-3.5 rounded border-slate-700 bg-slate-900"
                  />
                  <span className={`flex-1 text-sm font-medium ${isExcluded ? 'line-through text-slate-500' : 'text-slate-100'}`}>
                    {game.matchup}
                  </span>
                  {startTime && (
                    <span className="text-xs text-slate-500">{startTime}</span>
                  )}
                </label>
              )
            })}
          </div>
        </Card>
      )}

      {/* Lock/Ban Summary */}
      <div className="flex items-center justify-center gap-4 text-sm text-slate-400 py-2">
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-green-500" />
          Locked: {lockedCount}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
          Banned: {bannedCount}
        </span>
      </div>

      {/* Build Button */}
      <Button
        variant="primary"
        size="lg"
        fullWidth
        onClick={onStartBuild}
        disabled={!canBuild}
        loading={isBuilding}
      >
        {isBuilding ? 'Building...' : 'Generate Lineups'}
      </Button>
    </div>
  )
}
