import type { PoolPlayerWithOverrides, PlayerOverride } from '../../api/optimizer'
import { formatSalary } from '../../utils'

type SortKey = 'name' | 'team' | 'salary' | 'proj' | 'own_proj' | 'value'

interface PlayerPoolTableProps {
  players: PoolPlayerWithOverrides[]
  lockedIds: Set<string>
  bannedIds: Set<string>
  overrides: Map<string, PlayerOverride>
  showOverrides: boolean
  sortKey: SortKey
  sortDir: 'asc' | 'desc'
  onToggleSort: (key: SortKey) => void
  onToggleLock: (id: string) => void
  onToggleBan: (id: string) => void
  onUpdateOverride: (playerId: string, field: 'minutes' | 'fpts' | 'own' | 'is_out', value: number | boolean | null) => void
  getModelFppm: (player: PoolPlayerWithOverrides) => number
}

export function PlayerPoolTable({
  players,
  lockedIds,
  bannedIds,
  overrides,
  showOverrides,
  sortKey,
  sortDir,
  onToggleSort,
  onToggleLock,
  onToggleBan,
  onUpdateOverride,
  getModelFppm,
}: PlayerPoolTableProps) {
  const SortHeader = ({ label, sortKeyName }: { label: string; sortKeyName: SortKey }) => (
    <th
      onClick={() => onToggleSort(sortKeyName)}
      className="px-2 py-2 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider cursor-pointer hover:text-indigo-400 transition-colors whitespace-nowrap"
    >
      <span className="inline-flex items-center gap-1">
        {label}
        {sortKey === sortKeyName && (
          <span className="text-indigo-400">{sortDir === 'asc' ? '▲' : '▼'}</span>
        )}
      </span>
    </th>
  )

  return (
    <div className="overflow-x-auto bg-slate-800 border border-slate-700 rounded-xl">
      <table className="w-full text-sm">
        <thead className="bg-slate-700/30 sticky top-0 z-10">
          <tr>
            <th className="px-2 py-2 text-center text-xs font-semibold text-slate-400 uppercase w-10">
              <span title="Lock">🔒</span>
            </th>
            <th className="px-2 py-2 text-center text-xs font-semibold text-slate-400 uppercase w-10">
              <span title="Ban">🚫</span>
            </th>
            <SortHeader label="Player" sortKeyName="name" />
            <SortHeader label="Team" sortKeyName="team" />
            <th className="px-2 py-2 text-left text-xs font-semibold text-slate-400 uppercase">Pos</th>
            <SortHeader label="Salary" sortKeyName="salary" />
            {showOverrides && (
              <>
                <th className="px-2 py-2 text-right text-xs font-semibold text-slate-500 uppercase whitespace-nowrap">Model</th>
                <th className="px-2 py-2 text-center text-xs font-semibold text-blue-400 uppercase whitespace-nowrap">Min</th>
                <th className="px-2 py-2 text-center text-xs font-semibold text-blue-400 uppercase whitespace-nowrap">FPTS</th>
                <th className="px-2 py-2 text-center text-xs font-semibold text-blue-400 uppercase whitespace-nowrap">Own</th>
                <th className="px-2 py-2 text-center text-xs font-semibold text-slate-400 uppercase w-10">Out</th>
              </>
            )}
            <SortHeader label="Proj" sortKeyName="proj" />
            <SortHeader label="Own%" sortKeyName="own_proj" />
            <SortHeader label="Value" sortKeyName="value" />
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-700/50">
          {players.map((player) => {
            const isLocked = lockedIds.has(player.player_id)
            const isBanned = bannedIds.has(player.player_id)
            const override = overrides.get(player.player_id)
            const isOut = override?.is_out ?? false
            const hasOverride = override && (override.is_out || override.minutes != null || override.fpts != null || override.own != null)

            const rowClasses = [
              'transition-colors',
              isLocked ? 'bg-green-500/10' : '',
              isBanned ? 'bg-red-500/10 opacity-60' : '',
              isOut ? 'bg-slate-600/50 opacity-50' : '',
              hasOverride && !isLocked && !isBanned && !isOut ? 'bg-blue-500/5' : '',
              !isLocked && !isBanned && !isOut && !hasOverride ? 'hover:bg-slate-700/30' : '',
            ].filter(Boolean).join(' ')

            const textClass = isOut || isBanned ? 'line-through' : ''
            const value = (player.proj / (player.salary / 1000)).toFixed(2)

            return (
              <tr key={player.player_id} className={rowClasses}>
                {/* Lock */}
                <td className="px-2 py-1.5 text-center">
                  <input
                    type="checkbox"
                    checked={isLocked}
                    onChange={() => onToggleLock(player.player_id)}
                    className="w-4 h-4 rounded border-green-500/50 bg-slate-900 text-green-500 focus:ring-green-500 cursor-pointer"
                    title="Lock player"
                  />
                </td>
                {/* Ban */}
                <td className="px-2 py-1.5 text-center">
                  <input
                    type="checkbox"
                    checked={isBanned}
                    onChange={() => onToggleBan(player.player_id)}
                    className="w-4 h-4 rounded border-red-500/50 bg-slate-900 text-red-500 focus:ring-red-500 cursor-pointer"
                    title="Ban player"
                  />
                </td>
                {/* Name */}
                <td className={`px-2 py-1.5 font-medium text-slate-100 ${textClass}`}>
                  {player.name}
                </td>
                {/* Team */}
                <td className={`px-2 py-1.5 text-slate-400 ${textClass}`}>
                  {player.team}
                </td>
                {/* Positions */}
                <td className="px-2 py-1.5 text-slate-500 text-xs">
                  {player.positions.join('/')}
                </td>
                {/* Salary */}
                <td className={`px-2 py-1.5 text-amber-400 font-medium ${textClass}`}>
                  {formatSalary(player.salary)}
                </td>

                {showOverrides && (
                  <>
                    {/* Model FPTS */}
                    <td className="px-2 py-1.5 text-right text-slate-500 text-xs">
                      {(player.model_proj ?? player.proj)?.toFixed(1)}
                    </td>
                    {/* Override Minutes */}
                    <td className="px-1 py-1">
                      <input
                        type="number"
                        className={`w-14 px-1.5 py-1 text-xs text-center rounded border bg-slate-900 focus:ring-1 focus:ring-indigo-500 focus:border-transparent ${
                          override?.minutes != null ? 'border-blue-500 bg-blue-500/10 text-blue-400' : 'border-slate-600 text-slate-100'
                        }`}
                        placeholder="—"
                        step={0.5}
                        min={0}
                        max={48}
                        value={override?.minutes ?? ''}
                        onChange={(e) => {
                          const minutes = e.target.value === '' ? null : Number(e.target.value)
                          if (minutes == null) {
                            onUpdateOverride(player.player_id, 'minutes', null)
                            onUpdateOverride(player.player_id, 'fpts', null)
                          } else {
                            const fppm = getModelFppm(player)
                            const fpts = Number((minutes * fppm).toFixed(1))
                            onUpdateOverride(player.player_id, 'minutes', minutes)
                            onUpdateOverride(player.player_id, 'fpts', fpts)
                          }
                        }}
                      />
                    </td>
                    {/* Override FPTS */}
                    <td className="px-1 py-1">
                      <input
                        type="number"
                        className={`w-14 px-1.5 py-1 text-xs text-center rounded border bg-slate-900 focus:ring-1 focus:ring-indigo-500 focus:border-transparent ${
                          override?.fpts != null ? 'border-blue-500 bg-blue-500/10 text-blue-400' : 'border-slate-600 text-slate-100'
                        }`}
                        placeholder="—"
                        step={0.5}
                        min={0}
                        value={override?.fpts ?? ''}
                        onChange={(e) => onUpdateOverride(player.player_id, 'fpts', e.target.value === '' ? null : Number(e.target.value))}
                      />
                    </td>
                    {/* Override Own */}
                    <td className="px-1 py-1">
                      <input
                        type="number"
                        className={`w-14 px-1.5 py-1 text-xs text-center rounded border bg-slate-900 focus:ring-1 focus:ring-indigo-500 focus:border-transparent ${
                          override?.own != null ? 'border-blue-500 bg-blue-500/10 text-blue-400' : 'border-slate-600 text-slate-100'
                        }`}
                        placeholder="—"
                        step={0.5}
                        min={0}
                        max={100}
                        value={override?.own ?? ''}
                        onChange={(e) => onUpdateOverride(player.player_id, 'own', e.target.value === '' ? null : Number(e.target.value))}
                      />
                    </td>
                    {/* Out checkbox */}
                    <td className="px-2 py-1.5 text-center">
                      <input
                        type="checkbox"
                        checked={isOut}
                        onChange={(e) => onUpdateOverride(player.player_id, 'is_out', e.target.checked)}
                        className="w-4 h-4 rounded border-slate-600 bg-slate-900 cursor-pointer"
                        title="Mark player out"
                      />
                    </td>
                  </>
                )}

                {/* Projection */}
                <td className={`px-2 py-1.5 text-green-400 font-semibold ${textClass}`}>
                  {player.proj?.toFixed(1) ?? '—'}
                </td>
                {/* Ownership */}
                <td className={`px-2 py-1.5 text-violet-400 ${textClass}`}>
                  {player.own_proj != null ? `${player.own_proj.toFixed(1)}%` : '—'}
                </td>
                {/* Value */}
                <td className={`px-2 py-1.5 text-cyan-400 font-medium ${textClass}`}>
                  {value}
                </td>
              </tr>
            )
          })}
          {players.length === 0 && (
            <tr>
              <td colSpan={showOverrides ? 14 : 9} className="px-4 py-8 text-center text-slate-500">
                No players found
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
