import { Card, Badge, Checkbox } from '../ui'
import type { LineupRow, PoolPlayerWithOverrides } from '../../api/optimizer'

interface LineupCardNewProps {
  lineup: LineupRow
  index: number
  playerMap: Map<string, PoolPlayerWithOverrides>
  isSelected: boolean
  onToggleSelect: () => void
  filterText?: string
  getEffectiveProj: (player: PoolPlayerWithOverrides | undefined) => number
}

export function LineupCardNew({
  lineup,
  index,
  playerMap,
  isSelected,
  onToggleSelect,
  filterText = '',
  getEffectiveProj,
}: LineupCardNewProps) {
  const totalSalary = lineup.player_ids.reduce(
    (sum, id) => sum + (playerMap.get(id)?.salary ?? 0),
    0
  )
  const totalProj = lineup.player_ids.reduce(
    (sum, id) => sum + getEffectiveProj(playerMap.get(id)),
    0
  )
  const p90 = lineup.p90 ?? lineup.player_ids.reduce(
    (sum, id) => sum + (playerMap.get(id)?.p90 ?? 0),
    0
  )
  const totalOwn = lineup.player_ids.reduce(
    (sum, id) => sum + (playerMap.get(id)?.own_proj ?? 0),
    0
  )

  const filterLower = filterText.toLowerCase()

  return (
    <Card
      padding="sm"
      selected={isSelected}
      hover
      onClick={onToggleSelect}
      className="cursor-pointer"
    >
      {/* Header */}
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-2">
          <Checkbox
            checked={isSelected}
            onChange={(e) => {
              e.stopPropagation()
              onToggleSelect()
            }}
          />
          <span className="text-sm font-medium text-slate-400">#{index + 1}</span>
        </div>

        {/* Stats badges */}
        <div className="flex items-center gap-2 text-xs">
          <Badge variant="salary" size="sm">${totalSalary.toLocaleString()}</Badge>
          <Badge variant="projection" size="sm">{totalProj.toFixed(1)}</Badge>
          {p90 > 0 && <Badge variant="ceiling" size="sm">p90: {p90.toFixed(1)}</Badge>}
        </div>
      </div>

      {/* Players */}
      <div className="flex flex-wrap gap-1.5">
        {lineup.player_ids.map((id) => {
          const player = playerMap.get(id)
          if (!player) return null

          const isHighlighted = filterLower && player.name.toLowerCase().includes(filterLower)

          return (
            <span
              key={id}
              className={`
                inline-flex items-center gap-1 px-2 py-1
                text-xs rounded-md
                ${isHighlighted
                  ? 'bg-indigo-600/20 text-indigo-500 border border-indigo-500/30'
                  : 'bg-slate-700/50 text-slate-400'
                }
              `}
            >
              <span className="font-medium text-slate-100">{player.name}</span>
              <span className="text-slate-500">({player.positions[0]})</span>
            </span>
          )
        })}
      </div>

      {/* Footer stats */}
      <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-700/50 text-xs text-slate-500">
        <span>Own: <span className="text-violet-400 font-medium">{totalOwn.toFixed(1)}%</span></span>
        <span>Remaining: <span className="text-amber-400 font-medium">${(50000 - totalSalary).toLocaleString()}</span></span>
      </div>
    </Card>
  )
}
