import { Card, Badge, StatBadge, Checkbox } from '../ui'
import type { PoolPlayerWithOverrides, PlayerOverride } from '../../api/optimizer'
import { formatSalary } from '../../utils'

interface PlayerCardProps {
  player: PoolPlayerWithOverrides
  isLocked: boolean
  isBanned: boolean
  override?: PlayerOverride
  showOverrides: boolean
  onToggleLock: () => void
  onToggleBan: () => void
  onUpdateOverride?: (field: 'minutes' | 'fpts' | 'own' | 'is_out', value: number | boolean | null) => void
}

export function PlayerCard({
  player,
  isLocked,
  isBanned,
  override,
  showOverrides,
  onToggleLock,
  onToggleBan,
  onUpdateOverride,
}: PlayerCardProps) {
  const isOut = override?.is_out ?? false
  const hasOverride = override && (override.is_out || override.minutes != null || override.fpts != null || override.own != null)

  const getStatusStyles = () => {
    if (isOut) return 'border-slate-500/30 bg-slate-700/30 opacity-60'
    if (isLocked) return 'border-green-500/50 bg-green-500/5'
    if (isBanned) return 'border-red-500/50 bg-red-500/5 opacity-70'
    if (hasOverride) return 'border-blue-500/50 bg-blue-500/5'
    return ''
  }

  const value = (player.proj / (player.salary / 1000)).toFixed(2)

  return (
    <Card
      padding="sm"
      className={`
        relative
        ${getStatusStyles()}
        transition-all duration-150
      `}
    >
      {/* Header Row: Name + Position + Status */}
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex-1 min-w-0">
          <div className={`font-semibold text-slate-100 truncate ${isOut || isBanned ? 'line-through' : ''}`}>
            {player.name}
          </div>
          <div className="flex items-center gap-1.5 mt-0.5">
            <span className="text-xs text-slate-500">{player.team}</span>
            <span className="text-xs text-slate-500">·</span>
            <span className="text-xs text-slate-400">{player.positions.join('/')}</span>
          </div>
        </div>

        {/* Status Badges */}
        <div className="flex items-center gap-1">
          {isLocked && <Badge variant="success" size="sm">Locked</Badge>}
          {isBanned && <Badge variant="danger" size="sm">Banned</Badge>}
          {isOut && <Badge variant="default" size="sm">Out</Badge>}
        </div>
      </div>

      {/* Stats Row */}
      <div className="flex items-center gap-3 py-2 border-y border-slate-700/50">
        <StatBadge label="$" value={formatSalary(player.salary)} variant="salary" />
        <StatBadge label="Proj" value={player.proj?.toFixed(1) ?? '-'} variant="projection" />
        <StatBadge label="Own" value={player.own_proj ? `${player.own_proj.toFixed(1)}%` : '-'} variant="ownership" />
        <StatBadge label="Val" value={value} variant="ceiling" />
      </div>

      {/* Override Section (when enabled) */}
      {showOverrides && (
        <div className="mt-2 pt-2 space-y-2">
          <div className="grid grid-cols-3 gap-2">
            <div>
              <label className="block text-[10px] text-slate-500 uppercase mb-0.5">Min</label>
              <input
                type="number"
                className="w-full px-2 py-1 text-xs bg-slate-900 border border-slate-700 rounded focus:ring-1 focus:ring-indigo-500 focus:border-transparent"
                placeholder="—"
                step={0.5}
                min={0}
                max={48}
                value={override?.minutes ?? ''}
                onChange={(e) => onUpdateOverride?.('minutes', e.target.value === '' ? null : Number(e.target.value))}
              />
            </div>
            <div>
              <label className="block text-[10px] text-slate-500 uppercase mb-0.5">FPTS</label>
              <input
                type="number"
                className="w-full px-2 py-1 text-xs bg-slate-900 border border-slate-700 rounded focus:ring-1 focus:ring-indigo-500 focus:border-transparent"
                placeholder="—"
                step={0.5}
                min={0}
                value={override?.fpts ?? ''}
                onChange={(e) => onUpdateOverride?.('fpts', e.target.value === '' ? null : Number(e.target.value))}
              />
            </div>
            <div>
              <label className="block text-[10px] text-slate-500 uppercase mb-0.5">Own%</label>
              <input
                type="number"
                className="w-full px-2 py-1 text-xs bg-slate-900 border border-slate-700 rounded focus:ring-1 focus:ring-indigo-500 focus:border-transparent"
                placeholder="—"
                step={0.5}
                min={0}
                max={100}
                value={override?.own ?? ''}
                onChange={(e) => onUpdateOverride?.('own', e.target.value === '' ? null : Number(e.target.value))}
              />
            </div>
          </div>
        </div>
      )}

      {/* Actions Row */}
      <div className="flex items-center justify-between mt-3 pt-2 border-t border-slate-700/50">
        <div className="flex items-center gap-3">
          <Checkbox
            variant="lock"
            checked={isLocked}
            onChange={onToggleLock}
            title="Lock player"
          />
          <span className="text-xs text-slate-500">Lock</span>

          <Checkbox
            variant="ban"
            checked={isBanned}
            onChange={onToggleBan}
            title="Ban player"
          />
          <span className="text-xs text-slate-500">Ban</span>
        </div>

        {showOverrides && (
          <Checkbox
            checked={isOut}
            onChange={(e) => onUpdateOverride?.('is_out', (e.target as HTMLInputElement).checked)}
            label="Out"
          />
        )}
      </div>
    </Card>
  )
}
