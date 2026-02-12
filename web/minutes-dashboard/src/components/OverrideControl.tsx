import React from 'react'
import { OverrideMode, PlayerOverrideState } from '../api/gameview_v2'

type OverrideControlProps = {
    value?: PlayerOverrideState
    baselineMinutes: number
    resolvedMinutes: number
    onChange: (next: PlayerOverrideState) => void
    compact?: boolean
}

const MODES: OverrideMode[] = [
    'none',
    'lock',
    'band',
    'cap',
    'floor',
    'zero',
    'force_active',
    'force_inactive',
]

const normalizeMinutes = (n: number) => Number(Math.max(0, Math.min(48, n)).toFixed(1))

const nextForMode = (
    mode: OverrideMode,
    current: PlayerOverrideState | undefined,
    resolved: number,
    baselineMinutes: number,
): PlayerOverrideState => {
    if (mode === 'none') return { mode: 'none' }
    if (mode === 'lock') return { mode: 'lock', lock_value: normalizeMinutes(current?.lock_value ?? resolved) }
    if (mode === 'band') {
        const center = current?.lock_value ?? resolved
        return {
            mode: 'band',
            min_value: normalizeMinutes((current?.min_value ?? center) - 2),
            max_value: normalizeMinutes((current?.max_value ?? center) + 2),
        }
    }
    if (mode === 'cap') return { mode: 'cap', cap_value: normalizeMinutes(current?.cap_value ?? Math.max(0, resolved)) }
    if (mode === 'floor') return { mode: 'floor', floor_value: normalizeMinutes(current?.floor_value ?? Math.max(0, baselineMinutes * 0.5)) }
    if (mode === 'zero') return { mode: 'zero' }
    if (mode === 'force_active') return { mode: 'force_active' }
    return { mode: 'force_inactive' }
}

export const OverrideControl: React.FC<OverrideControlProps> = ({
    value,
    baselineMinutes,
    resolvedMinutes,
    onChange,
    compact = false,
}) => {
    const current = value ?? { mode: 'none' as OverrideMode }

    const setField = (patch: Partial<PlayerOverrideState>) => {
        onChange({ ...current, ...patch })
    }

    const setMode = (mode: OverrideMode) => {
        onChange(nextForMode(mode, current, resolvedMinutes, baselineMinutes))
    }

    return (
        <div className={`gv2-override-control ${compact ? 'compact' : ''}`} onClick={(e) => e.stopPropagation()}>
            <div className="gv2-override-row">
                <select
                    className="gv2-select"
                    value={current.mode}
                    onChange={(e) => setMode(e.target.value as OverrideMode)}
                >
                    {MODES.map((mode) => (
                        <option key={mode} value={mode}>
                            {mode}
                        </option>
                    ))}
                </select>

                {current.mode === 'lock' && (
                    <input
                        type="number"
                        className="gv2-input"
                        step={0.5}
                        min={0}
                        max={48}
                        value={current.lock_value ?? ''}
                        onChange={(e) => setField({ lock_value: e.target.value ? normalizeMinutes(Number(e.target.value)) : null })}
                    />
                )}

                {current.mode === 'band' && (
                    <>
                        <input
                            type="number"
                            className="gv2-input"
                            step={0.5}
                            min={0}
                            max={48}
                            value={current.min_value ?? ''}
                            onChange={(e) => setField({ min_value: e.target.value ? normalizeMinutes(Number(e.target.value)) : null })}
                        />
                        <input
                            type="number"
                            className="gv2-input"
                            step={0.5}
                            min={0}
                            max={48}
                            value={current.max_value ?? ''}
                            onChange={(e) => setField({ max_value: e.target.value ? normalizeMinutes(Number(e.target.value)) : null })}
                        />
                    </>
                )}

                {current.mode === 'cap' && (
                    <input
                        type="number"
                        className="gv2-input"
                        step={0.5}
                        min={0}
                        max={48}
                        value={current.cap_value ?? ''}
                        onChange={(e) => setField({ cap_value: e.target.value ? normalizeMinutes(Number(e.target.value)) : null })}
                    />
                )}

                {current.mode === 'floor' && (
                    <input
                        type="number"
                        className="gv2-input"
                        step={0.5}
                        min={0}
                        max={48}
                        value={current.floor_value ?? ''}
                        onChange={(e) => setField({ floor_value: e.target.value ? normalizeMinutes(Number(e.target.value)) : null })}
                    />
                )}
            </div>

            {!compact && (
                <div className="gv2-quick-actions">
                    <button type="button" onClick={() => onChange({ mode: 'zero' })}>0</button>
                    <button type="button" onClick={() => onChange({ mode: 'lock', lock_value: normalizeMinutes(resolvedMinutes) })}>Lock</button>
                    <button
                        type="button"
                        onClick={() =>
                            onChange({
                                mode: 'band',
                                min_value: normalizeMinutes(resolvedMinutes - 2),
                                max_value: normalizeMinutes(resolvedMinutes + 2),
                            })
                        }
                    >
                        Band ±2
                    </button>
                    <button type="button" onClick={() => onChange({ mode: 'cap', cap_value: 34 })}>Cap 34</button>
                    <button type="button" onClick={() => onChange({ mode: 'cap', cap_value: 28 })}>Cap 28</button>
                </div>
            )}
        </div>
    )
}
