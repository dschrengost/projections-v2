import React from 'react'
import { OverrideMode, PlayerOverrideState } from '../api/gameview_v2'

type OverrideControlProps = {
    value?: PlayerOverrideState
    baselineMinutes: number
    resolvedMinutes: number
    onChange: (next: PlayerOverrideState) => void
    compact?: boolean
}

const normalizeMinutes = (n: number) => Number(Math.max(0, Math.min(48, n)).toFixed(1))

const orderBand = (a: number, b: number) => {
    const min = normalizeMinutes(Math.min(a, b))
    const max = normalizeMinutes(Math.max(a, b))
    return { min, max }
}

const deriveBand = (
    current: PlayerOverrideState | undefined,
    resolvedMinutes: number,
    baselineMinutes: number,
): { min: number; max: number; sourceMode: OverrideMode } => {
    const mode = current?.mode ?? 'none'
    if (mode === 'band') {
        const min = normalizeMinutes(current?.min_value ?? Math.max(0, resolvedMinutes - 2))
        const max = normalizeMinutes(current?.max_value ?? Math.min(48, resolvedMinutes + 2))
        return { ...orderBand(min, max), sourceMode: mode }
    }
    if (mode === 'lock') {
        const lock = normalizeMinutes(current?.lock_value ?? resolvedMinutes)
        return { min: lock, max: lock, sourceMode: mode }
    }
    if (mode === 'cap') {
        const cap = normalizeMinutes(current?.cap_value ?? resolvedMinutes)
        return { ...orderBand(0, cap), sourceMode: mode }
    }
    if (mode === 'floor') {
        const floor = normalizeMinutes(current?.floor_value ?? Math.max(0, baselineMinutes * 0.5))
        return { ...orderBand(floor, 48), sourceMode: mode }
    }
    if (mode === 'zero' || mode === 'force_inactive') {
        return { min: 0, max: 0, sourceMode: mode }
    }
    return {
        ...orderBand(Math.max(0, resolvedMinutes - 2), Math.min(48, resolvedMinutes + 2)),
        sourceMode: mode,
    }
}

export const OverrideControl: React.FC<OverrideControlProps> = ({
    value,
    baselineMinutes,
    resolvedMinutes,
    onChange,
    compact = false,
}) => {
    const current = value ?? { mode: 'none' as OverrideMode }
    const band = deriveBand(current, resolvedMinutes, baselineMinutes)
    const isActive = current.mode !== 'none'

    const applyBand = (min: number, max: number) => {
        const ordered = orderBand(min, max)
        onChange({
            mode: 'band',
            min_value: ordered.min,
            max_value: ordered.max,
            protect_weight: current.protect_weight,
        })
    }

    const clearBand = () => {
        onChange({ mode: 'none', protect_weight: current.protect_weight })
    }

    return (
        <div className={`gv2-override-control ${compact ? 'compact' : ''}`} onClick={(e) => e.stopPropagation()}>
            <div className="gv2-override-row gv2-band-row">
                {!compact ? <span className="gv2-band-label">Band</span> : null}
                <input
                    type="number"
                    className="gv2-input"
                    step={0.5}
                    min={0}
                    max={48}
                    value={band.min}
                    onChange={(e) => {
                        if (!e.target.value) return
                        applyBand(Number(e.target.value), band.max)
                    }}
                    aria-label="Minimum mean minutes"
                />
                <span className="gv2-band-sep">to</span>
                <input
                    type="number"
                    className="gv2-input"
                    step={0.5}
                    min={0}
                    max={48}
                    value={band.max}
                    onChange={(e) => {
                        if (!e.target.value) return
                        applyBand(band.min, Number(e.target.value))
                    }}
                    aria-label="Maximum mean minutes"
                />
                {compact && isActive ? (
                    <button type="button" className="gv2-band-clear" onClick={clearBand} title="Clear band">
                        Clear
                    </button>
                ) : null}
            </div>

            {!compact && (
                <div className="gv2-quick-actions">
                    <button type="button" onClick={() => applyBand(resolvedMinutes - 2, resolvedMinutes + 2)}>μ ±2</button>
                    <button type="button" onClick={() => applyBand(resolvedMinutes, resolvedMinutes)}>Lock μ</button>
                    <button type="button" onClick={() => applyBand(0, 0)}>0</button>
                    <button type="button" onClick={clearBand}>Clear</button>
                </div>
            )}

            {!compact && band.sourceMode !== 'none' && band.sourceMode !== 'band' ? (
                <div className="muted gv2-band-note">
                    Existing `{band.sourceMode}` converted to band on edit.
                </div>
            ) : null}
        </div>
    )
}
