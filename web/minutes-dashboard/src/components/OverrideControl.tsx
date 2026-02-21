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

const toMaybeNum = (value: unknown): number | null => {
    if (value == null) return null
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
}

const deriveTarget = (current: PlayerOverrideState | undefined): { target: number | null; sourceMode: OverrideMode } => {
    const mode = current?.mode ?? 'none'
    if (mode === 'none') return { target: null, sourceMode: mode }
    if (mode === 'zero' || mode === 'force_inactive') return { target: 0, sourceMode: mode }
    if (mode === 'lock') {
        const lock = toMaybeNum(current?.lock_value)
        return { target: lock == null ? null : normalizeMinutes(lock), sourceMode: mode }
    }
    if (mode === 'band') {
        const min = toMaybeNum(current?.min_value)
        const max = toMaybeNum(current?.max_value)
        if (min != null && max != null) {
            return { target: normalizeMinutes(0.5 * (min + max)), sourceMode: mode }
        }
        return { target: null, sourceMode: mode }
    }
    // Legacy/advanced v2 modes: prefer showing something rather than an empty box.
    if (mode === 'cap') {
        const cap = toMaybeNum(current?.cap_value)
        return { target: cap == null ? null : normalizeMinutes(cap), sourceMode: mode }
    }
    if (mode === 'floor') {
        const floor = toMaybeNum(current?.floor_value)
        return { target: floor == null ? null : normalizeMinutes(floor), sourceMode: mode }
    }
    return { target: null, sourceMode: mode }
}

export const OverrideControl: React.FC<OverrideControlProps> = ({
    value,
    baselineMinutes,
    resolvedMinutes,
    onChange,
    compact = false,
}) => {
    const current = value ?? { mode: 'none' as OverrideMode }
    const derived = deriveTarget(current)
    const target = derived.target
    const isActive = (current.mode ?? 'none') !== 'none'

    const applyTarget = (val: number | null) => {
        if (val == null) {
            onChange({ mode: 'none', protect_weight: current.protect_weight })
            return
        }
        const normalized = normalizeMinutes(val)
        if (normalized <= 0) {
            onChange({ mode: 'zero', protect_weight: current.protect_weight })
            return
        }
        onChange({
            mode: 'lock',
            lock_value: normalized,
            protect_weight: current.protect_weight,
        })
    }

    const clearTarget = () => applyTarget(null)

    const placeholder = current.mode === 'none'
        ? normalizeMinutes(resolvedMinutes).toFixed(1)
        : target == null
            ? ''
            : target.toFixed(1)

    return (
        <div className={`gv2-override-control ${compact ? 'compact' : ''}`} onClick={(e) => e.stopPropagation()}>
            <div className="gv2-override-row gv2-target-row">
                {!compact ? <span className="gv2-band-label">Target</span> : null}
                <input
                    type="number"
                    className="gv2-input"
                    step={0.5}
                    min={0}
                    max={48}
                    value={target ?? ''}
                    placeholder={placeholder}
                    onChange={(e) => {
                        const raw = e.target.value
                        if (!raw) {
                            applyTarget(null)
                            return
                        }
                        const parsed = Number(raw)
                        if (!Number.isFinite(parsed)) return
                        applyTarget(parsed)
                    }}
                    aria-label="Target mean minutes"
                />
                {compact && isActive ? (
                    <button type="button" className="gv2-band-clear" onClick={clearTarget} title="Clear target">
                        Clear
                    </button>
                ) : null}
            </div>

            {!compact && (
                <div className="gv2-quick-actions">
                    <button type="button" onClick={() => applyTarget(resolvedMinutes)}>Use resolved</button>
                    <button type="button" onClick={() => applyTarget(baselineMinutes)}>Use baseline</button>
                    <button type="button" onClick={() => applyTarget(0)}>0</button>
                    <button type="button" onClick={clearTarget}>Clear</button>
                </div>
            )}

            {!compact && derived.sourceMode !== 'none' && derived.sourceMode !== 'lock' && derived.sourceMode !== 'zero' ? (
                <div className="muted gv2-band-note">
                    Existing `{derived.sourceMode}` converted to target on edit.
                </div>
            ) : null}
        </div>
    )
}
