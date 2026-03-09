import { useState } from 'react'
import { LateSwapMode, LateSwapPolicy } from '../../api/late_swap'

export interface LateSwapContestOption {
    contestId: string
    contestName: string
    entryCount: number
}

interface LateSwapPolicyPanelProps {
    policy: LateSwapPolicy
    contestOptions: LateSwapContestOption[]
    selectedContestIds: Set<string>
    onToggleContest: (contestId: string) => void
    onPolicyChange: (next: LateSwapPolicy) => void
    onCreateSession: () => void
    onApplyPolicy: () => void
    onPreview: () => void
    onCommit: () => void
    onExport: (includePreview: boolean) => void
    disabled?: boolean
}

const MODE_OPTIONS: Array<{ mode: LateSwapMode; label: string; summary: string }> = [
    { mode: 'preserve_targets', label: 'Preserve Targets', summary: 'Balanced EV and target preservation' },
    { mode: 'best_ev', label: 'Best EV', summary: 'Pure projection/EV with hard caps' },
    { mode: 'decorrelated_ev', label: 'Decorrelated', summary: 'EV plus overlap controls' },
    { mode: 'catch_up', label: 'Catch Up', summary: 'Lower ownership and higher leverage' },
    { mode: 'block', label: 'Block', summary: 'Conservative with higher swap cost' },
]

export function LateSwapPolicyPanel({
    policy,
    contestOptions,
    selectedContestIds,
    onToggleContest,
    onPolicyChange,
    onCreateSession,
    onApplyPolicy,
    onPreview,
    onCommit,
    onExport,
    disabled = false,
}: LateSwapPolicyPanelProps) {
    const [boundPlayerId, setBoundPlayerId] = useState('')
    const [boundMin, setBoundMin] = useState('')
    const [boundMax, setBoundMax] = useState('')

    const setNumeric = (field: keyof LateSwapPolicy, value: string) => {
        const parsed = Number(value)
        if (Number.isNaN(parsed)) return
        onPolicyChange({ ...policy, [field]: parsed })
    }

    const addOrUpdateBound = () => {
        const playerId = boundPlayerId.trim()
        if (!playerId) return
        const min = boundMin.trim() === '' ? null : Number(boundMin)
        const max = boundMax.trim() === '' ? null : Number(boundMax)
        onPolicyChange({
            ...policy,
            exposure_bounds: {
                ...policy.exposure_bounds,
                [playerId]: {
                    min: Number.isFinite(min ?? NaN) ? min : null,
                    max: Number.isFinite(max ?? NaN) ? max : null,
                },
            },
        })
        setBoundPlayerId('')
        setBoundMin('')
        setBoundMax('')
    }

    const removeBound = (playerId: string) => {
        const next = { ...policy.exposure_bounds }
        delete next[playerId]
        onPolicyChange({ ...policy, exposure_bounds: next })
    }

    return (
        <section className="late-swap-policy-panel">
            <h3>Session & Policy</h3>

            <div className="contest-picker">
                {contestOptions.map((contest) => {
                    const checked = selectedContestIds.has(contest.contestId)
                    return (
                        <label key={contest.contestId} className="contest-option">
                            <input
                                type="checkbox"
                                checked={checked}
                                onChange={() => onToggleContest(contest.contestId)}
                                disabled={disabled}
                            />
                            <span>{contest.contestName}</span>
                            <small>{contest.entryCount}</small>
                        </label>
                    )
                })}
            </div>

            <div className="mode-grid">
                {MODE_OPTIONS.map((opt) => (
                    <button
                        type="button"
                        key={opt.mode}
                        className={`mode-card ${policy.mode === opt.mode ? 'active' : ''}`}
                        onClick={() => onPolicyChange({ ...policy, mode: opt.mode })}
                        disabled={disabled}
                    >
                        <strong>{opt.label}</strong>
                        <span>{opt.summary}</span>
                    </button>
                ))}
            </div>

            <div className="policy-controls">
                <label>
                    Target Source
                    <select
                        value={policy.target_source}
                        onChange={(event) =>
                            onPolicyChange({
                                ...policy,
                                target_source: event.target.value as LateSwapPolicy['target_source'],
                            })
                        }
                        disabled={disabled}
                    >
                        <option value="source_portfolio">Source Portfolio</option>
                        <option value="current_entries">Current Entries</option>
                        <option value="explicit">Explicit</option>
                        <option value="none">None</option>
                    </select>
                </label>
                <label>
                    Candidates / Entry
                    <input
                        type="number"
                        min={6}
                        max={20}
                        value={policy.candidate_count_per_entry}
                        onChange={(event) => setNumeric('candidate_count_per_entry', event.target.value)}
                        disabled={disabled}
                    />
                </label>
                <label>
                    Min Uniques
                    <input
                        type="number"
                        min={0}
                        value={policy.min_uniques}
                        onChange={(event) => setNumeric('min_uniques', event.target.value)}
                        disabled={disabled}
                    />
                </label>
                <label>
                    Max Duplicates
                    <input
                        type="number"
                        min={1}
                        value={policy.max_duplicate_lineups}
                        onChange={(event) => setNumeric('max_duplicate_lineups', event.target.value)}
                        disabled={disabled}
                    />
                </label>
                <label>
                    Swap Cost λ
                    <input
                        type="number"
                        step={0.01}
                        value={policy.swap_cost_lambda}
                        onChange={(event) => setNumeric('swap_cost_lambda', event.target.value)}
                        disabled={disabled}
                    />
                </label>
                <label>
                    Target Deviation λ
                    <input
                        type="number"
                        step={0.01}
                        value={policy.target_deviation_lambda}
                        onChange={(event) => setNumeric('target_deviation_lambda', event.target.value)}
                        disabled={disabled}
                    />
                </label>
            </div>

            <div className="bound-editor">
                <h4>Player Exposure Bounds</h4>
                <div className="bound-row">
                    <input
                        type="text"
                        placeholder="Player ID"
                        value={boundPlayerId}
                        onChange={(event) => setBoundPlayerId(event.target.value)}
                        disabled={disabled}
                    />
                    <input
                        type="number"
                        placeholder="Min %"
                        value={boundMin}
                        onChange={(event) => setBoundMin(event.target.value)}
                        disabled={disabled}
                    />
                    <input
                        type="number"
                        placeholder="Max %"
                        value={boundMax}
                        onChange={(event) => setBoundMax(event.target.value)}
                        disabled={disabled}
                    />
                    <button type="button" onClick={addOrUpdateBound} disabled={disabled}>
                        Set
                    </button>
                </div>
                <ul>
                    {Object.entries(policy.exposure_bounds).map(([pid, bounds]) => (
                        <li key={pid}>
                            <span>{pid}</span>
                            <span>
                                min {bounds.min ?? '-'} / max {bounds.max ?? '-'}
                            </span>
                            <button type="button" onClick={() => removeBound(pid)} disabled={disabled}>
                                Remove
                            </button>
                        </li>
                    ))}
                </ul>
            </div>

            <div className="policy-actions">
                <button type="button" onClick={onCreateSession} disabled={disabled || selectedContestIds.size === 0}>
                    Create Session
                </button>
                <button type="button" onClick={onApplyPolicy} disabled={disabled}>
                    Save Policy
                </button>
                <button type="button" onClick={onPreview} disabled={disabled}>
                    Refresh Preview
                </button>
                <button type="button" onClick={onCommit} disabled={disabled}>
                    Commit Preview
                </button>
                <button type="button" onClick={() => onExport(true)} disabled={disabled}>
                    Export Preview CSV
                </button>
                <button type="button" onClick={() => onExport(false)} disabled={disabled}>
                    Export Committed CSV
                </button>
            </div>
        </section>
    )
}
