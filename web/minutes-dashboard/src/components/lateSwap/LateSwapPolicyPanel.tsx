import { useState } from 'react'
import { LateSwapMode, LateSwapPolicy } from '../../api/late_swap'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select'

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
        <Card className="late-swap-policy-panel">
            <CardHeader>
                <CardTitle>Session & Policy</CardTitle>
            </CardHeader>
            <CardContent>

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
                            <span className="contest-name">{contest.contestName}</span>
                            <Badge variant="muted">{contest.entryCount}</Badge>
                        </label>
                    )
                })}
            </div>

            <div className="mode-grid">
                {MODE_OPTIONS.map((opt) => (
                    <Button
                        type="button"
                        key={opt.mode}
                        variant={policy.mode === opt.mode ? 'secondary' : 'outline'}
                        className={`mode-card ${policy.mode === opt.mode ? 'active' : ''}`}
                        onClick={() => onPolicyChange({ ...policy, mode: opt.mode })}
                        disabled={disabled}
                    >
                        <strong>{opt.label}</strong>
                        <span>{opt.summary}</span>
                    </Button>
                ))}
            </div>

            <div className="policy-controls">
                <label>
                    <span>Target Source</span>
                    <Select
                        value={policy.target_source}
                        onValueChange={(value) =>
                            onPolicyChange({
                                ...policy,
                                target_source: value as LateSwapPolicy['target_source'],
                            })
                        }
                    >
                        <SelectTrigger disabled={disabled}>
                            <SelectValue placeholder="Target source" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="source_portfolio">Source Portfolio</SelectItem>
                            <SelectItem value="current_entries">Current Entries</SelectItem>
                            <SelectItem value="explicit">Explicit</SelectItem>
                            <SelectItem value="none">None</SelectItem>
                        </SelectContent>
                    </Select>
                </label>
                <label>
                    <span>Candidates / Entry</span>
                    <Input
                        type="number"
                        min={6}
                        max={20}
                        value={policy.candidate_count_per_entry}
                        onChange={(event) => setNumeric('candidate_count_per_entry', event.target.value)}
                        disabled={disabled}
                    />
                </label>
                <label>
                    <span>Min Uniques</span>
                    <Input
                        type="number"
                        min={0}
                        value={policy.min_uniques}
                        onChange={(event) => setNumeric('min_uniques', event.target.value)}
                        disabled={disabled}
                    />
                </label>
                <label>
                    <span>Max Duplicates</span>
                    <Input
                        type="number"
                        min={1}
                        value={policy.max_duplicate_lineups}
                        onChange={(event) => setNumeric('max_duplicate_lineups', event.target.value)}
                        disabled={disabled}
                    />
                </label>
                <label>
                    <span>Swap Cost λ</span>
                    <Input
                        type="number"
                        step={0.01}
                        value={policy.swap_cost_lambda}
                        onChange={(event) => setNumeric('swap_cost_lambda', event.target.value)}
                        disabled={disabled}
                    />
                </label>
                <label>
                    <span>Target Deviation λ</span>
                    <Input
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
                    <Input
                        type="text"
                        placeholder="Player ID"
                        value={boundPlayerId}
                        onChange={(event) => setBoundPlayerId(event.target.value)}
                        disabled={disabled}
                    />
                    <Input
                        type="number"
                        placeholder="Min %"
                        value={boundMin}
                        onChange={(event) => setBoundMin(event.target.value)}
                        disabled={disabled}
                    />
                    <Input
                        type="number"
                        placeholder="Max %"
                        value={boundMax}
                        onChange={(event) => setBoundMax(event.target.value)}
                        disabled={disabled}
                    />
                    <Button type="button" variant="secondary" onClick={addOrUpdateBound} disabled={disabled}>
                        Set
                    </Button>
                </div>
                <ul>
                    {Object.entries(policy.exposure_bounds).map(([pid, bounds]) => (
                        <li key={pid}>
                            <Badge variant="outline">{pid}</Badge>
                            <small>
                                min {bounds.min ?? '-'} / max {bounds.max ?? '-'}
                            </small>
                            <Button type="button" variant="ghost" size="sm" onClick={() => removeBound(pid)} disabled={disabled}>
                                Remove
                            </Button>
                        </li>
                    ))}
                </ul>
            </div>

            <div className="policy-actions">
                <Button type="button" onClick={onCreateSession} disabled={disabled || selectedContestIds.size === 0}>
                    Create Session
                </Button>
                <Button type="button" variant="secondary" onClick={onApplyPolicy} disabled={disabled}>
                    Save Policy
                </Button>
                <Button type="button" variant="outline" onClick={onPreview} disabled={disabled}>
                    Refresh Preview
                </Button>
                <Button type="button" variant="destructive" onClick={onCommit} disabled={disabled}>
                    Commit Preview
                </Button>
                <Button type="button" variant="outline" onClick={() => onExport(true)} disabled={disabled}>
                    Export Preview CSV
                </Button>
                <Button type="button" variant="outline" onClick={() => onExport(false)} disabled={disabled}>
                    Export Committed CSV
                </Button>
            </div>
            </CardContent>
        </Card>
    )
}
