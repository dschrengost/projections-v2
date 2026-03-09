import { useMemo, useState } from 'react'
import { LateSwapCandidate } from '../../api/late_swap'

interface EntryDetailDrawerProps {
    scopedEntryId: string | null
    candidates: LateSwapCandidate[]
    selectedCandidateId: string | null
    pinnedCandidateId: string | null
    onPinCandidate: (entryId: string, candidateId: string) => void
}

export function EntryDetailDrawer({
    scopedEntryId,
    candidates,
    selectedCandidateId,
    pinnedCandidateId,
    onPinCandidate,
}: EntryDetailDrawerProps) {
    const [localSelectedId, setLocalSelectedId] = useState<string | null>(null)
    const effectiveSelectedId = localSelectedId ?? selectedCandidateId

    const selectedCandidate = useMemo(
        () =>
            candidates.find((candidate) => candidate.candidate_id === effectiveSelectedId)
            ?? candidates[0]
            ?? null,
        [candidates, effectiveSelectedId],
    )

    if (!scopedEntryId || !selectedCandidate) {
        return (
            <aside className="late-swap-entry-detail">
                <h3>Entry Detail</h3>
                <p>Select an entry to inspect candidates.</p>
            </aside>
        )
    }

    return (
        <aside className="late-swap-entry-detail">
            <h3>Entry Detail</h3>
            <p className="entry-id">{scopedEntryId}</p>
            <div className="selected-metrics">
                <span>Selected: {selectedCandidate.generated_by}</span>
                <span>Swaps: {selectedCandidate.swap_count}</span>
                <span>Proj: {selectedCandidate.projected_score?.toFixed(2) ?? '-'}</span>
                <span>Own: {selectedCandidate.total_own?.toFixed(1) ?? '-'}</span>
            </div>

            <div className="diff-row">
                <div>
                    <h4>Players In</h4>
                    <ul>
                        {selectedCandidate.added_player_ids.map((pid) => (
                            <li key={pid}>{pid}</li>
                        ))}
                        {selectedCandidate.added_player_ids.length === 0 && <li>None</li>}
                    </ul>
                </div>
                <div>
                    <h4>Players Out</h4>
                    <ul>
                        {selectedCandidate.removed_player_ids.map((pid) => (
                            <li key={pid}>{pid}</li>
                        ))}
                        {selectedCandidate.removed_player_ids.length === 0 && <li>None</li>}
                    </ul>
                </div>
            </div>

            <div className="candidate-list">
                {candidates.map((candidate) => (
                    <label key={candidate.candidate_id} className="candidate-item">
                        <input
                            type="radio"
                            checked={effectiveSelectedId === candidate.candidate_id}
                            onChange={() => setLocalSelectedId(candidate.candidate_id)}
                        />
                        <span>{candidate.generated_by}</span>
                        <small>
                            swaps {candidate.swap_count} · proj {candidate.projected_score?.toFixed(2) ?? '-'}
                        </small>
                    </label>
                ))}
            </div>

            <button
                type="button"
                onClick={() => {
                    if (effectiveSelectedId) {
                        onPinCandidate(scopedEntryId, effectiveSelectedId)
                    }
                }}
            >
                {pinnedCandidateId ? 'Update Pin' : 'Pin Candidate'}
            </button>
        </aside>
    )
}
