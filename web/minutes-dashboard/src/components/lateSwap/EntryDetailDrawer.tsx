import { useMemo, useState } from 'react'
import { LateSwapCandidate } from '../../api/late_swap'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

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
            <Card className="late-swap-entry-detail">
                <CardHeader>
                    <CardTitle>Entry Detail</CardTitle>
                </CardHeader>
                <CardContent>Select an entry to inspect candidates.</CardContent>
            </Card>
        )
    }

    return (
        <Card className="late-swap-entry-detail">
            <CardHeader>
                <CardTitle>Entry Detail</CardTitle>
            </CardHeader>
            <CardContent>
                <p className="entry-id">{scopedEntryId}</p>
                <div className="selected-metrics">
                    <Badge variant="secondary">Selected: {selectedCandidate.generated_by}</Badge>
                    <Badge variant="muted">Swaps: {selectedCandidate.swap_count}</Badge>
                    <Badge variant="outline">Proj: {selectedCandidate.projected_score?.toFixed(2) ?? '-'}</Badge>
                    <Badge variant="outline">Own: {selectedCandidate.total_own?.toFixed(1) ?? '-'}</Badge>
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

                <Button
                    type="button"
                    onClick={() => {
                        if (effectiveSelectedId) {
                            onPinCandidate(scopedEntryId, effectiveSelectedId)
                        }
                    }}
                >
                    {pinnedCandidateId ? 'Update Pin' : 'Pin Candidate'}
                </Button>
            </CardContent>
        </Card>
    )
}
