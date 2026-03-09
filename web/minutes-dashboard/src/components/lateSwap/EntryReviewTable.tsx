import { LateSwapCandidate } from '../../api/late_swap'

interface EntryReviewTableProps {
    candidatesByEntryId: Record<string, LateSwapCandidate[]>
    selectedByEntryId: Record<string, string>
    pinnedByEntryId: Record<string, string>
    selectedEntryId: string | null
    onSelectEntry: (entryId: string) => void
}

function splitScopedEntryId(scoped: string): { contestId: string; entryId: string } {
    const idx = scoped.indexOf(':')
    if (idx < 0) return { contestId: '-', entryId: scoped }
    return {
        contestId: scoped.slice(0, idx),
        entryId: scoped.slice(idx + 1),
    }
}

export function EntryReviewTable({
    candidatesByEntryId,
    selectedByEntryId,
    pinnedByEntryId,
    selectedEntryId,
    onSelectEntry,
}: EntryReviewTableProps) {
    const rows = Object.entries(candidatesByEntryId).map(([entryId, candidates]) => {
        const selectedCandidateId = selectedByEntryId[entryId]
        const selected = candidates.find((candidate) => candidate.candidate_id === selectedCandidateId) ?? candidates[0]
        const pinned = Boolean(pinnedByEntryId[entryId])
        const state = pinned ? 'pinned' : selected.swap_count > 0 ? 'swapped' : 'held'
        const split = splitScopedEntryId(entryId)
        return {
            scopedEntryId: entryId,
            contestId: split.contestId,
            entryId: split.entryId,
            lockedSlots: selected.locked_slots.length,
            selected,
            state,
        }
    })

    return (
        <section className="late-swap-entry-table">
            <h3>Entry Review</h3>
            <div className="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Contest</th>
                            <th>Entry</th>
                            <th>Locked Slots</th>
                            <th>Selected Candidate</th>
                            <th>Swaps</th>
                            <th>Projected</th>
                            <th>State</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows.map((row) => (
                            <tr
                                key={row.scopedEntryId}
                                className={selectedEntryId === row.scopedEntryId ? 'active' : ''}
                                onClick={() => onSelectEntry(row.scopedEntryId)}
                            >
                                <td>{row.contestId}</td>
                                <td>{row.entryId}</td>
                                <td>{row.lockedSlots}</td>
                                <td>{row.selected.generated_by}</td>
                                <td>{row.selected.swap_count}</td>
                                <td>{row.selected.projected_score?.toFixed(2) ?? '-'}</td>
                                <td>{row.state}</td>
                            </tr>
                        ))}
                        {rows.length === 0 && (
                            <tr>
                                <td colSpan={7}>No entry previews available.</td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        </section>
    )
}
