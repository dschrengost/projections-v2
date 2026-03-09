import { LateSwapCandidate } from '../../api/late_swap'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table'

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
        <Card className="late-swap-entry-table">
            <CardHeader>
                <CardTitle>Entry Review</CardTitle>
            </CardHeader>
            <CardContent className="table-wrap">
                <Table>
                    <TableHeader>
                        <TableRow>
                            <TableHead>Contest</TableHead>
                            <TableHead>Entry</TableHead>
                            <TableHead>Locked Slots</TableHead>
                            <TableHead>Selected Candidate</TableHead>
                            <TableHead>Swaps</TableHead>
                            <TableHead>Projected</TableHead>
                            <TableHead>State</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {rows.map((row) => (
                            <TableRow
                                key={row.scopedEntryId}
                                className={selectedEntryId === row.scopedEntryId ? 'active' : ''}
                                onClick={() => onSelectEntry(row.scopedEntryId)}
                            >
                                <TableCell>{row.contestId}</TableCell>
                                <TableCell>{row.entryId}</TableCell>
                                <TableCell>{row.lockedSlots}</TableCell>
                                <TableCell>{row.selected.generated_by}</TableCell>
                                <TableCell>{row.selected.swap_count}</TableCell>
                                <TableCell>{row.selected.projected_score?.toFixed(2) ?? '-'}</TableCell>
                                <TableCell>
                                    <Badge variant={row.state === 'swapped' ? 'secondary' : row.state === 'pinned' ? 'default' : 'muted'}>
                                        {row.state}
                                    </Badge>
                                </TableCell>
                            </TableRow>
                        ))}
                        {rows.length === 0 && (
                            <TableRow>
                                <TableCell colSpan={7}>No entry previews available.</TableCell>
                            </TableRow>
                        )}
                    </TableBody>
                </Table>
            </CardContent>
        </Card>
    )
}
