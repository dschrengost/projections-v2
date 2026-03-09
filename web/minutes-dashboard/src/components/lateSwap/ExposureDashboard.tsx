import { useMemo, useState } from 'react'
import { ExposureStateRow } from '../../api/late_swap'
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

interface ExposureDashboardProps {
    rows: ExposureStateRow[]
}

export function ExposureDashboard({ rows }: ExposureDashboardProps) {
    const [onlyOverCap, setOnlyOverCap] = useState(false)
    const [onlyForced, setOnlyForced] = useState(false)
    const [onlyChanged, setOnlyChanged] = useState(false)
    const [top20, setTop20] = useState(true)

    const filtered = useMemo(() => {
        let out = rows.slice()
        if (onlyOverCap) {
            out = out.filter((row) => row.status === 'over_target')
        }
        if (onlyForced) {
            out = out.filter((row) => row.forced_over_cap_by_locks)
        }
        if (onlyChanged) {
            out = out.filter((row) => row.status !== 'within_target')
        }
        out.sort((left, right) => right.proposed_final_pct - left.proposed_final_pct)
        if (top20) {
            out = out.slice(0, 20)
        }
        return out
    }, [rows, onlyOverCap, onlyForced, onlyChanged, top20])

    return (
        <Card className="late-swap-exposure">
            <CardHeader className="panel-header-row">
                <CardTitle>Exposure Dashboard</CardTitle>
                <div className="filters">
                    <label>
                        <input
                            type="checkbox"
                            checked={onlyOverCap}
                            onChange={(event) => setOnlyOverCap(event.target.checked)}
                        />
                        Over Cap
                    </label>
                    <label>
                        <input
                            type="checkbox"
                            checked={onlyForced}
                            onChange={(event) => setOnlyForced(event.target.checked)}
                        />
                        Forced By Locks
                    </label>
                    <label>
                        <input
                            type="checkbox"
                            checked={onlyChanged}
                            onChange={(event) => setOnlyChanged(event.target.checked)}
                        />
                        Changed
                    </label>
                    <label>
                        <input
                            type="checkbox"
                            checked={top20}
                            onChange={(event) => setTop20(event.target.checked)}
                        />
                        Top 20
                    </label>
                </div>
            </CardHeader>
            <CardContent className="table-wrap">
                <Table>
                    <TableHeader>
                        <TableRow>
                            <TableHead>Player</TableHead>
                            <TableHead>Target%</TableHead>
                            <TableHead>Lock Floor%</TableHead>
                            <TableHead>Current%</TableHead>
                            <TableHead>Proposed%</TableHead>
                            <TableHead>Δ Target</TableHead>
                            <TableHead>Status</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {filtered.map((row) => (
                            <TableRow key={row.player_id}>
                                <TableCell>{row.player_name}</TableCell>
                                <TableCell>{row.source_target_pct?.toFixed(1) ?? '-'}</TableCell>
                                <TableCell>{row.locked_floor_pct.toFixed(1)}</TableCell>
                                <TableCell>{row.current_committed_pct.toFixed(1)}</TableCell>
                                <TableCell>{row.proposed_final_pct.toFixed(1)}</TableCell>
                                <TableCell>{row.delta_vs_target_pct?.toFixed(1) ?? '-'}</TableCell>
                                <TableCell>
                                    <Badge
                                        variant={
                                            row.forced_over_cap_by_locks
                                                ? 'default'
                                                : row.status === 'within_target'
                                                    ? 'muted'
                                                    : 'secondary'
                                        }
                                    >
                                        {row.status}
                                    </Badge>
                                </TableCell>
                            </TableRow>
                        ))}
                        {filtered.length === 0 && (
                            <TableRow>
                                <TableCell colSpan={7}>No exposures to display for current filters.</TableCell>
                            </TableRow>
                        )}
                    </TableBody>
                </Table>
            </CardContent>
        </Card>
    )
}
