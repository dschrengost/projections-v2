import { LateSwapSelectionSummary } from '../../api/late_swap'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

interface SwapSummaryPanelProps {
    summary: LateSwapSelectionSummary | null | undefined
}

export function SwapSummaryPanel({ summary }: SwapSummaryPanelProps) {
    if (!summary) {
        return (
            <Card className="late-swap-summary">
                <CardHeader>
                    <CardTitle>Swap Summary</CardTitle>
                </CardHeader>
                <CardContent>No preview summary yet.</CardContent>
            </Card>
        )
    }

    return (
        <Card className="late-swap-summary">
            <CardHeader>
                <CardTitle>Swap Summary</CardTitle>
            </CardHeader>
            <CardContent className="summary-grid">
                <div>
                    <span>Entries</span>
                    <strong>{summary.entries_total}</strong>
                </div>
                <div>
                    <span>Swapped</span>
                    <strong>{summary.entries_swapped}</strong>
                </div>
                <div>
                    <span>Held</span>
                    <strong>{summary.entries_held}</strong>
                </div>
                <div>
                    <span>Total Swaps</span>
                    <strong>{summary.total_swaps}</strong>
                </div>
                <div>
                    <span>Avg Swaps / Changed</span>
                    <strong>{summary.average_swaps_per_changed_entry.toFixed(2)}</strong>
                </div>
                <div>
                    <span>Projected Δ</span>
                    <strong>{summary.projected_delta_total?.toFixed(2) ?? '-'}</strong>
                </div>
                <div>
                    <span>Max Exposure Before</span>
                    <strong>{summary.max_exposure_before_pct?.toFixed(1) ?? '-'}%</strong>
                </div>
                <div>
                    <span>Max Exposure After</span>
                    <strong>{summary.max_exposure_after_pct?.toFixed(1) ?? '-'}%</strong>
                </div>
                <div>
                    <span>Infeasibilities</span>
                    <strong>{summary.infeasibility_count}</strong>
                </div>
            </CardContent>
        </Card>
    )
}
