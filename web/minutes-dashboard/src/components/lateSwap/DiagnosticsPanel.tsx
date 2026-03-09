import { LateSwapDiagnostics } from '../../api/late_swap'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

interface DiagnosticsPanelProps {
    diagnostics: LateSwapDiagnostics | null | undefined
}

export function DiagnosticsPanel({ diagnostics }: DiagnosticsPanelProps) {
    if (!diagnostics) {
        return (
            <Card className="late-swap-diagnostics">
                <CardHeader>
                    <CardTitle>Diagnostics</CardTitle>
                </CardHeader>
                <CardContent>No diagnostics available yet.</CardContent>
            </Card>
        )
    }

    const warningItems = [
        ...diagnostics.warnings,
        ...diagnostics.stale_reasons.map((item) => `stale: ${item}`),
    ]

    return (
        <Card className="late-swap-diagnostics">
            <CardHeader>
                <CardTitle>Diagnostics</CardTitle>
            </CardHeader>
            <CardContent className="diagnostic-columns">
                <div className="diag-card">
                    <h4>Warnings</h4>
                    <div className="diag-items">
                        {warningItems.map((warning, idx) => (
                            <Badge key={`${warning}-${idx}`} variant="muted" className="diag-badge">
                                {warning}
                            </Badge>
                        ))}
                        {warningItems.length === 0 && <span className="muted-empty">None</span>}
                    </div>
                </div>
                <div className="diag-card">
                    <h4>Errors</h4>
                    <div className="diag-items">
                        {diagnostics.errors.map((error, idx) => (
                            <Badge key={`${error}-${idx}`} variant="default" className="diag-badge">
                                {error}
                            </Badge>
                        ))}
                        {diagnostics.errors.length === 0 && <span className="muted-empty">None</span>}
                    </div>
                </div>
            </CardContent>
        </Card>
    )
}
