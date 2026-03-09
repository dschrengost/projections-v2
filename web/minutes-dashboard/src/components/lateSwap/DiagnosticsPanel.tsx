import { LateSwapDiagnostics } from '../../api/late_swap'

interface DiagnosticsPanelProps {
    diagnostics: LateSwapDiagnostics | null | undefined
}

export function DiagnosticsPanel({ diagnostics }: DiagnosticsPanelProps) {
    if (!diagnostics) {
        return (
            <section className="late-swap-diagnostics">
                <h3>Diagnostics</h3>
                <p>No diagnostics available yet.</p>
            </section>
        )
    }

    const warningItems = [
        ...diagnostics.warnings,
        ...diagnostics.stale_reasons.map((item) => `stale: ${item}`),
    ]

    return (
        <section className="late-swap-diagnostics">
            <h3>Diagnostics</h3>
            <div className="diagnostic-columns">
                <div>
                    <h4>Warnings</h4>
                    <ul>
                        {warningItems.map((warning, idx) => (
                            <li key={`${warning}-${idx}`}>{warning}</li>
                        ))}
                        {warningItems.length === 0 && <li>None</li>}
                    </ul>
                </div>
                <div>
                    <h4>Errors</h4>
                    <ul>
                        {diagnostics.errors.map((error, idx) => (
                            <li key={`${error}-${idx}`}>{error}</li>
                        ))}
                        {diagnostics.errors.length === 0 && <li>None</li>}
                    </ul>
                </div>
            </div>
        </section>
    )
}
