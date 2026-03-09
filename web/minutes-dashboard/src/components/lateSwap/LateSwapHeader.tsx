import { LateSwapSession } from '../../api/late_swap'

interface LateSwapHeaderProps {
    date: string
    onDateChange: (value: string) => void
    selectedContestCount: number
    selectedEntryCount: number
    session: LateSwapSession | null
}

export function LateSwapHeader({
    date,
    onDateChange,
    selectedContestCount,
    selectedEntryCount,
    session,
}: LateSwapHeaderProps) {
    return (
        <section className="late-swap-header">
            <div>
                <h2>Late Swap Workbench</h2>
                <p>
                    Sessionized grouped late swap with preview, diagnostics, and explicit commit.
                </p>
            </div>
            <div className="late-swap-header-meta">
                <label>
                    Date
                    <input
                        type="date"
                        value={date}
                        onChange={(event) => onDateChange(event.target.value)}
                    />
                </label>
                <div className="late-swap-chip">
                    Contests Selected: <strong>{selectedContestCount}</strong>
                </div>
                <div className="late-swap-chip">
                    Entries: <strong>{selectedEntryCount}</strong>
                </div>
                {session && (
                    <div className={`late-swap-chip status-${session.status}`}>
                        Session: <strong>{session.status}</strong>
                    </div>
                )}
            </div>
        </section>
    )
}
