import { useMemo, useState } from 'react'
import { ExposureStateRow } from '../../api/late_swap'

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
        <section className="late-swap-exposure">
            <div className="panel-header">
                <h3>Exposure Dashboard</h3>
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
            </div>
            <div className="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Player</th>
                            <th>Target%</th>
                            <th>Lock Floor%</th>
                            <th>Current%</th>
                            <th>Proposed%</th>
                            <th>Δ Target</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {filtered.map((row) => (
                            <tr key={row.player_id}>
                                <td>{row.player_name}</td>
                                <td>{row.source_target_pct?.toFixed(1) ?? '-'}</td>
                                <td>{row.locked_floor_pct.toFixed(1)}</td>
                                <td>{row.current_committed_pct.toFixed(1)}</td>
                                <td>{row.proposed_final_pct.toFixed(1)}</td>
                                <td>{row.delta_vs_target_pct?.toFixed(1) ?? '-'}</td>
                                <td>{row.status}</td>
                            </tr>
                        ))}
                        {filtered.length === 0 && (
                            <tr>
                                <td colSpan={7}>No exposures to display for current filters.</td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        </section>
    )
}
