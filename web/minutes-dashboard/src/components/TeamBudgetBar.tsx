import React from 'react'
import { TeamDiagnostics } from '../api/gameview_v2'

type TeamBudgetBarProps = {
    diagnostics?: TeamDiagnostics | null
    target?: number
}

export const TeamBudgetBar: React.FC<TeamBudgetBarProps> = ({ diagnostics, target = 240 }) => {
    const locked = diagnostics?.locked_minutes_total ?? 0
    const sumMu = diagnostics?.sum_mu ?? 0
    const sumLb = diagnostics?.sum_lb ?? 0
    const sumUb = diagnostics?.sum_ub ?? 0
    const flexible = Math.max(0, sumUb - sumLb)
    const remaining = Math.max(0, target - sumMu)
    const infeasibleReason = diagnostics?.infeasibility_reason

    const clampPct = (n: number) => Math.max(0, Math.min(100, (n / target) * 100))

    return (
        <div className="gv2-budget-wrap">
            <div className="gv2-budget-top">
                <span>Team Budget</span>
                <span>{sumMu.toFixed(1)} / {target}</span>
            </div>
            <div className="gv2-budget-bar">
                <div className="segment locked" style={{ width: `${clampPct(locked)}%` }} title={`Locked ${locked.toFixed(1)}`} />
                <div className="segment flexible" style={{ width: `${clampPct(flexible)}%` }} title={`Flexible ${flexible.toFixed(1)}`} />
                <div className="segment remaining" style={{ width: `${clampPct(remaining)}%` }} title={`Remaining ${remaining.toFixed(1)}`} />
            </div>
            <div className="gv2-budget-meta">
                <span>LB {sumLb.toFixed(1)}</span>
                <span>UB {sumUb.toFixed(1)}</span>
                <span>Locked {locked.toFixed(1)}</span>
            </div>
            {infeasibleReason && <div className="gv2-infeasible-banner">Infeasible: {infeasibleReason}</div>}
        </div>
    )
}
