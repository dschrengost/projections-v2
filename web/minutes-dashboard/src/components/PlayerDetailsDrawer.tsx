import React from 'react'
import { OpsGamePlayer, PlayerLastGameStat } from '../api/manualAvailability'

type DrawerMetric = {
  baseline: number | null
  resolved: number | null
  p10?: number | null
  p50?: number | null
  p90?: number | null
}

export type PlayerDrawerData = {
  player: OpsGamePlayer
  metrics: {
    minutes: DrawerMetric
    fpts: DrawerMetric
    pts: DrawerMetric
    reb: DrawerMetric
    ast: DrawerMetric
    stl: DrawerMetric
    blk: DrawerMetric
    to: DrawerMetric
  }
  lastGames: PlayerLastGameStat[]
  lastGamesLoading: boolean
  lastGamesError?: string | null
}

type PlayerDetailsDrawerProps = {
  open: boolean
  data: PlayerDrawerData | null
  onClose: () => void
}

const fmt = (value: number | null | undefined, digits = 1) =>
  value == null || Number.isNaN(value) ? '-' : value.toFixed(digits)

const MetricRow: React.FC<{ label: string; metric: DrawerMetric }> = ({ label, metric }) => {
  const hasQuantiles = metric.p10 != null || metric.p50 != null || metric.p90 != null
  return (
    <div className="gv2-metric-row">
      <div className="gv2-metric-head">
        <span>{label}</span>
        <span>
          {fmt(metric.baseline)} → {fmt(metric.resolved)}
        </span>
      </div>
      {hasQuantiles ? (
        <div className="gv2-metric-quantiles">
          <span>p10 {fmt(metric.p10)}</span>
          <span>p50 {fmt(metric.p50)}</span>
          <span>p90 {fmt(metric.p90)}</span>
        </div>
      ) : null}
    </div>
  )
}

export const PlayerDetailsDrawer: React.FC<PlayerDetailsDrawerProps> = ({
  open,
  data,
  onClose,
}) => {
  if (!open || !data) return null

  const { player, metrics, lastGames, lastGamesLoading, lastGamesError } = data
  const manualOverride = player.manual_override

  return (
    <>
      <div className="gv2-drawer-backdrop" onClick={onClose} />
      <aside className="gv2-drawer" role="dialog" aria-label="Player details">
        <div className="gv2-drawer-header">
          <div>
            <h3 className="gv2-drawer-player-title">{player.player_name || player.player_id}</h3>
            <div className="muted">
              {player.team_tricode || player.team_id}
              {player.minutes_effective?.status ? ` · ${player.minutes_effective.status}` : ''}
            </div>
          </div>
          <button type="button" onClick={onClose}>Close</button>
        </div>

        {manualOverride ? (
          <section className="gv2-drawer-section">
            <h4>Manual Availability</h4>
            <div className="gv2-why-changed">
              {manualOverride.override_type === 'force_out' ? 'Force OUT' : 'Force IN'}
              {manualOverride.entered_by ? ` · ${manualOverride.entered_by}` : ''}
              {manualOverride.reason_code ? ` · ${manualOverride.reason_code}` : ''}
            </div>
            {manualOverride.reason_text ? (
              <div className="gv2-why-changed" style={{ marginTop: '0.35rem' }}>{manualOverride.reason_text}</div>
            ) : null}
          </section>
        ) : null}

        <section className="gv2-drawer-section">
          <h4>Projections</h4>
          <MetricRow label="Minutes" metric={metrics.minutes} />
          <MetricRow label="FPTS" metric={metrics.fpts} />
          <MetricRow label="PTS" metric={metrics.pts} />
          <MetricRow label="REB" metric={metrics.reb} />
          <MetricRow label="AST" metric={metrics.ast} />
          <MetricRow label="STL" metric={metrics.stl} />
          <MetricRow label="BLK" metric={metrics.blk} />
          <MetricRow label="TO" metric={metrics.to} />
        </section>

        <section className="gv2-drawer-section">
          <h4>Last 5 Games (Actuals)</h4>
          {lastGamesLoading ? (
            <div className="muted">Loading game log…</div>
          ) : lastGamesError ? (
            <div className="gv2-why-changed">{lastGamesError}</div>
          ) : lastGames.length === 0 ? (
            <div className="muted">No recent games found.</div>
          ) : (
            <div className="gv2-last-games-wrap">
              <table className="gv2-last-games-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Opp</th>
                    <th>MIN</th>
                    <th>PTS</th>
                    <th>REB</th>
                    <th>AST</th>
                    <th>FPTS</th>
                  </tr>
                </thead>
                <tbody>
                  {lastGames.map((game) => (
                    <tr key={`${game.game_id}-${game.game_date}`}>
                      <td>{game.game_date}</td>
                      <td>
                        {game.team_tricode && game.opponent_tricode
                          ? `${game.team_tricode} vs ${game.opponent_tricode}`
                          : game.opponent_tricode || '-'}
                      </td>
                      <td>{fmt(game.minutes)}</td>
                      <td>{fmt(game.pts, 0)}</td>
                      <td>{fmt(game.reb, 0)}</td>
                      <td>{fmt(game.ast, 0)}</td>
                      <td>{fmt(game.fpts)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </aside>
    </>
  )
}
