import React, { useState, useEffect } from 'react';

interface FeatureStat {
    min: number | null;
    mean: number | null;
    max: number | null;
    missing: number;
}

interface Anomaly {
    feature: string;
    issue: string;
    expected_min?: number;
    expected_max?: number;
    count: number;
    severity: string;
}

interface FeatureSummary {
    date: string;
    row_count: number;
    starters: number;
    stats: Record<string, FeatureStat>;
    anomalies: Anomaly[];
    has_anomalies: boolean;
}

interface PlayerProjection {
    player_id: number;
    player_name: string | null;
    team_id: number;
    is_starter: number;
    minutes_mean?: number;
    minutes_sim_mean?: number;
    minutes_pred_p50?: number;
    dk_fpts_mean?: number;
    dk_fpts_p50?: number;
    dk_fpts_p10?: number;
    dk_fpts_p90?: number;
    pts_mean?: number;
    reb_mean?: number;
    ast_mean?: number;
    stl_mean?: number;
    blk_mean?: number;
    tov_mean?: number;
    season_fga_per_min?: number;
    season_ast_per_min?: number;
    season_reb_per_min?: number;
}

interface ProjectionsResponse {
    date: string;
    total_players: number;
    players: PlayerProjection[];
}

const DiagnosticsPage: React.FC = () => {
    const [date, setDate] = useState<string>(() => new Date().toISOString().split('T')[0]);
    const [summary, setSummary] = useState<FeatureSummary | null>(null);
    const [projections, setProjections] = useState<ProjectionsResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [selectedPlayer, setSelectedPlayer] = useState<number | null>(null);
    const [playerDetails, setPlayerDetails] = useState<any>(null);

    const fetchData = async () => {
        setLoading(true);
        setError(null);

        try {
            // Fetch feature summary
            const summaryRes = await fetch(`/api/diagnostics/feature-summary?date=${date}`);
            if (summaryRes.ok) {
                const summaryData = await summaryRes.json();
                setSummary(summaryData);
            } else if (summaryRes.status !== 404) {
                throw new Error('Failed to fetch feature summary');
            }

            // Fetch projections
            const projRes = await fetch(`/api/diagnostics/projections?date=${date}&top_n=30`);
            if (projRes.ok) {
                const projData = await projRes.json();
                setProjections(projData);
            } else if (projRes.status !== 404) {
                throw new Error('Failed to fetch projections');
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error');
        } finally {
            setLoading(false);
        }
    };

    const fetchPlayerDetails = async (playerId: number) => {
        try {
            const res = await fetch(`/api/diagnostics/player/${playerId}?date=${date}`);
            if (res.ok) {
                const data = await res.json();
                setPlayerDetails(data);
                setSelectedPlayer(playerId);
            }
        } catch (err) {
            console.error('Failed to fetch player details', err);
        }
    };

    useEffect(() => {
        fetchData();
    }, [date]);

    const formatNumber = (val: number | null | undefined, decimals: number = 2): string => {
        if (val === null || val === undefined) return '-';
        return val.toFixed(decimals);
    };

    return (
        <div className="diagnostics-page">
            <div className="page-header">
                <h1>Feature Diagnostics</h1>
                <div className="controls">
                    <input
                        type="date"
                        value={date}
                        onChange={(e) => setDate(e.target.value)}
                        className="date-input"
                    />
                    <button onClick={fetchData} disabled={loading} className="refresh-btn">
                        {loading ? 'Loading...' : 'Refresh'}
                    </button>
                </div>
            </div>

            {error && <div className="error-banner">{error}</div>}

            {/* Feature Summary Section */}
            {summary && (
                <div className="section">
                    <h2>Feature Summary</h2>
                    <div className="summary-cards">
                        <div className="stat-card">
                            <div className="stat-value">{summary.row_count}</div>
                            <div className="stat-label">Players</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value">{summary.starters}</div>
                            <div className="stat-label">Starters</div>
                        </div>
                        <div className={`stat-card ${summary.has_anomalies ? 'warning' : 'success'}`}>
                            <div className="stat-value">{summary.anomalies.length}</div>
                            <div className="stat-label">Anomalies</div>
                        </div>
                    </div>

                    {/* Anomalies */}
                    {summary.anomalies.length > 0 && (
                        <div className="anomalies-section">
                            <h3>⚠️ Detected Anomalies</h3>
                            <table className="data-table">
                                <thead>
                                    <tr>
                                        <th>Feature</th>
                                        <th>Issue</th>
                                        <th>Count</th>
                                        <th>Severity</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {summary.anomalies.map((a, i) => (
                                        <tr key={i} className={`severity-${a.severity}`}>
                                            <td><code>{a.feature}</code></td>
                                            <td>
                                                {a.issue === 'above_max'
                                                    ? `Above max (${a.expected_max})`
                                                    : `Below min (${a.expected_min})`}
                                            </td>
                                            <td>{a.count}</td>
                                            <td><span className={`badge badge-${a.severity}`}>{a.severity}</span></td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}

                    {/* Feature Stats */}
                    <div className="feature-stats">
                        <h3>Feature Statistics</h3>
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Feature</th>
                                    <th>Min</th>
                                    <th>Mean</th>
                                    <th>Max</th>
                                    <th>Missing</th>
                                </tr>
                            </thead>
                            <tbody>
                                {Object.entries(summary.stats).map(([feature, stat]) => (
                                    <tr key={feature}>
                                        <td><code>{feature}</code></td>
                                        <td>{formatNumber(stat.min, 4)}</td>
                                        <td>{formatNumber(stat.mean, 4)}</td>
                                        <td>{formatNumber(stat.max, 4)}</td>
                                        <td>{stat.missing}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* Projections Section */}
            {projections && (
                <div className="section">
                    <h2>Sim Projections (Top {projections.players.length})</h2>
                    <table className="data-table projections-table">
                        <thead>
                            <tr>
                                <th>Player</th>
                                <th>Team</th>
                                <th>Start</th>
                                <th>Min</th>
                                <th>FPTS Mean</th>
                                <th>FPTS P50</th>
                                <th>FPTS P10-P90</th>
                                <th>PTS</th>
                                <th>REB</th>
                                <th>AST</th>
                                <th>Season Rates</th>
                            </tr>
                        </thead>
                        <tbody>
                            {projections.players.map((p) => (
                                <tr
                                    key={p.player_id}
                                    onClick={() => fetchPlayerDetails(p.player_id)}
                                    className={`clickable ${selectedPlayer === p.player_id ? 'selected' : ''} ${p.player_name ? 'elite-player' : ''}`}
                                >
                                    <td>
                                        <span className="player-name">{p.player_name || p.player_id}</span>
                                    </td>
                                    <td>{p.team_id}</td>
                                    <td>{p.is_starter === 1 ? '✓' : '-'}</td>
                                    <td>{formatNumber(p.minutes_sim_mean || p.minutes_mean, 1)}</td>
                                    <td className="fpts-cell">{formatNumber(p.dk_fpts_mean, 1)}</td>
                                    <td>{formatNumber(p.dk_fpts_p50, 1)}</td>
                                    <td className="range-cell">
                                        {formatNumber(p.dk_fpts_p10, 0)}-{formatNumber(p.dk_fpts_p90, 0)}
                                    </td>
                                    <td>{formatNumber(p.pts_mean, 1)}</td>
                                    <td>{formatNumber(p.reb_mean, 1)}</td>
                                    <td>{formatNumber(p.ast_mean, 1)}</td>
                                    <td className="rates-cell">
                                        <small>
                                            FGA: {formatNumber(p.season_fga_per_min, 3)} |
                                            AST: {formatNumber(p.season_ast_per_min, 3)}
                                        </small>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {/* Player Details Modal/Panel */}
            {playerDetails && (
                <div className="player-details-panel">
                    <div className="panel-header">
                        <h3>{playerDetails.player_name || `Player ${playerDetails.player_id}`}</h3>
                        <button onClick={() => setPlayerDetails(null)} className="close-btn">×</button>
                    </div>
                    <div className="panel-content">
                        <h4>Features</h4>
                        <div className="details-grid">
                            {playerDetails.features && Object.entries(playerDetails.features)
                                .filter(([k]) => k.includes('season_') || k.includes('minutes'))
                                .map(([k, v]) => (
                                    <div key={k} className="detail-item">
                                        <span className="detail-label">{k}</span>
                                        <span className="detail-value">{formatNumber(v as number, 4)}</span>
                                    </div>
                                ))}
                        </div>
                        <h4>Projections</h4>
                        <div className="details-grid">
                            {playerDetails.projections && Object.entries(playerDetails.projections)
                                .filter(([k]) => k.includes('fpts') || k.includes('mean'))
                                .map(([k, v]) => (
                                    <div key={k} className="detail-item">
                                        <span className="detail-label">{k}</span>
                                        <span className="detail-value">{formatNumber(v as number, 2)}</span>
                                    </div>
                                ))}
                        </div>
                    </div>
                </div>
            )}

            <style>{`
        .diagnostics-page {
          padding: 20px;
          max-width: 1400px;
          margin: 0 auto;
        }
        
        .page-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 24px;
        }
        
        .page-header h1 {
          margin: 0;
          color: #1a1a2e;
        }
        
        .controls {
          display: flex;
          gap: 12px;
          align-items: center;
        }
        
        .date-input {
          padding: 8px 12px;
          border: 1px solid #ddd;
          border-radius: 6px;
          font-size: 14px;
        }
        
        .refresh-btn {
          padding: 8px 16px;
          background: #4361ee;
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-weight: 500;
        }
        
        .refresh-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }
        
        .error-banner {
          background: #fee2e2;
          color: #dc2626;
          padding: 12px;
          border-radius: 6px;
          margin-bottom: 16px;
        }
        
        .section {
          background: white;
          border-radius: 12px;
          padding: 20px;
          margin-bottom: 20px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .section h2 {
          margin: 0 0 16px 0;
          color: #1a1a2e;
          font-size: 18px;
        }
        
        .summary-cards {
          display: flex;
          gap: 16px;
          margin-bottom: 20px;
        }
        
        .stat-card {
          background: #f8fafc;
          padding: 16px 24px;
          border-radius: 8px;
          text-align: center;
          min-width: 100px;
        }
        
        .stat-card.warning {
          background: #fef3c7;
        }
        
        .stat-card.success {
          background: #d1fae5;
        }
        
        .stat-value {
          font-size: 28px;
          font-weight: bold;
          color: #1a1a2e;
        }
        
        .stat-label {
          font-size: 12px;
          color: #64748b;
          text-transform: uppercase;
          margin-top: 4px;
        }
        
        .data-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 13px;
        }
        
        .data-table th,
        .data-table td {
          padding: 10px 12px;
          text-align: left;
          border-bottom: 1px solid #e2e8f0;
        }
        
        .data-table th {
          background: #f8fafc;
          font-weight: 600;
          color: #475569;
        }
        
        .data-table tbody tr:hover {
          background: #f8fafc;
        }
        
        .data-table code {
          background: #e2e8f0;
          padding: 2px 6px;
          border-radius: 4px;
          font-size: 12px;
        }
        
        .badge {
          padding: 2px 8px;
          border-radius: 12px;
          font-size: 11px;
          font-weight: 500;
        }
        
        .badge-high {
          background: #fee2e2;
          color: #dc2626;
        }
        
        .badge-medium {
          background: #fef3c7;
          color: #d97706;
        }
        
        .badge-low {
          background: #dbeafe;
          color: #2563eb;
        }
        
        .projections-table {
          font-size: 12px;
        }
        
        .projections-table .clickable {
          cursor: pointer;
        }
        
        .projections-table .selected {
          background: #e0e7ff !important;
        }
        
        .projections-table .elite-player {
          font-weight: 500;
        }
        
        .fpts-cell {
          font-weight: 600;
          color: #059669;
        }
        
        .range-cell {
          color: #64748b;
          font-size: 11px;
        }
        
        .rates-cell {
          color: #64748b;
        }
        
        .player-details-panel {
          position: fixed;
          right: 0;
          top: 0;
          width: 400px;
          height: 100vh;
          background: white;
          box-shadow: -4px 0 20px rgba(0,0,0,0.15);
          z-index: 1000;
          overflow-y: auto;
        }
        
        .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 16px 20px;
          border-bottom: 1px solid #e2e8f0;
          position: sticky;
          top: 0;
          background: white;
        }
        
        .panel-header h3 {
          margin: 0;
        }
        
        .close-btn {
          background: none;
          border: none;
          font-size: 24px;
          cursor: pointer;
          color: #64748b;
        }
        
        .panel-content {
          padding: 20px;
        }
        
        .panel-content h4 {
          margin: 16px 0 12px 0;
          color: #475569;
          font-size: 14px;
        }
        
        .details-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px;
        }
        
        .detail-item {
          display: flex;
          justify-content: space-between;
          padding: 6px 8px;
          background: #f8fafc;
          border-radius: 4px;
        }
        
        .detail-label {
          font-size: 11px;
          color: #64748b;
        }
        
        .detail-value {
          font-weight: 500;
          font-size: 12px;
        }
      `}</style>
        </div>
    );
};

export default DiagnosticsPage;
