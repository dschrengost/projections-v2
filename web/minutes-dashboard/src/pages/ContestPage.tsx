import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  getContestDates,
  getContests,
  getContestDetail,
  getPairsAnalysis,
  getUserLineups,
  ContestSummary,
  ContestDetail,
  PlayerPair,
  UserLineupEntry,
  OwnershipMetrics,
  LeveragePlay,
  StackInfo,
  PlayerOwnership,
  LineupOwnershipEntry,
  OwnershipDistribution,
  DupeAnalysis,
  DuplicateLineup,
  DupesByOwnershipBand,
} from '../api/contest'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

type TabType = 'lineups' | 'distribution' | 'dupes' | 'pairs' | 'leverage' | 'stacks' | 'ownership' | 'mylineups'

const todayISO = () => new Date().toISOString().slice(0, 10)

const formatOwn = (value: number | null | undefined): string => {
  if (value == null) return '-'
  return `${value.toFixed(1)}%`
}

const formatScore = (value: number | null | undefined): string => {
  if (value == null) return '-'
  return value.toFixed(1)
}

const formatPct = (value: number | null | undefined): string => {
  if (value == null) return '-'
  return `${value.toFixed(1)}%`
}

const formatLeverage = (value: number): string => {
  if (value > 0) return `+${value.toFixed(2)}`
  return value.toFixed(2)
}

function OwnershipCard({
  label,
  metrics,
  score,
}: {
  label: string
  metrics: OwnershipMetrics | null
  score?: number | null
}) {
  return (
    <div className="card">
      <div className="label">{label}</div>
      <div className="value">{formatOwn(metrics?.avg_own)}</div>
      <div className="card-details">
        <span title="Players under 10% owned">{metrics?.num_under_10 ?? '-'} &lt;10%</span>
        <span title="Players under 5% owned">{metrics?.num_under_5 ?? '-'} &lt;5%</span>
      </div>
      {score != null && <div className="card-score">{formatScore(score)} pts</div>}
    </div>
  )
}

export default function ContestPage() {
  // Date selection
  const [availableDates, setAvailableDates] = useState<string[]>([])
  const [selectedDate, setSelectedDate] = useState(todayISO())
  const [datesLoading, setDatesLoading] = useState(false)

  // Contest selection
  const [contests, setContests] = useState<ContestSummary[]>([])
  const [contestsLoading, setContestsLoading] = useState(false)
  const [selectedContestId, setSelectedContestId] = useState<string | null>(null)

  // Contest detail
  const [contestDetail, setContestDetail] = useState<ContestDetail | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)

  // Pairs (lazy loaded)
  const [pairs, setPairs] = useState<PlayerPair[]>([])
  const [pairsLoading, setPairsLoading] = useState(false)
  const [pairsTopN, setPairsTopN] = useState(10)

  // User lineups
  const [userPattern, setUserPattern] = useState(() =>
    localStorage.getItem('contest_user_pattern') || ''
  )
  const [userLineups, setUserLineups] = useState<UserLineupEntry[]>([])
  const [userLineupsLoading, setUserLineupsLoading] = useState(false)
  const [userStats, setUserStats] = useState<{
    avgFinishPct: number | null
    bestRank: number | null
    bestPct: number | null
  }>({ avgFinishPct: null, bestRank: null, bestPct: null })

  // Active tab
  const [activeTab, setActiveTab] = useState<TabType>('lineups')

  // Error state
  const [error, setError] = useState<string | null>(null)

  // Player ownership sort
  const [ownershipSort, setOwnershipSort] = useState<'ownership' | 'fpts' | 'win' | 'top10'>('ownership')
  const [ownershipSortDir, setOwnershipSortDir] = useState<'asc' | 'desc'>('desc')

  // Load available dates on mount
  useEffect(() => {
    const loadDates = async () => {
      setDatesLoading(true)
      try {
        const dates = await getContestDates(60)
        setAvailableDates(dates)
        if (dates.length > 0 && !dates.includes(selectedDate)) {
          setSelectedDate(dates[0])
        }
      } catch (err) {
        setError((err as Error).message)
      } finally {
        setDatesLoading(false)
      }
    }
    void loadDates()
  }, [])

  // Load contests when date changes
  useEffect(() => {
    const loadContests = async () => {
      setContestsLoading(true)
      setContestDetail(null)
      setSelectedContestId(null)
      setPairs([])
      setError(null)
      try {
        const data = await getContests(selectedDate)
        setContests(data)
        if (data.length > 0) {
          setSelectedContestId(data[0].contest_id)
        }
      } catch (err) {
        setError((err as Error).message)
        setContests([])
      } finally {
        setContestsLoading(false)
      }
    }
    void loadContests()
  }, [selectedDate])

  // Load contest detail when contest changes
  useEffect(() => {
    if (!selectedContestId) {
      setContestDetail(null)
      return
    }
    const loadDetail = async () => {
      setDetailLoading(true)
      setError(null)
      try {
        const detail = await getContestDetail(selectedContestId, selectedDate)
        setContestDetail(detail)
      } catch (err) {
        setError((err as Error).message)
        setContestDetail(null)
      } finally {
        setDetailLoading(false)
      }
    }
    void loadDetail()
  }, [selectedContestId, selectedDate])

  // Load pairs when pairs tab is active
  useEffect(() => {
    if (activeTab !== 'pairs' || !selectedContestId) {
      return
    }
    const loadPairs = async () => {
      setPairsLoading(true)
      try {
        const data = await getPairsAnalysis(selectedContestId, selectedDate, pairsTopN)
        setPairs(data.pairs)
      } catch (err) {
        setError((err as Error).message)
        setPairs([])
      } finally {
        setPairsLoading(false)
      }
    }
    void loadPairs()
  }, [activeTab, selectedContestId, selectedDate, pairsTopN])

  // Load user lineups when tab is active and pattern exists
  const loadUserLineups = useCallback(async () => {
    if (!userPattern || userPattern.length < 2) {
      setUserLineups([])
      return
    }
    setUserLineupsLoading(true)
    localStorage.setItem('contest_user_pattern', userPattern)
    try {
      const data = await getUserLineups(selectedDate, userPattern)
      setUserLineups(data.entries)
      setUserStats({
        avgFinishPct: data.avg_finish_pct,
        bestRank: data.best_finish_rank,
        bestPct: data.best_finish_pct,
      })
    } catch (err) {
      setError((err as Error).message)
      setUserLineups([])
    } finally {
      setUserLineupsLoading(false)
    }
  }, [selectedDate, userPattern])

  useEffect(() => {
    if (activeTab === 'mylineups' && userPattern.length >= 2) {
      void loadUserLineups()
    }
  }, [activeTab, selectedDate])

  // Sorted player ownership
  const sortedOwnership = useMemo(() => {
    if (!contestDetail?.player_ownership) return []
    const sorted = [...contestDetail.player_ownership]
    sorted.sort((a, b) => {
      let aVal: number, bVal: number
      switch (ownershipSort) {
        case 'ownership':
          aVal = a.ownership_pct
          bVal = b.ownership_pct
          break
        case 'fpts':
          aVal = a.fpts ?? 0
          bVal = b.fpts ?? 0
          break
        case 'win':
          aVal = a.win_pct ?? 0
          bVal = b.win_pct ?? 0
          break
        case 'top10':
          aVal = a.top10_pct ?? 0
          bVal = b.top10_pct ?? 0
          break
        default:
          aVal = a.ownership_pct
          bVal = b.ownership_pct
      }
      return ownershipSortDir === 'desc' ? bVal - aVal : aVal - bVal
    })
    return sorted
  }, [contestDetail?.player_ownership, ownershipSort, ownershipSortDir])

  const toggleOwnershipSort = (key: typeof ownershipSort) => {
    if (ownershipSort === key) {
      setOwnershipSortDir(prev => (prev === 'desc' ? 'asc' : 'desc'))
    } else {
      setOwnershipSort(key)
      setOwnershipSortDir('desc')
    }
  }

  const selectedContest = contests.find(c => c.contest_id === selectedContestId)

  return (
    <div className="contest-page">
      {/* Header */}
      <header className="contest-header">
        <div>
          <h1>Contest Analyzer</h1>
          <p className="subtitle">Post-contest GPP analysis and insights</p>
        </div>
        <div className="contest-controls">
          <label>
            Date
            <select
              value={selectedDate}
              onChange={e => setSelectedDate(e.target.value)}
              disabled={datesLoading}
            >
              {availableDates.map(d => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </label>
          <label>
            Contest
            <select
              value={selectedContestId ?? ''}
              onChange={e => setSelectedContestId(e.target.value || null)}
              disabled={contestsLoading || contests.length === 0}
            >
              {contests.length === 0 && <option value="">No contests</option>}
              {contests.map(c => (
                <option key={c.contest_id} value={c.contest_id}>
                  {c.contest_name} ({c.total_entries.toLocaleString()})
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>

      {error && (
        <div className="contest-alert error">Error: {error}</div>
      )}

      {/* Summary Cards */}
      {contestDetail && (
        <>
          <div className="contest-summary">
            <OwnershipCard
              label="Winner"
              metrics={contestDetail.winner}
              score={contestDetail.winner_score}
            />
            <OwnershipCard label="Top 10" metrics={contestDetail.top10} />
            <OwnershipCard
              label="Top 1%"
              metrics={contestDetail.top1pct}
              score={contestDetail.top1pct_score}
            />
            <OwnershipCard
              label="Cash Line"
              metrics={contestDetail.cash_line}
              score={contestDetail.cash_line_score}
            />
          </div>

          <div className="contest-meta">
            <span>
              {contestDetail.total_entries.toLocaleString()} entries
              {selectedContest?.entry_fee != null && ` | $${selectedContest.entry_fee} entry`}
              {selectedContest?.contest_type && ` | ${selectedContest.contest_type}`}
            </span>
          </div>

          {/* Dupe Stats */}
          {contestDetail.dupe_analysis && (
            <div className="contest-meta">
              <span>
                {contestDetail.dupe_analysis.unique_lineups.toLocaleString()} unique lineups
                ({contestDetail.dupe_analysis.duplicate_entries.toLocaleString()} dupes,{' '}
                {contestDetail.dupe_analysis.duplicate_rate}% dupe rate)
              </span>
            </div>
          )}
        </>
      )}

      {/* Tabs */}
      <div className="contest-tabs">
        <button
          className={activeTab === 'lineups' ? 'active' : ''}
          onClick={() => setActiveTab('lineups')}
        >
          Top Lineups
        </button>
        <button
          className={activeTab === 'distribution' ? 'active' : ''}
          onClick={() => setActiveTab('distribution')}
        >
          Ownership Distribution
        </button>
        <button
          className={activeTab === 'dupes' ? 'active' : ''}
          onClick={() => setActiveTab('dupes')}
        >
          Dupes
        </button>
        <button
          className={activeTab === 'pairs' ? 'active' : ''}
          onClick={() => setActiveTab('pairs')}
        >
          Pairs
        </button>
        <button
          className={activeTab === 'leverage' ? 'active' : ''}
          onClick={() => setActiveTab('leverage')}
        >
          Leverage
        </button>
        <button
          className={activeTab === 'stacks' ? 'active' : ''}
          onClick={() => setActiveTab('stacks')}
        >
          Stacks
        </button>
        <button
          className={activeTab === 'ownership' ? 'active' : ''}
          onClick={() => setActiveTab('ownership')}
        >
          Player Own%
        </button>
        <button
          className={activeTab === 'mylineups' ? 'active' : ''}
          onClick={() => setActiveTab('mylineups')}
        >
          My Lineups
        </button>
      </div>

      {/* Tab Content */}
      <div className="contest-tab-content">
        {/* Top Lineups Tab */}
        {activeTab === 'lineups' && (
          <div className="lineups-tab">
            <p className="tab-description">
              Top 100 finishing lineups with total ownership (sum of 8 players' ownership %).
              Lower total ownership = more contrarian.
            </p>
            {detailLoading ? (
              <div className="loading">Loading lineup data...</div>
            ) : contestDetail?.top_lineups && contestDetail.top_lineups.length > 0 ? (
              <div className="table-wrapper">
                <table className="top-lineups-table">
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Entry Name</th>
                      <th>Score</th>
                      <th>Total Own%</th>
                      <th>Avg Own%</th>
                      <th>&lt;10%</th>
                      <th>&lt;5%</th>
                      <th>Lineup</th>
                    </tr>
                  </thead>
                  <tbody>
                    {contestDetail.top_lineups.map((lineup, i) => (
                      <tr key={i} className={lineup.rank <= 10 ? 'top10-lineup' : ''}>
                        <td>
                          <strong>#{lineup.rank}</strong>
                        </td>
                        <td>{lineup.entry_name}</td>
                        <td>{formatScore(lineup.points)}</td>
                        <td className="total-own-cell">
                          <strong>{formatOwn(lineup.total_own)}</strong>
                        </td>
                        <td>{formatOwn(lineup.avg_own)}</td>
                        <td>{lineup.num_under_10}</td>
                        <td>{lineup.num_under_5}</td>
                        <td className="lineup-players-cell">
                          <div className="lineup-players-list">
                            {lineup.lineup_players.map((p, j) => (
                              <span key={j} className="player-chip">
                                {p}
                              </span>
                            ))}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="muted">No lineup data available</div>
            )}
          </div>
        )}

        {/* Ownership Distribution Tab */}
        {activeTab === 'distribution' && (
          <div className="distribution-tab">
            <p className="tab-description">
              Distribution of total ownership across all lineups in the contest.
              Shows how many lineups fell into each ownership bucket.
            </p>
            {detailLoading ? (
              <div className="loading">Loading distribution...</div>
            ) : contestDetail?.ownership_distribution && contestDetail.ownership_distribution.length > 0 ? (
              <>
                <div className="chart-container" style={{ height: 400 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={contestDetail.ownership_distribution}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis
                        dataKey="bin_label"
                        stroke="#94a3b8"
                        fontSize={12}
                        angle={-45}
                        textAnchor="end"
                        height={80}
                      />
                      <YAxis
                        stroke="#94a3b8"
                        fontSize={12}
                        label={{ value: '# of Lineups', angle: -90, position: 'insideLeft' }}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                        labelStyle={{ color: '#f8fafc' }}
                      />
                      <Bar dataKey="count" fill="#4f46e5" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="table-wrapper" style={{ marginTop: '1rem' }}>
                  <table className="distribution-table">
                    <thead>
                      <tr>
                        <th>Total Own% Range</th>
                        <th>Count</th>
                        <th>% of Field</th>
                      </tr>
                    </thead>
                    <tbody>
                      {contestDetail.ownership_distribution.map((bin, i) => (
                        <tr key={i}>
                          <td><strong>{bin.bin_label}</strong></td>
                          <td>{bin.count.toLocaleString()}</td>
                          <td>{formatPct(bin.pct_of_total)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <div className="muted">No distribution data available</div>
            )}
          </div>
        )}

        {/* Dupes Tab */}
        {activeTab === 'dupes' && (
          <div className="dupes-tab">
            {detailLoading ? (
              <div className="loading">Loading duplicate analysis...</div>
            ) : contestDetail?.dupe_analysis ? (
              <>
                {/* Summary */}
                <div className="dupes-summary">
                  <div className="dupe-stat-card">
                    <div className="dupe-stat-value">{contestDetail.dupe_analysis.total_entries.toLocaleString()}</div>
                    <div className="dupe-stat-label">Total Entries</div>
                  </div>
                  <div className="dupe-stat-card">
                    <div className="dupe-stat-value">{contestDetail.dupe_analysis.unique_lineups.toLocaleString()}</div>
                    <div className="dupe-stat-label">Unique Lineups</div>
                  </div>
                  <div className="dupe-stat-card">
                    <div className="dupe-stat-value">{contestDetail.dupe_analysis.duplicate_entries.toLocaleString()}</div>
                    <div className="dupe-stat-label">Duplicate Entries</div>
                  </div>
                  <div className="dupe-stat-card highlight">
                    <div className="dupe-stat-value">{contestDetail.dupe_analysis.duplicate_rate}%</div>
                    <div className="dupe-stat-label">Dupe Rate</div>
                  </div>
                </div>

                {/* Most Duped Lineup */}
                {contestDetail.dupe_analysis.most_duped_lineup && (
                  <>
                    <h3>Most Duplicated Lineup</h3>
                    <div className="most-duped-card">
                      <div className="most-duped-header">
                        <span className="most-duped-count">
                          {contestDetail.dupe_analysis.most_duped_lineup.entry_count}x entries
                        </span>
                        <span>Best Rank: #{contestDetail.dupe_analysis.most_duped_lineup.best_rank}</span>
                        <span>Total Own: {formatOwn(contestDetail.dupe_analysis.most_duped_lineup.total_own)}</span>
                      </div>
                      <div className="most-duped-lineup">
                        {contestDetail.dupe_analysis.most_duped_lineup.lineup_players.map((p, i) => (
                          <span key={i} className="player-chip">{p}</span>
                        ))}
                      </div>
                      <div className="most-duped-entries">
                        <strong>Entry Names:</strong>{' '}
                        {contestDetail.dupe_analysis.most_duped_lineup.entry_names.slice(0, 10).join(', ')}
                        {contestDetail.dupe_analysis.most_duped_lineup.entry_names.length > 10 &&
                          ` ... and ${contestDetail.dupe_analysis.most_duped_lineup.entry_names.length - 10} more`}
                      </div>
                    </div>
                  </>
                )}

                {/* Top Duped Lineups Table */}
                <h3>Top Duplicated Lineups</h3>
                <div className="table-wrapper">
                  <table className="duped-lineups-table">
                    <thead>
                      <tr>
                        <th>Entries</th>
                        <th>Best Rank</th>
                        <th>Total Own%</th>
                        <th>Avg Own%</th>
                        <th>Lineup</th>
                      </tr>
                    </thead>
                    <tbody>
                      {contestDetail.dupe_analysis.top_duped_lineups.map((dupe, i) => (
                        <tr key={i}>
                          <td>
                            <strong>{dupe.entry_count}x</strong>
                          </td>
                          <td>#{dupe.best_rank}</td>
                          <td>{formatOwn(dupe.total_own)}</td>
                          <td>{formatOwn(dupe.avg_own)}</td>
                          <td className="lineup-players-cell">
                            <div className="lineup-players-list">
                              {dupe.lineup_players.map((p, j) => (
                                <span key={j} className="player-chip">{p}</span>
                              ))}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Dupes by Ownership Band */}
                {contestDetail.dupe_analysis.dupes_by_ownership.length > 0 && (
                  <>
                    <h3>Duplicates by Ownership Band</h3>
                    <div className="table-wrapper">
                      <table className="dupes-by-ownership-table">
                        <thead>
                          <tr>
                            <th>Ownership Band</th>
                            <th>Total Entries</th>
                            <th>Unique Lineups</th>
                            <th>Duped Entries</th>
                            <th>Dupe Rate</th>
                          </tr>
                        </thead>
                        <tbody>
                          {contestDetail.dupe_analysis.dupes_by_ownership.map((band, i) => (
                            <tr key={i}>
                              <td><strong>{band.ownership_band}</strong></td>
                              <td>{band.total_lineups.toLocaleString()}</td>
                              <td>{band.unique_lineups.toLocaleString()}</td>
                              <td>{band.duped_lineups.toLocaleString()}</td>
                              <td className={band.dupe_rate > 50 ? 'high-dupe-rate' : ''}>
                                {band.dupe_rate}%
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}
              </>
            ) : (
              <div className="muted">No dupe data available</div>
            )}
          </div>
        )}

        {/* Ownership Tab */}
        {activeTab === 'ownership' && (
          <div className="ownership-tab">
            {detailLoading ? (
              <div className="loading">Loading ownership data...</div>
            ) : sortedOwnership.length > 0 ? (
              <div className="table-wrapper">
                <table className="ownership-table">
                  <thead>
                    <tr>
                      <th
                        className="sortable"
                        onClick={() => toggleOwnershipSort('ownership')}
                      >
                        Player
                        {ownershipSort === 'ownership' && (
                          <span>{ownershipSortDir === 'desc' ? ' ▼' : ' ▲'}</span>
                        )}
                      </th>
                      <th
                        className="sortable"
                        onClick={() => toggleOwnershipSort('ownership')}
                      >
                        Own%
                        {ownershipSort === 'ownership' && (
                          <span>{ownershipSortDir === 'desc' ? ' ▼' : ' ▲'}</span>
                        )}
                      </th>
                      <th
                        className="sortable"
                        onClick={() => toggleOwnershipSort('fpts')}
                      >
                        FPTS
                        {ownershipSort === 'fpts' && (
                          <span>{ownershipSortDir === 'desc' ? ' ▼' : ' ▲'}</span>
                        )}
                      </th>
                      <th
                        className="sortable"
                        onClick={() => toggleOwnershipSort('win')}
                      >
                        Win%
                        {ownershipSort === 'win' && (
                          <span>{ownershipSortDir === 'desc' ? ' ▼' : ' ▲'}</span>
                        )}
                      </th>
                      <th
                        className="sortable"
                        onClick={() => toggleOwnershipSort('top10')}
                      >
                        Top10%
                        {ownershipSort === 'top10' && (
                          <span>{ownershipSortDir === 'desc' ? ' ▼' : ' ▲'}</span>
                        )}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedOwnership.map(p => (
                      <tr key={p.player}>
                        <td>
                          <strong>{p.player}</strong>
                        </td>
                        <td>{formatOwn(p.ownership_pct)}</td>
                        <td>{formatScore(p.fpts)}</td>
                        <td>{formatPct(p.win_pct)}</td>
                        <td>{formatPct(p.top10_pct)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="muted">No ownership data available</div>
            )}
          </div>
        )}

        {/* Pairs Tab */}
        {activeTab === 'pairs' && (
          <div className="pairs-tab">
            <div className="pairs-controls">
              <label>
                Analyze Top
                <select
                  value={pairsTopN}
                  onChange={e => setPairsTopN(Number(e.target.value))}
                >
                  <option value={1}>Winner Only</option>
                  <option value={10}>Top 10</option>
                  <option value={25}>Top 25</option>
                  <option value={50}>Top 50</option>
                  <option value={100}>Top 100</option>
                </select>
              </label>
            </div>
            {pairsLoading ? (
              <div className="loading">Loading pairs analysis...</div>
            ) : pairs.length > 0 ? (
              <div className="table-wrapper">
                <table className="pairs-table">
                  <thead>
                    <tr>
                      <th>Player 1</th>
                      <th>Player 2</th>
                      <th>Count</th>
                      <th>% of Top {pairsTopN}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pairs.map((p, i) => (
                      <tr key={i}>
                        <td>{p.player1}</td>
                        <td>{p.player2}</td>
                        <td>{p.count}</td>
                        <td>{formatPct(p.pct)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="muted">No pairs data available</div>
            )}
          </div>
        )}

        {/* Leverage Tab */}
        {activeTab === 'leverage' && (
          <div className="leverage-tab">
            <p className="tab-description">
              Players who appeared in winning lineups more than their ownership suggests.
              Positive leverage = underowned winners.
            </p>
            {detailLoading ? (
              <div className="loading">Loading leverage data...</div>
            ) : contestDetail?.leverage_plays && contestDetail.leverage_plays.length > 0 ? (
              <div className="table-wrapper">
                <table className="leverage-table">
                  <thead>
                    <tr>
                      <th>Player</th>
                      <th>Own%</th>
                      <th>Win Rate</th>
                      <th>Leverage</th>
                    </tr>
                  </thead>
                  <tbody>
                    {contestDetail.leverage_plays.map((p, i) => (
                      <tr key={i} className={p.leverage_score > 0 ? 'positive-leverage' : ''}>
                        <td>
                          <strong>{p.player}</strong>
                        </td>
                        <td>{formatOwn(p.ownership_pct)}</td>
                        <td>{formatPct(p.win_rate)}</td>
                        <td className={p.leverage_score > 0 ? 'leverage-positive' : 'leverage-negative'}>
                          {formatLeverage(p.leverage_score)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="muted">No leverage data available</div>
            )}
          </div>
        )}

        {/* Stacks Tab */}
        {activeTab === 'stacks' && (
          <div className="stacks-tab">
            <p className="tab-description">
              Team stacking patterns in top finishing lineups.
            </p>
            {detailLoading ? (
              <div className="loading">Loading stacks data...</div>
            ) : contestDetail?.stacks && contestDetail.stacks.length > 0 ? (
              <div className="table-wrapper">
                <table className="stacks-table">
                  <thead>
                    <tr>
                      <th>Team</th>
                      <th>Stack Size</th>
                      <th>Count</th>
                      <th>% of Top</th>
                    </tr>
                  </thead>
                  <tbody>
                    {contestDetail.stacks.map((s, i) => (
                      <tr key={i}>
                        <td>
                          <strong>{s.team}</strong>
                        </td>
                        <td>{s.stack_size}-stack</td>
                        <td>{s.count}</td>
                        <td>{formatPct(s.pct_of_top)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="muted">No stacking data available (requires team mapping)</div>
            )}

            {/* Game Correlations */}
            {contestDetail?.game_correlations && contestDetail.game_correlations.length > 0 && (
              <>
                <h3>Game Correlations</h3>
                <p className="tab-description">
                  Which teams/games produced the most players in winning lineups.
                </p>
                <div className="table-wrapper">
                  <table className="correlations-table">
                    <thead>
                      <tr>
                        <th>Team</th>
                        <th>Players in Winners</th>
                        <th>Total FPTS</th>
                        <th>Avg FPTS</th>
                      </tr>
                    </thead>
                    <tbody>
                      {contestDetail.game_correlations.map((g, i) => (
                        <tr key={i}>
                          <td>
                            <strong>{g.matchup}</strong>
                          </td>
                          <td>{g.players_in_winners}</td>
                          <td>{formatScore(g.total_fpts)}</td>
                          <td>{formatScore(g.avg_fpts)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        )}

        {/* My Lineups Tab */}
        {activeTab === 'mylineups' && (
          <div className="mylineups-tab">
            <div className="user-pattern-input">
              <label>
                Username Pattern
                <input
                  type="text"
                  value={userPattern}
                  onChange={e => setUserPattern(e.target.value)}
                  placeholder="Your DraftKings username"
                />
              </label>
              <button onClick={loadUserLineups} disabled={userPattern.length < 2}>
                Search
              </button>
            </div>

            {userLineupsLoading ? (
              <div className="loading">Searching for your entries...</div>
            ) : userLineups.length > 0 ? (
              <>
                <div className="user-summary">
                  <span>Found {userLineups.length} entries</span>
                  {userStats.bestRank != null && (
                    <span>Best: #{userStats.bestRank} ({formatPct(userStats.bestPct)})</span>
                  )}
                  {userStats.avgFinishPct != null && (
                    <span>Avg Finish: {formatPct(userStats.avgFinishPct)}</span>
                  )}
                </div>
                <div className="table-wrapper">
                  <table className="user-lineups-table">
                    <thead>
                      <tr>
                        <th>Contest</th>
                        <th>Entry</th>
                        <th>Rank</th>
                        <th>Finish %</th>
                        <th>Score</th>
                        <th>Avg Own%</th>
                        <th>&lt;10%</th>
                      </tr>
                    </thead>
                    <tbody>
                      {userLineups.map((entry, i) => (
                        <tr key={i}>
                          <td title={entry.contest_name}>
                            {entry.contest_name.length > 30
                              ? entry.contest_name.slice(0, 30) + '...'
                              : entry.contest_name}
                          </td>
                          <td>{entry.entry_name}</td>
                          <td>
                            #{entry.rank} / {entry.total_entries.toLocaleString()}
                          </td>
                          <td className={entry.pct_finish <= 20 ? 'cash-finish' : ''}>
                            {formatPct(entry.pct_finish)}
                          </td>
                          <td>{formatScore(entry.points)}</td>
                          <td>{formatOwn(entry.ownership_metrics.avg_own)}</td>
                          <td>{entry.ownership_metrics.num_under_10}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : userPattern.length >= 2 ? (
              <div className="muted">No entries found for "{userPattern}" on {selectedDate}</div>
            ) : (
              <div className="muted">Enter your username to search for your entries</div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
