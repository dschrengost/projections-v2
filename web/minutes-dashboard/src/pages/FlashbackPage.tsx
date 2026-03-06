import { useEffect, useMemo, useState } from 'react'

import {
  FlashbackCalibrationResponse,
  FlashbackContestSummary,
  FlashbackRunResponse,
  listFlashbackContests,
  runFlashback,
  runFlashbackCalibration,
} from '../api/flashback'
import { Badge } from '../components/ui/badge'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Input } from '../components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../components/ui/select'

type PreviewRow = Record<string, unknown>

const todayString = () => new Date().toISOString().slice(0, 10)
const ISO_DATE_RE = /^\d{4}-\d{2}-\d{2}$/

function normalizeGameDateInput(value: string): string {
  const text = value.trim()
  if (!text) return ''
  const isoMatch = text.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/)
  if (isoMatch) {
    const [, year, month, day] = isoMatch
    return `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`
  }
  const slashMatch = text.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/)
  if (slashMatch) {
    const [, month, day, year] = slashMatch
    return `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`
  }
  return text
}

function formatPrimitive(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return '—'
    if (Math.abs(value) >= 1000) return value.toFixed(0)
    if (Math.abs(value) >= 10) return value.toFixed(2)
    return value.toFixed(4)
  }
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  return String(value)
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function objectArray(value: unknown): PreviewRow[] {
  if (!Array.isArray(value)) return []
  return value.filter((item): item is PreviewRow => isPlainObject(item))
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.filter((item): item is string => typeof item === 'string').map((item) => item.trim()).filter(Boolean)
}

function scalarEntries(values: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(values).filter(([, value]) => {
      if (value === null || value === undefined || value === '') return false
      if (Array.isArray(value)) return false
      return !isPlainObject(value)
    }),
  )
}

function SummaryGrid({ summary }: { summary: Record<string, unknown> }) {
  const items = Object.entries(summary)
  if (!items.length) return null
  return (
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
      {items.map(([key, value]) => (
        <div key={key} className="rounded-md border border-[hsl(var(--border))] bg-[hsl(var(--muted))] p-3">
          <div className="text-[11px] uppercase tracking-[0.12em] text-[hsl(var(--muted-foreground))]">{key}</div>
          <div className="mt-1 text-sm font-medium text-[hsl(var(--foreground))] break-words">{formatPrimitive(value)}</div>
        </div>
      ))}
    </div>
  )
}

function PreviewTable({ title, rows }: { title: string; rows: PreviewRow[] }) {
  const columns = useMemo(() => {
    const colSet = new Set<string>()
    rows.forEach((row) => Object.keys(row).forEach((key) => colSet.add(key)))
    return Array.from(colSet).slice(0, 8)
  }, [rows])

  return (
    <Card className="overflow-hidden">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{rows.length} preview rows</CardDescription>
      </CardHeader>
      <CardContent>
        {!rows.length ? (
          <div className="text-sm text-[hsl(var(--muted-foreground))]">No rows yet.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-left text-sm">
              <thead>
                <tr className="border-b border-[hsl(var(--border))]">
                  {columns.map((column) => (
                    <th key={column} className="px-2 py-2 font-medium text-[hsl(var(--muted-foreground))]">
                      {column}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, idx) => (
                  <tr key={`${title}-${idx}`} className="border-b border-[hsl(var(--border))]/60">
                    {columns.map((column) => (
                      <td key={`${title}-${idx}-${column}`} className="px-2 py-2 align-top text-[hsl(var(--foreground))]">
                        {formatPrimitive(row[column])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function MetricCards({ title, values }: { title: string; values: Record<string, unknown> }) {
  const items = Object.entries(scalarEntries(values))
  if (!items.length) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <SummaryGrid summary={Object.fromEntries(items)} />
      </CardContent>
    </Card>
  )
}

function InsightList({ title, items }: { title: string; items: string[] }) {
  if (!items.length) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {items.map((item, idx) => (
            <div
              key={`${title}-${idx}`}
              className="rounded-md border border-[hsl(var(--border))] bg-[hsl(var(--muted))] px-3 py-2 text-sm text-[hsl(var(--foreground))]"
            >
              {item}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

function EnteredLineupsTable({ rows }: { rows: PreviewRow[] }) {
  const columns = [
    'entry_name',
    'realized_rank',
    'realized_points',
    'sim_roi',
    'sim_cash_rate',
    'sim_top1pct_rate',
    'sim_win_rate',
    'realized_score_sim_percentile',
    'opponent_dupe_count',
  ]
  return (
    <Card className="overflow-hidden">
      <CardHeader>
        <CardTitle>Your entered lineups</CardTitle>
        <CardDescription>Replay ROI and realized result for your entered contest lineups.</CardDescription>
      </CardHeader>
      <CardContent>
        {!rows.length ? (
          <div className="text-sm text-[hsl(var(--muted-foreground))]">No entered-lineup rows yet.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-left text-sm">
              <thead>
                <tr className="border-b border-[hsl(var(--border))]">
                  {columns.map((column) => (
                    <th key={column} className="px-2 py-2 font-medium text-[hsl(var(--muted-foreground))]">
                      {column}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, idx) => (
                  <tr key={`entered-${idx}`} className="border-b border-[hsl(var(--border))]/60">
                    {columns.map((column) => (
                      <td key={`entered-${idx}-${column}`} className="px-2 py-2 align-top text-[hsl(var(--foreground))]">
                        {formatPrimitive(row[column])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default function FlashbackPage() {
  const [gameDate, setGameDate] = useState(todayString())
  const [userPattern, setUserPattern] = useState('')
  const [contestId, setContestId] = useState('')
  const [draftGroupId, setDraftGroupId] = useState('')
  const [entryFee, setEntryFee] = useState('')
  const [contests, setContests] = useState<FlashbackContestSummary[]>([])
  const [selectedContestId, setSelectedContestId] = useState<string>('manual')
  const [loadingContests, setLoadingContests] = useState(false)
  const [runningReplay, setRunningReplay] = useState(false)
  const [runningCalibration, setRunningCalibration] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [flashback, setFlashback] = useState<FlashbackRunResponse | null>(null)
  const [calibration, setCalibration] = useState<FlashbackCalibrationResponse | null>(null)

  useEffect(() => {
    if (selectedContestId === 'manual') return
    const contest = contests.find((item) => item.contest_id === selectedContestId)
    if (!contest) return
    setContestId(contest.contest_id)
    setDraftGroupId(contest.draft_group_id ? String(contest.draft_group_id) : '')
    setEntryFee(contest.entry_fee != null ? String(contest.entry_fee) : '')
  }, [selectedContestId, contests])

  useEffect(() => {
    if (selectedContestId === 'manual') return
    const normalizedGameDate = normalizeGameDateInput(gameDate)
    if (!ISO_DATE_RE.test(normalizedGameDate)) return
    if (!contestId.trim() || !userPattern.trim()) return
    let cancelled = false
    setFlashback(null)
    ;(async () => {
      try {
        const response = await runFlashback({
          game_date: normalizedGameDate,
          contest_id: contestId.trim(),
          user_pattern: userPattern.trim(),
          draft_group_id: draftGroupId.trim() ? Number(draftGroupId) : undefined,
          entry_fee: entryFee.trim() ? Number(entryFee) : undefined,
          archetype: 'medium',
          worlds_source: 'gtv2',
          ownership_mode: 'field_only',
          modeled_field_version: 'v1_calibrated',
          include_modeled_field: true,
          load_existing_only: true,
        })
        if (cancelled) return
        setFlashback(response)
        setError(null)
        setSuccess('Loaded existing flashback artifacts.')
      } catch (err) {
        if (cancelled) return
        const message = err instanceof Error ? err.message : 'Failed to load existing flashback artifacts.'
        if (message.includes('No existing flashback artifacts found')) {
          return
        }
        setError(message)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [selectedContestId, gameDate, contestId, userPattern, draftGroupId, entryFee])

  const replaySummary = (flashback?.summary ?? {}) as Record<string, unknown>
  const replayCounts = (replaySummary.counts ?? {}) as Record<string, unknown>
  const replayArtifacts = (replaySummary.artifacts ?? {}) as Record<string, unknown>
  const replayResolution = (replaySummary.resolution ?? {}) as Record<string, unknown>
  const userReplaySummary = (replaySummary.user_replay_summary ?? {}) as Record<string, unknown>
  const fieldSummary = (replaySummary.field_summary ?? {}) as Record<string, unknown>
  const regretSummary = (replaySummary.regret_summary ?? {}) as Record<string, unknown>
  const attributionSummary = (replaySummary.attribution_summary ?? {}) as Record<string, unknown>
  const serverDecisionGuidance = stringArray(replaySummary.decision_guidance)
  const calibrationSummary = (calibration?.summary ?? {}) as Record<string, unknown>
  const calibrationArtifactCounts = isPlainObject(calibrationSummary.artifact_counts)
    ? calibrationSummary.artifact_counts
    : {}
  const unresolvedExamples = objectArray(replayResolution.unresolved_examples)
  const ambiguousExamples = objectArray(replayResolution.ambiguous_examples)
  const fuzzyExamples = objectArray(replayResolution.fuzzy_examples)

  const replayInsights = useMemo(() => {
    if (serverDecisionGuidance.length > 0) {
      return serverDecisionGuidance
    }
    const items: string[] = []
    const bestSimRoi = Number(userReplaySummary.best_sim_roi)
    if (Number.isFinite(bestSimRoi)) {
      items.push(
        bestSimRoi > 0
          ? `Your best entered lineup had positive sim ROI (${bestSimRoi.toFixed(3)}). A weak realized finish there is more likely variance than lineup quality.`
          : `Your best entered lineup had negative sim ROI (${bestSimRoi.toFixed(3)}). That points to an ex ante lineup-quality miss, not just variance.`,
      )
    }
    const avgSimRoi = Number(userReplaySummary.avg_sim_roi)
    if (Number.isFinite(avgSimRoi)) {
      items.push(`Average entered-lineup sim ROI was ${avgSimRoi.toFixed(3)} across this contest replay.`)
    }
    const selectionRegret = Number(regretSummary.selection_regret_roi)
    if (Number.isFinite(selectionRegret)) {
      items.push(
        selectionRegret > 0.02
          ? `Selection regret was ${selectionRegret.toFixed(3)} ROI. Better candidate lineups existed than the set you entered.`
          : `Selection regret was ${selectionRegret.toFixed(3)} ROI. Final lineup selection was not the main miss on this contest.`,
      )
    }
    const slotResolutionRate = Number(replayResolution.slot_resolution_rate)
    if (Number.isFinite(slotResolutionRate)) {
      items.push(
        slotResolutionRate >= 0.995
          ? `Player-slot resolution was strong at ${(slotResolutionRate * 100).toFixed(1)}%. Player-level replay diagnostics should be trustworthy.`
          : `Player-slot resolution was only ${(slotResolutionRate * 100).toFixed(1)}%. Review unresolved names before trusting player-level calibration.`,
      )
    }
    const ownershipMae = Number(fieldSummary.player_ownership_mae_pct)
    if (Number.isFinite(ownershipMae)) {
      items.push(`Modeled-vs-actual player ownership MAE was ${ownershipMae.toFixed(2)} percentage points.`)
    }
    return items
  }, [
    serverDecisionGuidance,
    fieldSummary.player_ownership_mae_pct,
    regretSummary.selection_regret_roi,
    replayResolution.slot_resolution_rate,
    userReplaySummary.avg_sim_roi,
    userReplaySummary.best_sim_roi,
  ])

  async function handleLoadContests() {
    const normalizedGameDate = normalizeGameDateInput(gameDate)
    setGameDate(normalizedGameDate)
    if (!normalizedGameDate || !userPattern.trim()) {
      setError('Date and user pattern are required to load contests.')
      return
    }
    if (!ISO_DATE_RE.test(normalizedGameDate)) {
      setError('Date must be in YYYY-MM-DD format.')
      return
    }
    setLoadingContests(true)
    setError(null)
    setSuccess(null)
    try {
      const result = await listFlashbackContests(normalizedGameDate, userPattern.trim())
      setContests(result)
      if (result.length > 0) {
        setSelectedContestId(result[0].contest_id)
      } else {
        setSelectedContestId('manual')
      }
      setSuccess(`Loaded ${result.length} contest candidates.`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load flashback contests.')
    } finally {
      setLoadingContests(false)
    }
  }

  async function handleRunReplay() {
    const normalizedGameDate = normalizeGameDateInput(gameDate)
    setGameDate(normalizedGameDate)
    if (!contestId.trim() || !normalizedGameDate.trim() || !userPattern.trim()) {
      setError('Date, contest id, and user pattern are required.')
      return
    }
    if (!ISO_DATE_RE.test(normalizedGameDate)) {
      setError('Date must be in YYYY-MM-DD format.')
      return
    }
    setRunningReplay(true)
    setError(null)
    setSuccess(null)
    try {
      const response = await runFlashback({
        game_date: normalizedGameDate,
        contest_id: contestId.trim(),
        user_pattern: userPattern.trim(),
        draft_group_id: draftGroupId.trim() ? Number(draftGroupId) : undefined,
        entry_fee: entryFee.trim() ? Number(entryFee) : undefined,
        archetype: 'medium',
        worlds_source: 'gtv2',
        ownership_mode: 'field_only',
        modeled_field_version: 'v1_calibrated',
        include_modeled_field: true,
      })
      setFlashback(response)
      setSuccess(response.loaded_from_cache ? 'Loaded existing flashback artifacts.' : 'Flashback replay completed.')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Flashback replay failed.')
    } finally {
      setRunningReplay(false)
    }
  }

  async function handleRunCalibration() {
    setRunningCalibration(true)
    setError(null)
    setSuccess(null)
    try {
      const response = await runFlashbackCalibration()
      setCalibration(response)
      setSuccess('Replay calibration artifacts updated.')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Calibration build failed.')
    } finally {
      setRunningCalibration(false)
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-4 px-4 py-4">
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <Badge variant="outline">Flashback</Badge>
          <Badge variant="muted">Post-contest replay</Badge>
          <Badge variant="muted">Replay analytics</Badge>
        </div>
        <h1 className="text-2xl font-semibold tracking-tight">Contest Flashback</h1>
        <p className="max-w-3xl text-sm text-[hsl(var(--muted-foreground))]">
          Run exact post-contest replay against the observed field, preview the replay analytics tables,
          and refresh the global calibration artifacts that feed upstream model tuning.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Replay controls</CardTitle>
          <CardDescription>Select one of your contests for a slate or enter the contest id manually.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
          <div className="space-y-2">
            <label className="text-xs font-medium uppercase tracking-[0.12em] text-[hsl(var(--muted-foreground))]">Date</label>
            <Input
              value={gameDate}
              onChange={(e) => setGameDate(normalizeGameDateInput(e.target.value))}
              placeholder="YYYY-MM-DD"
            />
          </div>
          <div className="space-y-2">
            <label className="text-xs font-medium uppercase tracking-[0.12em] text-[hsl(var(--muted-foreground))]">User pattern</label>
            <Input placeholder="EntryName match" value={userPattern} onChange={(e) => setUserPattern(e.target.value)} />
          </div>
          <div className="space-y-2 xl:col-span-2">
            <label className="text-xs font-medium uppercase tracking-[0.12em] text-[hsl(var(--muted-foreground))]">Contest</label>
            <Select value={selectedContestId} onValueChange={setSelectedContestId}>
              <SelectTrigger>
                <SelectValue placeholder="Load contests first" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="manual">Manual contest id</SelectItem>
                {contests.map((contest) => (
                  <SelectItem key={contest.contest_id} value={contest.contest_id}>
                    {contest.contest_name} ({contest.entry_count} entries)
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-end">
            <Button variant="secondary" className="w-full" onClick={handleLoadContests} disabled={loadingContests}>
              {loadingContests ? 'Loading…' : 'Load contests'}
            </Button>
          </div>

          <div className="space-y-2">
            <label className="text-xs font-medium uppercase tracking-[0.12em] text-[hsl(var(--muted-foreground))]">Contest id</label>
            <Input value={contestId} onChange={(e) => setContestId(e.target.value)} placeholder="188576982" />
          </div>
          <div className="space-y-2">
            <label className="text-xs font-medium uppercase tracking-[0.12em] text-[hsl(var(--muted-foreground))]">Draft group</label>
            <Input value={draftGroupId} onChange={(e) => setDraftGroupId(e.target.value)} placeholder="143344" />
          </div>
          <div className="space-y-2">
            <label className="text-xs font-medium uppercase tracking-[0.12em] text-[hsl(var(--muted-foreground))]">Entry fee</label>
            <Input value={entryFee} onChange={(e) => setEntryFee(e.target.value)} placeholder="3" />
          </div>
          <div className="flex items-end gap-2 xl:col-span-2">
            <Button className="flex-1" onClick={handleRunReplay} disabled={runningReplay}>
              {runningReplay ? 'Running flashback…' : 'Run flashback'}
            </Button>
            <Button variant="outline" className="flex-1" onClick={handleRunCalibration} disabled={runningCalibration}>
              {runningCalibration ? 'Building calibration…' : 'Run calibration'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {error ? (
        <Card className="border-[hsl(var(--destructive))]">
          <CardContent className="pt-3 text-sm text-[hsl(var(--destructive-foreground))]">{error}</CardContent>
        </Card>
      ) : null}

      {success ? (
        <Card>
          <CardContent className="pt-3 text-sm text-[hsl(var(--foreground))]">{success}</CardContent>
        </Card>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-2">
        <EnteredLineupsTable rows={flashback?.previews.entered_lineups ?? []} />
        <MetricCards title="Replay takeaways" values={userReplaySummary} />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <InsightList title="How to read this replay" items={replayInsights} />
        <MetricCards title="Match quality" values={replayResolution} />
        <MetricCards title="Regret summary" values={regretSummary} />
        <MetricCards title="Attribution summary" values={attributionSummary} />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <MetricCards title="Field summary" values={fieldSummary} />
        <Card>
          <CardHeader>
            <CardTitle>Replay summary</CardTitle>
            <CardDescription>Artifact counts and run metadata for the selected contest replay.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <SummaryGrid summary={replayCounts} />
            <SummaryGrid summary={replayArtifacts} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Calibration summary</CardTitle>
            <CardDescription>Global replay-calibration aggregate over all available flashback analytics runs.</CardDescription>
          </CardHeader>
          <CardContent>
            <SummaryGrid summary={calibrationArtifactCounts} />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <PreviewTable title="Unresolved player-name matches" rows={unresolvedExamples} />
        <PreviewTable title="Ambiguous player-name matches" rows={ambiguousExamples} />
        <PreviewTable title="Fuzzy player-name matches" rows={fuzzyExamples} />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <PreviewTable title="Lineup calibration preview" rows={flashback?.previews.lineup_calibration ?? []} />
        <PreviewTable title="Player calibration preview" rows={flashback?.previews.player_calibration ?? []} />
        <PreviewTable title="Field calibration preview" rows={flashback?.previews.field_calibration ?? []} />
        <PreviewTable title="Regret summary preview" rows={flashback?.previews.regret_summary ?? []} />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <PreviewTable title="Ownership recalibration preview" rows={calibration?.previews.ownership_recalibration ?? []} />
        <PreviewTable title="Field-model calibration preview" rows={calibration?.previews.field_model_calibration ?? []} />
        <PreviewTable title="Player FPTS calibration preview" rows={calibration?.previews.player_fpts_calibration ?? []} />
        <PreviewTable title="Optimizer regret-by-bucket preview" rows={calibration?.previews.optimizer_regret_by_bucket ?? []} />
      </div>
    </div>
  )
}
