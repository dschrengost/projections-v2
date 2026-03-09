import { useEffect, useMemo, useState } from 'react'
import { listEntryFiles, EntryFileSummary } from '../api/entry_manager'
import {
    LateSwapCandidate,
    LateSwapPolicy,
    LateSwapSession,
    commitLateSwapSession,
    createLateSwapSession,
    defaultLateSwapPolicy,
    exportLateSwapSession,
    getLateSwapSession,
    listLateSwapSessions,
    pinLateSwapCandidates,
    previewLateSwapSession,
    updateLateSwapPolicy,
} from '../api/late_swap'
import { useSlateDate } from '../hooks/useSlateDate'
import { DiagnosticsPanel } from '../components/lateSwap/DiagnosticsPanel'
import { EntryDetailDrawer } from '../components/lateSwap/EntryDetailDrawer'
import { EntryReviewTable } from '../components/lateSwap/EntryReviewTable'
import { ExposureDashboard } from '../components/lateSwap/ExposureDashboard'
import { LateSwapHeader } from '../components/lateSwap/LateSwapHeader'
import { LateSwapPolicyPanel } from '../components/lateSwap/LateSwapPolicyPanel'
import { SwapSummaryPanel } from '../components/lateSwap/SwapSummaryPanel'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select'
import './LateSwapPage.css'

function downloadBlob(blob: Blob, filename: string) {
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = filename
    anchor.click()
    URL.revokeObjectURL(url)
}

export default function LateSwapPage() {
    const [selectedDate, setSelectedDate] = useSlateDate()
    const [entryFiles, setEntryFiles] = useState<EntryFileSummary[]>([])
    const [sessions, setSessions] = useState<LateSwapSession[]>([])
    const [session, setSession] = useState<LateSwapSession | null>(null)
    const [policy, setPolicy] = useState<LateSwapPolicy>(defaultLateSwapPolicy())
    const [selectedContestIds, setSelectedContestIds] = useState<Set<string>>(new Set())
    const [candidatesByEntryId, setCandidatesByEntryId] = useState<Record<string, LateSwapCandidate[]>>({})
    const [selectedByEntryId, setSelectedByEntryId] = useState<Record<string, string>>({})
    const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const refreshEntryAndSessions = async () => {
        const [entries, sessionRows] = await Promise.all([
            listEntryFiles(selectedDate),
            listLateSwapSessions(selectedDate),
        ])
        setEntryFiles(entries)
        setSessions(sessionRows)
    }

    useEffect(() => {
        const load = async () => {
            setError(null)
            try {
                await refreshEntryAndSessions()
            } catch (err) {
                setError((err as Error).message)
            }
        }
        void load()
    }, [selectedDate])

    const contestOptions = useMemo(
        () =>
            entryFiles.map((entry) => ({
                contestId: entry.contest_id,
                contestName: entry.contest_name || entry.contest_id,
                entryCount: entry.entry_count,
            })),
        [entryFiles],
    )

    const selectedEntryCount = useMemo(
        () =>
            Array.from(selectedContestIds).reduce((sum, contestId) => {
                const item = entryFiles.find((entry) => entry.contest_id === contestId)
                return sum + (item?.entry_count ?? 0)
            }, 0),
        [entryFiles, selectedContestIds],
    )

    const handlePolicyChange = (next: LateSwapPolicy) => {
        if (next.mode !== policy.mode) {
            const modeDefaults = defaultLateSwapPolicy(next.mode)
            setPolicy({
                ...modeDefaults,
                ...next,
                mode: next.mode,
            })
            return
        }
        setPolicy(next)
    }

    const handleToggleContest = (contestId: string) => {
        setSelectedContestIds((prev) => {
            const next = new Set(prev)
            if (next.has(contestId)) {
                next.delete(contestId)
            } else {
                next.add(contestId)
            }
            return next
        })
    }

    const loadSession = async (sessionId: string, date?: string) => {
        setLoading(true)
        setError(null)
        try {
            const preview = await getLateSwapSession(sessionId, date ?? selectedDate)
            setSession(preview.session)
            setPolicy(preview.session.policy)
            setCandidatesByEntryId(preview.candidates_by_entry_id)
            setSelectedByEntryId(preview.selected_candidates_by_entry_id)
            const firstEntry = Object.keys(preview.candidates_by_entry_id)[0] ?? null
            setSelectedEntryId(firstEntry)
        } catch (err) {
            setError((err as Error).message)
        } finally {
            setLoading(false)
        }
    }

    const handleCreateSession = async () => {
        const contestIds = Array.from(selectedContestIds)
        if (contestIds.length === 0) return
        setLoading(true)
        setError(null)
        try {
            const created = await createLateSwapSession(selectedDate, contestIds, policy)
            setSession(created)
            await refreshEntryAndSessions()
            const preview = await previewLateSwapSession(created.session_id, {}, created.game_date)
            setSession(preview.session)
            setCandidatesByEntryId(preview.candidates_by_entry_id)
            setSelectedByEntryId(preview.selected_candidates_by_entry_id)
            setSelectedEntryId(Object.keys(preview.candidates_by_entry_id)[0] ?? null)
        } catch (err) {
            setError((err as Error).message)
        } finally {
            setLoading(false)
        }
    }

    const handleApplyPolicy = async () => {
        if (!session) return
        setLoading(true)
        setError(null)
        try {
            const updated = await updateLateSwapPolicy(session.session_id, policy, session.game_date)
            setSession(updated)
            await refreshEntryAndSessions()
        } catch (err) {
            setError((err as Error).message)
        } finally {
            setLoading(false)
        }
    }

    const handlePreview = async () => {
        if (!session) return
        setLoading(true)
        setError(null)
        try {
            const preview = await previewLateSwapSession(session.session_id, {}, session.game_date)
            setSession(preview.session)
            setCandidatesByEntryId(preview.candidates_by_entry_id)
            setSelectedByEntryId(preview.selected_candidates_by_entry_id)
            if (!selectedEntryId) {
                setSelectedEntryId(Object.keys(preview.candidates_by_entry_id)[0] ?? null)
            }
            await refreshEntryAndSessions()
        } catch (err) {
            setError((err as Error).message)
        } finally {
            setLoading(false)
        }
    }

    const handlePinCandidate = async (entryId: string, candidateId: string) => {
        if (!session) return
        setLoading(true)
        setError(null)
        try {
            const preview = await pinLateSwapCandidates(
                session.session_id,
                { [entryId]: candidateId },
                session.game_date,
            )
            setSession(preview.session)
            setCandidatesByEntryId(preview.candidates_by_entry_id)
            setSelectedByEntryId(preview.selected_candidates_by_entry_id)
        } catch (err) {
            setError((err as Error).message)
        } finally {
            setLoading(false)
        }
    }

    const handleCommit = async () => {
        if (!session) return
        setLoading(true)
        setError(null)
        try {
            const committed = await commitLateSwapSession(session.session_id, session.game_date)
            setSession(committed)
            await refreshEntryAndSessions()
        } catch (err) {
            setError((err as Error).message)
        } finally {
            setLoading(false)
        }
    }

    const handleExport = async (includePreview: boolean) => {
        if (!session) return
        setLoading(true)
        setError(null)
        try {
            const blob = await exportLateSwapSession(session.session_id, {
                includeUncommittedPreview: includePreview,
                date: session.game_date,
            })
            downloadBlob(
                blob,
                includePreview
                    ? `entries_${session.game_date}_${session.session_id}_preview.csv`
                    : `entries_${session.game_date}_${session.session_id}_committed.csv`,
            )
        } catch (err) {
            setError((err as Error).message)
        } finally {
            setLoading(false)
        }
    }

    const selectedEntryCandidates = selectedEntryId ? (candidatesByEntryId[selectedEntryId] ?? []) : []

    return (
        <div className="late-swap-page">
            <LateSwapHeader
                date={selectedDate}
                onDateChange={setSelectedDate}
                selectedContestCount={selectedContestIds.size}
                selectedEntryCount={selectedEntryCount}
                session={session}
            />

            <div className="late-swap-session-strip">
                <Card className="session-strip-card">
                    <CardContent className="session-strip-card-content">
                        <label className="late-swap-label">
                            <span>Session</span>
                            <Select
                                value={session?.session_id ?? '__none__'}
                                onValueChange={(value) => {
                                    if (value !== '__none__') void loadSession(value, selectedDate)
                                }}
                            >
                                <SelectTrigger className="w-[360px] max-w-full">
                                    <SelectValue placeholder="Select Session" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="__none__">Select Session</SelectItem>
                                    {sessions.map((item) => (
                                        <SelectItem key={item.session_id} value={item.session_id}>
                                            {item.session_id} · {item.status} · {item.contest_ids.length} contests
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </label>
                        {session && (
                            <Badge variant={session.status === 'failed' ? 'default' : 'outline'}>
                                {session.status}
                            </Badge>
                        )}
                    </CardContent>
                </Card>
                {error && <span className="late-swap-error">{error}</span>}
            </div>

            <div className="late-swap-layout">
                <div className="left-rail">
                    <LateSwapPolicyPanel
                        policy={policy}
                        contestOptions={contestOptions}
                        selectedContestIds={selectedContestIds}
                        onToggleContest={handleToggleContest}
                        onPolicyChange={handlePolicyChange}
                        onCreateSession={handleCreateSession}
                        onApplyPolicy={handleApplyPolicy}
                        onPreview={handlePreview}
                        onCommit={handleCommit}
                        onExport={handleExport}
                        disabled={loading}
                    />
                </div>

                <div className="center-pane">
                    <SwapSummaryPanel summary={session?.selection_summary ?? null} />
                    <ExposureDashboard rows={session?.diagnostics?.exposure_states ?? []} />
                    <EntryReviewTable
                        candidatesByEntryId={candidatesByEntryId}
                        selectedByEntryId={selectedByEntryId}
                        pinnedByEntryId={session?.pinned_candidates_by_entry_id ?? {}}
                        selectedEntryId={selectedEntryId}
                        onSelectEntry={setSelectedEntryId}
                    />
                    <DiagnosticsPanel diagnostics={session?.diagnostics} />
                </div>

                <div className="right-pane">
                    <EntryDetailDrawer
                        scopedEntryId={selectedEntryId}
                        candidates={selectedEntryCandidates}
                        selectedCandidateId={selectedEntryId ? selectedByEntryId[selectedEntryId] ?? null : null}
                        pinnedCandidateId={selectedEntryId ? session?.pinned_candidates_by_entry_id?.[selectedEntryId] ?? null : null}
                        onPinCandidate={handlePinCandidate}
                    />
                </div>
            </div>
        </div>
    )
}
