import { apiUrl } from './client'

export interface EntryFileSummary {
    contest_id: string
    contest_name: string
    entry_count: number
    created_at: string
    updated_at: string
}

export interface EntryFileState {
    game_date: string
    draft_group_id: number
    site: string
    contest_id: string
    contest_name: string
    entry_fee: string
    created_at: string
    updated_at: string
    client_revision: number
    header: string[]
    entries: Record<string, string>[]
}

export interface PlayerSwap {
    slot: string
    old_player: string
    new_player: string
    old_proj: number | null
    new_proj: number | null
}

export interface LineupAlternative {
    lineup_idx: number
    projected_score: number
    slot_values: Record<string, string>
    player_swaps: PlayerSwap[]
}

export interface EntryAlternatives {
    entry_id: string
    locked_slots: string[]
    alternatives: LineupAlternative[]
    selected_idx: number
}

export interface LateSwapResult {
    entry_state: EntryFileState
    locked_count: number
    updated_entries: number
    missing_locked_ids: string[]
    locked_slots_by_entry_id: Record<string, string[]>
    alternatives_by_entry_id: Record<string, EntryAlternatives>
    selection_summary?: {
        entries_total: number
        entries_swapped: number
        entries_held: number
        entries_unmapped: number
        entries_unknown: number
    }
    solver_summary?: {
        status_counts: Record<string, number>
        avg_gap?: number | null
        max_gap?: number | null
    }
}

export async function uploadEntries(
    date: string,
    draftGroupId: number,
    file: File,
): Promise<EntryFileSummary[]> {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(
        apiUrl(`/api/entry-manager/entries/upload?date=${date}&draft_group_id=${draftGroupId}`),
        {
            method: 'POST',
            body: form,
        },
    )
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Failed to upload entries: ${res.status}`)
    }
    return res.json()
}

export async function listEntryFiles(date: string): Promise<EntryFileSummary[]> {
    const res = await fetch(apiUrl(`/api/entry-manager/entries?date=${date}`))
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Failed to load entries: ${res.status}`)
    }
    return res.json()
}

export async function getEntryFile(date: string, contestId: string): Promise<EntryFileState> {
    const res = await fetch(
        apiUrl(`/api/entry-manager/entries/${contestId}?date=${date}`),
    )
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Failed to load entry file: ${res.status}`)
    }
    return res.json()
}

export async function applyBuildToEntries(
    date: string,
    contestId: string,
    buildSource: 'optimizer' | 'contest-sim',
    buildId: string,
    lineups?: string[][],
): Promise<EntryFileState> {
    const res = await fetch(
        apiUrl(`/api/entry-manager/entries/${contestId}/apply-build?date=${date}`),
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                build_source: buildSource,
                build_id: buildId,
                lineups: lineups ?? null,
            }),
        },
    )
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Failed to apply build: ${res.status}`)
    }
    return res.json()
}

export async function exportEntryFile(date: string, contestId: string): Promise<Blob> {
    const res = await fetch(
        apiUrl(`/api/entry-manager/entries/${contestId}/export?date=${date}`),
        { method: 'POST' },
    )
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Failed to export entries: ${res.status}`)
    }
    return res.blob()
}

export async function exportEntriesBatch(date: string, contestIds: string[]): Promise<Blob> {
    const res = await fetch(apiUrl(`/api/entry-manager/entries/export?date=${date}`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ contest_ids: contestIds }),
    })
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Failed to export entries: ${res.status}`)
    }
    return res.blob()
}

export async function runLateSwap(
    date: string,
    contestId: string,
    options?: {
        useUserOverrides?: boolean
        ownershipMode?: string
        runId?: string | null
        nAlternatives?: number
        randomnessPct?: number
    },
): Promise<LateSwapResult> {
    const res = await fetch(
        apiUrl(`/api/entry-manager/entries/${contestId}/late-swap?date=${date}`),
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                use_user_overrides: options?.useUserOverrides ?? true,
                ownership_mode: options?.ownershipMode ?? 'renormalize',
                run_id: options?.runId ?? null,
                n_alternatives: options?.nAlternatives ?? 5,
                randomness_pct: options?.randomnessPct ?? null,
            }),
        },
    )
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Failed to run late swap: ${res.status}`)
    }
    return res.json()
}

export async function selectLateSwapAlternative(
    date: string,
    contestId: string,
    entryId: string,
    alternativeIdx: number,
    slotValues: Record<string, string>,
): Promise<EntryFileState> {
    const res = await fetch(
        apiUrl(`/api/entry-manager/entries/${contestId}/select-alternative?date=${date}`),
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                entry_id: entryId,
                alternative_idx: alternativeIdx,
                slot_values: slotValues,
            }),
        },
    )
    if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Failed to select alternative: ${res.status}`)
    }
    return res.json()
}
