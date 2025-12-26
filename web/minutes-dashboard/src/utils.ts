export type StatusBadge = {
    label: string
    title: string
    className: string
}

const STATUS_BADGE_DEFINITIONS: Array<{
    matcher: (value: string) => boolean
    badge: StatusBadge
}> = [
        {
            matcher: (value) => value === 'q' || value === 'questionable',
            badge: {
                label: 'Q',
                title: 'Questionable',
                className: 'status-questionable',
            },
        },
        {
            matcher: (value) => value.startsWith('prob'),
            badge: {
                label: 'Prob',
                title: 'Probable',
                className: 'status-probable',
            },
        },
        {
            matcher: (value) => value.includes('doubt'),
            badge: {
                label: 'D',
                title: 'Doubtful',
                className: 'status-doubtful',
            },
        },
        {
            matcher: (value) => value.includes('gtd') || value.includes('game time'),
            badge: {
                label: 'GTD',
                title: 'Game time decision',
                className: 'status-gtd',
            },
        },
        {
            matcher: (value) => value === 'out',
            badge: {
                label: 'Out',
                title: 'Out',
                className: 'status-out',
            },
        },
    ]

export const getStatusBadge = (status?: string): StatusBadge | null => {
    const normalized = status?.trim().toLowerCase()
    if (!normalized) {
        return null
    }
    for (const definition of STATUS_BADGE_DEFINITIONS) {
        if (definition.matcher(normalized)) {
            return definition.badge
        }
    }
    return null
}

export const formatTime = (ts?: string, dateContext?: string) => {
    if (!ts) return ''
    try {
        let dateStr = ts
        // If ts is just a time (HH:MM:SS or HH:MM), append it to the date context
        if (!ts.includes('T') && !ts.includes(' ') && dateContext) {
            dateStr = `${dateContext}T${ts}`
        }

        // Ensure UTC if no timezone specified
        if (!dateStr.endsWith('Z') && !dateStr.includes('+') && !dateStr.includes('-')) {
            dateStr = `${dateStr}Z`
        }

        const date = new Date(dateStr)
        if (isNaN(date.getTime())) {
            return ts
        }
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', timeZoneName: 'short' })
    } catch (e) {
        return ts
    }
}

export const formatMinutes = (value?: number) =>
    typeof value === 'number' ? value.toFixed(1) : '—'
export const formatPercent = (value?: number) =>
    typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—'
export const formatFpts = (value?: number) =>
    typeof value === 'number' ? value.toFixed(1) : '—'
export const formatStat = (value?: number) =>
    typeof value === 'number' ? value.toFixed(1) : '—'
export const formatMinutesSim = (value?: number) =>
    typeof value === 'number' ? value.toFixed(1) : '—'
export const formatSalary = (value?: number) =>
    typeof value === 'number' ? `$${(value / 1000).toFixed(1)}k` : '—'
export const formatOwnership = (value?: number) =>
    typeof value === 'number' ? `${value.toFixed(1)}%` : '—'
export const formatValue = (value?: number) =>
    typeof value === 'number' ? value.toFixed(2) : '—'

/**
 * Convert run ID (YYYYMMDDTHHMMSSz) to Eastern time string
 * e.g., "20251224T020312Z" -> "9:03 PM EST"
 */
export const formatRunIdToEST = (runId: string): string => {
    if (!runId || runId.length < 15) return ''
    try {
        // Parse: 20251224T020312Z -> 2025-12-24T02:03:12Z
        const year = runId.slice(0, 4)
        const month = runId.slice(4, 6)
        const day = runId.slice(6, 8)
        const hour = runId.slice(9, 11)
        const minute = runId.slice(11, 13)
        const second = runId.slice(13, 15)

        const isoString = `${year}-${month}-${day}T${hour}:${minute}:${second}Z`
        const date = new Date(isoString)

        if (isNaN(date.getTime())) return ''

        return date.toLocaleTimeString('en-US', {
            hour: 'numeric',
            minute: '2-digit',
            timeZone: 'America/New_York',
            timeZoneName: 'short',
        })
    } catch {
        return ''
    }
}

/**
 * Format ISO timestamp to Eastern time
 */
export const formatTimestampToEST = (ts: string): string => {
    if (!ts) return ''
    try {
        const date = new Date(ts)
        if (isNaN(date.getTime())) return ''

        return date.toLocaleTimeString('en-US', {
            hour: 'numeric',
            minute: '2-digit',
            timeZone: 'America/New_York',
            timeZoneName: 'short',
        })
    } catch {
        return ''
    }
}
