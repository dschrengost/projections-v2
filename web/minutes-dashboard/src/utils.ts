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

