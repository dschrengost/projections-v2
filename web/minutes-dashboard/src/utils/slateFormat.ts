import type { Slate, GameInfo } from '../api/optimizer'

/**
 * Format a slate for display in dropdowns and headers.
 *
 * Examples:
 *   "Main (11g) 7:00 PM - BOS/NYK, MIA/CLE..."
 *   "Showdown (1g) 10:00 PM - LAL/GSW"
 *   "Turbo (3g) 9:30 PM - Late games"
 */
export function formatSlateLabel(slate: Slate): string {
    const parts: string[] = []

    // Slate type (capitalize)
    const slateType = slate.slate_type.charAt(0).toUpperCase() + slate.slate_type.slice(1)
    parts.push(slateType)

    // Game count
    const gameCount = slate.games?.length ?? 0
    if (gameCount > 0) {
        parts.push(`(${gameCount}g)`)
    } else if (slate.n_contests && slate.n_contests > 0) {
        parts.push(`(${slate.n_contests} contests)`)
    }

    // Start time
    const startTime = formatStartTime(slate.earliest_start)
    if (startTime) {
        parts.push(startTime)
    }

    // Matchups preview
    const matchups = formatMatchupsPreview(slate.games, 3)
    if (matchups) {
        parts.push(`- ${matchups}`)
    }

    return parts.join(' ')
}

/**
 * Format a shorter slate label for tight spaces.
 *
 * Examples:
 *   "Main 11g 7PM"
 *   "Showdown 1g 10PM"
 */
export function formatSlateShort(slate: Slate): string {
    const slateType = slate.slate_type.charAt(0).toUpperCase() + slate.slate_type.slice(1)
    const gameCount = slate.games?.length ?? 0
    const startTime = formatStartTimeShort(slate.earliest_start)

    const parts = [slateType]
    if (gameCount > 0) parts.push(`${gameCount}g`)
    if (startTime) parts.push(startTime)

    return parts.join(' ')
}

/**
 * Format a start time like "7:00 PM ET"
 */
function formatStartTime(isoString: string | undefined): string | null {
    if (!isoString) return null
    try {
        const date = new Date(isoString)
        if (isNaN(date.getTime())) return null
        return date.toLocaleTimeString('en-US', {
            hour: 'numeric',
            minute: '2-digit',
            hour12: true,
            timeZone: 'America/New_York',
        }) + ' ET'
    } catch {
        return null
    }
}

/**
 * Format a short start time like "7PM"
 */
function formatStartTimeShort(isoString: string | undefined): string | null {
    if (!isoString) return null
    try {
        const date = new Date(isoString)
        if (isNaN(date.getTime())) return null
        const hours = date.toLocaleTimeString('en-US', {
            hour: 'numeric',
            hour12: true,
            timeZone: 'America/New_York',
        })
        return hours.replace(' ', '')  // "7 PM" -> "7PM"
    } catch {
        return null
    }
}

/**
 * Format matchups preview like "BOS/NYK, MIA/CLE, LAL/GSW..."
 */
function formatMatchupsPreview(games: GameInfo[] | undefined, maxGames: number): string | null {
    if (!games || games.length === 0) return null

    const matchups = games
        .slice(0, maxGames)
        .map(g => g.matchup)
        .filter(Boolean)

    if (matchups.length === 0) return null

    let result = matchups.join(', ')
    if (games.length > maxGames) {
        result += '...'
    }
    return result
}

/**
 * Get a unique identifier string for a slate (for display alongside name)
 */
export function formatSlateId(slate: Slate): string {
    return `DG${slate.draft_group_id}`
}
