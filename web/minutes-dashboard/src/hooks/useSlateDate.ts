import { useCallback, useEffect, useState } from 'react'

/**
 * Calculate the current "slate date" - the DFS date we're working on.
 *
 * Uses local time but doesn't flip to the next day until 6AM.
 * This keeps us on the current slate during evening games (7PM-2AM).
 */
export function getSlateDate(): string {
    const now = new Date()
    // Subtract 6 hours - so midnight-6AM still counts as "yesterday's" slate
    const adjusted = new Date(now.getTime() - 6 * 60 * 60 * 1000)
    return adjusted.toISOString().slice(0, 10)
}

/**
 * Get date from URL search params, falling back to slate date.
 */
function getDateFromURL(): string {
    if (typeof window === 'undefined') return getSlateDate()
    const params = new URLSearchParams(window.location.search)
    const urlDate = params.get('date')
    // Validate it looks like a date
    if (urlDate && /^\d{4}-\d{2}-\d{2}$/.test(urlDate)) {
        return urlDate
    }
    return getSlateDate()
}

/**
 * Get slate (draft_group_id) from URL search params.
 */
function getSlateFromURL(): number | null {
    if (typeof window === 'undefined') return null
    const params = new URLSearchParams(window.location.search)
    const urlSlate = params.get('slate')
    if (urlSlate && /^\d+$/.test(urlSlate)) {
        return parseInt(urlSlate, 10)
    }
    return null
}

/**
 * Update URL search params without causing a page reload.
 */
function updateURL(date: string, slate: number | null): void {
    if (typeof window === 'undefined') return
    const url = new URL(window.location.href)
    url.searchParams.set('date', date)
    if (slate !== null) {
        url.searchParams.set('slate', String(slate))
    } else {
        url.searchParams.delete('slate')
    }
    window.history.replaceState({}, '', url.toString())
}

/**
 * Hook for managing the slate date with URL persistence.
 *
 * - Initializes from URL ?date= param, or falls back to slate date
 * - Updates URL when date changes (without page reload)
 * - All pages using this hook share the same date via URL
 *
 * Usage:
 *   const [selectedDate, setSelectedDate] = useSlateDate()
 */
export function useSlateDate(): [string, (date: string) => void] {
    const [selectedDate, setSelectedDateState] = useState(getDateFromURL)

    // Sync from URL on mount and when URL changes (e.g., back/forward)
    useEffect(() => {
        const handlePopState = () => {
            setSelectedDateState(getDateFromURL())
        }
        window.addEventListener('popstate', handlePopState)
        return () => window.removeEventListener('popstate', handlePopState)
    }, [])

    // Wrapper that updates both state and URL
    const setSelectedDate = useCallback((date: string) => {
        setSelectedDateState(date)
        updateURL(date, getSlateFromURL())
    }, [])

    return [selectedDate, setSelectedDate]
}

/**
 * Hook for managing both date and slate with URL persistence.
 *
 * Usage:
 *   const [selectedDate, setSelectedDate, selectedSlate, setSelectedSlate] = useSlateDateAndSlate()
 */
export function useSlateDateAndSlate(): [
    string,
    (date: string) => void,
    number | null,
    (slate: number | null) => void
] {
    const [selectedDate, setSelectedDateState] = useState(getDateFromURL)
    const [selectedSlate, setSelectedSlateState] = useState<number | null>(getSlateFromURL)

    // Sync from URL on mount and when URL changes (e.g., back/forward)
    useEffect(() => {
        const handlePopState = () => {
            setSelectedDateState(getDateFromURL())
            setSelectedSlateState(getSlateFromURL())
        }
        window.addEventListener('popstate', handlePopState)
        return () => window.removeEventListener('popstate', handlePopState)
    }, [])

    // Update date (clears slate since slates are date-specific)
    const setSelectedDate = useCallback((date: string) => {
        setSelectedDateState(date)
        setSelectedSlateState(null)
        updateURL(date, null)
    }, [])

    // Update slate
    const setSelectedSlate = useCallback((slate: number | null) => {
        setSelectedSlateState(slate)
        updateURL(getDateFromURL(), slate)
    }, [])

    return [selectedDate, setSelectedDate, selectedSlate, setSelectedSlate]
}

export default useSlateDate
