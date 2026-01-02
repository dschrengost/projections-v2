import { apiUrl } from './client'

export interface BookLine {
    book: string
    line: number
    over_odds: number | null
    under_odds: number | null
}

export interface PropLine {
    player_id: string
    player_name: string
    team: string
    opponent: string
    prop_type: string
    prediction: number | null
    prediction_std: number | null
    best_over_line: number | null
    best_over_odds: number | null
    best_over_book: string | null
    best_under_line: number | null
    best_under_odds: number | null
    best_under_book: string | null
    over_implied_prob: number | null
    over_true_prob: number | null
    over_ev: number | null
    over_edge: string | null
    under_implied_prob: number | null
    under_true_prob: number | null
    under_ev: number | null
    under_edge: string | null
    all_lines: BookLine[]
}

export interface BestEdge {
    player: string
    prop: string
    side: string
    ev: number
}

export interface PropsSummary {
    date: string
    total_props: number
    players_with_props: number
    props_with_edge: number
    best_edges: BestEdge[]
}

export const getPropsLines = async (
    date: string,
    propType?: string,
    minEdge?: number
): Promise<PropLine[]> => {
    let url = `/api/props/lines?date=${date}`
    if (propType) url += `&prop_type=${propType}`
    if (minEdge !== undefined) url += `&min_edge=${minEdge}`
    const res = await fetch(apiUrl(url))
    if (!res.ok) {
        if (res.status === 404) {
            return []
        }
        throw new Error(`Failed to fetch props: ${res.status}`)
    }
    return res.json()
}

export const getPropsSummary = async (date: string): Promise<PropsSummary> => {
    const res = await fetch(apiUrl(`/api/props/summary?date=${date}`))
    if (!res.ok) {
        throw new Error(`Failed to fetch props summary: ${res.status}`)
    }
    return res.json()
}

// Prop type display names
export const PROP_TYPE_LABELS: Record<string, string> = {
    pts: 'Points',
    reb: 'Rebounds',
    ast: 'Assists',
    threes: '3-Pointers',
    blk: 'Blocks',
    stl: 'Steals',
    turnovers: 'Turnovers',
    ptsrebast: 'PTS+REB+AST',
    ptsreb: 'PTS+REB',
    ptsast: 'PTS+AST',
    rebast: 'REB+AST',
    stlblk: 'STL+BLK',
}

// Book display names
export const BOOK_LABELS: Record<string, string> = {
    draftkings: 'DraftKings',
    fanduel: 'FanDuel',
    mgm: 'BetMGM',
    caesars: 'Caesars',
    betrivers: 'BetRivers',
    hardrock: 'Hard Rock',
    espnbet: 'ESPN Bet',
}
