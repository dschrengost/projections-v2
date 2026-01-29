export type LineupSlots = Record<string, string>

export type RealSwapResult = {
    outs: string[]
    ins: string[]
}

export const extractDraftableId = (playerValue: string): number | null => {
    if (!playerValue) return null
    const match = playerValue.match(/\((\d+)\)\s*$/)
    if (!match) return null
    const id = Number(match[1])
    return Number.isFinite(id) ? id : null
}

const buildPlayerMap = (lineup: LineupSlots, slots: string[]): Map<number, string> => {
    const map = new Map<number, string>()
    for (const slot of slots) {
        const value = lineup[slot]
        if (!value) continue
        const id = extractDraftableId(value)
        if (id === null) continue
        if (!map.has(id)) {
            map.set(id, value)
        }
    }
    return map
}

export const computeRealSwaps = (
    baselineLineup: LineupSlots,
    currentLineup: LineupSlots,
    slots: string[],
): RealSwapResult => {
    const baselineMap = buildPlayerMap(baselineLineup, slots)
    const currentMap = buildPlayerMap(currentLineup, slots)

    const outs: string[] = []
    for (const [id, value] of baselineMap.entries()) {
        if (!currentMap.has(id)) {
            outs.push(value)
        }
    }

    const ins: string[] = []
    for (const [id, value] of currentMap.entries()) {
        if (!baselineMap.has(id)) {
            ins.push(value)
        }
    }

    return { outs, ins }
}
