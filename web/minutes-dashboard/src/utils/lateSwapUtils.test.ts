import { computeRealSwaps } from './lateSwapUtils'

const DK_SLOTS = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']

export const runLateSwapUtilsTests = (): void => {
    const baseline = {
        PG: 'Alpha (101)',
        SG: 'Beta (102)',
        SF: 'Gamma (103)',
        PF: 'Delta (104)',
        C: 'Epsilon (105)',
        G: 'Zeta (106)',
        F: 'Eta (107)',
        UTIL: 'Theta (108)',
    }

    const shuffled = {
        PG: 'Beta (102)',
        SG: 'Alpha (101)',
        SF: 'Gamma (103)',
        PF: 'Delta (104)',
        C: 'Epsilon (105)',
        G: 'Zeta (106)',
        F: 'Eta (107)',
        UTIL: 'Theta (108)',
    }

    const swapped = {
        PG: 'Alpha (101)',
        SG: 'Beta (102)',
        SF: 'Gamma (103)',
        PF: 'Delta (104)',
        C: 'Epsilon (105)',
        G: 'Zeta (106)',
        F: 'Eta (107)',
        UTIL: 'Iota (109)',
    }

    const shuffleResult = computeRealSwaps(baseline, shuffled, DK_SLOTS)
    if (shuffleResult.outs.length !== 0 || shuffleResult.ins.length !== 0) {
        throw new Error('Expected no real swaps for position-only shuffle')
    }

    const swapResult = computeRealSwaps(baseline, swapped, DK_SLOTS)
    if (swapResult.outs[0] !== 'Theta (108)' || swapResult.ins[0] !== 'Iota (109)') {
        throw new Error('Expected real swaps to detect player changes')
    }
}
