import { LateSwapSession } from '../../api/late_swap'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select'
import { SiteCode } from '../../api/entry_manager'

interface LateSwapHeaderProps {
    date: string
    onDateChange: (value: string) => void
    site: SiteCode
    onSiteChange: (value: SiteCode) => void
    selectedContestCount: number
    selectedEntryCount: number
    session: LateSwapSession | null
}

export function LateSwapHeader({
    date,
    onDateChange,
    site,
    onSiteChange,
    selectedContestCount,
    selectedEntryCount,
    session,
}: LateSwapHeaderProps) {
    return (
        <Card className="late-swap-header">
            <CardHeader className="late-swap-header-head">
                <CardTitle>Late Swap Workbench</CardTitle>
                <CardDescription>
                    Sessionized grouped late swap with preview diagnostics and explicit commit.
                </CardDescription>
            </CardHeader>
            <CardContent className="late-swap-header-meta">
                <label className="late-swap-label">
                    <span>Date</span>
                    <Input
                        type="date"
                        value={date}
                        onChange={(event) => onDateChange(event.target.value)}
                        className="w-[170px]"
                    />
                </label>
                <label className="late-swap-label">
                    <span>Site</span>
                    <Select value={site} onValueChange={v => onSiteChange(v === 'fd' ? 'fd' : 'dk')}>
                        <SelectTrigger className="w-[110px]">
                            <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="dk">DraftKings</SelectItem>
                            <SelectItem value="fd">FanDuel</SelectItem>
                        </SelectContent>
                    </Select>
                </label>
                <Badge variant="outline">Contests: {selectedContestCount}</Badge>
                <Badge variant="outline">Entries: {selectedEntryCount}</Badge>
                {session && (
                    <Badge variant={session.status === 'failed' ? 'default' : 'secondary'}>
                        Session: {session.status}
                    </Badge>
                )}
            </CardContent>
        </Card>
    )
}
