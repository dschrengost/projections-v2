import React from 'react'

type GameTab = {
    game_id: string
    label: string
}

type GameTabsProps = {
    tabs: GameTab[]
    activeGameId: string
    onChange: (gameId: string) => void
}

export const GameTabs: React.FC<GameTabsProps> = ({ tabs, activeGameId, onChange }) => {
    return (
        <div className="gv2-tabs" role="tablist" aria-label="Games">
            {tabs.map((tab) => (
                <button
                    key={tab.game_id}
                    type="button"
                    role="tab"
                    aria-selected={tab.game_id === activeGameId}
                    className={`gv2-tab ${tab.game_id === activeGameId ? 'active' : ''}`}
                    onClick={() => onChange(tab.game_id)}
                >
                    {tab.label}
                </button>
            ))}
        </div>
    )
}
