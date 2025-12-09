export type PlayerRow = {
    game_date?: string
    tip_ts?: string
    game_id?: number | string
    player_id?: number | string
    player_name?: string
    team_id?: number | string
    status?: string
    team_name?: string
    team_tricode?: string
    opponent_team_id?: number | string
    opponent_team_name?: string
    opponent_team_tricode?: string
    starter_flag?: boolean
    is_projected_starter?: boolean
    is_confirmed_starter?: boolean
    play_prob?: number
    minutes_p10?: number
    minutes_p50?: number
    minutes_p90?: number
    minutes_p10_cond?: number
    minutes_p90_cond?: number
    proj_fpts?: number
    fpts_per_min_pred?: number
    scoring_system?: string
    sim_dk_fpts_mean?: number
    sim_dk_fpts_std?: number
    sim_dk_fpts_p05?: number
    sim_dk_fpts_p10?: number
    sim_dk_fpts_p25?: number
    sim_dk_fpts_p50?: number
    sim_dk_fpts_p75?: number
    sim_dk_fpts_p90?: number
    sim_dk_fpts_p95?: number
    sim_pts_mean?: number
    sim_reb_mean?: number
    sim_ast_mean?: number
    sim_stl_mean?: number
    sim_blk_mean?: number
    sim_tov_mean?: number
    sim_minutes_sim_mean?: number
    // Vegas lines (game-level, same for all players in game)
    total?: number
    spread_home?: number
    team_implied_total?: number
    opponent_implied_total?: number
    // Ownership projections
    salary?: number
    pred_own_pct?: number
    value?: number
}

export type MinutesResponse = {
    date: string
    count: number
    players: PlayerRow[]
}
