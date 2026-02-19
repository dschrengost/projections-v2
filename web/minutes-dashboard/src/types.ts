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
    // Ops manual overrides (GameView)
    ops_override_applied?: boolean
    ops_depth_role?: string
    minutes_delta?: number
    minutes_delta_applied?: boolean
    minutes_target?: number | null
    minutes_lock?: boolean | null
    minutes_target_eff?: number
    minutes_lock_eff?: boolean
    // Canonical play probability fields (explicit conditioning)
    p_play_raw?: number
    p_play_eff?: number
    // Sim availability / two-regime diagnostics
    play_prob_eff?: number
    rotation_lock?: boolean
    play_prob_policy_reason?: string
    sim_p_available?: number
    sim_p_rotation?: number
    bench_zero_p_zero?: number
    bench_zero_threshold_minutes?: number
    p_in_rotation?: number
    minutes_p10?: number
    minutes_p50?: number
    minutes_p90?: number
    minutes_final?: number
    minutes_p10_cond?: number
    minutes_p90_cond?: number
    minutes_mean_uncond?: number
    minutes_q05_uncond?: number
    minutes_q10_uncond?: number
    minutes_q25_uncond?: number
    minutes_q50_uncond?: number
    minutes_q75_uncond?: number
    minutes_q90_uncond?: number
    minutes_q95_uncond?: number
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
    // Canonical sim FPTS summaries (prefer *_uncond_* for decision metrics)
    fpts_sim_cond_mean?: number
    fpts_sim_cond_std?: number
    fpts_sim_cond_p05?: number
    fpts_sim_cond_p50?: number
    fpts_sim_cond_p90?: number
    fpts_sim_cond_p95?: number
    fpts_sim_uncond_mean?: number
    fpts_sim_uncond_std?: number
    fpts_sim_uncond_p05?: number
    fpts_sim_uncond_p50?: number
    fpts_sim_uncond_p90?: number
    fpts_sim_uncond_p95?: number
    sim_pts_mean?: number
    sim_reb_mean?: number
    sim_ast_mean?: number
    sim_stl_mean?: number
    sim_blk_mean?: number
    sim_tov_mean?: number
    sim_minutes_sim_mean?: number
    sim_minutes_sim_p10?: number
    sim_minutes_sim_p50?: number
    sim_minutes_sim_p90?: number
    sim_minutes_sim_std?: number
    sim_minutes_sim_mean_uncond?: number
    sim_minutes_sim_p10_uncond?: number
    sim_minutes_sim_p50_uncond?: number
    sim_minutes_sim_p90_uncond?: number
    sim_minutes_sim_std_uncond?: number
    // Canonical sim minutes summaries (prefer *_uncond_* for decision metrics)
    minutes_sim_p_active?: number
    minutes_sim_cond_mean?: number
    minutes_sim_cond_std?: number
    minutes_sim_cond_p50?: number
    minutes_sim_uncond_p10?: number
    minutes_sim_uncond_mean?: number
    minutes_sim_uncond_std?: number
    minutes_sim_uncond_p50?: number
    minutes_sim_uncond_p90?: number
    // Vegas lines (game-level, same for all players in game)
    total?: number
    spread_home?: number
    team_implied_total?: number
    opponent_implied_total?: number
    // Ownership projections
    salary?: number
    pred_own_pct?: number
    value?: number
    is_locked?: boolean
}

export type MinutesResponse = {
    date: string
    count: number
    players: PlayerRow[]
}
