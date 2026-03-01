-- DuckDB schema for DraftKings NBA Classic GPP analytics

create table if not exists contests (
  contest_id text primary key,
  contest_name text,
  start_time text,
  start_time_readable text,
  prize_pool double,
  first_place_prize text,
  entry_fee double,
  max_entries integer,
  current_entries integer,
  game_type text,
  is_guaranteed boolean,
  is_starred boolean,
  template_id text,
  draft_group_id text,
  scrape_time text
);

create table if not exists entries (
  contest_id text,
  entry_id text,
  user_handle text,
  entry_name text,
  rank integer,
  points double,
  winnings double,
  lineup_raw text,
  primary key (contest_id, entry_id)
);

create table if not exists lineup_players (
  contest_id text,
  entry_id text,
  slot text,
  player_name text
);

create table if not exists player_ownership (
  contest_id text,
  player_name text,
  roster_position text,
  pct_drafted double,
  fpts double
);

-- Player pool per contest extracted from results ZIP (contest-players-*.csv)
create table if not exists contest_players (
  contest_id text,
  player_name text,
  roster_position text,
  team text,
  opponent text,
  game_info text,
  salary integer,
  avg_fpts double
);

-- Contest payout curve (per contest, ranges of min_rank..max_rank)
create table if not exists payouts (
  contest_id text,
  min_rank integer,
  max_rank integer,
  amount double,
  payout_type text
);

-- Pre-lock player list by draft group (optional enrichment)
create table if not exists draft_group_players (
  draft_group_id text,
  player_id text,
  player_name text,
  team text,
  positions text,
  salary integer,
  game_id text,
  game_time text
);

-- Ingestion bookkeeping (optional)
create table if not exists ingestion_log (
  kind text,
  key text,
  value text,
  ts timestamp default current_timestamp
);
