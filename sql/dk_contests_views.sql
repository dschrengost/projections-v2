-- Metric views for duplication, ownership, archetypes, and contest KPIs

-- Number of entries per contest from final standings
create or replace view v_contest_entry_stats as
select contest_id, max(rank) as n_entries
from entries
group by contest_id;

-- Exploded lineup ownership join and aggregated lineup metrics
create or replace view v_lineup_agg as
with lp as (
  select
    l.contest_id,
    l.entry_id,
    lower(trim(l.player_name)) as player_name_norm,
    l.player_name,
    l.slot
  from lineup_players l
),
po as (
  select
    contest_id,
    lower(trim(player_name)) as player_name_norm,
    pct_drafted,
    fpts
  from player_ownership
)
select
  e.contest_id,
  e.entry_id,
  e.rank,
  e.points,
  e.entry_name,
  e.user_handle,
  e.lineup_raw,
  -- sum of ownership (in percentage points)
  sum(coalesce(po.pct_drafted, 0)) as ownership_sum,
  -- uniqueness index: higher is more unique
  sum(-ln((coalesce(po.pct_drafted,0) / 100.0) + 1e-9)) as uniqueness_index,
  -- signature for duplication detection
  string_agg(lp.player_name, '|' order by lp.player_name) as signature
from entries e
join lp on (lp.contest_id = e.contest_id and lp.entry_id = e.entry_id)
left join po on (po.contest_id = e.contest_id and po.player_name_norm = lp.player_name_norm)
group by 1,2,3,4,5,6,7;

-- Add duplication counts
create or replace view v_lineup_agg_with_dupes as
select
  v.*,
  count(*) over (partition by v.contest_id, v.signature) as dup_count
from v_lineup_agg v;

-- Chalk profile: counts of chalky and contrarian plays per lineup
create or replace view v_lineup_chalk_profile as
with lp as (
  select contest_id, entry_id, lower(trim(player_name)) as player_name_norm
  from lineup_players
),
po as (
  select contest_id, lower(trim(player_name)) as player_name_norm, pct_drafted
  from player_ownership
)
select l.contest_id, l.entry_id,
       sum(case when coalesce(po.pct_drafted, 0) >= 30 then 1 else 0 end) as n_gt_30,
       sum(case when coalesce(po.pct_drafted, 0) >= 20 and coalesce(po.pct_drafted, 0) < 30 then 1 else 0 end) as n_20_30,
       sum(case when coalesce(po.pct_drafted, 0) < 5 then 1 else 0 end) as n_lt_5
from lp l
left join po on po.contest_id = l.contest_id and po.player_name_norm = l.player_name_norm
group by l.contest_id, l.entry_id;

-- Top 0.1% cut per contest
create or replace view v_top_0p1_lineups as
select v.*, s.n_entries
from v_lineup_agg_with_dupes v
join v_contest_entry_stats s using (contest_id)
where v.rank <= greatest(1, floor(s.n_entries / 1000.0));

-- Duplication stats per contest
create or replace view v_dup_stats_per_contest as
with d as (
  select contest_id,
         sum(case when dup_count > 1 then 1 else 0 end)::double as duped,
         count(*)::double as total
  from v_lineup_agg_with_dupes
  group by contest_id
)
select contest_id,
       duped as duped_lineups,
       total as total_lineups,
       case when total>0 then duped/total else 0 end as duped_ratio
from d;

-- Winners (rank = 1) with metrics
create or replace view v_winners as
select * from v_lineup_agg_with_dupes where rank = 1;

-- Ownership sum and dup density by contest
create or replace view v_ownership_sum_bins as
with binned as (
  select contest_id,
         cast(floor(coalesce(ownership_sum, 0) / 20.0) as integer) as bin,
         ownership_sum,
         dup_count
  from v_lineup_agg_with_dupes
)
select contest_id,
       bin,
       avg(ownership_sum) as avg_ownership_sum,
       avg(dup_count) as avg_dup_count,
       count(*) as n
from binned
group by contest_id, bin
order by contest_id, bin;

-- Contest KPIs from contest file
create or replace view v_contest_kpis as
select
  contest_id,
  contest_name,
  start_time_readable,
  prize_pool,
  first_place_prize,
  entry_fee,
  max_entries,
  current_entries,
  case when max_entries is not null and max_entries>0 then 1 - (current_entries::double / max_entries) else null end as overlay_pct,
  case when entry_fee is not null and max_entries is not null then prize_pool / (entry_fee * max_entries) else null end as payout_efficiency,
  is_guaranteed,
  template_id,
  draft_group_id
from contests;

-- Optional salary enrichment (when contest_players is available)
create or replace view v_lineup_salary as
with cps as (
  select contest_id, lower(trim(player_name)) as player_name_norm, salary
  from contest_players
)
select l.contest_id, l.entry_id,
       sum(cps.salary) as salary_sum,
       avg(cps.salary) as salary_avg
from lineup_players l
left join cps on cps.contest_id = l.contest_id and cps.player_name_norm = lower(trim(l.player_name))
group by l.contest_id, l.entry_id;

create or replace view v_lineup_agg_full as
select a.*, s.salary_sum, s.salary_avg
from v_lineup_agg_with_dupes a
left join v_lineup_salary s using (contest_id, entry_id);
