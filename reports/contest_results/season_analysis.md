# Season-long DraftKings NBA Contest Results Analysis (2025-10-25 → 2026-01-17)

This report re-derives conclusions directly from the raw DraftKings contest result CSVs in
`/home/daniel/projections-data/` and treats the existing dashboard “Contest” tab as a *hypothesis only*.

## Executive Summary (Actionable)

1) **Stop “over-contrarian by default” in low-stakes GPPs.** Our lineups (username in raw files: `theangrydingo`) are *more contrarian than the median field* and far more contrarian than winners.
   - Low-stakes field median `own_total` ≈ **214.8**; low-stakes winners median ≈ **251.7**.
   - Our median `own_total` ≈ **193.4** (below field median; ~**−31.9** vs contest top-1% median in 76% of our contests).

2) **Add more correlation (game + team concentration) — we’re materially under-stacked.**
   - Low-stakes winners: **30.3%** have `max_from_game ≥ 5` vs low-stakes field **24.1%**.
   - We: **15.6%** have `max_from_game ≥ 5` (subset where salary/team/game could be joined).
   - Low-stakes winners: **24.2%** have `max_from_team ≥ 4` vs field **18.2%**; we’re **13.2%**.

3) **Spend (almost) all salary; don’t force exactly $50k.**
   - Winners (all contests with salary info): `salary_left` p50=**0**, p95≤**400**, p99≤**700**.
   - Low-stakes winners are slightly looser (p50=**100**, p99≤**800**), but `salary_left > 1000` is **~0.3%** of winners.

4) **Don’t over-index on “unique” — but actively avoid *mass-dup* lineups.**
   - Low-stakes winners: `dupe_count` p95≤**16**, max=**31** (ties included).
   - Our portfolio includes some *very* duped lineups (max `dupe_count`=**159**) even though most of ours are unique.

5) **Low-stakes vs flagship (within-slate): flagship top-1% is chalkier and less duped.**
   - Per-slate paired comparison (largest-field low-stakes contest vs slate flagship): low-stakes top-1% has **~−10.4** lower `own_total` and **+0.10** higher dupe rate (`is_dupe`) than flagship top-1%.

## Data Discovery & Inventory (required)

### Where the raw data lives

Contest results (CSV):
- `/home/daniel/projections-data/bronze/dk_contests/nba_gpp_data/<date>/results/contest_<contest_id>_results.csv`

Contest metadata (CSV):
- `/home/daniel/projections-data/bronze/dk_contests/nba_gpp_data/<date>/nba_gpp_<date>.csv`

Optional “draftables” snapshots (JSON; required for salary/team/game features):
- `/home/daniel/projections-data/bronze/dk/draftables/draftables_raw_<draft_group_id>.json`

### Inventory summary (season-to-date)

From `scripts/contest_results/analyze_contest_results.py` + saved parquet outputs:
- Dates: **71** (`2025-10-25` → `2026-01-17`)
- Contests with result files found: **2,759**
- Zip-like / corrupt results skipped: **34**
- Contests successfully parsed into the tidy dataset: **2,725**
- Low-stakes contests (empirical: entry_fee ≤ $5): **1,298**
- “Flagship” contests (per slate: max prize_pool): **195**

Output artifacts (reusable datasets):
- Inventory: `/home/daniel/projections-data/analytics/contest_results/contest_inventory_with_field_sizes.parquet`
- Tidy entries: `/home/daniel/projections-data/analytics/contest_results/contest_entries_tidy.parquet`
- Per-contest cohort summary: `/home/daniel/projections-data/analytics/contest_results/contest_cohort_summary.parquet`
- User entries (substring match `angrydingo` → `theangrydingo`): `/home/daniel/projections-data/analytics/contest_results/user_entries.parquet`

### Schema variations & unified loader

Do *not* assume column consistency. The loader:
- Normalizes BOM/whitespace in headers.
- Reads only needed columns when available and skips files missing required fields.
- Skips “zip-like” results (files beginning with `PK`).

The “results” CSVs are mostly consistent for non-corrupt files:
- Entry-level fields: `Rank`, `EntryId`, `EntryName`, `Points`, `Lineup`
- Player rows used to build lookups: `Player`, `%Drafted`, `FPTS`

## Cohorts (per contest; no cross-slate leakage)

For each contest (after filtering to valid 8-player lineups):
- **Winner**: `rank = 1` (ties included)
- **Top 0.1%**: `rank ≤ ceil(0.001 * N)`
- **Top 1%**: `rank ≤ ceil(0.01 * N)`
- **Top 5%**: `rank ≤ ceil(0.05 * N)`
- **Min-cash (proxy)**: `rank ≤ ceil(0.20 * N)` (payout structures aren’t available in the CSVs, so this uses top-20% as a defensible approximation)
- **Field median**: the per-contest median of each metric

## Lineup Feature Engineering

All features are computed at the **entry/lineup** level.

### Always available (from the results CSVs)

- **Dupes**: exact-8-player duplicates within the contest via `lineup_key`:
  - `dupe_count`, `is_dupe`
- **Ownership**: derived from `%Drafted`:
  - `own_total`, `own_avg`, `own_min`, `own_max`, `own_gini`
  - `own_num_lt10`, `own_num_lt5`, `own_num_gt50`

### Available when “draftables” join succeeds (partial coverage)

Coverage: **~51%** of the 13.0M entries had salary/team/game successfully attached.

- **Salary usage**:
  - `salary_total`, `salary_left`
  - `salary_gini`, `salary_min`, `salary_max`
- **Team concentration**:
  - `num_teams`, `max_from_team`
- **Game stacking + “bring-back”** (via DK competitionId):
  - `num_games`, `max_from_game`
  - `has_bring_back` (at least 2 players from same game, with both teams represented)

Limitations:
- Salary/team/game depend on name matches between lineup strings and draftables `displayName`.
- `has_bring_back` is a *proxy*; it doesn’t capture nuanced correlation (e.g., multiple games, late swap dynamics).

## Contest Segmentation

### Low-stakes

Defined empirically as **entry_fee ≤ $5**.

### Flagship

Per slate (`date` + `draft_group_id`), define “flagship” as the contest with **maximum prize_pool**
(tie-break by entries). This is designed for **within-slate** comparisons.

### Contest class (do not assume all are GPPs)

Classified from contest names:
- `gpp` (default)
- `cash` (double-ups, 50/50, H2H)
- `satellite` (qualifiers/tickets)
- `multiplier`

In the contests we entered (`theangrydingo`), all are classified `gpp`.

## Core Questions (with evidence)

### A) What lineup features are overrepresented among top finishes in low-stakes?

**Chalk level (ownership): winners are meaningfully chalkier than the field.**
- Low-stakes field median `own_total`: **214.8**
- Low-stakes winners median `own_total`: **251.7**
- Low-stakes winners have fewer ultra-contrarian builds:
  - Low-stakes winners `own_num_lt5` p90=**2** vs low-stakes field p90=**3**

**Correlation: winners are more game- and team-concentrated than the low-stakes field.**
*(Entries with salary/team/game attached only)*
- `max_from_game ≥ 5`: winners **30.3%** vs field **24.1%**
- `max_from_team ≥ 4`: winners **24.2%** vs field **18.2%**
- `max_from_team ≥ 5` is rare in winners (~**1.7%**); 5-man team stacks are not a staple.

**Salary usage: winners rarely “leave a lot.”**
- Low-stakes winners (with salary info): `salary_left` p50=**100**, p95≤**400**, p99≤**800**

**Dupes: winners tolerate duplication, but do not win with massively duplicated lineups.**
- Low-stakes winners: `dupe_count` p95≤**16**, max=**31**
- This supports “don’t force uniqueness,” but **avoid the mass-dup tail**.

### B) How does this differ in flagship contests?

Flagship contests have a **different baseline field**.

**Flagship field is chalkier and more stacked than low-stakes field.**
*(Entries with salary/team/game attached only)*
- Flagship field `max_from_game ≥ 5`: **29.4%** (vs low-stakes field **24.1%**)
- Flagship field `own_total` median: **228.0** (vs low-stakes field **214.8**)

**Flagship winners are even chalkier than low-stakes winners.**
- Flagship winners `own_total` median: **293.5** (low-stakes winners: **251.7**)
- Flagship winners use fewer <5% owned plays:
  - Flagship winners `own_num_lt5` p75=**0**, p90=**1**

**Dupes remain common for winners, but massive duplication still doesn’t win.**
- Flagship winners: `dupe_count` max=**11**

**Within-slate comparison (recommended lens):**
Using paired slates where we have both contests (flagship and a largest-field low-stakes contest):
- Low-stakes top-1% is **less chalky** than flagship top-1%:
  - `own_total` median delta (low − flagship) ≈ **−10.4** (positive in only **25.5%** of slates)
- Low-stakes top-1% is **more duped** than flagship top-1%:
  - `is_dupe` mean delta (low − flagship) ≈ **+0.10** (positive in **74.5%** of slates)

### C) How do our lineups differ from winners and top finishers?

User identification:
- The raw CSVs contain `user_name = theangrydingo`.
- `user_entries.parquet` is built via substring match on `angrydingo`.

Volume:
- Our entries: **2,733** across **87** contests (**85** low-stakes; **0** flagship)
- Best finish per contest: median **6.36%** (best observed **0.079%**, 15th/18922 in a $0.25 contest)

**We are too contrarian vs both winners and the field.**
- Our `own_total` median: **193.4**
- Low-stakes field median: **214.8**
- Low-stakes winners median: **251.7**
- Contest-by-contest vs top-1% median: our median delta ≈ **−31.9**, negative in **75.9%** of contests.

**We overuse ultra-low-owned plays.**
- Our `own_num_lt5` median: **1** (p75=**2**; p90≈**3**)
- Low-stakes winners: p75=**1**; p90=**2**

**We under-stack (game + team) on the subset we can measure.**
*(Only our entries with salary/team/game attached; n=623)*
- `max_from_game ≥ 5`: **15.6%** (low-stakes field **24.1%**, winners **30.3%**)
- `max_from_team ≥ 4`: **13.2%** (low-stakes field **18.2%**, winners **24.2%**)

**We are “too unique” in the wrong way and still occasionally land on mass-dup lineups.**
- Our dupe rate is low (is_dupe ≈ **0.149**) and is below top-1% in **83.9%** of our contests.
- But our worst-case duplication is extreme: `dupe_count` max=**159**, far above winner tails.

### D) Which differences look structural vs slate noise?

The following are **sign-consistent across many contests** (not single-slate anecdotes):
- `own_total` gap (us below top-1%): negative in **~76%** of our contests.
- `is_dupe` gap (us below top-1%): negative in **~84%** of our contests.
- Paired-slate low vs flagship: flagship top-1% is chalkier in **~75%** of slates.

Correlation gaps (stacking) are also large in magnitude, but measured on a smaller subset
due to partial salary/team/game coverage — treat those as “strong evidence but lower coverage.”

### E) Which commonly assumed DFS heuristics do NOT show evidence?

1) **“You must leave lots of salary to win.”** Not supported.
   - Winners almost always leave ≤ **$700**; leaving > **$1000** is ~**0.3%** of winners.

2) **“Never play duped lineups.”** Not supported.
   - Winners are often duped (especially in flagship), just not *massively* duped.

3) **“Bring-back is the edge.”** Not supported as a standalone rule.
   - Bring-back rates are high in the field; differences are small and not directionally consistent.
   - The edge shows up more reliably in **overall correlation / stacking intensity**.

4) **“Low-stakes winners are super contrarian.”** Not supported.
   - Low-stakes winners are *more chalky than the low-stakes median field*, and far chalkier than us.

## Translating to Optimizer Strategy (do not implement yet)

The goal is to turn these empirical findings into *defensible defaults*, without overfitting.
Use **ranges**, and prefer *soft penalties / portfolio mixes* over brittle hard constraints.

### Recommended: Low-stakes default configuration

- **Salary usage**: target `salary_left` **0–400** most of the time.
  - Practical: set `min_salary` around **49,600–49,900** (instead of 49,000) *if* it doesn’t kill diversity.
  - Warning: hard `min_salary=50,000` likely removes viable winner-like lineups (low-stakes p50 leftover is 100).

- **Correlation / stacking**:
  - Ensure meaningful correlation exists in the pool: include many lineups where `max_from_game` is **4–5**.
  - Do **not** hard-require a bring-back; instead, avoid generating large volumes of “8 different games” lineups.

- **Team limit**:
  - Keep `global_team_limit = 4` as the default.
  - Optional exploration: allow `5` only for very small slates, because 5-man team stacks are rare winners overall.

- **Ownership posture**:
  - Reduce “ultra-contrarian” frequency: aim for most lineups to have **0–1** players projected <5% owned.
  - Avoid pushing `own_total` far below the field median; winners are typically above it.
  - Warning: hard ownership constraints can reduce EV; prefer soft penalties and portfolio balancing.

- **Uniqueness**:
  - Do not chase uniqueness for its own sake.
  - Add a *mass-dup avoidance* heuristic: avoid constructions likely to be entered 50–100+ times.

- **Pool sizing**:
  - Generate **20k–40k** candidates, then select a final set with controlled ownership + correlation mix.

### Recommended: Flagship-style configuration

- **Expect a chalkier, more stacked baseline**; don’t try to “out-galaxy-brain” it.
- **Salary usage**: keep `salary_left` mostly **0–500** (winners almost never leave >500).
- **Ownership posture**: be comfortable with higher-chalk combinations; avoid lineups with multiple <5% plays.
- **Uniqueness**: use portfolio-level de-duplication:
  - Consider enabling near-duplicate filtering (`near_dup_jaccard` around **0.60–0.75**) for large-field flagship sets.
  - Warning: too strict can cap upside by eliminating correlated cores.

### Recommended: Exploration / diversification mode

Use this to discover slate-specific winners without turning them into hard rules:
- Mix multiple generation passes:
  - pass A: tighter salary (left 0–300), moderate chalk
  - pass B: slightly looser salary (left 0–700), include 1–2 low-owned pivots
  - pass C: higher correlation variants (more 4–5 game stacks)
- Keep hard constraints minimal; use diversity controls (`min_uniq`, `near_dup_jaccard`, jitter) to spread outcomes.

## Comparison to Existing Dashboard “Contest” Tab

### What it gets right (agrees with this analysis)

- **Ownership totals and “under 5% / under 10%” counts** are directionally consistent with our findings.
- **Duplicate lineup detection** (sorted lineup key) is conceptually aligned with this analysis.
- **Top-1% and cash-line (top 20%) slices** are reasonable first-order cohorts.

### Where it is misleading or incomplete (disagrees / blind spots)

- **Stack and game-correlation sections are not real.**
  - `projections/api/contest_service.py` uses `player_teams = {}` (empty), so team stacks and game correlations are placeholders.
  - This analysis joins DK “draftables” snapshots to compute salary/team/game features.

- **No within-slate segmentation.**
  - Aggregating across slates can hide that flagship contests have a different baseline (chalkier, more stacked).
  - This analysis explicitly compares low-stakes vs flagship *within the same slate*.

- **No salary usage analysis.**
  - The dashboard doesn’t quantify leftover salary distributions or identify “leave too much” lineups.

- **No user-vs-cohort diagnostics.**
  - The key actionable gap for us is “too contrarian + under-correlated,” which the dashboard doesn’t flag.

### Concrete dashboard improvements to implement next

1) Join `draftables_raw_<draft_group_id>.json` to add: salary usage, team stacks, game stacks, bring-back rates.
2) Add segmentation toggles: low-stakes vs flagship, SE vs 150-max, and field-size bands.
3) Add paired-slate views: compare low-stakes vs flagship on the same slate.
4) Add user diagnostics: per contest, show deltas vs winner/top-1% for `own_total`, `own_num_lt5`, `max_from_game`, and a “mass-dup risk” proxy.

## Repro / How to run

Build/refresh the season dataset:
- `uv run python scripts/contest_results/analyze_contest_results.py --write-dataset --user angrydingo --user-match contains`

Inventory-only (safe, fast):
- `uv run python scripts/contest_results/analyze_contest_results.py --inventory-only`

Rebuild user entries from the saved tidy parquet (fast; no reprocessing):
- `uv run python scripts/contest_results/analyze_contest_results.py --rebuild-user-entries --user angrydingo --user-match contains`
