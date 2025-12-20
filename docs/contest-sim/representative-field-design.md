# Contest Sim v2: Representative Field Simulation (Option A → Option C)

## TL;DR
Our current contest sim is “self-play” (your lineups compete against each other). This is useful for relative ranking but is not a realistic estimate of ROI/rates in a real DK GPP field.

This doc proposes a fast, practical upgrade:
- **Option A (start here):** build a **weighted field library** from many QuickBuild lineups (e.g. 30–40k generated quickly), then **compress to K ≈ 1k–5k unique lineups with weights** and simulate user lineups vs that weighted library.
- **Option C (plug-in ready):** extend the field library to a **mixture** of generators (optimizer-ish + human-ish + noisy) while keeping the same “weighted library” interface and caching.

We can calibrate the field weights and mixture proportions using our historical DK contest result CSVs in bronze (20–30 slates).

## Goals
- Simulate each lineup vs a **representative opponent field**, not only vs our own lineup set.
- Keep runtime low enough for pre-lock workflows (minutes-scale worst case).
- Prioritize **rate metrics** (win/top 1%/top 5%/cash) and **ROI** over contest-specific EV dollars.
- Make “field modeling” modular so we can iterate without rewriting the sim engine.

## Non-goals (for this phase)
- Perfect contest-specific EV across many buy-ins (we’ll standardize payout archetypes).
- Perfect ownership modeling (we’ll calibrate against historical data and allow ownership to be “helpful but not required”).
- Late-swap contest-aware “real-time contest file” sims (future).

## Current state (what we have)
### Pipeline
- Player outcomes are driven by `sim_v2` worlds:
  - `projections-data/artifacts/sim_v2/worlds_fpts_v2/game_date=YYYY-MM-DD/`
- Contest sim endpoint: `projections/api/contest_sim_api.py` → `projections/contest_sim/contest_sim_service.py`
- Current sim mode: **self-play**
  - `user_scores = lineup_scores`
  - `field_scores = lineup_scores`
  - weights are scaled up to “field size bucket”

### Key limitations
- Self-play cannot answer “how good is lineup A vs the field?”
- It produces unstable EV/ROI if we only simulate a narrow slice of possible opponents.

### Must-fix correctness issues (independent of field model)
Even if we keep the UI “rate-first”, we should fix these to avoid large ROI gaps:
1) **Self-play entrant accounting**: the payout engine treats total entrants as `user + field`. In self-play we currently pass the same set as both, which can effectively double-count entrants.
2) **Rake consistency**: payout tiers use rake from `contest_sim.yaml` defaults, while `ContestConfig` uses a different default. These should match.
3) **Dupe penalty application**: we compute `adjusted_expected_payout` but do not propagate that into ROI sorting/filtering consistently.

We can defer double-up semantics because we don’t play them, but the above affect GPP sims too.

## Core concept: a Weighted Field Library
All future approaches should produce the same shape:

```text
FieldLibrary {
  lineups: List[List[player_id]]   # K unique lineups
  weights: List[int]               # K integer weights, sum(weights) ≈ field_size
  meta: {
    method: "quickbuild" | "mixture" | ...,
    generated_at,
    calibration_version,
    params,
  }
}
```

Then contest sim becomes:
- user side: `U` lineups (often <= 150)
- field side: `K` lineups (target 1k–5k)
- payout engine uses scores + weights to estimate rates/ROI

**Why weights are required:** the payout engine sorts per world; it cannot handle 30k–150k *unique* opponent lineups across 10k+ worlds. We must keep `K` small and let weights represent frequency.

## Option A: QuickBuild Field Library (start here)
### Step A1: Generate a large candidate set quickly
Use our existing QuickBuild pipeline to generate many lineups fast (you noted ~30–40k in <2 minutes with light constraints).

Suggested sweep knobs (v0):
- different seeds
- different `randomness_pct`/`jitter`
- different `min_salary` (e.g. 49000, 49500, 50000)
- optionally a small ownership penalty (off by default until we trust it)

Output: `M` unique-ish candidate lineups (M ≈ 20k–80k).

### Step A2: Deduplicate and compress to K “representative” lineups
We need K ≈ 1k–5k unique lineups that cover the space.

Compression strategies (in increasing sophistication):
- **Stratified sampling by features**: bucket lineups by `(total_own_bin, salary_bin, #sub5own, #sub10own)` and sample evenly.
- **Greedy diversity**: start with top-N by “popularity weight” then iteratively add lineups that maximize Jaccard distance.
- **Clustering** (later): treat each lineup as an 8-player set; approximate clustering via MinHash signatures.

v0 recommendation: stratified sampling + diversity guard.

### Step A3: Assign weights (how often each lineup appears in the field)
We need a function: `popularity(lineup) -> nonnegative weight`.

**v0 heuristic weights** (no training required):
- Compute `sum_own` from whatever ownership projection is available (or fallback to priors).
- Set `raw_weight = exp(alpha * sum_own)` (higher sum_own ⇒ more common)
- Add penalties/boosts for:
  - salary spend (some slates show strong preference for leaving little unused)
  - “stars-and-scrubs” patterns (counts of min-salary punts)
  - lineup duplication risk proxy (optional)
- Normalize weights to sum to desired `field_size`.

**v1 calibrated weights** (recommended using bronze CSVs):
- From historical contests, estimate lineup “popularity” as:
  - `entry_count` for exact duplicates when available (best)
  - otherwise, infer distributional targets:
    - total ownership histogram (field, top 1%, winners)
    - dupe rate by ownership band
    - salary spend histogram
- Fit a simple model for `log(weight)` from lineup features:
  - `sum_own`, `salary_used`, `num_under_5`, `num_under_10`, `num_over_50`, `team_stack_size`, etc.
- Choose objective: minimize divergence between simulated field distributions and observed distributions.

### Step A4: Cache and reuse the field library
Store per slate (date + draft_group_id):
- `projections-data/field_libraries/game_date=YYYY-MM-DD/draft_group_id=<id>/field_library_<version>.json`
- include meta (method/version/params) and calibration metadata

Contest sim then becomes instant: only scoring and payout computation are per-request.

## Option C: Mixture Field (plug-in architecture)
Option C keeps the exact same `FieldLibrary` output, but we generate it as a mixture of generators.

### Generators in the mixture
1) **QuickBuild optimizer-like** (same as Option A source)
2) **Human-ish sampler + repair** (new)
   - sample players proportional to ownership (with noise)
   - build a lineup meeting DK roster/salary constraints via a “repair” routine
   - captures common but non-optimizer lineup shapes and more realistic dupes
3) **Noisy-near-optimal**
   - take optimizer-like lineups and randomly swap 1–3 players (biased by ownership)

### Mixture weights
Let `pi = (pi_opt, pi_human, pi_noisy)` sum to 1.

We calibrate `pi` using historical CSV-derived targets:
- match field `total_own` histogram
- match dupe rate by ownership band
- match salary spend histogram

Option C’s value is robustness: we’re less reliant on any single generator being “right”.

## Using historical contest CSVs (bronze) for calibration
We already have parsing/analysis utilities in `projections/api/contest_service.py` that compute:
- ownership distributions
- duplicate lineup analysis
- top finishers patterns

Calibration approach (practical):
1) Pick representative contests per slate (e.g. main slate GPPs).
2) Extract targets:
   - distribution of `total_own` across all entries and across top 1% / winners
   - dupe rate by `total_own` band
   - salary distribution and “unused salary”
3) For a candidate field library, compute the same distributions implied by weights.
4) Tune:
   - Option A: tune weight model parameters (e.g., alpha, salary preference)
   - Option C: tune mixture `pi` and generator parameters

We don’t need perfect per-contest matching; we want a field generator that captures consistent patterns.

### Calibration scope (decision)
- Focus calibration on **DraftKings NBA Classic main-slate large-field GPPs** with **field size ≥ 5,000 entries**.
- Treat this as the primary target distribution for “what the field looks like” (ownership + dupes + salary spend).
- Smaller fields (500–2k) and non-GPP formats (double-ups, SE, showdowns) are out of scope for this phase.

Practical selection heuristics from bronze contest CSVs:
- `total_entries >= 5000`
- contest name/type indicates NBA Classic main slate GPP (exclude showdown, tiers, pick’em, single-game)
- prefer contests with reliable `%Drafted` and lineup-string columns (for ownership + dupes)

## Worlds count and stability
10k worlds may be enough for stable *mean/percentiles*, but **tail metrics** (win rate, top 1%) are noisy.

Practical approach:
- Keep worlds at 10k for interactive runs.
- Offer a “High Precision” mode at 25k–50k worlds for final decisions (possibly async/cached).
- Consider variance reduction later (e.g., reuse worlds, antithetic sampling if synthetic).

## Metrics: focus on ROI and rates (de-emphasize EV dollars)
To make outputs portable across buy-ins:
- Standardize payouts to a small set of archetypes (already in `contest_sim.yaml`).
- Report:
  - ROI (assuming entry_fee = 1.0 or normalized)
  - win/top 1%/top 5%/cash rates
  - optional “top-X%” rates as primary ordering metrics

We can keep EV dollars internally but avoid anchoring decision-making on it.

## Implementation plan (incremental)
### Phase 0: Correctness + instrumentation (fast)
- Fix self-play entrant accounting. ✅
- Make rake consistent across payout tiers and reported config. ✅
- Apply dupe penalty consistently to ROI/rates where applicable. ✅
- Add debug outputs: `field_unique_k`, `field_size`, `weights_sum`, and library metadata. ✅

### Phase 1: Option A (QuickBuild field library)
- Implement `FieldLibraryBuilder` that:
  - runs QuickBuild sweeps to generate M candidates
  - compresses to K
  - assigns weights (heuristic v0)
  - persists + caches
- Add new contest sim mode: `field_mode = self_play | generated_field`.
✅ Implemented: `quickbuild_v0` (untrained) and `quickbuild_v1_calibrated` (learned weights when available).

### Phase 1.5: Product wiring (API + UI)
- Add API endpoints to manage cached field libraries:
  - list libraries for a slate
  - build/rebuild a library for a slate
- Add dashboard UI controls to:
  - toggle `field_mode`
  - build/rebuild the field library
  - tune K and candidate pool size (v0 knobs)
✅ Implemented.

### Phase 2: Calibrate weights from bronze CSVs
- Build a small training pipeline that produces:
  - target distributions per slate bucket
  - fitted parameters for `weight(lineup)`
- Add `calibration_version` to field library meta.
✅ Implemented: `projections/contest_sim/field_weight_calibration.py` writes `projections-data/gold/field_weight_model_v1.json`.

### Phase 3: Option C mixture plug-in
- Add human-ish sampler/repair generator.
- Fit mixture weights `pi` to historical targets.
- Ship as `field_mode = mixture_field`.

## Current implementation notes
### Field library build + cache
- Cached per slate under `projections-data/field_libraries/game_date=.../draft_group_id=.../field_library_<version>.json`
- Raw QuickBuild **candidate pools** are also cached alongside the library so we can recompress quickly when news changes projections/ownership.
- Methods:
  - `quickbuild_v0`: weights derived from QuickBuild sample frequency (fallback when no learned model or no ownership)
  - `quickbuild_v1_calibrated`: weights predicted by learned `field_weight_model_v1.json` and scaled to a fixed pseudo-count sum (default 100,000)

### Candidate pool cache (for fast recompression)
- Cached per slate under `projections-data/field_libraries/game_date=.../draft_group_id=.../candidate_pool_<version>.jsonl.gz`
- If cached candidates exist and meet `candidate_pool_size`, rebuilding the library reuses candidates (no QuickBuild) and only recomputes weights + compression.
- To force generating fresh candidates, set `rebuild_candidates=true` on the build request (or toggle “Force rebuild candidate pool” in the UI).

### Dupe penalty behavior
- Exact duplication is already handled by the payout engine when the modeled field contains the same lineup (tie-splitting via `field_weights`).
- To avoid double-counting, the ownership-based dupe penalty is only applied when:
  - ownership is available **and**
  - the user lineup is **not present** in the modeled field library.

## Calibration from bronze CSVs
This trains the lineup popularity model used by `quickbuild_v1_calibrated`.

Expected bronze inputs (under `projections-data/`):
- `bronze/dk_contests/nba_gpp_data/<YYYY-MM-DD>/nba_gpp_<YYYY-MM-DD>.csv`
- `bronze/dk_contests/nba_gpp_data/<YYYY-MM-DD>/results/contest_<id>_results.csv`
- `bronze/dk_contests/ownership_by_date/<YYYY-MM-DD>.parquet` (dates without this are skipped)

Run:
```bash
uv run python -m projections.contest_sim.field_weight_calibration --max-dates 30 --out /home/daniel/projections-data/gold/field_weight_model_v1.json
```

## Open questions
- Should we bucket large-field GPPs further (e.g., 5k–20k vs 20k+) or use a single pooled target?

***We should bucket them***

- Do we want the field library to be slate-specific (draft_group_id) or date-only?

***it would need to be slate specific***

- What K gives the best speed/accuracy tradeoff (likely 1k–3k)?

***we would need to test. we should get as close to realism as possible while maintaining a fast runtime***

- Should we normalize ownership inputs to sum to 800% before computing `sum_own` features?

***yes, our ownership model already sums to 800 fwiw (check this)***
