***This document captures improvement ideas for the live inference pipeline and adjacent system reliability work.***

Canonical spec:

- See [LIVE_PIPELINE_PRODUCTION_SPEC.md](./LIVE_PIPELINE_PRODUCTION_SPEC.md) for the living production-readiness spec.
- This file should remain the lightweight parking lot for ideas and incidents that may later be promoted into the canonical spec.

## Current High-Priority Problems

### 1. We rerun the full slate too often

Right now the live pipeline reprocesses every game on each run. That is expensive, slows time-to-publish, and increases the odds that we publish using stale inputs while a later injury or lineup update is still arriving.

We should move toward game-scoped inference:

- Detect meaningful deltas at the game level instead of blindly rescoring the full slate.
- Only rerun a game when one of its inputs materially changes.
- Keep the expensive model path focused on affected games only.

Examples of meaningful deltas:

- A player is ruled out or upgraded to active.
- A projected starter becomes a confirmed starter.
- A confirmed starter flips to out.
- A meaningful vegas move occurs.
- A player prop move materially changes implied usage or minutes.

### 2. Slow pipeline + stale snapshot selection near lock

This is now a known failure mode.

Observed example on the 2026-02-27 slate:

- The `2026-02-28T00:00:00Z` features build still had James Harden as `status=Q`, `is_out=0`.
- Dennis Schroder was already picked up as confirmed starter in that same build.
- The first Rotowire lineup artifact I found with Harden marked `out` arrived at `2026-02-28T00:15:07Z`.
- There was no later successful live features rebuild after the `00:00Z` run.

Plausible explanation:

- We likely started a run using the latest available injury snapshot at that moment, which may have still reflected the 6:15 PM ET state.
- The official 6:30 PM ET injury signal may have landed slightly late relative to run start.
- Because the pipeline is slow, we effectively published on stale inputs.
- Because the system later went down, we never got the corrective rerun that would have consumed the later lineup `out` signal.

Improvements needed here:

- Add a near-lock freshness gate for injuries and lineups.
- Refuse to score if the latest injury snapshot is older than the scheduled report window when we are inside a lock-critical window.
- Add a short grace/wait policy around official report times instead of immediately starting expensive inference.
- Split scraping/input-freeze from expensive inference so input freshness can be evaluated before model work starts.
- Add a priority fast-lane rerun path for confirmed `out` and confirmed starter changes.

### 3. We need better handling of Rotowire projected -> confirmed transitions

This is still an area of weakness.

We appear to catch some projected/confirmed starter changes, but the system is not clearly structured around starter confirmation as a first-class trigger. We should tighten:

- Promotion of projected starter -> confirmed starter.
- Demotion when a projected starter is explicitly ruled out.
- Clear precedence between official injury reports, Rotowire lineups, and fallback sources.
- Explicit alerts when lineup data and injury data disagree for the same player.

### 4. Pipeline recovery after a missed update is too weak

Once we miss a key event, recovery is too dependent on the next scheduled run succeeding.

We should add:

- Automatic rerun on material input change, not just fixed cadence.
- A health check that compares the current published run timestamp to the newest available injury/lineup inputs.
- Alerting when a newer authoritative input exists but published artifacts are still based on an older snapshot.
- A lightweight reconciler that can say: "published run is stale relative to inputs for CLE-DET" and trigger only that game.

## Proposed System Changes

### Input freshness and orchestration

- Track report windows explicitly: 5:30 PM ET, 6:30 PM ET, 7:30 PM ET, etc.
- Before starting inference, check whether the latest injury snapshot is at or beyond the expected report timestamp.
- Add a bounded wait loop near scheduled report windows so we prefer a slightly later fresh input over an immediate stale run.
- Stamp every published game with the exact injury snapshot ts, lineup snapshot ts, odds ts, and props ts used.
- Surface this freshness data in the dashboard/API so we can see stale-input situations immediately.

### Event-driven / delta-driven execution

- Compute per-game input digests and only rerun when a game's digest changes materially.
- Define a "material change" contract instead of treating all input changes equally.
- Separate cheap change detection from expensive model inference.
- Add a hot-path "late news" scorer that can rebuild a single game quickly.

### Source redundancy and cross-checking

- Keep the official injury source as highest priority, but do not rely on it alone near lock.
- Cross-check official injury reports against Rotowire lineups and ESPN injuries.
- If lineup source says player is `out` and official injury source still says `Q`, flag and optionally force conservative handling until rerun completes.
- Add a source disagreement monitor for players whose status materially affects minutes allocation.

### Model/runtime architecture

- The new production inference path is still experimental and appears to be too slow for live operations.
- We should define a latency budget for each stage: scrape, feature build, score, finalize, publish.
- Add per-stage timing breakdowns to every run artifact.
- Consider a fallback "fast conservative mode" for late news windows if the full model path is too slow.

### Storage and operational resilience

All production data currently living on a single SSD is a real operational risk.

We need redundant storage for:

- Raw scraped inputs.
- Run-scoped live artifacts.
- Published pointers/manifests.
- Model artifacts and configs.
- Historical labels/training data.

At minimum:

- Add automated backups to a second physical disk or network-attached storage.
- Keep regular snapshots of `/home/daniel/projections-data`.
- Version and back up configs and promoted model pointers separately from raw data.
- Periodically test restore, not just backup creation.

Longer term:

- Separate hot live data from archive/training data.
- Consider mirrored storage or RAID for local disk redundancy.
- Consider off-machine backup for disaster recovery.

## Other Improvements To Revisit

- Eval is serviceable for now, but should eventually be made more automatic and more tightly connected to promotion decisions.
- We scrape box scores, but the downstream value chain is still incomplete in places.
- We should build toward automated model retraining with clear guardrails, promotion checks, and rollback paths.

## Candidate Action Items

- Add a lock-window freshness gate for injuries and lineups.
- Add a short bounded wait around scheduled NBA injury report times.
- Add per-game delta detection and game-scoped reruns.
- Add an alert when published artifacts are older than the newest authoritative input for a live game.
- Add source disagreement diagnostics between official injury feed, Rotowire lineups, and ESPN injuries.
- Add per-stage latency instrumentation to every live run.
- Add a fast late-news fallback inference path.
- Add redundant storage and tested backup/restore for `projections-data`.
