RMH_v1.1 – Rotation Minutes Hurdle Model (Incremental Improvements)

Status: Proposed

Audience: Codex / Claude implementation agents

Baseline: RMH_v1 (rotation_minutes_hurdle_v1)

⸻

1. Objective

Build RMH_v1.1, an incremental upgrade to RMH_v1 that specifically targets the observed production pathology:

Hard downward hedging of minutes for Q / UNK players pre‑lineup, followed by sharp upward “snap” once lineups resolve.

RMH_v1.1 must:
	•	Preserve the correct hurdle decomposition (DNP mass vs conditional minutes)
	•	Reduce pessimistic early‑day bias without eliminating legitimate temporal improvement
	•	Improve conditional minutes distribution fidelity (tails) for simulation
	•	Add evaluation that reflects real production uncertainty slices, not global averages

This is not a new architecture—only high‑leverage refinements.

⸻

2. Core Changes (v1 → v1.1)

2.1 Redefine “Played” as “In Rotation”

Change
	•	Update hurdle target from:

y_play = 1[minutes >= 1]

to:

y_rot = 1[minutes >= 5]


	•	Make threshold configurable:

play_threshold_minutes: 5



Rationale
	•	Separates real rotation appearances from garbage‑time cameos
	•	Reduces label noise in both play head and conditional minutes head
	•	Aligns better with downstream rotation + sim usage

Notes
	•	Eval should still report P(minutes>=1) as a diagnostic, but the model optimizes for >=5.

⸻

3. Expanded Conditional Quantiles

3.1 Quantile Set

Upgrade conditional minutes head from:
	•	{q10, q50, q90}

To:
	•	{q05, q10, q25, q50, q75, q90, q95}

3.2 Non‑crossing Parameterization (Required)

Maintain monotonicity via cumulative softplus deltas:
	•	Predict q50
	•	Predict positive deltas via softplus

Example construction:

q25 = q50 - sp(d25)
q10 = q25 - sp(d10)
q05 = q10 - sp(d05)

q75 = q50 + sp(d75)
q90 = q75 + sp(d90)
q95 = q90 + sp(d95)

3.3 Loss
	•	Joint pinball loss across all quantiles
	•	Default: equal weighting
	•	Optional (configurable): mild tail upweighting (e.g., 1.2× on q05/q95)

Rationale
	•	Improves conditional CDF fidelity
	•	Reduces need for extrapolation in mixture quantile math
	•	Improves sim tail realism without architectural changes

⸻

4. Evaluation Additions (High Priority)

4.1 Non‑OUT Slice Metrics (Play Head)

Global AUC is dominated by OUT vs non‑OUT. Add explicit metrics on:

status != 'OUT'
status in {Ava, AVAIL, UNK, Q, PROB}

Report:
	•	AUC
	•	PR‑AUC
	•	ECE

4.2 UNK / Q Calibration Table

For status in {UNK, Q}:
	•	Bucket p_play into deciles
	•	Report empirical P(minutes >= play_threshold) per bucket

This is the primary guardrail against early‑day pessimism.

4.3 Conditional Minutes Coverage by Bucket

On played‑only rows (minutes >= threshold):

Bucket by realized minutes:
	•	5–10
	•	10–20
	•	20–30
	•	30+

Report coverage for:
	•	q10 / q50 / q90 (and extended quantiles if desired)

⸻

5. Hedge / Snap Diagnostic (New)

5.1 Purpose

Directly measure the symptom RMH is meant to fix.

5.2 Definition

Given two snapshots for the same slate:
	•	as_of_early (e.g. T‑60)
	•	as_of_late (e.g. T‑10 or post‑lineup)

For players with status in {UNK, Q} at as_of_early:

Compute:

Δ = mean_minutes_uncond_late - mean_minutes_uncond_early

Report:
	•	p50 / p90 / p99 of |Δ|
	•	Fraction of cases where Δ > 0 (downward hedge bias)

Interpretation
	•	RMH_v1.1 should reduce extreme negative early bias and reduce snap magnitude vs baseline

⸻

6. Decay / Recency Weighting (Carry Forward)
	•	Keep separate half‑lives for:
	•	play head
	•	conditional minutes head
	•	No architectural change required
	•	Ensure half‑lives are:
	•	logged
	•	stored in metrics.json

Note: tuning half‑lives is explicitly out of scope for this PR unless metrics regress badly.

⸻

7. Guardrails

7.1 Feature Leakage Test (Required)

Add a unit test asserting schema excludes known leaky columns:
	•	minutes_from_stints
	•	team_total_minutes_from_stints
	•	first_in_time_real
	•	last_out_time_real
	•	max_stint_len_real
	•	any _real (non‑prior) suffixes

Test should fail loudly if future refactors re‑introduce them.

⸻

8. Acceptance Criteria

RMH_v1.1 is acceptable if:
	•	Tests pass (existing + new)
	•	Non‑OUT play head metrics are reported and sane
	•	UNK/Q calibration shows no collapse toward zero
	•	Conditional minutes coverage is stable or improved vs v1
	•	Hedge / snap diagnostic shows reduced extreme |Δ| vs baseline

No requirement to deploy to production immediately.

⸻

9. Out of Scope (Explicitly)
	•	3‑class (DNP / garbage / rotation) softmax head
	•	Allocator‑aware or team‑structured training
	•	Snapshot‑specific model variants
	•	End‑to‑end sim optimization

These are v1.2+ concerns.

⸻

10. Repo / Workflow Conventions
	•	Work on feature branches only
	•	Commit small, logically scoped changes
	•	Push branch and open PR (targeting repo’s main/dev policy)
	•	Do not merge without review

⸻

End of Spec