Rotation Minutes Hurdle Model v1 (RMH_v1)

Status: Design spec (ready for implementation)

Primary Dataset: rotation_train_v1_boxscore_20260112
Secondary Dataset (validation / ablations): rotation_train_v1_boxscore_20260112_szn2025

Audience: Codex / Claude (implementation agents), human reviewers

⸻

1. Motivation & Problem Statement

Empirical audits of the rotation training datasets show that:
	•	Player status (Ava / OUT / AVAIL / UNK / Q / PROB) primarily affects probability of playing (mass at 0).
	•	Conditional minutes given play are remarkably stable across uncertainty statuses.
	•	~40% of rows are DNP (minutes = 0), dominated by OUT rows.

Training a single unconditional regression head for minutes causes systematic pre-tip hedging:
	•	Mean minutes are pulled downward when injury uncertainty exists.
	•	Predictions “snap” once lineups resolve.

This behavior is correct under MSE, but wrong for downstream rotation realism and simulation stability.

Conclusion: Minutes must be modeled as a zero-inflated / hurdle process.

⸻

2. Modeling Approach (Hurdle Decomposition)

We decompose minutes into two components:

2.1 Head A — Play / In-Rotation Probability

Target:

y_play = 1 if minutes >= T else 0

Where T is configurable (default T = 1, optional T = 5 for stricter rotation definition).

Output:
	•	p_play ∈ [0,1]

Loss:
	•	Binary cross-entropy (BCE)

⸻

2.2 Head B — Conditional Minutes Distribution | Play

Trained only on rows where y_play = 1.

Targets:
	•	Conditional minutes quantiles: q10, q50, q90

Loss:
	•	Joint pinball loss over selected quantiles

L = Σ_τ w_τ · Pinball(y, q_τ)

Quantiles are trained jointly with a shared trunk.

Non-crossing constraint (lightweight):
	•	Predict q50
	•	Predict positive deltas via softplus

q10 = q50 - softplus(d10)
q90 = q50 + softplus(d90)


⸻

2.3 Optional Head C — Starter Probability (Optional)

Target: starter_flag_label

Used only when starters are unknown; ignored once confirmed.

This head is optional and may be added in a follow-up iteration.

⸻

3. Unconditional Output Construction (Mixture Logic)

3.1 Mean Minutes

E[minutes | X] = p_play · E[minutes | play, X]

Where E[minutes | play] ≈ q50.

⸻

3.2 Unconditional Quantiles (Correct Mixture Math)

Let:
	•	p = p_play
	•	F_pos(y) = CDF of minutes | play

Then:

F(y) = (1 - p) · 1[y ≥ 0] + p · F_pos(y)

Unconditional quantile q_τ:
	•	If τ ≤ (1 - p) → q_τ = 0
	•	Else:

q_τ = q_pos((τ - (1 - p)) / p)

Important:
	•	Do not clamp adjusted quantiles to minimum conditional grid.
	•	This avoids the previously observed “floor inflation” bug.

⸻

4. Features & Semantics

4.1 Injury Semantics

Empirical finding:
	•	Ava ≈ no injury-feed row
	•	Non-Ava ≈ injury-feed row exists

Action:
	•	Introduce explicit boolean feature:

has_injury_row = (injury_as_of_ts IS NOT NULL) OR (status != 'Ava')

Do not rely on NaN semantics of prior_play_prob.

⸻

4.2 OUT Rows Handling
	•	OUT rows are included only in the play head.
	•	Conditional minutes head excludes OUT implicitly via y_play = 0.

This prevents OUT rows from dominating regression loss.

⸻

5. Recency Weighting

Each training row receives a recency weight:

w_i = exp(-Δt_i / half_life)

Key design choice:
	•	Separate half-lives per head:

Head	Default Half-Life
Play (BCE)	Short (e.g. 30 days)
Minutes	Play

Half-lives are tunable hyperparameters selected via walk-forward CV.

⸻

6. Training Procedure
	1.	Load dataset rotation_train_v1_boxscore_20260112
	2.	Build features (shared trunk)
	3.	Construct targets:
	•	y_play
	•	minutes | y_play = 1
	4.	Apply recency weights
	5.	Train:
	•	Head A: BCE
	•	Head B: joint pinball loss
	6.	Optional: train Head C (starter)
	7.	Save artifacts:
	•	model weights
	•	feature schema hash
	•	training config JSON

⸻

7. Evaluation Protocol

7.1 Play Head
	•	AUC / PR-AUC
	•	Calibration (ECE / reliability curves)
	•	Slice by status (Ava / AVAIL / UNK / Q / OUT)

7.2 Conditional Minutes Head
	•	Pinball loss on played-only slice
	•	Coverage (empirical P(y ≤ qτ))
	•	Slice by status and role

7.3 Unconditional Outputs
	•	RMSE / MAE (reported but not primary)
	•	DNP accuracy

⸻

8. Integration & Simulation

8.1 Allocator / 240-Minute Reconciliation

Model outputs are treated as priors.

Allocator enforces:
	•	team total = 240
	•	starter floors
	•	bench caps

⸻

8.2 World Generation

For each player:
	1.	Sample play ~ Bernoulli(p_play)
	2.	If play = 1, sample minutes from conditional distribution
	3.	Reconcile team totals

This removes artificial pre-tip hedging and produces realistic rotation variance.

⸻

9. Acceptance Criteria (Must-Pass)
	•	Conditional minutes distributions stable across statuses
	•	p_play calibration within ±3% ECE
	•	No upward bias in unconditional p10/p25 due to clamping
	•	Improved pre-tip stability vs current minutes model

⸻

10. Non-Goals (Explicit)
	•	Predicting exact coach substitution patterns
	•	Eliminating all DNP uncertainty
	•	Removing allocator / reconciliation logic

⸻

11. Future Extensions (Out of Scope)
	•	Team-level latent rotation volatility factor
	•	Copula-based minute correlation
	•	Starter-conditioned minutes distributions

⸻

12. Repository & Workflow Conventions

All implementation work must follow these conventions:
	•	Development is done only on dev (never directly on main).
	•	Create a feature branch off dev for this work (e.g. feat/rotation-minutes-hurdle-v1).
	•	Make small, logically scoped commits with clear messages.
	•	Push the branch and open a PR targeting dev.
	•	Do not merge without review or explicit approval.

Any deviation from this workflow should be called out explicitly in the PR description.