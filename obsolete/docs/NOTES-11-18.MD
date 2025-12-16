So, is this “good enough to move on”?

For now and within the limited window:
	•	Yes for:
	•	DFS-ready minutes for main rotation players.
	•	Moving on to usage / FPTS modeling, or to building recon + post-recon metrics.
	•	No if your bar is:
	•	“I never want to touch minutes calibration again.”
	•	“Bench buckets should be beautifully calibrated and symmetric.”

If I were to write your TODO list for when you come back to this:
	1.	Split evaluation into two dashboards:
	•	Conditional minutes only (ignore play_prob, focus on p10/p50/p90_conds).
	•	Play probability as its own metric set.
	2.	Targeted clean-up of buckets:
	•	Consider separate conformal config for bench vs starters (or tweak bucket definitions).
	•	Maybe loosen constraints or increase k for noisy bench buckets so they lean more on global offsets.
	3.	Post-reconcile metrics job (separate CLI):
	•	Take this trained bundle, run recon on a held-out window, and see:
	•	How much QP actually improves or hurts MAE/coverage.
	•	Whether team totals and caps behave as intended.
	4.	Drift check over time:
	•	Right now you’re looking at one calibration/val window. Before going “prod-prod”, I’d check:
	•	Early season vs mid-season vs late-season windows.
	•	Injury-heavy slates vs normal ones.

But for today’s concrete question — “analyze this metrics.json with a critical eye” — the verdict is:
	•	Core minutes behavior is legit.
	•	Bench behavior is noisy and conservative, but not catastrophic.
	•	Play-prob is decently calibrated but makes unconditional metrics look worse than they really are.

When you’re ready to revisit, we can build the post-reconcile scorer 
and then really see how the full stack behaves end-to-end. ***RUN 
WAS RAN WITH CALIBRATION-PROD.YAML***
artifact is v1_full_calibration
