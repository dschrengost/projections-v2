# Rotshare injury-regime eval (Nov 2025)

Goal: isolate the "next man up" failure mode (bench core under-called when starters are OUT) and verify whether `rotshare` fixes it, specifically on injury-regime slices.

## How to run

Generate candidate predictions (rotshare, no learned tau by default):

```bash
uv run python -m projections.cli.score_minutes_v1 \
  --date 2025-11-01 --end-date 2025-11-30 --mode historical \
  --bundle-dir artifacts/minutes_rotation_share/rotshare_smoke_20251229 \
  --artifact-root artifacts/rotshare_eval_preds_k10_e1 \
  --rotshare-min-players 10 \
  --rotshare-play-prob-exponent 1.0 \
  --rotshare-no-learned-tau
```

Evaluate vs current + baseline heuristic on two slices (injury-regime + overall):

```bash
uv run python -m projections.cli.eval_minutes_injury_regime \
  --start-date 2025-11-01 --end-date 2025-11-30 \
  --min-starters-out 1 --min-team-out 2 \
  --baseline-top-k 8 \
  --candidate-root artifacts/rotshare_eval_preds_k10_e1 \
  --out reports/minutes_injury_regime/2025-11_rotshare_k10_e1.json
```

Optional: evaluate learned tau (currently regresses on this month):

```bash
uv run python -m projections.cli.score_minutes_v1 \
  --date 2025-11-01 --end-date 2025-11-30 --mode historical \
  --bundle-dir artifacts/minutes_rotation_share/rotshare_tau_v4_20251229 \
  --artifact-root artifacts/rotshare_eval_preds_tau_v4 \
  --rotshare-min-players 10 \
  --rotshare-play-prob-exponent 1.0 \
  --rotshare-use-learned-tau

uv run python -m projections.cli.eval_minutes_injury_regime \
  --start-date 2025-11-01 --end-date 2025-11-30 \
  --min-starters-out 1 --min-team-out 2 \
  --baseline-top-k 8 \
  --candidate-root artifacts/rotshare_eval_preds_tau_v4 \
  --out reports/minutes_injury_regime/2025-11_rotshare_tau.json
```

## Results summary (injury-regime slice)

From `reports/minutes_injury_regime/2025-11_rotshare_k10_e1.json`:

- Bench core MAE: `5.52 -> 4.56` (Δ `-0.96`)
- Top-2 bench MAE: `5.45 -> 4.57` (Δ `-0.87`)
- Player MAE: `3.76 -> 3.01` (Δ `-0.75`)
- Top-7 minutes sum MAE: `12.37 -> 12.47` (Δ `+0.10`, effectively unchanged)
- Bench concentration errors improved: Gini MAE `0.066 -> 0.057`, HHI MAE `0.043 -> 0.033`

From `reports/minutes_injury_regime/2025-11_rotshare_tau.json` (learned tau enabled):

- Bench core MAE: `5.52 -> 6.84` (Δ `+1.32`, worse)
- Top-7 minutes sum MAE: `12.37 -> 14.37` (Δ `+1.99`, worse)

Conclusion: `rotshare` fixes the bench-core "next man up" failure mode on this slice; the current learned-tau prototype does not help yet and is disabled by default.

## Next steps

- If we still want a learned "rotation tightness" control, consider learning a discrete regime / rotation size (tight vs normal) instead of (or in addition to) continuous tau.
- Investigate persistent negative bench-core bias (under-calling) while MAE improves; likely a calibration issue (minutes magnitude) rather than rotation inclusion.
