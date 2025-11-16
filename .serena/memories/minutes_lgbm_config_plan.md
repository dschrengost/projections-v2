## Config wiring plan (Nov 17 2025)
- Current LightGBM + ridge baseline trainers only accept CLI options; no YAML parsing.
- Config goal: introduce yaml-defined runs that map onto existing Typer params (date windows, conformal, playable guardrails).
- Implementation direction: add dataclasses for config (trained/val windows + hyperparams), load YAML, let CLI flags override; Pandera remains for dataframe schemas, not config parsing.
- Next steps once reloaded: build config dataclasses, loader, and CLI flag precedence tests.