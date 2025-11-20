Live pipeline/service setup (Nov 19/20, 2025):
- run_live_pipeline.sh tip gating fixed: schedule path printf corrected; schedule envs passed into the tip-window python. Uses first/last tip from silver schedule for today and skips runs outside [first_tip - LIVE_TIP_LEAD_MINUTES, last_tip + LIVE_TIP_TAIL_MINUTES]. Defaults lead=60, tail=180.
- Lock guard in build_minutes_live via --lock-buffer-minutes (env LIVE_LOCK_BUFFER_MINUTES); default 15.
- LIVE_SCORE=1 drop-ins installed for live-pipeline, live-pipeline-evening, live-pipeline-weekend services. Env values in overrides: LIVE_SCORE=1, LIVE_TIP_LEAD_MINUTES=60, LIVE_TIP_TAIL_MINUTES=180, LIVE_LOCK_BUFFER_MINUTES=15, LIVE_FLAGS=.
- Permissions fixed on /home/daniel/projections-data/live/features_minutes_v1/2025-11-19/run=20251120T023000Z and artifacts/minutes_v1/daily/2025-11-19/run=20251120T023000Z (chowned to daniel).
- Added live-pipeline-fast.timer to run every 5 minutes (OnCalendar=*:0/5) targeting live-pipeline.service. Enabled and active.
- Services/timers restarted: live-pipeline.service, live-pipeline-evening.service, live-pipeline-weekend.service; timers live-pipeline-hourly.timer, live-pipeline-evening.timer, live-pipeline-weekend.timer, live-pipeline-fast.timer.
- live-pipeline.service successfully runs scrape+build+score (LIVE_SCORE=1); last run produced run=20251120T023000Z (53 rows) under artifacts/minutes_v1/daily/2025-11-19.
- Backend restart command: pkill -f "uvicorn projections.api.minutes_api" || true; env PROJECTIONS_DATA_ROOT=/home/daniel/projections-data UV_CACHE_DIR=/tmp/uv-cache uv run uvicorn projections.api.minutes_api:create_app --host 0.0.0.0 --port 8501 --factory
