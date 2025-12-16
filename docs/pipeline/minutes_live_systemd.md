# Minutes Live Systemd Templates

Use these snippets as starting points; adjust paths, users, and calendars for your host. All commands assume `uv` is on the PATH for the service user.

## Environment

- `PROJECTIONS_DATA_ROOT`: base data directory (bronze/silver/gold).
- `MINUTES_DAILY_ROOT`: location of scored projections (e.g., `/home/daniel/projects/projections-v2/artifacts/minutes_v1/daily`).
- `LIVE_BUNDLE_DIR`: minutes_v1 bundle used by `score_minutes_v1`.
- `MINUTES_PRIORS_ROOT`: where starter priors live (defaults to `$PROJECTIONS_DATA_ROOT/gold/minutes_priors`).

## Services / Timers

### Single consolidated daily unit (labels → features → priors → QC)

Use one unit to avoid overlapping services; everything runs serially.

```ini
[Unit]
Description=Minutes daily pipeline (labels, features, priors, QC)

[Service]
Type=oneshot
Environment=PROJECTIONS_DATA_ROOT=/home/daniel/projections-data
Environment=MINUTES_DAILY_ROOT=/home/daniel/projects/projections-v2/artifacts/minutes_v1/daily
Environment=MINUTES_PRIORS_ROOT=${PROJECTIONS_DATA_ROOT}/gold/minutes_priors
ExecStart=/bin/sh -c '\
  TARGET="$(date -Idate -d "yesterday")" && \
  /home/daniel/.local/bin/uv run python -m projections.cli.build_minutes_labels \
    --start-date "$TARGET" --end-date "$TARGET" --data-root "${PROJECTIONS_DATA_ROOT}" && \
  /home/daniel/.local/bin/uv run python -m projections.pipelines.build_features_minutes_v1 \
    --start-date "$TARGET" --end-date "$TARGET" --data-root "${PROJECTIONS_DATA_ROOT}" && \
  /home/daniel/.local/bin/uv run python -m projections.cli.build_starter_priors \
    --features-root "${PROJECTIONS_DATA_ROOT}/gold/features_minutes_v1" \
    --cutoff-date "$TARGET" \
    --out "${MINUTES_PRIORS_ROOT}/starter_slot_priors.parquet" \
    --history-out "${MINUTES_PRIORS_ROOT}/player_starter_history.parquet" && \
  /home/daniel/.local/bin/uv run python -m projections.cli.check_minutes_live \
    --game-date "$TARGET" \
    --salaries-path "/var/minutes/dk/${TARGET}.csv" \
    --projections-root "${MINUTES_DAILY_ROOT}" \
    --qc-root "/home/daniel/projects/projections-v2/artifacts/minutes_v1/live_qc" \
'

[Timer]
OnCalendar=*-*-* 03:30
Persistent=true
```

### minutes-boxscores-labels-nightly

```ini
[Unit]
Description=Freeze boxscore labels for yesterday

[Service]
Type=oneshot
Environment=PROJECTIONS_DATA_ROOT=/home/daniel/projections-data
ExecStart=/home/daniel/.local/bin/uv run python -m projections.cli.build_minutes_labels \
  --start-date "$(date -Idate -d 'yesterday')" \
  --end-date "$(date -Idate -d 'yesterday')" \
  --data-root "${PROJECTIONS_DATA_ROOT}"

[Timer]
OnCalendar=*-*-* 03:10
Persistent=true
```

### minutes-features-priors-nightly

```ini
[Unit]
Description=Rebuild gold features and starter priors nightly
After=minutes-boxscores-labels-nightly.service

[Service]
Type=oneshot
Environment=PROJECTIONS_DATA_ROOT=/home/daniel/projections-data
ExecStart=/bin/sh -c '\
  START="$(date -Idate -d "yesterday")" && \
  /home/daniel/.local/bin/uv run python -m projections.pipelines.build_features_minutes_v1 \
    --start-date "$START" --end-date "$START" --data-root "${PROJECTIONS_DATA_ROOT}" && \
  /home/daniel/.local/bin/uv run python -m projections.cli.build_starter_priors \
    --features-root "${PROJECTIONS_DATA_ROOT}/gold/features_minutes_v1" \
    --cutoff-date "$START" \
    --out "${PROJECTIONS_DATA_ROOT}/gold/minutes_priors/starter_slot_priors.parquet" \
    --history-out "${PROJECTIONS_DATA_ROOT}/gold/minutes_priors/player_starter_history.parquet" \
'

[Timer]
OnCalendar=*-*-* 04:00
Persistent=true
```

### live-minutes-qc

```ini
[Unit]
Description=Live projections QC for today

[Service]
Type=oneshot
Environment=MINUTES_DAILY_ROOT=/home/daniel/projects/projections-v2/artifacts/minutes_v1/daily
Environment=PROJECTIONS_DATA_ROOT=/home/daniel/projections-data
ExecStart=/bin/sh -c '\
  GAME_DATE="$(date -Idate)" && \
  SALARIES="/var/minutes/dk/${GAME_DATE}.csv" && \
  /home/daniel/.local/bin/uv run python -m projections.cli.check_minutes_live \
    --game-date "$GAME_DATE" \
    --salaries-path "$SALARIES" \
    --projections-root "${MINUTES_DAILY_ROOT}" \
    --qc-root "/home/daniel/projects/projections-v2/artifacts/minutes_v1/live_qc" \
'

[Timer]
OnCalendar=Mon-Sun 11:00,15:00,18:00
Persistent=true
```

> Tip: enable timers with `systemctl enable --now <unit>.timer` after copying service/timer files into `/etc/systemd/system/`.
