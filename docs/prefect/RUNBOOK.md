# Prefect Runbook

## Common Operations

### Trigger a Manual Run

```bash
source .venv/bin/activate
export PREFECT_API_URL=http://100.78.180.34:4200/api

# Run live-score now
prefect deployment run 'live-score/live-score'

# Run with parameters
prefect deployment run 'live-score/live-score' --param game_date=2025-12-19

# Run dk-salaries
prefect deployment run 'dk-salaries/dk-salaries'
```

### Pause/Resume Schedules

```bash
# Pause a deployment (stops scheduled runs, manual runs still work)
prefect deployment pause 'live-score/live-score'

# Resume
prefect deployment resume 'live-score/live-score'
```

### View Recent Runs

```bash
prefect flow-run ls --limit 10
```

### Check Deployment Status

```bash
prefect deployment ls
```

## Rollback to systemd

If Prefect has issues, rollback is simple:

```bash
# 1. Pause Prefect schedules
prefect deployment pause 'live-score/live-score'
prefect deployment pause 'dk-salaries/dk-salaries'

# 2. Re-enable systemd timers
sudo systemctl enable --now live-score.timer
sudo systemctl enable --now dk-salaries.timer
```

## Troubleshooting

### Worker Not Picking Up Jobs

1. Check worker is running:
   ```bash
   sudo systemctl status prefect-worker
   ```

2. Check work pool has capacity:
   ```bash
   prefect work-pool inspect projections-local
   ```

3. Restart worker:
   ```bash
   sudo systemctl restart prefect-worker
   ```

### Server Unreachable

1. Check server is running:
   ```bash
   sudo systemctl status prefect-server
   ```

2. Verify Tailscale:
   ```bash
   tailscale status
   curl http://100.78.180.34:4200/api/health
   ```

### Flow Fails Immediately

Check the flow run logs in the UI or via CLI:

```bash
prefect flow-run logs <FLOW_RUN_ID>
```

Common issues:
- Missing environment variables
- Script path issues
- Missing dependencies in venv

## Manifests

Each flow run writes a manifest to `/home/daniel/projections-data/manifests/`:

```
manifests/
└── 2025-12-19/
    ├── live-scrape/
    │   └── 20251219T045023Z.json
    ├── live-score/
    │   └── 20251219T050000Z.json
    └── dk-salaries/
        └── 20251219T130000Z.json
```

Manifest format:
```json
{
  "task_name": "live-score",
  "start_ts": "2025-12-19T05:00:00+00:00",
  "end_ts": "2025-12-19T05:02:30+00:00",
  "duration_s": 150.0,
  "exit_code": 0,
  "success": true,
  "output_paths": [...],
  "error_snippet": null,
  "game_date": "2025-12-19"
}
```
