# Prefect Setup

## Infrastructure

### Prefect Server

Self-hosted Prefect server running as a systemd service.

**Service file**: `/etc/systemd/system/prefect-server.service`

```ini
[Unit]
Description=Prefect Server (self-hosted)
After=network-online.target

[Service]
Type=simple
User=daniel
WorkingDirectory=/home/daniel/projects/projections-v2
Environment=PREFECT_PROFILE=selfhost
Environment=PREFECT_UI_API_URL=http://100.78.180.34:4200/api
Environment=PREFECT_UI_URL=http://100.78.180.34:4200
ExecStart=/home/daniel/projects/projections-v2/.venv/bin/prefect server start --host 100.78.180.34 --port 4200
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Access**: http://100.78.180.34:4200 (via Tailscale)

### Prefect Worker

Process-based worker polling the `projections-local` work pool.

**Service file**: `/etc/systemd/system/prefect-worker.service`

```ini
[Unit]
Description=Prefect Worker (process) - projections-local
After=network-online.target prefect-server.service

[Service]
Type=simple
User=daniel
WorkingDirectory=/home/daniel/projects/projections-v2
Environment=PREFECT_PROFILE=selfhost
Environment=PREFECT_API_URL=http://100.78.180.34:4200/api
ExecStart=/home/daniel/projects/projections-v2/.venv/bin/prefect worker start --pool projections-local --type process --limit 2 --with-healthcheck
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Concurrency**: 2 simultaneous jobs

## CLI Usage

All CLI commands require the venv and API URL:

```bash
source /home/daniel/projects/projections-v2/.venv/bin/activate
export PREFECT_API_URL=http://100.78.180.34:4200/api
```

Or create an alias in `~/.bashrc`:

```bash
alias prefect-cli='source /home/daniel/projects/projections-v2/.venv/bin/activate && PREFECT_API_URL=http://100.78.180.34:4200/api prefect'
```

## Deployments

Deployments are configured in `/prefect.yaml` and deployed via:

```bash
# Deploy a specific flow
prefect deploy -n live-score

# Deploy all flows
prefect deploy --all
```

## Data Persistence

- **Prefect DB**: SQLite at default location (managed by Prefect)
- **Run Manifests**: `/home/daniel/projections-data/manifests/{date}/{task}/`

Manifests are JSON files written by each flow run, providing a durable record independent of Prefect's database.
