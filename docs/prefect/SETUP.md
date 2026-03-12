# Prefect Setup

## Infrastructure

### Prefect Metadata DB (Postgres)

Use local Postgres for Prefect metadata. Do not run production on default SQLite.

```bash
docker run -d \
  --name prefect-postgres \
  --restart unless-stopped \
  -e POSTGRES_USER=prefect \
  -e POSTGRES_PASSWORD=prefect \
  -e POSTGRES_DB=prefect \
  -p 127.0.0.1:55432:5432 \
  -v prefect-postgres-data:/var/lib/postgresql/data \
  postgres:16-alpine

docker exec prefect-postgres pg_isready -U prefect -d prefect
```

### Prefect Server

Self-hosted Prefect server running as a systemd service.

**Service file**: `/etc/systemd/system/prefect-server.service`

```ini
[Unit]
Description=Prefect Server (self-hosted)
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=20

[Service]
Type=simple
User=daniel
WorkingDirectory=/home/daniel/prod/projections-v2
EnvironmentFile=-/home/daniel/.config/projections/prefect-server.env
Environment=PREFECT_PROFILE=selfhost
Environment=PREFECT_UI_API_URL=http://100.78.180.34:4200/api
Environment=PREFECT_UI_URL=http://100.78.180.34:4200
Environment=PREFECT_API_DATABASE_CONNECTION_URL=postgresql+asyncpg://prefect:prefect@127.0.0.1:55432/prefect
Environment=PREFECT_API_DATABASE_TIMEOUT=30.0
Environment=PREFECT_API_DATABASE_CONNECTION_TIMEOUT=30.0
Environment=PYTHONFAULTHANDLER=1
ExecStart=/home/daniel/prod/projections-v2/.venv311/bin/python -m prefect server start --host 100.78.180.34 --port 4200
Restart=always
RestartSec=5
TimeoutStopSec=30
KillMode=mixed

[Install]
WantedBy=multi-user.target
```

**Access**: http://100.78.180.34:4200 (via Tailscale)

### Prefect Worker

Process-based worker polling the `projections-local` work pool.

**Service file**: `/home/daniel/.config/systemd/user/prefect-worker.service`

```ini
[Unit]
Description=Prefect Worker for projections-local pool
After=network.target
StartLimitIntervalSec=300
StartLimitBurst=20

[Service]
Type=simple
WorkingDirectory=/home/daniel/prod/projections-v2
Environment=PREFECT_API_URL=http://100.78.180.34:4200/api
ExecStart=/home/daniel/prod/projections-v2/.venv311/bin/python -m prefect worker start --pool projections-local --type process --limit 1 --with-healthcheck
Restart=always
RestartSec=5
TimeoutStopSec=30
KillMode=mixed

[Install]
WantedBy=default.target
```

**Concurrency**: 1 simultaneous run

## Install/Update Services

```bash
# Install user worker unit template from repo
./scripts/deploy/install_prefect_worker_unit.sh

# Install root server unit template from repo
sudo ./scripts/deploy/install_prefect_server_unit.sh
```

## Hardening Bootstrap

Run this after reboots, host incidents, or major Prefect changes:

```bash
./scripts/deploy/harden_prefect_runtime.sh
```

What it does:

1. Ensures `prefect-postgres` container is running and healthy.
2. Sets `selfhost` Prefect profile DB settings to Postgres.
3. Bounces `prefect-server.service` and waits for `/api/health`.
4. Re-deploys all flows from `prefect.yaml`.
5. Re-applies deployment overrides (`collision_strategy=CANCEL_NEW`, limit 1).
6. Restarts `prefect-worker.service`.

## CLI Usage

All CLI commands require the venv and API URL:

```bash
source /home/daniel/prod/projections-v2/.venv311/bin/activate
export PREFECT_API_URL=http://100.78.180.34:4200/api
```

Or create an alias in `~/.bashrc`:

```bash
alias prefect-cli='source /home/daniel/prod/projections-v2/.venv311/bin/activate && PREFECT_API_URL=http://100.78.180.34:4200/api prefect'
```

## Deployments

Deployments are configured in `/prefect.yaml` and deployed via:

```bash
# Deploy all flows
prefect deploy --all

# Re-apply fields Prefect may drop on deploy
python tools/prefect_apply_deployment_overrides.py \
  --deployment nba-live-pipeline-v3/nba-live-pipeline \
  --collision-strategy CANCEL_NEW \
  --concurrency-limit 1
```

## Data Persistence

- **Prefect DB**: Postgres (`prefect-postgres`, mapped to `127.0.0.1:55432`)
- **Run Manifests**: `/home/daniel/projections-data/manifests/{date}/{task}/`

Manifests provide durable per-run records independent of Prefect metadata.
