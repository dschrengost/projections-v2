# Prefect Setup

## Infrastructure

### Prefect Metadata DB (Postgres)

Preferred: local Postgres for Prefect metadata. For recovery / minimal installs, the repo
also supports a SQLite DB on the data volume (see "SQLite recovery baseline" below).

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

Self-hosted Prefect server running as a `systemd --user` service and bound to `127.0.0.1`.
External access is provided via `tailscale serve`.

**Service file**: `~/.config/systemd/user/prefect-server.service`

```ini
[Unit]
Description=Prefect Server (self-hosted)
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=20

[Service]
Type=simple
WorkingDirectory=/home/daniel/prod/projections-v2
EnvironmentFile=-/home/daniel/.config/projections/prefect-server.env
Environment=PREFECT_PROFILE=selfhost
Environment=PREFECT_SERVER_ALLOW_EPHEMERAL_MODE=false
Environment=PREFECT_API_DATABASE_CONNECTION_URL=sqlite+aiosqlite:////home/daniel/projections-data/control_plane/prefect/prefect.db
Environment=PREFECT_API_DATABASE_TIMEOUT=30.0
Environment=PREFECT_API_DATABASE_CONNECTION_TIMEOUT=30.0
Environment=PYTHONFAULTHANDLER=1
ExecStart=/home/daniel/prod/projections-v2/.venv/bin/python -m prefect server start --host 127.0.0.1 --port 4200
Restart=always
RestartSec=5
TimeoutStopSec=30
KillMode=mixed

[Install]
WantedBy=default.target
```

**Access (tailnet)**:

```bash
tailscale serve --yes --https 4200 http://127.0.0.1:4200
```

Then open `https://<your-node-magicdns>:4200/`.

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
Environment=PREFECT_API_URL=http://127.0.0.1:4200/api
ExecStart=/home/daniel/prod/projections-v2/.venv/bin/python -m prefect worker start --pool projections-local --type process --limit 1 --with-healthcheck
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

# Install user server unit template from repo
./scripts/deploy/install_prefect_server_unit.sh
```

## Hardening Bootstrap

Run this after reboots, host incidents, or major Prefect changes:

```bash
./scripts/deploy/harden_prefect_runtime.sh
```

What it does:

1. Restarts `prefect-server.service` (user unit) and waits for `/api/health`.
2. Ensures work pool `projections-local` exists.
3. Re-deploys all flows from `prefect.yaml`.
4. Re-applies deployment overrides (`collision_strategy=CANCEL_NEW`, limit 1).
5. Restarts `prefect-worker.service`.

## CLI Usage

All CLI commands require the venv and API URL:

```bash
source /home/daniel/prod/projections-v2/.venv/bin/activate
export PREFECT_API_URL=http://127.0.0.1:4200/api
```

Or create an alias in `~/.bashrc`:

```bash
alias prefect-cli='source /home/daniel/prod/projections-v2/.venv/bin/activate && PREFECT_API_URL=http://127.0.0.1:4200/api prefect'
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

- **Prefect DB**: SQLite (recovery baseline) at `/home/daniel/projections-data/control_plane/prefect/prefect.db`
- **Run Manifests**: `/home/daniel/projections-data/manifests/{date}/{task}/`

Manifests provide durable per-run records independent of Prefect metadata.

## SQLite Recovery Baseline

If Docker/Postgres are not available yet, the recovery baseline is:

1. Run Prefect Server on SQLite (unit template does this by default).
2. Keep deployment concurrency at `1` (`CANCEL_NEW`).
3. Keep schedules paused until live runs are validated and storage guards are verified.

When Postgres is reintroduced, override `PREFECT_API_DATABASE_CONNECTION_URL` in:
`/home/daniel/.config/projections/prefect-server.env` and restart `prefect-server.service`.
